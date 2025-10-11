import os, sys
import numpy as np
import gymnasium as gym
import ipdb
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.common.utils import plot_total_reward


ACTION_SCALE = 1.0
LOG_STD_MIN, LOG_STD_MAX = -6, 2


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 32)
        self.ln2 = nn.LayerNorm(32)
        self.mean_head = nn.Linear(32, action_dim)
        self.log_std_head = nn.Linear(32, action_dim)

    def forward(self, x):
        h1 = self.ln1(torch.relu(self.fc1(x)))
        h2 = self.ln2(torch.relu(self.fc2(h1)))
        mu = self.mean_head(h2)
        log_std = self.log_std_head(h2)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)  # クランプ
        std = torch.exp(log_std)
        return mu, std


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 32)
        self.ln2 = nn.LayerNorm(32)
        self.mean_head = nn.Linear(32, 1)

    def forward(self, x):
        h1 = self.ln1(F.relu(self.fc1(x)))
        h2 = self.ln2(F.relu(self.fc2(h1)))
        val = self.mean_head(h2)
        return val


class Agent:
    def __init__(self, state_dim: int, action_dim: int):
        self.gamma = 0.99
        self.lr_pi = 3e-4
        self.lr_v = 3e-4

        self.pi = PolicyNet(state_dim, action_dim)
        self.v = ValueNet(state_dim)
        self.optimizer_pi = optimizer = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optimizer = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        with torch.no_grad():  # 収集はno_gradでOK
            s = torch.as_tensor(np.asarray(state).ravel(), dtype=torch.float32).unsqueeze(0)
            mu, std = self.pi(s)
            dist = Normal(mu, std)
            u = dist.rsample()
            a = torch.tanh(u) * ACTION_SCALE
            logp_u = dist.log_prob(u)
            two = torch.tensor(2.0, device=logp_u.device, dtype=logp_u.dtype)
            corr = 2.0 * (torch.log(two) - u - F.softplus(-2.0 * u))
            old_logp = (logp_u - corr).sum(dim=-1, keepdim=True)  # [1,1]

        return a, u.detach(), old_logp.cpu()

    def update_batch(
        self,
        buf,
        lam=0.95,
        clip_eps=0.2,
        epochs=16,
        minibatch=64,
        target_kl=0.02,
    ):
        S = torch.as_tensor(np.array(buf["s"]), dtype=torch.float32)
        S_next = torch.as_tensor(np.array(buf["s_next"]), dtype=torch.float32)
        U = torch.cat(buf["u"], dim=0)
        OLD = torch.cat(buf["old_logp"], dim=0).to(dtype=torch.float32)
        R = torch.as_tensor(buf["r"], dtype=torch.float32).unsqueeze(1)
        Term = torch.as_tensor(buf["term"], dtype=torch.float32).unsqueeze(1)

        # ----- GAE & 価値ターゲット -----
        with torch.no_grad():
            V = self.v(S)
            V_next = self.v(S_next)
            not_done = 1.0 - Term  # terminatedのみ0、truncatedは1
            delta = R + self.gamma * V_next * not_done - V

            adv = torch.zeros_like(delta)
            gae = 0.0
            for t in reversed(range(len(S))):
                gae = delta[t] + self.gamma * lam * not_done[t] * gae
                adv[t] = gae
            target = adv + V
            # 標準化＋外れ値クリップ：安定化
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv = adv.clamp_(-5.0, 5.0)

        T = len(S)
        idx_all = torch.arange(T)
        two = torch.tensor(2.0, device=U.device, dtype=U.dtype)

        # ----- PPO: 複数エポック × ミニバッチ -----
        for _ in range(epochs):
            perm = idx_all[torch.randperm(T)]
            for i in range(0, T, minibatch):
                b = perm[i : i + minibatch]

                mu, std = self.pi(S[b])
                dist = Normal(mu, std)
                logp_u = dist.log_prob(U[b])
                corr = 2.0 * (torch.log(two) - U[b] - F.softplus(-2.0 * U[b]))
                new_logp = (logp_u - corr).sum(dim=-1, keepdim=True)
                ratio = (new_logp - OLD[b].to(new_logp.device)).exp()

                # Policy loss
                a = adv[b].to(ratio.device)
                pg1 = ratio * a
                pg2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * a
                loss_pi = -torch.min(pg1, pg2).mean()

                # Value loss
                v_pred = self.v(S[b])
                loss_v = F.smooth_l1_loss(v_pred, target[b].to(v_pred.device))

                self.optimizer_pi.zero_grad(set_to_none=True)
                self.optimizer_v.zero_grad(set_to_none=True)
                loss_v.backward()
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.v.parameters(), 1.0)
                self.optimizer_pi.step()
                self.optimizer_v.step()

            # ----- KL発散ガード（早期打ち切り）-----
            with torch.no_grad():
                mu, std = self.pi(S)
                dist = Normal(mu, std)
                logp_u = dist.log_prob(U)
                corr = 2.0 * (torch.log(two) - U - F.softplus(-2.0 * U))
                new_logp_full = (logp_u - corr).sum(dim=-1, keepdim=True)
                approx_kl = (OLD.to(new_logp_full.device) - new_logp_full).mean().item()
            if approx_kl > 1.5 * target_kl:
                break


def train():
    episodes = 10000
    env = gym.make("Hopper-v5")

    agent = Agent(11, 3)
    reward_history = []

    T = 1024
    best_reward = 0
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        buffer = dict(s=[], s_next=[], r=[], u=[], old_logp=[], term=[])

        while not done:
            action, sample_u, old_logp = agent.get_action(state)
            action_np = action.detach().cpu().numpy().ravel()
            next_state, reward, term, trunc, _ = env.step(action_np)
            done = term or trunc

            # バッファへ追加
            buffer["s"].append(state)
            buffer["r"].append(reward)
            buffer["u"].append(sample_u)  # detach済み
            buffer["old_logp"].append(old_logp)  # CPUでOK
            buffer["s_next"].append(next_state)
            buffer["term"].append(term)
            state = next_state

            # T ステップたまったら更新
            if len(buffer["s"]) == T or done:
                agent.update_batch(buffer)
                buffer = dict(s=[], s_next=[], r=[], u=[], old_logp=[], term=[])
        if episode % 100 == 0:
            eval_rewards = []
            for _ in range(10):  # 決定論評価
                s, _ = env.reset()
                done = False
                total = 0
                while not done:
                    with torch.no_grad():
                        mu, _ = agent.pi(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
                        a = torch.tanh(mu) * ACTION_SCALE  # deterministic
                    s, r, term, trunc, _ = env.step(a.numpy().ravel())
                    total += r
                    done = term or trunc
                eval_rewards.append(total)
            avg_r = np.mean(eval_rewards)
            reward_history.append(avg_r)
            print(f"[Eval] episode={episode}, avg_reward={avg_r:.1f}")
            if avg_r > best_reward:
                best_reward = avg_r
                torch.save(agent.pi.state_dict(), "best_actor.pth")
                torch.save(agent.v.state_dict(), "best_critic.pth")
                print("✅ Saved new best model!")

    plot_total_reward(reward_history)


if __name__ == "__main__":
    train()
