import numpy as np
import gymnasium as gym
import ipdb
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

ACTION_SCALE = 1.0

writer = SummaryWriter("runs/ac_exp5")


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_std_init = 0.6
        self.action_var = torch.full((action_dim,), self.action_std_init * self.action_std_init)
        self.hidden_space1 = 64
        self.hidden_space2 = 64
        self.actor = nn.Sequential(
            nn.Linear(state_dim, self.hidden_space1),
            nn.ReLU(),
            nn.Linear(self.hidden_space1, self.hidden_space2),
            nn.ReLU(),
            nn.Linear(self.hidden_space2, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, self.hidden_space1),
            nn.ReLU(),
            nn.Linear(self.hidden_space1, self.hidden_space2),
            nn.ReLU(),
            nn.Linear(self.hidden_space2, 1),
        )
        self.apply(weights_init_)

    def forward(self, x):
        mu = self.actor(x.float())
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        state_val = self.critic(x.float())
        return mu, cov_mat, state_val

    # def set_action_std(self, new_action_std):
    #     self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)


class Agent:
    def __init__(self, state_dim: int, action_dim: int):
        # Hyperparameters
        self.lr_actor = 0.0003
        self.lr_critic = 0.001
        self.gamma = 0.99
        self.lamnda = 0.95
        self.eps = 1e-6
        self.eps_clip = 0.2
        self.K_epochs = 20
        # Init policy
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": self.lr_actor},
                {"params": self.policy.critic.parameters(), "lr": self.lr_critic},
            ]
        )
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.init_buffer()

    def init_buffer(self):
        self.buffer = dict(action=[], prob=[], state=[], state_val=[], reward=[], n_state=[], term=[], trunc=[])

    def sample_action(self, state: np.ndarray):
        state = torch.tensor(state)
        if state.ndim == 1:
            state = state.unsqueeze(0)  # [N, obs]

        with torch.no_grad():
            mu, cov_mat, state_val = self.policy_old(state)
            dist = MultivariateNormal(mu, cov_mat)
            action = dist.sample()
            prob = dist.log_prob(action)
            two = torch.tensor(2.0, device=prob.device, dtype=prob.dtype)
            corr = 2.0 * (torch.log(two) - action - F.softplus(-2.0 * action))
            prob -= corr.sum(dim=-1)
        return action, prob, state_val

    def update(self, n_envs: int, step: int):
        OLD_State = torch.tensor(np.array(self.buffer["state"]), dtype=torch.float32)
        OLD_V = torch.tensor(np.array(self.buffer["state_val"]), dtype=torch.float32)
        OLD_N_state = torch.tensor(np.array(self.buffer["n_state"]), dtype=torch.float32)
        OLD_Action = torch.tensor(self.buffer["action"], dtype=torch.float32)
        OLD_Prob = torch.tensor(self.buffer["prob"], dtype=torch.float32)
        Reward = torch.tensor(self.buffer["reward"], dtype=torch.float32).unsqueeze(1)
        Term = torch.tensor(self.buffer["term"], dtype=torch.float32).unsqueeze(1)
        Trunc = torch.tensor(self.buffer["trunc"], dtype=torch.float32).unsqueeze(1)

        B = OLD_State.shape[0]
        assert B % n_envs == 0, "収集数が n_envs で割り切れません"
        T = B // n_envs

        def time_env(x):
            # x: [B, *] -> [T, n_envs, *]
            return x.view(T, n_envs, *x.shape[1:])

        n_Reward = time_env(Reward)
        n_Term = time_env(Term)
        n_Trunc = time_env(Trunc)

        # GAE、価値ターゲット
        with torch.no_grad():
            n_V = time_env(OLD_V)
            _n_NV = time_env(self.policy_old.critic(OLD_N_state))
            # 1) ブートストラップ用マスク：termだけ0、truncは1
            m_boot = 1.0 - n_Term
            # 2) キャリー用マスク：term も trunc も 0（= 境界で GAE を切る）
            m_carry = 1.0 - torch.clamp(n_Term + n_Trunc, max=1.0)
            n_NV = _n_NV * m_boot
            delta = n_Reward + self.gamma * n_NV - n_V
            # delta = (delta - delta.mean()) / (delta.std() + 1e-7)

            adv = torch.zeros_like(delta)
            gae: torch.Tensor = torch.zeros((n_envs, 1), dtype=delta.dtype)
            for t in reversed(range(T)):
                gae = delta[t] + self.gamma * self.lamnda * m_carry[t] * gae
                adv[t] = gae
            target = adv + n_V  # λリターン
            # 安定化：標準化＋外れ値クリップ。要素数 1 の正規化は回避
            if B > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv = adv.clamp_(-10.0, 10.0)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            mu, cov_mat, vs = self.policy(OLD_State)
            dist = MultivariateNormal(mu, cov_mat)
            prob = dist.log_prob(OLD_Action)
            two = torch.tensor(2.0, device=prob.device, dtype=prob.dtype)
            corr = 2.0 * (torch.log(two) - OLD_Action - F.softplus(-2.0 * OLD_Action))
            prob -= corr.sum(dim=-1)
            ent = dist.entropy().mean()
            vs = vs.reshape(B)
            adv = adv.reshape(B)
            target = target.reshape(B)
            ratios = torch.exp(prob - OLD_Prob)
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            loss = (-(torch.min(surr1, surr2))).mean() + 0.5 * self.MseLoss(vs, target) - 0.01 * ent

            # Update the policy network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # Empty / zero out all episode-centric/related variables
        self.init_buffer()

        # logging
        writer.add_scalar("loss/policy+value", loss.item(), step)
        avg_mu = mu.mean(axis=0)
        actions_scaled = torch.tanh(OLD_Action) * ACTION_SCALE
        actions_scaled_mean = actions_scaled.mean(axis=0)
        actions_scaled_max = actions_scaled.max(axis=0).values
        actions_scaled_min = actions_scaled.min(axis=0).values
        writer.add_scalars("stats/mu", {f"d{i}": avg_mu[i].item() for i in range(len(avg_mu))}, step)
        writer.add_scalars(
            "stats/actions",
            {f"d{i}_mean": actions_scaled_mean[i].item() for i in range(len(actions_scaled_mean))},
            step,
        )
        writer.add_scalars(
            "stats/actions",
            {f"d{i}_max": actions_scaled_max[i].item() for i in range(len(actions_scaled_max))},
            step,
        )
        writer.add_scalars(
            "stats/actions",
            {f"d{i}_min": actions_scaled_min[i].item() for i in range(len(actions_scaled_min))},
            step,
        )


# 追加: ベクタ環境用のファクトリ
def make_env(seed_offset):
    def _thunk():
        env = gym.make("Hopper-v5")
        env.reset(seed=seed_offset)
        env = gym.wrappers.RecordEpisodeStatistics(env, 50)
        return env

    return _thunk


def calc_reward(env):
    # list(recent_returns)[-50:]
    queues = [subenv.return_queue for subenv in env.envs]  # 各 deque
    avg_latest = np.mean([np.mean(q) for q in queues if len(q) > 0])
    return avg_latest


def train():
    n_envs = 8
    env = gym.vector.SyncVectorEnv([make_env(seed_offset=i) for i in range(n_envs)])

    total_num_episodes = int(2e4)  # Total number of episodes
    obs_space_dims = env.single_observation_space.shape[0]
    action_space_dims = env.single_action_space.shape[0]
    T = 512
    steps_per_update = T * n_envs

    # for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
    best_reward = 0
    for seed in [1]:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        agent = Agent(obs_space_dims, action_space_dims)
        collected = 0

        for episode in range(total_num_episodes):
            states, _ = env.reset()
            for _ in range(T):
                actions, probs, state_vals = agent.sample_action(states)
                actions_scaled = torch.tanh(actions) * ACTION_SCALE
                actions_scaled = actions_scaled.numpy()
                n_states, rewards, terms, truncs, infos = env.step(actions_scaled)

                n_state_for_V = n_states.copy()
                finals = infos.get("final_observation", [None] * n_envs)
                for i in range(n_envs):
                    if truncs[i] and finals[i] is not None:
                        n_state_for_V[i] = finals[i]  # 時間切れは最終観測で価値を引き継ぐ

                # バッファへ追加
                agent.buffer["state"].extend(states.tolist())
                agent.buffer["state_val"].extend(state_vals.tolist())
                agent.buffer["action"].extend(actions.tolist())
                agent.buffer["prob"].extend(probs.tolist())
                agent.buffer["reward"].extend(rewards.tolist())
                agent.buffer["n_state"].extend(n_state_for_V.tolist())
                agent.buffer["term"].extend(terms.tolist())
                agent.buffer["trunc"].extend(truncs.tolist())

                collected += n_envs
                # 更新
                if len(agent.buffer["state"]) >= steps_per_update:
                    agent.update(n_envs, collected)
                # logging
                avg_reward = calc_reward(env)
                writer.add_scalar("stats/reward", avg_reward, collected)
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    print(f"[Eval] {collected=}, best_reward={avg_reward:.1f}")

                dones = terms | truncs
                if np.any(dones):
                    for i, done in enumerate(dones):
                        if done:
                            new_state, _ = env.envs[i].reset()  # その環境だけ更新
                            n_states[i] = new_state
                states = n_states

            if episode % 100 == 0:
                avg_latest = calc_reward(env)
                print("Episode:", episode, "Average Reward:", avg_latest)


if __name__ == "__main__":
    train()
