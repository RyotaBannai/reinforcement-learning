import numpy as np
import gymnasium as gym
import ipdb
from gymnasium import spaces
from gymnasium.wrappers import FrameStackObservation, AtariPreprocessing

# from gymnasium.wrappers.atari import ClipRewardEnv, EpisodicLifeEnv
from gymnasium.wrappers import TransformReward
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical
import ale_py
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter("runs/ac_exp10_GPU")

gym.register_envs(ale_py)


def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.constant_(m.weight, 0)
        torch.nn.init.constant_(m.bias, 0)


class NatureCNN(nn.Module):
    def __init__(self, in_ch=4, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=8, stride=4),
            nn.GroupNorm(8, 32),  # 8グループ (各グループ4ch)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.GroupNorm(16, 64),  # 16グループ (各4ch)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        return self.net(x.float())


class ActorCritic(nn.Module):
    def __init__(self, channels: int, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.features_dim = 512
        self.body = NatureCNN(channels, self.features_dim)
        self.net = nn.Linear(self.features_dim, action_dim + 1)
        self.softmax = nn.Softmax(dim=1)
        self.apply(orthogonal_init)

    def forward(self, x):
        z = self.body(x.float())
        policy, value = torch.split(
            self.net(z),
            [self.action_dim, 1],
            dim=1,
        )
        policy = self.softmax(policy)
        return policy, value


class Agent:
    def __init__(self, channels: int, action_dim: int, device: torch.device):
        # Hyperparameters
        self.lr = 2.5e-4
        self.gamma = 0.99
        self.lamnda = 0.995
        self.eps = 1e-6
        self.eps_clip = 0.2
        self.K_epochs = 10
        self.ent_coef = 0.02
        self.action_dim = action_dim
        self.device = device
        # Init policy
        self.policy = ActorCritic(channels, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.policy_old = ActorCritic(channels, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.init_buffer()

    def init_buffer(self):
        self.buffer = dict(
            action=[],
            prob=[],
            state=[],
            state_val=[],
            reward=[],
            n_state=[],
            term=[],
            trunc=[],
        )

    def sample_action(self, state: np.ndarray):
        state = torch.tensor(state, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            prob, state_val = self.policy_old(state)
            prob = prob.squeeze(0)
            dist = Categorical(probs=prob)
            action = dist.sample().item()

        return action, prob.detach().cpu(), state_val.squeeze(0).item()

    def update(self, step: int):
        OLD_State = torch.tensor(np.array(self.buffer["state"]), dtype=torch.float32, device=self.device)
        OLD_V = torch.tensor(np.array(self.buffer["state_val"]), dtype=torch.float32, device=self.device)
        OLD_N_state = torch.tensor(np.array(self.buffer["n_state"]), dtype=torch.float32, device=self.device)
        OLD_Action = torch.tensor(self.buffer["action"], dtype=torch.long, device=self.device)
        OLD_Prob = torch.stack(self.buffer["prob"]).to(self.device)  # CPU保存した場合は to(device)
        Reward = torch.tensor(self.buffer["reward"], dtype=torch.float32, device=self.device).unsqueeze(1)
        Term = torch.tensor(self.buffer["term"], dtype=torch.float32, device=self.device).unsqueeze(1)
        Trunc = torch.tensor(self.buffer["trunc"], dtype=torch.float32, device=self.device).unsqueeze(1)

        B = OLD_Prob.shape[0]

        # GAE、価値ターゲット
        with torch.no_grad():
            _, _NV = self.policy_old(OLD_N_state)
            m_boot = 1.0 - Term
            m_carry = 1.0 - torch.clamp(Term + Trunc, max=1.0)
            NV = _NV * m_boot
            OLD_V = OLD_V.unsqueeze(-1)
            delta = Reward + self.gamma * NV - OLD_V
            adv = torch.zeros_like(delta)
            gae = 0
            for t in reversed(range(len(delta))):
                gae = delta[t] + self.gamma * self.lamnda * m_carry[t] * gae
                adv[t] = gae
            target = adv + OLD_V  # λリターン
            adv = adv.clamp_(-5.0, 5.0)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            prob, vs = self.policy(OLD_State)
            dist = Categorical(probs=prob)
            ent = dist.entropy().mean()
            ratios = prob[torch.arange(B), OLD_Action] / (OLD_Prob[torch.arange(B), OLD_Action] + 1e-8)
            ratios = ratios.unsqueeze(-1)
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            loss = (-(torch.min(surr1, surr2))).mean() + 0.5 * self.MseLoss(vs, target) - 0.02 * ent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.init_buffer()

        # logging
        writer.add_scalar("loss/policy+value", loss.item(), step)


class SimplifiedAmidar(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(4)
        self.mapping = [2, 3, 4, 5]

    # 元のALEアクション番号に対応
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(self.mapping[action])
        return obs, reward, term, trunc, info


def make_env(
    seed: int | None = None,
    render: bool = False,
):
    env = gym.make(
        "ALE/Amidar-v5",
        obs_type="grayscale",
        frameskip=1,
        render_mode="human" if render else None,
    )
    if seed is not None:
        env.reset(seed=seed)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)
    env = FrameStackObservation(env, stack_size=4)
    env = SimplifiedAmidar(env)
    return env


def obs_change_metric(prev_obs, obs):
    # 前処理後の観測 (84x84 か 4x84x84 等) を想定
    a = np.array(prev_obs, dtype=np.float32, copy=True).ravel()
    b = np.array(obs, dtype=np.float32, copy=True).ravel()
    return float(np.mean(np.abs(a - b)))


def train():
    seed = 1
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device={device}")
    env = make_env(seed)
    total_num_episodes = int(2e4)
    channels = env.observation_space.shape[0]
    action_space_dims = env.action_space.n
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    agent = Agent(channels, action_space_dims, device)
    collected = 0
    best_reward = 0
    for episode in range(total_num_episodes):
        print(f"new episode: {episode}")
        state, info = env.reset()
        num = 11
        left = 3
        cum_reward = 0
        game_total_reward = 0
        while left != 0:
            action, prob, state_val = agent.sample_action(state)
            n_state, reward, term, trunc, info = env.step(action)
            reward = int(reward)
            diff = obs_change_metric(state, n_state)
            if diff == 0:
                state = n_state
                continue
            # print(f"diff={diff:.6f}") # 確認重要。

            # logging
            # a = action.item()
            meanings = ["UP", "RIGHT", "LEFT", "DOWN"]
            # log = f"A:{action.item()}({meanings[a]}), {term=}, "
            # if reward != 0:
            #     log += f"{reward=}"
            # print(log)  # 確認重要。
            # print(f"{info=}")

            writer.add_scalars(
                "stats/action_logits",
                {f"{meanings[i]}": prob[i].item() for i in range(len(prob))},
                collected,
            )

            cum_reward += reward
            game_total_reward += reward
            live = info["lives"]
            if num != live:
                num = live
                left -= 1
                term = True
                reward -= 5
                print(f"{cum_reward=}")
                if game_total_reward > best_reward:
                    best_reward = game_total_reward
                    print(f"[Eval] best_reward={best_reward:.1f}")
                    torch.save(agent.policy.state_dict(), "best_policy.pth")
                cum_reward = 0
                if left == 0:
                    game_total_reward = 0

            # バッファへ追加
            agent.buffer["state"].append(np.array(state, copy=False))
            agent.buffer["state_val"].append(float(state_val))
            agent.buffer["action"].append(int(action))
            agent.buffer["prob"].append(prob)
            agent.buffer["reward"].append(reward)
            agent.buffer["n_state"].append(np.array(n_state, copy=False))
            agent.buffer["term"].append(term)
            agent.buffer["trunc"].append(trunc)
            state = n_state

            collected += 1

        if len(agent.buffer["state"]) >= 8192:
            print(f"learning... data_len:{len(agent.buffer['state'])}")
            agent.update(collected)

        print("resetting env...")


if __name__ == "__main__":
    train()
