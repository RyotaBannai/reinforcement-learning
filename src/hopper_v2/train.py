import os, sys
import numpy as np
import gymnasium as gym
import ipdb
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

ACTION_SCALE = 1.0
LOG_SIG_MAX = 2
LOG_SIG_MIN = -2


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Policy_Network(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.hidden_space1 = 256
        self.hidden_space2 = 256
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, self.hidden_space1),
            nn.ReLU(),
            nn.Linear(self.hidden_space1, self.hidden_space2),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(self.hidden_space2, action_dim)
        self.log_std_head = nn.Linear(self.hidden_space2, action_dim)

        self.apply(weights_init_)

    def forward(self, x):
        shared_features = self.shared_net(x.float())
        mu = self.mean_head(shared_features)
        log_std = self.log_std_head(shared_features).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        std = log_std.exp()
        return mu, std


class Value_Network(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.hidden_space1 = 256
        self.hidden_space2 = 256
        self.net = nn.Sequential(
            nn.Linear(state_dim, self.hidden_space1),
            nn.ReLU(),
            nn.Linear(self.hidden_space1, self.hidden_space2),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(self.hidden_space2, 1)

        self.apply(weights_init_)

    def forward(self, x):
        features = self.net(x)
        val = self.mean_head(features)
        return val


class ActorCritic:
    """ActorCritic algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        # Hyperparameters
        self.learning_rate = 3e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.lamnda = 0.95
        self.eps = 1e-6  # small number for mathematical stability

        self.actor = Policy_Network(obs_space_dims, action_space_dims)
        self.critic = Value_Network(obs_space_dims)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        self.init_buffer()

    def init_buffer(self):
        self.buffer = dict(action=[], prob=[], state=[], reward=[], n_state=[], term=[])

    def sample_action(self, state: np.ndarray):
        state = torch.tensor(state)
        if state.ndim == 1:
            state = state.unsqueeze(0)  # [N, obs]
        mu, std = self.actor(state)
        distrib = Normal(mu, std + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)
        two = torch.tensor(2.0, device=prob.device, dtype=prob.dtype)
        corr = 2.0 * (torch.log(two) - action - F.softplus(-2.0 * action))
        prob -= corr
        action = torch.tanh(action) * ACTION_SCALE
        action = action.numpy()
        return action, prob

    def update(self, n_envs: int):
        State = torch.tensor(np.array(self.buffer["state"]), dtype=torch.float32)
        N_state = torch.tensor(np.array(self.buffer["n_state"]), dtype=torch.float32)
        Prob = torch.stack(self.buffer["prob"]).sum(dim=-1).unsqueeze(1)
        Reward = torch.tensor(self.buffer["reward"], dtype=torch.float32).unsqueeze(1)
        Term = torch.tensor(self.buffer["term"], dtype=torch.float32).unsqueeze(1)

        B = State.shape[0]
        assert B % n_envs == 0, "収集数が n_envs で割り切れません"
        T = B // n_envs

        # ----- 形状を [T, n_envs, …] に並べ替え（time-major）-----
        def time_env(x):
            # x: [B, *] -> [T, n_envs, *]
            return x.view(T, n_envs, *x.shape[1:])

        n_Reward = time_env(Reward)
        n_Term = time_env(Term)
        n_Prob = time_env(Prob)

        # ----- GAE & 価値ターゲット -----
        with torch.no_grad():
            n_V = time_env(self.critic(State))  # [T, n_envs, 1]
            n_NV = time_env(self.critic(N_state))  # [T, n_envs, 1]
            not_done = 1.0 - n_Term
            delta = n_Reward + self.gamma * n_NV * not_done - n_V

            adv = torch.zeros_like(delta)
            gae: torch.Tensor = torch.zeros((n_envs, 1), dtype=delta.dtype)
            for t in reversed(range(T)):
                gae = delta[t] + self.gamma * self.lamnda * not_done[t] * gae
                adv[t] = gae
            target = adv + n_V  # λリターン
            # 標準化＋外れ値クリップ：安定化
            if B > 1:  # 要素数 1 の正規化回避
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv = adv.clamp_(-15.0, 15.0)

        vs = self.critic(State)
        target = target.reshape(B, 1)
        v_loss = F.smooth_l1_loss(vs, target)

        # State: [B, obs], B=T*n_envs
        mu, std = self.actor(State)
        dist = Normal(mu, std.clamp_min(1e-6))
        ent = dist.entropy().sum(dim=-1, keepdim=True)  # [B,1]
        ent_t = ent.view(T, n_envs, 1)  # [T, n_envs, 1]
        entropy_coef = 0.02
        policy_loss = (-(n_Prob * adv) - entropy_coef * ent_t).mean()

        # Update the policy network
        self.optimizer_critic.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.optimizer_critic.step()
        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.optimizer_actor.step()

        # Empty / zero out all episode-centric/related variables
        self.init_buffer()


# 追加: ベクタ環境用のファクトリ
def make_env(seed_offset=0):
    def _thunk():
        env = gym.make("Hopper-v5")
        env.reset(seed=seed_offset)  # 各envに違うseedを与える
        env = gym.wrappers.Autoreset(env)
        env = gym.wrappers.RecordEpisodeStatistics(env, 50)
        return env

    return _thunk


def train():
    n_envs = 8
    env = gym.vector.SyncVectorEnv([make_env(seed_offset=i) for i in range(n_envs)])

    total_num_episodes = int(2e4)  # Total number of episodes
    obs_space_dims = env.single_observation_space.shape[0]
    action_space_dims = env.single_action_space.shape[0]
    rewards_over_seeds = []
    T = 128
    steps_per_update = T * n_envs

    # for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
    for seed in [1]:
        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        agent = ActorCritic(obs_space_dims, action_space_dims)
        collected = 0

        for episode in range(total_num_episodes):
            states, _ = env.reset()
            for _ in range(T):
                actions, probs = agent.sample_action(states)
                n_states, rewards, terms, _, _ = env.step(actions)
                # バッファへ追加
                agent.buffer["state"].extend(states.tolist())  # (n_envs, obs_dim)
                agent.buffer["action"].extend(actions.tolist())  # (n_envs, act_dim) ←今回は未使用でもOK
                # logpはtensorなので分解して保存
                agent.buffer["prob"].extend([lp for lp in probs])  # list of tensors（後でstackする）
                agent.buffer["reward"].extend(rewards.tolist())  # (n_envs,)
                agent.buffer["n_state"].extend(n_states.tolist())  # (n_envs, obs_dim)
                agent.buffer["term"].extend(terms.tolist())  # (n_envs,)

                collected += n_envs
                states = n_states

                if len(agent.buffer["state"]) >= steps_per_update:
                    agent.update(n_envs)

            if episode % 100 == 0:
                queues = [subenv.return_queue for subenv in env.envs]  # 各 deque
                avg_latest = np.mean([np.mean(q) for q in queues if len(q) > 0])
                print("Episode:", episode, "Average Reward:", avg_latest)

    df1 = pd.DataFrame(rewards_over_seeds).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(title="REINFORCE for Hopper-v5")
    plt.show()


if __name__ == "__main__":
    train()
