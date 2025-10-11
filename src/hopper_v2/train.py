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
LOG_SIG_MIN = -20


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
        self.std_head = nn.Linear(self.hidden_space2, action_dim)

        self.apply(weights_init_)

    def forward(self, x):
        shared_features = self.shared_net(x.float())
        mu = self.mean_head(shared_features)
        std = F.softplus(self.std_head(shared_features))
        std = torch.clamp(std, min=np.exp(LOG_SIG_MIN), max=np.exp(LOG_SIG_MAX))
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


class REINFORCE_baseline:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards
        self.obs = []  # Stores observations

        self.actor = Policy_Network(obs_space_dims, action_space_dims)
        self.critic = Value_Network(obs_space_dims)
        self.optimizer_actor = torch.optim.AdamW(self.actor.parameters(), lr=self.learning_rate)
        self.optimizer_critic = torch.optim.AdamW(self.critic.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.actor(state)
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)
        two = torch.tensor(2.0, device=prob.device, dtype=prob.dtype)
        corr = 2.0 * (torch.log(two) - action - F.softplus(-2.0 * action))
        prob -= corr
        action = torch.tanh(action) * ACTION_SCALE
        action = action.numpy()
        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)
        deltas = gs

        obs = torch.tensor(np.array(self.obs), dtype=torch.float)
        vs = self.critic(obs)
        v_loss = F.smooth_l1_loss(vs, torch.tensor(deltas).unsqueeze(1))

        v_baseline = vs.detach()
        deltas = torch.tensor(np.array(deltas)).unsqueeze(1)
        adv = deltas - v_baseline
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # 正規化
        log_probs = torch.stack(self.probs).squeeze().sum(dim=-1).unsqueeze(1)  # アクション次元で先に合計
        policy_loss = (-log_probs * adv).sum()

        # Update the policy network
        self.optimizer_critic.zero_grad()
        v_loss.backward()
        self.optimizer_critic.step()
        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        self.optimizer_actor.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []
        self.obs = []


def train():
    # Create and wrap the environment
    env = gym.make("Hopper-v5")
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

    total_num_episodes = int(2e4)  # Total number of episodes
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]
    rewards_over_seeds = []

    for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Reinitialize agent every seed
        agent = REINFORCE_baseline(obs_space_dims, action_space_dims)
        reward_over_episodes = []

        for episode in range(total_num_episodes):
            # gymnasium v26 requires users to set seed while resetting the environment
            obs, _ = wrapped_env.reset(seed=seed)

            done = False
            while not done:
                agent.obs.append(obs)
                action = agent.sample_action(obs)
                obs, reward, term, trunc, _ = wrapped_env.step(action)
                agent.rewards.append(reward)
                done = term or trunc

            reward_over_episodes.append(wrapped_env.return_queue[-1])
            agent.update()

            if episode % 1000 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print("Episode:", episode, "Average Reward:", avg_reward)

        rewards_over_seeds.append(reward_over_episodes)

    df1 = pd.DataFrame(rewards_over_seeds).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(title="REINFORCE for Hopper-v5")
    plt.show()


if __name__ == "__main__":
    train()
