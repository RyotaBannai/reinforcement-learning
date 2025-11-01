import torch
import gymnasium as gym
from train import ActorCritic, make_env
from torch.distributions import Categorical
import ipdb
import ale_py

gym.register_envs(ale_py)


# =========================
#  推 論 ル ー プ
# =========================
def run_eval(actor_ckpt, episodes):
    env = make_env(render=True)
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.n
    # モデル構築＆ロード
    policy = ActorCritic(obs_space_dims, action_space_dims)
    policy.load_state_dict(torch.load(actor_ckpt))
    policy.eval()
    for ep in range(episodes):
        state, _ = env.reset(seed=ep + 100)
        cum_reward = 0.0
        num = 11
        left = 3
        while left != 0:
            with torch.no_grad():
                st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits, *_ = policy(st)
                print(logits)
                # probablistic
                # dist = Categorical(logits=logits)
                # action = dist.sample().item()
                # deterministic
                action = logits.argmax().item()
            state, reward, *_, info = env.step(action)
            cum_reward += reward
            live = info["lives"]
            if num != live:
                num = live
                left -= 1
                print(f"{cum_reward=}")
                cum_reward = 0
    env.close()


if __name__ == "__main__":
    run_eval(
        actor_ckpt="best_policy.pth",
        episodes=10,
    )
