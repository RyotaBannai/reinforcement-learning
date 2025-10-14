import numpy as np
import torch
import gymnasium as gym
from train import ACTION_SCALE, ActorCritic
import ipdb


# =========================
#  推 論 ル ー プ
# =========================
def run_eval(actor_ckpt="best_policy.pth", episodes=5, render=False):
    env = gym.make("Hopper-v5", render_mode="human" if render else None)
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]
    # モデル構築＆ロード
    policy = ActorCritic(obs_space_dims, action_space_dims, 0.1)
    policy.load_state_dict(torch.load(actor_ckpt))
    policy.eval()
    returns = []
    for ep in range(episodes):
        s, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            s_in = np.asarray(s).ravel().astype(np.float32)
            with torch.no_grad():
                st = torch.tensor(s_in, dtype=torch.float32).unsqueeze(0)
                mu, *_ = policy(st)
                a = torch.tanh(mu) * ACTION_SCALE  # 決定論（平均）で行動（dist.sampleしない）
                a_np = a.squeeze(0).cpu().numpy()

            s, r, term, trunc, _ = env.step(a_np)
            total += r
            done = term or trunc
            if done:
                break

        returns.append(total)
        print(f"[Eval] Episode {ep}: return = {total:.1f}")

    print(f"Average return over {episodes} episodes: {np.mean(returns):.1f}")
    env.close()


if __name__ == "__main__":
    run_eval(
        actor_ckpt="best_policy.pth",
        episodes=10,
        render=True,
    )
