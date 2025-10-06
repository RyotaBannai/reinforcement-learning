# demo_hopper_rllib_newapi.py
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from ray.rllib.core.rl_module import RLModule


def run_demo(checkpoint_uri: str, episodes: int = 3, render: bool = False):
    # チェックポイントから RLModule（=方策ネット）を直接ロード
    ckpt = Path(checkpoint_uri)
    rl_module = RLModule.from_checkpoint(ckpt / "learner_group" / "learner" / "rl_module" / "default_policy")
    env = gym.make("Hopper-v5", render_mode="human" if render else None)

    def greedy_action_from_outputs(model_outputs):
        """
        連続アクション: action_dist_inputs は [mu, log_std] を返すのが既定。
        Hopper は3次元アクションなので、前半3つが平均（mu）。
        """
        params = model_outputs["action_dist_inputs"][0].detach().cpu().numpy()  # shape (2*act_dim,)
        act_dim = env.action_space.shape[0]  # = 3
        mu = params[:act_dim]  # 平均
        # 決定論行動（平均）をそのまま使い、環境の範囲にクリップ
        return np.clip(mu, env.action_space.low, env.action_space.high).astype(np.float32)

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            # 3) forward_inference でアクション分布パラメータを取得
            obs_batch = torch.from_numpy(obs).float().unsqueeze(0)  # [B=1, obs_dim]
            outs = rl_module.forward_inference({"obs": obs_batch})
            action = greedy_action_from_outputs(outs)

            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
            if render:
                env.render()

        returns.append(total)
        print(f"[Eval] Episode {ep}: return = {total:.1f}")

    print(f"Average return over {episodes} episodes: {np.mean(returns):.1f}")
    env.close()


if __name__ == "__main__":
    checkpoint_uri = "/Users/ryotabannai/dev/github.com/RyotaBannai/reinforcement-learning/src/hopper_rllib/checkpoints"
    run_demo(checkpoint_uri, episodes=5, render=True)
