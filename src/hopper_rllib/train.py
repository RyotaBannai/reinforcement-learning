import os, numpy as np, gymnasium as gym, ray
from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,  # これが旧 timesteps_total に相当
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)


def main():
    ray.init(ignore_reinit_error=True)

    config = (
        PPOConfig()
        .environment(env="Hopper-v5")
        .framework("torch")
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=4,
            # seed=0,
        )
        .training(
            gamma=0.99,
            lr=2e-4,
            train_batch_size=16384,
            minibatch_size=64,
            num_epochs=20,
            num_sgd_iter=10,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            use_gae=True,
            entropy_coeff=0.01,
            model={"fcnet_hiddens": [128, 64], "fcnet_activation": "tanh"},
        )
        .learners(
            num_learners=4,
            num_gpus_per_learner=0,  # GPU使うなら1,使わないなら0
        )
        .evaluation(
            evaluation_interval=5,  # ← 評価の頻度
            evaluation_duration=5,  # ← 何エピソード or 何ステップ評価するか
            evaluation_duration_unit="episodes",  # ← "timesteps" も可
            evaluation_config={
                "explore": False,  # 決定論的評価
                "render_env": False,
            },
        )
    )

    algo = config.build()

    best = -np.inf

    save_dir = os.path.abspath("checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    path = algo.save(save_dir)

    for it in range(1000):
        result = algo.train()
        mean_ret = result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
        total_sampled = result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_SAMPLED_LIFETIME]
        print(f"[Iter {it}] reward_mean={mean_ret:.1f}  ts_total={total_sampled}")
        if mean_ret > best:
            best = mean_ret
            ret_obj = algo.save(save_dir)
            print(f"✅ New best ({best:.1f}) saved to: {save_dir}")

    # 簡易評価（デモ）
    env = gym.make("Hopper-v5")
    policy = algo.get_policy()

    def run_episode():
        s, _ = env.reset()
        done, total = False, 0.0
        while not done:
            a, _, _ = policy.compute_single_action(s, explore=False)
            s, r, term, trunc, _ = env.step(a.astype(np.float32))
            total += r
            done = term or trunc
        return total

    print("[Final Eval]", np.mean([run_episode() for _ in range(5)]))
    env.close()
    ray.shutdown()


if __name__ == "__main__":
    main()
