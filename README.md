・概要

強化学習のコード

・内容・構成

src

- hopper：
  - 課題：[MuJoCo　Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/)
  - モデル：PPOのゼロ実装（Actor-critic）
- hopper_v2：
  - 課題：[MuJoCo　Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/)
  - モデル：PPOのゼロ実装（Actor-critic）
  - hopper/の実装内容とほとんど変わらないが、こっちの方が安定していて報酬も倍近い。
- hopper_rilib：
  - 課題：同上
  - モデル：RLlibを使ったhopperの学習
- pendulum：
  - 課題：[Classic Control　Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/)
  - モデル：PPOのゼロ実装（Actor-critic）

・実行/再現方法
```shell
poetry install # 初回のみ
cd src/{各課題フォルダ}
poetry run python train.py # 学習させる
poetry run python use_trained_model.py # 学習済みモデルを実行
```
