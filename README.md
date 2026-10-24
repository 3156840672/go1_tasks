# go1-tasks

本项目将两篇四足机器人强化学习论文的核心方法移植到 MuJoCo/mjlab 的
Unitree Go1 任务中，并提供平地、崎岖地形速度跟踪基线。运行所需的 `mjlab`
框架随仓库打包；克隆后无需额外下载源码即可安装、训练与播放。包含四个任务：

| key | 包内任务 ID | 说明 |
| --- | --- | --- |
| `rough` | `Go1-Velocity-Rough` | 崎岖地形速度跟踪基线 |
| `flat` | `Go1-Velocity-Flat` | 平地速度跟踪基线 |
| `mob` | `Go1-Velocity-MoB` | Walk These Ways 启发的多行为步态策略 |
| `cts` | `Go1-Velocity-CTS` | CTS 启发的并发师生策略 |

## 论文与复现范围

| 论文 | 对应任务 | 本仓库实现的核心方法 |
| --- | --- | --- |
| Margolis & Agrawal, [*Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior* (2023)](https://proceedings.mlr.press/v205/margolis23a.html) | `mob` | 用行为命令调节步态、姿态和摆腿参数，并训练条件化速度跟踪策略。 |
| Wang et al., [*CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion* (2024)](https://arxiv.org/abs/2405.10830) | `cts` | 在同一 PPO 训练过程中并发训练特权信息教师和本体感觉历史学生。 |

`rough` 和 `flat` 是本项目的 Go1 速度跟踪基线，不单独声称复现某篇论文。
`mob` 与 `cts` 复现的是上述论文的**核心思路**，并非逐项复现原作者的仿真器、
全部网络与奖励细节、训练设置、实验指标或真机结果。本仓库使用 MuJoCo/mjlab
及自己的 Go1 配置，需要重新训练和评估，不能直接沿用论文报告的性能数字。
具体实现、与论文的差异及使用方法见 [`docs/mob.md`](docs/mob.md) 和
[`docs/cts.md`](docs/cts.md)。

项目结构：

- `src/go1_tasks/`：Go1 任务定义（环境配置、CTS、MoB、奖励与课程）。
- `framework/mjlab/`：vendored 的 `mjlab` 框架（已移除其内置的 Go1 任务，
  只保留框架与资产），以 uv workspace 成员的形式随本项目一起安装。
- `vendor/mujoco_warp/`：vendored 的 `mujoco-warp` 源码，安装时不依赖
  GitHub 拉取。
- `docs/`、`tests/`：文档与测试。

## 克隆与安装

需要 [uv](https://docs.astral.sh/uv/) 与 Python 3.10–3.13。
将下方的 `OWNER` 替换为实际的 GitHub 用户名或组织名。

```bash
git clone https://github.com/OWNER/go1-tasks.git go1-tasks
cd go1-tasks
uv sync --locked
uv run go1-list-tasks
```

`uv sync` 会创建 `.venv`、解析并安装全部依赖，并把 `mjlab` 与 `go1_tasks`
两个包以可编辑方式安装进去。Linux 默认使用 CUDA 12.8 的 torch；若只需 CPU，
把 `pyproject.toml` 里的

```toml
torch = { index = "pytorch-cu128", marker = "sys_platform != 'darwin'" }
```

改为 `pytorch-cpu` 后重新 `uv sync`。

## 任务 ID

本项目同时注册两套 ID：

- 历史 ID（`Mjlab-Velocity-Rough/Flat/MoB/CTS-Unitree-Go1`）——由本包注册；
- 包内 ID（`Go1-Velocity-Rough/Flat/MoB/CTS`）——始终可用。

两套 ID 指向同一份配置。查看当前生效的 ID：

```bash
uv run go1-list-tasks
```

checkpoint 目录保持不变（`logs/rsl_rl/go1_velocity`、`go1_mob`、`go1_cts`），
已有 checkpoint 可以继续加载。

## 训练与播放

```bash
uv run go1-train Go1-Velocity-Rough --env.scene.num-envs 1024
uv run go1-play Go1-Velocity-Rough \
  --checkpoint-file logs/rsl_rl/go1_velocity/YOUR_RUN/model_1000.pt \
  --viewer viser
```

框架自带的 `train` / `play` 等命令也一并安装，并通过 `mjlab.tasks` 入口点
自动加载本包，因此也能直接使用：

```bash
uv run train Mjlab-Velocity-Rough-Unitree-Go1 --env.scene.num-envs 1024
```

论文相关任务的详细说明见 [`docs/cts.md`](docs/cts.md) 与
[`docs/mob.md`](docs/mob.md)。

## CTS 评估

```bash
uv run go1-eval-cts /path/to/model_5000.pt --vx 0.5 --seconds 5
```

`--branch teacher` 可以区分策略整体误差与学生编码器误差，`--randomized`
用于带域随机化的对比评估。

## 测试

```bash
uv run pytest
```

若系统 ROS 的 pytest 插件自动加载并报缺少 `lark`，可运行
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest`。

`tests/test_go1_constants.py` 校验 Go1 资产参数，`tests/test_go1_cts.py` 与
`tests/test_go1_mob.py` 校验任务配置与算法契约，
`tests/test_training_logging.py` 校验训练日志依赖的兼容性。

## sim2sim

独立的 MuJoCo sim2sim 部署项目（读取训练 checkpoint、不导入本仓库）单独
维护，训练完成后按其 README 运行即可，不在本仓库内。

## 许可与第三方代码

本项目自有代码采用 Apache-2.0 许可，见 [`LICENSE`](LICENSE)。随仓库
分发的第三方代码保留其原有许可：

- `framework/mjlab/` 为 [mjlab](https://github.com/mujocolab/mjlab) 的副本，
  主体采用 Apache-2.0；其中 3 个源文件保留了 Isaac Lab 的 BSD-3-Clause
  声明，许可文本见 [mjlab Apache 许可](framework/mjlab/LICENSE) 与
  [Isaac Lab BSD 许可](framework/mjlab/LICENSES/BSD-3-Clause.txt)；
- `vendor/mujoco_warp/` 为
  [mujoco_warp](https://github.com/google-deepmind/mujoco_warp) 的副本，
  Apache-2.0，许可见 `vendor/mujoco_warp/LICENSE`。

再分发时请保留相应的版权声明和许可文件。机器人模型资产的来源与授权还应
在公开发布前单独核对。
