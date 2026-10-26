# go1-tasks

本项目将三篇四足机器人强化学习论文的核心方法移植到 MuJoCo/mjlab 的
Unitree Go1 任务中，并提供平地、崎岖地形速度跟踪基线。运行所需的 `mjlab`
框架随仓库打包；克隆后无需额外下载源码即可安装、训练与播放。包含四个注册
任务和一套 RMA 两阶段训练流程：

| key | 包内任务 ID | 说明 |
| --- | --- | --- |
| `rough` | `Go1-Velocity-Rough` | 崎岖地形速度跟踪基线 |
| `flat` | `Go1-Velocity-Flat` | 平地速度跟踪基线 |
| `mob` | `Go1-Velocity-MoB` | Walk These Ways 启发的多行为步态策略 |
| `cts` | `Go1-Velocity-CTS` | CTS 启发的并发师生策略 |

RMA 使用独立的 `go1-train-rma` 命令训练 teacher 和 adaptation module，
不注册为普通 PPO 任务 ID。

## 论文与复现范围

| 论文 | 对应任务 | 本仓库实现的核心方法 |
| --- | --- | --- |
| Kumar et al., [*RMA: Rapid Motor Adaptation for Legged Robots* (2021)](https://www.roboticsproceedings.org/rss17/p011.html) | `rma` | 先用特权环境参数训练 teacher，再从本体感觉历史蒸馏 adaptation module。 |
| Margolis & Agrawal, [*Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior* (2023)](https://proceedings.mlr.press/v205/margolis23a.html) | `mob` | 用行为命令调节步态、姿态和摆腿参数，并训练条件化速度跟踪策略。 |
| Wang et al., [*CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion* (2024)](https://arxiv.org/abs/2405.10830) | `cts` | 在同一 PPO 训练过程中并发训练特权信息教师和本体感觉历史学生。 |

`rough` 和 `flat` 是本项目的 Go1 速度跟踪基线，不单独声称复现某篇论文。
`rma`、`mob` 与 `cts` 复现的是上述论文的**核心思路**，并非逐项复现原作者的仿真器、
全部网络与奖励细节、训练设置、实验指标或真机结果。本仓库使用 MuJoCo/mjlab
及自己的 Go1 配置，需要重新训练和评估，不能直接沿用论文报告的性能数字。
具体实现、与论文的差异及使用方法见 [`docs/rma.md`](docs/rma.md)、
[`docs/mob.md`](docs/mob.md) 和 [`docs/cts.md`](docs/cts.md)。

项目结构：

- `src/go1_tasks/`：Go1 任务定义（环境配置、RMA、CTS、MoB、奖励与课程）。
- `framework/mjlab/`：vendored 的 `mjlab` 框架（已移除其内置的 Go1 任务，
  只保留框架与资产），以 uv workspace 成员的形式随本项目一起安装。
- `vendor/mujoco_warp/`：vendored 的 `mujoco-warp` 源码，安装时不依赖
  GitHub 拉取。
- `sim2sim/`：独立 MuJoCo 推理场景、Go1 模型、地形资源与部署测试。
- `docs/`、`tests/`：文档与训练任务测试。

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

checkpoint 目录保持不变（`logs/rsl_rl/go1_velocity`、`go1_mob`、`go1_cts`、
`go1_rma`），已有 checkpoint 可以继续加载。

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

论文相关任务的详细说明见 [`docs/rma.md`](docs/rma.md)、
[`docs/cts.md`](docs/cts.md) 与 [`docs/mob.md`](docs/mob.md)。

## RMA 两阶段训练

RMA 先训练能读取仿真特权参数的 teacher，再冻结 teacher，用最近 50 帧
本体感觉观测训练 adaptation module。teacher 的策略输入由 48 维本体感觉和
8 维环境 latent 组成，输出 12 维关节动作；adaptation module 根据历史观测
预测同一个 8 维 latent。

### 1. 快速检查

首次运行前先确认命令已安装：

```bash
uv sync --locked
uv run go1-train-rma --help
```

可以用单环境、单次迭代检查完整训练链路。以下命令仅用于确认程序能运行，
生成的 checkpoint 没有实际控制效果：

```bash
uv run go1-train-rma teacher \
  --num-envs 1 \
  --iterations 1 \
  --steps 2 \
  --output /tmp/go1-rma-smoke

uv run go1-train-rma adaptation \
  --checkpoint /tmp/go1-rma-smoke/model_0.pt \
  --num-envs 1 \
  --iterations 1 \
  --steps 2 \
  --output /tmp/go1-rma-smoke/adaptation
```

### 2. 训练 teacher

```bash
uv run go1-train-rma teacher \
  --num-envs 128 \
  --iterations 10000 \
  --steps 32
```

teacher 在崎岖地形上用 PPO 训练。18 维特权信息包括载荷质量、质心偏移、
12 个电机增益比例、足端摩擦和局部足端离地高度。默认输出目录为：

```text
logs/rsl_rl/go1_rma/YYYY-MM-DD_HH-MM-SS/
├── model_0.pt
├── model_50.pt
├── ...
└── model_*.pt
```

训练规模可按显存调整。显存不足时先减小 `--num-envs`；`--steps` 是每次 PPO
更新前每个环境收集的控制步数。默认设备是检测到 CUDA 时使用 `cuda:0`，
否则使用 CPU，也可显式传入 `--device cuda:1` 或 `--device cpu`。通过
`--output PATH` 可以指定 teacher 输出目录。

### 3. 训练 adaptation module

选择一个训练完成的 teacher checkpoint：

```bash
uv run go1-train-rma adaptation \
  --checkpoint logs/rsl_rl/go1_rma/YOUR_RUN/model_9999.pt \
  --num-envs 128 \
  --iterations 1000 \
  --steps 32
```

这一阶段冻结 teacher 策略和特权 encoder，使用 adaptation module 预测 teacher
的环境 latent，并用预测结果控制采样。默认每 50 次迭代保存一次，输出到所选
teacher 旁边的 `adaptation/` 目录：

```text
logs/rsl_rl/go1_rma/YOUR_RUN/
├── model_9999.pt
└── adaptation/
    ├── adapter_50.pt
    ├── adapter_100.pt
    └── adapter_1000.pt
```

adapter checkpoint 会记录对应 teacher 的路径、48 维观测约定和 50 帧历史
长度。`--checkpoint` 在 adaptation 阶段是必需参数；`--output PATH` 可覆盖
默认的 adaptation 输出目录。

### 4. 在 sim2sim 中验证

先进行无窗口测试：

```bash
uv run python sim2sim/go1_sim.py \
  --teacher logs/rsl_rl/go1_rma/YOUR_RUN/model_9999.pt \
  --adapter logs/rsl_rl/go1_rma/YOUR_RUN/adaptation/adapter_1000.pt \
  --terrain rough \
  --headless \
  --steps 2000 \
  --command 0.3 0 0
```

需要交互查看时去掉 `--headless` 和 `--steps`：

```bash
uv run python sim2sim/go1_sim.py \
  --teacher logs/rsl_rl/go1_rma/YOUR_RUN/model_9999.pt \
  --adapter logs/rsl_rl/go1_rma/YOUR_RUN/adaptation/adapter_1000.pt \
  --terrain rough
```

可选地形为 `flat`、`rough`、`stairs` 和 `rl_sar`。窗口中按 `1` 启用策略、
`0` 进入被动站立、`R` 重置；`W/S`、`A/D`、`Q/E` 分别控制前后、横向和
转向，空格清零速度命令。`--terrain-seed N` 可改变随机崎岖地形；没有手柄时
可传入 `--joystick ''`。

teacher 与 adapter 必须来自同一训练组合；加载器会检查 adapter 记录的 teacher
文件名以及观测和历史长度。短冒烟训练生成的 checkpoint 只用于检查接口，
不能用来判断步态质量，更不能直接用于真机。

### 参数速查

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `stage` | 必填 | `teacher` 或 `adaptation` |
| `--num-envs` | `128` | 并行仿真环境数量 |
| `--iterations` | teacher `10000`；adaptation `1000` | 训练迭代次数 |
| `--steps` | `32` | 每次迭代、每个环境采集的控制步数 |
| `--checkpoint` | 无 | adaptation 使用的 teacher checkpoint |
| `--output` | 自动生成 | 覆盖 checkpoint 输出目录 |
| `--device` | 自动选择 | 例如 `cuda:0`、`cuda:1` 或 `cpu` |

实现细节和与原论文的差异见 [`docs/rma.md`](docs/rma.md)。

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

`tests/test_go1_constants.py` 校验 Go1 资产参数，`tests/test_go1_rma.py`、
`tests/test_go1_cts.py` 与 `tests/test_go1_mob.py` 校验任务配置与算法契约，
`tests/test_training_logging.py` 校验训练日志依赖的兼容性。
sim2sim 的场景和推理接口可运行：

```bash
PYTHONPATH=sim2sim uv run python -m unittest discover -s sim2sim/tests
```

## sim2sim

`sim2sim/` 已包含在仓库中，使用独立的 MuJoCo 场景，不依赖训练时的
`mjlab` 环境对象。可先在无检查点时测试模型与地形：

```bash
uv run python sim2sim/go1_sim.py --headless --steps 100
```

本仓库训练的 Flat（48 维观测）和 MoB（1710 维观测）检查点有对应推理脚本；
当前 Rough（235 维观测）和 CTS 检查点不支持直接运行。由本仓库
`go1-train-rma` 生成的 RMA teacher/adapter 检查点可由 `go1_sim.py` 加载。
命令、控制方式和检查点约定见 [sim2sim/README.md](sim2sim/README.md)。

## 许可与第三方代码

本项目自有代码采用 Apache-2.0 许可，见 [`LICENSE`](LICENSE)。随仓库
分发的第三方代码保留其原有许可：

- `framework/mjlab/` 为 [mjlab](https://github.com/mujocolab/mjlab) 的副本，
  主体采用 Apache-2.0；其中 3 个源文件保留了 Isaac Lab 的 BSD-3-Clause
  声明，许可文本见 [mjlab Apache 许可](framework/mjlab/LICENSE) 与
  [Isaac Lab BSD 许可](framework/mjlab/LICENSES/BSD-3-Clause.txt)；
- `vendor/mujoco_warp/` 为
  [mujoco_warp](https://github.com/google-deepmind/mujoco_warp) 的副本，
  Apache-2.0，许可见 `vendor/mujoco_warp/LICENSE`；
- `sim2sim/terrain_assets/` 中的 rl_sar 场景资源保留
  [上游许可](sim2sim/terrain_assets/RL_SAR_LICENSE)。

再分发时请保留相应的版权声明和许可文件。机器人模型资产的来源与授权还应
在公开发布前单独核对。
