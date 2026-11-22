# Go1 RMA 两阶段师生适应

RMA（Rapid Motor Adaptation）分两阶段：先训练一个能看到特权物理参数的教师（PPO + 环境
编码器），冻结教师后再训练一个只看 50 帧本体历史的适应模块，用它回归教师编码器输出的
8 维隐变量。它不是一个注册任务——没有 `Go1-Velocity-RMA` 这个 ID，只有 `go1-train-rma`
和 `sim2sim/go1_sim.py --teacher/--adapter`。和 `Go1-Velocity-CTS` 的区别是两阶段串行、
隐变量来自特权参数回归而不是与 PPO 联合训练；和 `Go1-Velocity-Rough` 的区别是部署时
不需要高度扫描。

## 任务 ID

| 项 | 值 |
| --- | --- |
| 任务 ID | 未注册；`go1-list-tasks` 里没有 RMA，只有 `go1-train-rma` 入口 |
| 环境配置 | `src/go1_tasks/rma_cfg.py::unitree_go1_rma_env_cfg`（派生自 `unitree_go1_rough_env_cfg`） |
| Runner | 阶段一 `MjlabOnPolicyRunner` + `go1_tasks.rma:RmaActor`；阶段二手写训练循环，无 runner |
| 检查点目录 | `logs/rsl_rl/go1_rma/<timestamp>/model_*.pt` 与 `<timestamp>/adaptation/adapter_*.pt` |

## 观测与动作

| 项 | 值 |
| --- | --- |
| 教师 actor | 48 维本体（rough 的 actor 去掉 height_scan）+ 18 维特权量；策略头实际输入 48 + 8 |
| 特权量 18 维 | trunk 质量增量 1（相对默认，kg）、trunk COM 偏移 3、12 个执行器增益比、足端滑动摩擦 1（绝对值）、足下地形高度最大值 1 |
| 教师网络 | 编码器 18 → 8（256, 128, ELU）；MLP 128-128-128，ELU |
| 学生历史 | 50 帧 × 48 维（由旧到新），复位或 done 时清零 |
| 适应模块 | 48 → 32 → 32（embed，ELU）→ Conv1d 8/stride 4 → Conv1d 5 → Conv1d 5（各带 ELU）→ Linear(96, 8) |
| 动作 | 12 维关节位置偏移；`GO1_ACTION_SCALE`：hip/thigh 0.3728、calf 0.2485；`Kp≈15.9/35.8`（rough 默认，未改） |
| 控制频率 | 50 Hz（物理步长 0.005 s，decimation 4） |
| episode | 20 s |
| 指令范围 | 沿用 rough：lin_vel_x/y ±1.0 m/s、ang_vel_z ±0.5 rad/s、heading ±π；重采样 3–8 s，含地形与速度课程 |

教师训练时额外加两个 startup 事件：trunk 附加质量 0–2 kg、PD 增益 0.9–1.1 倍；这两个量
正是特权向量里质量与增益的来源。

## 训练

```bash
# 阶段一：教师（输出目录默认 logs/rsl_rl/go1_rma/<timestamp>）
uv run go1-train-rma teacher --num-envs 128 --iterations 10000 --steps 32

# 阶段二：冻结教师，训练适应模块
uv run go1-train-rma adaptation \
  --checkpoint logs/rsl_rl/go1_rma/2026-10-28_13-38-32/model_9999.pt \
  --num-envs 128 --iterations 1000 --steps 32
```

阶段一超参：`hidden_dims=(128,128,128)`、ELU、lr 5e-4 adaptive、`num_steps_per_env` 由
`--steps` 覆盖为 32、默认 10000 轮、`save_interval=50`、`logger=tensorboard`、
`upload_model=False`。`--resume` 需要配 `--output` 指向已有 run 目录，程序取其中最新的
`model_*.pt` 继续。

阶段二是闭环的：每个仿真步用适应模块从 50 帧历史估出的隐变量驱动教师策略，
`history` 在 done 时清零；每轮采 32 步后，用 4 个 epoch、batch 2048 的 Adam（lr 5e-4）
回归同一步教师编码器的 `encoder(obs["privileged"])`，损失是普通 MSE。每 10 轮打印
`latent_mse`，每 50 轮存 `adapter_<iter>.pt`。

现有目录 `logs/rsl_rl/go1_rma/2026-10-28_13-38-32` 里有 `model_0/50/100/9950/9999.pt`
和 `adaptation/adapter_1000.pt`。

## 评估

没有 `go1-play`/`go1-eval-*` 入口（任务未注册），仓库也没有单独的 teacher/adapter 评估
脚本。可以量化的只有两处：

- 阶段二每个迭代打印的 `latent_mse`（适应模块对教师隐变量的拟合理想程度），仓库里没有
  记录这个 run 的收敛曲线。
- `sim2sim/go1_sim.py --teacher ... --adapter ...` 的闭环统计（见下节），它同时检验教师
  策略和适应模块。

要单独看教师行为，只能在 `train_rma.py adaptation` 的同一环境里观察；这条路径没有日志
导出，也不打印跟踪误差。

## sim2sim

```bash
uv run python sim2sim/go1_sim.py \
  --teacher logs/rsl_rl/go1_rma/2026-10-28_13-38-32/model_9999.pt \
  --adapter logs/rsl_rl/go1_rma/2026-10-28_13-38-32/adaptation/adapter_1000.pt \
  --terrain rough --command 0.5 0 0
```

`--teacher` 与 `--adapter` 必须成对给，`--policy`/`--cts` 不能同时用；`--terrain` 可选
`flat`、`rough`、`stairs`、`rl_sar`、`go2_rough`（Go1 默认 `rough`，`go2_rough` 是分布外
压力测试）。程序对 48 维观测维护最近 50 帧历史（最新帧在末尾），动作是 `FR, FL, RR, RL`
顺序的 12 维关节位置偏移。加载时校验两件事：adapter 里记录的 `obs_dim=48`、
`history_length=50`，以及 `teacher_checkpoint` 的文件名与 `--teacher` 一致——所以改名或
换目录会导致加载失败。

`go2_rough` 第 6 级 `open_stairs` 上的压力测试实测 `vx +0.402`（指令 0.5，400 步后半段
均值，见 [`sim2sim/README.md`](../../sim2sim/README.md)）。

## 已知限制

- 未注册任务：没有 `go1-play`，也没有 `go1-eval-*`；所有评估只能靠 sim2sim 或训练日志。
  [`docs/tasks/README.md`](README.md) 的索引把 RMA 写成"用 Flat 两阶段"，实际 `rma_cfg.py`
  派生自 **rough**（actor 去掉 187 维高度扫描后剩 48 维，critic 仍是 rough 的 259 维再拼
  18 维特权量）。
- 18 维特权量全部从仿真 model 字段直接读取（质量增量、COM、`actuator_gainprm`、
  `geom_friction`、足下地形高度），真机上不可得；适应模块只能从历史里近似它们。
- 阶段二在 rough 课程分布内闭环采样，没有检验地形切换或未见扰动下的隐变量误差；
  仓库里也没有记录这个 run 的 `latent_mse` 曲线或终值。
- 只有一次教师训练（2026-10-28）和一个 `adapter_1000.pt`，没有对比实验，也没有记录
  为什么在 9999 轮停止。
- adapter 与 teacher 的绑定只按文件名核对，同名不同内容的检查点不会被拒绝。

## 参考

- [`docs/rma.md`](../rma.md)（两阶段流程、特权量定义、与论文的差异）
- [`sim2sim/README.md`](../../sim2sim/README.md)（`--teacher/--adapter` 的观测与历史约定）
- 论文：Kumar, Fu, Pathak, Malik, *RMA: Rapid Motor Adaptation for Legged Robots*,
  RSS 2021 — <https://www.roboticsproceedings.org/rss17/p011.html>
- [相邻任务：Go1 CTS](go1_cts.md)、[Go1 崎岖地形](go1_rough.md)
