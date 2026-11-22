# Go1 MoB 行为条件步态

MoB（Mixture of Behaviors，Walk These Ways 的 Go1 移植）用一个 8 维行为指令条件化步态：
步态相位偏移（Pronk/Trot/Bound/Pace）、步频、机身高度、俯仰、站姿宽度、摆腿高度，同时跟踪
速度指令。策略输入是 30 帧本体历史外加 4 维步态相位 sin。和 `Go1-Velocity-Flat` 的区别是
多了行为指令、相位特征与历史栈；和 `Go1-Velocity-CTS` 的区别是它本来就只有一个盲策略，
不区分师生。

## 任务 ID

| 项 | 值 |
| --- | --- |
| 任务 ID | `Go1-Velocity-MoB`（历史 ID `Mjlab-Velocity-MoB-Unitree-Go1`，本机两者都已注册） |
| 环境配置 | `src/go1_tasks/env_cfgs.py::unitree_go1_mob_env_cfg` |
| Runner | `MobOnPolicyRunner`（保存/恢复速度课程状态），算法为 rsl-rl 的 `PPO` |
| 检查点目录 | `logs/rsl_rl/go1_mob/<run>/`（`experiment_name=go1_mob`） |

## 观测与动作

| 项 | 值 |
| --- | --- |
| actor 观测 | 57 维 × `history_length=30` = 1710：base_ang_vel 3、projected_gravity 3、joint_pos 12、joint_vel 12、actions 12、command 3、behavior 8、clock 4（四腿 sin(2π·phase)） |
| critic 观测 | 84 维：actor 的 57 维（把 base_lin_vel 加回来 = 60）+ foot_height 4 + foot_air_time 4 + foot_contact 4 + foot_contact_forces 12 |
| 动作 | 12 维关节位置偏移；尺度 hip 0.125、thigh 0.25、calf 0.25；执行器 `Kp=20`、`Kd=0.5` |
| 控制频率 / episode | 50 Hz（物理步长 0.005 s，decimation 4）；训练 20 s，play 模式 1e9 s |
| 指令范围 | 机体系 lin_vel_x ±1.0 m/s、lin_vel_y ±0.6 m/s、ang_vel_z ±1.0 rad/s，`heading_command=False`，重采样 3–8 s |

行为指令（`BehaviorCommand`，采样区间）：

| 分量 | 范围 |
| --- | --- |
| 步态相位偏移 | 四选一：Pronk (0, 0, 0)、Trot (0.5, 0, 0)、Bound (0, 0.5, 0)、Pace (0, 0, 0.5) |
| frequency | 1.5–4.0 Hz |
| body_height | 0.20–0.38 m |
| body_pitch | -0.15–0.15 rad |
| stance_width | 0.22–0.40 m |
| swing_height | 0.04–0.18 m |

## 训练

```bash
uv run go1-train Go1-Velocity-MoB \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard
```

配置默认 `num_envs=1024`；现有 `logs/rsl_rl/go1_mob/2026-10-27_15-04-14` 用 4096 个环境，
最后一个检查点是 `model_5050.pt`。

| 超参 | 值 |
| --- | --- |
| actor / critic | 512-256-128，ELU，不做观测归一化 |
| 动作分布 | Gaussian，`init_std=1.0`，scalar std |
| PPO | lr 1e-3 adaptive（`desired_kl=0.01`）、gamma 0.99、lam 0.95、clip 0.2、entropy 0.01 |
| 迭代 / 保存 | 10000 轮；每环境每轮 24 步；5 epochs、4 mini-batches；每 50 轮保存 |
| 奖励 | 单个 `mob` 项，权重 1.0 |

奖励是乘积式 `tracking × exp(auxiliary)`：`tracking = exp(-‖v_xy_err‖²/0.25) + 0.5·exp(-yaw_err²/0.25)`；
auxiliary 的主要系数是机身高度误差 -80、摆腿高度误差 -30、Raibert 足端目标误差 -10、摆腿期触地 -4、
支撑期足端滑动 -4、俯仰误差 -5、关节限位 -10、动作变化/加速度各 -0.1、躯干 vz² -0.02、足端接触速度 -0.04、
力矩与关节速度各 -1e-4、关节加速度 -2.5e-7、四肢碰撞 -5。

课程是网格自适应速度指令，没有地形课程：

| 项 | 值 |
| --- | --- |
| 网格 | `(11, 7, 11)` 对应 x / y / yaw 三个轴 |
| 初始范围 | x ±0.4、y ±0.3、yaw ±0.4（即 `initial_limits`） |
| 跟踪阈值 | 线速度 0.8、偏航角速度 0.5 |
| 扩张节奏 | 每 1.0 s 检查一次，达标格的邻居权重 +0.2 |
| 持久化 | 课程状态随检查点存取（`MobOnPolicyRunner`） |

域随机化：足端摩擦（滑动 0.3–1.5、自旋、滚动）、编码器偏置 ±0.015、trunk COM 偏移、
`pseudo_inertia` 质量/惯量 ±5%、PD 增益 ±10%、关节摩擦 0–0.05、reset 关节位置/速度
±0.1、每 4–8 s 施加 ±0.3 m/s 推挤。

## 评估

```bash
uv run go1-play Go1-Velocity-MoB \
  --checkpoint-file logs/rsl_rl/go1_mob/2026-10-27_15-04-14/model_5050.pt \
  --viewer viser
```

play 把行为指令的重采样周期设成 1e8（固定当前行为），速度指令仍按 3–8 s 重采样，也可在
Viser 的 `Commands` 面板改；行为分量要手动设。指标看 `Metrics/body_height_error`（机身高度与指令的偏差）、`Episode_Reward/mob`，以及
同一速度下换步态后是否还保持对角/同相接触节律。仓库里没有记录 MoB 的评测表。

## sim2sim

MoB 有独立入口（`go1_sim.py` 只认 48/45 维和 CTS，跑不了 1710 维）：

```bash
uv run python sim2sim/go1_mob_sim.py \
  --policy logs/rsl_rl/go1_mob/2026-10-27_15-04-14/model_5050.pt \
  --terrain flat --gait Trot --command 0.5 0 0
```

`--gait` 选 Pronk/Trot/Bound/Pace，另有 `--frequency`、`--body-height`、`--body-pitch`、
`--stance-width`、`--swing-height` 逐项覆盖（默认 2.5 Hz / 0.30 m / 0 / 0.30 m / 0.10 m）；
`--terrain` 非 `flat` 时脚本会警告
"训练只在平地"。该入口的执行器参数是 `Kp=20`、`Kd=0.5`、动作尺度 0.125/0.25/0.25，与训练
一致（不同于 `go1_sim.py` 用 armature 推导增益）。`--headless --steps 400` 打印后半段速度统计。

## 已知限制

- 只在平地训练；`rough`/`stairs`/`rl_sar`/`go2_rough` 都是分布外。`go2_rough` 平地块上
  实测 `vx +0.515`（指令 0.5），但仓库里没有记录系统性的地形评测。
- 复现的是论文的接口与奖励思路，不含论文的监督式速度/摩擦估计器、执行器网络、20 ms 动作
  延迟和按步态分阶段的课程；行为指令范围是移植时选的实验值，不是论文附录原值。
- 现有 run 是否在奖励修正之后训练，仓库里没有记录；`docs/mob.md` 明确说用旧奖励训练、
  连续行为指令恒为零的检查点不会展示这套配置的行为。
- 课程状态存在检查点里，用 `go1-play` 继续训练没问题，但把检查点挪到别的实验目录后
  课程会从零开始。

## 参考

- [`docs/mob.md`](../mob.md)（行为清单、奖励设计、保真度说明）
- [`docs/amp_mob_collection.md`](../amp_mob_collection.md)（用 MoB 检查点采集 AMP 参考动作）
- 论文：Margolis & Agrawal, *Walk These Ways: Tuning Robot Control for Generalization with
  Multiplicity of Behavior*, CoRL 2022 —
  <https://proceedings.mlr.press/v205/margolis23a.html>
- [相邻任务：Go1 AMP](go1_amp.md)、[Go1 平地](go1_flat.md)
