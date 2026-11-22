# Go1 CTS 师生并发训练

CTS（Concurrent Teacher-Student）把教师与学生放进同一批环境同时训练：教师用 259 维
critic 观测（高度扫描、足端接触力等特权信息）学出 32 维隐变量，学生只用 45 维本体观测加
5 帧历史去回归同一隐变量，两者共享策略头。和 `Go1-Velocity-Rough` 的区别是部署输入不含
高度扫描；和 `Go1-Velocity-AMP` 的区别是地形适应靠特权观测而不是运动数据。

## 任务 ID

| 项 | 值 |
| --- | --- |
| 任务 ID | `Go1-Velocity-CTS`（历史 ID `Mjlab-Velocity-CTS-Unitree-Go1`，本机两者都已注册） |
| 环境配置 | `src/go1_tasks/env_cfgs.py::unitree_go1_cts_env_cfg` |
| Runner | `VelocityOnPolicyRunner` + `go1_tasks.cts.CTSPPO`，actor 类为 `go1_tasks.cts.CTSActor` |
| 检查点目录 | `logs/rsl_rl/go1_cts/<run>/`（`experiment_name=go1_cts`，`save_interval=100`） |

## 观测与动作

| 项 | 值 |
| --- | --- |
| actor 观测 | 45 维：base_ang_vel 3 + projected_gravity 3 + joint_pos 12 + joint_vel 12 + actions 12 + command 3 |
| 学生输入 | `cts_history` 5 帧 × 45 = 225（term-major、每段由旧到新）+ `cts_student` 的 1 维 `mask`（训练时最后 1/4 环境为 1，play 恒为 1） |
| critic 观测 | 259 维：rough 的 critic（235 + 24 个足端项），含地形扫描与接触力 |
| 策略头 | 45 + 32 = 77 维输入；32 维隐变量 L2 归一化后与本体观测拼进 MLP |
| 动作 | 12 维关节位置偏移；hip 0.125、thigh/calf 0.25；`Kp=20`、`Kd=0.5`；`delay_max_lag=4`（0–20 ms 延迟） |
| 控制频率 / episode | 50 Hz（物理步长 0.005 s，decimation 4）；训练 20 s，play 模式 1e9 s |
| 指令范围 | 机体系 lin_vel_x/y、ang_vel_z 随课程扩张（见下表），heading ±π；重采样 3–8 s，10% 站立 / 10% 按朝向 / 10% 仅前进 |

## 训练

```bash
uv run go1-train Go1-Velocity-CTS \
  --env.scene.num-envs 256 \
  --agent.logger tensorboard
```

`num_envs` 必须是 4 的倍数（`student_group_mask` 划最后 1/4 为学生），配置默认 1024；现有
目录 `2026-10-21_18-59-37`（17799 轮）与 `2026-10-29_14-14-46`（256 环境、6300 轮）。

| 超参 | 值 |
| --- | --- |
| actor / critic | 512-256-128，ELU；编码器 512-256 → 32 |
| 学生编码器 | lr 1e-3、5 epochs、batch 512；每个 PPO 迭代后更新一次（256 环境时 64 个学生世界 × 24 步 = 1536 样本，约 15 步） |
| PPO | lr 1e-3 adaptive（`desired_kl=0.01`）、gamma 0.99、lam 0.95、clip 0.2、entropy 0.01 |
| 迭代 / 保存 | 10000 轮；每环境每轮 24 步；5 epochs、4 mini-batches；每 100 轮保存 |

环境改动：`nconmax=128`（rough 的 35 会在地形上丢接触）、`max_init_terrain_level=0`、楼梯
比例 55%（上/下各 27.5%）、坡度 ≤0.6、台阶高 0–0.08 m / 宽 0.35 m，终止条件加回 `fell_over`（70°）。

奖励（权重）：track_linear_velocity 6.0、track_angular_velocity 3.0、track_yaw_velocity 2.0、
upright 0.1、pose 1.0、dof_pos_limits -1.0、action_rate_l2 -0.1、action_acc_l2 -0.02、foot_slip -0.3、
soft_landing -1e-5、self_collisions -0.1、shank_collision -0.3、trunk_head_collision -0.1、
feet_regulation -0.05、standing_missing_foot_contacts -0.3、linear_velocity_error_l2 -1.0；
`air_time`、`foot_clearance`、`foot_swing_height` 已删除，抬脚高度由近地足端调节项约束。

速度课程（step = 轮数 × 24）：

| 触发（轮） | lin_vel_x | lin_vel_y | ang_vel_z |
| --- | --- | --- | --- |
| 0 | -0.3 – 0.6 | ±0.2 | ±0.15 |
| 1200 | -0.6 – 0.8 | ±0.4 | ±0.3 |
| 3600 | ±1.0 | ±0.7 | ±0.5 |
| 5600 | ±1.0 | ±1.0 | ±0.5 |

域随机化：质量/惯量 0.9–1.1、trunk 附加 -0.5 – +2 kg、COM 偏移 x ±0.05 m / y,z ±0.035 m、
足端滑动摩擦 0.3–1.5、PD 增益与力矩上限 0.8–1.2、编码器偏置 ±0.015、每 1–3 s ±0.5 m/s
推挤；play 只保留 rough 继承的 startup 随机化。

## 评估

```bash
uv run go1-play Go1-Velocity-CTS \
  --checkpoint-file logs/rsl_rl/go1_cts/2026-10-29_14-14-46/model_6300.pt \
  --viewer viser

uv run go1-eval-cts logs/rsl_rl/go1_cts/2026-10-29_14-14-46/model_6300.pt \
  --vx 0.5 --seconds 5

uv run go1-eval-cts logs/rsl_rl/go1_cts/2026-10-29_14-14-46/model_6300.pt \
  --vx 0.5 --branch teacher --randomized --seed 7
```

play 只走学生分支，但场景仍会构造地形与 critic 传感器。`go1-eval-cts` 强制平地、单环境、
默认剥掉 startup 域随机化；输出实际/指令速度、`mae_vx/vy/yaw`、`action_rms`、
`action_delta_rms`、shank 接触比例、足端接触比例与 site 高度、reset 次数。`--branch teacher`
区分跟踪失败出在策略还是学生编码器，`--randomized` 保留域随机化做鲁棒性对照；前向、反向、
横移、偏航要分别跑。

## sim2sim

学生分支在 ONNX 与 checkpoint 里都完整存在，`go1_sim.py --cts` 两种都收：

```bash
uv run python sim2sim/go1_sim.py \
  --cts logs/rsl_rl/go1_cts/2026-10-29_14-14-46/2026-10-29_14-14-46.onnx \
  --terrain rough --command 0.5 0 0
```

`--terrain` 可选 `flat`、`rough`、`stairs`、`rl_sar`、`go2_rough`。ONNX 是双输入
`proprio (1,45)` + `history (1,225)`，只在保存检查点时导出，所以只对应 run 目录最后一个
`model_*.pt`；中间检查点用 `.pt` 跑会不同（`model_6000.pt` 在 `rough` 上 `vx +0.205`，6300
及其导出是 `+0.456`）。该 run 在 `rough` 上实测 `vx +0.456`、误差 0.044、偏航漂移 `+0.359 rad/s`（见 [`docs/cts.md`](../cts.md)）。

历史必须 term-major：角速度 5 帧 → 重力 5 帧 → 关节角 5 帧 → 关节速度 5 帧 → 上一动作
5 帧 → 指令 5 帧，每段由旧到新；按 frame-major 拼仍能跑，但会明显横漂（`vy` 从 `+0.001` 变成 `+0.689`）。

## 已知限制

- `go1_sim.py` 对 `--cts` 用 armature 推导的 `Kp≈15.9/35.8`、`Kd≈1.01/2.28` 和
  `0.25·effort/Kp`（0.3728/0.2485）尺度，训练却是 `Kp=20`、`Kd=0.5`、0.125/0.25/0.25；
  偏航漂移有多少来自这里、偏航跟踪为何是弱项，仓库里都没有记录。
- 早期 run（`2026-10-19_23-32-45_cts_contact128_full` 的 `model_9800.pt`）平均摆腿离地
  只有 3.2 cm；现有奖励已删掉抬脚激励，旧检查点要继续训练才会改变步态。
- `num_envs` 不是 4 的倍数会断言失败；学生世界固定为最后 1/4，改比例要改代码。

## 参考

- [`docs/cts.md`](../cts.md)（观测布局、奖励逐项解释、实测与调参）
- [`sim2sim/README.md`](../../sim2sim/README.md)（`--cts` 的 ONNX/`.pt` 契约与历史布局）
- 论文：Wang et al., *Concurrent Teacher-Student Framework for Agile Quadrupedal Locomotion*, 2024 — <https://arxiv.org/abs/2405.10830>
- [相邻任务：Go1 崎岖地形](go1_rough.md)、[Go1 RMA](go1_rma.md)
