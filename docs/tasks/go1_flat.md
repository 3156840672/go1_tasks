# Go1 平地速度跟踪

在无限平面上训练 Go1 的速度跟踪基线：策略只吃本体感觉和速度指令，跟踪机身坐标系下的
x/y 线速度与 yaw 角速度。和 `Go1-Velocity-Rough` 的区别是去掉 187 维地形扫描、地形课程和
碰撞传感器；和 `Go1-Velocity-MoB` 的区别是只跟踪速度，没有行为指令与步态相位。

## 任务 ID

| 项 | 值 |
| --- | --- |
| 任务 ID | `Go1-Velocity-Flat`（历史 ID `Mjlab-Velocity-Flat-Unitree-Go1`，本机两者都已注册） |
| 环境配置 | `src/go1_tasks/env_cfgs.py::unitree_go1_flat_env_cfg` |
| Runner | `VelocityOnPolicyRunner`，算法为 rsl-rl 的 `PPO` |
| 检查点目录 | `logs/rsl_rl/go1_velocity/<run>/`（`experiment_name` 与 rough 共用） |

## 观测与动作

| 项 | 值 |
| --- | --- |
| actor 观测 | 48 维：base_lin_vel 3 + base_ang_vel 3 + projected_gravity 3 + joint_pos 12 + joint_vel 12 + actions 12 + command 3 |
| critic 观测 | 72 维：actor 的 48 维（无噪声）+ foot_height 4 + foot_air_time 4 + foot_contact 4 + foot_contact_forces 12 |
| 观测噪声 | 线速度 ±0.5 m/s、角速度 ±0.2 rad/s、重力 ±0.05、关节角 ±0.01 rad、关节速度 ±1.5 rad/s |
| 动作 | 12 维关节位置偏移；`GO1_ACTION_SCALE = 0.25·effort_limit/kp`，hip/thigh 0.3728、calf 0.2485 |
| 控制频率 | 50 Hz（物理步长 0.005 s，decimation 4） |
| episode | 训练 20 s；play 模式 1e9 s |
| 指令范围 | lin_vel_x/y ±1.0 m/s、ang_vel_z ±0.5 rad/s、heading ±π；重采样 3–8 s；10% 站立、30% 按朝向、20% 仅前进 |

## 训练

```bash
uv run go1-train Go1-Velocity-Flat \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard
```

环境配置里 `num_envs` 默认是 1，不显式给 `--env.scene.num-envs` 只会开一个环境。现有
`logs/rsl_rl/go1_velocity/2026-10-27_11-41-07`（平地，4096 环境，最后一个检查点
`model_5050.pt`）是仓库里唯一的平地运行目录；rough 没有运行目录。

| 超参 | 值 |
| --- | --- |
| actor / critic | 512-256-128，ELU，`obs_normalization=False` |
| 动作分布 | Gaussian，`init_std=1.0`，scalar std |
| 学习率 | 1e-3，`adaptive`（`desired_kl=0.01`） |
| 迭代数 | 10000；每环境每轮 24 步；5 epochs、4 mini-batches |
| PPO | gamma 0.99、lam 0.95、clip 0.2、entropy 0.01、max_grad_norm 1.0 |
| save_interval | 50 |
| 环境数 | 默认 1；现有 run 用 4096 |

课程只有速度指令 `command_vel`（单位 m/s、rad/s）：

| 触发（step） | lin_vel_x | ang_vel_z |
| --- | --- | --- |
| 0 | -1.0 – 1.0 | ±0.5 |
| 5000×24 | -1.5 – 2.0 | ±0.7 |
| 10000×24 | -2.0 – 3.0 | ±0.7（不再扩大） |

域随机化（全部在 startup/reset 采样）：足端滑动摩擦 0.3–1.5、自旋 1e-4–2e-2、滚动
1e-5–5e-3；编码器偏置 ±0.015；trunk COM 偏移 x/y ±0.025 m、z ±0.03 m；每 1–3 s 施加
±0.5 m/s 水平、±0.4 m/s 竖向速度、±0.52 rad/s 翻滚/俯仰、±0.78 rad/s 偏航的推挤。play
模式去掉推挤、`height_scan` 噪声与 actor 观测噪声。

## 评估

```bash
uv run go1-play Go1-Velocity-Flat \
  --checkpoint-file logs/rsl_rl/go1_velocity/2026-10-27_11-41-07/model_5050.pt \
  --viewer viser
```

`--checkpoint-file` 与 `--wandb-run-path` 必须给一个。看三件事：在 Viser 的 Commands 面板
改 x/y/yaw 后的跟踪误差、`Episode_Reward/track_linear_velocity` 与 `track_angular_velocity`
的收敛情况、以及 `Metrics/mean_action_acc` 是否随训练下降。`--no-terminations` 只用于观察
动作，不代表策略不摔。

## sim2sim

独立 MuJoCo 的 `--policy` 只认 48 维 Go1 平地格式：

```bash
uv run python sim2sim/go1_sim.py \
  --policy logs/rsl_rl/go1_velocity/2026-10-27_11-41-07/model_5050.pt \
  --terrain flat --command 0.5 0 0
```

`--terrain` 可选 `flat`、`rough`、`stairs`、`rl_sar`、`go2_rough`（后两者是分布外压力
测试）。加 `--headless --steps 400` 会打印后半段的 `mean vx/vy/wz` 与绝对误差。
`--policy` 只接受 48 维第一层；235 维 rough 检查点会被维度校验拒绝。

## 已知限制

- 环境配置默认 `num_envs=1`，忘了覆盖就退化成单环境训练。仓库里没有记录这是有意为之。
- play 的指令范围（x -1.5–2.0、yaw ±0.7）来自 rough 的 play 覆盖，比训练第 0 阶段的
  范围宽；低迭代数检查点在这两个端点上不可靠。
- 平地任务没有任何地形感知，上坡、楼梯、碎石都属分布外，对台阶没有先验。
- 独立 MuJoCo 场景的执行器与接触参数和 mjlab/MuJoCo-Warp 不同，sim2sim 的跟踪数字
  只能用来横向比较检查点；仓库里没有记录平地任务的 sim2sim 实测数字或 benchmark。

## 参考

- [仓库 README](../../README.md)（安装、任务总表、ID 注册规则）
- [`docs/cts.md`](../cts.md)（同一套 rough 奖励与地形定义的来历）
- [相邻任务：Go1 崎岖地形](go1_rough.md)、[Go1 MoB](go1_mob.md)
- 上游基础配置：`framework/mjlab/src/mjlab/tasks/velocity/velocity_env_cfg.py`
