# Go1 崎岖地形速度跟踪

Go1 的崎岖地形速度跟踪基线：本体观测外加机身自带 raycast 的 187 维高度扫描（不是 Go2
那种 45 维盲策略），在程序生成的地形块上跟踪 x/y 线速度与 yaw 角速度，靠地形等级与指令
范围两条课程逐步加难。
和 `Go1-Velocity-Flat` 的区别是多了 187 维 height_scan、地形课程和碰撞惩罚；和
`Go1-Velocity-CTS` 的区别是没有师生结构，actor 直接吃高度扫描（235 维）。

## 任务 ID

| 项 | 值 |
| --- | --- |
| 任务 ID | `Go1-Velocity-Rough`（历史 ID `Mjlab-Velocity-Rough-Unitree-Go1`，本机两者都已注册） |
| 环境配置 | `src/go1_tasks/env_cfgs.py::unitree_go1_rough_env_cfg` |
| Runner | `VelocityOnPolicyRunner`，算法为 rsl-rl 的 `PPO` |
| 检查点目录 | `logs/rsl_rl/go1_velocity/<run>/`（与 flat 共用 `experiment_name`，靠 run 名区分） |

## 观测与动作

| 项 | 值 |
| --- | --- |
| actor 观测 | 235 维：flat 的 48 维 + height_scan 187（1.6 m × 1.0 m 网格，0.1 m 间距，scale 1/5） |
| critic 观测 | 259 维：actor 的 235 维（无噪声）+ foot_height 4 + foot_air_time 4 + foot_contact 4 + foot_contact_forces 12 |
| 观测噪声 | 线速度 ±0.5 m/s、角速度 ±0.2 rad/s、重力 ±0.05、关节角 ±0.01 rad、关节速度 ±1.5 rad/s、高度扫描 ±0.1 |
| 动作 | 12 维关节位置偏移；`GO1_ACTION_SCALE`：hip/thigh 0.3728、calf 0.2485 |
| 控制频率 | 50 Hz（物理步长 0.005 s，decimation 4） |
| episode | 训练 20 s；play 模式 1e9 s |
| 指令范围 | lin_vel_x/y ±1.0 m/s、ang_vel_z ±0.5 rad/s、heading ±π；重采样 3–8 s；10% 站立、30% 按朝向、20% 仅前进 |

## 训练

```bash
uv run go1-train Go1-Velocity-Rough \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard
```

`num_envs` 默认是 1，必须显式覆盖。`logs/rsl_rl/` 下没有 rough 的运行目录（`go1_velocity`
里那次是平地），因此没有可引用的 rough 检查点。

| 超参 | 值 |
| --- | --- |
| actor / critic | 512-256-128，ELU，不做观测归一化 |
| 动作分布 | Gaussian，`init_std=1.0`，scalar std |
| 学习率 | 1e-3，`adaptive`（`desired_kl=0.01`） |
| 迭代数 | 10000；每环境每轮 24 步；5 epochs、4 mini-batches |
| PPO | gamma 0.99、lam 0.95、clip 0.2、entropy 0.01、max_grad_norm 1.0 |
| save_interval | 50 |
| 接触预算 | `nconmax=35`、`njmax=1500`、`contact_sensor_maxmatch=500`、`ccd_iterations=500` |

地形生成器（8 m × 8 m 块，10 行 × 20 列，`curriculum=True`，初始等级上限 5）：

| 子地形 | 比例 | 说明 |
| --- | --- | --- |
| flat | 0.20 | 平地块 |
| pyramid_stairs | 0.20 | 上金字塔楼梯，踏步高 0–0.10 m |
| pyramid_stairs_inv | 0.20 | 下金字塔楼梯，同上 |
| hf_pyramid_slope | 0.10 | 上坡，slope 0–1.0 |
| hf_pyramid_slope_inv | 0.10 | 下坡，同上 |
| random_rough | 0.10 | 随机高度场 |
| wave_terrain | 0.10 | 波浪地形 |

奖励（`RewardManager` 默认 `scale_by_dt=True`，权重再乘 dt=0.02 s）：track_linear_velocity 4.0、
track_angular_velocity 4.0、upright 1.0、pose 1.0、dof_pos_limits -1.0、action_rate_l2 -0.2、
foot_clearance -0.1、foot_slip -0.1、soft_landing -1e-5、self_collisions -0.1、shank_collision -0.1、
trunk_head_collision -0.1；`air_time`、`body_ang_vel`、`angular_momentum`、`foot_swing_height` 的权重
为 0，配置保留但不参与优化。

终止条件只有 `time_out`、`out_of_terrain_bounds` 和 `illegal_contact`（thigh 触地）。
rough 故意删掉了 `fell_over`：机器人需要能在大倾角地形上恢复，只靠出界和非法接触重置。
课程 `terrain_levels`（`mdp.terrain_levels_vel`）按位移升降等级（走满半个地形块 4 m 升级，不足指令距离一半降级），`command_vel` 按 step 把 x 从 ±1 扩到
-2.0–3.0、yaw 到 ±0.7。域随机化与 flat 相同（足端摩擦、编码器偏置、trunk COM、每 1–3 s 一次
推挤：水平/竖向速度 ±0.5/±0.4 m/s、翻滚/俯仰/偏航 ±0.52/±0.52/±0.78 rad/s）。

## 评估

```bash
uv run go1-play Go1-Velocity-Rough \
  --checkpoint-file logs/rsl_rl/go1_velocity/YOUR_RUN/model_5000.pt \
  --viewer viser
```

play 会重建 5×5、`border_width=10` 的随机地形块，关掉课程与推挤，每个 reset 换一块地形，
episode 设成 1e9，actor 观测不再加噪。看 `Episode_Reward/track_linear_velocity`、
`Curriculum/terrain_levels`（训练时）以及 `Metrics/mean_action_acc`；当前等级在
Viser 里不可见，只能用日志。仓库里没有 rough 的实测评测数字。

## sim2sim

**不支持。** 独立 MuJoCo 场景（`sim2sim/go1_sim.py`）只接受平地 48 维、CTS 学生和 RMA
teacher/adapter 三种检查点；rough 的 235 维输入需要 187 维 raycast 高度扫描，而
`sim2sim` 的独立 MJCF 没有 raycast 传感器，`--policy` 的维度校验会直接拒绝该检查点。
`--terrain rough` 是给 48 维策略用的分布外压力测试地形，不是这个任务的部署入口。

## 已知限制

- 训练接触上限 `nconmax=35` 在困难地形上可能丢接触（见 [`docs/cts.md`](../cts.md) 的分析，
  CTS 因此把它提到 128）。这会影响奖励与终止统计，无法从日志直接看出。
- play 的地形是随机 5×5 块且没有课程，出生等级不固定；同一检查点在不同 patch 上的表现
  差异很大，单次播放不能当作评测。
- 235 维策略无法进独立 MuJoCo，部署前必须重训成 blind（48/45 维）或 CTS 学生。
- 奖励权重里 `foot_clearance`、`soft_landing` 的数值很小，训练中是否真正约束了抬脚高度，
  仓库里没有记录验证结果。
- 现有日志没有 rough 检查点，也没有 rough 的 sim2sim 数字；仓库里没有记录。

## 参考

- [仓库 README](../../README.md)（任务总表与训练入口）
- [`docs/cts.md`](../cts.md)（rough 奖励/地形的详细分析，CTS 在其上派生）
- [相邻任务：Go1 平地](go1_flat.md)、[Go1 CTS](go1_cts.md)
- 上游基础配置：`framework/mjlab/src/mjlab/tasks/velocity/velocity_env_cfg.py`
