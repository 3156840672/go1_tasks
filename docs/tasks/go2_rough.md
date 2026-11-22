# Go2 崎岖地形盲走

`Go2-Velocity-Rough` 是 Go2 的主线盲策略：actor 只有 45 维本体感觉，不看地形也
不看机身线速度，靠足端接触走完 14 种崎岖地形；critic 保留线速度与 187 维高度扫描，
构成非对称 PPO。与 `Go2-Velocity-Rough-AMP` 的差别是没有 AMP 判别器，与
`Go2-Velocity-HIM` 的差别是没有历史编码器，输入是单帧。

## 任务 ID

| 项 | 值 |
| --- | --- |
| 任务 ID | `Go2-Velocity-Rough` |
| 环境配置 | `src/go2_tasks/env_cfgs.py::unitree_go2_rough_env_cfg` |
| Runner | `VelocityOnPolicyRunner`（PPO，非对称 actor/critic） |
| 检查点目录 | `logs/rsl_rl/go2_rough/<run>/`，仓库里有 `2026-11-07_20-12-16`（只存了 `model_9999.pt`） |

## 观测与动作

| 项 | 维数 | 说明 |
| --- | ---: | --- |
| `base_ang_vel` | 3 | 噪声 ±0.2 rad/s |
| `projected_gravity` | 3 | 噪声 ±0.05 |
| `joint_pos` | 12 | 相对默认角，噪声 ±0.01 rad |
| `joint_vel` | 12 | 噪声 ±1.5 rad/s |
| `actions` | 12 | 上一控制步动作 |
| `command` | 3 | `[vx, vy, wz]` |
| **actor 合计** | **45** | 单帧历史（`history_length=1`） |
| critic 额外项 | +214 | `base_lin_vel` 3、`height_scan` 187（1.6×1.0 m，0.1 m 网格）、足端四项 24 |
| **critic 合计** | **259** | 无噪声 |
| 动作 | 12 | `q_des = q_default + scale · a`，scale = `[0.125, 0.25, 0.25]` |
| 控制频率 | 50 Hz | 物理步 0.005 s，decimation 4 |
| episode 长度 | 20 s | play 模式为 `1e9` |
| 指令范围（训练） | x ±1.0，y ±1.0 m/s，yaw ±0.5 rad/s | 3–8 s 重采样，10% 站立、30% 航向控制、20% 只前进 |
| 指令范围（play） | 同训练 | 地形换成 5×5 随机块，无课程 |

仿真侧为刚体接触加严设置：`ccd_iterations=500`、`impratio=10`、椭圆锥、
`contact_sensor_maxmatch=500`、`nconmax=35`（CTS 与论文式任务改为 128）。

## 训练

与平地不同，rough 的注册默认环境数也是 1，训练要显式给 1024。

```bash
uv run go2-train Go2-Velocity-Rough \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard
```

| 超参 | 值 |
| --- | --- |
| 网络 | actor/critic 512-256-128，ELU，观测不做归一化 |
| lr | 1e-3，`adaptive`，`desired_kl=0.01` |
| PPO | 5 epochs、4 mini-batches、clip 0.2、entropy 0.01、γ 0.99、λ 0.95、grad-norm 1.0 |
| 采样 | 24 steps/env，`max_iterations=10000`，`save_interval=50` |
| 环境数 | 1024（需手动传） |

奖励（权重）：`track_linear_velocity` 4.0、`track_angular_velocity` 4.0、`upright`
1.0（相对地形高度）、`pose` 1.0、`dof_pos_limits` −1.0、`action_rate_l2` −0.2、
`foot_clearance` −0.1、`foot_slip` −0.1、`soft_landing` −1e−5、
`self_collisions`/`shank_collision`/`trunk_head_collision` 各 −0.1；
`body_ang_vel`、`angular_momentum`、`air_time`、`foot_swing_height` 权重为 0。

课程：

- **地形**：14 种地形（`flat` 0.10、上下金字塔台阶各 0.10、上下坡度各 0.07、
  `random_rough`/`wave_terrain` 各 0.07，其余 7 种各 0.06），10 行难度、每格 8×8 m，
  初始等级 0–5。`terrain_levels_vel` 按位移升降级（走满 4 m 升级，
  不足指令距离一半降级）。高度上限压在 `0.14 m`，只用本体感觉可穿越。
- **速度**：继承基础三段课程，0 步 x ±1.0/yaw ±0.5，`5000×24` 步 x −1.5…2.0/yaw ±0.7，
  `10000×24` 步 x −2.0…3.0。
- 不使用 AMP 参考动作，也没有教师-学生蒸馏。

终止条件：时间到、`illegal_contact`（大腿碰地）、`out_of_terrain_bounds`（越界算
time-out）。play 模式去掉越界与推力扰动、关闭观测噪声。

## 评估

```bash
uv run go2-play Go2-Velocity-Rough \
  --checkpoint-file logs/rsl_rl/go2_rough/2026-11-07_20-12-16/model_9999.pt \
  --num-envs 4 --viewer viser

uv run python tools/eval_go2_task.py \
  --task Go2-Velocity-Rough \
  --checkpoint logs/rsl_rl/go2_rough/2026-11-07_20-12-16/model_9999.pt \
  --command 0.5 0 0 --terrain-level 5 --steps 300
```

看 `error_vel_xy`、`terminated` 百分比，以及训练日志里的 `Curriculum/terrain_levels/mean`
（应逐步抬升）和 `Episode_Termination/illegal_contact`（过高说明还在摔）。
`--terrain-level` 会把所有环境固定在同一级并关掉地形课程，适合回归对比。

## sim2sim

`--policy` 自动识别 45 维检查点，`go2_rough` 是训练地形的逐格导出：

```bash
uv run python sim2sim/go2_sim.py \
  --policy logs/rsl_rl/go2_rough/2026-11-07_20-12-16/model_9999.pt \
  --terrain go2_rough --terrain-type all --terrain-level 6 \
  --terrain-column open_stairs --command 0.5 0 0
```

支持 `flat`/`rough`/`stairs`/`rl_sar`/`go2_rough` 五种地形；`go2_rough` 可选单列
（`--terrain-type <列名> --terrain-level 0-9`）或 4×4 整图（`--terrain-type all`
加 `--terrain-column`）。只接受 45/48 维 `model_*.pt`；带 187 维扫描的旧 235 维
Rough 检查点会被拒绝，因为独立 MJCF 没有 raycast 传感器。

## 已知限制

- 训练包含低矮台阶。金字塔上、下台阶的单级高度上限为 `0.10 m`，开放式和随机台阶的配置上限为 `0.14 m`。普通楼梯的通过能力需要单独验证。
- `nconmax=35` 在困难地形上可能丢接触；CTS 与论文式任务都用 128，本任务没有改。
- `--terrain rough` 的 2–10 cm 随机高度场比训练课程（约 5 级）更难，属分布外；
  `docs/go2_him.md` 记录平地 `vx=0.5` 误差 0.096 m/s 属正常量级。
- 日志里有两处 `error_vel_xy`（0.353 与 0.80），对应不同评测器与轮数，口径不一致，
  不能当成本任务的 benchmark。
- `2026-11-07_20-12-16` 只保留 `model_9999.pt`，没有中间检查点；sim2sim 与训练引擎的
  接触、摩擦和电机限幅实现不同，行为差异不能全部归因于策略。

## 参考

- [Go2 训练与部署约定](../go2.md)、[Go2 HIM 文档（含地形选型与同口径评测）](../go2_him.md)
- [CTS 文档（同地形上的对照任务）](../cts.md)
- sim2sim 地形与选格命令：[`sim2sim/README.md`](../../sim2sim/README.md)
