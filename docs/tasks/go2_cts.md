# Go2 并发教师-学生（CTS）

`Go2-Velocity-CTS` 用并发教师-学生结构训练可部署的盲策略：教师吃 259 维特权观测、
学生吃 5 帧本体历史（225 维），两者各编码 32 维隐变量，策略头看到「45 维本体 +
32 维隐变量」共 77 维。训练时 75% 环境用教师隐变量、25% 用学生隐变量，学生编码器以
教师隐变量为回归目标，不需要先训教师再蒸馏。

## 任务 ID

| 项 | 值 |
| --- | --- |
| 任务 ID | `Go2-Velocity-CTS` |
| 环境配置 | `src/go2_tasks/env_cfgs.py::unitree_go2_cts_env_cfg` |
| Runner | `VelocityOnPolicyRunner` + `go1_tasks.cts.CTSPPO`，actor 为 `CTSActor` |
| 检查点目录 | `logs/rsl_rl/go2_cts/<run>/`，仓库里有 `2026-10-31_10-43-13`（只保留 `model_9999.pt` 与同名 `.onnx`） |

## 观测与动作

| 项 | 维数 | 说明 |
| --- | ---: | --- |
| `actor` | 45 | 角速度 3、投影重力 3、关节角 12、关节速度 12、上一动作 12、指令 3 |
| `cts_history` | 225 | `actor` 组 5 帧历史，按 term 展开（term-major），段内从旧到新 |
| `cts_student` | 1 | 组掩码，训练时最后 25% 环境为 1 |
| `critic` | 259 | 45 + 线速度 3 + 高度扫描 187 + 足端四项 24 |
| 教师 / 学生编码器 | 259 → 32 / 225 → 32 | 输出做 L2 归一化 |
| 策略头输入 | 77 | 45 维本体 + 32 维隐变量 |
| 动作 | 12 | `q_des = q_default + scale · a`，scale = `[0.125, 0.25, 0.25]` |
| 控制 / episode | 50 Hz / 20 s | 物理步 0.005 s，decimation 4，执行器延迟 ≤4 步 |
| 指令范围 | 见下方四段课程 | 10% 站立、10% 航向控制、10% 只前进 |

## 训练

`num_envs` 必须是 4 的倍数（学生掩码按最后 25% 划分）。

```bash
uv run go2-train Go2-Velocity-CTS \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard
```

| 超参 | 值 |
| --- | --- |
| 网络 | 策略头 512-256-128，ELU；两个编码器各自独立 |
| lr | 策略 1e-3 adaptive；学生编码器 1e-3 |
| PPO | 5 epochs、4 mini-batches、clip 0.2、entropy 0.01、γ 0.99、λ 0.95 |
| 学生回归 | 5 epochs、批 512、归一化隐变量的 MSE、grad-norm 1.0 |
| 采样 | 24 steps/env，`max_iterations=10000`，`save_interval=100` |

- **地形**：与 rough 相同的 14 种地形、10 行难度、初始等级 0；出生比例按数值设置：
  上/下金字塔台阶各 0.275（台阶高 0–0.08 m、宽 0.35 m）、上/下坡面各 0.075
  （坡度 0–0.6）、`flat` 0.15、`random_rough` 与 `wave_terrain` 各 0.075；mjlab 按
  比例归一化后作为出生权重，14 种地形仍各占一列。`nconmax=128`。
- **速度**：0 步 x −0.3…0.6、y ±0.2、yaw ±0.15；`1200×24` 步 x −0.6…0.8、y ±0.4、
  yaw ±0.3；`3600×24` 步 x ±1.0、y ±0.7、yaw ±0.5；`5600×24` 步 y 扩到 ±1.0。
- **域随机化**：伪惯量、躯干负载 −0.5…2.0 kg、质心偏移 ±0.05/±0.035/±0.035 m、
  足底摩擦 0.3–1.5、`Kp/Kd` ×0.8–1.2、力矩上限 ×0.8–1.2、执行器延迟 ≤4 步。

奖励：`track_linear_velocity` 6.0、`track_angular_velocity` 3.0、`track_yaw_velocity`
2.0（σ=0.25）、`upright` 0.1、`pose` 1.0、`dof_pos_limits` −1.0、`action_rate_l2`
−0.1、`action_acc_l2` −0.02、`foot_slip` −0.3、`feet_regulation` −0.05、
`standing_missing_foot_contacts` −0.3、`linear_velocity_error_l2` −1.0、
三个碰撞项 −0.1/−0.3/−0.1、`soft_landing` −1e−5；`air_time`、`foot_clearance`、
`foot_swing_height` 已移除。

## 评估

play 模式把 `cts_student` 掩码全置 1，因此 `go2-play` 看到的是学生分支：

```bash
uv run go2-play Go2-Velocity-CTS \
  --checkpoint-file logs/rsl_rl/go2_cts/2026-10-31_10-43-13/model_9999.pt \
  --num-envs 4 --viewer viser

uv run python tools/eval_go2_task.py \
  --task Go2-Velocity-CTS \
  --checkpoint logs/rsl_rl/go2_cts/2026-10-31_10-43-13/model_9999.pt \
  --command 0.5 0 0 --steps 300
```

除 `error_vel_xy` 外要看 `student_reconstruction`：它只在学生掩码环境上计算，长期不降
说明学生编码器没跟上教师。课程进度看 `Curriculum/terrain_levels/mean` 与
`Curriculum/command_vel/*`。

## sim2sim

`--cts` 同时接受 run 目录下的 ONNX 和 `model_*.pt`（后者带 `student_encoder.*`，
两条路径跑出一致策略）：

```bash
uv run python sim2sim/go2_sim.py \
  --cts logs/rsl_rl/go2_cts/2026-10-31_10-43-13/2026-10-31_10-43-13.onnx \
  --terrain go2_rough --terrain-type all --terrain-level 6 \
  --terrain-column open_stairs --command 0.5 0 0
```

同一策略也可以直接把 `model_9999.pt` 传给 `--cts`（省掉 onnx 依赖）。

`sim2sim/README.md` 记录的实测（400 步，取后半段均值）：

| 地形 | 指令 | 实测 |
| --- | --- | --- |
| `go2_rough` 整图第 0 级 `flat` 格 | `0.5 0 0` | `vx +0.511`、误差 `0.011`、高度 `0.358 m` |
| 整图第 6 级 `open_stairs` 格 | `0.5 0 0` | `vx +0.521`、误差 `0.021`、高度 `0.932 m` |
| 整图第 4 级 `wave_terrain` 格 | `0.5 0 0` | `vx +0.473`、误差 `0.027` |
| 单列 `perlin_noise` 第 7 级 | `0.5 0 0` | `vx +0.471`、误差 `0.029` |
| 整图第 2 级 `flat` 格 | `0 0 0.5` | `wz +0.264`、误差 `0.236` |

## 已知限制

- 仓库只保留 `2026-10-31_10-43-13` 的 `model_9999.pt`；`.onnx` 只在保存时导出，只对应
  这个最后的 checkpoint，中间检查点没有导出可对照。
- 学生分支只在最后 25% 环境上采到 on-policy 数据，差距不体现在整体 `error_vel_xy` 里，
  必须单独看 `student_reconstruction` 或只跑学生分支的 sim2sim。
- 历史必须 term-major（角速度 5 帧 → 重力 5 帧 → 关节角 → 关节速度 → 动作 → 指令），
  顺序反了策略仍能动，但实测 `vy` 会从 `+0.001` 漂到 `+0.689`。
- sim2sim 偏航是弱项：指令 0.5 rad/s 只跟到 `+0.264`（同场景 Rough AMP 只有 `+0.013`），
  差距主要在独立引擎的执行器与摩擦标定；出生比例偏向楼梯/坡面，14 种地形仍各占一列。
- 执行器延迟上限 4 个物理步（80 ms），实机通信与滤波延迟更大时需重新确认。

## 参考

- [CTS 任务说明](../cts.md)、[Go2 训练与部署约定](../go2.md)
- [Go2 HIM 文档（同地形上的历史编码对照）](../go2_him.md)
- sim2sim CTS 入口与实测表：[`sim2sim/README.md`](../../sim2sim/README.md)
