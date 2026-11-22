# Go2 行为条件运动（MoB）

`Go2-Velocity-MoB` 沿用 Walk These Ways 的行为条件思路：除了 3 维速度指令，策略还
接收 8 维行为指令（相位、bound、pace、步频、机身高度、俯仰、站距、抬腿高度）和
4 维内部时钟，用 30 帧历史推断足端接触，从而在同一个网络里切换 Trot/Bound/Pace
等步态。与 `Go2-Velocity-Flat` 的差别是输入改成 1710 维历史、奖励换成 MoB 复合项，
并新增网格自适应速度课程。

## 任务 ID

| 项 | 值 |
| --- | --- |
| 任务 ID | `Go2-Velocity-MoB` |
| 环境配置 | `src/go2_tasks/env_cfgs.py::unitree_go2_mob_env_cfg`（在 Go1 MoB 配置上套用 Go2 机器人） |
| Runner | `MobOnPolicyRunner`（`mjlab.tasks.velocity.rl`，PPO） |
| 检查点目录 | `logs/rsl_rl/go2_mob/<run>/`（仓库里没有该目录） |

## 观测与动作

actor 是历史组：每帧 57 维，30 帧，按 term 展开后共 1710 维。

| 项 | 每帧维数 | 说明 |
| --- | ---: | --- |
| `base_ang_vel` | 3 | |
| `projected_gravity` | 3 | |
| `joint_pos` | 12 | |
| `joint_vel` | 12 | |
| `actions` | 12 | 上一控制步动作 |
| `command` | 3 | `[vx, vy, wz]` |
| `behavior` | 8 | `[phase, bound, pace, frequency, body_height, body_pitch, stance_width, swing_height]` |
| `clock` | 4 | 由行为相位生成的 4 维节律信号 |
| **actor 合计** | **1710** | 57 × 30 帧，训练时按 term 加噪 |
| critic | 84 | 72（含机身线速度与足端四项）+ `behavior` 8 + `clock` 4 |
| 动作 | 12 | `q_des = q_default + scale · a`，scale = `[0.125, 0.25, 0.25]` |
| 控制频率 | 50 Hz | 物理步 0.005 s，decimation 4 |
| episode 长度 | 20 s | |
| 速度指令 | x ±1.0、y ±0.6 m/s、yaw ±1.0 rad/s | 无航向控制；由网格课程决定各方向采样概率 |
| 行为指令 | 3–8 s 重采样 | `frequency` 1.5–4.0 Hz、`body_height` 0.20–0.38 m、`body_pitch` ±0.15 rad、`stance_width` 0.22–0.40 m、`swing_height` 0.04–0.18 m |

`_apply_go2_profile` 会把机器人换成 Go2 的 MJCF，因此最终 PD 是 `Kp=25`、`Kd=0.5`
（Go1 MoB 配置里的 `Kp=20` 被替换），力矩上限 23.5 Nm、速度上限 30 rad/s。
没有执行器延迟。

## 训练

```bash
uv run go2-train Go2-Velocity-MoB \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard
```

| 超参 | 值 |
| --- | --- |
| 网络 | actor/critic 512-256-128，ELU，不做观测归一化 |
| lr | 1e-3，`adaptive`，`desired_kl=0.01` |
| PPO | 5 epochs、4 mini-batches、clip 0.2、entropy 0.01、γ 0.99、λ 0.95 |
| 采样 | 24 steps/env，`max_iterations=10000`，`save_interval=50`，环境数默认 1024 |

课程与随机化：

- **速度网格**：`GridAdaptiveVelocityCommandCfg`，`[vx, vy, wz]` 各切 11×7×11 格，
  初始只开放 ±0.4/±0.3/±0.4；每格记录线速度与偏航跟踪分数，线速度 ≥0.8、偏航
  ≥0.5 时把相邻格的采样权重 +0.2（上限 1.0），最短 1 s 重采样一次。权重写入
  checkpoint，`--agent.resume` 会一起恢复；日志里看
  `Metrics/twist/curriculum_coverage`。
- **地形**：无限平面，无地形课程。
- **域随机化**：伪惯量 α ±0.05、`Kp/Kd` ×0.9–1.1、关节摩擦 0–0.05、初始关节角/速度
  ±0.1；推力扰动每 4–8 s 一次，±0.3 m/s、±0.3 rad/s。

奖励只有一个 `mob` 项（权重 1.0），内部是「任务项 × exp(辅助项)」：

- 任务项：`exp(−‖v_xy_cmd − v_xy‖² / 0.25) + 0.5·exp(−(w_z_cmd − w_z)² / 0.25)`。
- 辅助项：垂直速度 −0.02、roll/pitch 角速度 −0.001、支撑相足端滑移 −0.04、
  关节力矩 −1e−4、关节速度 −1e−4、关节加速度 −2.5e−7、软限位越界 −10、
  动作一阶/二阶差分各 −0.1、四肢碰撞 −5。
- 步态项：支撑相足端速度 −4、摆动相触地 −4、机身高度误差 −80、俯仰误差 −5、
  Raibert 落足点误差 −10、摆动高度误差 −30。

## 评估

```bash
uv run go2-play Go2-Velocity-MoB \
  --checkpoint-file logs/rsl_rl/go2_mob/YOUR_RUN/model_9999.pt \
  --num-envs 4 --viewer viser

uv run python tools/eval_go2_task.py \
  --task Go2-Velocity-MoB \
  --checkpoint logs/rsl_rl/go2_mob/YOUR_RUN/model_9999.pt \
  --command 0.6 0 0 --steps 300
```

除速度误差外重点看 `Episode_Metrics/body_height_error`（行为指令高度与实际机身高度
之差）和 `Metrics/twist/curriculum_coverage`（网格被点开的比例）。行为指令的效果要把
`frequency`、`body_height`、`stance_width` 分别扫一遍，单看速度误差无法判断步态
是否跟随。

## sim2sim

仓库里没有 Go2 MoB 的 sim2sim 入口：`sim2sim/go2_sim.py` 只接受 45/48 维
`--policy`、`--student`、`--him`、`--cts`，没有 1710 维的 `--mob`；`go1_mob_sim.py`
加载的是 Go1 的 MJCF 与 `Kp=20`，不能用来验证 Go2 检查点。要验证只能在
`go2-play` 里看，或另写一个按 30 帧 term-major 历史喂观测的入口。

## 已知限制

- 仓库里没有 `logs/rsl_rl/go2_mob/` 运行目录，也没有 Go2 MoB 的 sim2sim 入口，本任务
  的输入契约只有配置和代码可查，没有本地检查点或实测数据佐证。
- 1710 维观测要求按 term 展开的 30 帧历史，且帧内顺序、复位清空方式必须与训练一致；
  顺序错了策略仍能输出动作，但步态和高度会明显漂移。
- 行为指令与速度指令相互耦合：站距、俯仰、抬腿高度同时改变时，同一个速度误差
  不能说明跟踪能力，必须按时钟相位对齐后再比较。
- 网格课程依赖跟踪分数，早期偏航跟踪差会把偏航方向的格子锁住，导致后期只剩
  前进/后退；恢复训练时若 checkpoint 里的网格与配置尺寸不符会直接报错。
- 平地训练，`rough`/`stairs` 属于分布外；没有地形扫描，也没有教师-学生结构。
- 行为奖励是复合项乘性叠加，权重（如 −80 的高度误差）是按参考实现标定的，
  改动其中一项会同时改变步态与速度跟踪的平衡。

## 参考

- [Go2 训练与部署约定](../go2.md)
- [MoB 任务说明](../mob.md)
- [AMP 与 MoB 参考动作采集](../amp_mob_collection.md)
- [Go2 平地速度跟踪（无行为指令的对照任务）](go2_flat.md)
