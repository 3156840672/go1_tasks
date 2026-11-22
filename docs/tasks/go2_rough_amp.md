# Go2 崎岖地形 AMP

`Go2-Velocity-Rough-AMP` 把单阶段 AMP 搬到课程崎岖地形上：actor 仍是 45 维盲观测，
critic 用 259 维非对称观测，风格判别器与 PPO 同步训练。与 `Go2-Velocity-AMP` 的
差别是地形、地形课程和更保守的速度课程，与 `Go2-Velocity-Rough-AMP-Teacher`
的差别是没有教师-学生两阶段结构。

## 任务 ID

| 项 | 值 |
| --- | --- |
| 任务 ID | `Go2-Velocity-Rough-AMP` |
| 环境配置 | `src/go2_tasks/env_cfgs.py::unitree_go2_rough_amp_env_cfg` |
| Runner | `VelocityOnPolicyRunner` + `go1_tasks.amp.AmpPPO` |
| 检查点目录 | `logs/rsl_rl/go2_rough_amp/<run>/`，仓库里有 `2026-11-01_19-32-22`（含 `model_9999.pt`） |

## 观测与动作

| 项 | 维数 | 说明 |
| --- | ---: | --- |
| `base_ang_vel` | 3 | 噪声 ±0.2 rad/s |
| `projected_gravity` | 3 | 噪声 ±0.05 |
| `joint_pos` | 12 | 相对默认角 |
| `joint_vel` | 12 | 噪声 ±1.5 rad/s |
| `actions` | 12 | 上一控制步动作 |
| `command` | 3 | `[vx, vy, wz]` |
| **actor 合计** | **45** | 单帧，与部署契约一致 |
| critic 额外项 | +214 | `base_lin_vel` 3、`height_scan` 187、足端四项 24 |
| **critic 合计** | **259** | 无噪声 |
| 动作 | 12 | `q_des = q_default + scale · a`，scale = `[0.125, 0.25, 0.25]` |
| 控制频率 | 50 Hz | 物理步 0.005 s，decimation 4 |
| episode 长度 | 20 s | |
| 指令范围 | 见下方课程表 | 4–8 s 重采样，训练 5% 严格零指令，无航向控制 |

奖励沿用平地 AMP 的稀疏任务项，再把 `upright`、`foot_slip` 换成相对地形高度的
版本，并保留三个安全项：`track_linear_velocity` 2.0、`track_angular_velocity` 2.0
（σ²=0.15）、`upright` 0.2、`dof_pos_limits` −1.0、`action_rate_l2` −0.05、
`foot_slip` −0.05、`self_collisions`/`shank_collision`/`trunk_head_collision` 各 −0.1，
外加 `amp_reward_weight=2.0` 的风格奖励。

## 训练

```bash
uv run go2-train Go2-Velocity-Rough-AMP \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard
```

| 超参 | 值 |
| --- | --- |
| 网络 | actor/critic 512-256-128，ELU |
| lr | 1e-3 adaptive；判别器 1e-4 |
| PPO | 5 epochs、4 mini-batches、clip 0.2、entropy 0.01、γ 0.99、λ 0.95 |
| 采样 | 24 steps/env，`max_iterations=10000`，`save_interval=50`，环境数默认 1024 |
| 判别器 | `92 → 256 → 128 → 1`，ReLU，每轮 8 批、批 1024、梯度惩罚 10、replay 100 000 |

课程与随机化：

- **速度**：0 步 x ±0.3、y ±0.2 m/s、yaw ±0.55 rad/s；`3500×24` 步扩到 ±0.6、±0.4、
  ±0.75；`6000×24` 步到 ±1.0、±0.6、±1.0。起始范围按参考片段实测速度设定，
  让判别器不能只靠“速度快慢”分辨真假。
- **地形**：14 种地形、10 行难度、初始等级固定 0；`random_rough` 换成
  `CurriculumRandomRoughTerrainCfg`（噪声幅度按难度从 12.5% 缩放到 100%），
  升降级仍用位移口径的 `terrain_levels_vel`。
- **AMP**：参考动作 `logs/amp/go2_reference`（`GO2_AMP_REFERENCE_PATH`，17 段），
  50% reset 用参考帧初始化；`zero_command_amp_threshold=0`。
- **域随机化**：在 rough 已有的编码器偏置、足底摩擦、躯干质心之外，追加惯量
  扰动、`Kp/Kd` ×0.9–1.1、关节摩擦 0–0.05，推力扰动改为 4–8 s 一次、±0.3 m/s
  与 ±0.3 rad/s，初始关节角/速度 ±0.1。

## 评估

```bash
uv run go2-play Go2-Velocity-Rough-AMP \
  --checkpoint-file logs/rsl_rl/go2_rough_amp/2026-11-01_19-32-22/model_9999.pt \
  --num-envs 4 --viewer viser

uv run python tools/eval_go2_task.py \
  --task Go2-Velocity-Rough-AMP \
  --checkpoint logs/rsl_rl/go2_rough_amp/2026-11-01_19-32-22/model_9999.pt \
  --command 0.5 0 0 --steps 300
```

指标：`error_vel_xy`、`terminated`、训练日志里的 `Curriculum/terrain_levels/mean`
以及 `amp_expert_logit` / `amp_policy_logit`。地形能力不能只看短时摔倒率，要在
固定等级上分别测前进、后退、横移和偏航。

## sim2sim

45 维检查点走 `--policy`，`go2_rough` 就是训练地形导出：

```bash
uv run python sim2sim/go2_sim.py \
  --policy logs/rsl_rl/go2_rough_amp/2026-11-01_19-32-22/model_9999.pt \
  --terrain go2_rough --terrain-type all --terrain-level 6 \
  --terrain-column open_stairs --command 0.5 0 0
```

`sim2sim/README.md` 记录的同检查点 400 步实测（取后半段均值）：

| 场景 | 实测 |
| --- | --- |
| 4×4 整图第 6 级 `open_stairs` 格，指令 `0.5 0 0` | `vx +0.374`、误差 `0.126`、高度 `0.934 m` |
| 整图第 0 级 `flat` 格，指令 `0.4 0 0` | `vx +0.275`、误差 `0.125`、高度 `0.346 m` |
| 单列 `discrete_obstacles` 第 5 级，指令 `0.4 0 0` | `vx +0.188`、误差 `0.212` |

也支持 `flat`/`rough`/`stairs`/`rl_sar`；`--terrain-seed` 只影响生成地形。

## 已知限制

- sim2sim 里偏航明显偏弱：同一场景该检查点 `wz` 只有 `+0.013`（指令 0.5），
  与训练日志的偏航跟踪不能直接对照，差距主要在执行器/摩擦标定。
- 第 6 级 `open_stairs` 的 0.126 m/s 误差和 0.934 m 高度来自出生格本身被抬高，不能与
  平地数字横向比较；参考集只有低速独立方向片段，高速复合指令没有风格支撑。
- 每轮只更新 8 个判别器批次、参考集只有 3825 个转移，判别器容易饱和；训练早期应
  观察风格奖励是否已经接近 0。
- 地形等级的升降仍按位移判定，转向和横移指令下升级偏慢，与论文式教师的跟踪质量
  课程不同；仓库只有一个 9999 轮的目录，没有多种子或消融记录。

## 参考

- [Go2 训练与部署约定](../go2.md)、[Go2 论文式 AMP 与 LSTM 学生](../go2_paper_amp.md)
- [AMP 训练教程](../amp_tutorial.md)
- sim2sim 实测与地形选格：[`sim2sim/README.md`](../../sim2sim/README.md)
