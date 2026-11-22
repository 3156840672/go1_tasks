# Go2 论文式 Rough AMP 教师与 LSTM 学生

两阶段任务：教师按 Wu 等人的 RA-L 2023 结构训练（45 维本体 + 30 维特权 + 187 维地形，
AMP 判别器只看 31 维单帧状态），再用 DAgger 蒸馏成只吃 45 维本体观测、带 256 维 LSTM
记忆的学生；与 `Go2-Velocity-Rough-AMP` 的差别是多了特权/地形编码器与可导出的学生。

## 任务 ID

| 项 | 值 |
| --- | --- |
| 任务 ID | `Go2-Velocity-Rough-AMP-Teacher` |
| 环境配置 | `unitree_go2_paper_amp_baseline_env_cfg`（教师注册与地形评测）；`unitree_go2_paper_amp_env_cfg`（`baseline=False`，学生蒸馏与 `evaluate_go2_paper_amp.py`） |
| Runner | `VelocityOnPolicyRunner` + `AmpPPO`，actor 为 `go2_tasks.paper_amp.PaperAmpTeacherActor` |
| 检查点目录 | 教师 `logs/rsl_rl/go2_paper_amp_teacher/<run>/`，仓库里有 `2026-11-05_11-08-59`（最新 `model_4650.pt`）；学生 `logs/rsl_rl/go2_paper_amp_student/<run>/`，仓库里没有该目录 |

## 观测与动作

`obs_groups` 把 actor 映射到三个组，critic 仍是单组 259 维。

| 项 | 维数 | 内容 |
| --- | ---: | --- |
| actor 组 `actor` | 45 | 角速度 3、投影重力 3、关节角 12、关节速度 12、上一动作 12、指令 3 |
| actor 组 `privileged` | 30 | 机体系线速度 3、足端接触力 12（`sign·log1p`）、触地 4、足底摩擦 3、质量偏差 1、质心偏移 3、电机强度 1、碰撞指示 3 |
| actor 组 `terrain` | 187 | 1.6×1.0 m、0.1 m 网格的高度扫描 |
| **教师输入** | **45+30+187** | 经两个编码器压成 8 + 16 维 |
| critic | 259 | 45 + 线速度 3 + 扫描 187 + 足端 24 |
| 学生输入 | 45 | 另带 LSTM hidden/cell 各 256 |
| 动作 | 12 | `q_des = q_default + scale · a`，scale = `[0.125, 0.25, 0.25]`；50 Hz、20 s |
| 指令（注册 baseline） | ±0.3 / ±0.2 / ±0.55 | 4–8 s 重采样，5% 严格零指令，无航向控制 |
| 指令（非 baseline 环境） | ±1.0 / ±0.6 / ±1.0 | 20 s 固定重采样 + 100% 航向控制 |

## 训练

```bash
uv run go2-train Go2-Velocity-Rough-AMP-Teacher \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard \
  --agent.upload-model False

uv run python -m go2_tasks.train_terrain_amp --iterations 8000 --num-envs 1024 --seed 42
```

从旧判别器恢复时加 `--agent.resume True --agent.load-run <目录> --agent.load-checkpoint
model_<轮数>.pt --agent.algorithm.reset-discriminator-on-resume True`；
`--agent.max-iterations` 是追加轮数，不是总轮数。

| 超参 | 教师 |
| --- | --- |
| 编码器 | 30→64→32→8、187→256→128→16（ELU） |
| 策略 | 69→256→128→64→12，ELU；`bound_action_mean=False`，当前输出层是线性的 |
| PPO | critic 512-256-128，lr 1e-3 adaptive，5 epochs、4 mini-batches、γ 0.99、λ 0.95、entropy 0.01 |
| 采样 | 24 steps/env，`max_iterations=10000`，`save_interval=50` |
| 判别器 | `62 → 1024 → 512 → 1`，ELU，每轮 2 批、批 1024，trunk 权重衰减 1e-3、head 1e-1 |

- **参考动作**：`logs/amp/go2_paper_reference`，13 段（六类方向各 2 段 + 1.6 s 站立），
  每段取中间 2.5 s，合计 1580 个转移（1593 帧、31.9 s），站立占 5.06%。判别器只看 31 维
  （关节角 12、关节速度 12、机身线速度 3、角速度 3、相对地形高度 1），风格权重 2.0。
- **地形课程**：14 种地形、初始等级 0；`terrain_levels_from_tracking` 要求命令平移
  速度 > 0.1 m/s、没早摔、线速度跟踪 ≥ 0.85 且已走出格子中心才升级，跟踪 < 0.5 或
  位移不足一半降级，零速与纯转向不参与。
- **速度课程**：baseline 保留三段（0 步、`3500×24`、`6000×24`）；非 baseline 环境移除
  `command_vel`，直接用全域指令。

学生（DAgger，教师冻结）：

```bash
uv run go2-train-paper-amp-student \
  --teacher logs/rsl_rl/go2_paper_amp_teacher/2026-11-05_11-08-59/model_4650.pt \
  --num-envs 256 --iterations 4000 --sequence-length 50 --epochs 4 --batch-size 32
```

学生只训 `MSE(action, teacher_action) + latent_loss_weight · MSE(latent, teacher_latent)`
（默认 lr 5e-4、latent 权重 1.0、每 50 轮保存），不更新教师、不训判别器、不算风格
奖励，低层网络先从教师复制。输出 `student_<iter>.pt` 与同名 `.onnx`。

## 评估

```bash
uv run go2-play Go2-Velocity-Rough-AMP-Teacher \
  --checkpoint-file logs/rsl_rl/go2_paper_amp_teacher/2026-11-05_11-08-59/model_4650.pt \
  --num-envs 4 --viewer viser
```

地形脚本 `tools/evaluate_go2_paper_terrain.py <ckpt> --levels 0 2 4 --output x.json`
枚举 14 种地形 × 指定等级 × `--repeats` 个实例，前进 0.4 m/s，走出 4 m 记为穿越。
论文命令集的固定指令评测用 `tools/evaluate_go2_paper_amp.py <ckpt> --output x.json`。

另有两个只对教师有效的脚本（默认 `--device cuda:0`，无卡时传 `cpu`）：
`record_go2_paper_amp_preview.py <ckpt> --preview-command {stand,forward,turn} --output
x.json --video-output x.mp4`，以及按 `flat`/`rough` × 3 个种子评测的
`train_validate_go2_paper_amp.py`。

教师还要看 `amp_expert_logit` / `amp_policy_logit` 是否饱和、`amp_style_reward_raw` 是否
接近 0、`Curriculum/terrain_levels/mean` 是否上升；学生只看 `action_mse` / `latent_mse`。

## sim2sim

教师不支持：它需要 30 维仿真特权和 187 维 raycast 扫描，独立 MJCF 没有这些传感器。
学生走 `--student`：

```bash
uv run python sim2sim/go2_sim.py \
  --student logs/rsl_rl/go2_paper_amp_student/YOUR_RUN/student_4000.pt \
  --terrain rough --command 0.4 0 0 --headless --steps 2000
```

程序维护 256 维 hidden/cell，复位清零；可用地形 `flat`/`rough`/`stairs`/`rl_sar`/
`go2_rough`。ONNX 三输入（`proprio`、`hidden`、`cell`）三输出（`actions`、
`next_hidden`、`next_cell`），实机每 20 ms 把 `next_*` 原样回灌，急停/复位清零。

## 已知限制

- 仓库里的教师只到 `model_4650.pt`，学生目录不存在，蒸馏链路没有被完整跑通的证据。
- 30 维特权的字段与顺序是项目自定义的；判别器只给 31 维状态，与
  `Go2-Velocity-Rough-AMP` 的 92 维扩展判别器不可互换。
- 参考集只有 1580 个转移（1593 帧、31.9 s），且为低速独立方向；论文是 30 s Trot-only。
- `bound_action_mean` 目前为 `False`，输出层没有论文 Table III 的 `tanh`（恢复旧检查点
  时从 `amp_bound_action_mean` 读取）；学生用非 baseline 环境采样，能力以学生评测为准。

## 参考

- [Go2 论文式 Rough AMP 与 LSTM 学生长文档](../go2_paper_amp.md)、[Go2 训练与部署约定](../go2.md)
- Wu et al., *Learning Robust and Agile Legged Locomotion Using Adversarial Motion
  Priors*, RA-L 2023；sim2sim 学生入口见 [`sim2sim/README.md`](../../sim2sim/README.md)
