# Go2 平地 AMP 速度跟踪

`Go2-Velocity-AMP` 在平地上用单阶段 AMP 学 Trot 风格：actor 是 45 维本体观测
（与 Rough 同布局、不含机身线速度），奖励由速度跟踪 + 判别器风格奖励组成，
没有教师网络。与 `Go2-Velocity-Rough-AMP` 的差别是地形为平面、critic 无高度扫描；
与 `Go2-Velocity-Rough` 的差别是多了 AMP 判别器与参考状态初始化。

## 任务 ID

| 项 | 值 |
| --- | --- |
| 任务 ID | `Go2-Velocity-AMP` |
| 环境配置 | `src/go2_tasks/env_cfgs.py::unitree_go2_amp_env_cfg` |
| Runner | `VelocityOnPolicyRunner` + `go1_tasks.amp.AmpPPO` |
| 检查点目录 | `logs/rsl_rl/go2_amp/<run>/`，仓库里有 `2026-10-30_12-02-41`（最新 `model_2650.pt`） |

## 观测与动作

| 项 | 维数 | 说明 |
| --- | ---: | --- |
| `base_ang_vel` | 3 | 噪声 ±0.2 rad/s |
| `projected_gravity` | 3 | 噪声 ±0.05 |
| `joint_pos` | 12 | 相对默认角 |
| `joint_vel` | 12 | 噪声 ±1.5 rad/s |
| `actions` | 12 | 上一控制步动作 |
| `command` | 3 | `[vx, vy, wz]` |
| **actor 合计** | **45** | 单帧（`history_length=1`），与部署契约一致 |
| critic 额外项 | +27 | `base_lin_vel` 3、足端四项 24 |
| **critic 合计** | **72** | 平地任务无高度扫描 |
| 动作 | 12 | `q_des = q_default + scale · a`，scale = `[0.125, 0.25, 0.25]` |
| 控制频率 | 50 Hz | 物理步 0.005 s，decimation 4 |
| episode 长度 | 20 s | |
| 指令范围 | x ±1.0、y ±0.6 m/s、yaw ±1.0 rad/s | 4–8 s 重采样；训练 5% 严格零指令，play 为 0 |
| 航向控制 | 关闭 | `heading_command=False`，`rel_heading_envs=0` |

没有速度指令课程（`command_vel` 被移除）。奖励：`track_linear_velocity` 2.0
（σ²=0.25）、`track_angular_velocity` 2.0（σ²=0.15）、`upright` 0.2、
`dof_pos_limits` −1.0、`action_rate_l2` −0.05、`foot_slip` −0.05，再加
`amp_reward_weight=2.0` 的风格奖励（由 RewardManager 乘控制步长 0.02 s）。

## 训练

```bash
uv run go2-train Go2-Velocity-AMP \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard
```

| 超参 | 值 |
| --- | --- |
| 网络 | actor/critic 512-256-128，ELU |
| lr | 1e-3 adaptive；判别器 1e-4 |
| PPO | 5 epochs、4 mini-batches、clip 0.2、entropy 0.01、γ 0.99、λ 0.95 |
| 采样 | 24 steps/env，`max_iterations=10000`，`save_interval=50`，环境数默认 1024 |

AMP 部分：

- 参考动作 `GO2_AMP_REFERENCE_PATH = logs/amp/go2_reference`，递归读取 11 个类别共
  17 段 OCS2 片段（Trot、Flying Trot、Pace、两种 walk、左右转、左右横移、后退、
  站立），50 Hz；`docs/go2.md` 记录启动日志为 `17 clips, 3825 transitions`。
- 判别器输入是相邻两帧 46 维扩展状态（相对地形高度 1、投影重力 3、机体系线速度 3、
  角速度 3、关节角 12、关节速度 12、足端相对位置 12），拼成 92 维；结构
  `92 → 256 → 128 → 1`，ReLU，LSGAN 目标 +1/−1。
- 每轮 8 个判别器批次、批大小 1024、梯度惩罚 10、replay 容量 100 000，判别器
  1 epoch；风格奖励权重 2.0，`zero_command_amp_threshold=0`（零指令样本也参与风格奖励）。
- 参考状态初始化：50% 的 reset 用参考帧替换默认初始状态（`amp_reference_state_init`），
  只在训练时启用。

## 评估

```bash
uv run go2-play Go2-Velocity-AMP \
  --checkpoint-file logs/rsl_rl/go2_amp/2026-10-30_12-02-41/model_2650.pt \
  --num-envs 4 --viewer viser

uv run python tools/eval_go2_task.py \
  --task Go2-Velocity-AMP \
  --checkpoint logs/rsl_rl/go2_amp/2026-10-30_12-02-41/model_2650.pt \
  --command 0.8 0 0 --steps 300
```

除了 `error_vel_xy` / `error_vel_yaw`，训练日志里要同时看 `amp_expert_logit`、
`amp_policy_logit` 和 `amp_style_reward_raw`：前两者长期贴在 +1/−1 且风格奖励接近
0 说明判别器饱和，这时步态是任务奖励驱动的，继续训练也不会更像参考动作。

## sim2sim

45 维检查点走 `--policy`，地形默认 `flat`：

```bash
uv run python sim2sim/go2_sim.py \
  --policy logs/rsl_rl/go2_amp/2026-10-30_12-02-41/model_2650.pt \
  --terrain flat --command 0.8 0 0

uv run python sim2sim/go2_sim.py \
  --policy logs/rsl_rl/go2_amp/2026-10-30_12-02-41/model_2650.pt \
  --terrain flat --command 0 0 0.5 --headless --steps 400
```

支持五种地形，但 AMP 只在平面上训练，`go2_rough` 等属于分布外。无窗口输出给的是
后半段平均 `vx/vy/wz` 和绝对误差。

## 已知限制

- 仓库里的 `2026-10-30_12-02-41` 只训到 2650 轮（`model_2650.pt` 是最后一个），
  没有收敛记录，`docs/cts.md` 引用的 0.120 误差对应这个轮数，不宜当作该任务的
  上限。
- 参考集来自 OCS2 的实际轨迹，文件名里的速度大于实测值（前进约 0.25 m/s、
  横向约 0.19 m/s、偏航约 0.52 rad/s），风格先验只在低速独立方向上有支撑。
- 判别器显式看到机身线速度，参考分布外的高速/复合指令会更容易被判定为“假”，
  风格奖励随速度升高而减弱。
- sim2sim 里偏航是弱项：同一场景 AMP 检查点的 `wz` 只有 `+0.013`（指令 0.5），
  见 [`sim2sim/README.md`](../../sim2sim/README.md)，主要差距在独立引擎的执行器与
  摩擦标定。
- 参考动作目录必须只放 Go2 片段（读取器递归加载所有 `.npz`）；风格奖励不等于步态
  验收，判别器输出饱和时策略可能仍能走，但不像参考 Trot。

## 参考

- [Go2 训练与部署约定（含 OCS2 采集流程与类别表）](../go2.md)
- [AMP 训练教程](../amp_tutorial.md)、[AMP 代码教程](../amp_code_tutorial.md)
- [参考动作采集与整理](../amp_mob_collection.md)
- sim2sim 用法：[`sim2sim/README.md`](../../sim2sim/README.md)
