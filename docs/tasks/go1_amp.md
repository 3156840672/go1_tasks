# Go1 AMP 风格奖励 Trot

用对抗式运动先验（AMP）替代手写步态奖励：判别器区分"策略转移"与"MoB Trot 参考片段"，
风格奖励与跟踪奖励相加，策略从随机初始化开始学。和 `Go1-Velocity-MoB` 的区别是丢掉行为
指令与相位输入、参考动作只作奖励来源；和 `Go1-Velocity-Flat` 的区别是没有机身线速度
（45 维 vs 48 维），多了判别器与参考状态初始化。

## 任务 ID

| 项 | 值 |
| --- | --- |
| 任务 ID | `Go1-Velocity-AMP`（没有历史 ID，注册表里只有这一个名字） |
| 环境配置 | `src/go1_tasks/env_cfgs.py::unitree_go1_amp_env_cfg`（由 MoB 配置派生，只改观测/指令/奖励） |
| Runner | `VelocityOnPolicyRunner`，算法为 `go1_tasks.amp.AmpPPO`（PPO + 判别器） |
| 检查点目录 | `logs/rsl_rl/go1_amp/<run>/`（`experiment_name=go1_amp`） |

## 观测与动作

| 项 | 值 |
| --- | --- |
| actor 观测 | 45 维、`history_length=1`：base_ang_vel 3 + projected_gravity 3 + joint_pos 12 + joint_vel 12 + actions 12 + command 3 |
| critic 观测 | 72 维：actor 的 45 维（无噪声 + base_lin_vel 3 = 48）+ foot_height 4 + foot_air_time 4 + foot_contact 4 + foot_contact_forces 12 |
| 判别器输入 | `extended` 模式：单帧 46 维、单转移 92 维（含动作与足端接触特征），不是论文的 31 维 `paper` 模式 |
| 动作 | 12 维关节位置偏移；hip 0.125、thigh/calf 0.25；`Kp=20`、`Kd=0.5` |
| 控制频率 | 50 Hz（物理步长 0.005 s，decimation 4） |
| episode | 训练 20 s；play 模式 1e9 s |
| 指令范围 | 只给正向：lin_vel_x 0.2–0.8 m/s、lin_vel_y = 0、ang_vel_z = 0；重采样 4–8 s，无站立/朝向/前进子集 |

## 训练

参考动作来自 `logs/amp/mob_reference/trot`（4 个速度目录 `vx_+0p20`/`+0p40`/`+0p60`/
`+0p80`，每个 3 段 `clip_000..002.npz`，每段 251 帧 = 5 s @ 50 Hz + 1）。换目录用
`--agent.algorithm.reference-path`：

```bash
uv run go1-train Go1-Velocity-AMP \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard \
  --agent.upload-model False
```

配置默认 `num_envs=1024`，现有 `logs/rsl_rl/go1_amp/2026-10-29_09-53-48_lsgan_norm` 在
5500 轮手动停止（`save_interval` 覆盖为 250）。

| 超参 | 值 |
| --- | --- |
| actor / critic | 512-256-128，ELU |
| 学习率 | 1e-3 adaptive；判别器 1e-4 |
| 判别器 | 隐藏层 (256, 128)、ReLU、1 epoch、batch 1024、每轮 8 个 batch、梯度罚 10.0、replay 100000 |
| 判别器损失 | LSGAN 目标 ±1，`state_mode=extended`，margin 0、权重衰减 0 |
| AMP 奖励 | `2.0 × dt × clamp(1 - 0.25·(logit-1)², min=0)`，与任务奖励直接相加 |
| 迭代数 | 10000；每环境每轮 24 步 |
| save_interval | 250（本次 run）；配置默认 50 |

任务奖励只剩 5 项：track_linear_velocity 2.0、upright 0.2、dof_pos_limits -1.0、
action_rate_l2 -0.05、foot_slip -0.05。reset 时 `amp_reference_state_init` 以 0.5 的概率
把默认初始状态替换成随机参考帧，让策略早期就能看到合法的 Trot 状态。域随机化沿用 MoB：
摩擦、编码器偏置、COM、惯量 ±5%、PD 增益 ±10%、关节摩擦、4–8 s 推挤。没有课程。

## 评估

```bash
uv run go1-play Go1-Velocity-AMP \
  --checkpoint-file logs/rsl_rl/go1_amp/2026-10-29_09-53-48_lsgan_norm/model_5500.pt \
  --viewer viser
```

播放只加载 actor，不读参考文件；`--video` 可导出 mp4。定量对比用风格指标脚本，它把策略的
抬脚高/对角相关/相邻相关与同速度参考片段放在一起比：

```bash
uv run --no-sync python -m go1_tasks.eval_amp \
  logs/rsl_rl/go1_amp/2026-10-29_09-53-48_lsgan_norm/model_5500.pt \
  --vx 0.6 --seconds 5
```

`model_5500` 的实测（来源 [`docs/amp_code_tutorial.md`](../amp_code_tutorial.md) §8.3）：

| 命令 vx | 实际 vx | 抬脚高 FR/FL/RR/RL (m) | 对角相关 | 相邻相关 |
| ---: | ---: | --- | ---: | ---: |
| 0.2 | 0.164 | 0.091/0.087/0.077/0.084 | 0.985 | -0.660 |
| 0.4 | 0.364 | 0.087/0.085/0.085/0.087 | 0.987 | -0.706 |
| 0.6 | 0.578 | 0.088/0.081/0.075/0.090 | 0.987 | -0.749 |
| 0.8 | 0.808 | 0.094/0.079/0.087/0.094 | 0.985 | -0.818 |

同速度参考片段的抬脚高约 0.08 m、对角相关 0.99、相邻相关 -0.79。四档都复现了
FR–RL / FL–RR 对角同相、相邻交替的 Trot；0.2–0.6 m/s 实际速度偏低 0.02–0.04 m/s，
0.8 m/s 偏高 0.008 m/s。

## sim2sim

**不支持。** `sim2sim/go1_sim.py` 没有 AMP 入口：它的 `--policy` 只接受 48 维的 flat
格式，把 AMP 的 45 维检查点传进去会被维度校验拒绝；`--cts`、`--teacher/--adapter`
也都不适用于纯 PPO + 判别器结构。AMP 的 sim2sim 只对 Go2 实现（`go2_sim.py --policy`
自动按第一层 45/48 区分）。要在 MuJoCo 里看 Go1 Trot，只能用 `go1_mob_sim.py` 跑生成
参考动作的 MoB 检查点，那不是这个任务的策略。

## 已知限制

- 参考动作只覆盖正向前进 0.2–0.8 m/s，策略没有转向、横移、后退、跌倒恢复的风格样本。
- 只在平地训练；上坡楼梯没有参考数据，也无法用 `go1_mob_sim.py` 之外的工具验证。
- 参考片段由 MoB 检查点（`go1_mob/2026-10-27_15-04-14` 的 5050 轮）采集，风格上限受
  其步态质量限制；换参考目录后需要重新训练，旧判别器状态不再匹配。
- 训练在 5500/10000 轮手动停止，仓库里没有记录停止原因与收敛判据。
- 判别器、特征归一化和 replay 状态都随检查点保存，resume 时若同时换参考数据，需要
  `--agent.algorithm.reset-discriminator-on-resume True` 才能清掉旧判别器。

## 参考

- [`docs/amp_tutorial.md`](../amp_tutorial.md)（任务搭建与训练入口）
- [`docs/amp_code_tutorial.md`](../amp_code_tutorial.md)（判别器实现、实测表、调参手册）
- [`docs/amp_mob_collection.md`](../amp_mob_collection.md)（参考动作采集与可视化）
- 论文：Peng, Ma, Abbeel, Levine, Kanazawa, *AMP: Adversarial Motion Priors for
  Stylized Physics-Based Character Control*, SIGGRAPH 2021 —
  <https://xbpeng.github.io/projects/AMP/AMP_2021.pdf>
- [相邻任务：Go1 MoB](go1_mob.md)、[Go1 平地](go1_flat.md)
