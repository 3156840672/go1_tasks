# Go2 HIM（混合内部模型）

`Go2-Velocity-HIM` 按 HIMLoco（ICLR 2024）实现：可部署 actor 只吃最近 6 帧 45 维
本体观测（270 维，term-major 展开），内部估计器从这段历史回归 3 维机身线速度并编码
16 维隐变量（SwAV 32 原型），隐变量与本帧观测一起进策略。critic 保留 238 维特权
观测做非对称 PPO，因此推理时不需要任何特权信息。

## 任务 ID

| 项 | 值 |
| --- | --- |
| 任务 ID | `Go2-Velocity-HIM` |
| 环境配置 | `src/go2_tasks/env_cfgs.py::unitree_go2_him_env_cfg` |
| Runner | `VelocityOnPolicyRunner` + `go2_tasks.him.HIMPPO`，actor 为 `HIMActor` |
| 检查点目录 | `logs/rsl_rl/go2_him/<run>/`，仓库里有 `2026-11-07_13-21-07`（`model_5050.pt`）与 `2026-11-08_15-10-07`（`model_7400.pt`） |

## 观测与动作

| 项 | 维数 | 说明 |
| --- | ---: | --- |
| 观测布局 | 45 | 指令 3、角速度 3、投影重力 3、关节角 12、关节速度 12、上一动作 12 |
| 缩放 | — | 指令 ×`[2.0, 2.0, 0.25]`、角速度 ×0.25、关节速度 ×0.05 |
| actor | 270 | 6 帧 × 45，训练时按 term 展开加噪并展平 |
| critic | 238 | `proprio` 45 + `base_lin_vel` 3（×2.0）+ `base_lin_acc` 3 + `height_scan` 187 |
| 估计器输入 | 270 | 与 actor 同一段历史 |
| 估计器输出 | 3 + 16 | 线速度 3、隐变量 16（32 个原型、温度 3.0） |
| 策略头 | 61 → 12 | 45 维观测 + 16 维隐变量，MLP 512-256-128，ELU |
| 动作 | 12 | `q_des = q_default + scale · a`，scale = `[0.125, 0.25, 0.25]`，`clip_actions=100` |
| 控制频率 | 50 Hz | 物理步 0.005 s，decimation 4 |
| episode 长度 | 20 s | |
| 指令范围 | x ±1.0 → −1.5…2.0 → −2.0…3.0，y ±1.0，yaw ±0.5 rad/s | 3–8 s 重采样，10% 站立；另留 10% 环境做原地转向 `|yaw| ∈ [0.2, 0.5]` |

估计器的监督信号来自 critic 的固定切片：`critic[..., 45:48]` 是机身线速度真值，
`cat(critic[..., 3:45], critic[..., 45:48])` 是 successor state。因此 45 维
proprio 的 term 顺序不能改，改了估计器就学到错误的目标。

## 训练

```bash
uv run go2-train Go2-Velocity-HIM \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard
```

| 超参 | 值 |
| --- | --- |
| 网络 | 策略/价值 MLP 512-256-128，ELU；隐变量 16，原型 32 |
| lr | 策略 1e-3 adaptive；估计器 1e-3（跟随策略） |
| PPO | 5 epochs、4 mini-batches、clip 0.2、entropy 0.01、γ 0.99、λ 0.95、`clip_actions=100` |
| 估计器更新 | 4 epochs、批 512、grad-norm 10 |
| 采样 | 24 steps/env，`max_iterations=10000`，`save_interval=50` |
| 环境数 | 1024（默认） |

奖励：`track_linear_velocity` 4.0（σ²=0.25）、`track_angular_velocity` 2.0（函数
`track_yaw_velocity`，只取偏航分量，σ²=0.25）、`upright` 1.0（相对地形）、`pose` 1.0、
`dof_pos_limits` −1.0、`action_rate_l2` −0.04、`body_ang_vel` −0.2、
`linear_velocity_error_l2` −1.0、`foot_clearance` −0.1、`foot_slip` −0.1、
`soft_landing` −1e−5、三个碰撞项各 −0.1。

课程：地形与 rough 相同（14 种、10 行、初始等级 0–5，`terrain_levels_vel` 按位移
升降级）；速度课程只有 x 在 `5000×24`、`10000×24` 步两级扩张，y 与 yaw 不变，
所以偏航上限一直是 ±0.5 rad/s。检查点带 `him_contract_version=2`，加载器会拒绝
没有该标记的旧检查点。

## 评估

```bash
uv run go2-play Go2-Velocity-HIM \
  --checkpoint-file logs/rsl_rl/go2_him/2026-11-08_15-10-07/model_7400.pt \
  --num-envs 4 --viewer viser

uv run python tools/eval_go2_task.py \
  --task Go2-Velocity-HIM \
  --checkpoint logs/rsl_rl/go2_him/2026-11-08_15-10-07/model_7400.pt \
  --command 0.5 0 0 --steps 300
```

指标：`error_vel_xy`、`error_vel_yaw`、`terminated` 百分比、`mean height`；训练日志里
看 `Curriculum/terrain_levels/mean`、`Episode_Termination/*`、`estimator_velocity`（速度
回归误差）与 `estimator_swap`（SwAV 损失）。偏航要分别测 `+yaw` 与 `-yaw`，两侧不对称通常意味着原地转向样本不足。
`docs/go2_him.md` §10.4 记录过同口径对比（24 envs × 300 步）：HIM 0.176、
CTS 0.087、AMP 0.120，但轮数分别是 4000/10000/2650，只用于量级参照。

## sim2sim

`--him` 接受路径或 `latest`（取 `logs/rsl_rl/` 下最新 HIM `model_*.pt`）：

```bash
uv run python sim2sim/go2_sim.py --him latest \
  --terrain go2_rough --terrain-type all --terrain-level 6 \
  --command 0.5 0 0

uv run python sim2sim/go2_sim.py \
  --him logs/rsl_rl/go2_him/2026-11-08_15-10-07/model_7400.pt \
  --terrain flat --command 0.5 0 0 --headless --steps 400
```

程序维护最近 6 帧、最新帧在前的 45 维历史，复位时清空；观测顺序、缩放与
`him_actor_observation()` 一致。未缩放的旧 HIM 检查点会被拒绝。支持
`flat`/`rough`/`stairs`/`rl_sar`/`go2_rough` 五种地形。

## 已知限制

- 偏航指令上限只有 ±0.5 rad/s，远低于论文的高速转向范围；高速转向能力未训练。
- `docs/go2_him.md` 里的收敛曲线与对比数字对应的 run（`2026-11-06_21-47-59_rewfix`、
  `2026-11-07_09-34-01`）不在 `logs/rsl_rl/` 中，仓库只保留后两个 run 的检查点，
  这些数字不能与现有文件一一对应。
- 4000 轮时误差仍在缓慢下降，未到平台；与 CTS 的 10000 轮不可直接比较。
- sim2sim 的 `rough` 高度场比训练课程（约 5 级）更难，属分布外；平地 `vx=0.5`
  误差 0.096 m/s 属正常量级。
- 历史必须 term-major 且复位清空；顺序或缩放错一处，估计器会收敛到错误速度，
  表现为原地踏步或缓慢漂移。
- 训练显存随 270 维历史 + 238 维 critic 增长，1024 环境约多 50 MB 存储，环境数
  降到 512/256 时估计器收敛会更慢。
- 无硬件验证记录：本仓库没有实机测试数据。

## 参考

- [Go2 HIM 长文档（观测契约、估计器、同口径评测与旧实验记录）](../go2_him.md)
- [Go2 训练与部署约定](../go2.md)
- [CTS 文档（同为历史编码的可部署分支）](../cts.md)
- HIMLoco: *Learning Humanoid Locomotion with a Hybrid Internal Model*, ICLR 2024
- sim2sim HIM 入口：[`sim2sim/README.md`](../../sim2sim/README.md)
