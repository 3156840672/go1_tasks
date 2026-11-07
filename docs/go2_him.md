# Go2 上的 Hybrid Internal Model（HIMLoco, ICLR 2024）

本任务把 *Hybrid Internal Model: Learning Agile Legged Locomotion with
Simulated Robot Response*（Long et al., ICLR 2024）移植到本仓库的 mjlab
Go2 崎岖地形速度任务上，任务 ID 为 `Go2-Velocity-HIM`。

原实现基于 Isaac Gym + 旧版 rsl_rl（`OpenRobotLab/HIMLoco`），本仓库运行在
mjlab + rsl-rl 5.x 上。**方法的核心信息流逐项保留**，与参考代码的差异集中在
§6，主要由仿真器与训练框架不同造成。

## 1. 方法在做什么

HIM 的出发点是内部模型控制（IMC）：控制器不必显式识别外界扰动，只要能预测
"系统对外部输入的响应"，就可以从预测与实际响应的偏差中估计扰动。

四足机器人上做两点改写：

1. 环境属性（地形高度、摩擦、恢复系数）一律不显式建模，视为扰动；
2. 机器人唯一可靠的传感器是关节编码器和 IMU，所以内部模型只能从**本体感觉
   历史**出发。

于是策略不再接收地形扫描或真实速度，而是接收三部分：

```
[当前 45 维本体观测] + [估计的机身线速度 3 维] + [隐式内部嵌入 16 维]
```

其中：

- **速度回归**给出闭环反馈里"速度"这一显式分量；
- **隐式嵌入**携带无法直接观测的"稳定性/响应"信息，等价于对环境响应的压缩。

嵌入用**对比学习（SwAV）**训练：用 6 帧观测历史编码出的嵌入，去匹配"下一时刻
机器人状态"编码出的嵌入。下一时刻状态是仿真器给的**特权信息**（真实速度 +
下一帧本体观测），训练时才可见，部署时完全不需要。

## 2. 代码映射

| 论文组件 | HIMLoco 参考实现 | 本仓库 |
| --- | --- | --- |
| 历史编码器 / 目标网络 / 原型 | `rsl_rl/modules/him_estimator.py` | `src/go2_tasks/him.py::HIMEstimator` |
| 策略 + 估计器组合 | `modules/him_actor_critic.py` | `him.py::HIMActor` |
| PPO + 估计器优化 | `algorithms/him_ppo.py` | `him.py::HIMPPO` |
| 特权观测与历史缓冲 | `legged_gym/envs/base/legged_robot.py` | `src/go2_tasks/env_cfgs.py::unitree_go2_him_env_cfg` |
| 部署网络 | `play.py` 导出 | `him.py::_HIMExport`、`sim2sim/rma_policy.py::HimActor` |

## 3. 观测布局（严格按索引）

策略观测组 `actor`（**6 帧历史，mjlab 按旧→新拼接**）：

| 顺序 | 内容 | 维度 |
| --- | --- | ---: |
| 1 | 速度命令 × `[2.0, 2.0, 0.25]` | 3 |
| 2 | 机体角速度 × 0.25 | 3 |
| 3 | 投影重力 | 3 |
| 4 | 关节角 − 默认角 | 12 |
| 5 | 关节速度 × 0.05 | 12 |
| 6 | 上一时刻动作 | 12 |
|  | **合计每帧** | **45** |

6 帧 → 270 维。默认 50 Hz 控制，历史窗口 120 ms。

注意 mjlab 的观测组在 `concatenate_terms=True` 时是**按 term 排列**的，实际向量是

```
[command × 6][base_ang_vel × 6][projected_gravity × 6][joint_pos × 6][joint_vel × 6][actions × 6]
```

而不是"6 个完整 45 维帧"。`HIMActor.frame_major_history()` 按 `TERM_DIMS`
重新拆块并还原成 `[t-H … t]` 的帧顺序，再翻转为最新帧在前送进编码器。
部署端（ONNX / `sim2sim`）的输入约定则是直接给**帧优先、最新帧在前**的 270 维，
由部署代码自己维护环形缓冲。

Critic 观测组 `critic`（**单帧特权向量**，无噪声）：

| 索引 | 内容 | 维度 |
| --- | --- | ---: |
| 0:45 | 与 actor 同布局、同缩放的 45 维（无观测噪声） | 45 |
| 45:48 | 机体真实线速度 × 2.0（`imu_lin_vel`） | 3 |
| 48:51 | 扰动代理（`imu_lin_acc` 机体加速度） | 3 |
| 51:238 | 187 点地形高度扫描（1.6 m × 1.0 m，0.1 m 网格） | 187 |

估计器正是按这些固定偏移取数，和参考代码一致：

```python
velocity_target = critic[..., 45:48]
successor_state = cat(critic[..., 3:45], critic[..., 45:48])   # 45 维
```

注意 `3:45` 是"去掉命令的下一帧本体观测"：命令不是机器人响应的一部分；`45:48`
是下一帧的真实速度。两者拼起来恰好 45 维，这就是目标网络的输入维度。

> **扰动代理**：mjlab 没有"外力观测"，参考实现用的是仿真中施加到机身的推力。
> 这里改用机体线加速度（IMU 加速度计，仿真值），它对推挤、落地冲击、地形起伏
> 都有响应。它只进 critic，不影响部署。

## 4. 网络结构

```
270 -> 128 -> 64 -> 19        encoder（前 3 维 = 速度，后 16 维 L2 归一化）
45  -> 128 -> 64 -> 16        target（下一帧状态 -> 嵌入）
32  x 16                       prototype（SwAV 聚类中心）
64 = 45 + 3 + 16 -> 512 -> 256 -> 128 -> 12    actor（ELU，输出层线性）
238 -> 512 -> 256 -> 128 -> 1                  critic（ELU）
```

- 估计器前向在 `torch.no_grad()` 下进行，策略梯度不会回传到估计器；
- 估计器有自己的 Adam（默认 lr 跟随 PPO 的自适应 lr，梯度裁剪 10）；
- actor 不使用观测归一化（与参考实现一致）。

## 5. 训练目标

每个 mini-batch 上估计器的损失是两项之和：

1. **速度回归**：`MSE(v_hat, v_true)`，`v_true` 取下一帧缩放后的特权线速度；
2. **SwAV 交换预测**：
   `-0.5 * (q_s · log p_t + q_t · log p_s)`，
   其中 `q` 是 Sinkhorn 归一化后的软分配（stop-gradient），`p` 是
   `softmax(score / 3.0)`。原型权重每步归一化到单位球面。只有 `q`
   停梯度，`score` 必须保留梯度，使目标网络、原型和历史编码器都能训练。

原型分配必须"每个原型承担相同质量、每个样本分配相同质量"，Sinkhorn 迭代
（eps=0.05，3 次）实现这一点；批内统计正是对比学习能压制噪声的原因。
参考实现的 `swap_loss` 对样本和原型都取 `mean`，因此 32 个原型接近均匀
分配时，日志值约为 `ln(32)/32 ≈ 0.1083`；仅看这一数值不能判断隐变量学到了
什么，还要检查对编码器隐变量输出、目标网络及原型的梯度。

PPO 部分沿用本仓库默认：gamma 0.99、GAE 0.95、clip 0.2、entropy 0.01、
自适应 lr（desired_kl 0.01）、`num_steps_per_env = 24`、5 epochs × 4 minibatch。

## 6. 与参考实现的差异

| 项 | HIMLoco | 本仓库 | 原因 |
| --- | --- | --- | --- |
| 估计器更新时机 | 与每个 PPO mini-batch 交替 | 每次 PPO 更新后对同一批数据跑 4 个 epoch | PPO 实现不同；目标函数与数据相同，避免复制整个 PPO |
| episode 终止处 | 用终止前观测作为"下一帧" | 丢弃该 transition | mjlab 在 `step()` 内就地 reset，`extras` 不返回 reset 前观测 |
| 3 维扰动观测 | 施加外力 | 机体线加速度 | mjlab 无外力观测项 |
| critic 噪声 | 与 actor 共用噪声 | critic 无噪声 | estimator 目标更干净；部署不受影响 |
| 每轮步数 | 100 | 24 | 与本仓库其他任务一致 |
| 历史排列 | 帧优先、最新帧在前 | env 输出 term 优先，模型内重排为帧优先 | mjlab 观测组的 history 是逐 term 展开的 |
| action 裁剪 | clip 100 | `clip_actions=100` | 防止单个异常动作写回 `actions` 观测形成正反馈 |
| 跟踪权重 | lin 1.0 : ang 0.5，σ²=0.25 | lin 4.0 : ang 2.0，σ²=0.25 | 保留论文的相对权重，适配 mjlab 的奖励量级 |
| 角速度项 | yaw 跟踪奖励 + 单独的 roll/pitch 惩罚 | yaw 跟踪权重 2.0；roll/pitch 速度惩罚 -0.2 | 避免崎岖地形上的 roll/pitch 运动压低 yaw 奖励 |
| 动作变化率 | -0.01 | -0.04 | 与线速度奖励同样放大 4 倍；继承的 -0.2 在训练日志中远大于 yaw 奖励 |
| 命令采样 | `‖v_xy‖ < 0.2` 时把小线速度指令置零，保留航向/偏航指令 | `HimVelocityCommandCfg` 预留 10% 的 world 采「线速度恒 0、`\|yaw\| ≥ 0.2`」的原地转向 | mjlab 的均匀采样几乎采不到 `v=0 且 yaw≠0`；且 `_update_command` 会把 standing/heading/world-frame 世界的指令清零，所以转向 world 必须同时清掉这些标记 |
| 稠密跟踪项 | 无 | `-1.0·‖v_cmd − v_xy‖²` | 站立时 `exp(−e²/σ²)` 已饱和（e≈0.7 m/s），需要稠密项提供梯度；与 `go2_cts` 同项 |
| 训练环境 | Isaac Gym 4096 envs | mjlab，默认 1024 envs | 单机 6 GB 显存 |

以上差异不改变"用历史 + 特权下一帧状态训练内部模型、策略只看 45+3+16"的
信息结构；但**性能数字不能直接与论文对比**。

## 7. 运行

```bash
# 训练（默认 1024 环境；显存不足先降 --env.scene.num-envs）
uv run go2-train Go2-Velocity-HIM \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard \
  --agent.upload-model False

# 播放
uv run go2-play Go2-Velocity-HIM \
  --checkpoint-file logs/rsl_rl/go2_him/YOUR_RUN/model_9999.pt \
  --viewer viser

# 独立 MuJoCo 引擎验证（不依赖 mjlab）
uv run python sim2sim/go2_sim.py \
  --him logs/rsl_rl/go2_him/YOUR_RUN/model_9999.pt \
  --terrain flat --command 0.4 0 0
```

同分布评测（用训练环境自身的地形与命令分布，排除 reset 步，因此没有
`Metrics/twist/error_vel_xy` 那种按固定步数归一导致的偏差）：

```bash
uv run python tools/eval_go2_task.py \
  --checkpoint logs/rsl_rl/go2_him/YOUR_RUN/model_2000.pt \
  --num-envs 16 --steps 250
```

输出 `error_vel_xy`、`error_vel_yaw`、机身高度、地形等级与终止率，可直接与
`go2_rough`(0.80)、`go2_cts`(0.42)、`go2_amp`(0.29) 的收敛值对比。

日志中应出现：

```text
Loss/value, Loss/surrogate, Loss/estimator_velocity, Loss/estimator_swap
```

前两项应随训练下降；`estimator_velocity` 是速度回归 MSE（米²/秒²），
`estimator_swap` 是 SwAV 损失（理论下界 0，初始约 2.0 量级）。速度估计明显
不收敛时，优先检查 critic 的 45:48 是否真的是线速度。

`logs/rsl_rl/go2_him/<run>/` 中每次保存会同时产生 `model_*.pt` 与
`<run>.onnx`。

## 8. 部署契约

- 输入：**每帧 45 维**，顺序见 §3 第 2 列表；控制器维护最近 6 帧的环形缓冲，
  **帧优先、最新帧在前**拼接成 270 维；
- 输出：12 维动作，`q_target = default_pos + action_scale * action`，
  关节顺序 `FR, FL, RR, RL`（见 `go2_tasks/deployment.py`）；
- ONNX 模型只包含历史编码器 + 策略头，不含 target 网络和原型，因此不需要
  特权信息，也不需要估计器的优化器状态；
- 复位时必须清空历史缓冲（`HimActor.reset()` / `sim2sim` 的 `R` 键），
  否则会把上一段 episode 的观测带入。

构建单帧观测可用：

```python
from go2_tasks.deployment import him_actor_observation
obs45 = him_actor_observation(
    command=cmd, base_ang_vel=gyro, projected_gravity=gravity,
    joint_pos=q, joint_vel=dq, previous_action=last_action,
)
```

## 9. 常见问题

- **策略原地踏步不前进**：先看 `estimator_velocity` 是否收敛；速度估计偏差大时，
  策略会把噪声当作真实速度反馈。
- **sim2sim 与训练行为不一致**：确认历史缓冲是最新帧在前，且复位时清空；
  命令、角速度、关节速度的缩放必须与训练环境一致。训练时 actor 的噪声只在
  训练环境存在，`go2-play` / sim2sim 都不会加噪声。
- **显存不足**：HIMPPO 额外保存 270 维历史和 238 维特权观测（每轮
  `steps × envs`），1024 envs × 24 steps 约 50 MB，主要显存仍来自并行仿真；
  可先降到 512 或 256 环境。

## 10. 旧版训练记录（2026-11-06）

下面是修复 SwAV 梯度、估计器历史排列及观测缩放**之前**的实验记录。
当时的 `model_*.pt` 与当前观测契约不兼容，不能用来验证修复后的 HIM，
也不能直接用于当前 sim2sim 部署；需要从头训练。新检查点带有
`him_contract_version=2`，加载器会拒绝没有这个标记的旧检查点。

2026-11-07 的 `2026-11-07_09-34-01` run 已使用 v2 观测契约，但仍采用
原先耦合的角速度奖励与 `action_rate_l2=-0.2`。该 run 的后续 checkpoint 可
作为奖励调整前的对照；当前源码中新的奖励权重只对新启动的训练进程生效。

配置：1024 envs，4000 轮（约 1 小时 46 分），RTX 3060 Laptop（显存峰值
3.9 GB），`clip_actions=100`。日志目录
`logs/rsl_rl/go2_him/2026-11-06_21-47-59_rewfix/`。

### 10.1 两个已修复的实现问题

1. **历史排列**：mjlab 的观测组在 `concatenate_terms=True` 时按 term 展开，
   而模型最初按"整帧"重排，导致策略的"当前观测"实际取到动作历史的尾部。
   现由 `HIMActor.frame_major_history()` 按 `TERM_DIMS` 还原，测试
   `test_him_env_history_is_term_major` 与
   `test_him_actor_rebuilds_frame_order_from_term_major` 锁定该契约。
2. **动作裁剪**：缺少 HIMLoco 的 `clip_actions=100`，一次异常动作会写回
   `actions` 观测并进入历史编码器，形成正反馈。旧 run 在 690/705 轮出现
   `action_rate_l2 ≈ -3e16`、value loss `inf`，即由此引起。

### 10.2 跟踪权重 A/B

旧配置（mjlab 默认 `lin:ang = 4:4`，yaw 核 σ²=0.5）训练到 1400 轮后行为探针
显示：指令 0.4–0.6 m/s 时 `vx ≈ 0`，但 yaw 指令可跟到 0.414，`terrain_levels`
恒为 0——策略专门化到原地转向并蹲伏（高度 0.163 m）。改为论文比例
`4:2`、yaw σ²=0.25，并加入稠密项 `-1.0·‖v_cmd − v_xy‖²` 后：

| 同评测器（`tools/eval_go2_task.py`） | error_vel_xy |
| --- | ---: |
| 旧 run @1400 轮 | 0.689 m/s |
| 新 run @4000 轮 | 0.176 m/s |

训练日志里 `terrain_levels` 从 0 升到 5.4，`mean_episode_length` 稳定在 900。

### 10.3 收敛过程（固定指令 vx=0.5，25 轮平均）

| checkpoint | 1000 | 2000 | 3000 | 3999 |
| --- | --- | --- | --- | --- |
| error_vel_xy | 0.251 | 0.215 | 0.186 | 0.171 |

### 10.4 与其他任务同口径对比

同一评测器、各任务自身 play 地形与命令分布、24 envs × 300 步：

| 任务 | 训练轮数 | error_vel_xy |
| --- | ---: | ---: |
| `Go2-Velocity-HIM`（本任务） | 4000 | 0.176 |
| `Go2-Velocity-CTS` | 10000 | 0.087 |
| `Go2-Velocity-AMP`（平地） | 2650 | 0.120 |
| `Go2-Velocity-Rough`（带高度扫描） | 2500 | 0.353 |

注意各任务的命令范围与训练轮数不同，本表只用于同一评测器下的量级对照。
HIM 训练轮数只有 CTS 的 40%。原表的 `error_vel_yaw` 误用了机体线速度 z
分量，不能用于判断偏航跟踪；评测脚本现已改为读取机体角速度 z 分量。

### 10.5 已知缺口

- **偏航**：旧版评测数字无效，应重新以固定 yaw 指令测量真实角速度和绝对误差。
  当前训练命令课程最高只到 `±0.5 rad/s`，远低于论文的高速转向范围。
  训练环境另外预留 10% 的 world 做原地转向（`v_xy = 0`，`|yaw| ∈ [0.2, 0.5]`），
  否则均匀采样几乎不会出现"站着转身"的样本。评估时请分别看 `+yaw` 与 `-yaw`
  的绝对误差：两侧不对称通常意味着该类样本不足或奖励项有方向性偏置。
- **sim2sim 崎岖地形**：`go2_sim.py --terrain rough` 的 2–10 cm 随机高度场比
  训练课程（~5 级）更难，属分布外；平地 `vx=0.5` 误差 0.096 m/s 正常。
- **训练轮数**：4000 轮尚未收敛到平台（1000→4000 误差仍在缓慢下降），
  与 CTS 的 10000 轮不具可比性。
