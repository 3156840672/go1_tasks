# 在 Go2 上复现 DreamWaQ

本任务把论文的非对称 PPO、CENet 和 AdaBoot 接入现有 Go2/MuJoCo 训练环境。最终的策略可以在独立 MuJoCo 中运行。训练时用仿真信息教策略和估计器，部署时只需要关节编码器、IMU、速度指令和控制器自己保存的上一动作。

论文在 A1 和 Isaac Gym 上验证。这里复用 Go2 Rough 的电机、奖励、地形课程和域随机化。论文公式 (2) 给 critic 输入外加扰动力，Go2 版本使用 3 维机身线加速度作为扰动代理。帧数和网络输出的实现约定在下面逐项说明。[DreamWaQ v2](https://arxiv.org/abs/2301.10602v2) · [Go2 参考实现](https://github.com/curieuxjy/go2_dreamwaq)

## 第一步　理解一帧数据如何变成动作

### 为什么策略需要历史

论文第 II-A 节将盲走视为部分可观测决策过程。地形、接触状态、外力和机器人运动共同决定下一步会发生什么。策略每次只能看到这些因素在本体传感器上的一部分结果。例如，同样的关节角可能出现在平地正常支撑和前脚抵住台阶的两个瞬间。看过去几帧，就能看到当前动作发出后，关节有没有继续转动、机身有没有抬头、角速度有没有突变。这些变化提供了区分两种情况的线索。

公式 (1) 定义单帧 $o_t$。下表按本仓库实际送进网络的顺序列出字段。$t$ 以策略步计，每隔 20 ms 更新一次。当前帧和往前四帧合计五帧，跨度约 80 ms。五帧沿用开源 Go2 实现的设置。[论文第 II-A、II-B 节](https://arxiv.org/abs/2301.10602v2) · [Go2 历史配置](https://github.com/curieuxjy/go2_dreamwaq/blob/main/dreamwaq/legged_gym/legged_gym/envs/go2/go2_config.py#L229-L237)

| 数据 | 内容 | 在任务中的位置 |
| --- | --- | --- |
| 单帧 $o_t$ | 角速度 3、投影重力 3、相对默认姿态的关节角 12、关节角速度 12、上一动作 12、指令 $[v_x,v_y,\omega_z]$ 3 | 共 45 维 |
| 历史 $o_t^H$ | 从 $t-4$ 到 $t$ 的五帧 | mjlab 和 ONNX 接口共 225 维 |
| CENet 输出 | 估计机身坐标系下的线速度 $\hat v_t$ 3、隐变量 $z_t$ 16 | 用相同编码器提取历史特征 |
| actor 输入 | 当前 45 维、速度 3、隐变量 16 | 共 64 维，输出 12 维动作 |
| critic 输入 | 当前 45 维、仿真真实机身线速度 3、机身线加速度 3、地形高程 187 | 共 238 维，估计状态价值 |

速度指令 $c_t$ 表示希望机器人怎么走。比如 `[0.5, 0, 0]` 表示向前 0.5 m/s，横移与转向都为零。真实速度 $v_t$ 表示机器人实际上怎么动，估计速度 $\hat v_t$ 则是 CENet 从历史推断出的实际速度。上一动作 $a_{t-1}$ 告诉编码器机器人刚收到什么关节目标，配合随后的关节与 IMU 变化，可以判断机器人怎样响应了这条指令。

### 为什么 actor 和 critic 看见的数据不同

Actor 决定下一步怎么动。部署时，它接收当前观测、估计速度和隐变量。Critic 只在训练中参与学习，用仿真真实速度、地形高度等信息估计从当前状态出发还能获得多少回报。举例来说，机器人将要踩上台阶时，critic 可以根据高度扫描判断当前状态较难，帮助 PPO 评估实际采到的动作效果。部署时不需要运行 critic。这就是论文公式 (2) 对应的**非对称 actor–critic**。[论文第 II-B 节](https://arxiv.org/abs/2301.10602v2)

策略每 20 ms 输出 12 个关节偏移。Go2 按髋、大腿、小腿将偏移分别乘以 `0.125/0.25/0.25`，再加到默认站姿。比如一个大腿关节的动作是 `0.4`，目标就在该关节默认角度上加 `0.1 rad`。每个目标持续 4 个物理步，期间由 200 Hz 的 PD 跟踪，仿真增益为 `Kp=25, Kd=0.5`。这对应论文公式 (3) 的关节位置控制。[机器人及动作约定](../go2.md)

## 第二步　理解 CENet 如何学习

### 先看一次仿真转移

1. 在时刻 $t$ 读出历史 $o_t^H$，同时从仿真器记录机身真实线速度 $v_t$。这是**当前速度**标签。
2. CENet 读取历史，actor 根据编码结果选出动作 $a_t$。仿真器执行关节目标并返回奖励 $r_t$、下一帧 $o_{t+1}$ 和结束标记 `done`。
3. 保存 $o_{t+1}$ 作为**下一帧**标签。它和 $v_t$ 属于同一条转移，但对应两个不同的时刻。

例如，在 $t$ 时刻机身仍在向前，4 个物理步后前脚接触台阶，角速度和关节速度发生变化。速度分支仍应对齐接触前记录的 $v_t$，解码分支才学习预测接触后的 $o_{t+1}$。遇到 `done` 时，mjlab 返回的观测可能已经是复位后的新回合首帧，CENet 会跳过该条转移。新回合的历史用自己的首帧填满。

### 再看两条分支和三项损失

225 维历史先经共享编码器压缩。速度分支输出 $\hat v_t$，分别估计机身坐标系的前向、侧向和垂向线速度。隐变量分支为 16 个维度分别输出均值 $\mu$ 和对数方差 $\log\sigma^2$。训练时采样 $\epsilon\sim\mathcal N(0,I)$，再算 $z_t=\mu+\sigma\epsilon$。这个重参数化让采样有随机性，同时让预测误差的梯度能传回均值和方差。解码器从 $z_t$ 预测归一化的下一帧本体观测。

论文把这种机制称为隐式地形想象。训练没有给 $z_t$ 提供坡度、摩擦系数或台阶类别标签。当足端接触和机身运动的规律有助于预测下一帧时，编码器可以将这些规律保留在 $z_t$ 里。策略直接使用这个向量，向量的单个维度没有预先规定的物理意义。论文图 2 将解码器标为 48 维，公式 (1) 的六项合计 45 维，这里按公式 (1) 定义重建目标。[论文图 2 与公式 (5)–(7)](https://arxiv.org/abs/2301.10602v2)

$$
L_{\mathrm{CE}} = \operatorname{MSE}(\hat v_t,v_t)
+ \operatorname{MSE}(\hat o_{t+1},o_{t+1})
+ \beta D_{\mathrm{KL}}\big(q(z\mid o_t^H)\Vert\mathcal{N}(0,I)\big).
$$

- **速度 MSE** 让估计值接近仿真真值，单位是 `(m/s)²`。其平方根是按三个轴合并的 RMSE。
- **下一帧 MSE** 让历史编码保留有助于预测机器人运动的信息。目标使用归一化观测，数值没有统一的物理单位。
- **KL** 衡量编码得到的高斯分布与标准正态先验的差异。`β` 控制这项约束的强度，本实现默认 `0.5`。若 KL 长期趋近零，编码分布可能几乎不随输入变化。结合重建误差和速度跟踪表现，才能判断隐变量是否还带有任务相关信息。

PPO 先更新 actor/critic，随后用相同批次的有效转移更新 CENet。策略梯度只更新策略网络。训练采样隐变量，导出使用 $\mu$ 做确定性推理。训练 rollout 会保存生成该动作时实际用到的 64 维 actor 输入。这样 PPO 重算动作概率时会复用同一个隐变量和速度选择。

## 第三步　理解 PPO 和 AdaBoot 如何一起训练

### PPO 从走路结果学习动作

仿真每一步都给出奖励。论文表 I 里，跟踪线速度和偏航速度给正向奖励，机身晃动、关节加速度、动作变化和电机功率等带来惩罚。例如，指令前进 `0.5 m/s` 时，实际速度接近 `0.5 m/s` 的跟踪奖励更高。两种步态即使都能跟住指令，动作突变更大的步态也会因此损失回报。论文还通过地形课程逐步增加难度，并随机改变摩擦、负载等仿真条件。本仓库沿用 Go2 Rough 的具体奖励、课程和随机化配置。[论文表 I、表 II](https://arxiv.org/abs/2301.10602v2) · [Go2 Rough 配置说明](go2_rough.md)

一轮训练会先收集一批转移。PPO 结合实际奖励和 critic 的价值估计计算优势，衡量采到的动作比当前价值预期好多少。优势为正时，提高类似状态下采到该动作的概率。PPO 用概率比率的裁剪目标限制单次更新的幅度。完成 PPO 更新后，再对这一批转移训练 CENet。两者共享仿真经验，各自使用对应的训练信号。

### AdaBoot 逐步让 actor 依赖自己的速度估计

启用时，早期 actor 从真实线速度开始。训练框架在启动时随机设置首回合的已用时间，AdaBoot 会跳过这些首回合。之后每轮收集已结束回合的总回报，至少有两个完整样本时计算

$$
p_{\mathrm{est}} = 1 - \tanh\left(\frac{\operatorname{std}(R)}{|\operatorname{mean}(R)|+\epsilon}\right).
$$

公式的直觉是，回报波动大时，策略可能还没稳定学会走路，先更多地用真值速度。回报波动变小时，让策略更多地适应实际部署时的估计误差。例如，回报标准差是均值绝对值的 `0.2` 倍时，$p_{\mathrm{est}}\approx0.80$。比例为 `1.0` 时，概率约为 `0.24`。概率按每轮数据重新计算，可以上升或下降。当轮完整样本少于两个时，沿用上一轮的概率。

下一轮中，每个环境的每个控制步都会按这个概率选择 actor 的三个速度输入。隐变量仍由 CENet 生成，critic 和 CENet 的速度监督始终读取仿真真值。部署时不用这个开关，速度始终来自 CENet。[论文公式 (8)](https://arxiv.org/abs/2301.10602v2)

## 第四步　先跑一次 CPU 冒烟测试

在仓库根目录执行。现有 `.venv` 可用时可以给 `uv run` 加 `--no-sync`。下面把地形课程缩到两级、将每回合缩短到 0.16 s，让几轮训练就能覆盖采样、自动复位、AdaBoot、检查点和 ONNX 导出。

```bash
uv run go2-list-tasks

uv run go2-train Go2-Velocity-DreamWaQ \
  --gpu-ids None --env.scene.num-envs 2 \
  --env.scene.terrain.terrain-generator.num-rows 2 \
  --env.scene.terrain.max-init-terrain-level 0 \
  --env.episode-length-s 0.16 \
  --agent.num-steps-per-env 12 --agent.max-iterations 3 \
  --agent.algorithm.cenet-epochs 1 \
  --agent.logger tensorboard --agent.run-name cpu_smoke
```

成功时应看到 actor 观测 `(225,)`、critic 观测 `(238,)`，`cenet_valid_transitions` 大于 0，三项 CENet 损失为有限数。目录 `logs/rsl_rl/go2_dreamwaq/<run>/` 中有 `model_2.pt` 和 `<run>.onnx`。这个检查点只用于验证接口，其步态没有经过学习验证。若环境不允许写默认缓存，可在命令前设置 `WARP_CACHE_PATH=/tmp/go2-warp MPLCONFIGDIR=/tmp/go2-mpl`。

## 第五步　训练 DreamWaQ

以下命令在单卡上训练 Go2 DreamWaQ，默认使用第一张可见 GPU 并启用 AdaBoot。每轮从 `1024` 个环境各收集 `24` 个策略步，总共训练 `1000` 轮。显存不足时可以降低 `--env.scene.num-envs`。配置会保存在该次训练目录的 `params/` 下。

```bash
uv run go2-train Go2-Velocity-DreamWaQ \
  --env.scene.num-envs 1024 --agent.max-iterations 1000 \
  --agent.seed 42 --agent.logger tensorboard --agent.run-name dreamwaq_s42

uv run tensorboard --logdir logs/rsl_rl
```

首次在 GPU 上运行时，Warp 会编译碰撞等 CUDA 内核。若终端还在打印 `Module ... (compiled)`，请继续等待，看到 `Learning iteration 0/1000` 就表示已进入第一轮训练。若在编译期间按下 `Ctrl+C`，直接重跑训练命令即可，已成功写入缓存的内核一般会复用。

重点读 TensorBoard 的 `Loss/cenet_velocity_mse`、`Loss/cenet_next_obs_mse`、`Loss/cenet_kl`、`Loss/adaboot_probability`，以及地形课程、摔倒终止率和速度跟踪误差。`adaboot_probability` 受当轮完成的回合影响，可以上下波动。评估脚本固定速度指令和地形难度，可把学习曲线的变化与实际跟踪能力对上。

## 第六步　播放、评分和 sim2sim

把 `RUN` 换成 `logs/rsl_rl/go2_dreamwaq` 下实际训练完成的目录名。`model_999.pt` 对应从头运行 1000 轮的最后一个迭代。播放时看零速是否站稳，随后分别观察前进、侧移和原地转向。

训练地形有 14 类、10 级难度，其中包括金字塔上台阶、金字塔下台阶、开放式台阶和随机台阶。`go2-play` 默认重新随机生成 5×5 地图，少量机器人可能只出现在其中几类地形上。想单独看台阶时，用下面的 sim2sim 命令指定 `--terrain-type open_stairs`。交互窗口里按 `1` 启动策略。

```bash
RUN=替换成实际的训练目录名
DIR="logs/rsl_rl/go2_dreamwaq/$RUN"
CKPT="$DIR/model_999.pt"

uv run go2-play Go2-Velocity-DreamWaQ \
  --checkpoint-file "$CKPT" --num-envs 4 --viewer viser

uv run python tools/eval_go2_task.py \
  --task Go2-Velocity-DreamWaQ --checkpoint "$CKPT" \
  --device cuda:0 --seed 42 --num-envs 32 --steps 500 \
  --command 0.5 0 0 --terrain-level 3

uv run python sim2sim/go2_sim.py \
  --dreamwaq "$DIR/$RUN.onnx" --terrain flat --headless \
  --steps 2000 --command 0.3 0 0

uv run python sim2sim/go2_sim.py \
  --dreamwaq "$CKPT" --terrain go2_rough \
  --terrain-type open_stairs --terrain-level 2 \
  --steps 2000 --command 0.3 0 0 --headless
```

`--dreamwaq` 接受指定轮次的 `model_*.pt`，也接受训练目录中的 ONNX 导出。想检查某一轮时直接传该轮的 `.pt`。

分别测直行 `0.5 0 0`、侧移 `0 0.3 0`、原地转向 `0 0 0.3`，再提高地形难度。记录 `error_vel_xy`（平面线速度误差）、`error_vel_yaw`（偏航角速度误差）和 `terminated`（发生终止的环境步比例）。重复评估同一指令时固定 `--seed` 和 `--terrain-level`，便于追踪同一策略在不同检查点的表现。

sim2sim 的 `--steps` 是物理步数，2000 步约 10 s。对于无头运行，检查机身高度和三个方向的速度误差。去掉 `--headless` 可以看到脚是否卡在台阶上。ONNX 入口接受 `[1, 225]` 的 float32 历史，按观测项拼接，每项内部从最旧排到最新。例如先拼五帧角速度，再拼五帧投影重力，依次拼完六项。推理图包含训练得到的归一化参数。独立仿真会在取帧前更新传感器，按 ONNX 的 `action_clip` 元数据做动作限幅，并在复位或激活状态切换时清空历史。

## 第七步　接实机前核对接口

用已在 sim2sim 通过的 ONNX。先用实机的离线日志构建同样的 45 维观测，核对以下约定。

1. IMU 的角速度在机身坐标系，投影重力是 $R^T[0,0,-1]$。打印静止水平时的重力，检查其接近 `[0, 0, -1]`。
2. 12 个关节按 `FR, FL, RR, RL` 排列，每条腿按 `hip, thigh, calf`。逐项比对 SDK 电机索引、方向、零位与仿真模型。用部署常量中的 `policy_action_to_joint_targets` 计算目标角并限制在关节范围内。
3. `q_rel = q_meas - q_default`，上一动作使用上次实际送入关节目标变换的 12 维动作。上电、复位以及切换到策略时，用首个有效帧填满历史。
4. 先在驱动禁用时测接口和 50 Hz 推理周期，检查动作、目标角、限幅和超时处理。实机首次动作从悬挂状态、零指令开始，逐步过渡到低速平地。确认能可靠停机后再测坡面和台阶。

[Go2 部署常量与关节目标变换](../../src/go2_tasks/deployment.py) · [面向 Unitree Go2 的官方 SDK](https://github.com/unitreerobotics/unitree_sdk2)

## 阅读源码的顺序

1. [环境与观测配置](../../src/go2_tasks/env_cfgs.py) 的 `unitree_go2_dreamwaq_env_cfg`。
2. [CENet 和训练](../../src/go2_tasks/dreamwaq.py) 中的 `CENet.losses`、`DreamWaQPPO.act`、`process_env_step`、`update`。
3. [训练与导出回归测试](../../tests/test_go2_dreamwaq.py)，可见一个带复位的两环境样例，以及同一输入在 PyTorch、TorchScript 和 ONNX 中的输出比较。
4. [独立 MuJoCo ONNX 模块](../../sim2sim/quadruped_sim/policies/dreamwaq.py)，看到 225 维历史如何交给导出的策略。
