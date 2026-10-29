# Go1 AMP 全流程与代码详解

这份文档讲清楚本仓库里 Go1 AMP 的**每一步在做什么、代码在哪里、为什么这么写**。
它比 [AMP 操作教程](amp_tutorial.md) 更偏原理与代码，建议先能按操作教程跑通，
再回来对照阅读。

本文对应的代码状态：`src/go1_tasks/amp.py` 已改为最小二乘 GAN（LSGAN）判别器 +
有界风格奖励 + 特征归一化；训练脚本为 `go1-tasks` 自带的 `go1-train`。
文中 `文件:行号` 以当前工作区为准。

---

## 0. 文件地图

| 文件 | 作用 | 关键内容 |
| --- | --- | --- |
| `src/go1_tasks/collect_mob_amp.py` | 用已训练的 MoB 策略采集参考动作 | 连续状态快照、命令固定、片段校验 |
| `src/go1_tasks/visualize_amp_reference.py` | 把参考片段渲染成 GIF | MuJoCo 正运动学 + 足端一致性检查 |
| `src/go1_tasks/amp_data.py` | 特征工程与离线数据集 | 46 维帧、92 维转移、校验、采样 |
| `src/go1_tasks/amp_events.py` | 从参考帧重置环境 | 早期探索覆盖、软限位裁剪 |
| `src/go1_tasks/env_cfgs.py` | AMP 任务的环境配置 | `unitree_go1_amp_env_cfg` |
| `src/go1_tasks/rl_cfg.py` | AMP 的 PPO/判别器超参数 | `AmpPpoAlgorithmCfg` |
| `src/go1_tasks/amp.py` | AMP 算法本体 | 判别器、归一化、奖励、`AmpPPO` |
| `src/go1_tasks/eval_amp.py` | 定量评估步态 | 抬脚高度、对角/相邻相关 |
| `src/go1_tasks/__init__.py` | 任务注册 | `Go1-Velocity-AMP` |
| `tests/test_go1_amp.py` | 端到端短测试 | 数据流、奖励、保存/恢复 |

---

## 1. 一张图看懂全流程

```mermaid
flowchart LR
  A[已训练的 MoB 策略\nmodel_xxxx.pt] -->|collect_mob_amp.py| B[参考片段\nlogs/amp/mob_reference/trot/*.npz]
  B -->|AmpMotionDataset| C[专家转移\n(3000, 92)]
  D[AMP 策略\n随机初始化] --> E[Go1 仿真环境]
  E -->|采样状态| F[在线转移\n(s_t, s_t+1)]
  F --> G[判别器 D]
  C --> G
  G -->|有界风格奖励| H[PPO 更新]
  E -->|任务奖励| H
  H --> D
  G -->|判别器更新| G
```

四个阶段：

1. **采集**：MoB 策略在仿真里跑固定速度/步态，记录真实机器人状态，而不是策略观测。
2. **离线特征**：把每帧状态变成 46 维特征，把相邻两帧拼成 92 维转移。
3. **在线训练**：策略与环境交互；判别器同时看“专家转移”和“策略转移”，给策略风格奖励；PPO 用任务奖励 + 风格奖励更新策略；判别器自己也在更新。
4. **评估**：固定命令、固定种子跑一段，比较抬脚高度和足端相位相关性。

---

## 2. AMP 的核心思想

普通速度跟踪 RL 只告诉机器人“速度要对”，所以它可能找到速度正确但动作很怪的走法。
AMP 额外学习一个**动作风格打分器**：

- **专家转移**：来自参考动作的 `(s_t, s_t+1)`。
- **策略转移**：当前策略在仿真里产生的 `(s_t, s_t+1)`。
- **判别器** `D`：尽量把两者分开。
- **风格奖励**：判别器认为“像专家”时给正奖励，于是策略为了拿奖励去模仿专家。

关键点：AMP **不要求相位对齐**。它比较的是“这一步的状态转移像不像参考”，而不是
“第 t 帧必须跟参考第 t 帧一样”。所以策略可以自己决定何时迈步，只要步态统计特性一致。

本仓库使用的公式（`amp.py`）：

```text
判别器损失（LSGAN）:
  L_D = 0.5 * [ MSE(D(x_expert), +1) + MSE(D(x_policy), -1) ]
      + λ_gp * E_expert[ ||∂D/∂x||² ]

风格奖励（有界）:
  r_AMP = amp_reward_weight * step_dt * clamp(1 - 0.25 * (D(x) - 1)², min=0)

总奖励:
  r = r_task + r_AMP        （仅未终止的相邻状态对）
```

为什么是这套形式、而不是交叉熵 + `softplus`，见第 6 节。

---

## 3. 阶段 A：用 MoB 策略采集参考动作

### 3.1 为什么不直接保存 MoB 的策略观测

MoB 策略的观测含 30 帧历史、行为命令、上一动作，甚至观测噪声。AMP 判别器需要
**参考动作和在线策略使用完全相同的特征函数**。如果保存的是策略观测，在线时你
还得用同一套网络去复现它，既耦合又容易错。

所以采集脚本保存的是**真实仿真状态**：机身位姿/速度、关节位置/速度、四只脚的世界坐标。
这些量既能离线构造 46 维特征，也能在线从 `robot.data` 直接取，天然一致。

### 3.2 `collect_mob_amp.py` 逐段

**(a) 快照**（`collect_mob_amp.py:35`）

```python
def _snapshot(robot, foot_ids):
  data = robot.data
  return {
    "root_pos_w": data.root_link_pos_w[0],
    "root_quat_w": data.root_link_quat_w[0],
    "root_lin_vel_w": data.root_link_lin_vel_w[0],
    "root_ang_vel_w": data.root_link_ang_vel_w[0],
    "root_lin_vel_b": data.root_link_lin_vel_b[0],
    "root_ang_vel_b": data.root_link_ang_vel_b[0],
    "joint_pos": data.joint_pos[0],
    "joint_vel": data.joint_vel[0],
    "foot_pos_w": data.site_pos_w[0, foot_ids],
  }
```

同时保存世界系和机体系速度：世界系给“重置事件”写回仿真用，机体系给判别器特征用。
注意这里只取 `[0]`，因为采集环境只有一个 env。

**(b) 固定命令**（`collect_mob_amp.py:51`）

MoB 是一个行为条件策略，命令由 `twist`（速度）和 `behavior`（步态参数）两个
命令项组成。采集时必须把两者都固定住，否则片段里会混入不同步态：

```python
twist.vel_command_b[:] = command          # [vx, 0, 0]
behavior.values[:] = torch.tensor(
  (*GAITS["Trot"], 2.5, 0.30, 0.0, 0.30, 0.10)
)
behavior.cycle.zero_()                    # 相位从 0 开始
```

`GAITS["Trot"] = (0.5, 0.0, 0.0)`，后面依次是频率、身体高度、俯仰、站宽、抬腿高度。

**(c) 采集循环**（`collect_mob_amp.py:68`）

```python
wrapped.reset()
_set_commands(env, velocity, gait)
env.observation_manager.compute(update_history=True)
obs = wrapped.get_observations()

with torch.no_grad():
  for _ in range(warmup_steps):          # 至少 30 步，填满 MoB 的 30 帧历史
    obs, _, dones, _ = wrapped.step(policy(obs))
    if bool(dones[0].item()):
      return None                        # 跌倒就丢弃重试
  frames = [_snapshot(robot, foot_ids)]
  for _ in range(clip_steps):
    obs, _, dones, _ = wrapped.step(policy(obs))
    if bool(dones[0].item()):
      return None
    frames.append(_snapshot(robot, foot_ids))
```

三个细节：

- `cfg.auto_reset = False`：跌倒时保留终止帧，而不是被自动重置污染。
- 预热至少 30 步：把 MoB 的 30 帧历史观测填满，避免片段开头是随机历史。
- 一旦 `done` 立刻丢弃整段：保证保存的帧是**连续同一条轨迹**，不会把
  “重置前最后一帧”和“重置后第一帧”错接成转移。

**(d) 输出格式**

每段 `N = ceil(seconds × fps) + 1` 帧，50 Hz 下 5 秒是 251 帧。字段：

| 字段 | 形状 | 用途 |
| --- | --- | --- |
| `root_pos_w` / `root_lin_vel_w` / `root_ang_vel_w` | `(N,3)` | 重置事件写回世界系状态 |
| `root_quat_w` | `(N,4)` | `wxyz` 四元数 |
| `root_lin_vel_b` / `root_ang_vel_b` | `(N,3)` | 判别器特征 |
| `joint_pos` / `joint_vel` | `(N,12)` | 判别器特征 |
| `foot_pos_w` | `(N,4,3)` | 判别器特征（相对机身） |
| `fps` / `joint_names` / `foot_site_names` | 元数据 | 一致性校验 |

### 3.3 可视化检查

`visualize_amp_reference.py` 用 MuJoCo 对保存的根位姿和关节角做正运动学，
同时核对算出的脚位置与文件里的 `foot_pos_w` 是否一致：

```python
if not np.allclose(frames[out_idx, :, 3], clip["foot_pos_w"][frame_idx], atol=1e-4):
  raise ValueError(f"Frame {frame_idx}: FK feet differ from stored reference feet")
```

这一步能抓出关节顺序、根坐标系、XML 不一致的问题。只检查文件数量和形状
**不能**证明步态正确，一定要看 GIF 里对角足是否交替抬起。

### 3.4 采集命令

```bash
uv run --no-sync python -m go1_tasks.collect_mob_amp \
  --checkpoint logs/rsl_rl/go1_mob/2026-10-27_15-04-14/model_5050.pt \
  --gait Trot --speeds 0.2 0.4 0.6 0.8 \
  --clips-per-speed 3 --seconds 5 --warmup-seconds 1
```

---

## 4. 阶段 B：特征工程与离线数据集

### 4.1 单帧 46 维特征

| 顺序 | 特征 | 维度 | 坐标系 |
| --- | --- | ---: | --- |
| 1 | 机身高度 `z` | 1 | 世界系，平地地面 `z=0` |
| 2 | 重力方向 | 3 | 机体系 |
| 3 | 机身线速度 | 3 | 机体系 |
| 4 | 机身角速度 | 3 | 机体系 |
| 5 | 关节位置 | 12 | 采集文件 `joint_names` 顺序 |
| 6 | 关节速度 | 12 | 同上 |
| 7 | 四只脚相对机身位置 | 12 | 机体系，`FR,FL,RR,RL` |
| | **合计** | **46** | |

相邻两帧拼接得到 **92 维转移**。绝对 `x/y`、绝对朝向、速度命令、步态命令都
**不进入**判别器：判别器只应该比较“动作状态”，不应该因为机器人跑到不同位置
或收到不同速度命令而改变判断。

### 4.2 `amp_frame` 代码

`amp_data.py:36`：

```python
gravity_w = torch.zeros_like(root_pos_w)
gravity_w[:, 2] = -1.0
gravity_b = quat_apply_inverse(root_quat_w, gravity_w)

foot_rel_w = foot_pos_w - root_pos_w[:, None, :]
foot_rel_b = quat_apply_inverse(
  root_quat_w[:, None, :].expand(-1, 4, -1), foot_rel_w
).reshape(n, 12)

return torch.cat(
  (root_pos_w[:, 2:3], gravity_b, root_lin_vel_b, root_ang_vel_b,
   joint_pos, joint_vel, foot_rel_b),
  dim=-1,
)
```

- **重力方向**代替绝对朝向：四元数会随世界系 yaw 变化，而 `gravity_b` 只反映
  机身相对重力的倾角，这正是步态风格关心的量。
- **足端相对机身再转到机体系**：去掉世界位置和朝向后，剩下的是“腿相对身体
  摆在哪里”，这是 Trot 相位的直接编码。
- 高度只取 `z`，并且代码注释明确假设平地 `z=0`。以后做非平地运动，要改成相对
  当地地面的高度，否则判别器会把地形高度差当成风格差异。

### 4.3 在线特征 `amp_frame_from_robot`

`amp_data.py:84`：

```python
if joint_names is not None and tuple(robot.joint_names) != joint_names:
  raise ValueError("Live robot joint order differs from the reference motions")
if foot_ids is None:
  foot_ids, foot_names = robot.find_sites(FOOT_NAMES, preserve_order=True)
  if tuple(foot_names) != FOOT_NAMES:
    raise ValueError(f"Unexpected foot order: {foot_names}")
return amp_frame(
  data.root_link_pos_w, data.root_link_quat_w,
  data.root_link_lin_vel_b, data.root_link_ang_vel_b,
  data.joint_pos, data.joint_vel,
  data.site_pos_w[:, foot_ids],
)
```

`foot_ids` 可以在算法构造时缓存一次，避免每个控制步都做正则匹配。
`joint_names` 校验只在 `act()` 里做，因为那里能拿到 `motions.joint_names`。

### 4.4 `AmpMotionDataset`

`amp_data.py:113`。构造时做大量校验，任何一项不满足都直接报错：

- 必须有 `.npz` 文件；
- 必需字段齐全（`REQUIRED_FIELDS`）；
- 形状正确、数值有限；
- 四元数近似单位长度；
- 12 个关节、名字唯一；
- 足端顺序必须是 `FR, FL, RR, RL`；
- `fps > 0`；
- 所有片段关节顺序、fps 一致。

然后**只在单个文件内部**配对相邻帧：

```python
frame = amp_frame(*(torch.from_numpy(clip[key].astype(np.float32))
                    for key in REQUIRED_FIELDS))
transitions.append(torch.cat((frame[:-1], frame[1:]), dim=-1))
```

这一点很重要：如果把 `clip_000` 的末帧和 `clip_001` 的首帧拼起来，就会造出
一段物理上不存在的“瞬移”，污染专家数据。当前实现不会跨文件配对。

`fps` 参数还会和在线 `1/step_dt` 校验一致。也就是说，参考片段必须按训练环境的
控制频率采集；当前是 50 Hz。

---

## 5. 阶段 C-1：AMP 环境怎么搭

### 5.1 继承链

`unitree_go1_amp_env_cfg` 不是从零写，而是复用已有的 Go1 环境：

```text
unitree_go1_rough_env_cfg   # 崎岖地形、DR、碰撞传感器
        ↓
unitree_go1_flat_env_cfg    # 平地：去掉地形扫描/部分传感器，加 fell_over
        ↓
unitree_go1_mob_env_cfg     # 走路行为：behavior 命令、clock 观测、30 帧历史
        ↓
unitree_go1_amp_env_cfg     # AMP：删掉 behavior/clock，改命令范围，换奖励，加参考重置
```

`env_cfgs.py:653` 的核心改动：

```python
cfg.commands.pop("behavior")
actor.terms.pop("behavior"); actor.terms.pop("clock")
critic.terms.pop("behavior"); critic.terms.pop("clock")
actor.history_length = 1          # AMP 不需要 MoB 的 30 帧历史
```

MoB 的风格来自行为命令；AMP 的风格来自参考动作，所以不能保留 behavior 命令，
否则策略会收到一个训练时无意义的条件输入。

### 5.2 命令范围

```python
twist.ranges.lin_vel_x = (0.2, 0.8)
twist.ranges.lin_vel_y = (0.0, 0.0)
twist.ranges.ang_vel_z = (0.0, 0.0)
twist.rel_standing_envs = 0.0
```

参考数据只有正向 Trot，所以只采样正向速度，横向和偏航固定为 0。
这不是“策略输入少了两维”——actor 仍然接收完整 3 维命令，只是命令值不变化。
以后要转弯/侧移，先采集对应动作，再扩大范围。

### 5.3 奖励集

只保留 5 项，权重在 `env_cfgs.py:691`：

| 奖励 | 权重 | 作用 |
| --- | ---: | --- |
| `track_linear_velocity` | 2.0 | 速度跟踪（主任务） |
| `upright` | 0.2 | 保持直立 |
| `dof_pos_limits` | -1.0 | 关节限位 |
| `action_rate_l2` | -0.05 | 动作变化率 |
| `foot_slip` | -0.05 | 足端打滑 |

注意：mjlab 的 `RewardManager` 默认 `scale_by_dt=True`，所以环境返回的每一项
奖励都已经乘过 `step_dt`。这就是为什么 AMP 风格奖励也要乘 `step_dt`，两者单位
才一致（见 6.4）。

### 5.4 观测维度

- **Actor 45 维**：角速度 3 + 投影重力 3 + 关节位置偏移 12 + 关节速度 12 +
  上一动作 12 + 命令 3。
- **Critic 72 维**：额外含训练时可得的机身线速度、足端等信息。
- **判别器 92 维**：与 actor/critic 都是独立接口，不要混淆。

### 5.5 参考状态重置事件

`amp_events.py:13` 的 `reset_from_amp_reference` 会在每个 reset 时，按
`probability=0.5` 把一部分环境直接放到随机参考帧：

```python
selected = ids[torch.rand(len(ids), device=...) < self._probability]
frame_ids = torch.randint(len(self._frames["joint_pos"]), (len(selected),), ...)

root_pos = self._env.scene.env_origins[selected].clone()
root_pos[:, 2] += self._frames["root_pos_w"][frame_ids, 2]

self._robot.write_root_state_to_sim(root_state, selected)
limits = self._robot.data.soft_joint_pos_limits[selected]
joint_pos = self._frames["joint_pos"][frame_ids].clamp(
  limits[..., 0], limits[..., 1]
)
self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=selected)
```

为什么这么做：

- **早期探索**：从随机初始姿态开始，策略很难碰巧做出 Trot；直接放到参考状态里，
  它更容易体验到“像专家”的转移。
- **只覆盖一半**：保留 50% 的普通 reset，避免策略只会在参考分布附近活动。
- **只取参考的 `z`**：`x/y` 用环境自己的 `env_origins`，否则机器人会被放到世界
  别的角落。
- **软限位裁剪**：参考帧的关节角如果略超限（换机器人/换 DR 配置时可能发生），
  直接写入会给求解器注入非法状态；裁剪后更稳。

事件顺序：`_reset_idx` 先 `scene.reset`，再按 `cfg.events` 的插入顺序执行 reset
事件。`amp_reference_state_init` 是最后追加的，所以它覆盖前面的默认 reset。
这个事件只在训练配置里添加（`if not play`），播放/评估不会用到参考数据。

---

## 6. 阶段 C-2：判别器与风格奖励

### 6.1 判别器网络

`amp.py:19`：

```python
class AmpDiscriminator(nn.Module):
  def __init__(self, hidden_dims: tuple[int, ...] = (256, 128)) -> None:
    super().__init__()
    layers, in_dim = [], TRANSITION_DIM          # 92
    for hidden_dim in hidden_dims:
      layers.extend((nn.Linear(in_dim, hidden_dim), nn.ReLU()))
      in_dim = hidden_dim
    self.trunk = nn.Sequential(*layers)
    self.head = nn.Linear(in_dim, 1)

  def forward(self, transitions):
    return self.head(self.trunk(transitions)).squeeze(-1)
```

输出是**原始 logit**，没有 sigmoid。参考样本目标是 `+1`，策略样本目标是 `-1`。

### 6.2 为什么不用 BCE + softplus

早先版本用：

```text
loss = BCE(D, expert=1) + BCE(D, policy=0)
reward = softplus(D)
```

这套组合有一个致命问题：**BCE 的最优解是 logit 趋于 ±∞**。判别器越训练越自信，
策略样本的 logit 会一路变成 `-800`。而 `softplus(-800)` 在 float32 中直接下溢为
0，梯度也随之为 0。于是：

1. 策略拿到 0 风格奖励；
2. 风格项对策略参数的梯度也是 0；
3. 策略再也无法通过模仿专家来“变好”；
4. 判别器继续把策略判得更负，永久塌陷。

这正是旧 checkpoint `model_9400.pt` 上观察到的 `D(expert)=+16.6, D(policy)=-816`。
加谱归一化只能限制 Lipschitz，**限制不了 logit 的常数偏移**，所以不能根治。

LSGAN 的目标是有限值：专家 `+1`、策略 `-1`。判别器一旦把 logit 推到目标附近，
MSE 损失就接近 0，不会继续往无穷推。于是策略 logit 不会掉到 `-800`，风格奖励
的梯度也不会消失。

### 6.3 梯度惩罚

`amp.py:43`：

```python
def compute_grad_pen(self, transitions, weight):
  transitions = transitions.detach().requires_grad_(True)
  logits = self.forward(transitions)
  gradients = autograd.grad(logits.sum(), transitions, create_graph=True)[0]
  return weight * gradients.square().sum(-1).mean()
```

作用：让判别器在专家样本附近的输入梯度不要过大，避免判别边界过陡。
它和 LSGAN 一起用；谱归一化与 GP 功能重叠，因此当前实现**去掉了谱归一化**。
GP 在**未归一化的原始专家转移**上计算（与 AMP 参考实现一致）。

### 6.4 特征归一化 `AmpFeatureNormalizer`

不同特征尺度差很多（机身高度 std ≈ 0.006，关节速度 std ≈ 2.5），不归一化时
判别器可能只依赖几个大尺度维度。`amp.py:55` 实现了一个一维 running mean/var：

```python
@torch.no_grad()
def update(self, frames):
  batch_mean = frames.mean(0)
  batch_var = frames.var(0, unbiased=False)
  batch_count = frames.shape[0]
  delta = batch_mean - self.mean
  total = self.count + batch_count
  self.mean += delta * batch_count / total
  m2 = (self.var * self.count + batch_var * batch_count
        + delta.square() * self.count * batch_count / total)
  self.var = m2 / total
  self.count = total
```

这就是并行方差合并公式（Chan/Welford 形式），支持任意 batch 顺序更新。
`normalize` 先把方差下限截到 `1e-4`，再除以 `sqrt(var) + 1e-8`，最后裁剪到
`[-10, 10]`：

```python
std = torch.sqrt(self.var.clamp_min(1e-4))
return ((frames - self.mean) / (std + 1e-8)).clamp(-10.0, 10.0)
```

- 统计量**用参考数据初始化**，训练中再用专家 + 策略样本共同更新。
- 手机制：normalizer 的 mean/var 随 checkpoint 保存，恢复训练不会从零重估。

归一化只作用于判别器输入，不影响 actor/critic 观测。

### 6.5 有界风格奖励

`amp.py:115`：

```python
def amp_style_reward(logits):
  return torch.clamp(1.0 - 0.25 * (logits - 1.0).square(), min=0.0)
```

| `D(x)` | 风格奖励 |
| ---: | ---: |
| ≤ -1 | 0 |
| 0 | 0.75 |
| +1 | 1.0 |
| +3 | 0 |
| > 3 | 0 |

奖励有界让 PPO 的回报尺度稳定，不会因为判别器自信而爆炸。
实际每步加入环境的量是：

```python
rewards += amp_reward_weight * step_dt * amp_style_reward(logits) * valid
```

默认 `amp_reward_weight=2.0`、`step_dt=0.02`，所以每步最多加 `0.04`，与
已经乘过 `dt` 的任务奖励同量级。`valid = ~dones` 保证重置边界的那一步不拿风格
奖励、也不进入判别器训练（因为自动重置后取到的“下一帧”已经属于新 episode）。

---

## 7. 阶段 C-3：`AmpPPO` 如何接进 rsl_rl

### 7.1 类关系

```text
rsl_rl.algorithms.PPO
        ↑ 继承
go1_tasks.amp.AmpPPO
```

`AmpPPO` 只覆盖以下关键方法：

| 方法 | 作用 |
| --- | --- |
| `construct_algorithm` | 拿到 robot、step_dt、缓存的足端 id |
| `train_mode` | 构建离线数据集 + 初始化归一化 |
| `act` | 记录 `s_t` |
| `process_env_step` | 计算 `r_AMP(s_t,s_{t+1})`、存转移、调用父类 |
| `update` | 先 PPO，后判别器 |
| `save` / `load` | 额外存取判别器、归一化 |

### 7.2 `construct_algorithm`

`amp.py:206`：

```python
@staticmethod
def construct_algorithm(obs, env, cfg, device):
  algorithm = PPO.construct_algorithm(obs, env, cfg, device)  # 实际构造的是 AmpPPO
  assert isinstance(algorithm, AmpPPO)
  robot = env.unwrapped.scene["robot"]
  algorithm._robot = robot
  algorithm._step_dt = env.unwrapped.step_dt
  foot_ids, foot_names = robot.find_sites(FOOT_NAMES, preserve_order=True)
  if tuple(foot_names) != FOOT_NAMES:
    raise ValueError(f"Unexpected foot order: {foot_names}")
  algorithm._foot_ids = foot_ids
  return algorithm
```

`PPO.construct_algorithm` 会从 `cfg["algorithm"]["class_name"]` 解析出类名并构造
实例。任务配置里 `class_name = "go1_tasks.amp.AmpPPO"`，所以这里拿到的就是
`AmpPPO`。随后把运行期需要的 robot / dt / 足端索引挂到算法上。

### 7.3 `train_mode`

`amp.py:218`：

```python
def train_mode(self):
  super().train_mode()
  if self.motions is None:
    self.motions = AmpMotionDataset(
      self.reference_path,
      joint_names=tuple(self._robot.joint_names),
      fps=1.0 / self._step_dt,
    )
    print(f"[AMP] {self.motions.num_clips} clips, "
          f"{len(self.motions)} transitions, {self.motions.fps:g} Hz")
  if self.amp_normalizer is None:
    frames = torch.cat((self.motions.transitions[:, :FRAME_DIM],
                        self.motions.transitions[:, FRAME_DIM:]), dim=0)
    self.amp_normalizer = AmpFeatureNormalizer(FRAME_DIM, self.device)
    self.amp_normalizer.initialize(frames)
```

只在第一次 `train_mode` 时加载数据集；如果从 checkpoint 恢复了归一化统计量，
这里不会覆盖它。

### 7.4 `act` / `process_env_step` 的时序

这是 AMP 最容易写错的地方。一次控制步的顺序是：

```text
runner:  actions = alg.act(obs)          # 此时 robot.data 是 s_t
         obs, r_task, dones, extras = env.step(actions)
         # env 内部：物理积分 → 计算任务奖励 → 自动重置 done 环境 → forward
         alg.process_env_step(obs, r_task, dones, extras)
         # 此时 robot.data 是 s_{t+1}（done 环境已是重置后的新状态）
```

对应代码：

```python
def act(self, obs):
  self._previous_frame = amp_frame_from_robot(
    self._robot, joint_names=self.motions.joint_names,
    foot_ids=self._foot_ids).detach()
  return super().act(obs)

def process_env_step(self, obs, rewards, dones, extras):
  next_frame = amp_frame_from_robot(self._robot, foot_ids=self._foot_ids)
  transitions = torch.cat((self._previous_frame, next_frame), dim=-1)
  valid = ~dones.bool().reshape(-1)
  with torch.no_grad():
    logits = self.discriminator(self._normalize_transitions(transitions))
    imitation_reward = amp_style_reward(logits)
    rewards += self.amp_reward_weight * self._step_dt * imitation_reward * valid
  if bool(valid.any()):
    self._policy_transitions.append(transitions[valid].detach())
    self._style_reward_values.append(imitation_reward[valid].detach())
  self._previous_frame = None
  super().process_env_step(obs, rewards, dones, extras)
```

要点：

- **`s_t` 必须在 `env.step` 之前取**，否则会拿到 `s_{t+1}`。
- **done 环境要排除**：mjlab 在 `env.step` 内部自动重置，done 环境的
  `robot.data` 已经是新 episode 的状态，拼出来的转移是假的。
- **风格奖励在 `no_grad` 下算**：只作为奖励，不参与判别器反传。
- **判别器在 rollout 时是 eval 模式**，避免像 BatchNorm 那样的训练态行为
  （虽然当前网络没有 BN，但保持一致性）。
- `super().process_env_step` 会把奖励 clone 进 rollout storage，并做 PPO 的
  超时 bootstrap。

### 7.5 `update`：先 PPO，后判别器

`amp.py:284` 的顺序：

1. `super().update()`：标准 PPO 更新 actor/critic，清空 rollout storage。
2. 记录本轮风格奖励的均值（用于日志）。
3. 把本轮所有有效策略转移拼起来、`.clone()` 后放进 replay buffer。
4. 循环 `discriminator_epochs × discriminator_batches_per_iteration` 次：
   - 从专家数据集采样 1024 条；
   - 从 replay buffer 采样 1024 条；
   - 各自归一化，过判别器；
   - LSGAN MSE + 梯度惩罚；
   - 优化判别器；
   - 用原始专家/策略帧更新 normalizer。
5. 记录 `amp_discriminator` / `amp_expert_logit` / `amp_policy_logit`。

为什么 `.clone()`：

```python
# Rollout tensors were created under torch.inference_mode; clone here so
# autograd may use replay samples as discriminator inputs.
policy = torch.cat(self._policy_transitions).clone()
```

`act`/`process_env_step` 都在 runner 的 `torch.inference_mode()` 里执行，产生的
是 inference tensor；后面要拿 replay 样本做判别器输入时，克隆成普通 tensor 才能
安全地被 autograd 使用。

为什么先 PPO 后判别器：

- 本轮风格奖励必须用**更新前**的判别器算，否则 PPO 用的奖励和实际数据分布不一致。
- 判别器更新在 PPO 之后，逻辑上等价于“用旧 D 采数据 → 更新策略 → 再用新数据
  更新 D”，是标准的交替训练。

### 7.6 保存与恢复

`amp.py:353` / `amp.py:361`，checkpoint 额外包含：

```python
amp_discriminator_state_dict
amp_optimizer_state_dict
amp_normalizer_state      # mean / var / count
```

恢复训练：

```bash
uv run go1-train Go1-Velocity-AMP \
  --agent.resume True \
  --agent.load-run 2026-10-29_09-53-48_lsgan_norm \
  --agent.load-checkpoint model_5500.pt \
  --agent.max-iterations 3000
```

- `runner.load(path)` 默认 `load_cfg=None`，会同时恢复 actor、critic、判别器、
  判别器优化器和归一化统计量。
- `--agent.max-iterations 3000` 表示**再跑 3000 轮**，不是总轮数。
- replay buffer 不保存，恢复后从空开始积累；这是有意的，避免 checkpoint 过大。

### 7.7 训练超参数

`rl_cfg.py:20`：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `amp_reward_weight` | 2.0 | 风格奖励上限 `2.0 × 0.02` |
| `discriminator_learning_rate` | 1e-4 | 判别器单独优化器 |
| `discriminator_batch_size` | 1024 | 每次判别器更新样本数 |
| `discriminator_batches_per_iteration` | 8 | 每轮判别器梯度步数 |
| `discriminator_epochs` | 1 | 与上一项相乘 |
| `gradient_penalty_weight` | 10.0 | GP 系数 |
| `replay_capacity` | 100000 | 历史策略转移缓存 |

PPO 部分沿用仓库默认：5 epochs、4 minibatches、lr 1e-3（自适应 KL）、
`gamma=0.99`、`lam=0.95`。

---

## 8. 阶段 D：训练、评估与播放

### 8.1 训练

```bash
uv run go1-train Go1-Velocity-AMP \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 10000 \
  --agent.save-interval 250 \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --agent.run-name lsgan_norm
```

日志里要盯的字段：

| 字段 | 健康范围 | 含义 |
| --- | --- | --- |
| `Mean amp_style_reward_raw` | 0.2–0.8 | 有界风格奖励均值；长期接近 0 说明判别器赢了 |
| `Mean amp_expert_logit` | 0.5–1.0 | 专家 logit，应接近 +1 |
| `Mean amp_policy_logit` | -0.5–-1.0 | 策略 logit，应接近 -1 |
| `Mean amp_discriminator` | 0.2–1.0 | LSGAN 损失，不会像 BCE 一样单调到 0 |
| `Episode_Termination/fell_over` | → 0 | 跌倒次数 |
| `Mean episode length` | → 1000 | 20 s / 0.02 s = 1000 |
| `Metrics/twist/error_vel_xy` | 越小越好 | 速度跟踪误差 |

如果 `amp_style_reward_raw` 持续接近 0、`policy_logit` 远小于 -1，优先检查
判别器是否又被训练得过强（增加 `discriminator_batches_per_iteration` 前先确认
LSGAN 目标没被改回 BCE）。

### 8.2 定量评估

`eval_amp.py` 固定命令、固定种子跑 5 秒，统计四只脚的抬脚高度和相位相关性：

```bash
uv run --no-sync python -m go1_tasks.eval_amp \
  logs/rsl_rl/go1_amp/2026-10-29_09-53-48_lsgan_norm/model_5500.pt \
  --vx 0.6 --seconds 5
```

指标含义：

- **foot lift**：每只脚在窗口内的高度极差（`np.ptp`），反映抬腿幅度。
- **diagonal correlation**：`(corr(FR,RL) + corr(FL,RR)) / 2`。
  Trot 的对角腿同相，应该接近 +1。
- **adjacent correlation**：`(corr(FR,FL) + corr(RR,RL)) / 2`。
  Trot 的相邻腿反相，应该接近 -1。

单个相关系数不能代替动画验收，但它能快速区分“只是走得快”和“真的在 Trot”。

### 8.3 本次实测结果

`model_5500`（第 5500/10000 轮，训练手动停止）：

| 命令 vx | 实际 vx | 抬脚高 FR/FL/RR/RL (m) | 对角相关 | 相邻相关 | 跌倒 |
| ---: | ---: | --- | ---: | ---: | :--: |
| 0.2 | 0.164 | 0.091/0.087/0.077/0.084 | 0.985 | -0.660 | 否 |
| 0.4 | 0.364 | 0.087/0.085/0.085/0.087 | 0.987 | -0.706 | 否 |
| 0.6 | 0.578 | 0.088/0.081/0.075/0.090 | 0.987 | -0.749 | 否 |
| 0.8 | 0.808 | 0.094/0.079/0.087/0.094 | 0.985 | -0.818 | 否 |

参考片段（同速度）的抬脚高约 0.08 m、对角相关 0.99、相邻相关 -0.79。
策略在四档速度上都复现了 `FR–RL`、`FL–RR` 对角同相、相邻交替的 Trot；
0.2–0.4 m/s 实际速度略低于命令（约 0.02–0.04 m/s），0.8 m/s 几乎无误差。

### 8.4 播放

```bash
uv run go1-play Go1-Velocity-AMP \
  --checkpoint-file logs/rsl_rl/go1_amp/2026-10-29_09-53-48_lsgan_norm/model_5500.pt \
  --viewer viser
```

播放只加载 actor，不读取参考动作文件；参考数据只在继续 AMP 训练时需要。

---

## 9. 调参手册

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| 风格奖励长期接近 0 | 判别器过强 / 目标函数有界性被破坏 | 确认 LSGAN ±1；降低判别器更新次数或学习率 |
| `D(policy)` 远小于 -1 | 判别器过快区分 | 减少 `discriminator_batches_per_iteration`，或加 label 平滑 |
| 能走但步态不像 Trot | 风格奖励权重太低 / 参考数据本身不对 | 先看 GIF 确认参考；再把 `amp_reward_weight` 从 2.0 调高 |
| 速度跟踪差 | 任务奖励权重低 / 命令范围太宽 | 提高 `track_linear_velocity` 权重；缩小命令范围 |
| 早期频繁跌倒 | 参考重置比例太高或参考动作不稳 | 检查参考片段；把 `probability` 调到 0.3 |
| 训练很慢 | env 数 / batch 过大 | 降 `num-envs` 或 `discriminator_batch_size` |
| 想换速度/步态 | 参考数据不覆盖 | 用 `--gait Bound/Pace` 和新 `--speeds` 重新采集，再扩大命令范围 |
| 想上台阶/坡面 | 当前高度特征是平地 `z` | 改成相对地形高度，并采集对应动作 |

### 常见误区

1. **只看判别器损失**：LSGAN 损失低不代表策略好，必须同时看风格奖励和实际步态。
2. **用同一段参考反复过拟合**：参考片段之间要覆盖不同速度/相位。
3. **把重置边界样本喂给判别器**：自动重置后的“下一帧”属于新 episode，必须排除。
4. **忽略 dt**：mjlab 任务奖励按 `dt` 缩放，AMP 奖励也要按同一单位。
5. **跨文件拼相邻帧**：必须在单个 `.npz` 内部配对。

---

## 10. 扩展到新步态、新速度、非平地

### 10.1 新步态

```bash
uv run --no-sync python -m go1_tasks.collect_mob_amp \
  --checkpoint <mob.pt> --gait Bound \
  --speeds 0.2 0.4 0.6 --clips-per-speed 3
```

`GAITS` 支持 `Pronk / Trot / Bound / Pace`。然后把
`--agent.algorithm.reference-path` 指到新目录，或把 AMP 环境里的默认路径改掉。
不同步态不要混在同一个数据集目录里，判别器会学到一个“平均风格”。

### 10.2 新速度

- 采集：`--speeds` 加上目标速度；
- 环境：改 `twist.ranges.lin_vel_x`；
- 评估：`eval_amp --vx` 用对应速度（它会去找 `vx_+XpXX/clip_000.npz`）。

### 10.3 非平地

当前 `amp_frame` 的高度是世界系 `z`，并假设地面 `z=0`。上台阶/坡面时需要：

1. 采集时保存“相对当地地面的高度”，或保存地形高度；
2. 在线特征同样改为相对高度；
3. 判别器的输入维度会变，旧 checkpoint 不能直接复用。

---

## 11. 代码索引

| 主题 | 位置 |
| --- | --- |
| 采集快照 | `collect_mob_amp.py:35` |
| 固定 MoB 命令 | `collect_mob_amp.py:51` |
| 连续片段采集 | `collect_mob_amp.py:68` |
| 片段校验 | `collect_mob_amp.py:105` |
| 46 维特征 | `amp_data.py:36` |
| 在线特征 | `amp_data.py:84` |
| 数据集与配对 | `amp_data.py:113` |
| 重置事件 | `amp_events.py:13` |
| AMP 环境配置 | `env_cfgs.py:653` |
| AMP 超参数 | `rl_cfg.py:20` |
| 判别器 | `amp.py:19` |
| 梯度惩罚 | `amp.py:43` |
| 特征归一化 | `amp.py:55` |
| 有界奖励 | `amp.py:115` |
| replay buffer | `amp.py:120` |
| `AmpPPO` | `amp.py:151` |
| 时序取帧 | `amp.py:259` |
| 判别器更新 | `amp.py:284` |
| 保存/恢复 | `amp.py:353` |
| 定量评估 | `eval_amp.py:22` |
| 参考动作可视化 | `visualize_amp_reference.py:37` |
| 端到端测试 | `tests/test_go1_amp.py:42` |

---

## 12. 复现命令速查

```bash
# 1. 安装与列出任务
uv sync
uv run go1-list-tasks

# 2. 采集参考动作（MoB 检查点换成自己的）
uv run --no-sync python -m go1_tasks.collect_mob_amp \
  --checkpoint logs/rsl_rl/go1_mob/<RUN>/model_5050.pt \
  --gait Trot --speeds 0.2 0.4 0.6 0.8 \
  --clips-per-speed 3 --seconds 5 --warmup-seconds 1

# 3. 检查参考数据
uv run --no-sync python -c 'from go1_tasks.amp_data import AmpMotionDataset; \
d=AmpMotionDataset("logs/amp/mob_reference/trot", fps=50.0); \
print(d.num_clips, len(d), d.sample(4).shape)'

# 4. 看 GIF
uv run --no-sync python -m go1_tasks.visualize_amp_reference \
  logs/amp/mob_reference/trot/vx_+0p60/clip_000.npz \
  --output logs/amp/reference_preview_vx_0p60.gif --seconds 3

# 5. 训练
uv run go1-train Go1-Velocity-AMP \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 10000 \
  --agent.save-interval 250 \
  --agent.logger tensorboard \
  --agent.upload-model False

# 6. 定量评估
uv run --no-sync python -m go1_tasks.eval_amp \
  logs/rsl_rl/go1_amp/<RUN>/model_5500.pt --vx 0.6 --seconds 5

# 7. 播放
uv run go1-play Go1-Velocity-AMP \
  --checkpoint-file logs/rsl_rl/go1_amp/<RUN>/model_5500.pt \
  --viewer viser

# 8. 测试
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ -q
```

> 注：本机 ROS 的 pytest 插件会干扰 pytest 启动，所以测试命令加了
> `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。