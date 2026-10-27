# 使用 MoB 策略采集 AMP 参考动作

本文介绍 AMP 的参考数据采集、读取和训练入口。完整的操作顺序与原理见
[AMP 完整教程](amp_tutorial.md)。采集脚本在本仓库的
Go1 mjlab 环境中运行已训练的 MoB 策略，为每段轨迹固定步态和速度命令，保存
机器人的真实仿真状态，而不是 MoB 策略的 1710 维历史观测。

## 采集方法

在仓库根目录运行，并将 `--checkpoint` 指向已经播放验证过的 MoB 检查点。
以下命令使用当前训练结果：

```bash
uv run --no-sync python -m go1_tasks.collect_mob_amp \
  --checkpoint logs/rsl_rl/go1_mob/2026-10-27_15-04-14/model_5050.pt \
  --gait Trot \
  --speeds 0.2 0.4 0.6 0.8 \
  --clips-per-speed 3 \
  --seconds 5 \
  --warmup-seconds 1
```

默认输出目录为 `logs/amp/mob_reference/trot/`，每个速度、每段轨迹分别保存为
一个 `.npz` 文件。`logs/` 已被 Git 忽略。更换检查点时，需要显式指定新路径。
没有可用 GPU 时，可加 `--device cpu`。

采集脚本使用 50 Hz 的播放环境，并关闭自动重置。机器人在预热或录制期间跌倒
时，脚本丢弃当前片段并重试，默认最多重试五次。因此，输出文件不会把前一次
运行的末帧与重置后的首帧接在一起。脚本不会覆盖已有文件；中断后加
`--resume` 可校验并跳过已经保存的片段，再继续采集。

## 文件内容

每个文件包含 `N = ceil(采集秒数 × fps) + 1` 帧，因此可从同一段轨迹中得到
`N - 1` 组相邻帧状态转移：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `root_pos_w`、`root_lin_vel_w`、`root_ang_vel_w` | `(N, 3)` | 世界坐标系下的机身状态。 |
| `root_quat_w` | `(N, 4)` | 机身四元数，顺序为 `wxyz`。 |
| `root_lin_vel_b`、`root_ang_vel_b` | `(N, 3)` | 机身坐标系下的线速度和角速度。 |
| `joint_pos`、`joint_vel` | `(N, 12)` | 实际关节状态，顺序见 `joint_names`。 |
| `foot_pos_w` | `(N, 4, 3)` | 四只脚的世界坐标位置，顺序为 `FR、FL、RR、RL`。 |
| `velocity_command` | `(N, 3)` | 固定的 `[vx, vy, wz]` 速度命令。 |
| `behavior_command` | `(N, 8)` | 固定的 MoB 步态和行为参数。 |
| `fps`、`joint_names`、`foot_site_names`、`source_checkpoint`、`seed` | 元数据 | 用于核对数据来源、时间间隔及关节顺序。 |

AMP 数据读取器只从**同一个文件内**抽取相邻帧。参考动作与 AMP 策略产生的
轨迹必须经过同一套特征提取和坐标变换，才能交给判别器比较。

## 读取参考动作并构造 AMP 输入

`go1_tasks.amp_data` 已实现数据校验、特征提取和随机抽样。下面的路径就是上述
命令生成的目录：

```python
from go1_tasks.amp_data import AmpMotionDataset, amp_frame_from_robot

motions = AmpMotionDataset("logs/amp/mob_reference/trot", fps=50.0)
reference_batch = motions.sample(256, device="cpu")  # (256, 92)

# AMP 环境创建后，robot = env.scene["robot"]
policy_frame = amp_frame_from_robot(robot, joint_names=motions.joint_names)
# policy_frame: (环境数, 46)
```

每帧的 46 维依次是：机身世界高度 1 维、机身坐标系下的重力方向 3 维、
机身线速度 3 维、角速度 3 维、12 个关节位置、12 个关节速度，以及相对机身
的位置表示的四只脚坐标 12 维。把同一条轨迹中的第 `t` 帧和第 `t+1` 帧拼接，
得到判别器输入的 92 维。足端顺序固定为 `FR、FL、RR、RL`；关节顺序采用
采集文件中的 `joint_names`，在线提取时会校验顺序。

参考数据和在线状态共用 `amp_frame` 特征函数。特征不包含世界坐标 `x/y`、
绝对朝向或速度命令，因此判别器比较的是动作状态而不是行进地点或命令值。
当前高度采用世界坐标 `z`，适用于这里的平地数据；以后加入非平地运动时，
需要将参考数据与在线状态同时改成相对地面的高度。训练时还要确保环境步长为
`1 / motions.fps`，且在线两帧来自同一个环境的相邻时刻；环境重置后的跨段
样本不能送入判别器。

## 运行 AMP 训练

采集并核对上述参考动作后，在仓库根目录启动训练：

```bash
uv run go1-train Go1-Velocity-AMP --env.scene.num-envs 1024
```

默认参考目录是 `logs/amp/mob_reference/trot`。如果数据在其他位置，可添加
`--agent.algorithm.reference-path 目录路径`。新任务的 checkpoint 存在
`logs/rsl_rl/go1_amp/` 下。AMP 策略从随机参数开始训练；已有的 MoB 检查点
只用于生成参考动作，并不会直接作为 AMP 策略初值。

训练环境使用平地和 `0.2–0.8 m/s` 的前进命令，与当前采集范围一致。
AMP 策略采用单帧 45 维 actor 观测，不读取 MoB 的行为命令或步态时钟；
Trot 风格由参考动作提供。环境奖励保留速度跟踪、机身直立、关节限位、动作变化率
和足端打滑项；判别器额外对每个未终止的相邻状态转移给出模仿奖励。机器人跌倒
时仍按常规终止并重置，不训练跌倒恢复动作。

判别器输入为前述 92 维状态对，使用参考动作和策略轨迹分别训练正负样本，
并用历史策略转移的缓存减少样本变化。每轮先用当前判别器计算 AMP 奖励完成
PPO 数据采集与更新，再更新判别器。重置边界的转移既不进入判别器训练，
也不获得 AMP 奖励。判别器参数和优化器状态随策略一起写入 checkpoint。

现阶段验证覆盖一轮短训练的数据流和参数更新；最终动作质量仍需要完整训练，
再用播放和速度跟踪指标检查。若新增转弯、侧移或其他步态，先采集对应的
参考动作，再扩大训练命令范围。
