# Go2 训练与部署约定

Go2 任务位于 `src/go2_tasks`，模型来自用户提供的
`/home/yue/Desktop/go2_description/mjcf/go2.xml`，仓库中保存了自包含副本
（与源文件逐字节一致）。Go2 与 Go1 共用任务算法，但机器人模型、电机限制、
默认姿态和策略 I/O 顺序均由 Go2 profile 单独定义。

连杆质量/惯量、关节限位、被动阻尼、armature 和 `frictionloss` 保留源
`go2_description` MJCF 的值，总质量约 `15.2 kg`。训练执行器采用本机
`robot_lab` 中 `UNITREE_GO2_CFG` 的 DC 电机配置：全部 12 个关节统一
`Kp=25`、`Kd=0.5`、连续/饱和力矩上限 `23.5 Nm`、空载速度上限
`30 rad/s`。默认站姿 `[0.0, 0.8, -1.5]`，机身初始高度 `0.38 m`，
动作尺度 `[0.125, 0.25, 0.25]`。sim2sim 使用同一 PD 和力矩—速度限幅。

## 任务

```bash
uv run go2-list-tasks
uv run go2-train Go2-Velocity-Flat --env.scene.num-envs 1024
uv run go2-play Go2-Velocity-Flat --checkpoint-file /path/to/model.pt --viewer viser
```

Go2 提供 Rough、Flat、MoB、AMP、CTS 五个任务。AMP 使用
`logs/amp/go2_reference` 下的全部类别；读取器会递归加载所有 `.npz`，
因此不能与 Go1 动作混放。

### 从 OCS2 采集多步态参考动作

当前参考集为 50 Hz、每段截取 4.5 s，按 OCS2 实际支持的步态组织：

| 类别 | 片段 | 平均前进速度 | 对角足相关性 |
| --- | --- | ---: | ---: |
| `trot` | `trot_ly_0p30_01`, `trot_ly_0p60_01` | 0.088 / 0.249 m/s | +1.00 |
| `flying_trot` | `flying_trot_ly_0p30_01` | 0.118 m/s | +1.00 |
| `pace` | `pace_ly_0p15_01` | 0.057 m/s | -0.56 |
| `dynamic_walk` | `dynamic_walk_ly_0p30_01` | 0.111 m/s | -0.24 |
| `static_walk` | `static_walk_ly_0p30_01` | 0.105 m/s | -0.18 |
| `turn_left` | `turn_left_ly_0p00_01`, `turn_left_rx_0p60_01` | 约 0，偏航 +0.102 / +0.520 rad/s | +1.00 |
| `turn_right` | `turn_right_ly_0p00_01`, `turn_right_rx_0p60_01` | 约 0，偏航 -0.102 / -0.522 rad/s | +1.00 |
| `side_left` | `side_left_lx_0p30_01`, `side_left_lx_0p60_01` | 横向 +0.063 / +0.182 m/s | - |
| `side_right` | `side_right_lx_0p30_01`, `side_right_lx_0p60_01` | 横向 -0.060 / -0.187 m/s | - |
| `backward_trot` | `backward_trot_ly_0p30_01`, `backward_trot_ly_0p60_01` | -0.093 / -0.250 m/s | +1.00 |
| `stand` | `stand_01` | 约 0 | 四足持续接触 |

对角足相关性为 `(corr(FR,RL) + corr(FL,RR)) / 2`；Trot 类应接近 +1，
walk 类因为四拍支撑序列会接近 0 或为负，这是正常的。Pace 是同侧两拍
（`FR+RR` 与 `FL+RL`），对角相关性应为负、同侧相邻足正相关，并且机身
横向速度摆幅明显大于 Trot；采集器因此对 pace 单独放宽了 `|vy|` 门槛。

先在终端 A 启动 `/home/yue/Desktop/go2_ocs2_ws/verified_humble_ws` 的仿真：

```bash
cd /home/yue/Desktop/go2_ocs2_ws/verified_humble_ws
source env.sh
ros2 launch verified_go2_mujoco sim.launch.py headless:=true rviz:=false
```

终端 B 使用 ROS Humble 的 Python 运行自动采集器；此时不要同时运行键盘或手柄节点：

```bash
cd /home/yue/go1_tasks
source /home/yue/Desktop/go2_ocs2_ws/verified_humble_ws/env.sh
/usr/bin/python3 tools/record_go2_ocs2_trot.py \
  --output logs/amp/go2_ocs2_raw/<name> \
  --motion trot --ly 0.30 --seconds 5
```

`--motion` 可选 `trot`、`flying_trot`、`pace`、`dynamic_walk`、
`static_walk`、`turn_left`、`turn_right`、`side_left`、`side_right`、
`backward_trot`、`stand`。采集器先发送 stance 命令，
确认机身连续站稳约 2 秒，再切换步态；只有当最近约 3 秒内机身高度、倾角、
前进速度（转向时为偏航速度）稳定，且支撑相出现预期的足端交替后才开始
录包。录制中失稳会报错并拒绝这段数据。每次运行必须使用新的 `--output`。

录完的原始 bag 先抽取状态，再转换为 AMP 格式：

```bash
/usr/bin/python3 tools/extract_go2_ocs2_bag.py \
  logs/amp/go2_ocs2_raw/<name> logs/amp/go2_ocs2_raw/<name>_raw.npz
MPLCONFIGDIR=/tmp/mplcache .venv/bin/python tools/convert_go2_ocs2_amp.py \
  logs/amp/go2_ocs2_raw/<name>_raw.npz \
  logs/amp/go2_reference/<category>/<name>.npz --max-seconds 4.5
```

转换器要求关节顺序与 Go2 MJCF 一致（`FL, FR, RL, RR`），用 MuJoCo 正运动学
重新计算足端世界坐标，并输出 `root_*`、`joint_*`、`foot_pos_w`、`fps`、
`joint_names`、`foot_site_names` 字段。AMP 训练时
`Go2-Velocity-AMP` 通过 `GO2_AMP_REFERENCE_PATH` 读取整个目录。

生成前先用可视化检查片段（`--robot go2` 使用 Go2 模型做正运动学）：

```bash
uv run --no-sync python -m go1_tasks.visualize_amp_reference \
  logs/amp/go2_reference/trot/trot_ly_0p60_01.npz --robot go2 \
  --output logs/amp/reference_previews/go2/trot_0p60.gif
```

当前 17 段片段的预览已生成在 `logs/amp/reference_previews/go2/`。Trot 与
Flying Trot 的对角足同相、相邻足反相；Pace 是同侧足同相；Walk 是四拍
支撑；左右转向的偏航速度符号相反。

训练前运行完整数据检查；它会验证字段、50 Hz 帧率、关节/足端顺序、四元数、
世界/机体系速度变换、逐帧足端正运动学以及类别分布：

```bash
MPLCONFIGDIR=/tmp/mplcache .venv/bin/python \
  tools/validate_go2_amp_dataset.py logs/amp/go2_reference
```

Go2 AMP使用与MoB任务一致的正常速度范围：前后`-1.0–1.0 m/s`、侧向
`-0.6–0.6 m/s`、偏航`-1.0–1.0 rad/s`，并同时启用线速度与角速度跟踪
奖励。AMP是风格先验，任务命令不需要限制在参考动作的实测速度范围内；
速度跟踪奖励负责驱动策略超出短OCS2片段的速度，同时判别器约束关节、姿态、
足端和速度转移保持自然。

角速度跟踪奖励权重为`2.0`、高斯容差为`0.25 rad/s`。较小容差用于避免策略
在`±1 rad/s`命令范围内保持不转却仍获得较高奖励；线速度跟踪权重同为`2.0`。

连续均匀采样几乎不会产生严格的零速度命令，因此训练时单独将5%的环境设为
`[0, 0, 0]`，并在参考集中提供稳定站立片段。两者共同训练零命令保持静止；
播放时不随机强制站立，手动命令仍可自由切换。

Go2 AMP训练同时保留域随机化和外部扰动：编码器偏置`±0.015 rad`、机身
质心偏移、机身伪惯量`alpha ±0.05`、PD增益倍率`0.9–1.1`、关节摩擦
`0–0.05`，以及足端滑动/旋转/滚动摩擦随机化。关节重置位置和速度均加入
`±0.1`扰动；每个环境每`4–8 s`施加一次机身速度突变，`x/y/yaw`分别从
`[-0.3, 0.3]`采样。Go2质量为零的foot辅助体不会参与伪惯量随机化，避免
产生奇异惯量和NaN。

OCS2 本身支持横向目标，映射为 `v_y = -lx * targetDisplacementVelocity`。
采集左右侧移可分别使用：

```bash
/usr/bin/python3 tools/record_go2_ocs2_trot.py \
  --motion side_left --lx 0.3 --seconds 5 \
  --output logs/amp/go2_ocs2_raw/side_left_lx_0p30_01

/usr/bin/python3 tools/record_go2_ocs2_trot.py \
  --motion side_right --lx 0.3 --seconds 5 \
  --output logs/amp/go2_ocs2_raw/side_right_lx_0p30_01
```

必须先完成侧移录制、转换和稳定性检查，再根据实测速度决定课程学习的侧向
范围；不能把手柄输入 `lx=0.3` 直接当成 `0.3 m/s`。

完整训练使用：

```bash
uv run go2-train Go2-Velocity-AMP --env.scene.num-envs 1024
```

不使用 W&B 时增加 `--agent.logger tensorboard`。训练启动时应打印
`[AMP] 17 clips, 3825 transitions, 50 Hz`；ONNX 会与检查点一起写入
`logs/rsl_rl/go2_amp/<run>/`。

论文 [Adversarial Motion Priors Make Good Substitutes for Complex Reward
Functions](https://arxiv.org/html/2203.15103) 的参考数据总计约 4.5 秒，
由 pace、trot、canter 和原地转向的短片段组成；论文没有报告各类别的精确
时长占比。其[公开代码](https://github.com/escontra/AMP_for_hardware/tree/main/datasets/mocap_motions)
另提供六个文件，并为每个文件设置 `MotionWeight`，按文件权重抽样后再从
文件内抽时间。本仓库的每段 4.5 s 取自该量级，但类别更多，不能当作论文
原始比例的精确复刻。OCS2 没有 canter，Flying Trot 是最接近的可用步态。

## 实机接口

策略和部署端统一使用以下关节顺序：

```text
FR_hip, FR_thigh, FR_calf,
FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf,
RL_hip, RL_thigh, RL_calf
```

控制周期为 `0.02 s`（50 Hz）。策略动作转换为：

```text
q_des = q_default + action_scale * action
```

其中每条腿的默认角度为 `[0.0, 0.8, -1.5]`，动作尺度为
`[0.125, 0.25, 0.25]`。策略约定的 PD 参数为 `Kp=25`、`Kd=0.5`。

两组力矩数字的含义不同：

- 本项目 Go2 训练电机输出限幅：12 个关节均为 `23.5 Nm`，参考 RobotLab 的
  `DCMotorCfg`；并采用 `30 rad/s` 力矩—速度曲线。
- 源 Go2 MJCF/URDF 的电机/关节上限：hip/thigh `23.7 Nm`，calf/knee
  `45.43 Nm`；源模型仍保存这些数值，但训练时源 `<motor>` 被 DC 执行器替换。

部署转换常量和函数集中在 `go2_tasks.deployment`；真机控制器需按目标机器的
实测电机限制校验下发命令，并保留急停、通信超时、姿态保护、关节
软限位、速度限制和力矩饱和。未经吊架和低增益测试，不能直接在自由站立状态
加载新策略。

Flat 策略输入为 48 维：机体系线速度 3、角速度 3、投影重力 3、相对关节角
12、关节速度 12、上一动作 12、速度命令 3。所有关节量都必须先映射到上述固定
顺序。机体系线速度需要由实机状态估计器提供；若部署端没有可用的线速度估计，
应改用不含 `base_lin_vel` 的 AMP/CTS 策略（45 维）。MoB 使用历史观测，部署时
必须复用训练端的历史缓冲逻辑。

## sim2sim 验证

`sim2sim/go2_sim.py` 在独立 MuJoCo 引擎中加载同一份 Go2 MJCF，其控制常量
由测试与部署常量核对，不导入 `mjlab`：

```bash
uv run python sim2sim/go2_sim.py --headless --steps 100
uv run python sim2sim/go2_sim.py \
  --policy logs/rsl_rl/go2_flat/YOUR_RUN/model_9999.pt \
  --terrain flat
```

无检查点时机器人保持被动站姿，用于检查模型、地形和执行器接口。`go2_sim.py`
目前只接受 48 维 Flat/部署 actor；Rough 的 235 维高度扫描策略和 CTS 策略
需要单独的推理适配。真机部署前应先在 sim2sim 中验证，并保留急停、通信超时、
姿态保护、关节软限位、速度限制和力矩饱和。
