# Dog2 四足机器人训练配置

基于 [mjlab](https://github.com/xxx/mjlab) 框架，参照 Unitree Go1 和 Dog1 的经验，为自己的直驱四足机器人定制的 sim-to-real 训练配置。

## 文件结构

```
src/mjlab/asset_zoo/robots/dog2/
├── __init__.py              # 导出 get_dog2_robot_cfg, DOG2_ACTION_SCALE
├── dog2_constants.py        # 机器人物理参数、执行器、初始姿态、碰撞配置
└── xmls/
    ├── dog2.xml             # MuJoCo 模型（关节、几何、传感器）
    └── assets/              # STL 网格文件

src/mjlab/tasks/velocity/config/dog2/
├── __init__.py              # 注册训练任务 Mjlab-Velocity-Rough-Dog2 / Flat-Dog2
├── env_cfgs.py              # 环境配置（奖励权重、传感器、域随机化、终止条件）
└── rl_cfg.py                # PPO 训练超参数（网络结构、学习率、迭代次数）
```

## 硬件

| 项目 | 参数 |
|---|---|
| 电机 | 达妙 8009 直驱（无减速箱） |
| 关节 | 每腿 3 个：hip（横摆）、thigh（髋）、calf（膝） |
| 自由度 | 12 个驱动关节 + 6 自由度浮基 = 19 qpos |
| 传感器（部署） | IMU（陀螺仪 + 加速度计）+ 电机编码器 |

## 训练

```bash
# 粗糙地形训练（512 环境）
uv run train Mjlab-Velocity-Rough-Dog2 --env.scene.num-envs 512

# 续训
uv run train Mjlab-Velocity-Rough-Dog2 \
  --env.scene.num-envs 512 \
  --agent.resume True \
  --agent.load_run 2026-05-25_22-24-30 \
  --agent.load_checkpoint model_9999.pt

# 平坦地形训练（更多环境，更快）
uv run train Mjlab-Velocity-Flat-Dog2 --env.scene.num-envs 4096
```

## 可视化

```bash
uv run play Mjlab-Velocity-Rough-Dog2 \
  --checkpoint-file logs/rsl_rl/dog2_velocity/<run>/<model>.pt \
  --viewer viser
```

## Sim-to-Real 设计

训练中使用特权信息（terrain scan、contact force 等），但 **actor 策略仅暴露真实机器人可用的传感器**：

| 观测 | Actor（部署） | Critic（训练） |
|---|---|---|
| base_ang_vel | ✅ IMU 陀螺仪 | ✅ |
| projected_gravity | ✅ IMU 加速度计 | ✅ |
| joint_pos / joint_vel | ✅ 电机编码器 | ✅ |
| actions / command | ✅ | ✅ |
| base_lin_vel | ❌ IMU 无法可靠估计 | ✅ |
| height_scan | ❌ 无外部传感器 | ✅ |
| foot_height / contact | ❌ | ✅ |

## 奖励权重

| 奖励 | 权重 | 说明 |
|---|---|---|
| track_linear_velocity | 4.0 | 线速度指令跟踪 |
| track_angular_velocity | 4.0 | 角速度指令跟踪（std=0.5） |
| upright | 1.0 | 保持躯干竖直 |
| pose | 1.0 | 关节接近默认姿态 |
| air_time | 0.1 | 鼓励腾空步态 |
| foot_clearance | 0.3 | 抬脚目标 0.12m |
| foot_swing_height | -0.02 | 抬脚上限 0.2m |
| foot_slip | -0.1 | 惩罚支撑脚滑动 |
| action_rate_l2 | -0.1 | 动作平滑约束 |
| dof_pos_limits | -1.0 | 关节限位惩罚 |
| shank_collision | -0.1 | 惩罚小腿碰地 |
| trunk_head_collision | -0.1 | 惩罚躯干碰地 |

## 执行器配置

| 参数 | 值 |
|---|---|
| 类型 | `IdealPdActuatorCfg`（显式 PD 控制，对应真实力矩控制） |
| stiffness Kp | 40.0 Nm/rad |
| damping Kd | 1.0 Nm/(rad/s) |
| effort_limit (hip/thigh) | 20.0 Nm |
| effort_limit (calf) | 25.0 Nm |
| action_scale | 0.125 rad（hip/thigh），0.156 rad（calf） |

## 训练超参数

| 参数 | 值 |
|---|---|
| 网络结构 | [512, 256, 128] ELU |
| 初始探索 std | 1.0 |
| 学习率 | 1e-3（adaptive schedule） |
| PPO clip | 0.2 |
| 折扣因子 γ | 0.99 |
| GAE λ | 0.95 |
| 每环境步数 | 24 |
| 总迭代 | 15,000 |
| 总步数 | ~1.84 亿（15k × 512 env × 24 steps） |

## 训练结果参考

| 指标 | 最终值 |
|---|---|
| track_linear_velocity | 2.86 / 4.0 |
| terrain_levels mean | 5.77 / 9 |
| velocity error xy | ~0.66 m/s |
| air_time_mean | ~0.14 s |
| fell_over rate | ~0% |
