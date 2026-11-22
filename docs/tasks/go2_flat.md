# Go2 平地速度跟踪

`Go2-Velocity-Flat` 是 Go2 的平地基线任务：actor 看 48 维本体观测（含机身线速度，
不含地形扫描），只跟踪平面速度指令。与 `Go2-Velocity-Rough` 的差别是地形换成无限
平面、actor 多 3 维线速度；与 `Go2-Velocity-AMP` 的差别是保留线速度且没有风格判别器。

## 任务 ID

| 项 | 值 |
| --- | --- |
| 任务 ID | `Go2-Velocity-Flat` |
| 环境配置 | `src/go2_tasks/env_cfgs.py::unitree_go2_flat_env_cfg` |
| Runner | `VelocityOnPolicyRunner`（`mjlab.tasks.velocity.rl`，PPO） |
| 检查点目录 | `logs/rsl_rl/go2_flat/<run>/`（仓库里没有该目录） |

## 观测与动作

actor 的传感器与关节项都带均匀噪声（训练时），`actions`/`command` 与 critic 不加噪声。

| 项 | 维数 | 说明 |
| --- | ---: | --- |
| `base_lin_vel` | 3 | `imu_lin_vel`，噪声 ±0.5 m/s |
| `base_ang_vel` | 3 | `imu_ang_vel`，噪声 ±0.2 rad/s |
| `projected_gravity` | 3 | 噪声 ±0.05 |
| `joint_pos` | 12 | 相对默认角，噪声 ±0.01 rad |
| `joint_vel` | 12 | 噪声 ±1.5 rad/s |
| `actions` | 12 | 上一控制步动作 |
| `command` | 3 | `[vx, vy, wz]` |
| **actor 合计** | **48** | |
| critic 额外项 | 4 项 | `foot_height` 4、`foot_air_time` 4、`foot_contact` 4、`foot_contact_forces` 12 |
| **critic 合计** | **72** | 无 187 维高度扫描（平地任务移除了 raycast 传感器） |
| 动作 | 12 | `q_des = q_default + scale · a`，scale = `[0.125, 0.25, 0.25]`（hip/thigh/calf） |
| 控制频率 | 50 Hz | 物理步 0.005 s，decimation 4 |
| episode 长度 | 20 s | 1000 个控制步；play 模式为 `1e9` |
| 指令范围（训练） | x ±1.0 m/s，y ±1.0 m/s，yaw ±0.5 rad/s | 3–8 s 重采样，10% 站立、30% 航向控制、20% 只前进 |
| 指令范围（play） | x −1.5…2.0 m/s，yaw ±0.7 rad/s | `unitree_go1_flat_env_cfg` 的 play 覆盖 |

关节顺序 `FR, FL, RR, RL`，默认姿态 `[0.0, 0.8, −1.5]`（每腿），初始 `base_link`
高度 0.38 m；`Kp=25`、`Kd=0.5`、力矩上限 23.5 Nm、空载速度上限 30 rad/s。

## 训练

注册的默认 `num_envs` 是 1，训练必须显式指定。

```bash
uv run go2-train Go2-Velocity-Flat \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard
```

| 超参 | 值 |
| --- | --- |
| 网络 | actor/critic 均 512-256-128，ELU，不做观测归一化 |
| 动作分布 | 高斯，标量 std，初值 1.0 |
| lr | 1e-3，`adaptive`，`desired_kl=0.01` |
| PPO | 5 epochs、4 mini-batches、clip 0.2、entropy 0.01、γ 0.99、λ 0.95、grad-norm 1.0 |
| 采样 | 24 steps/env，`max_iterations=10000`，`save_interval=50` |
| 环境数 | 1024（需手动传） |

课程：只有速度指令课程（`command_vel`），没有地形课程。三个阶段按环境步数切换：
0 步 x ±1.0/yaw ±0.5，`5000×24` 步 x −1.5…2.0/yaw ±0.7，`10000×24` 步 x −2.0…3.0
（第二段改 x 与 yaw，第三段只有 x）。第 10000 轮结束时第三段刚触发，实际很少生效。

## 评估

```bash
uv run go2-play Go2-Velocity-Flat \
  --checkpoint-file logs/rsl_rl/go2_flat/YOUR_RUN/model_9999.pt \
  --num-envs 4 --viewer viser

uv run python tools/eval_go2_task.py \
  --task Go2-Velocity-Flat \
  --checkpoint logs/rsl_rl/go2_flat/YOUR_RUN/model_9999.pt \
  --command 0.8 0 0 --steps 300
```

看 `tools/eval_go2_task.py` 输出的 `error_vel_xy`（m/s）、`error_vel_yaw`（rad/s）、
`terminated` 百分比和平均机身高度；训练日志里看 `Episode_Metrics/mean_action_acc`、
`Episode_Termination/*` 与 `Curriculum/command_vel/*` 的当前段。注意该评测脚本对
采样指令统计的是整个 episode 的平均误差，不能和固定指令单点测试直接比。

## sim2sim

`sim2sim/go2_sim.py` 的 `--policy` 按第一层宽度自动识别 48 维 Flat 检查点：

```bash
uv run python sim2sim/go2_sim.py \
  --policy logs/rsl_rl/go2_flat/YOUR_RUN/model_9999.pt \
  --terrain flat --command 0.8 0 0 \
  --headless --steps 400
```

支持全部 5 种地形（`flat`、`rough`、`stairs`、`rl_sar`、`go2_rough`），但 Flat 只在
平面上训练过；`go2_rough` 的格子、出生点和级数由导出文件决定。无窗口模式会打印
后半段的平均 `vx/vy/wz` 与绝对误差，用来快速比较检查点。48 维策略和 45 维策略共用
一个入口，不需要额外开关。

## 已知限制

- 仓库里没有 `logs/rsl_rl/go2_flat/` 运行目录，本页的行为描述没有本地检查点验证。
- 48 维观测依赖机身线速度估计；部署端没有可靠的速度估计时应改用 45 维的
  `Go2-Velocity-Rough` / `Go2-Velocity-AMP`。
- 平地任务的终止条件只有 70° 倾角摔倒和时间到，没有 `illegal_contact`、
  `out_of_terrain_bounds`，也没有大腿/躯干接触惩罚，摔倒之外的异常姿态不会被终止。
- sim2sim 用独立 MuJoCo 引擎和手写 PD，接触与摩擦参数与 mjlab/MuJoCo-Warp 训练不同，
  同一检查点的速度误差不能跨引擎直接比较。
- 在 `rough`/`stairs`/`go2_rough` 上跑属于分布外测试，只能作为鲁棒性观察。

## 参考

- [Go2 训练与部署约定](../go2.md)
- [AMP 训练教程](../amp_tutorial.md)
- sim2sim 命令与地形说明：[`sim2sim/README.md`](../../sim2sim/README.md)
