# Go1 sim2sim

本目录随 `go1-tasks` 一起发布，在独立的 MuJoCo 场景中运行 Go1 策略；
推理脚本不导入 `mjlab` 或 `rsl_rl`。支持本仓库的 `Go1-Velocity-Flat`
和 `Go1-Velocity-MoB` 检查点，以及旧版 48 维 RMA 教师/适应器检查点。
**当前 `Go1-Velocity-Rough` 的 235 维高度扫描策略与 CTS 策略不受这些脚本支持。**
本仓库不附带训练检查点。

RMA 运行程序使用旧项目两阶段训练得到的检查点；本仓库没有 RMA 训练任务，
需要自行提供兼容的教师和适应器文件。运行脚本只需 Python、MuJoCo、NumPy、
PyTorch 和本目录的 Go1 模型文件。

## 安装与运行

```bash
cd go1-tasks
uv sync --locked
uv run python sim2sim/go1_sim.py --headless --steps 100
```

不提供检查点时，机器人保持默认站姿。已有兼容的 RMA 两阶段检查点时运行：

```bash
uv run python sim2sim/go1_sim.py \
  --teacher /path/to/model_9999.pt \
  --adapter /path/to/adapter_1000.pt \
  --terrain rough
```

运行本仓库不含高度扫描、输入为 48 维的 Go1 平地速度策略：

```bash
uv run python sim2sim/go1_sim.py \
  --policy /path/to/go1_velocity/RUN/model_9999.pt \
  --terrain flat
```

`--policy` 不能与 `--teacher` 或 `--adapter` 同时使用。包含 187 维高度扫描、
总输入为 235 维的当前 Rough 策略检查点会被拒绝。

默认地形是 `--terrain rough`：8 米见方的高度场，包含 2～10 厘米的随机台阶和一块平坦起步区域。`--terrain stairs` 提供高度递增至 20 厘米的 5 厘米台阶；`--terrain flat` 使用原始平面；`--terrain rl_sar` 使用完整的 rl_sar 障碍场。`--terrain-seed N` 可改变崎岖地形的随机布局。这些生成地形借鉴了训练配置，并不是训练时 10×20 课程地图的精确导出。

`rl_sar` 地形保留了原有平面、几何障碍、坡道、楼梯、随机方块场和两张图像高度场。复制的原始资源及 Apache-2.0 许可证位于 `terrain_assets/`。

可加 `--headless --steps 1000 --command 0.3 0 0` 进行无窗口冒烟测试。适应模块检查点必须对应所选教师策略。极短冒烟训练得到的检查点**不能**当作可正常行走的策略。

键盘操作：`1` 启用 RMA，`0` 被动站立，`R` 重置，`W/S` 前进/后退，`A/D` 横向移动，`Q/E` 转向，方向键上/下控制前进/后退，空格清零速度指令。Linux 手柄 `/dev/input/js0`：`A` 启用、`B` 被动模式、`X` 重置；左摇杆控制平面速度，右摇杆水平轴控制转向。传入 `--joystick ''` 可禁用手柄。MuJoCo 查看器仍可用鼠标控制相机。

在仓库根目录运行 `PYTHONPATH=sim2sim uv run python -m unittest discover -s sim2sim/tests`，
可以在没有训练好策略的情况下检查场景和推理接口。

## MoB 策略（1710 维观测）

在独立 MuJoCo 场景中运行可调步态的 Go1 策略：

```bash
uv run python sim2sim/go1_mob_sim.py \
  --checkpoint logs/rsl_rl/go1_mob/YOUR_RUN/model_6300.pt \
  --terrain flat
```

无窗口测试可在命令末尾添加 `--headless --steps 2000`。MoB 任务在平地上训练，因此建议从 `flat` 开始。`rough`、`stairs` 和 `rl_sar` 属于分布外测试，并非与训练条件一致的场景。MoB 运行程序使用独立的 `Kp=20`、`Kd=0.5` 和 `0.125/0.25/0.25` 关节动作尺度，不会改变原有 RMA 运行程序。

键盘操作：`1/2/3/4` 选择并启用 Pronk（四足同跳）/Trot（对角小跑）/Bound（前后足成对跳）/Pace（同侧足成对走），`0` 被动模式，`R` 重置，`W/S` 前进/后退，`A/D` 横向移动，`Q/E` 转向，空格停止，`Z/X` 调整机身高度，`C/V` 调整俯仰，`F/G` 调整步频，`B/N` 调整站姿宽度，`H/J` 调整摆腿高度。

MoB 手柄：`A` 启用、`B` 被动模式、`X` 重置、`Y` 或方向键右切换到下一步态、方向键左切换到上一步态；方向键上/下依次选择要调整的参数（机身高度、俯仰、步频、站姿宽度、摆腿高度），`LB/RB` 每按一次将当前参数减小/增大一档。默认选中机身高度；终端会显示当前选中的参数和值。左摇杆控制平面速度，右摇杆水平轴控制转向。方向键使用 Linux 手柄轴 6/7，按钮编号按常见 Xbox 布局；不同设备可能需要调整映射。`--joystick ''` 可禁用手柄。

策略网络接收按观测项排列的最近 30 个控制周期数据：机身角速度、重力方向、12 个关节位置、12 个关节速度、上一步 12 维动作、3 维速度指令、8 维行为指令和 4 维步态相位时钟。每个观测项的历史按从旧到新排列，合计 1710 维输入；策略以 50 Hz 输出 12 维关节位置偏移，物理仿真以 200 Hz 运行。部署观测不加噪声，训练时使用了观测噪声。

## RMA 观测约定

RMA 主策略接收当前 48 维观测；适应网络接收此前 50 帧观测（从旧到新），输出 8 维隐变量。动作是按 `FR, FL, RR, RL` 顺序排列的 12 维关节位置偏移。控制频率为 50 Hz，物理仿真频率为 200 Hz。部署时使用无噪声观测，训练时加入观测噪声进行增强。这里的 MuJoCo 引擎和场景与 `mjlab`/MuJoCo-Warp 训练环境不同；在考虑真机部署前，应先在 sim2sim 中验证运动表现。
