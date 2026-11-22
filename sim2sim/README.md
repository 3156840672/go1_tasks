# Quadruped sim2sim

在独立 MuJoCo 场景里运行本仓库训练的四足策略。推理脚本只依赖 Python、MuJoCo、
NumPy、PyTorch 和本目录的模型数据，不导入 `mjlab` 或 `rsl_rl`；本仓库不附带
训练好的检查点，下面的命令都要换成自己的路径。

| 脚本 | 策略 | 观测 |
| --- | --- | --- |
| `go1_sim.py` | Go1-Velocity-Flat、RMA 两阶段（teacher + adapter） | 48、48 + 8 维隐变量 |
| `go1_mob_sim.py` | Go1 MoB 可调步态策略 | 1710（30 帧 × 57 维） |
| `go2_sim.py` | Go2 Flat、AMP、盲 Rough、论文式 LSTM 学生、HIM、CTS 学生 | 48、45、45 + LSTM 状态、6 × 45 历史、45 + 5 × 45 历史 |

三个脚本共用同一套命令行：`--terrain`、`--terrain-seed`、`--terrain-type`、
`--terrain-column`、`--terrain-level`、`--headless`、`--steps`、`--command`、
`--camera`、`--follow-distance`、`--shadows/--no-shadows`、`--joystick` 的含义完全
一致（见 §1.1），只有"加载哪个检查点"的开关按策略格式不同。

地形也是通用的：`--terrain` 的 5 个取值对三个脚本都可用，`go2_rough` 这个训练地形
导出同样能在 Go1 和 MoB 上装载（出生点、光源跟随、渲染修正都走同一套代码），只是
Go1/MoB 的策略没在这套地形上训练过，属于分布外测试。Go2 侧能配 `go2_rough` 的是
45/48 维策略和带历史的策略，包括盲的 `Go2-Velocity-Rough`、`Go2-Velocity-Rough-AMP`、
HIM、论文式 LSTM 学生和 CTS 学生；旧的 235 维 Rough 检查点没有入口（它需要 187 维
地形扫描，而独立 MJCF 没有 raycast 传感器）。Go1 自己的训练地形没有导出。

三个入口脚本只做"解析参数 → 构造模拟器 → 启动循环"，实现都在 `quadruped_sim/`，
这个包同样不导入训练依赖：

| 模块 | 内容 |
| --- | --- |
| `quadruped_sim/cli.py` | 三个脚本共用的参数（地形、运行、查看）与无窗口统计输出 |
| `quadruped_sim/terrain.py` | 生成 `flat`/`rough`/`stairs`/`rl_sar`，装载 `go2_rough` 导出，列出可用列与级数 |
| `quadruped_sim/scene.py` | 导出地形的通用修正：光源跟随、`stat` 尺度、裁剪面、出生点 |
| `quadruped_sim/joystick.py` | `/dev/input/js*` 手柄轮询，按键 0/1/2 = 启用/被动/重置 |
| `quadruped_sim/viewer.py` | 键盘速度指令、相机摆放、按键分发、交互循环 |
| `quadruped_sim/robots/go1.py` | Go1 MJCF 装配（armature、执行器、地形）+ RMA/平地模拟器 |
| `quadruped_sim/robots/go1_mob.py` | MoB 的接触参数、步态/行为指令槽位和 1710 维观测 |
| `quadruped_sim/robots/go2.py` | Go2 MJCF 装配、电机限幅、渲染修正和四种策略共用的模拟器 |
| `quadruped_sim/policies/actor.py` | 通用 MLP actor 与检查点维度校验 |
| `quadruped_sim/policies/cts.py` | CTS 学生检查点与 ONNX 推理 |
| `quadruped_sim/policies/rma.py` | Go1 RMA teacher 与 adapter 网络 |
| `quadruped_sim/policies/go2.py` | Go2 论文式 LSTM 学生与 HIM 历史编码器 |
| `quadruped_sim/policies/mob.py` | MoB 的 1710 维输入网络 |

要加新机器人或新策略：常量和 `make_model` 放进 `robots/`，网络与检查点加载放进
`policies/`，入口脚本保持"`build_parser()` + `main()`"两段式，地形与查看参数一律
从 `cli.py` 取，不要自己再加一份。

## 1. 安装与冒烟测试

```bash
cd go1-tasks
uv sync --locked
uv run python sim2sim/go2_sim.py --headless --steps 100
uv run python sim2sim/go1_sim.py --headless --steps 100
uv run python sim2sim/go1_mob_sim.py --headless --steps 100
```

不提供检查点时机器人保持默认站姿，只验证模型、地形、执行器和观测接口能搭起来。
无窗口模式可加 `--command 0.3 0 0`，程序会统计后半段仿真的平均 `vx/vy/wz` 和
绝对跟踪误差，用来快速比较检查点；极短冒烟训练得到的检查点不能当作会走路的策略。

### 1.1 命令行约定

三个脚本的公共参数完全一致，同一行命令换个脚本名就能跑：

| 参数 | 含义 |
| --- | --- |
| `--terrain {flat,rough,stairs,rl_sar,go2_rough}` | 生成地形或训练地形导出；默认 Go1 `rough`，MoB/Go2 `flat` |
| `--terrain-seed N` | 生成地形的随机布局；`go2_rough` 忽略它 |
| `--terrain-type NAME` | `go2_rough` 的列名，或 `all` 表示整张 4×4 图 |
| `--terrain-column NAME` | `--terrain-type all` 时的出生格，缺省 `flat` |
| `--terrain-level 0-9` | 课程级数（难度取该级中点） |
| `--headless` / `--steps N` | 不开窗口跑 N 个物理步（每 4 步出一次策略动作） |
| `--command VX VY WZ` | 初始速度指令，上限按机器人：Go1 `1.0/1.0/0.5`，MoB 与 Go2 `1.0/0.6/1.0` |
| `--camera {free,track,chase}` | 相机模式；默认 Go2 `track`，Go1/MoB `free` |
| `--follow-distance M` | `track`/`chase` 的跟随距离，默认 2.2 m |
| `--shadows` / `--no-shadows` | 阴影开关，默认开 |
| `--joystick PATH` | 手柄设备，`''` 禁用 |

无窗口输出也统一成一行（有 `base_velocity()` 的模拟器会带上速度统计）：

```text
Completed 400 steps; height=0.934 m; mean vx/vy/wz=+0.374/-0.018/-0.001; abs error=0.126/0.018/0.001
```

只有检查点开关按策略格式区分：`go1_sim.py` 是 `--policy`、`--cts` 或
`--teacher`+`--adapter`，`go1_mob_sim.py` 是 `--policy`（旧写法 `--checkpoint` 仍可
用），`go2_sim.py` 是 `--policy`/`--student`/`--him`/`--cts`。每个脚本只接受其中
一个。哪个检查点该走哪个开关、对应的观测维数与训练命令见
[`docs/tasks/README.md`](../docs/tasks/README.md) 的任务表。

## 2. 地形

`--terrain` 有 5 个取值，三个脚本都支持。`flat`/`rough`/`stairs`/`rl_sar` 由
`quadruped_sim/terrain.py` 现场生成（Go1 默认 `rough`，MoB/Go2 默认 `flat`）；
`go2_rough` 是训练地形的精确导出，能装到任意一个机器人上。
当前导出与项目内 mjlab 1.6.0 的课程难度及颜色计算一致；旧版训练检查点
若需复现原地图，应保留当时导出的资产。

| 取值 | 内容 |
| --- | --- |
| `flat` | 无限平面 |
| `rough` | 8 m 见方高度场，0/2/…/10 cm 的 4×4 方块起伏，中心 1.4 m 见方是平坦起步区 |
| `stairs` | 5 cm 台阶向外递增到 20 cm，同样留平坦起步区 |
| `rl_sar` | rl_sar 障碍场：平面、几何障碍、坡道、楼梯、随机方块场、两张图像高度场 |

`--terrain-seed N` 改变生成地形的随机布局，对 `go2_rough` 无效。这些地形借鉴
训练配置，不是训练课程地图的复现。

### 2.1 单列：`--terrain-type <列名>`

Go2 rough 任务的 14 个地形列各导出一份 MJCF
（`terrain_assets/go2_rough/<列名>.xml`），每份 10 级难度，沿 +X 方向逐级变难：

```bash
uv run python sim2sim/go2_sim.py --him latest \
  --terrain go2_rough --terrain-type open_stairs --terrain-level 4 \
  --command 0.5 0 0
```

可用列：`flat`、`pyramid_stairs`、`pyramid_stairs_inv`、`hf_pyramid_slope`、
`hf_pyramid_slope_inv`、`random_rough`、`wave_terrain`、`discrete_obstacles`、
`box_random_grid`、`random_spread_boxes`、`open_stairs`、`random_stairs`、
`tilted_grid`、`perlin_noise`。高度上限都压在 0.14 m（约小腿长的 0.64 倍），
只用本体感知就能穿越，选型理由见 [`docs/go2_him.md`](../docs/go2_him.md)。

### 2.2 整图：`--terrain-type all`

每级难度一张 `level_<N>.xml`：4×4 个格子，14 种地形 + 2 格补平的 `flat`，每格
8 m，整张 32 m × 32 m、约 620 个几何体、0.39 MB，编译 0.1～0.2 s（原来的
14×10 整图是 112 m × 112 m，现在小 12 倍）：

```bash
uv run python sim2sim/go2_sim.py --him latest \
  --terrain go2_rough --terrain-type all \
  --terrain-level 7 --terrain-column open_stairs \
  --command 0.5 0 0
```

- `--terrain-level 0-9` 选地图；难度取该级中点 `(N + 0.5) / 10`（训练里第 N 级的
  实际难度在 `[N/10, (N+1)/10)` 内均匀分布）。
- `--terrain-column` 选出生格，不填则落在 `flat`。
- 格子之间不留缝：导出时对每格测 16 个方位、半径 `size/2 - 0.05` 处的边界高度，
  只有整圈都高于地面的地形才整体下沉——目前没有这种地形，`tilted_grid` 原来那道
  0.2 m 边框已在训练侧把 `border_height` 设成 0，楼梯和坡道则保留它们在边界处
  本来就该有的高度。
- 出生点是模型里的不可见 site：单列叫 `spawn_level_N`，整图叫 `spawn_<列名>`。

### 2.3 重新导出

训练地形配置改动后才需要这一步（只有它需要训练依赖）：

```bash
uv run python tools/export_sim2sim_terrain.py                         # 14 列 + 全部 10 级地图
uv run python tools/export_sim2sim_terrain.py --levels 3 7            # 只导这两级
uv run python tools/export_sim2sim_terrain.py --levels 3 --shape 2x7  # 自定义矩形
uv run python tools/export_sim2sim_terrain.py --types perlin_noise    # 只导这些列
uv run python tools/export_sim2sim_terrain.py --seed 7                # 换一套随机布局
```

高度场由 MuJoCo 的 `to_xml` 内联进 MJCF，只有颜色纹理落成 PNG；`manifest.json`
记录列名、级数和格子排布。`rl_sar` 的原始资源与 Apache-2.0 许可证在
`terrain_assets/` 下。想先看训练地形长什么样，`tools/render_go2_terrain.py` 把
14×10 网格和每种地形的最难一档渲染成 PNG（OSMesa，无需显示器），
`tools/visualize_go2_terrain.py` 起一个 `:8081` 的交互查看页。

## 3. 运行策略

### Go1 RMA 与平地速度策略

```bash
uv run python sim2sim/go1_sim.py \
  --teacher logs/rsl_rl/go1_rma/YOUR_RUN/model_9999.pt \
  --adapter logs/rsl_rl/go1_rma/YOUR_RUN/adapter_1000.pt \
  --terrain rough

uv run python sim2sim/go1_sim.py \
  --policy logs/rsl_rl/go1_velocity/YOUR_RUN/model_9999.pt \
  --terrain flat
```

`--policy` 不能与 `--teacher`/`--adapter` 同时使用；带 187 维高度扫描的当前
Rough 检查点会被拒绝。RMA 两阶段流程见 [`docs/rma.md`](../docs/rma.md)：主策略
输入 48 维当前观测，适应网络输入此前 50 帧观测（从旧到新）并输出 8 维隐变量，
动作是 `FR, FL, RR, RL` 顺序的 12 维关节位置偏移，50 Hz 输出、200 Hz 物理。

### Go2 Flat / AMP

Go2 与 Go1 共享算法，但用独立 MJCF、关节顺序 `FR, FL, RR, RL`、默认姿态
`[0, 0.8, -1.5]`、动作尺度 `[0.125, 0.25, 0.25]`、`Kp=25 / Kd=0.5`、初始
`base_link` 高度 0.38 m，以及 RobotLab 风格的 `23.5 Nm` / `30 rad/s` 电机限幅
（源 MJCF 的 hip/thigh/calf 上限 `23.7/23.7/45.43 Nm` 与训练限幅不同）。AMP、
Rough 与 Flat 共用一个入口，程序按网络第一层自动区分：45 维 = Flat 的 48 维去掉
机身线速度（AMP、Rough 和 Rough AMP 都是这个布局）。

```bash
uv run python sim2sim/go2_sim.py --policy logs/rsl_rl/go2_flat/YOUR_RUN/model_9999.pt --terrain flat
uv run python sim2sim/go2_sim.py --policy logs/rsl_rl/go2_amp/YOUR_RUN/model_9999.pt \
  --terrain flat --command 0 0 0.5
```

指令范围与训练一致：前后 `[-1, 1] m/s`、横向 `[-0.6, 0.6] m/s`、偏航角速度
`[-1, 1] rad/s`。

### Go2 论文式 LSTM 学生

```bash
PYTHONPATH=sim2sim uv run python sim2sim/go2_sim.py \
  --student logs/rsl_rl/go2_paper_amp_student/YOUR_RUN/student_4000.pt \
  --terrain rough --command 0.4 0 0
```

程序维护 256 维 hidden/cell，仿真复位时清零。

### Go2 rough 地形上的部署

`go2_rough` 不是随机生成的 8 m 高度场，而是训练课程的逐格导出：每列一条 10 级
条带，或每级一张 4×4 整图，出生点由导出时埋进模型的不可见 site 决定。它同样能装
到 Go1 和 MoB 上（`--terrain go2_rough` 加同样的选格参数），只是那两个策略没在上面
训练过。下面先说哪些 Go2 检查点能在上面跑，再说怎么选格子。

| 检查点 | 观测 | 能否在 `go2_rough` 上跑 |
| --- | --- | --- |
| `--policy` Go2 Flat / AMP | 48 / 45 | 能 |
| `--policy` `Go2-Velocity-Rough-AMP`（盲策略，训练用的就是这套地形） | 45 | 能 |
| `--him`（Go2-Velocity-HIM） | 6 × 45 历史 | 能 |
| `--student` 论文式 LSTM 学生 | 45 + LSTM 状态 | 能 |
| `--policy` `Go2-Velocity-Rough`（盲策略，与 AMP 同布局） | 45 | 能 |
| `--policy` 旧的 235 维 Rough 检查点 | 235 | 不能：需要 187 维地形扫描，独立 MJCF 没有 raycast 传感器 |
| `--cts` CTS 学生（`<run>.onnx` 或 `model_*.pt`） | 45 + 5 × 45 历史 | 能：checkpoint 里有 `student_encoder.*`，`.onnx` 与 `.pt` 跑出同一策略 |
| `--policy` CTS 的 `model_*.pt` | 77 | 不能：77 维头要拼 32 维隐变量，加载时会提示改用 `--cts` |

`--policy` 按第一层自动区分 45/48，所以 rough / rough-AMP 的检查点直接丢给
`--policy` 即可，不用额外开关。以 `go2_rough_amp` 的最新检查点为例（`--command 0.5 0 0`，
400 步，取后半段均值）：

```bash
uv run python sim2sim/go2_sim.py \
  --policy logs/rsl_rl/go2_rough_amp/YOUR_RUN/model_9999.pt \
  --terrain go2_rough --terrain-type all --terrain-level 6 \
  --terrain-column open_stairs --command 0.5 0 0
```

| 场景 | 实测 |
| --- | --- |
| 4×4 整图第 6 级 `open_stairs` 格 | `vx +0.374`、误差 `0.126`，高度 `0.934 m`（出生格本身抬高） |
| 4×4 整图第 0 级 `flat` 格，`--command 0.4 0 0` | `vx +0.275`、误差 `0.125`，高度 `0.346 m` |
| 单列 `discrete_obstacles` 第 5 级，`--command 0.4 0 0` | `vx +0.188`、误差 `0.212`（障碍密度高于平均，横向偏差 0.016） |

三种选格方式：

| 命令片段 | 场景 |
| --- | --- |
| `--terrain go2_rough` | 等价于 `--terrain-type flat --terrain-level 0`：平列起点 |
| `--terrain go2_rough --terrain-type perlin_noise --terrain-level 4` | 单列条带，落在该列第 5 级 |
| `--terrain go2_rough --terrain-type all --terrain-level 6 --terrain-column open_stairs` | 4×4 整图，从 `open_stairs` 格出生；不填 `--terrain-column` 就落在 `flat` 格 |

Go1 和 MoB 用同一套参数也能装这张图（分布外，仅作压力测试）：
`go1_sim.py --teacher ... --adapter ... --terrain go2_rough --terrain-type all
--terrain-column open_stairs --terrain-level 6 --command 0.5 0 0 --headless
--steps 400` 实测 `vx +0.402`；MoB 在 `flat` 列第 0 级 `vx +0.515`（指令 0.5）。

难度取该级中点 `(level + 0.5) / 10`，`--terrain-seed` 对它无效（布局在导出时已
固定）。可选列名和级数来自 `terrain_assets/go2_rough/manifest.json`，缺资源时按
§2.3 重新导出。这个地形把 `stat.extent` 撑到几十米，所以渲染上有三处修正，见 §4.2。

### Go2 HIM

```bash
uv run python sim2sim/go2_sim.py --him latest \
  --terrain go2_rough --terrain-type all --terrain-level 6 \
  --command 0.5 0 0
```

`--him latest` 取 `logs/rsl_rl/` 下最新的 HIM `model_*.pt`，也可以直接给路径。
程序维护最近 6 帧、每帧 45 维的历史（最新帧在前），复位时清空；观测顺序为命令、
角速度、投影重力、关节角、关节速度、上一动作，缩放为 `[2.0, 2.0, 0.25]`、
`0.25`、`0.05`，与训练环境和 `him_actor_observation()` 一致。旧版未缩放的 HIM
检查点会被拒绝。

### CTS（`--cts <run>.onnx` 或 `--cts model_*.pt`）

CTS 的 actor 是 77 维输入：45 维本体观测拼接 32 维隐变量，隐变量由两个编码器之一
给出——特权编码器吃 259 维 critic 观测，学生编码器吃 `cts_history`（actor 组的 5 帧
历史，共 225 维）。部署只走学生分支，而这一分支在两种文件里都完整存在：

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `<run>.onnx` | 学生编码器 + L2 归一化 + 策略头，双输入 `proprio (1,45)`、`history (1,225)` → `actions (1,12)` | 训练器顺手导出的部署图，跨框架、bit-exact |
| `model_*.pt` | `actor_state_dict` 里的 `student_encoder.*`、`mlp.*`、`distribution.std_param` | 任意中间 checkpoint 都能跑，不需要 `onnx` |

两者实测等价：同一批 200 帧随机观测上动作最大差 `1.9e-6`（float32 噪声），在
`go2_rough` 第 6 级 `open_stairs` 上 400 步统计完全一致（`vx +0.521`、误差 `0.021`、
高度 `0.932 m`）。所以 `--cts` 收两种后缀：

```bash
# Go2：用 run 目录里的 ONNX 导出
uv run python sim2sim/go2_sim.py \
  --cts logs/rsl_rl/go2_cts/YOUR_RUN/YOUR_RUN.onnx \
  --terrain go2_rough --terrain-type all --terrain-level 6 \
  --terrain-column open_stairs --command 0.5 0 0

# 同一策略，直接给 checkpoint（省掉 onnx 依赖）
uv run python sim2sim/go2_sim.py \
  --cts logs/rsl_rl/go2_cts/YOUR_RUN/model_9999.pt --terrain go2_rough \
  --terrain-type all --terrain-level 6 --terrain-column open_stairs

# Go1 CTS 同理
uv run python sim2sim/go1_sim.py \
  --cts logs/rsl_rl/go1_cts/YOUR_RUN/YOUR_RUN.onnx --terrain rough --command 0.5 0 0
```

两处实现细节决定了必须按下面的方式喂历史，写自己的部署代码时也一样：

- **历史是 term-major**：mjlab 给 actor 组的每个 term 各开一个环形缓冲，展平后是
  「角速度 5 帧 → 重力 5 帧 → 关节角 5 帧 → 关节速度 5 帧 → 上一动作 5 帧 →
  指令 5 帧」，不是「5 帧 × 45 维」的 frame-major。顺序反了策略仍能跑，但会明显
  横漂（实测 `vy` 从 `+0.001` 变成 `+0.689`）。
- **每段内部从旧到新**，复位后第一次推入会回填满 5 帧（mjlab `CircularBuffer`
  的 backfill 行为）。

注意 `.onnx` 只在保存 checkpoint 时导出，所以它对应 run 目录里的**最后一个**
`model_*.pt`；拿中间的 checkpoint 跑会有行为差异（Go1 的 `model_6000.pt` 在
`--terrain rough` 上是 `vx +0.205`，而 6300 和它的导出都是 `+0.456`）。

用 `model_*.pt` 重建时，checkpoint 里没有记录激活函数，一律按本仓库所有 CTS 任务的
ELU 处理；若 checkpoint 带 `obs_normalizer.*`，按 rsl-rl 的
`(x - mean) / (std + 0.01)` 归一化本体观测，没有就跳过。ONNX 路径仍用
`onnx.reference.ReferenceEvaluator`，所以不需要 onnxruntime。

用 `go2_cts/2026-10-31_10-43-13` 实测（400 步、取后半段均值）：

| 地形 | 指令 | 实测 |
| --- | --- | --- |
| `go2_rough` 整图第 0 级 `flat` 格 | `0.5 0 0` | `vx +0.511`、误差 `0.011`，高度 `0.358 m` |
| `go2_rough` 整图第 6 级 `open_stairs` 格 | `0.5 0 0` | `vx +0.521`、误差 `0.021`，高度 `0.932 m` |
| `go2_rough` 整图第 4 级 `wave_terrain` 格 | `0.5 0 0` | `vx +0.473`、误差 `0.027` |
| `go2_rough` 单列 `perlin_noise` 第 7 级 | `0.5 0 0` | `vx +0.471`、误差 `0.029` |
| 同上整图第 2 级 `flat` 格 | `-0.4 0 0` | `vx -0.359`、误差 `0.041` |
| 同上 | `0 0.4 0` | `vy +0.342`、误差 `0.058` |
| 同上 | `0 0 0.5` | `wz +0.264`、误差 `0.236` |

偏航是这套独立场景的弱项：同一个格子换成 `go2_rough_amp` 的 `--policy` 检查点，
`wz` 只有 `+0.013`，说明差距主要在 sim2sim 的 Go2 执行器/摩擦标定，而不是 CTS
训练本身。Go1 CTS（`go1_cts/2026-10-29_14-14-46`）在 `--terrain rough` 上
`vx +0.456`、误差 `0.044`，但偏航漂移 `+0.359 rad/s`，同样建议先在平地上核对
指令响应再用它评估地形能力。

### Go1 MoB

```bash
uv run python sim2sim/go1_mob_sim.py \
  --policy logs/rsl_rl/go1_mob/YOUR_RUN/model_6300.pt --terrain flat
```

1710 维输入 = 最近 30 个控制周期的角速度、重力方向、12 关节位置、12 关节速度、
上一步 12 维动作、3 维速度指令、8 维行为指令和 4 维步态相位（每项按旧→新排列），
50 Hz 输出、200 Hz 物理，部署观测不加噪声。MoB 在平地训练，`rough`/`stairs`/
`rl_sar` 属于分布外测试。该入口用自己的 `Kp=20`、`Kd=0.5` 和 `0.125/0.25/0.25`
动作尺度，不改变 RMA 运行程序。旧的 `--checkpoint` 仍可作为 `--policy` 的别名
使用。

## 4. 查看器

### 相机

三个脚本都接受 `--camera track`（跟随 `base_link`）、`chase`（额外跟随朝向，始终
待在身后）、`free`（鼠标放在哪就停在哪）和 `--follow-distance`（默认 2.2 m）；默认
Go2 用 `track`，Go1/MoB 用 `free`。MuJoCo 查看器本身的鼠标相机控制始终可用。

### 阴影与诊断标记

导出地形把 `stat.extent` 撑到 40 m（单列条带 150 m），而 MuJoCo 有若干默认值是按
场景尺度算的，所以 `go2_rough` 会额外收拾三处——**只影响渲染，物理不变**：

| 项目 | 问题 | 处理 |
| --- | --- | --- |
| 诊断标记粗细（接触点、力箭头、body 坐标系） | 由 `stat.meansize` 决定，导出地形把它从 mjlab 的 0.03 m 抬到数米 | 固定 `stat.meansize = 0.03` |
| 力箭头长度、鼠标扰动强度 | 由 `stat.meanmass` 决定，地形和光源的无质量 body 会把它拉低 | 只对机器人 body 取平均，排除 `terrain`、`shadow_light` |
| 阴影锯齿 | 定向光固定在地图中心，阴影投影铺满整张地图，每 texel 好几厘米 | 光源挂到跟随机器人的 mocap body，投影裁到机器人周围 4 m（`vis.map.shadowclip`） |

相机裁剪面按米设置：`go2_rough` 用 `0.2–80 m`，其他地形 `0.05–400 m`，跟随相机
贴近时不会把机器人切一半。阴影默认开启，`--no-shadows` 关闭（等价于把
`vis.quality.shadowsize` 置 0），`--shadows` 显式打开。

`Contact force` 等显示开关和力箭头的默认长度倍率保持 MuJoCo 原值：箭头长度 =
接触力 × `vis.map.force`（默认 0.005 m/N），Go2 站立时每只脚 30～55 N（箭头
0.15～0.28 m），单脚推进或落地可到 100 N 以上。开关都在查看器右侧 `Rendering`
面板里，默认关闭。

### 键位与手柄

`go1_sim.py` / `go2_sim.py`：`1` 启用策略，`0` 被动站立，`R` 重置，`W/S` 前后，
`A/D` 横移，`Q/E` 转向，方向键上/下控制前后，空格清零指令。Linux 手柄
`/dev/input/js0`：`A` 启用、`B` 被动模式、`X` 重置、左摇杆平面速度、右摇杆水平轴
转向；`--joystick ''` 禁用手柄。

`go1_mob_sim.py` 额外有步态与参数调节：`1/2/3/4` 选并启用 Pronk/Trot/Bound/Pace，
`Z/X` 机身高度，`C/V` 俯仰，`F/G` 步频，`B/N` 站姿宽度，`H/J` 摆腿高度；手柄上
`Y` 或方向键右切下一步态、方向键左切上一步态、方向键上/下选参数、`LB/RB` 调档，
终端会显示当前选中的参数和值。

## 5. 测试

```bash
uv run python -m pytest -q sim2sim/tests
uv run python -m pytest -q tests/test_go2_rough_terrain_mix.py
```

`sim2sim/conftest.py` 把 `sim2sim/` 加进 `sys.path`，所以在仓库根目录跑
`uv run python -m pytest sim2sim/tests -q` 也可以。覆盖内容：Go2 模型与执行器契约、
四种生成地形、`go2_rough` 的单列与 4×4 地图（同一级、接缝、几何体数量、出生点）、
相机裁剪面、诊断标记尺度、阴影跟随、追尾相机、共用键盘指令与查看器循环
（用假查看器驱动，不需要显示器）、三个脚本共用同一套地形/运行/查看参数且
`go2_rough` 在 Go1/MoB/Go2 上出生点一致，CTS 的 ONNX 签名校验、`.pt` 重建与导出
等价、term-major 历史布局与复位行为，以及 Flat/AMP/LSTM/HIM 的观测与推理契约
（含旧检查点与非有限值拒绝）和 sim2sim 常量与训练配置的一致性。
`PYTHONPATH=sim2sim uv run python -m unittest discover -s sim2sim/tests` 仍可用。
受限制环境中的 Warp 缓存和系统 pytest 插件设置见
[测试说明](../tests/README.md)。

这里的引擎与场景和 `mjlab`/MuJoCo-Warp 训练环境不同，上真机前应先在 sim2sim 里
验证运动表现。
