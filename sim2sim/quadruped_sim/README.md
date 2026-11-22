# `quadruped_sim` 包

这个包是 [sim2sim](../README.md) 的实现层；`sim2sim/go1_sim.py`、
`go1_mob_sim.py`、`go2_sim.py` 只负责解析参数并启动循环。运行时只依赖独立
MuJoCo 模型和策略检查点，不导入训练环境。

| 模块 | 约定 |
| --- | --- |
| [cli.py](cli.py) | 三个入口共享地形、运行和查看参数，以及无窗口统计输出 |
| [terrain.py](terrain.py) | 生成简单地形或装载已导出的课程地图 |
| [scene.py](scene.py) | 地形出生点、光源跟随和 MuJoCo 可视化统计尺度 |
| [viewer.py](viewer.py)、[joystick.py](joystick.py) | 交互查看器、键盘和手柄控制 |
| [robots/](robots) | Go1、Go1 MoB、Go2 的模型装配与物理控制 |
| [policies/](policies) | 与训练输出匹配的 actor、CTS、HIM、MoB、RMA 加载器 |

新增策略时，把网络和检查点解析放进 `policies/`，把机器人状态与控制放进
`robots/`。共用地形和相机参数应由 `cli.py`、`terrain.py`、`viewer.py` 提供，
避免三个入口出现不同语义。契约测试位于 [sim2sim/tests](../tests)。
