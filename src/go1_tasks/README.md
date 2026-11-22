# Go1 任务包

导入 `go1_tasks` 时会注册 `Go1-Velocity-*` 任务 ID；除 AMP 外还兼容
`Mjlab-Velocity-*-Unitree-Go1` 历史 ID。实际注册表在
[__init__.py](__init__.py)，命令入口在 [cli.py](cli.py)。

| 文件 | 职责 |
| --- | --- |
| [env_cfgs.py](env_cfgs.py) | Rough、Flat、MoB、AMP、CTS 环境配置 |
| [rl_cfg.py](rl_cfg.py) | PPO、AMP、CTS 的训练器配置 |
| [rma.py](rma.py)、[rma_cfg.py](rma_cfg.py)、[train_rma.py](train_rma.py) | RMA 网络、特权信息与两阶段训练 |
| [mob.py](mob.py)、[mob_curriculum.py](mob_curriculum.py)、[mob_runner.py](mob_runner.py) | 行为命令、课程和 MoB 训练器 |
| [amp.py](amp.py)、[amp_data.py](amp_data.py)、[amp_events.py](amp_events.py) | 动作先验、参考数据与重置事件 |
| [cts.py](cts.py) | 并发教师与学生策略 |

训练任务在 `mjlab` 环境中运行；独立 MuJoCo 推理代码位于
[sim2sim](../../sim2sim/README.md)。各任务的可执行命令、观测维数和限制见
[任务索引](../../docs/tasks/README.md)。修改 actor 的观测顺序或关节顺序时，
同时检查部署代码和 [契约测试](../../tests/README.md)。
