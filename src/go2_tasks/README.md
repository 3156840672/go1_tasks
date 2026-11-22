# Go2 任务包

[__init__.py](__init__.py) 注册 `Go2-Velocity-*` 任务；
[cli.py](cli.py) 提供训练、播放和论文式 AMP 学生训练入口。Go2 复用部分 Go1
算法实现，但机器人模型、关节常量、地形和部署约定在本包中单独定义。

| 路径 | 职责 |
| --- | --- |
| [robots/go2.py](robots/go2.py)、[assets/go2/go2.xml](assets/go2/go2.xml) | Go2 MJCF、关节顺序、碰撞和电机配置 |
| [env_cfgs.py](env_cfgs.py) | Flat、盲 Rough、MoB、AMP、CTS、HIM 环境配置 |
| [terrains.py](terrains.py) | 盲 Rough 地形种类、比例和难度范围 |
| [deployment.py](deployment.py) | 部署观测与动作的固定接口 |
| [him.py](him.py)、[him_commands.py](him_commands.py) | HIM 历史编码器与命令 |
| [paper_amp.py](paper_amp.py)、[paper_rewards.py](paper_rewards.py)、[paper_curriculum.py](paper_curriculum.py) | 论文式 AMP 教师、奖励与地形课程 |
| [train_paper_amp.py](train_paper_amp.py)、[train_terrain_amp.py](train_terrain_amp.py) | 学生蒸馏和地形 AMP 训练流程 |

注册的 `Go2-Velocity-Rough-AMP-Teacher` 使用窄指令起始、逐步扩速的 baseline
配置；`unitree_go2_paper_amp_env_cfg(baseline=False)` 是另一个未注册为任务 ID 的
实验配置。不要把两者的奖励和命令范围混用。

任务用法见 [任务索引](../../docs/tasks/README.md)，机器人与数据约定见
[Go2 说明](../../docs/go2.md)。地形变更后用
[导出脚本](../../tools/export_sim2sim_terrain.py) 更新 sim2sim 资产；部署验证见
[sim2sim/README.md](../../sim2sim/README.md)。
