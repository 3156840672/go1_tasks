# 工具脚本

以下脚本从仓库根目录运行，先执行 `uv sync --locked`。参数以各脚本的
`--help` 为准；多数脚本需要已训练检查点、参考动作或 OCS2 录制，不会由安装步骤生成。

| 工作流 | 脚本 |
| --- | --- |
| Go2 地形导出与查看 | [export_sim2sim_terrain.py](export_sim2sim_terrain.py)、[render_go2_terrain.py](render_go2_terrain.py)、[visualize_go2_terrain.py](visualize_go2_terrain.py) |
| Go2 检查点评估 | [eval_go2_task.py](eval_go2_task.py)、[evaluate_go2_paper_amp.py](evaluate_go2_paper_amp.py)、[evaluate_go2_paper_terrain.py](evaluate_go2_paper_terrain.py) |
| OCS2 参考动作采集与转换 | [record_go2_ocs2_trot.py](record_go2_ocs2_trot.py)、[extract_go2_ocs2_bag.py](extract_go2_ocs2_bag.py)、[convert_go2_ocs2_amp.py](convert_go2_ocs2_amp.py) |
| AMP 数据准备与校验 | [build_go2_paper_amp_dataset.py](build_go2_paper_amp_dataset.py)、[validate_go2_amp_dataset.py](validate_go2_amp_dataset.py) |
| 论文式 AMP 预览与持续验证 | [record_go2_paper_amp_preview.py](record_go2_paper_amp_preview.py)、[train_validate_go2_paper_amp.py](train_validate_go2_paper_amp.py)、[continue_validate_go2_paper_amp.py](continue_validate_go2_paper_amp.py) |

地形配置在 [go2_tasks/terrains.py](../src/go2_tasks/terrains.py)，导出的独立资产在
[sim2sim/terrain_assets](../sim2sim/terrain_assets/README.md)。训练和论文式 AMP 的
完整流程分别见 [任务索引](../docs/tasks/README.md) 与
[go2_paper_amp.md](../docs/go2_paper_amp.md)。
