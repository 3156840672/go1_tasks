# mjlab 地形

通用地形实现在 `primitive_terrains.py` 与 `terrain_generator.py`，常用参数预设在
`config.py`，环境中的出生点和课程等级由 `terrain_entity.py` 管理。

- `TerrainGeneratorCfg(curriculum=True)` 为每种子地形生成一列；`num_rows` 是
  难度级数，`proportion` 决定机器人出生到各列的权重。mjlab 1.6.0 在每行使用
  确定的难度，首尾两行对应 `difficulty_range` 的两个端点。
- `curriculum=False` 按权重随机选择每个格子的地形，并从 `difficulty_range`
  抽取难度；它不代表固定课程等级。
- 新地形先实现 `SubTerrainCfg.function()`，再在 `config.py` 中提供预设函数；
  高度场和几何体应返回正确的出生原点。

本项目的 Go2 盲地形组合定义在
[go2_tasks/terrains.py](../../../../../src/go2_tasks/terrains.py)，独立 MuJoCo
地图由 [导出脚本](../../../../../tools/export_sim2sim_terrain.py) 生成。修改通用
地形行为后，运行仓库根目录的地形与 sim2sim 测试。
