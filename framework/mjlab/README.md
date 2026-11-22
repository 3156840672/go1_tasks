# 项目内的 mjlab

此目录基于上游 [mjlab 1.6.0](https://github.com/mujocolab/mjlab/releases/tag/v1.6.0)，
保留项目需要的通用框架和 Go1 资产。它是 uv 工作区成员，由根目录的 `uv sync --locked` 以
可编辑方式安装。通用环境、传感器、地形生成器、强化学习组件和查看器在
`src/mjlab/`；Go1/Go2 的项目任务配置位于仓库根目录的 `src/`，独立部署代码位于
[sim2sim](../../sim2sim/README.md)。

本项目保留了地形边框高度、接触传感器多槽位统计、关节动作顺序和查看器等
本地修复；Go2 资产与任务仍在根目录的 `src/`。MuJoCo、MuJoCo Warp 和
rsl-rl 的版本由根目录锁文件管理。修改通用地形 API 时，应同时检查
[地形说明](src/mjlab/terrains/README.md)、Go1/Go2 环境配置、地形导出工具与
相关测试。第三方来源和许可证见[项目主页](../../README.md#许可与第三方代码)。

迁移后的任务与部署回归测试：

```bash
uv run python -m pytest -q tests sim2sim/tests
```
