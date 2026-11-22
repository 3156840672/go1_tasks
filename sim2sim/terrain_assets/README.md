# sim2sim terrain assets

`go2_rough/` 保存 Go2 训练课程的导出地图：各地形列的 10 级条带、每级一张
整图、颜色纹理 PNG 和 `manifest.json`。XML 中的纹理路径相对于 XML 文件，
整个目录可以随仓库移动。当前资产使用本地 mjlab 1.6.0 的地形生成器、
`--seed 42` 和任务配置重新导出。课程第 0～9 级分别覆盖配置难度范围的两个
端点；`manifest.json` 记录生成器版本和种子。升级前训练的检查点若需要严格
复现旧地图，应使用其当时导出的地图。
训练地形配置修改后，从仓库根目录重新生成：

```bash
uv run python tools/export_sim2sim_terrain.py
```

生成逻辑和运行参数见 [sim2sim 说明](../README.md#2-地形)。不要手工修改导出的
XML；应在 `src/go2_tasks/terrains.py` 或导出脚本中修改来源并重新导出。

## rl_sar

These files are copied from the `dog1_description/mjcf` obstacle course in
https://github.com/3156840672/rl_sar and are used by `--terrain rl_sar`.

The scene XML is parsed only for its terrain assets and world-body objects;
its Dog1 robot include is intentionally ignored. See `RL_SAR_LICENSE` for the
upstream Apache-2.0 license.
