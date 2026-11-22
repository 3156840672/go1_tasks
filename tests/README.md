# 测试

从仓库根目录运行全部项目测试和独立 MuJoCo 部署测试：

```bash
uv run python -m pytest -q tests sim2sim/tests
```

`tests/` 检查任务注册、机器人资产、观测与动作接口、地形课程及训练器约定；
`sim2sim/tests/` 检查三个入口脚本、独立 MJCF、检查点加载、地形资产、查看器和
策略历史。后者不需要训练好的检查点或显示器。

部分环境测试首次运行会编译 Warp 内核，需要可写缓存目录。受限制的环境可用：

```bash
WARP_CACHE_PATH=/tmp/go1_tasks_warp_cache \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  uv run python -m pytest -q tests sim2sim/tests
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 用于避开系统 ROS 安装的 pytest 插件与本项目
Python 环境冲突。运行较慢的环境测试时，缓存首次编译可能耗时较长。
