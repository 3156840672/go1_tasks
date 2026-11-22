# 任务索引

`docs/tasks/` 按任务拆分，每页固定结构：任务 ID、观测与动作、训练、评估、sim2sim、
已知限制，末尾附参考。跨任务的原理讲解、与论文的差异和踩坑记录留在 `docs/` 下的长文档里，任务页
只回答"怎么跑、看什么、哪里有坑"。

## Go1

| 任务 | 页面 | 任务 ID | actor 观测 | sim2sim |
| --- | --- | --- | --- | --- |
| 平地速度 | [go1_flat.md](go1_flat.md) | `Go1-Velocity-Flat` | 48 | `go1_sim.py --policy` |
| 崎岖地形速度 | [go1_rough.md](go1_rough.md) | `Go1-Velocity-Rough` | 235 | 不支持（需要 187 维地形扫描） |
| AMP（MoB 参考动作） | [go1_amp.md](go1_amp.md) | `Go1-Velocity-AMP` | 45 | 无入口（`go1_sim.py` 只认 48 维） |
| MoB 行为条件步态 | [go1_mob.md](go1_mob.md) | `Go1-Velocity-MoB` | 1710（30 × 57） | `go1_mob_sim.py --policy` |
| CTS 教师-学生 | [go1_cts.md](go1_cts.md) | `Go1-Velocity-CTS` | 45 + 5 × 45 历史 | `go1_sim.py --cts` |
| RMA 两阶段 | [go1_rma.md](go1_rma.md) | 未注册，复用 Rough 环境 | 48 + 18 特权 → 8 维隐变量 | `go1_sim.py --teacher/--adapter` |

## Go2

| 任务 | 页面 | 任务 ID | actor 观测 | sim2sim |
| --- | --- | --- | --- | --- |
| 平地速度 | [go2_flat.md](go2_flat.md) | `Go2-Velocity-Flat` | 48 | `go2_sim.py --policy` |
| 盲崎岖地形速度 | [go2_rough.md](go2_rough.md) | `Go2-Velocity-Rough` | 45 | `go2_sim.py --policy` |
| AMP | [go2_amp.md](go2_amp.md) | `Go2-Velocity-AMP` | 45 | `go2_sim.py --policy` |
| 崎岖地形 AMP | [go2_rough_amp.md](go2_rough_amp.md) | `Go2-Velocity-Rough-AMP` | 45 | `go2_sim.py --policy` |
| 论文式 Rough AMP + LSTM 学生 | [go2_paper_amp.md](go2_paper_amp.md) | `Go2-Velocity-Rough-AMP-Teacher` | 45 + LSTM 状态 | `go2_sim.py --student` |
| MoB 行为条件步态 | [go2_mob.md](go2_mob.md) | `Go2-Velocity-MoB` | 1710（30 × 57） | 无入口 |
| CTS 教师-学生 | [go2_cts.md](go2_cts.md) | `Go2-Velocity-CTS` | 45 + 5 × 45 历史 | `go2_sim.py --cts` |
| HIM | [go2_him.md](go2_him.md) | `Go2-Velocity-HIM` | 6 × 45 历史 | `go2_sim.py --him` |
| DreamWaQ | [go2_dreamwaq.md](go2_dreamwaq.md) | `Go2-Velocity-DreamWaQ` | 5 × 45 历史 | `go2_sim.py --dreamwaq` |

## 通用约定

| 项 | 约定 |
| --- | --- |
| 任务 ID | 包内 ID 以 `Go1-`/`Go2-` 开头；Go1 还保留 mjlab 的 `Mjlab-Velocity-*-Unitree-Go1` 旧 ID |
| 列出任务 | `uv run go1-list-tasks`、`uv run go2-list-tasks` |
| 训练 | `uv run go1-train <ID> --env.scene.num-envs 4096`，Go2 用 `go2-train` |
| 播放 | `uv run go1-play <ID> --checkpoint-file logs/rsl_rl/<run>/model_*.pt` |
| 检查点 | `logs/rsl_rl/<experiment_name>/<时间戳>/model_<迭代>.pt`，ONNX 由训练器在同目录写出 |
| sim2sim | 见 [sim2sim/README.md](../../sim2sim/README.md)；三个入口脚本的地形参数完全一致 |
| 测试 | `uv run pytest tests/ -q`，sim2sim 另有 `cd sim2sim && pytest tests/ -q` |

## 与 `docs/` 长文档的关系

| 长文档 | 内容 |
| --- | --- |
| [rma.md](../rma.md) | Go1 RMA 两阶段原理与参数 |
| [mob.md](../mob.md) | MoB 行为指令与步态奖励 |
| [cts.md](../cts.md) | CTS 教师-学生原理、评估与 sim2sim 实测 |
| [amp_tutorial.md](../amp_tutorial.md)、[amp_code_tutorial.md](../amp_code_tutorial.md) | Go1 AMP 完整流程与逐文件代码讲解 |
| [amp_mob_collection.md](../amp_mob_collection.md) | 用 MoB 策略采集 AMP 参考动作 |
| [go2.md](../go2.md) | Go2 训练与部署约定（关节顺序、常量、参考动作） |
| [go2_him.md](../go2_him.md) | HIMLoco 复现范围与差异 |
| [go2_paper_amp.md](../go2_paper_amp.md) | 论文式 Rough AMP、LSTM 蒸馏与地形排查记录 |
