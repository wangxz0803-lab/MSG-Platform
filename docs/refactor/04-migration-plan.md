# 04 - 迁移映射表

## 迁移状态图例

- P0 = 阻塞其他模块，必须先完成
- P1 = 重要，按依赖顺序推进
- P2 = 可延后
- 新建 / 重度改造 / 轻度改造 / 保留 / 丢弃

## P0 模块（阻塞路径）

| # | 模块 | 方式 | 源文件 | 目标位置 | 风险 | 状态 |
|---|------|------|--------|----------|------|------|
| 1 | core/ | 新建 | - | src/msg_embedding/core/ | 低 | **完成** |
| 2 | models/ | 重度改造 | model.py (245行) | src/msg_embedding/models/ | 中 | 待开始 |
| 3 | features/extractor | 重度改造 | tools.py (~700行) | src/msg_embedding/features/ | 中 | 待开始 |
| 4 | data/ | 轻度改造 | src/msg_embedding/data/ | src/msg_embedding/data/ | 低 | 待开始 |
| 5 | platform/backend | 重度改造 | platform/backend/ | platform/backend/ | 中 | 待开始 |

## P1 模块（重要）

| # | 模块 | 方式 | 源文件 | 目标位置 | 风险 | 状态 |
|---|------|------|--------|----------|------|------|
| 6 | channel_est/ | 保留 | src/.../channel_est/ | 同 | 低 | 待开始 |
| 7 | channel_models/ | 保留 | src/.../channel_models/ | 同 | 低 | 待开始 |
| 8 | ref_signals/ | 保留 | src/.../ref_signals/ | 同 | 低 | 待开始 |
| 9 | phy_sim/ | 保留 | src/.../phy_sim/ | 同 | 低 | 待开始 |
| 10 | topology/ | 保留 | src/.../topology/ | 同 | 低 | 待开始 |
| 11 | training/ | 轻度改造 | src/.../training/ | 同 | 中 | 待开始 |
| 12 | inference/ | 轻度改造 | src/.../inference/ | 同 | 低 | 待开始 |
| 13 | eval/ | 保留 | src/.../eval/ | 同 | 低 | 待开始 |
| 14 | features/pmi | 保留核心+轻改 | CsiChanProcFunc.py 等 | features/pmi/ | 高 | 待开始 |
| 15 | features/srs | 保留核心+轻改 | SrsChanProcFunc.py 等 | features/srs/ | 中 | 待开始 |
| 16 | features/ssb | 保留核心+轻改 | SsbChanProcFunc.py 等 | features/ssb/ | 中 | 待开始 |
| 17 | platform/worker | 轻度改造 | platform/worker/ | 同 | 低 | 待开始 |
| 18 | scripts/ | 轻度改造 | scripts/run_*.py | 同 | 低 | 待开始 |
| 19 | utils/ | 轻度改造 | src/.../utils/ | 同 | 低 | 待开始 |

## P2 模块（可延后）

| # | 模块 | 方式 | 源文件 | 目标位置 | 风险 | 状态 |
|---|------|------|--------|----------|------|------|
| 20 | report/ | 保留 | src/.../report/ | 同 | 低 | 待开始 |
| 21 | viz/ | 保留 | src/.../viz/ | 同 | 低 | 待开始 |
| 22 | platform/frontend | 原样保留 | platform/frontend/ | 同 | 低 | 待开始 |
| 23 | Docker + CI | 新建 | - | Dockerfile, .github/ | 低 | 待开始 |

## 丢弃清单

| 文件 | 原因 |
|------|------|
| config.py | 已废弃，常量迁至 Hydra YAML + protocol_spec.py |
| gol.py + GlobalData.py | 全局状态模式，替换为类内状态 |
| quadriga_single_legacy.py | 已被 quadriga_multi 完全替代 |
| pretrain.py (根目录) | 已被 training/pretrain.py 替代 |
| inference.py (根目录) | 已被 inference/ 模块替代 |
| online_train.py | 已被 training/finetune.py 替代 |
| bridge_channel_to_pretrain.py | 已被 data/bridge.py 替代 |
| transfer_channel_noise_to_mutli_npy.py | 一次性转换脚本 |
| _check.py | 调试残留 |
| past.py | 遗留算法，无调用者 |
| PmiTest.py, GetTotalPmi.py 等 | 测试/工具脚本，无生产调用 |

## 迁移决策记录

（每完成一个模块在此追加决策和踩坑记录）
