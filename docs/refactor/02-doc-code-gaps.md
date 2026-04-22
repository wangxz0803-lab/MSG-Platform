# 02 - 文档与代码差异

本文档记录 `D:\MSG\guide` 文档描述与实际代码实现之间的差异。
原则：**以代码为准**，但记录差异以免误导。

## 差异清单

### GAP-01: features/ 和 models/ 子模块状态

| | 文档描述 | 实际代码 |
|-|----------|----------|
| **文档** | 06_developer_reference.md 列出 14 个子模块，features/ 标记为 "Shell"，models/ 标记为 "Shell" | |
| **代码** | `src/msg_embedding/features/` 和 `src/msg_embedding/models/` 是空目录或仅有 `__init__.py` | |
| **影响** | 实际的模型定义在根目录 `model.py`（ChannelMAE），特征提取在根目录 `tools.py`（FeatureExtractor）。所有依赖方通过根目录导入。 |
| **评估** | 文档对此标注了 "Shell" 状态，基本准确。但未说明这导致了 `from config import *` 的耦合链。 |

### GAP-02: config.py 废弃状态

| | 文档描述 | 实际代码 |
|-|----------|----------|
| **文档** | 06_developer_reference.md 提到配置通过 Hydra 管理 | |
| **代码** | `config.py` 虽标记 `DeprecationWarning`，但仍被 `model.py` 第 7 行通过 `from config import *` 活跃导入。如果删除 config.py，model.py 会走 fallback 默认值路径（第 9-24 行）。 |
| **影响** | Hydra 配置和 config.py 硬编码值可能不同步。model.py 构造函数同时接受 `cfg` dict 参数和全局常量，存在二义性。 |
| **评估** | 文档未充分说明这一过渡期的双源配置问题。 |

### GAP-03: 日志系统迁移状态

| | 文档描述 | 实际代码 |
|-|----------|----------|
| **文档** | 06_developer_reference.md 提到使用 `msg_embedding.utils.logger` 统一日志 | |
| **代码** | `utils/logger.py` 模块文档注释自述 "mid-migration from print() calls"。`platform/backend/main.py` 直接 `from loguru import logger`，未经过统一封装。 |
| **影响** | 部分模块绕过统一日志，日志格式和输出不一致。 |
| **评估** | 文档描述了目标状态，实际仍在迁移中。 |

### GAP-04: 测试数量

| | 文档描述 | 实际代码 |
|-|----------|----------|
| **文档** | 01_project_overview.md 声称 870+ 测试 | |
| **代码** | Python 3.9 下 584 通过，Python 3.12 下 460 通过。实际可运行测试数取决于环境（MATLAB/Sionna/GPU 标记的测试在无对应环境时跳过）。 |
| **影响** | 870+ 可能是所有标记的测试总和（包含跳过的）。不影响迁移，但验收时需明确基线。 |
| **评估** | 差异不大，文档大体准确。 |

### GAP-05: 数据源完成度

| | 文档描述 | 实际代码 |
|-|----------|----------|
| **文档** | 02_data_pipeline.md 列出 5 种数据源，标记 field 为 "v3.1 placeholder" | |
| **代码** | `field.py` 抛出 `NotImplementedError`。另外还存在 `quadriga_single_legacy.py`（第 6 个适配器），文档未提及。 |
| **影响** | 迁移时需决定是否保留 `quadriga_single_legacy.py`。 |
| **评估** | 文档略有遗漏，以代码为准。 |

### GAP-06: Worker 通信机制

| | 文档描述 | 实际代码 |
|-|----------|----------|
| **文档** | 05_platform_guide.md 描述 Dramatiq + Redis 任务队列 | |
| **代码** | 实际使用文件系统队列（file-drop queue）作为主通信机制：Worker 轮询 `queue/` 目录下的 JSON 文件，通过 `progress/` 目录报告进度，通过 `cancel/` 目录接收取消信号。Dramatiq/Redis 作为补充通道。 |
| **影响** | 文件队列是实际的工作引擎，不仅是 fallback。迁移时需保留此机制。 |
| **评估** | 文档对此描述不充分。 |

### GAP-07: 前端 SPA 路由

| | 文档描述 | 实际代码 |
|-|----------|----------|
| **文档** | 05_platform_guide.md 描述前后端分离部署 | |
| **代码** | `backend/main.py` 同时挂载前端静态文件和 SPA catch-all 路由（第 77-85 行）。生产可通过 Nginx 独立部署（`deploy/nginx.conf`），开发时后端兼任静态服务。 |
| **影响** | 迁移时需决定是否保留后端内嵌的 SPA 路由。 |
| **评估** | 文档描述了理想架构，代码包含了开发便利的混合模式。两者均有效。 |

### GAP-08: 依赖版本

| | 文档描述 | 实际代码 |
|-|----------|----------|
| **文档** | 各文档提到的版本：torch 2.9.1, numpy 2.4.3, fastapi 0.115.x | |
| **代码** | `requirements.txt` 列出 torch 2.9.1, numpy 2.4.3 等。但 FastAPI、SQLAlchemy、Pydantic、Hydra、Loguru **不在** requirements.txt 中，而在 `platform/deploy/requirements/api.txt` 和 `worker.txt` 中独立管理。 |
| **影响** | 依赖分散在多个文件，需统一到 pyproject.toml。 |
| **评估** | 文档未说明依赖管理的分散现状。 |

## 总结

| 类型 | 数量 | 严重程度 |
|------|------|----------|
| 功能性差异（影响迁移决策） | 3 | GAP-01, GAP-02, GAP-06 |
| 信息遗漏（需补充理解） | 3 | GAP-05, GAP-07, GAP-08 |
| 精度偏差（不影响迁移） | 2 | GAP-03, GAP-04 |

文档整体质量较高，大部分内容与代码一致。主要差异集中在"过渡期状态"的描述不够明确。
