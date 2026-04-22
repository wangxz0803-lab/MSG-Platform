# ADR-002: 配置管理策略

## 状态
已采纳

## 背景
原工程配置分散在三套系统：
1. `config.py`（根目录，已标记废弃）：硬编码 Python 常量（PROTOCOL_SPEC、模型维度、训练超参）
2. `configs/*.yaml` + Hydra：ML 超参数，支持 multirun/sweep
3. `platform/*/settings.py`：pydantic-settings，管 DB/Redis/端口

`config.py` 被 `model.py` 和 `tools.py` 通过 `from config import *` 导入，
是当前最大的耦合点。

## 选项

### 选项 A：纯 pydantic-settings
- 优点：统一一套系统
- 缺点：丢失 Hydra multirun/sweep，ML 实验工作流受损

### 选项 B：Hydra ML + pydantic-settings 基础设施
- 优点：各管其长，保留 Hydra sweep 能力
- 缺点：两套系统需理解

### 选项 C：Dynaconf
- 优点：多源支持
- 缺点：新依赖，无 Hydra 等效

## 决策
选择 B。Hydra 继续管 ML 配置组（model/train/eval/infer），pydantic-settings 管基础设施
（DB URL、Redis URL、端口、路径）。`config.py` 废弃，常量分流到 Hydra YAML 或
`core/protocol_spec.py`。

## 后果
- 正面：保留 ML 实验 sweep 能力，基础设施配置有验证
- 负面：需理解两套系统边界
- 风险：需明确划分哪些参数归 Hydra、哪些归 pydantic-settings

## 日期
2026-04-22
