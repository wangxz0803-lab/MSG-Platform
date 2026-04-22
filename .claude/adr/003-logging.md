# ADR-003: 日志框架

## 状态
已采纳

## 背景
原工程使用 loguru，通过 `utils/logger.py` 提供 `get_logger(name)` 封装。
支持双输出（控制台彩色 + 文件轮转），但缺乏 JSON 结构化输出和关联 ID 绑定。
部分代码（如 `platform/backend/main.py`）直接 `from loguru import logger`，
未经过统一封装。

## 选项

### 选项 A：保留 loguru
- 优点：零迁移成本，团队已熟悉
- 缺点：缺乏原生 JSON 结构化和关联 ID

### 选项 B：structlog
- 优点：JSON 结构化输出、关联 ID 绑定、开发彩色/生产 JSON 双模式
- 缺点：迁移成本（find-replace import）

### 选项 C：标准 logging + json-formatter
- 优点：最小依赖
- 缺点：配置繁琐，功能较弱

## 决策
选择 B（structlog）。迁移面有限（`utils/logger.py` 封装层 + 少量直接导入），
`get_logger(name)` API 保持不变。

## 后果
- 正面：生产级日志聚合、请求关联追踪
- 负面：需全局 find-replace loguru 导入
- 风险：loguru 的 lazy formatting 语法需适配 structlog

## 日期
2026-04-22
