# MSG-Embedding 开发规范

## 项目概述
5G NR 信道表征学习平台（MAE + 对比学习），用于信道数据采集、特征提取、预训练、评估推理和模型导出。

## 技术栈
- Python >= 3.10, PyTorch, NumPy, SciPy
- FastAPI + SQLAlchemy + SQLite (Alembic 迁移)
- Dramatiq + Redis (任务队列)
- React 18 + TypeScript + Ant Design (前端)
- Hydra (ML 配置) + pydantic-settings (基础设施配置)
- structlog (日志) + pytest (测试)

## Python 环境
- **必须使用** `D:\MSG\.venv312\Scripts\python.exe`（已安装 sionna 2.0.1 + drjit + mitsuba CPU 后端）
- 新工程以 editable 模式安装在此 venv 中
- 所有 python/pytest/pip 命令必须用此 venv 的解释器，不要用系统 Python

## 开发命令
```bash
# 使用 venv Python
D:/MSG/.venv312/Scripts/python.exe -m pytest tests/ -x -q

make install          # 安装开发依赖
make lint             # ruff + black check
make format           # ruff fix + black format
make typecheck        # mypy
make test             # pytest unit tests
make test-integration # pytest integration tests
make serve            # 启动 FastAPI 开发服务器
make worker           # 启动 Dramatiq worker
make docker-up        # docker compose up
make migrate          # alembic upgrade head
```

## 代码规范

### 配置
- 基础设施配置（DB/Redis/端口/路径）走 `core/config.py` (pydantic-settings)
- ML 超参数（模型/训练/评估/推理）走 `configs/*.yaml` (Hydra)
- 3GPP 协议参数走 `core/protocol_spec.py`
- 禁止硬编码配置值，禁止 `from config import *`

### 日志
- 使用 `from msg_embedding.core.logging import get_logger`
- 禁止直接 `from loguru import logger` 或 `import logging`
- logger 实例：`logger = get_logger(__name__)`
- 带上下文字段：`logger.info("event", key=value)`

### 异常
- 所有自定义异常继承 `msg_embedding.core.exceptions.MSGError`
- 禁止裸 `raise Exception("...")`
- 数据源异常用 `DataSourceError`/`MATLABError`
- 特征提取异常用 `FeatureExtractionError`
- 模型异常用 `ModelError`

### 类型标注
- 所有公共函数和方法必须有类型标注
- 内部函数建议标注但不强制
- 使用 `from __future__ import annotations` 延迟求值

### 测试
- 每个模块必须有对应测试：`tests/unit/<module>/`
- 关键路径覆盖：正常 + 异常 + 边界
- 标记慢速/GPU/外部依赖测试：`@pytest.mark.slow/gpu/matlab/sionna`
- 数值一致性用 golden vector 测试

### 导入规范
- 标准库 → 第三方 → 本项目（isort 自动排序）
- 核心包内部用相对导入
- 平台层对核心包用绝对导入 `from msg_embedding.xxx import ...`

### 数据采集适配器
- 适配器中的重试/超时/降级/特殊字段处理是踩坑经验，只做规范化改造
- 禁止"优化掉"看起来丑的错误处理代码
- 新增适配器必须实现 `DataSource` ABC 并注册到 `SOURCE_REGISTRY`

## 文件组织
- 源码：`src/msg_embedding/`
- 平台：`platform/{backend,worker,frontend}/`
- 测试：`tests/{unit,integration}/`
- 配置：`configs/`（Hydra YAML）
- 脚本：`scripts/`（CLI 入口）
- 文档：`docs/`

## 禁止事项
- 不修改 `D:\MSG`（源工程只读）
- 不留 TODO 带过
- 不写"暂时跳过"
- 不在迁移某个模块时顺手改别的模块
- 不使用 `gol.py` 全局状态模式
