# 03 - 目标架构

## 设计原则

1. **分层迁移**：不重写，不简单搬代码，按层决策（新建/改造/保留/丢弃）
2. **保留隐性知识**：适配器中的重试/超时/降级逻辑是踩坑经验，只做规范化改造
3. **统一基础设施**：配置、日志、异常、DB 访问走新建的 core/ 层
4. **向后兼容**：checkpoint .pth 文件必须可加载，数值输出必须一致
5. **可验证**：每步都有明确的验收标准（lint/typecheck/test 全绿）

## 目标目录结构

```
D:\MSG平台_cc\
├── pyproject.toml                    # 统一依赖管理（替代分散的 requirements.txt）
├── alembic.ini                       # DB 迁移配置
├── Makefile                          # 一键 install/lint/test/serve/docker
├── Dockerfile                        # 多阶段构建（api + worker）
├── docker-compose.yml                # redis + api + worker + frontend
├── docker-compose.dev.yml            # 开发覆盖（热重载、暴露 Redis）
├── .github/workflows/ci.yml          # lint + unit test + typecheck
├── .pre-commit-config.yaml           # Ruff + Black + 文件检查
├── .env.example                      # 环境变量模板
├── CLAUDE.md                         # 编码规范和行为准则
│
├── configs/                          # Hydra YAML（从原工程保留）
│   ├── config.yaml                   # 根配置
│   ├── data/default.yaml             # 硬件/天线常量
│   ├── model/default.yaml            # Transformer 架构超参
│   ├── train/default.yaml            # 训练超参
│   ├── eval/default.yaml             # 评估配置
│   └── infer/default.yaml            # 推理配置
│
├── src/msg_embedding/
│   ├── __init__.py                   # 版本号
│   │
│   ├── core/                         # ★ 新建：基础设施层
│   │   ├── __init__.py
│   │   ├── config.py                 # MSGSettings(pydantic-settings) + load_hydra_config()
│   │   ├── protocol_spec.py          # PROTOCOL_SPEC 3GPP 参数表（从 config.py 迁入）
│   │   ├── logging.py                # structlog 配置（替代 loguru）
│   │   ├── exceptions.py             # 异常层级（MSGError 基类 + 子类）
│   │   └── types.py                  # 共享类型别名
│   │
│   ├── models/                       # ★ 填充：从根目录 model.py 迁入
│   │   ├── __init__.py               # 重导出 ChannelMAE, LatentAdapter
│   │   ├── channel_mae.py            # ChannelMAE（去除 from config import *）
│   │   ├── adapters.py               # LatentAdapter（~1,451 参数）
│   │   └── ema.py                    # SimpleEMA（从 tools.py 迁入）
│   │
│   ├── features/                     # ★ 填充：从根目录 tools.py + 信号处理模块迁入
│   │   ├── __init__.py               # 重导出 FeatureExtractor
│   │   ├── extractor.py              # FeatureExtractor（构造函数注入常量）
│   │   ├── normalizer.py             # ProtocolNormalizer（依赖 protocol_spec）
│   │   ├── losses.py                 # reconstruction_loss, contrastive_loss
│   │   ├── denormalize.py            # denormalize_reconstruction
│   │   ├── pipeline.py              # GetMSGEmbd_Inp 完整特征提取编排
│   │   ├── pmi/                      # CsiChanProcFunc + 权重计算（去除 gol）
│   │   │   ├── __init__.py
│   │   │   ├── codebook.py           # PMI 码本查找（VAM + Type-I）
│   │   │   └── csi_weight.py         # CSI-RS 权重（合并 3 个文件）
│   │   ├── srs/                      # SRS 信道处理
│   │   │   ├── __init__.py
│   │   │   ├── processing.py         # SrsChanProcFunc
│   │   │   └── precode.py            # Precode（Rhh 协方差）
│   │   └── ssb/                      # SSB 波束处理
│   │       ├── __init__.py
│   │       ├── processing.py         # SsbChanProcFunc
│   │       └── beam_transform.py     # BeamAntTrans（Kronecker DFT）
│   │
│   ├── data/                         # 保留 + 改进
│   │   ├── __init__.py
│   │   ├── contract.py               # ChannelSample（Pydantic v2，不动）
│   │   ├── bridge.py                 # Bridge 特征提取（SVD 稳定性代码不动）
│   │   ├── dataset.py                # ChannelDataset
│   │   ├── manifest.py               # Parquet manifest
│   │   ├── parallel.py               # 并行数据加载
│   │   ├── webdataset_shard.py        # 分片流式
│   │   └── sources/                  # 5 个适配器
│   │       ├── base.py               # DataSource ABC + 注册表
│   │       ├── internal_sim.py       # 内部模拟器（无外部依赖）
│   │       ├── sionna_rt.py          # Sionna RT（TDL 降级不动）
│   │       ├── quadriga_multi.py     # QuaDRiGa 多小区（.mat 读取）
│   │       ├── quadriga_real.py      # QuaDRiGa MATLAB 子进程（重试/超时不动）
│   │       ├── quadriga_single_legacy.py  # 遗留单小区
│   │       └── field.py              # v3.1 占位
│   │
│   ├── channel_est/                  # 保留
│   ├── channel_models/               # 保留
│   ├── ref_signals/                  # 保留
│   ├── phy_sim/                      # 保留
│   ├── topology/                     # 保留
│   ├── training/                     # 保留 + 更新导入路径
│   ├── inference/                    # 保留 + 更新导入路径
│   ├── eval/                         # 保留
│   ├── report/                       # 保留
│   ├── viz/                          # 保留
│   └── utils/                        # 保留（protocol_spec 迁至 core/）
│
├── platform/
│   ├── backend/
│   │   ├── app.py                    # FastAPI 工厂（原 main.py 重命名）
│   │   ├── settings.py               # BackendSettings（指向 core/config）
│   │   ├── db.py                     # SQLAlchemy 引擎 + Alembic 集成
│   │   ├── middleware/               # ★ 新建
│   │   │   ├── request_id.py         # X-Request-ID 关联
│   │   │   ├── error_handler.py      # 统一 JSON 错误响应
│   │   │   └── timing.py            # X-Response-Time 响应计时
│   │   ├── models/                   # ORM 模型（Job, Run, Sample, ModelRegistry）
│   │   ├── routes/                   # API 路由（加 /api/v1 前缀）
│   │   ├── schemas/                  # Pydantic 请求/响应 schema
│   │   ├── services/                 # 业务服务
│   │   └── migrations/              # ★ 新建：Alembic 迁移
│   │       ├── env.py
│   │       ├── script.py.mako
│   │       └── versions/
│   │           └── 001_initial_schema.py
│   │
│   ├── worker/                       # 保留 + 改进
│   │   ├── actors.py, broker.py, cli.py
│   │   ├── queue_watcher.py          # 文件队列轮询（核心机制，保留）
│   │   ├── settings.py
│   │   └── tasks/                    # 8 种任务类型
│   │
│   └── frontend/                     # 原样保留
│       ├── package.json
│       ├── vite.config.ts
│       └── src/                      # React 18 + Ant Design
│
├── scripts/                          # Hydra CLI 入口（保留）
├── tests/                            # pytest（迁移 + 扩展）
│   ├── conftest.py
│   ├── unit/
│   │   ├── core/                     # ★ 新增
│   │   ├── models/                   # ★ 新增
│   │   ├── features/                 # ★ 新增（含 golden vector 测试）
│   │   └── ...                       # 原有测试迁移
│   └── integration/
│
└── docs/
    └── refactor/                     # 迁移文档（本目录）
```

## 异常层级

```python
MSGError                              # 所有 MSG 应用异常的基类
├── ConfigError                       # 配置加载或验证失败
├── DataSourceError                   # 数据源初始化或迭代失败
│   └── MATLABError                   # MATLAB 子进程失败
├── FeatureExtractionError            # 特征提取管线失败（SVD、归一化等）
├── ModelError                        # 模型加载、checkpoint、前向传播失败
├── TrainingError                     # 训练循环失败（NaN loss、OOM）
├── InferenceError                    # 推理或导出失败
└── PlatformError                     # 平台基础设施错误
    └── JobError                      # 任务生命周期错误
```

## 配置统一方案

```
┌──────────────────────────────────────────┐
│ core/config.py                           │
│                                          │
│ MSGSettings(BaseSettings)                │
│   repo_root, data_dir, artifacts_dir     │  ← pydantic-settings
│   db_url, redis_url, api_port            │     环境变量 + .env
│   log_level, log_format                  │
│                                          │
│ load_hydra_config(overrides) → DictConfig│  ← Hydra compose
│   configs/data, model, train, eval, infer│     ML 超参数
│                                          │
│ core/protocol_spec.py                    │
│   PROTOCOL_SPEC dict                     │  ← 3GPP 物理层范围
│   （从废弃 config.py 迁入）                │     Python dict（类型复杂）
└──────────────────────────────────────────┘
```

## 迁移优先级执行顺序

```
Phase 2.1: core/               → 基础设施地基
Phase 2.2: 工程骨架             → pyproject.toml, Makefile, Docker, CI
    ↓
Phase 3.1: models/             → ChannelMAE + LatentAdapter + EMA
Phase 3.2: features/extractor  → FeatureExtractor + Normalizer + Losses
Phase 3.3: data/               → contract + bridge + sources
Phase 3.4: platform/backend    → FastAPI + Alembic + middleware
    ↓
Phase 3.5: channel_est, channel_models, ref_signals, phy_sim, topology
Phase 3.6: training/ + inference/ + eval/
Phase 3.7: features/pmi + features/srs + features/ssb（遗留信号处理）
Phase 3.8: platform/worker
    ↓
Phase 3.9: report/ + viz/
Phase 3.10: platform/frontend
Phase 3.11: scripts/ + Docker/CI 完善
```

## 已确认的技术选型

| 项目 | 选择 | ADR |
|------|------|-----|
| 数据库 | SQLite + Alembic | ADR-001 |
| 配置管理 | Hydra ML + pydantic-settings | ADR-002 |
| 日志框架 | structlog | ADR-003 |
| 任务队列 | Dramatiq + Redis（保留） | - |
| 前端 | React 18 + Ant Design（保留） | - |
| 遗留信号处理 | 迁入 features/ 子包 | - |
