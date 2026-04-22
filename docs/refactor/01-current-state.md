# 01 - 现状盘点

## 项目概况

| 项 | 值 |
|----|-----|
| 名称 | MSG-Embedding |
| 版本 | 0.3.0-dev |
| 用途 | 5G NR 信道表征学习平台（MAE + 对比学习） |
| Python | >= 3.10 |
| 核心框架 | PyTorch 2.9.1, FastAPI, React 18, Dramatiq + Redis |
| 代码规模 | ~12K LOC（核心包），870+ 测试 |
| 部署方式 | Docker Compose（redis + api + worker + frontend） |

## 架构分层

```
┌─────────────────────────────────────────────────────┐
│ PLATFORM LAYER（Web UI）                             │
│ React 18 + TypeScript + Ant Design（前端）            │
│ FastAPI + SQLAlchemy + SQLite（后端 API）             │
│ Dramatiq + Redis（异步任务队列 + Worker）              │
├─────────────────────────────────────────────────────┤
│ SCRIPT LAYER（CLI 编排）                              │
│ run_simulate → run_train → run_eval →                │
│ run_infer → run_export → run_report                  │
│ Hydra 驱动，支持 DDP/AMP                              │
├─────────────────────────────────────────────────────┤
│ MODULE LAYER（14 子模块，src/msg_embedding/）          │
│ data/ | channel_est/ | training/ | inference/        │
│ eval/ | topology/ | ref_signals/ | phy_sim/ | viz/   │
│ features/[空壳] | models/[空壳] | report/ | utils/    │
├─────────────────────────────────────────────────────┤
│ LEGACY ROOT（活跃降级层）                              │
│ model.py | tools.py | config.py[废弃] | pretrain.py  │
│ CsiChanProcFunc.py | SsbChanProcFunc.py | gol.py     │
│ 约 15 个 5G 信号处理模块                               │
└─────────────────────────────────────────────────────┘
```

## 核心数据流

```
DataSource.iter_samples()
  → ChannelSample (Pydantic v2 契约, 4D 复数矩阵 [T, RB, BS, UE])
  → bridge.sample_to_features() (16 tokens + 8 gates)
  → FeatureExtractor() (tools.py)
  → tokens [B, 16, 128]
  → ChannelMAE.encode() (model.py)
  → latent [B, 16] (L2 归一化嵌入)
```

## 模块清单

### 核心包 src/msg_embedding/ (14 子模块)

| # | 模块 | 状态 | 文件数 | 职责 |
|---|------|------|--------|------|
| 1 | `data/` | 完成 | 8 | ChannelSample 契约、Bridge、Dataset、Manifest |
| 2 | `data/sources/` | 4/5 完成 | 7 | 5 种数据源适配器（field 占位） |
| 3 | `channel_est/` | 完成 | 5 | LS/MMSE 信道估计 + 插值 |
| 4 | `channel_models/` | 完成 | 2 | TDL-A/B/C/D/E 参数表 |
| 5 | `ref_signals/` | 完成 | 8 | 3GPP 38.211 参考信号生成（87 单测） |
| 6 | `phy_sim/` | 完成 | 3 | SSB RSRP 测量、TDD 时隙配置 |
| 7 | `topology/` | 完成 | 4 | 六边形网格、PCI 规划、场景定义 |
| 8 | `training/` | 完成 | 7 | 预训练/微调、DDP、回调、实验追踪 |
| 9 | `inference/` | 完成 | 5 | 批量推理、ONNX/TorchScript 导出 |
| 10 | `eval/` | 完成 | 4 | CT/TW/kNN 指标、NMSE |
| 11 | `report/` | 完成 | 3 | 报告生成、训练对比 |
| 12 | `viz/` | 完成 | 4 | 数据集统计、潜空间可视化、训练曲线 |
| 13 | `features/` | **空壳** | - | 占位（逻辑在根目录 tools.py） |
| 14 | `models/` | **空壳** | - | 占位（逻辑在根目录 model.py） |

### 遗留根目录文件

| 文件 | 行数 | 状态 | 职责 |
|------|------|------|------|
| `model.py` | 245 | **活跃** | ChannelMAE + LatentAdapter（唯一模型定义） |
| `tools.py` | ~700 | **活跃** | FeatureExtractor + ProtocolNormalizer + 损失函数 + EMA |
| `config.py` | 165 | 已废弃 | PROTOCOL_SPEC + 硬编码常量，仍被 model.py 导入 |
| `gol.py` | 21 | 活跃 | 全局变量字典（被信号处理模块使用） |
| `CsiChanProcFunc.py` | ~300 | 活跃 | PMI 码本查找（VAM + Type-I） |
| `SsbChanProcFunc.py` | ~250 | 活跃 | SSB/DFT 波束赋形 |
| `GetCsirsWeight.py` | ~300 | 活跃 | CSI-RS 权重计算 |
| `GetCsiWeightCore.py` | ~350 | 活跃 | 核心权重算法 |
| `SrsChanProcFunc.py` | ~230 | 活跃 | SRS 信道处理 |
| `BeamAntTrans.py` | ~130 | 活跃 | 波束-天线变换（Kronecker DFT） |
| `Precode.py` | ~200 | 活跃 | Rhh 协方差块提取、频域 DFT 投影 |
| `GetMSGEmbd_Inp.py` | - | 活跃 | SRS/SSB/CSI 三路特征提取编排 |
| `pretrain.py` | 152 | 遗留 | 旧训练循环（已被 training/pretrain.py 替代） |
| `inference.py` | - | 遗留 | 旧推理工具（已被 inference/ 替代） |
| `bridge_channel_to_pretrain.py` | - | 遗留 | 旧数据桥接（已被 data/bridge.py 替代） |

### 平台层 platform/

| 组件 | 文件数 | 技术栈 | 职责 |
|------|--------|--------|------|
| `backend/` | ~20 | FastAPI + SQLAlchemy + SQLite | REST API、ORM、服务 |
| `worker/` | ~15 | Dramatiq + Redis | 异步任务执行、进度追踪 |
| `frontend/` | ~30 | React 18 + Ant Design + Vite | 9 个页面的 Web UI |
| `deploy/` | ~15 | Docker Compose + Nginx | 容器编排、入口脚本 |

## 数据源适配器

| 数据源 | 文件 | 外部依赖 | 特殊处理 |
|--------|------|----------|----------|
| internal_sim | `internal_sim.py` | 无（NumPy/SciPy） | 快速原型、CI/CD |
| sionna_rt | `sionna_rt.py` | Sionna 2.0.1（可选） | 缺 Sionna 时自动降级为 TDL |
| quadriga_multi | `quadriga_multi.py` | scipy + hdf5storage | 读取预生成 .mat 文件 |
| quadriga_real | `quadriga_real.py` | MATLAB R2020b+ 子进程 | 3 次重试、5 分钟超时、.mat/.h5 降级 |
| field | `field.py` | - | v3.1 占位（NotImplementedError） |

## 配置管理现状

```
config.py (废弃)              → from config import * → model.py, tools.py
                                                      → 硬编码 PROTOCOL_SPEC, 维度常量, 训练超参
configs/*.yaml + Hydra         → load_config()        → scripts/run_*.py, training/, data/
platform/backend/settings.py   → pydantic-settings     → API 端口, DB URL, 文件路径
platform/worker/settings.py    → pydantic-settings     → Redis URL, 轮询间隔, 超时
```

## 数据库现状

- 引擎：SQLite（`D:/MSG/platform/backend/msg.db`）
- ORM：SQLAlchemy 2.0 声明式基类
- 建表：`Base.metadata.create_all()`（无版本化迁移）
- 表：Job、Run、Sample、ModelRegistry
- 会话：`check_same_thread=False`，`autoflush=False`

## 测试现状

- 框架：pytest
- 总量：870+ 测试
- 标记：`slow`、`gpu`、`matlab`、`sionna`
- 覆盖：ref_signals(87)、channel_est(30)、data(27)、training(36)、platform(51)
- 集成测试：bridge smoke、pipeline、pretrain single GPU
- 缺少：core/ 基础设施测试、models/ 测试、features/ 测试（因为这两个是空壳）

## CI/CD 现状

- 无 GitHub Actions 或其他 CI 管线
- 仅有 `.pre-commit-config.yaml`（Ruff + Black + 文件检查）
- Docker Compose 语法校验脚本：`test_compose_syntax.sh`

## 代码质量工具

| 工具 | 配置位置 | 规则 |
|------|----------|------|
| Ruff | pyproject.toml | E/F/I/B/UP/SIM, line-length=100 |
| Black | pyproject.toml | line-length=100, py310 |
| mypy | pyproject.toml | ignore_missing_imports=true, check_untyped_defs=false |
| pre-commit | .pre-commit-config.yaml | trailing-ws, eof, ruff, black |
