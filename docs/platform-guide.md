# MSG-Embedding 平台说明书

## 概述

MSG-Embedding 平台是 5G NR 信道表征学习系统的 Web 界面，基于 FastAPI + React 18 + Ant Design 构建。提供数据采集配置、数据集管理、信道浏览、模型训练/评估/推理/导出的全流程可视化操作。

## 数据采集（CollectWizard）

### 数据源

| 数据源 | 说明 | 依赖 |
|--------|------|------|
| internal_sim | 3GPP 38.901 多小区统计模型 | 纯 Python |
| sionna_rt | Sionna 射线追踪仿真 | GPU + Sionna 2.0 |
| quadriga_multi | MATLAB 预生成 .mat 文件 | 无 |
| quadriga_real | MATLAB 实时生成 | 本地 MATLAB |
| internal_upload | 上传已有数据（暂未实现） | - |

### 链路方向与配对模式

- **UL（上行）**：仅采集上行信道，导频使用 SRS（ZC 序列）
- **DL（下行）**：仅采集下行信道，导频使用 CSI-RS（Gold-PRBS）
- **双向（BOTH）**：配对模式（`link_pairing="paired"`），同时采集：
  - UL 信道：理想信道 + 含邻区 SRS 干扰的估计信道
  - DL 信道：理想信道 + 含邻区 CSI-RS 干扰的估计信道
  - TDD 互易关系：H_UL = conj(H_DL^T) + 小扰动（校准误差）
  - GT Token：用理想信道计算的 ground-truth token（用于未来训练目标）

### 干扰建模参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_interfering_ues | 3 | 每个邻区最大干扰 UE 数（上行 SRS），实际数量逐样本随机 [0, N] |
| 邻区数 | 自动（K-1） | 下行 CSI-RS 干扰，随机选取 [0, K-1] 个邻区 |

干扰注入是物理建模，非等效高斯噪声：
- **UL**：邻区 UE 发送不同 ZC 根的 SRS，叠加在接收信号上
- **DL**：邻区发送各自 PCI 对应的 CSI-RS Gold 序列，叠加在接收信号上
- LS 估计器用服务小区导频解调，干扰残留在估计信道中

### 其他配置

- **信道估计模式**：ideal / ls_linear / ls_mmse
- **信道模型**：TDL-A~E（NLOS: A/B/C，LOS: D/E）
- **TDD 配比**：DDDSU / DDSUU 等
- **SRS 参数**：组跳频、序列跳频、周期、频域跳频
- **SSB 波束数**：4 / 8 / 16

### 移动性建模

| 运动模式 | 说明 | 适用场景 |
|----------|------|----------|
| static | 静止，UE 位置固定 | 室内固定终端、基准测试 |
| linear | 匀速直线运动，随机方向 | 高速公路、直线道路 |
| random_walk | 随机游走，每步随机转向（高斯） | 步行用户、城市漫游 |
| random_waypoint | 随机航路点（选目标→匀速移动→到达→再选） | 通用移动场景、RWP 标准模型 |

**核心机制：**
- 轨迹生成模块 (`_mobility.py`) 为每个 UE 生成 `[num_samples, 3]` 的连续位置序列
- 多普勒频移从位置差分自动推导（径向速度），非恒定值
- 大尺度参数（时延扩展、阴影衰落）沿轨迹空间相关：`corr(d) = exp(-d / d_decorr)`
- 边界约束：UE 不会跑出网络覆盖范围（反射式边界）
- 采样间隔可配置（默认 0.5ms = 1 slot @ 30kHz SCS）

**数据源支持：**
- internal_sim：Python 端完整轨迹建模 + 空间 LSP 相关
- sionna_rt：Python 端轨迹 + Sionna RT 信道计算（或 TDL 回退）
- quadriga_real：移动性参数传递给 MATLAB QuaDRiGa 引擎（原生轨迹支持）
- quadriga_multi：数据中已包含移动性信息（由 MATLAB 预生成）

## 数据集管理

### 数据集列表（/datasets）

显示每个数据源的聚合统计：
- 样本数、SNR/SIR/SINR 均值
- UL SIR 均值、DL SIR 均值（配对模式下）
- 链路类型标签 + 配对标记

### 数据集详情（/datasets/:source）

- 统计卡片：总样本数、SNR/SIR/UL SIR/DL SIR/SINR 均值
- SINR 分布直方图
- 样本列表：含 UL SIR、DL SIR、配对标记列

### 样本字段

| 字段 | 类型 | 说明 |
|------|------|------|
| snr_dB | float | 信噪比 |
| sir_dB | float | 信干比（总） |
| sinr_dB | float | 信干噪比 |
| ul_sir_dB | float? | 上行信干比（配对模式） |
| dl_sir_dB | float? | 下行信干比（配对模式） |
| num_interfering_ues | int? | 干扰 UE 数上限 |
| link_pairing | "single"/"paired" | 配对模式标识 |

## 信道浏览器（ChannelExplorer）

浏览 Bridge 处理后的 .pt 样本文件，显示：
- 信道幅度热力图（理想/估计/误差）
- 16 个 Token 的特征可视化
- 元数据：SNR/SIR/SINR、UL SIR/DL SIR、配对模式、干扰 UE 数

## Bridge 特征提取

16 个 Token 按方向分离计算（配对模式）：
- **UL 信道 → Token**：PDP[0]、SRS[1-4]、DFT[9-12]、RSRP_SRS[13]、RSRP_CB[14]
- **DL 信道 → Token**：PMI[5-8]、CQI gate
- **Cell RSRP[15]**：SSB 测量

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/datasets | 数据源聚合列表 |
| GET | /api/datasets/:source/samples | 样本分页查询 |
| POST | /api/datasets/collect | 触发数据采集任务 |
| DELETE | /api/datasets/:source | 删除数据源 |
| GET | /api/channels | Bridge 处理后的样本列表 |
| GET | /api/channels/:index | 单个样本完整数据 |
| GET/POST | /api/jobs/* | 任务管理 |
| POST | /api/topology/preview | 拓扑预览 |

## 数据库

SQLite + Alembic 迁移。Schema 保持 PostgreSQL 兼容。

### samples 表字段

核心字段：uuid, sample_id, shard_id, source, link, snr_db, sir_db, sinr_db, num_cells, ts, status, job_id, run_id, path, split

干扰/配对字段（v1.0 新增）：ul_sir_db, dl_sir_db, num_interfering_ues, link_pairing
