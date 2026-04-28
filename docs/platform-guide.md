# MSG-Embedding 平台说明书

## 概述

MSG-Embedding 平台是 5G NR 信道表征学习系统的 Web 界面，基于 FastAPI + React 18 + Ant Design 构建。提供数据采集配置、数据集管理、信道浏览、模型训练/评估/推理/导出的全流程可视化操作。

## 数据采集（CollectWizard）

### 数据源

| 数据源 | 说明 | 依赖 |
|--------|------|------|
| internal_sim | 3GPP 38.901 多小区统计模型 | 纯 Python |
| sionna_rt | Sionna 射线追踪仿真 | GPU + Sionna 2.0 |
| quadriga_real | MATLAB 实时生成（QuaDRiGa 引擎），RB 数/SCS/TDD/SRS/link/est_mode 配置完整透传 MATLAB，DL 通过 CSI-RS pipeline 独立估计 | 本地 MATLAB |
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
- **UL**：邻区 UE 按 3GPP TS 38.211 §6.4.1.4 标准 SRS 序列（各自 n_SRS_ID + cyclic shift 区分）发送，叠加在接收信号上
- **DL**：邻区发送各自 PCI 对应的 CSI-RS Gold 序列，叠加在接收信号上
- LS 估计器用服务小区导频解调，干扰残留在估计信道中

### OFDM 与频域资源

- **带宽**：5/10/15/20/25/30/40/50/60/70/80/90/100 MHz（NR FR1 标准值）
- **子载波间隔**：15/30/60/120 kHz
- **RB 数**：自动从带宽+SCS查 3GPP TS 38.101 Table 5.3.2-1 标准表（如 100MHz/30kHz → 273 RB）
- 不再使用 `bandwidth/(12*SCS)` 公式，非标准组合会 fallback 到公式并 clamp 到 [1, 275]

### TDD 时隙配置

TDD 时隙模式严格按照 3GPP TS 38.213 §11.1 实现，**已真实应用到信道生成中**：

| 配比 | 时隙序列 | 周期 | DL:UL 比例 |
|------|----------|------|-----------|
| DDDSU | D-D-D-S-U | 5ms | 44:16 |
| DDSUU | D-D-S-U-U | 5ms | 38:30 |
| DDDDDDDSUU | D×7-S-U-U | 10ms | 104:32 |
| DDDSUDDSUU | D×3-S-U-D×2-S-U-U | 10ms | 90:32 |
| DSUUD | D-S-U-U-D | 5ms | 20:32 |

- Special 槽内符号拆分：默认 10D+2G+2U（可定制）
- 每个 sample 映射到一个 TDD slot，导频放置严格遵循时隙方向：
  - SRS 导频仅在 UL 符号上发送
  - CSI-RS 导频仅在 DL 符号上发送
  - Guard 符号不放置导频
- 元数据中记录：`tdd_slot_index`, `tdd_slot_direction`, `tdd_symbol_map`

### SRS 参数（3GPP TS 38.211 §6.4.1.4）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| srs_periodicity | 10 | SRS 周期（时隙），严格使用 3GPP Table 6.4.1.4.4-1 合法值 |
| srs_group_hopping | false | 组跳频（与序列跳频互斥） |
| srs_sequence_hopping | false | 序列跳频（与组跳频互斥） |
| srs_comb | 2 | 传输梳齿 K_TC ∈ {2, 4, 8}，决定频域密度 |
| srs_c_srs | 3 | 带宽配置索引（Table 6.4.1.4.3-1），决定 SRS 总分配带宽和跳频树 |
| srs_b_srs | 1 | 带宽层级 0-3，越大单次发送越窄、跳频位置越多 |
| srs_n_rrc | 0 | RRC 配置的频域起始 RB 位置 |
| srs_b_hop | 0 | 频域跳频参数，b_hop < B_SRS 时启用跳频 |

### SRS 频域跳频（带宽树机制）

SRS 不再全带宽发送，而是按 3GPP TS 38.211 §6.4.1.4.3 的带宽树结构进行频域跳频：

- **C_SRS** 选定 Table 6.4.1.4.3-1 中的一行，确定 `m_SRS[0..3]` 和 `N[0..3]`
- **B_SRS** 选定树的层级：每次发送覆盖 `m_SRS[B_SRS]` 个 RB
- **跳频位置数** = `N[B_SRS]`：SRS 在不同时隙跳到不同频率位置
- 完整跳频周期覆盖 `m_SRS[0]` 个 RB

示例（默认配置 C_SRS=3, B_SRS=1）：
- m_SRS = (16, 4, 4, 4)，N = (1, 4, 1, 1)
- 每次发送 4 个 RB，4 个跳频位置，完整周期覆盖 16 个 RB（4× 跳频）

**UL 信道累积估计**：基站不使用单次 SRS（仅覆盖 m_SRS[B_SRS] 个 RB）做插值，
而是累积一个完整跳频周期的所有 SRS 观测后再估计 UL 信道。累积后观测覆盖 m_SRS[0] 个 RB，
无需在 SRS 带宽内做插值。仅 SRS 带宽之外的 RB 才由估计器插值。

- 跳频周期长度 = ∏ N[b]（b = b_hop+1 .. B_SRS），C_SRS=3/B_SRS=1 时为 4 次
- 累积 RBs = m_SRS[0]，C_SRS=3 时为 16 RBs，C_SRS=14 时为 52 RBs
- 对于 52-RB 系统（10 MHz），选 C_SRS=14 可实现 UL 全带宽 SRS 覆盖（13 次跳频）

SRS 导频生成使用 `srs.py` 的完整 3GPP 实现，包含：
- 基于 Gold PRBS 的组号/序列号跳频计算
- Zadoff-Chu 基序列 + 端口级循环移位
- 不同 slot 产生不同序列（跳频开启时）
- `srs_rb_indices()` 计算单次 SRS 的 RB 覆盖
- `srs_accumulated_rb_indices()` 计算完整跳频周期的累积 RB 覆盖
- `srs_hopping_cycle_length()` 返回跳频周期长度

### DL 预编码（SRS-based Beamforming）

TDD 系统下，BS 通过 UL SRS 估计上行信道，利用互易性推导 DL 信道，再通过 SVD 计算 DL 预编码权值：

1. **UL 信道估计**：BS 从 SRS 导频估计 H_UL_est `[T, RB, BS_ant, UE_ant]`
2. **TDD 互易**：H_DL = conj(H_UL)（contract 约定下无需转置）
3. **SVD 分解**：SVD(H_DL) = U S Vh，取 W_DL = U[:, :rank]
4. **Rank 选择**：基于奇异值分布，sigma_i > 0.1 * sigma_max 的层数，取中位数作为宽带 rank
5. **等效信道**：H_eff = W^H @ H_DL，shape `[T, RB, rank, UE_ant]`

| 参数 | 取值 | 说明 |
|------|------|------|
| max_rank | min(4, BS_ant, UE_ant) | 最大传输层数，由天线维度自动确定 |
| rank_threshold | 0.1 | 奇异值低于 0.1×最大值时截断 |
| precoding_type | SVD (per-RB) | 每个 RB 独立 SVD 计算权值 |

**输出存储**：
- `w_dl`: `[RB, BS_ant, rank]` complex64 — per-RB 预编码权值矩阵
- `dl_rank`: int — 传输 rank（1-4）
- CQI 基于预编码后等效信道的 SINR 计算

**注意**：UL 方向不做预编码（UE 端不使用 beamforming weights）。

### 多 UE 干扰信道建模

UL 干扰建模使用独立的 per-UE 信道（非复用 per-cell 信道）：
- 每个邻区生成 `num_interfering_ues` 个独立 UE 位置（均匀分布在邻区覆盖范围内）
- 每个干扰 UE 独立生成 H(UE_kn → BS_serving) 信道（含路损、衰落）
- 调度仿真：每样本随机激活 [0, num_interfering_ues] 个 UE（模拟调度碰撞不确定性）
- 已适配 internal_sim / sionna_rt；quadriga_real 的 MATLAB 端待后续更新

### 其他配置

- **信道估计模式**：ideal / ls_linear / ls_mmse
- **信道模型**：TDL-A~E（NLOS: A/B/C，LOS: D/E）
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
- quadriga_real：移动性参数传递给 MATLAB QuaDRiGa 引擎（原生轨迹支持），link/est_mode/tdd_pattern/num_interfering_ues 完整透传，DL 模式下调用 dl_csirs_pipeline 独立估计

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
| w_dl | complex64 ndarray? | [RB, BS_ant, rank] DL 预编码权值 |
| dl_rank | int? | DL 传输 rank（1-4） |

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

预编码字段（v1.1 新增）：w_dl (complex64 ndarray), dl_rank (int)
