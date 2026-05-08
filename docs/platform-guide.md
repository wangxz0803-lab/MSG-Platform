# ChannelHub 平台说明书（信道数据工场）

## 概述

ChannelHub 平台是 5G NR 信道表征学习系统的**信道数据采集·处理·管理一体化工场**，基于 FastAPI + React 18 + Ant Design 构建。平台负责数据采集/管理/导出 + 模型导入/推理/评估，训练在外部平台独立进行。

### 平台定位

- **数据工厂**：采集多源信道数据 → 管理 → 划分测试集 → 多格式导出（HDF5/WebDataset/pt_dir）供外部训练平台消费
- **评估引擎**：导入外部训练好的模型 → 在锁定测试集上评估 → 推理生成嵌入 → 排行榜对比
- **不包含训练**：训练在外部执行（支持任意 PyTorch 训练框架），平台只关注数据和评估

## 数据采集（CollectWizard）

### 数据源

| 数据源 | 说明 | 依赖 |
|--------|------|------|
| internal_sim | 3GPP 38.901 多小区统计模型 | 纯 Python |
| sionna_rt | Sionna 射线追踪仿真 | GPU + Sionna 2.0 |
| quadriga_real | MATLAB 实时生成（QuaDRiGa 引擎），50+ 参数完整透传（拓扑/移动/SRS 跳频/SSB/预编码/干扰），支持 hex/linear/custom 三种拓扑、5 种移动模式、HSR 列车场景、全 K 小区干扰信道、SSB DFT 波束扫描、SVD 预编码、4 种信道估计模式（ideal/ls_linear/ls_mmse/ls_hop_concat）；与 internal_sim/sionna_rt 输出完全对齐 | 本地 MATLAB |
| internal_upload | 上传已有数据（暂未实现） | - |

### 前端统一配置模型

CollectWizard 采用统一的 `DeviceConfig` 接口，三种数据源共享同一套表单和布局。

#### 场景预设系统

Step 2 顶部提供 6 个一键预设卡片，点击后自动填充拓扑 + 天线 + 射频参数：

| 预设 | 场景 | 频段 | 带宽 | ISD | 天线 | 数据源限制 |
|------|------|------|------|-----|------|-----------|
| 城市宏站 64T | UMa | 3.5 GHz | 100 MHz | 500 m | 8×4×2 = 64T64R | 全部 |
| 城市微站 32T | UMi | 3.5 GHz | 100 MHz | 200 m | 4×4×2 = 32T32R | 全部 |
| 室内热点 8T | InH | 3.5 GHz | 50 MHz | 50 m | 2×2×2 = 8T8R | 仅 QuaDRiGa |
| 农村宏站 4T | RMa | 700 MHz | 20 MHz | 1732 m | 2×2×1 = 4T4R | 仅 QuaDRiGa |
| 毫米波 28G | UMi | 28 GHz | 100 MHz | 100 m | 8×4×2 = 64T64R | 全部 |
| 高铁 350km/h | UMa-LOS · HyperCell | 2.6 GHz | 100 MHz | 1000 m | 8×4×2 = 64T64R | 全部 |

预设仅做表单预填充，用户可在此基础上自由修改。手动修改任意字段后预设高亮自动取消。

**数据源约束**：室内热点（InH）和农村宏站（RMa）的路径损耗模型在 sionna_rt / internal_sim 后端未实现，因此仅在选择 QuaDRiGa 数据源时显示。切换数据源后，不兼容的预设自动隐藏，已激活的预设高亮自动清除。

**高铁预设说明**：12 站线性部署 · ISD 1000m · HyperCell ×4 · track 固定轨迹 · 350 km/h · 22 dB 车体穿透损耗。UMa-LOS 场景对应铁塔对轨道视距传播，2.6 GHz n41 为国内高铁 5G 主力频段，46 dBm 满功率覆盖，20 UE 集中分布在一列 ~400m 长的列车车厢内（对应 16 编组 CRH/CR 动车组，车厢宽 3.4m）。拓扑预览中列车以橙色虚线矩形标注。

### 高铁（HSR）专属能力

| 特性 | 参数 | 说明 |
|------|------|------|
| 线性拓扑 | `topology_layout: "linear"` | 站点交替部署在轨道两侧（±track_offset_m），主扇区朝向轨道中心 |
| 轨道偏移 | `track_offset_m: 80` | 站点到轨道中心线的垂直距离（默认 80m） |
| HyperCell 组网 | `hypercell_size: N` | 每 N 个连续 RRH 共享同一 PCI 组，UE 在组内不触发切换 |
| 轨道固定轨迹 | `mobility_mode: "track"` | UE 从第一站出发沿轨道中心线匀速行驶，到头后折返（ping-pong） |
| 车体穿透损耗 | `train_penetration_loss_db: 22` | 叠加在路径损耗上的固定附加衰减（3GPP TR 38.913 Table 7.4.4-1） |
| 列车 UE 分布 | 自动 | 所有 UE（含干扰 UE）集中在 ~400m×3.4m 列车车厢内随列车整体移动 |
| 动态 Doppler | 自动 | 每个 snapshot 相对最近基站计算径向 Doppler，而非固定参考站 |

- **线性拓扑下站点数为自由输入**（不限于蜂窝的 1/3/7/19/37），前端和后端均已适配
- **PCI 分配**：`assign_pci_hypercell()` 确保同组内所有扇区 PCI 相同，不同组间 PCI 不冲突
- **拓扑预览**：`/api/topology/preview` 根据 `topology_layout` 参数自动切换六边形或线性布局渲染；列车以橙色虚线矩形标注
- **三源适配**：sionna_rt、internal_sim 均原生支持 HSR 参数（列车 UE 分布、动态 Doppler、车体穿透损耗）；quadriga_real 透传 MATLAB

#### Collapse 面板布局（Step 2）

| 面板 | 默认展开 | 字段 |
|------|----------|------|
| 拓扑配置 | 是 | num_sites, isd_m, sectors_per_site, tx_height_m |
| 天线阵列 | 是 | BS 预设下拉 + bs_ant_h/v/p, UE 预设下拉 + ue_ant_h/v/p, xpd_db |
| 射频参数 | 否 | carrier_freq_hz, bandwidth_hz, subcarrier_spacing, tx_power_dbm |
| 终端配置 | 否 | num_ues, ue_distribution, ue_speed_kmh, ue_tx_power_dbm, mobility_mode |
| QuaDRiGa 配置 | 否 | qr_scenario, num_snapshots, ues_per_shard, skip_generation（仅 quadriga_real） |

天线阵列面板提供常用预设下拉：BS 支持 64T/32T/16T/8T/4T/2T，UE 支持 4/2/1 天线。

#### 信道配置分层（Step 3）

核心参数（链路方向、信道估计模式、导频类型、采样数、场景）始终可见，SRS 高级参数（周期、跳频、梳齿、带宽树）默认折叠。

#### 确认页（Step 4）

顶部 Tag 横幅显示关键配置摘要（数据源、站点数、天线规格、频段、带宽、采样数），下方 3 列 Descriptions 展示完整参数。

#### 后端兼容字段

提交时自动生成：`bs_panel`/`ue_panel` 数组、`num_bs_tx_ant`/`num_bs_rx_ant`/`num_ue_tx_ant`/`num_ue_rx_ant` 计算值、`num_sites` + `num_cells` 双字段。

### 链路方向与配对模式

- **UL（上行）**：仅采集上行信道，导频使用 SRS（ZC 序列）
- **DL（下行）**：仅采集下行信道，导频使用 CSI-RS（Gold-PRBS）
- **双向（BOTH）**：配对模式（`link_pairing="paired"`），同时采集：
  - UL 信道：理想信道 + 含邻区 SRS 干扰的估计信道
  - DL 信道：理想信道 + 含邻区 CSI-RS 干扰的估计信道
  - TDD 互易关系：H_DL = H_UL（contract 约定下恒等映射）
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

| 参数 | 默认值 | 说明 |
|------|--------|------|
| apply_interferer_precoding | true | 对每个邻区 BS_k，用其对自身调度用户的 SVD 预编码 W_k 投影干扰信道 H(BS_k→Q)，等效模拟调度对干扰的方向性影响 |
| store_interferer_channels | false | 是否持久存储 h_interferers（默认不存，仅作为 h_serving_est 的中间产物） |

以上两个参数可在采集向导 Step 2（信道配置）中通过"邻区预编码投影"和"存储干扰信道"开关直接配置。

**邻区预编码投影**：邻区 BS_k 对本区用户 Q 的干扰 = H(BS_k→Q) @ W_k @ s_k。W_k 由 BS_k 到其调度用户 P_k 的信道 SVD 得到。投影法 H_proj = W_k @ W_k^H @ H(BS_k→Q) 保持 shape 不变，干扰功率降低约 10·log10(rank/BS_ant) dB。每个 sample 随机选取邻区调度用户以模拟调度随机性。

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
2. **TDD 互易**：H_DL = H_UL（contract 约定下恒等映射）
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
- 三种数据源全部已适配：internal_sim / sionna_rt / quadriga_real（MATLAB 端 ssb_measurement.m）

### 路损模型（38.901 Table 7.4.1-1）

根据 LOS/NLOS 概率自动选择对应路损公式：

| 场景 | LOS 公式 | NLOS 公式 | σ_SF (LOS/NLOS) |
|------|----------|-----------|-----------------|
| UMa | 28.0 + 22·log₁₀(d₃D) + 20·log₁₀(fc) (d₂D≤d'BP) | 13.54 + 39.08·log₁₀(d₃D) + 20·log₁₀(fc) | 4 / 6 dB |
| UMi | 32.4 + 21·log₁₀(d₃D) + 20·log₁₀(fc) (d₂D≤d'BP) | 22.4 + 35.3·log₁₀(d₃D) + 21.3·log₁₀(fc) | 4 / 7.82 dB |
| InF | 31.84 + 21.50·log₁₀(d₃D) + 19.0·log₁₀(fc) | 同左（简化） | 7.56 dB |

- 断点距离 d'BP = 4·h'BS·h'UT·fc/c，fc 取 Hz
- LOS 概率按 Table 7.4.2-1 自动判定，LOS 时同步切换 TDL 模型（TDL-D）和 LSP 参数
- **per-cell LOS 判定**：sionna_rt 中每个小区独立计算 LOS 概率（基于各自到 UE 的距离），不再共用 sites[0] 的 LOS 状态
- 已在三种数据源全部适配：internal_sim（TDL 路径）、sionna_rt（TDL fallback 路径）、quadriga_real（MATLAB 通道）

### 信道固有属性保障

平台在信道生成阶段通过以下机制保障信道物理可信性：

| 属性 | 机制 | 说明 |
|------|------|------|
| **空间一致性** | 位置量化确定性采样 | 大尺度参数（DS, SF）使用 UE 位置量化到相关距离网格的确定性 RNG，相邻位置共享相同 LSP |
| **TDD 互易性** | 频域平滑校准噪声 | H_DL = H_UL + 频域低秩插值噪声（非 i.i.d.），模拟真实 TDD 校准误差的频率相关性 |
| **per-cell LOS** | 独立 LOS 判定 | 每个小区根据各自到 UE 的距离独立判定 LOS/NLOS，避免错误路损模型选择 |
| **信道溯源** | channel_generation_mode 标记 | sionna_rt 在元数据中标记 `"sionna_rt"` 或 `"tdl_fallback"`，便于数据质量筛选 |
| **理想信道纯净性** | 无干扰保证 | h_serving_true 直接取自单 BS→UE 纯净信道，ideal 模式不叠加任何干扰 |

**LSP 空间相关距离**（来自 38.901 Table 7.5-6）：

| 场景 | DS (m) | SF (m) | 最小网格 |
|------|--------|--------|----------|
| UMa-NLOS | 40 | 50 | 40 m |
| UMi-NLOS | 10 | 13 | 10 m |
| InF | 10 | 10 | 10 m |
| UMa-LOS | 30 | 37 | 8 m |
| UMi-LOS | 7 | 10 | 7 m |

### UL 功率与 Pre-SINR

- **UE 发射功率**（`ue_tx_power_dbm`）：默认 23 dBm（3GPP 38.101 PC3），用于 UL 链路预算
- **DL SNR/SINR** 使用基站发射功率 `tx_power_dbm`（如 43 dBm）
- **UL SNR/SINR** 使用终端发射功率 `ue_tx_power_dbm`（如 23 dBm），差异约 20 dB
- **Pre-SINR**（`ul_pre_sinr_dB` / `ul_pre_sinr_per_rb`）：
  - 模拟真实基站从 SRS 测量中估计的上行 SINR
  - 计算方式：per-RB 信号功率 |h_ul_true|² 除以估计误差功率 |h_ul_est - h_ul_true|²
  - 宽带 Pre-SINR = Σ|h_true|² / Σ|h_est - h_true|²
  - 包含噪声、干扰、信道老化等效果，直接反映 SRS 估计质量
  - 存储为 ChannelSample 特征，可用于 MCS 选择、链路自适应等 ML 任务
- 三种数据源均已适配 UL SNR/SINR 和 Pre-SINR：internal_sim、sionna_rt、quadriga_real

### 双极化面阵天线

支持 3GPP 面阵天线模型（Panel Array），取代传统单极化 ULA：

- **配置方式**：`bs_panel: [N_H, N_V, N_P]`，如 `[8, 4, 2]` 表示 8H×4V×2Pol = 64T64R
- **空间相关矩阵**：R = R_H ⊗ R_V ⊗ R_P（Kronecker 结构）
  - R_H/R_V：指数衰减模型 ρ^|i-j|（水平/垂直方向独立）
  - R_P：极化相关矩阵 [[1, μ], [μ, 1]]，其中 μ = 10^(-XPD/10)
- **XPD**（交叉极化鉴别）：默认 8 dB (NLOS)，可通过 `xpd_db` 配置
- **导向矢量**：2D 面阵响应 a(φ,θ) = a_H(φ) ⊗ a_V(θ)，双极化端口共享空间相位
- **UE 天线**：`ue_panel: [1, 1, 2]` = 2 天线双极化（典型手机配置）
- **向后兼容**：不设置 `bs_panel` 时自动回退到单极化 ULA 模式

典型 TDD 配置：64T64R = 8×4×2 (bs_panel)，UE 2 天线 = 1×1×2 (ue_panel)

三种数据源均已适配面阵天线：
- **internal_sim**：面阵相关矩阵 + 导向矢量直接作用于 TDL 信道生成
- **sionna_rt**：TDL fallback 路径使用面阵相关，RT 路径使用 PlanarArray 传入面阵维度
- **quadriga_real**：面阵配置（`bs_ant_h/v/p`）透传 MATLAB，QuaDRiGa 原生支持双极化

### 其他配置

- **信道估计模式**：ideal / ls_linear / ls_mmse / ls_hop_concat
  - `ls_hop_concat`：SRS 跳频逐跳精确估计。每个 hop 独立做 LS（用该 hop 对应时刻的信道），然后按 RB 拼接，不做频域插值。过去 hop 的信道通过 Jakes 时域去相关模型生成（ρ = J₀(2πf_d·Δt)），干扰和噪声也逐跳独立。仅对 UL (SRS) 生效，DL (CSI-RS) 自动回退 ls_linear
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
- quadriga_real：MATLAB 端全流程实现（50+ 参数透传），支持 hex/linear/custom 拓扑、static/linear/random_walk/random_waypoint/track 五种移动模式、HSR 列车 UE + 动态 Doppler + 穿透损耗、SSB DFT 波束扫描（RSRP/RSRQ/SS-SINR）、SRS 3GPP 跳频（38.211 §6.4.1.4）、SVD 预编码 + rank 选择、全 K 小区干扰信道（h_interferers）、4 种信道估计模式（ideal/ls_linear/ls_mmse/ls_hop_concat）、UE 三种分布（uniform/clustered/hotspot）

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
- **DL 信道 → Token**：PMI[5-8]（38.214 Type I 码本搜索）、CQI gate
- **Cell RSRP[15]**：SSB 测量

### 处理状态追踪

每个样本具有 `stage` 字段，标识其在处理流水线中的阶段：
- **raw**：原始采集，尚未经过 Bridge 处理
- **bridged**：已完成 Bridge 特征提取，生成 16 Tokens + 8 Gates

数据处理页面展示每个数据源的 stage 分布（raw/bridged 数量）和处理进度条。
API 的 `GET /api/datasets` 响应中每个 `DatasetSummary` 包含 `stage_counts` 字典。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/datasets | 数据源聚合列表（含 stage_counts） |
| GET | /api/datasets/:source/samples | 样本分页查询（含 stage 字段） |
| POST | /api/datasets/collect | 触发数据采集任务 |
| DELETE | /api/datasets/:source | 删除数据源 |
| GET | /api/channels | Bridge 处理后的样本列表 |
| GET | /api/channels/:index | 单个样本完整数据 |
| GET/POST | /api/jobs/* | 任务管理 |
| POST | /api/topology/preview | 拓扑预览 |

## 数据集划分与测试集锁定

### 概念

平台定位为**数据工厂 + 评估引擎**，训练在外部独立平台完成。因此需要：

1. **固定测试集**：测试集一旦划定并锁定，后续新采集的数据只进训练集，保证评估基准一致。
2. **数据导出**：训练集可导出为 HDF5/WebDataset/pt 三种格式，外部训练平台直接消费。
3. **模型回流**：外部训练好的模型上传到平台，在锁定测试集上自动评估。

### Split 管理

#### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/datasets/split/status` | 查看当前 split 状态（锁定、版本、各 split 样本数） |
| POST | `/api/datasets/split` | 计算 split 并可选锁定。参数：`strategy`(random/by_position/by_beam)、`seed`、`ratios`[3]、`lock`(bool) |
| POST | `/api/datasets/split/unlock` | 解锁当前 split（管理员操作） |

#### 锁定机制

- 锁定后 `compute_split()` 不再重新划分，只将 `unassigned` 行分配到 `train`
- 新增数据（`manifest.append()`）自动标记为 `train`
- 测试集/验证集 UUID 记录在 `manifest_split_meta.json` 侧车文件中
- `split_version` 递增，评估报告绑定版本号

#### 前端操作

数据集页面 → "数据集划分与导出" 卡片：
- 选择划分策略（按位置/按波束/随机）、种子、比例
- 开关控制是否锁定
- 锁定后显示详细信息（策略、种子、比例、锁定时间）
- 解锁需要二次确认

### 数据导出

#### 支持格式

| 格式 | 文件 | 适用场景 |
|------|------|---------|
| **HDF5** (.h5) | 单文件，gzip 压缩 | 跨语言、numpy 原生，推荐用于 PyTorch/JAX 训练 |
| **WebDataset** (.tar) | tar 分片，每片 1000 样本 | 大规模流式训练（支持 `webdataset` 库） |
| **pt_dir** | 目录 + .pt 文件 | 与平台内部格式一致，最简单 |

#### 导出包内容

每个导出包都是**自包含**的：

```
msg_export_train_hdf5.h5
├── samples/           # 按序号存储的信道样本
│   ├── 0/            # h_serving_true, h_serving_est, scalars...
│   ├── 1/
│   └── ...
├── manifest_scalars/  # 筛选后的 manifest 列（uuid, source, snr_dB, ...）
└── attrs:            # contract_version, split_version, num_samples, readme_json
```

`README.json`（或 HDF5 root attrs）包含：
- 契约版本（`CONTRACT_VERSION = "1.0"`）
- Split 版本号
- 导出时间戳
- 信道维度（T, RB, BS_ant, UE_ant）
- Schema 描述（字段含义说明）
- 使用示例（Python 代码片段）

#### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/datasets/export` | 提交导出任务。参数：`format`、`split`、`source_filter`、`link_filter`、`min_snr`/`max_snr`、`include_interferers` |
| GET | `/api/datasets/exports` | 列出已完成的导出包（名称、格式、样本数、大小） |

#### CLI 导出

```bash
python scripts/run_dataset_export.py format=hdf5 split=train
python scripts/run_dataset_export.py format=webdataset split=train shard_size=500
python scripts/run_dataset_export.py format=pt_dir split=train include_interferers=true
```

#### 外部训练平台消费示例

```python
# HDF5 格式
import h5py
from msg_embedding.data.contract import ChannelSample

with h5py.File("msg_export_train_hdf5.h5", "r") as f:
    print(f"样本数: {f.attrs['num_samples']}, 契约版本: {f.attrs['contract_version']}")
    sample = ChannelSample.from_hdf5_group(f["samples/0"])
    h_true = sample.h_serving_true   # [T, RB, BS_ant, UE_ant] complex64
    h_est  = sample.h_serving_est    # 含干扰的估计信道
    pos    = sample.ue_position      # [x, y, z] — 评估标签

# WebDataset 格式
import pickle, tarfile
with tarfile.open("shard-000000.tar") as tf:
    for member in tf:
        data = pickle.loads(tf.extractfile(member).read())
        sample = ChannelSample.from_dict(data)
```

### 模型导入

#### 上传流程

1. 前端：模型仓库页 → "上传模型" → 拖拽 .pt/.pth/.ckpt 文件
2. 后端自动验证 ChannelMAE 兼容性（检查 encoder/decoder/latent_proj keys）
3. 创建 Run 记录 + ModelArtifact 注册
4. 可选：立即触发在锁定测试集上的评估

#### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/models/upload` | 上传模型 checkpoint（multipart/form-data）。字段：`file`、`run_id`(可选)、`tags`、`description` |
| POST | `/api/models/{run_id}/evaluate` | 在指定 split 上触发评估。参数：`test_split`(默认 test)、`limit`、`device` |
| POST | `/api/models/{run_id}/infer` | 推理生成嵌入。参数：`split`、`batch_size`、`limit`、`output_name` |
| GET | `/api/models/{run_id}/meta` | 获取模型训练元数据（epoch、loss、参数量、兼容性等） |
| GET | `/api/models/leaderboard` | 所有已评估模型的排行榜（按 KNN Acc 排序） |

#### 兼容性要求

上传的 checkpoint 必须是 `ChannelMAE` 的 state_dict 或包含 `{"model": state_dict}` 的字典。评估流程会：
1. 加载 checkpoint → `ChannelMAE`
2. 用 `FeatureExtractor` 提取 bridge features
3. 在锁定测试集上计算 channel charting 指标（trustworthiness, continuity, KNN consistency, Kendall tau）
4. 输出 `metrics.json` + `embeddings.parquet`

### E2E 工作流

```
1. 平台采集信道数据（三源：internal_sim / sionna_rt / quadriga_real）
2. 划分数据集 → 锁定测试集（版本 v1）
3. 导出训练集（HDF5/WebDataset）→ 外部训练平台下载
4. 继续采集新数据 → 自动进入训练集（测试集不变）
5. 重新导出扩充后的训练集 → 外部增量训练
6. 外部训练完成 → 上传 checkpoint 到平台
7. 平台在锁定测试集 v1 上自动评估 → 输出指标
8. 推理生成嵌入 → 排行榜对比 → 选最优模型
```

### 外部训练对接

#### 导出数据下载

导出完成后可通过 API 直接下载：
- `GET /api/datasets/exports` → 列出所有导出包（含 `download_url` 字段）
- `GET /api/datasets/exports/{name}/download` → 下载文件（HDF5 直接下载，目录格式自动打包为 .zip）
- 前端导出列表已集成"下载"链接

#### Python 消费示例

```python
import requests

# 1. 查看可用导出
exports = requests.get("http://platform:8000/api/datasets/exports").json()
for e in exports["exports"]:
    print(f"{e['name']}: {e['format']}, {e['num_samples']} samples")

# 2. 下载训练集
resp = requests.get(f"http://platform:8000{exports['exports'][0]['download_url']}", stream=True)
with open("train_data.h5", "wb") as f:
    for chunk in resp.iter_content(8192):
        f.write(chunk)

# 3. 训练完成后上传模型
with open("ckpt_best.pth", "rb") as f:
    result = requests.post("http://platform:8000/api/models/upload",
        files={"file": f},
        data={"tags": "mae-vicreg-v2"}).json()
print(f"Run ID: {result['run_id']}, Compatible: {result['compatible']}")

# 4. 触发评估
job = requests.post(f"http://platform:8000/api/models/{result['run_id']}/evaluate",
    json={"test_split": "test"}).json()
print(f"Eval job: {job['job_id']}")

# 5. 查看排行榜
lb = requests.get("http://platform:8000/api/models/leaderboard").json()
for i, e in enumerate(lb["entries"][:5]):
    print(f"#{i+1} {e['run_id']}: KNN={e['metrics'].get('knn_acc', '-')}")
```

## 数据库

SQLite + Alembic 迁移。Schema 保持 PostgreSQL 兼容。

### samples 表字段

核心字段：uuid, sample_id, shard_id, source, link, snr_db, sir_db, sinr_db, num_cells, ts, status, job_id, run_id, path, split

干扰/配对字段（v1.0 新增）：ul_sir_db, dl_sir_db, num_interfering_ues, link_pairing

预编码字段（v1.1 新增）：w_dl (complex64 ndarray), dl_rank (int)
