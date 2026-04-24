# MSG-Embedding 数据处理流水线技术报告

> 版本：1.0 | 日期：2026-04-23 | 状态：已验证（基于源码逐行核对）

---

## 1. 流水线总览

数据处理流水线负责将原始信道测量数据（`ChannelSample`）转换为模型可消费的定长 token 序列。整个过程分为三个阶段：

```
ChannelSample (pydantic 验证后的信道记录)
    │
    ├─ 阶段 1: Bridge (_build_feat_dict)
    │   原始信道 → 24 字段特征字典（16 token 字段 + 8 门控字段）
    │
    ├─ 阶段 2: FeatureExtractor.normalize_features()
    │   物理值 → [-1, 1] 归一化域（3GPP 协议感知）
    │
    └─ 阶段 3: FeatureExtractor.forward()
        归一化特征 → 门控 → 线性嵌入 → 定长序列 [B, 16, 128]
```

**最终输出**：`(tokens: [B, 16, 128], norm_stats: dict)` — 每个样本产生 16 个 128 维 token，供 ChannelMAE 编码器消费。

---

## 2. 输入规格：ChannelSample

> 源文件：`src/msg_embedding/data/contract.py`

### 2.1 核心信道数据

| 字段 | 形状 | 数据类型 | 物理含义 |
|------|------|----------|----------|
| `h_serving_true` | [T, RB, BS, UE] | complex64 | 理想服务小区信道 |
| `h_serving_est` | [T, RB, BS, UE] | complex64 | 估计服务小区信道（**Bridge 实际使用的输入**） |
| `h_interferers` | [K-1, T, RB, BS, UE] \| None | complex64 | 干扰小区信道（可选） |
| `interference_signal` | [T, N_RE_obs, ...] \| None | complex64 | PHY 层干扰信号（可选） |

**典型维度**：T=14 (OFDM 符号), RB=52~272 (子载波), BS=4~64 (基站天线), UE=1~4 (终端天线)

### 2.2 标量链路指标

| 字段 | 类型 | 单位 | 有效范围 | 说明 |
|------|------|------|----------|------|
| `noise_power_dBm` | float | dBm | — | 噪声底噪 |
| `snr_dB` | float | dB | [-50, 50] | 信噪比 |
| `sir_dB` | float \| None | dB | [-50, 50] | 信干比（可选） |
| `sinr_dB` | float | dB | [-50, 50] | 信干噪比 |

### 2.3 SSB 多小区测量（可选）

| 字段 | 形状 | 单位 | 说明 |
|------|------|------|------|
| `ssb_rsrp_dBm` | [K] | dBm | 各小区最佳波束 SS-RSRP |
| `ssb_rsrq_dB` | [K] | dB | 各小区 RSRQ |
| `ssb_sinr_dB` | [K] | dB | 各小区 SS-SINR |
| `ssb_best_beam_idx` | [K] | — | 各小区最佳 SSB 波束索引 |
| `ssb_pcis` | [K] | — | 物理小区 ID |

### 2.4 元数据

| 字段 | 类型 | 说明 |
|------|------|------|
| `link` | `"UL"` \| `"DL"` | 链路方向 |
| `channel_est_mode` | `"ideal"` \| `"ls_linear"` \| `"ls_mmse"` | 信道估计方式 |
| `serving_cell_id` | int | 服务小区 ID |
| `ue_position` | [3] float64 | UE 位置 (x, y, z) 米 |
| `source` | SourceType | 数据源（6 种） |
| `sample_id` | UUID4 str | 唯一标识 |
| `created_at` | datetime | 创建时间 |
| `meta` | dict | 任意附加信息 |

---

## 3. 阶段 1：Bridge — 特征提取

> 源文件：`src/msg_embedding/data/bridge.py` → `_build_feat_dict()`

Bridge 接收一个 `ChannelSample`，输出一个 24 字段的特征字典（16 个 token 字段 + 8 个门控字段）。

### 3.1 常量

```python
NORM_EPS = 1e-8
REF_POWER_OFFSET = -100.0      # RSRP 参考功率偏移 (dBm)
_TX_ANT_NUM_MAX = 64           # BS 天线维度统一为 64（不足补零、超出截断）
_PDP_TAPS = 64                 # PDP 延迟抽头数
_CELL_RSRP_DIM = 16            # 多小区 RSRP 向量长度
```

### 3.2 Step 1：时频平均信道

```
h_avg = mean(h_serving_est, axis=(T, RB))    →  [BS, UE] complex64
```

将信道矩阵在时域和频域上取平均，得到频率平坦的空间信道表示。用于后续 PMI 码本搜索和 DFT 波束计算。

---

### 3.3 Token 0：PDP（功率延迟谱）

> 函数：`_compute_pdp(h, n_taps=64)`

**计算步骤**：

```
1. h_avg_freq = mean(h, axis=T)           → [RB, BS, UE]     时间维平均
2. h_flat = h_avg_freq[:, :, 0]           → [RB, BS]          取第 1 根 UE 天线
3. pdp_raw = |IFFT(h_flat, axis=RB)|²     → [RB, BS]          沿频率轴 IFFT 得功率延迟谱
4. pdp = mean(pdp_raw, axis=BS)           → [RB]              对 BS 天线取均值
5. pdp_norm = pdp / max(pdp)              → [RB]              归一化到 [0, 1]
6. 截取/补零到 64 taps                     → [64] float32
```

**数学公式**：

$$\text{PDP}[\tau] = \frac{1}{N_{BS}} \sum_{b=0}^{N_{BS}-1} \left| \text{IFFT}_{f}\left\{ \bar{h}[f, b, 0] \right\} [\tau] \right|^2$$

其中 $\bar{h} = \frac{1}{T}\sum_t h[t, f, b, u]$ 是时间平均信道。

**输出**：`pdp_crop` → `[1, 64]` float32，范围 [0, 1]

**物理含义**：信道的多径时延扩展分布。峰值位置反映主径和反射径的时延，峰值数量反映多径丰富度。

---

### 3.4 Tokens 1–4：SRS（空间参考信号 — 协方差特征向量）

**Step A：空间协方差矩阵计算（IIR 滤波）**

> 函数：`_compute_spatial_covariance_iir(h, alpha=0.2)`

对每个时隙 t 计算空间协方差，然后做一阶 IIR 时间滤波：

```
对每个 t = 0, 1, ..., T-1:
    H_col = h[t].transpose(BS, RB, UE).reshape(BS, RB×UE)    → [BS, RB×UE]
    R_t = (H_col × H_col^H) / (RB × UE)                      → [BS, BS]
    若 t == 0:
        R_filt = R_t
    否则:
        R_filt = α × R_t + (1-α) × R_filt                     α = 0.2
```

**数学公式**：

$$R_t = \frac{1}{N_{RB} \cdot N_{UE}} \sum_{f,u} \mathbf{h}_{t,f,u} \, \mathbf{h}_{t,f,u}^H \quad \in \mathbb{C}^{N_{BS} \times N_{BS}}$$

$$R_{\text{filt}}[t] = \alpha \cdot R_t + (1-\alpha) \cdot R_{\text{filt}}[t-1], \quad \alpha=0.2$$

**输出**：`R_hh` → [BS, BS] complex128，Hermitian 正半定矩阵

**物理含义**：空间信道协方差矩阵。IIR 平滑（α=0.2）在保留准静态空间结构的同时降低方差。

**Step B：特征分解 → SRS token**

> 函数：`_srs_from_covariance(R_hh)`

```
1. eigvals, eigvecs = eigh(R_hh)           → 特征值升序
2. 翻转为降序：eigvals[::-1], eigvecs[:, ::-1]
3. srs[i] = eigvecs[:, i]                  → 取前 4 个特征向量
4. eigvals_top4 = max(eigvals[:4], 0)      → 确保非负
5. 每个 srs[i] 补零/截断到 64             → [64] complex64
```

**数学公式**：

$$R_{hh} = \mathbf{V} \Lambda \mathbf{V}^H, \quad \lambda_0 \geq \lambda_1 \geq \lambda_2 \geq \lambda_3$$

$$\text{srs}_i = \mathbf{v}_i \in \mathbb{C}^{N_{BS}}, \quad i = 0,1,2,3$$

**输出**：
- `srs1`~`srs4` → 各 `[1, 64]` complex64
- `srs_cov_eigvals` → [4] float64（供门控权重使用）

**物理含义**：信道的主空间模式。第一特征向量捕获主导空间方向，后续向量捕获正交子空间。特征值反映各方向能量强度。

---

### 3.5 门控字段：SRS 特征值权重

```
ev_sum = sum(eigvals_top4)
若 ev_sum > 1e-8:
    srs_w[i] = eigvals_top4[i] / ev_sum    → 归一化，∑ = 1
否则:
    srs_w[i] = 0.25                         → 均匀分布
```

**输出**：`srs_w1`~`srs_w4` → 各 `[1]` float32，范围 [0, 1]，和为 1

**物理含义**：反映能量在空间模式间的分布。`srs_w1` ≈ 1 表示信道近似秩-1（单一主导方向），`srs_w1` ≈ `srs_w2` ≈ 0.5 表示两个等强度散射路径。

---

### 3.6 Tokens 5–8：PMI（预编码矩阵指示）

提供两种模式：

#### 模式 A：DFT 码本搜索（默认，`use_legacy_pmi=False`）

> 函数：`_pmi_dft_codebook_search(h_avg, oversampling=4)`
>
> 3GPP 38.214 §5.2.2.2.1 Type-I 单面板码本

```
1. n_beams = N_tx × oversampling                           = BS × 4
2. 构建 DFT 码本：
   codebook[b, m] = exp(j·2π·b·m / n_beams) / √N_tx      b ∈ [0, n_beams), m ∈ [0, N_tx)
3. h_col = h_avg[:, 0]                                     取第 1 根 UE 天线
4. proj_power[b] = |codebook[b, :] · h_col|²              各波束投影功率
5. top4_idx = argsort(proj_power)[::-1][:4]                取功率最大的 4 个波束
6. pmi[i] = codebook[top4_idx[i], :]                       → [N_tx] complex64
```

**数学公式**：

$$\mathbf{c}_b[m] = \frac{1}{\sqrt{N_{tx}}} e^{j \frac{2\pi b m}{N_{beams}}}, \quad b \in [0, N_{beams}), \; m \in [0, N_{tx})$$

$$P_b = \left| \mathbf{c}_b^H \mathbf{h}_{\text{avg}}[:,0] \right|^2$$

$$\text{pmi}_i = \mathbf{c}_{b^*_i}, \quad b^*_i = \text{top-}4\text{ of } P_b$$

#### 模式 B：遗留 PMI（`use_legacy_pmi=True`）

> 函数：`_legacy_pmi_tokens(h_sim)`

调用遗留的 `CsiChanProcFunc.PmiCqiRiGenerator`，使用 CSV 码本文件。失败时回退到 SVD：用 `h_avg` 的右奇异向量 `Vh` 作为 PMI。

**输出**：`pmi1`~`pmi4` → 各 `[1, 64]` complex64（补零到 64）

**物理含义**：3GPP 标准化的发射预编码向量。DFT 码本穷举搜索波束方向，选择信号最强的 4 个方向。

---

### 3.7 Tokens 9–12：DFT 波束

```
1. n_dft = min(BS, 64)
2. dft_matrix = FFT(I_{n_dft}) / √n_dft                    → [n_dft, n_dft] 标准 DFT 矩阵
3. beam_response = dft_matrix × h_avg[:n_dft, 0]           → [n_dft] 波束域信道
4. beam_power = |beam_response|²                            → [n_dft] 各波束线性功率
5. top4_idx = argsort(beam_power)[::-1][:4]
6. dft[i] = dft_matrix[top4_idx[i], :]                     → [n_dft] → 补零到 [64]
```

**数学公式**：

$$\text{DFT}[k, m] = \frac{1}{\sqrt{N}} e^{j \frac{2\pi k m}{N}}$$

$$\text{beam\_power}[k] = \left| \sum_{m=0}^{N-1} \text{DFT}[k, m] \cdot h_{\text{avg}}[m, 0] \right|^2$$

**输出**：`dft1`~`dft4` → 各 `[1, 64]` complex64

**物理含义**：按能量排序的 top-4 空间特征波束。每个波束向量对应一个平面波方向；功率表征该方向的信道丰富度。

---

### 3.8 Token 13：RSRP_SRS（逐天线参考信号接收功率）

> 函数：`_compute_rsrp_srs(h)`

```
pwr = mean(|h|², axis=(T, RB, UE))           → [BS]  各 BS 天线的平均功率（线性）
rsrp_srs = 10 × log10(pwr + 1e-30) + (-100)  → [BS]  dBm
补零到 [64]，填充值 = -160.0 dBm
```

**数学公式**：

$$\text{RSRP\_SRS}[b] = 10 \log_{10}\!\left( \frac{1}{T \cdot N_{RB} \cdot N_{UE}} \sum_{t,f,u} |h[t,f,b,u]|^2 \right) + (-100) \quad [\text{dBm}]$$

**输出**：`rsrp_srs` → `[1, 64]` float32，范围 ≈ [-160, -60] dBm

**物理含义**：逐 BS 天线的平均接收功率。反映天线间功率分布和阵列增益模式。

---

### 3.9 Token 14：RSRP_CB（波束域参考信号接收功率）

> 函数：`_compute_rsrp_cb(beam_power)`

```
rsrp_cb = 10 × log10(beam_power + 1e-30) + (-100)   → [n_dft] dBm
补零到 [64]，填充值 = -160.0 dBm
```

**数学公式**：

$$\text{RSRP\_CB}[k] = 10 \log_{10}(\text{beam\_power}[k]) + (-100) \quad [\text{dBm}]$$

其中 `beam_power` 是 Step 3.7 中计算的 DFT 波束域功率。

**输出**：`rsrp_cb` → `[1, 64]` float32

**物理含义**：波束域 RSRP。指示哪些 DFT 波束捕获了最多的信号能量。

---

### 3.10 Token 15：Cell RSRP（多小区 RSRP）

```
cell_rsrp = [-160.0] × 16                    → 初始化全部为 -160 dBm（底噪）
若 ssb_rsrp_dBm 存在且非空：
    rsrp_sorted = sort(ssb_rsrp_dBm, 降序)
    cell_rsrp[:n_cells] = rsrp_sorted[:n_cells]
否则：
    cell_rsrp[0] = -110.0                     → 默认服务小区
```

**输出**：`cell_rsrp` → `[1, 16]` float32

**物理含义**：按功率降序排列的 top-16 小区 RSRP。反映多小区干扰环境；-160 dBm 填充值表示未检测到的小区。

---

### 3.11 门控字段：SINR 与 CQI

**srs_sinr** — SRS SINR：

```
sinr_db = clip(sample.sinr_dB, -20, 20)
```

**输出**：`srs_sinr` → `[1]` float32，范围 [-20, 20] dB

**srs_cb_sinr** — 码本 SINR：

```
srs_cb_sinr = sinr_db                        → 当前与 srs_sinr 相同
```

**输出**：`srs_cb_sinr` → `[1]` float32

**cqi** — 信道质量指示：

> 函数：`_cqi_from_sinr(sinr_db)`

```
sinr_lin = 10^(sinr_dB / 10)
SE = log2(1 + sinr_lin)                       → 香农容量 [bps/Hz]
CQI = round(SE × 15 / 7.4)                   → 映射到 [0, 15]
CQI = clip(CQI, 0, 15)
```

**数学公式**：

$$\text{CQI} = \text{clip}\!\left( \text{round}\!\left( \frac{\log_2(1 + 10^{SINR_{dB}/10}) \times 15}{7.4} \right),\; 0,\; 15 \right)$$

**输出**：`cqi` → `[1]` int64，范围 [0, 15]

**物理含义**：3GPP CQI 索引。CQI=15 对应 SE ≈ 7.4 bps/Hz（最高），CQI=0 对应 SE ≈ 0（最低）。

---

### 3.12 干扰上下文（可选，不进入 token 序列）

> 函数：`compute_interference_features(sample)`

当 `h_interferers` 存在时，计算干扰空间特征并存入 `norm_stats['interference']`，**不改变 16-slot token 布局**。

**干扰协方差**：

```
h = h_interferers.reshape(BS, K×T×RB×UE)     → [BS, N]
R_intf = (h × h^H) / N                        → [BS, BS]
```

**特征分解**：

```
eigvals, eigvecs = eigh(R_intf)               → 升序
取 top-4 特征值和特征向量（降序）
num_strong = count(eigvals > 0.1 × eigvals[0])  → 强干扰源数量
```

**到达方向（DoA）估计**：

- **MUSIC 谱**（首选）：

$$P_{\text{MUSIC}}(\theta) = \frac{1}{\mathbf{a}^H(\theta) \, \mathbf{E}_n \mathbf{E}_n^H \, \mathbf{a}(\theta)}$$

其中 $\mathbf{a}(\theta) = [1, e^{j2\pi d\sin\theta}, \ldots, e^{j2\pi(N_{BS}-1)d\sin\theta}]^T$，$d = 0.5\lambda$（ULA 半波长间距），$\mathbf{E}_n$ 是噪声子空间。

- **Capon 谱**（MUSIC 退化时回退）：

$$P_{\text{Capon}}(\theta) = \frac{1}{\mathbf{a}^H(\theta) \, R_{\text{intf}}^{-1} \, \mathbf{a}(\theta)}$$

取谱峰值的 top-4 角度作为 DoA 估计。

**输出字段**（存入 `norm_stats['interference']`）：

| 字段 | 类型/形状 | 说明 |
|------|-----------|------|
| `sir_linear` | float | 10^(SIR_dB/10) |
| `sinr_linear` | float | 10^(SINR_dB/10) |
| `intf_cov` | [BS, BS] complex64 | 干扰协方差矩阵 |
| `eigvals_top4` | [4] float32 | Top-4 特征值（降序） |
| `eigvecs_top4` | [BS, 4] complex64 | Top-4 特征向量 |
| `doa_peaks_deg` | [4] float32 | Top-4 DoA 峰值角度（度） |
| `num_strong` | int | 强干扰源数量 |
| `spectrum_kind` | str | `"music"` / `"capon"` / `"none"` |

---

### 3.13 Bridge 输出汇总

**24 字段 feat_dict 总表**：

| 槽位 | 字段名 | 形状 | 数据类型 | 值域 | 类别 |
|------|--------|------|----------|------|------|
| 0 | `pdp_crop` | [1, 64] | float32 | [0, 1] | Token |
| 1 | `srs1` | [1, 64] | complex64 | C | Token |
| 2 | `srs2` | [1, 64] | complex64 | C | Token |
| 3 | `srs3` | [1, 64] | complex64 | C | Token |
| 4 | `srs4` | [1, 64] | complex64 | C | Token |
| 5 | `pmi1` | [1, 64] | complex64 | C | Token |
| 6 | `pmi2` | [1, 64] | complex64 | C | Token |
| 7 | `pmi3` | [1, 64] | complex64 | C | Token |
| 8 | `pmi4` | [1, 64] | complex64 | C | Token |
| 9 | `dft1` | [1, 64] | complex64 | C | Token |
| 10 | `dft2` | [1, 64] | complex64 | C | Token |
| 11 | `dft3` | [1, 64] | complex64 | C | Token |
| 12 | `dft4` | [1, 64] | complex64 | C | Token |
| 13 | `rsrp_srs` | [1, 64] | float32 | dBm | Token |
| 14 | `rsrp_cb` | [1, 64] | float32 | dBm | Token |
| 15 | `cell_rsrp` | [1, 16] | float32 | dBm | Token |
| — | `srs_w1` | [1] | float32 | [0, 1] | Gate |
| — | `srs_w2` | [1] | float32 | [0, 1] | Gate |
| — | `srs_w3` | [1] | float32 | [0, 1] | Gate |
| — | `srs_w4` | [1] | float32 | [0, 1] | Gate |
| — | `srs_sinr` | [1] | float32 | [-20, 20] dB | Gate |
| — | `srs_cb_sinr` | [1] | float32 | [-20, 20] dB | Gate |
| — | `cqi` | [1] | int64 | [0, 15] | Gate |

另附 `context` 元数据：`bs_native`, `ue_native`, `t`, `rb`, `used_legacy_pmi`, `sample_id`。

---

## 4. 阶段 2：归一化

> 源文件：`src/msg_embedding/features/normalizer.py` + `src/msg_embedding/core/protocol_spec.py`

### 4.1 3GPP 协议值域规格

> 来源：3GPP TS 38.214 / 38.133 / 38.213

| 字段 | 最小值 | 最大值 | 单位 | 处理模式 |
|------|--------|--------|------|----------|
| `cqi` | 0 | 15 | index | 离散 min-max |
| `srs_sinr` | -20 | 20 | dB | 先转线性域 (0.01~100)，再 min-max |
| `srs_cb_sinr` | -20 | 20 | dB | 先转线性域 (0.01~100)，再 min-max |
| `rsrp_srs` | -160 | -60 | dBm | dB 域直接 min-max |
| `rsrp_cb` | -160 | -60 | dBm | dB 域直接 min-max |
| `cell_rsrp` | -160 | -60 | dBm | dB 域直接 min-max |
| `srs_w1`~`srs_w4` | 0 | 1 | linear | 线性域 min-max |

### 4.2 归一化公式

所有归一化均映射到 **[-1, 1]**。

#### dB 域字段（rsrp_srs, rsrp_cb, cell_rsrp, cqi）

$$x_{\text{norm}} = \text{clamp}\!\left( 2 \cdot \frac{x - x_{\min}}{x_{\max} - x_{\min}} - 1, \; -1, \; 1 \right)$$

**示例**（RSRP = -110 dBm）：

$$x_{\text{norm}} = 2 \times \frac{-110 - (-160)}{-60 - (-160)} - 1 = 2 \times \frac{50}{100} - 1 = 0.0$$

#### 线性域字段（srs_sinr, srs_cb_sinr, srs_w1~w4）

$$x_{\text{lin}} = \begin{cases} 10^{x_{dB}/10} & \text{if unit=dB} \\ x & \text{if unit=linear} \end{cases}$$

$$x_{\text{norm}} = \text{clamp}\!\left( 2 \cdot \frac{x_{\text{lin}} - x_{\text{lin,min}}}{x_{\text{lin,max}} - x_{\text{lin,min}}} - 1, \; -1, \; 1 \right)$$

**示例**（SINR = 5 dB）：

$$x_{\text{lin}} = 10^{0.5} \approx 3.16, \quad x_{\text{norm}} = 2 \times \frac{3.16 - 0.01}{100 - 0.01} - 1 \approx -0.937$$

#### 复数向量（srs, pmi, dft）— RMS 归一化

```python
power = |v|²                                  → [B, 64]
rms = sqrt(mean(power, dim=-1) + eps)         → [B, 1]
v_norm = v / (rms + eps)
```

$$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\sqrt{\frac{1}{N}\sum_{m=0}^{N-1}|v_m|^2} + \epsilon}$$

归一化后向量的 RMS 幅值 ≈ 1，保留相位和角度关系。`rms` 值存入 `norm_stats[key]["rms"]` 供反归一化使用。

#### PDP

```python
pdp_norm = clamp(pdp, 0, 1) × 2 - 1          → [0, 1] → [-1, 1]
```

---

## 5. 阶段 3：Token 嵌入

> 源文件：`src/msg_embedding/features/extractor.py` → `FeatureExtractor.forward()`

### 5.1 网络结构

```
嵌入层：
├─ complex_embed:  Linear(128 → 128)    # 实部+虚部拼接: 2×64=128
├─ pdp_embed:      Linear(64 → 128)
├─ real_embed:     Linear(64 → 128)     # RSRP 实数向量
└─ cell_embed:     Linear(16 → 128)     # 多小区 RSRP

门控网络：
├─ quality_gate_proj:  Linear(1→16) → GELU → Linear(16→1) → Sigmoid
├─ energy_gate_proj:   Linear(4→16) → GELU → Linear(16→4) → Sigmoid
└─ pmi_gate_proj:      Linear(1→16) → GELU → Linear(16→1) → Sigmoid
```

### 5.2 门控机制

门控机制根据链路质量和特征能量分布对 token 进行加权，低质量/低能量特征被衰减但不完全置零。

#### 质量门控（Quality Gate）— 基于 SINR

```
sinr_robust = tanh(srs_sinr_norm × 2.5)          → [-1, 1]
quality_gate = Sigmoid(MLP(sinr_robust))          → [0, 1]
quality_gate = quality_gate × 0.3 + 0.7           → [0.7, 1.0]
```

**行为**：高 SINR → gate ≈ 1.0（特征完整通过）；低 SINR → gate ≈ 0.7（衰减 30%）

#### 能量门控（Energy Gate）— 基于特征值权重

```
energy_sc = stack([srs_w1, srs_w2, srs_w3, srs_w4])  → [B, 4]
energy_gates = Sigmoid(MLP(energy_sc)) × 0.3 + 0.7    → [B, 4] ∈ [0.7, 1.0]
srs_gates = min(quality_gate.expand(4), energy_gates)  → [B, 4]
```

**行为**：对 SRS tokens 施加逐模式门控。主导模式（高 srs_w）gate ≈ 1.0，弱模式 gate 被压低。

#### PMI 门控（PMI Gate）— 基于 CQI

```
cqi_robust = tanh(cqi_norm × 2.0)
pmi_gate = Sigmoid(MLP(cqi_robust)) × 0.3 + 0.7       → [0.7, 1.0]
```

**行为**：高 CQI（信道好）→ PMI 特征权重高；低 CQI → PMI 特征被衰减。

### 5.3 Token 生成

#### Token 0：PDP

```python
sinr_linear = 10^(sinr_dB / 10)
weight = clamp((sinr_linear - 0.01) / (100 - 0.01), 0, 1)
pdp_weighted = pdp_norm × weight                       → [B, 64]
token_0 = Linear_64→128(pdp_weighted)                  → [B, 128]
```

#### Tokens 1–12：复数特征（SRS / PMI / DFT）

```python
# 对每个 key ∈ {srs1,..,srs4, pmi1,..,pmi4, dft1,..,dft4}:
v = norm_feat[key]                                     → [B, 64] complex

# 应用门控
if key.startswith("srs"):
    v_gated = v × srs_gates[:, i]                      → 逐模式门控
elif key.startswith("pmi"):
    v_gated = v × pmi_gate                             → CQI 门控
else:  # dft
    v_gated = v                                         → 不门控

# 补零/截断到 64 维
v_padded = pad_or_truncate(v_gated, 64)

# 拼接实部虚部
cat = [v_padded.real, v_padded.imag]                   → [B, 128]

# 线性嵌入
token = Linear_128→128(cat)                            → [B, 128]
```

#### Tokens 13–14：RSRP（SRS / CB）

```python
# RSRP 门控（独立于 quality_gate）
srs_sinr_gate = tanh(srs_sinr_norm × 2.5) × 0.3 + 0.7
cb_sinr_gate  = tanh(srs_cb_sinr_norm × 2.5) × 0.3 + 0.7

v_gated = rsrp_norm × gate                             → [B, 64]
token = Linear_64→128(v_gated)                         → [B, 128]
```

#### Token 15：Cell RSRP

```python
token = Linear_16→128(cell_rsrp_norm)                  → [B, 128]
```

### 5.4 序列组装

```python
tokens = stack([token_0, ..., token_15])               → [B, 16, 128]

# 确保精确 16 长度（截断或补零）
if tokens.shape[1] > 16: tokens = tokens[:, :16]
if tokens.shape[1] < 16: tokens = pad_zeros(tokens, target=16)

# Token mask: True = 该位置缺失（用于下游 attention mask）
token_mask = [not present[i] for i in range(16)]       → [B, 16] bool
```

**最终输出**：`(tokens: [B, 16, 128] float32, norm_stats: dict)`

---

## 6. 损失函数

> 源文件：`src/msg_embedding/features/losses.py`

### 6.1 重建损失

ChannelMAE 从 masked token 预测原始 token，用加权 MSE 评估重建质量。

**特征权重**：

| 特征 | 权重 | 理由 |
|------|------|------|
| SRS (srs1~srs4) | 2.0× | 空间模式是信道制图核心 |
| PMI (pmi1~pmi4) | 1.5× | 预编码质量直接影响吞吐量 |
| PDP, DFT, RSRP | 1.0× | 标准权重 |
| Cell RSRP | 0.5× | 辅助信息，权重降半 |

**公式**：

$$\mathcal{L}_{\text{recon}} = \sum_{k} w_k \cdot \text{MSE}(\hat{x}_k, x_k^{\text{target}})$$

其中对复数特征：

$$\text{MSE}_{\text{complex}} = \text{MSE}(\text{Re}(\hat{x}), \text{Re}(x)) + \text{MSE}(\text{Im}(\hat{x}), \text{Im}(x))$$

### 6.2 对比损失（SimCLR 风格）

$$\mathcal{L}_{\text{cont}} = \frac{1}{2} \left[ \text{CE}(\text{sim}_{12}, \text{labels}) + \text{CE}(\text{sim}_{21}, \text{labels}) \right]$$

其中：

$$\text{sim}_{12}[i, j] = \frac{\hat{\mathbf{z}}_1^{(i)T} \hat{\mathbf{z}}_2^{(j)}}{\tau}, \quad \hat{\mathbf{z}} = \frac{\mathbf{z}}{||\mathbf{z}||_2}, \quad \tau = 0.07$$

可选正则化项：正交性损失、平滑性损失、均匀性损失（权重 0.01）。

---

## 7. 反归一化

> 源文件：`src/msg_embedding/features/normalizer.py` → `denormalize()`

将模型输出从 [-1, 1] 域恢复到物理值域：

| 类型 | 反归一化公式 |
|------|-------------|
| dB 域 | $x = \frac{x_{\text{norm}} + 1}{2} (x_{\max} - x_{\min}) + x_{\min}$ |
| 线性域 | $x_{\text{lin}} = \frac{x_{\text{norm}} + 1}{2} (x_{\text{lin,max}} - x_{\text{lin,min}}) + x_{\text{lin,min}}$，若原始单位为 dB 则再取 $10\log_{10}$ |
| 离散 | 同 dB 域 + `round()` |
| 复数向量 | $\mathbf{v} = \mathbf{v}_{\text{norm}} \times (\text{rms} + \epsilon)$ |
| PDP | $\text{pdp} = \frac{x_{\text{norm}} + 1}{2}$ |

---

## 8. 端到端数据流示例

```
输入：
  h_serving_est: [14, 272, 16, 2] complex64
  sinr_dB: 5.0
  ssb_rsrp_dBm: [-80, -95, -110]

Bridge 输出 (feat_dict):
  pdp_crop:    [1, 64] float32 ∈ [0, 1]
  srs1~srs4:   [1, 64] complex64 (协方差特征向量, 补零到 64)
  pmi1~pmi4:   [1, 64] complex64 (DFT 码本 top-4 波束)
  dft1~dft4:   [1, 64] complex64 (DFT 波束域 top-4)
  rsrp_srs:    [1, 64] float32 ≈ [-140, -80] dBm
  rsrp_cb:     [1, 64] float32
  cell_rsrp:   [1, 16] = [-80, -95, -110, -160, ..., -160]
  srs_w1~w4:   [1] ∈ [0, 1], 和 = 1
  srs_sinr:    [1] = 5.0
  cqi:         [1] = 5 (SE ≈ 2.6 → round(2.6×15/7.4) ≈ 5)

归一化后:
  pdp_norm:    [1, 64] ∈ [-1, 1]
  srs_norm:    [1, 64] complex, RMS ≈ 1
  rsrp_norm:   [1, 64] ∈ [-1, 1]   (−110 dBm → 0.0)
  sinr_norm:   [1] ≈ -0.937        (3.16 线性)
  cqi_norm:    [1] ≈ -0.333        (5/15 × 2 - 1)

FeatureExtractor 输出:
  tokens:      [1, 16, 128] float32  → 送入 ChannelMAE 编码器
  token_mask:  [1, 16] bool          → 全 False（所有字段存在）
```

---

## 9. Token 布局速查表

```
┌──────┬──────────────┬──────────┬─────────────┬──────────────────────────────┐
│ 槽位 │ 字段         │ 嵌入层   │ 门控        │ 物理含义                     │
├──────┼──────────────┼──────────┼─────────────┼──────────────────────────────┤
│  0   │ pdp_crop     │ pdp_embed│ SINR 加权   │ 功率延迟谱                   │
│  1   │ srs1         │ complex  │ srs_gate[0] │ 主空间模式                   │
│  2   │ srs2         │ complex  │ srs_gate[1] │ 第 2 空间模式                │
│  3   │ srs3         │ complex  │ srs_gate[2] │ 第 3 空间模式                │
│  4   │ srs4         │ complex  │ srs_gate[3] │ 第 4 空间模式                │
│  5   │ pmi1         │ complex  │ pmi_gate    │ 最佳预编码方向               │
│  6   │ pmi2         │ complex  │ pmi_gate    │ 第 2 预编码方向              │
│  7   │ pmi3         │ complex  │ pmi_gate    │ 第 3 预编码方向              │
│  8   │ pmi4         │ complex  │ pmi_gate    │ 第 4 预编码方向              │
│  9   │ dft1         │ complex  │ 无          │ 最强 DFT 波束                │
│ 10   │ dft2         │ complex  │ 无          │ 第 2 DFT 波束               │
│ 11   │ dft3         │ complex  │ 无          │ 第 3 DFT 波束               │
│ 12   │ dft4         │ complex  │ 无          │ 第 4 DFT 波束               │
│ 13   │ rsrp_srs     │ real     │ SINR 门控   │ 逐天线接收功率               │
│ 14   │ rsrp_cb      │ real     │ CB SINR门控 │ 波束域接收功率               │
│ 15   │ cell_rsrp    │ cell     │ 无          │ 多小区 RSRP (top-16)         │
└──────┴──────────────┴──────────┴─────────────┴──────────────────────────────┘
```

---

## 10. 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 固定 16-slot 布局 | 所有样本统一 16 token | 保证 Transformer 编码器定长输入；缺失字段用零 token + mask 处理 |
| IIR 平滑 α=0.2 | 时间维低通滤波 | 在 10~100ms 信道相干时间内平衡方差降低与结构保留 |
| RMS 归一化 | 复数向量统一幅值 | 去除幅度变化，保留相位和角度关系 |
| Min-max → [-1, 1] | 统一 dB/线性域 | 裁剪防止极端异常值破坏梯度 |
| 门控最低 0.7 | `gate × 0.3 + 0.7` | 即使最差信道也保留 70% 信息流，避免梯度消失 |
| DFT 码本（默认） | 3GPP Type-I 单面板 | 标准兼容、无外部依赖；遗留路径保留向后兼容 |
| 天线维统一 64 | 补零/截断 | 确保嵌入层权重共享，支持 4~64 天线配置 |
| 干扰上下文不进 token | 存入 norm_stats | 保持 16-slot 布局不变，下游按需路由 |

---

## 11. 源文件索引

| 文件 | 核心函数/类 | 职责 |
|------|-------------|------|
| `data/contract.py` | `ChannelSample` | 数据契约（pydantic 验证） |
| `data/bridge.py` | `_build_feat_dict()` | 信道 → 24 字段特征字典 |
| | `_compute_pdp()` | PDP 计算（IFFT） |
| | `_compute_spatial_covariance_iir()` | 空间协方差（IIR 滤波） |
| | `_srs_from_covariance()` | 协方差特征分解 |
| | `_pmi_dft_codebook_search()` | DFT 码本 PMI 搜索 |
| | `_compute_rsrp_srs()` | 逐天线 RSRP |
| | `_compute_rsrp_cb()` | 波束域 RSRP |
| | `compute_interference_features()` | 干扰上下文（DoA/协方差） |
| | `sample_to_features()` | 单样本完整流水线 |
| | `batch_samples_to_features()` | 批量处理迭代器 |
| `features/extractor.py` | `FeatureExtractor` | Token 嵌入（归一化+门控+投影） |
| `features/normalizer.py` | `ProtocolNormalizer` | 3GPP 协议感知归一化 |
| `core/protocol_spec.py` | `PROTOCOL_SPEC` | 3GPP 值域规格表 |
| `features/losses.py` | `calculate_reconstruction_loss()` | MAE 重建损失 |
| | `contrastive_loss_fn()` | SimCLR 对比损失 |
