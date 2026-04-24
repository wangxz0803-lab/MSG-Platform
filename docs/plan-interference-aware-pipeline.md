# 修改计划：干扰感知的成对信道采集与处理流水线

> 日期：2026-04-23 | 状态：待审批

---

## 1. 目标

对于同一个 UE 点位，成对采集理想信道和含干扰估计信道：

```
同一 UE 点位
├─ UL 信道对 (h_ul_true, h_ul_est_interf)
│   → SRS tokens: 全带鲁棒权(srs1-4)、DFT 波束权(dft1-4)、
│     波束级 RSRP(rsrp_srs, rsrp_cb)、PDP(pdp_crop)、preSINR
│
└─ DL 信道对 (h_dl_true, h_dl_est_interf)
    → PMI tokens(pmi1-4)、CQI、RI
```

训练目标：模型从含干扰 token 恢复理想 token，使其具备降噪降干扰能力。

---

## 2. 现状问题

### 2.1 信道估计不含干扰

当前所有适配器的 `_estimate_channel` 构造接收信号时：

```python
# internal_sim.py:895-900 — 只有噪声，没有干扰
Y = h_serving_at_pilots * X_pilot + noise
ĥ_LS = Y / X_pilot = h_serving + noise/X_pilot
```

`h_interferers` 完全没参与 Y 的构造。

### 2.2 UL/DL 不区分 token 来源

当前 Bridge 对一个 `h_serving_est`（不区分 UL/DL）统一计算全部 16 个 token。实际应该：
- SRS/DFT/RSRP/PDP 从 **UL 信道**计算（基站侧协方差）
- PMI/CQI/RI 从 **DL 信道**计算（终端侧码本匹配）

### 2.3 没有成对数据结构

`ChannelSample` 只有一对 `(h_serving_true, h_serving_est)`，无法承载 UL+DL 成对信道。

---

## 3. 干扰物理模型

### 3.1 上行干扰（基站接收端）

多个 UE 同时发送 SRS，基站天线收到的信号是所有 UE 的叠加：

```
y_BS[t,f,b] = h_serving[t,f,b,:] · x_SRS_serving[f]
            + Σ_k h_intf_k[t,f,b,:] · x_SRS_k[f]     ← 其他 UE 的 SRS
            + n[t,f,b]

LS 估计：
ĥ_UL[t,f,b,:] = y_BS / x_SRS_serving
              = h_serving + Σ_k h_intf_k · (x_SRS_k / x_SRS_serving) + n/x_SRS_serving
                            ↑ 干扰残留（取决于 ZC 序列互相关）
```

干扰残留大小取决于：
- SRS ZC 根序列和循环移位的正交性
- 干扰 UE 的信道强度（SIR）
- 是否存在 SRS 资源碰撞（同 comb、同符号）

### 3.2 下行干扰（UE 接收端）

多个小区同时发送 CSI-RS/DMRS，UE 收到的信号是所有小区的叠加：

```
y_UE[t,f,u] = h_serving[t,f,:,u] · x_CSIRS_serving[f]
            + Σ_k h_intf_cell_k[t,f,:,u] · x_CSIRS_k[f]   ← 邻区 CSI-RS
            + n[t,f,u]

LS 估计：
ĥ_DL[t,f,:,u] = y_UE / x_CSIRS_serving
              = h_serving + Σ_k h_intf_cell_k · (x_CSIRS_k / x_CSIRS_serving) + n/x_CSIRS_serving
                            ↑ 邻区干扰残留（取决于 CSI-RS 序列互相关）
```

---

## 4. 修改计划

### Phase A：数据契约扩展

**文件**：`src/msg_embedding/data/contract.py`

**改动**：ChannelSample 新增字段以承载 UL+DL 成对数据

```python
class ChannelSample(BaseModel):
    # --- 现有字段保持不变（向后兼容） ---
    h_serving_true: NdArray       # [T, RB, BS, UE] 理想信道
    h_serving_est: NdArray        # [T, RB, BS, UE] 估计信道（将改为含干扰）
    h_interferers: NdArray | None
    ...

    # --- 新增：成对 UL/DL 信道 ---
    # 当 link_pairing="paired" 时，以下字段有效：
    link_pairing: Literal["single", "paired"] = "single"

    # UL 信道对（基站视角：[T, RB, BS_ant, UE_ant]）
    h_ul_true: NdArray | None = None        # UL 理想信道
    h_ul_est: NdArray | None = None         # UL 含干扰估计信道

    # DL 信道对（终端视角：[T, RB, BS_ant, UE_ant]）
    h_dl_true: NdArray | None = None        # DL 理想信道
    h_dl_est: NdArray | None = None         # DL 含干扰估计信道

    # 干扰场景参数（用于分析和复现）
    ul_sir_dB: float | None = None          # UL 信干比
    dl_sir_dB: float | None = None          # DL 信干比
    num_interfering_ues: int | None = None  # UL 干扰 UE 数量
```

**兼容策略**：`link_pairing="single"` 时走老路径（`h_serving_est`），`"paired"` 时 Bridge 使用新字段。

**验证器**：
- `link_pairing="paired"` 时 `h_ul_true/h_ul_est/h_dl_true/h_dl_est` 必须全部非空
- UL 信道形状 `[T, RB, BS, UE]` 一致
- DL 信道形状 `[T, RB, BS, UE]` 一致

---

### Phase B：干扰注入的信道估计

**新文件**：`src/msg_embedding/data/sources/_interference_estimation.py`

抽取公共的干扰注入逻辑，所有适配器共用。

```python
def estimate_channel_with_interference(
    h_serving_true: np.ndarray,          # [T, RB, BS, UE]
    h_interferers: np.ndarray | None,    # [K-1, T, RB, BS, UE]
    pilots_serving: np.ndarray,          # [RB] 服务信号导频
    pilots_interferers: list[np.ndarray] | None,  # 每个干扰源的导频 [RB]
    mode: str,                           # "ideal" | "ls_linear" | "ls_mmse"
    snr_db: float,
    rng: np.random.Generator,
    direction: Literal["UL", "DL"],
    *,
    pilot_rb_spacing: int = 1,
    pilot_sym_spacing: int = 4,
    sir_db: float | None = None,         # 可选：覆盖干扰功率缩放
) -> np.ndarray:
    """含干扰的信道估计。

    构造接收信号：
      UL: y_BS = h_serving · x_SRS_serving + Σ h_intf_k · x_SRS_k + noise
      DL: y_UE = h_serving · x_CSIRS_serving + Σ h_intf_cell_k · x_CSIRS_k + noise

    然后对 y 做 LS/MMSE 估计，返回 h_est（含干扰残留）。
    """
```

**UL 实现细节**：

```python
# 在 pilot 位置提取信道
h_at_pilots = h_serving_true[rs_time][:, rs_freq]  # [n_t, n_f, BS, UE]

# 服务信号：Y_serving = H · X
Y = h_at_pilots * X_serving_expanded

# 逐个干扰 UE 叠加
for k, h_intf_k in enumerate(h_interferers):
    h_intf_at_pilots = h_intf_k[rs_time][:, rs_freq]
    X_intf_k = pilots_interferers[k][rs_freq]
    Y += h_intf_at_pilots * X_intf_k_expanded     # 干扰 SRS 贡献

# 加噪声
Y += noise

# LS 估计（用服务导频解）
ĥ = LS_estimate(Y, X_serving)
# 结果自然包含：h_serving + 干扰残留 + 噪声
```

**DL 实现细节**：

```python
# 类似 UL，但角色互换：
# - 服务信号用该小区的 CSI-RS
# - 干扰来自邻区的 CSI-RS（不同 n_ID → 不同 Gold 序列）
for k, h_intf_cell_k in enumerate(h_interferers):
    X_csirs_k = generate_csirs(cell_id=neighbor_pci_k)
    Y += h_intf_at_pilots * X_csirs_k_expanded
```

---

### Phase C：适配器修改（5 个数据源）

每个适配器的 `_generate_one_sample` / `iter_samples` 修改为：

#### C1. internal_sim

**文件**：`src/msg_embedding/data/sources/internal_sim.py`

改动：
1. 当 `link="BOTH"` 或新增的 `paired=True` 时：
   - DL 信道估计：调用 `estimate_channel_with_interference(..., direction="DL")`，用邻区 CSI-RS 作为干扰导频
   - UL 信道：TDD 互易 `h_ul_true = conj(h_dl_true.T)`
   - UL 信道估计：调用 `estimate_channel_with_interference(..., direction="UL")`，生成多个干扰 UE 的 SRS（不同 ZC 根 / 循环移位）
2. 干扰 UE 的 SRS 导频生成：
   ```python
   # 服务 UE：ZC root u = cell_id % (Nzc-1) + 1
   # 干扰 UE k：不同 cyclic shift 或不同 root
   for k in range(num_intf_ues):
       pilots_intf.append(_generate_pilots_srs(RB, cell_id, cyclic_shift=k+1))
   ```
3. 构造 `ChannelSample(link_pairing="paired", h_ul_true=..., h_ul_est=..., h_dl_true=..., h_dl_est=...)`

#### C2. sionna_rt

**文件**：`src/msg_embedding/data/sources/sionna_rt.py`

改动同 internal_sim，结构完全一致（两个适配器的信道估计代码已高度对称）。

#### C3. quadriga_multi

**文件**：`src/msg_embedding/data/sources/quadriga_multi.py`

改动：
1. 此源从 MATLAB `.mat` 文件加载多小区信道，已有 `h_interferers`
2. 增加对干扰注入估计的调用
3. UL 信道通过互易生成
4. 注意：MATLAB 文件中信道形状是 `[K, BsAnt, UeAnt, RB, T]`，需要在 transpose 后再注入

#### C4. quadriga_real

**文件**：`src/msg_embedding/data/sources/quadriga_real.py`

改动：
1. 此源当前 `h_interferers=None`（单小区数据）
2. 方案 A：如果 `.mat` 文件中有多小区数据 → 提取干扰信道
3. 方案 B：如果确实没有干扰信道 → `link_pairing="single"`，走兼容路径
4. 根据实际 MATLAB 文件结构决定

#### C5. field（占位符）

**文件**：`src/msg_embedding/data/sources/field.py`

改动：接口预留，实际实现推迟到 v3.1。

---

### Phase D：Bridge 改造 — 分离 UL/DL token 计算

**文件**：`src/msg_embedding/data/bridge.py`

#### D1. 新增 `_build_feat_dict_paired` 函数

```python
def _build_feat_dict_paired(
    sample: ChannelSample,
    use_legacy_pmi: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """成对模式的特征构建。

    从 UL 信道计算 SRS 类 token，从 DL 信道计算 PMI 类 token。
    返回两套特征：(feat_from_est, feat_from_true)
    """
```

#### D2. Token 来源分离

| Token 槽位 | 字段 | 信道来源 | 估计/理想 | 计算方法 |
|:---:|--------|:---:|:---:|------|
| 0 | pdp_crop | UL est | 含干扰 | IFFT(h_ul_est) |
| 1-4 | srs1-4 | UL est | 含干扰 | eigh(R_hh(h_ul_est)) |
| 5-8 | pmi1-4 | DL est | 含干扰 | DFT_codebook(h_dl_est) 或 legacy |
| 9-12 | dft1-4 | UL est | 含干扰 | DFT_beam(h_ul_est) |
| 13 | rsrp_srs | UL est | 含干扰 | mean(\|h_ul_est\|²) per BS ant |
| 14 | rsrp_cb | UL est | 含干扰 | beam_power(h_ul_est) |
| 15 | cell_rsrp | SSB | — | ssb_rsrp_dBm（不变） |
| gate | srs_sinr | UL | — | from sinr_dB |
| gate | srs_cb_sinr | DL | — | from dl_sinr_dB |
| gate | cqi | DL est | 含干扰 | _cqi_from_sinr(dl_sinr) |
| gate | srs_w1-4 | UL est | 含干扰 | eigvals(R_hh(h_ul_est)) |

#### D3. Ground-truth 特征生成

同样的计算逻辑，但输入用 `h_ul_true` / `h_dl_true`：

```python
feat_gt = _build_feat_dict_from_channels(
    h_ul=sample.h_ul_true,     # 理想 UL
    h_dl=sample.h_dl_true,     # 理想 DL
    ...
)
feat_est = _build_feat_dict_from_channels(
    h_ul=sample.h_ul_est,      # 含干扰 UL
    h_dl=sample.h_dl_est,      # 含干扰 DL
    ...
)
```

训练时 `feat_est → tokens → MAE → recon → loss(recon, feat_gt)`。

#### D4. 新增 preSINR token

当前 `srs_sinr` 是标量门控。新增一个基于信道估计质量的 per-antenna preSINR：

```python
# preSINR[b] = |h_est[b]|² / (|h_est[b] - h_true[b]|² + eps)
# 训练时 h_true 可用；推理时用噪声方差估计替代
pre_sinr = compute_pre_sinr(h_ul_est, h_ul_true)  # [BS] dB
```

这个可以作为额外的门控信号或替换现有 srs_sinr。

---

### Phase E：FeatureExtractor 适配

**文件**：`src/msg_embedding/features/extractor.py`

改动较小：
1. `normalize_features` 和 `forward` 逻辑不变（输入仍然是 24 字段 feat_dict）
2. 新增一个 `paired` 模式开关：训练时同时产出 `(tokens_est, tokens_gt)` 两组 token
3. 可选：为 preSINR 新增一个门控分支

---

### Phase F：训练流程适配

**文件**：`src/msg_embedding/training/pretrain.py`、`src/msg_embedding/features/losses.py`

改动：
1. 数据加载识别 `link_pairing="paired"` 样本
2. `feat_est` 经 FeatureExtractor 得到 `tokens_est`（模型输入）
3. `feat_gt` 经 FeatureExtractor 得到 `tokens_gt`（训练目标）
4. 重建损失改为：`loss = MSE(MAE_recon(tokens_est), tokens_gt)`
   - 当前：`loss = MSE(recon, tokens_est)` （自重建）
   - 目标：`loss = MSE(recon, tokens_gt)` （去干扰重建）
5. 对比损失可以用 `(z_est, z_gt)` 作为正样本对

---

### Phase G：测试

| 测试类型 | 覆盖内容 |
|----------|----------|
| 单元测试 | `estimate_channel_with_interference` 的 UL/DL 两个方向 |
| 单元测试 | 干扰导频生成（不同 ZC root 的 SRS、不同 n_ID 的 CSI-RS） |
| 单元测试 | `_build_feat_dict_paired` 输出形状和字段完整性 |
| 数值一致性 | 干扰为零时 `h_est_interf` 退化为 `h_est_noise_only`（现有行为） |
| 数值一致性 | `link_pairing="single"` 时全部走老路径，结果不变 |
| 集成测试 | 成对样本 → Bridge → FeatureExtractor → MAE → loss 端到端 |
| Golden vector | 固定种子，h_true/h_est_interf 的 token 差异在预期范围内 |

---

## 5. 执行顺序与依赖

```
Phase A (contract)
    ↓
Phase B (干扰估计公共模块)  ← 不依赖 contract 改动，可并行开发
    ↓
Phase C (适配器改造)        ← 依赖 A + B
    ↓
Phase D (Bridge 改造)       ← 依赖 A
    ↓
Phase E (FeatureExtractor)  ← 依赖 D
    ↓
Phase F (训练流程)          ← 依赖 D + E
    ↓
Phase G (测试)              ← 贯穿全程，每个 Phase 完成后立即验证
```

建议先做 A → B → D（数据契约 + 干扰估计 + Bridge），形成可验证的中间产物，再铺开 C（适配器）和 F（训练）。

---

## 6. 风险与缓解

| 风险 | 级别 | 缓解 |
|------|:---:|------|
| ZC 序列互相关精度影响训练效果 | 中 | 提供 `interference_correlation` 参数控制：完美正交 / 部分相关 / 完全碰撞三档 |
| quadriga_real 没有多小区数据 | 低 | 走 `link_pairing="single"` 兼容路径，不阻塞其他源 |
| 成对数据使存储翻倍 | 低 | 同一个 h_serving_true 共享，只多存 h_est 的差异部分 |
| 向后兼容：老 checkpoint / 老 shard | 高 | `link_pairing` 默认 `"single"`；Bridge 检测到 `h_ul_true=None` 自动回退老路径 |
| preSINR 推理时无 h_true | 中 | 推理时用噪声方差估计 `σ²_n` 替代；训练时两种模式都练 |

---

## 7. 验收标准

1. **干扰可见**：对同一点位，`h_est`（含干扰）与 `h_true`（理想）的 NMSE 随 SIR 降低而增大
2. **token 差异可测**：`feat_est` 和 `feat_gt` 的 token-wise MSE 与 SIR 成反比
3. **模型有效**：训练后的 MAE 从含干扰 token 重建的结果比直接用含干扰 token 更接近理想 token（NMSE 降低 ≥ 3dB）
4. **全源适配**：internal_sim、sionna_rt、quadriga_multi 均产出 `link_pairing="paired"` 样本
5. **向后兼容**：`link_pairing="single"` 样本训练结果与改动前无差异
