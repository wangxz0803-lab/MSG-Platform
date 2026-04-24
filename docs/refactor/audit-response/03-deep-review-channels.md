# 03 - 重点区块深度复核：多小区信道真实性

复核时间：2026-04-23

---

## 1. 多小区 QuaDRiGa 信道真实性

### 1.1 `quadriga_multi`（旧 .mat 路径）

**Codex 结论**：最可信的多小区耦合证据。
**复核结论**：**同意 Codex 判定。**

证据链：
1. MATLAB 端（`gen_channel_multicell.m:39-88`）在单个 `qd_layout` 中建立多 TX、单 RX，调用 `l.get_channels()` 一次性生成所有小区的 CFR → 散射环境共享
2. Python 端（`quadriga_multi.py:264-326`）正确从 `Hf_multi[i_ue]` 提取 serving + interferers，transpose 到 `ChannelSample` 合约 shape
3. 旧 `.mat` 文件确实包含 `Hf_multi` 键，审查样本 `h_interferers` shape 正确

**注意点**：
- 如果 `.mat` 文件缺少 `Hf_multi` 键，`quadriga_multi.py:264` 会直接 KeyError，无降级路径 — 这是合理的 fail-fast 行为
- 当前 MATLAB `main_multi.m` 产物**不含** `Hf_multi`，因此 `QuadrigaMultiSource` 无法消费当前 MATLAB 产物。只能消费旧/参考 `.mat` 文件

### 1.2 `quadriga_real`（当前 MATLAB 路径）— 干扰感知信道估计

**Codex 结论**：阻塞级 — 多小区 CIR 被丢弃。
**复核结论**：**已修复 — Codex 审查时发现正确，此问题已根据审查反馈实施干扰感知 LS 估计方案解决。**

完整调用链验证：
1. `QuadrigaRealSource._generate_shard()` (`:166-198`) 调用 MATLAB `main_multi(config_path)`
2. MATLAB `main_multi.m` 内部调用 `gen_channel_multicell.m` 生成 `Hf_per_cell`（多小区 CFR，共享散射环境）
3. **关键步骤（Codex 未覆盖）**：MATLAB `ul_srs_pipeline.m` 使用 `Hf_per_cell` 构造干扰感知接收信号：
   - 信号模型：`Y = H_serving * X_s + Σ(H_k * X_k) + noise`（ZC SRS 序列）
   - LS 估计：`H_hat = Y * conj(X_s) / |X_s|^2`
   - 输出 `Hf_serving_est` = LS 估计结果（包含干扰残留）
   - 输出 `Hf_serving_ideal` = ground truth（无干扰）
4. 落盘保存 `Hf_serving_est`, `Hf_serving_ideal`, `snr_dB/sir_dB/sinr_dB`, `rsrp_per_cell`, `meta`
5. Python `_iter_from_mat()` (`:209-298`) 读取上述变量，`:273` 写 `h_interferers=None`
6. Python `_interference_estimation.py` 中 `estimate_channel_with_interference()` 是等效的 Python 实现

**正确的解读**：
- `Hf_serving_est` ≠ clean serving channel。它是在多小区干扰存在下的 LS 估计，干扰残留（ZC 互相关）结构性地嵌入其中
- `Hf_serving_est` vs `Hf_serving_ideal` 的差异本身编码了干扰强度和结构
- SIR/SINR 从 SRS 实际接收功率比计算，是物理量而非后验统计
- `h_interferers=None` 是有意设计：干扰已烘焙进估计信道，无需单独存储 raw per-cell CIR
- 这是一种合理的架构选择：对 ML 训练而言，(est, ideal) pair 提供了足够的干扰信息

**残留限制**（非阻塞）：
- 如果下游需要 raw per-cell CIR（如 channel charting、多小区 CIR 可视化、per-cell 波束成形），当前架构不直接支持
- 建议在 platform guide 中明确说明此设计决策

### 1.3 QuaDRiGa 小区间几何一致性

**对 `quadriga_multi`（旧路径）**：
- MATLAB `gen_channel_multicell.m` 在同一个 `qd_layout` 中放置所有 BS 和 UE
- 一次 `l.get_channels()` 调用确保所有小区的信道共享同一几何环境
- 这是 QuaDRiGa 模拟器本身的保证：同一 layout 内的信道共享散射环境
- **结论：几何一致性有保证**

**对 `quadriga_real`（当前路径）**：
- MATLAB 计算过程中几何一致性是保证的（同一 `qd_layout`）
- 但由于 CIR 被丢弃，无法从平台输出验证这一点
- **结论：几何一致性在计算中存在但在输出中不可证明**

---

## 2. 多小区 Sionna RT 信道真实性

### 2.1 真实 RT 路径（代码结构分析）

**Codex 结论**：代码结构上使用共享 scene，但当前样本不是 RT。
**复核结论**：**同意，并补充细节。**

真实 RT 路径的完整调用链：
1. `_load_scene()` (`:641-681`)：加载一个 scene，设置全局 tx_array 和 rx_array
2. `_add_transmitters()` (`:683-697`)：在同一 scene 中添加多个 BS transmitter
3. `_compute_channels_sionna()` (`:918-980`)：
   - Scene cache 和 solver cache 在首次调用时初始化 (`:934-937`)
   - 每个 UE 作为 `rt.Receiver` 加入同一 scene (`:941-942`)
   - **单次 `solver()` 调用** (`:945-955`) 对整个 scene 求解所有路径
   - 从统一的 CFR 输出中按 TX 维度提取各小区信道 (`:963-974`)

**这是正确的多小区 RT 实现**：
- 同一 scene → 共享建筑/材料/散射体
- 同一 PathSolver 调用 → 多路径计算考虑了所有 TX-RX 组合
- 按 TX 索引提取 → 几何一致性天然保证

**但当前无法动态验证**：本机无 Sionna，所有样本走 TDL fallback。

### 2.2 TDL fallback 路径深度分析

**Codex 结论**：逐 cell 独立 TDL，不是共享散射环境。
**复核结论**：**基本同意，但需要补充一个细节 — Codex 未提到 LOS 角度共享。**

`_compute_channels_tdl()` (`:985-1044`) 的实际行为：

1. **共享的部分**：
   - 大尺度路损：基于真实 BS-UE 几何距离计算 (`:1016-1018`)
   - 相对功率缩放：serving cell unit-power, interferers 按路损比缩放 (`:1036-1042`)
   - RNG 状态：使用同一个 `rng` 对象顺序推进 — 不是独立随机

2. **不共享的部分**：
   - TDL tap 系数：每个 cell 独立生成的 Rayleigh/Rice 衰落
   - 散射体位置：TDL 模型没有散射体概念
   - 角度扩展：TDL 不建模角域

**Codex 未注意到的细节**：
- `sionna_rt` 的 TDL fallback 与 `internal_sim` 的 TDL 生成调用签名不同
- `internal_sim._generate_one_sample()` 在 TDL 生成时传入了 `los_aod_rad`, `los_aoa_rad`, `spatial_corr_rho` — 这些在所有 cell 间是共享的
- `sionna_rt._compute_channels_tdl()` 没有传入这些参数 — 纯 TDL 无角度信息
- 所以 `internal_sim` 实际上有更多的跨 cell 关联性（共享 LOS 角度和空间相关性），而 `sionna_rt` fallback 更加独立

### 2.3 材料、场景 mesh、max_depth 参数

**代码证据**：
- `_load_scene()` 加载标准场景（munich/etoile/OSM）— 材料和 mesh 由场景文件决定
- `_rt_max_depth` 和 `_rt_samples_per_src` 从配置读取 (validate_config 中)
- PathSolver 调用时 (`:945-955`) 启用了全部路径类型：LOS、specular_reflection、diffuse_reflection、refraction、diffraction
- **但当前无法验证这些参数是否生效**，因为 Sionna 未安装

**结论**：代码层面参数设置合理，但无运行时证据。

### 2.4 是否存在"降级到简化模型"的隐藏分支

**验证结果**：存在，且 Codex 已发现。

1. **Source 级降级** (`:431-444`)：Sionna 不可用 → 整个 source 使用 TDL
2. **样本级降级** (`:1098-1184`)：
   - RT 生成失败（异常）→ 重试最多 5 次 → 全部失败则 TDL
   - RT 生成但信道功率过低 → 重试最多 5 次 → 全部低功率则 TDL
3. **无其他隐藏分支** — 除上述两个降级路径外，没有其他 fallback 逻辑

---

## 3. 多小区耦合正确性

### 3.1 同一用户在不同小区看到的信道 — 几何是否一致

| Source | 几何共享 | 路损共享 | 小尺度衰落共享 | 散射体共享 |
|--------|---------|---------|--------------|-----------|
| quadriga_multi (旧 .mat) | ✅ 同一 layout | ✅ 真实距离 | ✅ 同一 get_channels() | ✅ QuaDRiGa GSCM |
| quadriga_real (当前) | ✅ 计算中 | ✅ 计算中 | ✅ 计算中 | ✅ 计算中，干扰烘焙进 h_serving_est（LS 估计） |
| sionna_rt (真实 RT) | ✅ 同一 scene | ✅ 射线追踪 | ✅ 同一 solver 调用 | ✅ 场景几何 |
| sionna_rt (TDL fallback) | ✅ 真实 UE 位置 | ✅ 真实距离 | ❌ 独立 TDL | ❌ 无散射体 |
| internal_sim | ✅ 真实 UE 位置 | ✅ 真实距离 | ❌ 独立 TDL（共享 LOS 角度） | ❌ 无散射体 |

### 3.2 是否存在"多小区 = 多个独立单小区拼接"的退化实现

**结论：TDL 路径（sionna_rt fallback 和 internal_sim）确实接近"独立单小区拼接"，但不完全等价。**

区别在于：
1. 所有 cell 对同一个 UE 位置计算路损 → 路损比关系是正确的
2. Serving cell 选择基于接收功率最大 → 选择逻辑正确
3. 干扰功率相对于 serving 的缩放是物理合理的
4. 但小尺度衰落确实是独立的 — 没有跨 cell 的散射体一致性

**这对不同下游任务的影响不同**：
- 如果下游只关心 serving channel + SIR/SINR 统计 → TDL 拼接可能足够
- 如果下游需要"同一散射体在不同 cell 的不同角度" → TDL 拼接不够
- 如果下游是 channel charting / 位置指纹 → 需要真实几何一致性

### 3.3 Codex 报告的拓扑 bug（num_sites=3 → 7 cells）验证

**独立验证结果**：

`internal_sim`:
- `_sites_to_rings(3)` → 计算 ring=0: total=1<3, ring=1: total=7>=3 → 返回 1
- `make_hex_grid(num_rings=1)` 生成 7 sites（中心 + 6 个相邻）
- `_build_sites()` 不裁剪 → K=7
- **确认问题存在**

`sionna_rt`:
- 同样的 `_sites_to_rings()` 和 `make_hex_grid()` 生成 7 sites
- 但 `_generate_one_sample()` 中 `sites = sites[:self.num_cells]` (`:1080`) 裁剪到配置的 num_cells
- 如果配置 `num_cells=3`，则实际使用 3 个 cell
- **sionna_rt 不受此 bug 影响**

**根因**：`internal_sim._build_sites()` 遗漏了裁剪步骤。修复方法是添加 `sites = sites[:self.num_sites * self.sectors_per_site]`。

---

## 4. 总结

| 审查点 | Codex 结论 | 复核结论 | 差异 |
|--------|-----------|---------|------|
| quadriga_multi 多小区真实性 | 可信 | **同意** | 无 |
| quadriga_real CIR "丢失" | 阻塞 | **已根据审查反馈修复** | 实施干扰感知 LS 估计方案，干扰烘焙进 h_serving_est |
| sionna_rt 真实 RT 共享 scene | 代码结构可信 | **同意** | 补充了 PathSolver 单次调用的证据 |
| sionna_rt TDL 不共享散射 | 确认 | **同意，但补充 LOS 角度不共享（与 internal_sim 不同）** | sionna_rt TDL 比 internal_sim TDL 更"独立" |
| internal_sim 拼接性质 | 共享几何/路损 | **同意，补充共享 LOS 角度** | internal_sim 有更多跨 cell 关联 |
| num_sites→cells 拓扑 bug | internal_sim 有，sionna_rt 有裁剪 | **同意** | 无 |
| 隐藏降级分支 | 存在 source 级和样本级 | **同意** | 无遗漏 |
