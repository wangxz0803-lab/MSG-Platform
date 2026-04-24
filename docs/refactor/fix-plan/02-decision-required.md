# 需用户决策的问题

> 以下任务需要你做出决策后才能确定修复方案。

---

## 决策 D-001 — F-001: quadriga_real DL 支持策略

### 背景
`quadriga_real` 目前只走 UL SRS 管线。MATLAB 仓库里存在 `dl_csirs_pipeline.m`，但 `main_multi.m` 从未调用它。Python 包装层也不解析 `link` / `channel_est_mode`。

### 可选方案

**方案 A: 完整 MATLAB DL 管线接入**
- 在 `main_multi.m` 中接入 `dl_csirs_pipeline()`，输出 DL 估计/理想信道
- Python 层同时消费 UL 和 DL 输出
- 影响：改动量大（MATLAB + Python），但物理上最正确
- 风险：`dl_csirs_pipeline.m` 未经充分测试，可能有自己的 bug

**方案 B: TDD 互易性方案（与 internal_sim/sionna_rt 对齐）**
- MATLAB 端保持现状只输出 UL
- Python 层接收 UL 理想/估计信道后，用 TDD 互易 `H_DL = conj(H_UL^T)` 生成 DL
- 当请求 `link=DL` 或 `link=BOTH` 时，在 Python 端做配对
- 影响：改动集中在 Python，与平台已有 `estimate_paired_channels()` 复用逻辑一致
- 风险：QuaDRiGa 的 UL 信道本身可能与互易假设不完全一致

**方案 C: 仅修 Python 透传，DL 标记但不做物理转换**
- 让 Python 层读取上层传入的 `link` / `channel_est_mode`，正确写入 ChannelSample
- 但不改变底层物理：DL 样本仍基于 UL SRS 管线产出
- 影响：改动最小，但"DL"标签名不副实
- 风险：下一轮 Codex 审查仍会指出方向不真实

### 我的建议
**倾向方案 B**。理由：
1. 与 `internal_sim` / `sionna_rt` 的 paired 架构完全一致
2. TDD 互易是 3GPP 标准行为，物理上站得住
3. 已有成熟的 `estimate_paired_channels()` 函数可复用
4. 改动量可控，风险低
5. 方案 A 可作为后续增强，但不是本轮修复范围

---

## 决策 D-002 — F-004: sionna_rt 干扰不可观测的根因判断

### 背景
干扰等级从 baseline 到 high，SNR/SIR/SINR 全部钉在 49.9 dB。49.9 是代码中 `p_intf <= 1e-30` 时的 fallback 值，说明干扰功率在估计链路中实质为零。

### 可能的根因链
1. **RT 干扰信道本身极弱**：验证脚本在 `(30, 0, 1.5)` 处、3 cell、ISD=300m 的 RT 场景做实验。干扰基站距离远，RT 信道增益极小。
2. **归一化导致干扰被淹没**：`_generate_one_sample()` 将 serving channel 归一化到单位功率，干扰信道按相同因子缩放后变得极小。但 `estimate_channel_with_interference()` 接收的 `h_interferers` 是归一化后的值。
3. **SIR 计算口径问题**：ChannelSample 中报告的 `sir_dB` 来自 `_generate_one_sample()` 中基于 `rx_power_dbm` 计算的链路级 SIR，与估计链路内部的 pilot 级 SIR 可能不一致。

### 可选修复方向

**方案 A: 修复干扰链路使干扰真实可观测**
- 检查并修复 `h_interferers` 传入 `estimate_channel_with_interference()` 前的归一化问题
- 确保干扰功率在估计链路中物理合理
- 统一 SIR 计算口径（使用估计链路内部的实际干扰功率）

**方案 B: 确认这是物理现实，调整验证预期**
- 如果 RT 场景下干扰确实极弱（近场 serving 远强于远场 interferer），这可能是正确的物理行为
- 但需要用更密集的部署（更小 ISD）或更近的干扰源来验证干扰确实可被注入
- 停下来告诉 Codex "该场景下干扰确实不可观测是正确的"

### 我的建议
**先执行方案 A 的调查**。即使物理上干扰确实弱，SIR 钉在 49.9 dB 的 fallback 值是不合理的——应该报告真实的（即使很高的）SIR。调查重点：
1. 确认 `h_interferers` 传到估计函数时是否为零/None
2. 确认归一化后干扰信道的实际功率级
3. 如果功率确实极小但非零，让 SIR 报告真实值而不是 49.9 fallback

---

## 决策 D-003 — F-006: sionna_rt 动态 TDL 回退策略

### 背景
动态序列中某些 snapshot 的 RT 求解失败后自动回退到 TDL，导致同一序列混入两种物理模型。

### 可选方案

**方案 A: 添加 `allow_tdl_fallback=False` 配置选项**
- 默认为 True（保持现有行为）
- 设为 False 时，RT 失败直接 raise 而不回退
- 调用方可选择跳过该 snapshot 或终止

**方案 B: 仅元数据标记 + 输出隔离**
- 保持 TDL 回退，但在 ChannelSample.meta 中标记 `sionna_rt_used=False` + `fallback_reason`
- 让下游消费方可识别和过滤

**方案 C: 两者都做（推荐）**
- 既有配置选项控制是否允许回退
- 也在发生回退时做元数据标记

### 我的建议
**倾向方案 C**。代码已经有 `meta.sionna_rt_used` 字段，只需确保在回退时正确设为 False 并附加原因。添加配置选项让实验者可以选择"宁可报错也不混入 TDL"。

---

## 决策 D-004 — F-008: internal_sim 移动相关性是否本轮修复

### 背景
3.6 km/h 的相邻相关仅 0.135/0.167，物理上偏低。但根因是 TDL 模型按 snapshot 独立采样，要真实修复需要引入时间域相关 TDL（如 Jakes 模型或 Clarke 频谱采样），改动量大。

### 可选方案

**方案 A: 本轮修复——引入时间域连续 TDL**
- 在 `_generate_one_sample()` 中传入前一个 snapshot 的信道状态
- 用 Doppler 频率控制相邻 snapshot 的相关度
- 改动量大，可能引入新问题

**方案 B: 本轮不修——标记为已知限制，另起 issue**
- 在文档中明确标注 `internal_sim` 的 TDL 模型不支持时间域连续快衰落
- 记录为架构限制，后续作为独立增强项

### 我的建议
**倾向方案 B**。这是物理建模层面的增强（不是 bug），改动面大且有引入新问题的风险。本轮聚焦阻塞级和严重 bug 修复更合理。

---

## 决策汇总

| 编号 | 关联任务 | 决策要点 | 我的建议 |
|---|---|---|---|
| D-001 | F-001 | quadriga_real DL 支持策略 | 方案 B (TDD 互易) |
| D-002 | F-004 | 干扰不可观测根因 | 先调查后判断，修复 SIR 计算 |
| D-003 | F-006 | TDL 回退策略 | 方案 C (配置 + 元数据) |
| D-004 | F-008 | 移动相关性是否本轮修 | 方案 B (延后) |
