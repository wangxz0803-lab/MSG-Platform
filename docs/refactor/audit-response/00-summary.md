# 00 - 执行摘要

审查复核时间：2026-04-23
审查对象：D:\MSG平台_cc（新重构工程）
审查方报告：D:\MSG平台_codex\（Codex 独立审查）

---

## 1. Codex 报告整体质量

| 指标 | 数值 |
|------|------|
| 提出问题总数 | 30 条 |
| 完全确认 (Confirmed) | 20 条 (67%) |
| 部分成立 (Partially Valid) | 9 条 (30%) |
| 误判 (False Positive) | **0 条 (0%)** |
| 需决策 (Need Decision) | 1 条 (3%) |
| 严重程度需下调 | 6 条 |

**评价**：Codex 的审查质量很高。30 条问题全部真实存在，零误判。代码行号引用准确，动态验证方法论合理。其中 C-001（quadriga_real CIR 丢失）已根据审查反馈实施干扰感知 LS 估计方案修复。另有 14 条问题被自检发现而 Codex 未覆盖，主要集中在内部实现细节（SSB 假干扰、NaN 传播、静默异常吞掉等）— 这些需要实现者视角才能发现，属于正常的审查盲区。

---

## 2. 阻塞级问题

合并去重后共 **2 个 P0 阻塞项**：

> **2026-04-23 更正**：M-001（quadriga_real CIR）从 P0 移除。该问题已根据审查反馈修复 — 实施了干扰感知 LS 信道估计方案（`ul_srs_pipeline.m` / `_interference_estimation.py`），干扰烘焙进 `h_serving_est`。降级为 P3（残留限制，文档记录）。

### ~~P0-1（已修复）：`quadriga_real` 多小区 CIR~~ → P3
已实施干扰感知 LS 估计方案解决。

### P0-1：`sionna_rt` 静默 fallback 且标记不分 (M-002)
Sionna 不可用时自动降级 TDL，但样本仍标记 `source="sionna_rt"`，`run_meta.json` 固定写 `fallback=false`。下游无法区分真 RT 和 TDL 样本。
**修复方向**：新增 source 枚举值或 `physical_model_status` 字段。约 0.5-1 天。

### P0-2：参数边界不校验 (M-009)
`bandwidth=0`, `num_bs_ant=0`, `carrier_freq<0` 等不物理配置可生成样本或崩溃。
**修复方向**：`validate_config()` 添加正值校验。约 2-3 小时。

---

## 3. 最需要立即决策的事情

### ~~决策 1（已修复）~~：quadriga_real CIR
已根据审查反馈实施干扰感知 LS 估计方案，问题已解决。

### 决策 1（原决策 2）：sionna_rt fallback 的标记方案
- **方案 A**（推荐）：新增 `SourceType = "sionna_rt_tdl_fallback"`，简单直接
- **方案 B**：保持 `source="sionna_rt"`，增加 `physical_model_status` 字段

### 决策 2（原决策 3）：`sample_id` 统一类型
- contract 用 UUID 字符串，manifest 用 int32，ORM 用字符串
- 建议统一为 UUID 字符串（已有大量样本使用）

### 决策 3（原决策 4）：internal_sim 拓扑裁剪方案
- **方案 A**：允许任意 `num_sites` 然后裁剪到请求数量
- **方案 B**：限制 `num_sites` 只接受 hex ring 精确值（1/7/19/37）

---

## 4. 多小区信道真实性结论

| Source | 真实性 | 详细结论 |
|--------|--------|---------|
| **QuaDRiGa multi (旧 .mat)** | ✅ 可信 | 单一 `qd_layout` + 单次 `get_channels()` = 共享散射环境。`h_interferers` 正确保留。目前最可信的多小区数据。 |
| **QuaDRiGa real (当前)** | ✅ 可信（干扰感知估计） | MATLAB 计算过程正确（共享散射）。通过 `ul_srs_pipeline.m` LS 估计，干扰烘焙进 `h_serving_est`。(est, ideal) pair 编码了干扰信息。 |
| **Sionna RT (真实路径)** | ⚠️ 代码可信 / 无运行证据 | 代码结构正确：共享 scene + 单次 PathSolver。但当前环境无 Sionna，所有样本走 TDL fallback。 |
| **Sionna RT (TDL fallback)** | ❌ 不是真 RT | 逐 cell 独立 TDL，不共享散射体。共享大尺度路损/几何，但不等价于多小区 RT 信道。 |
| **Internal sim** | ⚠️ 统计级 | 共享几何/路损/LOS 角度，独立 TDL 小尺度衰落。可作为结构化测试数据，不能作为真实多小区 CIR 验收依据。 |

---

## 5. 修复工时估算

| 阶段 | 内容 | 问题数 | 预计工时 |
|------|------|--------|---------|
| P0 立即 | 阻塞项 | 2 | 1-1.5 天 |
| P1 本轮必修 | 严重 + 数据一致性 | 11 | 3-4 天 |
| P2 可延后 | 改进项 | 8 | 2-3 天 |
| P3 不修但记录 | 已知限制 | 7 | — |
| **总计** | | **28** | **6-8.5 天** |

其中 P0 + P1 = 13 条，约 4-5.5 个工作日。P2 可在后续迭代中逐步推进。M-001 移至 P3（设计如此），M-008 移至 P2。

---

## 附录：产出文件清单

| 文件 | 内容 |
|------|------|
| `01-codex-findings.md` | Codex 30 条问题逐条提取 |
| `02-verdicts.md` | 逐条独立裁决（Confirmed / Partially Valid / False Positive） |
| `03-deep-review-channels.md` | 多小区信道深度复核（QuaDRiGa / Sionna RT / Internal Sim） |
| `04-self-audit.md` | 独立自检 16 条风险 |
| `05-merged-issues.md` | 统一问题清单 28 条（M-001 ~ M-031） |
| `06-action-plan.md` | 修复路线图（P0→P1→P2→P3，含节奏和决策点） |
| `00-summary.md` | 本文 — 一页纸执行摘要 |
