# 06 - 修复路线图

制定时间：2026-04-23
基于：05-merged-issues.md

---

## 第一阶段：P0 阻塞项（立即修复）

**预计工时：1-1.5 天**
**目标：消除所有能让平台输出不可信或不安全数据的问题**

> **2026-04-23 更正**：M-001 从 P0 移除。该问题已根据审查反馈修复 — 实施了干扰感知 LS 信道估计方案（`ul_srs_pipeline.m` + `_interference_estimation.py`），干扰烘焙进 `h_serving_est`。P0 从 3 项缩减为 2 项。

### 修复顺序

```
M-009 参数校验 (2-3h)
  ↓ （解除基础安全风险后再动信道层）
M-002 sionna_rt fallback 标记 (4-6h)
```

### M-009：参数边界校验（第 1 天上午）

修复范围：
1. `internal_sim.py validate_config()`：添加 `bandwidth_hz > 0`, `carrier_freq_hz > 0`, `num_bs_ant >= 1`, `num_ue_ant >= 1` 检查
2. `sionna_rt.py validate_config()`：同上
3. `contract.py _check_shape_consistency()`：添加每个维度 > 0 的检查
4. `internal_sim.py _noise_power_dbm()`：`bandwidth_hz <= 0` 时 raise 而非用 0

测试要点：
- 确认零带宽/零天线/负频率/零距离被拒绝
- 确认正常配置不受影响

### M-002：sionna_rt fallback 标记（第 1 天下午）

修复范围：
1. `contract.py SourceType`：新增 `"sionna_rt_tdl_fallback"` 或在 meta 中添加 `physical_model_status` 强制字段
2. `sionna_rt.py _generate_one_sample()`：fallback 样本 `source` 改为区分值
3. `run_simulate.py`：`run_meta.json` 的 `fallback` 字段根据 source 实际状态动态设置
4. 可选：增加 `strict_rt=True` 配置项，Sionna 不可用时直接 fail

**需要人工决策**：方案 A（新增 SourceType 枚举值）还是方案 B（保持 source 不变 + 增加 `physical_model_status` 一等字段）？

- 方案 A 优点：下游只看 source 就能区分，最简单
- 方案 A 缺点：破坏现有按 source 做的过滤/分组
- 方案 B 优点：兼容性好
- 方案 B 缺点：下游必须同时看两个字段
- **建议方案 A**，因为当前数据量小，迁移成本低

---

## 第二阶段：P1 本轮必修（按依赖排序）

**预计工时：3-4 天**
**目标：修复所有严重问题和数据一致性问题**

### 修复顺序

```
M-005 pathloss 符号 (15min)
  ↓
M-003 internal_sim 拓扑裁剪 (1h)
  ↓
M-004 SSB 假干扰 + 异常处理 (2-3h)
  ↓
M-010 NaN 传播检查 (1-2h)
  ↓ （以上是信道层面修复，以下是数据管理层面）
M-008 h_interferers 一致性校验 (1-2h)
  ↓
M-006 元数据 provenance (4-6h)
  ↓
M-007 Schema 统一 (8-12h) → 需要人工决策
  ↓
M-011 采集入库 (4-6h)
  ↓
M-012 Channels Explorer 字段 (30min)
  ↓
M-014 注释更新 (30min)
  ↓
M-018 批量处理错误日志 (30min)
  ↓
M-019 MATLAB 诊断 (30min)
```

### 快速修复（半天内全部完成）

- **M-005**：`sionna_rt.py:1387` 改 `self.tx_power_dbm - rx_power_dbm[serving_idx]`
- **M-003**：`internal_sim._build_sites()` 添加 `sites = sites[:self.num_sites * self.sectors_per_site]`
- **M-012**：`channels.py:178` 改为读 `h_serving_true`，fallback 到 `channel_ideal`
- **M-014**：更新 `main_multi.m` 头注释和 `run_quadriga_real.py` 注释
- **M-018**：`post_quadriga_pipeline.py:372` 改为 `except Exception as e: logger.warning(...)`
- **M-019**：`quadriga_real.py:196` 记录 `result.stderr` 和异常

### M-004 + M-010（1 天）

- SSB 测量：`h_interferers=None` 时跳过多小区 SSB，只输出 serving cell RSRP
- `quadriga_multi.py:468` 改为 `except Exception as e: logger.warning("SSB failed: %s", e)`
- 所有 `.tolist()` 前加 `assert np.all(np.isfinite(arr))` 或 `arr = np.nan_to_num(arr, nan=-999.0)`

### M-008（1-2 小时）→ 移至 P2

> M-008 严重程度下调。`quadriga_real` 的 `h_interferers=None` 是有意设计（干扰感知估计），校验逻辑需排除此场景。移至 P2。

- `contract.py` 增加 `model_validator`：如果 `meta.get("num_cells", 1) > 1` 且 `h_interferers is None` 且 source 不是干扰感知估计类型 且 `meta.get("serving_only") != True`，则 raise

### M-006 + M-007（1.5-2 天）

**需要人工决策的关键点**：

1. `sample_id` 类型统一：UUID 字符串 vs 自增 int？
   - 建议 UUID 字符串（已有大量样本用 UUID）
2. `MANIFEST_SCHEMA` 是否完全重写还是渐进式适配？
   - 建议重写，保持与 `to_parquet_row()` 一致
3. `schema_version` 初始值？
   - 建议 `"1.0.0"`

### M-011（4-6 小时）

- `run_simulate.py` 完成后自动调用 `Manifest.append()` 将样本注册到 manifest
- 或增加 `--register` flag 控制是否注册

---

## 第三阶段：P2 可延后

**预计工时：2-3 天**
**触发条件：P0+P1 全部完成后**

| 编号 | 内容 | 预计工时 |
|------|------|---------|
| M-013 | 删除同步物理文件 | 2-3h |
| M-015 | 更新 platform guide 文档 | 1h |
| M-016 | 前端按 source 差异化 link 选项 | 1-2h |
| M-020 | UE 位置回退添加日志 | 30min |
| M-021 | 支持 canonical export | 1-2h |
| M-022 | 修正 samples_per_sec 计时 | 15min |
| M-023 | 扩展 API 筛选字段 | 4h |
| M-024 | Legacy PMI 封装 + NaN 默认值 | 4-6h |
| M-025 | to_dict 改 np.save + scene close | 4h |

---

## 第四阶段：P3 不修但记录

| 编号 | 理由 |
|------|------|
| M-026 | TDL 模型固有局限，需换模型才能解决 |
| M-027 | ASA/ASD/PDP 是导出量，可后处理 |
| M-028 | 代码已诚实标注 simplified |
| M-029 | 极端场景才触发，ROI 低 |
| M-030 | 架构层问题，随大重构解决 |
| M-031 | mock 仅影响开发者，风险极低 |

---

## 建议的修复节奏

```
Day 1 AM:  M-009 参数校验
Day 1 PM:  M-002 sionna_rt fallback 标记
Day 2 AM:  M-005 + M-003 + M-012 + M-014 + M-018 + M-019 (快速批量)
Day 2 PM:  M-004 + M-010 (SSB + NaN)
Day 3 AM:  M-006 (元数据 provenance)
Day 3-4:   M-007 (Schema 统一) → 需要人工决策
Day 5:     M-011 (采集入库)
Day 6+:    P2 项（含 M-008，按优先级逐步推进）
```

> **注**：M-001 已确认为设计如此（干扰感知 LS 估计），从 P0 移至 P3。M-008 从 P1 移至 P2。总工时减少约 1.5-2 天。

---

## 修复过程中需要我决策的关键点

1. ~~**M-001**：是否有权修改 `D:\MSG\matlab\main_multi.m`？~~ → **已修复**：已根据审查反馈实施干扰感知 LS 估计方案。
2. **M-002**：fallback 标记用新 SourceType 枚举还是新增 `physical_model_status` 字段？
3. **M-003**：拓扑裁剪方案 — 允许任意 `num_sites` 然后裁剪，还是限制为 hex ring 精确值？
4. **M-007**：`sample_id` 统一为 UUID 字符串还是 int？`MANIFEST_SCHEMA` 重写还是渐进适配？
5. **M-011**：采集后自动注册 manifest，还是保持手动/可选？
