# 修复任务清单

> 来源: Codex 第二轮审查 `D:\MSG平台_codex\round2\review\09-issues.md`
> 生成时间: 2026-04-24

---

## F-001 — quadriga_real 方向与估计模式未透传

| 字段 | 值 |
|---|---|
| 对应 Codex 编号 | R2-001 |
| 问题摘要 | 请求 DL/ideal 后，输出样本仍为 `link="UL"`, `channel_est_mode="ls_linear"` |
| 影响方向 | DL / 上下行关系 |
| 严重程度 | 阻塞 |
| 修复类型 | 代码缺陷修复 (Bug) + 上下行差异未正确实现 |
| 回归验证脚本 | `summarize_phase1()` → `phase1_sample_summary.json`; `analyze_pdp_examples()` → PDP 对照; `analyze_estimated_quality()` → NMSE |
| 预估修复代价 | 3–4 小时 |
| 修复依赖 | 无（基础性修复，其他 quadriga_real 任务依赖它） |
| 建议优先级 | **P0** |

**根因定位:**
1. Python 层 `quadriga_real.py:285` 硬编码 `link="UL"`, `channel_est_mode="ls_linear"`
2. MATLAB `main_multi.m:319-324` 只调用 `ul_srs_pipeline()`，未接线 `dl_csirs_pipeline()`
3. Python `validate_config()` 未解析上层传入的 `link` / `channel_est_mode` 参数

**代码位置:**
- `D:\MSG平台_cc\src\msg_embedding\data\sources\quadriga_real.py:125-170` (validate_config)
- `D:\MSG平台_cc\src\msg_embedding\data\sources\quadriga_real.py:276-302` (_iter_from_mat)
- `D:\MSG平台_cc\matlab\main_multi.m:319-324` (仅调用 ul_srs_pipeline)
- `D:\MSG平台_cc\matlab\dl_csirs_pipeline.m:1-114` (已存在但未使用)

---

## F-002 — quadriga_real 静态同点位生成失败

| 字段 | 值 |
|---|---|
| 对应 Codex 编号 | R2-002 |
| 问题摘要 | `mobility_mode=static` 的 MATLAB 生成直接失败，无法完成同一点位采集 |
| 影响方向 | 两者 |
| 严重程度 | 阻塞 |
| 修复类型 | 代码缺陷修复 (Bug) |
| 回归验证脚本 | `phase1_collect_samples.py` (quadriga static specs); `summarize_phase1()` |
| 预估修复代价 | 3–4 小时 |
| 修复依赖 | 无（与 F-001 同为 quadriga_real，但可独立修） |
| 建议优先级 | **P0** |

**根因定位:**
- MATLAB `main_multi.m:302,460` 的 `static` 轨迹构造与 `gen_channel_multicell` 的 reshape/snapshot 假设不一致
- 静态 track 的 `no_snapshots` 维度与 `fr(...)` 输出维度和保存张量的 `no_ss` 不统一

**代码位置:**
- `D:\MSG平台_cc\matlab\main_multi.m:201-236` (轨迹生成)
- `D:\MSG平台_cc\matlab\main_multi.m:302-307` (gen_channel_multicell 调用)
- `D:\MSG平台_cc\matlab\gen_channel_multicell.m:1-88` (信道生成)

---

## F-003 — quadriga_real 估计信道质量崩溃

| 字段 | 值 |
|---|---|
| 对应 Codex 编号 | R2-003 |
| 问题摘要 | estimated vs true NMSE 达 +29.22 dB，频域相关仅 0.033，几乎失去主结构 |
| 影响方向 | 两者 |
| 严重程度 | 严重 |
| 修复类型 | 代码缺陷修复 (Bug) — 张量轴顺序/归一化口径对齐 |
| 回归验证脚本 | `analyze_estimated_quality()` → `estimated_quality.json` |
| 预估修复代价 | 4–6 小时 |
| 修复依赖 | F-001（方向/模式透传修好后才能正确评估 NMSE） |
| 建议优先级 | **P1** |

**根因定位:**
- MATLAB 输出 `Hf_serving_est` / `Hf_serving_ideal` 的轴顺序为 `[no_ue, BsAnt, ue_ant, N_RB, no_ss]`
- Python `_iter_from_mat()` 做 `np.transpose(Hf_ideal[i_ue], (3, 2, 0, 1))` 将 `[BsAnt, ue_ant, N_RB, no_ss]` → `[no_ss, N_RB, BsAnt, ue_ant]`
- 但 LS 估计在 MATLAB 端按不同轴顺序运算，转换后 ideal 和 estimated 的物理含义可能已错位
- 此外 Python 端统一做了 `scale = sqrt(raw_gain)` 归一化（line 261-264），但 MATLAB 端的 LS 估计是在未归一化的信道上做的，归一化比例不一致

**代码位置:**
- `D:\MSG平台_cc\src\msg_embedding\data\sources\quadriga_real.py:253-265` (轴转换与归一化)
- `D:\MSG平台_cc\matlab\ul_srs_pipeline.m:1-119` (LS 估计)
- `D:\MSG平台_cc\matlab\main_multi.m:317-324` (存储)

---

## F-004 — sionna_rt 干扰等级变化不可观测

| 字段 | 值 |
|---|---|
| 对应 Codex 编号 | R2-004 |
| 问题摘要 | 干扰从 baseline 到 high，频域相关不变，有效秩不变，SNR/SIR/SINR 钉在 49.9 dB |
| 影响方向 | UL / DL / 上下行关系 |
| 严重程度 | 阻塞 |
| 修复类型 | 代码缺陷修复 (Bug) — 干扰注入/SNR 计算链路问题 |
| 回归验证脚本 | `run_interference_correlation()` → `corr_interference.json` |
| 预估修复代价 | 6–8 小时 |
| 修复依赖 | 无 |
| 建议优先级 | **P0** |

**根因定位:**
SIR 被钉在 49.9 dB 说明 `p_intf <= 1e-30`（`_interference_estimation.py:322-323`），即 `interference_total` 近零。可能原因链：
1. `sionna_rt._generate_one_sample()` 中 `h_interferers` 的干扰信道功率极弱（RT 场景下距离远的干扰基站信道增益本就极小）
2. 信道归一化（`serving_power` 归一化 at line 1224-1227）后，serving 和 interferer 都被同一因子缩放，但 serving 被重新设为单位功率，interferer 相对值变得极小
3. 在 `estimate_channel_with_interference()` 内，干扰信号 `H_intf * X_intf` 在 pilot 位置上叠加后，功率远低于 serving 信号
4. SNR/SIR/SINR 计算使用 `rx_power_dbm` 而非估计链路内部值，两者口径不一致

**代码位置:**
- `D:\MSG平台_cc\src\msg_embedding\data\sources\sionna_rt.py:1220-1250` (归一化与 SNR 计算)
- `D:\MSG平台_cc\src\msg_embedding\data\sources\_interference_estimation.py:244-354` (干扰注入与 SIR 计算)
- `D:\MSG平台_cc\src\msg_embedding\data\sources\_interference_estimation.py:316-323` (sir_dB 回退到 49.9)

---

## F-005 — sionna_rt 静止模式位置跳点

| 字段 | 值 |
|---|---|
| 对应 Codex 编号 | R2-005 |
| 问题摘要 | `mobility_mode=static`, `ue_speed_kmh=0` 下 8 个 snapshot 中出现多个离散坐标跳点 |
| 影响方向 | 两者 |
| 严重程度 | 阻塞 |
| 修复类型 | 代码缺陷修复 (Bug) |
| 回归验证脚本 | `run_mobility_correlation()` → `corr_mobility_probe_counts.json` (position 序列检查) |
| 预估修复代价 | 2–3 小时 |
| 修复依赖 | 无 |
| 建议优先级 | **P0** |

**根因定位:**
- `sionna_rt.py:1108-1161`：RT 重试逻辑在 `max_power <= 1e-20` 时会调用 `_place_ues_uniform()` 生成新 UE 位置
- 在 `ue_pos_override is None` 条件下（静态模式传入了 `ue_pos_override=None`），每次重试都可能换到不同位置
- 但静止模式 `mobility_mode=static` 时 `ue_pos_override` 确实是 None（`iter_samples()` 在 static 模式下不生成 trajectory），所以 RT 重试会改变位置
- 更根本地：即使 `ue_pos_override` 非 None，RT 重试仍然会在 override 为 None 时换位置（line 1151-1161）

**代码位置:**
- `D:\MSG平台_cc\src\msg_embedding\data\sources\sionna_rt.py:1108-1161` (RT 重试逻辑)
- `D:\MSG平台_cc\src\msg_embedding\data\sources\sionna_rt.py:1463-1495` (iter_samples 轨迹逻辑)

---

## F-006 — sionna_rt 动态序列 snapshot 级回退 TDL

| 字段 | 值 |
|---|---|
| 对应 Codex 编号 | R2-006 |
| 问题摘要 | 动态实验中出现 snapshot 级 RT 回退到 TDL，`medium UL/DL` 各有 1/8 回退 |
| 影响方向 | UL / DL |
| 严重程度 | 严重 |
| 修复类型 | 代码缺陷修复 (Bug) + 物理建模不真实 |
| 回归验证脚本 | `run_mobility_correlation()` → `corr_mobility_probe_counts.json` |
| 预估修复代价 | 3–4 小时 |
| 修复依赖 | 无 |
| 建议优先级 | **P1** |

**根因定位:**
- `sionna_rt.py:1162-1200`：RT 求解全部 `_MAX_RT_RETRIES=5` 次失败后，自动 fallback 到 `_compute_channels_tdl()`
- 动态序列中某些位置可能在场景地图中无法 RT 求解（如建筑内部、边界外），导致混入 TDL
- 同一序列混入两种物理模型，破坏了信道统计连续性

**代码位置:**
- `D:\MSG平台_cc\src\msg_embedding\data\sources\sionna_rt.py:1162-1200` (TDL 回退逻辑)
- `D:\MSG平台_cc\src\msg_embedding\data\sources\sionna_rt.py:1066-1068` (_generate_one_sample)

---

## F-007 — internal_sim 场景枚举残留 LOS 分支

| 字段 | 值 |
|---|---|
| 对应 Codex 编号 | R2-007 |
| 问题摘要 | 接口不接受 `UMa_LOS`，但代码仍保留 `UMa_LOS/UMi_LOS` 默认参数分支 |
| 影响方向 | 两者 |
| 严重程度 | 一般 |
| 修复类型 | 文档与代码不一致 (对齐) |
| 回归验证脚本 | `summarize_phase1()` 场景处理; `ideal_ul_checks.py` |
| 预估修复代价 | 1 小时 |
| 修复依赖 | 无 |
| 建议优先级 | **P2** |

**根因定位:**
- `internal_sim.py:628`：`_PATHLOSS_MODELS` 只包含 NLOS 和 InF，不含 LOS
- 但 `internal_sim.py:636-644`：`_default_tx_power` 仍包含 `UMa_LOS` / `UMi_LOS`
- 场景验证会 raise ValueError，但残余默认值分支会误导开发者

**代码位置:**
- `D:\MSG平台_cc\src\msg_embedding\data\sources\internal_sim.py:628` (scenario 验证)
- `D:\MSG平台_cc\src\msg_embedding\data\sources\internal_sim.py:636-644` (默认 TX 功率)

---

## F-008 — internal_sim 移动建模相邻相关偏低

| 字段 | 值 |
|---|---|
| 对应 Codex 编号 | R2-008 |
| 问题摘要 | 3.6 km/h 小 mobility 的 UL 相邻相关仅 0.135、DL 仅 0.167，去相关偏快 |
| 影响方向 | UL / DL |
| 严重程度 | 一般 |
| 修复类型 | 物理建模不真实 |
| 回归验证脚本 | `analyze_phase1_mobility()` → `mobility_phase1_summary.json` |
| 预估修复代价 | 4–6 小时 |
| 修复依赖 | 无 |
| 建议优先级 | **P2** |

**根因定位:**
- `internal_sim` 使用 TDL 模型独立生成每个 snapshot 的快衰落
- 相邻 snapshot 之间只共享大尺度参数（路损、场景），快衰落成分完全独立重新随机
- 物理上，3.6 km/h（1 m/s）在 10 ms 间隔下位移仅 0.01 m ≈ λ/8.6（3.5 GHz），小尺度衰落应有较强时间相关性
- 但 TDL 模型按 snapshot 独立采样，丢失了时间域连续性

**代码位置:**
- `D:\MSG平台_cc\src\msg_embedding\data\sources\internal_sim.py:952-1165` (_generate_one_sample)
- `D:\MSG平台_cc\src\msg_embedding\data\sources\_mobility.py:35-141` (轨迹生成，不含信道连续性)

---

## 汇总统计

| 优先级 | 任务数 | 任务编号 |
|---|---|---|
| P0（阻塞级） | 4 | F-001, F-002, F-004, F-005 |
| P1（本轮必修） | 2 | F-003, F-006 |
| P2（可延后） | 2 | F-007, F-008 |
| **合计** | **8** | |

| 影响方向 | 任务 |
|---|---|
| 仅 DL / 上下行关系 | F-001 |
| 仅 UL/DL | F-006, F-007, F-008 |
| 两者 | F-002, F-003, F-004, F-005 |

| 修复类型 | 任务 |
|---|---|
| 代码缺陷修复 (Bug) | F-001, F-002, F-003, F-004, F-005, F-006 |
| 物理建模不真实 | F-006, F-008 |
| 上下行差异未正确实现 | F-001 |
| 文档与代码不一致 | F-007 |
