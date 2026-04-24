# 05 - 统一问题清单（合并去重）

合并时间：2026-04-23
来源：Codex 30 条 (C-xxx) + 自检 16 条 (S-xxx)

## 编号规则
- M-xxx：合并后编号
- 原始来源：C-xxx = Codex 发现，S-xxx = 自检发现

---

### M-001 | `quadriga_real` 多小区 CIR 未单独存储（设计如此）

| 项目 | 内容 |
|------|------|
| 原始来源 | C-001 |
| 最终裁决 | 已修复（根据审查反馈） |
| 问题描述 | Codex 审查时正确发现 `quadriga_real.py:273` 写 `h_interferers=None`，多小区 CIR 未保留。**已修复**：实施干扰感知 LS 信道估计方案（MATLAB `ul_srs_pipeline.m` + Python `_interference_estimation.py`），`Y = H_serving*X_s + Σ(H_k*X_k) + noise → H_hat = Y*conj(X_s)/|X_s|^2`，干扰烘焙进 `h_serving_est`。 |
| 根因 | 原问题真实存在。已通过干扰感知估计方案解决，用 (est, ideal) pair 替代存储 raw per-cell CIR。 |
| 严重程度 | **已知限制**（非阻塞） |
| 修复代价 | — |
| 修复优先级 | **P3 不修但记录** |
| 修复方向 | 在 platform guide 中明确说明此设计决策。如果未来需要 raw per-cell CIR（channel charting 等），需扩展 MATLAB 保存策略。 |

---

### M-002 | `sionna_rt` 静默 fallback + 标记/元数据不分

| 项目 | 内容 |
|------|------|
| 原始来源 | C-002 + C-017 + C-003 |
| 最终裁决 | Confirmed |
| 问题描述 | Sionna 不可用时自动降级 TDL，样本 `source` 仍为 `"sionna_rt"`。`run_meta.json` 固定 `"fallback": false`。TDL fallback 多小区不共享散射环境。下游无法区分真 RT 和 fallback 样本。 |
| 根因 | 设计决策：让平台无 Sionna 也能跑通。执行缺陷：source 字段和 run_meta 未区分。 |
| 严重程度 | **阻塞** |
| 修复代价 | 0.5-1 天 |
| 修复优先级 | **P0 立即** |
| 修复方向 | (1) `source` 改为 `"sionna_rt_tdl_fallback"` 或增加 `SourceType` 枚举 (2) `run_meta.json` 动态写入实际 fallback 状态 (3) 增加 strict 模式：Sionna 不可用则 fail |

---

### M-003 | `internal_sim` 拓扑 num_sites→cells 不裁剪

| 项目 | 内容 |
|------|------|
| 原始来源 | C-011 + S-015 |
| 最终裁决 | Confirmed |
| 问题描述 | `num_sites=3` 通过 `_sites_to_rings(3)=1` 生成 7 sites。`sionna_rt` 有裁剪 (`sites[:num_cells]`)，`internal_sim` 没有。配置 `sectors_per_site=3` 时乘法关系进一步放大问题。 |
| 根因 | `internal_sim._build_sites()` 遗漏裁剪步骤。 |
| 严重程度 | **严重** |
| 修复代价 | 1 小时 |
| 修复优先级 | **P1 本轮必修** |
| 修复方向 | `_build_sites()` 末尾添加 `sites = sites[:self.num_sites * self.sectors_per_site]`；或限制 `num_sites` 只接受 hex ring 精确值（1/7/19/37）。**需要人工决策**：选哪种方案。 |

---

### M-004 | SSB 测量使用假干扰 + 异常静默吞掉

| 项目 | 内容 |
|------|------|
| 原始来源 | S-001 + S-002 |
| 最终裁决 | Confirmed（Codex 未发现） |
| 问题描述 | `quadriga_multi.py:445` 在 `h_interferers=None` 时用 `h_serving_true * 0.1` 作假干扰信道。SSB 异常被 `except Exception: pass` 吞掉（`:468-469`），无日志。 |
| 根因 | 为让 SSB 测量在所有场景下跑通，简化了缺失数据的处理。 |
| 严重程度 | **严重** |
| 修复代价 | 2-3 小时 |
| 修复优先级 | **P1 本轮必修** |
| 修复方向 | (1) `h_interferers=None` 时跳过 SSB 或只计算 serving RSRP (2) 将 `except: pass` 改为 `except Exception as e: logger.warning(...)` |

---

### M-005 | `sionna_rt` TDL fallback pathloss 符号错误

| 项目 | 内容 |
|------|------|
| 原始来源 | C-029 |
| 最终裁决 | Confirmed |
| 问题描述 | `sionna_rt.py:1387-1388` 计算 `rx_power - tx_power`（负值），`internal_sim` 记录正值 `pl_db`。两个 source 的 `pathloss_serving_db` 语义相反。 |
| 根因 | 编码时符号约定错误。 |
| 严重程度 | **严重** |
| 修复代价 | 15 分钟 |
| 修复优先级 | **P1 本轮必修** |
| 修复方向 | 改为 `self.tx_power_dbm - rx_power_dbm[serving_idx]` |

---

### M-006 | 元数据缺失 provenance 字段

| 项目 | 内容 |
|------|------|
| 原始来源 | C-004 + C-005 |
| 最终裁决 | Confirmed（C-005 严重程度调整为 HIGH） |
| 问题描述 | 所有样本缺少 `tool_version`, `code_version`, `git_commit`, `generated_at`。字段命名含单位后缀但无统一坐标系声明。 |
| 根因 | 重构优先实现核心功能，provenance 列为后续迭代。 |
| 严重程度 | **严重** |
| 修复代价 | 0.5-1 天 |
| 修复优先级 | **P1 本轮必修** |
| 修复方向 | `ChannelSample` 增加 `schema_version`, `producer_version`, `git_commit` 字段；构建时注入版本信息 |

---

### M-007 | Schema 无版本 + manifest/contract/ORM 三层不一致

| 项目 | 内容 |
|------|------|
| 原始来源 | C-006 + S-005 |
| 最终裁决 | Confirmed |
| 问题描述 | `to_parquet_row()` 输出、`MANIFEST_SCHEMA`、`run_full_pipeline.py` 手写 row 三者字段/类型各不相同。`sample_id` 在 contract 是 UUID 字符串、manifest 是 int32、ORM 是字符串。 |
| 根因 | 分阶段开发未统一。 |
| 严重程度 | **阻塞** |
| 修复代价 | 1-2 天（需要统一设计） |
| 修复优先级 | **P1 本轮必修** |
| 修复方向 | 以 `ChannelSample.to_parquet_row()` 为 source of truth，统一 `MANIFEST_SCHEMA` 和 ORM `Sample` 模型。**需要人工决策**：`sample_id` 用 UUID 还是 int。 |

---

### M-008 | `h_interferers=None` 与 `num_cells>1` 无一致性校验

| 项目 | 内容 |
|------|------|
| 原始来源 | C-022 |
| 最终裁决 | Partially Valid（严重程度下调） |
| 问题描述 | `ChannelSample` 允许 `meta.num_cells=7` + `h_interferers=None`，不报错不警告。 |
| 根因 | 对于 `quadriga_real`，`h_interferers=None` 是有意设计（干扰烘焙进 h_serving_est）。但对于其他 source，`num_cells>1` + `h_interferers=None` 可能表示数据不完整。 |
| 严重程度 | **一般** |
| 修复代价 | 1-2 小时 |
| 修复优先级 | **P2 可延后** |
| 修复方向 | 增加 model_validator：`num_cells>1` 时 `h_interferers` 不得为 None，除非 source 使用干扰感知估计（`quadriga_real`）或显式声明 `serving_only=True` |

---

### M-009 | 参数边界不校验（0 带宽 / 0 距离 / 0 天线 / 负频率）

| 项目 | 内容 |
|------|------|
| 原始来源 | C-007 + C-008 + C-009 + C-010 |
| 最终裁决 | Confirmed |
| 问题描述 | `bandwidth_hz=0` 生成 1 RB 样本；零距离 clamp 到 1m 但 meta 记录 0m；零天线生成 NaN；负频率运行时崩溃。 |
| 根因 | 缺少 `validate_config()` 中的物理约束检查。 |
| 严重程度 | **阻塞** |
| 修复代价 | 2-3 小时 |
| 修复优先级 | **P0 立即** |
| 修复方向 | 在 `validate_config()` 中添加：`bandwidth_hz > 0`, `carrier_freq_hz > 0`, `num_bs_ant >= 1`, `num_ue_ant >= 1`。可选：`ChannelSample` 增加维度 > 0 检查。 |

---

### M-010 | NaN 通过 `.tolist()` 传播到输出

| 项目 | 内容 |
|------|------|
| 原始来源 | S-008 |
| 最终裁决 | Confirmed（Codex 未发现） |
| 问题描述 | SSB 结果 `.tolist()` 不检查 NaN/Inf，可能将无效值写入 `ChannelSample` 的 `ssb_rsrp_dBm` 等字段。 |
| 根因 | 缺少输出数据有效性检查。 |
| 严重程度 | **严重** |
| 修复代价 | 1-2 小时 |
| 修复优先级 | **P1 本轮必修** |
| 修复方向 | SSB 结果转换前加 `np.isfinite` 检查；或在 `ChannelSample` validator 中校验 SSB 列表不含 NaN |

---

### M-011 | 采集样本不自动进入 manifest/DB

| 项目 | 内容 |
|------|------|
| 原始来源 | C-013 |
| 最终裁决 | Confirmed |
| 问题描述 | `run_simulate.py` 只写 `.pt` 文件，不写 manifest 或注册 DB。文档/UI 暗示采集后即可查看。 |
| 根因 | CLI 设计为轻量工具，manifest/DB 是独立步骤。 |
| 严重程度 | **阻塞** |
| 修复代价 | 0.5-1 天 |
| 修复优先级 | **P1 本轮必修** |
| 修复方向 | `run_simulate.py` 完成后可选调用 manifest 写入；或 worker task 中自动触发 |

---

### M-012 | Channels Explorer 读取旧字段名

| 项目 | 内容 |
|------|------|
| 原始来源 | C-014 |
| 最终裁决 | Confirmed |
| 问题描述 | `channels.py:178-184` 读 `channel_ideal/channel_est`，新样本字段是 `h_serving_true/h_serving_est`。 |
| 根因 | 重构时字段重命名未同步更新 Channels Explorer。 |
| 严重程度 | **阻塞** |
| 修复代价 | 30 分钟 |
| 修复优先级 | **P1 本轮必修** |
| 修复方向 | 更新 `channels.py` 读取新字段名；加向后兼容 fallback 到旧字段名 |

---

### M-013 | 删除操作只删 DB 不删文件

| 项目 | 内容 |
|------|------|
| 原始来源 | C-015 |
| 最终裁决 | Confirmed |
| 问题描述 | 删除 endpoint 只 `db.delete()`，不删 `.pt`/manifest/bridge 产出。 |
| 根因 | 数据管理生命周期未完整实现。 |
| 严重程度 | **一般** |
| 修复代价 | 2-3 小时 |
| 修复优先级 | **P2 可延后** |
| 修复方向 | 删除操作同步清理物理文件和 manifest 条目 |

---

### M-014 | MATLAB 注释/脚本常量与实际不一致

| 项目 | 内容 |
|------|------|
| 原始来源 | C-012 |
| 最终裁决 | Confirmed |
| 问题描述 | `main_multi.m` 声称输出 `Hf_multi` 实际不输出；`run_quadriga_real.py` 注释 "10x500=5000" 实际 20x50=1000。 |
| 根因 | 文档腐化。 |
| 严重程度 | **一般** |
| 修复代价 | 30 分钟 |
| 修复优先级 | **P1 本轮必修**（因为会误导维护者） |
| 修复方向 | 更新注释与实际代码一致 |

---

### M-015 | 队列协议文档与代码不一致

| 项目 | 内容 |
|------|------|
| 原始来源 | C-019 |
| 最终裁决 | Confirmed |
| 问题描述 | 文档描述 Redis 队列，实际是 file-drop + Dramatiq；进度格式也不一致。 |
| 根因 | 重构改了实现但文档未同步。 |
| 严重程度 | **一般** |
| 修复代价 | 1 小时 |
| 修复优先级 | **P2 可延后** |
| 修复方向 | 更新 platform guide 文档 |

---

### M-016 | `quadriga_multi` 不支持 `link=BOTH` 但 UI 展示

| 项目 | 内容 |
|------|------|
| 原始来源 | C-020 |
| 最终裁决 | Confirmed |
| 问题描述 | 前端对所有 source 展示 BOTH 选项，`quadriga_multi` 拒绝 BOTH。 |
| 根因 | 前端未按 source 差异化 UI。 |
| 严重程度 | **一般** |
| 修复代价 | 1-2 小时 |
| 修复优先级 | **P2 可延后** |
| 修复方向 | 前端按 source 配置可用的 link 选项 |

---

### M-017 | `sionna_rt_mock` 可见性不一致

| 项目 | 内容 |
|------|------|
| 原始来源 | C-021 |
| 最终裁决 | Confirmed |
| 问题描述 | mock source 在 registry 注册但不在 `SourceType` enum 和 API 中；mock 样本 source 写为 `"sionna_rt"`。 |
| 根因 | mock 是测试用，未与正式 source 体系对齐。 |
| 严重程度 | **轻微** |
| 修复代价 | 1 小时 |
| 修复优先级 | **P3 不修但记录** |
| 修复方向 | 将 mock 限制在测试环境，或加入 SourceType enum |

---

### M-018 | 批量处理静默跳过失败样本

| 项目 | 内容 |
|------|------|
| 原始来源 | S-006 |
| 最终裁决 | Confirmed（Codex 未发现） |
| 问题描述 | `post_quadriga_pipeline.py:372-373` 中 `except Exception: pass` 导致样本加载/特征提取失败时无任何日志或计数。 |
| 根因 | 快速原型代码的错误处理不完善。 |
| 严重程度 | **严重** |
| 修复代价 | 30 分钟 |
| 修复优先级 | **P1 本轮必修** |
| 修复方向 | 改为 `except Exception as e: logger.warning(...); error_count += 1`；末尾汇总 |

---

### M-019 | MATLAB 子进程失败无诊断信息

| 项目 | 内容 |
|------|------|
| 原始来源 | S-007 |
| 最终裁决 | Confirmed（Codex 未发现） |
| 问题描述 | `quadriga_real.py:196` 所有 MATLAB 异常被吞掉只返回 `False`。 |
| 根因 | 错误处理过于简化。 |
| 严重程度 | **一般** |
| 修复代价 | 30 分钟 |
| 修复优先级 | **P1 本轮必修** |
| 修复方向 | 记录 `result.stderr` 和异常 traceback |

---

### M-020 | UE 位置提取失败静默回退到占位坐标

| 项目 | 内容 |
|------|------|
| 原始来源 | S-004 |
| 最终裁决 | Confirmed（Codex 未发现） |
| 问题描述 | `quadriga_real.py:263` 失败时用 `[0, 0, 1.5]`；`quadriga_multi.py:420` 失败时用 `None`。无日志。 |
| 根因 | 防御性编码但缺少可观测性。 |
| 严重程度 | **一般** |
| 修复代价 | 30 分钟 |
| 修复优先级 | **P2 可延后** |
| 修复方向 | 添加 `logger.warning` 说明位置提取失败原因 |

---

### M-021 | `.pt` 文件不逐比特可复现

| 项目 | 内容 |
|------|------|
| 原始来源 | C-018 |
| 最终裁决 | Partially Valid |
| 问题描述 | UUID4 + 时间戳导致文件哈希不一致。物理数据 canonical hash 可复现。 |
| 根因 | 设计选择。 |
| 严重程度 | **一般** |
| 修复代价 | 1-2 小时 |
| 修复优先级 | **P2 可延后** |
| 修复方向 | 支持 canonical export 模式；或提供 content-hash 字段 |

---

### M-022 | `samples_per_sec` 低估端到端耗时

| 项目 | 内容 |
|------|------|
| 原始来源 | C-024 |
| 最终裁决 | Confirmed |
| 问题描述 | `t0` 设在 source 构造后，不含初始化时间。 |
| 根因 | 计时点选择不当。 |
| 严重程度 | **一般** |
| 修复代价 | 15 分钟 |
| 修复优先级 | **P2 可延后** |
| 修复方向 | 将 `t0` 移到 source 构造前；summary 同时报告 init_time 和 gen_time |

---

### M-023 | 检索/筛选范围有限

| 项目 | 内容 |
|------|------|
| 原始来源 | C-016 |
| 最终裁决 | Confirmed |
| 问题描述 | API 只支持 source/link/snr 筛选。与 M-006 关联。 |
| 根因 | ORM schema 字段有限。 |
| 严重程度 | **一般** |
| 修复代价 | 0.5 天 |
| 修复优先级 | **P2 可延后** |
| 修复方向 | 随 M-006 和 M-007 一起扩展 |

---

### M-024 | Legacy PMI 全局状态 + 干扰默认 NaN

| 项目 | 内容 |
|------|------|
| 原始来源 | S-009 + S-010 |
| 最终裁决 | Confirmed（Codex 未发现） |
| 问题描述 | `gol._init()` 全局状态在并发调用时竞态。干扰特征用 NaN 填充，下游 loss 需 NaN-aware。 |
| 根因 | Legacy 代码路径的固有限制。 |
| 严重程度 | **一般** |
| 修复代价 | 0.5-1 天 |
| 修复优先级 | **P2 可延后** |
| 修复方向 | 封装 `gol` 为实例状态；NaN 默认值改为 0 或加 mask 字段 |

---

### M-025 | `to_dict()` 内存膨胀 + Sionna scene cache 无释放

| 项目 | 内容 |
|------|------|
| 原始来源 | S-012 + S-014 |
| 最终裁决 | Confirmed（Codex 未发现） |
| 问题描述 | `.tolist()` 深拷贝大数组内存膨胀 10-20x。Sionna scene cache 无 `__del__` 或 context manager。 |
| 根因 | 序列化效率和资源管理的遗漏。 |
| 严重程度 | **一般** |
| 修复代价 | 0.5 天 |
| 修复优先级 | **P2 可延后** |
| 修复方向 | 大数组改用 `np.save` 而非 `.tolist()`；scene 增加 `close()` 方法 |

---

### P3 不修但记录

| 编号 | 问题 | 理由 |
|------|------|------|
| M-026 | TDL 不共享散射体 (C-003) | TDL 模型固有局限，修复需换模型 |
| M-027 | ASA/ASD/PDP 不在输出 (C-027+C-028) | 功能增强，可从 CFR 后处理导出 |
| M-028 | UMi/InF simplified (C-030) | 代码已标注 simplified，诚实声明 |
| M-029 | complex64 精度/SNR 截断 (S-011) | 影响极端场景，ROI 低 |
| M-030 | 跨模块隐式 shape 约定 (S-016) | 架构问题，需随大重构解决 |
| M-031 | `sionna_rt_mock` 可见性 (C-021/M-017) | 轻微，仅影响开发者 |

---

## 统计汇总

| 优先级 | 数量 | 涉及编号 |
|--------|------|---------|
| P0 立即 | 3 | M-001, M-002, M-009 |
| P1 本轮必修 | 12 | M-003~M-008, M-010~M-012, M-014, M-018~M-019 |
| P2 可延后 | 7 | M-013, M-015~M-016, M-020~M-023, M-024~M-025 |
| P3 不修但记录 | 6 | M-026~M-031 |
| **总计** | **28** | |
