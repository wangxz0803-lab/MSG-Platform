# 01 - Codex 审查问题清单（逐条提取）

提取时间：2026-04-23
来源：D:\MSG平台_codex\review\ 全部 13 份报告

## 提取原则

- 不合并、不省略，每条 Codex 提出的具体技术问题独立编号
- 跨报告重复出现的同一问题只编号一次，在备注中标注重复来源
- 严重程度使用 Codex 原始判定

---

| 编号 | 来源报告 | 问题类别 | Codex 严重程度 | Codex 原始描述摘要 | Codex 证据 |
|------|----------|----------|---------------|-------------------|------------|
| C-001 | FINAL-REPORT (B-01), 02-phase2 (B1), 12-multicell, 14-end-to-end (R-14-01) | 信道真实性 / 多小区耦合 | BLOCKER | `quadriga_real` MATLAB 生成 `Hf_per_cell` 但保存/适配层丢弃，平台输出 `h_interferers=None`，无法支持多小区 CIR 下游训练 | MATLAB `main_multi.m:288-309` 生成 `Hf_per_cell`；`:380-381` 未保存；Python `quadriga_real.py:270-274` 固定写 `None`；end-to-end 测试 failed |
| C-002 | FINAL-REPORT (B-02), 01-phase1 (F5), 02-phase2 (B2), 06-repro (R-06-03), 08-schema (R-08-03), 14-e2e (R-14-02), 15-perf (R-15-03) | 信道真实性 | BLOCKER | `sionna_rt` 缺少 Sionna 时静默 fallback 到 TDL，样本仍标记 `source="sionna_rt"`，导致数据混淆 | `sionna_rt.py:431-444` 不抛错；`:1098-1186` fallback 逻辑；`:1434` source 仍为 `"sionna_rt"`；所有 phase2/3 sionna 样本 `sionna_rt_used=False` |
| C-003 | 02-phase2 (B3), 12-multicell | 多小区耦合 | BLOCKER | Sionna TDL fallback 的多小区信道是逐 cell 独立 TDL 生成，不是共享散射环境 | `sionna_rt.py:1014-1032` 逐 cell 循环；真实 RT 路径才在同一 scene 中求解 `:918-965` |
| C-004 | FINAL-REPORT (B-03), 07-metadata (R-07-01), 06-repro (R-06-02) | 元数据 | BLOCKER | 全部 657 个样本缺少 `tool_version`, `code_version`, `git_commit`, `generated_at` 五个 provenance 字段 | `stage3_metadata.json` 中 657/657 缺失；`contract.py` 无对应字段 |
| C-005 | 07-metadata (R-07-02) | 元数据 | BLOCKER | 缺少统一 `units` 字段和坐标系声明 | 同上扫描结果 |
| C-006 | FINAL-REPORT (B-04), 01-phase1 (F3), 08-schema (R-08-01, R-08-02), 13-data (R-13-05), 14-e2e (R-14-04) | Schema | BLOCKER | `ChannelSample` 无 `schema_version`；`to_parquet_row()` 与 `MANIFEST_SCHEMA` 字段/类型不一致；`sample_id` 在 contract 是 UUID 字符串，manifest 是 int32 | `contract.py:112-194` 无 version；`contract.py:492-521` vs `manifest.py:37-63`；`sample.py:18-31` ORM 又是字符串 |
| C-007 | FINAL-REPORT (B-05), 10-robustness (R-10-01) | 鲁棒性 | BLOCKER | `bandwidth_hz=0` 静默生成样本（强制 1 RB），不报错 | `internal_sim.py:557-563` `max(1, int(0/...))=1`；动态样本 `zero_bandwidth/sample_000000.pt` |
| C-008 | FINAL-REPORT (B-05), 10-robustness (R-10-02) | 鲁棒性 | BLOCKER | 零距离 BS/UE 静默生成非物理路损样本（内部 clamp 到 1m 但 meta 记录 0m） | `internal_sim.py:85-89` clamp；`:1208-1211` meta 记录 0m；动态样本 `zero_distance/sample_000000.pt` |
| C-009 | FINAL-REPORT (B-05), 10-robustness (R-10-03) | 鲁棒性 | BLOCKER | 零 BS 天线生成 shape=[T,RB,0,UE] 且功率 NaN 的样本 | `internal_sim.py:566-574` 无 >=1 校验；`contract.py:269-280` shape validator 不检查每维>0 |
| C-010 | 10-robustness (R-10-04) | 鲁棒性 | MEDIUM | 负频率在运行时 `math domain error` 崩溃，非配置期友好报错 | `internal_sim.py:88` `log10(fc_ghz)` 对负数 |
| C-011 | 02-phase2 (S2), 12-multicell, 14-e2e (R-14-03) | 可复现性 / 语义一致性 | SEVERE | `internal_sim` `num_sites=3` 通过 hex ring rounding 实际生成 7 个 cell，语义不符直觉 | `internal_sim.py:54-69` `_sites_to_rings(3)=1`；`hex_grid.py:107-153` ring-1=7 sites；`sionna_rt` 有裁剪但 `internal_sim` 无 |
| C-012 | 02-phase2 (S1), 15-perf (R-15-02) | 语义一致性 | SEVERE | `main_multi.m` 文件头声称输出 `Hf_multi`，实际保存 `Hf_serving_*`；注释说 "10x500=5000" 实际常量 20x50=1000 | `main_multi.m:8-11` vs `:380-381`；`run_quadriga_real.py:1-4` vs `:22-23` |
| C-013 | 01-phase1 (F1), 13-data (R-13-01) | 数据管理 | BLOCKER | 采集生成的样本（`run_simulate.py`）不会自动写入 manifest/DB，与文档"采集即可查看"不符 | `run_simulate.py:99-109` 只写 .pt；无 manifest 写入调用 |
| C-014 | 13-data (R-13-02) | 数据管理 | BLOCKER | Channels Explorer 期望旧字段 `channel_ideal/channel_est`，新 `ChannelSample` 使用 `h_serving_true/h_serving_est` | `channels.py:178-184` 读 `channel_ideal`；`contract.py:131-146` 字段为 `h_serving_*` |
| C-015 | 13-data (R-13-03) | 数据管理 | HIGH | 删除 dataset 只删 DB 行，不同步删除 `.pt` 文件、manifest 或 bridge shard | `datasets.py:96-108` 只有 DB delete |
| C-016 | 13-data (R-13-04) | 数据管理 | HIGH | 检索/筛选只支持 source/link/min_snr/max_snr，无法按 provenance 查询 | `datasets.py:46-65`；与 C-004 关联 |
| C-017 | 07-metadata (R-07-03), 02-phase2 | 元数据 | BLOCKER | `run_meta.json` 的 `"fallback": false` 是硬编码，不反映 Sionna 实际 fallback 状态 | `run_simulate.py:136-144` 固定写 `false` |
| C-018 | 06-reproducibility (R-06-01) | 可复现性 | HIGH | 同配置同种子的 `.pt` 文件哈希不一致（UUID4 + 时间戳导致），非逐比特可复现 | `contract.py:240-249` UUID4；`stage3_reproducibility.json` file_hash≠ canonical_hash= |
| C-019 | 01-phase1 (F2) | 语义一致性 | SEVERE | 平台队列协议文档说 Redis→Worker，实际是 file-drop→Dramatiq；进度格式也不一致 | `job_dispatch.py:64-73` 写文件；`queue_watcher.py:83-120` 扫文件；`base.py:28` 进度格式 |
| C-020 | 01-phase1 (F4) | 语义一致性 | SEVERE | `quadriga_multi` 拒绝 `link=BOTH`，但文档和前端 UI 展示此选项 | `quadriga_multi.py:137-139` reject；`CollectWizard.tsx:1097-1104` 展示 |
| C-021 | 01-phase1 (F6) | 语义一致性 | GENERAL | `sionna_rt_mock` 在 registry 中注册但不在 `SourceType` enum 和平台 API 中，mock 样本以 `source="sionna_rt"` 写出 | `sionna_rt.py:1453-1462` 注册；`contract.py:58-65` 不含；`sionna_rt.py:1605-1622` source 写 sionna_rt |
| C-022 | 08-schema (R-08-04) | Schema | HIGH | `h_interferers` 可为 None 即使 `meta.num_cells>1`，无一致性校验 | `contract.py:139-142` 可选；`:282-292` 只在非 None 时检查 shape |
| C-023 | 09-scale (R-09-01) | 可复现性 | MEDIUM | 500 组小配置 (`internal_sim` 1x1 ant, 1 RB) 不能代表真实大规模能力 | `stage3_scale_stability.json` |
| C-024 | 09-scale (R-09-02), 15-perf (R-15-01) | 可复现性 | HIGH | `samples_per_sec` 不包含 source 初始化时间，低估端到端耗时（4.38s vs 7.53s） | `run_simulate.py:134-151` t0 在 source 构造之后 |
| C-025 | 09-scale (R-09-03) | 可复现性 | HIGH | 缺少内存/GPU 稳定性证据（`psutil` 不可用） | `stage3_scale_stability.json` |
| C-026 | 09-scale (R-09-04) | 可复现性 | HIGH | 并行数据隔离仅做弱验证（canonical hash 无碰撞，但无 run_id/manifest 事务隔离） | 同上 |
| C-027 | 11-3gpp (R-11-03) | 信道真实性 | HIGH | ASA/ASD 不在 `ChannelSample` 输出字段，无法从输出对标 38.901 | `contract.py:131-194` 无 ASA/ASD |
| C-028 | 11-3gpp (R-11-04) | 信道真实性 | HIGH | PDP（功率延迟谱）不是一等输出字段，只有频域 CFR | 同上 |
| C-029 | 11-3gpp (R-11-05) | 信道真实性 | BLOCKER | `sionna_rt` TDL fallback 的 `pathloss_serving_db` 符号错误（记录 `rx_power - tx_power`，为负值） | `sionna_rt.py:1387-1388`；`internal_sim` 记录正值 `pl_all[k]` |
| C-030 | 11-3gpp (R-11-06) | 信道真实性 | MEDIUM | UMi/InF 路损模型标注为 simplified，不能视为完整 38.901 实现 | `internal_sim.py:92-103` 注释 |

---

**合计 30 条具体问题**

- BLOCKER：14 条（C-001~C-009, C-013, C-014, C-017, C-029 + C-005 归入 BLOCKER）
- SEVERE：4 条（C-011, C-012, C-019, C-020）
- HIGH：9 条（C-015, C-016, C-018, C-022, C-024~C-028）
- MEDIUM：3 条（C-010, C-023, C-030）
- GENERAL：1 条（C-021）
