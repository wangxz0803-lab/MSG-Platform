# 02 - 逐条裁决

裁决时间：2026-04-23
裁决人：实现方（Claude Opus）
裁决标准：独立交叉验证代码、样本、Codex 证据后给出

## 裁决枚举

- **Confirmed**：Codex 说的对，证据充分，问题真实存在
- **Partially Valid**：现象存在，但描述或严重程度需修正
- **False Positive**：Codex 搞错了，证据说明不成立
- **Need Decision**：涉及设计取舍，需人工决策

---

### C-001 | `quadriga_real` 丢失多小区 CIR

**裁决：Partially Valid（严重程度需大幅下调，非阻塞）**

**2026-04-23 更正**：Codex 的发现在审查时完全正确。此问题已根据审查反馈修复 — 实现了干扰感知 LS 信道估计方案，原 BLOCKER 已解决。

**Codex 发现时的状态（已修复）**：
- `quadriga_real.py:270-273` 写 `h_interferers=None`
- `.mat` 文件中只有 `Hf_serving_est` / `Hf_serving_ideal`，没有 `Hf_per_cell`

**根据审查反馈实施的修复方案**：
- MATLAB `ul_srs_pipeline.m` 实现了干扰感知信道估计：信号模型 `Y = H_serving*X_s + Σ(H_k*X_k) + noise`，然后 LS 估计 `H_hat = Y * conj(X_s) / |X_s|^2`
- Python `_interference_estimation.py` 中 `estimate_channel_with_interference()` 是等效实现
- `Hf_serving_est` 现在不是 clean serving channel，而是在多小区干扰存在下的 LS 估计结果，天然包含干扰残留（ZC 序列互相关）
- `Hf_serving_ideal` 是 ground truth（无干扰）
- SIR/SINR 从 SRS 接收功率比计算
- `h_interferers=None` 现在是 **有意设计** — 干扰已烘焙进 `h_serving_est`

**当前状态**：问题已解决。平台选择了"干扰感知估计信道"而非"存储 raw per-cell CIR"的方案。(est, ideal) pair 编码了干扰信息，对下游 ML 训练有效。

**残留限制（P3）**：
- 如果未来需要 raw per-cell CIR（如 channel charting、多小区 CIR 可视化），当前架构不直接支持
- 建议在文档中明确说明此设计决策

---

### C-002 | `sionna_rt` 静默 fallback 但仍标记 `source="sionna_rt"`

**裁决：Confirmed**

**我的证据**：
- `sionna_rt.py:431-444`：构造函数在 Sionna 不可用时设置 `_use_real_sionna = False`，仅 log warning
- `sionna_rt.py:1185-1186`：fallback 路径进入 `_compute_channels_tdl()`
- `sionna_rt.py:1434`：`source="sionna_rt"` 是硬编码字符串，不区分 real/fallback
- `sionna_rt.py:1401`：`meta["sionna_rt_used"]` 记录了真实状态，但这是自由 meta 字段，不在 source enum 层面

**根因**：这是一个有意识的设计决策 — 目的是让平台在没有 Sionna（昂贵的 GPU 依赖）环境下也能跑通采集流程，用于开发、测试、CI。设计意图是合理的，但执行有缺陷：`source` 字段应该区分 `sionna_rt` 和 `sionna_rt_tdl_fallback`，或者至少在 `run_meta.json` 中如实记录 fallback 状态。

**Codex 判定阻塞级是合理的**：因为不看 meta 的下游消费者会把 TDL 样本当 RT 样本。

---

### C-003 | Sionna TDL fallback 多小区不是共享散射环境

**裁决：Confirmed，但 Codex 的 BLOCKER 严重程度需要讨论**

**我的证据**：
- `sionna_rt.py:1014-1032`：`_compute_channels_tdl()` 对每个 site 独立调用 `_generate_tdl_channel()`
- 每个 cell 的小尺度衰落使用独立 TDL tap 实现，不共享散射体
- 但几何（距离、路损、相对功率）是共享的：所有 cell 对同一个 UE 位置计算路损

**根因**：TDL 模型本身的固有局限 — 3GPP TR 38.901 的 TDL 是统计信道模型，不包含空间散射体的概念。要实现多小区共享散射环境，必须使用几何信道模型（GSCM，如 QuaDRiGa）或射线追踪（如 Sionna RT）。TDL fallback 本质上就只能做到"共享大尺度参数 + 独立小尺度衰落"。

**严重程度修正建议**：如果平台明确标记 fallback 样本（C-002 修复后），则此条降为"已知局限"而非阻塞。如果不标记，则确实阻塞。因此归结为 C-002 的修复。

---

### C-004 | 元数据缺少 `tool_version`, `code_version`, `git_commit`, `generated_at`

**裁决：Confirmed**

**我的证据**：
- `ChannelSample` 类（`contract.py:112-232`）没有这四个字段
- 各 source 的 `meta` dict 也没有包含这些信息
- `run_simulate.py:136-143` 的 `run_meta.json` 有 `started_at` 但没有 git commit 或 code version
- `created_at` 存在于 ChannelSample 中，但是样本级时间戳，不等于 `generated_at`（批次级）

**根因**：重构时优先实现信道生成核心逻辑，provenance 元数据列为后续迭代。属于"TODO 未完成"类问题。

**补充**：`created_at` 字段实际上承担了 `generated_at` 的部分角色。真正缺失的是 `git_commit` 和 `code_version`，这两个需要在构建/打包时注入。

---

### C-005 | 缺少统一 `units` 字段

**裁决：Partially Valid**

**我的证据**：
- 确实没有统一 `units` 字段
- 但字段命名已经包含单位后缀：`carrier_freq_hz`, `bandwidth_hz`, `distance_3d_m`, `pathloss_serving_db`, `noise_power_dBm`, `tx_power_dbm`
- 坐标系没有显式声明，但 `ue_position` 文档写明 `[3] float64 meters`
- 信道数据的单位是 complex64 无量纲（归一化后的传递函数），这是通信领域惯例

**根因**：设计选择 — 通过命名约定而非独立字段来表达单位。对于当前使用场景（单平台内部消费）这是可接受的，但对跨平台数据交换不够严谨。

**严重程度修正**：Codex 判 BLOCKER 偏重。建议降为 HIGH — 有改进空间但不阻塞核心功能。

---

### C-006 | Schema 无版本且多层不一致

**裁决：Confirmed**

**我的证据**：
- `ChannelSample` 无 `schema_version` — 确认
- `to_parquet_row()` 返回字段：`sample_id`(UUID str), `T`, `RB`, `BS_ant`, `UE_ant`, `num_interferers`, `array_path`, `meta_json` 等
- `MANIFEST_SCHEMA` 字段：`uuid`, `sample_id`(int32), `shard_id`(int32), `stage`, `status`, `hash`, `path` 等
- 两者几乎没有交集 — 不是"部分不一致"，是"两套完全不同的 schema"
- ORM `Sample` 模型又是第三套：`sample_id` 为字符串

**根因**：重构分阶段进行 — `MANIFEST_SCHEMA` 是从原工程继承的数据管理 schema，`to_parquet_row()` 是新 `ChannelSample` 添加的导出方法。两者未统一，因为 manifest 层和 contract 层在不同迁移阶段完成。

---

### C-007 | `bandwidth_hz=0` 静默生成样本

**裁决：Confirmed**

**我的证据**：
- `internal_sim.py:557-563`：`max(1, int(0 / (12 * 30000))) = max(1, 0) = 1`
- 带宽为零时 `_noise_power_dbm()` 返回 `kT + 0 + NF`（因为 `log10(0)` 走 else 分支返回 0）— 实际上 `bandwidth_hz > 0` 条件不满足，`bw_db = 0.0`
- 生成的样本有 1 个 RB、正常的信道矩阵、不合物理的噪声功率
- 没有任何环节拒绝这个配置

**根因**：缺少配置层物理约束校验。重构时专注于信道生成流程的正确性，未添加参数边界检查。

---

### C-008 | 零距离 BS/UE 静默生成非物理路损

**裁决：Partially Valid**

**我的证据**：
- `internal_sim.py:87`：`d_3d = max(d_3d, 1.0)` — 内部做了 clamp
- 路损计算本身是有效的（在 1m 距离下计算）
- 但 meta 记录 `distance_3d_m=0.0` 而路损基于 1m 计算 — meta 和实际计算不一致
- `pathloss_serving_db=-4.82` 对 1m 距离实际上也是物理合理的（极近距离的 UMa NLOS）

**根因**：代码确实做了防护（clamp 到 1m），但两个问题：(1) 没有对用户报错说明 0m 是无效配置 (2) meta 记录的是原始请求值 0m 而非实际使用值 1m。

**严重程度修正**：Codex 判 BLOCKER。考虑到内部有 clamp，问题主要在 meta 不一致，建议降为 HIGH。但零距离作为配置输入确实应该被拒绝。

---

### C-009 | 零 BS 天线生成 NaN 功率样本

**裁决：Confirmed**

**我的证据**：
- `internal_sim.py:566-574`：天线数直接取值，无 >=1 校验
- `num_bs_ant=0` → 信道 shape `[T, RB, 0, UE]` → `np.mean(np.abs(h)**2)` 在空数组上返回 NaN
- `contract.py:312-335` 的 `_check_shape_consistency` 只检查 4D/5D 和 shape 匹配，不检查每个维度 > 0
- NaN 功率的样本会污染所有下游统计

**根因**：缺少前置校验。这是典型的边界条件遗漏。

---

### C-010 | 负频率运行时崩溃

**裁决：Confirmed**

**我的证据**：
- `internal_sim.py:88`：`20.0 * math.log10(fc_ghz)` 对负数抛 `math domain error`
- 崩溃发生在样本生成过程中，而非配置解析阶段
- 应该在 `validate_config()` 中检查 `carrier_freq_hz > 0`

**根因**：缺少配置校验。Codex 判 MEDIUM 合理。

---

### C-011 | `num_sites=3` 生成 7 个 cell

**裁决：Confirmed**

**我的证据**：
- `internal_sim.py:54-69`：`_sites_to_rings(3)` → ring=1（因为 `1+3*1*2=7 >= 3`）
- `hex_grid.py` 生成 ring-1 = 7 sites
- `internal_sim._build_sites()` 不裁剪，直接返回全部 7 sites
- 对照 `sionna_rt.py:1080`：`sites = sites[:self.num_cells]` — sionna_rt 做了裁剪
- 所以同样的 `num_sites=3`，sionna_rt 正确生成 3 cell，internal_sim 生成 7 cell

**根因**：`internal_sim` 缺少裁剪逻辑。`sionna_rt` 有相同的 hex grid 生成路径但加了截断。这是 `internal_sim` 的遗漏，不是 hex_grid 模块的 bug。

---

### C-012 | MATLAB 注释与实际输出/常量不一致

**裁决：Confirmed**

**我的证据**：
- `main_multi.m` 文件头声称输出 `Hf_multi`，实际保存 `Hf_serving_*` — 确认
- `run_quadriga_real.py:2-3` 注释 "10 shards x 500 UEs = 5000"，实际 `NUM_SHARDS=20`, `UES_PER_SHARD=50` = 1000 — 确认
- 这是文档腐化（doc rot），注释没有随代码更新

**根因**：代码演进过程中注释未同步更新。这不是功能 bug 但会误导维护者和审查者。

---

### C-013 | 采集样本不自动进入 manifest/DB

**裁决：Confirmed**

**我的证据**：
- `run_simulate.py:99-109`：`_save_sample()` 只写 `.pt` 文件到 output_dir
- 没有任何代码调用 `Manifest.append()` 或数据库写入
- 后端 `/api/datasets` 从 SQLite 查询，需要 manifest → DB 同步过程
- 平台 guide 描述的"采集后可查看"流程在代码中不成立

**根因**：`run_simulate.py` 是轻量级 CLI 工具，设计为"只管生成"；manifest 写入和 DB 入库是后续 pipeline 步骤（`run_full_pipeline.py` 做了部分）。但平台文档和 UI 暗示这是自动化流程。

---

### C-014 | Channels Explorer 期望旧字段名

**裁决：Confirmed**

**我的证据**：
- `channels.py:178-184`：读取 `sample.get("channel_ideal", {})` 和 `sample.get("channel_est", {})`
- `ChannelSample.to_dict()` 输出 `h_serving_true` 和 `h_serving_est`
- 字段名完全不匹配，Explorer 加载新样本会显示空内容

**根因**：重构 `ChannelSample` 时重命名了字段（`channel_ideal` → `h_serving_true`），但未同步更新 Channels Explorer 的读取逻辑。典型的跨层重命名遗漏。

---

### C-015 | 删除只删 DB 不删文件

**裁决：Confirmed**

**我的证据**：
- `datasets.py:96-108`：`db.query(Sample).filter_by(id=sample_id).delete()` 只操作数据库
- 没有删除 `.pt` 文件、manifest 条目或 bridge 产出的代码

**根因**：数据管理层尚未实现完整的生命周期管理。重构优先建立了 CRUD API 框架，删除的物理同步属于后续迭代。

---

### C-016 | 检索/筛选范围有限

**裁决：Confirmed**

**我的证据**：
- `datasets.py:46-65`：过滤参数只有 `source`, `link`, `min_snr`, `max_snr`
- 由于 provenance 元数据缺失（C-004），即使扩展 API 也没有底层数据支撑

**根因**：与 C-004 关联。数据库 schema 反映的是当前 `Sample` ORM 的有限字段。

---

### C-017 | `run_meta.json` fallback 字段硬编码 false

**裁决：Confirmed**

**我的证据**：
- `run_simulate.py:138`：`"fallback": False` 是字面常量
- 应该根据 source 实例的实际状态（如 `sionna_rt._use_real_sionna`）来设置
- 但 `run_simulate.py` 在写 `run_meta.json` 时 source 已经实例化，可以检查

**根因**：`run_meta.json` 的写入逻辑过于简化。`fallback` 字段应该是运行后的统计（有多少样本 fallback），而不是预设值。但当前写入时机是 source 实例化后、样本生成前，此时尚不知道会不会 fallback（对 sionna_rt 的逐样本 fallback 而言）。

**补充**：对于 `sionna_rt` source 级 fallback（整个 Sionna 不可用），其实在 source 构造后就已知 `_use_real_sionna=False`，可以立刻设置 `fallback=True`。

---

### C-018 | `.pt` 文件不逐比特可复现

**裁决：Partially Valid**

**我的证据**：
- 每个样本包含 `sample_id=uuid4()` 和 `created_at=datetime.now()`，这两个是非确定性的
- 物理信道数据（canonical hash）在同配置同种子下一致
- 这是可追溯性设计（每个样本有唯一 ID 和时间戳）和可复现性之间的权衡

**严重程度修正**：Codex 判 HIGH 合理。但"逐比特可复现"在很多 ML 数据 pipeline 中不是硬性要求，canonical hash 可复现是更实用的保证。建议保持 HIGH 但不升为 BLOCKER。

**根因**：设计选择 — `sample_id` 和 `created_at` 是 per-instance 标识，不是 per-content 标识。

---

### C-019 | 队列协议文档与代码不一致

**裁决：Confirmed**

**我的证据**：
- 文档描述 Redis 队列，代码使用 file-drop + Dramatiq
- 进度格式 `[progress] pct=X step=Y` 也与文档示例不同
- 这是文档未更新的问题

**根因**：重构时将队列机制从纯 Redis 改为 file-drop + Dramatiq 双通道，但文档未同步。

---

### C-020 | `quadriga_multi` 拒绝 `link=BOTH` 但 UI 展示

**裁决：Confirmed**

**我的证据**：
- `quadriga_multi.py:137-139`：显式检查并 raise ValueError
- `CollectWizard.tsx` 对所有 source 展示 UL/DL/BOTH 选项
- 用户在 UI 选择 BOTH + quadriga_multi 会在后端崩溃

**根因**：前端 UI 对所有 source 使用统一的选项集，没有按 source 差异化禁用不支持的选项。

---

### C-021 | `sionna_rt_mock` 可见性不一致

**裁决：Confirmed**

**我的证据**：
- `sionna_rt.py:1453-1462`：`sionna_rt_mock` 注册在 SOURCE_REGISTRY
- `contract.py:58-65`：`SourceType` literal 不包含 `sionna_rt_mock`
- mock 样本 source 写为 `"sionna_rt"` — 因为 ChannelSample 不接受 `"sionna_rt_mock"`

**根因**：mock source 是测试/开发辅助，设计时没有考虑将其作为正式 source type。但注册在全局 registry 又意味着 CLI 可调用它。

---

### C-022 | `h_interferers=None` 与 `num_cells>1` 无一致性校验

**裁决：Confirmed**

**我的证据**：
- `contract.py:139-142`：`h_interferers` 默认 None
- `contract.py:324-334`：shape validator 只在 `h_interferers is not None` 时检查
- 没有 model_validator 检查 `meta.num_cells > 1` 时 `h_interferers` 不得为 None
- `quadriga_real.py` 确实产出 `num_cells=7` + `h_interferers=None` 的样本

**根因**：`ChannelSample` 的设计允许 serving-only 模式（`h_interferers=None`），这对单小区场景是正确的。但缺少"多小区声明与数据一致性"的校验逻辑。

---

### C-023 | 500 组小配置不代表大规模能力

**裁决：Partially Valid**

**我的证据**：
- 500 组 `internal_sim` 1x1 天线、1 RB 确实是极小配置
- 但这是审查方环境限制，不是平台 bug
- 平台本身没有声称"已验证大规模稳定性"

**严重程度修正**：这不是平台的问题（defect），而是审查覆盖不足。Codex 判 MEDIUM 合理，但这应该标记为"审查限制"而非"平台缺陷"。

---

### C-024 | `samples_per_sec` 低估端到端耗时

**裁决：Confirmed**

**我的证据**：
- `run_simulate.py:134`：`source = source_cls(cfg)` 在第 134 行
- `run_simulate.py:151`：`t0 = time.monotonic()` 在第 151 行
- source 构造（含 MATLAB 启动、Sionna import 等）不计入时间
- 对于 `internal_sim` 差异不大，但对 `quadriga_real`（MATLAB 启动）和 `sionna_rt`（Sionna+CUDA init）差异可能很大

**根因**：CLI 计时点选择不当。这是一个简单的代码修复。

---

### C-025 | 缺少内存/GPU 稳定性证据

**裁决：Partially Valid**

**我的证据**：
- `psutil` 不是必需依赖，审查环境未安装
- 平台代码本身不包含内存/GPU 监控功能
- 这更多是"功能缺失"而非"bug"

**严重程度修正**：Codex 判 HIGH。这是功能期望而非缺陷。建议降为 MEDIUM（改进建议）。

---

### C-026 | 并行数据隔离弱验证

**裁决：Partially Valid**

**我的证据**：
- 并行任务输出到不同目录，canonical hash 无碰撞 — 这已经是有效的隔离证据
- 缺少的是"manifest 事务隔离"，但当前 `run_simulate.py` 本身就不写 manifest
- 这是 C-013 的衍生问题，不是独立缺陷

**根因**：与 C-013 数据管理链路断裂相关。

---

### C-027 | ASA/ASD 不在输出字段

**裁决：Partially Valid**

**我的证据**：
- 确实不在 `ChannelSample` 输出中
- 但 ASA/ASD 是大尺度参数统计量，不是信道传递函数的直接属性
- `internal_sim` 使用 LSP 表生成 delay spread / angle spread，通过 TDL 模型隐式体现在信道中
- 要验证 ASA/ASD 需要从 CFR 反推，这是分析步骤而非数据合约问题

**严重程度修正**：Codex 判 HIGH。这更准确的说是"信道特征输出不完整"，而非数据错误。建议保持 HIGH 但标记为功能增强。

---

### C-028 | PDP 不是一等输出

**裁决：Partially Valid**

**我的证据**：
- `ChannelSample` 输出频域 CFR，不直接输出 PDP
- PDP 可以从 CFR 通过 IFFT 计算，是后处理步骤
- 通信系统中频域 CFR 是更通用的表示，PDP 是导出量

**严重程度修正**：Codex 判 HIGH。建议降为 MEDIUM — PDP 作为导出量可以在分析层面计算，不需要在数据合约中强制。

---

### C-029 | `sionna_rt` TDL fallback pathloss 符号错误

**裁决：Confirmed**

**我的证据**：
- `sionna_rt.py:1387-1388`：
  ```python
  "pathloss_serving_db": (
      float(rx_power_dbm[serving_idx] - self.tx_power_dbm) ...
  )
  ```
  计算的是 `Prx - Ptx`，得到负值（如 -131.72 dB）
- `internal_sim.py:1260`：
  ```python
  "pathloss_serving_db": float(pl_all[serving_idx]),
  ```
  `pl_all[k]` 来自 `_compute_pathloss()` 返回的正值路损
- 两个 source 的 `pathloss_serving_db` 语义相反：internal_sim 是正的路损，sionna_rt 是负的"接收增益"
- 混合统计会得到荒谬的中位数

**根因**：`sionna_rt` 在编写 meta 时使用了错误的符号约定。应改为 `self.tx_power_dbm - rx_power_dbm[serving_idx]`。

---

### C-030 | UMi/InF 标注为 simplified

**裁决：Partially Valid — 属于已知限制而非缺陷**

**我的证据**：
- `internal_sim.py:92-103`：注释明确写 "simplified"
- 代码本身是诚实的，没有声称完整实现 38.901
- 这不是 bug，是 scope 限制

**严重程度修正**：Codex 判 MEDIUM 合理。但这更应标记为"已知限制"。

---

## 裁决汇总统计

| 裁决 | 数量 | 编号 |
|------|------|------|
| Confirmed | 21 | C-001~C-004, C-006~C-007, C-009~C-017, C-019~C-022, C-024, C-029 |
| Partially Valid | 8 | C-005, C-008, C-018, C-023, C-025~C-028, C-030 |
| False Positive | 0 | — |
| Need Decision | 1 | C-003（取决于 C-002 修复后是否还需独立标记） |

**Codex 整体命中率**：30/30 条现象都真实存在（0 条误判）。其中 21 条完全确认，8 条部分成立（严重程度需调整），1 条需要看前置修复结果决定。

**严重程度调整建议**：
- C-005：BLOCKER → HIGH
- C-008：BLOCKER → HIGH
- C-023：MEDIUM → 审查限制（非平台缺陷）
- C-025：HIGH → MEDIUM
- C-027/C-028：HIGH → MEDIUM（功能增强）
- C-030：保持 MEDIUM（已知限制）
