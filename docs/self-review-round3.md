# Round 3 专家评审 — 自审结果与修改计划

审查日期：2026-04-26
审查对象：`D:\MSG平台_codex\round3\review\` 全部 8 份报告
自审方法：逐项对照源码验证，给出"确认属实 / 部分属实 / 不属实"判定

---

## 一、总体自评

专家的核心结论是：**平台更接近"研究脚本的管理壳层"，尚不能被认定为按 5G 协议配置真实执行的数据平台。**

经逐项核查，**这个总体判断是成立的**。具体来说：

- 14 个编号问题（R3-001 ~ R3-014）中，**13 个确认属实，1 个部分属实**
- 8 个阻塞级问题全部确认存在
- 文档-实现不一致（DG-001 ~ DG-008）中，**8 个全部确认属实**

下面按主题逐项给出核查结论。

---

## 二、协议符合性（对应 03-5g-protocol-compliance.md）

### R3-001：internal_sim / sionna_rt 协议参数名义存在但未真正执行

**判定：部分属实（TDD 部分已初步接入，SRS 跳频确实未接入）**

**TDD pattern 部分 —— 已部分接入但效果有限：**

代码确实存在 TDD 集成：
- `internal_sim.py:1012-1019` 读取 `_tdd_pattern`，计算 `slot_direction`、`symbol_map`、`dl/ul/guard_symbol_mask`
- `_interference_estimation.py:137-138, 487, 504` 在估计链中使用 `valid_symbol_mask` 过滤导频放置时域位置
- paired 模式通过 `dl_symbol_mask` / `ul_symbol_mask` 控制 UL/DL 导频分别放在正确方向的符号上

**但专家探针结果（DDDSU vs DDSUU 完全一致）仍然有道理**：探针用 `idx=0`，此时 `slot_idx = 0 % 5 = 0`，而 slot 0 在 DDDSU 和 DDSUU 中都是 'D' 全下行时隙，符号掩码完全相同。如果探针在 `idx=3`（DDDSU 是 S 特殊槽，DDSUU 是 U 上行槽）测试，结果会不同。

**根本局限**：TDD pattern 仅影响导频放置位置，不影响信道本身（`_generate_tdl_channel` 是 TDD 无关的统计衰落生成器）。真正的 5G 协议中，信道采样应按时隙方向调度（UL 时隙采 UL，DL 时隙采 DL），当前实现是一次性生成全部 T 个 OFDM 符号的信道矩阵，然后用掩码选择估计位置，这不等于真正的 TDD 时域调度。

**SRS 跳频部分 —— 确认未接入主链：**

- `srs.py` 中有完整的 `srs_group_number()` 和 `srs_freq_position()` 实现
- `srs_sequence()` 内部会调用 `srs_group_number()`，所以 ZC 序列的组号/序列号跳频在序列生成层面是生效的
- **但 `srs_freq_position()`（频域跳频位置计算）从未被主链调用**
- 验证方式：`grep -r "srs_freq_position" src/` 只在 `srs.py` 和 `__init__.py` 中出现
- 专家探针中 `group_hopping=True` 后估计信道与 baseline 完全一致（SHA256 相同），说明即便序列层面的跳频也没有产生可观测的输出差异

**结论：TDD 有初步集成但远非协议级调度；SRS 频域跳频完全未接入。专家判断基本成立。**

---

### R3-002：quadriga_real DL/BOTH 未形成可信主链

**判定：确认属实**

证据：
- `matlab/main_multi.m` 只调用 `ul_srs_pipeline()`（1次），`dl_csirs_pipeline()` 调用次数为 0
- Python 包装层 `quadriga_real.py:308-345` 通过 TDD 互易性（共轭转置）从 UL 信道派生 DL 信道，这不是真正的 DL CSI-RS 采集
- `_matlab_config`（`quadriga_real.py:200-219`）不包含 `link`、`channel_est_mode`、`tdd_pattern`、`num_interfering_ues` 等关键协议参数
- MATLAB 端 `dl_csirs_pipeline.m:88-92` 的 `ls_mmse` 模式直接回退到 `ls_linear` 并打印 TODO

---

### 前端 SSB / DMRS(PUSCH) 选项后端不支持

**判定：确认属实**

证据：
- `CollectWizard.tsx:342-349` 提供 `ssb` 和 `dmrs_pusch` 选项
- `internal_sim.py:600-604` 和 `sionna_rt.py:560-564` 只接受 `srs_zc` / `csi_rs_gold`，其他值直接抛 `ValueError`
- `ref_signals/dmrs.py` 有独立实现但主链从未调用（`grep -r "from.*dmrs import" src/` 搜索结果为空）

---

### 单链路 UL 的 pilot_type_ul 映射失败

**判定：确认属实**

证据：
- 前端在单链路 UL 时把 `pilot_type_ul` 映射到通用 `pilot_type`（`CollectWizard.tsx:528-535`）
- 但如果只设 `pilot_type_ul='srs_zc'` 而不设 `pilot_type`，源端仍使用默认 `csi_rs_gold`
- 专家探针验证：`observed_meta_pilot_type = "csi_rs_gold"`（应为 `srs_zc`）

---

## 三、数据格式（对应 04-channel-data-format.md）

### Paired UL/DL 维度语义不一致

**判定：确认属实**

证据：
- `contract.py:195-209` 文档写 `h_ul_est: [T, RB, BS_ant, UE_ant]`
- `bridge.py:527` 注释写 `h_ul_est stored as [T, RB, UE, BS]; transpose to [T, RB, BS, UE]`
- 实测 paired 样本：`h_serving_true = [14,85,64,4]`（BS=64, UE=4），`h_ul_true = [14,85,4,64]`（UE=4, BS=64）
- 合同文档与实际存储维度顺序相反，bridge 必须手动 transpose 才能使用

---

### Manifest 方言不一致

**判定：确认属实**

证据：
- `manifest.py:43` schema 定义 `shard_id: int32`、`sample_id: int32`
- `run_simulate.py:134` 写入 `shard_id: 0`（int）
- `run_full_pipeline.py:61` 写入 `shard_id: "shard_0000"`（str），`sample_id: s.sample_id`（UUID str）
- 类型不一致会导致 parquet 类型强制转换或数据丢失

---

### 元数据不完整

**判定：确认属实**

缺失项：
- 机器名、操作系统信息
- 代码版本 / git commit
- 关键库版本（QuaDRiGa / Sionna / NumPy）
- 数据 schema 版本号
- 哈希校验值（列存在但默认 `None`）
- 完整轨迹信息（只有当前 `ue_position`）

---

## 四、数据管理（对应 05-data-management.md / 06-issues.md）

### R3-003：run_bridge.py 缺失

**判定：确认属实**

证据：
- `D:\MSG平台_cc\scripts\run_bridge.py` 文件不存在（Glob 搜索无结果）
- `platform/worker/tasks/base.py:31` 把 `bridge` 映射到 `run_bridge.py`
- `DataProcess.tsx:190` 前端会提交 bridge 作业
- 提交 bridge job 后必然 `FileNotFoundError`

---

### R3-004：官方训练入口回退到 FakeDataset

**判定：确认属实**

证据：
- `pretrain.py:172-186` 的 `build_dataset(cfg)` 尝试 `ChannelDataset(cfg=cfg, split="train")`
- `dataset.py:55` 的 `ChannelDataset.__init__` 签名要求 `manifest` 参数，不接受 `cfg` 参数
- 任何异常都被 catch 并静默回退到 `_FakeChannelDataset`
- 用户无法从日志中明显察觉训练在吃假数据

---

### R3-005：官方评估入口对 raw dataset 不可用

**判定：确认属实**

证据：
- `runner.py:84-88` 的 `_feat_dict_from_record()` 要求 `record["feat"]`
- `dataset.py:99-127` 的 `__getitem__` 返回 `h_true/h_est/h_interferers/meta` 等，无 `feat` 键
- 结果：eval runner 对所有 raw 样本产出 `embeddings_used=0`

---

### R3-006：无数据集下载能力

**判定：确认属实**

证据：
- 后端路由中无任何 download/export 端点
- `endpoints.ts` 前端 API 封装中无 download 相关调用
- 仅有 `POST /api/models/{run_id}/export`（模型导出，非数据下载）

---

### R3-007：无平台级 split + 泄露防护不足

**判定：确认属实**

证据：
- 后端无 split 相关 API 路由
- `manifest.py:282-345` 的 `compute_split()` 只支持 `random / by_position / by_beam`
- 无 `user_id / trajectory_id` 等分组键
- 探针验证：单条轨迹在 `random` 策略下同时落入 train/val/test

---

### R3-008：paired UL/DL 元数据在 manifest/backend 中丢失

**判定：确认属实**

证据：
- `manifest_sync.py:22-80` 的 `_row_to_kwargs` 只映射 `uuid/sample_id/shard_id/source/link/snr_db/sir_db/sinr_db/num_cells/ts/status/job_id/run_id/path/split`
- **不映射**：`ul_sir_dB`、`dl_sir_dB`、`link_pairing`、`num_interfering_ues`
- DB 的 `Sample` 模型确实有这些列（`sample.py:26-29`），但 sync 逻辑不填充它们
- 结果：磁盘上有 4 个 paired 样本，但 `/api/datasets` 显示 `has_paired=false`

---

### R3-009：删除数据集后"僵尸复活"

**判定：确认属实**

证据：
- `datasets.py:116-128` DELETE 只删 DB 行，不删 manifest.parquet 和磁盘文件
- `datasets.py:70` GET 每次请求先调 `sync_manifest_to_db(db)`
- sync 从 manifest.parquet 读取并 upsert 回 DB → 刚删的数据被"复活"

---

### R3-010：50k 规模性能问题

**判定：确认属实**

证据：
- `datasets.py:70` 每次 GET 都全量 sync manifest
- `datasets.py:81-109` 聚合在 Python 内存做全表 `by_source` 分组
- 无 SQL 端聚合、无缓存、无增量 sync

---

### R3-011：manifest 路径硬编码

**判定：确认属实**

证据：
- `run_simulate.py:118` 硬编码 `bridge_out/manifest.parquet`
- 不利于多环境隔离和工作区管理

---

### R3-012：无权限/审计

**判定：确认属实**

证据：
- 所有路由无 auth dependency
- Job/Sample/Run 模型无 user_id/tenant_id 字段
- 删除/取消操作对任何调用方等权开放

---

### R3-013：前端 download_url 幻象字段

**判定：确认属实**

证据：
- `types.ts:143` 定义 `download_url?: string`
- `Models.tsx:40` 和 `RunDetail.tsx:101` 据此渲染下载链接
- `platform/backend/schemas/model.py:14-24` 无 `download_url` 字段

---

### R3-014：hash 概念存在但不执行

**判定：确认属实**

证据：
- `manifest.py:353` 有 `compute_content_hash()` helper
- `manifest.py:58` schema 有 `hash` 列
- `run_simulate.py` 写入 `hash=None`
- `manifest_sync.py` 不映射 hash 到 DB
- `dataset.py:105` 读取样本时无 hash 校验

---

## 五、文档-实现不一致（对应 02-doc-gap.md）

| 编号 | 专家发现 | 自审判定 |
|------|---------|---------|
| DG-001 | 文档写 Redis+Worker，实际是线程 | **属实** — `job_dispatch.py` 用 daemon thread，无 Redis |
| DG-002 | 文档写 quadriga_multi，实际无 Web 入口 | **属实** — CollectWizard 只有 quadriga_real |
| DG-003 | 文档写 `GET /api/datasets/:source`，实际不存在 | **属实** — 只有 `GET /api/datasets` |
| DG-004 | 文档写 `POST /api/runs`，实际不存在 | **属实** — runs 路由只有 GET/DELETE |
| DG-005 | 文档写 `GET /api/topology`，实际是 `POST /api/topology/preview` | **属实** |
| DG-006 | 信道浏览描述为"全量已入库样本"，实际是文件扫描 | **属实** |
| DG-007 | 模型下载声称有 ckpt/ONNX/TorchScript 下载 | **属实** — 后端无 download_url |
| DG-008 | `/jobs/new` simulate 会把 source 错发为 "simulate" | **属实** — 非法 source 枚举值 |

---

## 六、专家结论中需要补充说明的点

1. **TDD pattern 并非完全未接入**：代码层面已有初步集成（符号掩码传入估计链），但效果有限且探针设计未能触发差异场景。严格意义上是"部分接入但远非协议级调度"。

2. **SRS 序列级跳频已连通**：`srs_sequence()` → `srs_group_number()` 调用链存在，组号/序列号跳频在序列生成层面有代码实现。但频域跳频位置（`srs_freq_position()`）未接入，且即便序列跳频对估计输出也无可观测影响。

3. **平台定位问题**：专家评审按"商用可交付数据平台"标准评估，而当前平台实际定位更接近"单人研究环境"。auth/audit/tenant/性能等问题在单用户场景下优先级较低，但 bridge 缺失、训练回退假数据、eval 空输出等问题在任何场景下都是阻塞级的。

---

## 七、修改计划

按优先级分为 P0（阻塞级，必须立即修复）、P1（严重，短期修复）、P2（一般，中期完善）。

### P0 — 阻塞级修复（预计 1-2 周）

| 编号 | 问题 | 修复方案 | 涉及文件 |
|------|------|---------|---------|
| P0-1 | `run_bridge.py` 缺失，bridge 作业必然失败 | 补齐 `scripts/run_bridge.py`，调用 `bridge.py` 的 `BridgeProcessor`，接受 manifest/source/output_dir 参数，输出 featured shard 并注册到 manifest | `scripts/run_bridge.py`（新建）, `platform/worker/tasks/base.py` |
| P0-2 | 训练入口静默回退到 FakeDataset | 修改 `build_dataset()` 为显式 manifest 构造：`ChannelDataset(manifest=cfg.data.manifest_path, split=...)`；移除 silent fallback，改为 raise 并提示 manifest 路径 | `src/.../training/pretrain.py` |
| P0-3 | 评估入口对 raw dataset 产出 0 embedding | 在 `_feat_dict_from_record()` 中增加 raw→feat 转换路径：当 record 无 `feat` 时，调用 `BridgeProcessor.extract_features(record)` 即时转换 | `src/.../eval/runner.py` |
| P0-4 | 前端 SSB/DMRS 选项后端不支持 | 方案 A：前端移除 `ssb` / `dmrs_pusch` 选项（如果短期不打算实现）；方案 B：后端接入 `dmrs.py` 实现（工作量大）。**建议先用方案 A** | `CollectWizard.tsx` |
| P0-5 | quadriga_real 协议参数不传 MATLAB | 把 `link`、`channel_est_mode`、`tdd_pattern`、`num_interfering_ues` 加入 `_matlab_config`；MATLAB 端 `main_multi.m` 相应读取并条件调用 `dl_csirs_pipeline()` | `quadriga_real.py`, `matlab/main_multi.m` |
| P0-6 | JobCreate simulate source 错误 | `JobCreate.tsx` 中 `type === "simulate"` 分支改为选择具体 source 枚举值 | `JobCreate.tsx` |

### P1 — 严重问题修复（预计 2-4 周）

| 编号 | 问题 | 修复方案 | 涉及文件 |
|------|------|---------|---------|
| P1-1 | paired UL/DL 维度不一致 | 统一存储为 `[T, RB, BS_ant, UE_ant]`，在 `_interference_estimation.py` 输出时就 transpose 到合同维度，而非留给 bridge 处理 | `_interference_estimation.py`, `contract.py`, `bridge.py` |
| P1-2 | manifest 方言不一致 | 统一 `shard_id: int`, `sample_id: int`；`run_full_pipeline.py` 改用整数 shard 编号；`contract.py:to_parquet_row()` 与 `run_simulate.py` 对齐 | `run_full_pipeline.py`, `contract.py`, `run_simulate.py` |
| P1-3 | manifest sync 丢失 paired/interference 字段 | `_row_to_kwargs` 增加 `ul_sir_dB`、`dl_sir_dB`、`link_pairing`、`num_interfering_ues` 映射 | `manifest_sync.py` |
| P1-4 | 删除数据集"僵尸复活" | DELETE 时同步删除 manifest.parquet 中对应 source 的行；或改为软删除（标记 status=deleted 而非物理删除） | `datasets.py`, `manifest.py` |
| P1-5 | GET /api/datasets 性能问题 | 将 `sync_manifest_to_db` 从读路径移出（改为写入时触发或定时刷新）；聚合下推到 SQL `GROUP BY` | `datasets.py`, `manifest_sync.py` |
| P1-6 | 单链路 UL pilot_type_ul 映射失败 | 前端 → 后端的配置组装逻辑中，单链路模式下应把 `pilot_type_ul/dl` 正确映射到 `pilot_type` | `CollectWizard.tsx` |
| P1-7 | SRS 频域跳频未接入 | 在 `_generate_pilots_srs()` 中调用 `srs_freq_position()` 计算导频频域起始位置，使跳频真正影响估计输出 | `internal_sim.py`, `_interference_estimation.py` |
| P1-8 | 前端 download_url 幻象字段 | 后端 `ModelArtifactSchema` 添加 `download_url` 字段并生成真实下载 URL；或前端移除无后端支持的下载按钮 | `model.py`, `models.py` 路由, 或 `types.ts` |
| P1-9 | manifest 路径硬编码 | `run_simulate.py` 接受 `--manifest-path` 参数；默认值从 backend settings 或环境变量读取 | `run_simulate.py` |

### P2 — 中期完善（预计 1-2 月）

| 编号 | 问题 | 修复方案 |
|------|------|---------|
| P2-1 | 无数据集下载 API | 新增 `GET /api/datasets/{source}/download`，支持条件筛选后打包下载（tar/zip），附带 manifest 子集和 hash 校验 |
| P2-2 | 无平台级 split | 新增 `POST /api/datasets/{source}/split`，支持 random/by_position/by_beam + trajectory group，结果持久化到 manifest `split` 列并注册版本 |
| P2-3 | 泄露防护 | 引入 `trajectory_id` / `config_group_id` 作为 split group key；`compute_split()` 增加 group-aware 策略 |
| P2-4 | 元数据完整性 | 样本和 sidecar 中补充：git commit、库版本、机器名、schema 版本号 |
| P2-5 | hash 校验执行 | `run_simulate.py` 写入时计算 hash；`ChannelDataset` 读取时可选校验；manifest sync 映射 hash 到 DB |
| P2-6 | 文档-实现同步 | 全面更新 `D:\MSG\guide\05_platform_guide.md`，修正 API 速查、Worker 架构、source 枚举等不一致项 |
| P2-7 | TDD 真实调度 | 将信道生成改为按时隙方向调度：UL 时隙只生成 UL 信道、DL 时隙只生成 DL 信道，而非一次生成全部后用掩码选择 |
| P2-8 | 权限与审计 | 补 auth middleware（JWT/API key）、用户模型、操作审计日志 |
| P2-9 | 三元组一致性 | DB 建立 raw sample → featured sample → training run 的关联关系 |

---

## 八、修改计划执行路线图

```
Week 1-2 (P0):
  ├── P0-1  补 run_bridge.py
  ├── P0-2  修 build_dataset() 不走 FakeDataset
  ├── P0-3  修 eval runner 支持 raw dataset
  ├── P0-4  前端移除不支持的 pilot 选项
  ├── P0-5  quadriga_real 协议参数透传 MATLAB
  └── P0-6  修 JobCreate simulate source bug

Week 3-4 (P1 前半):
  ├── P1-1  统一 paired UL/DL 维度
  ├── P1-2  统一 manifest 方言
  ├── P1-3  manifest sync 补全 paired 字段
  └── P1-4  修复删除"僵尸复活"

Week 5-6 (P1 后半):
  ├── P1-5  GET /api/datasets 性能优化
  ├── P1-6  修 pilot_type_ul 映射
  ├── P1-7  SRS 频域跳频接入
  ├── P1-8  修 download_url 幻象字段
  └── P1-9  manifest 路径可配置

Month 2-3 (P2):
  ├── P2-1~3  下载/split/泄露防护
  ├── P2-4~5  元数据+hash
  ├── P2-6    文档同步
  └── P2-7~9  TDD 真实调度 + auth + 三元组
```

---

## 九、对专家建议优先级的认同与调整

专家建议的 P0 是"修 bridge/train/eval 真数据闭环"和"补 download + split"。我们的调整：

1. **完全认同 bridge/train/eval 闭环为 P0** — 这是平台作为数据平台的基本功能
2. **download/split 降为 P2** — 在单人研究场景下，短期可用文件系统直接访问；平台化下载/划分重要但不阻塞当前使用
3. **新增前端-后端一致性为 P0** — SSB/DMRS 幻象选项和 JobCreate bug 会让用户直接踩坑，应立即修复
4. **quadriga_real 协议透传提升到 P0** — 这是上一轮工作的遗留项，且专家已指出这是可信度核心问题
