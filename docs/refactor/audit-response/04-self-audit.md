# 04 - 独立自检报告

自检时间：2026-04-23
自检人：实现方（Claude Opus）
原则：坦诚 > 体面；即使 Codex 没发现，也主动交代

---

## 2.1 重构过程中的自我坦白

### S-001 | SSB 测量中使用假干扰信道（`h_serving * 0.1`）

**位置**：`quadriga_multi.py:442-446`

```python
h_cells.append(
    h_interferers[int_pos]
    if h_interferers is not None
    else h_serving_true * 0.1  # ← 假干扰
)
```

**坦白**：当 `h_interferers=None`（如 K=1 或数据缺失）但 `enable_ssb=True` 时，SSB 测量需要所有小区的信道。此处用 serving channel 缩放 0.1 作为假干扰。这个 magic number 没有物理依据，纯粹为了让 SSB 测量不崩溃。

**影响**：SSB RSRP/RSRQ/SINR 在单小区或缺失干扰时是不可信的。`quadriga_real`（`h_interferers=None` + `num_cells=7`）的 SSB 结果尤其可疑。

**Codex 是否发现**：未发现。

---

### S-002 | SSB 测量失败静默吞掉（`except Exception: pass`）

**位置**：
- `quadriga_multi.py:468-469`
- `internal_sim.py` 和 `sionna_rt.py` 的 SSB 块使用 `except Exception as exc: logger.warning(...)` — 至少有日志

**坦白**：`quadriga_multi` 的 SSB 异常处理是最差的一个 — 纯 `pass`，不打日志。如果 SSB 测量抛异常，所有 SSB 字段静默为 None，用户不知道发生了什么。

**与 S-001 的关联**：如果假干扰 `h_serving * 0.1` 导致 SSB 计算异常，这个 except 会吞掉错误，输出看起来正常但缺少 SSB 数据。

**Codex 是否发现**：未发现。

---

### S-003 | `_dict_get()` 静默吞掉配置读取异常

**位置**：
- `internal_sim.py:47-50`
- `quadriga_multi.py:59-62`
- `sionna_rt.py:71-75`

```python
try:
    return cfg.get(key, default)
except Exception:
    pass
```

**坦白**：三个 source 共用的 `_dict_get()` 辅助函数。当 `cfg.get()` 抛出任何异常时，静默回退到 `getattr(cfg, key, default)`。这是为了兼容 Hydra OmegaConf 对象和普通 dict 的差异。但如果配置文件有语法错误或类型错误，不会有任何报错。

**影响**：低 — 实际触发概率小。但违反了"配置错误应尽早暴露"的原则。

**Codex 是否发现**：未发现。

---

### S-004 | UE 位置提取失败时静默回退到占位坐标

**位置**：
- `quadriga_real.py:260-263`：回退到 `[0.0, 0.0, 1.5]`
- `quadriga_multi.py:407-420`：回退到 `None`

**坦白**：当 MATLAB `.mat` 文件中的 `ue_positions` 字段格式异常时，不报错，直接使用占位值。这是为了让 pipeline 不因位置解析失败而中断。但下游如果用 `ue_position` 做 channel charting 或位置指纹，会得到错误结果。

**Codex 是否发现**：未发现。

---

### S-005 | `to_parquet_row()` 和 `MANIFEST_SCHEMA` 从未被统一使用

**坦白**：这两个组件是在不同阶段开发的，从未在同一个完整 pipeline 中联合测试过。

- `MANIFEST_SCHEMA`（`manifest.py:37-63`）继承自原工程的数据管理设计
- `to_parquet_row()`（`contract.py:484-521`）是新 `ChannelSample` 添加的接口
- `run_full_pipeline.py:58-77` 手写 manifest row，既不用 `to_parquet_row()` 也不完全遵循 `MANIFEST_SCHEMA`

**影响**：三套不一致的 manifest 写入方式。这不是"看不懂老代码而省略"，而是"分阶段开发未统一"。

**Codex 是否发现**：是（C-006），但未意识到 `run_full_pipeline.py` 是第三条独立路径。

---

### S-006 | `post_quadriga_pipeline.py` 中样本加载失败静默跳过

**位置**：`scripts/post_quadriga_pipeline.py:372-373`

```python
except Exception:
    pass
```

**坦白**：批量处理样本时，单个样本加载/特征提取失败会被静默跳过。没有计数、没有日志、没有错误汇总。如果大量样本有格式问题，pipeline 会悄悄产出比预期少得多的数据，没有任何警告。

**Codex 是否发现**：未发现。

---

### S-007 | `run_simulate.py` 中 MATLAB 子进程失败不提供诊断信息

**位置**：`quadriga_real.py:186-198`

```python
except Exception:
    config_path.unlink(missing_ok=True)
    return False
```

**坦白**：MATLAB 进程超时、崩溃、内存不足等所有异常都被吞掉，只返回 `False`。调用方 `_generate_all()` 只打印 `"WARNING: shard X failed"`，不提供失败原因。

**影响**：MATLAB 调试极其困难 — 你只知道"某个 shard 失败了"，不知道为什么。

**Codex 是否发现**：未发现。

---

### S-008 | `.tolist()` 调用不检查 NaN/Inf

**位置**：
- `internal_sim.py:1211-1214`：SSB 结果
- `quadriga_multi.py:456-460`：SSB 结果
- `sionna_rt.py:1341-1344`：SSB 结果
- 所有 source 中功率/距离/路损的 float() 转换

**坦白**：SSB 测量产出的 RSRP/RSRQ/SINR 数组直接 `.tolist()` 转为 Python list，不检查是否包含 NaN 或 Inf。如果输入信道有问题（如零天线 C-009），SSB 结果可能包含 NaN，这些 NaN 会通过 `ChannelSample` 的验证（因为 SSB 字段是 `list[float] | None`，不检查 finite）。

**Codex 是否发现**：未发现（Codex 检查了零天线生成 NaN 功率，但没追踪到 SSB 传播路径）。

---

### S-009 | Legacy PMI 路径依赖全局状态 `gol.py`

**位置**：`data/bridge.py:250-286`

**坦白**：当 `use_legacy_pmi=True` 时，会 import 根目录的 `gol.py`（全局状态字典）和 `CsiChanProcFunc.py`。这条路径：
1. 修改 `sys.path` 注入根目录
2. 调用 `gol._init()` 初始化全局状态
3. 使用硬编码的 `"8H4V"` 天线配置字符串

如果并发或多次调用，`gol._init()` 会重置全局状态，可能导致竞态条件。

**Codex 是否发现**：未发现。Codex 知道 `gol.py` 存在但没有分析其在 bridge 中的实际调用。

---

### S-010 | `interference_defaults()` 使用 NaN 作为默认值

**位置**：`data/bridge.py:294-304`

```python
def _interference_defaults():
    return {
        "sir_linear": float("nan"),
        "sinr_linear": float("nan"),
        "eigvals_top4": np.full(_INTF_TOP_K, np.nan, dtype=np.float32),
        ...
    }
```

**坦白**：当样本没有干扰数据时，干扰特征用 NaN 填充。这是有意为之 — 让下游模型能区分"无干扰数据"和"零干扰"。但如果下游代码没有 NaN-aware 的处理，这些 NaN 会污染训练。

**影响**：取决于下游。如果 ChannelMAE 的 loss 没有 `nan_to_num` 或 mask，NaN 会导致梯度爆炸。

**Codex 是否发现**：未发现。

---

## 2.2 Codex 视角覆盖不到的薄弱点

### S-011 | 内部数据流中的隐式假设

1. **`ChannelSample` 假设 complex64 精度足够**：所有信道张量强制 complex64。对于高 SNR 场景（>40dB），float32 的精度（约 7 位有效数字）可能不足以区分信号和噪声。这在普通使用中不会暴露，但在极端配置下可能导致数值问题。

2. **`_clamp_db()` 在 ChannelSample 创建前截断 SNR/SIR/SINR 到 [-49.9, 49.9]**：如果真实 SNR 为 60dB（如近距离 LOS），会被截断。这是为了满足 `ChannelSample` 的 `[-50, 50]` 范围校验，但丢失了信息。

3. **`h_serving_est` 和 `h_serving_true` 被同比例归一化**：`quadriga_real.py:254-258` 用 `h_ideal` 的平均功率归一化两者。这改变了绝对功率语义，但对相对精度（估计误差分析）是正确的。下游如果假设信道有物理尺度的功率，会得到错误结果。

### S-012 | 性能陷阱

1. **`ChannelSample.to_dict()` 调用 `.tolist()` 进行深拷贝**：对大信道矩阵（如 273 RB × 14 OFDM × 32 天线 × 2），`.tolist()` 会生成大量 Python 嵌套列表，内存膨胀 10-20 倍。对单个样本不明显，但批量处理时可能 OOM。

2. **`_load_mat()` 的 `simplify_cells=True`**：scipy loadmat 的 simplify_cells 会自动 squeeze 单维数组，导致 `Hf_multi` 的维度不确定（4D/5D/6D 都可能）。代码处理了 4D 和 5D 的情况（`quadriga_multi.py:270-275`），但没有处理 3D 或更低维度的情况。

3. **`sionna_rt._compute_channels_sionna()` 缓存整个 scene**：scene cache (`_rt_scene_cache`) 在整个 source 生命周期内保持。如果 scene 占用大量 GPU 显存，且 source 实例不被 GC，显存泄漏。

### S-013 | 并发/线程安全问题

1. **`gol._init()` 全局状态**：如上所述，多线程/多进程调用 bridge 时会有竞态。

2. **`run_simulate.py` 的 `progress.json` 写入**：用 tmp+rename 的方式写 progress（`:88-96`），在单进程下安全，但如果 worker 和 watcher 同时读写同一目录，可能读到空文件。

3. **`sionna_rt` 的 scene cache**：`_rt_scene_cache` 是实例级属性，同一 source 实例的 `iter_samples()` 在同一线程内是安全的。但如果同一实例被多线程调用，scene 的 `add(rx)` / `remove(rx_name)` 不是线程安全的。

### S-014 | 资源释放隐患

1. **`quadriga_real.py:297-298`**：`del Hf_est, Hf_ideal, d; gc.collect()` — 显式 GC。这说明作者知道内存是问题，但 GC 的时机可能不够（因为 `ChannelSample` 持有的 ndarray 仍在内存中直到下游消费完）。

2. **Sionna scene cache**：如 S-012 所述，没有 `__del__` 或 context manager 释放 scene。

3. **MATLAB 子进程**：`subprocess.run(timeout=7200)` — 2 小时超时。如果 MATLAB hang 但不退出，占用进程资源。没有 kill 逻辑。

### S-015 | 配置项之间的隐含耦合

1. **`num_sites` vs `sectors_per_site`**：`num_sites=3, sectors_per_site=3` 会通过 hex grid 生成 7 sites × 3 sectors = 21 cells。这个乘法关系在文档中不明显。

2. **`num_cells` (sionna_rt/quadriga_real) vs `num_sites` (internal_sim)**：不同 source 用不同的名字表达类似概念。sionna_rt 的 `num_cells` 经过裁剪是准确的；internal_sim 的 `num_sites` 经过 hex grid 放大后不准确。

3. **`channel_est_mode` 与 `pilot_type` 的耦合**：ideal 模式忽略 pilot_type；ls_linear 和 ls_mmse 需要有效的 pilot 配置。但 `validate_config()` 不检查这种跨字段一致性。

4. **`link=BOTH` 的含义因 source 而异**：`quadriga_multi` 拒绝；`internal_sim` 和 `sionna_rt` 支持；`quadriga_real` 不检查但只输出 UL。

### S-016 | 跨模块的隐式约定

1. **Bridge 假设 `h_serving_true` shape 是 `[T, RB, BS, UE]`**：如果上游改变维度顺序，bridge 会静默产出错误特征。没有 shape assertion。

2. **训练代码假设 features 是 `[B, 16, 128]`**：ChannelMAE 的 patch_embed 硬编码了 token 数和维度。如果 bridge 的特征提取配置改变，模型不会报错但会产出无意义的结果。

3. **前端 CollectWizard 的配置映射**：前端构建的 JSON 配置直接传递给后端和 CLI。如果前端字段名与 source `validate_config()` 中的字段名不一致，错误会在生成阶段才暴露。

---

## 2.3 风险清单

| 编号 | 风险描述 | 触发场景 | 影响范围 | Codex 已发现 | 修复优先级 |
|------|---------|---------|---------|------------|-----------|
| S-001 | SSB 测量使用假干扰 `h_serving*0.1` | K>1 且 `h_interferers=None`（如 quadriga_real） | SSB RSRP/RSRQ/SINR 不可信 | ❌ | P1 |
| S-002 | quadriga_multi SSB 异常静默吞掉 | SSB 计算抛异常 | SSB 字段无预期地为 None | ❌ | P1 |
| S-003 | `_dict_get()` 吞掉配置异常 | 配置文件有语法/类型错误 | 使用默认值而非报错 | ❌ | P2 |
| S-004 | UE 位置回退到占位坐标 | .mat 中 ue_positions 格式异常 | channel charting/位置指纹错误 | ❌ | P2 |
| S-005 | manifest 写入三条独立路径 | 使用不同 pipeline 入口 | 数据不一致 | 部分（C-006） | P1 |
| S-006 | 批量处理静默跳过失败样本 | 样本文件损坏或格式不兼容 | 产出数据量少于预期 | ❌ | P1 |
| S-007 | MATLAB 失败无诊断信息 | MATLAB OOM/超时/license 失败 | 无法调试 | ❌ | P2 |
| S-008 | `.tolist()` 不检查 NaN/Inf | 零天线/零带宽/异常输入 | NaN 传播到下游 | ❌ | P1 |
| S-009 | Legacy PMI 依赖全局状态 | 并发调用 bridge + `use_legacy_pmi=True` | 竞态条件/数据污染 | ❌ | P2 |
| S-010 | 干扰默认值使用 NaN | 单小区样本或缺少干扰数据 | 训练时梯度爆炸（如果 loss 不 mask） | ❌ | P2 |
| S-011 | complex64 精度和 SNR 截断 | 高 SNR (>40dB) 或极近距离 | 数值精度损失 | ❌ | P3 |
| S-012 | `to_dict().tolist()` 内存膨胀 | 大天线阵列批量处理 | OOM | ❌ | P2 |
| S-013 | 并发安全问题 | 多线程/多进程调用 source | 竞态/数据污染 | ❌ | P3 |
| S-014 | Sionna scene cache 无释放机制 | 长时间运行 + GPU 场景 | 显存泄漏 | ❌ | P2 |
| S-015 | num_sites × sectors 隐含乘法 | 用户配置 sectors_per_site=3 | 实际 cell 数远超预期 | 部分（C-011） | P1 |
| S-016 | 跨模块隐式 shape/dim 约定 | 修改 bridge 或 model 配置 | 静默产出错误结果 | ❌ | P3 |
