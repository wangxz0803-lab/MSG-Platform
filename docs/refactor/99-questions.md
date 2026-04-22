# 99 - 疑问清单

记录迁移过程中看不懂或有疑问的代码/设计。
**禁止猜测，必须记录后向项目负责人确认。**

---

## 待确认

### Q-05: Worker 的 Python 路径检测逻辑

**位置**: `platform/worker/settings.py` 第 55-61 行

```python
_venv = Path("D:/MSG/.venv312/Scripts/python.exe")
python_exe: str = str(_venv) if _venv.exists() else sys.executable
```

**问题**: 生产部署时是否确实使用 `.venv312` 这个特定虚拟环境名？
Docker 环境中此路径不存在，会 fallback 到容器内的 Python。
迁移时这个路径应该完全配置化还是有特定约束？

**处理**: 迁移时改为完全配置化（`MSG_WORKER_PYTHON_EXE` 环境变量），默认 `sys.executable`。

---

### Q-06: CQI 表的 16 个 SINR 阈值来源

**位置**: `data/bridge.py` 中的 CQI 映射表

**问题**: 具体对应 3GPP TS 38.214 的哪个 Table？
**处理**: 迁移时原样保留数值，在 `protocol_spec.py` 中加注释标注来源。

---

## 已确认

### Q-01: quadriga_single_legacy.py 是否保留？
**答复**: 已被 quadriga_multi 完全替代，迁移时丢弃。(2026-04-22)

### Q-02: .pth checkpoint 向后兼容
**答复**: 现有 .pth 是测试产物，可不保留。ChannelMAE 迁移不需考虑旧 checkpoint 兼容。(2026-04-22)

### Q-03: 文件队列 vs Dramatiq
**答复**: 文件队列是主通道，Dramatiq 是辅助。迁移时两个都保留。(2026-04-22)

### Q-04: gol.py 生命周期
**答复**: 已通过代码追踪确认。gol 是单样本作用域的临时数据仓库——每次 PMI 计算都调 `_init()` 清空。
37 个键中大部分初始化为 None 从未读取，实际活跃键仅 5 个。迁移为类内状态安全可行。(2026-04-22)

### Q-07: 训练损失权重调度
**答复**: 不重要，作为 Hydra YAML 可配置参数保留即可。(2026-04-22)
