# 99 - 疑问清单

记录迁移过程中看不懂或有疑问的代码/设计。
**禁止猜测，必须记录后向项目负责人确认。**

---

## 待确认

### Q-01: quadriga_single_legacy.py 是否保留？

**位置**: `src/msg_embedding/data/sources/quadriga_single_legacy.py`

文档只列出 5 种数据源，未提及此文件。它是 QuaDRiGa 单小区的遗留适配器。
`quadriga_multi.py` 已覆盖单小区场景（选最强小区作 serving）。

**问题**: 迁移时是否保留此文件？还是已被 quadriga_multi 完全替代？

---

### Q-02: model.py 的 cfg dict 接口与 config.py 常量的优先级

**位置**: `D:\MSG\model.py` 第 6-24 行

```python
try:
    from config import *
except ImportError:
    TOKEN_DIM = 128
    LATENT_DIM = 256
    ...
```

ChannelMAE 构造函数同时接受 `cfg` dict 参数和使用模块级全局常量作为 `.get()` 的默认值。
当 `cfg` 中缺少某个键时，会 fallback 到 `config.py` 的全局值或上面的硬编码默认值。

**问题**: 现有训练出的 checkpoint .pth 文件中，模型是用 `cfg` dict 还是全局常量实例化的？
这决定了迁移时是否需要保持 fallback 链的行为一致。

---

### Q-03: 文件队列 vs Dramatiq 的实际使用关系

**位置**: `platform/worker/queue_watcher.py` + `platform/worker/actors.py`

Worker 有两套任务接收机制：
1. 文件队列（queue_watcher 轮询 `queue/` 目录下的 JSON 文件）
2. Dramatiq actor（通过 Redis 消息）

**问题**: 生产环境实际走哪条路径？两者是互补（不同任务类型走不同通道）还是文件队列是主路径？
这影响迁移时的保留策略。

---

### Q-04: gol.py 全局状态在信号处理链中的生命周期

**位置**: `D:\MSG\gol.py` + `CsiChanProcFunc.py` + 其他信号处理模块

`gol.py` 提供全局字典，被多个信号处理模块共享。

**问题**:
- `gol._init()` 在哪里调用？每个 sample 初始化一次还是全局一次？
- 哪些键被设置、被哪些模块读取？是否存在跨模块的隐式依赖？
- 迁移为类内状态时，是否会破坏信号处理链的调用顺序假设？

---

### Q-05: Worker 的 Python 路径检测逻辑

**位置**: `platform/worker/settings.py` 第 55-61 行

```python
_venv = Path("D:/MSG/.venv312/Scripts/python.exe")
python_exe: str = str(_venv) if _venv.exists() else sys.executable
```

**问题**: 生产部署时是否确实使用 `.venv312` 这个特定虚拟环境名？
Docker 环境中此路径不存在，会 fallback 到容器内的 Python。
迁移时这个路径应该完全配置化还是有特定约束？

---

### Q-06: CQI 表的 16 个 SINR 阈值来源

**位置**: `data/bridge.py` 中的 CQI 映射表

```python
cqi_thresholds = [0.1523, 0.2344, 0.3770, 0.6016, ...]  # 16 个值
```

**问题**: 这些值是 3GPP TS 38.214 Table 5.2.2.1-2/3/4 中的哪个？
是 CQI Table 1（QPSK/16QAM/64QAM）还是 Table 2（256QAM）还是 Table 3（1024QAM）？
迁移时需确认是否有其他 CQI 表变体需要支持。

---

### Q-07: 训练损失权重调度的物理含义

**位置**: `config.py` 第 148-153 行（训练配置）

损失权重调度：
- 前 10% epochs: recon=1.5, contrastive=0.2
- 10-30% epochs: recon=1.0, contrastive=0.5
- 30-100% epochs: recon=0.8, contrastive=1.0

**问题**: 这个调度是经验调参结果还是有理论依据？
迁移时应作为可配置参数（已在 Hydra YAML 中）还是作为硬编码的"已验证最佳实践"？

---

## 已确认

（迁移过程中确认的问题移到这里）
