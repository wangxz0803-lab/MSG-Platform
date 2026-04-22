# ADR-001: 数据库策略

## 状态
已采纳

## 背景
原工程使用 SQLite + `Base.metadata.create_all()` 自动建表，无版本化迁移。
手动迁移文件 `add_job_run_id.py` 使用 SQLite 特有的 `PRAGMA table_info`。
单写入者限制在当前单用户研究平台规模下可接受。

## 选项

### 选项 A：SQLite + Alembic
- 优点：零基础设施、便携、加 Alembic 即可版本化迁移
- 缺点：单写入者、无并发工作者、2GB 实际上限

### 选项 B：PostgreSQL only
- 优点：完整 ACID、并发写入、JSON 列、全文检索
- 缺点：需额外运行 PG 服务，当前规模过重

### 选项 C：SQLite 开发 + PG 生产
- 优点：两者兼得，渐进升级
- 缺点：需维护双 DB 兼容迁移

## 决策
选择 A（SQLite + Alembic）。Schema 设计保持 ANSI SQL 兼容，后续可无缝切换 PG。

## 后果
- 正面：零外部依赖，开发部署简单
- 负面：多用户并发写入受限
- 风险：如需多写入者场景，需升级到 PG

## 日期
2026-04-22
