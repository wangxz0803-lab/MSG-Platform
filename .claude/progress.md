# MSG-Embedding 迁移重构 - 总体进度

## 当前状态：阶段 2 - 规范固化 + 骨架搭建（进行中）

| 阶段 | 状态 | 开始日期 | 完成日期 | 备注 |
|------|------|----------|----------|------|
| 0 - 工作空间建立 | 完成 | 2026-04-22 | 2026-04-22 | 目录结构、状态文件、ADR |
| 1 - 双向理解 | 完成 | 2026-04-22 | 2026-04-22 | 4 份文档已交付，7 个问题已确认 |
| 2 - 规范固化 + 骨架搭建 | 进行中 | 2026-04-22 | - | core/ 已完成，待继续骨架 |
| 3 - 按模块迁移 | 未开始 | - | - | P0→P1→P2 逐模块迁移 |

## 阶段 2 子任务

| 任务 | 状态 | 验收 |
|------|------|------|
| CLAUDE.md | 完成 | 已写入 |
| core/config.py | 完成 | 6 tests pass |
| core/logging.py | 完成 | 4 tests pass |
| core/exceptions.py | 完成 | 6 tests pass |
| core/protocol_spec.py | 完成 | 6 tests pass |
| core/types.py | 完成 | 6 tests pass |
| pyproject.toml | 完成 | pip install -e 成功 |
| Makefile | 完成 | - |
| .pre-commit-config.yaml | 完成 | - |
| .env.example | 完成 | - |
| Hydra configs/ | 待迁移 | - |
| Docker 配置 | 未开始 | - |
| CI 配置 | 未开始 | - |
| 04-migration-plan.md | 未开始 | - |

## Git 检查点

| Commit | 内容 |
|--------|------|
| 50fa1de | Phase 0 + Phase 1：工作空间 + 现状文档 |
| 281de51 | Phase 2.1：core/ 基础设施 + 项目骨架 |

## 已确认的架构决策

- 数据库：SQLite + Alembic
- 任务队列：Dramatiq + Redis（保留）
- 配置管理：Hydra ML + pydantic-settings 基础设施
- 日志框架：structlog
- 前端：React 18 + Ant Design（原样保留）
- 遗留信号处理：迁入 features/pmi,srs,ssb 子包
- quadriga_single_legacy：丢弃（已被 quadriga_multi 替代）
- .pth checkpoint：不需要向后兼容（测试产物）
- gol.py：单样本作用域，迁移为类内状态安全
