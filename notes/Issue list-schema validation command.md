### 建议的 GitHub Projects 设定与任务分解

- **项目类型与视图**
  - 在 GitHub Projects（Beta）创建一个新 Project：`Schema Validation Command`
  - 视图
    - Board：按 `Status` 分列（Todo/In Progress/In Review/Done）
    - Table：显示自定义字段（Type、Area、Priority、Milestone、Risk、Coverage、ExitCodesVerified）
    - Roadmap：Group by `Milestone`
    - PRs：筛选关联 PR 的条目，便于跟踪评审与合并

- **自定义字段**
  - `Status`（单选）：Todo / In Progress / In Review / Done
  - `Type`（单选）：Epic / Feature / Task / Bug / Docs / CI
  - `Area`（单选）：CLI / Core / Shared / Docs / CI
  - `Priority`（单选）：P0 / P1 / P2
  - `Milestone`（单选）：vX.Y.0
  - `Risk`（单选）：Low / Medium / High
  - `Estimate`（数字，点数）
  - `Coverage`（数字，%，目标≥80）
  - `ExitCodesVerified`（复选）
  - `Blocks/Blocked by`（关系型，建立任务依赖）

- **统一标签（Labels）**
  - `type:feature`, `type:epic`, `type:task`
  - `area:cli`, `area:core`, `area:shared`, `area:docs`, `area:ci`
  - `priority:p0|p1|p2`
  - `risk:low|medium|high`
  - `good-first-task`（可选）

- **里程碑（Milestone）**
  - `vX.Y.0 - Schema Validation Command`（SemVer 次版本号，目标发布日期）

---

### Issue / 任务清单（可直接在项目中批量创建）

- Epic: Feature - Schema Validation Command
  - Type: Epic | Area: All | Priority: P0 | Milestone: vX.Y.0 | Risk: Medium
  - DoD
    - 所有验收标准满足，测试覆盖率≥80%，文档与变更日志更新，预提交钩子通过，遵循 PEP8/Black/isort/mypy

- Feature: CLI command `schema` – command skeleton
  - Type: Feature | Area: CLI | Priority: P0
  - AC
    - 新增命令 `vlite schema "data-source" --rules schema.json`
    - 解析基础参数，支持 `--output`, `--fail-on-error`, `--max-errors`, `--verbose`
    - 输出与 `check` 风格一致（table/JSON）
    - Exit codes: 0/1/2 实现并测试
  - Links: 设计文档、Feature Request

- Task: Rules file validation (minimal, no jsonschema)
  - Type: Task | Area: CLI | Priority: P1
  - AC
    - 仅支持单表，无顶层 `table`；若发现 `table` 字段，发出警告且忽略
    - 校验 `rules[*].field/type/required/enum/min/max` 的基本结构和类型
    - 错误时返回 Exit code 2

- Task: Decompose schema rules → atomic rules mapping
  - Type: Task | Area: CLI | Priority: P0
  - AC
    - 基于 JSON 将每条规则分解为：Schema(存在+类型)、not_null、range(min/max)、enum
    - 使用 `shared/enums` 中的类型枚举，不使用字符串字面量
    - 使用 `shared/utils` 的日志/错误工具

- Feature: Core – add `Schema` rule type
  - Type: Feature | Area: Core | Priority: P0 | Risk: Medium
  - AC
    - 新增 `Schema` 规则：校验字段存在与类型匹配（严格匹配，无类型转换）
    - 使用 `shared/enums` 类型枚举与 `shared/utils` 工具
    - 与既有引擎执行/注册流程无缝集成
    - 单元测试覆盖：存在/缺失、类型匹配/不匹配

- Task: Aggregation & Prioritization in CLI
  - Type: Task | Area: CLI | Priority: P0
  - AC
    - 每字段评估顺序：存在 → 类型 → not_null → range/enum
    - 字段缺失：报告 `FIELD_MISSING`，后续检查标记 `SKIPPED`
    - 类型不匹配：报告 `TYPE_MISMATCH`，后续检查 `SKIPPED`
    - 聚合输出：人类可读输出仅显示根因；JSON 输出包含 `SKIPPED` 信息

- Task: Output formatting + JSON schema for results
  - Type: Task | Area: CLI | Priority: P1
  - AC
    - table 默认输出；`--output json` 输出聚合后的结构
    - 汇总总检查数、失败数、字段级详情
    - 文本输出与既有 `check` 风格一致

- Task: Data-source resolution parity with `check`
  - Type: Task | Area: CLI | Priority: P1
  - AC
    - 复用/对齐 `check` 的数据源与表解析策略
    - 确保表名从 `data-source` 推导

- Tests: Core unit tests for `Schema` rule
  - Type: Task | Area: Core | Priority: P0
  - AC
    - 正常/边界/错误用例；严格类型检查；mypy 通过

- Tests: CLI unit tests for parsing/mapping/aggregation
  - Type: Task | Area: CLI | Priority: P0
  - AC
    - 参数解析、规则文件校验、分解映射、聚合优先级、输出格式、Exit codes
    - 使用 pytest/pytest-cov，`@pytest.mark.parametrize` 覆盖边界

- Tests: Integration – end-to-end `vlite schema`
  - Type: Task | Area: CLI/Core | Priority: P0
  - AC
    - 真实或模拟数据源上验证整条链路
    - 失败/跳过/通过路径皆覆盖
    - 覆盖率报告≥80%

- Docs: Update README/DEVELOPMENT/CHANGELOG
  - Type: Docs | Area: Docs | Priority: P0
  - AC
    - README 增加用法与示例
    - DEVELOPMENT 增加实现细节与测试说明
    - CHANGELOG 按 SemVer 记录

- CI: pre-commit, mypy, coverage gate
  - Type: Task | Area: CI | Priority: P0
  - AC
    - `requirements(-dev).txt` 添加/更新依赖并记录变更原因
    - 启用/确保 pre-commit（Black/isort/mypy/pytest）
    - 覆盖率阈值≥80%，低于阈值失败

- Security: Review against SECURITY.md
  - Type: Task | Area: Shared | Priority: P1
  - AC
    - 敏感信息走环境变量，最小权限
    - 日志中不泄漏敏感数据

- Release: version bump + tag
  - Type: Task | Area: CI | Priority: P1
  - AC
    - 次版本号 bump，生成 tag
    - 合并到主分支通过受保护策略与 PR

- 依赖关系（示例）
  - CLI skeleton → Decompose mapping → Aggregation/Output → Integration tests → Release
  - Core `Schema` rule → Core unit tests → Integration tests
  - CI gate、Docs → Release

---

### 自动化建议（Projects 工作流）

- 当 PR 引用/关闭某条目时，自动将 `Status` 流转为 `In Review` / `Done`
- 合并关闭后 14 天自动归档 `Done`
- 当 `Type=Epic` 时，Board 固定在顶部；子任务通过 Project 的 “Tracked by” 建立父子关系
- 在 PR 模板中强制关联 Project 条目、Milestone、Labels
- 添加 Rule：当 `ExitCodesVerified` 未勾选且 `Status=In Review` 时，显示提示卡片（手动检查）

---

### Issue 模板（简版，可作为项目内通用模板）

- 标题规范
  - Feature: …
  - Task: …
  - Tests: …
  - Docs: …
- 内容
  - Summary
  - Scope
  - Acceptance Criteria
  - Out of Scope
  - References（链接到 `docs/Feature Request- Schema Validation Command.md` 与设计文档）
  - Checklist
    - PEP8/Black/isort/mypy 通过
    - 使用 `shared/enums`、`shared/utils`
    - 覆盖率不回退
    - 变更记录与依赖变更原因已更新

---

### Definition of Done（Epic 级）

- 功能符合 `Feature Request` 与设计文档的 Scope/AC
- CLI 与 Core 全量单元/集成测试通过，覆盖率≥80%
- 文档与 CHANGELOG 更新，SemVer 次版本号 bump
- pre-commit、mypy、CI 通过；无新安全隐患
- Exit codes、聚合优先级、警告策略按规范验证并勾选 `ExitCodesVerified`

- 建议将上述清单直接创建为 Project 条目，并用依赖关系串联，保证从 Core → CLI → 测试 → 文档/CI → 发布的闭环推进。

- 我已经把任务拆分到可执行的粒度，并将项目字段、视图和自动化建议一起给出。你可以告诉我你使用的是 Projects Classic 还是 Projects（Beta），我可以按对应形态给你提供批量创建的脚本或更贴近你现状的配置说明。

- 关键落地点
  - 按 Area 与 Type 划分任务，保证 CLI 分解与 Core 规则并行推进
  - 用依赖链控制集成测试与发布节奏
  - 通过字段和自动化把 Exit codes、覆盖率与安全检查显式化

- 如果需要，我可以把上述每个 Issue 的模板正文（Summary/AC/Checklist）整理成可复制的清单，或生成 `gh` 命令行批量创建脚本。
