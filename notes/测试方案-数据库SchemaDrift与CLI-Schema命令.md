## 测试方案：数据库 Schema Drift 与 CLI `schema` 命令

本方案聚焦“数据库场景”的 Schema 漂移检测与 CLI `schema` 命令端到端验证。文件源（CSV/Excel/JSON → SQLite）仅做少量烟雾用例，主要关注常用数据库类型（MySQL、PostgreSQL、SQL Server）的类型映射、存在性检查、严格模式、多规则联动与输出契约。

### 目标与范围
- 目标
  - 验证 SCHEMA 规则在真实数据库下的行为：字段存在性、类型一致性、严格模式（额外列）。
  - 验证 CLI `schema` 优先级/跳过语义（缺失/类型不符导致依赖规则 SKIPPED）。
  - 验证 JSON/table 两种输出的聚合与可读性、退出码契约、规则负载校验与错误分类。
- 非目标
  - 文件源类型细化矩阵（仅留极简示例）。

### 环境与前置
- 数据库
  - MySQL、PostgreSQL、SQL Server（可选）通过 `docker-compose.test.yml` 或 CI 服务容器拉起。
  - 使用 `scripts/sql/*.sql` 初始化测试库与表（建表、插入少量样例数据）。
- 配置
  - 通过环境变量或测试配置注入连接信息（遵循 SECURITY.md，敏感信息不入库）。
  - 确保 `shared/database/database_dialect.py` 的方言映射在被测版本启用。
- 数据准备
  - 每库一套基础表：`users`（id, email, created_at, amount, is_active）。
  - 可能的漂移版本：缺列、改类型、加额外列、大小写差异列名等。

### 类型映射与判定基线
- 规范类型（RuleType → DataType）：STRING/INTEGER/FLOAT/BOOLEAN/DATE/DATETIME。
- 常见供应商类型映射指引（用于断言 TYPE_MISMATCH 与 PASS）：
  - MySQL: INT/INTEGER/BIGINT→INTEGER，VARCHAR/TEXT→STRING，DECIMAL/DOUBLE/FLOAT→FLOAT，TINYINT(1)→BOOLEAN，DATE→DATE，DATETIME/TIMESTAMP→DATETIME。
  - PostgreSQL: INTEGER/BIGINT→INTEGER，VARCHAR/TEXT→STRING，NUMERIC/REAL/DOUBLE PRECISION→FLOAT，BOOLEAN→BOOLEAN，DATE→DATE，TIMESTAMP/TIMESTAMPTZ→DATETIME。
  - SQL Server: INT/BIGINT→INTEGER，NVARCHAR/TEXT→STRING，DECIMAL/FLOAT→FLOAT，BIT→BOOLEAN，DATE→DATE，DATETIME2→DATETIME。

---

## 集成测试设计（tests/integration/...）

### 1) 方言元数据一致性与 SCHEMA 行为
- 场景
  - 存在性：全部存在；缺失列（FIELD_MISSING）；大小写不一致（case_insensitive=True/False）。
  - 类型一致性：全部匹配；单列/多列 TYPE_MISMATCH；长度/精度忽略后的匹配（VARCHAR(255)、DECIMAL(10,2)）。
  - 严格模式：存在 extras 列计入失败并在 `execution_plan.schema_details.extras` 输出。
- 验收
  - `ExecutionResultSchema.status` 与 `dataset_metrics.total_records/failed_records` 正确。
  - `execution_plan.schema_details.field_results[*]` 包含 `column/existence/type/failure_code`；`failure_code ∈ {FIELD_MISSING, TYPE_MISMATCH, NONE}`。
  - `extras`（严格模式）排序输出或与实现保持一致；计入失败计数。

### 2) 多规则联动（SCHEMA + NOT_NULL/RANGE/ENUM/...）
- 场景
  - 缺失列 → 依赖规则 SKIPPED: FIELD_MISSING。
  - 类型不符 → 依赖规则 SKIPPED: TYPE_MISMATCH。
- 验收
  - 依赖规则原始执行结果为 PASSED 也会被可视化覆盖为 SKIPPED（JSON 输出）；表格模式遵循“只显示根因”原则。

### 3) 错误传播与分类
- 场景
  - 表不存在、权限不足、SQL 语法错误、连接/超时问题。
- 验收
  - `status=ERROR`，`error_message` 含根因；`get_error_classification_hints()` 给出合理 `resource_type/error_type`（table/column, permission/timeout/syntax/connection 等）。

### 4) 大列量/多规则稳定性
- 场景
  - 100+ 列声明 + 数十条依赖规则；执行时间在合理阈值内完成（阈值宽松）。
- 验收
  - 不出现 OOM/超长阻塞；结果集合契约不变。

---

## E2E 测试设计（tests/e2e/cli_scenarios/...）

### 1) Happy Path（数据库 URL，table/json 双输出）
- 输入
  - `vlite-cli schema <db_url> --rules rules.json --output table`
  - 规则包含：SCHEMA 基线，少量 NOT_NULL/RANGE/ENUM。
- 验收
  - Exit code=0；table 输出按列汇总，“✓ <col>: OK”。
  - 切换 `--output json`：
    - `status=ok`，非空 `rules_count`；
    - `summary.total_rules/failed_rules/skipped_rules/total_failed_records/execution_time_s`；
    - `results[*].status` 合理；
    - `fields[*].checks` 至少含 `existence/type`，依赖检查按需补全。

### 2) Drift 套件（端到端）
- 缺失列（FIELD_MISSING）
  - table：`✗ <col>: missing (skipped dependent checks)`；
  - json：依赖检查 SKIPPED，`skip_reason=FIELD_MISSING`。
- 类型漂移（TYPE_MISMATCH）
  - table：`✗ <col>: type mismatch (skipped dependent checks)`；
  - json：依赖检查 SKIPPED，`skip_reason=TYPE_MISMATCH`。
- 严格模式（extras）
  - json：`schema_extras` 数组出现并排序；
  - table：不出现 `schema_extras` 键名，仅汇总列问题与 Summary。
- 大小写不一致
  - `case_insensitive=True` 通过；`False` 视为缺失。

### 3) 规则负载校验与退出码
- 不支持的 `type`、空 `enum`、非数字 `min/max`、顶层 `tables` 错误：
  - Exit code ≥ 2；错误文案清晰。
- `--fail-on-error`：即便规则全通过，也返回 1。

### 4) 输出契约与稳定性
- JSON Goldens（稳定子集）
  - 比较子集字段：`status/summary` 的计数类、`fields[].checks` 的 `status/skip_reason/failure_code`。
  - 忽略易变字段（时间戳、执行耗时），必要时对数组排序。

---

## 辅助与落地

### 测试组织与命名
- 目录
  - `tests/integration/database/`：方言/引擎集成。
  - `tests/e2e/cli_scenarios/`：CLI 全流程。
- 命名
  - `test_schema_drift_<db>_<scenario>.py`（如：`test_schema_drift_mysql_missing.py`）。

### 夹具与数据构建
- 复用 `tests/shared/builders/test_builders.py` 构造规则与连接配置。
- 为每 DB 准备 `setup/teardown` 夹具（创建/销毁测试表，或使用事务回滚）。
- 通过 SQL 脚本或 `QueryExecutor` 写入少量数据，保证可观测失败计数。

### 执行与门禁
- CI 任务拆分：快速单测/集成（MySQL/PG 必测）、E2E（至少 1 组完整覆盖）。
- 覆盖率目标 ≥ 80%，重点覆盖：SCHEMA 判定、skip 语义、JSON 聚合、退出码。

### 风险与缓解
- 方言细节差异大：以规范类型为准，供应商类型按映射收敛；在断言中允许长度/精度参数被忽略。
- 不稳定字段：严格限制金样对比字段集；数组/列名按字典序排序后断言。
- 外部依赖（数据库/网络）：尽量本地容器化；当容器不可用时跳过对应用例并标注原因。

---

## 里程碑与交付
1. 集成测试（MySQL/PG）：缺失/类型漂移/严格模式/大小写/多规则联动 — 可运行。
2. E2E（CLI）：Happy path + Drift（三件套）— 断言 table/json/退出码。
3. 文档与示例：在 README/docs 增加“数据库 schema drift 检测注意事项与规则书写建议”。

如需，我可以基于本方案先投放 2–3 个集成用例与 2 个 E2E 金丝雀场景作为起步样例。


