### Design: Schema Validation Command
Created on 09/01/2025

#### Overview
Introduce a `schema` CLI command that parses a JSON schema rules file, decomposes it to atomic rules, invokes the core rule engine, and aggregates results. Core adds a `Schema` rule that checks field existence and type only, keeping CLI responsible for expanding higher-level schema constraints into atomic checks.

#### Architecture

- CLI (`cli/`)
  - New module `cli/schema.py` with command wiring in existing CLI entry (e.g., `cli/main.py` or `main.py`).
  - Responsibilities:
    - Read and validate `--rules` file.
    - Normalize `data-source` and resolve `table` exclusively from `data-source` (v1). If the rules file contains `table`, ignore it and emit a warning.
    - Decompose schema entries into atomic rules:
      - `Schema(table, field, type)`
      - `NotNull(table, field)` for `required: true`
      - `Range(table, field, min, max)` for numeric bounds
      - `Enum(table, field, allowed_values)` for enumerations
    - Invoke core rule execution API with the rule batch.
    - Aggregate and format results using prioritization rules (see Aggregation section).
    - Use `shared/utils` for logging, errors, and date/time utilities; use `shared/enums` for types.
- Core (`core/`)
  - New `SchemaRule` implementing `Rule` interface/protocol.
  - Responsibilities:
    - For each `(table, field, expected_type)`:
      - Verify field exists in the data source.
      - Verify field’s data type matches `expected_type` as defined in `shared/enums`.
  - No decomposition logic in core; only atomic checks.
- Shared (`shared/`)
  - Ensure `DataType` enum (or equivalent) exists in `shared/enums` with STRING, INTEGER, FLOAT, BOOLEAN, DATE, DATETIME, etc.
  - Use `shared/utils` for logging and error handling, not standard library logging directly.

### RuleSchema Specification (authoritative)

This section defines the exact format of a rule object as consumed/produced by the system so that the CLI decomposition and the core rule engine share the same understanding.

- All rule objects must conform to `shared.schema.rule_schema.RuleSchema`, which extends `shared.schema.base.RuleBase`.
- All enum fields use values defined in `shared/enums` and must be serialized as uppercase strings.
- Connection context is NOT included at the rule level; it is supplied at execution time by the engine. Do not add any top-level `connection_id`.

Fields (top-level):

- id: string (UUID). Optional when creating; auto-generated if omitted.
- name: string (1..100), required.
- description: string (<=500), optional.
- type: `RuleType` (required). One of: SCHEMA, NOT_NULL, UNIQUE, RANGE, ENUM, REGEX, DATE_FORMAT.
- target: `RuleTarget` (required). Single-table in v1.
  - entities: array with exactly one `TargetEntity` in v1
    - database: string, required
    - table: string, required
    - column: string, optional for table-level rules
    - connection_id: null (reserved)
    - alias: null (reserved)
  - relationship_type: "single_table" in v1
  - join_conditions: [] (reserved)
- parameters: object, required (may be empty). The canonical keys per rule type are specified below.
- cross_db_config: null (reserved)
- threshold: number in [0.0, 100.0], optional. Interpreted as success threshold where supported.
- category: `RuleCategory` (required). CLI should derive from rule type as specified below.
- severity: `SeverityLevel` (required). Default MEDIUM if not specified.
- action: `RuleAction` (required). Default LOG if not specified.
- is_active: boolean, default true.
- tags: array of strings, optional.
- template_id: UUID, optional.
- validation_error: string, optional (execution-time use only).

Enum sources:

- RuleType: `shared/enums/rule_types.py`
- RuleCategory: `shared/enums/rule_categories.py`
- RuleAction: `shared/enums/rule_actions.py`
- SeverityLevel: `shared/enums/severity_levels.py`

Canonical parameter keys per rule type:

- SCHEMA (table-level): { columns: { [column_name]: { expected_type: DataType, max_length?: integer, precision?: integer, scale?: integer } }, strict_mode?: boolean, case_insensitive?: boolean }
  - Purpose: batch-validate existence, data type, and metadata (length/precision) for all declared columns of one table in a single rule execution.
  - columns is required; each entry requires expected_type (STRING|INTEGER|FLOAT|BOOLEAN|DATE|DATETIME).
  - Optional metadata validation:
    - max_length (integer): for STRING types, validate database column max_length matches
    - precision (integer): for FLOAT/DECIMAL types, validate database column precision matches  
    - scale (integer): for FLOAT/DECIMAL types, validate database column scale matches
  - strict_mode (optional): when true, fail if extra columns exist in the actual table that are not declared.
  - case_insensitive (optional): when true, compare column names case-insensitively.

- NOT_NULL: {}
- UNIQUE: {}
- RANGE: { min_value?: number, max_value?: number }
  - At least one of min_value/max_value must be present.
  - Numeric 0 is valid and must not be dropped.
- ENUM: { allowed_values: array }
  - Non-empty list required.
- REGEX: { pattern: string }
  - Must be a valid regex pattern for the target dialect/engine.
- DATE_FORMAT: { format: string }
  - A Python/strftime-compatible date format string understood by the engine.

Optional, cross-cutting parameter keys:

- filter_condition: string. Optional SQL-like predicate to pre-filter the dataset.

Category derivation from type (CLI default mapping):

- SCHEMA → SCHEMA
- NOT_NULL → COMPLETENESS
- UNIQUE → UNIQUENESS
- RANGE, ENUM → VALIDITY
- REGEX, DATE_FORMAT → VALIDITY or FORMAT. In v1 use:
  - REGEX → VALIDITY
  - DATE_FORMAT → FORMAT label for display is acceptable, but store category as VALIDITY unless a dedicated FORMAT category is introduced later.

Engine dictionary format (serialization used between layers) matches `RuleSchema.to_engine_dict()`:

```json
{
  "id": "<uuid>",
  "name": "<rule_name>",
  "type": "NOT_NULL|UNIQUE|RANGE|ENUM|REGEX|DATE_FORMAT",
  "target": {
    "database": "<db>",
    "table": "<table>",
    "column": "<column_or_null>"
  },
  "parameters": { /* see canonical keys above */ },
  "threshold": 0.0,
  "severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "action": "LOG|ALERT|BLOCK|QUARANTINE|CORRECT|IGNORE",
  "is_active": true,
  "validation_error": null
}
```

CLI decomposition rules → RuleSchema mapping

- Group schema file items by table. For each table, generate ONE SCHEMA rule with parameters.columns including all `{ field, type, metadata }` mappings:
  - Type: when `type` is present, add `columns[field] = { expected_type: <DataType> }` into the table's SCHEMA rule.
  - CLI maps input type strings to `DataType` and writes them as uppercase strings.
  - Metadata: when length/precision specified in CLI JSON, add to SCHEMA rule column definition:
    - `max_length: N` → `columns[field].max_length = N` (for STRING types)
    - `precision: P, scale: S` → `columns[field].precision = P, columns[field].scale = S` (for FLOAT types)
  - required: true → emit a separate NOT_NULL rule (per column) in addition to the table-level SCHEMA rule.
  - enum: [..] → emit a separate ENUM rule (per column).
  - min/max (numeric) → emit a separate RANGE rule (per column).
  - regex/date format (extended schema) → emit REGEX/DATE_FORMAT (per column).
  - Target mapping: for SCHEMA, set `target.entities[0].column = null` (table-level). For per-column rules (NOT_NULL/ENUM/RANGE/...), set column to the field name.
  - Category, severity, action defaults: derive category from type per mapping above; severity default MEDIUM; action default ALERT for CLI-generated rules unless specified by user flag.

**CLI JSON Schema Format Examples**:

Basic type definition:
```json
{"field": "name", "type": "string"}
```

With length constraint:
```json
{"field": "name", "type": "string", "max_length": 100}
```

With precision/scale:
```json  
{"field": "price", "type": "float", "precision": 10, "scale": 2}
```

Complex field with multiple constraints:
```json
{"field": "email", "type": "string", "max_length": 255, "required": true}
```

SchemaRule (existence/type, table-level) example

```json
{
  "name": "schema_users",
  "type": "SCHEMA",
  "target": {
    "entities": [
      { "database": "sales", "table": "users", "column": null, "connection_id": null, "alias": null }
    ],
    "relationship_type": "single_table",
    "join_conditions": []
  },
  "parameters": {
    "columns": {
      "id": { "expected_type": "INTEGER" },
      "email": { "expected_type": "STRING", "max_length": 255 },
      "name": { "expected_type": "STRING", "max_length": 100 },
      "price": { "expected_type": "FLOAT", "precision": 10, "scale": 2 },
      "created_at": { "expected_type": "DATETIME" }
    },
    "strict_mode": true,
    "case_insensitive": false
  },
  "category": "VALIDITY",
  "severity": "MEDIUM",
  "action": "ALERT",
  "is_active": true
}
```

Implementation note: introducing SCHEMA requires adding `SCHEMA` to `shared/enums/rule_types.py` and registering handling in the core engine. Core should fetch table metadata once, compare declared columns against actual columns, and compute failures. For result semantics, interpret `dataset_metrics.total_records` as number of declared columns and `failed_records` as number of mismatched/missing/extra columns (per `strict_mode`).

Examples

1) NOT_NULL rule

```json
{
  "name": "not_null_email",
  "type": "NOT_NULL",
  "target": {
    "entities": [
      { "database": "sales", "table": "users", "column": "email", "connection_id": null, "alias": null }
    ],
    "relationship_type": "single_table",
    "join_conditions": []
  },
  "parameters": {},
  "category": "COMPLETENESS",
  "severity": "MEDIUM",
  "action": "ALERT",
  "is_active": true
}
```

2) RANGE rule

```json
{
  "name": "range_age",
  "type": "RANGE",
  "target": {
    "entities": [
      { "database": "hr", "table": "employees", "column": "age", "connection_id": null, "alias": null }
    ],
    "relationship_type": "single_table",
    "join_conditions": []
  },
  "parameters": { "min_value": 0, "max_value": 120 },
  "category": "VALIDITY",
  "severity": "MEDIUM",
  "action": "ALERT",
  "is_active": true
}
```

3) ENUM rule with filter

```json
{
  "name": "enum_status",
  "type": "ENUM",
  "target": {
    "entities": [
      { "database": "sales", "table": "orders", "column": "status", "connection_id": null, "alias": null }
    ],
    "relationship_type": "single_table",
    "join_conditions": []
  },
  "parameters": { "allowed_values": ["NEW", "PAID", "CANCELLED"], "filter_condition": "deleted_at IS NULL" },
  "category": "VALIDITY",
  "severity": "HIGH",
  "action": "ALERT",
  "is_active": true
}
```

Validation rules (core enforcement):

- RANGE: at least one of min_value/max_value must be provided; if both, min_value <= max_value and both numeric.
- ENUM: allowed_values must be a non-empty list.
- REGEX: pattern must compile.

Notes

- RuleSchema introduces helper methods for compatibility and engine I/O, e.g., `to_engine_dict()` and `from_legacy_params()`. These do not change the canonical creation format above.
- CLI should always use `shared/enums` for enum values, and `shared/utils` for logging/error/now.

#### Data Types and Mapping
- Minimal canonical set in v1: STRING, INTEGER, FLOAT, BOOLEAN, DATE, DATETIME. Length/precision validation supported through SCHEMA rule metadata comparison.
- CLI maps JSON `type` strings to `shared/enums.DataType`:
  - `"string"` → STRING (with optional max_length)
  - `"integer"` → INTEGER  
  - `"float"` → FLOAT (with optional precision/scale)
  - `"boolean"` → BOOLEAN
  - `"date"` → DATE
  - `"datetime"` → DATETIME
- Strict typing by default; no implicit coercion.
- Vendor-specific types must be normalized to this minimal set by existing metadata adapters or a simple mapping layer; advanced coercion is out of v1 scope.

#### Files and Modules
- `cli/schema.py`: command implementation.
- `cli/main.py` (or entrypoint): add `schema` sub-command registration.
- `core/rules/schema_rule.py`: new rule type.
- `tests/cli/test_schema_command.py`: CLI tests.
- `tests/core/rules/test_schema_rule.py`: unit tests for `SchemaRule`.
- `docs/`:
  - `README.md`: usage section for `schema`.
  - `CHANGELOG.md`: new feature entry.
  - Optional: `docs/schemas/schema_rules.schema.json` and validation notes.

#### Dependencies
- No `jsonschema` in v1. Implement minimal validation in Python.
- Ensure entries in `requirements.txt` and `requirements-dev.txt`.
- Continue using Black, isort, mypy; update pre-commit if needed.

#### Error Handling and Logging
- All CLI and core errors go through `shared/utils` error and logging helpers.
- Clear error messages for:
  - Missing/invalid `--rules` file.
  - Invalid JSON format or unsupported fields/types.
  - Data source connection/metadata errors.
- Non-interactive behavior consistent with `check`.

#### Implementation Steps
1. Planning
   - Confirm supported data types enumeration in `shared/enums`; add missing ones if needed.
   - Decide strict typing policy (strict in v1).
2. Core
   - Add `SchemaRule` in `core/rules/schema_rule.py`:
     - Constructor: `(table: str, field: str, expected_type: DataType)`.
     - `execute(context)` obtains table metadata, checks existence and type, returns a standardized result object used across rules.
     - Use `shared/utils` for logging and errors.
   - Register `SchemaRule` with the rule engine (where rules are discovered/constructed).
3. CLI
   - Implement `cli/schema.py`:
     - Parse args (`data-source`, `--rules`).
      - Load JSON; validate minimal structure (`rules` array, each item has `field` and at least `type` or `enum`/`required`).
      - Resolve `table` from `data-source` only (ignore `table` in file with a warning).
     - Map JSON entries to:
       - `SchemaRule(table, field, mapped_type)` if `type` present.
       - `NotNullRule(table, field)` if `required: true`.
       - `RangeRule(table, field, min, max)` if numeric bounds present.
       - `EnumRule(table, field, values)` if `enum` present.
       - Length/precision constraints are embedded into SCHEMA rule parameters, not separate rules.
      - Execute all rules via the core API.
      - Aggregate per-field results for display; include totals and failures, applying prioritization and skip semantics.
     - Output formatting: table by default; JSON if requested.
     - Set exit code per spec.
4. Tests
   - Unit tests for `SchemaRule`:
     - Field exists and type matches.
     - Field missing.
     - Type mismatch (e.g., expected INTEGER, actual FLOAT).
    - CLI tests:
     - Valid schema file produces correct decomposition and passes.
     - Missing file/invalid JSON yields exit code 2.
      - Mixed results (some fields failing) yields exit code 1 and correct aggregation with root-cause prioritization and skipped dependents.
     - `--output json` format snapshot.
      - Warning emitted when `table` is present in rules file and ignored.
   - Integration:
     - End-to-end run against a mock or temp data source fixture used by `check`.
   - Keep coverage ≥80%.
5. Tooling and CI
   - Run Black, isort, mypy, pytest with coverage.
   - Ensure pre-commit hooks pass.
6. Docs and Changelog
   - Update `README.md` with usage and example.
   - Update `DEVELOPMENT.md` with testing instructions.
   - Update `CHANGELOG.md` (e.g., feat: add schema CLI).
7. Versioning
   - Bump minor version in `setup.py` or equivalent.
8. Optional: JSON Schema
   - Add `docs/schemas/schema_rules.schema.json`.
   - Validate rules file in CLI when `jsonschema` is available; otherwise, skip.

#### Test Plan (Pytest)
- Modules
  - `core/rules/schema_rule.py`
  - `cli/schema.py`
- Scenarios
  - Core `SchemaRule`
    - Normal: matching types and fields.
    - Edge: missing field; unmapped vendor type; nullability irrelevant here.
    - Error: metadata retrieval failure surfaces as handled error.
  - CLI `schema` command
    - Normal: valid file, all pass.
    - Mixed: some pass, some fail (`required`, `range`, `enum`).
    - Error: bad path, invalid JSON, unsupported type name.
- Cases
  - Parameterize across data types (STRING, INTEGER, FLOAT, BOOLEAN).
  - Range bounds inclusive behavior on edges (min, max).
  - Enum exact matching with ints and strings.
- Execution
  - `pytest -vv --cov`
  - Use `pytest-mock` for data source metadata where appropriate.
  - No mocking of internal logic; only external data source/IO.

#### Performance
- Batch rule execution where the core supports it.
- Fetch table metadata once per table and share for all rules to avoid repeated calls.

#### Aggregation and Skip Semantics
- Per field, enforce the following order and short-circuiting:
  1) Existence (SchemaRule existence)
  2) Type match (SchemaRule type)
  3) Not-null
  4) Range / Enum
- If 1) fails, record a single failure (code: FIELD_MISSING) and mark 2)-4) as SKIPPED.
- If 2) fails, record a single failure (code: TYPE_MISMATCH) and mark 3)-4) as SKIPPED.
- Only when 1) and 2) pass do we evaluate 3)-4).
- Human-readable output shows only the most fundamental failure per field; JSON output includes full detail with `status: PASSED|FAILED|SKIPPED` per atomic rule and `skip_reason` where applicable.

#### Security
- No secrets in files; rely on environment variables for credentials.
- Validate user-supplied file paths; avoid arbitrary file execution.

#### Rollout
- Behind a standard release; no feature flag required.
- Backward compatible with existing `check`.

#### Future Enhancements
- Multi-table rule files:
  - Support a top-level `tables` array with `{ table, rules[] }`.
- Additional constraints:
  - Enhanced SCHEMA rule with detailed metadata validation (length, precision, scale).
  - Regex, nullability warnings vs errors, cross-field logic.
- Type coercion policy configuration.

#### UX Notes
- Prioritizing root causes reduces noise and guides users to fix structural issues (missing fields, wrong types) before value-level constraints.

- Added a requirements doc for GitHub issue and a design/implementation doc with architecture, decomposition mapping, CLI spec, core rule responsibilities, dependencies, error handling, and a concrete step-by-step plan.
- Included a comprehensive test plan in line with your Pytest rules, coverage target, and workspace quality standards.
- Decisions: initial scope is single-table per file, strict typing, no inline schema; CLI performs decomposition, core adds `SchemaRule` for existence/type only.
