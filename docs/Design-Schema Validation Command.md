### Design: Schema Validation Command

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

#### Data Types and Mapping
- Minimal canonical set in v1: STRING, INTEGER, FLOAT, BOOLEAN, DATE, DATETIME. Length/precision are ignored.
- CLI maps JSON `type` strings to `shared/enums.DataType`:
  - `"string"` → STRING
  - `"integer"` → INTEGER
  - `"float"` → FLOAT
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
  - String length, regex, nullability warnings vs errors, cross-field logic.
- Type coercion policy configuration.

#### UX Notes
- Prioritizing root causes reduces noise and guides users to fix structural issues (missing fields, wrong types) before value-level constraints.

- Added a requirements doc for GitHub issue and a design/implementation doc with architecture, decomposition mapping, CLI spec, core rule responsibilities, dependencies, error handling, and a concrete step-by-step plan.
- Included a comprehensive test plan in line with your Pytest rules, coverage target, and workspace quality standards.
- Decisions: initial scope is single-table per file, strict typing, no inline schema; CLI performs decomposition, core adds `SchemaRule` for existence/type only.
