### Feature Request: Schema Validation Command

#### Summary
Add a new CLI command to validate dataset schema definitions against data sources. The command reads a JSON rules file, decomposes it into atomic validation rules, dispatches them to the core rule engine, and aggregates results for CLI output. No inline rules for schema are supported initially.

#### Motivation
- Ensure data sources conform to predefined schema (field presence and type).
- Reuse existing rule execution infrastructure while keeping CLI changes isolated.
- Provide a scalable path to higher-level schema authoring, while core focuses on atomic checks.

#### Scope
- New CLI command: `schema`.
- CLI-only rule decomposition from schema JSON to atomic rules.
- Core: add a new `Schema` rule type for field existence and data type matching.
- Output and error handling aligned with existing `check` behavior.
- Tests, docs, and CI integration to maintain coverage and quality.

#### CLI Specification
- Command
  - `vlite schema "data-source" --rules schema.json`
- Arguments
  - `data-source`: same format and resolution logic as `check` (e.g., connection string, path, table selector).
  - `--rules/-r`: path to a JSON rules file (no inline supported).
  - Table resolution: in v1 the table is derived exclusively from `data-source`. If a `table` field is present in the rules file, it is ignored and a warning is emitted.
  - Optional flags (matching existing conventions): `--output json|table`, `--fail-on-error`, `--max-errors N`, `--verbose`.
- Exit codes
  - 0: all validations passed.
  - 1: validation failures.
  - 2: CLI/configuration error (e.g., unreadable file, invalid JSON).
- Output
  - Human-readable table by default; JSON when `--output json` is used.
  - Aggregated result summarizing total checks, failures, and per-field details.

#### Rules File Format
- Single-table file (v1); do not include a top-level `table`. The target table is resolved from `data-source`.
- Example:
  ```json
  {
    "rules": [
      { "field": "id", "type": "integer", "required": true },
      { "field": "age", "type": "integer", "required": true, "min": 0, "max": 120 },
      { "field": "has_children", "enum": [0, 1] },
      { "field": "income", "type": "float", "required": true, "min": 0 },
      { "field": "job_category", "type": "string", "enum": ["engineer", "teacher", "doctor", "other"] }
    ]
  }
  ```
- Supported properties
  - `field` (string, required)
  - `type` (enum via `shared/enums`: STRING, INTEGER, FLOAT, BOOLEAN, DATE, DATETIME). Length/precision are not considered in v1.
  - `required` (boolean)
  - `enum` (array)
  - `min`/`max` (numeric; applies to numeric types)
- Limitations
  - No inline schema rules.
  - Initial version supports one table per file; multi-table files considered later.
  - No `jsonschema` dependency in v1; the CLI performs minimal manual validation of the rules file.

#### Behavior and Rule Decomposition
- CLI maps each entry into:
  - Schema rule: verifies field exists and type matches.
  - not_null rule: for `required: true`.
  - range rule: for numeric `min`/`max`.
  - enum rule: for enumerations.
- CLI sends decomposed rules to core, receives results, and aggregates them back into field-level outcomes.

##### Aggregation and Prioritization
- Evaluation order per field: existence → type → not_null → range/enum.
- If the field is missing, report a single failure for the field with reason "FIELD_MISSING" and mark dependent checks as "SKIPPED".
- If the type mismatches, report a single failure with reason "TYPE_MISMATCH" and mark not_null/range/enum as "SKIPPED".
- Only when existence and type pass will not_null/range/enum be executed and reported.
- CLI output aggregates per field, prioritizing the most fundamental cause; skipped dependents are visible in JSON output (when requested) with their skip reason, but are not duplicated as failures in human-readable output.

#### Acceptance Criteria
- New command works with valid JSON rule files and fails gracefully on invalid input.
- Core `Schema` rule verifies presence and type using `shared/enums` and `shared/utils`.
- CLI output mirrors `check` style; exit codes match spec.
- Unit and integration tests; ≥80% coverage maintained.
- Docs updated: `README.md`, `DEVELOPMENT.md`, `CHANGELOG.md`.
- Table name, if present in the rules file, is ignored with a warning; the table is derived from `data-source`.
- Aggregation behavior follows the prioritization rules above; dependent checks are marked as skipped when blocked.

#### Non-Goals
- Multi-table rule files (phase 2).
- Complex constraints (cross-field dependencies, length patterns, regex).
- Inline schema rules.

#### Risks/Trade-offs
- Single-table JSON is simpler but limits reuse; can expand later with a `tables` array format.
- Type coercion vs strict typing: initial version uses strict matching; coercion policy can be added later.
- Aggregation suppresses noisy duplicates, which improves UX but hides secondary failures until root causes are resolved.

#### Versioning and Docs
- SemVer: minor bump.
- Update docs and changelog.
- Add/adjust dev dependencies as needed in `requirements(-dev).txt`.
