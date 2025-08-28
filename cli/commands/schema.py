"""
Schema Command

Adds `vlite schema` command that parses parameters, performs minimal rules
file validation (supports both single-table and multi-table formats), and prints
output aligned with the existing CLI style.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import click

from cli.core.data_validator import DataValidator
from cli.core.source_parser import SourceParser
from shared.enums import RuleAction, RuleCategory, RuleType, SeverityLevel
from shared.enums.data_types import DataType
from shared.schema.base import RuleTarget, TargetEntity
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.rule_schema import RuleSchema
from shared.utils.console import safe_echo
from shared.utils.datetime_utils import now as _now
from shared.utils.logger import get_logger

logger = get_logger(__name__)


_ALLOWED_TYPE_NAMES: set[str] = {
    "string",
    "integer",
    "float",
    "boolean",
    "date",
    "datetime",
}


def _validate_multi_table_rules_payload(payload: Any) -> Tuple[List[str], int]:
    """Validate the structure of multi-table schema rules file.

    Multi-table format:
    {
      "table1": {
        "rules": [...],
        "strict_mode": true
      },
      "table2": {
        "rules": [...]
      }
    }

    Returns:
        warnings, total_rules_count
    """
    warnings: List[str] = []
    total_rules = 0

    if not isinstance(payload, dict):
        raise click.UsageError("Rules file must be a JSON object")

    # Check if this is a multi-table format (has table names as keys)
    table_names = [key for key in payload.keys() if key != "rules"]

    if table_names:
        # Multi-table format
        for table_name in table_names:
            table_schema = payload[table_name]
            if not isinstance(table_schema, dict):
                raise click.UsageError(f"Table '{table_name}' schema must be an object")

            table_rules = table_schema.get("rules")
            if not isinstance(table_rules, list):
                raise click.UsageError(
                    f"Table '{table_name}' must have a 'rules' array"
                )

            # Validate each rule in this table
            for idx, item in enumerate(table_rules):
                if not isinstance(item, dict):
                    raise click.UsageError(
                        f"Table '{table_name}' rules[{idx}] must be an object"
                    )

                # Validate rule fields
                _validate_single_rule_item(item, f"Table '{table_name}' rules[{idx}]")

            total_rules += len(table_rules)

            # Validate optional table-level switches
            if "strict_mode" in table_schema and not isinstance(
                table_schema["strict_mode"], bool
            ):
                raise click.UsageError(
                    f"Table '{table_name}' strict_mode must be a boolean"
                )
            if "case_insensitive" in table_schema and not isinstance(
                table_schema["case_insensitive"], bool
            ):
                raise click.UsageError(
                    f"Table '{table_name}' case_insensitive must be a boolean"
                )
    else:
        # Single-table format (backward compatibility)
        warnings.append(
            "Single-table format detected; consider using multi-table format for "
            "better organization"
        )
        if "rules" not in payload:
            raise click.UsageError("Single-table format must have a 'rules' array")

        rules = payload["rules"]
        if not isinstance(rules, list):
            raise click.UsageError("'rules' must be an array")

        for idx, item in enumerate(rules):
            if not isinstance(item, dict):
                raise click.UsageError(f"rules[{idx}] must be an object")
            _validate_single_rule_item(item, f"rules[{idx}]")

        total_rules = len(rules)

    return warnings, total_rules


def _validate_single_rule_item(item: Dict[str, Any], context: str) -> None:
    """Validate a single rule item from the rules array."""
    # field
    field_name = item.get("field")
    if not isinstance(field_name, str) or not field_name:
        raise click.UsageError(f"{context}.field must be a non-empty string")

    # type
    if "type" in item:
        type_name = item["type"]
        if not isinstance(type_name, str):
            raise click.UsageError(f"{context}.type must be a string when provided")
        if type_name.lower() not in _ALLOWED_TYPE_NAMES:
            allowed = ", ".join(sorted(_ALLOWED_TYPE_NAMES))
            raise click.UsageError(
                f"{context}.type '{type_name}' is not supported. " f"Allowed: {allowed}"
            )

    # required
    if "required" in item and not isinstance(item["required"], bool):
        raise click.UsageError(f"{context}.required must be a boolean when provided")

    # enum
    if "enum" in item and not isinstance(item["enum"], list):
        raise click.UsageError(f"{context}.enum must be an array when provided")

    # min/max
    for bound_key in ("min", "max"):
        if bound_key in item:
            value = item[bound_key]
            if not isinstance(value, (int, float)):
                raise click.UsageError(
                    f"{context}.{bound_key} must be numeric when provided"
                )


def _validate_rules_payload(payload: Any) -> Tuple[List[str], int]:
    """Validate the minimal structure of the schema rules file.

    This performs non-jsonschema checks for both single-table and multi-table formats.
    """
    return _validate_multi_table_rules_payload(payload)


def _map_type_name_to_datatype(type_name: str) -> DataType:
    """Map user-provided type string to DataType enum.

    Args:
        type_name: Input type name (case-insensitive), e.g. "string".

    Returns:
        DataType enum.

    Raises:
        click.UsageError: When the value is unsupported.
    """
    normalized = str(type_name).strip().lower()
    mapping: Dict[str, DataType] = {
        "string": DataType.STRING,
        "integer": DataType.INTEGER,
        "float": DataType.FLOAT,
        "boolean": DataType.BOOLEAN,
        "date": DataType.DATE,
        "datetime": DataType.DATETIME,
    }
    if normalized not in mapping:
        allowed = ", ".join(sorted(_ALLOWED_TYPE_NAMES))
        raise click.UsageError(f"Unsupported type '{type_name}'. Allowed: {allowed}")
    return mapping[normalized]


def _derive_category(rule_type: RuleType) -> RuleCategory:
    """Derive category from rule type per design mapping."""
    if rule_type == RuleType.SCHEMA:
        return RuleCategory.VALIDITY
    if rule_type == RuleType.NOT_NULL:
        return RuleCategory.COMPLETENESS
    if rule_type == RuleType.UNIQUE:
        return RuleCategory.UNIQUENESS
    # RANGE, LENGTH, ENUM, REGEX, DATE_FORMAT -> VALIDITY in v1
    return RuleCategory.VALIDITY


def _create_rule_schema(
    *,
    name: str,
    rule_type: RuleType,
    column: str | None,
    parameters: Dict[str, Any],
    description: str | None = None,
    severity: SeverityLevel = SeverityLevel.MEDIUM,
    action: RuleAction = RuleAction.ALERT,
) -> RuleSchema:
    """Create a `RuleSchema` with an empty target that will be completed later.

    The database and table will be filled by the validator based on the source.
    """
    target = RuleTarget(
        entities=[
            TargetEntity(
                database="", table="", column=column, connection_id=None, alias=None
            )
        ],
        relationship_type="single_table",
    )
    return RuleSchema(
        name=name,
        description=description,
        type=rule_type,
        target=target,
        parameters=parameters,
        cross_db_config=None,
        threshold=0.0,
        category=_derive_category(rule_type),
        severity=severity,
        action=action,
        is_active=True,
        tags=[],
        template_id=None,
        validation_error=None,
    )


def _decompose_schema_payload(
    payload: Dict[str, Any], source_config: ConnectionSchema
) -> List[RuleSchema]:
    """Decompose a schema payload into atomic RuleSchema objects.

    This function handles both single-table and multi-table formats in a
    source-agnostic way.
    """
    all_atomic_rules: List[RuleSchema] = []
    source_db = source_config.db_name or "unknown"

    is_multi_table_format = "rules" not in payload

    if is_multi_table_format:
        tables_in_rules = list(payload.keys())
        available_tables_from_source = set(source_config.available_tables or [])

        for table_name in tables_in_rules:
            if (
                available_tables_from_source
                and table_name not in available_tables_from_source
            ):
                logger.warning(
                    f"Skipping rules for table '{table_name}' as it is not available "
                    "in the source."
                )
                continue

            table_schema = payload[table_name]
            if not isinstance(table_schema, dict):
                logger.warning(
                    f"Definition for table '{table_name}' is not a valid object, "
                    "skipping."
                )
                continue

            table_rules = _decompose_single_table_schema(
                table_schema, source_db, table_name
            )
            all_atomic_rules.extend(table_rules)
    else:
        table_name = "unknown"
        if source_config.available_tables:
            table_name = source_config.available_tables[0]
        else:
            logger.warning(
                "Could not determine table name for single-table schema. "
                "Consider using multi-table format for database sources."
            )

        table_rules = _decompose_single_table_schema(payload, source_db, table_name)
        all_atomic_rules.extend(table_rules)

    return all_atomic_rules


def _decompose_single_table_schema(
    table_schema: Dict[str, Any], source_db: str, table_name: str
) -> List[RuleSchema]:
    """Decompose a single table's schema definition into atomic RuleSchema objects.

    Args:
        table_schema: The schema definition for a single table
        source_db: Database name from source
        table_name: Name of the table being validated
    """
    rules_arr = table_schema.get("rules", [])

    # Build SCHEMA columns mapping first
    columns_map: Dict[str, Dict[str, Any]] = {}
    atomic_rules: List[RuleSchema] = []

    for item in rules_arr:
        field_name = item.get("field")
        if not isinstance(field_name, str) or not field_name:
            # Should have been validated earlier; keep defensive check
            raise click.UsageError("Each rule item must have a non-empty 'field'")

        # SCHEMA: type contributes expected_type
        if "type" in item and item["type"] is not None:
            dt = _map_type_name_to_datatype(str(item["type"]))
            columns_map[field_name] = {"expected_type": dt.value}

        # NOT_NULL
        if bool(item.get("required", False)):
            atomic_rules.append(
                _create_rule_schema(
                    name=f"not_null_{field_name}",
                    rule_type=RuleType.NOT_NULL,
                    column=field_name,
                    parameters={},
                    description=f"CLI: required non-null for {field_name}",
                )
            )

        # RANGE
        has_min = "min" in item and isinstance(item.get("min"), (int, float))
        has_max = "max" in item and isinstance(item.get("max"), (int, float))
        if has_min or has_max:
            params: Dict[str, Any] = {}
            if has_min:
                params["min_value"] = item["min"]
            if has_max:
                params["max_value"] = item["max"]
            atomic_rules.append(
                _create_rule_schema(
                    name=f"range_{field_name}",
                    rule_type=RuleType.RANGE,
                    column=field_name,
                    parameters=params,
                    description=f"CLI: range for {field_name}",
                )
            )

        # ENUM
        if "enum" in item:
            values = item.get("enum")
            if not isinstance(values, list) or len(values) == 0:
                raise click.UsageError("'enum' must be a non-empty array when provided")
            atomic_rules.append(
                _create_rule_schema(
                    name=f"enum_{field_name}",
                    rule_type=RuleType.ENUM,
                    column=field_name,
                    parameters={"allowed_values": values},
                    description=f"CLI: enum for {field_name}",
                )
            )

    # Create one table-level SCHEMA rule if any columns were declared
    if columns_map:
        schema_params: Dict[str, Any] = {"columns": columns_map}
        # Optional switches at table level
        if isinstance(table_schema.get("strict_mode"), bool):
            schema_params["strict_mode"] = table_schema["strict_mode"]
        if isinstance(table_schema.get("case_insensitive"), bool):
            schema_params["case_insensitive"] = table_schema["case_insensitive"]

        atomic_rules.insert(
            0,
            _create_rule_schema(
                name=f"schema_{table_name}",
                rule_type=RuleType.SCHEMA,
                column=None,
                parameters=schema_params,
                description=f"CLI: table schema existence+type for {table_name}",
            ),
        )

    # Set the target table and database for all rules
    for rule in atomic_rules:
        if rule.target and rule.target.entities:
            rule.target.entities[0].database = source_db
            rule.target.entities[0].table = table_name

    return atomic_rules


def _build_prioritized_atomic_status(
    *,
    schema_results: List[Dict[str, Any]],
    atomic_rules: List[RuleSchema],
) -> Dict[str, Dict[str, str]]:
    """Return a mapping rule_id -> {status, skip_reason} applying prioritization."""
    mapping: Dict[str, Dict[str, str]] = {}
    schema_failures: Dict[str, str] = (
        {}
    )  # Key: f"{table}.{column}", Value: failure_code

    schema_rules_map = {
        str(rule.id): rule for rule in atomic_rules if rule.type == RuleType.SCHEMA
    }

    for res in schema_results:
        rule_id = str(res.get("rule_id", ""))
        rule = schema_rules_map.get(rule_id)
        if not rule:
            continue

        table = rule.get_target_info().get("table", "")
        details = (
            res.get("execution_plan", {})
            .get("schema_details", {})
            .get("field_results", [])
        )

        for item in details:
            code = item.get("failure_code")
            if code in ("FIELD_MISSING", "TYPE_MISMATCH"):
                col = item.get("column")
                if col:
                    schema_failures[f"{table}.{col}"] = code

    if not schema_failures:
        return {}

    for rule in atomic_rules:
        if rule.type == RuleType.SCHEMA:
            continue

        col = rule.get_target_column()
        table = rule.get_target_info().get("table", "")

        if col and f"{table}.{col}" in schema_failures:
            reason = schema_failures[f"{table}.{col}"]
            mapping[str(rule.id)] = {"status": "SKIPPED", "skip_reason": reason}

    return mapping


def _safe_echo(text: str, *, err: bool = False) -> None:
    """Compatibility shim; delegate to shared safe_echo."""
    safe_echo(text, err=err)


def _maybe_echo_analyzing(source: str, output: str) -> None:
    """Emit analyzing line unless JSON output."""
    if str(output).lower() != "json":
        _safe_echo(f"🔍 Analyzing source: {source}", err=True)


def _guard_empty_source_file(source: str) -> None:
    """Raise a ClickException if a provided file source is empty."""
    potential_path = Path(source)
    if potential_path.exists() and potential_path.is_file():
        if potential_path.stat().st_size == 0:
            raise click.ClickException(
                f"Error: Source file '{source}' is empty – nothing to validate."
            )


def _read_rules_payload(rules_file: str) -> Dict[str, Any]:
    """Read and parse JSON rules file, raising UsageError on invalid JSON."""
    try:
        with open(rules_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        raise click.UsageError(f"Invalid JSON in rules file: {rules_file}") from e
    return cast(Dict[str, Any], payload)


def _emit_warnings(warnings: List[str], output: str = "table") -> None:
    """Emit warnings only for non-JSON output to avoid polluting JSON output."""
    if output.lower() != "json":
        for msg in warnings:
            _safe_echo(f"⚠️ Warning: {msg}", err=True)


def _early_exit_when_no_rules(
    *, source: str, rules_file: str, output: str, fail_on_error: bool
) -> None:
    """Emit minimal output and exit when no rules are present."""
    if output.lower() == "json":
        payload = {
            "status": "ok",
            "source": source,
            "rules_file": rules_file,
            "rules_count": 0,
            "summary": {
                "total_rules": 0,
                "passed_rules": 0,
                "failed_rules": 0,
                "skipped_rules": 0,
                "total_failed_records": 0,
                "execution_time_s": 0.0,
            },
            "results": [],
            "fields": [],
        }
        _safe_echo(json.dumps(payload, default=str))
        raise click.exceptions.Exit(1 if fail_on_error else 0)
    else:
        _safe_echo(f"✓ Checking {source} (0 records)")
        raise click.exceptions.Exit(1 if fail_on_error else 0)


def _create_validator(
    *,
    source_config: Any,
    atomic_rules: List[RuleSchema] | List[Dict[str, Any]],
    core_config: Any,
    cli_config: Any,
) -> Any:
    try:
        return DataValidator(
            source_config=source_config,
            rules=cast(List[RuleSchema | Dict[str, Any]], atomic_rules),
            core_config=core_config,
            cli_config=cli_config,
        )
    except Exception as e:
        logger.error(f"Failed to create DataValidator: {str(e)}")
        raise click.UsageError(f"Failed to create validator: {str(e)}")


def _run_validation(validator: Any) -> Tuple[List[Any], float]:
    import asyncio

    start = _now()
    logger.debug("Starting validation")
    try:
        results = asyncio.run(validator.validate())
        logger.debug(f"Validation returned {len(results)} results")
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise
    exec_seconds = (_now() - start).total_seconds()
    return results, exec_seconds


def _extract_schema_results(
    *, atomic_rules: List[RuleSchema], results: List[Any]
) -> List[Dict[str, Any]]:
    """Extract all SCHEMA rule results from the list of validation results."""
    schema_results = []
    schema_rule_ids = {
        str(rule.id) for rule in atomic_rules if rule.type == RuleType.SCHEMA
    }
    if not schema_rule_ids:
        return []

    for r in results:
        if r is None:
            continue
        rid = ""
        if hasattr(r, "rule_id"):
            try:
                rid = str(getattr(r, "rule_id"))
            except Exception:
                rid = ""
        elif isinstance(r, dict):
            rid = str(r.get("rule_id", ""))

        if rid in schema_rule_ids:
            schema_results.append(
                r.model_dump() if hasattr(r, "model_dump") else cast(Dict[str, Any], r)
            )
    return schema_results


def _compute_skip_map(
    *, atomic_rules: List[RuleSchema], schema_results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, str]]:
    try:
        return _build_prioritized_atomic_status(
            schema_results=schema_results, atomic_rules=atomic_rules
        )
    except Exception:
        return {}


def _emit_json_output(
    *,
    source: str,
    rules_file: str,
    atomic_rules: List[RuleSchema],
    results: List[Any],
    skip_map: Dict[str, Dict[str, str]],
    schema_results: List[Dict[str, Any]],
    exec_seconds: float,
) -> None:
    enriched_results: List[Dict[str, Any]] = []
    for r in results:
        rd: Dict[str, Any]
        if hasattr(r, "model_dump"):
            try:
                rd = cast(Dict[str, Any], r.model_dump())
            except Exception:
                rd = {}
        elif isinstance(r, dict):
            rd = r
        else:
            rd = {}
        rule_id = str(rd.get("rule_id", ""))
        if rule_id in skip_map:
            rd["status"] = skip_map[rule_id]["status"]
            rd["skip_reason"] = skip_map[rule_id]["skip_reason"]
        enriched_results.append(rd)

    rule_map: Dict[str, RuleSchema] = {str(rule.id): rule for rule in atomic_rules}

    def _failed_records_of(res: Dict[str, Any]) -> int:
        if "failed_records" in res and isinstance(res.get("failed_records"), int):
            return int(res.get("failed_records") or 0)
        dm = res.get("dataset_metrics") or []
        total = 0
        for m in dm:
            if hasattr(m, "failed_records"):
                total += int(getattr(m, "failed_records", 0) or 0)
            elif isinstance(m, dict):
                total += int(m.get("failed_records", 0) or 0)
        return total

    fields: List[Dict[str, Any]] = []
    schema_fields_index: Dict[str, Dict[str, Any]] = {}

    schema_rules_map = {
        str(rule.id): rule for rule in atomic_rules if rule.type == RuleType.SCHEMA
    }

    for schema_result in schema_results:
        schema_plan = (schema_result or {}).get("execution_plan", {}) or {}
        schema_details = schema_plan.get("schema_details", {}) or {}
        field_results = schema_details.get("field_results", []) or []

        rule_id = str(schema_result.get("rule_id", ""))
        rule = schema_rules_map.get(rule_id)
        table_name = rule.get_target_info().get("table") if rule else "unknown"

        for item in field_results:
            col_name = str(item.get("column"))
            entry: Dict[str, Any] = {
                "column": col_name,
                "table": table_name,
                "checks": {
                    "existence": {
                        "status": item.get("existence", "UNKNOWN"),
                        "failure_code": item.get("failure_code", "NONE"),
                    },
                    "type": {
                        "status": item.get("type", "UNKNOWN"),
                        "failure_code": item.get("failure_code", "NONE"),
                    },
                },
            }
            fields.append(entry)
            schema_fields_index[f"{table_name}.{col_name}"] = entry

    for rule in atomic_rules:
        if rule.type == RuleType.SCHEMA:
            params = rule.parameters or {}
            declared_cols = (params.get("columns") or {}).keys()
            table_name = rule.get_target_info().get("table")
            for col in declared_cols:
                if f"{table_name}.{str(col)}" not in schema_fields_index:
                    entry = {
                        "column": str(col),
                        "table": table_name,
                        "checks": {
                            "existence": {"status": "UNKNOWN", "failure_code": "NONE"},
                            "type": {"status": "UNKNOWN", "failure_code": "NONE"},
                        },
                    }
                    fields.append(entry)
                    schema_fields_index[f"{table_name}.{str(col)}"] = entry

    def _ensure_check(entry: Dict[str, Any], name: str) -> Dict[str, Any]:
        checks: Dict[str, Dict[str, Any]] = entry.setdefault("checks", {})
        if name not in checks:
            checks[name] = {
                "status": (
                    "SKIPPED"
                    if name in {"not_null", "range", "enum", "regex", "date_format"}
                    else "UNKNOWN"
                )
            }
        return checks[name]

    for rd in enriched_results:
        rule_id = str(rd.get("rule_id", ""))
        rule = rule_map.get(rule_id)
        if not rule or rule.type == RuleType.SCHEMA:
            continue

        column_name = rule.get_target_column() or ""
        if not column_name:
            continue

        table_name = "unknown"
        if rule.target and rule.target.entities:
            table_name = rule.target.entities[0].table

        l_entry = schema_fields_index.get(f"{table_name}.{column_name}")
        if not l_entry:
            l_entry = {"column": column_name, "table": table_name, "checks": {}}
            fields.append(l_entry)
            schema_fields_index[f"{table_name}.{column_name}"] = l_entry
        else:
            l_entry["table"] = table_name

        t = rule.type
        if t == RuleType.NOT_NULL:
            key = "not_null"
        elif t == RuleType.RANGE:
            key = "range"
        elif t == RuleType.ENUM:
            key = "enum"
        elif t == RuleType.REGEX:
            key = "regex"
        elif t == RuleType.DATE_FORMAT:
            key = "date_format"
        else:
            key = t.value.lower()

        check = _ensure_check(l_entry, key)
        check["status"] = str(rd.get("status", "UNKNOWN"))
        if rule_id in skip_map:
            check["status"] = skip_map[rule_id]["status"]
            check["skip_reason"] = skip_map[rule_id]["skip_reason"]

        fr = _failed_records_of(rd)
        if fr:
            check["failed_records"] = fr

    total_rules = len(enriched_results)
    passed_rules = sum(
        1 for r in enriched_results if str(r.get("status", "")).upper() == "PASSED"
    )
    failed_rules = sum(
        1 for r in enriched_results if str(r.get("status", "")).upper() == "FAILED"
    )
    skipped_rules = sum(
        1 for r in enriched_results if str(r.get("status", "")).upper() == "SKIPPED"
    )
    total_failed_records = sum(_failed_records_of(r) for r in enriched_results)

    schema_extras: List[str] = []
    for schema_result in schema_results:
        try:
            extras = (
                (schema_result or {})
                .get("execution_plan", {})
                .get("schema_details", {})
                .get("extras", [])
            )
            if isinstance(extras, list):
                schema_extras.extend([str(x) for x in extras])
        except Exception:
            pass

    payload: Dict[str, Any] = {
        "status": "ok",
        "source": source,
        "rules_file": rules_file,
        "rules_count": len(atomic_rules),
        "summary": {
            "total_rules": total_rules,
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
            "skipped_rules": skipped_rules,
            "total_failed_records": total_failed_records,
            "execution_time_s": round(exec_seconds, 3),
        },
        "results": enriched_results,
        "fields": fields,
    }
    if schema_extras:
        payload["schema_extras"] = sorted(list(set(schema_extras)))
    _safe_echo(json.dumps(payload, default=str))


def _emit_table_output(
    *,
    source: str,
    atomic_rules: List[RuleSchema],
    results: List[Any],
    skip_map: Dict[str, Dict[str, str]],
    schema_results: List[Dict[str, Any]],
    exec_seconds: float,
) -> None:
    rule_map = {str(rule.id): rule for rule in atomic_rules}

    table_results: List[Dict[str, Any]] = []

    def _dataset_total(res: Dict[str, Any]) -> int:
        if isinstance(res.get("total_records"), int):
            return int(res.get("total_records") or 0)
        dm = res.get("dataset_metrics") or []
        total = 0
        for m in dm:
            if hasattr(m, "total_records"):
                total = max(total, int(getattr(m, "total_records", 0) or 0))
            elif isinstance(m, dict):
                total = max(total, int(m.get("total_records", 0) or 0))
        return total

    for r in results:
        rd: Dict[str, Any]
        if hasattr(r, "model_dump"):
            try:
                rd = cast(Dict[str, Any], r.model_dump())
            except Exception:
                rd = {}
        elif isinstance(r, dict):
            rd = r
        else:
            rd = {}
        rid = str(rd.get("rule_id", ""))
        rule = rule_map.get(rid)
        if rule is not None:
            rd["rule_type"] = rule.type.value
            rd["column_name"] = rule.get_target_column()
            rd.setdefault("rule_name", rule.name)
            if rule.target and rule.target.entities:
                rd["table_name"] = rule.target.entities[0].table
        if rid in skip_map:
            rd["status"] = skip_map[rid]["status"]
            rd["skip_reason"] = skip_map[rid]["skip_reason"]
        table_results.append(rd)

    table_records: Dict[str, int] = {}
    for rd in table_results:
        table_name = rd.get("table_name", "unknown")
        total = _dataset_total(rd)
        if total > 0:
            table_records[table_name] = max(table_records.get(table_name, 0), total)

    header_total_records = sum(table_records.values())

    def _calc_failed(res: Dict[str, Any]) -> int:
        if isinstance(res.get("failed_records"), int):
            return int(res.get("failed_records") or 0)
        dm = res.get("dataset_metrics") or []
        total = 0
        for m in dm:
            if hasattr(m, "failed_records"):
                total += int(getattr(m, "failed_records", 0) or 0)
            elif isinstance(m, dict):
                total += int(m.get("failed_records", 0) or 0)
        return total

    for rd in table_results:
        if "failed_records" not in rd:
            rd["failed_records"] = _calc_failed(rd)
        if "total_records" not in rd:
            rd["total_records"] = _dataset_total(rd)

    tables_grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for rd in table_results:
        if rd.get("rule_type") == RuleType.SCHEMA.value:
            continue
        table_name = rd.get("table_name", "unknown")
        if table_name not in tables_grouped:
            tables_grouped[table_name] = {}

        col = rd.get("column_name", "")
        if col:
            if col not in tables_grouped[table_name]:
                tables_grouped[table_name][col] = {"column": col, "issues": []}

            status: Any = str(rd.get("status", "UNKNOWN"))
            if rd.get("rule_type") == RuleType.NOT_NULL.value:
                key = "not_null"
            elif rd.get("rule_type") == RuleType.RANGE.value:
                key = "range"
            elif rd.get("rule_type") == RuleType.ENUM.value:
                key = "enum"
            else:
                key = rd.get("rule_type", "unknown").lower()

            if status in {"FAILED", "ERROR", "SKIPPED"}:
                tables_grouped[table_name][col]["issues"].append(
                    {
                        "check": key,
                        "status": status,
                        "failed_records": int(rd.get("failed_records", 0) or 0),
                        "skip_reason": rd.get("skip_reason"),
                    }
                )

    all_columns_by_table: Dict[str, List[str]] = {}
    for rule in atomic_rules:
        if rule.target and rule.target.entities:
            table_name = rule.target.entities[0].table
            if table_name not in all_columns_by_table:
                all_columns_by_table[table_name] = []

            if rule.type == RuleType.SCHEMA:
                if rule.parameters:
                    declared_cols = (rule.parameters.get("columns") or {}).keys()
                    for col in declared_cols:
                        if str(col) not in all_columns_by_table[table_name]:
                            all_columns_by_table[table_name].append(str(col))
            else:
                column_name = rule.get_target_column()
                if column_name and column_name not in all_columns_by_table[table_name]:
                    all_columns_by_table[table_name].append(column_name)

    for table_name, columns in all_columns_by_table.items():
        if table_name not in tables_grouped:
            tables_grouped[table_name] = {}
        for column_name in columns:
            if column_name not in tables_grouped[table_name]:
                tables_grouped[table_name][column_name] = {
                    "column": column_name,
                    "issues": [],
                }

    schema_rules_map = {
        str(rule.id): rule for rule in atomic_rules if rule.type == RuleType.SCHEMA
    }
    for schema_result in schema_results:
        rule_id = str(schema_result.get("rule_id", ""))
        rule = schema_rules_map.get(rule_id)
        if not rule:
            continue

        table_name = rule.get_target_info().get("table")
        if not table_name or table_name not in tables_grouped:
            continue

        execution_plan = schema_result.get("execution_plan") or {}
        schema_details = execution_plan.get("schema_details", {}) or {}
        details = schema_details.get("field_results", []) or []
        for item in details:
            col = str(item.get("column"))
            if col not in tables_grouped[table_name]:
                continue
            if item.get("failure_code") == "FIELD_MISSING":
                tables_grouped[table_name][col]["issues"].append(
                    {"check": "missing", "status": "FAILED"}
                )
            elif item.get("failure_code") == "TYPE_MISMATCH":
                tables_grouped[table_name][col]["issues"].append(
                    {"check": "type", "status": "FAILED"}
                )

    lines: List[str] = []
    lines.append(f"✓ Checking {source}")

    total_failed_records = sum(
        int(r.get("failed_records", 0) or 0) for r in table_results
    )

    sorted_tables = sorted(tables_grouped.keys())

    for table_name in sorted_tables:
        records = table_records.get(table_name, 0)
        lines.append(f"\n📋 Table: {table_name} ({records:,} records)")

        table_grouped = tables_grouped[table_name]
        ordered_columns = all_columns_by_table.get(table_name, [])

        # Fallback for columns that might appear in results but not in rules
        # (e.g., from a different source)
        result_columns = sorted(table_grouped.keys())
        for col in result_columns:
            if col not in ordered_columns:
                ordered_columns.append(col)

        for col in ordered_columns:
            if col not in table_grouped:
                lines.append(f"✓ {col}: OK")
                continue

            issues = table_grouped[col]["issues"]

            if not issues:
                lines.append(f"✓ {col}: OK")
                continue

            is_missing = any(
                i.get("check") == "missing" or i.get("skip_reason") == "FIELD_MISSING"
                for i in issues
            )

            if is_missing:
                lines.append(f"✗ {col}: missing (skipped dependent checks)")
                continue

            unique_issues: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for issue in issues:
                key_ = (str(issue.get("status")), str(issue.get("check")))
                if key_ not in unique_issues:
                    unique_issues[key_] = issue

            final_issues = sorted(
                unique_issues.values(), key=lambda x: str(x.get("check"))
            )

            issue_descs: List[str] = []
            for i in final_issues:
                status = i.get("status")
                check = i.get("check", "unknown")

                if status in {"FAILED", "ERROR"}:
                    fr = i.get("failed_records", 0)
                    if status == "ERROR":
                        issue_descs.append(f"{check} error")
                    else:
                        issue_descs.append(f"{check} failed ({fr} failures)")
                elif status == "SKIPPED":
                    skip_reason = i.get("skip_reason")
                    if skip_reason == "TYPE_MISMATCH":
                        issue_descs.append("type mismatch (skipped dependent checks)")
                    else:
                        reason_text = skip_reason or "unknown reason"
                        issue_descs.append(f"{check} skipped ({reason_text})")

            if not issue_descs:
                lines.append(f"✓ {col}: OK")
            else:
                lines.append(f"✗ {col}: { ', '.join(issue_descs)}")

    total_columns = sum(len(all_columns_by_table.get(t, [])) for t in sorted_tables)
    passed_columns = sum(
        sum(
            1
            for c in all_columns_by_table.get(t, [])
            if not tables_grouped.get(t, {}).get(c, {}).get("issues", [])
        )
        for t in sorted_tables
    )
    failed_columns = total_columns - passed_columns
    overall_error_rate = (
        0.0
        if header_total_records == 0
        else (total_failed_records / max(header_total_records, 1)) * 100
    )

    if len(tables_grouped) > 1:
        lines.append("\n📊 Multi-table Summary:")
        for table_name in sorted_tables:
            table_cols = all_columns_by_table.get(table_name, [])
            table_columns_count = len(table_cols)
            table_passed = sum(
                1
                for c in table_cols
                if not tables_grouped[table_name].get(c, {}).get("issues")
            )
            table_failed = table_columns_count - table_passed
            lines.append(
                f"  {table_name}: {table_passed} passed, {table_failed} failed"
            )

    lines.append(
        f"\nSummary: {passed_columns} passed, {failed_columns} failed"
        f" ({overall_error_rate:.2f}% overall error rate)"
    )
    lines.append(f"Time: {exec_seconds:.2f}s")

    _safe_echo("\n".join(lines))


@click.command("schema")
@click.option(
    "--conn",
    "connection_string",
    required=True,
    help="Database connection string or file path",
)
@click.option(
    "--rules",
    "rules_file",
    type=click.Path(exists=True, readable=True),
    required=True,
    help="Path to schema rules file (JSON) - supports both single-table "
    "and multi-table formats",
)
@click.option(
    "--output",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format",
)
@click.option(
    "--fail-on-error",
    is_flag=True,
    default=False,
    help="Return exit code 1 if any error occurs during execution",
)
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose output")
def schema_command(
    connection_string: str,
    rules_file: str,
    output: str,
    fail_on_error: bool,
    verbose: bool,
) -> None:
    """
    Schema validation command with support for both single-table
    and multi-table validation.
    """

    from cli.core.config import get_cli_config
    from core.config import get_core_config

    try:
        _maybe_echo_analyzing(connection_string, output)
        _guard_empty_source_file(connection_string)

        source_config = SourceParser().parse_source(connection_string)
        rules_payload = _read_rules_payload(rules_file)

        is_multi_table_rules = "rules" not in rules_payload
        if is_multi_table_rules:
            source_config.parameters["is_multi_table"] = True

        warnings, rules_count = _validate_rules_payload(rules_payload)
        _emit_warnings(warnings, output)

        atomic_rules = _decompose_schema_payload(rules_payload, source_config)

        if not atomic_rules:
            _early_exit_when_no_rules(
                source=connection_string,
                rules_file=rules_file,
                output=output,
                fail_on_error=fail_on_error,
            )
            return

        core_config = get_core_config()
        cli_config = get_cli_config()
        validator = _create_validator(
            source_config=source_config,
            atomic_rules=atomic_rules,
            core_config=core_config,
            cli_config=cli_config,
        )
        results, exec_seconds = _run_validation(validator)

        schema_results = _extract_schema_results(
            atomic_rules=atomic_rules, results=results
        )
        skip_map = _compute_skip_map(
            atomic_rules=atomic_rules, schema_results=schema_results
        )

        if output.lower() == "json":
            _emit_json_output(
                source=connection_string,
                rules_file=rules_file,
                atomic_rules=atomic_rules,
                results=results,
                skip_map=skip_map,
                schema_results=schema_results,
                exec_seconds=exec_seconds,
            )
        else:
            _emit_table_output(
                source=connection_string,
                atomic_rules=atomic_rules,
                results=results,
                skip_map=skip_map,
                schema_results=schema_results,
                exec_seconds=exec_seconds,
            )

        def _status_of(item: Any) -> str:
            if hasattr(item, "status"):
                try:
                    return str(getattr(item, "status") or "").upper()
                except Exception:
                    return ""
            if isinstance(item, dict):
                return str(item.get("status", "") or "").upper()
            return ""

        any_failed = any(_status_of(r) == "FAILED" for r in results)
        raise click.exceptions.Exit(1 if any_failed or fail_on_error else 0)

    except click.UsageError:
        raise
    except click.exceptions.Exit:
        raise
    except Exception as e:
        logger.error(f"Schema command error: {str(e)}")
        _safe_echo(f"❌ Error: {str(e)}", err=True)
        raise click.exceptions.Exit(1)
