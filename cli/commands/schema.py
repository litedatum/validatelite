"""
Schema Command

Adds `vlite-cli schema` command that parses parameters, performs minimal rules
file validation (single-table only, no jsonschema), and prints placeholder
output aligned with the existing CLI style.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Tuple, cast

import click

from cli.core.output_formatter import OutputFormatter
from cli.core.source_parser import SourceParser
from shared.enums import RuleAction, RuleCategory, RuleType, SeverityLevel
from shared.enums.data_types import DataType
from shared.schema.base import RuleTarget, TargetEntity
from shared.schema.rule_schema import RuleSchema
from shared.utils.console import safe_echo
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


def _validate_rules_payload(payload: Any) -> Tuple[List[str], int]:
    """Validate the minimal structure of the schema rules file.

    This performs non-jsonschema checks:
    - Top-level must be an object with a `rules` array
    - Warn and ignore top-level `table` if present
    - Validate each rule item fields and types:
      - field: required str
      - type: optional str in allowed set
      - required: optional bool
      - enum: optional list
      - min/max: optional numeric (int or float)

    Returns:
        warnings, rules_count

    Raises:
        click.UsageError: if structure or types are invalid
    """
    warnings: List[str] = []

    if not isinstance(payload, dict):
        raise click.UsageError("Rules file must be a JSON object with a 'rules' array")

    if "table" in payload:
        warnings.append(
            "Top-level 'table' is ignored; table is derived from data-source"
        )

    if "tables" in payload:
        # Explicitly reject multi-table format in v1
        raise click.UsageError(
            "'tables' is not supported in v1; use single-table 'rules' only"
        )

    rules = payload.get("rules")
    if not isinstance(rules, list):
        raise click.UsageError("'rules' must be an array")

    for idx, item in enumerate(rules):
        if not isinstance(item, dict):
            raise click.UsageError(f"rules[{idx}] must be an object")

        # field
        field_name = item.get("field")
        if not isinstance(field_name, str) or not field_name:
            raise click.UsageError(f"rules[{idx}].field must be a non-empty string")

        # type
        if "type" in item:
            type_name = item["type"]
            if not isinstance(type_name, str):
                raise click.UsageError(
                    f"rules[{idx}].type must be a string when provided"
                )
            if type_name.lower() not in _ALLOWED_TYPE_NAMES:
                allowed = ", ".join(sorted(_ALLOWED_TYPE_NAMES))
                raise click.UsageError(
                    f"rules[{idx}].type '{type_name}' is not supported. "
                    f"Allowed: {allowed}"
                )

        # required
        if "required" in item and not isinstance(item["required"], bool):
            raise click.UsageError(
                f"rules[{idx}].required must be a boolean when provided"
            )

        # enum
        if "enum" in item and not isinstance(item["enum"], list):
            raise click.UsageError(f"rules[{idx}].enum must be an array when provided")

        # min/max
        for bound_key in ("min", "max"):
            if bound_key in item:
                value = item[bound_key]
                if not isinstance(value, (int, float)):
                    raise click.UsageError(
                        f"rules[{idx}].{bound_key} must be numeric when provided"
                    )

    return warnings, len(rules)


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


def _decompose_to_atomic_rules(payload: Dict[str, Any]) -> List[RuleSchema]:
    """Decompose schema JSON payload into atomic RuleSchema objects.

    Rules per item:
    - type -> contributes to table-level SCHEMA columns mapping
    - required -> NOT_NULL(column)
    - min/max -> RANGE(column, min_value/max_value)
    - enum -> ENUM(column, allowed_values)
    """
    rules_arr = payload.get("rules", [])

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
        # Optional switches at top-level
        if isinstance(payload.get("strict_mode"), bool):
            schema_params["strict_mode"] = payload["strict_mode"]
        if isinstance(payload.get("case_insensitive"), bool):
            schema_params["case_insensitive"] = payload["case_insensitive"]

        atomic_rules.insert(
            0,
            _create_rule_schema(
                name="schema",
                rule_type=RuleType.SCHEMA,
                column=None,
                parameters=schema_params,
                description="CLI: table schema existence+type",
            ),
        )

    return atomic_rules


def _safe_echo(text: str, *, err: bool = False) -> None:
    """Compatibility shim; delegate to shared safe_echo."""
    safe_echo(text, err=err)


@click.command("schema")
@click.argument("source", required=True)
@click.option(
    "--rules",
    "rules_file",
    type=click.Path(exists=True, readable=True),
    required=True,
    help="Path to schema rules file (JSON)",
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
    help="Return exit code 1 if any error occurs during skeleton execution",
)
@click.option(
    "--max-errors",
    type=int,
    default=100,
    show_default=True,
    help="Maximum number of errors to collect (reserved; not used in skeleton)",
)
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose output")
def schema_command(
    source: str,
    rules_file: str,
    output: str,
    fail_on_error: bool,
    max_errors: int,
    verbose: bool,
) -> None:
    """Schema validation command with minimal rules file validation.

    Decomposition and execution are added in subsequent tasks.
    """

    import asyncio

    from cli.core.config import get_cli_config
    from cli.core.data_validator import DataValidator
    from core.config import get_core_config

    # start_time = now()
    try:
        # Validate source and get connection config (table resolved here)
        source_config = SourceParser().parse_source(source)

        # Validate and load rules file
        try:
            with open(rules_file, "r", encoding="utf-8") as f:
                rules_payload = json.load(f)
        except json.JSONDecodeError as e:  # Usage-level error in skeleton
            raise click.UsageError(f"Invalid JSON in rules file: {rules_file}") from e

        # Minimal structure/type validation
        warnings, rules_count = _validate_rules_payload(rules_payload)

        # Emit warnings
        for msg in warnings:
            _safe_echo(f"⚠️ Warning: {msg}", err=True)

        # Decompose into atomic rules per design
        atomic_rules = _decompose_to_atomic_rules(rules_payload)

        # Execute via core engine using DataValidator
        core_config = get_core_config()
        cli_config = get_cli_config()
        validator = DataValidator(
            source_config=source_config,
            rules=cast(List[RuleSchema | Dict[str, Any]], atomic_rules),
            core_config=core_config,
            cli_config=cli_config,
        )

        from shared.utils.datetime_utils import now as _now

        _exec_start = _now()
        results = asyncio.run(validator.validate())
        exec_seconds = (_now() - _exec_start).total_seconds()

        # Output
        if output.lower() == "json":
            payload = {
                "status": "ok",
                "source": source,
                "rules_file": rules_file,
                "rules_count": len(atomic_rules),
                "results": [
                    r.model_dump() if hasattr(r, "model_dump") else r for r in results
                ],
                "execution_time_s": round(exec_seconds, 3),
            }
            _safe_echo(json.dumps(payload))
        else:
            formatter = OutputFormatter(quiet=False, verbose=verbose)
            text = formatter.format_basic_output(
                source=source,
                total_records=0,
                results=results,
                execution_time=exec_seconds,
            )
            _safe_echo(text)

        # Exit code: fail if any rule failed
        any_failed = any((r.status or "").upper() == "FAILED" for r in results)
        sys.exit(1 if any_failed or fail_on_error else 0)

    except click.UsageError:
        # Propagate Click usage errors for standard exit code (typically 2)
        raise
    except Exception as e:  # Fallback: print concise error and return generic failure
        logger.error(f"Schema command error: {str(e)}")
        _safe_echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)
