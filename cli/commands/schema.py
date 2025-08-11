"""
Schema Command

Adds `vlite-cli schema` command that parses parameters, performs minimal rules
file validation (single-table only, no jsonschema), and prints placeholder
output aligned with the existing CLI style.
"""

from __future__ import annotations

import json
import sys
from typing import Any, List, Tuple

import click

from cli.core.output_formatter import OutputFormatter
from cli.core.source_parser import SourceParser
from shared.utils.console import safe_echo
from shared.utils.datetime_utils import now
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

    start_time = now()
    try:
        # Validate source format using existing parser for parity with `check`
        SourceParser().parse_source(source)

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

        # Produce output
        exec_seconds = (now() - start_time).total_seconds()
        if output.lower() == "json":
            payload = {
                "status": "ok",
                "message": "Schema command skeleton",
                "source": source,
                "rules_file": rules_file,
                "rules_count": rules_count,
                "execution_time_s": round(exec_seconds, 3),
            }
            _safe_echo(json.dumps(payload))
        else:
            formatter = OutputFormatter(quiet=False, verbose=verbose)
            text = formatter.format_basic_output(
                source=source,
                total_records=0,
                results=[],  # No execution yet
                execution_time=exec_seconds,
            )
            _safe_echo(text)

        # Exit code policy for skeleton
        exit_code = 1 if fail_on_error else 0
        sys.exit(exit_code)

    except click.UsageError:
        # Propagate Click usage errors for standard exit code (typically 2)
        raise
    except Exception as e:  # Fallback: print concise error and return generic failure
        logger.error(f"Schema command error: {str(e)}")
        _safe_echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)
