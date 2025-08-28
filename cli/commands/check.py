"""
Check Command Implementation

The core `vlite check` command for data quality validation.
Supports smart source identification, rule parsing, and formatted output.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Tuple, cast

import click

from cli.core.config import get_cli_config

# Import new configuration system
from core.config import get_core_config
from shared.exceptions import EngineError, OperationError, RuleExecutionError
from shared.utils.console import safe_echo
from shared.utils.datetime_utils import now
from shared.utils.logger import get_logger

from ..core.data_validator import DataValidator
from ..core.exception_handler import CliExceptionHandler
from ..core.output_formatter import OutputFormatter
from ..core.rule_parser import RuleParser
from ..core.source_parser import SourceParser

# ---------------------------------------------------------------
# Core helper classes (can be monkey-patched by pytest via
# ``patch('cli.commands.check.SourceParser')`` etc.)
# ---------------------------------------------------------------


# Initialize logger
logger = get_logger(__name__)


@click.command("check")
@click.option(
    "--conn",
    "connection_string",
    required=True,
    help="Database connection string or file path",
)
@click.option("--table", "table_name", required=True, help="Table name to validate")
@click.option(
    "--rule",
    "rules",
    multiple=True,
    help="Inline rule expression (can be used multiple times)",
)
@click.option(
    "--rules",
    "rules_file",
    type=click.Path(exists=True, readable=True),
    help="Path to rules file (JSON format)",
)
@click.option("--quiet", is_flag=True, default=False, help="Show summary only")
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Show detailed information and failure samples",
)
def check_command(
    connection_string: str,
    table_name: str,
    rules: Tuple[str, ...],
    rules_file: Optional[str],
    quiet: bool,
    verbose: bool,
) -> None:
    """
    Check data quality for the given source.

    NEW FORMAT:
        vlite check --conn <connection> --table <table_name> [options]

    SOURCE can be:
    - File path: users.csv, data.xlsx, records.json
    - Database URL: mysql://user:pass@host/db
    - SQLite file: sqlite:///path/to/file.db

    Examples:
        vlite check --conn users.csv --table users --rule "not_null(id)"
        vlite check --conn mysql://user:pass@host/db \
            --table users --rules validation.json
    """
    # Record start time
    start_time = now()
    logger.info(f"Starting data quality check for: {connection_string}")

    # Create exception handler
    exception_handler = CliExceptionHandler(verbose=verbose)

    # Initialize error variables
    cli_error = None
    schema_error = None
    engine_error = None
    results = None

    try:
        # Phase 1: CLI self-processing and Schema creation
        try:
            # Load configurations using new system
            core_config = get_core_config()
            cli_config = get_cli_config()

            # Initialize components (these may be monkey-patched in the test-suite
            # via ``patch('cli.commands.check.SourceParser')`` etc.)
            source_parser = SourceParser()
            rule_parser = RuleParser()
            output_formatter = OutputFormatter(quiet=quiet, verbose=verbose)

            # Validate inputs
            if not rules and not rules_file:
                raise click.UsageError(
                    "No rules specified. Use --rule for inline rules or "
                    "--rules for rules file."
                )

            # Parse source
            safe_echo(f"🔍 Analyzing source: {connection_string}")

            # Proactively verify that a provided file is not empty – this avoids
            # kicking off heavy validation logic only to discover the file is
            # useless.  The modern test-suite expects a graceful early-exit with a
            # clear error message in such a scenario.
            potential_path = Path(connection_string)
            if potential_path.exists() and potential_path.is_file():
                if potential_path.stat().st_size == 0:
                    raise click.ClickException(
                        f"Error: Source file '{connection_string}' is empty "
                        "– nothing to validate."
                    )

            # Parse source config - this may raise Schema creation error
            # (OperationError)
            source_config = source_parser.parse_source(connection_string, table_name)

            # Parse rules - this may raise Schema creation error
            # (RuleExecutionError)
            safe_echo("📋 Loading validation rules...")
            rule_configs = rule_parser.parse_rules(
                inline_rules=list(rules) if rules else [], rules_file=rules_file
            )

            if not rule_configs:
                raise click.UsageError("No valid rules found.")

            safe_echo(f"   Found {len(rule_configs)} validation rules")

            # Create data validator with new configuration system
            # Use cast to satisfy mypy type checker due to list invariance
            validator = DataValidator(
                source_config=source_config,
                rules=cast(list, rule_configs),
                core_config=core_config,
                cli_config=cli_config,
            )
        except (OperationError, RuleExecutionError) as e:
            # Catch Schema creation error
            schema_error = e
            raise
        except Exception as e:
            # Other errors are considered CLI errors
            cli_error = e
            raise

        # Phase 2: Core validation execution
        try:
            # Execute validation
            safe_echo("✅ Starting validation...")
            results = asyncio.run(validator.validate())
            # Ensure results is not None before converting to dicts
            results_dicts = (
                [r.model_dump() for r in results] if results is not None else []
            )
        except EngineError as e:
            engine_error = e
            raise
        logger.info(f"Results: {results}")

        # Phase 3: Result processing
        error_context = exception_handler.handle_complete_process(
            cli_error=cli_error,
            schema_error=schema_error,
            engine_error=engine_error,
            results=results,  # results can be None, which is valid for this handler
        )

        # Decide output and exit code based on error context
        if error_context.category != "success":
            # Show error message
            safe_echo(f"❌ {error_context.user_message}", err=True)

            # Show recovery suggestions
            if error_context.recovery_actions:
                safe_echo("\nSuggested actions:")
                for action in error_context.recovery_actions:
                    safe_echo(f"• {action}")

            # Show technical details (if verbose enabled)
            if verbose and error_context.technical_details:
                safe_echo(f"\nTechnical details:\n{error_context.technical_details}")

            sys.exit(error_context.exit_code)
        else:
            # On success, calculate execution time and display results
            end_time = now()
            execution_time = (end_time - start_time).total_seconds()

            # Format and display results
            output_formatter.display_results(
                results=results_dicts,
                rules=rule_configs,  # Pass as objects, not dicts
                source=connection_string,
                execution_time=execution_time,
                total_rules=len(rule_configs),
            )

            # Set exit code based on validation results
            has_failures = any(result["status"] == "FAILED" for result in results_dicts)

            if has_failures:
                logger.warning("Validation completed with failures")
                sys.exit(1)
            else:
                logger.info("All validations passed successfully")
                # Show success message
                safe_echo(f"✅ {error_context.user_message}")
                sys.exit(0)

    except click.UsageError:
        # Re-raise click errors to preserve CLI behavior
        raise

    except Exception as e:
        # Unified error handling
        # Check if it is EngineError
        if isinstance(e, EngineError):
            engine_error = e

        error_context = exception_handler.handle_complete_process(
            cli_error=cli_error,
            schema_error=schema_error,
            engine_error=engine_error,
            results=results,
        )

        if error_context.category == "success":
            # On success, calculate execution time and display results
            end_time = now()
            execution_time = (end_time - start_time).total_seconds()

            # Format and display results
            output_formatter.display_results(
                results=results_dicts,
                rules=rule_configs,  # Pass as objects, not dicts
                source=connection_string,
                execution_time=execution_time,
                total_rules=len(rule_configs),
            )

            # Show success message
            safe_echo(f"✅ {error_context.user_message}")
            sys.exit(0)
        else:
            # Show error message
            safe_echo(f"❌ {error_context.user_message}", err=True)

            if error_context.recovery_actions:
                safe_echo("\nSuggested actions:")
                for action in error_context.recovery_actions:
                    safe_echo(f"• {action}")

            if verbose and error_context.technical_details:
                safe_echo(f"\nTechnical details:\n{error_context.technical_details}")

            sys.exit(error_context.exit_code)


@click.command("rules-help")
def rules_help_command() -> None:
    """Show help for rule syntax and examples."""
    help_text = """
📋 ValidateLite Rule Syntax Guide

INLINE RULE SYNTAX:
  --rule "rule_type(column[,param1,param2,...])"

SUPPORTED RULE TYPES:
  not_null(column)              - Check for null/empty values
  length(column,min,max)        - Check string length constraints
  unique(column)                - Check for duplicate values
  range(column,min,max)         - Check numeric range constraints
  regex(column,pattern)         - Check regex pattern matching
  enum(column,value1,value2...) - Check allowed enum values

EXAMPLES:
  vlite check users.csv --rule "not_null(id)"
  vlite check users.csv --rule "length(name,2,50)"
  vlite check users.csv --rule "unique(email)"
  vlite check users.csv --rule "range(age,18,65)"
  vlite check users.csv --rule "regex(email,^[\\w.-]+@[\\w.-]+\\.[a-zA-Z]{2,}$)"

MULTIPLE RULES:
  vlite check users.csv --rule "not_null(id)" --rule "unique(email)"

RULES FILE:
  vlite check users.csv --rules validation.json

  Example validation.json:
  {
    "version": "1.0",
    "rules": [
      {"type": "not_null", "column": "id"},
      {"type": "length", "column": "name", "min": 2, "max": 50},
      {"type": "unique", "column": "email"}
    ]
  }

SOURCES SUPPORTED:
  CSV Files:     users.csv, data.xlsx, records.json
  Database URLs: mysql://user:pass@host/db.table
  SQLite Files:  sqlite:///path/to/file.db
    """
    safe_echo(help_text)
