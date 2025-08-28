"""
CLI Application Entry Point

Main CLI application using Click framework.
Provides the unified `vlite check` command for data quality validation.
"""

import sys

import click

from shared.config.logging_config import get_logging_config
from shared.utils.console import safe_echo
from shared.utils.error_handler import handle_exception
from shared.utils.logger import get_logger, setup_logging
from shared.utils.logging_setup import setup_logging as setup_logging_legacy

# load config first
setup_logging(get_logging_config().model_dump())

from .commands.check import check_command  # noqa: E402
from .commands.schema import schema_command  # noqa: E402

# Initialize logger for CLI
logger = get_logger(__name__)


def _setup_logging() -> None:
    """Setup logging configuration from config file"""
    try:
        # Get logging config from file
        logging_config = get_logging_config()

        # Convert to dict format for LoggerManager
        config_dict = {
            "level": logging_config.level,
            "format": logging_config.format,
            "to_file": logging_config.to_file,
            "log_file": logging_config.file_path,
            "max_bytes": logging_config.max_bytes,
            "backup_count": logging_config.backup_count,
            "to_console": True,
            # Use module levels from config file
            "module_levels": logging_config.module_levels,
        }

        # Setup logging with the config
        setup_logging(config_dict)

        # Clear logger cache to ensure new config takes effect
        from shared.utils.logger import get_logger_manager

        manager = get_logger_manager()
        manager.loggers.clear()

    except Exception as e:
        # Fallback to basic logging if config fails
        print(f"Warning: Failed to load logging config: {e}")
        # Use legacy setup as fallback
        try:
            logging_config = get_logging_config()
            setup_logging_legacy(logging_config)
        except Exception:
            # Final fallback - just set root logger level
            import logging

            logging.getLogger().setLevel(logging.WARNING)


@click.group(name="vlite", invoke_without_command=True)
@click.version_option(version="0.4.2", prog_name="vlite")
@click.pass_context
def cli_app(ctx: click.Context) -> None:
    """
    ValidateLite - Data Quality Validation Tool

    A command-line tool for validating data quality across various data sources.
    """
    # If no command provided, show help
    if ctx.invoked_subcommand is None:
        safe_echo(ctx.get_help())
        ctx.exit()


# Register the check command
cli_app.add_command(check_command)
cli_app.add_command(schema_command)


@cli_app.command("rules-help")
def rules_help() -> None:
    """Show detailed help for rule syntax"""
    help_text = """
    ValidateLite Rule Syntax Help
    =============================

    Available Rule Types:

    1. NOT_NULL - Check for null values
       Syntax: not_null(column_name)
       Example: --rule "not_null(id)"

    2. UNIQUE - Check for duplicate values
       Syntax: unique(column_name)
       Example: --rule "unique(email)"

    3. LENGTH - Check string length
       Syntax: length(column_name,min,max)
       Example: --rule "length(name,2,50)"

    4. RANGE - Check numeric range
       Syntax: range(column_name,min,max)
       Example: --rule "range(age,0,120)"

    5. ENUM - Check allowed values
       Syntax: enum(column_name,value1,value2,...)
       Example: --rule "enum(status,active,inactive,pending)"

    6. REGEX - Check pattern matching
       Syntax: regex(column_name,pattern)
       Example: --rule "regex(email,^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$)"

    Rule Files (JSON format):
    {
      "version": "1.0",
      "rules": [
        {
          "type": "not_null",
          "column": "id",
          "description": "Primary key cannot be null"
        },
        {
          "type": "length",
          "column": "name",
          "min": 2,
          "max": 50,
          "description": "Name length validation"
        }
      ]
    }

    Usage Examples:

    # Single rule
    vlite check --conn users.csv --rule "not_null(id)"

    # Multiple rules
    vlite check --conn users.csv --rule "not_null(id)" --rule "unique(email)"

    # Rules file
    vlite check --conn users.csv --rules validation.json

    # Database check
    vlite check --conn mysql://user:pass@host/db --table users --rule "not_null(id)"
    """
    safe_echo(help_text)


def main() -> None:
    """Main entry point for the CLI application"""
    # Setup logging first
    # _setup_logging()

    try:
        cli_app()
    except Exception as e:
        logger.error(f"CLI application error: {str(e)}")
        error_response = handle_exception(e, context="CLI Application", logger=logger)
        safe_echo(f"Error: {error_response['message']}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
