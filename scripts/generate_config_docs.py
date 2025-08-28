"""
Configuration Documentation Generator

Generates comprehensive documentation for all configuration options
using Pydantic model introspection.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Type

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from cli.core.config import CliConfig
from core.config import CoreConfig
from shared.config.logging_config import LoggingConfig

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_field_info(
    model: Type[BaseModel], field_name: str, field_info: FieldInfo
) -> Dict[str, Any]:
    """Extract field information from Pydantic field."""
    info = {
        "name": field_name,
        "type": str(field_info.annotation) if field_info.annotation else "Any",
        "default": field_info.default if field_info.default is not ... else "Required",
        "description": field_info.description or "No description available",
    }

    # Handle special cases
    if hasattr(field_info, "ge") and field_info.ge is not None:
        info["minimum"] = field_info.ge
    if hasattr(field_info, "le") and field_info.le is not None:
        info["maximum"] = field_info.le
    if hasattr(field_info, "min_length") and field_info.min_length is not None:
        info["min_length"] = field_info.min_length
    if hasattr(field_info, "max_length") and field_info.max_length is not None:
        info["max_length"] = field_info.max_length

    return info


def generate_model_docs(model_class: type[BaseModel], config_name: str) -> str:
    """Generate documentation for a Pydantic model."""
    docs = f"## {config_name} Configuration\n\n"
    docs += f"Model: `{model_class.__name__}`\n\n"

    if model_class.__doc__:
        docs += f"{model_class.__doc__.strip()}\n\n"

    docs += "### Configuration Options\n\n"
    docs += "| Option | Type | Default | Description |\n"
    docs += "|--------|------|---------|-------------|\n"

    # Get model fields
    model_fields = model_class.model_fields

    for field_name, field_info in model_fields.items():
        field_details = get_field_info(model_class, field_name, field_info)

        # Format type name
        type_name = (
            field_details["type"]
            .replace("typing.", "")
            .replace("<class '", "")
            .replace("'>", "")
        )

        # Format default value
        default_val = field_details["default"]
        if isinstance(default_val, str):
            default_val = f'`"{default_val}"`'
        elif default_val is None:
            default_val = "`None`"
        else:
            default_val = f"`{default_val}`"

        docs += (
            f"| `{field_name}` | {type_name} | {default_val} | "
            f"{field_details['description']} |\n"
        )

    docs += "\n"
    return docs


def generate_config_file_examples() -> str:
    """Generate example configuration files."""
    docs = "## Configuration File Examples\n\n"

    docs += "### Core Configuration (`config/core.toml`)\n\n"
    docs += "```toml\n"
    docs += "# Core engine configuration\n"
    docs += "execution_timeout = 300  # seconds\n"
    docs += "table_size_threshold = 10000  # records\n"
    docs += "merge_execution_enabled = true\n"
    docs += "monitoring_enabled = false\n"
    docs += "```\n\n"

    docs += "### CLI Configuration (`config/cli.toml`)\n\n"
    docs += "```toml\n"
    docs += "# CLI application configuration\n"
    docs += "debug_mode = false\n"
    docs += "default_sample_size = 10000\n"
    docs += "max_file_size_mb = 100\n"
    docs += "query_timeout = 300\n"
    docs += "\n"
    docs += "[database]\n"
    docs += 'url = "sqlite:///temp.db"  # Optional database URL\n'
    docs += "connect_timeout = 30\n"
    docs += "echo_queries = false\n"
    docs += "```\n\n"

    docs += "### Logging Configuration (`config/logging.toml`)\n\n"
    docs += "```toml\n"
    docs += "# Logging configuration\n"
    docs += 'level = "INFO"\n'
    docs += 'format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"\n'
    docs += "to_file = false\n"
    docs += 'file_path = "logs/app.log"\n'
    docs += "max_bytes = 10485760  # 10MB\n"
    docs += "backup_count = 5\n"
    docs += "```\n\n"

    return docs


def generate_environment_variables_docs() -> str:
    """Generate documentation for environment variables."""
    docs = "## Environment Variables\n\n"
    docs += (
        "Configuration file paths can be overridden using environment variables:\n\n"
    )
    docs += "| Variable | Description | Default |\n"
    docs += "|----------|-------------|----------|\n"
    docs += (
        "| `CORE_CONFIG_PATH` | Path to core configuration file | "
        "`config/core.toml` |\n"
    )
    docs += (
        "| `CLI_CONFIG_PATH` | Path to CLI configuration file | `config/cli.toml` |\n"
    )
    docs += (
        "| `LOGGING_CONFIG_PATH` | Path to logging configuration file | "
        "`config/logging.toml` |\n"
    )
    docs += "\n"
    docs += "### Example Usage\n\n"
    docs += "```bash\n"
    docs += "# Use custom configuration files\n"
    docs += "export CORE_CONFIG_PATH=/path/to/custom/core.toml\n"
    docs += "export CLI_CONFIG_PATH=/path/to/custom/cli.toml\n"
    docs += "export LOGGING_CONFIG_PATH=/path/to/custom/logging.toml\n"
    docs += "\n"
    docs += "# Run the application\n"
    docs += 'vlite check data.csv --rule "not_null(id)"\n'
    docs += "```\n\n"

    return docs


def generate_full_documentation() -> str:
    """Generate complete configuration documentation."""
    docs = "# Configuration Reference\n\n"
    docs += "This document provides a comprehensive reference for all configuration "
    docs += "options available in the ValidateLite data quality application.\n\n"
    docs += "available in the ValidateLite data quality application.\n\n"
    docs += "## Overview\n\n"
    docs += "The application uses a decentralized configuration system with separate "
    docs += "configuration files for different concerns:\n\n"
    docs += "- **Core Configuration**: Settings for the rule engine\n"
    docs += "- **CLI Configuration**: Settings for the command-line interface\n"
    docs += "- **Logging Configuration**: Settings for application logging\n\n"
    docs += "All configuration files use the TOML format for readability and "
    docs += "maintainability.\n\n"

    # Generate model documentation
    docs += generate_model_docs(CoreConfig, "Core")
    docs += generate_model_docs(CliConfig, "CLI")
    docs += generate_model_docs(LoggingConfig, "Logging")

    # Generate examples
    docs += generate_config_file_examples()
    docs += generate_environment_variables_docs()

    docs += "## Configuration Loading Order\n\n"
    docs += "Configuration values are loaded in the following order of precedence "
    docs += "(later sources override earlier ones):\n\n"
    docs += "1. **Default Values**: Defined in Pydantic models\n"
    docs += "2. **Configuration Files**: Loaded from TOML files\n"
    docs += "3. **Environment Variables**: For production overrides\n"
    docs += "4. **Command-Line Arguments**: For temporary overrides\n\n"

    docs += "## Validation\n\n"
    docs += (
        "All configuration values are automatically validated using Pydantic models. "
    )
    docs += "Invalid values will result in clear error messages at startup.\n\n"

    docs += "---\n"
    docs += "*This documentation was automatically generated from the configuration "
    docs += "models.*\n"

    return docs


def main() -> None:
    """Generate and save configuration documentation."""
    print("Generating configuration documentation...")

    # Generate documentation
    docs = generate_full_documentation()

    # Save to file
    docs_path = project_root / "CONFIG_REFERENCE.md"
    with open(docs_path, "w", encoding="utf-8") as f:
        f.write(docs)

    print(f"Configuration documentation saved to: {docs_path}")
    print(f"Total lines: {len(docs.splitlines())}")


if __name__ == "__main__":
    main()
