# Configuration Reference

This document provides a comprehensive reference for all configuration options available in the ValidateLite data quality application.

## Overview

The application uses a decentralized configuration system with separate configuration files for different concerns:

- **Core Configuration**: Settings for the rule engine
- **CLI Configuration**: Settings for the command-line interface
- **Logging Configuration**: Settings for application logging

All configuration files use the TOML format for readability and maintainability.

## Core Configuration

Model: `CoreConfig`

Core configuration for the Vlite project.

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `execution_timeout` | int | `300` | Default timeout for a single rule execution in seconds. |
| `table_size_threshold` | int | `10000` | Threshold for large table optimizations. |
| `rule_count_threshold` | int | `2` | Minimum number of rules required for merge optimization. |
| `max_rules_per_merge` | int | `10` | Maximum number of rules to merge in a single SQL query. |
| `merge_execution_enabled` | bool | `True` | Enable rule merging for optimized execution. |
| `monitoring_enabled` | bool | `False` | Enable performance monitoring for the rule engine. |
| `sample_data_enabled` | bool | `True` | Enable collection of sample data for failed rules. |
| `sample_data_max_records` | int | `5` | Maximum number of sample records to collect for failed rules. |
| `independent_rule_types` | List[str] | `['UNIQUE', 'CUSTOM_SQL', 'FOREIGN_KEY']` | Rule types that should always be executed independently. |
| `TABLE_SIZE_THRESHOLD` | int | `10000` | No description available |
| `RULE_COUNT_THRESHOLD` | int | `2` | No description available |
| `MAX_RULES_PER_MERGE` | int | `10` | No description available |
| `MAX_CONCURRENT_EXECUTIONS` | int | `8` | No description available |
| `MERGE_EXECUTION_ENABLED` | bool | `True` | No description available |

## CLI Configuration

Model: `CliConfig`

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `debug_mode` | bool | `False` | Enable debug mode for verbose error output. |
| `default_sample_size` | int | `10000` | Number of records to sample for analysis. |
| `max_file_size_mb` | int | `100` | Maximum file size in MB to load into memory. |
| `database` | cli.core.config.DatabaseConfig | `PydanticUndefined` | No description available |
| `query_timeout` | int | `300` | Timeout for database queries initiated by the CLI. |

## Logging Configuration

Model: `LoggingConfig`

Logging configuration settings.

    Controls the behavior of application logging, including log levels,
    formatting, and output destinations.

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `level` | str | `"INFO"` | Logging level (e.g., DEBUG, INFO, WARNING). |
| `format` | str | `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"` | Log message format string. |
| `to_file` | bool | `False` | Enable logging to a file. |
| `file_path` | str | `"logs/app.log"` | Path to the log file. |
| `max_bytes` | int | `10485760` | Maximum log file size in bytes (10MB default). |
| `backup_count` | int | `5` | Number of backup log files to keep. |

## Configuration File Examples

### Core Configuration (`config/core.toml`)

```toml
# Core engine configuration
execution_timeout = 300  # seconds
table_size_threshold = 10000  # records
merge_execution_enabled = true
monitoring_enabled = false
```

### CLI Configuration (`config/cli.toml`)

```toml
# CLI application configuration
debug_mode = false
default_sample_size = 10000
max_file_size_mb = 100
query_timeout = 300

[database]
url = "sqlite:///temp.db"  # Optional database URL
connect_timeout = 30
echo_queries = false
```

### Logging Configuration (`config/logging.toml`)

```toml
# Logging configuration
level = "INFO"
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
to_file = false
file_path = "logs/app.log"
max_bytes = 10485760  # 10MB
backup_count = 5
```

## Environment Variables

Configuration file paths can be overridden using environment variables:

| Variable | Description | Default |
|----------|-------------|----------|
| `CORE_CONFIG_PATH` | Path to core configuration file | `config/core.toml` |
| `CLI_CONFIG_PATH` | Path to CLI configuration file | `config/cli.toml` |
| `LOGGING_CONFIG_PATH` | Path to logging configuration file | `config/logging.toml` |

### Example Usage

```bash
# Use custom configuration files
export CORE_CONFIG_PATH=/path/to/custom/core.toml
export CLI_CONFIG_PATH=/path/to/custom/cli.toml
export LOGGING_CONFIG_PATH=/path/to/custom/logging.toml

# Run the application
vlite check --conn data.csv --table data --rule "not_null(id)"
```

## Configuration Loading Order

Configuration values are loaded in the following order of precedence (later sources override earlier ones):

1. **Default Values**: Defined in Pydantic models
2. **Configuration Files**: Loaded from TOML files
3. **Environment Variables**: For production overrides
4. **Command-Line Arguments**: For temporary overrides

## Validation

All configuration values are automatically validated using Pydantic models. Invalid values will result in clear error messages at startup.

---
*This documentation was automatically generated from the configuration models.*
