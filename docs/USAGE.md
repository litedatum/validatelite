# ValidateLite - User Manual

This document provides detailed instructions on how to use ValidateLite for data validation tasks. For installation and initial setup, please refer to the [README.md](../README.md).

---

## Table of Contents

- [Installation](#installation)
- [Core Command: `check`](#core-command-check)
- [Specifying the Data Source](#specifying-the-data-source)
  - [File-Based Sources](#file-based-sources)
  - [Database Sources](#database-sources)
- [Specifying Validation Rules](#specifying-validation-rules)
  - [Inline Rules (`--rule`)](#inline-rules---rule)
  - [Rule Files (`--rules`)](#rule-files---rules)
- [Command Options](#command-options)
- [Output Interpretation](#output-interpretation)
- [Practical Examples](#practical-examples)
- [Error Handling](#error-handling)
- [Configuration](#configuration)

---

## Installation

### Option 1: Install from PyPI (Recommended)
```bash
pip install validatelite
```

### Option 2: Install from pre-built package
```bash
# Download the latest release from GitHub
pip install validatelite-0.1.0-py3-none-any.whl
```

### Option 3: Run from source
```bash
git clone https://github.com/litedatum/validatelite.git
cd validatelite
pip install -r requirements.txt
```

After installation, you can use the CLI with either:
- `vlite` (if installed via pip)
- `python cli_main.py` (if running from source)

---

## Core Command: `check`

The primary command for running data validation is `check`.

### Basic Syntax

```bash
vlite check <data_source> [options]
```

- **`<data_source>`**: The path to a file or a database connection string. (Required)
- **`[options]`**: Options to specify rules, control output verbosity, etc.

---

## Specifying the Data Source

The tool intelligently recognizes the source type based on the input string.

### File-Based Sources

You can directly provide the path to a local file.

- **Supported formats**: CSV, TSV, Excel (.xls, .xlsx), JSON, JSONL.

**Example:**
```bash
# Validate a local CSV file
vlite check data/customers.csv --rule "not_null(id)"

# Validate a local Excel file
vlite check test_data/customers.xlsx --rule "unique(id)"
```

### Database Sources

Connect to databases using a standard URL connection string.

- **Supported databases**: MySQL, PostgreSQL, SQLite.

#### URL Format

The general format is:
`dialect://user:password@host:port/database.table`

- **`dialect`**: `mysql`, `postgresql`, `sqlite`
- **`database`**: The name of the database.
- **`table`**: The name of the table to validate.

#### Examples

**MySQL:**
```bash
vlite check "mysql://root:root123@localhost:3306/data_quality.customers" --rules /path/to/rules.json
```

**PostgreSQL:**
```bash
vlite check "postgresql://user:pass@host:5432/data_quality.customers" --rules /path/to/rules.json
```

**SQLite:**
The path to the SQLite database file.
```bash
vlite check "sqlite:///path/to/my_database.db.table_name" --rule "not_null(id)"
```

---

## Specifying Validation Rules

You can define validation rules either directly on the command line or through a JSON file.

### Inline Rules (`--rule`)

Use the `--rule` option to specify a single rule. You can use this option multiple times to apply several rules.

#### Syntax
`--rule "rule_type(parameter1,parameter2,...)"`

#### Supported Inline Rules

| Rule Type     | Syntax                                      | Description                                      |
|---------------|---------------------------------------------|--------------------------------------------------|
| `not_null`    | `not_null(column_name)`                     | Checks for `NULL` or empty values in a column.   |
| `unique`      | `unique(column_name)`                       | Checks for duplicate values in a column.         |
| `length`      | `length(column_name, min, max)`             | Checks if string length is within a range.       |
| `range`       | `range(column_name, min, max)`              | Checks if numerical value is within a range.     |
| `enum`        | `enum(column_name, 'val1', 'val2', ...)`    | Checks if column value is in the specified set.  |
| `regex`       | `regex(column_name, 'pattern')`             | Checks if column value matches a regex pattern.  |
| `date_format` | `date_format(column_name, 'format_string')` | Checks if date strings match a specified format. |

**Note**: The **date_format** rule currently only supports the MySQL database.

**Example:**
```bash
vlite check data.csv \
  --rule "not_null(name)" \
  --rule "unique(id)" \
  --rule "range(age, 18, 99)"
```

### Rule Files (`--rules`)

For complex scenarios, you can define all your rules in a single JSON file and pass it using the `--rules` option.

#### JSON File Structure

The file must contain a JSON object with a top-level `rules` key, which holds an array of rule objects.

```json
{
  "rules": [
    {
      "type": "not_null",
      "column": "id",
      "description": "ID must not be null."
    },
    {
      "type": "length",
      "column": "product_code",
      "params": {
        "min": 8,
        "max": 12
      }
    },
    {
      "type": "enum",
      "column": "status",
      "params": {
        "values": ["active", "inactive", "pending"]
      }
    },
    {
      "type": "regex",
      "column": "email",
      "params": {
        "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
      }
    }
  ]
}
```

#### Example Usage
```bash
vlite check "mysql://..." --rules "test_data/validate_merge.json"
```

---

## Command Options

| Option      | Description                                                                  |
|-------------|------------------------------------------------------------------------------|
| `--verbose` | Shows detailed results, including samples of failing data for each rule.     |
| `--quiet`   | Suppresses detailed output and shows only a final summary table.             |
| `--help`    | Displays a help message with all available commands and options.             |

---

## Output Interpretation

- **Standard Output**: By default, the CLI prints a summary table showing each rule, its parameters, and the final status (`PASSED` or `FAILED`).
- **Verbose Output (`--verbose`)**: In addition to the summary, it provides details for failed rules, including the number of failed rows and a sample of the invalid data.
- **Exit Codes**:
  - `0`: All rules passed.
  - `1`: One or more rules failed.
  - `>1`: An application error occurred (e.g., invalid connection, file not found).

---

## Practical Examples

Here are some scenarios based on the E2E test suite.

#### 1. Check for `NULL` values in a specific column
```bash
# Expect: PASSED if 'name' has no nulls, FAILED otherwise.
vlite check test_data/customers.xlsx --rule "not_null(name)"
```

#### 2. Check for uniqueness and valid email format with verbose output
```bash
# Expect: FAILED for both rules, with sample data showing duplicate and invalid emails.
vlite check test_data/customers.xlsx \
  --rule "unique(email)" \
  --rule "regex(email, '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')" \
  --verbose
```

#### 3. Run a comprehensive validation using a rules file
```bash
# Use a file with multiple rules for a complete check
vlite check "mysql://root:root123@localhost:3306/data_quality.customers" \
  --rules "test_data/validate_merge.json" \
  --verbose
```

#### 4. Validate multiple rules on a CSV file
```bash
vlite check examples/sample_data.csv \
  --rule "not_null(customer_id)" \
  --rule "unique(customer_id)" \
  --rule "length(email, 5, 100)" \
  --rule "regex(email, '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')" \
  --verbose
```

---

## Error Handling

The CLI is designed to handle common issues gracefully.

- **File Not Found**: If the specified data source file does not exist, it will exit with an error.
- **Connection Error**: For database sources, it will report an error if the connection cannot be established (e.g., wrong credentials, host unreachable).
- **Invalid Rule Syntax**: If a rule provided via `--rule` or in a rules file is malformed, the CLI will report a parsing error.
- **No Rules Specified**: The command will fail if you do not provide at least one rule via `--rule` or `--rules`.

### Common Error Messages and Solutions

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `File not found` | The specified file path is incorrect | Check the file path and ensure the file exists |
| `Connection failed` | Database connection parameters are wrong | Verify database credentials and connection string |
| `Invalid rule syntax` | Rule format is incorrect | Check the rule syntax against the supported formats |
| `No rules specified` | No validation rules were provided | Add at least one `--rule` or `--rules` parameter |

---

## Configuration

ValidateLite supports configuration through TOML files and environment variables.

### Configuration Files

Example configuration files are available in the `config/` directory:
- `cli.toml.example` - CLI-specific configuration
- `core.toml.example` - Core engine configuration
- `logging.toml.example` - Logging configuration

To use these configurations:

1. Copy the example files:
```bash
cp config/cli.toml.example config/cli.toml
cp config/core.toml.example config/core.toml
cp config/logging.toml.example config/logging.toml
```

2. Modify the files according to your needs.

### Environment Variables

For sensitive information like database credentials, use environment variables:

```bash
export DB_HOST=localhost
export DB_USER=myuser
export DB_PASSWORD=mypassword
export DB_NAME=mydatabase
```

### Configuration Reference

For detailed configuration options, see [CONFIG_REFERENCE.md](../CONFIG_REFERENCE.md).

---

## Getting Help

- **Command Help**: `vlite --help` or `vlite check --help`
- **Documentation**: [README.md](../README.md) for installation and basic usage
- **Development**: [DEVELOPMENT_SETUP.md](DEVELOPMENT_SETUP.md) for contributors
- **Issues**: Report bugs and feature requests on the project's GitHub page
