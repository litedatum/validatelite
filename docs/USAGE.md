# ValidateLite - User Manual

This document provides comprehensive instructions on how to use ValidateLite for data validation tasks. ValidateLite is a lightweight, zero-config Python CLI tool for data quality validation across files and SQL databases.

---

## Table of Contents

- [Quick Start Guide](#quick-start-guide)
  - [Installation](#installation)
  - [First Validation Example](#first-validation-example)
- [Core Concepts](#core-concepts)
  - [Command Syntax Overview](#command-syntax-overview)
  - [Data Source Types](#data-source-types)
  - [Rule Types Overview](#rule-types-overview)
- [Commands Reference](#commands-reference)
  - [The `check` Command - Rule-Based Validation](#the-check-command---rule-based-validation)
  - [The `schema` Command - Schema Validation](#the-schema-command---schema-validation)
- [Advanced Usage](#advanced-usage)
  - [Data Source Configuration](#data-source-configuration)
  - [Validation Rules Deep Dive](#validation-rules-deep-dive)
  - [Output & Reporting](#output--reporting)
- [Configuration & Environment](#configuration--environment)
- [Troubleshooting](#troubleshooting)
- [Getting Help](#getting-help)

---

## Quick Start Guide

### Installation

**Option 1: Install from PyPI (Recommended)**
```bash
pip install validatelite
```

**Option 2: Install from pre-built package**
```bash
pip install validatelite-0.4.0-py3-none-any.whl
```

**Option 3: Run from source**
```bash
git clone https://github.com/litedatum/validatelite.git
cd validatelite
pip install -r requirements.txt
```

After installation, you can use the CLI with either:
- `vlite` (if installed via pip)
- `python cli_main.py` (if running from source)

### First Validation Example

Let's start with a simple validation to check that all records in a CSV file have non-null IDs:

```bash
# Validate a CSV file
vlite check examples/sample_data.csv --rule "not_null(customer_id)"

# Validate a database table
vlite check "mysql://user:pass@localhost:3306/mydb.customers" --rule "unique(email)"

# Validate against a schema file
vlite schema "mysql://user:pass@localhost:3306/mydb.customers" --rules schema.json
```

---

## Core Concepts

### Command Syntax Overview

ValidateLite provides two main commands:

1. **`vlite check`** - Rule-based validation with flexible, granular rules
2. **`vlite schema`** - Schema-based validation with structured JSON schema files

Both commands follow this general pattern:
```bash
vlite <command> <data_source> [options]
```

### Data Source Types

ValidateLite supports multiple data source types:

| Type | Format | Example |
|------|--------|---------|
| **Local Files** | CSV, Excel, JSON, JSONL | `data/customers.csv` |
| **MySQL** | Connection string | `mysql://user:pass@host:3306/db.table` |
| **PostgreSQL** | Connection string | `postgresql://user:pass@host:5432/db.table` |
| **SQLite** | File path with table | `sqlite:///path/to/db.sqlite.table` |

### Rule Types Overview

| Category | Rule Types | Description |
|----------|------------|-------------|
| **Completeness** | `not_null` | Check for missing/null values |
| **Uniqueness** | `unique` | Check for duplicate values |
| **Validity** | `regex`, `date_format`, `enum` | Check data format and values |
| **Consistency** | `range`, `length` | Check data bounds and constraints |
| **Schema** | `schema` (auto-generated) | Check field existence and types |

---

## Commands Reference

### The `check` Command - Rule-Based Validation

The `check` command allows you to specify validation rules either inline or through JSON files for flexible, granular data validation.

#### Basic Syntax & Parameters

```bash
vlite check <data_source> [options]
```

**Required Parameters:**
- `<data_source>` - Path to file or database connection string

**Options:**
| Option | Description |
|--------|-------------|
| `--rule "rule_spec"` | Specify inline validation rule (can be used multiple times) |
| `--rules <file.json>` | Specify JSON file containing validation rules |
| `--verbose` | Show detailed results with failure samples |
| `--quiet` | Show only summary information |
| `--help` | Display command help |

#### Specifying Rules

**Inline Rules (`--rule`)**

Use `--rule` for simple, quick validations:

```bash
# Single rule
vlite check data.csv --rule "not_null(id)"

# Multiple rules
vlite check data.csv \
  --rule "not_null(name)" \
  --rule "unique(id)" \
  --rule "range(age, 18, 99)"
```

**Supported Inline Rule Types:**

| Rule Type | Syntax | Description |
|-----------|--------|-------------|
| `not_null` | `not_null(column)` | No NULL or empty values |
| `unique` | `unique(column)` | No duplicate values |
| `length` | `length(column, min, max)` | String length within range |
| `range` | `range(column, min, max)` | Numeric value within range |
| `enum` | `enum(column, 'val1', 'val2', ...)` | Value in specified set |
| `regex` | `regex(column, 'pattern')` | Matches regex pattern |
| `date_format` | `date_format(column, 'format')` | Date format validation (MySQL only) |

**JSON Rule Files (`--rules`)**

For complex validations, use JSON files:

```json
{
  "rules": [
    {
      "type": "not_null",
      "column": "id",
      "description": "ID must not be null"
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

#### Output Formats & Interpretation

**Standard Output** - Summary table showing rule status:
```
Rule                    Parameters              Status   Failed Records
not_null(id)           column=id               PASSED   0/1000
unique(email)          column=email            FAILED   15/1000
range(age, 18, 99)     column=age, min=18...   PASSED   0/1000
```

**Verbose Output** (`--verbose`) - Includes failure samples:
```
Rule: unique(email)
Status: FAILED
Failed Records: 15/1000
Sample Failed Data:
  Row 23: john@example.com
  Row 45: john@example.com
  Row 67: mary@test.com
```

#### Practical Examples

**1. Basic file validation:**
```bash
vlite check test_data/customers.xlsx --rule "not_null(name)"
```

**2. Multiple rules with verbose output:**
```bash
vlite check test_data/customers.xlsx \
  --rule "unique(email)" \
  --rule "regex(email, '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')" \
  --verbose
```

**3. Comprehensive validation using rules file:**
```bash
vlite check "mysql://root:password@localhost:3306/data_quality.customers" \
  --rules "validation_rules.json" \
  --verbose
```

**4. CSV file with multiple constraints:**
```bash
vlite check examples/sample_data.csv \
  --rule "not_null(customer_id)" \
  --rule "unique(customer_id)" \
  --rule "length(email, 5, 100)" \
  --rule "regex(email, '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')" \
  --verbose
```

#### Exit Codes

- `0` - All rules passed
- `1` - One or more rules failed
- `>1` - Application error (invalid connection, file not found, etc.)

---

### The `schema` Command - Schema Validation

The `schema` command validates tables against JSON schema files, automatically decomposing schemas into atomic rules with intelligent prioritization and aggregation.

#### Basic Syntax & Parameters

```bash
vlite schema <data_source> --rules <schema_file.json> [options]
```

**Required Parameters:**
- `<data_source>` - Database/table identifier (table derived from URL)
- `--rules <file.json>` - Path to JSON schema file

**Options:**
| Option | Description |
|--------|-------------|
| `--output table\|json` | Output format (default: table) |
| `--verbose` | Show detailed information in table mode |
| `--help` | Display command help |

#### Schema File Structure (v1)

**Minimal Structure:**
```json
{
  "rules": [
    { "field": "id", "type": "integer", "required": true },
    { "field": "age", "type": "integer", "min": 0, "max": 120 },
    { "field": "gender", "type": "string", "enum": ["M", "F"] },
    { "field": "email", "type": "string", "required": true },
    { "field": "created_at", "type": "datetime" }
  ],
  "strict_mode": true,
  "case_insensitive": false
}
```

**Supported Field Types:**
- `string`, `integer`, `float`, `boolean`, `date`, `datetime`

**Schema Properties:**
- `field` - Column name (required)
- `type` - Data type (required)
- `required` - Generate NOT_NULL rule if true
- `min`/`max` - Generate RANGE rule for numeric types
- `enum` - Generate ENUM rule with allowed values
- `strict_mode` - Report extra columns as violations
- `case_insensitive` - Case-insensitive column matching

#### Rule Decomposition Logic

The schema command automatically converts each field definition into atomic validation rules:

```
Schema Field → Generated Rules
═══════════════════════════════
{ "field": "age", "type": "integer", "required": true, "min": 0, "max": 120 }
                ↓
1. SCHEMA rule: Check "age" field exists and is integer type
2. NOT_NULL rule: Check "age" has no null values
3. RANGE rule: Check "age" values between 0 and 120
```

**Execution Priority & Skip Logic:**
1. **Field Missing** → Report FIELD_MISSING, skip all other checks for that field
2. **Type Mismatch** → Report TYPE_MISMATCH, skip dependent checks (NOT_NULL, RANGE, ENUM)
3. **All Other Rules** → Execute normally if field exists and type matches

#### Output Formats

**Table Mode (default)** - Column-grouped summary:
```
Column Validation Results
═════════════════════════
Column: id
  ✓ Field exists (integer)
  ✓ Not null constraint

Column: age
  ✓ Field exists (integer)
  ✗ Range constraint (0-120): 5 violations

Column: status
  ✗ Field missing
  ⚠ Dependent checks skipped
```

**JSON Mode** (`--output json`) - Machine-readable format:
```json
{
  "summary": {
    "total_checks": 8,
    "passed": 5,
    "failed": 2,
    "skipped": 1
  },
  "results": [...],
  "fields": {
    "id": { "status": "passed", "checks": [...] },
    "age": { "status": "failed", "checks": [...] }
  },
  "schema_extras": ["unknown_column"]
}
```

#### Practical Examples

**1. Basic schema validation:**
```bash
vlite schema "mysql://root:password@localhost:3306/data_quality.customers" \
  --rules test_data/schema.json
```

**2. JSON output for automation:**
```bash
vlite schema "mysql://user:pass@host:3306/sales.users" \
  --rules schema.json \
  --output json
```

**3. Verbose table output:**
```bash
vlite schema "postgresql://user:pass@localhost:5432/app.customers" \
  --rules customer_schema.json \
  --verbose
```

#### Exit Codes

- `0` - All schema checks passed
- `1` - One or more schema violations found (or --fail-on-error triggered)
- `≥2` - Usage error (invalid JSON, unsupported schema structure, etc.)

---

## Advanced Usage

### Data Source Configuration

#### File-Based Sources

**Supported Formats:**
- CSV, TSV (comma/tab separated values)
- Excel (.xls, .xlsx)
- JSON, JSONL (JSON Lines)

**Examples:**
```bash
# CSV with custom delimiter (auto-detected)
vlite check data/customers.csv --rule "not_null(id)"

# Excel file (auto-detects first sheet)
vlite check reports/monthly_data.xlsx --rule "unique(transaction_id)"

# JSON Lines file
vlite check logs/events.jsonl --rule "not_null(timestamp)"
```

#### Database Sources

**Connection String Formats:**

**MySQL:**
```
mysql://[username[:password]@]host[:port]/database.table
```

**PostgreSQL:**
```
postgresql://[username[:password]@]host[:port]/database.table
```

**SQLite:**
```
sqlite:///[absolute_path_to_file].table
sqlite://[relative_path_to_file].table
```

**Connection Examples:**
```bash
# MySQL with authentication
vlite check "mysql://admin:secret123@db.company.com:3306/sales.customers" --rule "unique(id)"

# PostgreSQL with default port
vlite check "postgresql://analyst@analytics-db/warehouse.orders" --rules validation.json

# SQLite local file
vlite check "sqlite:///data/local.db.users" --rule "not_null(email)"
```

### Validation Rules Deep Dive

#### Rule Parameters & Behavior

**Completeness Rules:**
```bash
# Check for NULL, empty strings, or whitespace-only values
--rule "not_null(email)"
```

**Uniqueness Rules:**
```bash
# Check for exact duplicates (case-sensitive)
--rule "unique(customer_id)"
```

**Validity Rules:**
```bash
# Regex pattern matching
--rule "regex(phone, '^\+?[1-9]\d{1,14}$')"

# Enumerated values (case-sensitive)
--rule "enum(status, 'active', 'inactive', 'pending')"

# Date format validation (MySQL only)
--rule "date_format(created_at, '%Y-%m-%d %H:%i:%s')"
```

**Consistency Rules:**
```bash
# Numeric ranges (inclusive)
--rule "range(age, 0, 150)"
--rule "range(salary, 20000.00, 500000.00)"

# String length constraints
--rule "length(product_code, 8, 12)"
```

#### JSON Rule File Best Practices

**Well-structured rules file:**
```json
{
  "rules": [
    {
      "type": "not_null",
      "column": "customer_id",
      "description": "Customer ID is required for all records"
    },
    {
      "type": "unique",
      "column": "customer_id",
      "description": "Customer ID must be unique across all records"
    },
    {
      "type": "regex",
      "column": "email",
      "params": {
        "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
      },
      "description": "Email must be in valid format"
    },
    {
      "type": "enum",
      "column": "subscription_type",
      "params": {
        "values": ["free", "basic", "premium", "enterprise"]
      },
      "description": "Subscription type must be one of the defined tiers"
    }
  ]
}
```

**Tips:**
- Always include descriptive messages
- Group related rules together
- Use consistent parameter naming
- Validate your JSON syntax before use

### Output & Reporting

#### Understanding Results

**Rule Status Meanings:**
- `PASSED` - All records satisfy the rule
- `FAILED` - Some records violate the rule
- `SKIPPED` - Rule was not executed (dependency failed)

**Failed Record Counts:**
- Format: `failed_count/total_count`
- Example: `15/1000` means 15 out of 1000 records failed

**Sample Data in Verbose Mode:**
- Shows actual values that caused failures
- Limited to first few samples to avoid clutter
- Includes row numbers for easy debugging

#### JSON Output Schema

For the `schema` command with `--output json`, the response follows this structure:

```json
{
  "summary": {
    "total_checks": 12,
    "passed": 8,
    "failed": 3,
    "skipped": 1,
    "execution_time_ms": 1250
  },
  "results": [
    {
      "rule_type": "SCHEMA",
      "column": "age",
      "status": "PASSED",
      "message": "Field exists with correct type",
      "failed_count": 0,
      "total_count": 1000
    }
  ],
  "fields": {
    "age": {
      "status": "passed",
      "checks": ["existence", "type", "not_null", "range"]
    },
    "unknown_field": {
      "status": "extra",
      "checks": []
    }
  },
  "schema_extras": ["unknown_field"]
}
```

**Full JSON schema definition:** `docs/schemas/schema_results.schema.json`

---

## Configuration & Environment

### Configuration Files

ValidateLite uses TOML configuration files for advanced settings. Example files are provided in the `config/` directory:

**Setup:**
```bash
# Copy example configurations
cp config/cli.toml.example config/cli.toml
cp config/core.toml.example config/core.toml
cp config/logging.toml.example config/logging.toml
```

**CLI Configuration (`config/cli.toml`):**
```toml
# Default command options
default_verbose = false
default_quiet = false
max_sample_size = 5

# Output formatting
table_max_width = 120
json_indent = 2
```

**Core Configuration (`config/core.toml`):**
```toml
# Database settings
connection_timeout = 30
query_timeout = 300
max_connections = 10

# Rule execution
parallel_execution = true
batch_size = 1000
```

**Logging Configuration (`config/logging.toml`):**
```toml
level = "INFO"
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
to_file = false
file_path = "logs/validatelite.log"
```

### Environment Variables

**Configuration Path Overrides:**
```bash
export CORE_CONFIG_PATH=/path/to/custom/core.toml
export CLI_CONFIG_PATH=/path/to/custom/cli.toml
export LOGGING_CONFIG_PATH=/path/to/custom/logging.toml
```

**Database Credentials:**
```bash
# Use environment variables for sensitive information
export DB_HOST=localhost
export DB_USER=myuser
export DB_PASSWORD=mypassword
export DB_NAME=mydatabase

# Full connection URLs
export MYSQL_DB_URL="mysql://user:pass@host:3306/db"
export POSTGRESQL_DB_URL="postgresql://user:pass@host:5432/db"
```

**Configuration Loading Order:**
1. Default values (in Pydantic models)
2. Configuration files (TOML)
3. Environment variables
4. Command-line arguments

---

## Troubleshooting

### Common Error Messages

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `File not found: data.csv` | Incorrect file path | Verify file exists and path is correct |
| `Connection failed: Access denied` | Wrong database credentials | Check username/password in connection string |
| `Invalid rule syntax: not_nul(id)` | Typo in rule specification | Fix rule syntax: `not_null(id)` |
| `No rules specified` | Missing --rule or --rules | Add at least one validation rule |
| `Unsupported database type: oracle` | Database not supported | Use MySQL, PostgreSQL, or SQLite |
| `JSON parse error in rules file` | Malformed JSON | Validate JSON syntax in rules file |

### Connection Issues

**Database Connection Problems:**

1. **Test connection manually:**
```bash
# MySQL
mysql -h host -u user -p database

# PostgreSQL
psql -h host -U user -d database
```

2. **Check firewall/network:**
```bash
# Test port connectivity
telnet database_host 3306  # MySQL
telnet database_host 5432  # PostgreSQL
```

3. **Verify credentials:**
- Ensure user has SELECT permissions
- Check password special characters are URL-encoded
- Confirm database and table names are correct

**File Access Problems:**
```bash
# Check file permissions
ls -la data/customers.csv

# Verify file format
file data/customers.csv
head -n 5 data/customers.csv
```

### Performance Tips

**For Large Datasets:**
1. **Use database sources when possible** - Direct database queries are typically faster than loading entire files
2. **Enable batching in config** - Set appropriate `batch_size` in core configuration
3. **Limit sample output** - Use `--quiet` for large-scale validation
4. **Optimize rules** - Put fast rules (like `not_null`) before expensive ones (like `regex`)

**Memory Management:**
```toml
# In config/core.toml
batch_size = 10000        # Process in smaller chunks
max_connections = 5       # Limit concurrent database connections
query_timeout = 600       # Increase timeout for large queries
```

**Parallel Processing:**
```toml
# In config/core.toml
parallel_execution = true # Enable parallel rule execution
```

---

## Getting Help

### Command Line Help
```bash
# General help
vlite --help

# Command-specific help
vlite check --help
vlite schema --help
```

### Documentation Resources
- **[README.md](../README.md)** - Installation and quick start
- **[DEVELOPMENT_SETUP.md](DEVELOPMENT_SETUP.md)** - Development environment setup
- **[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)** - Complete configuration reference
- **[CHANGELOG.md](../CHANGELOG.md)** - Version history and changes

### Support Channels
- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - Questions and community support
- **Documentation** - Comprehensive guides and examples

### Example Files
The project includes working examples in the `examples/` directory:
- `sample_data.csv` - Sample dataset for testing
- `sample_rules.json` - Example validation rules
- `basic_usage.py` - Python API examples

---

*For more advanced usage patterns and API documentation, visit the project repository.*
