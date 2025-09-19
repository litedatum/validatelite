# ValidateLite User Guide

A practical tool for checking data quality and validating type conversions.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Sources](#data-sources)
  - [File Sources](#file-sources)
  - [Database Sources](#database-sources)
  - [Environment Variables](#environment-variables)
- [Command Overview](#command-overview)
  - [vlite check command](#vlite-check-command)
  - [vlite schema command](#vlite-schema-command)
- [Using check command](#using-check-command)
  - [Rule Types](#rule-types)
  - [Completeness Rules](#completeness-rules)
  - [Uniqueness Rules](#uniqueness-rules)
  - [Format Validation Rules](#format-validation-rules)
  - [Value Validation Rules](#value-validation-rules)
  - [Range Validation Rules](#range-validation-rules)
  - [JSON Rule Files](#json-rule-files)
- [Using schema command](#using-schema-command)
  - [Basic Type System](#basic-type-system)
  - [Data Type Definition Syntax](#data-type-definition-syntax)
  - [Desired Type Feature](#desired-type-feature)
  - [Type Compatibility Analysis](#type-compatibility-analysis)
  - [Conversion Validation Strategy](#conversion-validation-strategy)
- [Use Cases](#use-cases)
  - [Case 1: Customer Data Quality Check](#case-1-customer-data-quality-check)
  - [Case 2: E-commerce Order Validation](#case-2-e-commerce-order-validation)
  - [Case 3: Excel Financial Report Validation](#case-3-excel-financial-report-validation)
  - [Case 4: Pre-migration Data Validation](#case-4-pre-migration-data-validation)
  - [Case 5: Legacy System Data Cleanup](#case-5-legacy-system-data-cleanup)
  - [Case 6: API Data Interface Validation](#case-6-api-data-interface-validation)
  - [Case 7: Batch File Validation](#case-7-batch-file-validation)
  - [Case 8: Data Validation in CI/CD](#case-8-data-validation-in-cicd)
  - [Case 9: Data Science Preprocessing Validation](#case-9-data-science-preprocessing-validation)
- [Output and Results](#output-and-results)
  - [Table Output Format](#table-output-format)
  - [JSON Output Format](#json-output-format)
  - [Status Codes](#status-codes)
  - [Output Redirection](#output-redirection)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables-1)
  - [Connection Strings](#connection-strings)
  - [Performance Settings](#performance-settings)
- [Troubleshooting](#troubleshooting)
  - [Common Errors](#common-errors)
  - [Connection Issues](#connection-issues)
  - [Type Conversion Errors](#type-conversion-errors)

---

## Overview

ValidateLite is a Python command-line tool designed for data quality validation. It provides two main validation approaches:

**Quick validation with `vlite check`**
- Perfect for ad-hoc data checks and exploration
- Single rule validation with immediate feedback
- Great for debugging and development

**Schema-based validation with `vlite schema`**
- Comprehensive validation using JSON schema files
- Batch processing for multiple rules and tables
- Features the powerful **Desired Type** functionality for type conversion validation

**What makes ValidateLite special?**

The standout feature is **Desired Type validation** - it doesn't just check if your data fits a schema, it tells you whether your data can be safely converted to a different type. This is invaluable for:
- Data migration planning
- System upgrades
- ETL process validation
- Data quality assessment before transformations

**Supported data sources:**
- Files: CSV, Excel, JSON
- Databases: MySQL, PostgreSQL, SQLite

---

## Installation

### Install from PyPI (Recommended)

```bash
pip install validatelite
```

### Install from Source

```bash
git clone https://github.com/litedatum/validatelite.git
cd validatelite
pip install -e .
```

### Verify Installation

```bash
vlite --version
```

### Dependencies

ValidateLite works with:
- Python 3.8+
- pandas (for Excel/CSV processing)
- SQLAlchemy (for database connections)
- Click (for CLI interface)

Database drivers are optional:
- MySQL: `pip install pymysql`
- PostgreSQL: `pip install psycopg2-binary`
- SQLite: Built into Python

---

## Quick Start

Here are some simple examples to get you started:

### Basic Data Check

```bash
# Check for missing email addresses
vlite check --conn customers.csv --table customers --rule "not_null(email)"
```

### Multiple Checks

```bash
# Run several checks at once
vlite check --conn data.csv --table data \
  --rule "not_null(id)" \
  --rule "unique(email)" \
  --rule "range(age, 18, 99)"
```

### Schema Validation with Type Conversion

```bash
# Check if string data can be converted to proper types
vlite schema --conn messy_data.csv --rules cleanup_schema.json
```

**Sample schema file** (`cleanup_schema.json`):
```json
{
  "rules": [
    {
      "field": "user_id",
      "type": "string",
      "desired_type": "integer",
      "required": true
    },
    {
      "field": "salary",
      "type": "string",
      "desired_type": "float(10,2)",
      "required": true
    }
  ]
}
```

This will tell you exactly which records can't be converted from string to integer/float.

---

## Data Sources

ValidateLite connects to various data sources with a simple connection string approach.

### File Sources

**CSV Files:**
```bash
--conn data.csv
--conn /path/to/data.csv
--conn file://data.csv
```

**Excel Files:**
```bash
--conn report.xlsx
--conn /path/to/report.xlsx

# For multi-sheet Excel files, specify the sheet
--conn report.xlsx --table "Sheet1"
```

**JSON Files:**
```bash
--conn data.json
--conn /path/to/data.json
```

### Database Sources

**MySQL:**
```bash
--conn "mysql://username:password@host:port/database"
--conn "mysql://user:pass@localhost:3306/sales"
```

**PostgreSQL:**
```bash
--conn "postgresql://username:password@host:port/database"
--conn "postgres://user:pass@localhost:5432/analytics"
```

**SQLite:**
```bash
--conn "sqlite:///path/to/database.db"
--conn "sqlite:///data/local.db"
```

### Environment Variables

Keep sensitive connection details out of your commands:

```bash
# Set environment variables
export DB_HOST="localhost"
export DB_USER="analyst"
export DB_PASSWORD="secret123"
export DB_NAME="sales"

# Build connection string
export MYSQL_URL="mysql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:3306/${DB_NAME}"

# Use in commands
vlite check --conn "$MYSQL_URL" --table users --rule "not_null(email)"
```

---

## Command Overview

ValidateLite offers two commands for different validation needs.

### vlite check command

Quick data quality checks for immediate feedback:

```bash
vlite check --conn <data_source> --table <table_name> --rule "<rule>" [options]
```

**Key features:**
- Instant validation without config files
- Flexible inline rule definitions
- Fast feedback for development and debugging
- One rule at a time execution

**Best for:**
- Development phase testing
- Data exploration and analysis
- Quick data quality checks
- Debugging and troubleshooting

### vlite schema command

Comprehensive validation using schema files:

```bash
# Single table validation
vlite schema --conn <data_source> --table <table_name> --rules <schema.json> [options]

# Multi-table validation (tables defined in schema)
vlite schema --conn <data_source> --rules <schema.json> [options]
```

**Key features:**
- Schema-driven with JSON schema files
- Batch validation for multiple tables and rules
- Type conversion analysis with Desired Type functionality
- Structured configuration for reuse and version control

**Best for:**
- Production data quality monitoring
- Pre-migration data validation
- ETL pipeline data validation
- Automated testing in CI/CD

**Schema file syntax differences:**

When using `--table` parameter, your schema should contain field-level rules:
```json
{
  "rules": [
    {
      "field": "email",
      "type": "string(255)",
      "desired_type": "string(100)",
      "required": true
    }
  ]
}
```

When not using `--table` parameter, your schema should contain table-level definitions:
```json
{
  "tables": [
    {
      "name": "users",
      "fields": [
        {
          "field": "email",
          "type": "string(255)",
          "desired_type": "string(100)",
          "required": true
        }
      ]
    }
  ]
}
```

---

## Using check command

ValidateLite provides comprehensive validation rules covering all aspects of data quality.

### Rule Types

| Category | Rule Type | Purpose |
|----------|-----------|---------|
| Completeness | NOT_NULL | Check for missing values |
| Uniqueness | UNIQUE | Find duplicate values |
| Format | REGEX | Validate patterns |
| Format | DATE_FORMAT | Check date formats |
| Value | ENUM | Validate against allowed values |
| Range | RANGE | Check numeric ranges |

### Completeness Rules

**Check for missing values:**

```bash
# Basic not-null check
--rule "not_null(email)"

# With custom message
--rule "not_null(customer_id, 'Customer ID is required')"

# Check multiple columns
--rule "not_null(first_name)"
--rule "not_null(last_name)"
--rule "not_null(email)"
```

### Uniqueness Rules

**Find duplicate records:**

```bash
# Check for duplicate emails
--rule "unique(email)"

# Check for duplicate combinations
--rule "unique(first_name, last_name, birth_date)"

# Check with filter conditions
--rule "unique(username) WHERE status = 'active'"
```

### Format Validation Rules

**REGEX pattern validation:**

```bash
# Email format validation
--rule "regex(email, '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')"

# Phone number format
--rule "regex(phone, '^\\+?1?[0-9]{10,14}$')"

# Product code format
--rule "regex(product_code, '^[A-Z]{2}[0-9]{4}$')"
```

**DATE_FORMAT validation:**

```bash
# Basic syntax
--rule "date_format(column_name, 'format_pattern')"
```

**Supported date format patterns:**

| Pattern | Example | Description |
|---------|---------|-------------|
| `YYYY-MM-DD` | 2023-12-25 | ISO date format |
| `MM/DD/YYYY` | 12/25/2023 | US date format |
| `DD/MM/YYYY` | 25/12/2023 | European date format |
| `YYYYMMDD` | 20231225 | Compact date format |
| `DD.MM.YYYY` | 25.12.2023 | German date format |
| `YYYY-MM-DD HH:MI:SS` | 2023-12-25 14:30:00 | DateTime format |
| `DD-MMM-YYYY` | 25-Dec-2023 | Month abbreviation format |
| `YYYY/MM/DD` | 2023/12/25 | Slash-separated format |

**Format components:**
- `YYYY` or `yyyy` - Four-digit year
- `MM` or `mm` - Two-digit month (01-12)
- `DD` or `dd` - Two-digit day (01-31)
- `HH` or `hh` - Two-digit hour (00-23)
- `MI` or `mi` - Two-digit minute (00-59)
- `SS` or `ss` - Two-digit second (00-59)

```bash
# Examples
--rule "date_format(created_at, 'YYYY-MM-DD HH:MI:SS')"
--rule "date_format(birth_date, 'MM/DD/YYYY')"
--rule "date_format(event_date, 'DD.MM.YYYY')"
```

**Database support:**
- MySQL: Native support for all formats
- PostgreSQL: Uses regex pre-validation + Python verification
- SQLite: Uses custom function validation

### Value Validation Rules

**ENUM (allowed values) validation:**

```bash
# Status field validation
--rule "enum(status, 'active', 'inactive', 'pending')"

# Priority levels
--rule "enum(priority, 'low', 'medium', 'high', 'critical')"

# Boolean-like values
--rule "enum(is_verified, 'true', 'false', '1', '0')"
```

### Range Validation Rules

**Numeric range validation:**

```bash
# Age validation
--rule "range(age, 0, 120)"

# Price validation with decimals
--rule "range(price, 0.01, 999999.99)"

# Percentage validation
--rule "range(completion_rate, 0.0, 100.0)"

# Year validation
--rule "range(birth_year, 1900, 2024)"
```

### JSON Rule Files

For complex validation scenarios, use JSON rule files:

**Basic rule file** (`validation_rules.json`):
```json
{
  "rules": [
    {
      "name": "email_required",
      "type": "NOT_NULL",
      "target": {
        "database": "sales_db",
        "table": "customers",
        "column": "email"
      },
      "severity": "HIGH"
    },
    {
      "name": "unique_customer_email",
      "type": "UNIQUE",
      "target": {
        "database": "sales_db",
        "table": "customers",
        "column": "email"
      },
      "severity": "HIGH"
    },
    {
      "name": "valid_age_range",
      "type": "RANGE",
      "target": {
        "database": "sales_db",
        "table": "customers",
        "column": "age"
      },
      "parameters": {
        "min_value": 18,
        "max_value": 99
      },
      "severity": "MEDIUM"
    }
  ]
}
```

**Using rule files:**
```bash
vlite check --conn "mysql://user:pass@host:3306/sales_db" \
  --table customers --rules validation_rules.json
```

---

## Using schema command

This is where ValidateLite really shines! ValidateLite provides industry-leading type system and data conversion validation capabilities.

### Basic Type System

ValidateLite supports these fundamental data types:

| Type | Description | Examples |
|------|-------------|----------|
| `string` | Text data | "John", "Hello World" |
| `integer` | Whole numbers | 42, -17, 0 |
| `float` | Decimal numbers | 3.14, -0.5, 100.00 |
| `boolean` | True/false values | true, false |
| `date` | Date values | 2023-12-25 |
| `datetime` | Date and time values | 2023-12-25 14:30:00 |

### Data Type Definition Syntax

ValidateLite provides intuitive data type definition syntax with precise type constraints:

#### String Type Definitions

```json
{
  "field": "username",
  "type": "string(50)",          // Max length 50 characters
  "required": true
}
```

**String type definition syntax:**
- `string(100)` - Max length 100 characters
- `string(10,50)` - Length between 10-50 characters
- `string` - No length restrictions

#### Float Type Definitions

```json
{
  "field": "price",
  "type": "float(10,2)",         // Precision 10, scale 2
  "required": true
}
```

**Float type definition syntax:**
- `float(10,2)` - Precision 10, scale 2 decimal places
- `float(8,3)` - Precision 8, scale 3 decimal places
- `float` - Standard float

#### DateTime Type Definitions

```json
{
  "field": "created_at",
  "type": "datetime('YYYY-MM-DD HH:MI:SS')",    // Specific datetime format
  "required": true
}
```

**DateTime type definition syntax:**
- `datetime('YYYY-MM-DD HH:MI:SS')` - Specific datetime format
- `date('YYYY-MM-DD')` - Specific date format
- `datetime` - Standard datetime format

### Desired Type Feature

**Desired Type** is ValidateLite's most valuable feature! It lets you validate whether data can be safely converted to a target type, which is crucial for data migration, system upgrades, and data cleaning scenarios.

#### Why Desired Type Matters

Traditional validation just checks if data matches a schema. Desired Type goes further - it tells you if your messy string data can actually be converted to proper types like integers or dates.

**Example scenario:**
You have a CSV file where everything is stored as strings:
- `user_id: "123"` (should be integer)
- `salary: "75000.50"` (should be float)
- `join_date: "2023-01-15"` (should be date)

Desired Type validation will tell you exactly which records can be converted and which ones will cause problems.

#### Using Desired Type

Desired Type uses the same type definition syntax for precise validation:

```json
{
  "transactions": {
    "rules": [
      {
        "field": "amount",
        "type": "string",                         // Current: string data
        "desired_type": "float(12,2)",           // Target: decimal with 12 precision, 2 scale
        "required": true
      },
      {
        "field": "transaction_date",
        "type": "string",                         // Current: string data
        "desired_type": "datetime('YYYY-MM-DD')",   // Target: specific datetime format
        "required": true
      },
      {
        "field": "description",
        "type": "string(500)",                   // Current: long strings
        "desired_type": "string(200)",           // Target: shorter strings
        "required": true
      }
    ]
  }
}
```

#### Application in Desired Type

Desired Type supports the same type definition syntax for precise validation:

```json
{
  "migration_analysis": {
    "rules": [
      {
        "field": "legacy_id",
        "type": "string(50)",                    // Current: string with max 50 chars
        "desired_type": "integer",               // Target: integer
        "required": true
      },
      {
        "field": "legacy_amount",
        "type": "string",                        // Current: free-form string
        "desired_type": "float(10,2)",           // Target: precise decimal
        "required": true
      },
      {
        "field": "legacy_timestamp",
        "type": "string",                        // Current: string timestamp
        "desired_type": "datetime('YYYY-MM-DD HH:MI:SS')", // Target: structured datetime
        "required": true
      }
    ]
  }
}
```

**What you get from Desired Type validation:**
- Count of records that can be converted successfully
- Count of problematic records that would fail conversion
- Sample data showing exactly what the problems are
- Conversion feasibility percentage
- Specific error patterns in your data

### Type Compatibility Analysis

ValidateLite analyzes type conversion compatibility and reports three possible outcomes:

#### Compatible Conversion
All data can be safely converted to the desired type.

**Example:**
```
Field: user_id
Current Type: string → Desired Type: integer
Result: ✅ COMPATIBLE (500/500 records can be converted)
```

#### Partial Conversion
Some data can be converted, but some records have issues.

**Example:**
```
Field: salary
Current Type: string → Desired Type: float(10,2)
Result: ⚠️ PARTIAL (487/500 records can be converted)
Issues: 13 records contain non-numeric characters
```

#### Incompatible Conversion
Most or all data cannot be converted to the desired type.

**Example:**
```
Field: comments
Current Type: string → Desired Type: integer
Result: ❌ INCOMPATIBLE (0/500 records can be converted)
Issues: Text data cannot be converted to integers
```

### Conversion Validation Strategy

ValidateLite uses smart conversion validation strategies:

#### String to Numeric Conversion
- Removes common formatting (spaces, commas, currency symbols)
- Handles scientific notation
- Validates decimal precision and scale
- Checks for overflow conditions

#### String to Date/DateTime Conversion
- Attempts multiple common date formats
- Validates actual date values (no Feb 31st)
- Handles timezone considerations
- Checks for impossible dates

#### String Length Validation
- Measures actual character length
- Considers UTF-8 encoding
- Validates against target length constraints

#### Type Downgrading Validation
- Checks if larger types can fit into smaller ones
- Validates precision/scale requirements for decimals
- Ensures no data loss during conversion

**Comprehensive validation output:**
When you run Desired Type validation, you get detailed information about:
- Which fields can be safely converted
- Which data needs cleaning
- Specific failure samples and suggested fixes

---

## Use Cases

This section provides complete usage scenarios showcasing Desired Type functionality.

### Case 1: Customer Data Quality Check

**Background:** You have a customer database that's been collecting data for years. Data quality has declined and you need to assess what can be cleaned up.

**Dataset:** Customer table with mixed data quality

```csv
customer_id,name,email,phone,age,registration_date,is_premium
1,John Smith,john@email.com,555-1234,25,2023-01-15,true
2,"Jane, Doe",jane@email.com,,35,01/15/2023,1
3,Bob Johnson,invalid-email,555-ABCD,age_unknown,2023/1/15,yes
4,"Mike Wilson",mike@email.com,5551234567,45,2023-01-15,false
```

**Quick validation with check command:**

```bash
# Check for basic data quality issues
vlite check --conn customers.csv --table customers \
  --rule "not_null(customer_id)" \
  --rule "unique(email)" \
  --rule "regex(email, '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')" \
  --rule "not_null(phone)" \
  --verbose
```

**Schema validation for cleanup planning:**

Create `customer_cleanup.json`:
```json
{
  "rules": [
    {
      "field": "customer_id",
      "type": "string",
      "desired_type": "integer",
      "required": true
    },
    {
      "field": "age",
      "type": "string",
      "desired_type": "integer",
      "required": false,
      "min": 18,
      "max": 100
    },
    {
      "field": "registration_date",
      "type": "string",
      "desired_type": "date('YYYY-MM-DD')",
      "required": true
    },
    {
      "field": "is_premium",
      "type": "string",
      "desired_type": "boolean",
      "required": true
    }
  ]
}
```

```bash
# Analyze what can be cleaned up
vlite schema --conn customers.csv --rules customer_cleanup.json --verbose
```

This tells you exactly which customer records have data quality issues and what types of problems exist.

### Case 2: E-commerce Order Validation

**Background:** Validate daily order data before processing payments and shipments.

```bash
# Comprehensive order validation
vlite check --conn "mysql://user:pass@db:3306/ecommerce" --table orders \
  --rule "not_null(order_id)" \
  --rule "unique(order_id)" \
  --rule "not_null(customer_id)" \
  --rule "range(total_amount, 0.01, 999999.99)" \
  --rule "enum(status, 'pending', 'paid', 'shipped', 'delivered', 'cancelled')" \
  --rule "date_format(created_at, 'YYYY-MM-DD HH:MI:SS')" \
  --verbose
```

### Case 3: Excel Financial Report Validation

**Background:** Monthly financial reports come in Excel format and need validation before importing into the accounting system.

**Excel file structure** (`monthly_report.xlsx`):
- Sheet: "Revenue"
- Columns: transaction_id, amount, currency, transaction_date, category

**Multi-sheet validation:**

First, check what sheets are available:
```bash
vlite schema --conn monthly_report.xlsx --rules basic_schema.json
```

Then validate specific sheets:
```bash
# Validate Revenue sheet
vlite schema --conn monthly_report.xlsx --table "Revenue" --rules revenue_schema.json

# Validate Expenses sheet
vlite schema --conn monthly_report.xlsx --table "Expenses" --rules expense_schema.json
```

**Revenue validation schema** (`revenue_schema.json`):
```json
{
  "rules": [
    {
      "field": "transaction_id",
      "type": "string",
      "desired_type": "string(20)",
      "required": true
    },
    {
      "field": "amount",
      "type": "string",
      "desired_type": "float(15,2)",
      "required": true,
      "min": 0.01
    },
    {
      "field": "transaction_date",
      "type": "string",
      "desired_type": "date('YYYY-MM-DD')",
      "required": true
    }
  ],
  "strict_mode": true
}
```

### Case 4: Pre-migration Data Validation

**Background:** Before migrating from a legacy system to a modern database, you need to validate that all data can be properly converted and identify cleanup requirements.

**Legacy system data characteristics:**
- Everything stored as VARCHAR
- Inconsistent date formats
- Mixed boolean representations
- Unreliable numeric formatting

**Migration readiness schema** (`migration_readiness.json`):
```json
{
  "users": {
    "rules": [
      {
        "field": "user_id",
        "type": "string(50)",
        "desired_type": "integer",
        "required": true
      },
      {
        "field": "email",
        "type": "string(500)",
        "desired_type": "string(255)",
        "required": true
      },
      {
        "field": "created_date",
        "type": "string",
        "desired_type": "date('YYYY-MM-DD')",      // Target: standard date format
        "required": true
      },
      {
        "field": "last_login",
        "type": "string",
        "desired_type": "datetime('YYYY-MM-DD HH:MI:SS')", // Target: standard datetime
        "required": false
      },
      {
        "field": "is_active",
        "type": "string",
        "desired_type": "boolean",
        "required": true
      }
    ],
    "strict_mode": false
  }
}
```

```bash
# Analyze migration readiness
vlite schema --conn "mysql://legacy:pass@old-db:3306/legacy_db" \
  --rules migration_readiness.json --output json > migration_report.json

# Get detailed conversion analysis
vlite schema --conn "mysql://legacy:pass@old-db:3306/legacy_db" \
  --rules migration_readiness.json --verbose
```

**Expected output:**
```
Migration Readiness Report
==========================

Table: users
Total records: 10,543

Type conversion analysis:
┌─────────────────┬──────────┬──────────┬──────────┬─────────────────┐
│ Field           │ From     │ To       │ Status   │ Issues          │
├─────────────────┼──────────┼──────────┼──────────┼─────────────────┤
│ user_id         │ string   │ integer  │ ✅ OK     │ -               │
│ email           │ string   │ string   │ ⚠️ WARN   │ 12 too long     │
│ created_date    │ string   │ date     │ ⚠️ WARN   │ 45 bad formats  │
│ last_login      │ string   │ datetime │ ❌ ISSUES │ 234 bad formats │
│ is_active       │ string   │ boolean  │ ⚠️ WARN   │ 8 unclear values│
└─────────────────┴──────────┴──────────┴──────────┴─────────────────┘

Field: created_date
  ✓ Field exists (string)
  ✓ Non-null constraint
  ✗ Type conversion validation (string → date('YYYY-MM-DD')): 156 incompatible records

Failure samples:
  Row 12: "2023/12/25"    (slash format, needs standardization)
  Row 34: "Dec 25, 2023"  (English format)
  Row 67: "25.12.2023"    (European format)

Recommended cleanup:
1. Standardize date formats to YYYY-MM-DD
2. Trim email fields that exceed 255 characters
3. Normalize boolean values (true/false only)
4. Fix malformed datetime values
```

This gives you a complete roadmap for data cleanup before migration.

### Case 5: Legacy System Data Cleanup

**Background:** You inherit a legacy system with years of accumulated data quality issues. You need to understand the scope of cleanup required.

**Legacy data issues:**
- Mixed encodings
- Inconsistent data entry
- No validation for years
- Multiple date formats
- Currency symbols in numeric fields

**Cleanup assessment schema** (`legacy_cleanup.json`):
```json
{
  "rules": [
    {
      "field": "customer_id",
      "type": "string",
      "desired_type": "integer",
      "required": true
    },
    {
      "field": "first_name",
      "type": "string(1000)",
      "desired_type": "string(50)",
      "required": true
    },
    {
      "field": "salary",
      "type": "string",
      "desired_type": "float(10,2)",
      "required": false,
      "min": 0
    },
    {
      "field": "hire_date",
      "type": "string",
      "desired_type": "date('YYYY-MM-DD')",
      "required": true
    },
    {
      "field": "department_id",
      "type": "string",
      "desired_type": "integer",
      "required": true
    }
  ],
  "strict_mode": false
}
```

**Cleanup process:**

```bash
# Step 1: Assess current state
vlite schema --conn legacy_data.csv --rules legacy_cleanup.json \
  --output json > cleanup_assessment.json

# Step 2: Get detailed samples
vlite schema --conn legacy_data.csv --rules legacy_cleanup.json \
  --verbose > cleanup_details.txt

# Step 3: Validate after initial cleanup
# (after running data cleaning scripts)
vlite schema --conn cleaned_data.csv --rules legacy_cleanup.json \
  --verbose
```

**Sample output showing improvement:**
```
Before cleanup:
  salary field: 1,234 records with currency symbols ($, €, £)
  hire_date field: 567 records with inconsistent formats

After cleanup:
  salary field: 23 records still need manual review
  hire_date field: 12 records still need manual review
```

### Case 6: API Data Interface Validation

**Background:** Validate data received from external APIs before processing.

**API validation schema** (`api_validation.json`):
```json
{
  "rules": [
    {
      "field": "user_id",
      "type": "string",
      "desired_type": "integer",
      "required": true
    },
    {
      "field": "timestamp",
      "type": "string",
      "desired_type": "datetime('YYYY-MM-DD HH:MI:SS')", // Internal: standard format
      "required": true
    },
    {
      "field": "amount",
      "type": "string",
      "desired_type": "float(12,2)",
      "required": true,
      "min": 0
    }
  ]
}
```

```bash
# Validate API response data
vlite schema --conn api_response.json --rules api_validation.json
```

### Case 7: Batch File Validation

**Background:** Process multiple files in a batch operation.

```bash
#!/bin/bash
# validate_batch.sh

for file in data_files/*.csv; do
  echo "Validating $file..."
  vlite schema --conn "$file" --rules batch_schema.json \
    --output json > "reports/$(basename "$file" .csv)_report.json"
done

echo "Validation complete. Check reports/ directory for results."
```

### Case 8: Data Validation in CI/CD

**Background:** Integrate data quality checks into your CI/CD pipeline to catch data compatibility issues before they reach production.

**Create `.github/workflows/data-validation.yml`:**

```yaml
name: Data Quality and Type Conversion Validation
on:
  push:
    paths:
      - 'data/**'
      - 'schemas/**'
  pull_request:
    paths:
      - 'data/**'
      - 'schemas/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Install ValidateLite
        run: pip install validatelite

      - name: Basic data quality validation
        run: |
          vlite check --conn data/customers.csv --table customers \
            --rules schemas/customer_rules.json

      - name: Type conversion feasibility analysis
        run: |
          vlite schema --conn data/legacy_data.xlsx \
            --rules schemas/modernization_schema.json \
            --output json > type_conversion_report.json

      - name: Check conversion compatibility
        run: |
          # Check for incompatible type conversions
          python scripts/check_conversion_feasibility.py type_conversion_report.json

      - name: Upload validation reports
        uses: actions/upload-artifact@v2
        with:
          name: validation-reports
          path: |
            type_conversion_report.json
            validation_*.log
```

**Helper script** (`scripts/check_conversion_feasibility.py`):

```python
#!/usr/bin/env python3
import json
import sys

def check_conversion_feasibility(report_file):
    """Check type conversion feasibility"""
    with open(report_file, 'r') as f:
        report = json.load(f)

    failed_conversions = []
    for result in report.get('results', []):
        if result.get('rule_type') == 'DESIRED_TYPE' and result.get('status') == 'FAILED':
            failed_conversions.append({
                'field': result.get('column'),
                'failed_count': result.get('failed_count'),
                'total_count': result.get('total_count'),
                'failure_rate': result.get('failed_count', 0) / result.get('total_count', 1)
            })

    if failed_conversions:
        print("❌ Type conversion issues found:")
        for conversion in failed_conversions:
            print(f"  - Field {conversion['field']}: {conversion['failed_count']}/{conversion['total_count']} "
                  f"records cannot convert ({conversion['failure_rate']:.1%})")

        # Block merge if failure rate exceeds threshold
        max_failure_rate = max(c['failure_rate'] for c in failed_conversions)
        if max_failure_rate > 0.05:  # 5% threshold
            print(f"❌ Type conversion failure rate {max_failure_rate:.1%} exceeds 5% threshold. Blocking merge.")
            sys.exit(1)
        else:
            print(f"⚠️  Type conversion failure rate {max_failure_rate:.1%} is within acceptable range.")
    else:
        print("✅ All type conversion validations passed.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python check_conversion_feasibility.py <report.json>")
        sys.exit(1)

    check_conversion_feasibility(sys.argv[1])
```

This CI/CD pipeline provides:
1. **Early problem detection** - Find data compatibility issues before code merge
2. **Automated validation** - No manual data quality checks needed
3. **Block problematic merges** - Prevent incompatible data changes from reaching main branch
4. **Detailed reporting** - Help developers understand specific issues

### Case 9: Data Science Preprocessing Validation

**Background:** Data scientists need to preprocess raw data including cleaning, type conversion, and format standardization. Before starting model development, it's crucial to validate data quality and assess conversion feasibility.

**Raw survey dataset** (`raw_survey_data.csv`):

```csv
id,age,income,satisfaction_score,join_date,is_premium,location
1,25.5,50000.0,8.2,2023-01-15,True,New York
2,,"60K",7.8,15/01/2023,1,California
3,thirty,75000,nine,2023-1-20,yes,Texas
4,45,$85000,6.5,2023/01/22,0,Florida
5,52,95000.50,4.9,Jan 25 2023,false,Washington
```

**Preprocessing requirements:**
1. Age field needs conversion to integer (handle text and decimals)
2. Income field needs standardization to numeric (remove currency symbols and letters)
3. Satisfaction scores need conversion to standard numeric values
4. Date formats need standardization
5. Boolean fields need standardization
6. Geographic locations need standardization

**Create preprocessing validation schema** (`preprocessing_schema.json`):

```json
{
  "rules": [
    {
      "field": "id",
      "type": "string",
      "desired_type": "integer",
      "required": true,
      "description": "Unique user identifier"
    },
    {
      "field": "age",
      "type": "string",
      "desired_type": "integer",
      "required": true,
      "min": 18,
      "max": 100,
      "description": "Age needs conversion to integer, range 18-100"
    },
    {
      "field": "income",
      "type": "string",
      "desired_type": "float(10,2)",
      "required": true,
      "min": 0,
      "description": "Income needs conversion to numeric, remove non-digit characters"
    },
    {
      "field": "satisfaction_score",
      "type": "string",
      "desired_type": "float(3,1)",
      "required": true,
      "min": 1.0,
      "max": 10.0,
      "description": "Satisfaction score, 1-10 scale"
    },
    {
      "field": "join_date",
      "type": "string",
      "desired_type": "date('YYYY-MM-DD')",
      "required": true,
      "description": "Join date, standardize to YYYY-MM-DD format"
    },
    {
      "field": "is_premium",
      "type": "string",
      "desired_type": "boolean",
      "required": true,
      "description": "Premium status, standardize to true/false"
    },
    {
      "field": "location",
      "type": "string(50)",
      "desired_type": "string(20)",
      "required": true,
      "description": "Geographic location, standardize length"
    }
  ],
  "strict_mode": false,
  "case_insensitive": true
}
```

**Run preprocessing validation:**

```bash
# Step 1: Check data quality and conversion feasibility
vlite schema --conn raw_survey_data.csv \
  --rules preprocessing_schema.json \
  --output json > preprocessing_report.json

# Step 2: Analyze conversion issues
vlite schema --conn raw_survey_data.csv \
  --rules preprocessing_schema.json \
  --verbose
```

**Expected output:**

```
Data Preprocessing Validation Report
====================================

Table: raw_survey_data
Total records: 5

Conversion validation results:
┌─────────────────────┬──────────┬──────────┬──────────┬────────────────┐
│ Field               │ From     │ To       │ Status   │ Issues         │
├─────────────────────┼──────────┼──────────┼──────────┼────────────────┤
│ id                  │ string   │ integer  │ ✅ OK     │ -              │
│ age                 │ string   │ integer  │ ⚠️ PARTIAL│ 2 text values  │
│ income              │ string   │ float    │ ⚠️ PARTIAL│ Format issues  │
│ satisfaction_score  │ string   │ float    │ ⚠️ PARTIAL│ 1 text value   │
│ join_date          │ string   │ date     │ ❌ ISSUES │ Multiple formats│
│ is_premium         │ string   │ boolean  │ ⚠️ PARTIAL│ Format issues  │
│ location           │ string   │ string   │ ✅ OK     │ -              │
└─────────────────────┴──────────┴──────────┴──────────┴────────────────┘

Detailed issue analysis:
• age field: Row 2 (empty), Row 3 ("thirty") cannot convert to integer
• income field: Row 2 ("60K"), Row 4 ("$85000") contain non-numeric characters
• satisfaction_score field: Row 3 ("nine") cannot convert to numeric
• join_date field: Detected 3 different date formats, needs standardization
• is_premium field: Multiple boolean representations (True/1/yes/0/false)

Data cleaning recommendations:
1. Establish missing value strategy for age field
2. Standardize income field format, remove symbols and units
3. Create text-to-numeric mapping rules (nine→9)
4. Standardize date format parsing rules
5. Unify boolean value representation standards
```

**Create data cleaning script** (`clean_data.py`):

```python
import pandas as pd
import re
from datetime import datetime

def clean_survey_data(input_file, output_file):
    """Clean survey data"""
    df = pd.read_csv(input_file)

    # Clean age field
    def clean_age(age):
        if pd.isna(age):
            return None
        if str(age).lower() == 'thirty':
            return 30
        try:
            return int(float(str(age)))
        except:
            return None

    # Clean income field
    def clean_income(income):
        if pd.isna(income):
            return None
        # Remove all non-digit characters (except decimal point)
        cleaned = re.sub(r'[^\d.]', '', str(income))
        try:
            return float(cleaned)
        except:
            return None

    # Clean satisfaction score
    def clean_satisfaction(score):
        if pd.isna(score):
            return None
        if str(score).lower() == 'nine':
            return 9.0
        try:
            return float(score)
        except:
            return None

    # Clean date field
    def clean_date(date_str):
        if pd.isna(date_str):
            return None

        # Try multiple date formats
        formats = ['%Y-%m-%d', '%d/%m/%Y', '%Y-%m-%d', '%Y/%m/%d', '%b %d %Y']
        for fmt in formats:
            try:
                return datetime.strptime(str(date_str), fmt).strftime('%Y-%m-%d')
            except:
                continue
        return None

    # Clean boolean field
    def clean_boolean(value):
        if pd.isna(value):
            return False
        str_val = str(value).lower()
        return str_val in ['true', '1', 'yes', 'y']

    # Apply cleaning rules
    df['age'] = df['age'].apply(clean_age)
    df['income'] = df['income'].apply(clean_income)
    df['satisfaction_score'] = df['satisfaction_score'].apply(clean_satisfaction)
    df['join_date'] = df['join_date'].apply(clean_date)
    df['is_premium'] = df['is_premium'].apply(clean_boolean)
    df['location'] = df['location'].str.strip()

    # Save cleaned data
    df.to_csv(output_file, index=False)
    print(f"Cleaning complete, results saved to {output_file}")

if __name__ == '__main__':
    clean_survey_data('raw_survey_data.csv', 'cleaned_survey_data.csv')
```

**Validate cleaned data:**

```bash
# Validate cleaned data
vlite schema --conn cleaned_survey_data.csv \
  --rules preprocessing_schema.json \
  --verbose

# Output should show all conversion validations passing
```

**Workflow script** (`data_preprocessing_workflow.sh`):

```bash
#!/bin/bash

echo "Starting data preprocessing workflow..."

# 1. Initial data quality assessment
echo "Step 1: Assess raw data quality"
vlite schema --conn raw_survey_data.csv \
  --rules preprocessing_schema.json \
  --output json > initial_assessment.json

# 2. Execute data cleaning
echo "Step 2: Execute data cleaning"
python clean_data.py

# 3. Validate cleaning results
echo "Step 3: Validate cleaning results"
vlite schema --conn cleaned_survey_data.csv \
  --rules preprocessing_schema.json \
  --output json > final_validation.json

# 4. Generate data quality report
echo "Step 4: Generate data quality report"
python generate_quality_report.py initial_assessment.json final_validation.json

echo "Data preprocessing workflow complete!"
```

This scenario shows data scientists how to use ValidateLite for:
1. **Data quality assessment** - Understanding raw data issues
2. **Conversion feasibility analysis** - Evaluating cleaning strategy effectiveness
3. **Cleaning validation** - Ensuring processed data meets modeling requirements
4. **Automated workflow** - Standardized data preprocessing pipeline

---

## Output and Results

ValidateLite provides two main output formats: table format and JSON format. Understanding the output helps you quickly identify data quality issues.

### Table Output Format

**Default table output** provides a clear overview:

```
Data Validation Results
=======================

Connection: customers.csv
Table: customers
Rules executed: 5
Validation time: 1.23s

┌─────────────────┬──────────┬──────────┬──────────┬─────────────────┐
│ Rule            │ Type     │ Status   │ Failed   │ Details         │
├─────────────────┼──────────┼──────────┼──────────┼─────────────────┤
│ email_required  │ NOT_NULL │ ✅ PASS   │ 0/1000   │ All records OK  │
│ unique_email    │ UNIQUE   │ ❌ FAIL   │ 12/1000  │ 12 duplicates   │
│ valid_age       │ RANGE    │ ⚠️ WARN   │ 3/1000   │ 3 out of range  │
│ phone_format    │ REGEX    │ ✅ PASS   │ 0/1000   │ All valid       │
│ status_enum     │ ENUM     │ ❌ FAIL   │ 25/1000  │ Invalid values  │
└─────────────────┴──────────┴──────────┴──────────┴─────────────────┘

Overall Status: FAILED (2 rules failed)
```

**Verbose table output** includes sample data:

```bash
vlite check --conn data.csv --table users --rule "unique(email)" --verbose
```

```
Validation Results (Verbose)
============================

Rule: unique_email
Type: UNIQUE
Status: ❌ FAILED
Failed records: 12 out of 1000 total

Sample failures:
┌─────┬─────────────────────┬─────────────┐
│ Row │ Email               │ Occurrences │
├─────┼─────────────────────┼─────────────┤
│ 145 │ john@email.com      │ 3           │
│ 298 │ mary@email.com      │ 2           │
│ 456 │ bob@company.com     │ 2           │
│ 789 │ admin@system.com    │ 5           │
└─────┴─────────────────────┴─────────────┘

Recommendation: Review duplicate email addresses and decide on deduplication strategy.
```

### JSON Output Format

**JSON output** is perfect for automation and integration:

```bash
vlite schema --conn data.csv --rules schema.json --output json
```

```json
{
  "validation_summary": {
    "connection": "data.csv",
    "table": "users",
    "total_rules": 5,
    "passed_rules": 3,
    "failed_rules": 2,
    "warning_rules": 0,
    "validation_time": "1.23s",
    "overall_status": "FAILED"
  },
  "results": [
    {
      "rule_id": "email_required",
      "rule_type": "NOT_NULL",
      "column": "email",
      "status": "PASSED",
      "total_count": 1000,
      "failed_count": 0,
      "failure_rate": 0.0,
      "message": "All records have non-null email values"
    },
    {
      "rule_id": "email_unique",
      "rule_type": "UNIQUE",
      "column": "email",
      "status": "FAILED",
      "total_count": 1000,
      "failed_count": 12,
      "failure_rate": 0.012,
      "message": "Found 12 duplicate email addresses",
      "sample_data": [
        {"row": 145, "email": "john@email.com", "occurrences": 3},
        {"row": 298, "email": "mary@email.com", "occurrences": 2}
      ]
    },
    {
      "rule_id": "salary_conversion",
      "rule_type": "DESIRED_TYPE",
      "column": "salary",
      "status": "FAILED",
      "current_type": "string",
      "desired_type": "float(10,2)",
      "total_count": 1000,
      "failed_count": 45,
      "failure_rate": 0.045,
      "message": "45 records cannot be converted from string to float(10,2)",
      "conversion_analysis": {
        "compatible_records": 955,
        "incompatible_records": 45,
        "common_issues": [
          "Currency symbols ($, €, £)",
          "Thousands separators (,)",
          "Text values (N/A, TBD)"
        ]
      }
    }
  ]
}
```

### Status Codes

ValidateLite uses clear exit codes for automation:

| Exit Code | Meaning | Description |
|-----------|---------|-------------|
| 0 | Success | All validations passed |
| 1 | Validation Failed | One or more rules failed |
| 2 | Usage Error | Invalid command line arguments |
| 3 | Connection Error | Cannot connect to data source |
| 4 | File Error | File not found or permission issues |
| 5 | Configuration Error | Invalid schema or rule format |

**Using exit codes in scripts:**

```bash
#!/bin/bash

vlite check --conn data.csv --table users --rule "not_null(email)"
exit_code=$?

case $exit_code in
  0)
    echo "✅ Data validation passed"
    ;;
  1)
    echo "❌ Data validation failed - check the output above"
    exit 1
    ;;
  *)
    echo "💥 Validation error (code: $exit_code)"
    exit $exit_code
    ;;
esac
```

### Output Redirection

**Save results to files:**

```bash
# Save table output
vlite check --conn data.csv --table users --rule "unique(email)" > validation_report.txt

# Save JSON output
vlite schema --conn data.csv --rules schema.json --output json > results.json

# Save both stdout and stderr
vlite check --conn data.csv --table users --rule "unique(email)" &> full_output.log

# Append to existing files
vlite check --conn data.csv --table users --rule "range(age, 0, 120)" >> daily_checks.log
```

**Parse JSON results:**

```python
import json

# Load validation results
with open('results.json', 'r') as f:
    results = json.load(f)

# Check overall status
if results['validation_summary']['overall_status'] == 'FAILED':
    print("Validation failed!")

    # Get failed rules
    failed_rules = [r for r in results['results'] if r['status'] == 'FAILED']
    for rule in failed_rules:
        print(f"Rule {rule['rule_id']}: {rule['failed_count']} failures")
```

---

## Configuration

ValidateLite supports various configuration methods, from simple command-line parameters to complex configuration files for different usage scenarios.

### Environment Variables

**Database connections:**
```bash
# MySQL connection
export DB_HOST="production-db.company.com"
export DB_USER="data_analyst"
export DB_PASSWORD="secure_password"
export DB_NAME="analytics"
export MYSQL_URL="mysql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:3306/${DB_NAME}"

# PostgreSQL connection
export PG_HOST="warehouse.company.com"
export PG_USER="reporting_user"
export PG_PASSWORD="another_secure_password"
export PG_NAME="data_warehouse"
export POSTGRES_URL="postgresql://${PG_USER}:${PG_PASSWORD}@${PG_HOST}:5432/${PG_NAME}"

# Use in commands
vlite check --conn "$MYSQL_URL" --table customers --rule "not_null(email)"
```

**Performance tuning:**
```bash
# Query timeouts (in seconds)
export VLITE_QUERY_TIMEOUT=300
export VLITE_CONNECTION_TIMEOUT=30

# Memory limits
export VLITE_MAX_SAMPLE_SIZE=1000
export VLITE_BATCH_SIZE=10000

# Parallel processing
export VLITE_MAX_WORKERS=4
```

### Connection Strings

**Advanced connection string options:**

```bash
# MySQL with SSL
--conn "mysql://user:pass@host:3306/db?ssl_ca=/path/to/ca.pem&ssl_cert=/path/to/cert.pem"

# PostgreSQL with connection pool
--conn "postgresql://user:pass@host:5432/db?pool_size=10&max_overflow=20"

# SQLite with custom timeout
--conn "sqlite:///data.db?timeout=20"
```

**Connection string with table specification:**
```bash
# Include table name in connection string
--conn "mysql://user:pass@host:3306/database.table_name"

# Override with command line parameter
--conn "mysql://user:pass@host:3306/database.table_name" --table "different_table"
```

### Performance Settings

**For large datasets:**

```json
{
  "performance": {
    "query_timeout": 600,
    "sample_size": 5000,
    "batch_size": 50000,
    "parallel_workers": 8,
    "memory_limit": "2GB"
  },
  "rules": [
    {
      "field": "user_id",
      "type": "string",
      "desired_type": "integer",
      "required": true
    }
  ]
}
```

**For development/testing:**

```json
{
  "performance": {
    "query_timeout": 30,
    "sample_size": 100,
    "batch_size": 1000,
    "parallel_workers": 2
  }
}
```

---

## Troubleshooting

This section helps you solve common issues when using ValidateLite, especially with type conversion validation.

### Common Errors

#### Connection Issues

| Error Message | Possible Cause | Solution |
|---------------|----------------|----------|
| `Connection timeout` | Database unreachable | Check host, port, and network connectivity |
| `Authentication failed` | Wrong credentials | Verify username and password |
| `Database not found` | Wrong database name | Check database name in connection string |
| `File not found: data.csv` | Wrong file path | Use absolute path or check current directory |
| `Permission denied` | File access rights | Check file permissions or run with proper rights |

#### Schema and Rule Errors

| Error Message | Possible Cause | Solution |
|---------------|----------------|----------|
| `Invalid JSON schema` | Malformed JSON | Validate JSON syntax with a JSON validator |
| `Unknown rule type: INVALID` | Typo in rule type | Use valid rule types: NOT_NULL, UNIQUE, RANGE, etc. |
| `Missing required field: field` | Schema missing field name | Add "field" property to rule definition |
| `Table 'users' not found` | Wrong table name | Check table name and database connection |

#### Type Conversion Errors

| Error Message | Possible Cause | Solution |
|---------------|----------------|----------|
| `Invalid type syntax: float(10)` | Wrong type definition format | Use correct format: `float(10,2)` |
| `Conflicting conversion: datetime to integer` | Impossible type conversion | Check desired_type setting for reasonableness |
| `Type conversion timeout` | Conversion validation timeout | Increase `conversion_timeout` config or reduce data size |
| `Precision must be greater than scale` | Wrong float precision config | Ensure precision > scale |

### Connection Issues

**Debug connection problems:**

```bash
# Test basic connectivity
vlite check --conn "mysql://user:pass@host:3306/db" --table "information_schema.tables" --rule "not_null(table_name)"

# Verbose connection debugging
vlite check --conn data.csv --table nonexistent --rule "not_null(id)" --verbose
```

**Common connection string fixes:**

```bash
# Wrong: Missing protocol
--conn "user:pass@host:3306/database"
# Right: Include protocol
--conn "mysql://user:pass@host:3306/database"

# Wrong: Incorrect port for PostgreSQL
--conn "postgresql://user:pass@host:3306/database"
# Right: Use PostgreSQL default port
--conn "postgresql://user:pass@host:5432/database"

# Wrong: Relative path issues
--conn "data/file.csv"
# Right: Use absolute path
--conn "/full/path/to/data/file.csv"
```

### Type Conversion Errors

**Debug type conversion issues:**

```bash
# Check what types are detected
vlite schema --conn data.csv --rules schema.json --verbose

# Test conversion with smaller sample
vlite schema --conn data.csv --rules schema.json --sample-size 100
```

**Common type conversion fixes:**

```json
// Wrong: Impossible conversion
{
  "field": "description",
  "type": "string",
  "desired_type": "integer"  // Text cannot become numbers
}

// Right: Reasonable conversion
{
  "field": "description",
  "type": "string(1000)",
  "desired_type": "string(500)"  // Truncate long text
}

// Wrong: Invalid precision/scale
{
  "field": "amount",
  "type": "string",
  "desired_type": "float(2,10)"  // Scale > precision
}

// Right: Valid precision/scale
{
  "field": "amount",
  "type": "string",
  "desired_type": "float(12,2)"  // Precision > scale
}
```

**Handle problematic data:**

```python
# Script to identify problematic records
import json

with open('validation_results.json') as f:
    results = json.load(f)

for result in results['results']:
    if result['rule_type'] == 'DESIRED_TYPE' and result['status'] == 'FAILED':
        print(f"Field: {result['column']}")
        print(f"Conversion: {result['current_type']} → {result['desired_type']}")
        print(f"Failed: {result['failed_count']}/{result['total_count']}")

        if 'sample_data' in result:
            print("Sample problematic values:")
            for sample in result['sample_data'][:5]:
                print(f"  Row {sample['row']}: {sample['value']}")
        print()
```

**Get help:**

```bash
# Show command help
vlite check --help
vlite schema --help

# Show version
vlite --version

# Test with minimal example
vlite check --conn /dev/null --table test --rule "not_null(id)" 2>&1
```

If you're still having issues, the most common problems are:
1. **Connection strings** - Double-check your database connection details
2. **File paths** - Use absolute paths when in doubt
3. **Type definitions** - Make sure your desired_type conversions make sense
4. **JSON syntax** - Validate your schema files with a JSON checker

ValidateLite is designed to give you clear error messages, so read them carefully - they usually point directly to the problem!
