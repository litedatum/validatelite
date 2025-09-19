# ValidateLite

[![PyPI version](https://badge.fury.io/py/validatelite.svg)](https://badge.fury.io/py/validatelite)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://img.shields.io/badge/coverage-80%25-green.svg)](https://github.com/litedatum/validatelite)

**ValidateLite: A lightweight, scenario-driven data validation tool for modern data practitioners.**

Whether you're a data scientist cleaning a messy CSV, a data engineer building robust pipelines, or a developer needing a quick check, ValidateLite provides powerful, focused commands for your use case:

*   **`vlite check`**: For quick, ad-hoc data checks. Need to verify if a column is unique or not null *right now*? The `check` command gets you an answer in seconds, zero config required.

*   **`vlite schema`**: For robust, repeatable, and automated validation. Define your data's contract in a JSON schema and let ValidateLite verify everything from data types and ranges to complex type-conversion feasibility.

---

## Who is it for?

### For the Data Scientist: Preparing Data for Analysis

You have a messy dataset (`legacy_data.csv`) where everything is a `string`. Before you can build a model, you need to clean it up and convert columns to their proper types (`integer`, `float`, `date`). How much work will it be?

Instead of writing complex cleaning scripts first, use `vlite schema` to **assess the feasibility of the cleanup**.

**1. Define Your Target Schema (`rules.json`)**

Create a schema file that describes the *current* type and the *desired* type.

```json
{
  "legacy_users": {
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
      },
      {
        "field": "bio",
        "type": "string",
        "desired_type": "string(500)",
        "required": false
      }
    ]
  }
}
```

**2. Run the Validation**

```bash
vlite schema --conn legacy_data.csv --rules rules.json
```

ValidateLite will generate a report telling you exactly what can and cannot be converted, saving you hours of guesswork.

```
FIELD VALIDATION RESULTS
========================

Field: user_id
  ✓ Field exists (string)
  ✓ Not Null constraint
  ✗ Type Conversion Validation (string → integer): 15 incompatible records found

Field: salary
  ✓ Field exists (string)
  ✗ Type Conversion Validation (string → float(10,2)): 8 incompatible records found

Field: bio
  ✓ Field exists (string)
  ✓ Length Constraint Validation (string → string(500)): PASSED
```

### For the Data Engineer: Ensuring Data Integrity in CI/CD

You need to prevent breaking schema changes and bad data from ever reaching production. Embed ValidateLite into your CI/CD pipeline to act as a quality gate.

**Example Workflow (`.github/workflows/ci.yml`)**

This workflow automatically validates the database schema on every pull request.

```yaml
jobs:
  validate-db-schema:
    name: Validate Database Schema
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install ValidateLite
        run: pip install validatelite

      - name: Run Schema Validation
        run: |
          vlite schema --conn "mysql://${{ secrets.DB_USER }}:${{ secrets.DB_PASS }}@${{ secrets.DB_HOST }}/sales" \
                       --rules ./schemas/customers_schema.json \
                       --fail-on-error
```
This same approach can be used to monitor data quality at every stage of your ETL/ELT pipelines, preventing "garbage in, garbage out."

---

## Quick Start: Ad-Hoc Checks with `check`

For temporary, one-off validation needs, the `check` command is your best friend. You can run multiple rules on any supported data source (files or databases) directly from the command line.

**1. Install (if you haven't already):**
```bash
pip install validatelite
```

**2. Run a check:**

```bash
# Check for nulls and uniqueness in a CSV file
vlite check --conn "customers.csv" --table customers \
  --rule "not_null(id)" \
  --rule "unique(email)"

# Check value ranges and formats in a database table
vlite check --conn "mysql://user:pass@host/db" --table customers \
  --rule "range(age, 18, 99)" \
  --rule "enum(status, 'active', 'inactive')"
```

---

## Learn More

- **[Usage Guide (docs/usage.md)](docs/usage.md)**: Learn about all commands, data sources, rule types, and advanced features like the **Desired Type** system.
- **[Configuration Reference (docs/CONFIG_REFERENCE.md)](docs/CONFIG_REFERENCE.md)**: See how to configure the tool via `toml` files.
- **[Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)**: We welcome contributions!

---

## 📝 Development Blog

Follow the journey of building ValidateLite through our development blog posts:

- **[DevLog #1: Building a Zero-Config Data Validation Tool](https://blog.litedatum.com/posts/Devlog01-data-validation-tool/)**
- **[DevLog #2: Why I Scrapped My Half-Built Data Validation Platform](https://blog.litedatum.com/posts/Devlog02-Rethinking-My-Data-Validation-Tool/)
- **[Rule-Driven Schema Validation: A Lightweight Solution](https://blog.litedatum.com/posts/Rule-Driven-Schema-Validation/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE)
