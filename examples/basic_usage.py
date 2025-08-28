#!/usr/bin/env python3
"""
Basic Usage Example for ValidateLite

This example demonstrates how to use ValidateLite for basic data quality validation.
"""

import json
import tempfile
from typing import Any, Dict, List


# Example 1: Basic CSV validation
def example_csv_validation() -> None:
    """Example of validating a CSV file with basic rules."""

    # Create sample CSV data
    csv_content = """id,name,email,age
1,John Doe,john@example.com,30
2,Jane Smith,jane@example.com,25
3,Bob Johnson,bob@example.com,35
4,Alice Brown,alice@example.com,28
5,Charlie Wilson,charlie@example.com,32"""

    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_file = f.name

    # Define validation rules
    rules = [
        {
            "name": "id_not_null",
            "type": "NOT_NULL",
            "column": "id",
            "description": "ID field must not be null",
        },
        {
            "name": "id_unique",
            "type": "UNIQUE",
            "column": "id",
            "description": "ID field must be unique",
        },
        {
            "name": "email_format",
            "type": "REGEX",
            "column": "email",
            "parameters": {
                "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            },
            "description": "Email must be in valid format",
        },
        {
            "name": "age_range",
            "type": "RANGE",
            "column": "age",
            "parameters": {"min_value": 18, "max_value": 100},
            "description": "Age must be between 18 and 100",
        },
    ]

    # Create temporary rules file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(rules, f, indent=2)
        rules_file = f.name

    print("Example 1: CSV Validation")
    print(f"CSV file: {csv_file}")
    print(f"Rules file: {rules_file}")
    print("Run command:")
    print(
        f"python cli_main.py check --conn {csv_file} --table data --rules {rules_file}"
    )
    print()


# Example 2: Database validation
def example_database_validation() -> None:
    """Example of validating a database table."""

    # Define database connection and rules,
    # replace <your_user> and <your_password> with your actual credentials
    # Suggest to use environment variables to store credentials
    db_connection = (
        "mysql://<your_user>:<your_password>@localhost:3306/testdb.customers"
    )

    rules = [
        {
            "name": "customer_id_primary",
            "type": "NOT_NULL",
            "column": "customer_id",
            "description": "Customer ID must not be null",
        },
        {
            "name": "email_unique",
            "type": "UNIQUE",
            "column": "email",
            "description": "Email addresses must be unique",
        },
        {
            "name": "phone_format",
            "type": "REGEX",
            "column": "phone",
            "parameters": {"pattern": r"^\+?[\d\s\-\(\)]{10,}$"},
            "description": "Phone number must be in valid format",
        },
    ]

    # Create temporary rules file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(rules, f, indent=2)
        rules_file = f.name

    print("Example 2: Database Validation")
    print(f"Database: {db_connection}")
    print(f"Rules file: {rules_file}")
    print("Run command:")
    print(
        f'python cli_main.py check --conn "{db_connection}" --table customers '
        f"--rules {rules_file}"
    )
    print()


# Example 3: Excel file validation
def example_excel_validation() -> None:
    """Example of validating an Excel file."""

    # Define rules for Excel validation
    rules: List[Dict[str, Any]] = [
        {
            "name": "product_id_not_null",
            "type": "NOT_NULL",
            "column": "product_id",
            "description": "Product ID must not be null",
        },
        {
            "name": "price_positive",
            "type": "RANGE",
            "column": "price",
            "parameters": {"min_value": 0.01},
            "description": "Price must be positive",
        },
        {
            "name": "category_valid",
            "type": "ENUM",
            "column": "category",
            "parameters": {
                "allowed_values": ["Electronics", "Clothing", "Books", "Home", "Sports"]
            },
            "description": "Category must be one of the allowed values",
        },
    ]

    print("Example 3: Excel Validation")
    print("Rules for Excel file validation:")
    for rule in rules:
        print(f"  - {rule['name']}: {rule['description']}")
    print("Run command:")
    print(
        "python cli_main.py check --conn products.xlsx --table products "
        "--rules rules.json"
    )
    print()


# Example 4: Custom SQL validation
def example_custom_sql_validation() -> None:
    """Example of using custom SQL for validation."""

    # Define custom SQL rules
    rules = [
        {
            "name": "revenue_consistency",
            "type": "CUSTOM_SQL",
            "parameters": {
                "sql": """
                SELECT COUNT(*) as invalid_count
                FROM sales
                WHERE revenue != (quantity * unit_price)
                """
            },
            "description": "Revenue must equal quantity times unit price",
        },
        {
            "name": "data_freshness",
            "type": "CUSTOM_SQL",
            "parameters": {
                "sql": """
                SELECT COUNT(*) as stale_count
                FROM orders
                WHERE created_at < DATE_SUB(NOW(), INTERVAL 30 DAY)
                """
            },
            "description": "Check for orders older than 30 days",
        },
    ]

    print("Example 4: Custom SQL Validation")
    print("Custom SQL rules:")
    for rule in rules:
        print(f"  - {rule['name']}: {rule['description']}")
    print("Run command:")
    print(
        "python cli_main.py check --conn "
        '"mysql://<your_user>:<your_password>@localhost:3306/testdb.sales" '
        "--rules custom_rules.json"
    )
    print()


if __name__ == "__main__":
    print("ValidateLite Basic Usage Examples")
    print("=" * 50)
    print()

    example_csv_validation()
    example_database_validation()
    example_excel_validation()
    example_custom_sql_validation()

    print("For more detailed examples, see the documentation at:")
    print("https://github.com/litedatum/validatelite#readme")
