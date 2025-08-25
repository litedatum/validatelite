#!/usr/bin/env python3
"""
Debug script for schema command
"""

import json
import subprocess
from pathlib import Path


def test_schema_command():
    # Create a temporary rules file similar to the test
    rules = {
        "rules": [
            {"field": "id", "type": "integer", "required": True},
            {"field": "email", "type": "string"},
            {"field": "age", "type": "integer", "min": 0, "max": 150},
        ],
        "strict_mode": False,
        "case_insensitive": True,
    }

    # Write rules to a temporary file
    rules_file = Path("debug_rules.json")
    with open(rules_file, "w") as f:
        json.dump(rules, f)

    try:
        # Test with a simple file source first
        print("=== Testing with file source ===")
        command = [
            "python",
            "cli_main.py",
            "schema",
            "--conn",
            "test_data/customers.xlsx",
            "--table",
            "customers",
            "--rules",
            str(rules_file),
            "--output",
            "table",
        ]

        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)

        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")

        # Test with database connection
        print("\n=== Testing with database connection ===")
        db_command = [
            "python",
            "cli_main.py",
            "schema",
            "--conn",
            "mysql://root:root123@localhost:3306/data_quality",
            "--table",
            "customers",
            "--rules",
            str(rules_file),
            "--output",
            "table",
        ]

        print(f"Running command: {' '.join(db_command)}")
        db_result = subprocess.run(db_command, capture_output=True, text=True)

        print(f"Return code: {db_result.returncode}")
        print(f"STDOUT: {db_result.stdout}")
        print(f"STDERR: {db_result.stderr}")

    finally:
        # Clean up
        if rules_file.exists():
            rules_file.unlink()


if __name__ == "__main__":
    test_schema_command()
