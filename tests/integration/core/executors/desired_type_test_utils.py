"""
Shared utilities for desired_type validation integration tests.

This module provides common patterns, data builders, and helper functions
used across multiple desired_type validation test files to improve maintainability
and reduce code duplication.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import pandas as pd
import pytest

# Ensure proper project root path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestDataBuilder:
    """Unified test data builder for all desired_type validation tests."""

    @staticmethod
    def create_multi_table_excel(
        file_path: str, include_validation_issues: bool = True
    ) -> None:
        """
        Create Excel file with multiple tables for comprehensive testing.

        Args:
            file_path: Path where Excel file should be created
            include_validation_issues: Whether to include data that should fail validation
        """
        # Products table - Test float(4,1) validation
        products_data = {
            "product_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "product_name": [
                "Widget A",
                "Widget B",
                "Widget C",
                "Widget D",
                "Widget E",
                "Widget F",
                "Widget G",
                "Widget H",
            ],
            "price": [
                123.4,  # ✓ Valid: 4 digits total, 1 decimal place
                12.3,  # ✓ Valid: 3 digits total, 1 decimal place
                1.2,  # ✓ Valid: 2 digits total, 1 decimal place
                0.5,  # ✓ Valid: 1 digit total, 1 decimal place
                999.99 if include_validation_issues else 999.9,  # ✗/✓ Invalid/Valid
                1234.5 if include_validation_issues else 123.4,  # ✗/✓ Invalid/Valid
                12.34 if include_validation_issues else 12.3,  # ✗/✓ Invalid/Valid
                10.0,  # ✓ Valid: 3 digits total, 1 decimal place
            ],
            "category": ["electronics"] * 8,
        }

        # Orders table - Test cross-type float->integer(2) validation
        orders_data = {
            "order_id": [1, 2, 3, 4, 5, 6],
            "user_id": [101, 102, 103, 104, 105, 106],
            "total_amount": [
                89.0,  # ✓ Valid: can convert to integer(2)
                12.0,  # ✓ Valid: can convert to integer(2)
                5.0,  # ✓ Valid: can convert to integer(2)
                999.99 if include_validation_issues else 99.0,  # ✗/✓ Invalid/Valid
                123.45 if include_validation_issues else 12.0,  # ✗/✓ Invalid/Valid
                1000.0 if include_validation_issues else 10.0,  # ✗/✓ Invalid/Valid
            ],
            "order_status": ["pending"] * 6,
        }

        # Users table - Test integer(2) and string(10) validation
        users_data = {
            "user_id": [101, 102, 103, 104, 105, 106, 107],
            "name": [
                "Alice",  # ✓ Valid: length 5 <= 10
                "Bob",  # ✓ Valid: length 3 <= 10
                "Charlie",  # ✓ Valid: length 7 <= 10
                "David",  # ✓ Valid: length 5 <= 10
                (
                    "VeryLongName" if include_validation_issues else "Eve"
                ),  # ✗/✓ Invalid/Valid
                "X",  # ✓ Valid: length 1 <= 10
                (
                    "TenCharName" if include_validation_issues else "Frank"
                ),  # ✗/✓ Invalid/Valid
            ],
            "age": [
                25,  # ✓ Valid: 2 digits
                30,  # ✓ Valid: 2 digits
                5,  # ✓ Valid: 1 digit
                99,  # ✓ Valid: 2 digits
                123 if include_validation_issues else 23,  # ✗/✓ Invalid/Valid
                8,  # ✓ Valid: 1 digit
                150 if include_validation_issues else 50,  # ✗/✓ Invalid/Valid
            ],
            "email": [
                "alice@test.com",
                "bob@test.com",
                "charlie@test.com",
                "david@test.com",
                "eve@test.com",
                "x@test.com",
                "frank@test.com",
            ],
        }

        # Write to Excel file with multiple sheets
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            pd.DataFrame(products_data).to_excel(
                writer, sheet_name="products", index=False
            )
            pd.DataFrame(orders_data).to_excel(writer, sheet_name="orders", index=False)
            pd.DataFrame(users_data).to_excel(writer, sheet_name="users", index=False)

    @staticmethod
    def create_boundary_test_data(file_path: str, test_type: str) -> None:
        """
        Create Excel file with boundary test cases for specific data types.

        Args:
            file_path: Path where Excel file should be created
            test_type: Type of boundary test ('float', 'integer', 'string', 'null', 'conversion',
                      'float_precision', 'precision_equals_scale', 'cross_type')
        """
        if test_type == "float":
            test_data = {
                "id": list(range(1, 13)),
                "description": [
                    "Exact precision match",
                    "Zero value",
                    "Negative value",
                    "Very small positive",
                    "Very small negative",
                    "Trailing zeros",
                    "Leading zeros",
                    "Maximum valid",
                    "Boundary case - precision",
                    "Boundary case - scale",
                    "Scientific notation",
                    "Edge boundary",
                ],
                "test_value": [
                    999.9,
                    0.0,
                    -99.9,
                    0.1,
                    -0.1,
                    10.0,
                    9.9,
                    999.9,
                    1000.0,
                    99.99,
                    1.23e2,
                    999.95,
                ],
            }
        elif test_type == "integer":
            test_data = {
                "id": list(range(1, 11)),
                "description": [
                    "Single digit",
                    "Two digits max",
                    "Zero",
                    "Negative single",
                    "Negative two digits",
                    "Three digits - boundary",
                    "Large positive",
                    "Large negative",
                    "Edge case 99",
                    "Edge case 100",
                ],
                "test_value": [1, 99, 0, -1, -99, 123, 9999, -123, 99, 100],
            }
        elif test_type == "string":
            test_data = {
                "id": list(range(1, 13)),
                "description": [
                    "Empty string",
                    "Single character",
                    "Exactly 10 chars",
                    "Unicode characters",
                    "Special characters",
                    "Whitespace only",
                    "Leading/trailing spaces",
                    "Exactly 11 chars",
                    "Very long",
                    "Mixed case",
                    "Numbers as string",
                    "Punctuation",
                ],
                "test_value": [
                    "",
                    "A",
                    "1234567890",
                    "café",
                    "!@#$%",
                    "   ",
                    " hello ",
                    "12345678901",
                    "This is a very long string that exceeds limit",
                    "MixedCase",
                    "1234567890",
                    "Hello,World!",
                ],
            }
        elif test_type == "null":
            test_data = {
                "id": [1, 2, 3, 4, 5, 6],
                "float_value": [123.4, None, float("nan"), 0.0, -0.0, ""],
                "int_value": [42, None, 0, -1, "", "NULL"],
                "str_value": ["valid", None, "", "NULL", "null", "   "],
            }
        elif test_type == "conversion":
            test_data = {
                "id": list(range(1, 11)),
                "description": [
                    "Float as integer",
                    "String number",
                    "Boolean as number",
                    "Date as string",
                    "Scientific notation",
                    "Infinity",
                    "Very small number",
                    "Very large number",
                    "String with spaces",
                    "Mixed content",
                ],
                "mixed_value": [
                    42.0,
                    "123",
                    True,
                    "2023-12-01",
                    1.23e-10,
                    float("inf"),
                    1e-100,
                    1e100,
                    " 42 ",
                    "abc123",
                ],
            }
        elif test_type == "float_precision":
            # Specialized float precision boundary test for float(4,1) validation
            test_data = {
                "id": list(range(1, 13)),
                "description": [
                    "Maximum valid float(4,1)",
                    "Minimum positive",
                    "Zero boundary",
                    "Negative maximum",
                    "Scale boundary valid",
                    "Scale boundary invalid",
                    "Precision boundary valid",
                    "Precision boundary invalid",
                    "Combined boundary valid",
                    "Combined boundary invalid",
                    "Scientific notation valid",
                    "Scientific notation invalid",
                ],
                "test_value": [
                    999.9,  # ✓ Valid: exactly float(4,1) maximum
                    0.1,  # ✓ Valid: minimum positive with scale 1
                    0.0,  # ✓ Valid: zero boundary
                    -99.9,  # ✓ Valid: negative maximum for float(4,1)
                    123.4,  # ✓ Valid: within precision and scale
                    123.45,  # ✗ Invalid: exceeds scale (2 decimal places)
                    999.9,  # ✓ Valid: exactly at precision boundary
                    1000.0,  # ✗ Invalid: exceeds precision (5 digits total)
                    99.9,  # ✓ Valid: within both boundaries
                    9999.9,  # ✗ Invalid: exceeds precision (6 digits total)
                    1.2e2,  # ✓ Valid: 120.0 converted to 120.0 (within bounds)
                    1.23e3,  # ✗ Invalid: 1230.0 exceeds precision
                ],
            }
        elif test_type == "precision_equals_scale":
            # Edge case test for when precision equals scale (e.g., float(1,1))
            test_data = {
                "id": list(range(1, 9)),
                "description": [
                    "Valid float(1,1) - 0.9",
                    "Invalid float(1,1) - 1.0",
                    "Valid float(1,1) - 0.1",
                    "Invalid float(1,1) - 1.5",
                    "Valid float(2,2) - 0.99",
                    "Invalid float(2,2) - 1.00",
                    "Edge case zero",
                    "Edge case negative",
                ],
                "test_value": [
                    0.9,  # ✓ Valid for float(1,1): 1 digit total, 1 after decimal
                    1.0,  # ✗ Invalid for float(1,1): 2 digits total (1.0)
                    0.1,  # ✓ Valid for float(1,1): 1 digit total, 1 after decimal
                    1.5,  # ✗ Invalid for float(1,1): 2 digits total
                    0.99,  # ✓ Valid for float(2,2): 2 digits total, 2 after decimal
                    1.00,  # ✗ Invalid for float(2,2): 3 digits total (1.00)
                    0.0,  # ✓ Valid: special case for zero
                    -0.9,  # ✓ Valid for float(1,1): negative with 1 digit total
                ],
            }
        elif test_type == "cross_type":
            # Cross-type validation scenarios (e.g., float to integer conversion)
            test_data = {
                "id": list(range(1, 11)),
                "description": [
                    "Float to int valid",
                    "Float to int invalid - decimal",
                    "Float to int invalid - range",
                    "String to int valid",
                    "String to int invalid",
                    "Boolean to int valid",
                    "Large float to small int",
                    "Negative conversion",
                    "Zero conversion",
                    "Scientific notation conversion",
                ],
                "cross_value": [
                    42.0,  # ✓ Valid: converts cleanly to integer(2)
                    12.5,  # ✗ Invalid: has decimal component
                    123.0,  # ✗ Invalid: too large for integer(2) (3 digits)
                    "89",  # ✓ Valid: string converts to integer(2)
                    "abc",  # ✗ Invalid: non-numeric string
                    True,  # ✓ Valid: boolean True converts to 1
                    999.0,  # ✗ Invalid: too large for integer(2)
                    -12.0,  # ✓ Valid: negative converts to integer(2)
                    0.0,  # ✓ Valid: zero converts cleanly
                    1.2e1,  # ✓ Valid: 12.0 scientific notation converts to 12
                ],
            }
        else:
            raise ValueError(f"Unknown test_type: {test_type}")

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df = pd.DataFrame(test_data)
            # Keep sheet names under 31 characters to avoid Excel compatibility issues
            sheet_name_mapping = {
                "float_precision": "float_precision_tests",
                "precision_equals_scale": "precision_scale_tests",
                "cross_type": "cross_type_tests",
                "float": "float_boundary_tests",
                "integer": "integer_boundary_tests",
                "string": "string_boundary_tests",
                "null": "null_boundary_tests",
                "conversion": "conversion_tests",
            }
            sheet_name = sheet_name_mapping.get(test_type, f"{test_type}_tests")
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    @staticmethod
    def create_rules_definition() -> Dict[str, Any]:
        """
        Create rules definition for multi-table testing.

        Returns:
            Rules definition dictionary with products, orders, and users tables
        """
        return {
            "t_products": {
                "rules": [
                    {"field": "product_id", "type": "integer", "required": True},
                    {"field": "product_name", "type": "string", "required": True},
                    {
                        "field": "price",
                        "type": "float",
                        "required": True,
                        "desired_type": "float(4,1)",
                    },
                    {"field": "category", "type": "string", "required": True},
                ]
            },
            "t_orders": {
                "rules": [
                    {"field": "order_id", "type": "integer", "required": True},
                    {"field": "user_id", "type": "integer", "required": True},
                    {
                        "field": "total_amount",
                        "type": "float",
                        "required": True,
                        "desired_type": "integer(2)",
                    },
                    {"field": "order_status", "type": "string", "required": True},
                ]
            },
            "t_users": {
                "rules": [
                    {"field": "user_id", "type": "integer", "required": True},
                    {
                        "field": "name",
                        "type": "string",
                        "required": True,
                        "desired_type": "string(10)",
                    },
                    {
                        "field": "age",
                        "type": "integer",
                        "required": True,
                        "desired_type": "integer(2)",
                    },
                    {"field": "email", "type": "string", "required": True},
                ]
            },
        }

    @staticmethod
    def create_schema_definition(
        float_precision: Tuple[int, int] = (4, 1),
        integer_digits: int = 2,
        string_length: int = 10,
        include_additional_constraints: bool = False,
    ) -> Dict[str, Any]:
        """
        Create schema definition for testing.

        Args:
            float_precision: Tuple of (precision, scale) for float validation
            integer_digits: Maximum digits for integer validation
            string_length: Maximum length for string validation
            include_additional_constraints: Whether to include additional validation rules

        Returns:
            Schema definition dictionary
        """
        precision, scale = float_precision
        schema = {
            "tables": [
                {
                    "name": "products",
                    "columns": [
                        {
                            "name": "product_id",
                            "type": "integer",
                            "nullable": False,
                            "primary_key": True,
                        },
                        {"name": "product_name", "type": "string", "nullable": False},
                        {
                            "name": "price",
                            "type": "float",
                            "nullable": False,
                            "desired_type": f"float({precision},{scale})",
                            "min": 0.0,
                        },
                        {"name": "category", "type": "string", "nullable": False},
                    ],
                },
                {
                    "name": "orders",
                    "columns": [
                        {
                            "name": "order_id",
                            "type": "integer",
                            "nullable": False,
                            "primary_key": True,
                        },
                        {"name": "user_id", "type": "integer", "nullable": False},
                        {
                            "name": "total_amount",
                            "type": "float",
                            "nullable": False,
                            "desired_type": f"integer({integer_digits})",
                        },
                        {"name": "order_status", "type": "string", "nullable": False},
                    ],
                },
                {
                    "name": "users",
                    "columns": [
                        {
                            "name": "user_id",
                            "type": "integer",
                            "nullable": False,
                            "primary_key": True,
                        },
                        {
                            "name": "name",
                            "type": "string",
                            "nullable": False,
                            "desired_type": f"string({string_length})",
                        },
                        {
                            "name": "age",
                            "type": "integer",
                            "nullable": False,
                            "desired_type": f"integer({integer_digits})",
                        },
                        {"name": "email", "type": "string", "nullable": False},
                    ],
                },
            ]
        }

        if include_additional_constraints:
            # Add regex constraint to email
            cast(Dict[str, Any], schema["tables"][2]["columns"][3])[
                "pattern"
            ] = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

            # Add enum constraint to category
            cast(Dict[str, Any], schema["tables"][0]["columns"][3])["enum"] = [
                "electronics",
                "books",
                "clothing",
                "home",
            ]

            # Add range constraint to age
            cast(Dict[str, Any], schema["tables"][2]["columns"][2])["min"] = 0
            cast(Dict[str, Any], schema["tables"][2]["columns"][2])["max"] = 150

        return schema


class TestAssertionHelpers:
    """Helper methods for common test assertions."""

    @staticmethod
    def assert_validation_results(
        results: List[Dict],
        expected_failed_tables: Optional[List[str]] = None,
        expected_passed_tables: Optional[List[str]] = None,
        min_total_anomalies: int = 0,
    ) -> None:
        """
        Assert validation results meet expectations.

        Args:
            results: List of validation result dictionaries
            expected_failed_tables: Tables that should have validation failures
            expected_passed_tables: Tables that should pass validation
            min_total_anomalies: Minimum total number of anomalies expected
        """
        assert isinstance(results, list), "Results should be a list"
        assert len(results) > 0, "Results should not be empty"

        # Group results by table
        table_results: dict = {}
        total_anomalies = 0

        for result in results:
            table_name = result.get("target_table", result.get("table", "unknown"))
            if table_name not in table_results:
                table_results[table_name] = []
            table_results[table_name].append(result)
            # Count anomalies
            if "dataset_metrics" in result:
                for metric in result["dataset_metrics"]:
                    total_anomalies += metric.get("failed_records", 0)
            elif "failed_records" in result:
                total_anomalies += result["failed_records"]
            elif "checks" in result:
                # Handle CLI JSON fields format - extract failed_records from checks
                for check_name, check_result in result["checks"].items():
                    if (
                        isinstance(check_result, dict)
                        and "failed_records" in check_result
                    ):
                        total_anomalies += check_result.get("failed_records", 0)

        # Check expected failures
        if expected_failed_tables:
            for table in expected_failed_tables:
                assert (
                    table in table_results
                ), f"Expected table {table} to have validation results"
                table_has_failures = any(
                    TestAssertionHelpers._result_has_failures(r)
                    for r in table_results[table]
                )
                assert (
                    table_has_failures
                ), f"Expected table {table} to have validation failures"

        # Check expected passes
        if expected_passed_tables:
            for table in expected_passed_tables:
                if table in table_results:
                    table_has_failures = any(
                        TestAssertionHelpers._result_has_failures(r)
                        for r in table_results[table]
                    )
                    assert (
                        not table_has_failures
                    ), f"Expected table {table} to pass validation"

        # Check minimum anomalies
        if min_total_anomalies > 0:
            assert (
                total_anomalies >= min_total_anomalies
            ), f"Expected at least {min_total_anomalies} anomalies, got {total_anomalies}"

    @staticmethod
    def _result_has_failures(result: Dict) -> bool:
        """Check if a single result indicates validation failures."""
        if "dataset_metrics" in result:
            return any(
                metric.get("failed_records", 0) > 0
                for metric in result["dataset_metrics"]
            )
        elif "checks" in result:
            # Handle both old format (direct failed_records) and new format (status-based)
            for check_name, check_result in result["checks"].items():
                if isinstance(check_result, dict):
                    # Check for failed_records count
                    if check_result.get("failed_records", 0) > 0:
                        return True
                    # Check for FAILED status
                    if check_result.get("status", "").upper() == "FAILED":
                        return True
            return False
        elif "status" in result:
            return result["status"].lower() in ["failed", "error"]
        return False

    @staticmethod
    def assert_sqlite_function_behavior(
        function_name: str, test_cases: List[Tuple[Any, ...]]
    ) -> None:
        """
        Assert SQLite custom function behaves as expected.

        Args:
            function_name: Name of the SQLite function to test
            test_cases: List of (input_args..., expected_result, description) tuples
        """
        try:
            func: Any = None
            if function_name == "validate_float_precision":
                from shared.database.sqlite_functions import (
                    validate_float_precision,
                )

                func = validate_float_precision
            elif function_name == "validate_string_length":
                from shared.database.sqlite_functions import (
                    validate_string_length,
                )

                func = validate_string_length
            elif function_name == "validate_integer_range_by_digits":
                from shared.database.sqlite_functions import (
                    validate_integer_range_by_digits,
                )

                func = validate_integer_range_by_digits
            else:
                pytest.skip(
                    f"SQLite function {function_name} not available for testing"
                )

        except ImportError as e:
            pytest.skip(f"Cannot import SQLite function {function_name}: {e}")

        for test_case in test_cases:
            *args, expected, description = test_case
            try:
                result = func(*args)
                assert result == expected, (
                    f"{function_name} test failed for {description}: "
                    f"args={args}, expected={expected}, got={result}"
                )
            except Exception as e:
                pytest.fail(f"{function_name} test error for {description}: {e}")


class TestSetupHelpers:
    """Helper methods for common test setup patterns."""

    @staticmethod
    def setup_temp_files(
        tmp_path: Path, include_validation_issues: bool = True
    ) -> Tuple[Path, Path]:
        """
        Set up temporary Excel and schema files for testing.

        Args:
            tmp_path: pytest tmp_path fixture
            include_validation_issues: Whether test data should include validation issues

        Returns:
            Tuple of (excel_file_path, schema_file_path)
        """
        excel_file = tmp_path / "test_data.xlsx"
        schema_file = tmp_path / "test_schema.json"

        # Create test data
        TestDataBuilder.create_multi_table_excel(
            str(excel_file), include_validation_issues
        )

        # Create schema definition
        schema = TestDataBuilder.create_schema_definition()
        with open(schema_file, "w") as f:
            json.dump(schema, f, indent=2)

        return excel_file, schema_file

    @staticmethod
    def skip_if_dependencies_unavailable(*module_names: str) -> None:
        """
        Skip test if required dependencies are not available.

        Args:
            module_names: Names of modules that must be importable
        """
        for module_name in module_names:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.skip(f"Required dependency not available: {module_name} - {e}")

    @staticmethod
    def get_database_connection_params(db_type: str) -> Optional[str]:
        """
        Get database connection string from environment or defaults.

        Args:
            db_type: Type of database ('mysql', 'postgresql', 'sqlite')

        Returns:
            Connection string or None if not available
        """
        if db_type == "mysql":
            host = os.getenv("MYSQL_HOST", "localhost")
            port = os.getenv("MYSQL_PORT", "3306")
            user = os.getenv("MYSQL_USER", "test_user")
            password = os.getenv("MYSQL_PASSWORD", "test_password")
            database = os.getenv("MYSQL_DATABASE", "test_database")
            return f"mysql://{user}:{password}@{host}:{port}/{database}"
        elif db_type == "postgresql":
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = os.getenv("POSTGRES_PORT", "5432")
            user = os.getenv("POSTGRES_USER", "test_user")
            password = os.getenv("POSTGRES_PASSWORD", "test_password")
            database = os.getenv("POSTGRES_DATABASE", "test_database")
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        elif db_type == "sqlite":
            return ":memory:"
        else:
            return None


# Export main classes for easy importing
__all__ = ["TestDataBuilder", "TestAssertionHelpers", "TestSetupHelpers"]
