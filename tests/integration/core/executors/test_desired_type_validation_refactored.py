"""
Refactored integration tests for desired_type validation.

Tests the complete end-to-end desired_type validation pipeline using the Click CLI interface.
Covers Excel files (SQLite backend), MySQL, and PostgreSQL databases.
Uses shared utilities for maintainable and consistent test scenarios.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pytest
from click.testing import CliRunner

from cli.app import cli_app
from tests.integration.core.executors.desired_type_test_utils import (
    TestAssertionHelpers,
    TestDataBuilder,
    TestSetupHelpers,
)

logger = logging.getLogger(__name__)


def _write_tmp_file(tmp_path: Path, name: str, content: str) -> str:
    """Write content to a temporary file and return its path."""
    file_path = tmp_path / name
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


@pytest.mark.integration
class TestDesiredTypeValidationExcelRefactored:
    """Test desired_type validation with Excel files using the CLI interface."""

    def test_float_precision_validation_comprehensive(self, tmp_path: Path) -> None:
        """Test comprehensive float(4,1) precision validation using CLI."""
        runner = CliRunner()

        # Set up test files
        excel_path, schema_path = TestSetupHelpers.setup_temp_files(tmp_path)
        TestDataBuilder.create_multi_table_excel(excel_path)

        # Create multi-table schema definition (CLI format)
        schema_definition = {
            "users": {
                "rules": [
                    { "field": "user_id", "type": "integer", "required": True },
                    { "field": "name", "type": "string", "required": True, "desired_type": "string(10)" },
                    { "field": "age", "type": "integer", "required": True, "desired_type": "integer(2)" },
                    { "field": "email", "type": "string", "required": True }
                ]
            },
            "products": {
                "rules": [
                    { "field": "product_id", "type": "integer", "required": True },
                    { "field": "product_name", "type": "string", "required": True },
                    { "field": "price", "type": "float", "required": True, "desired_type": "float(4,1)", "min": 0.0 },
                    { "field": "category", "type": "string", "required": True }
                ]
            },
            "orders": {
                "rules": [
                    { "field": "order_id", "type": "integer", "required": True },
                    { "field": "user_id", "type": "integer", "required": True },
                    { "field": "total_amount", "type": "float", "required": True, "desired_type": "integer(2)" },
                    { "field": "order_status", "type": "string", "required": True }
                ]
            }
        }
        with open(schema_path, 'w') as f:
            json.dump(schema_definition, f, indent=2)

        # Execute validation using CLI
        result = runner.invoke(
            cli_app,
            ["schema", "--conn", str(excel_path), "--rules", str(schema_path), "--output", "json"]
        )

        # Parse results
        assert result.exit_code == 1, f"Expected validation failures, got exit code {result.exit_code}. Output: {result.output}"
        payload = json.loads(result.output)
        assert payload["status"] == "ok"

        print("Payload = ", payload["fields"])
        # Verify comprehensive validation results
        TestAssertionHelpers.assert_validation_results(
            results=payload["fields"],
            expected_failed_tables=['products', 'orders', 'users'],
            min_total_anomalies=8
        )

    def test_float_precision_boundary_cases(self, tmp_path: Path) -> None:
        """Test boundary conditions for float precision validation using CLI."""
        runner = CliRunner()

        # Create boundary test data
        excel_path = tmp_path / "boundary_test_data.xlsx"
        schema_path = tmp_path / "boundary_schema.json"

        TestDataBuilder.create_boundary_test_data(str(excel_path), "float_precision")

        # Create multi-table schema definition (CLI format)
        schema_definition = {
            "users": {
                "rules": [
                    { "field": "user_id", "type": "integer", "required": True },
                    { "field": "name", "type": "string", "required": True, "desired_type": "string(10)" },
                    { "field": "age", "type": "integer", "required": True, "desired_type": "integer(2)" },
                    { "field": "email", "type": "string", "required": True }
                ]
            },
            "products": {
                "rules": [
                    { "field": "product_id", "type": "integer", "required": True },
                    { "field": "product_name", "type": "string", "required": True },
                    { "field": "price", "type": "float", "required": True, "desired_type": "float(4,1)", "min": 0.0 },
                    { "field": "category", "type": "string", "required": True }
                ]
            },
            "orders": {
                "rules": [
                    { "field": "order_id", "type": "integer", "required": True },
                    { "field": "user_id", "type": "integer", "required": True },
                    { "field": "total_amount", "type": "float", "required": True, "desired_type": "integer(2)" },
                    { "field": "order_status", "type": "string", "required": True }
                ]
            }
        }
        with open(schema_path, 'w') as f:
            json.dump(schema_definition, f, indent=2)

        # Execute validation using CLI
        result = runner.invoke(
            cli_app,
            ["schema", "--conn", str(excel_path), "--rules", str(schema_path), "--output", "json"]
        )

        # Parse results
        assert result.exit_code == 1, f"Expected validation failures for boundary cases. Output: {result.output}"
        payload = json.loads(result.output)
        assert payload["status"] == "ok"

        # Verify boundary cases are handled correctly
        TestAssertionHelpers.assert_validation_results(
            results=payload,
            expected_failed_tables=['boundary_test'],
            min_total_anomalies=3  # Expected boundary violations
        )

    def test_sqlite_custom_functions_directly(self) -> None:
        """Test SQLite custom validation functions directly."""
        # Test float precision function with key validation cases
        float_test_cases = [
            (999.9, 4, 1, True, "Maximum valid float(4,1)"),
            (1000.0, 4, 1, False, "Exceeds precision"),
            (99.99, 4, 1, False, "Exceeds scale"),
            (0.9, 1, 1, True, "Precision equals scale edge case"),
            (1.0, 1, 1, False, "Invalid when precision equals scale"),
        ]

        TestAssertionHelpers.assert_sqlite_function_behavior(
            'validate_float_precision',
            float_test_cases
        )

    def test_precision_equals_scale_edge_case(self, tmp_path: Path) -> None:
        """Test the precision==scale edge case fix using CLI."""
        runner = CliRunner()

        # Create test data specifically for precision==scale case
        excel_path = tmp_path / "precision_scale_test.xlsx"
        schema_path = tmp_path / "precision_scale_schema.json"

        TestDataBuilder.create_boundary_test_data(str(excel_path), "precision_equals_scale")

        # Create multi-table schema definition (CLI format)
        schema_definition = {
            "users": {
                "rules": [
                    { "field": "user_id", "type": "integer", "required": True },
                    { "field": "name", "type": "string", "required": True, "desired_type": "string(10)" },
                    { "field": "age", "type": "integer", "required": True, "desired_type": "integer(2)" },
                    { "field": "email", "type": "string", "required": True }
                ]
            },
            "products": {
                "rules": [
                    { "field": "product_id", "type": "integer", "required": True },
                    { "field": "product_name", "type": "string", "required": True },
                    { "field": "price", "type": "float", "required": True, "desired_type": "float(4,1)", "min": 0.0 },
                    { "field": "category", "type": "string", "required": True }
                ]
            },
            "orders": {
                "rules": [
                    { "field": "order_id", "type": "integer", "required": True },
                    { "field": "user_id", "type": "integer", "required": True },
                    { "field": "total_amount", "type": "float", "required": True, "desired_type": "integer(2)" },
                    { "field": "order_status", "type": "string", "required": True }
                ]
            }
        }
        with open(schema_path, 'w') as f:
            json.dump(schema_definition, f, indent=2)

        # Execute validation using CLI
        result = runner.invoke(
            cli_app,
            ["schema", "--conn", str(excel_path), "--rules", str(schema_path), "--output", "json"]
        )

        # Parse results
        assert result.exit_code == 1, f"Expected some validation failures. Output: {result.output}"
        payload = json.loads(result.output)
        assert payload["status"] == "ok"

        # Should pass for 0.9 with float(1,1), fail for 1.0 with float(1,1)
        TestAssertionHelpers.assert_validation_results(
            results=payload,
            expected_failed_tables=['precision_scale_test'],
            min_total_anomalies=1  # Only 1.0 should fail for float(1,1)
        )

    def test_cross_type_validation_scenarios(self, tmp_path: Path) -> None:
        """Test validation scenarios involving type conversions using CLI."""
        runner = CliRunner()

        # Create test data with cross-type scenarios
        excel_path = tmp_path / "cross_type_test.xlsx"
        schema_path = tmp_path / "cross_type_schema.json"

        TestDataBuilder.create_boundary_test_data(str(excel_path), "cross_type")

        # Create multi-table schema definition (CLI format)
        schema_definition = {
            "users": {
                "rules": [
                    { "field": "user_id", "type": "integer", "required": True },
                    { "field": "name", "type": "string", "required": True, "desired_type": "string(10)" },
                    { "field": "age", "type": "integer", "required": True, "desired_type": "integer(2)" },
                    { "field": "email", "type": "string", "required": True }
                ]
            },
            "products": {
                "rules": [
                    { "field": "product_id", "type": "integer", "required": True },
                    { "field": "product_name", "type": "string", "required": True },
                    { "field": "price", "type": "float", "required": True, "desired_type": "float(4,1)", "min": 0.0 },
                    { "field": "category", "type": "string", "required": True }
                ]
            },
            "orders": {
                "rules": [
                    { "field": "order_id", "type": "integer", "required": True },
                    { "field": "user_id", "type": "integer", "required": True },
                    { "field": "total_amount", "type": "float", "required": True, "desired_type": "integer(2)" },
                    { "field": "order_status", "type": "string", "required": True }
                ]
            }
        }
        with open(schema_path, 'w') as f:
            json.dump(schema_definition, f, indent=2)

        # Execute validation using CLI
        result = runner.invoke(
            cli_app,
            ["schema", "--conn", str(excel_path), "--rules", str(schema_path), "--output", "json"]
        )

        # Parse results
        assert result.exit_code == 1, f"Expected validation failures for cross-type scenarios. Output: {result.output}"
        payload = json.loads(result.output)
        assert payload["status"] == "ok"

        # Should detect validation failures in cross-type columns
        TestAssertionHelpers.assert_validation_results(
            results=payload,
            expected_failed_tables=['cross_type_test'],
            min_total_anomalies=2  # Expected failures
        )


@pytest.mark.integration
@pytest.mark.database
class TestDesiredTypeValidationMySQLRefactored:
    """Test desired_type validation with MySQL database using CLI."""

    def test_mysql_float_precision_validation(
        self, tmp_path: Path, mysql_connection_params: Dict[str, object]
    ) -> None:
        """Test MySQL desired_type validation using CLI."""
        if not mysql_connection_params:
            pytest.skip("MySQL connection parameters not available")

        runner = CliRunner()

        # Set up schema file
        schema_path = tmp_path / "mysql_schema.json"
        schema_definition = TestDataBuilder.create_schema_definition()
        with open(schema_path, 'w') as f:
            json.dump(schema_definition, f, indent=2)

        # Create MySQL connection string
        mysql_url = TestSetupHelpers.get_database_connection_params("mysql")
        if not mysql_url:
            pytest.skip("MySQL connection not available")

        # Execute validation using CLI
        result = runner.invoke(
            cli_app,
            ["schema", "--conn", mysql_url, "--rules", str(schema_path), "--output", "json"]
        )

        # Parse results
        if result.exit_code != 0:
            # This is expected if there are validation failures
            payload = json.loads(result.output)
            assert payload["status"] == "ok"

            TestAssertionHelpers.assert_validation_results(
                results=payload,
                expected_failed_tables=['products'],
                min_total_anomalies=3
            )


@pytest.mark.integration
@pytest.mark.database
class TestDesiredTypeValidationPostgreSQLRefactored:
    """Test desired_type validation with PostgreSQL database using CLI."""

    def test_postgresql_float_precision_validation(
        self, tmp_path: Path, postgres_connection_params: Dict[str, object]
    ) -> None:
        """Test PostgreSQL desired_type validation using CLI."""
        if not postgres_connection_params:
            pytest.skip("PostgreSQL connection parameters not available")

        runner = CliRunner()

        # Set up schema file
        schema_path = tmp_path / "postgres_schema.json"
        schema_definition = TestDataBuilder.create_schema_definition()
        with open(schema_path, 'w') as f:
            json.dump(schema_definition, f, indent=2)

        # Create PostgreSQL connection string
        postgres_url = TestSetupHelpers.get_database_connection_params("postgresql")
        if not postgres_url:
            pytest.skip("PostgreSQL connection not available")

        # Execute validation using CLI
        result = runner.invoke(
            cli_app,
            ["schema", "--conn", postgres_url, "--rules", str(schema_path), "--output", "json"]
        )

        # Parse results
        if result.exit_code != 0:
            # This is expected if there are validation failures
            payload = json.loads(result.output)
            assert payload["status"] == "ok"

            TestAssertionHelpers.assert_validation_results(
                results=payload,
                expected_failed_tables=['products'],
                min_total_anomalies=3
            )


@pytest.mark.integration
class TestDesiredTypeValidationRegressionRefactored:
    """Regression tests for specific bug fixes using CLI."""

    def test_regression_bug_fixes_comprehensive(self, tmp_path: Path) -> None:
        """Test all major bug fixes in the desired_type validation pipeline using CLI."""
        runner = CliRunner()

        # Set up test files specifically designed to trigger the original bugs
        excel_path, schema_path = TestSetupHelpers.setup_temp_files(tmp_path)
        TestDataBuilder.create_multi_table_excel(excel_path)

        # Create multi-table schema definition (CLI format)
        schema_definition = {
            "users": {
                "rules": [
                    { "field": "user_id", "type": "integer", "required": True },
                    { "field": "name", "type": "string", "required": True, "desired_type": "string(10)" },
                    { "field": "age", "type": "integer", "required": True, "desired_type": "integer(2)" },
                    { "field": "email", "type": "string", "required": True }
                ]
            },
            "products": {
                "rules": [
                    { "field": "product_id", "type": "integer", "required": True },
                    { "field": "product_name", "type": "string", "required": True },
                    { "field": "price", "type": "float", "required": True, "desired_type": "float(4,1)", "min": 0.0 },
                    { "field": "category", "type": "string", "required": True }
                ]
            },
            "orders": {
                "rules": [
                    { "field": "order_id", "type": "integer", "required": True },
                    { "field": "user_id", "type": "integer", "required": True },
                    { "field": "total_amount", "type": "float", "required": True, "desired_type": "integer(2)" },
                    { "field": "order_status", "type": "string", "required": True }
                ]
            }
        }
        with open(schema_path, 'w') as f:
            json.dump(schema_definition, f, indent=2)

        # Execute validation using CLI
        result = runner.invoke(
            cli_app,
            ["schema", "--conn", str(excel_path), "--rules", str(schema_path), "--output", "json"]
        )

        # Parse results - should detect all the issues that were previously missed
        assert result.exit_code == 1, f"Expected validation failures for regression test. Output: {result.output}"
        payload = json.loads(result.output)
        assert payload["status"] == "ok"

        # Should detect all the issues that the original bugs would have missed
        TestAssertionHelpers.assert_validation_results(
            results=payload,
            expected_failed_tables=['products', 'orders', 'users'],
            min_total_anomalies=8  # Should find the issues that were previously missed
        )

        logger.info("Regression test passed - all major bug fixes verified")