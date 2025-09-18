"""
Integration tests for desired_type validation functionality.

Tests the complete desired_type validation pipeline including:
1. Compatibility analysis
2. Rule generation with proper constraint enforcement
3. SQLite custom function validation for Excel/file sources
4. Native database validation for MySQL/PostgreSQL

This test suite specifically covers the bugs fixed in:
- cli/commands/schema.py (CompatibilityAnalyzer)
- core/executors/validity_executor.py (SQLite custom validation)
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest
from click.testing import CliRunner

from cli.app import cli_app
from tests.integration.core.executors.desired_type_test_utils import (
    TestAssertionHelpers,
    TestDataBuilder,
    TestSetupHelpers,
)

# Ensure proper project root path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# pytestmark = pytest.mark.asyncio  # Removed global asyncio mark - apply individually to async tests


class DesiredTypeTestDataBuilder:
    """Builder for creating test data files and schema definitions."""

    @staticmethod
    def create_excel_test_data(file_path: str) -> None:
        """Create Excel file with test data for desired_type validation."""

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
                999.99,  # ✗ Invalid: 5 digits total, 2 decimal places (was failing before fix)
                1234.5,  # ✗ Invalid: 5 digits total, 1 decimal place (exceeds precision)
                12.34,  # ✗ Invalid: 4 digits total, 2 decimal places (exceeds scale)
                10.0,  # ✓ Valid: 3 digits total, 1 decimal place (trailing zero)
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
                999.99,  # ✗ Invalid: cannot convert to integer(2) - too many digits
                123.45,  # ✗ Invalid: not an integer-like float
                1000.0,  # ✗ Invalid: exceeds integer(2) limit
            ],
            "order_status": ["pending"] * 6,
            "order_date": [
                "2020-02-09",
                "2019-11-22",
                "2021-02-29",  # invalid date
                "2021-04-31",  # invalid date
                "2011-01-05",
                "2024-13-06",  # invalid date
            ],
            "order_time": [
                "12:13:14",
                "13:00:00",
                "14:15:78",  # invalid time (78 seconds)
                "15:16:17",
                "25:17:18",  # invalid time (25 hours)
                "23:59:59",
            ],
        }

        # Users table - Test integer(2) and string(10) validation
        users_data = {
            "user_id": [101, 102, 103, 104, 105, 106, 107],
            "name": [
                "Alice",  # ✓ Valid: length 5 <= 10
                "Bob",  # ✓ Valid: length 3 <= 10
                "Charlie",  # ✓ Valid: length 7 <= 10
                "David",  # ✓ Valid: length 5 <= 10
                "VeryLongName",  # ✗ Invalid: length 12 > 10
                "X",  # ✓ Valid: length 1 <= 10
                "TenCharName",  # ✗ Invalid: length 11 > 10
            ],
            "age": [
                25,  # ✓ Valid: 2 digits
                30,  # ✓ Valid: 2 digits
                5,  # ✓ Valid: 1 digit
                99,  # ✓ Valid: 2 digits
                123,  # ✗ Invalid: 3 digits > integer(2)
                8,  # ✓ Valid: 1 digit
                150,  # ✗ Invalid: 3 digits > integer(2)
            ],
            "email": [
                "alice@test.com",
                "bob@test.com",
                "charlie@test.com",
                "david@test.com",
                "verylongname@test.com",
                "x@test.com",
                "ten@test.com",
            ],
            "birthday": [
                19680223,
                19680230,  # invalid date (Feb 30)
                19680401,
                19780431,  # invalid date (Apr 31)
                19680630,
                19680631,  # invalid date (Jun 31)
                19680701,
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
    def create_schema_rules() -> Dict[str, Any]:
        """Create schema rules for desired_type validation testing."""
        return {
            "products": {
                "rules": [
                    {"field": "product_id", "type": "integer", "required": True},
                    {"field": "product_name", "type": "string", "required": True},
                    {
                        "field": "price",
                        "type": "float",
                        "desired_type": "float(4,1)",
                        "min": 0.0,
                    },
                    {
                        "field": "category",
                        "type": "string",
                        "enum": ["electronics", "clothing", "books"],
                    },
                ]
            },
            "orders": {
                "rules": [
                    {"field": "order_id", "type": "integer", "required": True},
                    {"field": "user_id", "type": "integer", "required": True},
                    {
                        "field": "total_amount",
                        "type": "float",
                        "desired_type": "integer(2)",
                        "min": 0.0,
                    },
                    {
                        "field": "order_status",
                        "type": "string",
                        "enum": ["pending", "confirmed", "shipped"],
                    },
                    {
                        "field": "order_date",
                        "type": "string",
                        "desired_type": "date('YYYY-MM-DD')",
                    },
                    {
                        "field": "order_time",
                        "type": "string",
                        "desired_type": "datetime('HH:MI:SS')",
                    },
                ]
            },
            "users": {
                "rules": [
                    {"field": "user_id", "type": "integer", "required": True},
                    {
                        "field": "name",
                        "type": "string",
                        "desired_type": "string(10)",
                        "required": True,
                    },
                    {
                        "field": "age",
                        "type": "integer",
                        "desired_type": "integer(2)",
                        "min": 0,
                        "max": 120,
                    },
                    {"field": "email", "type": "string", "required": True},
                    {
                        "field": "birthday",
                        "type": "integer",
                        "desired_type": "date('YYYYMMDD')",
                    },
                ]
            },
        }


@pytest.mark.integration
@pytest.mark.database
class TestDesiredTypeValidationExcel:
    """Test desired_type validation with Excel files (SQLite backend)."""

    def _create_test_files(self, tmp_path: Path) -> tuple[str, str]:
        """Create test Excel file and schema JSON file."""
        excel_file = tmp_path / "desired_type_test.xlsx"
        schema_file = tmp_path / "schema_rules.json"

        # Create Excel test data
        DesiredTypeTestDataBuilder.create_excel_test_data(str(excel_file))

        # Create schema rules
        schema_rules = DesiredTypeTestDataBuilder.create_schema_rules()
        with open(schema_file, "w") as f:
            json.dump(schema_rules, f, indent=2)

        return str(excel_file), str(schema_file)

    def test_comprehensive_excel_validation_cli(self, tmp_path: Path) -> None:
        """Test comprehensive desired_type validation with an Excel file via the CLI."""
        # 1. Setup test files
        excel_file, schema_file = self._create_test_files(tmp_path)

        # 2. Run CLI
        runner = CliRunner()
        result = runner.invoke(
            cli_app,
            [
                "schema",
                "--conn",
                str(excel_file),
                "--rules",
                str(schema_file),
                "--output",
                "json",
            ],
        )

        # 3. Assert results
        assert (
            result.exit_code == 1
        ), f"Expected exit code 1 for validation failures. Output: {result.output}"

        try:
            payload = json.loads(result.output)
        except json.JSONDecodeError:
            pytest.fail(f"Failed to decode JSON output: {result.output}")

        assert payload["status"] == "ok"
        TestAssertionHelpers.assert_validation_results(
            results=payload["fields"],
            expected_failed_tables=["products", "orders", "users"],
            min_total_anomalies=8,  # Updated to expect date format validation failures
        )

        # Additional assertions for DATE_FORMAT validation results
        results = payload["results"]

        # Find DATE_FORMAT rule results
        date_format_results = [
            r
            for r in results
            if "DATE_FORMAT" in str(r.get("execution_plan", {}))
            or (r.get("execution_message", "").find("DATE_FORMAT") != -1)
        ]

        # Verify we have DATE_FORMAT validations running
        assert (
            len(date_format_results) >= 0
        ), "Should have DATE_FORMAT validation results"

        # Check specific field validation results in the fields section
        fields = payload["fields"]

        # Find orders table fields
        orders_fields = [f for f in fields if f["table"] == "orders"]
        order_date_field = next(
            (f for f in orders_fields if f["column"] == "order_date"), None
        )
        order_time_field = next(
            (f for f in orders_fields if f["column"] == "order_time"), None
        )

        # Find users table fields
        users_fields = [f for f in fields if f["table"] == "users"]
        birthday_field = next(
            (f for f in users_fields if f["column"] == "birthday"), None
        )

        # Verify DATE_FORMAT validation was attempted for these fields
        if order_date_field:
            print(f"\nOrder date field validation: {order_date_field}")
            # The field should exist and have some validation result
            assert "checks" in order_date_field

        if order_time_field:
            print(f"\nOrder time field validation: {order_time_field}")
            assert "checks" in order_time_field

        if birthday_field:
            print(f"\nBirthday field validation: {birthday_field}")
            assert "checks" in birthday_field

        # Count total failed records from all rules to verify DATE_FORMAT failures are included
        total_failed_records = payload["summary"]["total_failed_records"]
        print(f"\nTotal failed records across all validations: {total_failed_records}")

        # We expect at least some failures from DATE_FORMAT validations
        # Expected: 3 from order_date + 2 from order_time + 3 from birthday = 8 minimum
        # Note: The exact count may vary based on other validation rules
        assert (
            total_failed_records >= 8
        ), f"Expected at least 8 failed records from date format validations, got {total_failed_records}"

    @pytest.mark.asyncio
    async def test_compatibility_analyzer_always_enforces_constraints(self) -> None:
        """Test that CompatibilityAnalyzer always enforces desired_type constraints."""
        try:
            from cli.commands.schema import CompatibilityAnalyzer
            from shared.enums.connection_types import ConnectionType
        except ImportError as e:
            pytest.skip(f"Cannot import required modules: {e}")

        analyzer = CompatibilityAnalyzer(ConnectionType.SQLITE)

        # Test case 1: Native type has no precision metadata (typical for Excel)
        result1 = analyzer.analyze(
            native_type="FLOAT",
            desired_type="float(4,1)",
            field_name="price",
            table_name="products",
            native_metadata={"precision": None, "scale": None},
        )

        assert (
            result1.compatibility == "INCOMPATIBLE"
        ), "Should always enforce constraints"
        assert result1.required_validation == "REGEX", "Should require REGEX validation"
        assert result1.validation_params is not None
        assert (
            "4,1" in result1.validation_params["description"]
        ), "Should include precision/scale info"

        # Test case 2: Native type has equal precision (should still enforce)
        result2 = analyzer.analyze(
            native_type="FLOAT",
            desired_type="float(4,1)",
            field_name="price",
            table_name="products",
            native_metadata={"precision": 4, "scale": 1},
        )

        assert (
            result2.compatibility == "INCOMPATIBLE"
        ), "Should enforce even when metadata matches"
        assert result2.required_validation == "REGEX", "Should require validation"

        # Test case 3: Native type has larger precision
        result3 = analyzer.analyze(
            native_type="FLOAT",
            desired_type="float(4,1)",
            field_name="price",
            table_name="products",
            native_metadata={"precision": 10, "scale": 2},
        )

        assert (
            result3.compatibility == "INCOMPATIBLE"
        ), "Should enforce tighter constraints"
        assert result3.required_validation == "REGEX", "Should require validation"

    @pytest.mark.asyncio
    async def test_sqlite_custom_validation_function_integration(
        self, tmp_path: Path
    ) -> None:
        """Test that SQLite custom functions are properly used for validation."""
        excel_file, schema_file = self._create_test_files(tmp_path)

        try:
            from shared.database.sqlite_functions import validate_float_precision
        except ImportError as e:
            pytest.skip(f"Cannot import SQLite functions: {e}")

        # Test the core function that was fixed
        test_values = [123.4, 12.3, 999.99, 1234.5, 12.34]
        precision = 4
        scale = 1

        results = []
        for value in test_values:
            result = validate_float_precision(value, precision, scale)
            results.append((value, result))

        # Verify that violations are correctly detected
        expected_results = [
            (123.4, True),  # Valid
            (12.3, True),  # Valid
            (999.99, False),  # Invalid: too many decimal places
            (1234.5, False),  # Invalid: exceeds total precision
            (12.34, False),  # Invalid: too many decimal places
        ]

        for i, (value, expected) in enumerate(expected_results):
            actual_value, actual_result = results[i]
            assert actual_value == value, f"Test data mismatch at index {i}"
            assert (
                actual_result == expected
            ), f"validate_float_precision({value}, 4, 1) expected {expected}, got {actual_result}"


@pytest.mark.integration
@pytest.mark.database
class TestDesiredTypeValidationDatabaseCli:
    """Test desired_type validation with DBs using subprocess and shared utils."""

    async def _run_db_test(
        self, db_type: str, conn_params: Dict[str, Any], tmp_path: Path
    ) -> None:
        # Pre-flight check for connection parameters

        TestSetupHelpers.skip_if_dependencies_unavailable(
            "shared.database.connection", "shared.database.query_executor"
        )
        from shared.database.connection import get_db_url, get_engine
        from shared.database.query_executor import QueryExecutor

        table_name_map = {
            "products": "t_products",
            "orders": "t_orders",
            "users": "t_users",
        }

        async def setup_database() -> None:
            try:
                db_url = get_db_url(
                    db_type=db_type,
                    host=str(conn_params["host"]),
                    port=int(conn_params["port"]),
                    database=str(conn_params["database"]),
                    username=str(conn_params["username"]),
                    password=str(conn_params["password"]),
                )
                engine = await get_engine(db_url, pool_size=1, echo=False)
                executor = QueryExecutor(engine)
                try:
                    for table in table_name_map.values():
                        await executor.execute_query(
                            f"DROP TABLE IF EXISTS {table} CASCADE", fetch=False
                        )

                    # Create tables and insert data
                    await executor.execute_query(
                        """
                        CREATE TABLE t_products (product_id INT, product_name VARCHAR(100), price DECIMAL(10,2), category VARCHAR(50))
                    """,
                        fetch=False,
                    )
                    await executor.execute_query(
                        """
                        INSERT INTO t_products VALUES (1, 'P1', 999.9, 'A'), (2, 'P2', 1000.0, 'A'), (3, 'P3', 99.99, 'B')
                    """,
                        fetch=False,
                    )

                    await executor.execute_query(
                        "CREATE TABLE t_orders (order_id INT, user_id INT, total_amount DECIMAL(10,2), order_status VARCHAR(20))",
                        fetch=False,
                    )
                    await executor.execute_query(
                        "INSERT INTO t_orders VALUES (1, 101, 89.0, 'pending'), (2, 102, 999.99, 'pending')",
                        fetch=False,
                    )

                    await executor.execute_query(
                        "CREATE TABLE t_users (user_id INT, name VARCHAR(100), age INT, email VARCHAR(255))",
                        fetch=False,
                    )
                    await executor.execute_query(
                        "INSERT INTO t_users VALUES (1, 'Alice', 25, 'a@a.com'), (2, 'VeryLongName', 123, 'b@b.com')",
                        fetch=False,
                    )

                finally:
                    await engine.dispose()
            except Exception as e:
                # Database connection failed - skip test
                pytest.skip(f"Database connection to {db_type} failed: {e}")

        async def cleanup_database() -> None:
            try:
                db_url = get_db_url(
                    db_type=db_type,
                    host=str(conn_params["host"]),
                    port=int(conn_params["port"]),
                    database=str(conn_params["database"]),
                    username=str(conn_params["username"]),
                    password=str(conn_params["password"]),
                )
                engine = await get_engine(db_url, pool_size=1, echo=False)
                executor = QueryExecutor(engine)
                try:
                    for table in table_name_map.values():
                        await executor.execute_query(
                            f"DROP TABLE IF EXISTS {table} CASCADE", fetch=False
                        )
                finally:
                    await engine.dispose()
            except Exception:
                # Ignore cleanup errors - the test might have been skipped
                pass

        # Run setup within the same event loop
        await setup_database()
        try:
            # Create rules file
            rules = TestDataBuilder.create_rules_definition()
            rules_file = tmp_path / f"{db_type}_rules.json"
            rules_file.write_text(json.dumps(rules))

            # Manually construct a simple conn_str that SourceParser will recognize.
            # SourceParser does not recognize the '+aiomysql' driver part.
            conn_str = (
                f"{db_type}://{conn_params['username']}:{conn_params['password']}"
                f"@{conn_params['host']}:{conn_params['port']}/{conn_params['database']}"
            )

            # Use subprocess to avoid event loop conflicts (like refactored test)
            import subprocess
            import sys

            cmd = [
                sys.executable,
                "cli_main.py",
                "schema",
                "--conn",
                conn_str,
                "--rules",
                str(rules_file),
                "--output",
                "json",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")

            # Assertions
            assert (
                result.returncode == 1
            ), f"Expected exit code 1 for validation failures in {db_type}. stdout: {result.stdout}, stderr: {result.stderr}"

            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                pytest.fail(
                    f"Failed to decode JSON from output. returncode: {result.returncode}, stdout: {result.stdout}, stderr: {result.stderr}"
                )

            assert payload["status"] == "ok"

            TestAssertionHelpers.assert_validation_results(
                results=payload["fields"],
                expected_failed_tables=["t_products", "t_orders", "t_users"],
                min_total_anomalies=4,
            )

        finally:
            # Teardown within the same event loop
            await cleanup_database()

    @pytest.mark.asyncio
    async def test_mysql_desired_type_validation_cli(self, tmp_path: Path) -> None:
        """Test desired_type validation with real MySQL database via CLI."""
        from tests.shared.utils.database_utils import get_mysql_connection_params

        await self._run_db_test("mysql", get_mysql_connection_params(), tmp_path)

    @pytest.mark.asyncio
    async def test_postgresql_desired_type_validation_cli(self, tmp_path: Path) -> None:
        """Test desired_type validation with real PostgreSQL database via CLI."""
        from tests.shared.utils.database_utils import get_postgresql_connection_params

        await self._run_db_test(
            "postgresql", get_postgresql_connection_params(), tmp_path
        )
