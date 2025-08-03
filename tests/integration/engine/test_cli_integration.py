"""
Command-line interface (CLI) integration tests: These tests verify the performance and behavior of the completeness executor within various CLI usage scenarios.

This test simulates real-world command-line interface (CLI) usage scenarios.
CSV file validation
2. Database Connection Verification
Rule parsing and execution.
4. Format and output the results.
5. Error Handling
"""

import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from core.executors.completeness_executor import CompletenessExecutor
from shared.enums.rule_actions import RuleAction
from shared.enums.rule_categories import RuleCategory
from shared.enums.rule_types import RuleType
from shared.enums.severity_levels import SeverityLevel
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.result_schema import ExecutionResultSchema
from shared.schema.rule_schema import RuleSchema


class TestCLIIntegration:
    """Command-Line Interface (CLI) Integration Test Suite"""

    @pytest.fixture
    def sample_csv_data(self) -> pd.DataFrame:
        """Create CSV data for testing purposes."""
        return pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", None, "Diana", "Eve"],
                "email": [
                    "alice@test.com",
                    "bob@test.com",
                    "charlie@test.com",
                    None,
                    "eve@test.com",
                ],
                "age": [25, 30, 35, 28, 32],
                "description": [
                    "Short",
                    "Medium length text",
                    "Very long description that exceeds normal limits",
                    "OK",
                    "Good",
                ],
            }
        )

    @pytest.fixture
    def temp_csv_file(
        self, sample_csv_data: pd.DataFrame
    ) -> Generator[str, None, None]:
        """Create a temporary CSV file."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        sample_csv_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        yield temp_file.name
        # Cleanup.
        Path(temp_file.name).unlink(missing_ok=True)

    @pytest.fixture
    def temp_sqlite_db(
        self, sample_csv_data: pd.DataFrame
    ) -> Generator[str, None, None]:
        """Create a temporary SQLite database."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_file.close()

        # Write data.
        conn = sqlite3.connect(temp_file.name)
        sample_csv_data.to_sql("users", conn, index=False, if_exists="replace")
        conn.close()

        yield temp_file.name
        # Cleanup.
        Path(temp_file.name).unlink(missing_ok=True)

    @pytest.fixture
    def mock_connection(self, temp_sqlite_db: str) -> ConnectionSchema:
        """Simulates a command-line interface (CLI) connection configuration."""
        connection = Mock()
        connection.connection_id = uuid.uuid4()
        connection.connection_type = "sqlite"
        connection.database = temp_sqlite_db
        connection.host = None
        return connection

    def create_cli_rule(self, rule_type: str, column: str, **params: Any) -> Mock:
        """Create rules that mimic the structure and output of a command-line interface (CLI), simulating the results of CLI parsing."""
        rule = Mock(spec=RuleSchema)
        rule.id = str(uuid.uuid4())
        rule.name = f"cli_{rule_type}_{column}"
        rule.type = RuleType(rule_type)
        rule.category = RuleCategory.COMPLETENESS
        rule.severity = SeverityLevel.HIGH
        rule.action = RuleAction.ALERT
        rule.threshold = 0.0
        rule.is_active = True

        rule.get_target_info.return_value = {
            "database": "main",
            "table": "users",
            "column": column,
        }

        rule.get_rule_config.return_value = params
        rule.get_filter_condition.return_value = None

        return rule

    @pytest.mark.asyncio
    async def test_cli_scenario_csv_not_null_success(
        self, mock_connection: ConnectionSchema
    ) -> None:
        """Test CLI scenario: Successful validation of the NOT NULL rule for a CSV file."""
        # This command executes a lightweight validation process using `validate-lite`. It processes the data within the `users.csv` file and applies a rule ensuring that the `id` column contains no null values.

        executor = CompletenessExecutor(mock_connection)
        rule = self.create_cli_rule("NOT_NULL", "id")

        # Simulates queries against data after it has been migrated from CSV to SQLite.
        mock_query_result = [
            {"failed_count": 0}
        ]  # The "id" column contains no null values.
        mock_total_result = [{"total_count": 5}]

        with patch.object(executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            with patch(
                "shared.database.query_executor.QueryExecutor"
            ) as mock_query_executor_class:
                mock_query_executor = AsyncMock()
                mock_query_executor_class.return_value = mock_query_executor

                mock_query_executor.execute_query.side_effect = [
                    (mock_query_result, None),
                    (mock_total_result, None),
                ]

                result = await executor.execute_rule(rule)

                # Expected results of the Command Line Interface (CLI).
                assert result.status == "PASSED"
                assert result.dataset_metrics[0].total_records == 5
                assert result.dataset_metrics[0].failed_records == 0
                assert result.execution_message is not None
                assert "NOT_NULL check passed" in result.execution_message

                # Verify the generated SQL (which will be displayed to the user by the CLI).
                expected_sql = (
                    "SELECT COUNT(*) AS failed_count FROM users WHERE id IS NULL"
                )
                generated_sql = executor._generate_not_null_sql(rule)
                assert generated_sql == expected_sql

    @pytest.mark.asyncio
    async def test_cli_scenario_csv_not_null_failure(
        self, mock_connection: ConnectionSchema
    ) -> None:
        """Testing the command-line interface (CLI) scenario where validation of the NOT NULL rule fails for a CSV file."""
        # This command executes a lightweight validation process using `validate-lite`. It processes the data within the `users.csv` file, applying a rule that checks if the `name` field is not null (meaning it must contain a value).

        executor = CompletenessExecutor(mock_connection)
        rule = self.create_cli_rule("NOT_NULL", "name")

        # Simulates a scenario where the "name" column has one missing value.
        mock_query_result = [{"failed_count": 1}]
        mock_total_result = [{"total_count": 5}]

        with patch.object(executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            with patch(
                "shared.database.query_executor.QueryExecutor"
            ) as mock_query_executor_class:
                mock_query_executor = AsyncMock()
                mock_query_executor_class.return_value = mock_query_executor

                mock_query_executor.execute_query.side_effect = [
                    (mock_query_result, None),
                    (mock_total_result, None),
                ]

                result = await executor.execute_rule(rule)

                # The command-line interface (CLI) should indicate a failure status when appropriate.
                assert result.status == "FAILED"
                assert result.dataset_metrics[0].total_records == 5
                assert result.dataset_metrics[0].failed_records == 1
                assert result.execution_message is not None
                assert "found 1 null records" in result.execution_message
                assert result.error_message == "Found 1 null records"

    @pytest.mark.asyncio
    async def test_cli_scenario_length_validation(
        self, mock_connection: ConnectionSchema
    ) -> None:
        """Testing command-line interface (CLI) scenarios: validation of the LENGTH rule."""
        # This command executes a lightweight validation process on the data within the `users.csv` file. The validation rule enforces that the `description` field for each record must have a length between 5 and 20 characters, inclusive.

        executor = CompletenessExecutor(mock_connection)
        rule = self.create_cli_rule(
            "LENGTH", "description", min_length=5, max_length=20
        )

        # Simulates a scenario where one record has a length exceeding the allowed limit.
        mock_query_result = [{"failed_count": 1}]
        mock_total_result = [{"total_count": 5}]

        with patch.object(executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            with patch(
                "shared.database.query_executor.QueryExecutor"
            ) as mock_query_executor_class:
                mock_query_executor = AsyncMock()
                mock_query_executor_class.return_value = mock_query_executor

                mock_query_executor.execute_query.side_effect = [
                    (mock_query_result, None),
                    (mock_total_result, None),
                ]

                result = await executor.execute_rule(rule)

                # Command-line interface (CLI) results validation.
                assert result.status == "FAILED"
                assert result.dataset_metrics[0].failed_records == 1
                assert result.execution_message is not None
                assert "found 1 length anomaly records" in result.execution_message

                # SQL generation for validating the LENGTH rule.
                expected_sql = "SELECT COUNT(*) AS failed_count FROM users WHERE ((LENGTH(description) < 5 OR LENGTH(description) > 20) OR description IS NULL)"
                generated_sql = executor._generate_length_sql(rule)
                assert generated_sql == expected_sql

    @pytest.mark.asyncio
    async def test_cli_scenario_multiple_rules_batch(
        self, mock_connection: ConnectionSchema
    ) -> None:
        """Test CLI scenarios: Batch rule validation."""
        # This command executes a lightweight validation process using the `validate-lite` tool. It processes the data within the `users.csv` file, applying two validation rules:  first, ensuring that the `email` field is not null; and second, verifying that the `name` field has a length between 2 and 50 characters (inclusive).

        executor = CompletenessExecutor(mock_connection)

        # Create multiple rules.
        not_null_rule = self.create_cli_rule("NOT_NULL", "email")
        length_rule = self.create_cli_rule(
            "LENGTH", "name", min_length=2, max_length=50
        )

        results = []

        # Execute each rule individually, simulating batch processing from the command-line interface (CLI).
        for rule in [not_null_rule, length_rule]:
            if rule.type == RuleType.NOT_NULL:
                mock_query_result = [
                    {"failed_count": 1}
                ]  # There is one null value in the "email" field.
            else:
                mock_query_result = [
                    {"failed_count": 0}
                ]  # All names are of the required length.

            mock_total_result = [{"total_count": 5}]

            with patch.object(executor, "get_engine") as mock_get_engine:
                mock_engine = AsyncMock()
                mock_get_engine.return_value = mock_engine

                with patch(
                    "shared.database.query_executor.QueryExecutor"
                ) as mock_query_executor_class:
                    mock_query_executor = AsyncMock()
                    mock_query_executor_class.return_value = mock_query_executor

                    mock_query_executor.execute_query.side_effect = [
                        (mock_query_result, None),
                        (mock_total_result, None),
                    ]

                    result = await executor.execute_rule(rule)
                    results.append(result)

        # Command-line interface (CLI) batch results validation.
        assert len(results) == 2

        # The NOT NULL constraint failed.
        assert results[0].status == "FAILED"
        assert results[0].dataset_metrics[0].failed_records == 1

        # The LENGTH rule check passed successfully.
        assert results[1].status == "PASSED"
        assert results[1].dataset_metrics[0].failed_records == 0

    @pytest.mark.asyncio
    async def test_cli_scenario_empty_data_handling(
        self, mock_connection: ConnectionSchema
    ) -> None:
        """Test CLI scenario: Handling empty data files."""
        # Scenario: The user uploaded an empty CSV file.

        executor = CompletenessExecutor(mock_connection)
        rule = self.create_cli_rule("NOT_NULL", "name")

        # Simulates an empty dataset.
        mock_query_result: List[Dict[str, Any]] = []  # The query returned no results.
        mock_total_result = [{"total_count": 0}]

        with patch.object(executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            with patch(
                "shared.database.query_executor.QueryExecutor"
            ) as mock_query_executor_class:
                mock_query_executor = AsyncMock()
                mock_query_executor_class.return_value = mock_query_executor

                mock_query_executor.execute_query.side_effect = [
                    (mock_query_result, None),
                    (mock_total_result, None),
                ]

                result = await executor.execute_rule(rule)

                # The command-line interface (CLI) should handle empty data gracefully.
                assert (
                    result.status == "PASSED"
                )  # An empty dataset is considered a passing condition (or satisfies the criteria).
                assert result.dataset_metrics[0].total_records == 0
                assert result.dataset_metrics[0].failed_records == 0
                assert result.execution_message is not None
                assert "NOT_NULL check passed" in result.execution_message

    def test_cli_rule_parsing_simulation(self) -> None:
        """Testing the command-line interface (CLI) rule parsing simulation."""
        # Simulates command-line interface (CLI) argument parsing.
        cli_rules = ["not_null(email)", "length(name,3,50)", "not_null(id)"]

        parsed_rules = []
        for rule_str in cli_rules:
            if rule_str.startswith("not_null(") and rule_str.endswith(")"):
                column = rule_str[9:-1]  # Extract column names.
                rule = self.create_cli_rule("NOT_NULL", column)
                parsed_rules.append(rule)
            elif rule_str.startswith("length(") and rule_str.endswith(")"):
                params = rule_str[7:-1].split(",")  # Parse arguments.
                column = params[0]
                min_length = int(params[1]) if len(params) > 1 else None
                max_length = int(params[2]) if len(params) > 2 else None
                rule = self.create_cli_rule(
                    "LENGTH", column, min_length=min_length, max_length=max_length
                )
                parsed_rules.append(rule)

        # Verify the parsing results.
        assert len(parsed_rules) == 3
        assert parsed_rules[0].type == RuleType.NOT_NULL
        assert parsed_rules[1].type == RuleType.LENGTH
        assert parsed_rules[2].type == RuleType.NOT_NULL

        # Validate the LENGTH rule parameter.
        length_config = parsed_rules[1].get_rule_config()
        assert length_config["min_length"] == 3
        assert length_config["max_length"] == 50

    @pytest.mark.asyncio
    async def test_cli_error_reporting(self, mock_connection: ConnectionSchema) -> None:
        """Test the formatting of command-line interface (CLI) error reports."""
        # Scenario: The command-line interface (CLI) needs to provide user-friendly error messages.

        executor = CompletenessExecutor(mock_connection)

        # Testing unsupported rule types.
        unsupported_rule = Mock(spec=RuleSchema)
        unsupported_rule.type = (
            RuleType.UNIQUE
        )  # CompletenessExecutor is not supported.
        unsupported_rule.name = "test_unique_rule"

        from shared.exceptions.exception_system import RuleExecutionError

        with pytest.raises(RuleExecutionError, match="Unsupported rule type"):
            await executor.execute_rule(unsupported_rule)

        # Testing for errors in the LENGTH rule parameter.
        invalid_length_rule = self.create_cli_rule(
            "LENGTH", "name"
        )  # There are no minimum or maximum length restrictions.
        invalid_length_rule.get_rule_config.return_value = {}  # Empty configuration.

        with pytest.raises(
            RuleExecutionError, match="LENGTH rule requires min_length or max_length"
        ):
            executor._generate_length_sql(invalid_length_rule)

    def test_cli_output_formatting_simulation(self) -> None:
        """Testing the formatting of simulated command-line interface (CLI) output."""
        # Simulates the formatting of command-line interface (CLI) output.

        # Successful result.
        success_result = Mock(spec=ExecutionResultSchema)
        success_result.status = "PASSED"
        success_result.rule_id = "not_null_email"
        success_result.execution_message = "NOT_NULL检查通过"
        success_result.dataset_metrics = [Mock(total_records=100, failed_records=0)]
        success_result.execution_time = 0.05

        # Unsuccessful result.
        failure_result = Mock(spec=ExecutionResultSchema)
        failure_result.status = "FAILED"
        failure_result.rule_id = "length_description"
        failure_result.execution_message = "LENGTH检查完成，发现 3 条长度异常记录"
        failure_result.error_message = "发现 3 条长度异常记录"
        failure_result.dataset_metrics = [Mock(total_records=100, failed_records=3)]
        failure_result.execution_time = 0.12

        # Simulates the output format of a command-line interface (CLI).
        def format_result_for_cli(result: ExecutionResultSchema) -> str:
            if result.status == "PASSED":
                return f"✅ {result.rule_id}: {result.execution_message} ({result.execution_time:.3f}s)"
            else:
                return f"❌ {result.rule_id}: {result.error_message} ({result.execution_time:.3f}s)"

        # Verify the output format.
        success_output = format_result_for_cli(success_result)
        failure_output = format_result_for_cli(failure_result)

        assert "✅" in success_output
        assert "NOT_NULL检查通过" in success_output
        assert "0.050s" in success_output

        assert "❌" in failure_output
        assert "发现 3 条长度异常记录" in failure_output
        assert "0.120s" in failure_output


class TestCLIExecutorCompatibility:
    """CLI executor compatibility tests."""

    @pytest.mark.asyncio
    async def test_executor_supports_cli_rule_types(self) -> None:
        """The test runner supports common rule types used in command-line interfaces (CLIs)."""
        mock_connection = Mock()
        mock_connection.connection_type = "sqlite"
        executor = CompletenessExecutor(mock_connection)

        # The most commonly used rule types in the Command Line Interface (CLI).
        cli_common_types = ["NOT_NULL", "LENGTH"]

        for rule_type in cli_common_types:
            assert executor.supports_rule_type(
                rule_type
            ), f"CLI common rule type {rule_type} should be supported"

        # Unsupported types.
        unsupported_types = ["UNIQUE", "RANGE", "ENUM"]
        for rule_type in unsupported_types:
            assert not executor.supports_rule_type(
                rule_type
            ), f"Rule type {rule_type} should not be supported by CompletenessExecutor"

    def test_cli_sql_generation_compatibility(self) -> None:
        """Test the compatibility of the CLI-generated SQL."""
        mock_connection = Mock()
        mock_connection.connection_type = "sqlite"
        executor = CompletenessExecutor(mock_connection)

        # Simulates the rules of command-line interface (CLI) parsing.
        rule = Mock(spec=RuleSchema)
        rule.get_target_info.return_value = {"table": "users", "column": "email"}
        rule.get_filter_condition.return_value = None

        # Test basic SQL generation.
        sql = executor._generate_not_null_sql(rule)

        # SQL features required by the command-line interface (CLI).
        assert "SELECT COUNT(*)" in sql
        assert "failed_count" in sql
        assert "users" in sql
        assert "email IS NULL" in sql
        assert sql.count("SELECT") == 1  # Contains a single SELECT statement.
