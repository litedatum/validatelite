"""
Rule engine error classification test

Validate the new error classification system.
System-level errors (categorized as EngineError) will halt all execution by raising an exception.
2. Operational Errors (OperationError) ->  An incorrect result is returned, but execution continues with other operations.
3. Resource-level errors (RuleExecutionError) -> Return an error result but continue processing other resources.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy.engine import Engine

from core.engine.rule_engine import RuleEngine, RuleGroup
from shared.enums import (
    ConnectionType,
    RuleAction,
    RuleCategory,
    RuleType,
    SeverityLevel,
)
from shared.exceptions import EngineError, OperationError, RuleExecutionError
from shared.schema.base import RuleTarget, TargetEntity
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.result_schema import ExecutionResultSchema
from shared.schema.rule_schema import RuleSchema
from tests.shared.builders.test_builders import TestDataBuilder


class TestErrorClassification:
    """Testing the misclassification system."""

    def test_classify_system_errors(self) -> None:
        """Testing system-level error categorization."""
        # Test various system-level errors.
        system_errors = [
            EngineError("Connection failed"),
            EngineError("Authentication failed"),
            EngineError("Network error"),
            EngineError("Host not found"),
            EngineError("Database not found"),
        ]

        for error in system_errors:
            assert isinstance(error, EngineError), f"Should be EngineError: {error}"
            assert (
                error.get_impact_level() == "SYSTEM"
            ), f"Should be SYSTEM level: {error}"
            assert error.should_stop_execution(), f"Should stop execution: {error}"

    def test_classify_operation_errors(self) -> None:
        """Testing operational error classification."""
        # Test various operational-level errors.
        operation_errors = [
            OperationError("SQL syntax error"),
            OperationError("Query timeout"),
            OperationError("Invalid parameter"),
            OperationError("Type mismatch"),
            OperationError("Data type conversion error"),
        ]

        for error in operation_errors:
            assert isinstance(
                error, OperationError
            ), f"Should be OperationError: {error}"
            assert (
                error.get_impact_level() == "OPERATION"
            ), f"Should be OPERATION level: {error}"
            assert (
                not error.should_stop_execution()
            ), f"Should not stop execution: {error}"

    def test_classify_resource_errors(self) -> None:
        """Testing resource-level error classification."""
        # Test various resource-level errors.
        resource_errors = [
            RuleExecutionError("Table not found"),
            RuleExecutionError("Column does not exist"),
            RuleExecutionError("Field missing"),
            RuleExecutionError("Constraint violation"),
            RuleExecutionError("Foreign key error"),
        ]

        for error in resource_errors:
            assert isinstance(
                error, RuleExecutionError
            ), f"Should be RuleExecutionError: {error}"
            assert (
                error.get_impact_level() == "RESOURCE"
            ), f"Should be RESOURCE level: {error}"
            assert (
                not error.should_stop_execution()
            ), f"Should not stop execution: {error}"

    def test_exception_hierarchy_classification(self) -> None:
        """Testing the exception inheritance hierarchy."""
        # Create different types of exceptions.
        engine_error = EngineError("System failure")
        operation_error = OperationError("Operation failed")
        resource_error = RuleExecutionError("Resource not found")

        # Test exception type checking.
        assert isinstance(engine_error, EngineError)
        assert isinstance(operation_error, OperationError)
        assert isinstance(resource_error, RuleExecutionError)

        # Base class check tests.
        from shared.exceptions import DataQualityException

        assert isinstance(engine_error, DataQualityException)
        assert isinstance(operation_error, DataQualityException)
        assert isinstance(resource_error, DataQualityException)


class TestRuleGroupErrorHandling:
    """Testing Rule Group error handling."""

    @pytest.mark.asyncio
    async def test_rule_error_during_execution_creates_error_results(self) -> None:
        """
        Tests that a RuleExecutionError from an executor is caught
        and converted into an error result, allowing other groups to proceed.
        This is an integration test for the RuleGroup's error handling logic.
        """
        # 1. Setup
        connection = TestDataBuilder.mysql_connection()
        rule = TestDataBuilder.rule().with_name("rule_1").as_not_null_rule().build()

        # Instantiate RuleGroup *after* the config is patched
        rule_group = RuleGroup(
            table_name="test_table", database="test_db", connection=connection
        )
        rule_group.add_rule(rule)

        # 2. Prepare Mocks for the execution layer
        # This is the deepest point of mocking. We want everything else to be real.
        mock_executor_instance = AsyncMock()
        mock_executor_instance.execute_rules.side_effect = RuleExecutionError(
            "Query timeout error"
        )

        MockExecutorClass = Mock(return_value=mock_executor_instance)

        # 3. Patch the executor registry in the module where it's USED. #
        #    In this case, the return function is CompletenessExecutor.execute_rules
        patch_target = (
            "core.engine.rule_engine.executor_registry.get_executor_for_rule_type"
        )
        with patch(patch_target, return_value=MockExecutorClass) as mock_get_executor:

            # 4. Execute and Assert
            mock_engine = AsyncMock()
            results = await rule_group.execute(mock_engine)

            # Verification
            assert len(results) == 1
            result = results[0]
            assert isinstance(result, ExecutionResultSchema)
            assert result.status == "ERROR"

            # The error message is wrapped by the RuleGroup's error handler
            assert result.error_message is not None
            assert "Rule execution failed: Query timeout error" in result.error_message

            # Verify that the registry was indeed called to get the executor
            mock_get_executor.assert_called_once_with(str(rule.type))


class TestRuleEngineErrorHandling:
    """Testing Rule Engine error handling."""

    @pytest.mark.asyncio
    async def test_engine_connection_failure_raises_exception(self) -> None:
        """Verifies that an exception is thrown when the connection to the engine fails."""
        connection = TestDataBuilder.mysql_connection()
        rules = [TestDataBuilder.rule().with_name("rule_1").as_not_null_rule().build()]

        mock_engine = Mock(spec=Engine)
        # Simulates a connection failure.
        engine = RuleEngine(connection)
        with patch.object(
            engine, "_get_engine", side_effect=EngineError("Connection failed")
        ):

            with pytest.raises(EngineError) as exc_info:
                await engine.execute(rules)

            assert "Connection failed" in str(exc_info.value)
            assert exc_info.value.get_impact_level() == "SYSTEM"

    @pytest.mark.asyncio
    async def test_rule_level_errors_return_error_results(self) -> None:
        """Tests that resource-level errors return an error result."""
        connection = TestDataBuilder.mysql_connection()
        rules = [
            TestDataBuilder.rule()
            .with_name("rule_1")
            .with_target("test_db", "test_table", "nonexistent_column")
            .as_not_null_rule()
            .build(),
            TestDataBuilder.rule()
            .with_name("rule_2")
            .with_target("test_db", "test_table", "test_column")
            .as_not_null_rule()
            .build(),
        ]

        engine = RuleEngine(connection)
        mock_engine = Mock(spec=Engine)
        prevalidate_results = {
            "tables": {"test_db.test_table": True},
            "columns": {
                "test_db.test_table.nonexistent_column": False,
                "test_db.test_table.test_column": True,
            },
        }
        group_result = [
            TestDataBuilder.result()
            .with_rule(rule_id="rule_2")
            .with_status("PASSED")
            .with_counts(0, 100)
            .with_timing(0.2)
            .build()
        ]
        # The simulation engine was successfully created, but a resource-level error occurred during execution.
        with patch.object(engine, "_get_engine", return_value=mock_engine):
            # Simulates rule group execution returning an erroneous result.
            with patch.object(
                engine, "_batch_prevalidate_rules", return_value=prevalidate_results
            ):
                with patch.object(RuleGroup, "execute", return_value=group_result):

                    results = await engine.execute(rules)

                    # Validate results.
                    assert len(results) == 2
                    assert results[0].status == "ERROR"
                    assert (
                        results[1].status == "PASSED"
                    )  # The `ExecutionResultSchema.create_success_result` method uses a status of `PASSED`.
                    assert results[0].error_message is not None
                    assert (
                        "Column" in results[0].error_message
                        and "does not exist" in results[0].error_message
                    )

    @pytest.mark.asyncio
    async def test_mixed_results_handling(self) -> None:
        """Testing mixed results handling."""
        connection = TestDataBuilder.mysql_connection()
        rules = [
            TestDataBuilder.rule()
            .with_name("rule_1")
            .with_target("test_db", "test_table", "test_column1")
            .as_not_null_rule()
            .build(),
            TestDataBuilder.rule()
            .with_name("rule_2")
            .with_target("test_db", "test_table", "nonexistent_column")
            .as_not_null_rule()
            .build(),
            TestDataBuilder.rule()
            .with_name("rule_3")
            .with_target("test_db", "test_table", "test_column2")
            .as_not_null_rule()
            .build(),
        ]

        engine = RuleEngine(connection)
        mock_engine = Mock(spec=Engine)
        prevalidate_results = {  # Simulates the result returned by `prevalidate_rules`.
            "tables": {"test_db.test_table": True},
            "columns": {
                "test_db.test_table.test_column1": True,
                "test_db.test_table.nonexistent_column": False,
                "test_db.test_table.test_column2": True,
            },
        }
        # Because the second rule is invalid, only two of the normal rules can be executed, resulting in only two entries in `group_result`.
        group_result = [
            TestDataBuilder.result()
            .with_rule(rule_id="rule_1")
            .with_status("PASSED")
            .with_counts(0, 100)
            .with_timing(0.2)
            .build(),
            TestDataBuilder.result()
            .with_rule(rule_id="rule_3")
            .with_status("PASSED")
            .with_counts(0, 100)
            .with_timing(0.2)
            .build(),
        ]

        # The simulation engine was successfully created.
        with patch.object(engine, "_get_engine", return_value=mock_engine):
            with patch.object(
                engine, "_batch_prevalidate_rules", return_value=prevalidate_results
            ):
                with patch.object(RuleGroup, "execute", return_value=group_result):

                    results = await engine.execute(rules)

                    # Verify the results.  Rule 2 is prioritized because erroneous results are handled first.
                    assert len(results) == 3
                    assert results[0].status == "ERROR"
                    assert results[0].rule_id == "rule_2"
                    assert results[1].status == "PASSED"
                    assert results[2].status == "PASSED"
                    assert results[0].error_message is not None
                    assert (
                        "Column" in results[0].error_message
                        and "does not exist" in results[0].error_message
                    )


class TestErrorClassificationIntegration:
    """Testing the integration of the misclassification system."""

    @pytest.mark.asyncio
    async def test_end_to_end_error_handling(self) -> None:
        """Testing end-to-end error handling."""
        # Create a test connection.
        connection = TestDataBuilder.mysql_connection()

        # Create test rules.
        rule = (
            TestDataBuilder.rule()
            .with_name("Integration Test Rule")
            .with_target("test_db", "test_table", "test_column")
            .as_not_null_rule()
            .with_severity(SeverityLevel.MEDIUM)
            .build()
        )
        rule.id = "integration_test_rule"

        engine = RuleEngine(connection)
        mock_engine = Mock(spec=Engine)

        # Simulate various error scenarios.
        test_scenarios: List[Dict[str, Any]] = [
            {
                "name": "System Error",
                "error": EngineError("Database connection failed"),
                "should_raise": True,
                "expected_type": EngineError,
            },
            {
                "name": "Operation Error",
                "error": OperationError("SQL syntax error"),
                "should_raise": False,
                "expected_status": "ERROR",
            },
            {
                "name": "Resource Error",
                "error": RuleExecutionError("Table not found"),
                "should_raise": False,
                "expected_status": "ERROR",
            },
        ]

        for scenario in test_scenarios:
            with patch.object(engine, "_get_engine") as mock_get_engine:
                if scenario["should_raise"]:
                    # System-level errors should raise exceptions.
                    mock_get_engine.side_effect = scenario["error"]
                    expected_type = scenario["expected_type"]
                    assert isinstance(expected_type, type)
                    with pytest.raises(expected_type):
                        await engine.execute([rule])
                else:
                    # Operation-level and resource-level errors should return an error result.
                    mock_get_engine.return_value = mock_engine

                    with patch.object(
                        RuleGroup, "execute", side_effect=scenario["error"]
                    ):

                        results = await engine.execute([rule])

                        assert len(results) == 1
                        assert results[0].status == scenario["expected_status"]
                        assert results[0].error_message is not None
                        assert str(scenario["error"]) in results[0].error_message
