"""
Executor error handling test module

Tests the error handling functionality in various executors:
1. BaseExecutor error handling
2. CompletenessExecutor error handling
3. ValidityExecutor error handling
4. UniquenessExecutor error handling
"""

import time
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from core.executors.base_executor import BaseExecutor
from core.executors.completeness_executor import CompletenessExecutor
from core.executors.uniqueness_executor import UniquenessExecutor
from core.executors.validity_executor import ValidityExecutor
from shared.enums.connection_types import ConnectionType
from shared.enums.rule_actions import RuleAction
from shared.enums.rule_categories import RuleCategory
from shared.enums.rule_types import RuleType
from shared.enums.severity_levels import SeverityLevel
from shared.exceptions import EngineError, OperationError, RuleExecutionError
from shared.schema.base import RuleTarget, TargetEntity
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.rule_schema import RuleSchema


class TestBaseExecutorErrorHandling:
    """Test BaseExecutor error handling functionality"""

    @pytest.fixture
    def connection_schema(self) -> ConnectionSchema:
        """Create a test connection schema"""
        return ConnectionSchema(
            name="test_connection",
            connection_type=ConnectionType.SQLITE,
            file_path=":memory:",
        )

    @pytest.fixture
    def executor(self, connection_schema: ConnectionSchema) -> CompletenessExecutor:
        """Create a test executor instance"""
        return CompletenessExecutor(connection_schema, test_mode=True)

    @pytest.fixture
    def rule_schema(self) -> RuleSchema:
        """Create a test rule schema"""
        return RuleSchema(
            id="test_rule_123",
            name="Test Rule",
            description="Test rule description",
            type=RuleType.NOT_NULL,
            target=RuleTarget(
                entities=[
                    TargetEntity(
                        database="test_db", table="test_table", column="test_column"
                    )
                ]
            ),
            category=RuleCategory.COMPLETENESS,
            severity=SeverityLevel.MEDIUM,
            action=RuleAction.LOG,
            parameters={},
        )

    @pytest.mark.asyncio
    async def test_get_engine_connection_failure_raises_engine_error(
        self, executor: CompletenessExecutor
    ) -> None:
        """Test that connection failures raise EngineError"""
        with patch(
            "core.executors.base_executor.get_engine",
            side_effect=EngineError("Connection failed"),
        ):
            with pytest.raises(EngineError) as exc_info:
                await executor.get_engine()

            assert "Connection failed" in str(exc_info.value)
            assert exc_info.value.get_impact_level() == "SYSTEM"

    @pytest.mark.asyncio
    async def test_handle_execution_error_system_error_raises_exception(
        self, executor: CompletenessExecutor, rule_schema: RuleSchema
    ) -> None:
        """Test that system errors are re-raised"""
        start_time = time.time()
        table_name = "test_table"

        # Mock a system error
        system_error = EngineError("Database connection lost")

        with pytest.raises(EngineError):
            await executor._handle_execution_error(
                system_error, rule_schema, start_time, table_name
            )

    @pytest.mark.asyncio
    async def test_handle_execution_error_operation_error_returns_result(
        self, executor: CompletenessExecutor, rule_schema: RuleSchema
    ) -> None:
        """Test that operation errors return error results"""
        start_time = time.time()
        table_name = "test_table"

        # Mock an operation-level error
        operation_error = OperationError("SQL syntax error")

        with patch(
            "shared.schema.result_schema.ExecutionResultSchema.create_error_result"
        ) as mock_create:
            mock_result = Mock()
            mock_create.return_value = mock_result

            result = await executor._handle_execution_error(
                operation_error, rule_schema, start_time, table_name
            )

            assert result == mock_result
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_execution_error_resource_error_returns_result(
        self, executor: CompletenessExecutor, rule_schema: RuleSchema
    ) -> None:
        """Test that resource errors return error results"""
        start_time = time.time()
        table_name = "test_table"

        # Mock a resource-level error
        resource_error = RuleExecutionError("Table does not exist")

        with patch(
            "shared.schema.result_schema.ExecutionResultSchema.create_error_result"
        ) as mock_create:
            mock_result = Mock()
            mock_create.return_value = mock_result

            result = await executor._handle_execution_error(
                resource_error, rule_schema, start_time, table_name
            )

            assert result == mock_result
            mock_create.assert_called_once()


class TestCompletenessExecutorErrorHandling:
    """Test CompletenessExecutor error handling"""

    @pytest.fixture
    def connection_schema(self) -> ConnectionSchema:
        """Create a test connection schema"""
        return ConnectionSchema(
            name="test_connection",
            connection_type=ConnectionType.SQLITE,
            file_path=":memory:",
        )

    @pytest.fixture
    def executor(self, connection_schema: ConnectionSchema) -> CompletenessExecutor:
        """Create a test executor instance"""
        return CompletenessExecutor(connection_schema, test_mode=True)

    @pytest.fixture
    def rule_schema(self) -> RuleSchema:
        """Create a test rule schema"""
        return RuleSchema(
            id="test_rule_123",
            name="Test NOT_NULL Rule",
            description="Test NOT_NULL rule description",
            type=RuleType.NOT_NULL,
            target=RuleTarget(
                entities=[
                    TargetEntity(
                        database="test_db", table="test_table", column="test_column"
                    )
                ]
            ),
            category=RuleCategory.COMPLETENESS,
            severity=SeverityLevel.MEDIUM,
            action=RuleAction.LOG,
            parameters={},
        )

    @pytest.mark.asyncio
    async def test_execute_not_null_rule_connection_error_raises_exception(
        self, executor: CompletenessExecutor, rule_schema: RuleSchema
    ) -> None:
        """Test that connection errors in NOT_NULL rule execution raise exceptions"""
        # Mock get_engine to raise a connection error
        with patch.object(
            executor, "get_engine", side_effect=EngineError("Connection failed")
        ):
            with pytest.raises(EngineError):
                await executor._execute_not_null_rule(rule_schema)

    @pytest.mark.asyncio
    async def test_execute_not_null_rule_table_error_returns_result(
        self, executor: CompletenessExecutor, rule_schema: RuleSchema
    ) -> None:
        """Test that table-level errors return error results"""
        # Mock get_engine to succeed
        mock_engine = Mock()
        with patch.object(executor, "get_engine", return_value=mock_engine):
            # Mock QueryExecutor to raise table error
            with patch(
                "shared.database.query_executor.QueryExecutor"
            ) as mock_executor_class:
                mock_executor = Mock()
                mock_executor_class.return_value = mock_executor
                mock_executor.execute_query.side_effect = RuleExecutionError(
                    "Table 'test_table' doesn't exist"
                )

                # Mock the error handling method
                with patch.object(executor, "_handle_execution_error") as mock_handle:
                    mock_result = Mock()
                    mock_handle.return_value = mock_result

                    result = await executor._execute_not_null_rule(rule_schema)

                    assert result == mock_result
                    mock_handle.assert_called_once()


class TestValidityExecutorErrorHandling:
    """Test ValidityExecutor error handling"""

    @pytest.fixture
    def connection_schema(self) -> ConnectionSchema:
        """Create a test connection schema"""
        return ConnectionSchema(
            name="test_connection",
            connection_type=ConnectionType.SQLITE,
            file_path=":memory:",
        )

    @pytest.fixture
    def executor(self, connection_schema: ConnectionSchema) -> ValidityExecutor:
        """Create a test executor instance"""
        return ValidityExecutor(connection_schema, test_mode=True)

    @pytest.fixture
    def rule_schema(self) -> RuleSchema:
        """Create a test rule schema"""
        return RuleSchema(
            id="test_rule_123",
            name="Test RANGE Rule",
            description="Test RANGE rule description",
            type=RuleType.RANGE,
            target=RuleTarget(
                entities=[
                    TargetEntity(
                        database="test_db", table="test_table", column="test_column"
                    )
                ]
            ),
            category=RuleCategory.VALIDITY,
            severity=SeverityLevel.MEDIUM,
            action=RuleAction.LOG,
            parameters={"min_value": 0, "max_value": 100},
        )

    @pytest.mark.asyncio
    async def test_execute_range_rule_uses_error_handler(
        self,
        executor: ValidityExecutor,
        rule_schema: RuleSchema,
    ) -> None:
        """Test that RANGE rule execution uses error handler"""
        # Mock get_engine to succeed
        mock_engine = Mock()
        with patch.object(executor, "get_engine", return_value=mock_engine):
            # Mock QueryExecutor to raise an error
            with patch(
                "shared.database.query_executor.QueryExecutor"
            ) as mock_executor_class:
                mock_executor = Mock()
                mock_executor_class.return_value = mock_executor
                mock_executor.execute_query.side_effect = OperationError(
                    "SQL syntax error"
                )

                # Mock the error handling method
                with patch.object(executor, "_handle_execution_error") as mock_handle:
                    mock_result = Mock()
                    mock_handle.return_value = mock_result

                    result = await executor._execute_range_rule(rule_schema)

                    assert result == mock_result
                    mock_handle.assert_called_once()


class TestUniquenessExecutorErrorHandling:
    """Test UniquenessExecutor error handling"""

    @pytest.fixture
    def connection_schema(self) -> ConnectionSchema:
        """Create a test connection schema"""
        return ConnectionSchema(
            name="test_connection",
            connection_type=ConnectionType.SQLITE,
            file_path=":memory:",
        )

    @pytest.fixture
    def executor(self, connection_schema: ConnectionSchema) -> UniquenessExecutor:
        """Create a test executor instance"""
        return UniquenessExecutor(connection_schema, test_mode=True)

    @pytest.fixture
    def rule_schema(self) -> RuleSchema:
        """Create a test rule schema"""
        return RuleSchema(
            id="test_rule_123",
            name="Test UNIQUE Rule",
            description="Test UNIQUE rule description",
            type=RuleType.UNIQUE,
            target=RuleTarget(
                entities=[
                    TargetEntity(
                        database="test_db", table="test_table", column="test_column"
                    )
                ]
            ),
            category=RuleCategory.UNIQUENESS,
            severity=SeverityLevel.MEDIUM,
            action=RuleAction.LOG,
            parameters={},
        )

    @pytest.mark.asyncio
    async def test_execute_unique_rule_uses_error_handler(
        self,
        executor: UniquenessExecutor,
        rule_schema: RuleSchema,
    ) -> None:
        """Test that UNIQUE rule execution uses error handler"""
        # Mock get_engine to succeed
        mock_engine = Mock()
        with patch.object(executor, "get_engine", return_value=mock_engine):
            # Mock QueryExecutor to raise an error
            with patch(
                "shared.database.query_executor.QueryExecutor"
            ) as mock_executor_class:
                mock_executor = Mock()
                mock_executor_class.return_value = mock_executor
                mock_executor.execute_query.side_effect = RuleExecutionError(
                    "Column does not exist"
                )

                # Mock the error handling method
                with patch.object(executor, "_handle_execution_error") as mock_handle:
                    mock_result = Mock()
                    mock_handle.return_value = mock_result

                    result = await executor._execute_unique_rule(rule_schema)

                    assert result == mock_result
                    mock_handle.assert_called_once()


class TestExecutorErrorClassification:
    """Test error classification in executors"""

    def test_system_error_classification(self) -> None:
        """Test that system errors are properly classified"""
        error = EngineError("Database connection failed")

        assert isinstance(error, EngineError)
        assert error.get_impact_level() == "SYSTEM"
        assert error.should_stop_execution()

    def test_operation_error_classification(self) -> None:
        """Test that operation errors are properly classified"""
        error = OperationError("SQL syntax error")

        assert isinstance(error, OperationError)
        assert error.get_impact_level() == "OPERATION"
        assert not error.should_stop_execution()

    def test_resource_error_classification(self) -> None:
        """Test that resource errors are properly classified"""
        error = RuleExecutionError("Table not found")

        assert isinstance(error, RuleExecutionError)
        assert error.get_impact_level() == "RESOURCE"
        assert not error.should_stop_execution()

    def test_error_context_information(self) -> None:
        """Test that errors contain proper context information"""
        error = RuleExecutionError(
            message="Column does not exist",
            rule_id="test_rule_123",
            entity_name="test_db.test_table",
            resource_type="column",
        )

        context_info = error.get_context_info()

        assert context_info["message"] == "Column does not exist"
        assert context_info["impact_level"] == "RESOURCE"
        assert context_info["should_stop"] == False
        assert error.context["rule_id"] == "test_rule_123"
        assert error.context["entity_name"] == "test_db.test_table"
        assert error.context["resource_type"] == "column"
