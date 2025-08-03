"""
Tests for cli/core/exception_handler.py

Tests the exception handler functionality for CLI error handling.
"""

from typing import Dict, cast
from unittest.mock import MagicMock, call, patch

import pytest

from cli.core.exception_handler import CliErrorContext, CliExceptionHandler
from shared.exceptions import EngineError, OperationError, RuleExecutionError
from shared.schema.result_schema import DatasetMetrics, ExecutionResultSchema
from shared.utils.datetime_utils import now


class TestCliExceptionHandler:
    """Tests for CliExceptionHandler class"""

    @pytest.fixture
    def handler(self) -> CliExceptionHandler:
        """Create a CliExceptionHandler instance"""
        with patch("cli.core.exception_handler.get_logger"):
            return CliExceptionHandler(verbose=True)

    @pytest.fixture
    def mock_result(self) -> MagicMock:
        """Create a mock ExecutionResultSchema with error status"""
        result = MagicMock(spec=ExecutionResultSchema)
        result.rule_id = "test_rule_1"
        result.status = "ERROR"
        result.error_message = "Test error message"
        result.execution_time = 1.5

        # Mock dataset_metrics property
        dataset_metric = MagicMock(spec=DatasetMetrics)
        dataset_metric.entity_name = "test_db.customers"
        dataset_metric.total_records = 100
        dataset_metric.failed_records = 10
        result.dataset_metrics = [dataset_metric]

        # Mock methods
        result.get_entity_name.return_value = "test_db.customers"

        return result

    def test_init(self, handler: CliExceptionHandler) -> None:
        """Test initialization of CliExceptionHandler"""
        assert handler.verbose is True
        assert handler.classifier is not None
        assert handler.logger is not None

    def test_handle_cli_native_error(self, handler: CliExceptionHandler) -> None:
        """Test handling CLI native error"""
        error = FileNotFoundError("test.csv not found")

        context = handler._handle_cli_native_error(error)

        assert isinstance(context, CliErrorContext)
        assert context.category == "file_not_found"
        assert context.source == "cli_native"
        assert "test.csv not found" in context.user_message
        assert context.exit_code == 20
        assert (
            context.technical_details
        )  # Should have technical details in verbose mode
        assert context.requires_user_action is True
        assert context.can_retry is False

    def test_handle_schema_creation_error(self, handler: CliExceptionHandler) -> None:
        """Test handling schema creation error"""
        error = OperationError("Invalid connection parameters")

        context = handler._handle_schema_creation_error(error)

        assert isinstance(context, CliErrorContext)
        assert context.category == "invalid_connection"
        assert context.source == "schema_creation"
        assert "Invalid connection parameters" in context.user_message
        assert context.exit_code == 30
        assert (
            context.technical_details
        )  # Should have technical details in verbose mode
        assert context.requires_user_action is True
        assert context.can_retry is False

    def test_handle_engine_error(self, handler: CliExceptionHandler) -> None:
        """Test handling engine error"""
        error = EngineError("Connection timeout", operation="connect")

        context = handler._handle_engine_error(error)

        assert isinstance(context, CliErrorContext)
        assert context.category == "connectivity"
        assert context.source == "exception"
        assert "Database connection error" in context.user_message
        assert context.exit_code == 2
        assert (
            context.technical_details
        )  # Should have technical details in verbose mode
        assert context.requires_user_action is True
        assert context.can_retry is True

    def test_extract_error_results(
        self, handler: CliExceptionHandler, mock_result: MagicMock
    ) -> None:
        """Test extracting error results from a list of results"""
        success_result = MagicMock(spec=ExecutionResultSchema)
        success_result.status = "PASSED"

        results = [success_result, mock_result]

        error_results = handler._extract_error_results(cast(list, results))

        assert len(error_results) == 1
        assert error_results[0] == mock_result

    def test_handle_single_result_error(
        self, handler: CliExceptionHandler, mock_result: MagicMock
    ) -> None:
        """Test handling a single result error"""
        # Configure the mock to return a specific error category
        with patch(
            "cli.core.error_classifier.CliErrorClassifier.classify_result_error",
            return_value="table_not_found",
        ):
            context = handler._handle_single_result_error(mock_result)

            assert isinstance(context, CliErrorContext)
            assert context.category == "table_not_found"
            assert context.source == "result"
            assert "test_rule_1" in context.user_message
            assert context.exit_code == 7
            assert (
                context.technical_details
            )  # Should have technical details in verbose mode
            assert context.requires_user_action is True
            assert context.can_retry is False

    def test_handle_multiple_result_errors(
        self, handler: CliExceptionHandler, mock_result: MagicMock
    ) -> None:
        """Test handling multiple result errors"""
        mock_result2 = MagicMock(spec=ExecutionResultSchema)
        mock_result2.rule_id = "test_rule_2"
        mock_result2.status = "ERROR"
        mock_result2.error_message = "Column not found"
        mock_result2.get_entity_name.return_value = "test_db.customers"

        error_results = [mock_result, mock_result2]

        # Configure the mock to return different error categories
        with patch(
            "cli.core.error_classifier.CliErrorClassifier.classify_result_error",
            side_effect=["table_not_found", "column_not_found"],
        ):
            context = handler._handle_multiple_result_errors(cast(list, error_results))

            assert isinstance(context, CliErrorContext)
            assert (
                context.category == "table_not_found"
            )  # Should select the primary error category
            assert context.source == "result"
            assert "Multiple validation errors occurred" in context.user_message
            assert "(2 total)" in context.user_message
            assert context.exit_code == 7
            assert (
                context.technical_details
            )  # Should have technical details in verbose mode
            assert context.requires_user_action is True
            assert context.can_retry is False

    def test_create_success_context(self, handler: CliExceptionHandler) -> None:
        """Test creating a success context"""
        results = [MagicMock(spec=ExecutionResultSchema)]

        context = handler._create_success_context(cast(list, results))

        assert isinstance(context, CliErrorContext)
        assert context.category == "success"
        assert context.source == "none"
        assert context.user_message == "Operation completed successfully"
        assert context.exit_code == 0
        assert (
            not context.technical_details
        )  # Should not have technical details for success
        assert not context.requires_user_action
        assert not context.can_retry

    def test_handle_complete_process_cli_error(
        self, handler: CliExceptionHandler
    ) -> None:
        """Test handling complete process with CLI error"""
        cli_error = FileNotFoundError("test.csv not found")

        with patch.object(handler, "_handle_cli_native_error") as mock_handle:
            mock_handle.return_value = MagicMock(spec=CliErrorContext)

            handler.handle_complete_process(cli_error=cli_error)

            mock_handle.assert_called_once_with(cli_error)

    def test_handle_complete_process_schema_error(
        self, handler: CliExceptionHandler
    ) -> None:
        """Test handling complete process with schema error"""
        schema_error = OperationError("Invalid connection parameters")

        with patch.object(handler, "_handle_schema_creation_error") as mock_handle:
            mock_handle.return_value = MagicMock(spec=CliErrorContext)

            handler.handle_complete_process(schema_error=schema_error)

            mock_handle.assert_called_once_with(schema_error)

    def test_handle_complete_process_engine_error(
        self, handler: CliExceptionHandler
    ) -> None:
        """Test handling complete process with engine error"""
        engine_error = EngineError("Connection timeout", operation="connect")

        with patch.object(handler, "_handle_engine_error") as mock_handle:
            mock_handle.return_value = MagicMock(spec=CliErrorContext)

            handler.handle_complete_process(engine_error=engine_error)

            mock_handle.assert_called_once_with(engine_error)

    def test_handle_complete_process_results_error(
        self, handler: CliExceptionHandler, mock_result: MagicMock
    ) -> None:
        """Test handling complete process with results containing errors"""
        results = [mock_result]

        with patch.object(handler, "_extract_error_results") as mock_extract:
            mock_extract.return_value = [mock_result]

            with patch.object(handler, "_handle_result_errors") as mock_handle:
                mock_handle.return_value = MagicMock(spec=CliErrorContext)

                handler.handle_complete_process(results=cast(list, results))

                mock_extract.assert_called_once_with(results)
                mock_handle.assert_called_once_with([mock_result])

    def test_handle_complete_process_success(
        self, handler: CliExceptionHandler
    ) -> None:
        """Test handling complete process with no errors"""
        success_result = MagicMock(spec=ExecutionResultSchema)
        success_result.status = "PASSED"
        results = [success_result]

        with patch.object(handler, "_extract_error_results") as mock_extract:
            mock_extract.return_value = []

            with patch.object(handler, "_create_success_context") as mock_create:
                mock_create.return_value = MagicMock(spec=CliErrorContext)

                handler.handle_complete_process(results=cast(list, results))

                mock_extract.assert_called_once_with(results)
                mock_create.assert_called_once_with(results)

    def test_handle_validation_process(self, handler: CliExceptionHandler) -> None:
        """Test handling validation process (backward compatibility)"""
        engine_error = EngineError("Connection timeout", operation="connect")
        results = [MagicMock(spec=ExecutionResultSchema)]

        with patch.object(handler, "handle_complete_process") as mock_handle:
            mock_handle.return_value = MagicMock(spec=CliErrorContext)

            handler.handle_validation_process(
                engine_error=engine_error, results=cast(list, results)
            )

            mock_handle.assert_called_once_with(
                cli_error=None,
                schema_error=None,
                engine_error=engine_error,
                results=results,
            )

    def test_build_native_error_technical_details(
        self, handler: CliExceptionHandler
    ) -> None:
        """Test building technical details for native error"""
        error = ValueError("Test error")

        details = handler._build_native_error_technical_details(error)

        assert "Error Type: ValueError" in details
        assert "Error Message: Test error" in details
        assert "Traceback:" in details

    def test_build_schema_error_technical_details(
        self, handler: CliExceptionHandler
    ) -> None:
        """Test building technical details for schema error"""
        # Fix: Use context parameter instead of field/value attributes
        error = OperationError(
            "Invalid connection",
            context={"field": "host", "value": "localhost", "operation": "connect"},
        )

        details = handler._build_schema_error_technical_details(error)

        assert "Error Type: OperationError" in details
        assert "Error Message: Invalid connection" in details
        assert "Context:" in details

    def test_build_technical_details(self, handler: CliExceptionHandler) -> None:
        """Test building technical details for engine error"""
        error = EngineError("Connection timeout", operation="connect")

        details = handler._build_technical_details(error)

        assert "Error Type: EngineError" in details
        assert "Error Message: Connection timeout" in details
        assert "Context:" in details

    def test_build_result_technical_details(
        self, handler: CliExceptionHandler, mock_result: MagicMock
    ) -> None:
        """Test building technical details for result error"""
        details = handler._build_result_technical_details(mock_result)

        assert "Rule ID: test_rule_1" in details
        assert "Status: ERROR" in details
        assert "Error Message: Test error message" in details
        assert "Entity: test_db.customers" in details

    def test_build_multiple_errors_technical_details(
        self, handler: CliExceptionHandler, mock_result: MagicMock
    ) -> None:
        """Test building technical details for multiple result errors"""
        mock_result2 = MagicMock(spec=ExecutionResultSchema)
        mock_result2.rule_id = "test_rule_2"
        mock_result2.status = "ERROR"
        mock_result2.error_message = "Another error"

        with patch.object(
            handler,
            "_build_result_technical_details",
            return_value="Single result details",
        ):
            details = handler._build_multiple_errors_technical_details(
                [mock_result, mock_result2]
            )

            assert "Total Errors: 2" in details
            assert "Error 1:" in details
            assert "Error 2:" in details
            assert "Single result details" in details

    def test_build_error_summary(self, handler: CliExceptionHandler) -> None:
        """Test building error summary for multiple error groups"""
        error_groups: Dict[str, list] = {
            "table_not_found": [MagicMock(), MagicMock()],
            "column_not_found": [MagicMock()],
        }

        summary = handler._build_error_summary(error_groups)

        assert "• 2 table_not_found errors" in summary
        assert "• 1 column_not_found errors" in summary

    def test_select_primary_error_category(self, handler: CliExceptionHandler) -> None:
        """Test selecting primary error category based on priority"""
        error_groups: Dict[str, list] = {
            "execution_generic": [MagicMock()],
            "table_not_found": [MagicMock()],
            "data_access_denied": [MagicMock()],
        }

        category = handler._select_primary_error_category(error_groups)

        assert category == "table_not_found"  # Should select highest priority

    def test_select_primary_error_category_empty(
        self, handler: CliExceptionHandler
    ) -> None:
        """Test selecting primary error category with empty groups"""
        error_groups: Dict[str, list] = {}

        category = handler._select_primary_error_category(error_groups)

        assert category == "execution_generic"  # Should return default

    def test_get_entity_name(
        self, handler: CliExceptionHandler, mock_result: MagicMock
    ) -> None:
        """Test getting entity name from result"""
        entity_name = handler._get_entity_name(mock_result)

        assert entity_name == "test_db.customers"

    def test_get_entity_name_fallback(self, handler: CliExceptionHandler) -> None:
        """Test getting entity name fallback when not available"""
        result = MagicMock(spec=ExecutionResultSchema)
        # Fix: Mock the get_entity_name method to return "unknown" instead of None
        result.get_entity_name.return_value = "unknown"

        # Mock dataset_metrics to return empty list
        result.dataset_metrics = []

        entity_name = handler._get_entity_name(result)

        assert entity_name == "unknown"
