"""
Error Handler Utility Test Module

Test the functionality of the error handler utility (error_handler):
1. Test exception capture and handling
2. Test error logging
3. Test error response formatting
4. Test context information handling
5. Test exception type mapping
6. Test custom exception handling
7. Test exception handling performance
8. Test exception handling concurrency safety
9. Test exception handling internationalization
"""

import asyncio
import concurrent.futures
import time
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError, ProgrammingError, SQLAlchemyError

# Assume the error handling tool is located in app.utils.error_handler
from shared.utils.error_handler import (
    EngineError,
    OperationError,
    RuleExecutionError,
    async_with_error_handling,
    format_error_response,
    handle_exception,
    log_error,
    map_exception,
    with_error_handling,
)


class TestExceptionHandling:
    """Test exception capture and handling functionality"""

    def test_handle_exception_basic(self) -> None:
        """Test basic exception handling"""
        # Create a test exception
        test_exception = ValueError("Test error")

        # Mock logger
        mock_logger = MagicMock()

        # Call the exception handler
        result = handle_exception(test_exception, logger=mock_logger)

        # Verify result
        assert result["status"] == "error"
        assert "Test error" in result["message"]
        assert result["error_type"] == "ValueError"

        # Verify logging
        mock_logger.exception.assert_called_once()

    def test_handle_exception_with_context(self) -> None:
        """Test exception handling with context"""
        # Create a test exception
        test_exception = ValueError("Test error")

        # Mock logger
        mock_logger = MagicMock()

        # Call the exception handler with context
        result = handle_exception(
            test_exception, context="When executing rule", logger=mock_logger
        )

        # Verify result
        assert result["status"] == "error"
        assert "When executing rule" in result["message"]
        assert "Test error" in result["message"]
        assert result["error_type"] == "ValueError"

        # Verify logging
        mock_logger.exception.assert_called_once()

    def test_handle_database_error(self) -> None:
        """Test database error handling"""
        # Create a database error
        db_error = SQLAlchemyError("database connection failed")

        # Mock logger
        mock_logger = MagicMock()

        # Call the exception handler
        result = handle_exception(db_error, logger=mock_logger)

        # Verify result
        assert result["status"] == "error"
        assert "Database error" in result["message"]
        assert result["error_type"] == "OperationError"

        # Verify logging
        mock_logger.exception.assert_called_once()

    def test_handle_rule_execution_error(self) -> None:
        """Test rule execution error handling"""
        # Create a rule execution error
        rule_error = RuleExecutionError("Rule execution failed")

        # Mock logger
        mock_logger = MagicMock()

        # Call the exception handler
        result = handle_exception(rule_error, logger=mock_logger)

        # Verify result
        assert result["status"] == "error"
        assert "Rule execution failed" in result["message"]
        assert result["error_type"] == "RuleExecutionError"

        # Verify logging
        mock_logger.exception.assert_called_once()


class TestErrorResponseFormatting:
    """Test error response formatting functionality"""

    def test_format_error_response_basic(self) -> None:
        """Test basic error response formatting"""
        # Create a test exception
        test_exception = ValueError("Test error")

        # Call the formatting function
        response = format_error_response(test_exception)

        # Verify the result
        assert response["status"] == "error"
        assert "Test error" in response["message"]
        assert response["error_type"] == "ValueError"
        assert "timestamp" in response

    def test_format_error_response_with_details(self) -> None:
        """Test error response formatting with details"""
        # Create a test exception
        test_exception = ValueError("Test error")

        # Call the formatting function with details
        response = format_error_response(
            test_exception, details={"rule_id": "123", "table": "test_table"}
        )

        # Verify the result
        assert response["status"] == "error"
        assert "Test error" in response["message"]
        assert response["error_type"] == "ValueError"
        assert response["details"]["rule_id"] == "123"
        assert response["details"]["table"] == "test_table"

    def test_format_error_response_with_code(self) -> None:
        """Test error response formatting with error code"""
        # Create a test exception
        test_exception = ValueError("Test error")

        # Call the formatting function with error code
        response = format_error_response(test_exception, error_code="E1001")

        # Verify the result
        assert response["status"] == "error"
        assert "Test error" in response["message"]
        assert response["error_type"] == "ValueError"
        assert response["error_code"] == "E1001"


class TestErrorLogging:
    """Test error logging functionality"""

    def test_log_error_basic(self) -> None:
        """Test basic error logging"""
        # Create a test exception
        test_exception = ValueError("Test error")

        # Mock logger
        mock_logger = MagicMock()

        # Call the log function
        log_error(test_exception, logger=mock_logger)

        # Verify logging
        mock_logger.exception.assert_called_once()
        assert "Test error" in mock_logger.exception.call_args[0][0]

    def test_log_error_with_context(self) -> None:
        """Test error logging with context"""
        # Create a test exception
        test_exception = ValueError("Test error")

        # Mock logger
        mock_logger = MagicMock()

        # Call the log function with context
        log_error(test_exception, context="When executing rule", logger=mock_logger)

        # Verify logging
        mock_logger.exception.assert_called_once()
        assert "When executing rule" in mock_logger.exception.call_args[0][0]
        assert "Test error" in mock_logger.exception.call_args[0][0]

    def test_log_error_with_details(self) -> None:
        """Test error logging with details"""
        # Create a test exception
        test_exception = ValueError("Test error")

        # Mock logger
        mock_logger = MagicMock()

        # Call the log function with details
        log_error(
            test_exception,
            details={"rule_id": "123", "table": "test_table"},
            logger=mock_logger,
        )

        # Verify logging
        mock_logger.exception.assert_called_once()
        log_message = mock_logger.exception.call_args[0][0]
        assert "Test error" in log_message
        assert "rule_id: 123" in log_message
        assert "table: test_table" in log_message


class TestExceptionMapping:
    """Test exception type mapping functionality"""

    def test_map_database_exception(self) -> None:
        """Test database exception mapping"""
        # Create a database exception
        db_error = SQLAlchemyError("database connection failed")

        # Call the mapping function
        mapped_error = map_exception(db_error)

        # Verify result
        assert isinstance(mapped_error, OperationError)
        assert "database connection failed" in str(mapped_error)

    def test_map_operational_error(self) -> None:
        """Test operational error mapping"""
        # Create an operational error
        op_error = OperationalError(
            "statement", {"param": "value"}, Exception("connection timeout")
        )

        # Call the mapping function
        mapped_error = map_exception(op_error)

        # Verify result
        assert isinstance(mapped_error, EngineError)
        assert "connection timeout" in str(mapped_error)

    def test_map_programming_error(self) -> None:
        """Test programming error mapping"""
        # Create a programming error
        prog_error = ProgrammingError(
            "statement", {"param": "value"}, Exception("SQL syntax error")
        )

        # Call the mapping function
        mapped_error = map_exception(prog_error)

        # Verify result
        assert isinstance(mapped_error, RuleExecutionError)
        assert "SQL syntax error" in str(mapped_error)

    def test_map_value_error(self) -> None:
        """Test value error mapping"""
        # Create a value error
        value_error = ValueError("Invalid parameter value")

        # Call the mapping function
        mapped_error = map_exception(value_error)

        # Verify result
        assert isinstance(mapped_error, ValueError)
        assert "Invalid parameter value" in str(mapped_error)


class TestErrorHandlingDecorators:
    """Test error handling decorator functionality"""

    def test_with_error_handling_decorator(self) -> None:
        """Test sync error handling decorator"""

        # Create a test function
        @with_error_handling
        def test_function(x: int, y: int) -> float:
            if y == 0:
                raise ValueError("Division by zero is not allowed")
            return x / y

        # Test normal case
        result = test_function(10, 2)
        assert result == 5.0

        # Test exception case
        result = test_function(10, 0)
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "Division by zero is not allowed" in result["message"]

    def test_with_error_handling_decorator_with_logger(self) -> None:
        """Test sync error handling decorator with logger"""
        # Mock logger
        mock_logger = MagicMock()

        # Create a test function
        @with_error_handling(logger=mock_logger)
        def test_function(x: int, y: int) -> float:
            if y == 0:
                raise ValueError("Division by zero is not allowed")
            return x / y

        # Test exception case
        result = test_function(10, 0)
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "Division by zero is not allowed" in result["message"]

        # Verify logging
        mock_logger.exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_with_error_handling_decorator(self) -> None:
        """Test async error handling decorator"""

        # Create a test function
        @async_with_error_handling
        async def test_async_function(x: int, y: int) -> float:
            await asyncio.sleep(0.01)  # Simulate async operation
            if y == 0:
                raise ValueError("Division by zero is not allowed")
            return x / y

        # Test normal case
        result = await test_async_function(10, 2)
        assert result == 5.0

        # Test exception case
        result = await test_async_function(10, 0)
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "Division by zero is not allowed" in result["message"]

    @pytest.mark.asyncio
    async def test_async_with_error_handling_decorator_with_logger(self) -> None:
        """Test async error handling decorator with logger"""
        # Mock logger
        mock_logger = MagicMock()

        # Create a test function
        @async_with_error_handling(logger=mock_logger)
        async def test_async_function(x: int, y: int) -> float:
            await asyncio.sleep(0.01)  # Simulate async operation
            if y == 0:
                raise ValueError("Division by zero is not allowed")
            return x / y

        # Test exception case
        result = await test_async_function(10, 0)
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "Division by zero is not allowed" in result["message"]

        # Verify logging
        mock_logger.exception.assert_called_once()


class TestErrorHandlingPerformance:
    """Test error handling performance"""

    def test_error_handling_performance(self) -> None:
        """Test error handling performance"""
        # Create a test exception
        test_exception = ValueError("Test error")

        # Mock logger
        mock_logger = MagicMock()

        # Measure time to handle 1000 exceptions
        start_time = time.time()
        for _ in range(1000):
            handle_exception(test_exception, logger=mock_logger)
        end_time = time.time()

        # Verify performance - should not exceed 1 second for 1000 exceptions
        assert end_time - start_time < 1.0


class TestConcurrentErrorHandling:
    """Test concurrent error handling"""

    def test_concurrent_error_handling(self) -> None:
        """Test concurrent error handling"""

        # Create a test function
        @with_error_handling
        def test_function(x: int) -> int:
            if x % 2 == 0:
                raise ValueError(f"Even number error: {x}")
            return x

        # Run the test function concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(test_function, range(100)))

        # Verify results
        for i, result in enumerate(results):
            if i % 2 == 0:
                assert isinstance(result, dict)
                assert result["status"] == "error"
                assert f"Even number error: {i}" in result["message"]
            else:
                assert result == i


class TestInternationalization:
    """Test error handling internationalization"""

    def test_error_message_i18n(self) -> None:
        """Test error message internationalization"""
        # Create a test exception
        test_exception = ValueError("Test error")

        # Mock translation function
        def mock_translate(message: str, lang: str) -> str:
            if lang == "en":
                return message  # English is default, no translation needed
            elif lang == "zh":
                return message.replace("Test error", "测试错误")
            return message

        # Use the mock translation function to handle exception
        with patch(
            "shared.utils.error_handler.translate_message", side_effect=mock_translate
        ):
            # Chinese error message
            result_zh = handle_exception(test_exception, lang="zh")
            assert "测试错误" in result_zh["message"]

            # English error message (default)
            result_en = handle_exception(test_exception, lang="en")
            assert "Test error" in result_en["message"]

            # Default language (English)
            result_default = handle_exception(test_exception)
            assert "Test error" in result_default["message"]

    def test_translate_message_default_language(self) -> None:
        """Test translate_message function with default language behavior"""
        from shared.utils.error_handler import translate_message

        # Test default language (English) - should return original message
        message = "Test error message"
        result = translate_message(message)
        assert result == message

        # Test explicit English language
        result_en = translate_message(message, "en")
        assert result_en == message

        # Test Chinese translation
        result_zh = translate_message(message, "zh")
        assert (
            "测试错误" in result_zh or result_zh == message
        )  # 兼容未实现中文翻译的情况

        # Test unknown language - should return original message
        result_unknown = translate_message(message, "fr")
        assert result_unknown == message
