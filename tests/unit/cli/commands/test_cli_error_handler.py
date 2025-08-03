"""
🧙‍♂️ Error Handler TDD Tests - Modern Testing Architecture

Features:
- Comprehensive exception flow testing
- User-friendly error message validation
- Edge case error handling
- Recovery mechanism testing
- Error logging and reporting verification
"""

import sys
import traceback
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cli.core.error_handler import ErrorHandler
from cli.exceptions import (
    ConnectionError,
    DatabaseError,
    RuleParsingError,
    ValidationError,
)
from shared.enums import ExecutionStatus, SeverityLevel
from tests.shared.builders.test_builders import TestDataBuilder


class TestErrorHandler:
    """Modern Error Handler Test Suite - Testing Ghost's Architecture"""

    @pytest.fixture
    def error_handler(self) -> ErrorHandler:
        """Error handler instance"""
        return ErrorHandler()

    @pytest.fixture
    def mock_logger(self) -> Generator[MagicMock, None, None]:
        """Mock logger for testing error logging"""
        with patch("cli.core.error_handler.logger") as mock_log:
            yield mock_log

    # === BASIC ERROR HANDLING TESTS ===

    def test_handle_file_not_found_error(
        self, error_handler: ErrorHandler, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test user-friendly file not found error handling"""
        error = FileNotFoundError("No such file or directory: 'users.csv'")

        result = error_handler.handle_error(error, context="checking file")

        captured = capsys.readouterr()

        assert result.exit_code == 1
        assert result.user_message is not None
        assert "users.csv" in result.user_message
        assert "file not found" in result.user_message.lower()
        assert "please check" in result.user_message.lower()

        # Should suggest helpful actions
        assert any(
            suggestion in result.user_message.lower()
            for suggestion in [
                "check the file path",
                "verify the file exists",
                "ensure correct spelling",
            ]
        )

    def test_handle_validation_error(
        self, error_handler: ErrorHandler, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test validation error with helpful suggestions"""
        error = ValidationError("Invalid rule syntax: 'not_nul(id)'")

        result = error_handler.handle_error(error, context="parsing rules")

        assert result.exit_code == 1
        assert "not_nul" in result.user_message
        assert "rule syntax" in result.user_message.lower()

        # Should suggest correct syntax
        assert (
            "not_null" in result.user_message
            or "correct syntax" in result.user_message.lower()
        )

    def test_handle_database_connection_error(
        self, error_handler: ErrorHandler
    ) -> None:
        """Test database connection error with troubleshooting help"""
        error = ConnectionError(
            "Failed to connect to mysql://user:***@localhost:3306/testdb"
        )

        result = error_handler.handle_error(error, context="connecting to database")

        assert result.exit_code == 2  # Different exit code for connection errors
        assert "connection failed" in result.user_message.lower()
        assert "localhost:3306" in result.user_message
        assert "testdb" in result.user_message

        # Should provide troubleshooting steps
        troubleshooting_hints = [
            "check if database server is running",
            "verify connection parameters",
            "ensure network connectivity",
            "check firewall settings",
        ]
        assert any(
            hint in result.user_message.lower() for hint in troubleshooting_hints
        )

    def test_handle_rule_parsing_error(self, error_handler: ErrorHandler) -> None:
        """Test rule parsing error with syntax help"""
        error = RuleParsingError(
            "Unknown rule type 'custom_rule'", rule_expression="custom_rule(column)"
        )

        result = error_handler.handle_error(error, context="parsing inline rules")

        assert result.exit_code == 1
        assert "custom_rule" in result.user_message
        assert "unknown rule type" in result.user_message.lower()

        # Should list supported rule types
        supported_rules = ["not_null", "unique", "length", "range", "regex"]
        assert any(rule in result.user_message.lower() for rule in supported_rules)

    def test_handle_permission_error(self, error_handler: ErrorHandler) -> None:
        """Test permission error handling"""
        error = PermissionError("Permission denied: '/restricted/data.csv'")

        result = error_handler.handle_error(error, context="reading file")

        assert result.exit_code == 1
        assert "permission denied" in result.user_message.lower()
        assert "/restricted/data.csv" in result.user_message

        # Should suggest permission solutions
        permission_hints = [
            "check file permissions",
            "run with appropriate privileges",
            "contact system administrator",
        ]
        assert any(hint in result.user_message.lower() for hint in permission_hints)

    # === BOUNDARY CONDITION TESTS ===

    def test_handle_none_error(self, error_handler: ErrorHandler) -> None:
        """Test handling of None error (should not crash)"""
        result = error_handler.handle_error(None, context="unknown")

        assert result.exit_code == 1
        assert "unknown error occurred" in result.user_message.lower()
        assert result.technical_details is not None

    def test_handle_empty_error_message(self, error_handler: ErrorHandler) -> None:
        """Test handling of empty error message"""
        error = Exception("")  # Empty message

        result = error_handler.handle_error(error, context="testing")

        assert result.exit_code == 1
        assert len(result.user_message) > 0  # Should generate meaningful message
        assert "error occurred" in result.user_message.lower()

    def test_handle_unicode_error_message(self, error_handler: ErrorHandler) -> None:
        """Test handling of Unicode error messages"""
        unicode_errors = [
            Exception("文件未找到: 用户数据.csv"),  # Chinese
            Exception("Archivo no encontrado: données.csv"),  # Spanish/French mix
            Exception("Файл не найден: данные.csv"),  # Russian
        ]

        for error in unicode_errors:
            result = error_handler.handle_error(error, context="file operation")

            assert result.exit_code == 1
            assert result.user_message is not None
            assert len(result.user_message) > 0
            # Should preserve Unicode characters
            assert any(char in result.user_message for char in str(error))

    def test_handle_extremely_long_error_message(
        self, error_handler: ErrorHandler
    ) -> None:
        """Test handling of extremely long error messages"""
        long_message = "Error: " + "A" * 10000  # Very long error message
        error = Exception(long_message)

        result = error_handler.handle_error(error, context="processing")

        assert result.exit_code == 1
        # Should truncate appropriately for user display
        assert len(result.user_message) < 1000  # Reasonable length for user
        assert (
            "..." in result.user_message or "truncated" in result.user_message.lower()
        )
        # But preserve full details for technical logging
        assert long_message in result.technical_details

    @given(st.integers(min_value=-999, max_value=999))
    def test_property_based_exit_codes(
        self, error_handler: ErrorHandler, exit_code: int
    ) -> None:
        """Property-based test for exit code handling"""
        error = Exception("Test error")

        result = error_handler.handle_error(
            error, context="testing", suggested_exit_code=exit_code
        )

        # Exit codes should be normalized to valid range
        assert 0 <= result.exit_code <= 255
        if exit_code < 0:
            assert result.exit_code == 1  # Default error code
        elif exit_code > 255:
            assert result.exit_code == 255  # Max valid exit code
        else:
            assert result.exit_code == exit_code

    # === ERROR RECOVERY TESTS ===

    def test_suggest_recovery_actions_file_errors(
        self, error_handler: ErrorHandler
    ) -> None:
        """Test recovery action suggestions for file errors"""
        file_errors = [
            (FileNotFoundError("users.csv not found"), "create the file"),
            (PermissionError("access denied"), "check permissions"),
            (IsADirectoryError("path is directory"), "specify file path"),
            (OSError("disk full"), "free up disk space"),
        ]

        for error, expected_suggestion in file_errors:
            result = error_handler.handle_error(error, context="file operation")

            assert any(
                word in result.user_message.lower()
                for word in expected_suggestion.split()
            )
            assert result.recovery_suggestions is not None
            assert len(result.recovery_suggestions) > 0

    def test_suggest_recovery_actions_database_errors(
        self, error_handler: ErrorHandler
    ) -> None:
        """Test recovery action suggestions for database errors"""
        db_errors = [
            (ConnectionError("connection refused"), "check database server"),
            (DatabaseError("table not found"), "verify table name"),
            (DatabaseError("access denied"), "check credentials"),
            (DatabaseError("timeout"), "retry with longer timeout"),
        ]

        for error, expected_suggestion in db_errors:
            result = error_handler.handle_error(error, context="database operation")

            assert any(
                word in result.user_message.lower()
                for word in expected_suggestion.split()
            )
            assert len(result.recovery_suggestions) > 0

    def test_auto_retry_mechanism(self, error_handler: ErrorHandler) -> None:
        """Test automatic retry mechanism for transient errors"""
        transient_error = ConnectionError("connection timeout")

        # Mock retry configuration
        with patch.object(error_handler, "is_retryable_error", return_value=True):
            with patch.object(error_handler, "attempt_recovery", return_value=True):
                result = error_handler.handle_error(
                    transient_error, context="database query", allow_retry=True
                )

                assert result.retry_attempted is True
                assert (
                    "retrying" in result.user_message.lower()
                    or "retry" in result.user_message.lower()
                )

    # === ERROR LOGGING TESTS ===

    def test_error_logging_with_context(
        self, error_handler: ErrorHandler, mock_logger: MagicMock
    ) -> None:
        """Test comprehensive error logging with context"""
        error = ValidationError("Invalid rule")
        context = {
            "operation": "rule_parsing",
            "file": "test.json",
            "line": 42,
            "user": "test_user",
        }

        result = error_handler.handle_error(error, context=context)

        # Verify logging calls
        mock_logger.error.assert_called()
        logged_message = mock_logger.error.call_args[0][0]

        assert "ValidationError" in logged_message
        assert "rule_parsing" in logged_message
        assert "test.json" in logged_message
        assert "42" in str(logged_message)

    def test_error_aggregation_multiple_errors(
        self, error_handler: ErrorHandler
    ) -> None:
        """Test aggregation of multiple related errors"""
        errors: List[Exception] = [
            ValidationError("Rule 1 invalid"),
            ValidationError("Rule 2 invalid"),
            ValidationError("Rule 3 invalid"),
        ]

        result = error_handler.handle_multiple_errors(
            errors, context="batch validation"
        )

        assert result.exit_code == 1
        assert (
            "3 validation errors" in result.user_message
            or "multiple errors" in result.user_message.lower()
        )
        assert all(f"Rule {i}" in result.technical_details for i in [1, 2, 3])

        # Should provide summary rather than overwhelming details
        assert len(result.user_message) < 500  # Reasonable summary length

    def test_error_severity_classification(self, error_handler: ErrorHandler) -> None:
        """Test error severity classification"""
        severity_tests = [
            (FileNotFoundError("file missing"), SeverityLevel.HIGH),
            (ValidationError("rule syntax"), SeverityLevel.MEDIUM),
            (ConnectionError("db timeout"), SeverityLevel.HIGH),
            (Warning("deprecated feature"), SeverityLevel.LOW),
        ]

        for error, expected_severity in severity_tests:
            result = error_handler.handle_error(error, context="testing")

            assert result.severity == expected_severity

            # Exit codes should reflect severity
            if expected_severity == SeverityLevel.HIGH:
                assert result.exit_code in [1, 2]  # Critical errors
            elif expected_severity == SeverityLevel.LOW:
                assert result.exit_code == 0  # Warnings shouldn't fail

    # === INTEGRATION AND PERFORMANCE TESTS ===

    def test_error_handling_performance(self, error_handler: ErrorHandler) -> None:
        """Test error handling performance with complex errors"""

        # Create complex error with deep stack trace
        def deep_function_call(depth: int) -> None:
            if depth == 0:
                raise ValueError("Deep error with complex traceback")
            return deep_function_call(depth - 1)

        try:
            deep_function_call(50)  # 50 levels deep
        except ValueError as deep_error:
            import time

            start_time = time.time()

            result = error_handler.handle_error(deep_error, context="deep call")

            end_time = time.time()
            handling_time = end_time - start_time

            # Should handle complex errors quickly (< 0.1 seconds)
            assert (
                handling_time < 0.1
            ), f"Error handling took {handling_time:.3f}s, expected < 0.1s"
            assert result.exit_code == 1
            assert "Deep error" in result.technical_details

    def test_concurrent_error_handling(self, error_handler: ErrorHandler) -> None:
        """Test concurrent error handling safety"""
        import threading
        import time

        results = []
        errors = []

        def handle_error_thread(thread_id: int) -> None:
            try:
                error = Exception(f"Thread {thread_id} error")
                result = error_handler.handle_error(
                    error, context=f"thread_{thread_id}"
                )
                results.append((thread_id, result.exit_code))
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(20):
            thread = threading.Thread(target=handle_error_thread, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Concurrent error handling failed: {errors}"
        assert len(results) == 20
        assert all(exit_code == 1 for _, exit_code in results)

    def test_memory_usage_with_large_errors(self, error_handler: ErrorHandler) -> None:
        """Test memory usage with large error data"""
        # Create error with large data
        large_data = "x" * (1024 * 1024)  # 1MB of data
        error = Exception(f"Large error: {large_data}")

        import os

        import psutil

        process = psutil.Process(os.getpid())

        memory_before = process.memory_info().rss

        result = error_handler.handle_error(error, context="large data")

        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before

        # Should not significantly increase memory (< 10MB)
        assert (
            memory_increase < 10 * 1024 * 1024
        ), f"Memory increase: {memory_increase / 1024 / 1024:.2f}MB"
        assert result.exit_code == 1

        # User message should be truncated, technical details preserved
        assert len(result.user_message) < 1000
        assert large_data in result.technical_details

    # === SPECIALIZED ERROR SCENARIOS ===

    def test_keyboard_interrupt_handling(
        self, error_handler: ErrorHandler, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test graceful handling of user interruption"""
        error: Optional[BaseException] = KeyboardInterrupt()

        result = error_handler.handle_error(error, context="user operation")

        captured = capsys.readouterr()

        assert result.exit_code == 130  # Standard exit code for SIGINT
        assert (
            "interrupted" in result.user_message.lower()
            or "cancelled" in result.user_message.lower()
        )
        assert (
            "progress saved" in result.user_message.lower()
            or "cleanup" in result.user_message.lower()
        )

    def test_memory_error_handling(self, error_handler: ErrorHandler) -> None:
        """Test handling of memory errors"""
        error = MemoryError("not enough memory")

        result = error_handler.handle_error(error, context="large dataset processing")

        assert result.exit_code == 1
        assert "memory" in result.user_message.lower()
        assert any(
            suggestion in result.user_message.lower()
            for suggestion in [
                "reduce dataset size",
                "increase available memory",
                "process in smaller batches",
            ]
        )

    def test_network_timeout_handling(self, error_handler: ErrorHandler) -> None:
        """Test network timeout error handling"""
        import socket

        error = socket.timeout("connection timed out")

        result = error_handler.handle_error(error, context="database connection")

        assert result.exit_code == 2
        assert "timeout" in result.user_message.lower()
        assert (
            "retry" in result.user_message.lower()
            or "connection" in result.user_message.lower()
        )

    # === ERROR REPORTING TESTS ===

    def test_generate_error_report(self, error_handler: ErrorHandler) -> None:
        """Test comprehensive error report generation"""
        error = ValidationError("Complex validation failure")

        result = error_handler.handle_error(
            error, context="validation", generate_report=True
        )

        assert result.error_report is not None
        report = result.error_report

        # Report should contain essential information
        assert "error_type" in report
        assert "timestamp" in report
        assert "context" in report
        assert "user_message" in report
        assert "technical_details" in report
        assert "recovery_suggestions" in report

        # Should be JSON serializable
        import json

        json_report = json.dumps(report)
        assert len(json_report) > 0

    def test_error_trend_analysis(self, error_handler: ErrorHandler) -> None:
        """Test error trend analysis and pattern detection"""
        # Simulate repeated similar errors
        similar_errors = [
            ValidationError("Rule syntax error in rule 1"),
            ValidationError("Rule syntax error in rule 2"),
            ValidationError("Rule syntax error in rule 3"),
        ]

        for error in similar_errors:
            error_handler.handle_error(error, context="rule parsing")

        trends = error_handler.get_error_trends()

        assert "ValidationError" in trends
        assert trends["ValidationError"]["count"] == 3
        assert "rule syntax" in trends["ValidationError"]["common_patterns"]
