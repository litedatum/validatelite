"""
Performance Monitoring System Unit Tests

Focus on testing the core functionality of performance monitoring.
Perform statistical collection testing.
Performance metric calculation test.
3. Data Storage Monitoring Tests
Performance decorator testing.
"""

import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch

import pytest

from shared.utils.logger import (
    LoggerManager,
    get_logger,
    get_logger_manager,
    log_database_operation,
    log_performance,
    log_rule_execution,
    performance_monitor,
)


class TestPerformanceMonitoring:
    """Test basic performance monitoring functionality."""

    @pytest.fixture
    def logger_manager(self) -> LoggerManager:
        """Logging manager fixture."""
        config = {
            "to_file": False,
            "to_console": False,  # Suppress log output during testing.
            "level": "INFO",
        }
        return LoggerManager(config)

    def test_performance_log_creation(self, logger_manager: LoggerManager) -> None:
        """Testing performance log creation."""
        # Performance log testing
        logger_manager.log_performance("test_operation", 1.5)

        # Verify that the performance logger is present.
        perf_logger = logger_manager.get_logger("performance")
        assert perf_logger is not None

    def test_performance_log_with_metadata(self, logger_manager: LoggerManager) -> None:
        """Performance testing of logs with metadata."""
        metadata = {
            "table": "test_table",
            "rows_processed": 1000,
            "memory_usage": "50MB",
        }

        logger_manager.log_performance("complex_operation", 2.3, **metadata)

        # Verify successful logging.
        perf_logger = logger_manager.get_logger("performance")
        assert perf_logger is not None

    def test_database_operation_log(self, logger_manager: LoggerManager) -> None:
        """Test database operation logs."""
        # Successful database operation.
        logger_manager.log_database_operation(
            operation="SELECT",
            table="users",
            duration=0.5,
            success=True,
            rows_affected=100,
        )

        # Failed database operation.
        logger_manager.log_database_operation(
            operation="UPDATE",
            table="orders",
            duration=1.2,
            success=False,
            error="Connection timeout",
        )

        # Verify the database logger is present.
        db_logger = logger_manager.get_logger("database")
        assert db_logger is not None

    def test_rule_execution_log(self, logger_manager: LoggerManager) -> None:
        """Test rule execution log"""
        logger_manager.log_rule_execution(
            rule_id="rule_001",
            rule_type="NOT_NULL",
            table="customers",
            duration=0.8,
            result_count=50,
            anomaly_rate=5.0,
        )

        # Verify that the rule execution logger exists.
        rule_logger = logger_manager.get_logger("rule_execution")
        assert rule_logger is not None

    def test_audit_log(self, logger_manager: LoggerManager) -> None:
        """Test audit log"""
        logger_manager.log_audit(
            user="admin",
            action="CREATE_RULE",
            message="Created new validation rule",
            rule_id="rule_002",
            table="products",
        )

        # Verify that the audit log logger is present.
        audit_logger = logger_manager.get_logger("audit")
        assert audit_logger is not None


class TestPerformanceDecorator:
    """Test the performance monitoring decorator."""

    def test_performance_decorator_basic(self) -> None:
        """Tests the base performance decorator."""

        @performance_monitor()
        def test_function() -> str:
            time.sleep(0.1)
            return "test_result"

        # Simulated performance logging.
        with patch("shared.utils.logger.log_performance") as mock_log:
            result = test_function()

            assert result == "test_result"
            mock_log.assert_called_once()

            # Validate the calling arguments.  Or, slightly more formally: Validate the arguments passed to the function/method call.
            call_args = mock_log.call_args
            assert len(call_args[0]) == 2  # operation_name, duration
            assert call_args[0][0].endswith("test_function")
            assert isinstance(call_args[0][1], float)
            assert call_args[0][1] >= 0.1

    def test_performance_decorator_with_name(self) -> None:
        """Test the performance decorator with a custom name."""

        @performance_monitor("custom_operation")
        def test_function() -> int:
            time.sleep(0.05)
            return 42

        with patch("shared.utils.logger.log_performance") as mock_log:
            result = test_function()

            assert result == 42
            mock_log.assert_called_once()

            # Validate the operation name.
            call_args = mock_log.call_args
            assert call_args[0][0] == "custom_operation"

    def test_performance_decorator_with_exception(self) -> None:
        """Test the performance decorator under exceptional circumstances."""

        @performance_monitor("error_operation")
        def failing_function() -> None:
            time.sleep(0.02)
            raise ValueError("Test error")

        with patch("shared.utils.logger.log_performance") as mock_log:
            with pytest.raises(ValueError, match="Test error"):
                failing_function()

            mock_log.assert_called_once()

            # Verify that the error message is logged.
            call_args = mock_log.call_args
            assert call_args[0][0] == "error_operation"
            assert "error" in call_args[1]
            assert call_args[1]["error"] == "Test error"

    def test_performance_decorator_with_args(self) -> None:
        """A performance decorator for testing functions with arguments."""

        @performance_monitor()
        def function_with_args(a: float, b: float, c: Optional[float] = None) -> float:
            time.sleep(0.01)
            return a + b + (c or 0)

        with patch("shared.utils.logger.log_performance") as mock_log:
            result = function_with_args(1, 2, c=3)

            assert result == 6
            mock_log.assert_called_once()


class TestPerformanceStatistics:
    """Test the performance statistics functionality."""

    def test_concurrent_performance_logging(self) -> None:
        """Testing concurrent performance logging."""
        results = []

        def log_performance_task(task_id: int) -> None:
            # Verify thread safety by calling the function directly instead of using a mock.
            try:
                log_performance(f"task_{task_id}", 0.1 * task_id)
                results.append(
                    1
                )  # Successfully logged. / Log entry successful. / Record successfully saved.
            except Exception:
                results.append(0)  # Failure.

        # Create multiple threads to concurrently log performance data.
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_performance_task, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete.
        for thread in threads:
            thread.join()

        # Verify that performance logs have been recorded for all tasks.
        assert len(results) == 5
        assert all(count == 1 for count in results)

    def test_performance_metrics_calculation(self) -> None:
        """Testing performance metric calculations."""
        # Simulates the execution time of a series of operations.
        operations = [
            ("operation_1", 0.5),
            ("operation_1", 0.7),
            ("operation_1", 0.3),
            ("operation_2", 1.2),
            ("operation_2", 0.8),
        ]

        # Mocks the `log_performance` method of the `LoggerManager` class.
        with patch("shared.utils.logger.get_logger_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager

            for op_name, duration in operations:
                log_performance(op_name, duration)

            # Verify that all operations are logged.
            assert mock_manager.log_performance.call_count == 5

            # Validate the calling arguments.  / Verify the calling arguments.
            calls = mock_manager.log_performance.call_args_list
            for i, (op_name, duration) in enumerate(operations):
                assert calls[i][0][0] == op_name
                assert calls[i][0][1] == duration

    def test_performance_threshold_monitoring(self) -> None:
        """Performance threshold monitoring test"""
        # Simulates a slow operation.
        slow_operations = [
            ("slow_query", 5.0),
            ("normal_query", 0.5),
            ("very_slow_query", 10.0),
        ]

        # Mocks the `log_performance` method of the `LoggerManager` class.
        with patch("shared.utils.logger.get_logger_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager

            for op_name, duration in slow_operations:
                log_performance(op_name, duration, threshold=2.0)

            # Verify that all operations are logged.
            assert mock_manager.log_performance.call_count == 3

            # Verify that long-running operations are flagged or marked.
            calls = mock_manager.log_performance.call_args_list
            for i, (op_name, duration) in enumerate(slow_operations):
                call_kwargs = calls[i][1] if len(calls[i]) > 1 else {}
                if duration > 2.0:
                    assert "threshold" in call_kwargs


class TestPerformanceIntegration:
    """Test the performance monitoring integration."""

    def test_rule_execution_performance_tracking(self) -> None:
        """Performance tracking of rule execution."""
        rule_data: Dict[str, Any] = {
            "rule_id": "test_rule_001",
            "rule_type": "RANGE",
            "table": "test_table",
            "duration": 1.5,
            "result_count": 100,
        }

        # This code mocks the `log_rule_execution` method of the `LoggerManager` class.
        with patch("shared.utils.logger.get_logger_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager

            log_rule_execution(
                rule_data["rule_id"],
                rule_data["rule_type"],
                rule_data["table"],
                rule_data["duration"],
                rule_data["result_count"],
            )

            # Validate the positional arguments passed to the function.
            mock_manager.log_rule_execution.assert_called_once_with(
                "test_rule_001", "RANGE", "test_table", 1.5, 100
            )

    def test_database_operation_performance_tracking(self) -> None:
        """Performance monitoring of database operations."""
        db_operations: List[Dict[str, Any]] = [
            {
                "operation": "SELECT",
                "table": "users",
                "duration": 0.3,
                "success": True,
                "rows_returned": 50,
            },
            {
                "operation": "INSERT",
                "table": "orders",
                "duration": 0.8,
                "success": True,
                "rows_affected": 1,
            },
            {
                "operation": "UPDATE",
                "table": "products",
                "duration": 2.1,
                "success": False,
                "error": "Timeout",
            },
        ]

        # Mocks the `log_database_operation` method of the `LoggerManager` class.
        with patch("shared.utils.logger.get_logger_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager

            for op_data in db_operations:
                log_database_operation(
                    op_data["operation"],
                    op_data["table"],
                    op_data["duration"],
                    op_data["success"],
                )

            assert mock_manager.log_database_operation.call_count == 3

            # Validate the calling arguments for each operation.
            calls = mock_manager.log_database_operation.call_args_list
            expected_calls = [
                ("SELECT", "users", 0.3, True),
                ("INSERT", "orders", 0.8, True),
                ("UPDATE", "products", 2.1, False),
            ]

            for i, expected_args in enumerate(expected_calls):
                call_args, call_kwargs = calls[i]
                assert call_args == expected_args

    def test_performance_monitoring_error_handling(self) -> None:
        """Test performance monitoring error handling."""
        # Ensure that logging failures do not interrupt or affect the main application flow.
        with patch(
            "shared.utils.logger.get_logger_manager",
            side_effect=Exception("Logger error"),
        ):
            # These calls are expected to succeed without throwing exceptions.
            log_performance("test_op", 1.0)
            log_database_operation("SELECT", "test_table", 0.5)
            log_rule_execution("rule_1", "NOT_NULL", "table_1", 0.3, 10)

    def test_logger_manager_singleton(self) -> None:
        """Test the singleton pattern implementation of the log manager."""
        manager1 = get_logger_manager()
        manager2 = get_logger_manager()

        # This should be the same instance.
        assert manager1 is manager2

    def test_logger_manager_thread_safety(self) -> None:
        """Test the thread safety of the log manager."""
        managers = []

        def get_manager() -> None:
            manager = get_logger_manager()
            managers.append(manager)

        # Create multiple threads to access the manager concurrently.
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_manager)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete.
        for thread in threads:
            thread.join()

        # Verify that all threads are accessing the same instance.
        assert len(managers) == 10
        assert all(manager is managers[0] for manager in managers)


class TestPerformanceConfiguration:
    """Test the performance monitoring configuration."""

    def test_logger_manager_configuration(self) -> None:
        """Testing the log manager configuration."""
        config = {
            "to_file": True,
            "to_console": False,
            "level": "DEBUG",
            "log_dir": "/tmp/test_logs",
            "max_file_size": "10MB",
            "backup_count": 5,
        }

        manager = LoggerManager(config)
        assert manager.config["to_file"] == True
        assert manager.config["to_console"] == False
        assert manager.config["level"] == "DEBUG"

    def test_logger_manager_default_configuration(self) -> None:
        """Test the default configuration of the log manager."""
        manager = LoggerManager()

        # Verify the existence of the default configuration.
        assert "to_file" in manager.config
        assert "to_console" in manager.config
        assert "level" in manager.config

    def test_logger_manager_config_update(self) -> None:
        """Testing log manager configuration updates."""
        manager = LoggerManager({"level": "INFO"})

        # Update the configuration.
        new_config = {"level": "DEBUG", "to_file": True}
        manager.update_config(new_config)

        # Verify the configuration has been updated.
        assert manager.config["level"] == "DEBUG"
        assert manager.config["to_file"] == True


class TestPerformanceMetrics:
    """Test performance metrics."""

    def test_operation_timing_accuracy(self) -> None:
        """Test the accuracy of operation timing."""

        @performance_monitor("timing_test")
        def timed_operation() -> float:
            sleep_time = 0.1
            time.sleep(sleep_time)
            return sleep_time

        with patch("shared.utils.logger.log_performance") as mock_log:
            expected_time = timed_operation()

            mock_log.assert_called_once()
            call_args = mock_log.call_args
            actual_duration = call_args[0][1]

            # Verify timing accuracy (allowing for some tolerance).
            assert abs(actual_duration - expected_time) < 0.05

    def test_memory_usage_tracking(self) -> None:
        """Memory usage tracking test."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Simulates a memory-intensive operation.
        @performance_monitor("memory_test")
        def memory_intensive_operation() -> int:
            # Create a large list.
            data = [i for i in range(100000)]
            return len(data)

        with patch("shared.utils.logger.log_performance") as mock_log:
            result = memory_intensive_operation()

            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            assert result == 100000
            mock_log.assert_called_once()

            # Memory usage information can be logged in the actual implementation.
            # This only verifies basic functionality.

    def test_error_rate_tracking(self) -> None:
        """Tracking error rates."""
        error_count = 0
        total_count = 0

        @performance_monitor("error_rate_test")
        def operation_with_errors(should_fail: bool = False) -> str:
            nonlocal error_count, total_count
            total_count += 1

            if should_fail:
                error_count += 1
                raise ValueError("Simulated error")

            return "success"

        with patch("shared.utils.logger.log_performance") as mock_log:
            # Executes the successful operation.  Or, more naturally:  Performs the successful operation.
            for _ in range(7):
                operation_with_errors(False)

            # Execute the failing operation.  Or, if more context is appropriate:  Retry/re-attempt the failed operation.
            for _ in range(3):
                try:
                    operation_with_errors(True)
                except ValueError:
                    pass

            # Verify that all operations are logged.
            assert mock_log.call_count == 10

            # Calculate the error rate.
            error_rate = error_count / total_count
            assert error_rate == 0.3  # An error rate of 30%.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
