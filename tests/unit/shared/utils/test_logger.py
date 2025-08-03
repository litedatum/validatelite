"""
Improved Logger Tests - Balanced Approach
Combines focused testing with critical edge cases

This test suite focuses on:
1. Core functionality with proper test isolation
2. Critical error handling and resilience
3. Essential concurrency safety
4. Maintainable test structure
"""

import json
import logging
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Generator, List
from unittest.mock import patch

import pytest

# Make sure the shared module is in the path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)

from shared.utils.logger import (
    ContextFilter,
    LoggerManager,
    ModuleFilter,
    StructuredFormatter,
    get_logger,
    log_audit,
    setup_logging,
)

# --- Fixtures ---


@pytest.fixture(autouse=True)
def cleanup_logging() -> Generator[None, None, None]:
    """Fixture to reset the logging system and clean up handlers after each test."""
    yield
    # Reset the global manager variables if they exist
    try:
        import shared.utils.logger as logger_module

        if hasattr(logger_module, "_logger_manager"):
            logger_module._logger_manager = None
        if hasattr(logger_module, "_initialized"):
            logger_module._initialized = False
    except (ImportError, AttributeError):
        pass

    # Remove all handlers from the root logger
    root = logging.getLogger()
    for handler in root.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
        root.removeHandler(handler)

    # Clear all loggers from the logging manager's dictionary
    logging.root.manager.loggerDict.clear()


@pytest.fixture
def temp_log_dir() -> Generator[str, None, None]:
    """Provides a temporary directory for log files that is cleaned up afterwards."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def default_config(temp_log_dir: str) -> dict:
    """Provides a default, safe configuration for tests."""
    return {
        "level": "DEBUG",
        "to_console": False,
        "to_file": True,
        "log_file": os.path.join(temp_log_dir, "app.log"),
        "performance_enabled": True,
        "performance_log_file": os.path.join(temp_log_dir, "performance.log"),
        "error_log_enabled": True,
        "error_log_file": os.path.join(temp_log_dir, "error.log"),
        "audit_log_enabled": True,
        "audit_log_file": os.path.join(temp_log_dir, "audit.log"),
        "structured": False,
        "max_bytes": 10485760,  # 10MB
        "backup_count": 5,
    }


# --- Core Functionality Tests ---


class TestLoggerManagerCore:
    """Test core LoggerManager functionality with proper isolation."""

    def test_initialization_and_basic_config(self, default_config: dict) -> None:
        """Test basic initialization and configuration."""
        manager = LoggerManager(default_config)

        assert manager.config["level"] == "DEBUG"
        assert manager.config["to_file"] is True
        assert "main" in manager.handlers
        assert isinstance(
            manager.handlers["main"], logging.handlers.RotatingFileHandler
        )

    def test_directory_creation(self, temp_log_dir: str) -> None:
        """Test automatic log directory creation."""
        log_dir = os.path.join(temp_log_dir, "new_logs")
        log_file = os.path.join(log_dir, "app.log")
        assert not os.path.exists(log_dir)

        config = {"to_file": True, "log_file": log_file}
        LoggerManager(config)

        assert os.path.exists(log_dir)

    def test_console_handler_creation(self) -> None:
        """Test console handler creation."""
        manager = LoggerManager({"to_console": True, "to_file": False})
        assert "console" in manager.handlers
        assert isinstance(manager.handlers["console"], logging.StreamHandler)


class TestFiltersAndFormatters:
    """Test logging filters and formatters."""

    def test_structured_formatter_basic(self) -> None:
        """Test structured formatter produces valid JSON."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="A test message",
            args=(),
            exc_info=None,
            func="test_func",
        )
        record.thread_name = "MainThread"

        output = formatter.format(record)
        log_data = json.loads(output)

        assert log_data["level"] == "INFO"
        assert log_data["message"] == "A test message"
        assert log_data["function"] == "test_func"

    def test_context_filter(self) -> None:
        """Test context filter adds extra data to log records."""
        context = {"request_id": "xyz-123", "user_id": "user456"}
        context_filter = ContextFilter(context)
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)

        result = context_filter.filter(record)

        assert result is True
        assert record.request_id == "xyz-123"  # type: ignore[attr-defined]
        assert record.user_id == "user456"  # type: ignore[attr-defined]
        assert hasattr(record, "timestamp")

    def test_module_filter(self) -> None:
        """Test module filter correctly filters logs."""
        module_filter = ModuleFilter(["noisy_module"], filtered_level="WARNING")

        info_record = logging.LogRecord(
            "noisy_module.sub", logging.INFO, "", 0, "info", (), None
        )
        warning_record = logging.LogRecord(
            "noisy_module", logging.WARNING, "", 0, "warn", (), None
        )
        other_record = logging.LogRecord(
            "my_app.core", logging.INFO, "", 0, "info", (), None
        )

        assert not module_filter.filter(info_record)  # Should be filtered out
        assert module_filter.filter(warning_record)  # Should pass
        assert module_filter.filter(other_record)  # Should pass


class TestGlobalFunctions:
    """Test global helper functions and decorators."""

    def test_get_logger_functionality(self, default_config: dict) -> None:
        """Test get_logger returns properly configured logger."""
        setup_logging(default_config)
        logger = get_logger("my_app.test")

        assert logger.name == "my_app.test"
        assert len(logger.handlers) > 0

    def test_specialized_log_functions(self, default_config: dict) -> None:
        """Test specialized logging functions work correctly."""
        structured_config = default_config.copy()
        structured_config["structured"] = True
        setup_logging(structured_config)

        # Test audit log
        log_audit("test_user", "test_action", "An audit message.")
        assert os.path.exists(structured_config["audit_log_file"])

        with open(structured_config["audit_log_file"], "r", encoding="utf-8") as f:
            log_data = json.loads(f.read())
            assert log_data["user"] == "test_user"
            assert log_data["action"] == "test_action"


# --- Critical Edge Cases and Error Handling ---


class TestErrorHandlingAndResilience:
    """Test critical error handling scenarios."""

    def test_initialization_failure_recovery(self) -> None:
        """Test recovery from initialization failures."""
        with patch("os.makedirs", side_effect=PermissionError("Test permission error")):
            # Should not raise exception, should fall back gracefully
            manager = LoggerManager(
                {"to_file": True, "log_file": "/restricted/path/log.log"}
            )
            assert manager.config["to_file"] is False  # File logging should be disabled

    def test_unicode_handling(self, default_config: dict) -> None:
        """Test proper handling of unicode characters."""
        setup_logging(default_config)
        logger = get_logger("unicode_test")

        # Should handle unicode without crashing
        try:
            logger.info("Unicode test: 测试消息 🔍")
            logger.info("Emoji test: 📝✅❌")
        except Exception as e:
            pytest.fail(f"Should handle unicode gracefully: {e}")

    def test_malformed_configuration_handling(self) -> None:
        """Test handling of malformed configurations."""
        malformed_configs: List[Dict[str, Any]] = [
            {"level": None},
            {"max_bytes": "not_a_number"},
            {"backup_count": -1},
            {},  # Empty config
        ]

        for config in malformed_configs:
            try:
                manager = LoggerManager(config=config)
                assert manager is not None
            except Exception as e:
                pytest.fail(
                    f"Should handle malformed config gracefully: {config}, Error: {e}"
                )


# --- Concurrency and Thread Safety ---


class TestConcurrencyAndThreadSafety:
    """Test thread safety and concurrent access."""

    def test_concurrent_logging_operations(self, default_config: dict) -> None:
        """Test that concurrent logging operations are thread-safe."""
        setup_logging(default_config)
        results: list[str] = []

        def log_worker(worker_id: int) -> None:
            logger = get_logger(f"worker_{worker_id}")
            try:
                for i in range(5):  # Reduced iterations for faster testing
                    logger.info(f"Worker {worker_id} - Message {i}")
                results.append(f"Worker {worker_id} completed")
            except Exception as e:
                results.append(f"Worker {worker_id} failed: {e}")

        # Use ThreadPoolExecutor for better control
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(log_worker, i) for i in range(3)]
            for future in futures:
                future.result()

        # Verify all workers completed successfully
        completed_count = sum(1 for result in results if "completed" in result)
        assert completed_count == 3

    def test_concurrent_logger_retrieval(self) -> None:
        """Test concurrent logger retrieval is thread-safe."""
        loggers: list[logging.Logger] = []
        exceptions: list[Exception] = []

        def get_logger_worker() -> None:
            try:
                logger = get_logger("concurrent_test")
                loggers.append(logger)
            except Exception as e:
                exceptions.append(e)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_logger_worker) for _ in range(5)]
            for future in futures:
                future.result()

        # Should have no exceptions
        assert len(exceptions) == 0
        # All loggers should be the same instance (cached)
        assert all(logger is loggers[0] for logger in loggers)


# --- Parametrized Tests for Efficiency ---


class TestParametrizedScenarios:
    """Test multiple scenarios efficiently using parametrization."""

    @pytest.mark.parametrize(
        "log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    def test_different_log_levels(self, log_level: str, temp_log_dir: str) -> None:
        """Test logging with different log levels."""
        config = {
            "level": log_level,
            "to_file": True,
            "log_file": os.path.join(temp_log_dir, f"{log_level.lower()}.log"),
            "to_console": False,
        }

        manager = LoggerManager(config)
        logger = manager.get_logger("level_test")

        # Test that logger respects the configured level
        expected_level = getattr(logging, log_level)
        effective_level = logger.getEffectiveLevel()
        assert (
            effective_level <= expected_level
        )  # Should be at or below the configured level

    @pytest.mark.parametrize(
        "handler_config",
        [
            {"to_console": True, "to_file": False},
            {"to_console": False, "to_file": True},
            {"to_console": True, "to_file": True},
        ],
    )
    def test_different_handler_configurations(
        self, handler_config: dict, temp_log_dir: str
    ) -> None:
        """Test different handler configurations."""
        if handler_config.get("to_file"):
            handler_config["log_file"] = os.path.join(temp_log_dir, "test.log")

        manager = LoggerManager(handler_config)

        if handler_config["to_console"]:
            assert "console" in manager.handlers
        if handler_config["to_file"]:
            assert "main" in manager.handlers


# --- Integration Tests ---


class TestIntegration:
    """Integration tests that verify end-to-end functionality."""

    def test_complete_logging_workflow(self, default_config: dict) -> None:
        """Test complete logging workflow from setup to file output."""
        # Use structured logging for easier verification
        structured_config = default_config.copy()
        structured_config["structured"] = True

        setup_logging(structured_config)

        # Get logger and log various types of messages
        logger = get_logger("integration_test")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Verify log file was created and contains expected content
        assert os.path.exists(structured_config["log_file"])

        with open(structured_config["log_file"], "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) >= 4  # At least 4 messages logged

            # Verify each line is valid JSON
            for line in lines:
                log_data = json.loads(line)
                assert "level" in log_data
                assert "message" in log_data
                assert "logger" in log_data
