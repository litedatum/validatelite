"""
Tests for the logging_setup module.

Verifies that logging is correctly configured based on configuration.
"""

import logging
import os
import tempfile
from pathlib import Path

import pytest

from shared.config.logging_config import LoggingConfig
from shared.utils.logging_setup import setup_logging


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_log_level_setting(self) -> None:
        """Test that log level is correctly set."""
        # Create config with DEBUG level
        config = LoggingConfig(level="DEBUG")

        # Setup logging
        setup_logging(config)

        # Check root logger level
        assert logging.getLogger().level == logging.DEBUG

        # Reset logging to default
        logging.basicConfig(level=logging.INFO)

    def test_log_format_setting(self) -> None:
        """Test that log format is correctly set."""
        # Create config with custom format
        custom_format = "%(levelname)s - %(message)s"
        config = LoggingConfig(format=custom_format)

        # Setup logging
        setup_logging(config)

        # Check format of handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                formatter = handler.formatter
                # mypy: formatter may be None or not a logging.Formatter
                if isinstance(formatter, logging.Formatter):
                    assert formatter._fmt == custom_format
                else:
                    assert (
                        False
                    ), "Handler formatter is not a logging.Formatter instance"

        # Reset logging to default
        logging.basicConfig(format="%(message)s")

    def test_file_handler_added(self) -> None:
        """Test that FileHandler is added when to_file=True."""
        # Create temp directory for log file
        temp_dir = tempfile.mkdtemp()
        try:
            log_path = os.path.join(temp_dir, "test.log")

            # Create config with file logging enabled
            config = LoggingConfig(to_file=True, file_path=log_path)

            # Setup logging
            setup_logging(config)

            # Get root logger
            root_logger = logging.getLogger()

            # Check if a RotatingFileHandler was added
            has_file_handler = any(
                hasattr(handler, "baseFilename") and handler.baseFilename == log_path
                for handler in root_logger.handlers
            )

            assert has_file_handler, "RotatingFileHandler not added"

            # Test logging to file
            test_message = "Test log message"
            logging.info(test_message)

            # Check if message was written to file
            with open(log_path, "r") as f:
                log_content = f.read()
                assert test_message in log_content

            # Clean up handlers before removing directory
            for handler in list(root_logger.handlers):
                if (
                    hasattr(handler, "baseFilename")
                    and handler.baseFilename == log_path
                ):
                    handler.close()
                    root_logger.removeHandler(handler)
        finally:
            # Clean up temp directory
            try:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass

    def test_log_directory_creation(self) -> None:
        """Test that log directory is created if it doesn't exist."""
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a nested path that doesn't exist
            nested_dir = os.path.join(temp_dir, "logs", "nested")
            log_path = os.path.join(nested_dir, "test.log")

            # Verify directory doesn't exist yet
            assert not os.path.exists(nested_dir)

            # Create config with file logging enabled
            config = LoggingConfig(to_file=True, file_path=log_path)

            # Setup logging
            setup_logging(config)

            # Verify directory was created
            assert os.path.exists(nested_dir)

            # Clean up handlers
            root_logger = logging.getLogger()
            for handler in list(root_logger.handlers):
                if (
                    hasattr(handler, "baseFilename")
                    and handler.baseFilename == log_path
                ):
                    handler.close()
                    root_logger.removeHandler(handler)

    def test_third_party_loggers_level(self) -> None:
        """Test that third-party loggers have their levels set."""
        # Create config
        config = LoggingConfig(level="DEBUG")

        # Setup logging
        setup_logging(config)

        # Check third-party logger levels
        assert logging.getLogger("pydantic").level == logging.WARNING
        assert logging.getLogger("toml").level == logging.WARNING
        assert logging.getLogger("sqlalchemy").level == logging.WARNING
