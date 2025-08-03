"""
Unit tests for logging configuration module.

Tests the LoggingConfig model and related functionality.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from shared.config.logging_config import LoggingConfig, get_logging_config


class TestLoggingConfigModel:
    """Tests for the LoggingConfig model"""

    def test_default_values(self) -> None:
        """Test that default values are correctly set"""
        config = LoggingConfig()

        assert config.level == "INFO"
        assert config.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert config.to_file is False
        assert config.file_path == "logs/app.log"
        assert config.max_bytes == 10485760  # 10MB
        assert config.backup_count == 5

    def test_custom_values(self) -> None:
        """Test setting custom values"""
        config = LoggingConfig(
            level="DEBUG",
            format="%(levelname)s: %(message)s",
            to_file=True,
            file_path="custom/path/app.log",
            max_bytes=5242880,  # 5MB
            backup_count=3,
        )

        assert config.level == "DEBUG"
        assert config.format == "%(levelname)s: %(message)s"
        assert config.to_file is True
        assert config.file_path == "custom/path/app.log"
        assert config.max_bytes == 5242880
        assert config.backup_count == 3


class TestGetLoggingConfig:
    """Tests for the get_logging_config function"""

    def test_load_from_default_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading config from default path"""

        # Create a mock load_config function
        def mock_load_config(
            path: str, model_class: type[LoggingConfig]
        ) -> LoggingConfig:
            assert path == "config/logging.toml"
            assert model_class == LoggingConfig
            return LoggingConfig(level="DEBUG", to_file=True)

        # Apply the mock
        monkeypatch.setattr(
            "shared.config.logging_config.load_config", mock_load_config
        )

        # Load config (should use default path)
        config = get_logging_config()

        # Verify loaded values
        assert config.level == "DEBUG"
        assert config.to_file is True

    def test_load_from_environment_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test loading config from environment variable path"""
        # Set environment variable
        monkeypatch.setenv("LOGGING_CONFIG_PATH", "/custom/path/logging.toml")

        # Create a mock load_config function
        def mock_load_config(
            path: str, model_class: type[LoggingConfig]
        ) -> LoggingConfig:
            assert path == "/custom/path/logging.toml"
            assert model_class == LoggingConfig
            return LoggingConfig(level="WARNING", backup_count=10)

        # Apply the mock
        monkeypatch.setattr(
            "shared.config.logging_config.load_config", mock_load_config
        )

        # Load config (should use environment variable path)
        config = get_logging_config()

        # Verify loaded values
        assert config.level == "WARNING"
        assert config.backup_count == 10

    def test_file_not_found_returns_default_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that default config is returned when file is not found"""

        # Mock load_config to raise FileNotFoundError
        def mock_load_config(path: str, model_class: type[LoggingConfig]) -> None:
            raise FileNotFoundError(f"File not found: {path}")

        # Apply the mock
        monkeypatch.setattr(
            "shared.config.logging_config.load_config", mock_load_config
        )

        # Capture print output
        with patch("builtins.print") as mock_print:
            # Load config (should return default config)
            config = get_logging_config()

            # Verify default values
            assert config.level == "INFO"
            assert config.to_file is False
            assert config.file_path == "logs/app.log"

            # Verify warning was printed
            mock_print.assert_called_once()
            assert "Warning" in mock_print.call_args[0][0]
            assert "config/logging.toml" in mock_print.call_args[0][0]

    def test_load_from_real_file(self) -> None:
        """Test loading config from a real file"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".toml", delete=False) as f:
            f.write(
                """
            # Logging Configuration
            level = "DEBUG"
            format = "%(levelname)s: %(message)s"
            to_file = true
            file_path = "custom/path/app.log"
            max_bytes = 5242880
            backup_count = 3
            """
            )
            f.flush()

            try:
                # Set environment variable to point to temp file
                with patch.object(os, "getenv", return_value=f.name):
                    # Load config
                    config = get_logging_config()

                    # Verify loaded values
                    assert config.level == "DEBUG"
                    assert config.format == "%(levelname)s: %(message)s"
                    assert config.to_file is True
                    assert config.file_path == "custom/path/app.log"
                    assert config.max_bytes == 5242880
                    assert config.backup_count == 3
            finally:
                os.unlink(f.name)
