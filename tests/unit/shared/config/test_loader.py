"""
Unit tests for the shared configuration loader module.

Tests the functionality of the configuration loader utility.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from shared.config.loader import load_config
from shared.config.logging_config import LoggingConfig
from shared.exceptions.exception_system import OperationError


class TestConfigLoader:
    """Tests for the configuration loader utility"""

    def test_load_valid_toml_file(self) -> None:
        """Test loading a valid TOML configuration file"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".toml", delete=False) as f:
            f.write(
                """
            # Test configuration
            level = "DEBUG"
            format = "%(levelname)s: %(message)s"
            to_file = true
            file_path = "logs/test.log"
            max_bytes = 5242880
            backup_count = 3
            """
            )
            f.flush()

            try:
                # Load the configuration
                config = load_config(f.name, LoggingConfig)

                # Verify loaded values
                assert isinstance(config, LoggingConfig)
                assert config.level == "DEBUG"
                assert config.format == "%(levelname)s: %(message)s"
                assert config.to_file is True
                assert config.file_path == "logs/test.log"
                assert config.max_bytes == 5242880
                assert config.backup_count == 3
            finally:
                os.unlink(f.name)

    def test_file_not_found(self) -> None:
        """Test handling of non-existent configuration file"""
        non_existent_path = "/path/to/non/existent/config.toml"

        with pytest.raises(OperationError) as excinfo:
            load_config(non_existent_path, LoggingConfig)

        assert "not found" in str(excinfo.value)
        assert non_existent_path in str(excinfo.value)

    def test_toml_decode_error(self) -> None:
        """Test handling of TOML decode errors"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".toml", delete=False) as f:
            f.write(
                """
            # Invalid TOML syntax
            level = "DEBUG"
            format = "%(levelname)s: %(message)s
            """
            )  # Missing closing quote
            f.flush()

            try:
                with pytest.raises(Exception) as excinfo:
                    load_config(f.name, LoggingConfig)

                assert "Error decoding TOML" in str(excinfo.value)
            finally:
                os.unlink(f.name)

    def test_validation_error(self) -> None:
        """Test handling of Pydantic validation errors"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".toml", delete=False) as f:
            f.write(
                """
            # Invalid value type
            level = 123  # Should be a string
            format = "%(levelname)s: %(message)s"
            to_file = true
            """
            )
            f.flush()

            try:
                with pytest.raises(Exception) as excinfo:
                    load_config(f.name, LoggingConfig)

                assert "Error validating configuration" in str(excinfo.value)
            finally:
                os.unlink(f.name)

    def test_empty_config_file(self) -> None:
        """Test loading an empty configuration file"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".toml", delete=False) as f:
            # Write an empty file
            f.write("")
            f.flush()

            try:
                # Should load with default values
                config = load_config(f.name, LoggingConfig)

                # Verify default values
                assert isinstance(config, LoggingConfig)
                assert config.level == "INFO"
                assert config.to_file is False
                assert config.file_path == "logs/app.log"
            finally:
                os.unlink(f.name)

    def test_partial_config_file(self) -> None:
        """Test loading a partial configuration file"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".toml", delete=False) as f:
            f.write(
                """
            # Partial configuration
            level = "DEBUG"
            to_file = true
            """
            )
            f.flush()

            try:
                # Should load with specified values and defaults for the rest
                config = load_config(f.name, LoggingConfig)

                # Verify loaded and default values
                assert isinstance(config, LoggingConfig)
                assert config.level == "DEBUG"  # Specified
                assert config.to_file is True  # Specified
                assert (
                    config.format
                    == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )  # Default
                assert config.file_path == "logs/app.log"  # Default
            finally:
                os.unlink(f.name)

    def test_extra_fields_in_config(self) -> None:
        """Test handling of extra fields in configuration file"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".toml", delete=False) as f:
            f.write(
                """
            # Configuration with extra fields
            level = "DEBUG"
            format = "%(levelname)s: %(message)s"
            to_file = true
            file_path = "logs/test.log"

            # Extra field not in the model
            extra_field = "This should be ignored"
            """
            )
            f.flush()

            try:
                # Should load without error, ignoring extra fields
                config = load_config(f.name, LoggingConfig)

                # Verify loaded values
                assert isinstance(config, LoggingConfig)
                assert config.level == "DEBUG"
                assert config.format == "%(levelname)s: %(message)s"
                assert config.to_file is True
                assert config.file_path == "logs/test.log"

                # Verify extra field was ignored
                with pytest.raises(AttributeError):
                    getattr(config, "extra_field")
            finally:
                os.unlink(f.name)
