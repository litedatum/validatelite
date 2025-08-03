"""
Unit tests for core configuration module.

Tests the CoreConfig model and related functionality.
"""

import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.config import CoreConfig, get_core_config


class TestCoreConfigModel:
    """Tests for the CoreConfig model"""

    def test_default_values(self) -> None:
        """Test that default values are correctly set"""
        config = CoreConfig()

        # Performance & Resource Management
        assert config.execution_timeout == 300
        assert config.table_size_threshold == 10000
        assert config.rule_count_threshold == 2
        assert config.max_rules_per_merge == 10

        # Feature Flags
        assert config.merge_execution_enabled is True
        assert config.monitoring_enabled is False

        # Rule Type Settings
        assert "UNIQUE" in config.independent_rule_types
        assert "CUSTOM_SQL" in config.independent_rule_types
        assert "FOREIGN_KEY" in config.independent_rule_types

        # Sample Data Configuration
        assert config.sample_data_enabled is True
        assert config.sample_data_max_records == 5

        # Backwards compatibility attributes
        assert config.TABLE_SIZE_THRESHOLD == 10000
        assert config.RULE_COUNT_THRESHOLD == 2
        assert config.MAX_RULES_PER_MERGE == 10
        assert config.MAX_CONCURRENT_EXECUTIONS == 8
        assert config.MERGE_EXECUTION_ENABLED is True

    def test_should_enable_merge(self) -> None:
        """Test the should_enable_merge method logic"""
        config = CoreConfig()

        # Both conditions met - should enable merge
        assert config.should_enable_merge(10000, 2) is True
        assert config.should_enable_merge(20000, 5) is True

        # Table size below threshold - should not enable merge
        assert config.should_enable_merge(9999, 2) is False
        assert config.should_enable_merge(5000, 5) is False

        # Rule count below threshold - should not enable merge
        assert config.should_enable_merge(10000, 1) is False
        assert config.should_enable_merge(20000, 1) is False

        # Both conditions below threshold - should not enable merge
        assert config.should_enable_merge(5000, 1) is False

        # Feature flag disabled - should not enable merge regardless of conditions
        config.merge_execution_enabled = False
        assert config.should_enable_merge(10000, 2) is False
        assert config.should_enable_merge(20000, 5) is False

    def test_get_retry_config(self) -> None:
        """Test the get_retry_config method"""
        config = CoreConfig()
        retry_config = config.get_retry_config()

        assert isinstance(retry_config, dict)
        assert retry_config["enabled"] is True
        assert retry_config["max_attempts"] == 3
        assert retry_config["delay"] == 1.0

    def test_get_fallback_config(self) -> None:
        """Test the get_fallback_config method"""
        config = CoreConfig()
        fallback_config = config.get_fallback_config()

        assert isinstance(fallback_config, dict)
        assert fallback_config["enabled"] is True
        assert fallback_config["on_error"] is True
        assert fallback_config["on_timeout"] is True

    def test_validate_config(self) -> None:
        """Test the validate_config method"""
        config = CoreConfig()
        assert config.validate_config() is True

    def test_sample_data_configuration(self) -> None:
        """Test sample data configuration settings"""
        # Test default values
        config = CoreConfig()
        assert config.sample_data_enabled is True
        assert config.sample_data_max_records == 5

        # Test custom values
        config = CoreConfig(sample_data_enabled=False, sample_data_max_records=10)
        assert config.sample_data_enabled is False
        assert config.sample_data_max_records == 10

        # Test edge cases
        config = CoreConfig(sample_data_max_records=1)
        assert config.sample_data_max_records == 1

        config = CoreConfig(sample_data_max_records=100)
        assert config.sample_data_max_records == 100


class TestGetCoreConfig:
    """Tests for the get_core_config function"""

    def test_load_from_default_path(self, monkeypatch: Any) -> None:
        """Test loading config from default path"""

        # Create a mock load_config function
        def mock_load_config(path: str, model_class: Any) -> CoreConfig:
            assert path == "config/core.toml"
            assert model_class == CoreConfig
            return CoreConfig(execution_timeout=500, monitoring_enabled=True)

        # Apply the mock
        monkeypatch.setattr("core.config.load_config", mock_load_config)

        # Load config (should use default path)
        config = get_core_config()

        # Verify loaded values
        assert config.execution_timeout == 500
        assert config.monitoring_enabled is True

    def test_load_from_environment_variable(self, monkeypatch: Any) -> None:
        """Test loading config from environment variable path"""
        # Set environment variable
        monkeypatch.setenv("CORE_CONFIG_PATH", "/custom/path/core.toml")

        # Create a mock load_config function
        def mock_load_config(path: str, model_class: Any) -> CoreConfig:
            assert path == "/custom/path/core.toml"
            assert model_class == CoreConfig
            return CoreConfig(execution_timeout=600, monitoring_enabled=True)

        # Apply the mock
        monkeypatch.setattr("core.config.load_config", mock_load_config)

        # Load config (should use environment variable path)
        config = get_core_config()

        # Verify loaded values
        assert config.execution_timeout == 600
        assert config.monitoring_enabled is True

    def test_file_not_found_returns_default_config(self, monkeypatch: Any) -> None:
        """Test that default config is returned when file is not found"""

        # Mock load_config to raise FileNotFoundError
        def mock_load_config(path: str, model_class: Any) -> CoreConfig:
            raise FileNotFoundError(f"File not found: {path}")

        # Apply the mock
        monkeypatch.setattr("core.config.load_config", mock_load_config)

        # Capture print output
        with patch("builtins.print") as mock_print:
            # Load config (should return default config)
            config = get_core_config()

            # Verify default values
            assert config.execution_timeout == 300
            assert config.table_size_threshold == 10000
            assert config.merge_execution_enabled is True

            # Verify warning was printed
            mock_print.assert_called_once()
            assert "Warning" in mock_print.call_args[0][0]
            assert "config/core.toml" in mock_print.call_args[0][0]

    def test_load_from_real_file(self) -> None:
        """Test loading config from a real file"""
        temp_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".toml", delete=False)
        try:
            temp_file.write(
                """
            # Core Engine Configuration
            execution_timeout = 400
            table_size_threshold = 20000
            rule_count_threshold = 3
            max_rules_per_merge = 5
            merge_execution_enabled = false
            monitoring_enabled = true
            independent_rule_types = ["UNIQUE", "CUSTOM_SQL", "FOREIGN_KEY", "CUSTOM_PYTHON"]
            """
            )
            temp_file.close()  # Close the file before accessing it

            # Set environment variable to point to temp file
            with patch.object(os, "getenv", return_value=temp_file.name):
                # Load config
                config = get_core_config()

                # Verify loaded values
                assert config.execution_timeout == 400
                assert config.table_size_threshold == 20000
                assert config.rule_count_threshold == 3
                assert config.max_rules_per_merge == 5
                assert config.merge_execution_enabled is False
                assert config.monitoring_enabled is True
                assert "CUSTOM_PYTHON" in config.independent_rule_types
        finally:
            # Make sure the file is closed before trying to delete it
            if not temp_file.closed:
                temp_file.close()
            os.unlink(temp_file.name)
