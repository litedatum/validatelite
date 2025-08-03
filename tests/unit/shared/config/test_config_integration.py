"""
Configuration Management System Integration Tests

These tests focus on the integration between different configuration modules:
1. Configuration loading and validation across modules
2. Configuration registry functionality
3. Cross-module configuration interactions
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from cli.core.config import CliConfig, DatabaseConfig, get_cli_config
from core.config import CoreConfig, get_core_config
from shared.config import get_config, get_typed_config, register_config
from shared.config.loader import load_config
from shared.config.logging_config import LoggingConfig, get_logging_config
from shared.exceptions.exception_system import OperationError


class TestConfigLoader:
    """Tests for the configuration loader"""

    def test_load_config_with_valid_file(self) -> None:
        """Test loading a valid configuration file"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".toml", delete=False) as f:
            f.write(
                """
            level = "INFO"
            format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            to_file = false
            file_path = "logs/app.log"
            max_bytes = 10485760
            backup_count = 5
            """
            )
            f.flush()

            try:
                config = load_config(f.name, LoggingConfig)
                assert isinstance(config, LoggingConfig)
                assert config.level == "INFO"
                assert config.to_file == False
            finally:
                os.unlink(f.name)

    def test_load_config_file_not_found(self) -> None:
        """Test handling of non-existent configuration file"""
        with pytest.raises(OperationError):
            load_config("/path/to/nonexistent/file.toml", LoggingConfig)

    @patch("builtins.open")
    def test_load_config_toml_decode_error(self, mock_open: MagicMock) -> None:
        """Test handling of TOML decode errors"""
        mock_open.side_effect = Exception("TOML decode error")

        with pytest.raises(Exception, match="Error.*TOML"):
            load_config("mock_file.toml", LoggingConfig)


class TestConfigRegistry:
    """Tests for the configuration registry"""

    def test_register_and_get_config(self) -> None:
        """Test registering and retrieving configuration"""
        config = LoggingConfig()
        register_config("test_logging", config)

        retrieved = get_config("test_logging")
        assert retrieved is config

    def test_get_nonexistent_config(self) -> None:
        """Test retrieving non-existent configuration"""
        config = get_config("nonexistent")
        assert config is None

    def test_get_typed_config(self) -> None:
        """Test retrieving typed configuration"""
        config = LoggingConfig()
        register_config("typed_logging", config)

        retrieved = get_typed_config("typed_logging", LoggingConfig)
        assert retrieved is config
        assert isinstance(retrieved, LoggingConfig)

    def test_get_typed_config_wrong_type(self) -> None:
        """Test retrieving configuration with wrong type"""
        config = LoggingConfig()
        register_config("wrong_type", config)

        retrieved = get_typed_config("wrong_type", CoreConfig)
        assert retrieved is None


class TestConfigIntegration:
    """Tests for configuration integration across modules"""

    def test_config_registry_integration(self) -> None:
        """Test integration of all configuration modules with the registry"""
        # Register all configurations
        core_config = CoreConfig()
        cli_config = CliConfig()
        logging_config = LoggingConfig()

        register_config("core", core_config)
        register_config("cli", cli_config)
        register_config("logging", logging_config)

        # Verify all configurations can be retrieved
        assert get_config("core") is core_config
        assert get_config("cli") is cli_config
        assert get_config("logging") is logging_config

        # Verify typed retrieval
        assert get_typed_config("core", CoreConfig) is core_config
        assert get_typed_config("cli", CliConfig) is cli_config
        assert get_typed_config("logging", LoggingConfig) is logging_config

    def test_cross_module_config_access(self) -> None:
        """Test accessing configuration from different modules"""
        # Set up configurations
        core_config = CoreConfig(execution_timeout=500, monitoring_enabled=True)
        cli_config = CliConfig(debug_mode=True, default_sample_size=5000)
        logging_config = LoggingConfig(level="DEBUG", to_file=True)

        register_config("core", core_config)
        register_config("cli", cli_config)
        register_config("logging", logging_config)

        # Simulate access from different modules
        def core_module_function() -> bool:
            # Core module accessing its own config
            core = get_typed_config("core", CoreConfig)
            assert core is not None
            assert core.execution_timeout == 500

            # Core module accessing logging config
            logging = get_typed_config("logging", LoggingConfig)
            assert logging is not None
            assert logging.level == "DEBUG"

            return True

        def cli_module_function() -> bool:
            # CLI module accessing its own config
            cli = get_typed_config("cli", CliConfig)
            assert cli is not None
            assert cli.debug_mode is True

            # CLI module accessing core config
            core = get_typed_config("core", CoreConfig)
            assert core is not None
            assert core.monitoring_enabled is True

            return True

        assert core_module_function()
        assert cli_module_function()

    def test_application_bootstrap_simulation(self) -> None:
        """Test simulating application bootstrap with configuration loading"""
        # Create mock configs
        mock_core_config = CoreConfig(execution_timeout=400)
        mock_cli_config = CliConfig(debug_mode=True)
        mock_logging_config = LoggingConfig(level="DEBUG")

        # Mock the get_*_config functions
        with patch(
            "core.config.get_core_config", return_value=mock_core_config
        ) as mock_get_core:
            with patch(
                "cli.core.config.get_cli_config", return_value=mock_cli_config
            ) as mock_get_cli:
                with patch(
                    "shared.config.logging_config.get_logging_config",
                    return_value=mock_logging_config,
                ) as mock_get_logging:
                    # Simulate bootstrap
                    def bootstrap() -> bool:
                        # 1. Load all configurations
                        logging_config = get_logging_config()
                        core_config = get_core_config()
                        cli_config = get_cli_config()

                        # 2. Register configurations in the registry
                        register_config("logging", logging_config)
                        register_config("core", core_config)
                        register_config("cli", cli_config)

                        # 3. Return success
                        return True

                    # Run bootstrap
                    assert bootstrap() is True

                    # Verify all get_*_config functions were called
                    mock_get_core.assert_called_once()
                    mock_get_cli.assert_called_once()
                    mock_get_logging.assert_called_once()

                    # Verify configs were registered
                    assert get_config("core") is mock_core_config
                    assert get_config("cli") is mock_cli_config
                    assert get_config("logging") is mock_logging_config

    def test_config_interaction_between_modules(self) -> None:
        """Test configuration interaction between different modules"""
        # Create configurations
        core_config = CoreConfig(
            execution_timeout=300,
            table_size_threshold=10000,
            rule_count_threshold=2,
            merge_execution_enabled=True,
        )

        cli_config = CliConfig(
            debug_mode=True, default_sample_size=5000, max_file_size_mb=100
        )

        # Register configurations
        register_config("core", core_config)
        register_config("cli", cli_config)

        # Simulate rule engine using both configurations
        def rule_engine_execution() -> bool:
            # Get configurations
            core = get_typed_config("core", CoreConfig)
            cli = get_typed_config("cli", CliConfig)
            assert core is not None
            assert cli is not None

            # Use configuration values from both modules
            table_size = cli.default_sample_size
            should_merge = core.should_enable_merge(table_size, 3)

            # With default sample size of 5000 and threshold of 10000, should not merge
            assert should_merge is False

            # With larger sample, should merge
            should_merge = core.should_enable_merge(15000, 3)
            assert should_merge is True

            return True

        assert rule_engine_execution() is True
