"""
Unit tests for the shared configuration registry module.

Tests the functionality of the configuration registry system.
"""

from unittest.mock import patch

import pytest

from cli.core.config import CliConfig
from core.config import CoreConfig
from shared.config import get_config, get_typed_config, register_config
from shared.config.logging_config import LoggingConfig


class TestConfigRegistry:
    """Tests for the configuration registry functionality"""

    def setup_method(self) -> None:
        """Setup before each test - clear the registry"""
        # Access and clear the private _config_registry
        import shared.config

        shared.config._config_registry = {}

    def test_register_and_get_config(self) -> None:
        """Test registering and retrieving a configuration"""
        # Create a test config
        config = LoggingConfig(level="DEBUG", to_file=True)

        # Register it
        register_config("test_logging", config)

        # Retrieve it
        retrieved = get_config("test_logging")

        # Verify it's the same object
        assert retrieved is config
        assert retrieved.level == "DEBUG"
        assert retrieved.to_file is True

    def test_register_multiple_configs(self) -> None:
        """Test registering and retrieving multiple configurations"""
        # Create test configs
        logging_config = LoggingConfig(level="DEBUG")
        core_config = CoreConfig(execution_timeout=500)
        cli_config = CliConfig(debug_mode=True)

        # Register them
        register_config("logging", logging_config)
        register_config("core", core_config)
        register_config("cli", cli_config)

        # Retrieve them
        retrieved_logging = get_config("logging")
        retrieved_core = get_config("core")
        retrieved_cli = get_config("cli")

        # Verify they're the same objects
        assert retrieved_logging is logging_config
        assert retrieved_core is core_config
        assert retrieved_cli is cli_config

    def test_get_nonexistent_config(self) -> None:
        """Test retrieving a non-existent configuration"""
        # Try to get a config that doesn't exist
        config = get_config("nonexistent")

        # Should return None
        assert config is None

    def test_register_overwrite(self) -> None:
        """Test overwriting a registered configuration"""
        # Create and register a config
        config1 = LoggingConfig(level="INFO")
        register_config("logging", config1)

        # Create and register another config with the same name
        config2 = LoggingConfig(level="DEBUG")
        register_config("logging", config2)

        # Retrieve it
        retrieved = get_config("logging")

        # Should be the second config
        assert retrieved is config2
        assert retrieved.level == "DEBUG"

    def test_get_typed_config_correct_type(self) -> None:
        """Test retrieving a typed configuration with correct type"""
        # Create and register a config
        config = LoggingConfig(level="DEBUG")
        register_config("logging", config)

        # Retrieve it with correct type
        retrieved = get_typed_config("logging", LoggingConfig)

        # Should return the config
        assert retrieved is config
        assert isinstance(retrieved, LoggingConfig)

    def test_get_typed_config_wrong_type(self) -> None:
        """Test retrieving a typed configuration with wrong type"""
        # Create and register a config
        config = LoggingConfig(level="DEBUG")
        register_config("logging", config)

        # Retrieve it with wrong type
        retrieved = get_typed_config("logging", CoreConfig)

        # Should return None
        assert retrieved is None

    def test_get_typed_config_nonexistent(self) -> None:
        """Test retrieving a non-existent typed configuration"""
        # Try to get a typed config that doesn't exist
        config = get_typed_config("nonexistent", LoggingConfig)

        # Should return None
        assert config is None

    def test_registry_isolation(self) -> None:
        """Test that the registry is isolated between tests"""
        # This test relies on setup_method clearing the registry

        # Verify registry is empty
        import shared.config

        assert len(shared.config._config_registry) == 0

        # Register a config
        config = LoggingConfig()
        register_config("test", config)

        # Verify registry has one item
        assert len(shared.config._config_registry) == 1
