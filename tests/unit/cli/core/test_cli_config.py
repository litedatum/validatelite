"""
Unit tests for CLI configuration module.

Tests the CliConfig model and related functionality.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from cli.core.config import CliConfig, DatabaseConfig, get_cli_config


class TestDatabaseConfigModel:
    """Tests for the DatabaseConfig model"""

    def test_default_values(self) -> None:
        """Test that default values are correctly set"""
        config = DatabaseConfig()

        assert config.url is None
        assert config.connect_timeout == 30
        assert config.echo_queries is False

    def test_custom_values(self) -> None:
        """Test setting custom values"""
        config = DatabaseConfig(
            url="sqlite:///test.db", connect_timeout=60, echo_queries=True
        )

        assert config.url == "sqlite:///test.db"
        assert config.connect_timeout == 60
        assert config.echo_queries is True


class TestCliConfigModel:
    """Tests for the CliConfig model"""

    def test_default_values(self) -> None:
        """Test that default values are correctly set"""
        config = CliConfig()

        # General
        assert config.debug_mode is False

        # Data Source
        assert config.default_sample_size == 10000
        assert config.max_file_size_mb == 100

        # Database Connection
        assert isinstance(config.database, DatabaseConfig)
        assert config.query_timeout == 300

    def test_custom_values(self) -> None:
        """Test setting custom values"""
        db_config = DatabaseConfig(
            url="sqlite:///custom.db", connect_timeout=45, echo_queries=True
        )

        config = CliConfig(
            debug_mode=True,
            default_sample_size=5000,
            max_file_size_mb=200,
            database=db_config,
            query_timeout=600,
        )

        # General
        assert config.debug_mode is True

        # Data Source
        assert config.default_sample_size == 5000
        assert config.max_file_size_mb == 200

        # Database Connection
        assert config.database is db_config
        assert config.database.url == "sqlite:///custom.db"
        assert config.database.connect_timeout == 45
        assert config.database.echo_queries is True
        assert config.query_timeout == 600


class TestGetCliConfig:
    """Tests for the get_cli_config function"""

    def test_load_from_default_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading config from default path"""

        # Create a mock load_config function
        def mock_load_config(path: str, model_class: type) -> CliConfig:
            assert path == "config/cli.toml"
            assert model_class == CliConfig
            return CliConfig(debug_mode=True, default_sample_size=5000)

        # Apply the mock
        monkeypatch.setattr("cli.core.config.load_config", mock_load_config)

        # Load config (should use default path)
        config = get_cli_config()

        # Verify loaded values
        assert config.debug_mode is True
        assert config.default_sample_size == 5000

    def test_load_from_environment_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test loading config from environment variable path"""
        # Set environment variable
        monkeypatch.setenv("CLI_CONFIG_PATH", "/custom/path/cli.toml")

        # Create a mock load_config function
        def mock_load_config(path: str, model_class: type) -> CliConfig:
            assert path == "/custom/path/cli.toml"
            assert model_class == CliConfig
            return CliConfig(debug_mode=True, max_file_size_mb=500)

        # Apply the mock
        monkeypatch.setattr("cli.core.config.load_config", mock_load_config)

        # Load config (should use environment variable path)
        config = get_cli_config()

        # Verify loaded values
        assert config.debug_mode is True
        assert config.max_file_size_mb == 500

    def test_file_not_found_returns_default_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that default config is returned when file is not found"""

        # Mock load_config to raise FileNotFoundError
        def mock_load_config(path: str, model_class: type) -> CliConfig:
            raise FileNotFoundError(f"File not found: {path}")

        # Apply the mock
        monkeypatch.setattr("cli.core.config.load_config", mock_load_config)

        # Capture print output
        with patch("builtins.print") as mock_print:
            # Load config (should return default config)
            config = get_cli_config()

            # Verify default values
            assert config.debug_mode is False
            assert config.default_sample_size == 10000
            assert config.max_file_size_mb == 100

            # Verify warning was printed
            mock_print.assert_called_once()
            assert "Warning" in mock_print.call_args[0][0]
            assert "config/cli.toml" in mock_print.call_args[0][0]

    def test_load_from_real_file(self) -> None:
        """Test loading config from a real file"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".toml", delete=False) as f:
            f.write(
                """
            # CLI Application Configuration
            debug_mode = true
            default_sample_size = 5000
            max_file_size_mb = 200
            query_timeout = 450

            [database]
            url = "sqlite:///test.db"
            connect_timeout = 45
            echo_queries = true
            """
            )
            f.flush()

            try:
                # Set environment variable to point to temp file
                with patch.object(os, "getenv", return_value=f.name):
                    # Load config
                    config = get_cli_config()

                    # Verify loaded values
                    assert config.debug_mode is True
                    assert config.default_sample_size == 5000
                    assert config.max_file_size_mb == 200
                    assert config.query_timeout == 450
                    assert config.database.url == "sqlite:///test.db"
                    assert config.database.connect_timeout == 45
                    assert config.database.echo_queries is True
            finally:
                os.unlink(f.name)
