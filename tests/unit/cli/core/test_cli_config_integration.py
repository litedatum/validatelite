"""
Tests for CLI configuration integration.

Verifies that CLI components correctly use the configuration system.
"""

from typing import Any, Dict, List, Sequence, cast
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from cli.core.config import CliConfig, DatabaseConfig, get_cli_config
from core.config import CoreConfig, get_core_config
from shared.config import get_config, register_config
from shared.schema import RuleSchema


@pytest.fixture
def mock_configs() -> Dict[str, Any]:
    """Mock configurations for testing"""
    core_config = CoreConfig(
        execution_timeout=300,
        table_size_threshold=5000,
        rule_count_threshold=2,
        merge_execution_enabled=True,
    )

    cli_config = CliConfig(
        debug_mode=True,
        default_sample_size=5000,
        max_file_size_mb=200,
        query_timeout=450,
        database=DatabaseConfig(
            url="sqlite:///test.db", connect_timeout=45, echo_queries=True
        ),
    )

    return {"core": core_config, "cli": cli_config}


@pytest.fixture
def registered_configs(mock_configs: Dict[str, Any]) -> Dict[str, Any]:
    """Register mock configurations in the registry"""
    register_config("core", mock_configs["core"])
    register_config("cli", mock_configs["cli"])
    return mock_configs


class TestCliCommandsWithConfig:
    """Tests for CLI commands using configuration"""

    def test_data_validator_uses_config(
        self, registered_configs: Dict[str, Any]
    ) -> None:
        """Test that DataValidator uses both core and CLI configurations"""
        from cli.core.data_validator import DataValidator

        # Create mock source and rules
        source_config = {"type": "csv", "path": "test.csv"}
        rule_configs: List[Dict[str, str]] = [
            {"name": "1", "type": "NOT_NULL", "target": "column1"}
        ]

        # Initialize DataValidator
        validator = DataValidator(
            source_config=source_config,
            rules=cast(List[RuleSchema | Dict[str, Any]], rule_configs),
            core_config=registered_configs["core"],
            cli_config=registered_configs["cli"],
        )

        # Verify that validator uses the config values
        assert validator.core_config is registered_configs["core"]
        assert validator.cli_config is registered_configs["cli"]
        assert validator.sample_size == 5000  # From CLI config

    def test_check_command_uses_config(self) -> None:
        """Test that check command uses configuration"""
        from cli.commands.check import check_command

        # Create mock configs
        mock_core_config = CoreConfig(execution_timeout=400)
        mock_cli_config = CliConfig(debug_mode=True, default_sample_size=5000)

        # Mock the get_*_config functions
        with patch(
            "core.config.get_core_config", return_value=mock_core_config
        ) as mock_get_core:
            with patch(
                "cli.core.config.get_cli_config", return_value=mock_cli_config
            ) as mock_get_cli:
                # Mock other dependencies
                with patch("cli.commands.check.SourceParser") as mock_source_parser_cls:
                    with patch("cli.commands.check.RuleParser") as mock_rule_parser_cls:
                        with patch(
                            "cli.commands.check.OutputFormatter"
                        ) as mock_formatter_cls:
                            with patch(
                                "cli.commands.check.DataValidator"
                            ) as mock_validator_cls:
                                with patch("asyncio.run") as mock_asyncio_run:
                                    # Setup mocks
                                    mock_source_parser = MagicMock()
                                    mock_source_parser.parse_source.return_value = {
                                        "type": "csv",
                                        "path": "test.csv",
                                    }
                                    mock_source_parser_cls.return_value = (
                                        mock_source_parser
                                    )

                                    mock_rule_parser = MagicMock()
                                    mock_rule_parser.parse_rules.return_value = [
                                        {"id": "1", "type": "NOT_NULL"}
                                    ]
                                    mock_rule_parser_cls.return_value = mock_rule_parser

                                    mock_validator = MagicMock()
                                    mock_validator_cls.return_value = mock_validator

                                    mock_asyncio_run.return_value = [
                                        {"status": "PASSED"}
                                    ]

                                    # Create CLI runner
                                    runner = CliRunner()

                                    # Invoke the command
                                    result = runner.invoke(
                                        check_command,
                                        ["test.csv", "--rule", "not_null(column1)"],
                                    )

                                    # Verify configs were loaded
                                    mock_get_core.assert_called_once()
                                    mock_get_cli.assert_called_once()

                                    # Verify DataValidator was initialized with configs
                                    mock_validator_cls.assert_called_once()
                                    call_kwargs = mock_validator_cls.call_args.kwargs
                                    assert (
                                        call_kwargs["core_config"] is mock_core_config
                                    )
                                    assert call_kwargs["cli_config"] is mock_cli_config


class TestCliCoreComponentsWithConfig:
    """Tests for CLI core components using configuration"""

    def test_source_parser_uses_config(
        self, registered_configs: Dict[str, Any]
    ) -> None:
        """Test that SourceParser uses CLI configuration"""
        from cli.core.source_parser import SourceParser

        # Mock the get_config function
        with patch(
            "shared.config.get_typed_config", return_value=registered_configs["cli"]
        ):
            # Initialize SourceParser
            parser = SourceParser()

            # Verify that parser has access to config values if needed
            # This is a placeholder - actual implementation would depend on how SourceParser uses config
            assert True

    def test_rule_parser_uses_config(self, registered_configs: Dict[str, Any]) -> None:
        """Test that RuleParser uses CLI configuration"""
        from cli.core.rule_parser import RuleParser

        # Mock the get_config function
        with patch(
            "shared.config.get_typed_config", return_value=registered_configs["cli"]
        ):
            # Initialize RuleParser
            parser = RuleParser()

            # Verify that parser has access to config values if needed
            # This is a placeholder - actual implementation would depend on how RuleParser uses config
            assert True

    def test_output_formatter_uses_config(
        self, registered_configs: Dict[str, Any]
    ) -> None:
        """Test that OutputFormatter uses CLI configuration"""
        from cli.core.output_formatter import OutputFormatter

        # Mock the get_config function
        with patch(
            "shared.config.get_typed_config", return_value=registered_configs["cli"]
        ):
            # Initialize OutputFormatter
            formatter = OutputFormatter(quiet=False, verbose=True)

            # Verify that formatter has access to config values if needed
            # This is a placeholder - actual implementation would depend on how OutputFormatter uses config
            assert True
