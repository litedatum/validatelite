"""
🧙‍♂️ Check Command New Interface Tests

Tests for the new --conn and --table options in the check command.
This file focuses on testing the new interface functionality.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest
from click.testing import CliRunner

from cli.commands.check import check_command
from tests.shared.builders.test_builders import TestDataBuilder
from tests.shared.contracts.test_contracts import MockContract


class TestCheckCommandNewInterface:
    """Test suite for the new --conn and --table interface"""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """CLI test runner"""
        return CliRunner()

    @pytest.fixture
    def mock_components(self) -> Dict[str, Any]:
        """Mock components using Contract Testing"""
        return {
            "config_manager": MockContract.create_config_manager_mock(),
            "source_parser": MockContract.create_source_parser_mock(),
            "rule_parser": MockContract.create_rule_parser_mock(),
            "data_validator": MockContract.create_data_validator_mock(),
            "output_formatter": MockContract.create_output_formatter_mock(),
        }

    @pytest.fixture
    def sample_csv_data(self) -> str:
        """CSV test data"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,name,email,age\n")
            f.write("1,John,john@test.com,25\n")
            f.write("2,Jane,jane@test.com,30\n")
            temp_file = f.name
        return temp_file

    @pytest.fixture
    def sample_rules_file(self) -> str:
        """Sample rules file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "rules": [
                        {"field": "id", "type": "integer", "required": True},
                        {"field": "name", "type": "string", "required": True},
                    ]
                },
                f,
            )
            temp_file = f.name
        return temp_file

    # === NEW INTERFACE TESTS ===

    @patch("cli.commands.check.get_cli_config")
    @patch("cli.commands.check.get_core_config")
    @patch("cli.commands.check.SourceParser")
    @patch("cli.commands.check.RuleParser")
    @patch("cli.commands.check.DataValidator")
    @patch("cli.commands.check.OutputFormatter")
    def test_new_interface_with_conn_and_table(
        self,
        mock_formatter: Mock,
        mock_validator: Mock,
        mock_rule_parser: Mock,
        mock_source_parser: Mock,
        mock_core_config: Mock,
        mock_cli_config: Mock,
        runner: CliRunner,
        sample_csv_data: str,
        sample_rules_file: str,
        mock_components: Dict[str, Any],
    ):
        """Test the new --conn and --table interface"""
        # Setup mocks using the same pattern as successful tests
        mock_cli_config.return_value = Mock()
        mock_core_config.return_value = Mock()

        # Source parsing mock
        source_connection = Mock()
        mock_source_parser.return_value.parse_source.return_value = source_connection

        # Rule parsing mock
        rules = [Mock()]  # Create a mock rule
        mock_rule_parser.return_value.parse_rules.return_value = rules

        # Validation results mock
        validation_results = [Mock()]
        mock_validator_instance = AsyncMock()
        mock_validator_instance.validate.return_value = validation_results
        mock_validator.return_value = mock_validator_instance

        # Formatter mock
        mock_formatter.return_value = Mock()

        # Execute command with new interface
        result = runner.invoke(
            check_command,
            [
                "--conn",
                sample_csv_data,
                "--table",
                "users",
                "--rules",
                sample_rules_file,
            ],
        )

        # Verify success
        assert result.exit_code == 0
        assert "Starting validation" in result.output

    @patch("cli.commands.check.get_cli_config")
    @patch("cli.commands.check.get_core_config")
    @patch("cli.commands.check.SourceParser")
    @patch("cli.commands.check.RuleParser")
    @patch("cli.commands.check.DataValidator")
    @patch("cli.commands.check.OutputFormatter")
    def test_new_interface_missing_table(
        self,
        mock_formatter: Mock,
        mock_validator: Mock,
        mock_rule_parser: Mock,
        mock_source_parser: Mock,
        mock_core_config: Mock,
        mock_cli_config: Mock,
        runner: CliRunner,
        sample_csv_data: str,
    ):
        """Test that --table is required when using --conn"""
        # Execute command with --conn but no --table
        result = runner.invoke(check_command, ["--conn", sample_csv_data])

        # Verify error
        assert result.exit_code == 2  # Click error exit code
        assert "Missing option '--table'" in result.output

    @patch("cli.commands.check.get_cli_config")
    @patch("cli.commands.check.get_core_config")
    @patch("cli.commands.check.SourceParser")
    @patch("cli.commands.check.RuleParser")
    @patch("cli.commands.check.DataValidator")
    @patch("cli.commands.check.OutputFormatter")
    def test_new_interface_missing_conn(
        self,
        mock_formatter: Mock,
        mock_validator: Mock,
        mock_rule_parser: Mock,
        mock_source_parser: Mock,
        mock_core_config: Mock,
        mock_cli_config: Mock,
        runner: CliRunner,
    ):
        """Test that --conn is required when using --table"""
        # Execute command with --table but no --conn
        result = runner.invoke(check_command, ["--table", "users"])

        # Verify error
        assert result.exit_code == 2  # Click error exit code
        assert "Missing option '--conn'" in result.output

    @patch("cli.commands.check.get_cli_config")
    @patch("cli.commands.check.get_core_config")
    @patch("cli.commands.check.SourceParser")
    @patch("cli.commands.check.RuleParser")
    @patch("cli.commands.check.DataValidator")
    @patch("cli.commands.check.OutputFormatter")
    def test_new_interface_with_inline_rules(
        self,
        mock_formatter: Mock,
        mock_validator: Mock,
        mock_rule_parser: Mock,
        mock_source_parser: Mock,
        mock_core_config: Mock,
        mock_cli_config: Mock,
        runner: CliRunner,
        sample_csv_data: str,
        mock_components: Dict[str, Any],
    ):
        """Test new interface with inline rules"""
        # Setup mocks using the same pattern as successful tests
        mock_cli_config.return_value = Mock()
        mock_core_config.return_value = Mock()

        # Source parsing mock
        source_connection = Mock()
        mock_source_parser.return_value.parse_source.return_value = source_connection

        # Rule parsing mock
        rules = [Mock()]  # Create a mock rule
        mock_rule_parser.return_value.parse_rules.return_value = rules

        # Validation results mock
        validation_results = [Mock()]
        mock_validator_instance = AsyncMock()
        mock_validator_instance.validate.return_value = validation_results
        mock_validator.return_value = mock_validator_instance

        # Formatter mock
        mock_formatter.return_value = Mock()

        # Execute command with new interface and inline rules
        result = runner.invoke(
            check_command,
            [
                "--conn",
                sample_csv_data,
                "--table",
                "users",
                "--rule",
                "not_null(id)",
                "--rule",
                "length(name, 2, 50)",
            ],
        )

        # Verify success
        assert result.exit_code == 0
        assert "Starting validation" in result.output

    @patch("cli.commands.check.get_cli_config")
    @patch("cli.commands.check.get_core_config")
    @patch("cli.commands.check.SourceParser")
    @patch("cli.commands.check.RuleParser")
    @patch("cli.commands.check.DataValidator")
    @patch("cli.commands.check.OutputFormatter")
    def test_new_interface_with_database_connection(
        self,
        mock_formatter: Mock,
        mock_validator: Mock,
        mock_rule_parser: Mock,
        mock_source_parser: Mock,
        mock_core_config: Mock,
        mock_cli_config: Mock,
        runner: CliRunner,
        sample_rules_file: str,
        mock_components: Dict[str, Any],
    ):
        """Test new interface with database connection"""
        # Setup mocks using the same pattern as successful tests
        mock_cli_config.return_value = Mock()
        mock_core_config.return_value = Mock()

        # Source parsing mock
        source_connection = Mock()
        mock_source_parser.return_value.parse_source.return_value = source_connection

        # Rule parsing mock
        rules = [Mock()]  # Create a mock rule
        mock_rule_parser.return_value.parse_rules.return_value = rules

        # Validation results mock
        validation_results = [Mock()]
        mock_validator_instance = AsyncMock()
        mock_validator_instance.validate.return_value = validation_results
        mock_validator.return_value = mock_validator_instance

        # Formatter mock
        mock_formatter.return_value = Mock()

        # Execute command with database connection
        result = runner.invoke(
            check_command,
            [
                "--conn",
                "mysql://user:pass@host/db",
                "--table",
                "customers",
                "--rules",
                sample_rules_file,
            ],
        )

        # Verify success
        assert result.exit_code == 0
        assert "Starting validation" in result.output

    @patch("cli.commands.check.get_cli_config")
    @patch("cli.commands.check.get_core_config")
    @patch("cli.commands.check.SourceParser")
    @patch("cli.commands.check.RuleParser")
    @patch("cli.commands.check.DataValidator")
    @patch("cli.commands.check.OutputFormatter")
    def test_new_interface_with_sqlite_file(
        self,
        mock_formatter: Mock,
        mock_validator: Mock,
        mock_rule_parser: Mock,
        mock_source_parser: Mock,
        mock_core_config: Mock,
        mock_cli_config: Mock,
        runner: CliRunner,
        sample_rules_file: str,
        mock_components: Dict[str, Any],
    ):
        """Test new interface with SQLite file"""
        # Setup mocks using the same pattern as successful tests
        mock_cli_config.return_value = Mock()
        mock_core_config.return_value = Mock()

        # Source parsing mock
        source_connection = Mock()
        mock_source_parser.return_value.parse_source.return_value = source_connection

        # Rule parsing mock
        rules = [Mock()]  # Create a mock rule
        mock_rule_parser.return_value.parse_rules.return_value = rules

        # Validation results mock
        validation_results = [Mock()]
        mock_validator_instance = AsyncMock()
        mock_validator_instance.validate.return_value = validation_results
        mock_validator.return_value = mock_validator_instance

        # Formatter mock
        mock_formatter.return_value = Mock()

        # Execute command with SQLite file
        result = runner.invoke(
            check_command,
            [
                "--conn",
                "sqlite:///path/to/database.db",
                "--table",
                "orders",
                "--rules",
                sample_rules_file,
            ],
        )

        # Verify success
        assert result.exit_code == 0
        assert "Starting validation" in result.output

    # === ERROR HANDLING TESTS ===

    @patch("cli.commands.check.get_cli_config")
    @patch("cli.commands.check.get_core_config")
    @patch("cli.commands.check.SourceParser")
    @patch("cli.commands.check.RuleParser")
    @patch("cli.commands.check.DataValidator")
    @patch("cli.commands.check.OutputFormatter")
    def test_new_interface_no_rules_specified(
        self,
        mock_formatter: Mock,
        mock_validator: Mock,
        mock_rule_parser: Mock,
        mock_source_parser: Mock,
        mock_core_config: Mock,
        mock_cli_config: Mock,
        runner: CliRunner,
        sample_csv_data: str,
        mock_components: Dict[str, Any],
    ):
        """Test error when no rules are specified"""
        # Execute command without rules
        result = runner.invoke(
            check_command, ["--conn", sample_csv_data, "--table", "users"]
        )

        # Verify error
        assert result.exit_code == 2  # Click error exit code
        assert "No rules specified" in result.output

    @patch("cli.commands.check.get_cli_config")
    @patch("cli.commands.check.get_core_config")
    @patch("cli.commands.check.SourceParser")
    @patch("cli.commands.check.RuleParser")
    @patch("cli.commands.check.DataValidator")
    @patch("cli.commands.check.OutputFormatter")
    def test_new_interface_empty_file(
        self,
        mock_formatter: Mock,
        mock_validator: Mock,
        mock_rule_parser: Mock,
        mock_source_parser: Mock,
        mock_core_config: Mock,
        mock_cli_config: Mock,
        runner: CliRunner,
        sample_rules_file: str,
        mock_components: Dict[str, Any],
    ):
        """Test error when source file is empty"""
        # Create empty file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_file = f.name

        # Execute command with empty file
        result = runner.invoke(
            check_command,
            ["--conn", temp_file, "--table", "users", "--rules", sample_rules_file],
        )

        # Verify error
        assert result.exit_code > 0  # Any non-zero exit code indicates error
        assert "is empty" in result.output

        # Cleanup
        Path(temp_file).unlink(missing_ok=True)

    def test_table_name_parameter_passed_to_source_parser(
        self,
        runner: CliRunner,
    ):
        """Test that table_name parameter is correctly passed to SourceParser.parse_source"""
        with patch("cli.commands.check.SourceParser") as mock_source_parser_class:
            # Setup mock
            mock_source_parser = Mock()
            mock_source_parser_class.return_value = mock_source_parser

            # Create mock source config
            mock_source_config = Mock()
            mock_source_parser.parse_source.return_value = mock_source_config

            # Mock other components
            with patch("cli.commands.check.RuleParser") as mock_rule_parser_class:
                with patch("cli.commands.check.DataValidator") as mock_validator_class:
                    with patch(
                        "cli.commands.check.OutputFormatter"
                    ) as mock_formatter_class:
                        with patch(
                            "cli.commands.check.get_cli_config"
                        ) as mock_cli_config:
                            with patch(
                                "cli.commands.check.get_core_config"
                            ) as mock_core_config:
                                with patch("asyncio.run") as mock_asyncio_run:
                                    # Setup mocks
                                    mock_cli_config.return_value = Mock()
                                    mock_core_config.return_value = Mock()

                                    # Create mock rule
                                    mock_rule = Mock()
                                    mock_rule_parser_class.return_value.parse_rules.return_value = [
                                        mock_rule
                                    ]

                                    # Create mock validation result
                                    mock_result = Mock()
                                    mock_validator_instance = Mock()
                                    mock_validator_instance.validate.return_value = [
                                        mock_result
                                    ]
                                    mock_validator_class.return_value = (
                                        mock_validator_instance
                                    )

                                    # Create mock formatter
                                    mock_formatter = Mock()
                                    mock_formatter_class.return_value = mock_formatter

                                    # Mock asyncio.run
                                    mock_asyncio_run.return_value = [mock_result]

                                    # Run the command
                                    result = runner.invoke(
                                        check_command,
                                        [
                                            "--conn",
                                            "test.csv",
                                            "--table",
                                            "customers",
                                            "--rule",
                                            "not_null(id)",
                                        ],
                                    )

                                    # Verify that parse_source was called with both connection_string and table_name
                                    mock_source_parser.parse_source.assert_called_once_with(
                                        "test.csv", "customers"
                                    )

                                    # Verify success
                                    assert result.exit_code == 0

    def test_table_name_parameter_with_database_connection(
        self,
        runner: CliRunner,
    ):
        """Test that table_name parameter is correctly passed when using database connection"""
        with patch("cli.commands.check.SourceParser") as mock_source_parser_class:
            # Setup mock
            mock_source_parser = Mock()
            mock_source_parser_class.return_value = mock_source_parser

            # Create mock source config
            mock_source_config = Mock()
            mock_source_parser.parse_source.return_value = mock_source_config

            # Mock other components
            with patch("cli.commands.check.RuleParser") as mock_rule_parser_class:
                with patch("cli.commands.check.DataValidator") as mock_validator_class:
                    with patch(
                        "cli.commands.check.OutputFormatter"
                    ) as mock_formatter_class:
                        with patch(
                            "cli.commands.check.get_cli_config"
                        ) as mock_cli_config:
                            with patch(
                                "cli.commands.check.get_core_config"
                            ) as mock_core_config:
                                with patch("asyncio.run") as mock_asyncio_run:
                                    # Setup mocks
                                    mock_cli_config.return_value = Mock()
                                    mock_core_config.return_value = Mock()

                                    # Create mock rule
                                    mock_rule = Mock()
                                    mock_rule_parser_class.return_value.parse_rules.return_value = [
                                        mock_rule
                                    ]

                                    # Create mock validation result
                                    mock_result = Mock()
                                    mock_validator_instance = Mock()
                                    mock_validator_instance.validate.return_value = [
                                        mock_result
                                    ]
                                    mock_validator_class.return_value = (
                                        mock_validator_instance
                                    )

                                    # Create mock formatter
                                    mock_formatter = Mock()
                                    mock_formatter_class.return_value = mock_formatter

                                    # Mock asyncio.run
                                    mock_asyncio_run.return_value = [mock_result]

                                    # Run the command with database connection
                                    db_url = "postgresql://user:pass@host/db"
                                    table_name = "customers"

                                    result = runner.invoke(
                                        check_command,
                                        [
                                            "--conn",
                                            db_url,
                                            "--table",
                                            table_name,
                                            "--rule",
                                            "not_null(id)",
                                        ],
                                    )

                                    # Verify that parse_source was called with both db_url and table_name
                                    mock_source_parser.parse_source.assert_called_once_with(
                                        db_url, table_name
                                    )

                                    # Verify success
                                    assert result.exit_code == 0

    def test_table_name_parameter_overrides_url_table(
        self,
        runner: CliRunner,
    ):
        """Test that --table parameter overrides table name from URL when both are present"""
        with patch("cli.commands.check.SourceParser") as mock_source_parser_class:
            # Setup mock
            mock_source_parser = Mock()
            mock_source_parser_class.return_value = mock_source_parser

            # Create mock source config
            mock_source_config = Mock()
            mock_source_parser.parse_source.return_value = mock_source_config

            # Mock other components
            with patch("cli.commands.check.RuleParser") as mock_rule_parser_class:
                with patch("cli.commands.check.DataValidator") as mock_validator_class:
                    with patch(
                        "cli.commands.check.OutputFormatter"
                    ) as mock_formatter_class:
                        with patch(
                            "cli.commands.check.get_cli_config"
                        ) as mock_cli_config:
                            with patch(
                                "cli.commands.check.get_core_config"
                            ) as mock_core_config:
                                with patch("asyncio.run") as mock_asyncio_run:
                                    # Setup mocks
                                    mock_cli_config.return_value = Mock()
                                    mock_core_config.return_value = Mock()

                                    # Create mock rule
                                    mock_rule = Mock()
                                    mock_rule_parser_class.return_value.parse_rules.return_value = [
                                        mock_rule
                                    ]

                                    # Create mock validation result
                                    mock_result = Mock()
                                    mock_validator_instance = Mock()
                                    mock_validator_instance.validate.return_value = [
                                        mock_result
                                    ]
                                    mock_validator_class.return_value = (
                                        mock_validator_instance
                                    )

                                    # Create mock formatter
                                    mock_formatter = Mock()
                                    mock_formatter_class.return_value = mock_formatter

                                    # Mock asyncio.run
                                    mock_asyncio_run.return_value = [mock_result]

                                    # Run the command with URL that already contains table name
                                    # URL has "users" table, but we specify "customers" table
                                    db_url_with_table = (
                                        "postgresql://user:pass@host/db.users"
                                    )
                                    override_table_name = "customers"

                                    result = runner.invoke(
                                        check_command,
                                        [
                                            "--conn",
                                            db_url_with_table,
                                            "--table",
                                            override_table_name,
                                            "--rule",
                                            "not_null(id)",
                                        ],
                                    )

                                    # Verify that parse_source was called with URL and override table name
                                    # The --table parameter should take precedence over URL table
                                    mock_source_parser.parse_source.assert_called_once_with(
                                        db_url_with_table, override_table_name
                                    )

                                    # Verify success
                                    assert result.exit_code == 0
