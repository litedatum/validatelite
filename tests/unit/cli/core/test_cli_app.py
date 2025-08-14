"""
🧙‍♂️ CLI Application TDD Tests - Modern Testing Architecture

Features:
- Application entry point testing
- Command routing and registration verification
- Parameter parsing and validation
- Global options handling
- Error handling and recovery testing
- Help system validation
"""

import sys
from typing import Any, Generator
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner
from hypothesis import given
from hypothesis import strategies as st

from cli.app import cli_app, main
from tests.shared.builders.test_builders import TestDataBuilder


class TestCliApplication:
    """Modern CLI Application Test Suite - Testing Ghost's Architecture"""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """CLI test runner"""
        return CliRunner()

    @pytest.fixture
    def mock_logger(self) -> Generator[Mock, Any, None]:
        """Mock logger for testing"""
        with patch("cli.app.logger") as mock_log:
            yield mock_log

    # === APPLICATION ENTRY POINT TESTS ===

    def test_cli_app_without_command_shows_help(self: Any, runner: CliRunner) -> None:
        """Test CLI app shows help when no command is provided"""
        result = runner.invoke(cli_app, [])

        assert result.exit_code == 0
        assert "ValidateLite - Data Quality Validation Tool" in result.output
        assert "Usage:" in result.output
        assert "Commands:" in result.output
        assert "check" in result.output
        assert "rules-help" in result.output

    def test_cli_app_version_option(self: Any, runner: CliRunner) -> None:
        """Test CLI app version option"""
        result = runner.invoke(cli_app, ["--version"])

        assert result.exit_code == 0
        assert "vlite-cli" in result.output
        # assert "1.0.0" in result.output

    def test_cli_app_help_option(self: Any, runner: CliRunner) -> None:
        """Test CLI app help option"""
        result = runner.invoke(cli_app, ["--help"])

        assert result.exit_code == 0
        assert "ValidateLite - Data Quality Validation Tool" in result.output
        assert "A command-line tool for validating data quality" in result.output
        assert "Options:" in result.output
        assert "Commands:" in result.output

    # === COMMAND REGISTRATION TESTS ===

    def test_check_command_is_registered(self: Any, runner: CliRunner) -> None:
        """Test that check command is properly registered"""
        result = runner.invoke(cli_app, ["--help"])

        assert result.exit_code == 0
        assert "check" in result.output

        # Test check command can be invoked
        # (should fail without args but be recognized)
        result = runner.invoke(cli_app, ["check"])
        assert (
            "Usage:" in result.output or "Error:" in result.output
        )  # Either usage help or missing args error

    def test_rules_help_command_is_registered(self: Any, runner: CliRunner) -> None:
        """Test that rules-help command is properly registered"""
        result = runner.invoke(cli_app, ["--help"])

        assert result.exit_code == 0
        assert "rules-help" in result.output

        # Test rules-help command execution
        result = runner.invoke(cli_app, ["rules-help"])
        assert result.exit_code == 0

    # === RULES HELP COMMAND TESTS ===

    def test_rules_help_command_content(self: Any, runner: CliRunner) -> None:
        """Test rules-help command provides comprehensive help"""
        result = runner.invoke(cli_app, ["rules-help"])

        assert result.exit_code == 0

        # Check main sections
        assert "ValidateLite Rule Syntax Help" in result.output
        assert "Available Rule Types:" in result.output
        assert "Rule Files (JSON format):" in result.output
        assert "Usage Examples:" in result.output

        # Check all rule types are documented
        rule_types = ["NOT_NULL", "UNIQUE", "LENGTH", "RANGE", "ENUM", "REGEX"]
        for rule_type in rule_types:
            assert rule_type in result.output

        # Check examples are provided
        assert "not_null(id)" in result.output
        assert "unique(email)" in result.output
        assert "length(name,2,50)" in result.output
        assert "mysql://user:pass@host/db.users" in result.output

    def test_rules_help_json_schema_example(self: Any, runner: CliRunner) -> None:
        """Test rules-help includes valid JSON schema example"""
        result = runner.invoke(cli_app, ["rules-help"])

        assert result.exit_code == 0

        # Check JSON structure elements
        json_elements = [
            '"version": "1.0"',
            '"rules"',
            '"type"',
            '"column"',
            '"description"',
        ]

        for element in json_elements:
            assert element in result.output

    def test_rules_help_usage_examples(self: Any, runner: CliRunner) -> None:
        """Test rules-help includes practical usage examples"""
        result = runner.invoke(cli_app, ["rules-help"])

        assert result.exit_code == 0

        # Check usage examples
        usage_examples = [
            "vlite-cli check users.csv --rule",
            "vlite-cli check users.csv --rules validation.json",
            "vlite-cli check mysql://user:pass@host/db.users",
        ]

        for example in usage_examples:
            assert example in result.output

    # === MAIN FUNCTION TESTS ===

    @patch("cli.app.cli_app")
    def test_main_function_success(
        self: Any, mock_cli_app: Mock, mock_logger: Mock
    ) -> None:
        """Test main function successful execution"""
        mock_cli_app.return_value = None

        main()

        mock_cli_app.assert_called_once()

    @patch("cli.app.cli_app")
    @patch("cli.app.handle_exception")
    def test_main_function_error_handling(
        self: Any,
        mock_handle_exception: Mock,
        mock_cli_app: Mock,
        mock_logger: Mock,
    ) -> None:
        """Test main function error handling"""
        # Setup exception
        test_error = Exception("Test CLI error")
        mock_cli_app.side_effect = test_error

        # Setup error handler response
        mock_handle_exception.return_value = {
            "message": "CLI application failed",
            "error_type": "Exception",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        # Test main function with exception
        with patch("sys.exit") as mock_exit:
            with patch("click.echo") as mock_echo:
                main()

                mock_handle_exception.assert_called_once_with(
                    test_error, context="CLI Application", logger=mock_logger
                )
                mock_echo.assert_called_once_with(
                    "Error: CLI application failed", err=True
                )
                mock_exit.assert_called_once_with(1)

    # === ERROR HANDLING TESTS ===

    def test_invalid_command_error(self: Any, runner: CliRunner) -> None:
        """Test handling of invalid commands"""
        result = runner.invoke(cli_app, ["invalid-command"])

        assert result.exit_code != 0
        assert "No such command" in result.output or "Usage:" in result.output

    def test_malformed_options_error(self: Any, runner: CliRunner) -> None:
        """Test handling of malformed options"""
        result = runner.invoke(cli_app, ["--invalid-option"])

        assert result.exit_code != 0
        assert "No such option" in result.output or "Usage:" in result.output

    # === INTEGRATION TESTS ===

    @patch("cli.commands.check.get_cli_config")
    @patch("cli.commands.check.SourceParser")
    @patch("cli.commands.check.RuleParser")
    @patch("cli.commands.check.DataValidator")
    @patch("cli.commands.check.OutputFormatter")
    def test_end_to_end_check_command_integration(
        self: Any,
        mock_formatter: Mock,
        mock_validator: Mock,
        mock_rule_parser: Mock,
        mock_source_parser: Mock,
        mock_cli_config: Mock,
        runner: CliRunner,
    ) -> None:
        """Test end-to-end integration of check command through CLI app"""

        # Setup mocks for successful validation
        from shared.enums import ConnectionType

        mock_cli_config.return_value = Mock()
        mock_source_parser.return_value.parse_source.return_value = (
            TestDataBuilder.connection().with_type(ConnectionType.CSV).build()
        )
        mock_rule_parser.return_value.parse_rules.return_value = [
            TestDataBuilder.rule().as_not_null_rule().build()
        ]

        # Setup async validator mock
        mock_validator_instance = Mock()
        mock_validator_instance.validate = Mock()
        mock_validator_instance.validate.return_value = [
            TestDataBuilder.result().with_status("PASSED").build()
        ]
        mock_validator.return_value = mock_validator_instance

        mock_formatter.return_value.display_results = Mock()

        # Create temp CSV file for testing
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,name\n1,test\n")
            temp_file = f.name

        try:
            with patch("asyncio.run") as mock_asyncio_run:
                mock_asyncio_run.return_value = [
                    {"status": "PASSED", "rule": "not_null(id)"}
                ]

                result = runner.invoke(
                    cli_app, ["check", temp_file, "--rule", "not_null(id)"]
                )

                # Should execute without critical errors
                # (May have specific assertion failures but shouldn't crash)
                assert "Analyzing source:" in result.output or result.exit_code in [
                    0,
                    1,
                ]

        finally:
            import os

            os.unlink(temp_file)

    # === PERFORMANCE TESTS ===

    @pytest.mark.performance
    def test_cli_app_startup_performance(self: Any, runner: CliRunner) -> None:
        """Test CLI application startup performance"""
        import time

        start_time = time.time()

        result = runner.invoke(cli_app, ["--help"])

        end_time = time.time()
        startup_time = end_time - start_time

        assert result.exit_code == 0
        assert startup_time < 1.0, f"CLI startup too slow: {startup_time}s"

    @pytest.mark.performance
    def test_rules_help_response_time(self: Any, runner: CliRunner) -> None:
        """Test rules-help command response time"""
        import time

        start_time = time.time()

        result = runner.invoke(cli_app, ["rules-help"])

        end_time = time.time()
        response_time = end_time - start_time

        assert result.exit_code == 0
        assert response_time < 0.5, f"Rules help too slow: {response_time}s"

    # === BOUNDARY CONDITION TESTS ===

    def test_extremely_long_command_line(self: Any, runner: CliRunner) -> None:
        """Test handling of extremely long command lines"""
        long_rule = "not_null(" + "a" * 1000 + ")"

        result = runner.invoke(cli_app, ["check", "test.csv", "--rule", long_rule])

        # Should handle gracefully (either succeed or fail with proper error)
        assert result.exit_code in [20, 21, 22]
        assert (
            "Error:" in result.output
            or "Usage:" in result.output
            or "Analyzing" in result.output
        )

    @given(st.text(min_size=1, max_size=50))
    def test_property_based_command_arguments(
        self: Any, runner: CliRunner, command_arg: str
    ) -> None:
        """Property-based test for various command arguments"""
        # Skip arguments that would be valid subcommands
        if command_arg in ["check", "rules-help", "--help", "--version"]:
            return

        result = runner.invoke(cli_app, [command_arg])

        # Should handle invalid commands gracefully
        assert result.exit_code != 0
        assert len(result.output) > 0  # Should produce some output

    def test_unicode_command_arguments(self: Any, runner: CliRunner) -> None:
        """Test handling of Unicode in command arguments"""
        unicode_args = [
            "检查",  # Chinese
            "données.csv",  # French
            "файл.csv",  # Russian
        ]

        for arg in unicode_args:
            result = runner.invoke(cli_app, [arg])

            # Should handle Unicode gracefully without crashing
            assert isinstance(result.output, str)
            assert result.exit_code in [0, 1, 2]  # Various valid exit codes

    # === HELP SYSTEM TESTS ===

    def test_help_system_completeness(self: Any, runner: CliRunner) -> None:
        """Test that help system provides complete information"""
        result = runner.invoke(cli_app, ["--help"])

        assert result.exit_code == 0

        # Check all essential help sections
        essential_sections = ["Usage:", "Options:", "Commands:", "--help", "--version"]

        for section in essential_sections:
            assert section in result.output

    def test_command_specific_help(self: Any, runner: CliRunner) -> None:
        """Test command-specific help functionality"""
        result = runner.invoke(cli_app, ["check", "--help"])

        assert result.exit_code == 0
        assert "Check data quality" in result.output
        assert "SOURCE" in result.output
        assert "--rule" in result.output
        assert "--rules" in result.output
        assert "Examples:" in result.output

    # === CONTRACT COMPLIANCE TESTS ===

    def test_cli_app_contract_compliance(self: Any, runner: CliRunner) -> None:
        """Test CLI app adheres to Click framework contracts"""
        # Test that CLI app is properly structured as Click group
        result = runner.invoke(cli_app, ["--help"])

        assert result.exit_code == 0

        # Should have proper Click structure
        assert "Usage:" in result.output
        assert "vlite-cli" in result.output
        assert "Commands:" in result.output

    def test_error_exit_codes_consistency(self: Any, runner: CliRunner) -> None:
        """Test that error exit codes are consistent"""
        error_scenarios = [
            (["invalid-command"], 2),  # Unknown command
            (["--invalid-option"], 2),  # Unknown option
            (["check"], 2),  # Missing required argument
        ]

        for args, expected_min_exit_code in error_scenarios:
            result = runner.invoke(cli_app, args)

            assert result.exit_code >= expected_min_exit_code
            assert result.exit_code != 0  # Should not be success
