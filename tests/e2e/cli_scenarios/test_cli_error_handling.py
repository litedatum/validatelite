"""
E2E CLI Scenarios - Error Handling

This module tests the CLI's error handling scenarios, ensuring that the `check` command
-fails gracefully with invalid inputs and provides informative error messages.
"""

import os
import subprocess
from pathlib import Path

import pytest

# Mark as E2E test
pytestmark = pytest.mark.e2e


def run_cli_command(command: list[str]) -> subprocess.CompletedProcess:
    """Runs a CLI command and returns the result."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        ["python", "cli_main.py"] + command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )


class TestCliErrorHandling:
    """
    Tests error handling for the CLI `check` command.
    """

    def test_cli_check_command_no_rules(self, tmp_path: Path) -> None:
        """
        Tests that the `check` command fails when no rules are provided.
        """
        # Arrange
        sample_data_file = tmp_path / "sample-data.csv"
        sample_data_file.write_text("id,name\n1,Alice")
        command = ["check", "--conn", str(sample_data_file), "--table", "sample-data"]

        # Act
        result = run_cli_command(command)

        # Assert
        assert result.returncode != 0, "CLI command should fail without rules"
        assert "No rules specified" in result.stderr

    def test_cli_check_command_invalid_rule(self, tmp_path: Path) -> None:
        """
        Tests that the `check` command fails with an invalid rule.
        """
        # Arrange
        sample_data_file = tmp_path / "sample-data.csv"
        sample_data_file.write_text("id,name\n1,Alice")
        command = [
            "check",
            "--conn",
            str(sample_data_file),
            "--table",
            "sample-data",
            "--rule",
            "invalid_rule(name)",
        ]

        # Act
        result = run_cli_command(command)

        # Assert
        assert result.returncode != 0, "CLI command should fail with an invalid rule"
        assert "Invalid rule syntax" in result.stderr

    def test_cli_check_command_nonexistent_file(self) -> None:
        """
        Tests that the `check` command fails with a nonexistent source file.
        """
        # Arrange
        command = [
            "check",
            "--conn",
            "nonexistent.csv",
            "--table",
            "nonexistent",
            "--rule",
            "not_null(name)",
        ]

        # Act
        result = run_cli_command(command)

        # Assert
        assert result.returncode != 0, "CLI command should fail with a nonexistent file"
        assert "File not found" in result.stderr

    def test_cli_check_command_empty_file(self, tmp_path: Path) -> None:
        """
        Tests that the `check` command fails with an empty source file.
        """
        # Arrange
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")
        command = [
            "check",
            "--conn",
            str(empty_file),
            "--table",
            "empty",
            "--rule",
            "not_null(name)",
        ]

        # Act
        result = run_cli_command(command)

        # Assert
        assert result.returncode != 0, "CLI command should fail with an empty file"
        assert "is empty" in result.stderr
