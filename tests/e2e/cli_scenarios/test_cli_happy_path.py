"""
E2E CLI Scenarios - Happy Path

This module tests the CLI's happy path scenarios, ensuring that the `check` command
works as expected with valid inputs and a real database connection.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
from pytest import TempPathFactory

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


@pytest.fixture(scope="module")
def sample_data_file(tmp_path_factory: TempPathFactory) -> str:
    """Creates a temporary CSV file with sample data for the tests."""
    data = "id,name,email,age,gender,created_at\n"
    data += "1,Alice,alice@example.com,30,1,2023-01-01\n"
    data += "2,,thirty-five,0,2023-01-02\n"
    data += "3,Charlie,charlie#invalid.com,25,1,2023-01-03\n"
    data += "4,David,david@example.com,-5,3,2023-01-04\n"
    data += "5,Eve,eve@example.com,150,0,2023-01-05\n"
    data_file = tmp_path_factory.mktemp("data") / "sample-data.csv"
    data_file.write_text(data)
    return str(data_file)


class TestCliHappyPath:
    """
    Tests the happy path for the CLI `check` command.
    """

    def test_cli_check_command_success_inline_rules(
        self, sample_data_file: str
    ) -> None:
        """
        Tests that the `check` command runs successfully with inline rules.
        """
        # Arrange
        command = [
            "check",
            "--conn",
            sample_data_file,
            "--table",
            "sample-data",
            "--rule",
            "not_null(name)",
            "--rule",
            "length(email,5,100)",
        ]

        # Act
        result = run_cli_command(command)

        # Assert
        assert (
            result.returncode == 1
        ), f"CLI command should exit with 1 on validation failure: {result.stderr}"
        assert "Results:" in result.stdout
        assert "not_null(name)" in result.stdout
        assert "length(email)" in result.stdout

    def test_cli_check_command_success_rules_file(
        self, sample_data_file: str, tmp_path: Path
    ) -> None:
        """
        Tests that the `check` command runs successfully with a rules file.
        """
        # Arrange
        rules = {
            "version": "1.0",
            "rules": [
                {"type": "not_null", "column": "name"},
                {"type": "length", "column": "email", "min": 5, "max": 100},
            ],
        }
        rules_file = tmp_path / "rules.json"
        with open(rules_file, "w") as f:
            json.dump(rules, f)

        command = [
            "check",
            "--conn",
            sample_data_file,
            "--table",
            "sample-data",
            "--rules",
            str(rules_file),
        ]

        # Act
        result = run_cli_command(command)

        # Assert
        assert (
            result.returncode == 1
        ), f"CLI command should exit with 1 on validation failure: {result.stderr}"
        assert "Results:" in result.stdout
        assert "not_null(name)" in result.stdout
        assert "length(email)" in result.stdout
