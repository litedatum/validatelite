"""
E2E Test Utilities

Utility classes and functions for E2E tests.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class CLITestResult:
    """Result of a CLI command execution."""

    returncode: int
    stdout: str
    stderr: str
    execution_time: float


class E2ETestUtils:
    """Utility class for E2E tests."""

    @staticmethod
    def run_cli_command(command: List[str], timeout: int = 60) -> CLITestResult:
        """
        Runs a CLI command and returns the result.

        Args:
            command: List of command arguments
            timeout: Command timeout in seconds

        Returns:
            CLITestResult with execution details
        """
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        start_time = time.time()
        result = subprocess.run(
            ["python", "cli_main.py"] + command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=timeout,
        )
        end_time = time.time()

        return CLITestResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            execution_time=end_time - start_time,
        )

    @staticmethod
    def assert_rule_result(
        result: CLITestResult, rule_name: str, expected_status: str
    ) -> None:
        """
        Asserts that a specific rule has the expected status.

        Args:
            result: CLI execution result
            rule_name: Name of the rule to check
            expected_status: Expected status ("PASSED" or "FAILED")
        """
        if expected_status == "PASSED":
            assert (
                result.returncode == 0
            ), f"Rule {rule_name} should PASS but got return code {result.returncode}"
            assert (
                rule_name in result.stdout
            ), f"Rule {rule_name} should be mentioned in output"
        else:  # FAILED
            assert (
                result.returncode == 1
            ), f"Rule {rule_name} should FAIL but got return code {result.returncode}"
            assert (
                rule_name in result.stdout
            ), f"Rule {rule_name} should be mentioned in output"

    @staticmethod
    def assert_verbose_output(
        result: CLITestResult, should_have_samples: bool = True
    ) -> None:
        """
        Asserts verbose output contains expected content.

        Args:
            result: CLI execution result
            should_have_samples: Whether sample data should be present
        """
        if should_have_samples:
            assert (
                "Sample data:" in result.stdout
                or "showing first" in result.stdout
                or "Sample failures" in result.stdout
            ), "Verbose output should contain sample data"
        assert "Results:" in result.stdout, "Output should contain results section"

    @staticmethod
    def assert_performance_acceptable(
        result: CLITestResult, max_time: float = 30.0
    ) -> None:
        """
        Asserts that execution time is within acceptable limits.

        Args:
            result: CLI execution result
            max_time: Maximum acceptable execution time in seconds
        """
        assert (
            result.execution_time < max_time
        ), f"Execution took too long: {result.execution_time:.2f} seconds"

    @staticmethod
    def assert_error_handling(
        result: CLITestResult, expected_error_type: str = "error"
    ) -> None:
        """
        Asserts that error handling works correctly.

        Args:
            result: CLI execution result
            expected_error_type: Type of error expected ("error", "connection", "file", etc.)
        """
        assert result.returncode != 0, "Should fail due to error"
        error_output = result.stderr.lower()
        assert (
            expected_error_type in error_output
        ), f"Error output should contain '{expected_error_type}'"

    @staticmethod
    def create_temp_rules_file(rules: List[Dict[str, Any]], temp_dir: Path) -> str:
        """
        Creates a temporary rules file for testing.

        Args:
            rules: List of rule dictionaries
            temp_dir: Temporary directory to create file in

        Returns:
            Path to the created rules file
        """
        rules_data = {"rules": rules}
        rules_file = temp_dir / "temp_rules.json"
        with open(rules_file, "w") as f:
            json.dump(rules_data, f, indent=2)
        return str(rules_file)

    @staticmethod
    def extract_rule_results(result: CLITestResult) -> Dict[str, str]:
        """
        Extracts rule results from CLI output.

        Args:
            result: CLI execution result

        Returns:
            Dictionary mapping rule names to their status
        """
        rule_results = {}
        lines = result.stdout.split("\n")
        in_results_section = False

        for line in lines:
            if "Results:" in line:
                in_results_section = True
                continue

            if in_results_section and line.strip():
                # Parse rule result line
                if "PASSED" in line or "FAILED" in line:
                    # Extract rule name and status
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "PASSED" in part or "FAILED" in part:
                            rule_name = " ".join(parts[:i])
                            status = "PASSED" if "PASSED" in part else "FAILED"
                            rule_results[rule_name] = status
                            break

        return rule_results

    @staticmethod
    def assert_rule_count(result: CLITestResult, expected_count: int) -> None:
        """
        Asserts that the expected number of rules were processed.

        Args:
            result: CLI execution result
            expected_count: Expected number of rules
        """
        rule_results = E2ETestUtils.extract_rule_results(result)
        actual_count = len(rule_results)
        assert (
            actual_count == expected_count
        ), f"Expected {expected_count} rules, got {actual_count}"

    @staticmethod
    def assert_sample_data_present(result: CLITestResult, rule_name: str) -> None:
        """
        Asserts that sample data is present for a specific rule.

        Args:
            result: CLI execution result
            rule_name: Name of the rule to check
        """
        # Check if sample data section exists for the rule
        assert (
            rule_name in result.stdout
        ), f"Rule {rule_name} should be mentioned in output"

        # Look for sample data indicators
        sample_indicators = [
            "Sample failures",
            "showing first",
            "Sample records:",
            "Examples:",
        ]
        has_sample_data = any(
            indicator in result.stdout for indicator in sample_indicators
        )
        assert has_sample_data, f"Sample data should be present for rule {rule_name}"
