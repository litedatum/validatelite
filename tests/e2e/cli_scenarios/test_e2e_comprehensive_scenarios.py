"""
E2E Comprehensive Test Scenarios

This module implements comprehensive E2E tests based on the test case specifications.
Tests cover all three data sources (SQLite, MySQL, PostgreSQL) and all validation scenarios.
"""

import pytest

from tests.shared.utils.database_utils import (
    get_mysql_test_url,
    get_postgresql_test_url,
)
from tests.shared.utils.e2e_test_utils import E2ETestUtils

# Mark as E2E test
pytestmark = pytest.mark.e2e


class TestE2EComprehensiveScenarios:
    """
    Comprehensive E2E tests covering all scenarios from the test case specifications.
    """

    # Test data sources
    SQLITE_DATA_SOURCE = "test_data/customers.xlsx"
    MYSQL_DATA_SOURCE = get_mysql_test_url() + ".customers"
    POSTGRES_DATA_SOURCE = get_postgresql_test_url() + ".customers"

    @pytest.mark.parametrize(
        "data_source", [SQLITE_DATA_SOURCE, MYSQL_DATA_SOURCE, POSTGRES_DATA_SOURCE]
    )
    def test_not_null_name_rule(self, data_source: str) -> None:
        """
        Test: check *data_source* --rule="not_null(name)"
        Expected: PASSED
        """
        command = ["check", data_source, "--rule", "not_null(name)"]
        result = E2ETestUtils.run_cli_command(command)

        E2ETestUtils.assert_rule_result(result, "not_null(name)", "PASSED")
        E2ETestUtils.assert_performance_acceptable(result)

    @pytest.mark.parametrize(
        "data_source", [SQLITE_DATA_SOURCE, MYSQL_DATA_SOURCE, POSTGRES_DATA_SOURCE]
    )
    def test_not_null_email_rule(self, data_source: str) -> None:
        """
        Test: check *data_source* --rule="not_null(email)"
        Expected: FAILED
        """
        command = ["check", data_source, "--rule", "not_null(email)"]
        result = E2ETestUtils.run_cli_command(command)

        E2ETestUtils.assert_rule_result(result, "not_null(email)", "FAILED")
        E2ETestUtils.assert_performance_acceptable(result)

    @pytest.mark.parametrize(
        "data_source", [SQLITE_DATA_SOURCE, MYSQL_DATA_SOURCE, POSTGRES_DATA_SOURCE]
    )
    def test_unique_id_rule(self, data_source: str) -> None:
        """
        Test: check *data_source* --rule="unique(id)"
        Expected: PASSED
        """
        command = ["check", data_source, "--rule", "unique(id)"]
        result = E2ETestUtils.run_cli_command(command)

        E2ETestUtils.assert_rule_result(result, "unique(id)", "PASSED")
        E2ETestUtils.assert_performance_acceptable(result)

    @pytest.mark.parametrize(
        "data_source", [SQLITE_DATA_SOURCE, MYSQL_DATA_SOURCE, POSTGRES_DATA_SOURCE]
    )
    def test_unique_name_rule_verbose(self, data_source: str) -> None:
        """
        Test: check *data_source* --rule="unique(name)" --verbose
        Expected: FAILED with sample data
        """
        command = ["check", data_source, "--rule", "unique(name)", "--verbose"]
        result = E2ETestUtils.run_cli_command(command)

        E2ETestUtils.assert_rule_result(result, "unique(name)", "FAILED")
        E2ETestUtils.assert_verbose_output(result, should_have_samples=True)
        E2ETestUtils.assert_sample_data_present(result, "unique(name)")

    @pytest.mark.parametrize(
        "data_source", [SQLITE_DATA_SOURCE, MYSQL_DATA_SOURCE, POSTGRES_DATA_SOURCE]
    )
    def test_range_age_rule_verbose(self, data_source: str) -> None:
        """
        Test: check *data_source* --rule="range(age,0,120)" --verbose
        Expected: FAILED with sample data
        """
        command = ["check", data_source, "--rule", "range(age,0,120)", "--verbose"]
        result = E2ETestUtils.run_cli_command(command)

        E2ETestUtils.assert_rule_result(result, "range(age)", "FAILED")
        E2ETestUtils.assert_verbose_output(result, should_have_samples=True)
        E2ETestUtils.assert_sample_data_present(result, "range(age)")

    @pytest.mark.parametrize(
        "data_source", [SQLITE_DATA_SOURCE, MYSQL_DATA_SOURCE, POSTGRES_DATA_SOURCE]
    )
    def test_multiple_rules_verbose(self, data_source: str) -> None:
        """
        Test: check *data_source* --rule="length(name,1,30)" --rule="enum(gender,0,1)" --verbose
        Expected: PASSED + FAILED, failed rules return sample data
        """
        command = [
            "check",
            data_source,
            "--rule",
            "length(name,1,30)",
            "--rule",
            "enum(gender,0,1)",
            "--verbose",
        ]
        result = E2ETestUtils.run_cli_command(command)

        # Should have mixed results
        assert result.returncode == 1, "Should fail due to at least one rule failure"
        assert "length(name)" in result.stdout
        assert "enum(gender)" in result.stdout
        E2ETestUtils.assert_verbose_output(result, should_have_samples=True)
        E2ETestUtils.assert_rule_count(result, 2)

    @pytest.mark.parametrize(
        "data_source", [SQLITE_DATA_SOURCE, MYSQL_DATA_SOURCE, POSTGRES_DATA_SOURCE]
    )
    def test_regex_email_rule_verbose(self, data_source: str) -> None:
        """
        Test: check *data_source* --rule="regex(email,'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')" --verbose
        Expected: FAILED with sample data
        """
        command = [
            "check",
            data_source,
            "--rule",
            "regex(email,'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')",
            "--verbose",
        ]
        result = E2ETestUtils.run_cli_command(command)

        E2ETestUtils.assert_rule_result(result, "regex(email", "FAILED")
        E2ETestUtils.assert_verbose_output(result, should_have_samples=True)
        E2ETestUtils.assert_sample_data_present(result, "regex(email")

    @pytest.mark.parametrize(
        "data_source", [SQLITE_DATA_SOURCE, MYSQL_DATA_SOURCE, POSTGRES_DATA_SOURCE]
    )
    def test_validate_merge_rules_file(self, data_source: str) -> None:
        """
        Test: check *data_source* --rules="test_data/validate_merge.json" --verbose
        Expected: 2 rules PASSED, 5 rules FAILED with sample data
        """
        command = [
            "check",
            data_source,
            "--rules",
            "test_data/validate_merge.json",
            "--verbose",
        ]
        result = E2ETestUtils.run_cli_command(command)

        # Should fail due to multiple rule failures
        assert result.returncode == 1, "Should fail due to rule failures"

        # Check that all rules are mentioned
        expected_rules = [
            "unique(email)",
            "range(age)",
            "not_null(email)",
            "regex(email)",
            "not_null(name)",
            "length(name)",
            "enum(gender)",
        ]
        for rule in expected_rules:
            assert rule in result.stdout, f"Rule {rule} should be mentioned in output"

        E2ETestUtils.assert_verbose_output(result, should_have_samples=True)
        E2ETestUtils.assert_rule_count(result, 7)

    @pytest.mark.parametrize(
        "data_source", [SQLITE_DATA_SOURCE, MYSQL_DATA_SOURCE, POSTGRES_DATA_SOURCE]
    )
    def test_validate_invi_rules_file(self, data_source: str) -> None:
        """
        Test: check *data_source* --rules="test_data/validate_invi.json" --verbose
        Expected: Both rules FAILED with sample data
        """
        command = [
            "check",
            data_source,
            "--rules",
            "test_data/validate_invi.json",
            "--verbose",
        ]
        result = E2ETestUtils.run_cli_command(command)

        # Should fail due to rule failures
        assert result.returncode == 1, "Should fail due to rule failures"

        # Check that both rules are mentioned
        expected_rules = ["unique(email)", "regex(email)"]
        for rule in expected_rules:
            assert rule in result.stdout, f"Rule {rule} should be mentioned in output"

        E2ETestUtils.assert_verbose_output(result, should_have_samples=True)
        E2ETestUtils.assert_rule_count(result, 2)

    def test_connection_timeout_handling(self) -> None:
        """
        Test connection timeout handling for database sources.
        """

        # Test with invalid connection parameters
        # Create a completely invalid MySQL connection string that doesn't depend on environment variables
        invalid_source = (
            "mysql://invalid-user:invalid-pass@invalid-host:3306/invalid-db.customers"
        )
        command = ["check", invalid_source, "--rule", "not_null(name)"]
        result = E2ETestUtils.run_cli_command(command)

        E2ETestUtils.assert_error_handling(result, "connection")

    @pytest.mark.parametrize(
        "data_source", [SQLITE_DATA_SOURCE, MYSQL_DATA_SOURCE, POSTGRES_DATA_SOURCE]
    )
    def test_invalid_rule_syntax(self, data_source: str) -> None:
        """
        Test handling of invalid rule syntax.
        """
        command = ["check", data_source, "--rule", "invalid_rule_type(column)"]
        result = E2ETestUtils.run_cli_command(command)

        E2ETestUtils.assert_error_handling(result, "invalid")

    @pytest.mark.parametrize(
        "data_source", [SQLITE_DATA_SOURCE, MYSQL_DATA_SOURCE, POSTGRES_DATA_SOURCE]
    )
    def test_missing_data_source(self, data_source: str) -> None:
        """
        Test handling of missing data source.
        """
        command = ["check", "nonexistent_file.csv", "--rule", "not_null(name)"]
        result = E2ETestUtils.run_cli_command(command)

        E2ETestUtils.assert_error_handling(result, "file")

    @pytest.mark.parametrize(
        "data_source", [SQLITE_DATA_SOURCE, MYSQL_DATA_SOURCE, POSTGRES_DATA_SOURCE]
    )
    def test_empty_rules_list(self, data_source: str) -> None:
        """
        Test handling of empty rules list.
        """
        command = ["check", data_source]
        result = E2ETestUtils.run_cli_command(command)

        E2ETestUtils.assert_error_handling(result, "rule")

    @pytest.mark.parametrize(
        "data_source", [SQLITE_DATA_SOURCE, MYSQL_DATA_SOURCE, POSTGRES_DATA_SOURCE]
    )
    def test_large_dataset_performance(self, data_source: str) -> None:
        """
        Test performance with large dataset (basic timing check).
        """
        command = ["check", data_source, "--rule", "not_null(name)", "--verbose"]
        result = E2ETestUtils.run_cli_command(command)

        E2ETestUtils.assert_performance_acceptable(result, max_time=30.0)
        E2ETestUtils.assert_verbose_output(result, should_have_samples=False)

    @pytest.mark.parametrize(
        "data_source", [SQLITE_DATA_SOURCE, MYSQL_DATA_SOURCE, POSTGRES_DATA_SOURCE]
    )
    def test_concurrent_rule_execution(self, data_source: str) -> None:
        """
        Test multiple rules execution to ensure proper rule merging and separation.
        """
        command = [
            "check",
            data_source,
            "--rule",
            "not_null(name)",
            "--rule",
            "not_null(email)",
            "--rule",
            "unique(id)",
            "--rule",
            "range(age,0,120)",
            "--rule",
            "enum(gender,0,1)",
            "--verbose",
        ]
        result = E2ETestUtils.run_cli_command(command)

        # Should handle multiple rules
        assert result.returncode == 1, "Should fail due to some rule failures"

        # Check that all rules are processed
        expected_rules = [
            "not_null(name)",
            "not_null(email)",
            "unique(id)",
            "range(age)",
            "enum(gender)",
        ]
        for rule in expected_rules:
            assert rule in result.stdout, f"Rule {rule} should be mentioned in output"

        E2ETestUtils.assert_verbose_output(result, should_have_samples=True)
        E2ETestUtils.assert_rule_count(result, 5)
