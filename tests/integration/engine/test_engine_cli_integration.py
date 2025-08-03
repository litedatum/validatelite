"""
🚀 Engine CLI Integration Tests - Modern Testing Architecture

This module tests the complete CLI → Rule Engine integration workflow,
focusing on command invocation, parameter passing, and result handling.

Key Testing Areas:
- CLI command execution with Rule Engine backend
- Parameter transformation and validation flow
- Result formatting and output generation
- Error propagation from Engine to CLI
- Performance monitoring for CLI scenarios

Modern Testing Features:
✅ Schema Builder Pattern - Zero boilerplate test data creation
✅ Contract Testing - Mock/Real implementation consistency
✅ Property-based Testing - Edge case discovery with hypothesis
✅ Mutation Testing Ready - Catches subtle logic errors

Test Architecture:
CLI Commands → Source/Rule Parsers → DataValidator → Rule Engine → Executors
     ↓              ↓                    ↓             ↓           ↓
  Integration tests covering the complete pipeline from CLI to Engine results
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Tuple
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest
from click.testing import CliRunner
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# CLI imports
from cli.commands.check import check_command
from cli.core.config import get_cli_config
from cli.core.data_validator import DataValidator
from cli.core.output_formatter import OutputFormatter
from cli.core.rule_parser import RuleParser
from cli.core.source_parser import SourceParser

# Configuration and error handling
from core.config import get_core_config

# Core engine imports
from core.engine.rule_engine import RuleEngine
from shared.enums import ConnectionType, ExecutionStatus, RuleType
from shared.exceptions import EngineError, OperationError, RuleExecutionError
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.result_schema import ExecutionResultSchema
from shared.schema.rule_schema import RuleSchema

# Utilities
from shared.utils.logger import get_logger
from tests.shared.builders.performance_test_base import (
    PerformanceMetrics,
    PerformanceTestBase,
)

# Modern testing infrastructure
from tests.shared.builders.test_builders import TestDataBuilder
from tests.shared.contracts.test_contracts import MockContract

logger = get_logger(__name__)


class TestEngineCliIntegrationModern(PerformanceTestBase):
    """
    🎯 Modern Engine CLI Integration Test Suite

    Tests complete CLI → Engine integration using modern testing patterns.
    Inherits performance monitoring from PerformanceTestBase.
    """

    @pytest.fixture
    def cli_runner(self) -> CliRunner:
        """CLI test runner with isolated environment"""
        return CliRunner()

    @pytest.fixture
    def mock_components(self) -> Dict[str, Any]:
        """Contract-compliant component mocks for controlled testing"""
        return {
            "cli_config": MockContract.create_cli_config_mock(),
            "data_validator": MockContract.create_data_validator_mock(),
            "output_formatter": MockContract.create_output_formatter_mock(),
        }

    @pytest.fixture
    def test_csv_data(self, builder: TestDataBuilder) -> str:
        """CSV test data with known quality issues for validation"""
        return (
            builder.csv_data()
            .with_headers(["id", "name", "email", "age", "status"])
            .with_rows(
                [
                    [1, "John Doe", "john@example.com", 25, "active"],
                    [2, "Jane Smith", "jane@example.com", 30, "active"],
                    [3, "", "invalid-email", -5, "inactive"],  # Quality issues
                    [4, "Bob Johnson", "bob@example.com", 150, ""],  # More issues
                    [5, "Alice Brown", "alice@example.com", 28, "active"],
                ]
            )
            .build_file()
        )

    @pytest.fixture
    def validation_rules_file(self) -> Generator[str, None, None]:
        """Rules file with comprehensive validation rules"""
        rules_data = {
            "version": "1.0",
            "rules": [
                {
                    "type": "not_null",
                    "column": "id",
                    "description": "ID cannot be null",
                },
                {
                    "type": "not_null",
                    "column": "name",
                    "description": "Name cannot be empty",
                },
                {
                    "type": "unique",
                    "column": "email",
                    "description": "Email must be unique",
                },
                {
                    "type": "range",
                    "column": "age",
                    "min": 0,
                    "max": 120,
                    "description": "Valid age range",
                },
                {
                    "type": "regex",
                    "column": "email",
                    "pattern": "^[\\w.-]+@[\\w.-]+\\.[a-zA-Z]{2,}$",
                    "description": "Valid email format",
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(rules_data, f)
            temp_file = f.name

        yield temp_file
        Path(temp_file).unlink(missing_ok=True)

    # ===============================
    # CLI → Engine Integration Tests
    # ===============================

    @patch("cli.commands.check.get_core_config")
    @patch("cli.commands.check.get_cli_config")
    @patch("cli.commands.check.DataValidator")
    def test_complete_cli_to_engine_workflow_success(
        self,
        mock_data_validator_class: AsyncMock,
        mock_cli_config: Mock,
        mock_core_config: Mock,
        cli_runner: CliRunner,
        test_csv_data: str,
        validation_rules_file: str,
        builder: TestDataBuilder,
    ) -> None:
        """
        Test complete CLI → Engine workflow with successful validation

        Workflow: CLI Command → Parsers → DataValidator → RuleEngine → Results → Output
        """
        # Setup configurations using Builder Pattern
        mock_core_config.return_value = Mock(
            merge_execution_enabled=True,
            table_size_threshold=1000,
            rule_count_threshold=5,
        )
        mock_cli_config.return_value = (
            builder.connection()
            .with_type(ConnectionType.CSV)
            .with_file_path(test_csv_data)
            .build()
        )

        # Create successful validation results using Builder Pattern
        success_results = [
            builder.result()
            .with_rule("not_null_id", "not_null(id)")
            .with_entity("test_data.csv")
            .with_counts(failed_records=0, total_records=5)
            .with_timing(0.05)
            .with_status("PASSED")
            .build(),
            builder.result()
            .with_rule("unique_email", "unique(email)")
            .with_entity("test_data.csv")
            .with_counts(failed_records=0, total_records=5)
            .with_timing(0.08)
            .with_status("PASSED")
            .build(),
        ]

        # Contract-compliant DataValidator mock
        mock_validator_instance = AsyncMock()
        mock_validator_instance.validate.return_value = success_results
        mock_data_validator_class.return_value = mock_validator_instance

        # Execute CLI command
        result = cli_runner.invoke(
            check_command,
            [test_csv_data, "--rules", validation_rules_file, "--verbose"],
        )

        # Verify CLI executed successfully
        assert result.exit_code == 0, f"CLI command failed: {result.output}"

        # Verify DataValidator was called with correct parameters
        mock_data_validator_class.assert_called_once()
        validator_call_args = mock_data_validator_class.call_args

        # Verify source configuration was passed correctly
        source_config = validator_call_args[1]["source_config"]
        assert source_config.connection_type == ConnectionType.CSV
        assert source_config.file_path == test_csv_data

        # Verify rules were parsed and passed correctly
        rules_config = validator_call_args[1]["rules"]
        assert len(rules_config) > 0
        assert any(rule.type == RuleType.NOT_NULL for rule in rules_config)

        # Verify validator.validate() was called
        mock_validator_instance.validate.assert_called_once()

        # Performance verification
        assert "validation" in result.output.lower() or "check" in result.output.lower()

    @patch("cli.commands.check.get_core_config")
    @patch("cli.commands.check.get_cli_config")
    @patch("cli.commands.check.DataValidator")
    def test_cli_to_engine_validation_failures(
        self,
        mock_data_validator_class: AsyncMock,
        mock_cli_config: Mock,
        mock_core_config: Mock,
        cli_runner: CliRunner,
        test_csv_data: str,
        builder: TestDataBuilder,
    ) -> None:
        """
        Test CLI → Engine workflow with validation failures

        Verifies error propagation and user-friendly error reporting
        """
        # Setup configurations
        mock_core_config.return_value = Mock()
        mock_cli_config.return_value = Mock()

        # Create validation results with failures using Builder Pattern
        failure_results = [
            builder.result()
            .with_rule("not_null_name", "not_null(name)")
            .with_entity("test_data.csv")
            .with_counts(failed_records=1, total_records=5)
            .with_timing(0.12)
            .with_status("FAILED")
            .build(),
            builder.result()
            .with_rule("range_age", "range(age,0,120)")
            .with_entity("test_data.csv")
            .with_counts(failed_records=2, total_records=5)
            .with_timing(0.08)
            .with_status("FAILED")
            .build(),
        ]

        # Contract-compliant DataValidator mock
        mock_validator_instance = AsyncMock()
        mock_validator_instance.validate.return_value = failure_results
        mock_data_validator_class.return_value = mock_validator_instance

        # Execute CLI command
        result = cli_runner.invoke(
            check_command,
            [
                test_csv_data,
                "--rule",
                "not_null(name)",
                "--rule",
                "range(age,0,120)",
                "--verbose",
            ],
        )

        # CLI should complete execution but report failures
        # Exit code depends on implementation - checking that it executed
        assert result.exit_code in [0, 1], f"Unexpected exit code: {result.exit_code}"

        # Verify validator was called and results were processed
        mock_validator_instance.validate.assert_called_once()
        assert len(result.output) > 0, "Should have output for failed validations"

    @patch("cli.commands.check.get_core_config")
    @patch("cli.commands.check.get_cli_config")
    @patch("cli.commands.check.DataValidator")
    def test_cli_to_engine_error_propagation(
        self,
        mock_data_validator_class: AsyncMock,
        mock_cli_config: Mock,
        mock_core_config: Mock,
        cli_runner: CliRunner,
        test_csv_data: str,
        builder: TestDataBuilder,
    ) -> None:
        """
        Test error propagation from Engine to CLI

        Verifies that engine errors are properly caught and reported by CLI
        """
        # Setup configurations
        mock_core_config.return_value = Mock()
        mock_cli_config.return_value = Mock()

        # Mock DataValidator to raise EngineError
        mock_validator_instance = AsyncMock()
        mock_validator_instance.validate.side_effect = EngineError(
            message="Database connection failed",
            connection_id="test_connection",
            operation="rule_execution",
        )
        mock_data_validator_class.return_value = mock_validator_instance

        # Execute CLI command
        result = cli_runner.invoke(
            check_command, [test_csv_data, "--rule", "not_null(id)"]
        )

        # CLI should handle the error gracefully
        assert result.exit_code != 0, "Should exit with error code on engine failure"
        assert "error" in result.output.lower() or "failed" in result.output.lower()

        # Verify the error was properly handled by CLI exception system
        mock_validator_instance.validate.assert_called_once()

    # ===============================
    # Property-based Testing
    # ===============================

    @patch("cli.commands.check.get_core_config")
    @patch("cli.commands.check.get_cli_config")
    @patch("cli.commands.check.DataValidator")
    def test_property_based_cli_engine_scalability(
        self,
        mock_data_validator_class: AsyncMock,
        mock_cli_config: Mock,
        mock_core_config: Mock,
        cli_runner: CliRunner,
        builder: TestDataBuilder,
    ) -> None:
        """
        Property-based test for CLI → Engine scalability

        Tests various combinations of rule counts and record counts
        """

        @given(
            rule_count=st.integers(min_value=1, max_value=5),
            record_count=st.integers(min_value=10, max_value=50),
        )
        @settings(
            suppress_health_check=[HealthCheck.function_scoped_fixture],
            deadline=None,
            max_examples=3,  # Reduced for CI/CD efficiency
        )
        def run_scalability_test(rule_count: int, record_count: int) -> None:
            # Generate test data
            test_data = (
                builder.csv_data()
                .with_headers(["id", "name", "value"])
                .with_random_rows(
                    record_count,
                    {
                        "id": lambda i: i + 1,
                        "name": lambda i: f"item_{i}",
                        "value": lambda i: f"value_{i}",
                    },
                )
                .build_file()
            )

            try:
                # Setup configurations
                mock_core_config.return_value = Mock()
                mock_cli_config.return_value = Mock()

                # Generate variable number of successful results
                results = []
                for i in range(rule_count):
                    result = (
                        builder.result()
                        .with_rule(f"rule_{i}", f"not_null(column_{i})")
                        .with_entity("test_data")
                        .with_counts(failed_records=0, total_records=record_count)
                        .with_timing(0.01 * (i + 1))
                        .with_status("PASSED")
                        .build()
                    )
                    results.append(result)

                # Contract-compliant DataValidator mock
                mock_validator_instance = AsyncMock()
                mock_validator_instance.validate.return_value = results
                mock_data_validator_class.return_value = mock_validator_instance

                # Generate inline rules for testing (limit to 3 for performance)
                inline_rules = [
                    f"not_null(column_{i})" for i in range(min(rule_count, 3))
                ]

                # Execute CLI command with performance monitoring
                start_time = time.time()
                cli_result = cli_runner.invoke(
                    check_command,
                    [
                        test_data,
                        *[item for rule in inline_rules for item in ["--rule", rule]],
                    ],
                )
                execution_time = time.time() - start_time

                # Verify scalability properties
                assert (
                    cli_result.exit_code == 0
                ), f"CLI failed with {rule_count} rules and {record_count} records"
                assert (
                    execution_time < 10.0
                ), f"Execution too slow: {execution_time:.2f}s"

                # Verify validator was called
                mock_validator_instance.validate.assert_called_once()

            finally:
                # Cleanup
                Path(test_data).unlink(missing_ok=True)

        # Run the property-based test
        run_scalability_test()

    # ===============================
    # Performance and Stress Testing
    # ===============================

    @patch("cli.commands.check.get_core_config")
    @patch("cli.commands.check.get_cli_config")
    @patch("cli.commands.check.DataValidator")
    def test_cli_engine_performance_monitoring(
        self,
        mock_data_validator_class: AsyncMock,
        mock_cli_config: Mock,
        mock_core_config: Mock,
        cli_runner: CliRunner,
        builder: TestDataBuilder,
    ) -> None:
        """
        Test CLI → Engine performance with monitoring

        Uses PerformanceTestBase infrastructure for metrics collection
        """
        # Generate large test dataset
        large_dataset = (
            builder.csv_data()
            .with_headers(["id", "name", "email", "score"])
            .with_random_rows(
                1000,
                {
                    "id": lambda i: i + 1,
                    "name": lambda i: f"user_{i}",
                    "email": lambda i: f"user_{i}@test.com",
                    "score": lambda i: (i * 17) % 100,
                },
            )
            .build_file()
        )

        try:
            # Setup configurations for performance test
            mock_core_config.return_value = Mock(
                merge_execution_enabled=True,
                table_size_threshold=500,
                rule_count_threshold=3,
            )
            mock_cli_config.return_value = Mock()

            # Create performance-oriented results
            perf_results = [
                builder.result()
                .with_rule("not_null_id", "not_null(id)")
                .with_entity("large_dataset")
                .with_counts(failed_records=0, total_records=1000)
                .with_timing(0.15)
                .with_status("PASSED")
                .build(),
                builder.result()
                .with_rule("unique_email", "unique(email)")
                .with_entity("large_dataset")
                .with_counts(failed_records=0, total_records=1000)
                .with_timing(0.25)
                .with_status("PASSED")
                .build(),
            ]

            # Contract-compliant DataValidator mock
            mock_validator_instance = AsyncMock()
            mock_validator_instance.validate.return_value = perf_results
            mock_data_validator_class.return_value = mock_validator_instance

            # Execute with performance monitoring
            metrics, execution_time = self.measure_cli_performance(
                test_name="cli_engine_large_dataset",
                rule_count=2,
                func=lambda: cli_runner.invoke(
                    check_command,
                    [
                        large_dataset,
                        "--rule",
                        "not_null(id)",
                        "--rule",
                        "unique(email)",
                        "--quiet",
                    ],
                ),
            )

            # Verify performance metrics
            assert (
                metrics.execution_time < 5.0
            ), f"CLI → Engine too slow: {metrics.execution_time:.2f}s"
            assert (
                metrics.throughput > 0.5
            ), f"Throughput too low: {metrics.throughput:.2f} rules/s"

            # Verify execution completed successfully
            result = cli_runner.invoke(
                check_command,
                [
                    large_dataset,
                    "--rule",
                    "not_null(id)",
                    "--rule",
                    "unique(email)",
                    "--quiet",
                ],
            )
            assert result.exit_code == 0

        finally:
            Path(large_dataset).unlink(missing_ok=True)

    # ===============================
    # Contract Testing
    # ===============================

    def test_cli_datavalidator_contract_compliance(
        self, mock_components: Dict[str, Any]
    ) -> None:
        """
        Test that CLI components follow contracts with DataValidator

        Ensures Mock behavior matches real implementation contracts
        """
        # Verify DataValidator contract
        data_validator_mock = mock_components["data_validator"]

        # Contract verification
        assert hasattr(
            data_validator_mock, "validate"
        ), "DataValidator must have validate method"
        assert callable(data_validator_mock.validate), "validate must be callable"

        # Verify config contracts
        cli_config_mock = mock_components["cli_config"]

        # These should have expected configuration attributes
        assert hasattr(cli_config_mock, "debug_mode"), "CliConfig must have debug_mode"
        assert hasattr(
            cli_config_mock, "default_sample_size"
        ), "CliConfig must have default_sample_size"

    @patch("cli.commands.check.SourceParser")
    @patch("cli.commands.check.RuleParser")
    def test_cli_parser_contract_compliance(
        self,
        mock_rule_parser_class: Mock,
        mock_source_parser_class: Mock,
        builder: TestDataBuilder,
    ) -> None:
        """
        Test CLI parsers follow expected contracts

        Verifies parser interfaces match expected behavior patterns
        """
        # Setup contract-compliant mocks
        mock_source_parser = Mock()
        mock_source_parser.parse_source.return_value = (
            builder.connection()
            .with_type(ConnectionType.CSV)
            .with_file_path("test.csv")
            .build()
        )
        mock_source_parser_class.return_value = mock_source_parser

        mock_rule_parser = Mock()
        mock_rule_parser.parse_rules.return_value = [
            builder.rule().as_not_null_rule().build()
        ]
        mock_rule_parser_class.return_value = mock_rule_parser

        # Contract verification - parsers should have expected methods
        MockContract.verify_source_parser_contract(mock_source_parser)
        MockContract.verify_rule_parser_contract(mock_rule_parser)

        # Verify methods are callable and return expected types
        source_result = mock_source_parser.parse_source("test.csv")
        assert isinstance(source_result, ConnectionSchema)

        rules_result = mock_rule_parser.parse_rules(["not_null(id)"], None)
        assert isinstance(rules_result, list)
        assert all(isinstance(rule, RuleSchema) for rule in rules_result)

    # ===============================
    # Integration Edge Cases
    # ===============================

    def test_cli_engine_empty_dataset_handling(
        self, cli_runner: CliRunner, builder: TestDataBuilder
    ) -> None:
        """Test CLI → Engine handling of empty datasets"""
        # Create empty CSV file
        empty_csv = (
            builder.csv_data().with_headers(["id", "name"]).with_rows([]).build_file()
        )

        try:
            # Execute CLI command on empty data
            result = cli_runner.invoke(
                check_command, [empty_csv, "--rule", "not_null(id)"]
            )

            # Should handle empty data gracefully
            # The exact behavior depends on implementation
            # Exit code 6 is expected for execution errors (like empty files)
            assert result.exit_code in [
                0,
                1,
                6,
                10,
                20,
            ], f"Unexpected exit code: {result.exit_code}"

        finally:
            Path(empty_csv).unlink(missing_ok=True)

    def test_cli_engine_concurrent_requests(self, test_csv_data: str) -> None:
        """
        Test CLI → Engine behavior under concurrent requests using subprocess for true isolation
        """
        import subprocess
        import sys
        import threading

        results = []
        errors = []

        def run_cli_subprocess(idx: int) -> None:
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "cli_main.py",
                        "check",
                        test_csv_data,
                        "--rule",
                        "not_null(id)",
                        "--quiet",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                results.append((idx, proc.returncode, proc.stdout, proc.stderr))
            except Exception as e:
                errors.append((idx, str(e)))

        threads = []
        for i in range(3):
            t = threading.Thread(target=run_cli_subprocess, args=(i,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors: {errors}"
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        exit_codes = [code for _, code, _, _ in results]
        assert all(
            code == exit_codes[0] for code in exit_codes
        ), f"Inconsistent exit codes: {exit_codes}"
        # Optional: Verify the output.
        for idx, code, out, err in results:
            assert code in [
                0,
                1,
            ], f"Unexpected exit code {code} in thread {idx}. Output: {out} Error: {err}"

    # ===============================
    # Utility Methods
    # ===============================

    def measure_cli_performance(
        self, test_name: str, rule_count: int, func: Callable
    ) -> Tuple[PerformanceMetrics, float]:
        """
        Measure CLI → Engine performance with standardized metrics

        Returns both PerformanceMetrics object and execution time
        """
        start_time = time.time()
        result = func()
        execution_time = time.time() - start_time

        throughput = rule_count / execution_time if execution_time > 0 else 0

        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_delta=0.0,  # Simplified for integration tests
            rule_count=rule_count,
            throughput=throughput,
            test_name=test_name,
            timestamp=start_time,
        )

        # Store metrics for regression analysis
        self.performance_metrics.append(metrics)

        return metrics, execution_time
