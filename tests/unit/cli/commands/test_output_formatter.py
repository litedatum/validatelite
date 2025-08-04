"""
🧙‍♂️ Output Formatter TDD Tests - Modern Testing Architecture

Features:
- Zero boilerplate using Builder Pattern
- Comprehensive boundary condition testing
- Unicode and special character handling
- Performance tests for large datasets
- Exception flow coverage
"""

import io
import sys
from typing import Any, Dict, List

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cli.core.output_formatter import OutputFormatter
from shared.enums import ExecutionStatus, SeverityLevel
from shared.schema import ExecutionResultSchema
from tests.shared.builders.test_builders import TestDataBuilder


class TestOutputFormatter:
    """Modern Output Formatter Test Suite - Testing Ghost's Architecture"""

    @pytest.fixture
    def formatter(self) -> OutputFormatter:
        """Output formatter instance"""
        return OutputFormatter()

    @pytest.fixture
    def success_results(self) -> List[ExecutionResultSchema]:
        """Success validation results using Builder Pattern"""
        return [
            TestDataBuilder.result()
            .with_rule("not_null_id", "not_null(id)")
            .with_entity("users")
            .with_counts(failed_records=0, total_records=1000)
            .with_timing(0.05)
            .with_status("PASSED")
            .build(),
            TestDataBuilder.result()
            .with_rule("unique_email", "unique(email)")
            .with_entity("users")
            .with_counts(failed_records=0, total_records=1000)
            .with_timing(0.08)
            .with_status("PASSED")
            .build(),
        ]

    @pytest.fixture
    def mixed_results(self) -> List[ExecutionResultSchema]:
        """Mixed validation results (success + failures)"""
        return [
            TestDataBuilder.result()
            .with_rule("not_null_id", "not_null(id)")
            .with_entity("users")
            .with_counts(failed_records=0, total_records=1000)
            .with_timing(0.05)
            .with_status("PASSED")
            .build(),
            TestDataBuilder.result()
            .with_rule("length_name", "length(name,2,50)")
            .with_entity("users")
            .with_counts(failed_records=3, total_records=1000)
            .with_timing(0.12)
            .with_status("FAILED")
            .with_message("3 records failed validation")
            .build(),
            TestDataBuilder.result()
            .with_rule("unique_email", "unique(email)")
            .with_entity("users")
            .with_counts(failed_records=0, total_records=1000)
            .with_timing(0.08)
            .with_status("PASSED")
            .build(),
        ]

    @pytest.fixture
    def failure_samples(self) -> List[Dict[str, Any]]:
        """Sample failure data for detailed output"""
        return [
            {
                "row": 125,
                "column": "name",
                "value": "X",
                "expected": "length 2-50",
                "actual": "length 1",
            },
            {
                "row": 856,
                "column": "name",
                "value": "",
                "expected": "length 2-50",
                "actual": "length 0",
            },
            {
                "row": 1001,
                "column": "name",
                "value": "ThisNameIsTooLongForValidation...",
                "expected": "length 2-50",
                "actual": "length 67",
            },
        ]

    # === BASIC OUTPUT FORMATTING TESTS ===

    def test_format_basic_success_output(
        self,
        formatter: OutputFormatter,
        success_results: List[ExecutionResultSchema],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test basic success output formatting"""
        output = formatter.format_basic_output(
            source="users.csv",
            total_records=1000,
            results=success_results,
            execution_time=0.15,
        )

        assert "✓ Checking users.csv (1,000 records)" in output
        assert "✓ not_null(id): PASSED (0 failures)" in output
        assert "✓ unique(email): PASSED (0 failures)" in output
        assert "Summary: 2 passed, 0 failed" in output
        assert "Time: 0.15s" in output

    def test_format_basic_mixed_output(
        self, formatter: OutputFormatter, mixed_results: List[ExecutionResultSchema]
    ) -> None:
        """Test basic output with mixed results"""
        output = formatter.format_basic_output(
            source="users.csv",
            total_records=1000,
            results=mixed_results,
            execution_time=0.25,
        )

        assert "✓ not_null(id): PASSED (0 failures)" in output
        assert "✗ length(name,2,50): FAILED (3 failures)" in output
        assert "✓ unique(email): PASSED (0 failures)" in output
        assert "Summary: 2 passed, 1 failed (0.30% overall error rate)" in output
        assert "Time: 0.25s" in output

    def test_format_verbose_output_with_samples(
        self,
        formatter: OutputFormatter,
        mixed_results: List[ExecutionResultSchema],
        failure_samples: List[Dict[str, Any]],
    ) -> None:
        """Test verbose output including failure samples"""
        output = formatter.format_verbose_output(
            source="users.csv",
            total_records=1000,
            results=mixed_results,
            execution_time=0.25,
            failure_samples={"length_name": failure_samples},
        )

        assert "✓ Checking users.csv (1,000 records)" in output
        assert "Source: file://users.csv" in output
        assert "Processing: 1,000 records in" in output
        assert "Rules: 3 validation rules loaded" in output

        # Check detailed rule output
        assert "✗ length(name,2,50): FAILED (3 failures)" in output
        assert "Failure rate: 0.30% (3 out of 1,000)" in output
        assert "Failed records (showing first 20 of 3):" in output
        assert "Row 125: name='X' (length=1, expected 2-50)" in output
        assert "Row 856: name='' (length=0, expected 2-50)" in output

        # Check summary
        assert "Processing time: 0.25s" in output
        assert "Memory used:" in output  # Should include memory info

    def test_format_quiet_mode_summary_only(
        self, formatter: OutputFormatter, mixed_results: List[ExecutionResultSchema]
    ) -> None:
        """Test quiet mode shows only summary"""
        output = formatter.format_quiet_output(
            source="users.csv",
            total_records=1000,
            results=mixed_results,
            execution_time=0.25,
        )

        # Should be very concise
        lines = output.strip().split("\n")
        assert len(lines) <= 3  # Maximum 3 lines for quiet mode

        assert "users.csv: 2 passed, 1 failed (0.30% error rate)" in output
        assert "Time: 0.25s" in output

        # Should NOT contain detailed rule information
        assert "Processing:" not in output
        assert "Failed records" not in output

    # === BOUNDARY CONDITION TESTS ===

    def test_empty_results_handling(self, formatter: OutputFormatter) -> None:
        """Test handling of empty results"""
        output = formatter.format_basic_output(
            source="empty.csv", total_records=0, results=[], execution_time=0.01
        )

        assert "✓ Checking empty.csv (0 records)" in output
        assert "No validation rules executed" in output
        assert "Time: 0.01s" in output

    def test_zero_records_but_rules_exist(
        self, formatter: OutputFormatter, success_results: List[ExecutionResultSchema]
    ) -> None:
        """Test zero records with rules defined"""
        output = formatter.format_basic_output(
            source="empty.csv",
            total_records=0,
            results=success_results,
            execution_time=0.01,
        )

        assert "✓ Checking empty.csv (0 records)" in output
        assert "Warning: No records to validate" in output
        assert "Time: 0.01s" in output

    def test_extremely_large_numbers_formatting(
        self, formatter: OutputFormatter
    ) -> None:
        """Test formatting of extremely large numbers"""
        large_result = (
            TestDataBuilder.result()
            .with_counts(failed_records=1234567, total_records=123456789)
            .build()
        )

        output = formatter.format_basic_output(
            source="large.csv",
            total_records=123456789,
            results=[large_result],
            execution_time=300.5,
        )

        assert "123,456,789 records" in output
        assert "1,234,567 failures" in output
        assert "Time: 5m 0.5s" in output  # Should format long times nicely

    def test_unicode_source_names(
        self, formatter: OutputFormatter, success_results: List[ExecutionResultSchema]
    ) -> None:
        """Test Unicode characters in source names"""
        unicode_sources = [
            "用户数据.csv",  # Chinese
            "données_utilisateur.csv",  # French with accents
            "пользователи.csv",  # Cyrillic
            "файл_данных.xlsx",  # Mixed Unicode
        ]

        for source in unicode_sources:
            output = formatter.format_basic_output(
                source=source,
                total_records=100,
                results=success_results,
                execution_time=0.1,
            )

            assert source in output
            assert "✓ Checking" in output
            assert "100 records" in output

    def test_special_characters_in_rule_names(self, formatter: OutputFormatter) -> None:
        """Test special characters in rule names and descriptions"""
        special_result = TestDataBuilder.result()
        result_dict = special_result.build().model_dump()
        result_dict["rule_name"] = "email-validation_rule"
        result_dict["rule_description"] = "Check email format: user@domain.com"

        special_results = [ExecutionResultSchema(**result_dict)]

        output = formatter.format_basic_output(
            source="test.csv",
            total_records=100,
            results=special_results,
            execution_time=0.1,
        )

        assert "email-validation_rule" in output
        assert "user@domain.com" in output or "email format" in output

    @given(st.floats(min_value=0.001, max_value=3600.0))
    def test_property_based_execution_time_formatting(
        self, formatter: OutputFormatter, execution_time: float
    ) -> None:
        """Property-based test for execution time formatting"""
        result = TestDataBuilder.result().build()

        output = formatter.format_basic_output(
            source="test.csv",
            total_records=100,
            results=[result],
            execution_time=execution_time,
        )

        assert "Time:" in output
        # Time should be formatted appropriately (seconds, minutes, etc.)
        if execution_time < 60:
            assert (
                f"{execution_time:.2f}s" in output or f"{execution_time:.1f}s" in output
            )
        else:
            assert "m" in output  # Should show minutes for longer times

    # === ERROR HANDLING TESTS ===

    def test_malformed_result_handling(self, formatter: OutputFormatter) -> None:
        """Test handling of malformed results"""
        malformed_results = [
            {"invalid": "structure"},  # Missing required fields
            None,  # None result
            {"rule_name": "test", "status": "UNKNOWN"},  # Unknown status
        ]

        # Should handle gracefully without crashing
        try:
            output = formatter.format_basic_output(
                source="test.csv",
                total_records=100,
                results=malformed_results,
                execution_time=0.1,
            )
            # Should include error information
            assert "Error formatting result" in output or "Invalid result" in output
        except Exception as e:
            pytest.fail(f"Formatter should handle malformed results gracefully: {e}")

    def test_negative_values_handling(self, formatter: OutputFormatter) -> None:
        """Test handling of negative/invalid values"""
        invalid_result = (
            TestDataBuilder.result()
            .with_counts(failed_records=-1, total_records=100)  # Invalid negative count
            .build()
        )

        output = formatter.format_basic_output(
            source="test.csv",
            total_records=100,
            results=[invalid_result],
            execution_time=-0.5,  # Invalid negative time
        )

        # Should handle gracefully
        assert "test.csv" in output
        # Should either correct negative values or show error message
        assert "Time:" in output

    def test_division_by_zero_error_rate_calculation(
        self, formatter: OutputFormatter
    ) -> None:
        """Test division by zero in error rate calculation"""
        result = (
            TestDataBuilder.result()
            .with_counts(failed_records=5, total_records=0)  # Division by zero scenario
            .build()
        )

        output = formatter.format_basic_output(
            source="test.csv", total_records=0, results=[result], execution_time=0.1
        )

        # Should handle division by zero gracefully
        assert "test.csv" in output
        # Should not crash and should handle the error rate appropriately
        assert "error rate" not in output or "N/A" in output or "0%" in output

    # === PERFORMANCE TESTS ===

    def test_large_failure_samples_performance(
        self, formatter: OutputFormatter
    ) -> None:
        """Test performance with large failure samples"""
        # Create 10,000 failure samples
        large_failure_samples = [
            {
                "row": i,
                "column": "name",
                "value": f"invalid_{i}",
                "expected": "valid format",
            }
            for i in range(10000)
        ]

        large_result = (
            TestDataBuilder.result()
            .with_rule("validation_rule", "custom_rule(name)")
            .with_counts(failed_records=10000, total_records=100000)
            .build()
        )

        import time

        start_time = time.time()

        output = formatter.format_verbose_output(
            source="large.csv",
            total_records=100000,
            results=[large_result],
            execution_time=5.0,
            failure_samples={"validation_rule": large_failure_samples},
        )

        end_time = time.time()
        format_time = end_time - start_time

        # Should limit samples to reasonable number (20-50) and format quickly
        assert (
            "showing first 20 of 10,000" in output
            or "showing first 50 of 10,000" in output
        )
        assert format_time < 0.5, f"Formatting took {format_time:.2f}s, expected < 0.5s"

        # Should not include all 10,000 samples in output
        sample_lines = [
            line for line in output.split("\n") if "Row" in line and "invalid_" in line
        ]
        assert len(sample_lines) <= 50, f"Too many sample lines: {len(sample_lines)}"

    def test_memory_usage_with_large_output(self, formatter: OutputFormatter) -> None:
        """Test memory usage with large output"""
        # Create many results
        many_results = [
            (
                TestDataBuilder.result()
                .with_rule(f"rule_{i}", f"test_rule_{i}(column_{i})")
                .with_entity("large_table")
                .with_counts(failed_records=i % 10, total_records=1000)
                .build()
            )
            for i in range(1000)
        ]

        import os

        import psutil

        process = psutil.Process(os.getpid())

        memory_before = process.memory_info().rss

        output = formatter.format_basic_output(
            source="large.csv",
            total_records=1000000,
            results=many_results,
            execution_time=10.0,
        )

        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before

        # Memory increase should be reasonable (< 50MB for 1000 results)
        assert (
            memory_increase < 50 * 1024 * 1024
        ), f"Memory increase: {memory_increase / 1024 / 1024:.2f}MB"
        assert len(output) > 0  # Should produce output

    # === OUTPUT STREAM TESTS ===

    def test_output_to_different_streams(
        self, formatter: OutputFormatter, success_results: List[ExecutionResultSchema]
    ) -> None:
        """Test output to different streams (stdout, stderr, file)"""
        # Test stdout
        stdout_buffer = io.StringIO()
        formatter.write_output(output="Test output to stdout", stream=stdout_buffer)
        assert stdout_buffer.getvalue() == "Test output to stdout\n"

        # Test stderr
        stderr_buffer = io.StringIO()
        formatter.write_output(
            output="Test error output", stream=stderr_buffer, is_error=True
        )
        assert stderr_buffer.getvalue() == "Test error output\n"

    def test_color_output_handling(
        self, formatter: OutputFormatter, mixed_results: List[ExecutionResultSchema]
    ) -> None:
        """Test colored output handling"""
        # Test with color enabled
        colored_output = formatter.format_basic_output(
            source="test.csv",
            total_records=100,
            results=mixed_results,
            execution_time=0.1,
            use_colors=True,
        )

        # Should contain ANSI color codes for success/failure
        assert (
            "✓" in colored_output or "\033[32m" in colored_output
        )  # Green for success
        assert "✗" in colored_output or "\033[31m" in colored_output  # Red for failure

        # Test with color disabled
        plain_output = formatter.format_basic_output(
            source="test.csv",
            total_records=100,
            results=mixed_results,
            execution_time=0.1,
            use_colors=False,
        )

        # Should not contain ANSI color codes
        assert "\033[" not in plain_output

    def test_progress_indicator_formatting(self, formatter: OutputFormatter) -> None:
        """Test progress indicator formatting"""
        # Test progress during long operations
        progress_output = formatter.format_progress_indicator(
            current=250, total=1000, operation="Validating records"
        )

        assert "Validating records" in progress_output
        assert "25%" in progress_output or "250/1000" in progress_output

        # Test completed progress
        completed_output = formatter.format_progress_indicator(
            current=1000, total=1000, operation="Validation complete"
        )

        assert "100%" in completed_output or "complete" in completed_output.lower()

    # === INTEGRATION TESTS ===

    def test_formatter_with_real_console_output(
        self,
        formatter: OutputFormatter,
        mixed_results: List[ExecutionResultSchema],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test formatter with real console output"""
        # This tests actual console interaction
        formatter.print_results(
            source="integration_test.csv",
            total_records=1000,
            results=mixed_results,
            execution_time=0.5,
            mode="basic",
        )

        captured = capsys.readouterr()

        assert "integration_test.csv" in captured.out
        assert "PASSED" in captured.out or "✓" in captured.out
        assert "FAILED" in captured.out or "✗" in captured.out
        assert "Time: 0.5s" in captured.out

    def test_formatter_error_output_to_stderr(
        self, formatter: OutputFormatter, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test error output goes to stderr."""
        formatter.print_error("Test error message")
        captured = capsys.readouterr()
        assert "Test error message" in captured.err
        assert "Test error message" not in captured.out

    # ==============================
    # New Architecture Tests (dataset_metrics)
    # ==============================

    def test_display_results_with_new_architecture_dataset_metrics(
        self, formatter: OutputFormatter, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test display_results method with new architecture using dataset_metrics."""
        from shared.schema.base import DatasetMetrics

        # Create results using new architecture (dataset_metrics)
        new_format_results = [
            {
                "rule_id": "unique_name",
                "status": "FAILED",
                "dataset_metrics": [
                    DatasetMetrics(
                        entity_name="data_quality.customers",
                        total_records=1000,
                        failed_records=2,
                        processing_time=0.03,
                    )
                ],
                "execution_time": 0.03,
                "sample_data": [
                    {"name": "Huhansan3982", "dup_cnt": 2},
                    {"name": "Yang6646", "dup_cnt": 2},
                ],
            },
            {
                "rule_id": "not_null_gender",
                "status": "FAILED",
                "dataset_metrics": [
                    DatasetMetrics(
                        entity_name="data_quality.customers",
                        total_records=1000,
                        failed_records=259,
                        processing_time=0.01,
                    )
                ],
                "execution_time": 0.01,
                "sample_data": [
                    {"id": 3, "name": "Emy3737", "gender": None},
                    {"id": 7, "name": "Huhansan9960", "gender": None},
                ],
            },
        ]

        # Create mock rules
        rules = [
            TestDataBuilder.rule()
            .with_name("unique_name")
            .with_target("data_quality", "customers", "name")
            .as_unique_rule()
            .build(),
            TestDataBuilder.rule()
            .with_name("not_null_gender")
            .with_target("data_quality", "customers", "gender")
            .as_not_null_rule()
            .build(),
        ]

        # Capture output
        with capsys.disabled():
            formatter.display_results(
                results=new_format_results,
                rules=rules,
                source="mysql://root:root123@localhost:3306/data_quality.customers",
                execution_time=0.25,
                total_rules=2,
            )

    def test_calculate_stats_with_new_architecture(
        self, formatter: OutputFormatter
    ) -> None:
        """Test _calculate_stats method with new architecture dataset_metrics."""
        from shared.schema.base import DatasetMetrics

        # Create new format results
        new_format_results = [
            {
                "rule_id": "unique_name",
                "status": "FAILED",
                "dataset_metrics": [
                    DatasetMetrics(
                        entity_name="test_db.test_table",
                        total_records=1000,
                        failed_records=2,
                        processing_time=0.03,
                    )
                ],
                "execution_time": 0.03,
            },
            {
                "rule_id": "not_null_gender",
                "status": "FAILED",
                "dataset_metrics": [
                    DatasetMetrics(
                        entity_name="test_db.test_table",
                        total_records=1000,
                        failed_records=259,
                        processing_time=0.01,
                    )
                ],
                "execution_time": 0.01,
            },
        ]

        # Calculate stats
        stats = formatter._calculate_stats(new_format_results)

        # Verify correct calculation from dataset_metrics
        assert stats["total_rules"] == 2
        assert stats["passed_rules"] == 0
        assert stats["failed_rules"] == 2
        assert stats["total_records"] == 1000  # Should use max from dataset_metrics
        assert stats["total_failures"] == 261  # 2 + 259
        assert stats["overall_error_rate"] == 26.1  # (261/1000)*100

    def test_display_rule_result_with_new_architecture(
        self, formatter: OutputFormatter, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test _display_rule_result method with new architecture dataset_metrics."""
        from shared.schema.base import DatasetMetrics

        # Create new format result
        new_format_result = {
            "rule_id": "unique_name",
            "status": "FAILED",
            "dataset_metrics": [
                DatasetMetrics(
                    entity_name="test_db.test_table",
                    total_records=1000,
                    failed_records=2,
                    processing_time=0.03,
                )
            ],
            "execution_time": 0.03,
            "sample_data": [
                {"name": "Huhansan3982", "dup_cnt": 2},
                {"name": "Yang6646", "dup_cnt": 2},
            ],
        }

        # Create mock rule map
        rule_map = {
            "unique_name": TestDataBuilder.rule()
            .with_name("unique_name")
            .with_target("test_db", "test_table", "name")
            .as_unique_rule()
            .build()
        }

        # Test verbose mode to show samples
        formatter.verbose = True

        # Capture output
        with capsys.disabled():
            formatter._display_rule_result(
                result=new_format_result, rule_map=rule_map, show_samples=True
            )

    def test_mixed_old_and_new_architecture_results(
        self, formatter: OutputFormatter
    ) -> None:
        """Test handling of mixed old and new architecture results."""
        from shared.schema.base import DatasetMetrics

        # Mix of old and new format results
        mixed_results = [
            # Old format (legacy)
            {
                "rule_id": "old_rule",
                "status": "PASSED",
                "failed_records": 0,
                "total_records": 1000,
                "execution_time": 0.05,
            },
            # New format (dataset_metrics)
            {
                "rule_id": "new_rule",
                "status": "FAILED",
                "dataset_metrics": [
                    DatasetMetrics(
                        entity_name="test_db.test_table",
                        total_records=1000,
                        failed_records=5,
                        processing_time=0.02,
                    )
                ],
                "execution_time": 0.02,
            },
        ]

        # Calculate stats
        stats = formatter._calculate_stats(mixed_results)

        # Verify correct handling of mixed formats
        assert stats["total_rules"] == 2
        assert stats["passed_rules"] == 1
        assert stats["failed_rules"] == 1
        assert stats["total_records"] == 1000
        assert stats["total_failures"] == 5  # 0 + 5
        assert stats["overall_error_rate"] == 0.5  # (5/1000)*100

    def test_new_architecture_with_multiple_dataset_metrics(
        self, formatter: OutputFormatter
    ) -> None:
        """Test handling of multiple dataset_metrics in new architecture."""
        from shared.schema.base import DatasetMetrics

        # Result with multiple dataset_metrics
        multi_dataset_result = {
            "rule_id": "multi_table_rule",
            "status": "FAILED",
            "dataset_metrics": [
                DatasetMetrics(
                    entity_name="db1.table1",
                    total_records=500,
                    failed_records=3,
                    processing_time=0.01,
                ),
                DatasetMetrics(
                    entity_name="db2.table2",
                    total_records=300,
                    failed_records=2,
                    processing_time=0.01,
                ),
            ],
            "execution_time": 0.02,
        }

        # Calculate stats
        stats = formatter._calculate_stats([multi_dataset_result])

        # Verify aggregation across multiple datasets
        assert stats["total_rules"] == 1
        assert stats["failed_rules"] == 1
        assert (
            stats["total_records"] == 800
        )  # Should sum total_records from all datasets (500 + 300)
        assert stats["total_failures"] == 5  # 3 + 2
        assert stats["overall_error_rate"] == 0.625  # (5/800)*100

    def test_new_architecture_zero_records_edge_case(
        self, formatter: OutputFormatter
    ) -> None:
        """Test edge case with zero records in new architecture."""
        from shared.schema.base import DatasetMetrics

        # Result with zero records
        zero_records_result = {
            "rule_id": "empty_table_rule",
            "status": "PASSED",
            "dataset_metrics": [
                DatasetMetrics(
                    entity_name="test_db.empty_table",
                    total_records=0,
                    failed_records=0,
                    processing_time=0.0,
                )
            ],
            "execution_time": 0.0,
        }

        # Calculate stats
        stats = formatter._calculate_stats([zero_records_result])

        # Verify handling of zero records
        assert stats["total_rules"] == 1
        assert stats["passed_rules"] == 1
        assert stats["failed_rules"] == 0
        assert stats["total_records"] == 0
        assert stats["total_failures"] == 0
        assert stats["overall_error_rate"] == 0.0  # Should handle division by zero

    def test_new_architecture_model_dump_compatibility(
        self, formatter: OutputFormatter
    ) -> None:
        """Test that model_dump() results work correctly with new architecture."""
        from shared.schema.base import DatasetMetrics

        # Create ExecutionResultSchema with new architecture
        result_schema = ExecutionResultSchema(
            rule_id="test_rule",
            status="FAILED",
            dataset_metrics=[
                DatasetMetrics(
                    entity_name="test_db.test_table",
                    total_records=1000,
                    failed_records=10,
                    processing_time=0.05,
                )
            ],
            execution_time=0.05,
        )

        # Convert to dict using model_dump()
        result_dict = result_schema.model_dump()

        # Verify the dict contains dataset_metrics but not failed_records/total_records
        assert "dataset_metrics" in result_dict
        assert "failed_records" not in result_dict
        assert "total_records" not in result_dict

        # Test that our formatter can handle this correctly
        stats = formatter._calculate_stats([result_dict])

        # Verify correct calculation from dataset_metrics
        assert stats["total_rules"] == 1
        assert stats["failed_rules"] == 1
        assert stats["total_records"] == 1000
        assert stats["total_failures"] == 10
        assert stats["overall_error_rate"] == 1.0  # (10/1000)*100
