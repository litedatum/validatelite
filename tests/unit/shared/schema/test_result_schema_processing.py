"""
👻 Result Schema Processing Tests - The Testing Ghost's Comprehensive Suite

As the Testing Ghost, I uncover all edge cases in result processing, status mapping,
and data integrity validation. No corner case escapes my spectral attention!

This file tests:
1. Result Creation & Factory Methods - All creation paths with edge cases
2. Status Processing & Validation - Status transitions, invalid states
3. Data Formatting & Serialization - JSON output, legacy compatibility
4. Multi-Dataset Support - Single table vs multi-table results
5. Cross-Database Metrics - Hook functionality and future compatibility
6. Calculation Methods - Success rates, failure rates, error handling
7. Sample Data Handling - Large samples, null data, malformed data
8. Time & Performance Metrics - Duration calculations, timestamp validation
9. Property-Based Testing - Random data validation
10. Contract Testing - Builder pattern compliance
"""

import json
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Property-based testing
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, composite
from pydantic import ValidationError

from shared.enums.execution_status import ExecutionStatus
from shared.schema.base import CrossDbMetrics, DatasetMetrics, ExecutionResultBase

# Core imports
from shared.schema.result_schema import ExecutionResultSchema
from shared.utils.datetime_utils import format_datetime, now

# Test utilities
from tests.shared.builders.test_builders import TestDataBuilder


class TestResultSchemaCreation:
    """👻 Result creation tests - Factory methods and constructors"""

    def test_create_success_result_basic(self, builder: TestDataBuilder) -> None:
        """Test basic success result creation"""
        result = ExecutionResultSchema.create_success_result(
            rule_id="test_rule_1",
            entity_name="test_db.users",
            total_count=1000,
            error_count=0,
        )

        assert result.rule_id == "test_rule_1"
        assert result.status == ExecutionStatus.PASSED.value
        assert result.total_count == 1000
        assert result.error_count == 0
        assert result.get_success_rate() == 1.0
        assert result.is_success()
        assert not result.is_failure()
        assert not result.is_error()

    def test_create_success_result_with_failures(
        self, builder: TestDataBuilder
    ) -> None:
        """Test success result with some failures (status should be FAILED)"""
        result = ExecutionResultSchema.create_success_result(
            rule_id="test_rule_2",
            entity_name="test_db.orders",
            total_count=1000,
            error_count=50,
        )

        assert result.rule_id == "test_rule_2"
        assert result.status == ExecutionStatus.FAILED.value
        assert result.total_count == 1000
        assert result.error_count == 50
        assert result.get_success_rate() == 0.95
        # 👻 GHOST FIX: Use approximate comparison for floating point precision
        assert abs(result.get_failure_rate() - 0.05) < 1e-10
        assert not result.is_success()
        assert result.is_failure()

    def test_create_error_result_complete(self, builder: TestDataBuilder) -> None:
        """Test error result creation with all fields"""
        result = ExecutionResultSchema.create_error_result(
            rule_id="test_rule_error",
            entity_name="test_db.problematic_table",
            error_message="Database connection failed",
            execution_time=2.5,
        )

        assert result.rule_id == "test_rule_error"
        assert result.status == ExecutionStatus.ERROR.value
        assert result.error_message == "Database connection failed"
        assert result.execution_time == 2.5
        assert result.total_count == 0
        assert result.error_count == 0
        assert result.is_error()
        assert not result.is_success()

    def test_create_from_legacy_compatibility(self, builder: TestDataBuilder) -> None:
        """Test legacy format compatibility"""
        result = ExecutionResultSchema.create_from_legacy(
            rule_id="legacy_rule",
            status="PASSED",
            total_count=500,
            error_count=0,
            execution_time=1.2,
            database="legacy_db",
            table="legacy_table",
            execution_message="Legacy execution successful",
            sample_data=[{"sample_key": "sample_value"}],
        )

        assert result.rule_id == "legacy_rule"
        assert result.status == "PASSED"
        assert result.total_count == 500
        assert result.sample_data == [{"sample_key": "sample_value"}]
        assert len(result.dataset_metrics) == 1
        assert result.dataset_metrics[0].entity_name == "legacy_db.legacy_table"

    def test_create_result_zero_records_edge_case(
        self, builder: TestDataBuilder
    ) -> None:
        """Zero records edge case"""
        result = ExecutionResultSchema.create_success_result(
            rule_id="zero_test", entity_name="empty_table", total_count=0, error_count=0
        )

        # Zero records but no errors should still be success
        assert result.is_success()
        assert result.get_success_rate() == 1.0
        assert result.get_failure_rate() == 0.0

    def test_create_result_invalid_error_count(self, builder: TestDataBuilder) -> None:
        """Error count larger than total count"""
        # This should work but be logically inconsistent - let's see how it handles it
        result = ExecutionResultSchema.create_success_result(
            rule_id="invalid_test",
            entity_name="bad_table",
            total_count=100,
            error_count=150,  # More errors than total records!
        )

        # Should handle gracefully but produce negative success rate
        assert result.total_count == 100
        assert result.error_count == 150
        assert result.get_success_rate() == -0.5  # (100-150)/100


class TestResultSchemaStatusProcessing:
    """👻 Status processing and validation tests"""

    def test_status_detection_methods(self, builder: TestDataBuilder) -> None:
        """Test all status detection methods"""
        # Test PASSED status
        passed_result = (
            builder.result().with_status(ExecutionStatus.PASSED.value).build()
        )
        assert passed_result.is_success()
        assert not passed_result.is_failure()
        assert not passed_result.is_error()

        # Test FAILED status
        failed_result = (
            builder.result().with_status(ExecutionStatus.FAILED.value).build()
        )
        assert not failed_result.is_success()
        assert failed_result.is_failure()
        assert not failed_result.is_error()

        # Test ERROR status
        error_result = builder.result().with_status(ExecutionStatus.ERROR.value).build()
        assert not error_result.is_success()
        assert not error_result.is_failure()
        assert error_result.is_error()

    @pytest.mark.parametrize(
        "status,expected_success,expected_failure,expected_error",
        [
            (ExecutionStatus.PASSED.value, True, False, False),
            (ExecutionStatus.FAILED.value, False, True, False),
            (ExecutionStatus.ERROR.value, False, False, True),
            (
                ExecutionStatus.WARNING.value,
                False,
                False,
                False,
            ),  # Warning is neither success nor failure
            (ExecutionStatus.CANCELLED.value, False, False, False),
        ],
    )
    def test_status_detection_parametrized(
        self,
        builder: TestDataBuilder,
        status: str,
        expected_success: bool,
        expected_failure: bool,
        expected_error: bool,
    ) -> None:
        """Parametrized test for all status types"""
        result = builder.result().with_status(status).build()

        assert result.is_success() == expected_success
        assert result.is_failure() == expected_failure
        assert result.is_error() == expected_error


class TestResultSchemaCalculations:
    """👻 Calculation methods and mathematical edge cases"""

    def test_success_rate_calculations(self, builder: TestDataBuilder) -> None:
        """Test success rate calculations with various scenarios"""
        # Perfect success
        result = (
            builder.result().with_counts(failed_records=0, total_records=1000).build()
        )
        assert result.get_success_rate() == 1.0
        assert result.get_failure_rate() == 0.0

        # 50% success
        result = (
            builder.result().with_counts(failed_records=500, total_records=1000).build()
        )
        assert result.get_success_rate() == 0.5
        assert result.get_failure_rate() == 0.5

        # Complete failure
        result = (
            builder.result()
            .with_counts(failed_records=1000, total_records=1000)
            .build()
        )
        assert result.get_success_rate() == 0.0
        assert result.get_failure_rate() == 1.0

    def test_zero_division_edge_case(self, builder: TestDataBuilder) -> None:
        """Zero division in success rate calculation"""
        result = builder.result().with_counts(failed_records=0, total_records=0).build()

        # Should handle zero total gracefully
        assert (
            result.get_success_rate() == 1.0
        )  # By design, zero records is considered success
        assert result.get_failure_rate() == 0.0

    @pytest.mark.parametrize(
        "total,failed,expected_success_rate",
        [
            (100, 0, 1.0),
            (100, 1, 0.99),
            (100, 50, 0.5),
            (100, 99, 0.01),
            (100, 100, 0.0),
            (1, 0, 1.0),
            (1, 1, 0.0),
        ],
    )
    def test_success_rate_precision(
        self,
        builder: TestDataBuilder,
        total: int,
        failed: int,
        expected_success_rate: float,
    ) -> None:
        """Test success rate calculation precision"""
        result = (
            builder.result()
            .with_counts(failed_records=failed, total_records=total)
            .build()
        )
        assert abs(result.get_success_rate() - expected_success_rate) < 0.001

    def test_floating_point_precision_issues(self, builder: TestDataBuilder) -> None:
        """Floating point precision issues"""
        # Test with numbers that might cause floating point issues
        result = builder.result().with_counts(failed_records=1, total_records=3).build()
        success_rate = result.get_success_rate()
        failure_rate = result.get_failure_rate()

        # Should sum to 1.0 (within floating point precision)
        assert abs((success_rate + failure_rate) - 1.0) < 1e-10


class TestResultSchemaDataFormatting:
    """👻 Data formatting and serialization tests"""

    def test_to_engine_dict_complete(self, builder: TestDataBuilder) -> None:
        """Test complete engine dictionary conversion"""
        result = ExecutionResultSchema.create_success_result(
            rule_id="dict_test",
            entity_name="test_db.test_table",
            total_count=1000,
            error_count=50,
            execution_time=2.5,
            message="Custom message",
        )

        engine_dict = result.to_engine_dict()

        assert engine_dict["rule_id"] == "dict_test"
        assert engine_dict["status"] == ExecutionStatus.FAILED.value
        assert engine_dict["total_records"] == 1000
        assert engine_dict["failed_records"] == 50
        assert engine_dict["success_rate"] == 0.95
        assert engine_dict["execution_time"] == 2.5
        assert engine_dict["message"] == "Custom message"
        assert "executed_at" in engine_dict

    def test_to_engine_dict_with_error(self, builder: TestDataBuilder) -> None:
        """Test engine dict conversion with error result"""
        result = ExecutionResultSchema.create_error_result(
            rule_id="error_dict_test",
            entity_name="error_table",
            error_message="Connection timeout",
            execution_time=30.0,
        )

        engine_dict = result.to_engine_dict()

        assert engine_dict["rule_id"] == "error_dict_test"
        assert engine_dict["status"] == ExecutionStatus.ERROR.value
        assert engine_dict["error_message"] == "Connection timeout"
        assert engine_dict["total_records"] == 0
        assert engine_dict["failed_records"] == 0

    def test_json_serialization_compatibility(self, builder: TestDataBuilder) -> None:
        """Test JSON serialization doesn't break"""
        result = builder.result().build()

        # Should be JSON serializable
        json_str = json.dumps(result.to_engine_dict())
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Should be deserializable
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_datetime_formatting_in_dict(self, builder: TestDataBuilder) -> None:
        """Datetime formatting consistency"""
        result = builder.result().build()
        engine_dict = result.to_engine_dict()

        if engine_dict["executed_at"]:
            # Should be in ISO format with Z suffix
            executed_at = engine_dict["executed_at"]
            assert isinstance(executed_at, str)
            assert executed_at.endswith("Z")

            # Should be parseable back to datetime
            from datetime import datetime

            parsed_dt = datetime.fromisoformat(executed_at.replace("Z", "+00:00"))
            assert isinstance(parsed_dt, datetime)


class TestResultSchemaSummaryMethods:
    """👻 Summary and message generation tests"""

    def test_get_summary_success(self, builder: TestDataBuilder) -> None:
        """Test summary for successful result"""
        result = ExecutionResultSchema.create_success_result(
            rule_id="summary_test",
            entity_name="test_table",
            total_count=1000,
            error_count=0,
        )

        summary = result.get_summary()
        assert "Success" in summary
        assert "1000" in summary
        assert "passed" in summary

    def test_get_summary_failure(self, builder: TestDataBuilder) -> None:
        """Test summary for failed result"""
        result = ExecutionResultSchema.create_success_result(
            rule_id="failure_summary_test",
            entity_name="test_table",
            total_count=1000,
            error_count=50,
        )

        summary = result.get_summary()
        assert "Failure" in summary
        assert "50/1000" in summary or "50" in summary and "1000" in summary

    def test_get_summary_error(self, builder: TestDataBuilder) -> None:
        """Test summary for error result"""
        result = ExecutionResultSchema.create_error_result(
            rule_id="error_summary_test",
            entity_name="error_table",
            error_message="Database connection failed",
        )

        summary = result.get_summary()
        assert "Error" in summary
        assert "Database connection failed" in summary

    def test_get_detailed_message_complete(self, builder: TestDataBuilder) -> None:
        """Test detailed message generation"""
        result = ExecutionResultSchema.create_success_result(
            rule_id="detailed_test",
            entity_name="test_table",
            total_count=1000,
            error_count=50,
            execution_time=2.5,
        )

        detailed = result.get_detailed_message()
        assert "Failure" in detailed  # Status summary
        assert "95.00%" in detailed or "0.95" in detailed  # Success rate
        assert "2.5" in detailed or "2.500" in detailed  # Execution time

    def test_get_detailed_message_zero_time(self, builder: TestDataBuilder) -> None:
        """Detailed message with zero execution time"""
        result = ExecutionResultSchema.create_success_result(
            rule_id="zero_time_test",
            entity_name="test_table",
            total_count=100,
            execution_time=0.0,
        )

        detailed = result.get_detailed_message()
        # Should handle zero time gracefully (might not include time info)
        assert isinstance(detailed, str)
        assert len(detailed) > 0


class TestResultSchemaMultiDatasetSupport:
    """👻 Multi-dataset and cross-database functionality tests"""

    def test_single_dataset_access(self, builder: TestDataBuilder) -> None:
        """Test single dataset access methods"""
        result = builder.result().with_entity("test_db.users").build()

        dataset = result.get_dataset_by_name("test_db.users")
        assert dataset is not None
        assert dataset.entity_name == "test_db.users"

        # Non-existent dataset should return None
        non_existent = result.get_dataset_by_name("non_existent.table")
        assert non_existent is None

    def test_multi_dataset_support_structure(self, builder: TestDataBuilder) -> None:
        """Test multi-dataset support structure"""
        result = builder.result().build()

        # Test adding additional dataset metrics
        additional_metric = DatasetMetrics(
            entity_name="another_db.another_table", total_records=500, failed_records=25
        )
        result.add_dataset_metric(additional_metric)

        assert len(result.dataset_metrics) == 2
        found_dataset = result.get_dataset_by_name("another_db.another_table")
        assert found_dataset is not None
        assert found_dataset.total_records == 500

    def test_check_multi_table_support_single_table(
        self, builder: TestDataBuilder
    ) -> None:
        """Test multi-table support check for single table"""
        result = builder.result().build()

        support_check = result.check_multi_table_support()

        assert support_check["supported"] is True
        assert support_check["dataset_count"] == 1
        assert support_check["is_cross_database"] is False

    def test_check_multi_table_support_multi_dataset(
        self, builder: TestDataBuilder
    ) -> None:
        """Test multi-table support check for multiple datasets"""
        result = builder.result().build()

        # Add multiple datasets
        result.add_dataset_metric(
            DatasetMetrics(
                entity_name="db2.table2", total_records=200, failed_records=10
            )
        )

        support_check = result.check_multi_table_support()

        assert support_check["supported"] is True
        assert support_check["dataset_count"] == 2
        assert support_check["is_cross_database"] is False

    def test_cross_db_summary_none_when_no_metrics(
        self, builder: TestDataBuilder
    ) -> None:
        """Test cross-database summary returns None when no cross-db metrics"""
        result = builder.result().build()

        cross_db_summary = result.get_cross_db_summary()
        assert cross_db_summary is None

    def test_cross_db_summary_with_metrics(self, builder: TestDataBuilder) -> None:
        """Test cross-database summary with metrics (Hook functionality)"""
        result = builder.result().build()

        # Manually set cross-db metrics for testing
        result.cross_db_metrics = CrossDbMetrics(
            strategy_used="memory_dataframe",
            data_transfer_time=1.5,
            total_processing_time=3.0,
            temp_data_size_mb=25,
        )

        summary = result.get_cross_db_summary()

        assert summary is not None
        assert summary["strategy"] == "memory_dataframe"
        assert summary["transfer_time"] == 1.5
        assert summary["processing_time"] == 3.0
        assert summary["data_size_mb"] == 25


class TestResultSchemaSampleDataHandling:
    """👻 Sample data processing and edge cases"""

    def test_sample_data_preservation(self, builder: TestDataBuilder) -> None:
        """Test sample data is preserved correctly"""
        sample_data: list[dict[str, Any]] = [
            {"id": 1, "name": None, "issue": "null_value"},
            {"id": 2, "name": "", "issue": "empty_string"},
        ]

        result = ExecutionResultSchema.create_from_legacy(
            rule_id="sample_test",
            status="FAILED",
            total_count=1000,
            error_count=2,
            execution_time=1.0,
            database="test_db",
            table="test_table",
            sample_data=sample_data,
        )

        assert result.sample_data == sample_data
        assert result.sample_data[0]["issue"] == "null_value"
        assert result.sample_data[1]["issue"] == "empty_string"

    def test_large_sample_data_handling(self, builder: TestDataBuilder) -> None:
        """Large sample data handling"""
        # Create large sample data
        large_sample: list[dict[str, Any]] = [
            {"id": i, "data": f"data_{i}"} for i in range(1000)
        ]
        result = builder.result().build()
        result.sample_data = large_sample

        # Should handle large data without issues
        engine_dict = result.to_engine_dict()
        assert isinstance(engine_dict["sample_data"], list)
        assert len(engine_dict["sample_data"]) == 1000

    def test_malformed_sample_data_handling(self, builder: TestDataBuilder) -> None:
        """Malformed sample data"""
        malformed_data: list[dict[str, Any]] = [
            {"failed_rows": "not_a_list", "sample_count": "not_a_number"}
        ]
        result = builder.result().build()
        result.sample_data = malformed_data

        # Should not crash when converting to dict
        engine_dict = result.to_engine_dict()
        assert engine_dict["sample_data"] == malformed_data

    def test_null_sample_data_handling(self, builder: TestDataBuilder) -> None:
        """Test null sample data handling"""
        result = builder.result().build()
        result.sample_data = None

        engine_dict = result.to_engine_dict()
        assert engine_dict["sample_data"] is None


class TestResultSchemaTimeAndPerformanceMetrics:
    """👻 Time calculations and performance metrics"""

    def test_execution_time_precision(self, builder: TestDataBuilder) -> None:
        """Test execution time precision handling"""
        result = builder.result().with_timing(1.23456789).build()

        assert result.execution_time == 1.23456789

        engine_dict = result.to_engine_dict()
        assert engine_dict["execution_time"] == 1.23456789

    def test_zero_execution_time(self, builder: TestDataBuilder) -> None:
        """Test zero execution time handling"""
        result = builder.result().with_timing(0.0).build()

        assert result.execution_time == 0.0

        detailed_message = result.get_detailed_message()
        # Should handle zero time gracefully
        assert isinstance(detailed_message, str)

    def test_negative_execution_time_edge_case(self, builder: TestDataBuilder) -> None:
        """Negative execution time (clock issues)"""
        # Some systems might have clock issues leading to negative times
        result = builder.result().with_timing(-0.1).build()

        # Should preserve the value (might be valid in some contexts)
        assert result.execution_time == -0.1

    @pytest.mark.parametrize(
        "execution_time",
        [
            0.001,  # Very small time
            0.999,  # Just under 1 second
            1.0,  # Exactly 1 second
            60.0,  # 1 minute
            3600.0,  # 1 hour
            86400.0,  # 1 day
        ],
    )
    def test_various_execution_times(
        self, builder: TestDataBuilder, execution_time: float
    ) -> None:
        """Test various execution time values"""
        result = builder.result().with_timing(execution_time).build()

        assert result.execution_time == execution_time

        detailed = result.get_detailed_message()
        assert str(execution_time) in detailed or f"{execution_time:.3f}" in detailed

    def test_timestamp_consistency(self, builder: TestDataBuilder) -> None:
        """Test timestamp consistency in results"""
        result = ExecutionResultSchema.create_success_result(
            rule_id="timestamp_test", entity_name="test_table", total_count=100
        )

        assert result.started_at is not None
        assert result.ended_at is not None

        # Should be reasonable timestamps (within last few seconds)
        current_time = now()
        time_diff = abs((current_time - result.started_at).total_seconds())
        assert time_diff < 60  # Should be within 1 minute


class TestResultSchemaPropertyBased:
    """👻 Property-based testing for comprehensive validation"""

    @staticmethod
    @composite
    def result_data(draw: DrawFn) -> dict[str, Any]:
        """Generate random but valid result data"""
        total_records = draw(st.integers(min_value=0, max_value=1000000))
        failed_records = draw(st.integers(min_value=0, max_value=total_records))
        execution_time = draw(
            st.floats(
                min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False
            )
        )
        return {
            "total_records": total_records,
            "failed_records": failed_records,
            "execution_time": execution_time,
        }

    @given(result_data())
    def test_success_rate_invariants(self, data: Dict[str, Any]) -> None:
        """Property-based test for success rate invariants"""
        result = ExecutionResultSchema.create_success_result(
            rule_id="property_test",
            entity_name="test_table",
            total_count=data["total_records"],
            error_count=data["failed_records"],
            execution_time=data["execution_time"],
        )
        success_rate = result.get_success_rate()
        failure_rate = result.get_failure_rate()
        # Invariants that must always hold
        assert 0.0 <= success_rate <= 1.0
        assert 0.0 <= failure_rate <= 1.0
        assert abs((success_rate + failure_rate) - 1.0) < 1e-10
        # If no failures, success rate should be 1.0
        if data["failed_records"] == 0:
            assert success_rate == 1.0
        # If all records failed, success rate should be 0.0
        if (
            data["total_records"] > 0
            and data["failed_records"] == data["total_records"]
        ):
            assert success_rate == 0.0

    @given(st.text(min_size=1, max_size=100))
    def test_rule_id_handling(self, rule_id: str) -> None:
        """Property-based test for rule ID handling"""
        assume(rule_id.strip())  # Skip empty or whitespace-only strings
        result = ExecutionResultSchema.create_success_result(
            rule_id=rule_id, entity_name="test_table", total_count=100
        )
        assert result.rule_id == rule_id
        engine_dict = result.to_engine_dict()
        assert engine_dict["rule_id"] == rule_id


class TestResultSchemaContractCompliance:
    """👻 Contract testing - Builder pattern and interface compliance"""

    def test_result_builder_contract(self, builder: TestDataBuilder) -> None:
        """Test result builder contract compliance"""
        # Builder should support method chaining
        result = (
            builder.result()
            .with_rule("contract_test", "Contract Test Rule")
            .with_entity("contract_db.contract_table")
            .with_counts(failed_records=10, total_records=100)
            .with_timing(1.5)
            .with_status(ExecutionStatus.FAILED.value)
            .with_message("Contract test message")
            .build()
        )

        assert result.rule_id == "contract_test"
        assert result.total_count == 100
        assert result.error_count == 10
        assert result.execution_time == 1.5
        assert result.status == ExecutionStatus.FAILED.value

    def test_builder_reset_behavior(self, builder: TestDataBuilder) -> None:
        """Builder reset behavior"""
        # First build
        result1 = (
            builder.result()
            .with_rule("test1")
            .with_counts(failed_records=5, total_records=50)
            .build()
        )
        # Second build should not be affected by first
        result2 = (
            builder.result()
            .with_rule("test2")
            .with_counts(failed_records=10, total_records=100)
            .build()
        )

        assert result1.rule_id == "test1"
        assert result2.rule_id == "test2"
        assert result1.total_count == 50
        assert result2.total_count == 100

    def test_result_schema_base_compliance(self, builder: TestDataBuilder) -> None:
        """Test compliance with ExecutionResultBase interface"""
        result = builder.result().build()

        # Should inherit from ExecutionResultBase
        assert isinstance(result, ExecutionResultBase)

        # Should have all required base properties
        assert hasattr(result, "rule_id")
        assert hasattr(result, "status")
        assert hasattr(result, "dataset_metrics")
        assert hasattr(result, "execution_time")
        assert hasattr(result, "total_count")
        assert hasattr(result, "error_count")


class TestResultSchemaEdgeCasesAndBoundaries:
    """👻 Edge cases and boundary conditions"""

    def test_extremely_large_numbers(self, builder: TestDataBuilder) -> None:
        """Extremely large record counts"""
        large_number = 999_999_999
        result = ExecutionResultSchema.create_success_result(
            rule_id="large_test",
            entity_name="massive_table",
            total_count=large_number,
            error_count=large_number // 2,
        )

        assert result.total_count == large_number
        assert result.error_count == large_number // 2

        # Success rate calculation should still work
        success_rate = result.get_success_rate()
        assert 0.0 <= success_rate <= 1.0

    def test_unicode_and_special_characters(self, builder: TestDataBuilder) -> None:
        """Test Unicode and special characters in various fields"""
        unicode_rule_id = "规则_测试_🔍"
        unicode_entity = "测试数据库.用户表"
        unicode_message = "测试消息 with émojis 🎯✅❌"
        result = ExecutionResultSchema.create_success_result(
            rule_id=unicode_rule_id,
            entity_name=unicode_entity,
            total_count=100,
            message=unicode_message,
        )

        assert result.rule_id == unicode_rule_id
        assert result.execution_message == unicode_message

        # Should be JSON serializable
        engine_dict = result.to_engine_dict()
        json_str = json.dumps(engine_dict, ensure_ascii=False)
        assert unicode_rule_id in json_str

    def test_precision_edge_cases(self, builder: TestDataBuilder) -> None:
        """Floating point precision edge cases"""
        # Test with numbers that cause floating point precision issues
        result = ExecutionResultSchema.create_success_result(
            rule_id="precision_test",
            entity_name="precision_table",
            total_count=7,
            error_count=1,
            execution_time=0.1 + 0.2,  # This is famously 0.30000000000000004
        )

        success_rate = result.get_success_rate()
        expected_rate = 6.0 / 7.0

        # Should be close enough for practical purposes
        assert abs(success_rate - expected_rate) < 1e-10

    def test_empty_string_handling(self, builder: TestDataBuilder) -> None:
        """Empty string handling in various fields"""
        # Empty rule ID is actually allowed by the current implementation
        # This is a potential validation gap that should be documented
        result = ExecutionResultSchema.create_success_result(
            rule_id="", entity_name="test_table", total_count=100
        )
        assert result.rule_id == ""
        # 👻 GHOST DISCOVERY: Empty entity name is also allowed
        # This represents a validation gap in the current implementation
        result2 = ExecutionResultSchema.create_success_result(
            rule_id="test_rule", entity_name="", total_count=100
        )
        assert result2.dataset_metrics[0].entity_name == ""

    def test_null_handling_comprehensive(self, builder: TestDataBuilder) -> None:
        """Test null/None handling in optional fields"""
        result = ExecutionResultSchema.create_success_result(
            rule_id="null_test",
            entity_name="test_table",
            total_count=100,
            message=None,  # Explicitly null message
        )
        #  GHOST DISCOVERY: Even with None message,
        #  create_success_result generates a default message
        # This is by design - the factory method provides helpful defaults
        assert result.execution_message is not None
        assert "Validation completed, success rate 100.0%" in result.execution_message
        engine_dict = result.to_engine_dict()
        assert engine_dict["message"] is not None


# Test fixtures
@pytest.fixture
def builder() -> TestDataBuilder:
    """Provide TestDataBuilder instance"""
    return TestDataBuilder()
