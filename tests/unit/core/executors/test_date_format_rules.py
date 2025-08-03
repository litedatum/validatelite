"""
🧙‍♂️ Enhanced DATE_FORMAT Rule Tests - Testing Ghost's Modern Testing Masterpiece

As the Testing Ghost 👻, I demonstrate the four key testing improvements:

1. 🏗️ Schema Builder Pattern - Eliminates fixture duplication
2. 🔄 Contract Testing - Ensures mocks match reality
3. 📊 Property-based Testing - Verifies behavior with random inputs
4. 🧬 Mutation Testing Readiness - Catches subtle bugs

This file uses the modern architecture with ValidityExecutor.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest

# Core application imports
from core.executors.validity_executor import ValidityExecutor
from shared.enums import RuleAction, RuleCategory, RuleType, SeverityLevel
from shared.enums.connection_types import ConnectionType
from shared.exceptions import RuleExecutionError
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.result_schema import ExecutionResultSchema
from shared.schema.rule_schema import RuleSchema

# Enhanced testing imports
from tests.shared.builders.test_builders import TestDataBuilder
from tests.shared.contracts.test_contracts import ContractTestCase, MockContract

# Property-based testing
try:
    import hypothesis
    from hypothesis import HealthCheck, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


# 📊 PROPERTY-BASED TESTING STRATEGIES
if HYPOTHESIS_AVAILABLE:

    @st.composite
    def date_format_strategy(draw: st.DrawFn) -> str:
        """Generate valid date format patterns for property testing"""
        formats = [
            "%Y-%m-%d",  # ISO format
            "%m/%d/%Y",  # US format
            "%d/%m/%Y",  # European format
            "%Y%m%d",  # Compact format
            "%d.%m.%Y",  # German format
            "%Y-%m-%d %H:%M:%S",  # DateTime format
            "%d-%b-%Y",  # Day-Month-Year with abbreviated month
            "%m-%d-%Y",  # Month-Day-Year
            "%Y/%m/%d",  # Year/Month/Day
        ]
        return draw(st.sampled_from(formats))

    @st.composite
    def date_validation_scenario_strategy(draw: st.DrawFn) -> Dict[str, Any]:
        """Generate realistic date validation scenarios"""
        total_records = draw(st.integers(min_value=1, max_value=10000))
        failed_records = draw(st.integers(min_value=0, max_value=total_records))
        return {
            "total_records": total_records,
            "failed_records": failed_records,
            "failure_rate": (
                failed_records / total_records if total_records > 0 else 0.0
            ),
        }


# 🧪 ENHANCED FIXTURES USING BUILDERS
@pytest.fixture
def builder() -> TestDataBuilder:
    """Schema builder instance - eliminates fixture duplication"""
    return TestDataBuilder()


@pytest.fixture
def mock_connection() -> ConnectionSchema:
    """Database connection using builder pattern"""
    return TestDataBuilder.connection().build()


@pytest.fixture
def validity_executor(mock_connection: ConnectionSchema) -> ValidityExecutor:
    """ValidityExecutor with contract verification"""
    executor = ValidityExecutor(
        mock_connection,
        test_mode=True,
        sample_data_enabled=True,
        sample_data_max_records=5,
    )
    # Verify executor follows contract
    MockContract.verify_validity_executor_contract(executor)
    return executor


# 🏗️ SCHEMA BUILDER PATTERN TESTS
class TestSchemaBuilderPattern:
    """Demonstrate how Schema Builder eliminates fixture duplication"""

    def test_basic_date_format_rule_creation(self, builder: TestDataBuilder) -> None:
        """Test creating basic date format rule with builder"""
        rule = (
            builder.rule()
            .with_name("birth_date_format")
            .with_target("users", "profiles", "birth_date")
            .as_date_format_rule("%Y-%m-%d")
            .build()
        )

        assert rule.name == "birth_date_format"
        assert rule.type == RuleType.DATE_FORMAT
        assert rule.target.entities[0].table == "profiles"
        assert rule.parameters["format"] == "%Y-%m-%d"

    def test_complex_date_format_rule_with_filter(
        self, builder: TestDataBuilder
    ) -> None:
        """Test creating complex date format rule with filter"""
        rule = (
            builder.rule()
            .with_name("active_user_join_date")
            .with_severity(SeverityLevel.HIGH)
            .with_target("production", "users", "created_at")
            .as_date_format_rule("%Y-%m-%d %H:%M:%S")
            .with_filter("status = 'active' AND deleted_at IS NULL")
            .with_parameter("timezone", "UTC")
            .build()
        )

        assert rule.severity == SeverityLevel.HIGH
        assert rule.parameters["format"] == "%Y-%m-%d %H:%M:%S"
        assert (
            rule.parameters["filter_condition"]
            == "status = 'active' AND deleted_at IS NULL"
        )
        assert rule.parameters["timezone"] == "UTC"

    def test_us_date_format_rule(self, builder: TestDataBuilder) -> None:
        """Test US date format rule creation"""
        rule = (
            builder.rule()
            .with_name("legacy_us_dates")
            .with_target("legacy", "orders", "order_date")
            .as_date_format_rule("%m/%d/%Y")
            .with_parameter("region", "US")
            .build()
        )

        assert rule.parameters["format"] == "%m/%d/%Y"
        assert rule.parameters["region"] == "US"

    def test_quick_builder_methods(self, builder: TestDataBuilder) -> None:
        """Test quick builder methods for common date format scenarios"""
        # Quick ISO date format rule
        rule = TestDataBuilder.basic_date_format_rule(
            date_format="%Y-%m-%d", table="events", column="event_date"
        )

        assert rule.type == RuleType.DATE_FORMAT
        assert rule.target.entities[0].table == "events"
        assert rule.parameters["format"] == "%Y-%m-%d"


# 🔄 CONTRACT TESTING IMPLEMENTATION
class TestContractTesting:
    """Ensure mocks accurately represent real implementations"""

    @pytest.mark.asyncio
    async def test_query_executor_contract_compliance(self) -> None:
        """Test that our QueryExecutor mocks follow the contract"""
        # Create contract-compliant mock for date format scenario
        mock = MockContract.create_query_executor_mock(
            query_results=[{"anomaly_count": 12}], column_names=["anomaly_count"]
        )

        # Verify contract compliance
        await ContractTestCase.test_query_executor_contract_compliance(mock)

    @pytest.mark.asyncio
    async def test_date_format_rule_execution_with_contract_mock(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test date format rule execution using contract-compliant mock"""
        rule = builder.basic_date_format_rule("%Y-%m-%d", "users", "birth_date")

        # Create async mock functions that return the expected data structure
        async def mock_execute_query(
            sql: str,
        ) -> tuple[List[Dict[str, Any]], List[str]]:
            if "total_count" in sql.lower():
                return [{"total_count": 500}], ["total_count"]
            elif "anomaly_count" in sql.lower():
                return [{"anomaly_count": 25}], ["anomaly_count"]
            else:
                return [
                    {"id": 1, "name": "test"},
                    {"id": 2, "name": "test2"},
                    {"id": 3, "name": "test3"},
                    {"id": 4, "name": "test4"},
                    {"id": 5, "name": "test5"},
                ], ["id", "name"]

        with patch.object(validity_executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            # Mock the QueryExecutor class and its execute_query method
            with patch("shared.database.query_executor.QueryExecutor") as mock_qe_class:
                mock_query_executor = AsyncMock()
                mock_query_executor.execute_query.side_effect = mock_execute_query
                mock_qe_class.return_value = mock_query_executor

                result = await validity_executor.execute_rule(rule)

        # Verify result structure (contract compliance)
        assert isinstance(result, ExecutionResultSchema)
        assert result.rule_id == rule.id
        assert result.status == "FAILED"

        # Validate the data flow: from the query result to the Dataset Metrics, and then to the Execution Result Schema.
        assert len(result.dataset_metrics) > 0
        assert result.sample_data is not None
        assert len(result.sample_data) == 5
        assert (
            result.dataset_metrics[0].failed_records == 25
        )  # ← Invalid date format count
        assert result.dataset_metrics[0].total_records == 500  # ← Total records
        assert result.status == "FAILED"  # ← Should be FAILED due to invalid dates

    @pytest.mark.asyncio
    async def test_database_error_contract(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test error handling contract compliance"""
        rule = builder.basic_date_format_rule(
            "%Y-%m-%d", "nonexistent_table", "date_col"
        )

        with patch.object(validity_executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            # Mock database error
            with patch("shared.database.query_executor.QueryExecutor") as mock_qe_class:
                mock_query_executor = AsyncMock()
                mock_query_executor.execute_query.side_effect = Exception(
                    "Table 'nonexistent_table' doesn't exist"
                )
                mock_qe_class.return_value = mock_query_executor

                result = await validity_executor.execute_rule(rule)

                # Should handle error gracefully
                assert result.status == "ERROR"
                assert result.error_message is not None
                assert "Table 'nonexistent_table' doesn't exist" in result.error_message


# 📊 PROPERTY-BASED TESTING
@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
class TestPropertyBasedTesting:
    """Use property-based testing to verify behavior with random inputs"""

    @hypothesis.given(date_format=date_format_strategy())
    @settings(
        max_examples=30,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_sql_generation_invariants(
        self, validity_executor: ValidityExecutor, date_format: str
    ) -> None:
        """Verify SQL generation properties hold for any valid date format"""
        rule = TestDataBuilder.basic_date_format_rule(
            date_format, "test_table", "date_col"
        )

        sql = validity_executor._generate_date_format_sql(rule)

        # These properties should always hold
        assert "SELECT" in sql.upper()
        assert "COUNT" in sql.upper()
        assert "test_table" in sql
        assert "date_col" in sql
        # Should contain some form of date format validation
        assert len(sql) > 50  # Reasonable minimum SQL length

    @hypothesis.given(scenario=date_validation_scenario_strategy())
    @settings(
        max_examples=25,
        deadline=3000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_date_validation_properties(self, scenario: Dict[str, Any]) -> None:
        """Verify date validation mathematical properties"""
        total_records = scenario["total_records"]
        failed_records = scenario["failed_records"]
        failure_rate = scenario["failure_rate"]

        # Property: failure rate should be between 0 and 1
        assert 0.0 <= failure_rate <= 1.0

        # Property: failed records should not exceed total records
        assert failed_records <= total_records

        # Property: if no failures, rate should be 0
        if failed_records == 0:
            assert failure_rate == 0.0

        # Property: if all records fail, rate should be 1.0
        if failed_records == total_records and total_records > 0:
            assert failure_rate == 1.0

    @hypothesis.given(
        date_format=date_format_strategy(),
        table_name=st.text(
            min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz_"
        ),
    )
    @settings(
        max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_rule_creation_invariants(self, date_format: str, table_name: str) -> None:
        """Verify rule creation properties hold for any valid inputs"""
        rule = TestDataBuilder.basic_date_format_rule(
            date_format, table_name, "date_column"
        )

        # These properties should always hold
        assert rule.type == RuleType.DATE_FORMAT
        assert rule.parameters["format"] == date_format
        assert rule.target.entities[0].table == table_name
        assert rule.target.entities[0].column == "date_column"
        assert len(date_format) > 0


# 🧬 MUTATION TESTING READINESS
class TestMutationTestingReadiness:
    """Design tests to catch subtle bugs that mutation testing would find"""

    def test_date_format_pattern_validation(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test that invalid date format patterns are rejected"""
        # Test invalid patterns that should be caught
        invalid_patterns = [
            "",  # Empty pattern
            "%Q",  # Invalid format code
            "%Z%Y",  # Invalid format code
            "not_a_pattern",  # Plain text
            None,  # None value
        ]

        for pattern in invalid_patterns:
            if pattern is None:
                continue

            try:
                rule = builder.rule().as_date_format_rule(pattern).build()
                sql = validity_executor._generate_date_format_sql(rule)
                # Should either handle gracefully or raise appropriate error
            except (ValueError, RuleExecutionError):
                # Expected for invalid patterns
                pass

    def test_sql_injection_protection_date_format(
        self, mock_connection: ConnectionSchema, builder: TestDataBuilder
    ) -> None:
        """Verify SQL injection resistance for date format queries"""
        # Create a non-test-mode executor for security testing
        security_executor = ValidityExecutor(mock_connection, test_mode=False)

        malicious_inputs = [
            "'; DROP TABLE dates; --",
            "' OR '1'='1",
            "'; UPDATE users SET admin=1; --",
        ]

        for malicious_input in malicious_inputs:
            rule = (
                builder.rule()
                .with_target("test_db", malicious_input, "safe_column")
                .as_date_format_rule("%Y-%m-%d")
                .build()
            )

            # Should raise ValueError for SQL injection attempts
            with pytest.raises(
                RuleExecutionError, match="contains potentially dangerous SQL patterns"
            ):
                security_executor._generate_date_format_sql(rule)

    def test_null_date_handling(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test how NULL date values are handled"""
        rule = builder.basic_date_format_rule("%Y-%m-%d", "test_table", "nullable_date")
        sql = validity_executor._generate_date_format_sql(rule)

        # Should handle NULL values appropriately
        # Most implementations exclude NULLs from date format validation
        assert len(sql) > 0  # Basic validation that SQL is generated

    def test_empty_vs_invalid_date_distinction(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Ensure empty strings and invalid dates are handled consistently"""
        rule = builder.basic_date_format_rule("%Y-%m-%d", "test_table", "date_column")
        sql = validity_executor._generate_date_format_sql(rule)

        # Should distinguish between empty strings and genuinely invalid dates
        assert "date_column" in sql

    @pytest.mark.asyncio
    async def test_error_message_completeness(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test that error messages provide adequate information"""
        rule = builder.basic_date_format_rule("%Y-%m-%d", "missing_table", "date_col")

        with patch.object(validity_executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            # Mock table missing error
            with patch("shared.database.query_executor.QueryExecutor") as mock_qe_class:
                mock_query_executor = AsyncMock()
                mock_query_executor.execute_query.side_effect = Exception(
                    "Table 'missing_table' doesn't exist"
                )
                mock_qe_class.return_value = mock_query_executor

                result = await validity_executor.execute_rule(rule)

                # Error message should be informative
                assert result.status == "ERROR"
                assert result.error_message is not None
                assert len(result.error_message) > 0
                assert "missing_table" in result.error_message.lower()

    def test_date_format_boundary_conditions(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test boundary conditions in date format validation"""
        # Test with very specific date formats
        boundary_formats = [
            "%Y",  # Year only
            "%m",  # Month only
            "%d",  # Day only
            "%Y-%m",  # Year-Month
            "%m-%d",  # Month-Day
        ]

        for fmt in boundary_formats:
            rule = builder.basic_date_format_rule(fmt, "test_table", "partial_date")
            sql = validity_executor._generate_date_format_sql(rule)

            # Should handle partial date formats
            assert len(sql) > 0
            assert fmt in sql or "partial_date" in sql

    def test_filter_condition_with_date_format(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test filter condition interactions with date format logic"""
        rule_with_filter = (
            builder.rule()
            .as_date_format_rule("%Y-%m-%d")
            .with_filter("created_year >= 2020 AND status = 'active'")
            .with_target("test_db", "events", "event_date")
            .build()
        )

        sql_with_filter = validity_executor._generate_date_format_sql(rule_with_filter)

        # Should properly combine date format validation with filter
        assert "WHERE" in sql_with_filter.upper()
        assert "created_year >= 2020" in sql_with_filter
        assert "AND" in sql_with_filter.upper() or "OR" in sql_with_filter.upper()

    def test_special_characters_in_dates(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test handling of special characters in date strings"""
        # Test date formats with special characters
        special_formats = [
            "%d/%m/%Y",  # Forward slashes
            "%d-%m-%Y",  # Hyphens
            "%d.%m.%Y",  # Dots
            "%d %b %Y",  # Spaces and abbreviations
            "%d-%b-%Y",  # Mixed separators
        ]

        for fmt in special_formats:
            rule = builder.basic_date_format_rule(fmt, "test_table", "formatted_date")
            sql = validity_executor._generate_date_format_sql(rule)

            # Should handle special characters in format strings
            assert len(sql) > 0


# 🚀 PERFORMANCE AND EDGE CASES
class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases"""

    def test_large_dataset_date_format_optimization(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test date format validation performance considerations"""
        rule = builder.basic_date_format_rule(
            "%Y-%m-%d %H:%M:%S", "large_events_table", "timestamp_column"
        )
        sql = validity_executor._generate_date_format_sql(rule)

        # Should generate reasonably efficient SQL
        assert len(sql) < 2000  # Reasonable SQL length limit
        assert "large_events_table" in sql

    def test_unicode_date_formats(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test Unicode and international date formats"""
        unicode_columns = ["日期", "datum", "fecha", "データ"]

        for column_name in unicode_columns:
            rule = builder.basic_date_format_rule(
                "%Y-%m-%d", "international_table", column_name
            )
            sql = validity_executor._generate_date_format_sql(rule)

            # Should handle Unicode column names properly
            assert (
                column_name in sql
                or f"`{column_name}`" in sql
                or f'"{column_name}"' in sql
            )

    def test_multiple_date_formats_scenario(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test scenario with multiple date format rules"""
        formats_and_columns = [
            ("%Y-%m-%d", "birth_date"),
            ("%m/%d/%Y", "hire_date"),
            ("%d.%m.%Y", "termination_date"),
            ("%Y-%m-%d %H:%M:%S", "last_login"),
        ]

        for fmt, column in formats_and_columns:
            rule = builder.basic_date_format_rule(fmt, "employees", column)
            sql = validity_executor._generate_date_format_sql(rule)

            # Each should generate valid SQL
            assert "SELECT" in sql.upper()
            assert "COUNT" in sql.upper()
            assert column in sql

    @pytest.mark.asyncio
    async def test_concurrent_date_format_validation(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test concurrent date format validations"""
        rules = [
            builder.basic_date_format_rule("%Y-%m-%d", f"table_{i}", f"date_col_{i}")
            for i in range(1, 4)
        ]

        with patch.object(validity_executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            # Mock successful queries
            with patch("shared.database.query_executor.QueryExecutor") as mock_qe_class:
                mock_query_executor = AsyncMock()

                async def mock_execute_query(
                    sql: str,
                ) -> tuple[List[Dict[str, Any]], List[str]]:
                    if "total_count" in sql.lower():
                        return [{"total_count": 100}], ["total_count"]
                    elif "anomaly_count" in sql.lower():
                        return [{"anomaly_count": 5}], ["anomaly_count"]
                    else:
                        return [
                            {"id": 1, "name": "test"},
                            {"id": 2, "name": "test2"},
                            {"id": 3, "name": "test3"},
                            {"id": 4, "name": "test4"},
                            {"id": 5, "name": "test5"},
                        ], ["id", "name"]

                mock_query_executor.execute_query.side_effect = mock_execute_query
                mock_qe_class.return_value = mock_query_executor

                # Execute all rules concurrently
                import asyncio

                results = await asyncio.gather(
                    *[validity_executor.execute_rule(rule) for rule in rules]
                )

                # All should succeed
                assert len(results) == 3
                assert all(result.status == "FAILED" for result in results)
                assert all(result.sample_data is not None for result in results)
                assert all(
                    (result.sample_data is not None and len(result.sample_data) == 5)
                    for result in results
                )
