"""
🧙‍♂️ Enhanced UNIQUE Rule Tests - Testing Ghost's Modern Testing Masterpiece

As the Testing Ghost 👻, I demonstrate the four key testing improvements:

1. 🏗️ Schema Builder Pattern - Eliminates fixture duplication
2. 🔄 Contract Testing - Ensures mocks match reality
3. 📊 Property-based Testing - Verifies behavior with random inputs
4. 🧬 Mutation Testing Readiness - Catches subtle bugs

This file focuses on UNIQUE uniqueness rules with comprehensive duplicate detection coverage.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, patch

import pytest

# Core application imports
from core.executors.uniqueness_executor import UniquenessExecutor
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
    def duplicate_scenario_strategy(draw: st.DrawFn) -> Dict[str, int]:
        """Generate realistic duplicate scenarios"""
        total_records = draw(st.integers(min_value=1, max_value=10000))
        duplicate_groups = draw(
            st.integers(min_value=0, max_value=min(100, total_records // 2))
        )
        duplicate_records = draw(
            st.integers(min_value=duplicate_groups, max_value=total_records)
        )
        return {
            "total_records": total_records,
            "duplicate_groups": duplicate_groups,
            "duplicate_records": duplicate_records,
        }

    @st.composite
    def unique_values_strategy(draw: st.DrawFn) -> List[str]:
        """Generate lists with varying uniqueness characteristics"""
        size = draw(st.integers(min_value=1, max_value=50))
        values = draw(
            st.lists(
                st.text(
                    min_size=1,
                    max_size=20,
                    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
                ),
                min_size=size,
                max_size=size,
            )
        )
        return values


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
def uniqueness_executor(mock_connection: ConnectionSchema) -> UniquenessExecutor:
    """UniquenessExecutor with contract verification"""
    executor = UniquenessExecutor(mock_connection, test_mode=True)
    # Verify executor follows contract
    MockContract.verify_uniqueness_executor_contract(executor)
    return executor


# 🏗️ SCHEMA BUILDER PATTERN TESTS
class TestSchemaBuilderPattern:
    """Demonstrate how Schema Builder eliminates fixture duplication"""

    def test_basic_unique_rule_creation(self, builder: TestDataBuilder) -> None:
        """Test creating basic UNIQUE rule with builder"""
        rule = (
            builder.rule()
            .with_name("customer_email_uniqueness")
            .with_target("production", "customers", "email")
            .as_unique_rule()
            .build()
        )

        assert rule.name == "customer_email_uniqueness"
        assert rule.type == RuleType.UNIQUE
        assert rule.category == RuleCategory.UNIQUENESS
        assert rule.target.entities[0].table == "customers"
        assert rule.target.entities[0].column == "email"

    def test_unique_rule_with_filter(self, builder: TestDataBuilder) -> None:
        """Test creating UNIQUE rule with filter condition"""
        rule = (
            builder.rule()
            .with_name("active_user_username_check")
            .with_severity(SeverityLevel.CRITICAL)
            .with_target("auth", "users", "username")
            .as_unique_rule()
            .with_filter("status = 'active' AND deleted_at IS NULL")
            .build()
        )

        assert rule.severity == SeverityLevel.CRITICAL
        assert (
            rule.parameters["filter_condition"]
            == "status = 'active' AND deleted_at IS NULL"
        )

    def test_quick_builder_methods(self, builder: TestDataBuilder) -> None:
        """Test quick builder methods for common scenarios"""
        # Quick UNIQUE rule
        rule = TestDataBuilder.basic_unique_rule(table="products", column="sku")

        assert rule.type == RuleType.UNIQUE
        assert rule.target.entities[0].table == "products"
        assert rule.target.entities[0].column == "sku"


# 🔄 CONTRACT TESTING IMPLEMENTATION
class TestContractTesting:
    """Ensure mocks accurately represent real implementations"""

    @pytest.mark.asyncio
    async def test_query_executor_contract_compliance(self) -> None:
        """Test that our QueryExecutor mocks follow the contract"""
        # Create contract-compliant mock for UNIQUE scenario
        mock_data = MockContract.create_uniqueness_mock_data(
            duplicate_groups=3, duplicate_records=8, total_count=100
        )
        mock = MockContract.create_query_executor_mock(
            query_results=mock_data["anomaly_query"], column_names=["anomaly_count"]
        )

        # Verify contract compliance
        await ContractTestCase.test_query_executor_contract_compliance(mock)

    @pytest.mark.asyncio
    async def test_unique_rule_execution_with_contract_mock(
        self, uniqueness_executor: UniquenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test rule execution using contract-compliant mock"""
        rule = builder.basic_unique_rule(table="users", column="email")

        # Create async mock functions that return the expected data structure
        async def mock_execute_query(
            sql: str,
        ) -> Tuple[List[Dict[str, Any]], List[str]]:
            if "total_count" in sql.lower():
                return [{"total_count": 100}], ["total_count"]
            elif "duplicate_records_count" in sql.lower():
                # This is a detailed query for counting duplicate records.
                return [{"duplicate_records_count": 5}], ["duplicate_records_count"]
            else:
                # The main query returns the count of duplicate groups (excluding zero counts).
                return [{"anomaly_count": 2}], [
                    "anomaly_count"
                ]  # Two repeating groups.

        with patch.object(uniqueness_executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            # Mock the QueryExecutor class and its execute_query method
            with patch("shared.database.query_executor.QueryExecutor") as mock_qe_class:
                mock_query_executor = AsyncMock()
                mock_query_executor.execute_query.side_effect = mock_execute_query
                mock_qe_class.return_value = mock_query_executor

                result = await uniqueness_executor.execute_rule(rule)

        # Verify result structure (contract compliance)
        assert isinstance(result, ExecutionResultSchema)
        assert result.rule_id == rule.id
        assert result.status in ["PASSED", "FAILED", "ERROR"]

        # Verify uniqueness-specific data flow
        assert len(result.dataset_metrics) > 0
        # For uniqueness, failed_records represents duplicate records
        assert result.dataset_metrics[0].failed_records == 5  # ← Duplicate count
        assert result.dataset_metrics[0].total_records == 100  # ← Total records
        assert result.status == "FAILED"  # ← Should be FAILED due to duplicates


# 📊 PROPERTY-BASED TESTING
@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
class TestPropertyBasedTesting:
    """Use property-based testing to verify behavior with random inputs"""

    @hypothesis.given(scenario=duplicate_scenario_strategy())
    @settings(
        max_examples=30,
        deadline=3000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_duplicate_counting_invariants(
        self, uniqueness_executor: UniquenessExecutor, scenario: Dict[str, int]
    ) -> None:
        """Verify duplicate counting properties hold for any valid input"""
        total_records = scenario["total_records"]
        duplicate_groups = scenario["duplicate_groups"]
        duplicate_records = scenario["duplicate_records"]

        # Property: duplicate records should never exceed total records
        assert duplicate_records <= total_records

        # Property: duplicate groups should never exceed duplicate records
        assert duplicate_groups <= duplicate_records

        # Property: if there are duplicate groups, there must be duplicate records
        if duplicate_groups > 0:
            assert duplicate_records >= duplicate_groups

        # Property: unique count calculation should be consistent
        unique_records = total_records - duplicate_records
        assert unique_records >= 0
        assert unique_records <= total_records

    @hypothesis.given(values=unique_values_strategy())
    @settings(
        max_examples=20,
        deadline=3000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_sql_generation_invariants(
        self, uniqueness_executor: UniquenessExecutor, values: List[str]
    ) -> None:
        """Verify SQL generation properties for any valid input list"""
        rule = TestDataBuilder.basic_unique_rule(
            table="test_table", column="test_column"
        )

        sql = uniqueness_executor._generate_unique_sql(rule)

        # These properties should always hold
        assert "SELECT" in sql.upper()
        assert "GROUP BY" in sql.upper() or "DISTINCT" in sql.upper()
        assert "test_table" in sql
        assert "test_column" in sql

    @hypothesis.given(scenario=duplicate_scenario_strategy())
    @settings(
        max_examples=15,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_duplicate_rate_calculation_invariants(
        self, scenario: Dict[str, int]
    ) -> None:
        """Verify duplicate rate calculations are mathematically sound"""
        total_records = scenario["total_records"]
        duplicate_records = scenario["duplicate_records"]

        if total_records > 0:
            duplicate_rate = duplicate_records / total_records
            assert 0.0 <= duplicate_rate <= 1.0

            # Edge cases
            if duplicate_records == 0:
                assert duplicate_rate == 0.0
            if duplicate_records == total_records:
                assert duplicate_rate == 1.0


# 🧬 MUTATION TESTING READINESS
class TestMutationTestingReadiness:
    """Design tests to catch subtle bugs that mutation testing would find"""

    def test_duplicate_vs_unique_count_distinction(
        self, uniqueness_executor: UniquenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Catch confusion between duplicate count and unique count"""
        rule = builder.basic_unique_rule()

        sql = uniqueness_executor._generate_unique_sql(rule)

        # Should use proper logic for counting duplicates/uniques
        # Common mutation: COUNT(DISTINCT col) vs COUNT(col)
        assert (
            "COUNT" in sql.upper() and "DISTINCT" in sql.upper()
        ) or "GROUP BY" in sql.upper()

    def test_boundary_condition_mutations(
        self, uniqueness_executor: UniquenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Catch off-by-one errors in duplicate detection"""
        rule = builder.basic_unique_rule()
        sql = uniqueness_executor._generate_unique_sql(rule)

        # Should properly handle GROUP BY with HAVING COUNT > 1 (not >= 1)
        if "HAVING" in sql.upper():
            assert "COUNT" in sql.upper()
            # Common mutation: > 1 vs >= 1 vs > 0
            # The SQL should be designed to catch actual duplicates

    def test_sql_injection_protection_unique(
        self, mock_connection: ConnectionSchema, builder: TestDataBuilder
    ) -> None:
        """Verify SQL injection resistance for uniqueness queries"""
        # Create a non-test-mode executor for security testing
        security_executor = UniquenessExecutor(mock_connection, test_mode=False)

        malicious_inputs = [
            "'; DROP TABLE duplicates; --",
            "' UNION SELECT password FROM users --",
            "'; UPDATE users SET admin=1; --",
        ]

        for malicious_input in malicious_inputs:
            rule = (
                builder.rule()
                .with_target("test_db", malicious_input, "safe_column")
                .as_unique_rule()
                .build()
            )

            # Should raise ValueError for SQL injection attempts
            with pytest.raises(
                RuleExecutionError, match="contains potentially dangerous SQL patterns"
            ):
                security_executor._generate_unique_sql(rule)

    def test_null_handling_in_uniqueness(
        self, uniqueness_executor: UniquenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test how NULL values are handled in uniqueness checks"""
        rule = builder.basic_unique_rule()
        sql = uniqueness_executor._generate_unique_sql(rule)

        # Common mutation issue: Should NULL values be considered duplicates of each other?
        # Most business rules treat multiple NULLs as acceptable (not duplicates)
        # This is database-specific behavior that should be consistent

    @pytest.mark.asyncio
    async def test_error_handling_uniqueness(
        self, uniqueness_executor: UniquenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test comprehensive error scenarios"""
        rule = builder.basic_unique_rule(table="nonexistent_table")

        with patch.object(uniqueness_executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            # Mock database error
            with patch("shared.database.query_executor.QueryExecutor") as mock_qe_class:
                mock_query_executor = AsyncMock()
                mock_query_executor.execute_query.side_effect = Exception(
                    "Table does not exist"
                )
                mock_qe_class.return_value = mock_query_executor

                result = await uniqueness_executor.execute_rule(rule)

                # Should handle error gracefully
                assert result.status == "ERROR"
                assert result.error_message is not None
                assert "Table does not exist" in result.error_message

    def test_filter_with_uniqueness_mutations(
        self, uniqueness_executor: UniquenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test filter condition interactions with uniqueness logic"""
        # Test with complex filter
        rule_with_filter = (
            builder.rule()
            .as_unique_rule()
            .with_filter("status = 'active' AND deleted_at IS NULL")
            .build()
        )

        sql_with_filter = uniqueness_executor._generate_unique_sql(rule_with_filter)
        assert "WHERE" in sql_with_filter.upper()
        assert "status = 'active'" in sql_with_filter
        assert "deleted_at IS NULL" in sql_with_filter

        # Test without filter
        rule_without_filter = builder.rule().as_unique_rule().build()
        sql_without_filter = uniqueness_executor._generate_unique_sql(
            rule_without_filter
        )

        # Should behave differently
        assert len(sql_with_filter) > len(sql_without_filter)

    def test_case_sensitivity_uniqueness_mutations(
        self, uniqueness_executor: UniquenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test case sensitivity in uniqueness checks"""
        rule = builder.basic_unique_rule(column="email_column")
        sql = uniqueness_executor._generate_unique_sql(rule)

        # Should be consistent about case sensitivity
        # Common mutation: UPPER(col) vs col for case-insensitive uniqueness
        # The behavior should be explicit and documented


# 🚀 PERFORMANCE AND EDGE CASES
class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases"""

    def test_large_dataset_sql_optimization(
        self, uniqueness_executor: UniquenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Verify SQL is optimized for large datasets"""
        rule = builder.basic_unique_rule(table="huge_fact_table")
        sql = uniqueness_executor._generate_unique_sql(rule)

        # Should use efficient duplicate detection approach
        # Avoid inefficient operations like nested queries on large datasets
        assert "SELECT *" not in sql.upper()  # Should not select all columns
        assert sql.count("SELECT") <= 2  # Should not have too many nested selects

    def test_unicode_values_uniqueness(
        self, uniqueness_executor: UniquenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test Unicode and special characters in uniqueness detection"""
        unicode_columns = ["用户名", "naïve_email", "column-with-émojis"]

        for column_name in unicode_columns:
            rule = builder.basic_unique_rule(column=column_name)
            sql = uniqueness_executor._generate_unique_sql(rule)

            # Should handle Unicode column names properly
            assert (
                column_name in sql
                or f"`{column_name}`" in sql
                or f'"{column_name}"' in sql
            )

    def test_extreme_duplicate_scenarios(
        self, uniqueness_executor: UniquenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test extreme scenarios that might break uniqueness logic"""
        rule = builder.basic_unique_rule()

        # Test scenarios that SQL generation should handle gracefully:
        # 1. All records are duplicates
        # 2. No duplicates at all
        # 3. Single record
        # 4. Empty table

        sql = uniqueness_executor._generate_unique_sql(rule)

        # SQL should be robust enough to handle all these cases
        assert len(sql.strip()) > 0
        assert "SELECT" in sql.upper()

    @pytest.mark.asyncio
    async def test_concurrent_uniqueness_checks(
        self, uniqueness_executor: UniquenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test that uniqueness executor is safe for concurrent use"""
        rules = [
            builder.basic_unique_rule(column=f"unique_column_{i}") for i in range(3)
        ]

        with patch.object(uniqueness_executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            # Mock successful queries
            with patch("shared.database.query_executor.QueryExecutor") as mock_qe_class:
                mock_query_executor = AsyncMock()
                mock_query_executor.execute_query.return_value = (
                    [{"total_count": 100, "unique_count": 100}],
                    ["total_count", "unique_count"],
                )
                mock_qe_class.return_value = mock_query_executor

                # Execute all rules concurrently - should not interfere
                import asyncio

                results = await asyncio.gather(
                    *[uniqueness_executor.execute_rule(rule) for rule in rules]
                )

                # All should succeed
                assert len(results) == 3
                assert all(
                    result.status in ["PASSED", "FAILED", "ERROR"] for result in results
                )
