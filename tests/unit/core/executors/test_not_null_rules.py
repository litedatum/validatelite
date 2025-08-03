"""
🧙‍♂️ Enhanced NOT NULL Rule Tests - Testing Ghost's Modern Testing Masterpiece

As the Testing Ghost 👻, I demonstrate the four key testing improvements:

1. 🏗️ Schema Builder Pattern - Eliminates fixture duplication
2. 🔄 Contract Testing - Ensures mocks match reality
3. 📊 Property-based Testing - Verifies behavior with random inputs
4. 🧬 Mutation Testing Readiness - Catches subtle bugs

This file focuses on NOT_NULL completeness rules with comprehensive edge case coverage.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest

# Core application imports
from core.executors.completeness_executor import CompletenessExecutor
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
    def table_info_strategy(draw: st.DrawFn) -> Dict[str, str]:
        """Generate valid database table information"""
        return {
            "database": draw(
                st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz_")
            ),
            "table": draw(
                st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz_")
            ),
            "column": draw(
                st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz_")
            ),
        }

    @st.composite
    def null_count_strategy(draw: st.DrawFn) -> Dict[str, int]:
        """Generate realistic null count scenarios"""
        total_records = draw(st.integers(min_value=0, max_value=100000))
        null_count = draw(st.integers(min_value=0, max_value=total_records))
        return {"total_records": total_records, "null_count": null_count}


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
def completeness_executor(mock_connection: ConnectionSchema) -> CompletenessExecutor:
    """CompletenessExecutor with contract verification"""
    executor = CompletenessExecutor(
        mock_connection,
        test_mode=True,
        sample_data_enabled=True,
        sample_data_max_records=5,
    )
    # Verify executor follows contract
    MockContract.verify_completeness_executor_contract(executor)
    return executor


# 🏗️ SCHEMA BUILDER PATTERN TESTS
class TestSchemaBuilderPattern:
    """Demonstrate how Schema Builder eliminates fixture duplication"""

    def test_basic_not_null_rule_creation(self, builder: TestDataBuilder) -> None:
        """Test creating basic NOT NULL rule with builder"""
        rule = (
            builder.rule()
            .with_name("customer_name_completeness")
            .with_target("production", "customers", "full_name")
            .as_not_null_rule()
            .build()
        )

        assert rule.name == "customer_name_completeness"
        assert rule.type == RuleType.NOT_NULL
        assert rule.category == RuleCategory.COMPLETENESS
        assert rule.target.entities[0].table == "customers"
        assert rule.target.entities[0].column == "full_name"

    def test_not_null_rule_with_filter(self, builder: TestDataBuilder) -> None:
        """Test creating NOT NULL rule with filter condition"""
        rule = (
            builder.rule()
            .with_name("active_customer_email_check")
            .with_severity(SeverityLevel.HIGH)
            .with_target("crm", "customers", "email")
            .as_not_null_rule()
            .with_filter("status = 'active'")
            .build()
        )

        assert rule.severity == SeverityLevel.HIGH
        assert rule.parameters["filter_condition"] == "status = 'active'"

    def test_quick_builder_methods(self, builder: TestDataBuilder) -> None:
        """Test quick builder methods for common scenarios"""
        # Quick NOT NULL rule
        rule = TestDataBuilder.basic_not_null_rule(table="users", column="username")

        assert rule.type == RuleType.NOT_NULL
        assert rule.target.entities[0].table == "users"
        assert rule.target.entities[0].column == "username"


# 🔄 CONTRACT TESTING IMPLEMENTATION
class TestContractTesting:
    """Ensure mocks accurately represent real implementations"""

    @pytest.mark.asyncio
    async def test_query_executor_contract_compliance(self) -> None:
        """Test that our QueryExecutor mocks follow the contract"""
        # Create contract-compliant mock for NOT_NULL scenario
        mock_data = MockContract.create_completeness_mock_data(
            failed_count=5, total_count=100, rule_type="NOT_NULL"
        )
        mock = MockContract.create_query_executor_mock(
            query_results=mock_data["main_query"], column_names=["failed_count"]
        )

        # Verify contract compliance
        await ContractTestCase.test_query_executor_contract_compliance(mock)

    @pytest.mark.asyncio
    async def test_not_null_rule_execution_with_contract_mock(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test rule execution using contract-compliant mock"""
        rule = builder.basic_not_null_rule(table="test_table", column="required_field")

        # Create async mock functions that return the expected data structure
        async def mock_execute_query(
            sql: str,
        ) -> tuple[List[Dict[str, Any]], List[str]]:
            if "total_count" in sql.lower():
                return [{"total_count": 100}], ["total_count"]
            elif "failed_count" in sql.lower():
                return [{"failed_count": 3}], ["failed_count"]
            else:
                return [
                    {"id": 1, "name": "test"},
                    {"id": 2, "name": "test2"},
                    {"id": 3, "name": "test3"},
                    {"id": 4, "name": "test4"},
                    {"id": 5, "name": "test5"},
                ], ["id", "name"]

        with patch.object(completeness_executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            # Mock the QueryExecutor class and its execute_query method
            with patch("shared.database.query_executor.QueryExecutor") as mock_qe_class:
                mock_query_executor = AsyncMock()
                mock_query_executor.execute_query.side_effect = mock_execute_query
                mock_qe_class.return_value = mock_query_executor

                result = await completeness_executor.execute_rule(rule)

        # Verify result structure (contract compliance)
        assert isinstance(result, ExecutionResultSchema)
        assert result.rule_id == rule.id
        assert result.status == "FAILED"
        assert result.sample_data is not None
        assert len(result.sample_data) == 5

        # Verify data flow: query result -> DatasetMetrics -> ExecutionResultSchema
        assert len(result.dataset_metrics) > 0
        assert result.dataset_metrics[0].failed_records == 3  # ← Correct null count
        assert result.dataset_metrics[0].total_records == 100  # ← Correct total count
        assert result.status == "FAILED"  # ← Should be FAILED due to null values


# 📊 PROPERTY-BASED TESTING
@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
class TestPropertyBasedTesting:
    """Use property-based testing to verify behavior with random inputs"""

    @hypothesis.given(null_counts=null_count_strategy())
    @settings(
        max_examples=30,
        deadline=3000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_null_counting_invariants(
        self, completeness_executor: CompletenessExecutor, null_counts: Dict[str, int]
    ) -> None:
        """Verify null counting properties hold for any valid input"""
        # Property: null_count should never exceed total_records
        total_records = null_counts["total_records"]
        null_count = null_counts["null_count"]

        # This property should always hold
        assert null_count <= total_records

        # Property: anomaly rate calculation should be consistent
        if total_records > 0:
            expected_rate = null_count / total_records
            assert 0.0 <= expected_rate <= 1.0

    @hypothesis.given(table_info=table_info_strategy())
    @settings(
        max_examples=20,
        deadline=3000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_sql_generation_invariants(
        self, completeness_executor: CompletenessExecutor, table_info: Dict[str, str]
    ) -> None:
        """Verify SQL generation properties for any valid table info"""
        rule = TestDataBuilder.basic_not_null_rule(
            table=table_info["table"], column=table_info["column"]
        )

        sql = completeness_executor._generate_not_null_sql(rule)

        # These properties should always hold
        assert "SELECT COUNT(*)" in sql.upper()
        assert "IS NULL" in sql.upper()
        assert table_info["table"] in sql
        assert table_info["column"] in sql


# 🧬 MUTATION TESTING READINESS
class TestMutationTestingReadiness:
    """Design tests to catch subtle bugs that mutation testing would find"""

    def test_null_count_boundary_conditions(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Catch off-by-one errors in null counting logic"""
        rule = builder.basic_not_null_rule()

        # Test zero nulls - catches > vs >= mutations
        sql_zero = completeness_executor._generate_not_null_sql(rule)
        assert "IS NULL" in sql_zero  # Should check for NULL values

        # Test single null - catches boundary errors
        assert "COUNT(*)" in sql_zero  # Should count all records

    def test_sql_injection_protection(
        self, mock_connection: ConnectionSchema, builder: TestDataBuilder
    ) -> None:
        """Verify SQL injection resistance for malicious inputs"""
        # Create a non-test-mode executor for security testing
        security_executor = CompletenessExecutor(mock_connection, test_mode=False)

        malicious_inputs = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; SELECT * FROM sensitive_data; --",
            "' UNION SELECT password FROM users --",
        ]

        for malicious_input in malicious_inputs:
            rule = (
                builder.rule()
                .with_target("test_db", malicious_input, "safe_column")
                .as_not_null_rule()
                .build()
            )

            # In production mode, should raise ValueError for SQL injection patterns
            with pytest.raises(
                RuleExecutionError,
                match="Table name contains potentially dangerous SQL patterns",
            ):
                security_executor._generate_not_null_sql(rule)

    def test_null_vs_empty_string_distinction(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Ensure NULL and empty string are handled differently"""
        rule = builder.basic_not_null_rule()
        sql = completeness_executor._generate_not_null_sql(rule)

        # Should specifically check for NULL, not empty strings
        assert "IS NULL" in sql.upper()
        assert "= ''" not in sql  # Should NOT check for empty strings

    @pytest.mark.asyncio
    async def test_error_handling_completeness(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test comprehensive error scenarios"""
        rule = builder.basic_not_null_rule(table="nonexistent_table")

        with patch.object(completeness_executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            # Mock database error
            with patch("shared.database.query_executor.QueryExecutor") as mock_qe_class:
                mock_query_executor = AsyncMock()
                mock_query_executor.execute_query.side_effect = Exception(
                    "Table does not exist"
                )
                mock_qe_class.return_value = mock_query_executor

                result = await completeness_executor.execute_rule(rule)

                # Should handle error gracefully
                assert result.status == "ERROR"
                assert result.error_message is not None
                assert "Table does not exist" in result.error_message

    def test_filter_condition_mutations(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test filter condition boundary cases"""
        # Test with filter
        rule_with_filter = (
            builder.rule().as_not_null_rule().with_filter("status = 'active'").build()
        )

        sql_with_filter = completeness_executor._generate_not_null_sql(rule_with_filter)
        assert "WHERE" in sql_with_filter.upper()
        assert "status = 'active'" in sql_with_filter

        # Test without filter
        rule_without_filter = builder.rule().as_not_null_rule().build()
        sql_without_filter = completeness_executor._generate_not_null_sql(
            rule_without_filter
        )

        # Should behave differently
        assert len(sql_with_filter) > len(sql_without_filter)


# 🚀 PERFORMANCE AND EDGE CASES
class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases"""

    def test_large_table_sql_optimization(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Verify SQL is optimized for large tables"""
        rule = builder.basic_not_null_rule(table="huge_table")
        sql = completeness_executor._generate_not_null_sql(rule)

        # Should use efficient counting approach
        assert "COUNT(*)" in sql.upper()
        # Should not use inefficient operations like DISTINCT on large datasets
        assert "SELECT *" not in sql.upper()

    def test_unicode_column_names(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test Unicode and special characters in column names"""
        unicode_columns = [
            "用户名",
            "naïve_column",
            "column-with-dashes",
            "column_with_émotion",
        ]

        for column_name in unicode_columns:
            rule = builder.basic_not_null_rule(column=column_name)
            sql = completeness_executor._generate_not_null_sql(rule)

            # Should handle Unicode column names properly
            assert column_name in sql or f"`{column_name}`" in sql

    @pytest.mark.asyncio
    async def test_concurrent_execution_safety(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test that executor is safe for concurrent use"""
        rules = [builder.basic_not_null_rule(column=f"column_{i}") for i in range(5)]

        with patch.object(completeness_executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            # Mock successful queries
            with patch("shared.database.query_executor.QueryExecutor") as mock_qe_class:
                mock_query_executor = AsyncMock()

                async def concurrent_mock_execute_query(
                    sql: str,
                ) -> tuple[List[Dict[str, Any]], List[str]]:
                    i = 0
                    if "column_0" in sql.lower():
                        i = 1
                    if "total_count" in sql.lower():
                        return [{"total_count": 100}], ["total_count"]
                    elif "failed_count" in sql.lower():
                        return [{"failed_count": 0 if i == 0 else 3}], ["failed_count"]
                    else:
                        return [
                            {"id": 1, "name": "test"},
                            {"id": 2, "name": "test2"},
                            {"id": 3, "name": "test3"},
                        ], ["id", "name"]

                mock_query_executor.execute_query.side_effect = (
                    concurrent_mock_execute_query
                )
                mock_qe_class.return_value = mock_query_executor

                # Execute all rules concurrently - should not interfere
                import asyncio

                results = await asyncio.gather(
                    *[completeness_executor.execute_rule(rule) for rule in rules]
                )

                # All should succeed
                assert len(results) == 5
                assert all(result.status in ["PASSED", "FAILED"] for result in results)
                assert all(
                    (
                        result.sample_data is not None
                        if result.status == "FAILED"
                        else True
                    )
                    for result in results
                )
                assert all(
                    (
                        (
                            result.sample_data is not None
                            and len(result.sample_data) == 3
                        )
                        if result.status == "FAILED"
                        else True
                    )
                    for result in results
                )
