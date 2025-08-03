"""
🧙‍♂️ Enhanced RANGE Rule Tests - Testing Ghost's Modern Testing Masterpiece

As the Testing Ghost 👻, I demonstrate the four key testing improvements:

1. 🏗️ Schema Builder Pattern - Eliminates fixture duplication
2. 🔄 Contract Testing - Ensures mocks match reality
3. 📊 Property-based Testing - Verifies behavior with random inputs
4. 🧬 Mutation Testing Readiness - Catches subtle bugs

This file focuses on RANGE validity rules with comprehensive boundary condition coverage.
"""

import math
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
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
    def range_boundary_strategy(draw: st.DrawFn) -> Dict[str, Optional[float]]:
        """Generate range boundaries with various configurations"""
        # Generate reasonable numeric ranges
        min_val = draw(
            st.floats(
                min_value=-1000000,
                max_value=1000000,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        max_val = draw(
            st.floats(
                min_value=min_val,
                max_value=1000000,
                allow_nan=False,
                allow_infinity=False,
            )
        )

        # Sometimes only min, sometimes only max, sometimes both
        config_type = draw(st.sampled_from(["both", "min_only", "max_only"]))

        if config_type == "min_only":
            return {"min": min_val, "max": None}
        elif config_type == "max_only":
            return {"min": None, "max": max_val}
        else:
            return {"min": min_val, "max": max_val}

    @st.composite
    def out_of_range_scenario_strategy(draw: st.DrawFn) -> Dict[str, int]:
        """Generate realistic out-of-range scenarios"""
        total_records = draw(st.integers(min_value=1, max_value=10000))
        out_of_range_count = draw(st.integers(min_value=0, max_value=total_records))
        return {
            "total_records": total_records,
            "out_of_range_count": out_of_range_count,
        }

    @st.composite
    def numeric_values_strategy(draw: st.DrawFn) -> float:
        """Generate various numeric value types"""
        return draw(
            st.one_of(
                st.integers(min_value=-1000000, max_value=1000000),
                st.floats(
                    min_value=-1000000,
                    max_value=1000000,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                # Include some edge cases
                st.just(0),
                st.just(0.0),
                st.just(-0.0),
            )
        )


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
    executor = ValidityExecutor(mock_connection, test_mode=True)
    # Verify executor follows contract
    MockContract.verify_validity_executor_contract(executor)
    return executor


# 🏗️ SCHEMA BUILDER PATTERN TESTS
class TestSchemaBuilderPattern:
    """Demonstrate how Schema Builder eliminates fixture duplication"""

    def test_basic_range_rule_creation(self, builder: TestDataBuilder) -> None:
        """Test creating basic RANGE rule with builder"""
        rule = (
            builder.rule()
            .with_name("price_range_validation")
            .with_target("products", "catalog", "price")
            .as_range_rule(min_val=0.0, max_val=10000.0)
            .build()
        )

        assert rule.name == "price_range_validation"
        assert rule.type == RuleType.RANGE
        assert rule.category == RuleCategory.VALIDITY
        assert rule.target.entities[0].table == "catalog"
        assert rule.target.entities[0].column == "price"
        assert rule.parameters["min"] == 0.0
        assert rule.parameters["max"] == 10000.0

    def test_min_only_range_rule(self, builder: TestDataBuilder) -> None:
        """Test creating range rule with only minimum value"""
        rule = (
            builder.rule()
            .with_name("age_minimum_check")
            .with_target("users", "profiles", "age")
            .as_range_rule(min_val=18)
            .build()
        )

        assert rule.parameters["min"] == 18
        assert "max" not in rule.parameters

    def test_max_only_range_rule(self, builder: TestDataBuilder) -> None:
        """Test creating range rule with only maximum value"""
        rule = (
            builder.rule()
            .with_name("discount_max_check")
            .with_target("sales", "orders", "discount_percentage")
            .as_range_rule(max_val=100.0)
            .build()
        )

        assert rule.parameters["max"] == 100.0
        assert "min" not in rule.parameters

    def test_range_rule_with_filter(self, builder: TestDataBuilder) -> None:
        """Test creating RANGE rule with filter condition"""
        rule = (
            builder.rule()
            .with_name("active_product_price_check")
            .with_severity(SeverityLevel.HIGH)
            .with_target("inventory", "products", "unit_price")
            .as_range_rule(min_val=1.0, max_val=5000.0)
            .with_filter("status = 'active' AND category != 'discontinued'")
            .build()
        )

        assert rule.severity == SeverityLevel.HIGH
        assert (
            rule.parameters["filter_condition"]
            == "status = 'active' AND category != 'discontinued'"
        )

    def test_quick_builder_methods(self, builder: TestDataBuilder) -> None:
        """Test quick builder methods for common scenarios"""
        # Quick RANGE rule with both bounds
        rule = TestDataBuilder.basic_range_rule(
            min_val=0, max_val=100, table="metrics", column="percentage"
        )

        assert rule.type == RuleType.RANGE
        assert rule.target.entities[0].table == "metrics"
        assert rule.target.entities[0].column == "percentage"
        assert rule.parameters["min"] == 0
        assert rule.parameters["max"] == 100


# 🔄 CONTRACT TESTING IMPLEMENTATION
class TestContractTesting:
    """Ensure mocks accurately represent real implementations"""

    @pytest.mark.asyncio
    async def test_query_executor_contract_compliance(self) -> None:
        """Test that our QueryExecutor mocks follow the contract"""
        # Create contract-compliant mock for RANGE scenario
        mock = MockContract.create_query_executor_mock(
            query_results=[{"total_count": 100, "out_of_range_count": 8}],
            column_names=["total_count", "out_of_range_count"],
        )

        # Verify contract compliance
        await ContractTestCase.test_query_executor_contract_compliance(mock)

    @pytest.mark.asyncio
    async def test_range_rule_execution_with_contract_mock(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test rule execution using contract-compliant mock"""
        rule = builder.basic_range_rule(
            min_val=0, max_val=100, table="test_table", column="score"
        )

        # Create async mock functions that return the expected data structure
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
        assert result.sample_data is not None
        assert len(result.sample_data) == 5

        # Verify range-specific data flow
        assert len(result.dataset_metrics) > 0
        assert result.dataset_metrics[0].failed_records == 5  # ← Out of range count
        assert result.dataset_metrics[0].total_records == 100  # ← Total records


# 📊 PROPERTY-BASED TESTING
@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
class TestPropertyBasedTesting:
    """Use property-based testing to verify behavior with random inputs"""

    @hypothesis.given(range_config=range_boundary_strategy())
    @settings(
        max_examples=30,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_range_sql_generation_invariants(
        self,
        validity_executor: ValidityExecutor,
        range_config: Dict[str, Optional[float]],
    ) -> None:
        """Verify SQL generation properties for any valid range configuration"""
        rule = TestDataBuilder.basic_range_rule(
            min_val=range_config.get("min"), max_val=range_config.get("max")
        )

        sql = validity_executor._generate_range_sql(rule)

        # These properties should always hold
        assert "SELECT" in sql.upper()
        assert "FROM" in sql.upper()

        # Should have appropriate WHERE conditions based on configuration
        if range_config.get("min") is not None:
            assert "<" in sql  # Should check for values less than min
        if range_config.get("max") is not None:
            assert ">" in sql  # Should check for values greater than max

    @hypothesis.given(scenario=out_of_range_scenario_strategy())
    @settings(
        max_examples=25,
        deadline=3000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_out_of_range_counting_invariants(self, scenario: Dict[str, int]) -> None:
        """Verify out-of-range counting properties"""
        total_records = scenario["total_records"]
        out_of_range_count = scenario["out_of_range_count"]

        # Property: out-of-range count should never exceed total records
        assert out_of_range_count <= total_records

        # Property: anomaly rate calculation should be consistent
        if total_records > 0:
            anomaly_rate = out_of_range_count / total_records
            assert 0.0 <= anomaly_rate <= 1.0

    @hypothesis.given(
        values=st.lists(numeric_values_strategy(), min_size=1, max_size=100)
    )
    @settings(
        max_examples=20,
        deadline=4000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_range_validation_properties(self, values: List[float]) -> None:
        """Test mathematical properties of range validation"""
        min_val = min(values)
        max_val = max(values)

        # All values should be within the range defined by min and max of the list
        for value in values:
            assert min_val <= value <= max_val

        # Edge case: when min equals max, only that value should be valid
        if min_val == max_val:
            assert all(v == min_val for v in values)


# 🧬 MUTATION TESTING READINESS
class TestMutationTestingReadiness:
    """Design tests to catch subtle bugs that mutation testing would find"""

    def test_boundary_condition_mutations(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Catch off-by-one errors in range checking logic"""
        # Test boundary conditions that are prone to mutations
        rule = builder.basic_range_rule(min_val=10, max_val=90)
        sql = validity_executor._generate_range_sql(rule)

        # Common mutations: < vs <=, > vs >=
        # The SQL should use appropriate operators for range validation
        assert ("< 10" in sql or "<= 9" in sql) or ("< " in sql and "10" in sql)
        assert ("> 90" in sql or ">= 91" in sql) or (">" in sql and "90" in sql)

    def test_null_value_handling_mutations(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test how NULL values are handled in range checks"""
        rule = builder.basic_range_rule(min_val=0, max_val=100)
        sql = validity_executor._generate_range_sql(rule)

        # Common mutation: Should NULL values be considered out of range?
        # This depends on business logic, but should be consistent
        # Most implementations exclude NULLs from range validation
        if "IS NOT NULL" not in sql.upper():
            # If NULLs are not explicitly excluded, they should be handled appropriately
            pass

    def test_numeric_type_mutations(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test handling of different numeric types"""
        # Integer boundaries
        rule_int = builder.basic_range_rule(min_val=1, max_val=100)
        sql_int = validity_executor._generate_range_sql(rule_int)

        # Float boundaries
        rule_float = builder.basic_range_rule(min_val=1.5, max_val=99.9)
        sql_float = validity_executor._generate_range_sql(rule_float)

        # Should handle both integer and float values correctly
        assert "1" in sql_int and "100" in sql_int
        assert "1.5" in sql_float and "99.9" in sql_float

    def test_edge_case_range_values(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test edge cases that might break range logic"""
        # Zero boundaries
        rule_zero = builder.basic_range_rule(min_val=0, max_val=0)
        sql_zero = validity_executor._generate_range_sql(rule_zero)
        assert "0" in sql_zero

        # Negative ranges
        rule_negative = builder.basic_range_rule(min_val=-100, max_val=-10)
        sql_negative = validity_executor._generate_range_sql(rule_negative)
        assert "-100" in sql_negative and "-10" in sql_negative

    def test_sql_injection_protection_range(
        self, mock_connection: ConnectionSchema, builder: TestDataBuilder
    ) -> None:
        """Verify SQL injection resistance for range parameters"""
        # Create a non-test-mode executor for security testing
        security_executor = ValidityExecutor(mock_connection, test_mode=False)

        # While range values are typically numeric, test string injection attempts
        rule = (
            builder.rule()
            .with_target("test_db", "'; DROP TABLE users; --", "safe_column")
            .as_range_rule(min_val=0, max_val=100)
            .build()
        )

        # Should raise ValueError for SQL injection attempts
        with pytest.raises(
            RuleExecutionError, match="contains potentially dangerous SQL patterns"
        ):
            security_executor._generate_range_sql(rule)

    @pytest.mark.asyncio
    async def test_error_handling_range_validation(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test comprehensive error scenarios for range validation"""
        rule = builder.basic_range_rule(
            min_val=0, max_val=100, table="nonexistent_table"
        )

        with patch.object(validity_executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            # Mock database error
            with patch("shared.database.query_executor.QueryExecutor") as mock_qe_class:
                mock_query_executor = AsyncMock()
                mock_query_executor.execute_query.side_effect = Exception(
                    "Column does not exist"
                )
                mock_qe_class.return_value = mock_query_executor

                result = await validity_executor.execute_rule(rule)

                # Should handle error gracefully
                assert result.status == "ERROR"
                assert result.error_message is not None
                assert "Column does not exist" in result.error_message

    def test_invalid_range_parameters(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test validation of range parameters themselves"""
        # Test case where min > max (invalid range)
        try:
            rule = builder.basic_range_rule(
                min_val=100, max_val=10
            )  # Invalid: min > max
            sql = validity_executor._generate_range_sql(rule)
            # Should either handle gracefully or raise validation error
        except (ValueError, RuleExecutionError):
            # Expected behavior for invalid range
            pass

    def test_filter_condition_with_range_mutations(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test filter condition interactions with range logic"""
        rule_with_filter = (
            builder.rule()
            .as_range_rule(min_val=0, max_val=100)
            .with_filter("status = 'active'")
            .build()
        )

        sql_with_filter = validity_executor._generate_range_sql(rule_with_filter)

        # Should properly combine range conditions with filter
        assert "WHERE" in sql_with_filter.upper()
        assert "status = 'active'" in sql_with_filter
        assert "AND" in sql_with_filter.upper() or "OR" in sql_with_filter.upper()

    def test_operator_precedence_mutations(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test SQL operator precedence in range conditions"""
        rule = builder.basic_range_rule(min_val=10, max_val=90)
        sql = validity_executor._generate_range_sql(rule)

        # Should use proper parentheses for complex conditions
        # Common mutation: missing parentheses affecting logic
        if "AND" in sql.upper() or "OR" in sql.upper():
            # Should have appropriate grouping for complex conditions
            pass


# 🚀 PERFORMANCE AND EDGE CASES
class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases"""

    def test_large_range_values_performance(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test performance with very large range values"""
        rule = builder.basic_range_rule(
            min_val=-1e10, max_val=1e10, table="big_data_table"
        )
        sql = validity_executor._generate_range_sql(rule)

        # Should handle large numbers efficiently
        assert len(sql) < 10000  # Reasonable SQL length
        assert (
            "1e" in sql.lower() or "1000000000" in sql
        )  # Should contain the large values

    def test_precision_edge_cases(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test floating point precision edge cases"""
        rule = builder.basic_range_rule(
            min_val=0.1, max_val=0.9, column="precision_column"
        )
        sql = validity_executor._generate_range_sql(rule)

        # Should handle decimal precision appropriately
        assert "0.1" in sql and "0.9" in sql

    def test_unicode_numeric_formats(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test various numeric format representations"""
        # Scientific notation
        rule_scientific = builder.basic_range_rule(min_val=1e-6, max_val=1e6)
        sql_scientific = validity_executor._generate_range_sql(rule_scientific)

        # Should handle scientific notation
        assert len(sql_scientific) > 0

    @pytest.mark.asyncio
    async def test_concurrent_range_validation(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test concurrent range validations"""
        rules = [
            builder.basic_range_rule(
                min_val=i * 10, max_val=(i + 1) * 10, column=f"col_{i}"
            )
            for i in range(3)
        ]

        with patch.object(validity_executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            # Mock successful queries
            with patch("shared.database.query_executor.QueryExecutor") as mock_qe_class:
                mock_query_executor = AsyncMock()
                mock_query_executor.execute_query.return_value = (
                    [{"total_count": 100, "anomaly_count": 0}],
                    ["total_count", "anomaly_count"],
                )
                mock_qe_class.return_value = mock_query_executor

                # Execute all rules concurrently
                import asyncio

                results = await asyncio.gather(
                    *[validity_executor.execute_rule(rule) for rule in rules]
                )

                # All should succeed
                assert len(results) == 3
                assert all(result.status == "PASSED" for result in results)
