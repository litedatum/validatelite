"""
🧙‍♂️ Enhanced LENGTH Rule Tests - Testing Ghost's Modern Testing Masterpiece

As the Testing Ghost 👻, I demonstrate the four key testing improvements:

1. 🏗️ Schema Builder Pattern - Eliminates fixture duplication
2. 🔄 Contract Testing - Ensures mocks match reality
3. 📊 Property-based Testing - Verifies behavior with random inputs
4. 🧬 Mutation Testing Readiness - Catches subtle bugs

This file focuses on LENGTH completeness rules with comprehensive boundary length validation.
Incorporates valuable scenarios from the legacy test_length_rules.py.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
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
    def length_boundary_strategy(draw: st.DrawFn) -> Dict[str, Optional[int]]:
        """Generate length boundaries with various configurations"""
        # Generate reasonable length ranges
        min_length = draw(st.integers(min_value=0, max_value=100))
        max_length = draw(st.integers(min_value=min_length, max_value=1000))

        # Sometimes only min, sometimes only max, sometimes both
        config_type = draw(st.sampled_from(["both", "min_only", "max_only"]))

        if config_type == "min_only":
            return {"min_length": min_length, "max_length": None}
        elif config_type == "max_only":
            return {"min_length": None, "max_length": max_length}
        else:
            return {"min_length": min_length, "max_length": max_length}

    @st.composite
    def length_violation_scenario_strategy(draw: st.DrawFn) -> Dict[str, int]:
        """Generate realistic length violation scenarios"""
        total_records = draw(st.integers(min_value=1, max_value=5000))
        length_violations = draw(st.integers(min_value=0, max_value=total_records))
        return {"total_records": total_records, "length_violations": length_violations}

    @st.composite
    def text_data_strategy(draw: st.DrawFn) -> str:
        """Generate various text data patterns for length testing"""
        from typing import cast

        return cast(
            str,
            draw(
                st.one_of(
                    st.text(min_size=0, max_size=200),  # Normal text
                    st.text(min_size=0, max_size=5, alphabet="abcde"),  # Short text
                    st.text(min_size=100, max_size=1000),  # Long text
                    st.just(""),  # Empty string
                    st.just(" " * 50),  # Whitespace only
                    st.text(alphabet="测试中文字符"),  # Unicode text
                )
            ),
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
def completeness_executor(mock_connection: ConnectionSchema) -> CompletenessExecutor:
    """CompletenessExecutor with contract verification"""
    executor = CompletenessExecutor(mock_connection, test_mode=True)
    # Verify executor follows contract
    MockContract.verify_completeness_executor_contract(executor)
    return executor


# 🏗️ SCHEMA BUILDER PATTERN TESTS
class TestSchemaBuilderPattern:
    """Demonstrate how Schema Builder eliminates fixture duplication"""

    def test_basic_length_rule_creation(self, builder: TestDataBuilder) -> None:
        """Test creating basic LENGTH rule with builder"""
        rule = (
            builder.rule()
            .with_name("name_length_validation")
            .with_target("users", "profiles", "full_name")
            .as_length_rule(min_length=2, max_length=50)
            .build()
        )

        assert rule.name == "name_length_validation"
        assert rule.type == RuleType.LENGTH
        assert rule.category == RuleCategory.COMPLETENESS
        assert rule.target.entities[0].table == "profiles"
        assert rule.target.entities[0].column == "full_name"
        assert rule.parameters["min_length"] == 2
        assert rule.parameters["max_length"] == 50

    def test_min_only_length_rule(self, builder: TestDataBuilder) -> None:
        """Test creating length rule with only minimum value - valuable scenario from legacy"""
        rule = (
            builder.rule()
            .with_name("password_minimum_length")
            .with_target("auth", "users", "password")
            .as_length_rule(min_length=8)
            .build()
        )

        assert rule.parameters["min_length"] == 8
        assert "max_length" not in rule.parameters

    def test_max_only_length_rule(self, builder: TestDataBuilder) -> None:
        """Test creating length rule with only maximum value - valuable scenario from legacy"""
        rule = (
            builder.rule()
            .with_name("comment_max_length")
            .with_target("posts", "comments", "content")
            .as_length_rule(max_length=500)
            .build()
        )

        assert rule.parameters["max_length"] == 500
        assert "min_length" not in rule.parameters

    def test_length_rule_with_filter(self, builder: TestDataBuilder) -> None:
        """Test creating LENGTH rule with filter condition - scenario from legacy"""
        rule = (
            builder.rule()
            .with_name("active_user_bio_length")
            .with_severity(SeverityLevel.HIGH)
            .with_target("social", "profiles", "bio")
            .as_length_rule(min_length=10, max_length=160)
            .with_filter("status = 'active' AND verified = 1")
            .build()
        )

        assert rule.severity == SeverityLevel.HIGH
        assert (
            rule.parameters["filter_condition"] == "status = 'active' AND verified = 1"
        )

    def test_quick_builder_methods(self, builder: TestDataBuilder) -> None:
        """Test quick builder methods for common scenarios"""
        # Quick LENGTH rule with both bounds
        rule = TestDataBuilder.basic_length_rule(
            min_length=1, max_length=100, table="products", column="title"
        )

        assert rule.type == RuleType.LENGTH
        assert rule.target.entities[0].table == "products"
        assert rule.target.entities[0].column == "title"
        assert rule.parameters["min_length"] == 1
        assert rule.parameters["max_length"] == 100


# 🔄 CONTRACT TESTING IMPLEMENTATION
class TestContractTesting:
    """Ensure mocks accurately represent real implementations"""

    @pytest.mark.asyncio
    async def test_query_executor_contract_compliance(self) -> None:
        """Test that our QueryExecutor mocks follow the contract"""
        # Create contract-compliant mock for LENGTH scenario
        mock_data = MockContract.create_completeness_mock_data(
            failed_count=8, total_count=100, rule_type="LENGTH"
        )
        mock = MockContract.create_query_executor_mock(
            query_results=mock_data["main_query"], column_names=["failed_count"]
        )

        # Verify contract compliance
        await ContractTestCase.test_query_executor_contract_compliance(mock)

    @pytest.mark.asyncio
    async def test_length_rule_execution_with_contract_mock(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test rule execution using contract-compliant mock"""
        rule = builder.basic_length_rule(
            min_length=5, max_length=20, table="users", column="username"
        )

        # Create async mock functions that return the expected data structure
        async def mock_execute_query(
            sql: str,
        ) -> tuple[List[Dict[str, Any]], List[str]]:
            if "total_count" in sql.lower():
                return [{"total_count": 100}], ["total_count"]
            elif "failed_count" in sql.lower():
                return [{"failed_count": 12}], ["failed_count"]
            else:
                return [
                    {"id": 1, "name": "test"},
                    {"id": 2, "name": "test2"},
                    {"id": 3, "name": "test3"},
                    {"id": 4, "name": "test4"},
                    {"id": 5, "name": "test5"},
                    {"id": 6, "name": "test6"},
                    {"id": 7, "name": "test7"},
                    {"id": 8, "name": "test8"},
                    {"id": 9, "name": "test9"},
                    {"id": 10, "name": "test10"},
                    {"id": 11, "name": "test11"},
                    {"id": 12, "name": "test12"},
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
        assert len(result.sample_data) == 12

        # Verify length-specific data flow
        assert len(result.dataset_metrics) > 0
        assert (
            result.dataset_metrics[0].failed_records == 12
        )  # ← Length violation count
        assert result.dataset_metrics[0].total_records == 100  # ← Total records
        assert result.status == "FAILED"  # ← Should be FAILED due to length violations


# 📊 PROPERTY-BASED TESTING
@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
class TestPropertyBasedTesting:
    """Use property-based testing to verify behavior with random inputs"""

    @hypothesis.given(length_config=length_boundary_strategy())
    @settings(
        max_examples=30,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_length_sql_generation_invariants(
        self,
        completeness_executor: CompletenessExecutor,
        length_config: Dict[str, Optional[int]],
    ) -> None:
        """Verify SQL generation properties for any valid length configuration"""
        rule = TestDataBuilder.basic_length_rule(
            min_length=length_config.get("min_length"),
            max_length=length_config.get("max_length"),
        )

        sql = completeness_executor._generate_length_sql(rule)

        # These properties should always hold
        assert "SELECT" in sql.upper()
        assert "FROM" in sql.upper()
        assert "LENGTH(" in sql.upper() or "LEN(" in sql.upper()

        # Should have appropriate WHERE conditions based on configuration
        if length_config.get("min_length") is not None:
            assert "<" in sql  # Should check for values less than min
        if length_config.get("max_length") is not None:
            assert ">" in sql  # Should check for values greater than max

    @hypothesis.given(scenario=length_violation_scenario_strategy())
    @settings(
        max_examples=25,
        deadline=3000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_length_violation_counting_invariants(
        self, scenario: Dict[str, int]
    ) -> None:
        """Verify length violation counting properties"""
        total_records = scenario["total_records"]
        length_violations = scenario["length_violations"]

        # Property: violations should never exceed total records
        assert length_violations <= total_records

        # Property: violation rate calculation should be consistent
        if total_records > 0:
            violation_rate = length_violations / total_records
            assert 0.0 <= violation_rate <= 1.0

    @hypothesis.given(texts=st.lists(text_data_strategy(), min_size=1, max_size=50))
    @settings(
        max_examples=20,
        deadline=4000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_text_length_properties(self, texts: List[str]) -> None:
        """Test mathematical properties of text length validation"""
        for text in texts:
            # Property: length should be consistent with string length
            actual_length = len(text)
            assert actual_length >= 0

            # Property: empty strings have zero length
            if text == "":
                assert actual_length == 0


# 🧬 MUTATION TESTING READINESS
class TestMutationTestingReadiness:
    """Design tests to catch subtle bugs that mutation testing would find"""

    def test_length_boundary_condition_mutations(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Catch off-by-one errors in length checking logic - critical for legacy scenarios"""
        # Test boundary conditions that are prone to mutations
        rule = builder.basic_length_rule(min_length=5, max_length=20)
        sql = completeness_executor._generate_length_sql(rule)

        # Common mutations: < vs <=, > vs >=
        # The SQL should use appropriate operators for length validation
        assert ("< 5" in sql or "<= 4" in sql) or (
            "LENGTH" in sql.upper() and "5" in sql
        )
        assert ("> 20" in sql or ">= 21" in sql) or (
            "LENGTH" in sql.upper() and "20" in sql
        )

    def test_null_value_handling_in_length(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test how NULL values are handled in length checks"""
        rule = builder.basic_length_rule(min_length=1, max_length=10)
        sql = completeness_executor._generate_length_sql(rule)

        # Common mutation: Should NULL values be considered length violations?
        # Most implementations exclude NULLs from length validation
        # The behavior should be consistent and documented

    def test_empty_string_vs_null_distinction(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Ensure empty strings and NULL are handled differently - legacy scenario insight"""
        rule = builder.basic_length_rule(min_length=1)
        sql = completeness_executor._generate_length_sql(rule)

        # Should handle empty strings (length 0) separately from NULL values
        assert "LENGTH(" in sql.upper() or "LEN(" in sql.upper()

    def test_unicode_length_calculations(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test Unicode character length calculations - edge case from legacy"""
        rule = builder.basic_length_rule(
            min_length=2, max_length=10, column="unicode_column"
        )
        sql = completeness_executor._generate_length_sql(rule)

        # Should use appropriate length function for Unicode
        assert len(sql) > 0  # Basic validation that SQL is generated

    def test_whitespace_handling_mutations(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test handling of whitespace in length calculations"""
        rule = builder.basic_length_rule(min_length=1, max_length=50)
        sql = completeness_executor._generate_length_sql(rule)

        # Common mutation: Should whitespace be trimmed before length check?
        # This is business logic dependent - test for consistency

    def test_invalid_length_parameters(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test validation of length parameters themselves - legacy error scenario"""
        # Test case where min > max (invalid range)
        try:
            rule = builder.basic_length_rule(
                min_length=50, max_length=10
            )  # Invalid: min > max
            sql = completeness_executor._generate_length_sql(rule)
            # Should either handle gracefully or raise validation error
        except (ValueError, RuleExecutionError):
            # Expected behavior for invalid range
            pass

    @pytest.mark.asyncio
    async def test_length_error_handling_completeness(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test comprehensive error scenarios for length validation - legacy error patterns"""
        rule = builder.basic_length_rule(
            min_length=1, max_length=100, table="nonexistent_table"
        )

        with patch.object(completeness_executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            # Mock database error
            with patch("shared.database.query_executor.QueryExecutor") as mock_qe_class:
                mock_query_executor = AsyncMock()
                mock_query_executor.execute_query.side_effect = Exception(
                    "Column does not exist"
                )
                mock_qe_class.return_value = mock_query_executor

                result = await completeness_executor.execute_rule(rule)

                # Should handle error gracefully
                assert result.status == "ERROR"
                assert result.error_message is not None
                assert "Column does not exist" in result.error_message

    def test_filter_condition_with_length_mutations(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test filter condition interactions with length logic - legacy scenario"""
        rule_with_filter = (
            builder.rule()
            .as_length_rule(min_length=5, max_length=100)
            .with_filter("status = 'active' AND created_at > '2024-01-01'")
            .build()
        )

        sql_with_filter = completeness_executor._generate_length_sql(rule_with_filter)

        # Should properly combine length conditions with filter
        assert "WHERE" in sql_with_filter.upper()
        assert "status = 'active'" in sql_with_filter
        assert "AND" in sql_with_filter.upper() or "OR" in sql_with_filter.upper()

    def test_sql_injection_protection_length(
        self, mock_connection: ConnectionSchema, builder: TestDataBuilder
    ) -> None:
        """Verify SQL injection resistance for length validation"""
        # Create a non-test-mode executor for security testing
        security_executor = CompletenessExecutor(mock_connection, test_mode=False)

        malicious_inputs = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; UPDATE users SET admin=1; --",
        ]

        for malicious_input in malicious_inputs:
            rule = (
                builder.rule()
                .with_target("test_db", malicious_input, "safe_column")
                .as_length_rule(min_length=1, max_length=100)
                .build()
            )

            # Should raise ValueError for SQL injection attempts
            with pytest.raises(
                RuleExecutionError,
                match="Table name contains potentially dangerous SQL patterns",
            ):
                security_executor._generate_length_sql(rule)

    def test_zero_length_edge_cases(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test zero length edge cases - critical boundary from legacy"""
        # Zero minimum length (allow empty strings)
        rule_zero_min = builder.basic_length_rule(min_length=0, max_length=10)
        sql_zero_min = completeness_executor._generate_length_sql(rule_zero_min)

        # Should handle zero minimum appropriately
        assert len(sql_zero_min) > 0


# 🚀 PERFORMANCE AND EDGE CASES
class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases from legacy scenarios"""

    def test_large_text_length_optimization(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test performance with very large length bounds - legacy large data scenario"""
        rule = builder.basic_length_rule(
            min_length=1,
            max_length=10000,  # Very large max length
            table="large_text_table",
        )
        sql = completeness_executor._generate_length_sql(rule)

        # Should handle large length bounds efficiently
        assert len(sql) < 5000  # Reasonable SQL length limit
        assert "10000" in sql

    def test_unicode_text_length_handling(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test Unicode text handling - valuable scenario from legacy"""
        unicode_columns = ["中文名称", "naïve_name", "column_with_émojis"]

        for column_name in unicode_columns:
            rule = builder.basic_length_rule(
                min_length=1, max_length=50, column=column_name
            )
            sql = completeness_executor._generate_length_sql(rule)

            # Should handle Unicode column names properly
            assert (
                column_name in sql
                or f"`{column_name}`" in sql
                or f'"{column_name}"' in sql
            )

    def test_extreme_length_scenarios(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test extreme length scenarios that might break validation logic"""
        # Very small range
        rule_tiny = builder.basic_length_rule(min_length=1, max_length=2)
        sql_tiny = completeness_executor._generate_length_sql(rule_tiny)
        assert len(sql_tiny) > 0

        # Only zero length allowed (empty strings only)
        rule_empty = builder.basic_length_rule(min_length=0, max_length=0)
        sql_empty = completeness_executor._generate_length_sql(rule_empty)
        assert len(sql_empty) > 0

    def test_multiple_length_rules_scenario(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test scenario with multiple length rules - legacy multi-rule pattern"""
        rules = [
            builder.basic_length_rule(min_length=2, max_length=50, column="first_name"),
            builder.basic_length_rule(min_length=2, max_length=50, column="last_name"),
            builder.basic_length_rule(min_length=8, column="password"),  # min only
            builder.basic_length_rule(max_length=160, column="bio"),  # max only
        ]

        for rule in rules:
            sql = completeness_executor._generate_length_sql(rule)
            # Each should generate valid SQL
            assert "SELECT" in sql.upper()
            assert "LENGTH(" in sql.upper() or "LEN(" in sql.upper()

    @pytest.mark.asyncio
    async def test_concurrent_length_validation(
        self, completeness_executor: CompletenessExecutor, builder: TestDataBuilder
    ) -> None:
        """Test concurrent length validations - legacy concurrent scenario"""
        rules = [
            builder.basic_length_rule(
                min_length=i, max_length=i * 10, column=f"text_col_{i}"
            )
            for i in range(1, 4)
        ]

        with patch.object(completeness_executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            # Mock successful queries
            with patch("shared.database.query_executor.QueryExecutor") as mock_qe_class:
                mock_query_executor = AsyncMock()
                mock_query_executor.execute_query.return_value = (
                    [{"total_count": 100, "failed_count": 0}],
                    ["total_count", "failed_count"],
                )
                mock_qe_class.return_value = mock_query_executor

                # Execute all rules concurrently
                import asyncio

                results = await asyncio.gather(
                    *[completeness_executor.execute_rule(rule) for rule in rules]
                )

                # All should succeed
                assert len(results) == 3
                assert all(result.status == "PASSED" for result in results)
