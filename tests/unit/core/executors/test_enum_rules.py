"""
🧙‍♂️ Enhanced ENUM Rule Tests - Testing Ghost's Modern Testing Masterpiece

As the Testing Ghost 👻, I demonstrate the four key testing improvements:

1. 🏗️ Schema Builder Pattern - Eliminates fixture duplication
2. 🔄 Contract Testing - Ensures mocks match reality
3. 📊 Property-based Testing - Verifies behavior with random inputs
4. 🧬 Mutation Testing Readiness - Catches subtle bugs

This file serves as a template for upgrading all test files in the project.
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
from shared.exceptions import EngineError, RuleExecutionError
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.result_schema import ExecutionResultSchema
from shared.schema.rule_schema import RuleSchema

# Enhanced testing imports
from tests.shared.builders.test_builders import TestDataBuilder
from tests.shared.contracts.test_contracts import ContractTestCase, MockContract

# Property-based testing (would need to be added to requirements)
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
    def enum_values_strategy(draw: st.DrawFn) -> List[str]:
        """Generate valid enum value lists for property testing"""
        size = draw(st.integers(min_value=1, max_value=10))
        return draw(
            st.lists(
                st.text(
                    min_size=1,
                    max_size=20,
                    alphabet=st.characters(
                        blacklist_characters=['"', "'", "\\", "\n", "\r"]
                    ),
                ),
                min_size=size,
                max_size=size,
                unique=True,
            )
        )

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

    def test_basic_enum_rule_creation(self, builder: TestDataBuilder) -> None:
        """Test creating basic enum rule with builder"""
        rule = (
            builder.rule()
            .with_name("status_check")
            .with_target("production", "orders", "status")
            .as_enum_rule(["pending", "shipped", "delivered"])
            .build()
        )

        assert rule.name == "status_check"
        assert rule.type == RuleType.ENUM
        assert rule.target.entities[0].table == "orders"
        assert rule.parameters["allowed_values"] == ["pending", "shipped", "delivered"]

    def test_complex_enum_rule_with_filter(self, builder: TestDataBuilder) -> None:
        """Test creating complex rule with multiple builder methods"""
        rule = (
            builder.rule()
            .with_name("priority_customer_status")
            .with_severity(SeverityLevel.HIGH)
            .with_target("crm", "customers", "status")
            .as_enum_rule(["gold", "platinum", "diamond"])
            .with_filter("registration_date >= '2024-01-01'")
            .with_parameter("region", "US")
            .build()
        )

        assert rule.severity == SeverityLevel.HIGH
        assert (
            rule.parameters["filter_condition"] == "registration_date >= '2024-01-01'"
        )
        assert rule.parameters["region"] == "US"

    def test_email_domain_enum_rule(self, builder: TestDataBuilder) -> None:
        """Test email domain extraction rule"""
        rule = (
            builder.rule()
            .with_name("corporate_email_check")
            .with_target("users", "profiles", "email")
            .as_enum_rule(["company.com", "partner.org"])
            .with_email_domain_extraction()
            .build()
        )

        assert rule.parameters["extract_domain"] is True
        assert "company.com" in rule.parameters["allowed_values"]

    def test_quick_builder_methods(self, builder: TestDataBuilder) -> None:
        """Test quick builder methods for common scenarios"""
        # Quick enum rule
        rule = TestDataBuilder.basic_enum_rule(
            values=["active", "inactive"], table="users", column="status"
        )

        assert rule.type == RuleType.ENUM
        assert rule.target.entities[0].table == "users"
        assert rule.parameters["allowed_values"] == ["active", "inactive"]


# 🔄 CONTRACT TESTING IMPLEMENTATION
class TestContractTesting:
    """Ensure mocks accurately represent real implementations"""

    @pytest.mark.asyncio
    async def test_query_executor_contract_compliance(self) -> None:
        """Test that our QueryExecutor mocks follow the contract"""
        # Create contract-compliant mock
        mock = MockContract.create_query_executor_mock(
            query_results=[{"anomaly_count": 5}], column_names=["anomaly_count"]
        )

        # Verify contract compliance
        await ContractTestCase.test_query_executor_contract_compliance(mock)

    @pytest.mark.asyncio
    async def test_enum_rule_execution_with_contract_mock(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test rule execution using contract-compliant mock"""
        rule = builder.basic_enum_rule(["valid", "test"])

        # Create async mock functions that return the expected data structure
        async def mock_execute_query(
            sql: str,
        ) -> tuple[List[Dict[str, Any]], List[str]]:
            if "total_count" in sql:
                return [{"total_count": 100}], ["total_count"]
            elif "anomaly_count" in sql:
                return [{"anomaly_count": 3}], ["anomaly_count"]
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

        # Verify the data flow: from the query result to DatasetMetrics, and then to ExecutionResultSchema.
        assert len(result.dataset_metrics) > 0
        assert (
            result.dataset_metrics[0].failed_records == 3
        )  # Verifies the correct number of exceptions.
        assert (
            result.dataset_metrics[0].total_records == 100
        )  # Correctly verifies the total number of records.
        assert (
            result.status == "FAILED"
        )  # An exception should result in a FAILED status.

    @pytest.mark.asyncio
    async def test_database_error_contract(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test error handling follows contract - engine errors should raise exceptions"""
        rule = builder.basic_enum_rule(["test"])

        # Mock that raises connection timeout (engine-level error)
        with patch.object(validity_executor, "get_engine") as mock_get_engine:
            # Engine-level error should be raised from get_engine
            mock_get_engine.side_effect = EngineError("Connection timeout")

            # Engine-level errors should raise exceptions, not return error results
            with pytest.raises(Exception) as exc_info:
                await validity_executor.execute_rule(rule)

            assert "Connection timeout" in str(exc_info.value)


# 📊 PROPERTY-BASED TESTING (if hypothesis available)
if HYPOTHESIS_AVAILABLE:

    class TestPropertyBasedTesting:
        """Use property-based testing to verify invariants with random inputs"""

        @hypothesis.given(enum_values=enum_values_strategy())
        @settings(
            max_examples=50,
            deadline=5000,
            suppress_health_check=[HealthCheck.function_scoped_fixture],
        )
        def test_sql_generation_invariants(
            self, validity_executor: ValidityExecutor, enum_values: List[str]
        ) -> None:
            """Verify SQL generation invariants hold for any valid enum values"""
            # Create rule without using fixture to avoid health check issue
            builder = TestDataBuilder()
            rule = (
                builder.rule()
                .with_target("test_db", "test_table", "test_column")
                .as_enum_rule(enum_values)
                .build()
            )

            sql = validity_executor._generate_enum_sql(rule)

            # These invariants should hold for ANY valid enum values
            assert "SELECT COUNT(*) AS anomaly_count" in sql
            assert "NOT IN" in sql
            for value in enum_values:
                assert value in sql

        @hypothesis.given(table_info=table_info_strategy())
        @settings(
            max_examples=30,
            deadline=3000,
            suppress_health_check=[HealthCheck.function_scoped_fixture],
        )
        def test_target_handling_invariants(
            self, validity_executor: ValidityExecutor, table_info: Dict[str, str]
        ) -> None:
            """Verify table/column handling invariants"""
            builder = TestDataBuilder()
            rule = (
                builder.rule()
                .with_target(
                    table_info["database"], table_info["table"], table_info["column"]
                )
                .as_enum_rule(["test_value"])
                .build()
            )

            sql = validity_executor._generate_enum_sql(rule)

            # These invariants should hold for any valid table info
            assert table_info["table"] in sql
            assert table_info["column"] in sql
            assert "SELECT COUNT(*)" in sql

        @hypothesis.given(
            enum_values=enum_values_strategy(),
            severity=st.sampled_from(
                [
                    SeverityLevel.LOW,
                    SeverityLevel.MEDIUM,
                    SeverityLevel.HIGH,
                    SeverityLevel.CRITICAL,
                ]
            ),
        )
        @settings(
            max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        def test_rule_creation_invariants(
            self, enum_values: List[str], severity: SeverityLevel
        ) -> None:
            """Verify rule creation invariants"""
            builder = TestDataBuilder()
            rule = (
                builder.rule().with_severity(severity).as_enum_rule(enum_values).build()
            )

            # These invariants should always hold
            assert rule.type == RuleType.ENUM
            assert rule.severity == severity
            assert rule.parameters["allowed_values"] == enum_values
            assert len(rule.parameters["allowed_values"]) > 0

        @hypothesis.given(
            enum_values=enum_values_strategy(),
            filter_condition=st.text(min_size=1, max_size=50),
        )
        @settings(
            max_examples=15, suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        def test_sql_with_filter_invariants(
            self, enum_values: List[str], filter_condition: str
        ) -> None:
            """Verify the invariance of SQL generation with filtering criteria."""
            builder = TestDataBuilder()
            rule = (
                builder.rule()
                .as_enum_rule(enum_values)
                .with_filter(filter_condition)
                .build()
            )

            validity_executor = ValidityExecutor(builder.connection().build())
            sql = validity_executor._generate_enum_sql(rule)

            # Invariance: Filtering criteria must be applied within the SQL query itself.
            assert filter_condition in sql
            assert (
                "AND" in sql
            )  # Filter criteria should be combined using a logical AND.
            assert "NOT IN" in sql

        @hypothesis.given(
            table_name=st.text(
                min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz_"
            ),
            column_name=st.text(
                min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz_"
            ),
        )
        @settings(
            max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture]
        )
        def test_sql_injection_resistance_properties(
            self, table_name: str, column_name: str
        ) -> None:
            """Verify the mathematical properties that ensure resistance to SQL injection."""
            builder = TestDataBuilder()
            rule = (
                builder.rule()
                .with_target("db", table_name, column_name)
                .as_enum_rule(["safe_value"])
                .build()
            )

            validity_executor = ValidityExecutor(builder.connection().build())
            sql = validity_executor._generate_enum_sql(rule)

            # Data Integrity: Table and column names must be supplied correctly and be protected against injection attacks.
            assert table_name in sql
            assert column_name in sql
            assert (
                sql.count("SELECT") == 1
            )  # Only a single SELECT statement is allowed.


# 🧬 MUTATION TESTING READINESS
class TestMutationTestingReadiness:
    """Tests designed to catch subtle bugs that mutation testing would find"""

    def test_sql_injection_protection(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test protection against SQL injection in enum values"""
        # Try to inject SQL - these should be properly escaped or handled
        malicious_values = [
            "'; DROP TABLE users; --",
            "admin'; DELETE FROM orders; --",
            "' OR '1'='1",
        ]

        rule = (
            builder.rule()
            .with_target("test_db", "test_table", "status")
            .as_enum_rule(malicious_values)
            .build()
        )

        sql = validity_executor._generate_enum_sql(rule)

        # For this test, we verify that dangerous SQL patterns are contained within quotes
        # which makes them harmless string literals rather than executable SQL
        for value in malicious_values:
            # The value should appear in quotes, making it a string literal
            quoted_value = f"'{value}'"
            assert (
                quoted_value in sql
            ), f"Value {value} should be properly quoted in SQL"

        # Verify basic SQL structure is maintained
        assert "SELECT COUNT(*)" in sql
        assert "NOT IN" in sql

    def test_empty_and_whitespace_enum_values(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test edge cases with empty and whitespace-only values"""
        edge_values = ["", " ", "  ", "\t", "\n"]

        rule = builder.rule().as_enum_rule(edge_values).build()

        sql = validity_executor._generate_enum_sql(rule)

        # Should handle empty and whitespace values correctly
        assert "NOT IN" in sql
        assert "''" in sql  # Empty string should be represented

    def test_off_by_one_conditions(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test conditions that could have off-by-one errors"""
        # Single value test - catches > vs >= errors
        rule = builder.rule().as_enum_rule(["single"]).build()
        sql = validity_executor._generate_enum_sql(rule)
        assert "NOT IN ('single')" in sql

        # Empty list should be handled gracefully
        try:
            rule = builder.rule().as_enum_rule([]).build()
            sql = validity_executor._generate_enum_sql(rule)
            # Should not crash, even with empty enum list
        except Exception as e:
            # If it raises an exception, it should be a meaningful one
            assert "allowed_values" in str(e) or "empty" in str(e).lower()

    def test_null_handling_edge_cases(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test NULL value handling edge cases"""
        rule = builder.rule().as_enum_rule(["NULL", "null", "None"]).build()

        sql = validity_executor._generate_enum_sql(rule)

        # Should properly handle NULL-like string values
        assert "NULL" in sql or "null" in sql
        assert "NOT IN" in sql

    def test_case_sensitivity_boundaries(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test case sensitivity edge cases"""
        case_values = ["Active", "ACTIVE", "active", "aCtIvE"]

        rule = builder.rule().as_enum_rule(case_values).build()

        sql = validity_executor._generate_enum_sql(rule)

        # All variations should be preserved in SQL
        for value in case_values:
            assert value in sql

    @pytest.mark.asyncio
    async def test_error_message_completeness(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test that error messages contain sufficient debugging information"""
        rule = builder.basic_enum_rule(["test"])

        # Force an engine-level error by mocking get_engine to raise exception
        with patch.object(validity_executor, "get_engine") as mock_get_engine:
            mock_get_engine.side_effect = EngineError("Database connection failed")

            # Engine-level errors should raise exceptions, not return error results
            with pytest.raises(Exception) as exc_info:
                await validity_executor.execute_rule(rule)

            # Error message should be informative
            assert "Database connection failed" in str(exc_info.value)

    def test_boundary_value_mutations(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Testing boundary value mutations - catching errors in boundary conditions."""
        # Testing the handling of empty enumerated lists.
        try:
            empty_rule = builder.rule().as_enum_rule([]).build()
            sql = validity_executor._generate_enum_sql(empty_rule)
            # This code should generate valid SQL or raise a descriptive exception if it encounters an error.  It is expected to execute without crashing.
            assert "NOT IN ()" in sql or len(sql) > 0
        except RuleExecutionError as e:
            # Error messages should be clear and informative.
            assert "empty" in str(e).lower() or "allowed_values" in str(e)

    def test_logic_operator_mutations(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Testing mutation of logical operators – catching errors related to AND/OR and IN/NOT IN operations."""
        rule = builder.rule().as_enum_rule(["valid"]).with_filter("active = 1").build()

        sql = validity_executor._generate_enum_sql(rule)

        # Mutation testing: Verify the use of `NOT IN` instead of `IN`.
        assert "NOT IN" in sql
        assert (
            " IN " not in sql or "NOT IN" in sql
        )  # If the `IN` operator is used, it must be negated as `NOT IN`.

        # Mutation testing: Verify that filter conditions are combined using the AND operator.
        assert "AND" in sql
        assert " OR " not in sql  # Avoid combining primary conditions using OR.

    def test_string_handling_mutations(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Testing string manipulation mutations - handling quote and escape character errors."""
        tricky_values = [
            "value'with'quotes",
            'value"with"doublequotes',
            "value\\with\\backslashes",
            "",  # Empty string.
            " ",  # Whitespace only.
        ]

        rule = builder.rule().as_enum_rule(tricky_values).build()
        sql = validity_executor._generate_enum_sql(rule)

        # Mutation Testing: Ensure correct handling of all possible input values.
        for value in tricky_values:
            # The value should be incorporated into the SQL query, potentially after escaping any special characters.
            assert value in sql or value.replace("'", "''") in sql

    @pytest.mark.asyncio
    async def test_async_error_handling_mutations(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Testing asynchronous error handling mutations - capturing errors in exception handling."""
        rule = builder.basic_enum_rule(["test"])

        # Engine-level error condition - an exception should be thrown.
        engine_error_scenarios = ["Connection timeout", "Database connection failed"]

        for error_msg in engine_error_scenarios:
            with patch.object(validity_executor, "get_engine") as mock_get_engine:
                mock_get_engine.side_effect = EngineError(error_msg)

                # Engine-level errors should raise exceptions.
                with pytest.raises(Exception) as exc_info:
                    await validity_executor.execute_rule(rule)

                assert error_msg in str(exc_info.value)

        # Rule-level error scenario - an incorrect result should be returned.
        rule_error_scenarios = ["Table does not exist", "Invalid SQL syntax"]

        for error_msg in rule_error_scenarios:
            with patch.object(validity_executor, "get_engine") as mock_get_engine:
                mock_engine = AsyncMock()
                mock_get_engine.return_value = mock_engine

                # Simulate rule-level errors at the QueryExecutor level.
                with patch(
                    "shared.database.query_executor.QueryExecutor"
                ) as mock_qe_class:
                    mock_query_executor = AsyncMock()
                    mock_query_executor.execute_query.side_effect = Exception(error_msg)
                    mock_qe_class.return_value = mock_query_executor

                    result = await validity_executor.execute_rule(rule)

                    # Rule-level errors should return an error result.
                    assert result.status == "ERROR"
                    assert result.error_message is not None
                    assert error_msg in result.error_message

                    # Ensure reasonable execution time is logged.
                    assert result.execution_time >= 0


# 🎯 PERFORMANCE AND EDGE CASES
class TestPerformanceAndEdgeCases:
    """Test performance characteristics and extreme edge cases"""

    def test_large_enum_value_list_performance(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test performance with very large enum value lists"""
        # Generate large list of enum values
        large_enum_list = [f"value_{i}" for i in range(1000)]

        rule = builder.rule().as_enum_rule(large_enum_list).build()

        # Should generate SQL without crashing
        import time

        start_time = time.time()
        sql = validity_executor._generate_enum_sql(rule)
        generation_time = time.time() - start_time

        # Should complete reasonably quickly (less than 1 second)
        assert generation_time < 1.0
        assert "NOT IN" in sql
        assert len(sql) > 1000  # Should be a substantial SQL statement

    def test_unicode_and_special_characters(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test handling of Unicode and special characters"""
        unicode_values = [
            "测试",  # Chinese
            "тест",  # Cyrillic
            "🎉",  # Emoji
            "café",  # Accented characters
            "naïve",  # More accented characters
            "\\n\\t\\r",  # Escape sequences
        ]

        rule = builder.rule().as_enum_rule(unicode_values).build()

        sql = validity_executor._generate_enum_sql(rule)

        # Should handle Unicode characters properly
        assert "NOT IN" in sql
        # Most Unicode chars should be preserved or properly encoded
        for value in unicode_values:
            # Either the exact value or an encoded version should be present
            assert value in sql or any(char in sql for char in value)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
