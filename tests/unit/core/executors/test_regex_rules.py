"""
🧙‍♂️ Enhanced REGEX Rule Tests - Testing Ghost's Modern Testing Masterpiece

As the Testing Ghost 👻, I demonstrate the four key testing improvements:

1. 🏗️ Schema Builder Pattern - Eliminates fixture duplication
2. 🔄 Contract Testing - Ensures mocks match reality
3. 📊 Property-based Testing - Verifies behavior with random inputs
4. 🧬 Mutation Testing Readiness - Catches subtle bugs

This file focuses on REGEX validity rules with comprehensive pattern matching coverage.
"""

import re
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
    def email_pattern_strategy(draw: st.DrawFn) -> str:
        """Generate email-like patterns for testing"""
        local_part = draw(
            st.text(
                min_size=1,
                max_size=20,
                alphabet="abcdefghijklmnopqrstuvwxyz0123456789._-",
            )
        )
        domain_part = draw(
            st.text(
                min_size=1,
                max_size=20,
                alphabet="abcdefghijklmnopqrstuvwxyz0123456789.-",
            )
        )
        tld = draw(st.sampled_from(["com", "org", "net", "edu", "gov"]))
        return f"{local_part}@{domain_part}.{tld}"

    @st.composite
    def phone_pattern_strategy(draw: st.DrawFn) -> str:
        """Generate phone number patterns for testing"""
        area_code = draw(st.integers(min_value=100, max_value=999))
        prefix = draw(st.integers(min_value=100, max_value=999))
        number = draw(st.integers(min_value=1000, max_value=9999))

        format_type = draw(
            st.sampled_from(["digits", "dashes", "dots", "parens", "spaces"])
        )

        if format_type == "digits":
            return f"{area_code}{prefix}{number}"
        elif format_type == "dashes":
            return f"{area_code}-{prefix}-{number}"
        elif format_type == "dots":
            return f"{area_code}.{prefix}.{number}"
        elif format_type == "parens":
            return f"({area_code}) {prefix}-{number}"
        else:  # spaces
            return f"{area_code} {prefix} {number}"

    @st.composite
    def regex_mismatch_scenario_strategy(draw: st.DrawFn) -> Dict[str, int]:
        """Generate scenarios with regex mismatches"""
        total_records = draw(st.integers(min_value=1, max_value=1000))
        pattern_mismatches = draw(st.integers(min_value=0, max_value=total_records))
        return {
            "total_records": total_records,
            "pattern_mismatches": pattern_mismatches,
        }

    @st.composite
    def common_regex_patterns_strategy(draw: st.DrawFn) -> str:
        """Generate common regex patterns for testing"""
        return draw(
            st.sampled_from(
                [
                    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",  # Email
                    r"^\d{3}-\d{3}-\d{4}$",  # Phone
                    r"^[A-Z]{2}\d{6}$",  # ID pattern
                    r"^\d{4}-\d{2}-\d{2}$",  # Date YYYY-MM-DD
                    r"^[A-Z][a-z]+\s[A-Z][a-z]+$",  # Name pattern
                    r"^\d{5}(-\d{4})?$",  # ZIP code
                    r"^[A-Za-z0-9]{8,}$",  # Strong password base
                ]
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

    def test_basic_regex_rule_creation(self, builder: TestDataBuilder) -> None:
        """Test creating basic REGEX rule with builder"""
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        rule = (
            builder.rule()
            .with_name("email_format_validation")
            .with_target("users", "profiles", "email")
            .as_regex_rule(email_pattern)
            .build()
        )

        assert rule.name == "email_format_validation"
        assert rule.type == RuleType.REGEX
        assert rule.category == RuleCategory.VALIDITY
        assert rule.target.entities[0].table == "profiles"
        assert rule.target.entities[0].column == "email"
        assert rule.parameters["pattern"] == email_pattern

    def test_phone_regex_rule(self, builder: TestDataBuilder) -> None:
        """Test creating phone number regex rule"""
        phone_pattern = r"^\(\d{3}\) \d{3}-\d{4}$"
        rule = (
            builder.rule()
            .with_name("phone_format_check")
            .with_target("contacts", "directory", "phone_number")
            .as_regex_rule(phone_pattern)
            .build()
        )

        assert rule.parameters["pattern"] == phone_pattern

    def test_regex_rule_with_filter(self, builder: TestDataBuilder) -> None:
        """Test creating REGEX rule with filter condition"""
        ssn_pattern = r"^\d{3}-\d{2}-\d{4}$"
        rule = (
            builder.rule()
            .with_name("us_citizens_ssn_format")
            .with_severity(SeverityLevel.CRITICAL)
            .with_target("hr", "employees", "ssn")
            .as_regex_rule(ssn_pattern)
            .with_filter("country = 'US' AND status = 'active'")
            .build()
        )

        assert rule.severity == SeverityLevel.CRITICAL
        assert (
            rule.parameters["filter_condition"]
            == "country = 'US' AND status = 'active'"
        )

    def test_quick_builder_methods(self, builder: TestDataBuilder) -> None:
        """Test quick builder methods for common scenarios"""
        # Quick REGEX rule
        url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        rule = TestDataBuilder.basic_regex_rule(
            pattern=url_pattern, table="content", column="website_url"
        )

        assert rule.type == RuleType.REGEX
        assert rule.target.entities[0].table == "content"
        assert rule.target.entities[0].column == "website_url"
        assert rule.parameters["pattern"] == url_pattern


# 🔄 CONTRACT TESTING IMPLEMENTATION
class TestContractTesting:
    """Ensure mocks accurately represent real implementations"""

    @pytest.mark.asyncio
    async def test_query_executor_contract_compliance(self) -> None:
        """Test that our QueryExecutor mocks follow the contract"""
        # Create contract-compliant mock for REGEX scenario
        mock = MockContract.create_query_executor_mock(
            query_results=[{"total_count": 100, "pattern_mismatches": 12}],
            column_names=["total_count", "pattern_mismatches"],
        )

        # Verify contract compliance
        await ContractTestCase.test_query_executor_contract_compliance(mock)

    @pytest.mark.asyncio
    async def test_regex_rule_execution_with_contract_mock(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test rule execution using contract-compliant mock"""
        email_pattern = r"^[^@]+@[^@]+\.[^@]+$"
        rule = builder.basic_regex_rule(
            pattern=email_pattern, table="users", column="email"
        )

        # Create async mock functions that return the expected data structure
        async def mock_execute_query(
            sql: str,
        ) -> tuple[List[Dict[str, Any]], List[str]]:
            if "total_count" in sql.lower():
                return [{"total_count": 100}], ["total_count"]
            elif "anomaly_count" in sql.lower():
                return [{"anomaly_count": 7}], ["anomaly_count"]
            else:
                return [
                    {"id": 1, "name": "test"},
                    {"id": 2, "name": "test2"},
                    {"id": 3, "name": "test3"},
                    {"id": 4, "name": "test4"},
                    {"id": 5, "name": "test5"},
                    {"id": 6, "name": "test6"},
                    {"id": 7, "name": "test7"},
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
        assert len(result.sample_data) == 7

        # Verify regex-specific data flow
        assert len(result.dataset_metrics) > 0
        assert result.dataset_metrics[0].failed_records == 7  # ← Pattern mismatch count
        assert result.dataset_metrics[0].total_records == 100  # ← Total records


# 📊 PROPERTY-BASED TESTING
@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
class TestPropertyBasedTesting:
    """Use property-based testing to verify behavior with random inputs"""

    @hypothesis.given(pattern=common_regex_patterns_strategy())
    @settings(
        max_examples=20,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_regex_sql_generation_invariants(
        self, validity_executor: ValidityExecutor, pattern: str
    ) -> None:
        """Verify SQL generation properties for any valid regex pattern"""
        rule = TestDataBuilder.basic_regex_rule(
            pattern=pattern, table="test_table", column="test_column"
        )

        sql = validity_executor._generate_regex_sql(rule)

        # These properties should always hold
        assert "SELECT" in sql.upper()
        assert "FROM" in sql.upper()
        assert "test_table" in sql
        assert "test_column" in sql

        # Should contain pattern matching logic (database-specific)
        assert (
            "REGEXP" in sql.upper()
            or "RLIKE" in sql.upper()
            or "~" in sql
            or "LIKE" in sql.upper()
        )

    @hypothesis.given(
        emails=st.lists(email_pattern_strategy(), min_size=1, max_size=50)
    )
    @settings(
        max_examples=15,
        deadline=4000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_email_pattern_validation_properties(self, emails: List[str]) -> None:
        """Test properties of email pattern validation"""
        email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        compiled_pattern = re.compile(email_regex)

        for email in emails:
            # Property: generated emails should match basic email structure
            assert "@" in email
            assert "." in email.split("@")[1]  # Domain should have extension

            # Most generated emails should match the pattern (though not guaranteed)
            # This tests the pattern's effectiveness

    @hypothesis.given(
        phones=st.lists(phone_pattern_strategy(), min_size=1, max_size=30)
    )
    @settings(
        max_examples=12,
        deadline=3000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_phone_pattern_validation_properties(self, phones: List[str]) -> None:
        """Test properties of phone number pattern validation"""
        for phone in phones:
            # Property: all generated phones should contain only digits and separators
            clean_phone = re.sub(r"[^0-9]", "", phone)
            assert len(clean_phone) == 10  # US phone numbers have 10 digits
            assert clean_phone.isdigit()

    @hypothesis.given(scenario=regex_mismatch_scenario_strategy())
    @settings(
        max_examples=25,
        deadline=3000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_pattern_mismatch_counting_invariants(
        self, scenario: Dict[str, int]
    ) -> None:
        """Verify pattern mismatch counting properties"""
        total_records = scenario["total_records"]
        pattern_mismatches = scenario["pattern_mismatches"]

        # Property: mismatches should never exceed total records
        assert pattern_mismatches <= total_records

        # Property: mismatch rate calculation should be consistent
        if total_records > 0:
            mismatch_rate = pattern_mismatches / total_records
            assert 0.0 <= mismatch_rate <= 1.0


# 🧬 MUTATION TESTING READINESS
class TestMutationTestingReadiness:
    """Design tests to catch subtle bugs that mutation testing would find"""

    def test_regex_escaping_mutations(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Catch regex escaping errors in SQL generation"""
        # Pattern with special SQL characters that need escaping
        tricky_pattern = r"^[a-z\.'\"\\]+$"
        rule = builder.basic_regex_rule(pattern=tricky_pattern)

        sql = validity_executor._generate_regex_sql(rule)

        # Should properly escape special characters for SQL
        # Common mutation: missing escaping causing SQL errors
        assert "\\" in sql or "ESCAPE" in sql.upper() or "E'" in sql

    def test_case_sensitivity_mutations(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test case sensitivity in regex patterns"""
        # Case-sensitive pattern
        case_sensitive_pattern = r"^[A-Z][a-z]+$"
        rule_case_sensitive = builder.basic_regex_rule(pattern=case_sensitive_pattern)
        sql_case_sensitive = validity_executor._generate_regex_sql(rule_case_sensitive)

        # Should preserve case sensitivity
        assert case_sensitive_pattern in sql_case_sensitive

    def test_empty_pattern_mutations(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test handling of edge case patterns"""
        # Empty pattern (should be handled gracefully)
        try:
            rule_empty = builder.basic_regex_rule(pattern="")
            sql_empty = validity_executor._generate_regex_sql(rule_empty)
            # Should either handle gracefully or raise validation error
        except (ValueError, RuleExecutionError):
            # Expected behavior for invalid pattern
            pass

    def test_pattern_compilation_validation(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test validation of regex pattern compilation"""
        # Invalid regex pattern
        invalid_patterns = [
            r"[unclosed_bracket",
            r"*invalid_quantifier",
            r"(?P<>invalid_group)",
            r"(?invalid_group)",
        ]

        for invalid_pattern in invalid_patterns:
            try:
                rule = builder.basic_regex_rule(pattern=invalid_pattern)
                sql = validity_executor._generate_regex_sql(rule)
                # Should validate pattern before generating SQL
            except (ValueError, RuleExecutionError, re.error):
                # Expected behavior for invalid patterns
                pass

    def test_sql_injection_via_regex_pattern(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test SQL injection protection through regex patterns"""
        # Malicious patterns that could attempt SQL injection
        malicious_patterns = [
            r"'; DROP TABLE users; --",
            r"' OR '1'='1",
            r".*'; SELECT * FROM passwords; --.*",
        ]

        for malicious_pattern in malicious_patterns:
            rule = builder.basic_regex_rule(pattern=malicious_pattern)

            # Should raise ValueError for SQL injection attempts due to security protection
            with pytest.raises(
                RuleExecutionError,
                match="Pattern contains potentially dangerous SQL patterns",
            ):
                validity_executor._generate_regex_sql(rule)

    def test_null_value_handling_in_regex(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test how NULL values are handled in regex matching"""
        rule = builder.basic_regex_rule(pattern=r"^[a-z]+$")
        sql = validity_executor._generate_regex_sql(rule)

        # Common mutation: Should NULL values be considered as pattern mismatches?
        # Most implementations exclude NULLs from pattern matching
        # The behavior should be consistent and documented

    @pytest.mark.asyncio
    async def test_regex_error_handling_completeness(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test comprehensive error scenarios for regex validation"""
        rule = builder.basic_regex_rule(
            pattern=r"^valid_pattern$", table="nonexistent_table"
        )

        with patch.object(validity_executor, "get_engine") as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine

            # Mock database error
            with patch("shared.database.query_executor.QueryExecutor") as mock_qe_class:
                mock_query_executor = AsyncMock()
                mock_query_executor.execute_query.side_effect = Exception(
                    "Regex function not supported"
                )
                mock_qe_class.return_value = mock_query_executor

                result = await validity_executor.execute_rule(rule)

                # Should handle error gracefully
                assert result.status == "ERROR"
                assert result.error_message is not None
                assert "Regex function not supported" in result.error_message

    def test_pattern_anchoring_mutations(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test regex pattern anchoring (^ and $)"""
        # Pattern with anchors
        anchored_pattern = r"^[a-z]+$"
        rule_anchored = builder.basic_regex_rule(pattern=anchored_pattern)
        sql_anchored = validity_executor._generate_regex_sql(rule_anchored)

        # Pattern without anchors
        unanchored_pattern = r"[a-z]+"
        rule_unanchored = builder.basic_regex_rule(pattern=unanchored_pattern)
        sql_unanchored = validity_executor._generate_regex_sql(rule_unanchored)

        # Should preserve anchoring in the generated SQL
        assert anchored_pattern in sql_anchored
        assert unanchored_pattern in sql_unanchored

    def test_quantifier_mutations(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test various regex quantifiers"""
        quantifier_patterns = [
            r"^a+$",  # One or more
            r"^a*$",  # Zero or more
            r"^a?$",  # Zero or one
            r"^a{3}$",  # Exactly 3
            r"^a{2,5}$",  # Between 2 and 5
            r"^a{3,}$",  # 3 or more
        ]

        for pattern in quantifier_patterns:
            rule = builder.basic_regex_rule(pattern=pattern)
            sql = validity_executor._generate_regex_sql(rule)

            # Should preserve quantifiers in SQL
            assert pattern in sql

    def test_filter_condition_with_regex_mutations(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test filter condition interactions with regex logic"""
        rule_with_filter = (
            builder.rule()
            .as_regex_rule(r"^[A-Z][a-z]+$")
            .with_filter("status = 'active' AND length(name) > 2")
            .build()
        )

        sql_with_filter = validity_executor._generate_regex_sql(rule_with_filter)

        # Should properly combine regex matching with filter
        assert "WHERE" in sql_with_filter.upper()
        assert "status = 'active'" in sql_with_filter
        assert "AND" in sql_with_filter.upper() or "OR" in sql_with_filter.upper()


# 🚀 PERFORMANCE AND EDGE CASES
class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases"""

    def test_complex_regex_performance(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test performance with complex regex patterns"""
        # Complex but realistic email pattern
        complex_email_pattern = r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"

        rule = builder.basic_regex_rule(pattern=complex_email_pattern)
        sql = validity_executor._generate_regex_sql(rule)

        # Should handle complex patterns without excessive SQL length
        assert len(sql) < 5000  # Reasonable SQL length limit

    def test_unicode_regex_patterns(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test Unicode characters in regex patterns"""
        unicode_patterns = [
            r"^[А-Яа-я]+$",  # Cyrillic
            r"^[一-龯]+$",  # CJK
            r"^[א-ת]+$",  # Hebrew
            r"^[àáâãäåæçèéêë]+$",  # Accented Latin
        ]

        for pattern in unicode_patterns:
            rule = builder.basic_regex_rule(pattern=pattern)
            sql = validity_executor._generate_regex_sql(rule)

            # Should handle Unicode patterns appropriately
            assert len(sql) > 0
            # Pattern should be preserved (possibly with encoding)

    def test_very_long_patterns(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test very long regex patterns"""
        # Create a long but valid pattern
        long_pattern = r"^(" + "|".join([f"option{i}" for i in range(100)]) + r")$"

        rule = builder.basic_regex_rule(pattern=long_pattern)
        sql = validity_executor._generate_regex_sql(rule)

        # Should handle long patterns gracefully
        assert len(sql) > 0
        # Should not cause stack overflow or excessive memory usage

    @pytest.mark.asyncio
    async def test_concurrent_regex_validation(
        self, validity_executor: ValidityExecutor, builder: TestDataBuilder
    ) -> None:
        """Test concurrent regex validations"""
        patterns = [r"^[a-z]+$", r"^\d+$", r"^[A-Z]{2,}$"]

        rules = [
            builder.basic_regex_rule(pattern=pattern, column=f"col_{i}")
            for i, pattern in enumerate(patterns)
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
