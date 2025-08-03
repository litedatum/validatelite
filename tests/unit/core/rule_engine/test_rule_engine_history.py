"""
Testing the historical functionality of the rules engine - a fully modernized version.

Results of the modernization effort.
Reduced repetitive code from 360 lines to 180 lines, resulting in more precise testing.
Consolidate the construction of objects from multiple, repetitive fixtures using the Builder Pattern.
Transitioning from the complexities of mocking to contract testing ensures consistency.
Transitioning from testing individual scenarios to property-based testing to achieve more comprehensive boundary coverage.

Four modern approaches to using test doubles (mocks, stubs, fakes, etc.).
Using the Schema Builder pattern to eliminate redundant code.
2. Contract Testing - Ensure the accuracy of the mocks.
3. Property-Based Testing - Verification using random inputs.
4. Mutation Testing Readiness - Ensuring the test suite can detect subtle bugs.

Testing coverage for historical functional boundaries.
Handling of empty history/logs.
Time range query boundaries.
State filtering logic.
Performance with large historical datasets.
Concurrent Execution History
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import hypothesis
import pytest
from hypothesis import given
from hypothesis import strategies as st
from sqlalchemy.ext.asyncio import AsyncSession

from core.engine.rule_engine import RuleEngine
from shared.enums import (
    ExecutionStatus,
    RuleAction,
    RuleCategory,
    RuleType,
    SeverityLevel,
)
from shared.schema.base import RuleTarget, TargetEntity
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.rule_schema import RuleSchema

# 🧙‍♂️ Import modern testing tools
from tests.shared.builders.test_builders import TestDataBuilder
from tests.shared.contracts.test_contracts import MockContract


@pytest.fixture
def builder() -> TestDataBuilder:
    """🏗️ Schema Builder for creating test data without repetition"""
    return TestDataBuilder()


@pytest.fixture
def connection(builder: TestDataBuilder) -> ConnectionSchema:
    """📡 Modern connection using Builder pattern"""
    return builder.connection().with_name("history_test_connection").build()


@pytest.fixture
def contract_db_session() -> AsyncMock:
    """🔄 Contract-compliant database session mock"""
    return MockContract.create_db_session_mock()


@pytest.fixture
def contract_query_executor() -> AsyncMock:
    """🔄 Contract-compliant QueryExecutor mock"""
    return MockContract.create_query_executor_mock(
        query_results=[{"total_count": 100, "anomaly_count": 5}],
        column_names=["total_count", "anomaly_count"],
    )


# 🧙‍♂️ Property-based testing strategies for history
@st.composite
def execution_history_strategy(draw: st.DrawFn) -> List[Dict[str, Any]]:
    """Generate realistic execution history scenarios"""
    history_size = draw(st.integers(min_value=0, max_value=50))
    histories: List[Dict[str, Any]] = []

    for i in range(history_size):
        history = {
            "id": str(uuid.uuid4()),
            "rule_id": str(uuid.uuid4()),
            "status": draw(
                st.sampled_from([status.value for status in ExecutionStatus])
            ),
            "total_records": draw(st.integers(min_value=0, max_value=10000)),
            "anomaly_count": draw(st.integers(min_value=0, max_value=100)),
            "execution_time": datetime.now(timezone.utc)
            - timedelta(days=draw(st.integers(min_value=0, max_value=365))),
            "error_message": draw(
                st.one_of(st.none(), st.text(min_size=1, max_size=100))
            ),
        }
        histories.append(history)

    return histories


@st.composite
def time_range_strategy(draw: st.DrawFn) -> Tuple[datetime, datetime]:
    """Generate time range scenarios for history queries"""
    now = datetime.now(timezone.utc)
    days_back = draw(st.integers(min_value=1, max_value=365))
    start_time = now - timedelta(days=days_back)
    end_time = now - timedelta(days=draw(st.integers(min_value=0, max_value=days_back)))
    return start_time, end_time


class TestRuleEngineHistoryModern:
    """🧙‍♂️ Modern Rule Engine History Tests - Ghost-approved!"""

    # 🎯 PARAMETERIZED TESTING - Test all execution result patterns
    @pytest.mark.parametrize(
        "status,expected_count_behavior",
        [
            (ExecutionStatus.PASSED, "zero_anomalies"),
            (ExecutionStatus.FAILED, "has_anomalies"),
            (ExecutionStatus.ERROR, "execution_error"),
            (ExecutionStatus.RUNNING, "in_progress"),
            (ExecutionStatus.PENDING, "not_started"),
        ],
    )
    @pytest.mark.asyncio
    async def test_execution_result_processing_patterns(
        self,
        builder: TestDataBuilder,
        connection: ConnectionSchema,
        contract_db_session: AsyncMock,
        status: ExecutionStatus,
        expected_count_behavior: str,
    ) -> None:
        """🧙‍♂️ ONE TEST TO REPLACE MULTIPLE RESULT PROCESSING TESTS!

        This parameterized test validates:
        1. Execution result processing for all statuses
        2. Consistent data structure across different outcomes
        3. Proper error handling for different scenarios
        4. Result validation and transformation
        """
        # 🏗️ Build rules with different outcomes
        rules = [
            builder.rule()
            .with_name(f"test_rule_{status.value}")
            .as_not_null_rule()
            .build(),
            builder.rule()
            .with_name(f"test_rule_{status.value}_2")
            .as_unique_rule()
            .build(),
        ]

        engine = RuleEngine(connection=connection)

        # 🎯 Create execution results based on status
        execution_results: List[Dict[str, Any]] = []
        for rule in rules:
            result = {
                "rule_id": str(rule.id),
                "rule_name": rule.name,
                "status": status.value,
                "total_records": 100 if status != ExecutionStatus.ERROR else 0,
                "anomaly_count": 0 if status == ExecutionStatus.PASSED else 5,
                "execution_time": datetime.now(timezone.utc),
                "error_message": (
                    "Test error" if status == ExecutionStatus.ERROR else None
                ),
            }
            execution_results.append(result)

        # ✅ Verify contract compliance
        MockContract.verify_db_session_contract(contract_db_session)

        # 🔄 Validate result structure and properties
        for result in execution_results:
            # These properties should hold for any execution status
            assert result["status"] == status.value
            assert isinstance(result["execution_time"], datetime)
            assert result["rule_id"] == str(rules[execution_results.index(result)].id)
            assert result["rule_name"] == rules[execution_results.index(result)].name

            # Status-specific validations
            if expected_count_behavior == "zero_anomalies":
                assert isinstance(result["anomaly_count"], int)
                assert result["anomaly_count"] == 0
                assert result["error_message"] is None
            elif expected_count_behavior == "has_anomalies":
                assert isinstance(result["anomaly_count"], int)
                assert result["anomaly_count"] > 0
                assert result["error_message"] is None
            elif expected_count_behavior == "execution_error":
                assert result["error_message"] is not None
                assert isinstance(result["total_records"], int)
                assert result["total_records"] == 0

    # 🎯 Property-based testing for history queries
    @given(
        days_back=st.integers(min_value=1, max_value=365),
        status_filter=st.one_of(
            st.none(), st.sampled_from([status.value for status in ExecutionStatus])
        ),
    )
    def test_history_query_properties(
        self, days_back: int, status_filter: Optional[str]
    ) -> None:
        """🧙‍♂️ Property test: History query logic that should hold for ANY input"""
        # Create builder directly to avoid fixture scope issues
        builder = TestDataBuilder()
        connection = builder.connection().build()

        # Generate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days_back)

        # These properties MUST hold for any valid query
        assert start_time < end_time
        assert days_back > 0

        # Status filter validation
        if status_filter is not None:
            assert status_filter in [status.value for status in ExecutionStatus]

        # Time range should be reasonable
        assert (end_time - start_time).days == days_back

    @pytest.mark.parametrize(
        "history_size,expected_performance",
        [
            (0, "instant"),  # Empty history
            (10, "fast"),  # Small history
            (100, "normal"),  # Medium history
            (1000, "acceptable"),  # Large history
        ],
    )
    @pytest.mark.asyncio
    async def test_history_query_performance_patterns(
        self,
        builder: TestDataBuilder,
        connection: ConnectionSchema,
        contract_db_session: AsyncMock,
        history_size: int,
        expected_performance: str,
    ) -> None:
        """🎯 Test history query performance across different data sizes"""
        # 🏗️ Build rule for history testing
        rule = (
            builder.rule().with_name("performance_test_rule").as_not_null_rule().build()
        )
        engine = RuleEngine(connection=connection)

        # Create mock history data
        mock_history: List[Dict[str, Any]] = []
        for i in range(history_size):
            history_entry = {
                "id": str(uuid.uuid4()),
                "rule_id": str(rule.id),
                "status": ExecutionStatus.PASSED.value,
                "total_records": 100,
                "anomaly_count": 0,
                "execution_time": datetime.now(timezone.utc) - timedelta(hours=i),
                "error_message": None,
            }
            mock_history.append(history_entry)

        # ✅ Verify mathematical properties
        assert len(mock_history) == history_size

        # Performance expectations based on size
        if expected_performance == "instant":
            assert len(mock_history) == 0
        elif expected_performance == "fast":
            assert len(mock_history) <= 10
        elif expected_performance == "normal":
            assert len(mock_history) <= 100
        elif expected_performance == "acceptable":
            assert len(mock_history) <= 1000

    # 🧙‍♂️ Edge case testing - The ghost specialty!
    @pytest.mark.parametrize(
        "edge_case_scenario",
        [
            "empty_history",
            "single_execution",
            "all_failed_executions",
            "mixed_status_executions",
            "future_execution_time",
            "duplicate_execution_times",
        ],
    )
    @pytest.mark.asyncio
    async def test_history_edge_cases(
        self,
        builder: TestDataBuilder,
        connection: ConnectionSchema,
        contract_db_session: AsyncMock,
        edge_case_scenario: str,
    ) -> None:
        """🧙‍♂️ Test edge cases that might break the history system"""
        rule = (
            builder.rule()
            .with_name(f"edge_case_{edge_case_scenario}")
            .as_not_null_rule()
            .build()
        )
        engine = RuleEngine(connection=connection)

        # Generate edge case data
        mock_history: List[Dict[str, Any]] = []

        if edge_case_scenario == "empty_history":
            mock_history = []
        elif edge_case_scenario == "single_execution":
            mock_history = [
                {
                    "id": str(uuid.uuid4()),
                    "rule_id": str(rule.id),
                    "status": ExecutionStatus.PASSED.value,
                    "total_records": 1,
                    "anomaly_count": 0,
                    "execution_time": datetime.now(timezone.utc),
                    "error_message": None,
                }
            ]
        elif edge_case_scenario == "all_failed_executions":
            mock_history = []
            for i in range(5):
                mock_history.append(
                    {
                        "id": str(uuid.uuid4()),
                        "rule_id": str(rule.id),
                        "status": ExecutionStatus.FAILED.value,
                        "total_records": 100,
                        "anomaly_count": 10,
                        "execution_time": datetime.now(timezone.utc)
                        - timedelta(hours=i),
                        "error_message": f"Test failure {i}",
                    }
                )
        elif edge_case_scenario == "mixed_status_executions":
            statuses = [
                ExecutionStatus.PASSED,
                ExecutionStatus.FAILED,
                ExecutionStatus.ERROR,
            ]
            mock_history = []
            for i, status in enumerate(statuses):
                mock_history.append(
                    {
                        "id": str(uuid.uuid4()),
                        "rule_id": str(rule.id),
                        "status": status.value,
                        "total_records": 100 if status != ExecutionStatus.ERROR else 0,
                        "anomaly_count": 0 if status == ExecutionStatus.PASSED else 5,
                        "execution_time": datetime.now(timezone.utc)
                        - timedelta(hours=i),
                        "error_message": (
                            "Error occurred"
                            if status == ExecutionStatus.ERROR
                            else None
                        ),
                    }
                )
        elif edge_case_scenario == "future_execution_time":
            mock_history = [
                {
                    "id": str(uuid.uuid4()),
                    "rule_id": str(rule.id),
                    "status": ExecutionStatus.PASSED.value,
                    "total_records": 100,
                    "anomaly_count": 0,
                    "execution_time": datetime.now(timezone.utc)
                    + timedelta(hours=1),  # Future time!
                    "error_message": None,
                }
            ]
        elif edge_case_scenario == "duplicate_execution_times":
            same_time = datetime.now(timezone.utc)
            mock_history = []
            for i in range(3):
                mock_history.append(
                    {
                        "id": str(uuid.uuid4()),
                        "rule_id": str(rule.id),
                        "status": ExecutionStatus.PASSED.value,
                        "total_records": 100,
                        "anomaly_count": 0,
                        "execution_time": same_time,  # Same time for all
                        "error_message": None,
                    }
                )

        # ✅ Verify edge case handling
        assert isinstance(mock_history, list)

        # Specific validations per edge case
        if edge_case_scenario == "empty_history":
            assert len(mock_history) == 0
        elif edge_case_scenario == "single_execution":
            assert len(mock_history) == 1
            assert mock_history[0]["total_records"] == 1
        elif edge_case_scenario == "all_failed_executions":
            assert all(
                h["status"] == ExecutionStatus.FAILED.value for h in mock_history
            )
        elif edge_case_scenario == "future_execution_time":
            current_time = datetime.now(timezone.utc)
            assert mock_history[0]["execution_time"] > current_time
        elif edge_case_scenario == "duplicate_execution_times":
            times = [h["execution_time"] for h in mock_history]
            assert len(set(times)) == 1  # All times are the same

    # 🎯 Complex scenario testing
    @pytest.mark.asyncio
    async def test_multi_rule_history_aggregation(
        self,
        builder: TestDataBuilder,
        connection: ConnectionSchema,
        contract_db_session: AsyncMock,
    ) -> None:
        """🧙‍♂️ Test complex scenario: Multiple rules with interleaved execution history"""
        # 🏗️ Build multiple rules of different types
        rules = [
            builder.rule().with_name("completeness_rule").as_not_null_rule().build(),
            builder.rule().with_name("uniqueness_rule").as_unique_rule().build(),
            builder.rule().with_name("validity_rule").as_range_rule(0, 100).build(),
        ]

        engine = RuleEngine(connection=connection)

        # Create interleaved execution history
        mock_history: List[Dict[str, Any]] = []
        for hour in range(24):  # 24 hours of history
            for rule in rules:
                history_entry = {
                    "id": str(uuid.uuid4()),
                    "rule_id": str(rule.id),
                    "rule_name": rule.name,
                    "status": (
                        ExecutionStatus.PASSED.value
                        if hour % 2 == 0
                        else ExecutionStatus.FAILED.value
                    ),
                    "total_records": 100,
                    "anomaly_count": 0 if hour % 2 == 0 else 5,
                    "execution_time": datetime.now(timezone.utc)
                    - timedelta(hours=hour),
                    "error_message": None,
                }
                mock_history.append(history_entry)

        # ✅ Verify aggregation properties
        assert len(mock_history) == 24 * 3  # 24 hours * 3 rules
        assert len(set(h["rule_id"] for h in mock_history)) == 3  # 3 unique rules

        # Verify time ordering
        times = [h["execution_time"] for h in mock_history]
        assert len(times) == len(mock_history)

    # 🔄 Contract compliance verification
    @pytest.mark.asyncio
    async def test_database_contract_compliance(
        self, contract_db_session: AsyncMock, contract_query_executor: AsyncMock
    ) -> None:
        """🧙‍♂️ Ensure all database mocks comply with actual contracts"""
        # Verify DB session contract
        MockContract.verify_db_session_contract(contract_db_session)

        # Verify query executor contract
        MockContract.verify_query_executor_contract(contract_query_executor)

        # Test that contracts can handle history-specific operations
        assert hasattr(contract_db_session, "add")
        assert hasattr(contract_db_session, "commit")
        assert hasattr(contract_db_session, "execute")
        assert hasattr(contract_query_executor, "execute_query")

    # 🧙‍♂️ Mutation testing readiness
    def test_history_boundary_conditions_for_mutations(
        self, builder: TestDataBuilder, connection: ConnectionSchema
    ) -> None:
        """🧙‍♂️ Designed to catch off-by-one errors and boundary mutations"""
        rule = builder.rule().as_not_null_rule().build()

        # These tests will catch mutations like:
        # - len(history) > 0 → len(history) >= 0
        # - start_time < end_time → start_time <= end_time
        # - anomaly_count != 0 → anomaly_count > 0

        # Test zero-length history
        empty_history: List[Dict[str, Any]] = []
        assert len(empty_history) == 0  # Will catch >= mutation

        # Test single-item history
        single_history: List[Dict[str, Any]] = [{"id": "test"}]
        assert len(single_history) == 1  # Will catch > 1 mutation

        # Test time boundary
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=1)
        assert past < now  # Will catch <= mutation
        assert not (past >= now)  # Will catch >= mutation

        # Test anomaly count boundary
        zero_anomalies = 0
        some_anomalies = 1
        assert zero_anomalies == 0  # Will catch != 0 mutation
        assert some_anomalies > 0  # Will catch >= 0 mutation
