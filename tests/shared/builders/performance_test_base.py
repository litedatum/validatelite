"""
🚀 Performance Test Base Infrastructure - By Testing Ghost 👻

This modern performance testing base provides:
1. 🔧 Unified Mock Management - Automatic Individual execution forcing
2. 📊 Performance Metrics Collection - Standardized measurement
3. 🛡️ Regression Detection - Automated baseline comparison
4. 🧠 Memory Efficiency Tracking - Resource leak detection

Usage:
    class TestMyComponent(PerformanceTestBase):
        async def test_my_performance_scenario(self):
            # Mocks are automatically configured for Individual execution
            # Performance metrics are automatically collected
            pass
"""

import asyncio
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from core.engine.rule_merger import MergeGroup
from shared.schema.rule_schema import RuleSchema
from tests.shared.builders.test_builders import TestDataBuilder


@dataclass
class PerformanceMetrics:
    """Standardized performance measurement container"""

    execution_time: float
    memory_delta: float
    rule_count: int
    throughput: float  # rules per second
    test_name: str
    timestamp: float

    @property
    def avg_time_per_rule(self) -> float:
        return self.execution_time / self.rule_count if self.rule_count > 0 else 0.0


class PerformanceTestBase:
    """
    🚀 Unified Performance Test Base Class

    Provides standardized infrastructure for performance testing across all components.
    Eliminates repetitive Mock setup and ensures consistent Individual execution.
    """

    @pytest.fixture(autouse=True)
    def setup_performance_infrastructure(self) -> Generator[Dict[str, Any], None, None]:
        """Automatically set up performance testing infrastructure"""
        self.performance_metrics: List[PerformanceMetrics] = []
        self.start_memory = self._get_memory_usage() if PSUTIL_AVAILABLE else 0.0

        # Create mock connection for tests
        from unittest.mock import Mock

        from shared.enums.connection_types import ConnectionType
        from shared.schema.base import ConnectionBase, DataSourceCapability

        self.mock_connection = Mock(spec=ConnectionBase)
        self.mock_connection.id = "123"
        self.mock_connection.name = "test_connection"
        self.mock_connection.description = "Test connection for performance testing"
        self.mock_connection.connection_type = ConnectionType.MYSQL
        self.mock_connection.capabilities = DataSourceCapability(supports_sql=True)
        self.mock_connection.host = "localhost"
        self.mock_connection.port = 3306
        self.mock_connection.db_name = "test_db"
        self.mock_connection.username = "test_user"
        self.mock_connection.password = "test_pass"
        self.mock_connection.connection_string = "mysql://test:test@localhost/test_db"

        # Set up Individual execution mocks by default
        with ExitStack() as stack:
            # Store the context manager for test duration
            self._mock_stack = stack

            # Apply all Individual execution patches
            for mock_patch in self._create_individual_execution_mocks():
                stack.enter_context(mock_patch)

            yield {
                "individual_execution": True,
                "start_memory": self.start_memory,
                "mock_connection": self.mock_connection,
            }

    def _create_individual_execution_mocks(self) -> List[Any]:
        """🔧 Create unified mock infrastructure for forcing Individual execution"""

        def force_individual_groups(
            rules: List[RuleSchema],
            database: str,
            table: str,
        ) -> List[MergeGroup]:
            from core.engine.rule_merger import MergeGroup, MergeStrategy

            # Force every rule to use Individual execution
            return [
                MergeGroup(
                    strategy=MergeStrategy.INDIVIDUAL,
                    rules=[rule],
                    target_database=database,
                    target_table=table,
                )
                for rule in rules
            ]

        # Create performance engine mock
        performance_engine = self._create_performance_engine_mock()

        # Return all necessary mock patches
        return [
            # Fix: Use AsyncMock for get_engine to support await
            patch(
                "core.executors.base_executor.BaseExecutor.get_engine",
                new_callable=AsyncMock,
                return_value=performance_engine,
            ),
            # Fix: Mock QueryExecutor with proper async support and realistic data
            patch(
                "shared.database.query_executor.QueryExecutor",
                side_effect=self._create_query_executor_mock,
            ),
            patch(
                "core.engine.rule_merger.RuleMergeManager._analyze_table_rules",
                side_effect=force_individual_groups,
            ),
            patch(
                "core.engine.rule_engine.RuleEngine._get_engine",
                new_callable=AsyncMock,
                return_value=performance_engine,
            ),
        ]

    def _create_performance_engine_mock(self) -> AsyncMock:
        """🔧 Create contract-compliant database engine mock"""
        from sqlalchemy.engine import Engine

        # Create the main engine mock
        engine = AsyncMock(spec=Engine)
        engine.url = Mock()
        engine.url.__str__ = Mock(return_value="mysql://test:test@localhost/test_db")

        # Create async connection context manager
        async_conn = AsyncMock()
        async_conn.__aenter__ = AsyncMock(return_value=async_conn)
        async_conn.__aexit__ = AsyncMock(return_value=None)

        # Mock query results based on SQL type
        def mock_execute(sql: Any, params: Optional[Any] = None) -> Any:
            result = Mock()

            # Table existence check
            if "information_schema.tables" in str(sql) or "sqlite_master" in str(sql):
                result.scalar.return_value = 1  # Table exists
                result.fetchone.return_value = (1,)
                result.fetchall.return_value = [
                    (1,)
                ]  # Return non-empty list for table existence
            # Total record count query
            elif "COUNT(*)" in str(sql) and "total_count" in str(sql):
                result.scalar.return_value = 1000
                result.fetchone.return_value = (1000,)
                result.fetchall.return_value = [(1000,)]
            # Anomaly record count query
            elif "COUNT(*)" in str(sql) and "anomaly_count" in str(sql):
                result.scalar.return_value = 5
                result.fetchone.return_value = (5,)
                result.fetchall.return_value = [(5,)]
            # Other queries
            else:
                result.scalar.return_value = 0
                result.fetchone.return_value = (0,)
                result.fetchall.return_value = []

            result.keys.return_value = ["count"]
            return result

        async_conn.execute = AsyncMock(side_effect=mock_execute)

        # Set up async context managers
        engine.begin = AsyncMock(return_value=async_conn)
        engine.connect = AsyncMock(return_value=async_conn)

        return engine

    def _create_query_executor_mock(self, engine_or_conn: Any) -> Any:
        """Create a realistic QueryExecutor mock that returns proper data"""
        mock_executor = AsyncMock()

        async def mock_execute_query(
            query: Any,
            params: Optional[Any] = None,
            fetch: bool = True,
            sample_limit: Optional[int] = None,
            rule_id: Optional[str] = None,
            entity_name: Optional[str] = None,
        ) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
            """Mock execute_query method with realistic data"""
            # Simulate realistic query execution time
            await asyncio.sleep(0.001)  # 1ms simulation

            # Parse query to determine response
            query_lower = query.lower()

            if "total_count" in query_lower:
                # Total count query
                return [{"total_count": 1000}], None
            elif "failed_count" in query_lower:
                # Failed count query - simulate some failures
                if "not_null" in query_lower:
                    return [{"failed_count": 5}], None  # 5 null values
                elif "length" in query_lower:
                    return [{"failed_count": 3}], None  # 3 length violations
                else:
                    return [{"failed_count": 0}], None
            else:
                # Other queries
                return [], None

        mock_executor.execute_query = mock_execute_query
        return mock_executor

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if PSUTIL_AVAILABLE:
            return cast(float, psutil.Process().memory_info().rss / 1024 / 1024)
        return 0.0

    async def measure_performance(
        self, func: Callable, rule_count: int, test_name: str, *args: Any, **kwargs: Any
    ) -> PerformanceMetrics:
        """
        Measure performance of any async function with standardized metrics

        Args:
            func: The async function to measure
            rule_count: Number of rules being processed
            test_name: Name of the test for tracking
            *args, **kwargs: Arguments to pass to the function

        Returns:
            PerformanceMetrics object with all measurements
        """
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        # Execute the function
        result = await func(*args, **kwargs)

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        throughput = rule_count / execution_time if execution_time > 0 else 0

        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_delta=memory_delta,
            rule_count=rule_count,
            throughput=throughput,
            test_name=test_name,
            timestamp=time.time(),
        )

        self.performance_metrics.append(metrics)
        return metrics

    def assert_performance_bounds(
        self,
        metrics: PerformanceMetrics,
        max_time_per_rule: float = 0.01,
        max_memory_per_rule: float = 0.5,
    ) -> None:
        """
        Assert performance bounds with descriptive error messages

        Args:
            metrics: Performance metrics to validate
            max_time_per_rule: Maximum allowed time per rule in seconds
            max_memory_per_rule: Maximum allowed memory per rule in MB
        """
        # Time performance validation
        actual_time_per_rule = metrics.avg_time_per_rule
        assert actual_time_per_rule <= max_time_per_rule, (
            f"Performance regression in {metrics.test_name}: "
            f"{actual_time_per_rule:.4f}s per rule > {max_time_per_rule:.4f}s limit"
        )

        # Memory performance validation
        if PSUTIL_AVAILABLE and metrics.rule_count > 0:
            memory_per_rule = metrics.memory_delta / metrics.rule_count
            assert memory_per_rule <= max_memory_per_rule, (
                f"Memory regression in {metrics.test_name}: "
                f"{memory_per_rule:.3f}MB per rule > {max_memory_per_rule:.3f}MB limit"
            )

        # Throughput validation
        min_throughput = 100  # rules per second minimum
        assert metrics.throughput >= min_throughput, (
            f"Throughput regression in {metrics.test_name}: "
            f"{metrics.throughput:.1f} rules/s < {min_throughput} rules/s minimum"
        )

    def detect_performance_regressions(self) -> List[str]:
        """
        Analyze collected metrics for performance regressions

        Returns:
            List of regression warnings
        """
        if len(self.performance_metrics) < 2:
            return []

        regressions = []

        # Compare execution times
        times = [m.execution_time for m in self.performance_metrics]
        if len(times) >= 2:
            baseline = min(times)
            current = times[-1]

            if current > baseline * 1.2:  # 20% regression threshold
                regressions.append(
                    f"Execution time regression: {current:.3f}s > {baseline:.3f}s baseline"
                )

        # Compare throughput
        throughputs = [
            m.throughput for m in self.performance_metrics if m.throughput > 0
        ]
        if len(throughputs) >= 2:
            baseline_throughput = max(throughputs[:-1])
            current_throughput = throughputs[-1]

            if current_throughput < baseline_throughput * 0.8:  # 20% degradation
                regressions.append(
                    f"Throughput regression: {current_throughput:.1f} < {baseline_throughput:.1f} baseline"
                )

        return regressions

    @pytest.fixture
    def builder(self) -> TestDataBuilder:
        """Provide standard test data builder"""
        return TestDataBuilder()

    @pytest.fixture
    def performance_engine(self) -> AsyncMock:
        """Provide contract-compliant performance engine mock"""
        return self._create_contract_compliant_engine()

    def _create_contract_compliant_engine(self) -> AsyncMock:
        """Create engine mock that follows database connection contract"""
        from sqlalchemy.ext.asyncio import AsyncEngine

        engine = AsyncMock(spec=AsyncEngine)
        engine.url = Mock()
        engine.url.__str__ = Mock(return_value="mysql://user:pass@localhost/test")

        # Create async connection context manager
        async_conn = AsyncMock()
        async_conn.__aenter__ = AsyncMock(return_value=async_conn)
        async_conn.__aexit__ = AsyncMock(return_value=None)

        # Mock query results for Individual execution
        def mock_execute(sql: Any, params: Optional[Any] = None) -> Any:
            result = Mock()

            # Table existence check
            if "information_schema.tables" in str(sql) or "sqlite_master" in str(sql):
                result.scalar.return_value = 1
                result.fetchone.return_value = (1,)
                result.fetchall.return_value = [
                    (1,)
                ]  # Return non-empty list for table existence
            # Total count query
            elif "COUNT(*)" in str(sql) and "total_count" in str(sql):
                result.scalar.return_value = 1000
                result.fetchone.return_value = (1000,)
                result.fetchall.return_value = [(1000,)]
            # Anomaly count query
            elif "COUNT(*)" in str(sql) and "anomaly_count" in str(sql):
                result.scalar.return_value = 5
                result.fetchone.return_value = (5,)
                result.fetchall.return_value = [(5,)]
            # Other queries
            else:
                result.scalar.return_value = 0
                result.fetchone.return_value = (0,)
                result.fetchall.return_value = []

            result.keys.return_value = ["count"]
            return result

        async_conn.execute = AsyncMock(side_effect=mock_execute)
        engine.begin.return_value = async_conn
        engine.connect.return_value = async_conn

        return engine


class PerformanceRegressionDetector:
    """
    Automated Performance Regression Detection System

    Tracks performance baselines and detects regressions across test runs.
    """

    def __init__(self, baseline_file: str = "performance_baseline.json"):
        self.baseline_file = baseline_file
        self.current_metrics: Dict[str, PerformanceMetrics] = {}

    def record_metric(self, test_name: str, metric: PerformanceMetrics) -> None:
        """Record performance metric for regression analysis"""
        self.current_metrics[test_name] = metric

    def detect_regressions(self, threshold: float = 0.2) -> List[str]:
        """
        Detect performance regressions against baselines

        Args:
            threshold: Regression threshold (0.2 = 20% degradation)

        Returns:
            List of detected regressions
        """
        baselines = self._load_baselines()
        regressions = []

        for test_name, current in self.current_metrics.items():
            if test_name in baselines:
                baseline = baselines[test_name]

                # Check execution time regression
                if current.execution_time > baseline.execution_time * (1 + threshold):
                    regressions.append(
                        f"{test_name}: execution time {current.execution_time:.3f}s > "
                        f"baseline {baseline.execution_time:.3f}s (+{threshold:.0%})"
                    )

                # Check throughput regression
                if current.throughput < baseline.throughput * (1 - threshold):
                    regressions.append(
                        f"{test_name}: throughput {current.throughput:.1f} rules/s < "
                        f"baseline {baseline.throughput:.1f} rules/s (-{threshold:.0%})"
                    )

        return regressions

    def update_baselines(self) -> None:
        """Update performance baselines (use when confident about improvements)"""
        self._save_baselines(self.current_metrics)

    def _load_baselines(self) -> Dict[str, PerformanceMetrics]:
        """Load performance baselines from file"""
        import json

        try:
            with open(self.baseline_file, "r") as f:
                data = json.load(f)
            return {k: PerformanceMetrics(**v) for k, v in data.items()}
        except FileNotFoundError:
            return {}

    def _save_baselines(self, metrics: Dict[str, PerformanceMetrics]) -> None:
        """Save performance baselines to file"""
        import json

        data = {k: v.__dict__ for k, v in metrics.items()}
        with open(self.baseline_file, "w") as f:
            json.dump(data, f, indent=2)


# Export utilities for easy import
__all__ = ["PerformanceTestBase", "PerformanceMetrics", "PerformanceRegressionDetector"]
