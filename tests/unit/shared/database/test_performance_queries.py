"""
Performance Testing

This tests the query performance of the `QueryExecutor` class, including:
Basic query performance.
2. Performance of bulk queries/queries on large datasets.
3. DataFrame Query Performance
4. Transaction Performance
5. Concurrent Query Performance
"""

import time
from typing import Tuple
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from shared.database.query_executor import QueryExecutor

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_async_engine() -> Tuple[MagicMock, MagicMock]:
    """Create asynchronous mock database engine and connection objects."""
    mock_engine: MagicMock = MagicMock(spec=AsyncEngine)
    mock_conn: MagicMock = MagicMock(spec=AsyncConnection)
    mock_engine.connect.return_value.__aenter__.return_value = mock_conn
    mock_conn.execute = AsyncMock()
    return mock_engine, mock_conn


@pytest.fixture
def query_executor(mock_async_engine: Tuple[MagicMock, MagicMock]) -> QueryExecutor:
    mock_engine, _ = mock_async_engine
    return QueryExecutor(mock_engine)


class TestPerformanceQueries:
    """Test query performance."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_async_engine: Tuple[MagicMock, MagicMock]) -> None:
        self.mock_engine, self.mock_conn = mock_async_engine
        self.query_executor = QueryExecutor(self.mock_engine)

    async def test_basic_query_performance(self) -> None:
        """Test basic query performance."""
        mock_result = MagicMock()
        mock_result.keys.return_value = ["id", "name"]
        mock_result.fetchall = MagicMock(
            return_value=[(i, f"name_{i}") for i in range(1000)]
        )
        mock_result.rowcount = 1000
        mock_result.returns_rows = True
        self.mock_conn.execute.return_value = mock_result

        start_time = time.time()
        results, _ = await self.query_executor.execute_query(
            "SELECT id, name FROM large_table"
        )
        execution_time = time.time() - start_time
        assert len(results) == 1000
        assert execution_time < 1.0

    async def test_large_result_set_performance(self) -> None:
        """Test performance with large result sets."""
        mock_result = MagicMock()
        mock_result.keys.return_value = ["id", "name", "value", "description"]
        mock_result.fetchall = MagicMock(
            return_value=[
                (i, f"name_{i}", i * 1.5, f"description_{i}" * 10) for i in range(50000)
            ]
        )
        mock_result.rowcount = 50000
        mock_result.returns_rows = True
        self.mock_conn.execute.return_value = mock_result

        start_time = time.time()
        results, _ = await self.query_executor.execute_query(
            "SELECT * FROM very_large_table"
        )
        execution_time = time.time() - start_time
        assert len(results) == 50000
        assert execution_time < 5.0

    async def test_complex_query_performance(self) -> None:
        """Test complex query performance."""
        mock_result = MagicMock()
        mock_result.keys.return_value = ["id", "name", "total", "avg_value"]
        mock_result.fetchall = MagicMock(
            return_value=[(i, f"name_{i}", i * 100, i * 1.5) for i in range(1000)]
        )
        mock_result.rowcount = 1000
        mock_result.returns_rows = True
        self.mock_conn.execute.return_value = mock_result

        complex_query = """
            SELECT
                u.id,
                u.name,
                COUNT(o.id) as total,
                AVG(o.value) as avg_value
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            WHERE u.status = 'active'
            GROUP BY u.id, u.name
            HAVING COUNT(o.id) > 0
            ORDER BY total DESC
        """
        start_time = time.time()
        results, _ = await self.query_executor.execute_query(complex_query)
        execution_time = time.time() - start_time
        assert len(results) == 1000
        assert execution_time < 2.0
