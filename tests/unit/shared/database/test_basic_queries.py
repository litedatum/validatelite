"""
Basic query execution test.

This tests the basic query execution functionality of the QueryExecutor class, including:
Basic SELECT query.
2. Parameterized Queries
3. Queries with a LIMIT clause.
Queries that do not return results.
5. Error Handling
"""

from typing import Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from shared.database.query_executor import QueryExecutor
from shared.exceptions.exception_system import OperationError

pytestmark = pytest.mark.asyncio


# A common mock fixture for all tests.
@pytest.fixture
def mock_async_engine() -> Tuple[MagicMock, MagicMock]:
    """Create asynchronous mock database engine and connection objects."""
    mock_engine: MagicMock = MagicMock(spec=AsyncEngine)
    mock_conn: MagicMock = MagicMock(spec=AsyncConnection)
    mock_engine.connect.return_value.__aenter__.return_value = mock_conn
    mock_conn.execute = AsyncMock()
    return mock_engine, mock_conn


class TestBasicQueries:
    """Test basic query execution functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_async_engine: Tuple[MagicMock, MagicMock]) -> None:
        self.mock_engine, self.mock_conn = mock_async_engine
        self.query_executor = QueryExecutor(self.mock_engine)

    async def test_basic_select_query(self) -> None:
        """Test basic SELECT queries."""
        # Simulates query results.
        mock_result = MagicMock()
        mock_result.keys.return_value = ["id", "name", "age"]
        mock_result.fetchall = MagicMock(
            return_value=[(1, "Alice", 25), (2, "Bob", 30), (3, "Charlie", 35)]
        )
        mock_result.rowcount = 3
        mock_result.returns_rows = True
        self.mock_conn.execute.return_value = mock_result

        # mock_engine.connect.return_value.__enter__.return_value.execute.return_value = mock_result

        # Execute the query.
        query = "SELECT id, name, age FROM users"
        results, affected_rows = await self.query_executor.execute_query(query)

        # Validate the results.
        assert len(results) == 3
        assert results[0] == {"id": 1, "name": "Alice", "age": 25}
        assert results[1] == {"id": 2, "name": "Bob", "age": 30}
        assert results[2] == {"id": 3, "name": "Charlie", "age": 35}
        assert affected_rows is None

    async def test_query_with_params(self) -> None:
        """Test parameterized queries."""
        # Simulates query results.
        mock_result = MagicMock()
        mock_result.keys.return_value = ["id", "name"]
        mock_result.fetchall = MagicMock(return_value=[(1, "Alice")])
        mock_result.rowcount = 1
        mock_result.returns_rows = True
        self.mock_conn.execute.return_value = mock_result

        # mock_engine.connect.return_value.__enter__.return_value.execute.return_value = mock_result

        # Execute the query.
        query = "SELECT id, name FROM users WHERE age > :min_age"
        params = {"min_age": 20}
        results, affected_rows = await self.query_executor.execute_query(query, params)

        # Validate the results.
        assert len(results) == 1
        assert results[0] == {"id": 1, "name": "Alice"}
        assert affected_rows is None

    async def test_query_with_limit(self) -> None:
        """Test queries that utilize a LIMIT clause."""
        # Simulates query results.
        mock_result = MagicMock()
        mock_result.keys.return_value = ["id", "name"]
        mock_result.fetchall = MagicMock(return_value=[(1, "Alice"), (2, "Bob")])
        mock_result.rowcount = 2
        mock_result.returns_rows = True
        self.mock_conn.execute.return_value = mock_result

        # mock_engine.connect.return_value.__enter__.return_value.execute.return_value = mock_result

        # Execute the query.
        query = "SELECT id, name FROM users"
        results, affected_rows = await self.query_executor.execute_query(
            query, sample_limit=2
        )

        # Verify the results.
        assert len(results) == 2
        assert results[0] == {"id": 1, "name": "Alice"}
        assert results[1] == {"id": 2, "name": "Bob"}
        assert affected_rows is None

    async def test_query_without_results(self) -> None:
        """Test queries that do not return results."""
        # Simulates query results.
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_result.returns_rows = False
        self.mock_conn.execute.return_value = mock_result

        # mock_engine.connect.return_value.__enter__.return_value.execute.return_value = mock_result

        # Executes the query.
        query = "UPDATE users SET status = 'inactive' WHERE last_login < :date"
        params = {"date": "2024-01-01"}
        results, affected_rows = await self.query_executor.execute_query(
            query, params, fetch=False
        )

        # Verify the results.
        assert len(results) == 0
        assert affected_rows == 0

    async def test_query_error_handling(self) -> None:
        """Test query error handling."""
        # Simulates a query error.
        self.mock_conn.execute.side_effect = OperationError("Query error")

        # Execute the query.
        query = "SELECT * FROM non_existent_table"
        with pytest.raises(OperationError) as exc_info:
            await self.query_executor.execute_query(query)

        # Validate error messages.
        assert "Query error" in str(exc_info.value)

    async def test_query_with_empty_result(self) -> None:
        """Test query for empty results."""
        # Simulates an empty query result.
        mock_result = MagicMock()
        mock_result.keys.return_value = ["id", "name"]
        mock_result.fetchall = MagicMock(return_value=[])
        mock_result.rowcount = 0
        mock_result.returns_rows = True
        self.mock_conn.execute.return_value = mock_result

        # Execute the query.
        query = "SELECT id, name FROM users WHERE age > 100"
        results, affected_rows = await self.query_executor.execute_query(query)

        # Verify the results.
        assert len(results) == 0
        assert affected_rows is None

    async def test_query_with_special_characters(self) -> None:
        """Test queries containing special characters."""
        # Simulates query results.
        mock_result = MagicMock()
        mock_result.keys.return_value = ["id", "name"]
        mock_result.fetchall = MagicMock(return_value=[(1, "O'Connor")])
        mock_result.rowcount = 1
        mock_result.returns_rows = True
        self.mock_conn.execute.return_value = mock_result

        # mock_engine.connect.return_value.__enter__.return_value.execute.return_value = mock_result

        # Execute the query.
        query = "SELECT id, name FROM users WHERE name LIKE :pattern"
        params = {"pattern": "O'Connor%"}
        results, affected_rows = await self.query_executor.execute_query(query, params)

        # Validate the results.
        assert len(results) == 1
        assert results[0] == {"id": 1, "name": "O'Connor"}
        assert affected_rows is None
