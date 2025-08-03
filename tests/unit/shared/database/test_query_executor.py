"""
Tests for QueryExecutor

Tests the functionality of the QueryExecutor class
"""

from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.exc import OperationalError, ProgrammingError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from shared.database.database_dialect import DatabaseType, get_dialect
from shared.database.query_executor import QueryExecutor
from shared.exceptions import EngineError, OperationError, RuleExecutionError

pytestmark = pytest.mark.asyncio


# Common mock fixture, used by all tests.
@pytest.fixture
def mock_async_engine() -> Tuple[MagicMock, MagicMock]:
    """Create asynchronous mock database engine and connection objects."""
    mock_engine: MagicMock = MagicMock(spec=AsyncEngine)
    mock_conn: MagicMock = MagicMock(spec=AsyncConnection)
    mock_engine.connect.return_value.__aenter__.return_value = mock_conn
    mock_conn.execute = AsyncMock()
    return mock_engine, mock_conn


class TestQueryExecutor:
    """Testing the main workflow of the QueryExecutor."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_async_engine: Tuple[MagicMock, MagicMock]) -> None:
        self.mock_engine, self.mock_conn = mock_async_engine
        self.query_executor = QueryExecutor(self.mock_engine)

    async def test_execute_query_success(self) -> None:
        """Test successful execution of a standard query."""
        mock_result = MagicMock()
        mock_result.keys.return_value = ["id", "name"]
        mock_result.fetchall = MagicMock(return_value=[(1, "Test"), (2, "Test2")])
        mock_result.rowcount = 2
        mock_result.returns_rows = True
        self.mock_conn.execute.return_value = mock_result

        results, affected_rows = await self.query_executor.execute_query(
            "SELECT * FROM test"
        )
        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[0]["name"] == "Test"
        assert results[1]["id"] == 2
        assert results[1]["name"] == "Test2"
        assert affected_rows is None
        self.mock_conn.execute.assert_awaited_once()

    async def test_execute_query_with_params(self) -> None:
        """Test parameterized queries."""
        mock_result = MagicMock()
        mock_result.keys.return_value = ["id", "name"]
        mock_result.fetchall = MagicMock(return_value=[(1, "Test")])
        mock_result.rowcount = 1
        mock_result.returns_rows = True
        self.mock_conn.execute.return_value = mock_result

        params = {"id": 1}
        results, affected_rows = await self.query_executor.execute_query(
            "SELECT * FROM test WHERE id = :id", params=params
        )
        assert len(results) == 1
        assert results[0]["id"] == 1
        assert results[0]["name"] == "Test"
        assert affected_rows is None
        self.mock_conn.execute.assert_awaited_once()

    async def test_execute_query_no_fetch(self) -> None:
        """Test updating without fetching results."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_result.returns_rows = False
        self.mock_conn.execute.return_value = mock_result

        results, affected_rows = await self.query_executor.execute_query(
            "UPDATE test SET name = 'Updated' WHERE id = 1", fetch=False
        )
        assert results == []
        assert affected_rows == 1
        self.mock_conn.execute.assert_awaited_once()

    async def test_execute_query_with_sample_limit(self) -> None:
        """Testing queries with a sample limit."""
        mock_result = MagicMock()
        mock_result.keys.return_value = ["id", "name"]
        mock_result.fetchall = MagicMock(return_value=[(1, "Test")])
        mock_result.rowcount = 1
        mock_result.returns_rows = True
        self.mock_conn.execute.return_value = mock_result

        results, _ = await self.query_executor.execute_query(
            "SELECT * FROM test", sample_limit=1
        )
        assert len(results) == 1
        self.mock_conn.execute.assert_awaited_once()
        args, _ = self.mock_conn.execute.call_args
        assert "LIMIT 1" in str(args[0])

    async def test_execute_query_error(self) -> None:
        """Test query exceptions."""
        self.mock_conn.execute.side_effect = SQLAlchemyError("Test error")
        with pytest.raises(OperationError):
            await self.query_executor.execute_query("SELECT * FROM test")

    async def test_get_database_list(self) -> None:
        """Test retrieving the database list."""
        mock_result = MagicMock()
        mock_result.keys.return_value = ["Database"]
        mock_result.fetchall = MagicMock(return_value=[("db1",), ("db2",)])
        mock_result.rowcount = 2
        mock_result.returns_rows = True
        self.mock_conn.execute.return_value = mock_result

        databases = await self.query_executor.get_database_list()
        assert len(databases) == 2
        assert "db1" in databases
        assert "db2" in databases
        self.mock_conn.execute.assert_awaited()

    async def test_get_table_list(self) -> None:
        """Test retrieving the list of tables."""
        mock_result = MagicMock()
        mock_result.keys.return_value = ["table_name", "table_type"]
        mock_result.fetchall = MagicMock(
            return_value=[("table1", "BASE TABLE"), ("table2", "BASE TABLE")]
        )
        mock_result.rowcount = 2
        mock_result.returns_rows = True
        self.mock_conn.execute.return_value = mock_result

        tables = await self.query_executor.get_table_list("test_db")
        assert len(tables) == 2
        assert tables[0] == {
            "name": "table1",
            "type": "table",
            "schema": None,
            "database": "test_db",
        }
        assert tables[1] == {
            "name": "table2",
            "type": "table",
            "schema": None,
            "database": "test_db",
        }
        self.mock_conn.execute.assert_awaited()

    async def test_get_column_list(self) -> None:
        """Test retrieval of the field list."""
        mock_result = MagicMock()
        mock_result.keys.return_value = [
            "column_name",
            "data_type",
            "is_nullable",
            "column_default",
            "key",
            "extra",
            "column_comment",
        ]
        mock_result.fetchall = MagicMock(
            return_value=[
                ("id", "INTEGER", "NO", None, "PRI", "auto_increment", ""),
                ("name", "VARCHAR", "YES", None, "", "", ""),
            ]
        )
        mock_result.rowcount = 2
        mock_result.returns_rows = True
        self.mock_conn.execute.return_value = mock_result

        columns = await self.query_executor.get_column_list("test_table", "test_db")
        assert len(columns) == 2
        assert columns[0] == {
            "name": "id",
            "type": "INTEGER",
            "key": "PRI",
            "extra": "auto_increment",
            "nullable": False,
            "default": None,
            "original": {
                "column_name": "id",
                "data_type": "INTEGER",
                "key": "PRI",
                "extra": "auto_increment",
                "is_nullable": "NO",
                "column_default": None,
                "column_comment": "",
            },
        }
        assert columns[1] == {
            "name": "name",
            "type": "VARCHAR",
            "key": "",
            "extra": "",
            "nullable": True,
            "default": None,
            "original": {
                "column_name": "name",
                "data_type": "VARCHAR",
                "key": "",
                "extra": "",
                "is_nullable": "YES",
                "column_default": None,
                "column_comment": "",
            },
        }
        self.mock_conn.execute.assert_awaited()

    async def test_get_dialect_info(self) -> None:
        """Test getting dialect information for different database types"""
        # Test MySQL dialect info
        mysql_dialect = get_dialect("mysql")
        mysql_executor = QueryExecutor(self.mock_engine, dialect=mysql_dialect)
        mysql_info = mysql_executor.get_dialect_info()
        assert mysql_info["database_type"] == DatabaseType.MYSQL
        assert "dialect_name" in mysql_info
        assert "supports_schemas" in mysql_info
        assert "quote_character" in mysql_info
        assert "regex_operator" in mysql_info
        assert "length_function" in mysql_info
        assert mysql_info["regex_operator"] == "REGEXP"
        assert mysql_info["quote_character"] == "`"
        assert mysql_info["length_function"] == "CHAR_LENGTH"

        # Test PostgreSQL dialect info
        postgres_dialect = get_dialect("postgresql")
        postgres_executor = QueryExecutor(self.mock_engine, dialect=postgres_dialect)
        postgres_info = postgres_executor.get_dialect_info()
        assert postgres_info["database_type"] == DatabaseType.POSTGRESQL
        assert postgres_info["regex_operator"] == "~"
        assert postgres_info["quote_character"] == '"'
        assert postgres_info["length_function"] == "LENGTH"

        # Test SQLite dialect info
        sqlite_dialect = get_dialect("sqlite")
        sqlite_executor = QueryExecutor(self.mock_engine, dialect=sqlite_dialect)
        sqlite_info = sqlite_executor.get_dialect_info()
        assert sqlite_info["database_type"] == DatabaseType.SQLITE
        assert sqlite_info["regex_operator"] == "REGEXP"
        assert sqlite_info["quote_character"] == '"'
        assert sqlite_info["length_function"] == "LENGTH"

        # Verify all dialect info contains required fields
        for dialect_info in [mysql_info, postgres_info, sqlite_info]:
            assert isinstance(dialect_info["dialect_name"], str)
            assert isinstance(dialect_info["database_type"], DatabaseType)
            assert isinstance(dialect_info["supports_schemas"], bool)
            assert isinstance(dialect_info["quote_character"], str)
            assert isinstance(dialect_info["regex_operator"], str)
            assert isinstance(dialect_info["length_function"], str)

    # Test batch insert methods
    async def test_execute_batch_insert_basic(self) -> None:
        """Test basic batch insert functionality"""
        # Mock the connection behavior for batch insert
        # With batch_size=2 and 3 records, we'll have 2 batches: [2 records] + [1 record]
        mock_results = [MagicMock(), MagicMock()]
        mock_results[0].rowcount = 2  # First batch: 2 records
        mock_results[1].rowcount = 1  # Second batch: 1 record
        self.mock_conn.execute.side_effect = mock_results
        self.mock_conn.begin.return_value.__aenter__.return_value = MagicMock()

        # Prepare test data
        test_data: List[Dict[str, Any]] = [
            {"id": 1, "name": "test1", "value": 100},
            {"id": 2, "name": "test2", "value": 200},
            {"id": 3, "name": "test3", "value": 300},
        ]

        # Execute batch insert
        inserted_count = await self.query_executor.execute_batch_insert(
            table_name="batch_test",
            data_list=test_data,
            batch_size=2,
            use_transaction=True,
        )

        # Verify results: 2 + 1 = 3 total records
        assert inserted_count == 3

        # Verify that execute was called twice (for 2 batches)
        assert self.mock_conn.execute.call_count == 2

    async def test_execute_batch_insert_empty_data(self) -> None:
        """Test batch insert with empty data"""
        # Test with empty list
        result = await self.query_executor.execute_batch_insert(
            table_name="test_table", data_list=[], batch_size=1000
        )
        assert result == 0

    async def test_execute_batch_insert_inconsistent_keys(self) -> None:
        """Test batch insert with inconsistent keys"""
        # Test data with inconsistent keys
        test_data: List[Dict[str, Any]] = [
            {"id": 1, "name": "test1"},
            {"id": 2, "value": 200},  # Missing 'name', has 'value'
        ]

        # Should raise ValueError
        with pytest.raises(ValueError, match="different keys"):
            await self.query_executor.execute_batch_insert(
                table_name="test_table", data_list=test_data
            )

    async def test_batch_insert_performance_comparison(self) -> None:
        """Compare performance of batch vs single insert (mocked)"""
        # Mock the connection behavior
        # With batch_size=5 and 10 records, we'll have 2 batches of 5 records each
        mock_results = [MagicMock(), MagicMock()]
        mock_results[0].rowcount = 5  # First batch: 5 records
        mock_results[1].rowcount = 5  # Second batch: 5 records
        self.mock_conn.execute.side_effect = mock_results
        self.mock_conn.begin.return_value.__aenter__ = AsyncMock()

        # Prepare test data (smaller size for unit test)
        test_data: List[Dict[str, Any]] = [
            {"id": i + 1, "name": f"name_{i + 1}", "value": i * 10}
            for i in range(10)  # Smaller dataset for mocked test
        ]

        # Test batch insert
        batch_count = await self.query_executor.execute_batch_insert(
            table_name="perf_test",
            data_list=test_data,
            batch_size=5,
            use_transaction=True,
        )

        # Verify batch insert worked: 5 + 5 = 10 total records
        assert batch_count == 10
        assert self.mock_conn.execute.call_count == 2

    async def test_batch_insert_transaction_handling(self) -> None:
        """Test transaction handling in batch insert"""
        # Test data
        test_data: List[Dict[str, Any]] = [
            {"id": 1, "name": "test1"},
            {"id": 2, "name": "test2"},
        ]

        # Test with transaction enabled (default)
        # Reset mock for first test
        mock_result1 = MagicMock()
        mock_result1.rowcount = 2
        self.mock_conn.execute.return_value = mock_result1
        self.mock_conn.begin.return_value.__aenter__.return_value = MagicMock()

        inserted_count = await self.query_executor.execute_batch_insert(
            table_name="transaction_test", data_list=test_data, use_transaction=True
        )
        assert inserted_count == 2

        # Reset mock for second test
        self.mock_conn.reset_mock()
        mock_result2 = MagicMock()
        mock_result2.rowcount = 2
        self.mock_conn.execute.return_value = mock_result2

        # Test without transaction
        inserted_count = await self.query_executor.execute_batch_insert(
            table_name="transaction_test", data_list=test_data, use_transaction=False
        )
        assert inserted_count == 2

    async def test_execute_bulk_insert_values_basic(self) -> None:
        """Test basic bulk VALUES insert functionality"""
        # Mock the connection behavior for bulk VALUES insert
        # With batch_size=2 and 4 records, we'll have 2 batches: [2 records] + [2 records]
        mock_result1 = MagicMock()
        mock_result1.rowcount = 2  # First batch: 2 records
        mock_result2 = MagicMock()
        mock_result2.rowcount = 2  # Second batch: 2 records
        self.mock_conn.execute.side_effect = [mock_result1, mock_result2]
        self.mock_conn.begin.return_value.__aenter__.return_value = MagicMock()

        # Prepare test data
        test_data: List[Dict[str, Any]] = [
            {"id": 1, "name": "item1", "category": "A", "score": 95.5},
            {"id": 2, "name": "item2", "category": "B", "score": 87.2},
            {"id": 3, "name": "item3", "category": "A", "score": 92.8},
            {"id": 4, "name": "item4", "category": "C", "score": 78.9},
        ]

        # Execute bulk VALUES insert
        inserted_count = await self.query_executor.execute_bulk_insert_values(
            table_name="bulk_test",
            data_list=test_data,
            batch_size=2,  # Small batch size to test batching
        )

        # Verify results: 2 + 2 = 4 total records
        assert inserted_count == 4
        # Verify that execute was called twice (for 2 batches)
        assert self.mock_conn.execute.call_count == 2

    async def test_execute_bulk_insert_values_large_batch(self) -> None:
        """Test bulk VALUES insert with larger dataset"""
        # Mock the connection behavior
        mock_result = MagicMock()
        mock_result.rowcount = 25  # Per batch
        self.mock_conn.execute.return_value = mock_result
        self.mock_conn.begin.return_value.__aenter__.return_value = MagicMock()

        # Prepare larger test data
        test_data: List[Dict[str, Any]] = [
            {"id": i + 1, "data": f"data_{i + 1}"} for i in range(100)
        ]

        # Execute bulk VALUES insert
        inserted_count = await self.query_executor.execute_bulk_insert_values(
            table_name="large_bulk_test", data_list=test_data, batch_size=25
        )

        # Verify results (4 batches * 25 records each = 100)
        assert inserted_count == 100
        assert self.mock_conn.execute.called


class TestQueryExecutorErrorHandling:
    """Testing exception handling in the QueryExecutor."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_async_engine: Tuple[MagicMock, MagicMock]) -> None:
        self.mock_engine, self.mock_conn = mock_async_engine
        self.query_executor = QueryExecutor(self.mock_engine)

    async def test_sql_syntax_error(self) -> None:
        self.mock_conn.execute.side_effect = ProgrammingError(
            "syntax error", None, Exception()
        )
        with pytest.raises(OperationError):
            await self.query_executor.execute_query("SELECT * FROM invalid_syntax")

    async def test_connection_error(self) -> None:
        self.mock_engine.connect.side_effect = OperationalError(
            "connection error", None, Exception()
        )
        with pytest.raises(EngineError):
            await self.query_executor.execute_query("SELECT * FROM test_table")

    async def test_invalid_parameters(self) -> None:
        self.mock_conn.execute.side_effect = ProgrammingError(
            "parameter error", None, Exception()
        )
        with pytest.raises(OperationError):
            await self.query_executor.execute_query(
                "SELECT * FROM test WHERE id = :id", params={"invalid_param": 1}
            )

    async def test_table_not_exists(self) -> None:
        self.mock_conn.execute.side_effect = ProgrammingError(
            "table does not exist", None, Exception()
        )
        with pytest.raises(RuleExecutionError):
            await self.query_executor.execute_query("SELECT * FROM non_existent_table")

    async def test_column_not_exists(self) -> None:
        self.mock_conn.execute.side_effect = ProgrammingError(
            "column does not exist", None, Exception()
        )
        with pytest.raises(RuleExecutionError):
            await self.query_executor.execute_query(
                "SELECT non_existent_column FROM test_table"
            )

    async def test_permission_error(self) -> None:
        self.mock_conn.execute.side_effect = ProgrammingError(
            "permission denied", None, Exception()
        )
        with pytest.raises(OperationError):
            await self.query_executor.execute_query("SELECT * FROM restricted_table")

    async def test_timeout_error(self) -> None:
        self.mock_conn.execute.side_effect = OperationalError(
            "timeout", None, Exception()
        )
        with pytest.raises(EngineError):
            await self.query_executor.execute_query("SELECT * FROM large_table")
