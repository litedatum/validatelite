"""
Integration tests for the QueryExecutor dialect system.

Test the integration of the QueryExecutor with the database dialect system.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from shared.database.database_dialect import get_dialect
from shared.database.query_executor import QueryExecutor


class TestQueryExecutorDialectIntegration:
    """Integration tests for the QueryExecutor dialect system."""

    @pytest.fixture
    def mock_engine(self) -> Any:
        """Create the simulation engine."""
        engine = AsyncMock()
        # Set the default URL to use MySQL.
        engine.url = Mock()
        engine.url.__str__ = Mock(return_value="mysql://user:pass@localhost/test")
        return engine

    @pytest.fixture
    def mock_connection(self) -> Any:
        """Creates a mock connection."""
        conn = AsyncMock()
        conn.engine = Mock()
        conn.engine.url = Mock()
        conn.engine.url.__str__ = Mock(return_value="mysql://user:pass@localhost/test")
        return conn

    def test_dialect_auto_detection_mysql(self, mock_engine: Any) -> None:
        """Test MySQL dialect auto-detection."""
        # Ensure the URL is set before creating the QueryExecutor instance.
        mock_engine.url.__str__.return_value = "mysql://user:pass@localhost/test"
        executor = QueryExecutor(mock_engine)
        assert executor.dialect.__class__.__name__ == "MySQLDialect"

    def test_dialect_auto_detection_postgresql(self, mock_engine: Any) -> None:
        """Test PostgreSQL dialect auto-detection."""
        # Reset the URL to PostgreSQL.
        mock_engine.engine.url.__str__.return_value = (
            "postgresql://user:pass@localhost/test"
        )
        executor = QueryExecutor(mock_engine)
        assert executor.dialect.__class__.__name__ == "PostgreSQLDialect"

    def test_dialect_auto_detection_sqlite(self, mock_engine: Any) -> None:
        """Testing automatic dialect detection for SQLite."""
        # Directly mock the `_detect_dialect` method.
        # with patch.object(QueryExecutor, '_detect_dialect', return_value=get_dialect('sqlite')):
        mock_engine.engine.url.__str__.return_value = "sqlite:///test.db"
        executor = QueryExecutor(mock_engine)
        assert executor.dialect.__class__.__name__ == "SQLiteDialect"

    def test_manual_dialect_setting(self, mock_engine: Any) -> None:
        """Test manual dialect setting."""
        dialect = get_dialect("postgresql")
        executor = QueryExecutor(mock_engine, dialect=dialect)
        assert executor.dialect.__class__.__name__ == "PostgreSQLDialect"

    @pytest.mark.asyncio
    async def test_get_database_list_with_dialect(self, mock_engine: Any) -> None:
        """Test retrieving the database list using a dialect."""
        executor = QueryExecutor(mock_engine)

        # Simulates query results.
        mock_results = [{"Database": "test_db1"}, {"Database": "test_db2"}]

        with patch.object(executor, "execute_query", return_value=(mock_results, None)):
            databases = await executor.get_database_list()
            assert databases == ["test_db1", "test_db2"]

    @pytest.mark.asyncio
    async def test_get_table_list_with_dialect(self, mock_engine: Any) -> None:
        """Test retrieving the list of tables using a dialect."""
        executor = QueryExecutor(mock_engine)

        # Simulates query results.
        mock_results = [
            {"table_name": "users", "table_type": "BASE TABLE"},
            {"table_name": "orders", "table_type": "BASE TABLE"},
            {"table_name": "user_view", "table_type": "VIEW"},
        ]

        with patch.object(executor, "execute_query", return_value=(mock_results, None)):
            tables = await executor.get_table_list("test_db")
            assert len(tables) == 3
            assert tables[0]["name"] == "users"
            assert tables[0]["type"] == "table"
            assert tables[2]["name"] == "user_view"
            assert tables[2]["type"] == "view"

    @pytest.mark.asyncio
    async def test_get_column_list_with_dialect(self, mock_engine: Any) -> None:
        """Test retrieving the list of columns using a dialect."""
        executor = QueryExecutor(mock_engine)

        # Simulates query results.
        mock_results = [
            {
                "column_name": "id",
                "data_type": "int",
                "is_nullable": "NO",
                "column_default": None,
                "column_key": "PRI",
            },
            {
                "column_name": "name",
                "data_type": "varchar",
                "is_nullable": "YES",
                "column_default": None,
                "column_key": "",
            },
        ]

        with patch.object(executor, "execute_query", return_value=(mock_results, None)):
            columns = await executor.get_column_list("users")
            assert len(columns) == 2
            assert columns[0]["name"] == "id"
            assert columns[0]["type"] == "int"
            assert columns[0]["nullable"] == False
            assert columns[1]["name"] == "name"
            assert columns[1]["nullable"] == True

    @pytest.mark.asyncio
    async def test_table_exists_with_dialect(self, mock_engine: Any) -> None:
        """Test for the existence of the dialect checklist."""
        executor = QueryExecutor(mock_engine)

        # Test case for table existence.
        with patch.object(
            executor, "execute_query", return_value=([{"exists": 1}], None)
        ):
            exists = await executor.table_exists("users")
            assert exists == True

        # Test the scenario where the table does not exist.
        with patch.object(executor, "execute_query", return_value=([], None)):
            exists = await executor.table_exists("nonexistent_table")
            assert exists == False

    def test_get_dialect_info(self, mock_engine: Any) -> None:
        """Test retrieval of dialect information."""
        executor = QueryExecutor(mock_engine)
        dialect_info = executor.get_dialect_info()

        assert "dialect_name" in dialect_info
        assert "database_type" in dialect_info
        assert "supports_schemas" in dialect_info
        assert "quote_character" in dialect_info
        assert "regex_operator" in dialect_info
        assert "length_function" in dialect_info

    @pytest.mark.asyncio
    async def test_error_handling_in_metadata_queries(self, mock_engine: Any) -> None:
        """Testing error handling for metadata queries."""
        executor = QueryExecutor(mock_engine)

        # Simulates a query exception.
        with patch.object(
            executor, "execute_query", side_effect=Exception("Database error")
        ):
            # Testing error handling for various metadata queries.
            with pytest.raises(Exception):
                databases = await executor.get_database_list()

            with pytest.raises(Exception):
                tables = await executor.get_table_list()

            with pytest.raises(Exception):
                columns = await executor.get_column_list("test_table")

            with pytest.raises(Exception):
                exists = await executor.table_exists("test_table")

    def test_dialect_detection_fallback(self, mock_engine: Any) -> None:
        """Testing the fallback mechanism when dialect detection fails."""
        # Simulates an unknown database type.
        mock_engine.url.__str__.return_value = "unknown://localhost/test"

        executor = QueryExecutor(mock_engine)
        # The system should revert to using the MySQL dialect.
        assert executor.dialect.__class__.__name__ == "MySQLDialect"

    def test_dialect_detection_exception_handling(self, mock_engine: Any) -> None:
        """Test exception handling for dialect detection."""
        # Simulates a URL retrieval exception.
        mock_engine.url.__str__.side_effect = Exception("URL error")

        executor = QueryExecutor(mock_engine)
        # The system should fall back to the MySQL dialect.
        assert executor.dialect.__class__.__name__ == "MySQLDialect"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
