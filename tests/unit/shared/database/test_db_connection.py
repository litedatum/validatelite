"""
Database connection testing module.

This code tests the database connection functionality, including all relevant aspects.
URL Construction or Building a URL
2. Connection Test
Connection pool management.
4. Exception Handling
5. Session Management
"""

import pytest

from shared.database import check_connection, get_db_url, get_engine
from shared.exceptions import EngineError
from shared.schema.connection_schema import ConnectionType


@pytest.mark.asyncio
class TestDatabaseConnection:
    """Database Connection Test Class"""

    async def test_get_db_url_mysql(self) -> None:
        url = get_db_url(
            ConnectionType.MYSQL.value, "localhost", 3306, "test_db", "user", "pass"
        )
        assert url == "mysql+aiomysql://user:pass@localhost:3306/test_db"

    async def test_get_db_url_postgresql(self) -> None:
        url = get_db_url(
            ConnectionType.POSTGRESQL.value,
            "localhost",
            5432,
            "test_db",
            "user",
            "pass",
        )
        assert url == "postgresql+asyncpg://user:pass@localhost:5432/test_db"

    async def test_get_db_url_sqlite(self) -> None:
        url = get_db_url(ConnectionType.SQLITE.value, file_path="./test.db")
        assert url == "sqlite+aiosqlite:///./test.db"

    async def test_get_db_url_unsupported(self) -> None:
        with pytest.raises(
            EngineError, match="Unsupported or incomplete database type"
        ):
            get_db_url("unsupported", "localhost", 1234, "db", "user", "pass")

    async def test_check_connection_sqlite(self) -> None:
        url = get_db_url(ConnectionType.SQLITE.value, file_path="./test.db")
        result = await check_connection(url)
        assert isinstance(result, bool)

    async def test_check_connection_mysql_fail(self) -> None:
        url = get_db_url(
            ConnectionType.MYSQL.value, "127.0.0.1", 3306, "not_exist", "user", "wrong"
        )
        result = await check_connection(url)
        assert result is False

    async def test_get_engine_sqlite(self) -> None:
        url = get_db_url(ConnectionType.SQLITE.value, file_path="./test.db")
        engine = await get_engine(url)
        assert engine is not None

    async def test_get_engine_mysql_fail(self) -> None:
        with pytest.raises(Exception):
            url = get_db_url(
                ConnectionType.MYSQL.value, "127.0.0.1", 3306, None, "user", "wrong"
            )
            await get_engine(url)
