"""
Test configuration file.

Defines the fixtures and environment configuration required for testing.
"""

import logging as _logging
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, Generator, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import text

# Add the project root directory to the Python path.
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the test modules.
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

# Import the database connection management module.
from shared.database.connection import close_all_engines

# ---------------------------------------------------------------------------
# Hypothesis global configuration – suppress HealthCheck for function-scoped
# fixtures used in property-based tests (see OutputFormatter tests).
# ---------------------------------------------------------------------------

try:
    from hypothesis import HealthCheck, settings

    # Register and immediately activate a profile that suppresses the specific
    # health check causing failures when using pytest fixtures inside
    # @given-decorated tests.
    settings.register_profile(
        "ci", suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    settings.load_profile("ci")
except ImportError:  # Hypothesis not installed in minimal environments
    pass

# Modify the test database connection URL - use the real MySQL database.
TEST_DATABASE_URL = "mysql+pymysql://root:root123@localhost:3306/data_quality"
TEST_ASYNC_DATABASE_URL = "mysql+aiomysql://root:root123@localhost:3306/data_quality"

# Add PostgreSQL configuration - use the Neon database.
TEST_POSTGRES_DATABASE_URL = "postgresql://neondb_owner:npg_O2uKXTdwQE4i@ep-flat-moon-a8g567ek-pooler.eastus2.azure.neon.tech/neondb?sslmode=require&channel_binding=require"
TEST_POSTGRES_ASYNC_DATABASE_URL = "postgresql+asyncpg://neondb_owner:npg_O2uKXTdwQE4i@ep-flat-moon-a8g567ek-pooler.eastus2.azure.neon.tech/neondb?sslmode=require&channel_binding=require"

# ---------------------------------------------------------------------------
# Global logging configuration – hide DEBUG messages in test output to keep
# reports readable.  Individual tests can override via the standard `caplog`
# fixture when they *need* to assert on DEBUG output.
# ---------------------------------------------------------------------------

_logging.getLogger().setLevel(_logging.INFO)


class AsyncMockEngine:
    """A mock engine that supports asynchronous context managers."""

    def __init__(self) -> None:
        self.begin_mock = AsyncMock()
        self.connect_mock = AsyncMock()
        # Add a URL attribute.
        self.url = "mysql+aiomysql://test:test@localhost:3306/test_db"

    def begin(self) -> "AsyncMockConnection":
        """Simulates an asynchronous `begin` method, returning an instance of `AsyncMockConnection`."""
        return AsyncMockConnection()

    async def connect(self) -> "AsyncMockConnection":
        """Simulates an asynchronous connect method."""
        return AsyncMockConnection()

    async def dispose(self) -> None:
        """Simulates an asynchronous dispose method."""
        pass


class AsyncMockConnection:
    """A mock connection that supports asynchronous context managers."""

    def __init__(self) -> None:
        self.execute_mock = AsyncMock()
        self.fetchone_mock = AsyncMock()
        self.scalar_mock = AsyncMock()

    async def __aenter__(self) -> "AsyncMockConnection":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> None:
        pass

    async def execute(self, query: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        """Simulates an asynchronous execution method."""
        result = MagicMock()
        result.scalar.return_value = 1
        result.fetchone.return_value = (100,)  # Simulates the total number of records.
        result.fetchall.return_value = (
            []
        )  # Returns an empty list instead of a coroutine.
        result.keys.return_value = []
        return result

    async def run_sync(self, func: Callable[..., Any]) -> Any:
        """Simulates the `run_sync` method."""
        return func(self)


@pytest.fixture
def mock_async_engine() -> AsyncMockEngine:
    """Creates a mock engine that supports asynchronous context management."""
    return AsyncMockEngine()


@pytest.fixture(scope="session")
def db_engine() -> Generator[Engine, Any, None]:
    """Create a test database engine."""
    engine = create_engine(TEST_DATABASE_URL, echo=False)
    yield engine
    engine.dispose()


# @pytest_asyncio.fixture(scope="function")
# async def async_db_engine() -> AsyncGenerator[AsyncEngine, None]:
#     engine = create_async_engine(TEST_ASYNC_DATABASE_URL, echo=False)
#     yield engine
#     await engine.dispose()

# @pytest.fixture
# def db_session(db_engine: Engine) -> Generator[Session, Any, None]:
#     """Create a database session."""
#     connection = db_engine.connect()
#     transaction = connection.begin()
#     Session = sessionmaker(bind=connection)
#     session = Session()

#     # Create all test tables here.
#     from sqlalchemy import inspect
#     from shared.schema.base import Base

#     # Create all tables.
#     inspector = inspect(db_engine)

#     # Create all tables if they do not already exist.
#     if not inspector.has_table('connections'):
#         Base.metadata.create_all(db_engine)

#     yield session

#     # Cleanup: The transaction is rolled back instead of committed to prevent test data from polluting the production database.
#     session.close()
#     transaction.rollback()


@pytest_asyncio.fixture(scope="function")
async def async_db_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create an asynchronous test database engine."""
    engine = create_async_engine(
        TEST_ASYNC_DATABASE_URL,
        echo=False,
        poolclass=NullPool,  # Use a NullPool to prevent cleanup issues at the end of a session.
        pool_pre_ping=True,  # Enable connection pre-checks.
        pool_recycle=3600,  # The connection will be recycled after one hour.
    )
    # Perform table creation here (one-time setup).
    # async with engine.connect() as conn:
    #     await conn.run_sync(Base.metadata.create_all)
    #     await conn.commit()
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def test_async_engine() -> AsyncGenerator[AsyncEngine, None]:
    """
    Create a dedicated asynchronous engine for testing purposes,
    utilizing a connection pool configured for enhanced security.
    Prevent connection pool issues during multiple test runs.
    """
    engine = create_async_engine(
        TEST_ASYNC_DATABASE_URL,
        echo=False,
        poolclass=NullPool,  # Use a NullPool to prevent connection pooling issues.
        pool_pre_ping=True,  # Enable connection health checks.
        pool_recycle=300,  # Recycle the connection after 5 minutes (using a shorter recycle interval).
        pool_timeout=20,  # Connection timeout after 20 seconds.
        # Add MySQL-specific connection parameters.
        connect_args={
            "connect_timeout": 10,  # Connection timeout period.
            "autocommit": False,  # Automatic commit is disabled.
            "charset": "utf8mb4",  # Specifies the character encoding.
        },
    )
    try:
        # Test if the connection is working correctly.
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        yield engine
    finally:
        # Ensure the engine is shut down correctly.
        await engine.dispose()


@pytest_asyncio.fixture
async def async_db_session(
    async_db_engine: AsyncEngine,
) -> AsyncGenerator[AsyncSession, None]:
    TestingSessionLocal = async_sessionmaker(
        bind=async_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with TestingSessionLocal() as session:
        try:
            yield session
        finally:
            await session.rollback()


from tests.shared.utils.database_utils import (
    get_mysql_connection_params,
    get_postgresql_connection_params,
)


@pytest.fixture
def mysql_connection_params() -> Dict[str, object]:
    """MySQL connection parameters from environment variables."""
    return get_mysql_connection_params()


@pytest.fixture
def postgres_connection_params() -> Dict[str, object]:
    """PostgreSQL connection parameters from environment variables."""
    return get_postgresql_connection_params()


# Add a fixture for cleaning up the connection pool between tests.
@pytest_asyncio.fixture(scope="function", autouse=True)
async def cleanup_connection_pool() -> AsyncGenerator[None, None]:
    """
    Automatically clears the connection pool cache to ensure isolation between tests.
    Prevent a "'NoneType' object has no attribute 'send'" error.
    """
    # Clear the connection pool before and after each test.
    yield
    # Clean up after testing.
    try:
        await close_all_engines()
    except Exception as e:
        # Log any data cleaning errors encountered, but do not allow them to affect the test results.
        print(f"Warning: Error during connection pool cleanup: {e}")


# Add pytest configuration options.
def pytest_configure(config: Any) -> None:
    """Configure the pytest testing environment."""
    # Added tags for integration testing.
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "database: mark test as database test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "serial: mark test to run serially")


# Add pytest collection and execution hooks.
def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    """Modified the test collection logic to include special handling for database tests."""
    for item in items:
        # Add a marker to enforce serial execution for all database integration tests.
        if "integration" in item.keywords and "database" in str(item.fspath):
            item.add_marker(pytest.mark.serial)
        # Add a marker to serialize tests involving "real_database".
        if "real_database" in item.name or "RealDatabase" in str(item.cls):
            item.add_marker(pytest.mark.serial)


# @pytest_asyncio.fixture(scope="function")
# async def real_database_engine() -> AsyncGenerator[AsyncEngine, None]:
#     """
#     Create an asynchronous engine for testing against a live database.
#     Verify that the database connection is established and the table schema is correctly configured.
#     """
#     engine = create_async_engine(
#         TEST_ASYNC_DATABASE_URL,
#         echo=False,
#         poolclass=NullPool,
#         pool_pre_ping=True,
#         pool_recycle=3600,
#     )

#     # Create the necessary test table structure.
#     try:
#         async with engine.connect() as conn:
#             # Verify the existence of, and create if necessary, the `customers` table.
#             result = await conn.execute(
#                 text(
#                     """
#                 SELECT COUNT(*) as table_count
#                 FROM information_schema.tables
#                 WHERE table_schema = 'data_quality'
#                 AND table_name = 'customers'
#             """
#                 )
#             )
#             row = result.fetchone()
#             table_exists = row[0] > 0 if row else False

#             if not table_exists:
#                 # Create the `customers` table.
#                 conn.execute(
#                     text(
#                         """
#                     CREATE TABLE customers (
#                         id INT PRIMARY KEY AUTO_INCREMENT,
#                         name VARCHAR(255),
#                         email VARCHAR(255),
#                         age INT,
#                         gender INT COMMENT '0=female, 1=male, 3=invalid',
#                         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#                     ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
#                 """
#                     )
#                 )

#                 # Add index.
#                 await conn.execute(
#                     text("CREATE INDEX idx_customers_email ON customers(email)")
#                 )
#                 await conn.execute(
#                     text("CREATE INDEX idx_customers_age ON customers(age)")
#                 )
#                 await conn.execute(
#                     text("CREATE INDEX idx_customers_gender ON customers(gender)")
#                 )

#                 # Submit the create operation.
#                 await conn.commit()

#                 # Load testing data (a subset of the full dataset is used for testing purposes).
#                 sample_data = [
#                     (7, "Ivy6997", "bob303@example.com", 90, 1, "2025-07-02 01:12:28"),
#                     (18, "Alice3822", "eve277@test.org", -34, 0, "2025-07-02 01:12:28"),
#                     (66, "Grace6425", None, 74, 0, "2025-07-02 01:12:28"),
#                     (
#                         71,
#                         "Frank5182",
#                         "bob188@example.com",
#                         None,
#                         0,
#                         "2025-07-02 01:12:28",
#                     ),
#                     (205, "David4580", "eve268@mail.com", 39, 3, "2025-07-02 01:15:54"),
#                     (
#                         466,
#                         "Alice6855",
#                         "david209@example.com",
#                         81,
#                         None,
#                         "2025-07-02 01:15:09",
#                     ),
#                     (525, "Helen2145", "Helen#invalid", 70, 1, "2025-07-02 01:12:28"),
#                     (610, "Bob5392", "Bob#invalid", 77, 0, "2025-07-02 01:12:28"),
#                 ]

#                 for data in sample_data:
#                     await conn.execute(
#                         text(
#                             """
#                         INSERT INTO customers (id, name, email, age, gender, created_at)
#                         VALUES (%s, %s, %s, %s, %s, %s)
#                     """
#                         ),
#                         {"id": data[0], "name": data[1], "email": data[2], "age": data[3], "gender": data[4], "created_at": data[5]},
#                     )

#                 await conn.commit()

#     except Exception as e:
#         print(f"Warning: Failed to set up test database: {e}")

#     yield engine
#     await engine.dispose()
