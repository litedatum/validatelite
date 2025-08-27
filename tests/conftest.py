"""
Test configuration file.

Defines the fixtures and environment configuration required for testing.
"""

import logging as _logging
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

import pytest
import pytest_asyncio

# Add the project root directory to the Python path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config.loader import load_config

# Import the database connection management module.
from shared.database.connection import close_all_engines

# Load test-specific logging configuration
try:
    test_logging_config = load_config("logging.test.toml")
    if test_logging_config:
        # Apply test logging configuration
        for module, level in test_logging_config.get("module_levels", {}).items():
            _logging.getLogger(module).setLevel(getattr(_logging, level.upper()))
except Exception:
    # Fallback to default configuration if test config not found
    pass

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

# ---------------------------------------------------------------------------
# Global logging configuration – hide DEBUG messages in test output to keep
# reports readable.  Individual tests can override via the standard `caplog`
# fixture when they *need* to assert on DEBUG output.
# ---------------------------------------------------------------------------

_logging.getLogger().setLevel(_logging.INFO)


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
