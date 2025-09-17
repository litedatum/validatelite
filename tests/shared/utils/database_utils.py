"""
Database utilities for testing.

This module provides utility functions for generating database URLs
from environment variables for use in E2E and integration tests.
"""

import os
from typing import Dict, Optional
from urllib.parse import urlparse

from shared.database.connection import get_db_url
from shared.enums.connection_types import ConnectionType


def get_mysql_test_url() -> str:
    """Generate MySQL test database URL from environment variables."""
    # Try to get URL from environment variable first
    mysql_url = os.getenv("MYSQL_DB_URL")
    if mysql_url:
        return mysql_url

    # Fallback to individual environment variables
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    database = os.getenv("MYSQL_DATABASE", "test_db")
    username = os.getenv("MYSQL_USERNAME", "root")
    password = os.getenv("MYSQL_PASSWORD", "password")

    return get_db_url(ConnectionType.MYSQL, host, port, database, username, password)


def get_postgresql_test_url() -> str:
    """Generate PostgreSQL test database URL from environment variables."""
    # Try to get URL from environment variable first
    postgresql_url = os.getenv("POSTGRESQL_DB_URL")
    if postgresql_url:
        return postgresql_url

    # Fallback to individual environment variables
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    database = os.getenv("POSTGRES_DB", "test_db")
    username = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "password")

    return get_db_url(
        ConnectionType.POSTGRESQL, host, port, database, username, password
    )


def parse_database_url(url: str) -> Dict[str, str]:
    """Parse database URL into components."""
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "host": parsed.hostname or "localhost",
        "port": str(parsed.port or (3306 if parsed.scheme == "mysql" else 5432)),
        "database": parsed.path.lstrip("/"),
        "username": parsed.username or "",
        "password": parsed.password or "",
    }


def get_mysql_connection_params() -> Dict[str, object]:
    """Get MySQL connection parameters as a dictionary."""
    mysql_url = os.getenv("MYSQL_DB_URL")

    if mysql_url:
        params = parse_database_url(mysql_url)
        return {
            "db_type": ConnectionType.MYSQL.value,
            "host": params["host"],
            "port": int(params["port"]),
            "database": params["database"],
            "username": params["username"],
            "password": params["password"],
        }

    # Only return params if explicit environment variables are set
    # This ensures tests skip when database is not configured
    host = os.getenv("MYSQL_HOST")
    port = os.getenv("MYSQL_PORT")
    database = os.getenv("MYSQL_DATABASE")
    username = os.getenv("MYSQL_USERNAME")
    password = os.getenv("MYSQL_PASSWORD")

    if not all([host, database, username]):
        # Return dict with None values to trigger test skip
        return {
            "db_type": ConnectionType.MYSQL.value,
            "host": None,
            "port": None,
            "database": None,
            "username": None,
            "password": None,
        }

    return {
        "db_type": ConnectionType.MYSQL.value,
        "host": host,
        "port": int(port) if port else 3306,
        "database": database,
        "username": username,
        "password": password or "",
    }


def get_postgresql_connection_params() -> Dict[str, object]:
    """Get PostgreSQL connection parameters as a dictionary."""
    postgresql_url = os.getenv("POSTGRESQL_DB_URL")
    if postgresql_url:
        params = parse_database_url(postgresql_url)
        return {
            "db_type": ConnectionType.POSTGRESQL.value,
            "host": params["host"],
            "port": int(params["port"]),
            "database": params["database"],
            "username": params["username"],
            "password": params["password"],
        }

    # Only return params if explicit environment variables are set
    # This ensures tests skip when database is not configured
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    database = os.getenv("POSTGRES_DB")
    username = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    if not all([host, database, username]):
        # Return dict with None values to trigger test skip
        return {
            "db_type": ConnectionType.POSTGRESQL.value,
            "host": None,
            "port": None,
            "database": None,
            "username": None,
            "password": None,
        }

    return {
        "db_type": ConnectionType.POSTGRESQL.value,
        "host": host,
        "port": int(port) if port else 5432,
        "database": database,
        "username": username,
        "password": password or "",
    }


def get_test_database_urls() -> Dict[str, str]:
    """Get all test database URLs."""
    return {
        "mysql": get_mysql_test_url(),
        "postgresql": get_postgresql_test_url(),
    }


def get_test_connection_params() -> Dict[str, Dict[str, object]]:
    """Get all test connection parameters."""
    return {
        "mysql": get_mysql_connection_params(),
        "postgresql": get_postgresql_connection_params(),
    }


def is_ci_environment() -> bool:
    """Check if running in CI environment."""
    return os.getenv("CI", "false").lower() == "true"


def get_database_type_from_env() -> Optional[str]:
    """Get database type from environment variable."""
    return os.getenv("TEST_DATABASE_TYPE", "mysql").lower()


def get_available_databases() -> list[str]:
    """Get list of available databases based on environment variables."""
    available = []

    # Check MySQL availability
    if os.getenv("MYSQL_DB_URL") or all(
        [
            os.getenv("MYSQL_HOST"),
            os.getenv("MYSQL_DATABASE"),
            os.getenv("MYSQL_USERNAME"),
        ]
    ):
        available.append("mysql")

    # Check PostgreSQL availability
    if os.getenv("POSTGRESQL_DB_URL") or all(
        [
            os.getenv("POSTGRES_HOST"),
            os.getenv("POSTGRES_DB"),
            os.getenv("POSTGRES_USER"),
        ]
    ):
        available.append("postgresql")

    return available
