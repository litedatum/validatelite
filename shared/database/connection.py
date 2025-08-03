"""
Database connection module

Provides functions for database connection management:
1. Connection URL generation
2. Connection testing
3. Engine creation and management
4. Connection pooling
5. Connection object handling
"""

import asyncio  # For locking
from enum import Enum
from typing import Any, Dict, Optional, Union

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from shared.exceptions.exception_system import EngineError
from shared.utils.logger import get_logger


# Assuming ConnectionType is defined elsewhere, e.g., app.schemas.connection
# For standalone testing, let's define a simple version
class ConnectionType:
    """Connection type enum"""

    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    MSSQL = "mssql"
    ORACLE = "oracle"
    SQLITE = "sqlite"
    CSV = "csv"
    EXCEL = "excel"


# Configure logger
logger = get_logger(__name__)

# Engine cache to avoid creating multiple engines for the same connection
_engine_cache: Dict[str, AsyncEngine] = {}
_engine_creation_lock = (
    asyncio.Lock()
)  # To prevent race conditions during engine creation


def get_db_url(
    db_type: Union[ConnectionType, str],
    host: Optional[str] = None,
    port: Optional[int] = None,
    database: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    file_path: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Generate database connection URL"""
    if isinstance(db_type, ConnectionType):
        db_type_enum_val = db_type
    else:
        db_type_str = db_type.lower()
        if hasattr(
            ConnectionType, db_type_str.upper()
        ):  # Check if it's a known enum member name
            db_type_enum_val = getattr(ConnectionType, db_type_str.upper())
        else:
            raise EngineError(f"Unsupported or incomplete database type: {db_type_str}")

    if db_type_enum_val == ConnectionType.MYSQL:
        if not all([host, port, database, username, password]):
            raise EngineError("Missing required parameters for MySQL connection")
        return f"mysql+aiomysql://{username}:{password}@{host}:{port}/{database}"
    elif db_type_enum_val == ConnectionType.POSTGRESQL:
        if not all([host, port, database, username, password]):
            raise EngineError("Missing required parameters for PostgreSQL connection")
        return f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
    elif db_type_enum_val == ConnectionType.MSSQL:
        if not all([host, port, database, username, password]):
            raise EngineError("Missing required parameters for MSSQL connection")
        driver = kwargs.get("driver", "ODBC+Driver+17+for+SQL+Server").replace(" ", "+")
        return (
            f"mssql+aioodbc://{username}:{password}@{host}:{port}/"
            f"{database}?driver={driver}"
        )
    elif db_type_enum_val == ConnectionType.ORACLE:
        # Note: sqlalchemy docs recommend python-oracledb for async,
        # not asyncpg for oracle
        # Example: "oracle+oracledb://user:pass@host:port/service_name"
        # Your original "oracle+asyncpg" is likely incorrect for Oracle.
        # For now, I'll keep it as you had, but flag it.
        if not all([host, port, database, username, password]):
            raise EngineError("Missing required parameters for Oracle connection")
        logger.warning("Consider using 'oracle+oracledb' dialect for async Oracle.")
        return f"oracle+asyncpg://{username}:{password}@{host}:{port}/{database}"
        # CHECK THIS DIALECT
    elif db_type_enum_val == ConnectionType.SQLITE:
        if not file_path:
            raise EngineError("Missing file_path parameter for SQLite connection")
        return f"sqlite+aiosqlite:///{file_path}"
    elif db_type_enum_val in [ConnectionType.CSV, ConnectionType.EXCEL]:
        # These are not typically handled by SQLAlchemy engines directly for pooling.
        # This part of get_db_url might be for a different purpose.
        if not file_path:
            raise EngineError(
                f"Missing file_path parameter for {db_type_str} connection"
            )
        return f"{db_type_str}://{file_path}"  # This URL won't work with
        # create_async_engine
    else:
        # Allow generic pass-through for other sqlalchemy URLs if needed
        # For example, if user provides a full "dialect+driver://..." string
        # for db_type and no other args, it could be a pre-formed URL.
        # However, based on current logic, it will raise an error.
        # This path needs clarification if it's meant to support arbitrary URLs.
        raise EngineError(f"Unsupported or incomplete database type: {db_type_str}")


async def check_connection(db_url: str) -> bool:
    """Test database connection. This should create a TEMPORARY engine."""
    engine: Optional[AsyncEngine] = None
    try:
        logger.debug(
            f"Attempting to check connection for URL: "
            f"{db_url[:db_url.find('@') if '@' in db_url else 50]}..."
        )
        if db_url.startswith("sqlite"):
            engine = create_async_engine(db_url)
        elif db_url.startswith(ConnectionType.CSV) or db_url.startswith(
            ConnectionType.EXCEL
        ):
            # These are not database URLs that SQLAlchemy can connect to.
            # This function might need to handle these differently,
            # or they shouldn't reach here.
            logger.warning(f"Cannot test connection for non-SQLAlchemy URL: {db_url}")
            # For now, assume if we get such a URL, it's a configuration error
            # for this function.
            # Depending on requirements, this could be a `pass` or `return True`
            # if file existence is enough.
            return False  # Or raise NotImplementedError

        else:
            engine = create_async_engine(
                db_url,
                poolclass=NullPool,
                echo=False,
            )
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.debug(
            f"Connection test successful for URL: "
            f"{db_url[:db_url.find('@') if '@' in db_url else 50]}..."
        )
        return True
    except SQLAlchemyError as e:
        logger.error(
            f"Connection test failed for URL "
            f"{db_url[:db_url.find('@') if '@' in db_url else 50]}...: {str(e)}"
        )
        return False
    except Exception as e:  # Catch other potential errors
        # like invalid URL format for create_async_engine
        logger.error(
            f"Generic error during connection test for URL "
            f"{db_url[:db_url.find('@') if '@' in db_url else 50]}...: {str(e)}"
        )
        return False
    finally:
        if engine:
            await engine.dispose()


async def get_engine(
    db_url: str,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: int = 30,
    pool_recycle: int = 1800,  # 30 minutes
    echo: bool = False,
) -> AsyncEngine:
    """
    Get or create a database engine with connection pooling.
    Uses a cache to return existing engines for the same URL.
    """
    # Quick check without lock (most common case if engine already exists)
    if db_url in _engine_cache:
        return _engine_cache[db_url]

    # If not in cache, acquire lock to prevent multiple creations for the same URL
    async with _engine_creation_lock:
        # Double-check if another coroutine created it while we were waiting
        # for the lock
        if db_url in _engine_cache:
            return _engine_cache[db_url]

        logger.info(
            f"Creating new engine for URL: "
            f"{db_url[:db_url.find('@') if '@' in db_url else 50]}..."
        )
        try:
            if db_url.startswith("sqlite"):
                # SQLite specific: often doesn't need complex pooling,
                # but echo is useful.
                engine = create_async_engine(
                    db_url,
                    echo=echo,
                    poolclass=NullPool,  # Use NullPool for SQLite
                    # to avoid connection issues
                    pool_pre_ping=True,  # Enable connection health checks
                )
            elif db_url.startswith(ConnectionType.CSV) or db_url.startswith(
                ConnectionType.EXCEL
            ):
                raise ValueError(
                    f"Cannot create SQLAlchemy engine for file type: {db_url}"
                )
            else:
                # For MySQL/PostgreSQL/etc., use safer connection pool settings
                connect_args: Dict[str, Any] = (
                    {
                        # Connection parameters supported by MySQL/aiomysql
                        "connect_timeout": 10,  # Connection timeout
                        "autocommit": False,  # Disable autocommit
                    }
                    if db_url.startswith("mysql")
                    else (
                        {
                            # Connection parameters supported by PostgreSQL/asyncpg
                            "command_timeout": 30,  # Command timeout
                            "server_settings": {
                                "jit": "off"  # Disable JIT to improve stability
                            },
                        }
                        if db_url.startswith("postgresql")
                        else {}
                    )
                )
                engine = create_async_engine(
                    db_url,
                    pool_size=pool_size,
                    max_overflow=max_overflow,
                    pool_timeout=pool_timeout,
                    pool_recycle=pool_recycle,
                    echo=echo,
                    # Add more secure connection pool configuration
                    pool_pre_ping=True,  # Enable connection health check
                    pool_reset_on_return="commit",  # Reset transaction state
                    # when connection returns to pool
                    connect_args=connect_args,
                )
            _engine_cache[db_url] = engine
            logger.debug(
                f"Engine created and cached for URL: "
                f"{db_url[:db_url.find('@') if '@' in db_url else 50]}."
            )
            return engine
        except SQLAlchemyError as e:
            logger.error(
                f"Failed to create engine for "
                f"{db_url[:db_url.find('@') if '@' in db_url else 50]}: {str(e)}"
            )
            raise EngineError(
                f"Failed to create engine for "
                f"{db_url[:db_url.find('@') if '@' in db_url else 50]}: {str(e)}"
            )
        except Exception as e:  # Catch other potential errors
            logger.error(
                f"Generic error creating engine for "
                f"{db_url[:db_url.find('@') if '@' in db_url else 50]}: {str(e)}"
            )
            raise EngineError(
                f"Generic error creating engine for "
                f"{db_url[:db_url.find('@') if '@' in db_url else 50]}: {str(e)}"
            )


async def close_all_engines() -> None:
    """
    Dispose all cached engines. Call this on application shutdown.

    Ensures the entire process of collecting, clearing, and disposing
    is done atomically with respect to engine creation.
    """
    logger.debug("Initiating closure of all cached database engines...")

    async with _engine_creation_lock:  # Lock acquired for the entire operation
        engines_to_dispose_with_urls = list(_engine_cache.items())
        _engine_cache.clear()  # Clear the cache immediately after copying

        logger.debug(
            f"Collected {len(engines_to_dispose_with_urls)} engines for disposal. "
            "Cache is now empty."
        )

        if not engines_to_dispose_with_urls:
            logger.debug("No cached engines to dispose.")
            return  # Early exit if nothing to do

        for url, engine_instance in engines_to_dispose_with_urls:
            try:
                if not hasattr(engine_instance, "dispose"):
                    logger.warning(
                        f"Engine for URL {url} does not have dispose method "
                        f"(type: {type(engine_instance)})"
                    )
                    continue

                logger.info(
                    "Disposing engine for URL: "
                    f"{url[:url.find('@') if '@' in url else 50]}..."
                )

                # Check if engine has sync attribute (avoid calling corrupted engine)
                if hasattr(engine_instance, "sync") and engine_instance.sync is None:
                    logger.warning(
                        f"Engine for URL {url} appears to have corrupted sync "
                        "attribute, skipping disposal"
                    )
                    continue

                # Add timeout handling
                try:
                    await asyncio.wait_for(engine_instance.dispose(), timeout=30.0)
                    logger.debug(
                        "Successfully disposed engine for URL: "
                        f"{url[:url.find('@') if '@' in url else 50]}."
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Timeout during disposal of engine for URL {url}")
                except Exception as dispose_error:
                    logger.error(
                        f"Error during engine.dispose() for URL {url}: "
                        f"{dispose_error}"
                    )

            except Exception as e:
                # Log the error but continue to try and dispose other engines
                logger.error(
                    f"Error during disposal of engine for URL {url}: {str(e)}",
                    exc_info=True,  # Includes stack trace for better debugging
                )

    logger.info(
        f"Completed attempt to close all {len(engines_to_dispose_with_urls)} "
        "cached engines."
    )


async def retry_connection(
    db_url: str,
    max_retries: int = 3,
    retry_interval: int = 2,
    **engine_kwargs: Any,  # To pass pool_size, echo etc. to get_engine
) -> Optional[AsyncEngine]:
    """
    Retry database connection with exponential backoff.

    Uses the cached get_engine.
    """
    for attempt in range(max_retries):
        try:
            # Try to get/create engine using the cached mechanism
            engine = await get_engine(db_url, **engine_kwargs)

            # Test connection (optional, get_engine itself might fail if URL is bad)
            # For a more robust test, you might want a light query here.
            # However, `create_async_engine` doesn't connect immediately.
            # A simple test is to try acquiring a connection.
            async with engine.connect() as conn:
                await conn.execute(
                    text("SELECT 1")
                )  # Verifies pool can give a connection

            logger.debug(
                f"Connection successful for "
                f"{db_url[:db_url.find('@') if '@' in db_url else 50]} "
                f"on attempt {attempt + 1}."
            )
            return engine
        except (
            Exception
        ) as e:  # Catch SQLAlchemyError and other exceptions from connection
            logger.warning(
                f"Connection attempt {attempt + 1}/{max_retries} for "
                f"{db_url[:db_url.find('@') if '@' in db_url else 50]} failed: {str(e)}"
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_interval * (2**attempt))
            else:
                logger.error(
                    "All connection retries failed for "
                    f"{db_url[:db_url.find('@') if '@' in db_url else 50]}."
                )
    return None


async def check_connection_object(connection_details: Any) -> bool:
    """
    Test database connection using a connection object like structure.
    `connection_details` is expected to have attributes like `connection_type`,
    `host`, etc.
    """
    try:
        logger.debug(
            f"check_connection_object received: type={type(connection_details)}, "
            f"details={connection_details}"
        )
        connection_type_val = getattr(connection_details, "connection_type", None)
        # connection_type_val = getattr(connection_details, 'connection_type', '')
        # Convert to string if it's an enum or other type
        # connection_type_str = str(connection_type_val)
        logger.debug(
            f"Raw connection_type_val: type={type(connection_type_val)}, "
            f"value='{connection_type_val}'"
        )

        if connection_type_val is None:
            logger.error(
                "connection_type attribute is missing or None on connection_details."
            )

        # Handle case where connection_type_val might be an enum member
        # e.g. <ConnectionType.POSTGRESQL: 'postgresql'>
        # We want the 'postgresql' part for get_db_url
        if isinstance(
            connection_type_val, Enum
        ):  # hasattr(connection_type_val, 'value'):
            connection_type_str = str(connection_type_val.value)
        else:
            connection_type_str = str(connection_type_val)

        logger.debug(
            f"Derived connection_type_str for get_db_url: '{connection_type_str}'"
        )

        # get_db_url expects the string 'postgresql', not 'ConnectionType.POSTGRESQL'
        db_url = get_db_url(
            db_type=connection_type_str,  # Pass the string value
            host=getattr(connection_details, "host", None),
            port=getattr(connection_details, "port", None),
            database=getattr(connection_details, "db_name", None),  # or 'database'
            username=getattr(connection_details, "username", None),
            password=getattr(connection_details, "password", None),
            file_path=getattr(connection_details, "file_path", None),
            # Potentially pass other kwargs if get_db_url uses them
            # (e.g. driver for mssql)
            # driver=getattr(connection_details, 'driver', None)
        )
        logger.debug(f"Constructed db_url: '{db_url}'")
        return await check_connection(db_url)
    except EngineError as ve:  # Catch errors from get_db_url (e.g. missing params)
        logger.error(f"Failed to construct DB URL for connection test: {str(ve)}")
        return False
    except Exception as e:
        logger.error(f"Connection test from object failed: {str(e)}")
        return False
