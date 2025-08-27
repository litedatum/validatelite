"""
ConnectionSchema - Based on refactored ConnectionBase

Inherits from shared.schema.base.ConnectionBase, adds methods and properties
for connection management.
Supports current version's database connections, reserves hooks for
cross-database features.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from shared.enums import ConnectionType
from shared.exceptions.exception_system import OperationError
from shared.schema.base import ConnectionBase, DataSourceCapability
from shared.utils.logger import get_logger

logger = get_logger(__name__)


class ConnectionSchema(ConnectionBase):
    """
    ConnectionSchema - Database connection interface

    Based on refactored ConnectionBase, provides connection management features:
    1. Unified database connection interface
    2. Reserved hooks for cross-database capabilities
    3. Connection string construction and validation
    """

    # Add unique identifier for connection management
    id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for the connection"
    )
    available_tables: Optional[List[str]] = Field(
        default=None, description="List of available tables for file-based sources"
    )

    # ==================== Convenient methods ====================

    def get_connection_string(self) -> str:
        """Build connection string"""
        if self.connection_type == ConnectionType.SQLITE:
            # Handle file path to avoid double slashes
            file_path = self.file_path or ":memory:"
            if file_path.startswith("/") or (
                len(file_path) > 1 and file_path[1] == ":"
            ):
                # Absolute path (Unix or Windows)
                return f"sqlite://{file_path}"
            else:
                # Relative path
                return f"sqlite:///{file_path}"

        elif self.connection_type == ConnectionType.MYSQL:
            auth_part = ""
            if self.username:
                auth_part = f"{self.username}"
                if self.password:
                    auth_part += f":{self.password}"
                auth_part += "@"

            port_part = f":{self.port}" if self.port else ""
            db_part = f"/{self.db_name}" if self.db_name else ""

            return f"mysql://{auth_part}{self.host}{port_part}{db_part}"

        elif self.connection_type == ConnectionType.POSTGRESQL:
            auth_part = ""
            if self.username:
                auth_part = f"{self.username}"
                if self.password:
                    auth_part += f":{self.password}"
                auth_part += "@"

            port_part = f":{self.port}" if self.port else ""
            db_part = f"/{self.db_name}" if self.db_name else ""

            return f"postgresql://{auth_part}{self.host}{port_part}{db_part}"

        else:
            raise OperationError(
                f"Unsupported connection type for connection string: "
                f"{self.connection_type}"
            )

    def get_dsn_dict(self) -> Dict[str, Any]:
        """Get DSN dictionary"""
        if self.connection_type == ConnectionType.SQLITE:
            return {"driver": "sqlite3", "database": self.file_path or ":memory:"}

        elif self.connection_type in [ConnectionType.MYSQL, ConnectionType.POSTGRESQL]:
            dsn = {
                "host": self.host,
                "port": self.port,
                "database": self.db_name,
                "username": self.username,
                "password": self.password,
            }

            if self.db_schema:
                dsn["schema"] = self.db_schema

            # Add extra parameters
            if self.parameters:
                dsn.update(self.parameters)

            return dsn

        else:
            raise OperationError(
                f"Unsupported connection type for DSN: {self.connection_type}"
            )

    def to_engine_dict(self) -> Dict[str, Any]:
        """Convert to rule engine-specific dictionary format"""
        return {
            "type": self.connection_type.value,
            "host": self.host,
            "port": self.port,
            "database": self.db_name,
            "username": self.username,
            "password": self.password,
            "file_path": self.file_path,
            "parameters": self.parameters or {},
        }

    @classmethod
    def from_connection_string(
        cls, name: str, connection_string: str
    ) -> "ConnectionSchema":
        """Create from connection string"""
        from urllib.parse import urlparse

        parsed = urlparse(connection_string)

        if parsed.scheme == "sqlite":
            return cls(
                name=name,
                description=f"SQLite connection from {connection_string}",
                connection_type=ConnectionType.SQLITE,
                file_path=parsed.path,
                capabilities=DataSourceCapability(supports_sql=True),
            )

        elif parsed.scheme == "mysql":
            return cls(
                name=name,
                description=f"MySQL connection from {connection_string}",
                connection_type=ConnectionType.MYSQL,
                host=parsed.hostname,
                port=parsed.port
                or ConnectionType.get_default_port(ConnectionType.MYSQL),
                db_name=parsed.path.lstrip("/") if parsed.path else None,
                username=parsed.username,
                password=parsed.password,
                capabilities=DataSourceCapability(supports_sql=True),
            )

        elif parsed.scheme in ["postgresql", "postgres"]:
            return cls(
                name=name,
                description=f"PostgreSQL connection from {connection_string}",
                connection_type=ConnectionType.POSTGRESQL,
                host=parsed.hostname,
                port=parsed.port
                or ConnectionType.get_default_port(ConnectionType.POSTGRESQL),
                db_name=parsed.path.lstrip("/") if parsed.path else None,
                username=parsed.username,
                password=parsed.password,
                capabilities=DataSourceCapability(supports_sql=True),
            )

        else:
            raise OperationError(
                f"Unsupported connection string scheme: {parsed.scheme}"
            )

    @classmethod
    def create_sqlite_memory(cls, name: str = "memory_db") -> "ConnectionSchema":
        """Create in-memory SQLite connection"""
        return cls(
            name=name,
            description="In-memory SQLite database",
            connection_type=ConnectionType.SQLITE,
            file_path=":memory:",
            capabilities=DataSourceCapability(supports_sql=True),
        )

    @classmethod
    def create_sqlite_file(cls, name: str, file_path: str) -> "ConnectionSchema":
        """Create file-based SQLite connection"""
        return cls(
            name=name,
            description=f"SQLite file database: {file_path}",
            connection_type=ConnectionType.SQLITE,
            file_path=file_path,
            capabilities=DataSourceCapability(supports_sql=True),
        )

    # ==================== Validation methods ====================

    @model_validator(mode="after")
    def validate_connection_consistency(self) -> "ConnectionSchema":
        """Connection consistency validation"""
        # For SQLite we *always* need a file path ("/:memory:" accepted).  For
        # plain CSV / Excel connections the caller *might* provide the data
        # dynamically (e.g. via an in-memory DataFrame) – this is exactly what
        # the unit-test harness does.  Therefore we only enforce the presence
        # of ``file_path`` for SQLite.

        if self.connection_type == ConnectionType.SQLITE and not self.file_path:
            raise OperationError("File path is required for sqlite connections")

        # Database connection validation
        elif self.connection_type in [
            ConnectionType.MYSQL,
            ConnectionType.POSTGRESQL,
            ConnectionType.MSSQL,
            ConnectionType.ORACLE,
        ]:
            if not self.host:
                raise OperationError(
                    f"Host is required for {self.connection_type.value} connections"
                )
            if not self.port:
                raise OperationError(
                    f"Port is required for {self.connection_type.value} connections"
                )
            if not self.db_name:
                logger.warning(
                    f"Database name not specified for {self.connection_type.value} "
                    "connection"
                )

        return self

    async def test_connection(self) -> Dict[str, Any]:
        """Test connection"""
        import time

        from shared.database.connection import check_connection, get_db_url

        try:
            start_time = time.time()

            # Build connection URL based on connection type
            if self.connection_type == ConnectionType.SQLITE:
                db_url = get_db_url(
                    db_type=self.connection_type.value, file_path=self.file_path
                )
            elif self.connection_type in [
                ConnectionType.MYSQL,
                ConnectionType.POSTGRESQL,
                ConnectionType.MSSQL,
                ConnectionType.ORACLE,
            ]:
                db_url = get_db_url(
                    db_type=self.connection_type.value,
                    host=self.host,
                    port=self.port,
                    database=self.db_name,
                    username=self.username,
                    password=self.password,
                )
            elif self.connection_type in [ConnectionType.CSV, ConnectionType.EXCEL]:
                # For file type connections, check if file exists
                import os

                if not self.file_path or not os.path.exists(self.file_path):
                    return {
                        "success": False,
                        "message": f"File not found: {self.file_path}",
                        "response_time": 0.0,
                        "database_version": "N/A",
                    }
                else:
                    return {
                        "success": True,
                        "message": f"File {self.file_path} exists and is accessible",
                        "response_time": (time.time() - start_time) * 1000,
                        "database_version": "file",
                    }
            else:
                return {
                    "success": False,
                    "message": f"Unsupported connection type: {self.connection_type}",
                    "response_time": 0.0,
                    "database_version": "N/A",
                }

            # Test database connection
            success = await check_connection(db_url)
            latency_ms = (time.time() - start_time) * 1000

            if success:
                return {
                    "success": True,
                    "message": f"Connection to {self.connection_type.value} "
                    "successful",
                    "response_time": latency_ms,
                    "database_version": "unknown",
                    # Can be extended to get database version later
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to connect to {self.connection_type.value}",
                    "response_time": latency_ms,
                    "database_version": "N/A",
                }

        except Exception as e:
            logger.error(f"Error testing connection {self.name}: {str(e)}")
            return {
                "success": False,
                "message": f"Connection test error: {str(e)}",
                "response_time": 0.0,
                "database_version": "N/A",
            }

    # ==================== Hook check methods ====================

    def check_cross_db_capabilities(self) -> Dict[str, Any]:
        """Check cross-database capabilities"""
        result: Dict[str, Any] = {
            "supports_export": self.capabilities.supports_batch_export,
            "supports_streaming": self.capabilities.supports_streaming,
            "max_export_rows": self.capabilities.max_export_rows,
            "estimated_throughput": self.capabilities.estimated_throughput,
            "current_version_support": False,  # Not supported in current version
        }

        # Current version limitations
        result["limitations"] = [
            "Cross-database functionality will be supported in future versions",
            "Current version only supports single database connection",
        ]

        return result

    def supports_rule_type(self, rule_type: str) -> bool:
        """Check if the specified rule type is supported"""
        # All database types support basic rules in current version
        if self.connection_type in [
            ConnectionType.SQLITE,
            ConnectionType.MYSQL,
            ConnectionType.POSTGRESQL,
        ]:
            return True

        # File types have limited support
        elif self.connection_type in [ConnectionType.CSV, ConnectionType.EXCEL]:
            supported_types = ["NOT_NULL", "RANGE", "ENUM", "REGEX", "LENGTH"]
            return rule_type in supported_types

        return False

    def get_supported_rule_types(self) -> list[str]:
        """Get list of supported rule types"""
        if self.connection_type in [
            ConnectionType.SQLITE,
            ConnectionType.MYSQL,
            ConnectionType.POSTGRESQL,
        ]:
            return [
                "NOT_NULL",
                "UNIQUE",
                "RANGE",
                "ENUM",
                "REGEX",
                "EMAIL",
                "PHONE",
                "LENGTH",
                "CUSTOM_SQL",
            ]

        elif self.connection_type in [ConnectionType.CSV, ConnectionType.EXCEL]:
            return ["NOT_NULL", "RANGE", "ENUM", "REGEX", "LENGTH"]

        return []
