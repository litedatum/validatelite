"""
Connection type enumeration

Defines all types of data source connections, supporting databases, files,
and big data platforms.
Provides connection type judgment and default port acquisition functions.
"""

from enum import Enum

from shared.exceptions.exception_system import OperationError


class ConnectionType(str, Enum):
    """
    Connection type enumeration

    Defines supported data source connection types:
    - Relational databases: SQLite, MySQL, PostgreSQL, etc.
    - File formats: CSV, Parquet, JSON, etc.
    - Big data platforms: Hive, Spark, etc.
    """

    # Relational databases
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    ORACLE = "oracle"
    MSSQL = "mssql"

    # File formats
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    EXCEL = "excel"

    # Big data platforms
    HIVE = "hive"
    SPARK = "spark"
    HDFS = "hdfs"

    # Cloud storage (not supported yet, reserved)
    # S3 = "S3"
    # GCS = "GCS"
    # AZURE_BLOB = "AZURE_BLOB"

    @classmethod
    def is_database(cls, conn_type: "ConnectionType") -> bool:
        """
        Determine if it is a database connection

        Args:
            conn_type: Connection type

        Returns:
            bool: Whether it is a database connection
        """
        db_types = [cls.SQLITE, cls.MYSQL, cls.POSTGRESQL, cls.ORACLE, cls.MSSQL]
        return conn_type in db_types

    @classmethod
    def is_file_based(cls, conn_type: "ConnectionType") -> bool:
        """
        Determine if it is a file type

        Args:
            conn_type: Connection type

        Returns:
            bool: Whether it is a file type
        """
        file_types = [cls.CSV, cls.PARQUET, cls.JSON, cls.EXCEL]
        return conn_type in file_types

    @classmethod
    def is_big_data(cls, conn_type: "ConnectionType") -> bool:
        """
        Determine if it is a big data platform

        Args:
            conn_type: Connection type

        Returns:
            bool: Whether it is a big data platform
        """
        big_data_types = [cls.HIVE, cls.SPARK, cls.HDFS]
        return conn_type in big_data_types

    @classmethod
    def get_default_port(cls, conn_type: "ConnectionType") -> int:
        """
        Get default port

        Args:
            conn_type: Connection type

        Returns:
            int: Default port number, 0 means no default port
        """
        default_ports = {
            cls.MYSQL: 3306,
            cls.POSTGRESQL: 5432,
            cls.ORACLE: 1521,
            cls.MSSQL: 1433,
            cls.HIVE: 10000,
            cls.SPARK: 7077,
        }
        return default_ports.get(conn_type, 0)

    @classmethod
    def get_driver_name(cls, conn_type: "ConnectionType") -> str:
        """
        Get driver name

        Args:
            conn_type: Connection type

        Returns:
            str: Driver name
        """
        drivers = {
            cls.SQLITE: "sqlite3",
            cls.MYSQL: "mysql+pymysql",
            cls.POSTGRESQL: "postgresql+psycopg2",
            cls.ORACLE: "oracle+cx_oracle",
            cls.MSSQL: "mssql+pyodbc",
        }
        return drivers.get(conn_type, "")

    @classmethod
    def requires_credentials(cls, conn_type: "ConnectionType") -> bool:
        """
        Determine if credentials are required

        Args:
            conn_type: Connection type

        Returns:
            bool: Whether credentials are required
        """
        # SQLite and file types usually do not require authentication
        no_auth_types = [cls.SQLITE, cls.CSV, cls.PARQUET, cls.JSON, cls.EXCEL]
        return conn_type not in no_auth_types

    @classmethod
    def supports_schema(cls, conn_type: "ConnectionType") -> bool:
        """
        Determine if schema concept is supported

        Args:
            conn_type: Connection type

        Returns:
            bool: Whether schema is supported
        """
        schema_types = [cls.POSTGRESQL, cls.ORACLE, cls.MSSQL]
        return conn_type in schema_types

    @classmethod
    def get_file_extensions(cls, conn_type: "ConnectionType") -> list[str]:
        """
        Get file type corresponding extensions

        Args:
            conn_type: Connection type

        Returns:
            list[str]: List of supported file extensions
        """
        extensions = {
            cls.CSV: [".csv", ".tsv"],
            cls.PARQUET: [".parquet", ".pq"],
            cls.JSON: [".json", ".jsonl"],
            cls.EXCEL: [".xlsx", ".xls"],
        }
        return extensions.get(conn_type, [])

    @classmethod
    def from_string(cls, value: str) -> "ConnectionType":
        """Case-insensitive conversion from string to ``ConnectionType``.

        The helper first attempts to map the *name* of the enum (e.g. ``CSV``)
        and then falls back to matching the *value* (e.g. ``csv``) ignoring
        case.  This makes the API tolerant to variations such as "CSV", "csv",
        or "Csv" – which is required by the modernised test-suite.
        """
        if isinstance(value, cls):
            # Already a ConnectionType – return as-is.
            return value

        if not isinstance(value, str):
            raise OperationError(
                "ConnectionType must be created from a string or ConnectionType "
                "instance"
            )

        value = value.strip()

        # 1. Try enum *name* (CSV, MYSQL …)
        upper_name = value.upper()
        if upper_name in cls.__members__:
            return cls[upper_name]

        # 2. Try enum *value* (csv, mysql …) case-insensitively
        for conn_type in cls:
            if conn_type.value.lower() == value.lower():
                return conn_type

        # 3. No match – raise a helpful error.
        valid_types = ", ".join([t.value for t in cls])
        raise OperationError(
            f"Unsupported connection type: {value}. Valid types: {valid_types}"
        )

    @classmethod
    def from_file_extension(cls, extension: str) -> "ConnectionType":
        """
        Infer connection type from file extension

        Args:
            extension: File extension (including dot)

        Returns:
            ConnectionType: Corresponding connection type

        Raises:
            OperationError: If the extension is not supported
        """
        extension = extension.lower()

        for conn_type in cls:
            if cls.is_file_based(conn_type):
                extensions = cls.get_file_extensions(conn_type)
                if extension in extensions:
                    return conn_type

        raise OperationError(f"Unsupported file extension: {extension}")
