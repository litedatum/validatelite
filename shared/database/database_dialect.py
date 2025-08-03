"""
Database dialect system

Provides a database abstraction layer, supporting compatibility handling
for different databases:
- SQL syntax differences
- Data type mapping
- Function and operator compatibility
- Table existence check
- Index and constraint handling
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from shared.exceptions.exception_system import OperationError
from shared.utils.logger import get_logger


class DatabaseType(Enum):
    """Database type enumeration"""

    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    SQLSERVER = "sqlserver"


class DatabaseDialect(ABC):
    """Database dialect base class"""

    def __init__(self, database_type: DatabaseType):
        """Initialize DatabaseDialect"""
        self.database_type = database_type
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def get_table_exists_sql(
        self, database: str, table: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Get SQL to check if table exists"""
        pass

    @abstractmethod
    def get_column_info_sql(
        self, database: str, table: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Get SQL to retrieve column information"""
        pass

    @abstractmethod
    def get_count_sql(self, database: str, table: str, where_clause: str = "") -> str:
        """Get SQL to count records"""
        pass

    @abstractmethod
    def get_limit_sql(self, sql: str, limit: int, offset: int = 0) -> str:
        """Get SQL with limit clause"""
        pass

    @abstractmethod
    def quote_identifier(self, identifier: str) -> str:
        """Quote identifier"""
        pass

    @abstractmethod
    def format_datetime(self, datetime_str: str) -> str:
        """Format datetime string"""
        pass

    @abstractmethod
    def get_string_length_function(self) -> str:
        """Get string length function"""
        pass

    @abstractmethod
    def get_substring_function(self, column: str, start: int, length: int) -> str:
        """Get substring function"""
        pass

    @abstractmethod
    def get_regex_operator(self) -> str:
        """Get regular expression operator"""
        pass

    @abstractmethod
    def get_not_regex_operator(self) -> str:
        """Get NOT regular expression operator"""
        pass

    @abstractmethod
    def get_case_insensitive_like(self, column: str, pattern: str) -> str:
        """Get case-insensitive LIKE operator"""
        pass

    @abstractmethod
    def get_date_clause(self, column: str, format_pattern: str) -> str:
        """Get date formatting function"""
        pass

    @abstractmethod
    def is_supported_date_format(self) -> bool:
        """Determine if date formatting is supported"""
        pass

    @abstractmethod
    def get_date_functions(self) -> Dict[str, str]:
        """Get date function mapping"""
        pass

    @abstractmethod
    def get_data_type_mapping(self) -> Dict[str, str]:
        """Get data type mapping"""
        pass

    @abstractmethod
    def get_database_list_sql(self) -> Tuple[str, Dict[str, Any]]:
        """Get SQL to retrieve database list"""
        pass

    @abstractmethod
    def get_table_list_sql(
        self, database: str, schema: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Get SQL to retrieve table list"""
        pass

    @abstractmethod
    def get_column_list_sql(
        self, table: str, database: str, schema: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Get SQL to retrieve column list"""
        pass

    def escape_string(self, value: str) -> str:
        """Escape string"""
        return value.replace("'", "''")

    def build_where_clause(self, conditions: List[str]) -> str:
        """Build WHERE clause"""
        if not conditions:
            return ""
        return f"WHERE {' AND '.join(conditions)}"

    def build_full_table_name(self, database: str, table: str) -> str:
        """Build full table name"""
        return f"{self.quote_identifier(database)}.{self.quote_identifier(table)}"

    def get_length_function(self) -> str:
        """Get string length function name (alias)"""
        return self.get_string_length_function()


class MySQLDialect(DatabaseDialect):
    """MySQL dialect"""

    def __init__(self) -> None:
        """MySQL dialect"""
        super().__init__(DatabaseType.MYSQL)
        self.quote_character = "`"  # MySQL uses backticks

    def get_table_exists_sql(
        self, database: str, table: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Get MySQL table exists"""
        sql = """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = :database
            AND table_name = :table
        """
        params = {"database": database, "table": table}
        return sql.strip(), params

    def get_column_info_sql(
        self, database: str, table: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Get MySQL column info"""
        sql = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length,
                numeric_precision,
                numeric_scale
            FROM information_schema.columns
            WHERE table_schema = :database
            AND table_name = :table
            ORDER BY ordinal_position
        """
        params = {"database": database, "table": table}
        return sql.strip(), params

    def get_count_sql(self, database: str, table: str, where_clause: str = "") -> str:
        """MySQL uses COUNT(*) for count"""
        full_table = self.build_full_table_name(database, table)
        if where_clause:
            return f"SELECT COUNT(*) FROM {full_table} WHERE {where_clause}"
        return f"SELECT COUNT(*) FROM {full_table}"

    def get_limit_sql(self, sql: str, limit: int, offset: int = 0) -> str:
        """MySQL uses LIMIT/OFFSET for pagination"""
        if offset > 0:
            return f"{sql} LIMIT {offset}, {limit}"
        return f"{sql} LIMIT {limit}"

    def quote_identifier(self, identifier: str) -> str:
        """MySQL uses backticks for identifiers"""
        return f"`{identifier}`"

    def format_datetime(self, datetime_str: str) -> str:
        """MySQL uses single quotes for datetime strings"""
        return f"'{datetime_str}'"

    def get_string_length_function(self) -> str:
        """MySQL uses CHAR_LENGTH for string length"""
        return "CHAR_LENGTH"

    def get_substring_function(self, column: str, start: int, length: int) -> str:
        """MySQL uses SUBSTRING for substring"""
        return f"SUBSTRING({column}, {start}, {length})"

    def get_regex_operator(self) -> str:
        """MySQL uses REGEXP for regex"""
        return "REGEXP"

    def get_not_regex_operator(self) -> str:
        """MySQL uses NOT REGEXP for not regex"""
        return "NOT REGEXP"

    def get_case_insensitive_like(self, column: str, pattern: str) -> str:
        """MySQL uses LOWER for case-insensitive LIKE"""
        return f"LOWER({column}) LIKE LOWER('{pattern}')"

    def get_date_clause(self, column: str, format_pattern: str) -> str:
        """MySQL uses STR_TO_DATE for date formatting"""
        return f"STR_TO_DATE({column}, '{format_pattern}')"

    def is_supported_date_format(self) -> bool:
        """MySQL supports date formats"""
        return True

    def get_date_functions(self) -> Dict[str, str]:
        """Get MySQL date functions"""
        return {
            "now": "NOW()",
            "today": "CURDATE()",
            "year": "YEAR",
            "month": "MONTH",
            "day": "DAY",
            "hour": "HOUR",
            "minute": "MINUTE",
            "second": "SECOND",
            "date_add": "DATE_ADD",
            "date_sub": "DATE_SUB",
            "datediff": "DATEDIFF",
        }

    def get_data_type_mapping(self) -> Dict[str, str]:
        """Get MySQL data type mapping"""
        return {
            "string": "VARCHAR",
            "text": "TEXT",
            "integer": "INT",
            "bigint": "BIGINT",
            "float": "FLOAT",
            "double": "DOUBLE",
            "decimal": "DECIMAL",
            "boolean": "BOOLEAN",
            "date": "DATE",
            "datetime": "DATETIME",
            "timestamp": "TIMESTAMP",
            "time": "TIME",
            "json": "JSON",
            "blob": "BLOB",
        }

    def get_database_list_sql(self) -> Tuple[str, Dict[str, Any]]:
        """Get MySQL database list"""
        sql = "SHOW DATABASES"
        return sql, {}

    def get_table_list_sql(
        self, database: str, schema: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Get MySQL table list"""
        sql = """
            SELECT
                table_name as name,
                table_type,
                table_schema as schema_name,
                table_catalog as database_name
            FROM information_schema.tables
            WHERE table_schema = :database
            ORDER BY table_name
        """
        params = {"database": database}
        return sql.strip(), params

    def get_column_list_sql(
        self, table: str, database: str, schema: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Get MySQL column list"""
        sql = (
            f"SHOW COLUMNS FROM {self.quote_identifier(database)}."
            f"{self.quote_identifier(table)}"
        )
        return sql, {}


class PostgreSQLDialect(DatabaseDialect):
    """PostgreSQL dialect"""

    def __init__(self) -> None:
        """PostgreSQL dialect"""
        super().__init__(DatabaseType.POSTGRESQL)
        self.quote_character = '"'  # PostgreSQL uses double quotes

    def get_table_exists_sql(
        self, database: str, table: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Get PostgreSQL table exists"""
        sql = """
            SELECT 1
            FROM information_schema.tables
            WHERE table_catalog = :database
            AND table_name = :table
            AND table_schema = 'public'
        """
        params = {"database": database, "table": table}
        return sql.strip(), params

    def get_column_info_sql(
        self, database: str, table: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Get PostgreSQL column info"""
        sql = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length,
                numeric_precision,
                numeric_scale
            FROM information_schema.columns
            WHERE table_catalog = :database
            AND table_name = :table
            AND table_schema = 'public'
            ORDER BY ordinal_position
        """
        params = {"database": database, "table": table}
        return sql.strip(), params

    def build_full_table_name(self, database: str, table: str) -> str:
        """Build full table name - PostgreSQL does not use database as schema prefix"""
        return self.quote_identifier(table)

    def get_count_sql(self, database: str, table: str, where_clause: str = "") -> str:
        """PostgreSQL uses COUNT(*) for count"""
        # In PostgreSQL, database is actually schema
        full_table = f'"{table}"'
        if where_clause:
            return f"SELECT COUNT(*) FROM {full_table} WHERE {where_clause}"
        return f"SELECT COUNT(*) FROM {full_table}"

    def get_limit_sql(self, sql: str, limit: int, offset: int = 0) -> str:
        """PostgreSQL uses LIMIT/OFFSET for pagination"""
        if offset > 0:
            return f"{sql} LIMIT {limit} OFFSET {offset}"
        return f"{sql} LIMIT {limit}"

    def quote_identifier(self, identifier: str) -> str:
        """PostgreSQL uses double quotes for identifiers"""
        return f'"{identifier}"'

    def format_datetime(self, datetime_str: str) -> str:
        """PostgreSQL uses timestamp for datetime"""
        return f"'{datetime_str}'::timestamp"

    def get_string_length_function(self) -> str:
        """PostgreSQL uses LENGTH for string length"""
        return "LENGTH"

    def get_substring_function(self, column: str, start: int, length: int) -> str:
        """PostgreSQL uses SUBSTRING for substring"""
        return f"SUBSTRING({column} FROM {start} FOR {length})"

    def get_regex_operator(self) -> str:
        """PostgreSQL uses ~ for regex"""
        return "~"

    def get_not_regex_operator(self) -> str:
        """PostgreSQL uses !~ for not regex"""
        return "!~"

    def get_case_insensitive_like(self, column: str, pattern: str) -> str:
        """PostgreSQL uses LOWER for case-insensitive LIKE"""
        return f"LOWER({column}) LIKE LOWER('{pattern}')"

    def get_date_clause(self, column: str, format_pattern: str) -> str:
        """PostgreSQL uses TO_TIMESTAMP for date formatting"""
        return f"TO_TIMESTAMP({column}, '{format_pattern}')"

    def is_supported_date_format(self) -> bool:
        """PostgreSQL does not support date formats"""
        return False

    def get_date_functions(self) -> Dict[str, str]:
        """Get PostgreSQL date functions"""
        return {
            "now": "NOW()",
            "today": "CURRENT_DATE",
            "year": "EXTRACT(YEAR FROM",
            "month": "EXTRACT(MONTH FROM",
            "day": "EXTRACT(DAY FROM",
            "hour": "EXTRACT(HOUR FROM",
            "minute": "EXTRACT(MINUTE FROM",
            "second": "EXTRACT(SECOND FROM",
            "date_add": "DATE_ADD",  # Requires custom function
            "date_sub": "DATE_SUB",  # Requires custom function
            "datediff": "DATE_PART('day', AGE(",
        }

    def get_data_type_mapping(self) -> Dict[str, str]:
        """Get PostgreSQL data type mapping"""
        return {
            "string": "VARCHAR",
            "text": "TEXT",
            "integer": "INTEGER",
            "bigint": "BIGINT",
            "float": "REAL",
            "double": "DOUBLE PRECISION",
            "decimal": "DECIMAL",
            "boolean": "BOOLEAN",
            "date": "DATE",
            "datetime": "TIMESTAMP",
            "timestamp": "TIMESTAMP",
            "time": "TIME",
            "json": "JSONB",
        }

    def get_database_list_sql(self) -> Tuple[str, Dict[str, Any]]:
        """Get PostgreSQL database list"""
        sql = "SELECT datname FROM pg_database WHERE datistemplate = false"
        return sql, {}

    def get_table_list_sql(
        self, database: str, schema: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Get PostgreSQL table list"""
        if schema:
            sql = """
                SELECT table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = :schema
                ORDER BY table_name
            """
            params = {"schema": schema}
        else:
            sql = """
                SELECT table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """
            params = {}
        return sql.strip(), params

    def get_column_list_sql(
        self, table: str, database: str, schema: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Get PostgreSQL column list"""
        if schema:
            sql = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = :table AND table_schema = :schema
                ORDER BY ordinal_position
            """
            params = {"table": table, "schema": schema}
        else:
            sql = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = :table AND table_schema = 'public'
                ORDER BY ordinal_position
            """
            params = {"table": table}
        return sql.strip(), params


class SQLiteDialect(DatabaseDialect):
    """SQLite dialect"""

    def __init__(self) -> None:
        """SQLite dialect"""
        super().__init__(DatabaseType.SQLITE)
        self.quote_character = '"'  # SQLite uses double quotes

    def get_table_exists_sql(
        self, database: str, table: str
    ) -> Tuple[str, Dict[str, Any]]:
        """SQLite uses sqlite_master to check table existence"""
        sql = """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table'
            AND name = :table
        """
        params = {"table": table}
        return sql.strip(), params

    def get_column_info_sql(
        self, database: str, table: str
    ) -> Tuple[str, Dict[str, Any]]:
        """SQLite uses PRAGMA command to get column info"""
        # SQLite uses PRAGMA command to get column info
        sql = f"PRAGMA table_info({table})"
        params: Dict[str, Any] = {}
        return sql, params

    def get_count_sql(self, database: str, table: str, where_clause: str = "") -> str:
        """SQLite does not need database prefix"""
        # SQLite does not need database prefix
        quoted_table = self.quote_identifier(table)
        if where_clause:
            return f"SELECT COUNT(*) FROM {quoted_table} WHERE {where_clause}"
        return f"SELECT COUNT(*) FROM {quoted_table}"

    def get_limit_sql(self, sql: str, limit: int, offset: int = 0) -> str:
        """SQLite uses LIMIT/OFFSET for pagination"""
        if offset > 0:
            return f"{sql} LIMIT {limit} OFFSET {offset}"
        return f"{sql} LIMIT {limit}"

    def quote_identifier(self, identifier: str) -> str:
        """SQLite uses double quotes for identifiers"""
        return f'"{identifier}"'

    def format_datetime(self, datetime_str: str) -> str:
        """SQLite uses single quotes for datetime strings"""
        return f"'{datetime_str}'"

    def get_string_length_function(self) -> str:
        """Get SQLite string length function"""
        """SQLite uses LENGTH for string length"""
        return "LENGTH"

    def get_substring_function(self, column: str, start: int, length: int) -> str:
        """SQLite uses SUBSTR for substring"""
        return f"SUBSTR({column}, {start}, {length})"

    def get_regex_operator(self) -> str:
        """SQLite uses REGEXP for regex"""
        return "REGEXP"  # Requires extension loading

    def get_not_regex_operator(self) -> str:
        """SQLite does not have built-in regex"""
        return "NOT REGEXP"

    def get_case_insensitive_like(self, column: str, pattern: str) -> str:
        """SQLite uses COLLATE NOCASE for case-insensitive LIKE"""
        return f"{column} LIKE '{pattern}' COLLATE NOCASE"

    def get_date_clause(self, column: str, format_pattern: str) -> str:
        """SQLite uses strftime for date formatting"""
        fmt_map = {
            "yyyy": "%Y",
            "MM": "%m",
            "dd": "%d",
            "HH": "%H",
            "mm": "%M",
            "ss": "%S",
        }
        for k, v in fmt_map.items():
            format_pattern = format_pattern.replace(k, v)
        return f"strftime('{format_pattern}', {column})"

    def is_supported_date_format(self) -> bool:
        """SQLite does not support date formats"""
        return False

    def get_date_functions(self) -> Dict[str, str]:
        """Get SQLite date functions"""
        return {
            "now": "datetime('now')",
            "today": "date('now')",
            "year": "strftime('%Y',",
            "month": "strftime('%m',",
            "day": "strftime('%d',",
            "hour": "strftime('%H',",
            "minute": "strftime('%M',",
            "second": "strftime('%S',",
            "date_add": "datetime(",  # Requires special handling
            "date_sub": "datetime(",  # Requires special handling
            "datediff": "julianday(",  # Requires special handling
        }

    def get_data_type_mapping(self) -> Dict[str, str]:
        """Get SQLite data type mapping"""
        return {
            "string": "TEXT",
            "text": "TEXT",
            "integer": "INTEGER",
            "bigint": "INTEGER",
            "float": "REAL",
            "double": "REAL",
            "decimal": "REAL",
            "boolean": "INTEGER",
            "date": "TEXT",
            "datetime": "TEXT",
            "timestamp": "TEXT",
            "time": "TEXT",
            "json": "TEXT",
        }

    def get_database_list_sql(self) -> Tuple[str, Dict[str, Any]]:
        """Get SQLite database list (SQLite has only one database file)"""
        sql = "SELECT 'main' as name"
        return sql, {}

    def get_table_list_sql(
        self, database: str, schema: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Get SQLite table list"""
        sql = (
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
        return sql, {}

    def get_column_list_sql(
        self, table: str, database: str, schema: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Get SQLite column list"""
        sql = f"PRAGMA table_info({self.quote_identifier(table)})"
        return sql, {}


class SQLServerDialect(DatabaseDialect):
    """SQL Server dialect"""

    def __init__(self) -> None:
        """SQL Server dialect"""
        super().__init__(DatabaseType.SQLSERVER)
        self.quote_character = "["  # SQL Server uses square brackets

    def get_table_exists_sql(
        self, database: str, table: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Get SQL Server table exists"""
        sql = """
            SELECT 1
            FROM sys.tables t
            JOIN sys.schemas s ON t.schema_id = s.schema_id
            WHERE s.name = :schema
            AND t.name = :table
        """
        params = {"schema": database, "table": table}
        return sql.strip(), params

    def get_column_info_sql(
        self, database: str, table: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Get SQL Server column info"""
        sql = """
            SELECT
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                c.character_maximum_length,
                c.numeric_precision,
                c.numeric_scale
            FROM information_schema.columns c
            WHERE c.table_schema = :schema
            AND c.table_name = :table
            ORDER BY c.ordinal_position
        """
        params = {"schema": database, "table": table}
        return sql.strip(), params

    def get_count_sql(self, database: str, table: str, where_clause: str = "") -> str:
        """SQL Server uses COUNT(*) for count"""
        full_table = self.build_full_table_name(database, table)
        if where_clause:
            return f"SELECT COUNT(*) FROM {full_table} WHERE {where_clause}"
        return f"SELECT COUNT(*) FROM {full_table}"

    def get_limit_sql(self, sql: str, limit: int, offset: int = 0) -> str:
        """SQL Server uses OFFSET/FETCH for pagination"""
        if offset > 0:
            return f"{sql} OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY"
        return f"{sql} OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"

    def quote_identifier(self, identifier: str) -> str:
        """SQL Server uses square brackets for identifiers"""
        return f"[{identifier}]"

    def format_datetime(self, datetime_str: str) -> str:
        """SQL Server uses single quotes for datetime strings"""
        return f"'{datetime_str}'"

    def get_string_length_function(self) -> str:
        """SQL Server uses LEN for string length"""
        return "LEN"

    def get_substring_function(self, column: str, start: int, length: int) -> str:
        """SQL Server uses SUBSTRING for substring"""
        return f"SUBSTRING({column}, {start}, {length})"

    def get_regex_operator(self) -> str:
        """SQL Server does not have built-in regex"""
        return "LIKE"  # SQL Server does not have built-in regex

    def get_not_regex_operator(self) -> str:
        """SQL Server does not have built-in regex"""
        return "NOT LIKE"  # SQL Server does not have built-in regex

    def get_case_insensitive_like(self, column: str, pattern: str) -> str:
        """SQL Server uses LOWER for case-insensitive LIKE"""
        return f"LOWER({column}) LIKE LOWER('{pattern}')"

    def get_date_clause(self, column: str, format_pattern: str) -> str:
        """SQL Server uses CAST to convert date strings"""
        return f"CAST({column} AS DATETIME)"

    def is_supported_date_format(self) -> bool:
        """SQL Server does not support date formats"""
        return False

    def get_date_functions(self) -> Dict[str, str]:
        """Get SQL Server date functions"""
        return {
            "now": "GETDATE()",
            "today": "CAST(GETDATE() AS DATE)",
            "year": "YEAR",
            "month": "MONTH",
            "day": "DAY",
            "hour": "DATEPART(HOUR,",
            "minute": "DATEPART(MINUTE,",
            "second": "DATEPART(SECOND,",
            "date_add": "DATEADD",
            "date_sub": "DATEADD",  # Uses negative numbers
            "datediff": "DATEDIFF",
        }

    def get_data_type_mapping(self) -> Dict[str, str]:
        """Get SQL Server data type mapping"""
        return {
            "string": "NVARCHAR",
            "text": "NTEXT",
            "integer": "INT",
            "bigint": "BIGINT",
            "float": "FLOAT",
            "double": "FLOAT",
            "decimal": "DECIMAL",
            "boolean": "BIT",
            "date": "DATE",
            "datetime": "DATETIME2",
            "timestamp": "DATETIME2",
            "time": "TIME",
            "json": "NVARCHAR(MAX)",  # SQL Server 2016+ supports JSON
        }

    def get_database_list_sql(self) -> Tuple[str, Dict[str, Any]]:
        """Get SQL Server database list"""
        sql = "SELECT name FROM sys.databases WHERE database_id > 4"
        # Exclude system databases
        return sql, {}

    def get_table_list_sql(
        self, database: str, schema: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Get SQL Server table list"""
        if schema:
            sql = """
                SELECT TABLE_NAME, TABLE_TYPE
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_CATALOG = ? AND TABLE_SCHEMA = ?
                ORDER BY TABLE_NAME
            """
            params = {"database": database, "schema": schema}
        else:
            sql = """
                SELECT TABLE_NAME, TABLE_TYPE
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_CATALOG = ?
                ORDER BY TABLE_NAME
            """
            params = {"database": database}
        return sql.strip(), params

    def get_column_list_sql(
        self, table: str, database: str, schema: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Get SQL Server column list"""
        if schema:
            sql = """
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = ? AND TABLE_CATALOG = ? AND TABLE_SCHEMA = ?
                ORDER BY ORDINAL_POSITION
            """
            params = {"table": table, "database": database, "schema": schema}
        else:
            sql = """
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = ? AND TABLE_CATALOG = ?
                ORDER BY ORDINAL_POSITION
            """
            params = {"table": table, "database": database}
        return sql.strip(), params


class DatabaseDialectFactory:
    """Database dialect factory"""

    _dialects: Dict[DatabaseType, DatabaseDialect] = {}

    @classmethod
    def get_dialect(cls, database_type: Union[str, DatabaseType]) -> DatabaseDialect:
        """Get database dialect"""
        if isinstance(database_type, str):
            try:
                database_type = DatabaseType(database_type.lower())
            except ValueError:
                raise OperationError(f"Unsupported database type: {database_type}")

        if database_type not in cls._dialects:
            if database_type == DatabaseType.MYSQL:
                cls._dialects[database_type] = MySQLDialect()
            elif database_type == DatabaseType.POSTGRESQL:
                cls._dialects[database_type] = PostgreSQLDialect()
            elif database_type == DatabaseType.SQLITE:
                cls._dialects[database_type] = SQLiteDialect()
            elif database_type == DatabaseType.SQLSERVER:
                cls._dialects[database_type] = SQLServerDialect()
            else:
                raise OperationError(f"Unsupported database type: {database_type}")

        return cls._dialects[database_type]

    @classmethod
    def register_dialect(
        cls, database_type: DatabaseType, dialect: DatabaseDialect
    ) -> None:
        """Register custom dialect"""
        cls._dialects[database_type] = dialect

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get supported database types"""
        return [db_type.value for db_type in DatabaseType]


def get_dialect(database_type: Union[str, DatabaseType]) -> DatabaseDialect:
    """Convenience function to get database dialect"""
    return DatabaseDialectFactory.get_dialect(database_type)
