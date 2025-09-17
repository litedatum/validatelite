"""
Database dialect system unit tests.

Focused on testing the core functionality of the database dialect.
MySQL dialect testing.
PostgreSQL dialect testing.
SQLite dialect testing.
Dialect Factory Tests
SQL Generation Testing
"""

from typing import Any, Dict

import pytest

from shared.database.database_dialect import (
    DatabaseDialect,
    DatabaseDialectFactory,
    DatabaseType,
    MySQLDialect,
    PostgreSQLDialect,
    SQLiteDialect,
    SQLServerDialect,
    get_dialect,
)
from shared.exceptions.exception_system import OperationError


class TestMySQLDialect:
    """Test the MySQL dialect."""

    @pytest.fixture
    def mysql_dialect(self) -> MySQLDialect:
        """MySQL dialect fixture."""
        return MySQLDialect()

    def test_mysql_dialect_creation(self, mysql_dialect: MySQLDialect) -> None:
        """Testing MySQL dialect creation."""
        assert mysql_dialect.database_type == DatabaseType.MYSQL
        assert isinstance(mysql_dialect, DatabaseDialect)

    def test_table_exists_sql(self, mysql_dialect: MySQLDialect) -> None:
        """SQL query to check for table existence."""
        sql, params = mysql_dialect.get_table_exists_sql("test_db", "test_table")

        assert "information_schema.tables" in sql.lower()
        assert "table_schema" in sql.lower()
        assert "table_name" in sql.lower()
        assert params["database"] == "test_db"
        assert params["table"] == "test_table"

    def test_column_info_sql(self, mysql_dialect: MySQLDialect) -> None:
        """SQL query for retrieving column information."""
        sql, params = mysql_dialect.get_column_info_sql("test_db", "test_table")

        assert "information_schema.columns" in sql.lower()
        assert "column_name" in sql.lower()
        assert "data_type" in sql.lower()
        assert params["database"] == "test_db"
        assert params["table"] == "test_table"

    def test_count_sql(self, mysql_dialect: MySQLDialect) -> None:
        """Test SQL query generation for counts."""
        # Unconditional count.  Or, depending on the context, you might say "Count all instances." or  "Total count".
        sql = mysql_dialect.get_count_sql("test_db", "test_table")
        assert "SELECT COUNT(*)" in sql
        assert "`test_db`.`test_table`" in sql

        # Conditional counting.
        sql_with_where = mysql_dialect.get_count_sql(
            "test_db", "test_table", "id > 100"
        )
        assert "WHERE id > 100" in sql_with_where

    def test_limit_sql(self, mysql_dialect: MySQLDialect) -> None:
        """Test SQL generation limits."""
        base_sql = "SELECT * FROM test_table"

        # Only the limit (is applicable/relevant/used).
        limited_sql = mysql_dialect.get_limit_sql(base_sql, 10)
        assert "LIMIT 10" in limited_sql

        # Limit and offset (for database queries).
        limited_offset_sql = mysql_dialect.get_limit_sql(base_sql, 10, 20)
        assert "LIMIT 20, 10" in limited_offset_sql

    def test_quote_identifier(self, mysql_dialect: MySQLDialect) -> None:
        """Testing identifier references."""
        quoted = mysql_dialect.quote_identifier("test_table")
        assert quoted == "`test_table`"

    def test_string_functions(self, mysql_dialect: MySQLDialect) -> None:
        """Testing string functions."""
        # Length function.
        length_func = mysql_dialect.get_string_length_function()
        assert length_func == "CHAR_LENGTH"

        # Substring function
        substr_func = mysql_dialect.get_substring_function("column_name", 1, 10)
        assert "SUBSTRING(column_name, 1, 10)" == substr_func

    def test_regex_operator(self, mysql_dialect: MySQLDialect) -> None:
        """Testing regular expression operations."""
        regex_op = mysql_dialect.get_regex_operator()
        assert regex_op == "REGEXP"

    def test_case_insensitive_like(self, mysql_dialect: MySQLDialect) -> None:
        """Test case-insensitive LIKE operation."""
        like_expr = mysql_dialect.get_case_insensitive_like("column_name", "pattern")
        assert "LOWER(column_name)" in like_expr
        assert "LOWER('pattern')" in like_expr

    def test_date_functions(self, mysql_dialect: MySQLDialect) -> None:
        """Testing the date function mapping."""
        date_funcs = mysql_dialect.get_date_functions()

        assert date_funcs["now"] == "NOW()"
        assert date_funcs["today"] == "CURDATE()"
        assert date_funcs["year"] == "YEAR"
        assert date_funcs["datediff"] == "DATEDIFF"

    def test_data_type_mapping(self, mysql_dialect: MySQLDialect) -> None:
        """Testing data type mapping."""
        type_mapping = mysql_dialect.get_data_type_mapping()

        assert type_mapping["string"] == "VARCHAR"
        assert type_mapping["integer"] == "INT"
        assert type_mapping["boolean"] == "BOOLEAN"
        assert type_mapping["json"] == "JSON"


class TestPostgreSQLDialect:
    """Test the PostgreSQL dialect."""

    @pytest.fixture
    def postgresql_dialect(self) -> PostgreSQLDialect:
        """PostgreSQL dialect fixture."""
        return PostgreSQLDialect()

    def test_postgresql_dialect_creation(
        self, postgresql_dialect: PostgreSQLDialect
    ) -> None:
        """Testing PostgreSQL dialect creation."""
        assert postgresql_dialect.database_type == DatabaseType.POSTGRESQL
        assert isinstance(postgresql_dialect, DatabaseDialect)

    def test_table_exists_sql(self, postgresql_dialect: PostgreSQLDialect) -> None:
        """SQL query to check for table existence."""
        sql, params = postgresql_dialect.get_table_exists_sql("test_db", "test_table")

        assert "information_schema.tables" in sql.lower()
        assert "table_catalog" in sql.lower()
        assert "table_schema = 'public'" in sql.lower()
        assert params["database"] == "test_db"
        assert params["table"] == "test_table"

    def test_limit_sql(self, postgresql_dialect: PostgreSQLDialect) -> None:
        """Test limitations of SQL generation."""
        base_sql = "SELECT * FROM test_table"

        # Only the `LIMIT` value (or clause) is present/specified/used.
        limited_sql = postgresql_dialect.get_limit_sql(base_sql, 10)
        assert "LIMIT 10" in limited_sql

        # Limit and Offset (for database queries)
        limited_offset_sql = postgresql_dialect.get_limit_sql(base_sql, 10, 20)
        assert "LIMIT 10 OFFSET 20" in limited_offset_sql

    def test_quote_identifier(self, postgresql_dialect: PostgreSQLDialect) -> None:
        """Test identifier resolution."""
        quoted = postgresql_dialect.quote_identifier("test_table")
        assert quoted == '"test_table"'

    def test_string_functions(self, postgresql_dialect: PostgreSQLDialect) -> None:
        """Testing string functions."""
        # Length function.
        length_func = postgresql_dialect.get_string_length_function()
        assert length_func == "LENGTH"

        # Substring function.
        substr_func = postgresql_dialect.get_substring_function("column_name", 1, 10)
        assert "SUBSTRING(column_name FROM 1 FOR 10)" == substr_func

    def test_regex_operator(self, postgresql_dialect: PostgreSQLDialect) -> None:
        """Testing regular expression operations."""
        regex_op = postgresql_dialect.get_regex_operator()
        assert regex_op == "~"

    def test_date_functions(self, postgresql_dialect: PostgreSQLDialect) -> None:
        """Testing the date function mapping."""
        date_funcs = postgresql_dialect.get_date_functions()

        assert date_funcs["now"] == "NOW()"
        assert date_funcs["today"] == "CURRENT_DATE"
        assert "EXTRACT(YEAR FROM" in date_funcs["year"]
        assert "EXTRACT(MONTH FROM" in date_funcs["month"]

    def test_data_type_mapping(self, postgresql_dialect: PostgreSQLDialect) -> None:
        """Data type mapping test/tests."""
        type_mapping = postgresql_dialect.get_data_type_mapping()

        assert type_mapping["string"] == "VARCHAR"
        assert type_mapping["integer"] == "INTEGER"
        assert type_mapping["double"] == "DOUBLE PRECISION"
        assert type_mapping["json"] == "JSONB"


class TestSQLiteDialect:
    """Testing the SQLite dialect."""

    @pytest.fixture
    def sqlite_dialect(self) -> SQLiteDialect:
        """SQLite dialect fixture"""
        return SQLiteDialect()

    def test_sqlite_dialect_creation(self, sqlite_dialect: SQLiteDialect) -> None:
        """Testing SQLite dialect creation."""
        assert sqlite_dialect.database_type == DatabaseType.SQLITE
        assert isinstance(sqlite_dialect, DatabaseDialect)

    def test_table_exists_sql(self, sqlite_dialect: SQLiteDialect) -> None:
        """SQL query to check for table existence."""
        sql, params = sqlite_dialect.get_table_exists_sql("test_db", "test_table")

        assert "sqlite_master" in sql.lower()
        assert "type = 'table'" in sql.lower()
        assert "name = :table" in sql.lower()
        assert params["table"] == "test_table"

    def test_column_info_sql(self, sqlite_dialect: SQLiteDialect) -> None:
        """SQL query for retrieving column information."""
        sql, params = sqlite_dialect.get_column_info_sql("test_db", "test_table")

        assert "PRAGMA table_info(test_table)" in sql
        assert params == {}

    def test_count_sql(self, sqlite_dialect: SQLiteDialect) -> None:
        """Testing SQL query generation for counts."""
        # Unconditional count.
        sql = sqlite_dialect.get_count_sql("test_db", "test_table")
        assert "SELECT COUNT(*)" in sql
        assert '"test_table"' in sql

        # Conditional counting.
        sql_with_where = sqlite_dialect.get_count_sql(
            "test_db", "test_table", "id > 100"
        )
        assert "WHERE id > 100" in sql_with_where

    def test_quote_identifier(self, sqlite_dialect: SQLiteDialect) -> None:
        """Test identifier reference."""
        quoted = sqlite_dialect.quote_identifier("test_table")
        assert quoted == '"test_table"'

    def test_string_functions(self, sqlite_dialect: SQLiteDialect) -> None:
        """Testing string functions."""
        # Length function
        length_func = sqlite_dialect.get_string_length_function()
        assert length_func == "LENGTH"

        # Substring function
        substr_func = sqlite_dialect.get_substring_function("column_name", 1, 10)
        assert "SUBSTR(column_name, 1, 10)" == substr_func

    def test_case_insensitive_like(self, sqlite_dialect: SQLiteDialect) -> None:
        """Test case-insensitive LIKE operation."""
        like_expr = sqlite_dialect.get_case_insensitive_like("column_name", "pattern")
        assert "COLLATE NOCASE" in like_expr

    def test_date_functions(self, sqlite_dialect: SQLiteDialect) -> None:
        """Testing the date function mapping."""
        date_funcs = sqlite_dialect.get_date_functions()

        assert date_funcs["now"] == "datetime('now')"
        assert date_funcs["today"] == "date('now')"
        assert "strftime('%Y'," in date_funcs["year"]
        assert "strftime('%m'," in date_funcs["month"]

    def test_data_type_mapping(self, sqlite_dialect: SQLiteDialect) -> None:
        """Testing data type mapping."""
        type_mapping = sqlite_dialect.get_data_type_mapping()

        assert type_mapping["string"] == "TEXT"
        assert type_mapping["integer"] == "INTEGER"
        assert type_mapping["boolean"] == "INTEGER"
        assert type_mapping["json"] == "TEXT"


class TestSQLServerDialect:
    """Test the SQL Server dialect."""

    @pytest.fixture
    def sqlserver_dialect(self) -> SQLServerDialect:
        """SQL Server dialect fixture"""
        return SQLServerDialect()

    def test_sqlserver_dialect_creation(
        self, sqlserver_dialect: SQLServerDialect
    ) -> None:
        """Testing SQL Server dialect creation."""
        assert sqlserver_dialect.database_type == DatabaseType.SQLSERVER
        assert isinstance(sqlserver_dialect, DatabaseDialect)

    def test_limit_sql(self, sqlserver_dialect: SQLServerDialect) -> None:
        """Testing limitations of SQL generation."""
        base_sql = "SELECT * FROM test_table"

        # Only the `LIMIT` value (or clause) is present/specified/applied.
        limited_sql = sqlserver_dialect.get_limit_sql(base_sql, 10)
        assert "OFFSET 0 ROWS FETCH NEXT 10 ROWS ONLY" in limited_sql

        # LIMIT and OFFSET clauses.
        limited_offset_sql = sqlserver_dialect.get_limit_sql(base_sql, 10, 20)
        assert "OFFSET 20 ROWS FETCH NEXT 10 ROWS ONLY" in limited_offset_sql

    def test_quote_identifier(self, sqlserver_dialect: SQLServerDialect) -> None:
        """Test identifier resolution."""
        quoted = sqlserver_dialect.quote_identifier("test_table")
        assert quoted == "[test_table]"

    def test_string_functions(self, sqlserver_dialect: SQLServerDialect) -> None:
        """Testing string functions."""
        # Length function.
        length_func = sqlserver_dialect.get_string_length_function()
        assert length_func == "LEN"

        # Substring function.
        substr_func = sqlserver_dialect.get_substring_function("column_name", 1, 10)
        assert "SUBSTRING(column_name, 1, 10)" == substr_func

    def test_date_functions(self, sqlserver_dialect: SQLServerDialect) -> None:
        """Testing the date function mapping."""
        date_funcs = sqlserver_dialect.get_date_functions()

        assert date_funcs["now"] == "GETDATE()"
        assert date_funcs["today"] == "CAST(GETDATE() AS DATE)"
        assert date_funcs["year"] == "YEAR"
        assert date_funcs["datediff"] == "DATEDIFF"

    def test_data_type_mapping(self, sqlserver_dialect: SQLServerDialect) -> None:
        """Testing data type mappings."""
        type_mapping = sqlserver_dialect.get_data_type_mapping()

        assert type_mapping["string"] == "NVARCHAR"
        assert type_mapping["integer"] == "INT"
        assert type_mapping["boolean"] == "BIT"
        assert type_mapping["json"] == "NVARCHAR(MAX)"


class TestDatabaseDialectFactory:
    """Test database dialect factory."""

    def test_get_mysql_dialect(self) -> None:
        """Test retrieving the MySQL dialect."""
        dialect = DatabaseDialectFactory.get_dialect("mysql")
        assert isinstance(dialect, MySQLDialect)
        assert dialect.database_type == DatabaseType.MYSQL

    def test_get_postgresql_dialect(self) -> None:
        """Test retrieving the PostgreSQL dialect."""
        dialect = DatabaseDialectFactory.get_dialect("postgresql")
        assert isinstance(dialect, PostgreSQLDialect)
        assert dialect.database_type == DatabaseType.POSTGRESQL

    def test_get_sqlite_dialect(self) -> None:
        """Test retrieving the SQLite dialect."""
        dialect = DatabaseDialectFactory.get_dialect("sqlite")
        assert isinstance(dialect, SQLiteDialect)
        assert dialect.database_type == DatabaseType.SQLITE

    def test_get_sqlserver_dialect(self) -> None:
        """Test retrieving the SQL Server dialect."""
        dialect = DatabaseDialectFactory.get_dialect("sqlserver")
        assert isinstance(dialect, SQLServerDialect)
        assert dialect.database_type == DatabaseType.SQLSERVER

    def test_get_dialect_with_enum(self) -> None:
        """Test dialect retrieval using enumeration."""
        dialect = DatabaseDialectFactory.get_dialect(DatabaseType.MYSQL)
        assert isinstance(dialect, MySQLDialect)

    def test_get_unsupported_dialect(self) -> None:
        """Test retrieval of an unsupported dialect."""
        with pytest.raises(OperationError, match="Unsupported database type"):
            DatabaseDialectFactory.get_dialect("unsupported_db")

    def test_dialect_singleton(self) -> None:
        """Testing the dialect singleton pattern."""
        dialect1 = DatabaseDialectFactory.get_dialect("mysql")
        dialect2 = DatabaseDialectFactory.get_dialect("mysql")
        assert dialect1 is dialect2

    def test_get_supported_types(self) -> None:
        """Test retrieval of supported database types."""
        supported_types = DatabaseDialectFactory.get_supported_types()

        assert "mysql" in supported_types
        assert "postgresql" in supported_types
        assert "sqlite" in supported_types
        assert "sqlserver" in supported_types

    def test_register_custom_dialect(self) -> None:
        """Test registration of a custom dialect."""
        # Creating a custom dialect.
        custom_dialect = MySQLDialect()  # Using the MySQL dialect as an example.

        # Register a custom dialect.
        DatabaseDialectFactory.register_dialect(DatabaseType.MYSQL, custom_dialect)

        # Verify successful registration.
        retrieved_dialect = DatabaseDialectFactory.get_dialect(DatabaseType.MYSQL)
        assert retrieved_dialect is custom_dialect


class TestDialectConvenienceFunction:
    """Test utility functions for dialects."""

    def test_get_dialect_function(self) -> None:
        """Test the `get_dialect` helper function."""
        dialect = get_dialect("mysql")
        assert isinstance(dialect, MySQLDialect)

    def test_get_dialect_with_enum(self) -> None:
        """Test the `get_dialect` function with enumerated values."""
        dialect = get_dialect(DatabaseType.POSTGRESQL)
        assert isinstance(dialect, PostgreSQLDialect)


class TestDialectCommonFunctionality:
    """Testing dialect-agnostic functionality."""

    @pytest.fixture(params=["mysql", "postgresql", "sqlite", "sqlserver"])
    def dialect(self, request: pytest.FixtureRequest) -> DatabaseDialect:
        """Parameterized dialect fixture."""
        return get_dialect(request.param)

    def test_escape_string(self, dialect: DatabaseDialect) -> None:
        """Testing string escaping."""
        escaped = dialect.escape_string("test'string")
        assert "test''string" == escaped

    def test_build_where_clause(self, dialect: DatabaseDialect) -> None:
        """Testing WHERE clause construction."""
        # Empty or null condition.  Alternatively, a condition that always evaluates to false.
        where_clause = dialect.build_where_clause([])
        assert where_clause == ""

        # Single condition.
        where_clause = dialect.build_where_clause(["id > 100"])
        assert where_clause == "WHERE id > 100"

        # Multiple conditions.
        where_clause = dialect.build_where_clause(["id > 100", "name IS NOT NULL"])
        assert where_clause == "WHERE id > 100 AND name IS NOT NULL"

    def test_build_full_table_name(self, dialect: DatabaseDialect) -> None:
        """Testing the construction of fully qualified table names."""
        full_name = dialect.build_full_table_name("test_db", "test_table")

        # Verifies the inclusion of the database and table names.
        if not isinstance(
            dialect, (PostgreSQLDialect, SQLiteDialect)
        ):  # PostgreSQL does not support database name in table name
            assert "test_db" in full_name
        assert "test_table" in full_name

        # Verify that the correct quotation marks are used.
        if isinstance(dialect, MySQLDialect):
            assert "`test_db`.`test_table`" == full_name
        elif isinstance(dialect, PostgreSQLDialect):
            assert '"test_table"' == full_name
        elif isinstance(dialect, SQLiteDialect):
            assert '"test_table"' == full_name
        elif isinstance(dialect, SQLServerDialect):
            assert "[test_db].[test_table]" == full_name


class TestDialectSQLGeneration:
    """Testing SQL dialect generation."""

    def test_cross_dialect_table_exists(self) -> None:
        """Verify table existence checks across different dialects."""
        dialects = {
            "mysql": get_dialect("mysql"),
            "postgresql": get_dialect("postgresql"),
            "sqlite": get_dialect("sqlite"),
            "sqlserver": get_dialect("sqlserver"),
        }

        for db_type, dialect in dialects.items():
            sql, params = dialect.get_table_exists_sql("test_db", "test_table")

            # Verify that the SQL query is not empty.
            assert sql.strip() != ""

            # Validates the table name parameter.
            if db_type == "sqlite":
                assert "table" in params
            elif db_type == "sqlserver":
                assert "schema" in params and "table" in params
            else:
                assert "database" in params and "table" in params

    def test_cross_dialect_count_sql(self) -> None:
        """Testing count SQL queries across different dialects."""
        dialects = {
            "mysql": get_dialect("mysql"),
            "postgresql": get_dialect("postgresql"),
            "sqlite": get_dialect("sqlite"),
            "sqlserver": get_dialect("sqlserver"),
        }

        for db_type, dialect in dialects.items():
            sql = dialect.get_count_sql("test_db", "test_table")

            # Verify the inclusion of `COUNT(*)`.
            assert "COUNT(*)" in sql.upper()

            # Verifies the inclusion of the table name.
            assert "test_table" in sql

    def test_cross_dialect_limit_sql(self) -> None:
        """Testing cross-dialect SQL limitations."""
        base_sql = "SELECT * FROM test_table"
        dialects = {
            "mysql": get_dialect("mysql"),
            "postgresql": get_dialect("postgresql"),
            "sqlite": get_dialect("sqlite"),
            "sqlserver": get_dialect("sqlserver"),
        }

        for db_type, dialect in dialects.items():
            limited_sql = dialect.get_limit_sql(base_sql, 10, 5)

            # Validates the inclusion of raw SQL.
            assert "SELECT * FROM test_table" in limited_sql

            # Validates the input against a restricted syntax.
            if db_type == "mysql":
                assert "LIMIT 5, 10" in limited_sql
            elif db_type in ["postgresql", "sqlite"]:
                assert "LIMIT 10 OFFSET 5" in limited_sql
            elif db_type == "sqlserver":
                assert "OFFSET 5 ROWS FETCH NEXT 10 ROWS ONLY" in limited_sql


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
