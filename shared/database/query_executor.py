"""
Query executor module

Encapsulates SQL query execution logic, provides a unified interface, supports:
1. Execute SQL queries and return results
2. Table and column metadata query and validation
3. Database dialect system integration
"""

import json
import time
from logging import Logger
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from shared.database.database_dialect import DatabaseDialect, get_dialect
from shared.exceptions import DatabaseExceptionConverter, DataQualityException
from shared.utils.logger import get_logger

T = TypeVar("T")


class TransactionError(Exception):
    """Transaction related error"""

    pass


class QueryExecutor:
    """
    Asynchronous query executor class

    Only supports AsyncEngine/AsyncConnection, no longer supports synchronous objects.
    Integrates database dialect system, provides cross-database compatible
    query functions.
    """

    ISOLATION_LEVELS = {
        "READ UNCOMMITTED",
        "READ COMMITTED",
        "REPEATABLE READ",
        "SERIALIZABLE",
    }

    def __init__(
        self,
        engine_or_conn: Union[AsyncEngine, AsyncConnection],
        logger: Optional[Logger] = None,
        dialect: Optional[DatabaseDialect] = None,
    ):
        """
        Initialize query executor

        Args:
            engine_or_conn: SQLAlchemy engine or connection object
            logger: Logger instance
            dialect: Database dialect instance, if not provided will be auto-detected
        """
        self.engine_or_conn = engine_or_conn
        self.logger = logger or get_logger(__name__)
        self._savepoint_id = 0

        # Initialize database dialect
        if dialect:
            self.dialect = dialect
        else:
            # Auto-detect database type
            self.dialect = self._detect_dialect()

        # Get connection ID for exception context
        self.connection_id = self._get_connection_id()

    def _detect_dialect(self) -> DatabaseDialect:
        """
        Automatically detect database type and return corresponding dialect

        Returns:
            DatabaseDialect: Database dialect instance
        """
        try:
            if isinstance(self.engine_or_conn, AsyncEngine):
                db_url = str(self.engine_or_conn.url)
            else:
                # AsyncConnection - get engine URL
                db_url = str(self.engine_or_conn.engine.url)

            if "mysql" in db_url.lower():
                return get_dialect("mysql")
            elif "postgresql" in db_url.lower():
                return get_dialect("postgresql")
            elif "sqlite" in db_url.lower():
                return get_dialect("sqlite")
            else:
                # Default to MySQL dialect
                self.logger.warning(
                    f"Unknown database type in URL: {db_url}, using MySQL dialect "
                    "as default"
                )
                return get_dialect("mysql")

        except Exception as e:
            self.logger.warning(
                f"Failed to detect database dialect: {str(e)}, using MySQL dialect "
                "as default"
            )
            return get_dialect("mysql")

    def _get_connection_id(self) -> str:
        """Get connection ID for exception context"""
        try:
            if isinstance(self.engine_or_conn, AsyncEngine):
                db_url = str(self.engine_or_conn.url)
                return f"engine:{db_url.split('@')[-1] if '@' in db_url else db_url}"
            else:
                # AsyncConnection
                db_url = str(self.engine_or_conn.engine.url)
                return (
                    f"connection:{db_url.split('@')[-1] if '@' in db_url else db_url}"
                )
        except Exception:
            return "unknown_connection"

    def _handle_database_exception(
        self,
        error: Exception,
        operation: str,
        rule_id: Optional[str] = None,
        entity_name: Optional[str] = None,
        sql: Optional[str] = None,
    ) -> DataQualityException:
        """
        Handle database exception, convert to data quality exception

        Args:
            error: Original exception
            operation: Operation type
            rule_id: Rule ID (optional)
            entity_name: Entity name (optional)
            sql: SQL statement (optional)

        Returns:
            DataQualityException: Converted data quality exception
        """
        context = {"operation": operation, "connection_id": self.connection_id}

        if rule_id:
            context["rule_id"] = rule_id
        if entity_name:
            context["entity_name"] = entity_name
        if sql:
            context["sql"] = sql[:500]  # Limit SQL length

        # More precise error classification
        error_msg = str(error).lower()

        # Permission error
        if any(
            keyword in error_msg
            for keyword in [
                "permission denied",
                "access denied",
                "unauthorized",
                "forbidden",
            ]
        ):
            from shared.exceptions import OperationError

            return OperationError(
                message=f"Permission denied: {str(error)[:200]}",
                rule_id=rule_id,
                operation=operation,
                sql=sql,
                context=context,
                cause=error,
            )

        # Timeout error
        elif any(
            keyword in error_msg
            for keyword in ["timeout", "timed out", "connection timeout"]
        ):
            from shared.exceptions import EngineError

            return EngineError(
                message=f"Database timeout: {str(error)[:200]}",
                connection_id=self.connection_id,
                operation=operation,
                context=context,
                cause=error,
            )

        # Connection error
        elif any(
            keyword in error_msg
            for keyword in ["connection", "host", "authentication", "network"]
        ):
            from shared.exceptions import EngineError

            return EngineError(
                message=f"Database connection error: {str(error)[:200]}",
                connection_id=self.connection_id,
                operation=operation,
                context=context,
                cause=error,
            )

        # Use generic converter for other exceptions
        else:
            return DatabaseExceptionConverter.convert_sqlalchemy_error(error, context)

    def _preprocess_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-process query parameters for engine specific quirks.

        Currently handles:
        * PostgreSQL JSON/JSONB encoding – encode Python dict into JSON strings
        * PostgreSQL arrays – keep Python lists as-is for array columns

        Args:
            params: Original parameters passed by caller.

        Returns:
            Dict[str, Any]: Transformed parameters suitable for SQLAlchemy execution.
        """
        if not params:
            return params

        try:
            # local import to avoid cycles
            from shared.database.database_dialect import DatabaseType

            if self.dialect.database_type == DatabaseType.POSTGRESQL:
                transformed: Dict[str, Any] = {}
                for key, value in params.items():
                    # Only convert dicts to JSON strings for JSONB columns
                    # Keep lists as-is for PostgreSQL arrays
                    if isinstance(value, dict):
                        transformed[key] = json.dumps(value)
                    else:
                        transformed[key] = value
                return transformed
        except Exception as exc:
            # If preprocessing fails, log and fall back to original params
            self.logger.debug(f"Parameter preprocessing failed: {exc}")
        return params

    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        fetch: bool = True,
        sample_limit: Optional[int] = None,
        rule_id: Optional[str] = None,
        entity_name: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """
        Execute SQL query and return results

        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results
            sample_limit: Result limit count

        Returns:
            Tuple[List[Dict], Optional[int]]: Result list and affected row count

        Raises:
            SQLAlchemyError: If query execution fails
        """
        start_time = time.time()
        self.logger.debug(f"Executing query: {query}")

        if sample_limit:
            # Add LIMIT clause if not present
            if "LIMIT" not in query.upper():
                query = f"{query} LIMIT {sample_limit}"

        try:
            prepared_params = self._preprocess_params(params or {})
            result: Any
            if isinstance(self.engine_or_conn, AsyncEngine):
                async with self.engine_or_conn.connect() as conn:
                    result = await conn.execute(text(query), prepared_params)
                    is_select = query.strip().upper().startswith("SELECT")
                    affected_rows = None if is_select else result.rowcount
                    if fetch and result.returns_rows:
                        columns = result.keys()
                        rows = result.fetchall()
                        results = [dict(zip(columns, row)) for row in rows]
                    else:
                        results = []
                    # Explicitly commit any data-modifying or DDL statements so that
                    # subsequent operations executed on new connections can observe
                    # the changes (required because the implicit transaction would
                    # otherwise be rolled back when the connection context closes).
                    if not is_select:
                        try:
                            await conn.commit()
                        except AttributeError:
                            # Some dialects/autocommit modes may not expose commit
                            # (e.g. SQLite memory db)
                            # In those cases, rely on autocommit behaviour.
                            self.logger.debug(
                                "Connection object has no commit method – relying on "
                                "autocommit."
                            )
            elif isinstance(self.engine_or_conn, AsyncConnection):
                result = await self.engine_or_conn.execute(text(query), prepared_params)
                is_select = query.strip().upper().startswith("SELECT")
                affected_rows = None if is_select else result.rowcount
                if fetch and result.returns_rows:
                    columns = result.keys()
                    rows = result.fetchall()
                    results = [dict(zip(columns, row)) for row in rows]
                else:
                    results = []
            else:
                raise NotImplementedError(
                    "QueryExecutor only supports AsyncEngine or AsyncConnection"
                )
            execution_time = time.time() - start_time
            self.logger.debug(
                f"Query executed in {execution_time:.3f}s, "
                f"returned {len(results)} rows"
            )
            return results, affected_rows
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(
                f"Query execution failed in {execution_time:.3f}s: {str(e)}\n"
                f"Query: {query}\n"
                f"Params: {params}"
            )
            # Use new exception converter
            converted_exception = self._handle_database_exception(
                error=e,
                operation="query_execution",
                rule_id=rule_id,
                entity_name=entity_name,
                sql=query,
            )
            raise converted_exception

    # Metadata query methods

    async def execute_batch_insert(
        self,
        table_name: str,
        data_list: List[Dict[str, Any]],
        batch_size: int = 1000,
        use_transaction: bool = True,
    ) -> int:
        """
        Execute batch INSERT operations for improved performance

        Args:
            table_name: Table name
            data_list: List of data dictionaries to insert
            batch_size: Number of records to insert per batch
            use_transaction: Whether to wrap operations in a transaction

        Returns:
            int: Total number of inserted records

        Raises:
            ValueError: If data_list is empty or has inconsistent keys
            SQLAlchemyError: If insertion fails
        """
        if not data_list:
            return 0

        # Validate that all records have the same keys
        first_keys = set(data_list[0].keys())
        for i, record in enumerate(data_list):
            if set(record.keys()) != first_keys:
                raise ValueError(f"Record {i} has different keys than the first record")

        start_time = time.time()
        total_inserted = 0

        self.logger.debug(
            f"Starting batch insert of {len(data_list)} records into {table_name}"
        )

        try:
            if isinstance(self.engine_or_conn, AsyncEngine):
                if use_transaction:
                    async with self.engine_or_conn.connect() as conn:
                        async with conn.begin():
                            total_inserted = await self._execute_batch_insert_impl(
                                conn, table_name, data_list, batch_size
                            )
                else:
                    async with self.engine_or_conn.connect() as conn:
                        total_inserted = await self._execute_batch_insert_impl(
                            conn, table_name, data_list, batch_size
                        )
            elif isinstance(self.engine_or_conn, AsyncConnection):
                if use_transaction:
                    async with self.engine_or_conn.begin():
                        total_inserted = await self._execute_batch_insert_impl(
                            self.engine_or_conn, table_name, data_list, batch_size
                        )
                else:
                    total_inserted = await self._execute_batch_insert_impl(
                        self.engine_or_conn, table_name, data_list, batch_size
                    )
            else:
                raise NotImplementedError(
                    "QueryExecutor only supports AsyncEngine or AsyncConnection"
                )

            execution_time = time.time() - start_time
            records_per_sec = (
                total_inserted / execution_time if execution_time > 0 else 0
            )
            self.logger.info(
                f"Batch insert completed: {total_inserted} records in "
                f"{execution_time:.3f}s "
                f"({records_per_sec:.1f} records/sec)"
            )
            return total_inserted

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(
                f"Batch insert failed after {execution_time:.3f}s: {str(e)}"
            )
            raise

    async def _execute_batch_insert_impl(
        self,
        conn: AsyncConnection,
        table_name: str,
        data_list: List[Dict[str, Any]],
        batch_size: int,
        rule_id: Optional[str] = None,
        entity_name: Optional[str] = None,
    ) -> int:
        """
        Internal implementation of batch insert

        Args:
            conn: Database connection
            table_name: Table name
            data_list: List of data dictionaries
            batch_size: Batch size

        Returns:
            int: Number of inserted records
        """
        if not data_list:
            return 0

        # Get column names from first record
        columns = list(data_list[0].keys())
        columns_str = ", ".join(columns)
        placeholders = ", ".join([f":{col}" for col in columns])

        query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"

        total_inserted = 0
        start_time = time.time()
        # Process in batches
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i : i + batch_size]

            try:
                # Use executemany for efficient batch insertion
                result: Any = await conn.execute(text(query), batch)
                batch_inserted = result.rowcount if result.rowcount else len(batch)
                total_inserted += batch_inserted
                execution_time = time.time() - start_time
                self.logger.debug(
                    f"Inserted batch {i//batch_size + 1}: {batch_inserted} records "
                    f"in {execution_time:.3f}s"
                )

            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(
                    f"Failed to insert batch starting at record {i}: {str(e)} "
                    f"after {execution_time:.3f}s"
                )
                # Use new exception converter
                converted_exception = self._handle_database_exception(
                    error=e,
                    operation="batch_insert",
                    rule_id=rule_id,
                    entity_name=entity_name,
                    sql=query,
                )
                raise converted_exception

        return total_inserted

    async def execute_bulk_insert_values(
        self, table_name: str, data_list: List[Dict[str, Any]], batch_size: int = 1000
    ) -> int:
        """
        Execute bulk INSERT using VALUES clause for maximum performance

        This method constructs large INSERT statements with multiple VALUES clauses
        which can be significantly faster than individual INSERT statements.

        Args:
            table_name: Table name
            data_list: List of data dictionaries to insert
            batch_size: Number of records per INSERT statement

        Returns:
            int: Total number of inserted records

        Raises:
            ValueError: If data_list is empty or has inconsistent keys
            SQLAlchemyError: If insertion fails
        """
        if not data_list:
            return 0

        # Validate that all records have the same keys
        first_keys = set(data_list[0].keys())
        for i, record in enumerate(data_list):
            if set(record.keys()) != first_keys:
                raise ValueError(f"Record {i} has different keys than the first record")

        start_time = time.time()
        total_inserted = 0

        self.logger.debug(
            f"Starting bulk VALUES insert of {len(data_list)} records into {table_name}"
        )

        columns = list(data_list[0].keys())
        columns_str = ", ".join(columns)

        try:
            if isinstance(self.engine_or_conn, AsyncEngine):
                async with self.engine_or_conn.connect() as conn:
                    async with conn.begin():
                        total_inserted = await self._execute_bulk_values_impl(
                            conn,
                            table_name,
                            columns_str,
                            columns,
                            data_list,
                            batch_size,
                        )
            elif isinstance(self.engine_or_conn, AsyncConnection):
                async with self.engine_or_conn.begin():
                    total_inserted = await self._execute_bulk_values_impl(
                        self.engine_or_conn,
                        table_name,
                        columns_str,
                        columns,
                        data_list,
                        batch_size,
                    )
            else:
                raise NotImplementedError(
                    "QueryExecutor only supports AsyncEngine or AsyncConnection"
                )

            execution_time = time.time() - start_time
            records_per_sec = (
                total_inserted / execution_time if execution_time > 0 else 0
            )
            self.logger.info(
                f"Bulk VALUES insert completed: {total_inserted} records in "
                f"{execution_time:.3f}s "
                f"({records_per_sec:.1f} records/sec)"
            )
            return total_inserted

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(
                f"Bulk VALUES insert failed after {execution_time:.3f}s: {str(e)}"
            )
            raise

    async def _execute_bulk_values_impl(
        self,
        conn: AsyncConnection,
        table_name: str,
        columns_str: str,
        columns: List[str],
        data_list: List[Dict[str, Any]],
        batch_size: int,
        rule_id: Optional[str] = None,
        entity_name: Optional[str] = None,
    ) -> int:
        """
        Internal implementation of bulk VALUES insert

        Args:
            conn: Database connection
            table_name: Table name
            columns_str: Comma-separated column names
            columns: List of column names
            data_list: List of data dictionaries
            batch_size: Batch size

        Returns:
            int: Number of inserted records
        """
        total_inserted = 0
        start_time = time.time()
        # Process in batches
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i : i + batch_size]

            # Build VALUES clause with parameters
            values_clauses = []
            params = {}

            for j, record in enumerate(batch):
                param_names = []
                for col in columns:
                    param_name = f"{col}_{i}_{j}"
                    param_names.append(f":{param_name}")
                    params[param_name] = record[col]

                values_clauses.append(f"({', '.join(param_names)})")

            query = (
                f"INSERT INTO {table_name} ({columns_str}) VALUES "
                f"{', '.join(values_clauses)}"
            )

            try:
                result: Any = await conn.execute(text(query), params)
                batch_inserted = result.rowcount if result.rowcount else len(batch)
                total_inserted += batch_inserted
                execution_time = time.time() - start_time
                self.logger.debug(
                    f"Inserted VALUES batch {i//batch_size + 1}: {batch_inserted} "
                    f"records in {execution_time:.3f}s"
                )

            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(
                    f"Failed to insert VALUES batch starting at record {i}: "
                    f"{str(e)} after {execution_time:.3f}s"
                )
                # Use new exception converter
                converted_exception = self._handle_database_exception(
                    error=e,
                    operation="bulk_insert_values",
                    rule_id=rule_id,
                    entity_name=entity_name,
                    sql=query,
                )
                raise converted_exception

        return total_inserted

    async def get_database_list(self) -> List[str]:
        """
        Get list of databases using dialect-specific queries

        Returns:
            List[str]: List of database names

        Raises:
            SQLAlchemyError: If query execution fails
        """
        try:
            # Use dialect system to get database list
            sql, params = self.dialect.get_database_list_sql()
            results, _ = await self.execute_query(sql, params)

            # Parse results based on dialect type
            if not results:
                return []

            # Get first column value as database name
            first_key = next(iter(results[0].keys()))
            return [row[first_key] for row in results]

        except Exception as e:
            self.logger.error(f"Failed to get database list: {str(e)}")
            raise e

    async def get_table_list(
        self, database: Optional[str] = None, schema: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of tables using dialect-specific queries

        Args:
            database: Database name (optional for some database types)
            schema: Schema name (optional)

        Returns:
            List[Dict[str, Any]]: List of tables with metadata

        Raises:
            SQLAlchemyError: If query execution fails
        """
        try:
            # Use dialect system to get table list
            sql, params = self.dialect.get_table_list_sql(database or "main", schema)
            results, _ = await self.execute_query(sql, params)

            if not results:
                return []

            # Standardize result format
            tables = []
            for row in results:
                table_info = {
                    "name": row.get("table_name", row.get("name", "")),
                    "type": (
                        "view"
                        if (
                            row.get("table_type", "").upper() == "VIEW"
                            or row.get("type", "") == "view"
                        )
                        else "table"
                    ),
                    "schema": row.get("table_schema", schema),
                    "database": row.get("table_catalog", database),
                }
                tables.append(table_info)

            return tables

        except Exception as e:
            self.logger.error(f"Failed to get table list: {str(e)}")
            raise e

    async def get_column_list(
        self,
        table_name: str,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        entity_name: Optional[str] = None,
        resource_type: Optional[str] = None,
        rule_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get column information of a table

        Args:
            table_name: Table name
            database: Database name
            schema: Schema name
            entity_name: Entity name (for error context)
            resource_type: Resource type (for error context)
            rule_id: Rule ID (for error context)

        Returns:
            List[Dict[str, Any]]: List of column information, each dict contains
            standardized detailed info.
        """
        try:
            # Use dialect to get column info SQL
            query, params = self.dialect.get_column_list_sql(
                table_name, database or "main", schema
            )

            # Execute query
            result, _ = await self.execute_query(
                query=query,
                params=params,
                fetch=True,
                rule_id=rule_id,
                entity_name=(
                    entity_name or f"{database}.{table_name}"
                    if database
                    else table_name
                ),
            )

            # Standardize result format
            standardized_result = []
            for col in result:
                # Different database dialects may use different key names
                name = col.get("Field") or col.get("name") or col.get("column_name")
                type_ = col.get("Type") or col.get("data_type") or col.get("type")

                if not name:
                    # If column name not found, skip this column with a warning
                    self.logger.warning(
                        f"Could not determine column name from result: {col}"
                    )
                    continue

                if not type_:
                    type_ = "unknown"

                # Create standardized column info
                std_col = {
                    "name": name,
                    "type": type_,
                    "nullable": (
                        col.get("Null", col.get("is_nullable", "YES")).upper() == "YES"
                    ),
                    "key": col.get("Key", col.get("key", "")),
                    "default": col.get(
                        "Default", col.get("column_default", col.get("default"))
                    ),
                    "extra": col.get("Extra", col.get("extra", "")),
                    # Keep original data for future needs
                    "original": col,
                }
                standardized_result.append(std_col)

            return standardized_result

        except Exception as e:
            # Build detailed error context
            error_context = {
                "operation": "get_column_list",
                "entity_name": (
                    entity_name or f"{database}.{table_name}"
                    if database
                    else table_name
                ),
                "resource_type": resource_type or "table",
                "connection_id": self.connection_id,
                "table_name": table_name,
                "database": database,
                "schema": schema,
            }

            if rule_id:
                error_context["rule_id"] = rule_id

            # If already a data quality exception, raise directly
            if isinstance(e, DataQualityException):
                raise e
            else:
                # Convert to data quality exception
                converted_error = self._handle_database_exception(
                    e, "get_column_list", rule_id, entity_name
                )
                raise converted_error

    async def table_exists(
        self,
        table_name: str,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        entity_name: Optional[str] = None,
        resource_type: Optional[str] = None,
        rule_id: Optional[str] = None,
    ) -> bool:
        """
        Check if table exists

        Args:
            table_name: Table name
            database: Database name
            schema: Schema name
            entity_name: Entity name (for error context)
            resource_type: Resource type (for error context)
            rule_id: Rule ID (for error context)

        Returns:
            bool: Whether the table exists
        """
        try:
            # Use dialect to get SQL for checking table existence
            query, params = self.dialect.get_table_exists_sql(
                database or "main", table_name
            )

            # Execute query
            result, _ = await self.execute_query(
                query=query,
                params=params,
                fetch=True,
                rule_id=rule_id,
                entity_name=(
                    entity_name or f"{database}.{table_name}"
                    if database
                    else table_name
                ),
            )

            # Parse results based on dialect type
            if self.dialect.database_type.value == "mysql":
                return len(result) > 0
            elif self.dialect.database_type.value == "postgresql":
                return len(result) > 0
            else:  # SQLite
                return len(result) > 0

        except Exception as e:
            # Build detailed error context
            error_context = {
                "operation": "table_existence_check",
                "entity_name": (
                    entity_name or f"{database}.{table_name}"
                    if database
                    else table_name
                ),
                "resource_type": resource_type or "table",
                "connection_id": self.connection_id,
                "table_name": table_name,
                "database": database,
                "schema": schema,
            }

            if rule_id:
                error_context["rule_id"] = rule_id

            # If already a data quality exception, raise directly
            if isinstance(e, DataQualityException):
                raise e
            else:
                # Convert to data quality exception
                converted_error = self._handle_database_exception(
                    e, "table_existence_check", rule_id, entity_name
                )
                raise converted_error

    async def column_exists(
        self,
        table_name: str,
        column_name: str,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        entity_name: Optional[str] = None,
        resource_type: Optional[str] = None,
        rule_id: Optional[str] = None,
    ) -> bool:
        """
        Check if column exists

        Args:
            table_name: Table name
            column_name: Column name
            database: Database name
            schema: Schema name
            entity_name: Entity name (for error context)
            resource_type: Resource type (for error context)
            rule_id: Rule ID (for error context)

        Returns:
            bool: Whether the column exists
        """
        try:
            # First check if table exists
            if not await self.table_exists(
                table_name, database, schema, entity_name, resource_type, rule_id
            ):
                return False

            # Get column list and check if target column exists
            columns = await self.get_column_list(
                table_name, database, schema, entity_name, resource_type, rule_id
            )
            # Use standardized 'name' field
            column_names = {col["name"].lower() for col in columns}

            return column_name.lower() in column_names

        except Exception as e:
            # Build detailed error context
            error_context = {
                "operation": "column_existence_check",
                "entity_name": (
                    entity_name or f"{database}.{table_name}.{column_name}"
                    if database
                    else f"{table_name}.{column_name}"
                ),
                "resource_type": resource_type or "column",
                "connection_id": self.connection_id,
                "table_name": table_name,
                "column_name": column_name,
                "database": database,
                "schema": schema,
            }

            if rule_id:
                error_context["rule_id"] = rule_id

            # If already a data quality exception, raise directly
            if isinstance(e, DataQualityException):
                raise e
            else:
                # Convert to data quality exception
                converted_error = self._handle_database_exception(
                    e, "column_existence_check", rule_id, entity_name
                )
                raise converted_error

    def get_dialect_info(self) -> Dict[str, Any]:
        """
        Get information about the current database dialect

        Returns:
            Dict[str, Any]: Dialect information
        """
        return {
            "dialect_name": self.dialect.__class__.__name__,
            "database_type": getattr(self.dialect, "database_type", "unknown"),
            "supports_schemas": getattr(self.dialect, "supports_schemas", False),
            "quote_character": getattr(self.dialect, "quote_character", '"'),
            "regex_operator": self.dialect.get_regex_operator(),
            "length_function": self.dialect.get_length_function(),
        }
