"""
Base executor - unified rule execution logic entry point

Based on the mature validation logic of the existing Rule model,
provides a unified execution interface
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncEngine

from shared.database.connection import get_db_url, get_engine
from shared.database.database_dialect import DatabaseDialect, get_dialect
from shared.exceptions import EngineError, RuleExecutionError
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.result_schema import ExecutionResultSchema
from shared.schema.rule_schema import RuleSchema
from shared.utils.logger import get_logger


class BaseExecutor(ABC):
    """
    Base class for rule executors

    Design principles:
    1. Unified execution interface
    2. Based on mature validation logic
    3. Single responsibility, each executor is responsible for a specific rule type
    4. Reuse existing connection management functionality
    """

    def __init__(
        self,
        connection: ConnectionSchema,
        test_mode: Optional[bool] = False,
        sample_data_enabled: Optional[bool] = None,
        sample_data_max_records: Optional[int] = None,
    ) -> None:
        """Initialize BaseExecutor"""
        self.connection = connection
        self.test_mode = test_mode  # Test mode flag
        self.dialect = self._get_dialect()

        # Read sampling settings from global config, use default values
        # if not specified
        if sample_data_enabled is None or sample_data_max_records is None:
            try:
                from core.config import get_core_config

                core_config = get_core_config()
                self.sample_data_enabled = (
                    sample_data_enabled
                    if sample_data_enabled is not None
                    else core_config.sample_data_enabled
                )
                self.sample_data_max_records = (
                    sample_data_max_records
                    if sample_data_max_records is not None
                    else core_config.sample_data_max_records
                )
            except Exception:
                # If config loading fails, use default values
                self.sample_data_enabled = (
                    sample_data_enabled if sample_data_enabled is not None else True
                )
                self.sample_data_max_records = (
                    sample_data_max_records
                    if sample_data_max_records is not None
                    else 5
                )
        else:
            self.sample_data_enabled = sample_data_enabled
            self.sample_data_max_records = sample_data_max_records

        self.logger = get_logger(__name__)

    async def get_engine(self) -> AsyncEngine:
        """
        Get AsyncEngine - reuse existing connection management functionality

        Use get_engine function from shared.database.connection
        Engine-level error handling: connection failure will raise EngineError
        """
        try:
            # Build database URL
            db_url = get_db_url(
                db_type=self.connection.connection_type.value,
                host=self.connection.host,
                port=self.connection.port,
                database=self.connection.db_name,
                username=self.connection.username,
                password=self.connection.password,
                file_path=self.connection.file_path,
            )

            # Use existing get_engine function, already includes connection pool
            # management
            return await get_engine(
                db_url=db_url,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600,
                echo=False,
            )
        except Exception as e:
            # Connection failure is an engine-level error, raise EngineError
            error_msg = f"Failed to connect to database: {str(e)}"
            self.logger.error(error_msg)
            raise EngineError(
                error_msg,
                connection_id=(
                    self.connection.name
                    if hasattr(self.connection, "name")
                    else str(self.connection.host)
                ),
            )

    @abstractmethod
    def supports_rule_type(self, rule_type: str) -> bool:
        """Check if the specified rule type is supported"""
        pass

    @abstractmethod
    async def execute_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """Execute a single rule"""
        pass

    async def execute_rules(
        self, rules: List[RuleSchema]
    ) -> List[ExecutionResultSchema]:
        """
        Execute multiple rules - batch optimization

        For multiple rules with the same connection, reuse the database connection
        """
        results = []
        for rule in rules:
            if self.supports_rule_type(rule.type.value):
                result = await self.execute_rule(rule)
                results.append(result)
        return results

    def _build_table_name(self, rule: RuleSchema) -> str:
        """Build complete table name"""
        full_table_name = rule.get_full_table_name()
        return full_table_name

    def _build_where_clause(
        self, condition: str, filter_condition: Optional[str] = None
    ) -> str:
        """Build WHERE clause"""
        if filter_condition:
            return f"WHERE ({condition}) AND ({filter_condition})"
        return f"WHERE {condition}"

    def _safe_get_table_name(self, rule: RuleSchema) -> str:
        """Safely get table name

        - optimized error handling + SQL injection protection
        """
        target_info = rule.get_target_info()
        table_name = target_info.get("table", "")

        if not table_name:
            raise RuleExecutionError(f"Rule {rule.name} missing table name")

        # SQL injection protection: check for dangerous characters
        # when not in test mode
        if not self.test_mode and self._contains_sql_injection_patterns(table_name):
            raise RuleExecutionError(
                f"Table name contains potentially dangerous SQL patterns: "
                f"{table_name}"
            )

        # Basic SQL identifier validation (relaxed in test mode)
        if not self.test_mode and not self._is_valid_sql_identifier(table_name):
            raise RuleExecutionError(f"Invalid table name format: {table_name}")

        return table_name

    def _safe_get_column_name(self, rule: RuleSchema) -> str:
        """Safely get column name

        - optimized error handling + SQL injection protection
        """
        target_info = rule.get_target_info()
        column_name = target_info.get("column", "")
        if not column_name:
            raise RuleExecutionError(f"Rule {rule.name} missing column name")

        # SQL injection protection: check for dangerous characters
        # when not in test mode
        if not self.test_mode and self._contains_sql_injection_patterns(column_name):
            raise RuleExecutionError(
                f"Column name contains potentially dangerous SQL patterns: "
                f"{column_name}"
            )

        # Basic SQL identifier validation (relaxed in test mode)
        if not self.test_mode and not self._is_valid_sql_identifier(column_name):
            raise RuleExecutionError(f"Invalid column name format: {column_name}")

        return column_name

    def _contains_sql_injection_patterns(self, value: str) -> bool:
        """Check if the string contains SQL injection patterns"""
        import re

        # Dangerous SQL patterns
        dangerous_patterns = [
            r";\s*DROP\s+TABLE",
            r";\s*DELETE\s+FROM",
            r";\s*INSERT\s+INTO",
            r";\s*UPDATE\s+",
            r";\s*CREATE\s+",
            r";\s*ALTER\s+",
            r"--",  # SQL comment
            r"/\*.*\*/",  # Multi-line comment
            r"UNION\s+SELECT",
            r"OR\s+1\s*=\s*1",
            r"'\s*OR\s+'",
        ]

        value_upper = value.upper()
        for pattern in dangerous_patterns:
            if re.search(pattern, value_upper, re.IGNORECASE):
                return True

        return False

    async def _handle_execution_error(
        self, error: Exception, rule: RuleSchema, start_time: float, table_name: str
    ) -> ExecutionResultSchema:
        """
        Unified error handling method

        Decide whether to raise an exception or return an error result
        based on error type:
        - Engine-level error: log and re-raise exception
        - Rule-level error: return error result
        """
        execution_time = time.time() - start_time

        if isinstance(error, EngineError):
            # Engine-level error: log and re-raise
            self.logger.error(f"Engine error in rule {rule.id}: {str(error)}")
            raise error
        else:
            # Rule-level error: return error result
            self.logger.warning(f"Rule error in rule {rule.id}: {str(error)}")
            return ExecutionResultSchema.create_error_result(
                rule_id=rule.id,
                entity_name=table_name,
                error_message=str(error),
                execution_time=execution_time,
            )

    def _is_valid_sql_identifier(self, identifier: str) -> bool:
        """Validate whether SQL identifier format is valid

        - supports Unicode characters
        """
        import re

        # SQL identifier rules: support Unicode characters
        # 1. Standard identifier: starts with a letter (including Unicode), followed by
        #    letters, numbers, underscores
        # 2. MySQL backtick identifier
        # 3. Double-quote identifier (standard SQL)
        # 4. Square bracket identifier (SQL Server)
        patterns = [
            r"^[\w\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-"
            r"\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\u3300-\u33ff\ufe30-\ufe4f\uf900-"
            r"\ufaff\u2f800-\u2fa1f][\w\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df"
            r"\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\u3300-"
            r"\u33ff\ufe30-\ufe4f\uf900-\ufaff\u2f800-\u2fa1f\d]*$",  # Unicode
            r"^[a-zA-Z_][a-zA-Z0-9_]*$",  # Standard ASCII identifier
            r"^`[^`]+`$",  # MySQL backtick identifier
            r'^"[^"]+"$',  # Double-quote identifier
            r"^\[[^\]]+\]$",  # SQL Server square bracket identifier
        ]

        for pattern in patterns:
            if re.match(pattern, identifier):
                return True

        return False

    async def _generate_sample_data(
        self, rule: RuleSchema, count_sql: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate sample data

        Convert COUNT query to sample query to get some failed records as samples

        Args:
            rule: Rule object
            count_sql: Original COUNT query SQL

        Returns:
            List of sample data, or None if no failed records
        """
        from shared.database.query_executor import QueryExecutor

        # Check sample data switch
        if not self.sample_data_enabled:
            return None

        try:
            # Convert COUNT query to sample query
            # Support different field names: failed_count, anomaly_count, etc.
            sample_sql = count_sql
            if "SELECT COUNT(*) AS failed_count" in count_sql:
                sample_sql = count_sql.replace(
                    "SELECT COUNT(*) AS failed_count", "SELECT *"
                )
            elif "SELECT COUNT(*) AS anomaly_count" in count_sql:
                sample_sql = count_sql.replace(
                    "SELECT COUNT(*) AS anomaly_count", "SELECT *"
                )
            else:
                # General replacement: match any COUNT(*) AS xxx pattern
                import re

                sample_sql = re.sub(r"SELECT COUNT\(\*\) AS \w+", "SELECT *", count_sql)

            # Add LIMIT clause
            sample_sql += f" LIMIT {self.sample_data_max_records}"

            # Execute sample query
            engine = await self.get_engine()
            query_executor = QueryExecutor(engine)
            sample_result, _ = await query_executor.execute_query(sample_sql)

            # Return sample data
            return sample_result if sample_result else None

        except Exception as e:
            # Log warning if sampling fails but do not affect main flow
            self.logger.warning(
                f"Failed to generate sample data for rule {rule.id}: {str(e)}"
            )
            return None

    def _get_dialect(self) -> DatabaseDialect:
        """Get database dialect"""
        return get_dialect(self.connection.connection_type)
