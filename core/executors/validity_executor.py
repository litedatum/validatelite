"""
Validity rule executor - based on mature existing logic

Ported from mature validation logic in app/models/rule.py
Unified handling: RANGE, ENUM, REGEX and similar rules
"""

from datetime import datetime
from typing import Any, Dict, Optional

from shared.database.query_executor import QueryExecutor
from shared.enums.rule_types import RuleType
from shared.exceptions.exception_system import RuleExecutionError
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.result_schema import ExecutionResultSchema
from shared.schema.rule_schema import RuleSchema

from .base_executor import BaseExecutor


class ValidityExecutor(BaseExecutor):
    """
    Validity rule executor

    Based on mature logic in app.models.rule.Rule
    Unified handling: RANGE, ENUM, REGEX and similar rules
    """

    SUPPORTED_TYPES = [
        RuleType.RANGE,
        RuleType.ENUM,
        RuleType.REGEX,
        RuleType.DATE_FORMAT,
    ]

    def __init__(
        self,
        connection: ConnectionSchema,
        test_mode: Optional[bool] = False,
        sample_data_enabled: Optional[bool] = None,
        sample_data_max_records: Optional[int] = None,
    ) -> None:
        """Initialize ValidityExecutor"""
        super().__init__(
            connection, test_mode, sample_data_enabled, sample_data_max_records
        )

    def supports_rule_type(self, rule_type: str) -> bool:
        """Check if the rule type is supported"""
        return rule_type in [t.value for t in self.SUPPORTED_TYPES]

    async def execute_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """Execute validity rule"""
        if rule.type == RuleType.RANGE:
            return await self._execute_range_rule(rule)
        elif rule.type == RuleType.ENUM:
            return await self._execute_enum_rule(rule)
        elif rule.type == RuleType.REGEX:
            return await self._execute_regex_rule(rule)
        elif rule.type == RuleType.DATE_FORMAT:
            return await self._execute_date_format_rule(rule)
        else:
            raise RuleExecutionError(f"Unsupported rule type: {rule.type}")

    async def _execute_range_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """Execute RANGE rule, based on mature logic from Rule._generate_range_sql"""
        import time

        from shared.database.query_executor import QueryExecutor
        from shared.schema.base import DatasetMetrics

        start_time = time.time()
        table_name = self._safe_get_table_name(rule)

        try:
            # Generate validation SQL
            sql = self._generate_range_sql(rule)

            # Execute SQL and get result
            engine = await self.get_engine()
            query_executor = QueryExecutor(engine)

            # Get failed record count
            result, _ = await query_executor.execute_query(sql)
            failed_count = (
                result[0]["anomaly_count"] if result and len(result) > 0 else 0
            )

            # Get total record count
            filter_condition = rule.get_filter_condition()
            total_sql = f"SELECT COUNT(*) as total_count FROM {table_name}"
            if filter_condition:
                total_sql += f" WHERE {filter_condition}"

            total_result, _ = await query_executor.execute_query(total_sql)
            total_count = (
                total_result[0]["total_count"]
                if total_result and len(total_result) > 0
                else 0
            )

            execution_time = time.time() - start_time

            # Build standardized result
            status = "PASSED" if failed_count == 0 else "FAILED"

            # Generate sample data (only on failure)
            sample_data = None
            if failed_count > 0:
                sample_data = await self._generate_sample_data(rule, sql)

            # Build dataset metrics
            dataset_metric = DatasetMetrics(
                entity_name=table_name,
                total_records=total_count,
                failed_records=failed_count,
                processing_time=execution_time,
            )

            return ExecutionResultSchema(
                rule_id=rule.id,
                status=status,
                dataset_metrics=[dataset_metric],
                execution_time=execution_time,
                execution_message=(
                    f"RANGE check completed, found {failed_count} "
                    "out-of-range records"
                    if failed_count > 0
                    else "RANGE check passed"
                ),
                error_message=None,
                sample_data=sample_data,
                cross_db_metrics=None,
                execution_plan={"sql": sql, "execution_type": "single_table"},
                started_at=datetime.fromtimestamp(start_time),
                ended_at=datetime.fromtimestamp(time.time()),
            )

        except Exception as e:
            # Use unified error handling method
            # - distinguish engine-level and rule-level errors
            return await self._handle_execution_error(e, rule, start_time, table_name)

    async def _execute_enum_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """Execute ENUM rule, based on mature logic from Rule._generate_enum_sql"""
        import time

        from shared.database.query_executor import QueryExecutor
        from shared.schema.base import DatasetMetrics

        start_time = time.time()
        table_name = self._safe_get_table_name(rule)

        try:
            # Generate validation SQL
            sql = self._generate_enum_sql(rule)

            # Execute SQL and get result
            engine = await self.get_engine()
            query_executor = QueryExecutor(engine)

            # Get failed record count
            result, _ = await query_executor.execute_query(sql)
            failed_count = (
                result[0]["anomaly_count"] if result and len(result) > 0 else 0
            )

            # Get total record count
            filter_condition = rule.get_filter_condition()
            total_sql = f"SELECT COUNT(*) as total_count FROM {table_name}"
            if filter_condition:
                total_sql += f" WHERE {filter_condition}"

            total_result, _ = await query_executor.execute_query(total_sql)
            total_count = (
                total_result[0]["total_count"]
                if total_result and len(total_result) > 0
                else 0
            )

            execution_time = time.time() - start_time

            # Build standardized result
            status = "PASSED" if failed_count == 0 else "FAILED"

            # Generate sample data (only on failure)
            sample_data = None
            if failed_count > 0:
                sample_data = await self._generate_sample_data(rule, sql)

            # Build dataset metrics
            dataset_metric = DatasetMetrics(
                entity_name=table_name,
                total_records=total_count,
                failed_records=failed_count,
                processing_time=execution_time,
            )

            return ExecutionResultSchema(
                rule_id=rule.id,
                status=status,
                dataset_metrics=[dataset_metric],
                execution_time=execution_time,
                execution_message=(
                    f"ENUM check completed, found {failed_count} "
                    "illegal enum value records"
                    if failed_count > 0
                    else "ENUM check passed"
                ),
                error_message=None,
                sample_data=sample_data,
                cross_db_metrics=None,
                execution_plan={"sql": sql, "execution_type": "single_table"},
                started_at=datetime.fromtimestamp(start_time),
                ended_at=datetime.fromtimestamp(time.time()),
            )

        except Exception as e:
            # Use unified error handling method
            # - distinguish engine-level and rule-level errors
            return await self._handle_execution_error(e, rule, start_time, table_name)

    async def _execute_regex_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """Execute REGEX rule, based on mature logic from Rule._generate_regex_sql"""
        import time

        from shared.database.query_executor import QueryExecutor
        from shared.schema.base import DatasetMetrics

        start_time = time.time()
        table_name = self._safe_get_table_name(rule)

        # Check if database supports regex operations
        if not self.dialect.supports_regex():
            # For SQLite, try to use custom functions to replace REGEX
            if (
                hasattr(self.dialect, "can_use_custom_functions")
                and self.dialect.can_use_custom_functions()
            ):
                return await self._execute_sqlite_custom_regex_rule(rule)
            else:
                raise RuleExecutionError(
                    f"REGEX rule is not supported for "
                    f"{self.dialect.__class__.__name__}"
                )

        try:
            # Generate validation SQL
            sql = self._generate_regex_sql(rule)

            # Execute SQL and get result
            engine = await self.get_engine()
            query_executor = QueryExecutor(engine)

            # Get failed record count
            result, _ = await query_executor.execute_query(sql)
            failed_count = (
                result[0]["anomaly_count"] if result and len(result) > 0 else 0
            )

            # Get total record count
            filter_condition = rule.get_filter_condition()
            total_sql = f"SELECT COUNT(*) as total_count FROM {table_name}"
            if filter_condition:
                total_sql += f" WHERE {filter_condition}"

            total_result, _ = await query_executor.execute_query(total_sql)
            total_count = (
                total_result[0]["total_count"]
                if total_result and len(total_result) > 0
                else 0
            )

            execution_time = time.time() - start_time

            # Build standardized result
            status = "PASSED" if failed_count == 0 else "FAILED"

            # Generate sample data (only on failure)
            sample_data = None
            if failed_count > 0:
                sample_data = await self._generate_sample_data(rule, sql)

            # Build dataset metrics
            dataset_metric = DatasetMetrics(
                entity_name=table_name,
                total_records=total_count,
                failed_records=failed_count,
                processing_time=execution_time,
            )

            return ExecutionResultSchema(
                rule_id=rule.id,
                status=status,
                dataset_metrics=[dataset_metric],
                execution_time=execution_time,
                execution_message=(
                    f"REGEX check completed, found {failed_count} "
                    "format mismatch records"
                    if failed_count > 0
                    else "REGEX check passed"
                ),
                error_message=None,
                sample_data=sample_data,
                cross_db_metrics=None,
                execution_plan={"sql": sql, "execution_type": "single_table"},
                started_at=datetime.fromtimestamp(start_time),
                ended_at=datetime.fromtimestamp(time.time()),
            )

        except Exception as e:
            # Use unified error handling method
            # - distinguish engine-level and rule-level errors
            return await self._handle_execution_error(e, rule, start_time, table_name)

    async def _execute_date_format_rule(
        self, rule: RuleSchema
    ) -> ExecutionResultSchema:
        """
        Execute DATE_FORMAT rule with database-specific strategies:
        - MySQL: Uses STR_TO_DATE (existing implementation)
        - PostgreSQL: Uses two-stage validation (regex + Python)
        - SQLite: Uses custom functions
        """
        import time

        from shared.database.database_dialect import DatabaseType
        from shared.database.query_executor import QueryExecutor
        from shared.schema.base import DatasetMetrics

        start_time = time.time()
        table_name = self._safe_get_table_name(rule)

        try:
            # Check if date format is supported for this database
            if not self.dialect.is_supported_date_format():
                raise RuleExecutionError(
                    "DATE_FORMAT rule is not supported for this database"
                )

            # Get database engine and query executor
            engine = await self.get_engine()
            query_executor = QueryExecutor(engine)

            # Database-specific execution strategies
            if self.dialect.database_type == DatabaseType.POSTGRESQL:
                failed_count, total_count, sample_data = (
                    await self._execute_postgresql_date_format(rule, query_executor)
                )
            elif self.dialect.database_type == DatabaseType.SQLITE:
                failed_count, total_count, sample_data = (
                    await self._execute_sqlite_date_format(rule, query_executor, engine)
                )
            else:
                # MySQL and other databases use the original implementation
                failed_count, total_count, sample_data = (
                    await self._execute_standard_date_format(rule, query_executor)
                )

            execution_time = time.time() - start_time

            # Build standardized result
            status = "PASSED" if failed_count == 0 else "FAILED"

            # Build dataset metrics
            dataset_metric = DatasetMetrics(
                entity_name=table_name,
                total_records=total_count,
                failed_records=failed_count,
                processing_time=execution_time,
            )

            return ExecutionResultSchema(
                rule_id=rule.id,
                status=status,
                dataset_metrics=[dataset_metric],
                execution_time=execution_time,
                execution_message=(
                    f"DATE_FORMAT check completed, found {failed_count} "
                    "date format anomaly records"
                    if failed_count > 0
                    else "DATE_FORMAT check passed"
                ),
                error_message=None,
                sample_data=sample_data,
                cross_db_metrics=None,
                execution_plan={
                    "execution_type": f"{self.dialect.database_type.value}_date_format"
                },
                started_at=datetime.fromtimestamp(start_time),
                ended_at=datetime.fromtimestamp(time.time()),
            )

        except Exception as e:
            # Use unified error handling method
            return await self._handle_execution_error(e, rule, start_time, table_name)

    def _generate_range_sql(self, rule: RuleSchema) -> str:
        """
        Generate RANGE validation SQL

        Ported from app/models/rule.Rule._generate_range_sql
        """
        # Use safe method to get table and column names
        table = self._safe_get_table_name(rule)
        column = self._safe_get_column_name(rule)
        rule_config = rule.get_rule_config()
        filter_condition = rule.get_filter_condition()

        # Get range values from parameters (supports multiple parameter formats)
        # 🔒 Fix: Correctly handle 0 values, avoid falsy values being skipped
        params = rule.parameters if hasattr(rule, "parameters") else {}

        min_value = None
        if "min" in params and params["min"] is not None:
            min_value = params["min"]
        elif "min_value" in params and params["min_value"] is not None:
            min_value = params["min_value"]
        elif "min" in rule_config and rule_config["min"] is not None:
            min_value = rule_config["min"]
        elif "min_value" in rule_config and rule_config["min_value"] is not None:
            min_value = rule_config["min_value"]

        max_value = None
        if "max" in params and params["max"] is not None:
            max_value = params["max"]
        elif "max_value" in params and params["max_value"] is not None:
            max_value = params["max_value"]
        elif "max" in rule_config and rule_config["max"] is not None:
            max_value = rule_config["max"]
        elif "max_value" in rule_config and rule_config["max_value"] is not None:
            max_value = rule_config["max_value"]

        conditions = []

        # Add NULL value check, as NULL values should be considered anomalies
        conditions.append(f"{column} IS NULL")

        # Handle range conditions, particularly boundary cases
        if min_value is not None and max_value is not None:
            if min_value == max_value:
                # Special case: min = max, but still use standard range check
                # format to meet test expectations
                # This ensures that < and > symbols are included in the SQL
                conditions.append(f"({column} < {min_value} OR {column} > {max_value})")
            else:
                # Standard range check: value must be within [min, max]
                conditions.append(f"({column} < {min_value} OR {column} > {max_value})")
        elif min_value is not None:
            # Only minimum value limit
            conditions.append(f"{column} < {min_value}")
        elif max_value is not None:
            # Only maximum value limit
            conditions.append(f"{column} > {max_value}")
        else:
            # If no range values, only check for NULL values
            pass

        # Build complete WHERE clause
        if len(conditions) == 0:
            # Should theoretically not reach here
            where_clause = "WHERE 1=0"  # Empty result
        elif len(conditions) == 1:
            where_clause = f"WHERE {conditions[0]}"
        else:
            where_clause = f"WHERE ({' OR '.join(conditions)})"

        if filter_condition:
            where_clause += f" AND ({filter_condition})"

        return f"SELECT COUNT(*) AS anomaly_count FROM {table} {where_clause}"

    def _generate_enum_sql(self, rule: RuleSchema) -> str:
        """
        Generate ENUM validation SQL

        Ported from app/models/rule.Rule._generate_enum_sql
        """
        # Use safe method to get table and column names
        table = self._safe_get_table_name(rule)
        column = self._safe_get_column_name(rule)
        rule_config = rule.get_rule_config()
        filter_condition = rule.get_filter_condition()

        # Get allowed value list from parameters
        params = rule.parameters if hasattr(rule, "parameters") else {}
        allowed_values = params.get("allowed_values") or rule_config.get(
            "allowed_values", []
        )

        if not allowed_values:
            raise RuleExecutionError("ENUM rule requires allowed_values")

        # Check if email domain extraction is needed
        extract_domain = rule_config.get("extract_domain", False)

        if extract_domain:
            # Use SUBSTRING_INDEX to check email domain
            domain_column = f"SUBSTRING_INDEX({column}, '@', -1)"
            values_str = ", ".join(
                [f"'{v}'" if isinstance(v, str) else str(v) for v in allowed_values]
            )
            where_clause = (
                f"WHERE {column} IS NOT NULL AND {column} LIKE '%@%' AND "
                f"{domain_column} NOT IN ({values_str})"
            )
        else:
            # Standard enum value check
            values_str = ", ".join(
                [f"'{v}'" if isinstance(v, str) else str(v) for v in allowed_values]
            )
            where_clause = f"WHERE {column} NOT IN ({values_str})"

        if filter_condition:
            where_clause += f" AND ({filter_condition})"

        return f"SELECT COUNT(*) AS anomaly_count FROM {table} {where_clause}"

    def _generate_regex_sql(self, rule: RuleSchema) -> str:
        """
        Generate REGEX validation SQL

        Ported from app/models/rule.Rule._generate_regex_sql
        """
        # Use safe method to get table and column names
        table = self._safe_get_table_name(rule)
        column = self._safe_get_column_name(rule)
        rule_config = rule.get_rule_config()
        filter_condition = rule.get_filter_condition()

        # Get regex pattern from parameters
        params = rule.parameters if hasattr(rule, "parameters") else {}
        pattern = params.get("pattern") or rule_config.get("pattern")

        if not pattern:
            raise RuleExecutionError("REGEX rule requires pattern")

        # SQL injection protection: check if pattern contains potentially
        # dangerous SQL keywords
        dangerous_patterns = [
            "DROP TABLE",
            "DELETE FROM",
            "UPDATE SET",
            "INSERT INTO",
            "TRUNCATE",
            "ALTER TABLE",
            "CREATE TABLE",
            "DROP DATABASE",
            "--",
            "/*",
            "*/",
            "UNION SELECT",
            "'; ",
            " OR '",
            "1=1",
        ]

        pattern_upper = pattern.upper()
        for dangerous in dangerous_patterns:
            if dangerous in pattern_upper:
                raise RuleExecutionError(
                    f"Pattern contains potentially dangerous SQL patterns: {dangerous}"
                )

        # Escape single quotes to prevent SQL injection
        escaped_pattern = pattern.replace("'", "''")
        regex_op = self.dialect.get_not_regex_operator()

        # Cast column for regex operations if needed (PostgreSQL requires casting
        # for non-text columns)
        regex_column = self.dialect.cast_column_for_regex(column)

        # Generate REGEXP expression using the dialect
        where_clause = f"WHERE {regex_column} {regex_op} '{escaped_pattern}'"

        if filter_condition:
            where_clause += f" AND ({filter_condition})"

        return f"SELECT COUNT(*) AS anomaly_count FROM {table} {where_clause}"

    async def _execute_postgresql_date_format(
        self, rule: RuleSchema, query_executor: QueryExecutor
    ) -> tuple[int, int, list]:
        """Execute PostgreSQL two-stage date format validation"""

        from typing import cast

        from shared.database.database_dialect import PostgreSQLDialect

        postgres_dialect = cast(PostgreSQLDialect, self.dialect)
        table_name = self._safe_get_table_name(rule)
        column = self._safe_get_column_name(rule)
        format_pattern = self._get_format_pattern(rule)
        filter_condition = rule.get_filter_condition()

        # Stage 1: Get regex-based failures and candidates for Python validation
        stage1_sql, stage2_sql = postgres_dialect.get_two_stage_date_validation_sql(
            column, format_pattern, table_name, filter_condition
        )

        # Execute stage 1: get regex failures
        stage1_result, _ = await query_executor.execute_query(stage1_sql)
        regex_failed_count = (
            stage1_result[0]["regex_failed_count"] if stage1_result else 0
        )

        # Execute stage 2: get candidates for Python validation
        stage2_result, _ = await query_executor.execute_query(stage2_sql)
        candidates = [row[column] for row in stage2_result] if stage2_result else []

        # Stage 3: Python validation for semantic correctness
        python_failed_candidates = []
        normalized_pattern = self._normalize_format_pattern(format_pattern)

        for candidate in candidates:
            if candidate and not self._validate_date_in_python(
                candidate, normalized_pattern
            ):
                python_failed_candidates.append(candidate)

        # Stage 4: Count records with Python-detected failures
        python_failed_count = 0
        if python_failed_candidates:
            # Build SQL to count records with semantically invalid dates
            # Handle both string and integer candidates properly
            escaped_candidates = []
            for candidate in python_failed_candidates:
                if isinstance(candidate, str):
                    escaped_candidates.append(candidate.replace("'", "''"))
                else:
                    # For integer and other types, convert to string
                    #  (no escaping needed for integers)
                    escaped_candidates.append(str(candidate))

            values_list = "', '".join(escaped_candidates)
            python_count_where = f"WHERE {column} IN ('{values_list}')"
            if filter_condition:
                python_count_where += f" AND ({filter_condition})"

            python_count_sql = (
                f"SELECT COUNT(*) as python_failed_count "
                f"FROM {table_name} {python_count_where}"
            )
            python_result, _ = await query_executor.execute_query(python_count_sql)
            python_failed_count = (
                python_result[0]["python_failed_count"] if python_result else 0
            )

        # Get total record count
        total_sql = f"SELECT COUNT(*) as total_count FROM {table_name}"
        if filter_condition:
            total_sql += f" WHERE {filter_condition}"
        total_result, _ = await query_executor.execute_query(total_sql)
        total_count = int(total_result[0]["total_count"]) if total_result else 0

        # Generate sample data
        total_failed = int(regex_failed_count) + int(python_failed_count)
        if total_failed > 0:
            sample_data = await self._generate_postgresql_sample_data(
                rule, query_executor, python_failed_candidates
            )

        if sample_data is None:
            sample_data = []
        return total_failed, total_count, sample_data

    async def _execute_sqlite_date_format(
        self, rule: RuleSchema, query_executor: QueryExecutor, engine: Any
    ) -> tuple[int, int, list]:
        """Execute SQLite date format validation with custom functions"""

        table_name = self._safe_get_table_name(rule)
        # format_pattern = self._get_format_pattern(rule)

        # Use the custom function for validation
        sql = self._generate_date_format_sql(rule)

        # Execute SQL and get result
        result, _ = await query_executor.execute_query(sql)
        failed_count = result[0]["anomaly_count"] if result and len(result) > 0 else 0

        # Get total record count
        filter_condition = rule.get_filter_condition()
        total_sql = f"SELECT COUNT(*) as total_count FROM {table_name}"
        if filter_condition:
            total_sql += f" WHERE {filter_condition}"
        total_result, _ = await query_executor.execute_query(total_sql)
        total_count = int(total_result[0]["total_count"]) if total_result else 0

        # Generate sample data

        if failed_count > 0:
            sample_data = await self._generate_sample_data(rule, sql)

        if sample_data is None:
            sample_data = []
        return failed_count, total_count, sample_data

    async def _execute_standard_date_format(
        self, rule: RuleSchema, query_executor: QueryExecutor
    ) -> tuple[int, int, list]:
        """Execute standard date format validation (MySQL and others)"""
        # Original implementation for MySQL and other databases
        sql = self._generate_date_format_sql(rule)

        # Execute SQL and get result
        result, _ = await query_executor.execute_query(sql)
        failed_count = (
            int(result[0]["anomaly_count"]) if result and len(result) > 0 else 0
        )

        # Get total record count
        table_name = self._safe_get_table_name(rule)
        filter_condition = rule.get_filter_condition()
        total_sql = f"SELECT COUNT(*) as total_count FROM {table_name}"
        if filter_condition:
            total_sql += f" WHERE {filter_condition}"
        total_result, _ = await query_executor.execute_query(total_sql)
        total_count = int(total_result[0]["total_count"]) if total_result else 0

        # Generate sample data
        # sample_data = []
        if failed_count > 0:
            sample_data = await self._generate_sample_data(rule, sql)

        if sample_data is None:
            sample_data = []
        return failed_count, total_count, sample_data

    def _validate_date_in_python(self, date_value: Any, format_pattern: str) -> bool:
        """Validate date value in Python for semantic correctness"""
        from datetime import datetime

        try:
            # Convert to string if it's not already
            #  (handles integer date values like 19680223)
            if isinstance(date_value, int):
                date_str = str(date_value)
            elif isinstance(date_value, str):
                date_str = date_value
            else:
                # Convert other types to string
                date_str = str(date_value)

            # Parse date using the specified format
            parsed_date = datetime.strptime(date_str, format_pattern)
            # Round-trip validation to catch semantic errors like 2000-02-31
            return parsed_date.strftime(format_pattern) == date_str
        except (ValueError, TypeError):
            return False

    def _get_format_pattern(self, rule: RuleSchema) -> str:
        """Extract format pattern from rule parameters"""
        params = rule.parameters if hasattr(rule, "parameters") else {}
        format_pattern = (
            params.get("format_pattern")
            or params.get("format")
            or rule.get_rule_config().get("format_pattern")
            or rule.get_rule_config().get("format")
        )

        if not format_pattern:
            raise RuleExecutionError("DATE_FORMAT rule requires format_pattern")

        return str(format_pattern)

    def _normalize_format_pattern(self, format_pattern: str) -> str:
        """Normalize format pattern for Python datetime"""
        # Handle both case variations (YYYY/yyyy, MM/mm, etc.)
        pattern_map = {
            "YYYY": "%Y",
            "yyyy": "%Y",
            "MM": "%m",
            "mm": "%m",
            "DD": "%d",
            "dd": "%d",
            "HH": "%H",
            "hh": "%H",
            "MI": "%M",
            "mi": "%M",
            "SS": "%S",
            "ss": "%S",
        }

        normalized = format_pattern
        # Sort by length (descending) to avoid partial replacements
        for fmt in sorted(pattern_map.keys(), key=len, reverse=True):
            normalized = normalized.replace(fmt, pattern_map[fmt])

        return normalized

    async def _generate_postgresql_sample_data(
        self,
        rule: RuleSchema,
        query_executor: QueryExecutor,
        python_failed_candidates: list,
    ) -> list | None:
        """Generate sample data for PostgreSQL date format failures"""
        try:
            from core.config import get_core_config

            try:
                core_config = get_core_config()
                max_samples = (
                    core_config.sample_data_max_records
                    if core_config.sample_data_max_records
                    else 5
                )
            except Exception:
                max_samples = 5

            table_name = self._safe_get_table_name(rule)
            column = self._safe_get_column_name(rule)
            format_pattern = self._get_format_pattern(rule)
            filter_condition = rule.get_filter_condition()

            # Get sample data from both regex failures and Python failures
            from typing import cast

            from shared.database.database_dialect import PostgreSQLDialect

            postgres_dialect = cast(PostgreSQLDialect, self.dialect)
            regex_pattern = postgres_dialect._format_pattern_to_regex(format_pattern)

            # Sample data from regex failures
            # Cast column for regex operations to handle integer columns
            cast_column = postgres_dialect.cast_column_for_regex(column)
            regex_sample_where = (
                f"WHERE {column} IS NOT NULL AND {cast_column} !~ '{regex_pattern}'"
            )
            if filter_condition:
                regex_sample_where += f" AND ({filter_condition})"

            regex_sample_sql = (
                f"SELECT * FROM {table_name} {regex_sample_where} LIMIT {max_samples}"
            )
            regex_samples, _ = await query_executor.execute_query(regex_sample_sql)

            # Sample data from Python failures
            python_samples: list[dict[str, Any]] = []
            if python_failed_candidates:
                escaped_candidates = [
                    candidate.replace("'", "''")
                    for candidate in python_failed_candidates
                ]
                values_list = "', '".join(escaped_candidates)
                python_sample_where = f"WHERE {column} IN ('{values_list}')"
                if filter_condition:
                    python_sample_where += f" AND ({filter_condition})"

                python_sample_sql = (
                    f"SELECT * FROM {table_name} {python_sample_where} LIMIT "
                    f"{max_samples}"
                )
                python_samples, _ = await query_executor.execute_query(
                    python_sample_sql
                )

            # Combine samples intelligently
            regex_count = len(regex_samples) if regex_samples else 0
            python_count = len(python_samples) if python_samples else 0

            if regex_count == 0 and python_count == 0:
                return []
            elif regex_count == 0:
                # Only Python failures, take all up to max_samples
                return python_samples[:max_samples]
            elif python_count == 0:
                # Only regex failures, take all up to max_samples
                return regex_samples[:max_samples]
            else:
                # Both samples, try to balance them while ensuring total <= max_samples
                # Calculate how to split samples to ensure both types are represented
                half_samples = max_samples // 2

                # Take at least 1 from each type if available, then fill remaining space
                if regex_count >= half_samples and python_count >= half_samples:
                    # Both have enough samples, take half from each
                    combined_samples = (
                        regex_samples[:half_samples] + python_samples[:half_samples]
                    )
                elif regex_count < half_samples:
                    # Regex has fewer samples, take all regex + fill with python
                    remaining_slots = max_samples - regex_count
                    combined_samples = regex_samples + python_samples[:remaining_slots]
                else:
                    # Python has fewer samples, take all python + fill with regex
                    remaining_slots = max_samples - python_count
                    combined_samples = regex_samples[:remaining_slots] + python_samples

                return combined_samples[:max_samples]

        except Exception as e:
            self.logger.warning(f"Failed to generate PostgreSQL sample data: {e}")
            return None

    def _generate_date_format_sql(self, rule: RuleSchema) -> str:
        """
        Generate DATE_FORMAT validation SQL

        Ported from app/models/rule.Rule._generate_date_format_sql
        """
        # Use safe method to get table and column names
        table = self._safe_get_table_name(rule)
        column = self._safe_get_column_name(rule)
        rule_config = rule.get_rule_config()
        filter_condition = rule.get_filter_condition()

        # Get date format pattern from parameters
        params = rule.parameters if hasattr(rule, "parameters") else {}
        format_pattern = (
            params.get("format_pattern")
            or params.get("format")
            or rule_config.get("format_pattern")
            or rule_config.get("format")
        )

        if not format_pattern:
            raise RuleExecutionError("DATE_FORMAT rule requires format_pattern")

        date_clause = self.dialect.get_date_clause(column, format_pattern)
        # Generate date format check using the dialect. Dates that cannot be parsed
        # return NULL
        where_clause = f"WHERE {date_clause} IS NULL"

        if filter_condition:
            where_clause += f" AND ({filter_condition})"

        return f"SELECT COUNT(*) AS anomaly_count FROM {table} {where_clause}"

    async def _execute_sqlite_custom_regex_rule(
        self, rule: RuleSchema
    ) -> ExecutionResultSchema:
        """
        Use SQLite custom functions to execute REGEX rules as
        an alternative solution

        """
        import time

        from shared.database.query_executor import QueryExecutor
        from shared.schema.base import DatasetMetrics

        start_time = time.time()
        table_name = self._safe_get_table_name(rule)

        try:
            # Generate SQL using custom functions
            sql = self._generate_sqlite_custom_validation_sql(rule)

            # Execute SQL and get result
            engine = await self.get_engine()
            query_executor = QueryExecutor(engine)

            # Get failed record count
            result, _ = await query_executor.execute_query(sql)
            failed_count = (
                result[0]["anomaly_count"] if result and len(result) > 0 else 0
            )

            # Get total record count
            filter_condition = rule.get_filter_condition()
            total_sql = f"SELECT COUNT(*) as total_count FROM {table_name}"
            if filter_condition:
                total_sql += f" WHERE {filter_condition}"

            total_result, _ = await query_executor.execute_query(total_sql)
            total_count = (
                total_result[0]["total_count"]
                if total_result and len(total_result) > 0
                else 0
            )

            execution_time = time.time() - start_time

            # Build standardized result
            status = "PASSED" if failed_count == 0 else "FAILED"

            # Generate sample data (only on failure)
            sample_data = None
            if failed_count > 0:
                sample_data = await self._generate_sample_data(rule, sql)

            # Build dataset metrics
            dataset_metric = DatasetMetrics(
                entity_name=table_name,
                total_records=total_count,
                failed_records=failed_count,
                processing_time=execution_time,
            )

            return ExecutionResultSchema(
                rule_id=rule.id,
                status=status,
                dataset_metrics=[dataset_metric],
                execution_time=execution_time,
                execution_message=(
                    f"Custom validation completed, found {failed_count} "
                    "format mismatch records"
                    if failed_count > 0
                    else "Custom validation passed"
                ),
                error_message=None,
                sample_data=sample_data,
                cross_db_metrics=None,
                execution_plan={"sql": sql, "execution_type": "single_table"},
                started_at=datetime.fromtimestamp(start_time),
                ended_at=datetime.fromtimestamp(time.time()),
            )

        except Exception as e:
            # Use unified error handling method
            return await self._handle_execution_error(e, rule, start_time, table_name)

    def _generate_sqlite_custom_validation_sql(self, rule: RuleSchema) -> str:
        """
        Generate validation SQL using custom functions for SQLite
        - refactored version

        Remove hardcoded logic, dynamically determine validation type based
        on rule configuration
        """
        table = self._safe_get_table_name(rule)
        column = self._safe_get_column_name(rule)
        filter_condition = rule.get_filter_condition()

        # Dynamically determine validation type and parameters
        validation_info = self._determine_validation_type_from_rule(rule)

        # Generate validation conditions based on validation type
        validation_condition = self._generate_validation_condition_by_type(
            validation_info, column
        )

        # Build WHERE clause
        where_clause = f"WHERE {validation_condition}"
        if filter_condition:
            where_clause += f" AND ({filter_condition})"

        return f"SELECT COUNT(*) AS anomaly_count FROM {table} {where_clause}"

    def _determine_validation_type_from_rule(self, rule: RuleSchema) -> dict:
        """
        Dynamically determine validation type and
          parameters based on rule configuration
        """
        params = getattr(rule, "parameters", {})
        rule_config = rule.get_rule_config()

        # Priority to get validation type information from rule configuration
        validation_info: Dict[str, Any] = {
            "type": None,
            "parameters": {},
        }

        # 1. Check if there is explicit validation type configuration
        if "validation_type" in params:
            validation_info["type"] = params["validation_type"]
            validation_info["parameters"] = params
        elif "validation_type" in rule_config:
            validation_info["type"] = rule_config["validation_type"]
            validation_info["parameters"] = rule_config

        # 2. Infer validation type from desired_type field (this is key missing logic)
        elif "desired_type" in params:
            validation_info = self._infer_validation_from_desired_type(
                params["desired_type"]
            )
            validation_info["parameters"].update(params)
        elif "desired_type" in rule_config:
            validation_info = self._infer_validation_from_desired_type(
                rule_config["desired_type"]
            )
            validation_info["parameters"].update(rule_config)

        # 3. Infer validation type based on pattern
        elif "pattern" in params:
            validation_info = self._infer_validation_from_pattern(params["pattern"])
            # If pattern inference fails, try description inference
            if validation_info["type"] is None and "description" in params:
                validation_info = self._infer_validation_from_description(
                    params["description"]
                )
            # Merge other parameters
            validation_info["parameters"].update(params)

        # 4. Infer validation type based on description
        elif "description" in params:
            validation_info = self._infer_validation_from_description(
                params["description"]
            )
            validation_info["parameters"].update(params)

        return validation_info

    def _infer_validation_from_desired_type(self, desired_type: str) -> dict:
        """
        Infer validation type from desired_type field
        (e.g.: 'integer(2)', 'float(4,1)', 'string(10)'))
        """
        import re

        # Parse integer(N) format
        int_match = re.match(r"integer\((\d+)\)", desired_type)
        if int_match:
            max_digits = int(int_match.group(1))
            return {"type": "integer_digits", "parameters": {"max_digits": max_digits}}

        # Parse float(precision,scale) format
        float_match = re.match(r"float\((\d+),(\d+)\)", desired_type)
        if float_match:
            precision = int(float_match.group(1))
            scale = int(float_match.group(2))
            return {
                "type": "float_precision",
                "parameters": {"precision": precision, "scale": scale},
            }

        # Parse string(N) format
        string_match = re.match(r"string\((\d+)\)", desired_type)
        if string_match:
            max_length = int(string_match.group(1))
            return {"type": "string_length", "parameters": {"max_length": max_length}}

        # Parse basic types
        if desired_type == "integer":
            return {"type": "integer_format", "parameters": {}}
        elif desired_type == "float":
            return {"type": "float_format", "parameters": {}}
        elif desired_type == "string":
            return {"type": "string_length", "parameters": {}}

        return {"type": None, "parameters": {}}

    def _infer_validation_from_pattern(self, pattern: str) -> dict:
        """Infer validation type from regex pattern"""
        import re

        # Integer digit validation: ^-?\\d{1,N}$ or ^-?[0-9]{1,N}$
        int_digits_match = re.search(
            r"\\\\d\\{1,(\\d+)\\}|\\[0-9\\]\\{1,(\\d+)\\}", pattern
        )
        if int_digits_match:
            max_digits = int(int_digits_match.group(1) or int_digits_match.group(2))
            return {"type": "integer_digits", "parameters": {"max_digits": max_digits}}

        # String length validation: ^.{0,N}$
        str_length_match = re.search(r"\\.\\{0,(\\d+)\\}", pattern)
        if str_length_match:
            max_length = int(str_length_match.group(1))
            return {"type": "string_length", "parameters": {"max_length": max_length}}

        # Float validation: contains decimal point pattern
        if r"\\." in pattern and any(x in pattern for x in [r"\\d", "[0-9]"]):
            # Check if it's float to integer conversion (contains .0* pattern)
            if r"\\.0\\*" in pattern or r"\\.0+" in pattern:
                return {"type": "float_to_integer", "parameters": {}}
            return {"type": "float_format", "parameters": {}}

        return {"type": None, "parameters": {}}

    def _infer_validation_from_description(self, description: str) -> dict:
        """Infer validation type from description"""
        import re

        description_lower = description.lower()

        # Float precision/scale validation - fix regex expression
        if "precision/scale validation" in description_lower:
            # Match "Float precision/scale validation for (4,1)" format
            match = re.search(r"validation for \((\d+),(\d+)\)", description)
            if match:
                precision = int(match.group(1))
                scale = int(match.group(2))
                return {
                    "type": "float_precision",
                    "parameters": {"precision": precision, "scale": scale},
                }

        # Integer format validation
        if "integer" in description_lower and "format validation" in description_lower:
            return {"type": "integer_format", "parameters": {}}

        # Integer digits validation
        if "integer" in description_lower and any(
            word in description_lower for word in ["precision", "digits"]
        ):
            # Try to extract digit count
            match = re.search(r"max (\d+).*?digit", description_lower)
            if match:
                max_digits = int(match.group(1))
                return {
                    "type": "integer_digits",
                    "parameters": {"max_digits": max_digits},
                }
            return {"type": "integer_digits", "parameters": {}}

        # Float validation
        if "float" in description_lower:
            return {"type": "float_format", "parameters": {}}

        # String length validation
        if "string" in description_lower or "length" in description_lower:
            match = re.search(r"max (\d+).*?character", description_lower)
            if match:
                max_length = int(match.group(1))
                return {
                    "type": "string_length",
                    "parameters": {"max_length": max_length},
                }
            return {"type": "string_length", "parameters": {}}

        return {"type": None, "parameters": {}}

    def _generate_validation_condition_by_type(
        self, validation_info: dict, column: str
    ) -> str:
        """Generate validation conditions based on validation type information"""
        validation_type = validation_info.get("type")
        params = validation_info.get("parameters", {})

        if not validation_type:
            return "1=0"  # No validation condition

        from typing import cast

        from shared.database.database_dialect import SQLiteDialect

        sqlite_dialect = cast(SQLiteDialect, self.dialect)

        if validation_type == "integer_digits":
            max_digits = params.get("max_digits")
            if not max_digits:
                # Try to extract from other methods
                max_digits = self._extract_digits_from_params(params)
            if max_digits:
                return sqlite_dialect.generate_custom_validation_condition(
                    "integer_digits", column, max_digits=max_digits
                )
            return (
                f"typeof({column}) NOT IN ('integer', 'real') OR {column} "
                f"!= CAST({column} AS INTEGER)"
            )

        elif validation_type == "string_length":
            max_length = params.get("max_length")
            if not max_length:
                # Try to extract from other methods
                max_length = self._extract_length_from_params(params)
            if max_length:
                return sqlite_dialect.generate_custom_validation_condition(
                    "string_length", column, max_length=max_length
                )
            return "1=0"

        elif validation_type == "float_precision":
            precision = params.get("precision")
            scale = params.get("scale")
            if precision is not None and scale is not None:
                return sqlite_dialect.generate_custom_validation_condition(
                    "float_precision", column, precision=precision, scale=scale
                )
            return f"typeof({column}) NOT IN ('integer', 'real')"

        elif validation_type == "float_format":
            return f"typeof({column}) NOT IN ('integer', 'real')"

        elif validation_type == "integer_format":
            return (
                f"typeof({column}) NOT IN ('integer', 'real') OR {column} "
                f"!= CAST({column} AS INTEGER)"
            )

        elif validation_type == "float_to_integer":
            # Special case: float to integer validation, check if it's an integer
            return (
                f"typeof({column}) NOT IN ('integer', 'real') OR {column} "
                f"!= CAST({column} AS INTEGER)"
            )

        return "1=0"

    def _extract_digits_from_params(self, params: dict) -> Optional[int]:
        """Extract digit count information from parameters"""
        if "max_digits" in params:
            return int(params["max_digits"])

        # Try to extract from pattern parameter
        if "pattern" in params:
            pattern = params["pattern"]
            import re

            # Match \\d{1,number} format
            match = re.search(r"\\\\d\\{1,(\\d+)\\}", pattern)
            if match:
                return int(match.group(1))
            # Match [0-9]{1,number} format
            match = re.search(r"\\[0-9\\]\\{1,(\\d+)\\}", pattern)
            if match:
                return int(match.group(1))

        return None

    def _extract_length_from_params(self, params: dict) -> Optional[int]:
        """Extract string length information from parameters"""
        if "max_length" in params:
            return int(params["max_length"])

        # Try to extract from pattern parameter
        if "pattern" in params:
            pattern = params["pattern"]
            import re

            match = re.search(r"\\.\\{0,(\\d+)\\}", pattern)
            if match:
                return int(match.group(1))

        return None

    def _extract_digits_from_rule(self, rule: RuleSchema) -> Optional[int]:
        """Extract digit count information from rule"""
        # First try to extract from parameters
        params = getattr(rule, "parameters", {})
        if "max_digits" in params:
            return int(params["max_digits"])

        # Try to extract from pattern parameter (applicable to REGEX rules)
        if "pattern" in params:
            pattern = params["pattern"]
            # Find digits in patterns like '^-?\\d{1,5}$' or '^-?[0-9]{1,2}$'
            import re

            # Match \d{1,number} format
            match = re.search(r"\\d\{1,(\d+)\}", pattern)
            if match:
                return int(match.group(1))
            # Match [0-9]{1,number} format
            match = re.search(r"\[0-9\]\{1,(\d+)\}", pattern)
            if match:
                return int(match.group(1))

        # Try to extract from rule name
        if hasattr(rule, "name") and rule.name:
            # Find patterns like "integer(5)" or "integer_digits_5"
            import re

            match = re.search(r"integer.*?(\d+)", rule.name)
            if match:
                return int(match.group(1))

        # Try to extract from description
        description = params.get("description", "")
        if description:
            import re

            # Find patterns like "max 5 digits" or "validation for max 5 integer digits"
            match = re.search(r"max (\d+).*?digit", description)
            if match:
                return int(match.group(1))

        return None

    def _extract_length_from_rule(self, rule: RuleSchema) -> Optional[int]:
        """Extract string length information from rule"""
        # First try to extract from parameters
        params = getattr(rule, "parameters", {})
        if "max_length" in params:
            return int(params["max_length"])

        # Try to extract from pattern parameter (applicable to REGEX rules)
        if "pattern" in params:
            pattern = params["pattern"]
            # Find digits in patterns like '^.{0,10}$'
            import re

            match = re.search(r"\{0,(\d+)\}", pattern)
            if match:
                return int(match.group(1))

        # Try to extract from rule name
        if hasattr(rule, "name") and rule.name:
            # Find patterns like "string(10)" or "length_10"
            import re

            match = re.search(r"(?:string|length).*?(\d+)", rule.name)
            if match:
                return int(match.group(1))

        # Try to extract from description
        description = params.get("description", "")
        if description:
            import re

            # Find patterns like "max 10 characters" or "length validation for max 10"
            match = re.search(r"max (\d+).*?character", description)
            if match:
                return int(match.group(1))

        return None

    def _extract_float_precision_scale_from_description(
        self, description: str
    ) -> tuple[Optional[int], Optional[int]]:
        """Extract float precision and scale information from description"""
        import re

        # Find patterns like "Float precision/scale validation for (4,1)"
        match = re.search(r"validation for \((\d+),(\d+)\)", description)
        if match:
            precision: Optional[int] = int(match.group(1))
            scale: Optional[int] = int(match.group(2))
            return precision, scale

        # Find patterns like "precision=4, scale=1"
        precision_match = re.search(
            r"precision[=:]?\s*(\d+)", description, re.IGNORECASE
        )
        scale_match = re.search(r"scale[=:]?\s*(\d+)", description, re.IGNORECASE)

        precision = int(precision_match.group(1)) if precision_match else None
        scale = int(scale_match.group(1)) if scale_match else None

        return precision, scale
