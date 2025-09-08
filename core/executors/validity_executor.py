"""
Validity rule executor - based on mature existing logic

Ported from mature validation logic in app/models/rule.py
Unified handling: RANGE, ENUM, REGEX and similar rules
"""

from datetime import datetime
from typing import Optional

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
        Execute DATE_FORMAT rule, based on mature logic from
        Rule._generate_date_format_sql
        """
        import time

        from shared.database.query_executor import QueryExecutor
        from shared.schema.base import DatasetMetrics

        start_time = time.time()
        table_name = self._safe_get_table_name(rule)

        try:
            # Check if date format is supported for this database. Some
            # databases will raise an error for invalid date formats.
            if not self.dialect.is_supported_date_format():
                raise RuleExecutionError(
                    "DATE_FORMAT rule is not supported for this database"
                )

            # Generate validation SQL
            sql = self._generate_date_format_sql(rule)

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
                    f"DATE_FORMAT check completed, found {failed_count} "
                    "date format anomaly records"
                    if failed_count > 0
                    else "DATE_FORMAT check passed"
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

        # Generate REGEXP expression using the dialect
        where_clause = f"WHERE {column} {regex_op} '{escaped_pattern}'"

        if filter_condition:
            where_clause += f" AND ({filter_condition})"

        return f"SELECT COUNT(*) AS anomaly_count FROM {table} {where_clause}"

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
