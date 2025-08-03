"""
Completeness rule executor - based on mature existing logic

Unified handling: NOT_NULL, LENGTH and similar rules
"""

from typing import Optional

from shared.enums.rule_types import RuleType
from shared.exceptions.exception_system import RuleExecutionError
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.result_schema import ExecutionResultSchema
from shared.schema.rule_schema import RuleSchema

from .base_executor import BaseExecutor


class CompletenessExecutor(BaseExecutor):
    """
    Completeness rule executor

    Unified handling: NOT_NULL, LENGTH and similar rules
    """

    SUPPORTED_TYPES = [RuleType.NOT_NULL, RuleType.LENGTH]

    def __init__(
        self,
        connection: ConnectionSchema,
        test_mode: Optional[bool] = False,
        sample_data_enabled: Optional[bool] = None,
        sample_data_max_records: Optional[int] = None,
    ) -> None:
        """Initialize CompletenessExecutor"""
        super().__init__(
            connection, test_mode, sample_data_enabled, sample_data_max_records
        )

    def supports_rule_type(self, rule_type: str) -> bool:
        """Check if the rule type is supported"""
        return rule_type in [t.value for t in self.SUPPORTED_TYPES]

    async def execute_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """Execute completeness rule"""
        if rule.type == RuleType.NOT_NULL:
            return await self._execute_not_null_rule(rule)
        elif rule.type == RuleType.LENGTH:
            return await self._execute_length_rule(rule)
        else:
            raise RuleExecutionError(f"Unsupported rule type: {rule.type}")

    async def _execute_not_null_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """
        Execute NOT_NULL rule

        Based on mature logic from Rule._generate_not_null_sql
        """
        import time

        from shared.database.query_executor import QueryExecutor

        start_time = time.time()
        table_name = self._safe_get_table_name(rule)

        try:
            # Generate validation SQL
            sql = self._generate_not_null_sql(rule)

            # Execute SQL and get result
            engine = await self.get_engine()
            query_executor = QueryExecutor(engine)

            # Get failed record count - directly execute generated SQL
            result, _ = await query_executor.execute_query(sql)
            failed_count = result[0]["failed_count"] if result else 0

            # Get total record count - use safe method
            filter_condition = rule.get_filter_condition()

            # Build total record count query SQL
            total_sql = f"SELECT COUNT(*) as total_count FROM {table_name}"
            if filter_condition:
                total_sql += f" WHERE {filter_condition}"

            total_result, _ = await query_executor.execute_query(total_sql)
            total_count = total_result[0]["total_count"] if total_result else 0

            execution_time = time.time() - start_time

            # Build standardized result
            from shared.schema.base import DatasetMetrics

            status = "PASSED" if failed_count == 0 else "FAILED"

            # Generate sample data (only on failure)
            sample_data = None
            if failed_count > 0:
                sample_data = await self._generate_sample_data(rule, sql)

            # Build dataset metrics
            from datetime import datetime

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
                    f"NOT_NULL check completed, found {failed_count} null records"
                    if failed_count > 0
                    else "NOT_NULL check passed"
                ),
                error_message=(
                    None if failed_count == 0 else f"Found {failed_count} null records"
                ),
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

    async def _execute_length_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """
        Execute LENGTH rule

        Based on mature logic from Rule._generate_length_sql
        """
        import time
        from datetime import datetime

        from shared.database.query_executor import QueryExecutor
        from shared.schema.base import DatasetMetrics

        start_time = time.time()
        table_name = self._safe_get_table_name(rule)

        try:
            # Generate validation SQL
            sql = self._generate_length_sql(rule)

            # Execute SQL and get result
            engine = await self.get_engine()
            query_executor = QueryExecutor(engine)

            # Get failed record count
            result, _ = await query_executor.execute_query(sql)
            failed_count = (
                result[0]["failed_count"] if result and len(result) > 0 else 0
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
                    f"LENGTH check completed, found {failed_count} "
                    "length anomaly records"
                    if failed_count > 0
                    else "LENGTH check passed"
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

    def _generate_not_null_sql(self, rule: RuleSchema) -> str:
        """
        Generate NOT_NULL validation SQL

        Ported from app/models/rule.Rule._generate_not_null_sql
        """
        # Use safe method to get table and column names
        table = self._safe_get_table_name(rule)
        column = self._safe_get_column_name(rule)
        filter_condition = rule.get_filter_condition()

        # Base WHERE condition
        where_clause = f"{column} IS NULL"

        # Add filter condition
        if filter_condition:
            where_clause = f"({where_clause}) AND ({filter_condition})"

        return f"SELECT COUNT(*) AS failed_count FROM {table} WHERE {where_clause}"

    def _generate_length_sql(self, rule: RuleSchema) -> str:
        """
        Generate LENGTH validation SQL

        Ported from app/models/rule.Rule._generate_length_sql
        🔒 Enhanced with SQL injection protection
        """
        # Use safe method to get table and column names
        table = self._safe_get_table_name(rule)
        column = self._safe_get_column_name(rule)
        rule_config = rule.get_rule_config()
        filter_condition = rule.get_filter_condition()

        min_length = rule_config.get("min_length")
        max_length = rule_config.get("max_length")

        conditions = []

        if min_length is not None:
            conditions.append(f"LENGTH({column}) < {min_length}")

        if max_length is not None:
            conditions.append(f"LENGTH({column}) > {max_length}")

        if not conditions:
            raise RuleExecutionError("LENGTH rule requires min_length or max_length")

        # Combine conditions, also check NULL values (LENGTH(NULL) returns NULL)
        length_conditions = f"({' OR '.join(conditions)})"
        null_condition = f"{column} IS NULL"
        where_clause = f"({length_conditions} OR {null_condition})"

        # Add filter condition
        if filter_condition:
            where_clause = f"({where_clause}) AND ({filter_condition})"

        return f"SELECT COUNT(*) AS failed_count FROM {table} WHERE {where_clause}"
