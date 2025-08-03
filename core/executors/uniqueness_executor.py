"""
Uniqueness rule executor - based on mature existing logic

Ported from mature validation logic in app/models/rule.py
Unified handling: UNIQUE and similar rules
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from shared.enums.rule_types import RuleType
from shared.exceptions.exception_system import RuleExecutionError
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.result_schema import ExecutionResultSchema
from shared.schema.rule_schema import RuleSchema

from .base_executor import BaseExecutor


class UniquenessExecutor(BaseExecutor):
    """
    Uniqueness rule executor

    Based on mature logic in app.models.rule.Rule
    Unified handling: UNIQUE and similar rules
    """

    SUPPORTED_TYPES = [RuleType.UNIQUE]

    def __init__(
        self,
        connection: ConnectionSchema,
        test_mode: Optional[bool] = False,
        sample_data_enabled: Optional[bool] = None,
        sample_data_max_records: Optional[int] = None,
    ) -> None:
        """Initialize UniquenessExecutor"""
        super().__init__(
            connection, test_mode, sample_data_enabled, sample_data_max_records
        )

    def supports_rule_type(self, rule_type: str) -> bool:
        """Check if the rule type is supported"""
        return rule_type in [t.value for t in self.SUPPORTED_TYPES]

    async def execute_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """Execute uniqueness rule"""
        if rule.type == RuleType.UNIQUE:
            return await self._execute_unique_rule(rule)
        else:
            raise RuleExecutionError(f"Unsupported rule type: {rule.type}")

    async def _execute_unique_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """
        Execute UNIQUE rule

        Based on mature logic from Rule._generate_unique_sql
        """
        import time

        from shared.database.query_executor import QueryExecutor
        from shared.schema.base import DatasetMetrics

        start_time = time.time()
        table_name = self._safe_get_table_name(rule)

        try:
            # Generate validation SQL
            sql = self._generate_unique_sql(rule)

            # Execute SQL and get result
            engine = await self.get_engine()
            query_executor = QueryExecutor(engine)

            # Get duplicate group count (this SQL returns the number of duplicate
            # groups)
            result, _ = await query_executor.execute_query(sql)
            duplicate_groups = (
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

            # If there are duplicate groups, need to calculate the actual number
            # of duplicate records
            failed_count = 0
            if duplicate_groups > 0:
                # Calculate the actual number of duplicate records
                column = self._safe_get_column_name(rule)
                detailed_sql = f"""
                SELECT SUM(cnt - 1) as duplicate_records_count
                FROM (
                    SELECT {column}, COUNT(*) as cnt
                    FROM {table_name}
                    WHERE {column} IS NOT NULL
                """
                if filter_condition:
                    detailed_sql += f" AND ({filter_condition})"

                detailed_sql += f"""
                    GROUP BY {column}
                    HAVING COUNT(*) > 1
                ) duplicates
                """

                detailed_result, _ = await query_executor.execute_query(detailed_sql)
                failed_count = (
                    detailed_result[0]["duplicate_records_count"]
                    if detailed_result and len(detailed_result) > 0
                    else 0
                )
                failed_count = failed_count or 0  # Ensure not None

            execution_time = time.time() - start_time

            # Build standardized result
            status = "PASSED" if failed_count == 0 else "FAILED"

            # Generate sample data (only on failure)
            sample_data = None
            if failed_count > 0:
                sample_data = await self._generate_unique_sample_data(
                    rule, table_name, filter_condition
                )

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
                    f"UNIQUE check completed, found {failed_count} duplicate records"
                    if failed_count > 0
                    else "UNIQUE check passed"
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

    def _generate_unique_sql(self, rule: RuleSchema) -> str:
        """
        Generate UNIQUE validation SQL

        Ported from app/models/rule.Rule._generate_unique_sql
        """
        # Use safe method to get table and column names
        table = self._safe_get_table_name(rule)
        column = self._safe_get_column_name(rule)
        filter_condition = rule.get_filter_condition()

        # Base query: find duplicate values
        base_sql = f"""
        SELECT {column}, COUNT(*) as cnt
        FROM {table}
        WHERE {column} IS NOT NULL
        """

        # Add filter condition
        if filter_condition:
            base_sql += f" AND ({filter_condition})"

        # Complete duplicate check SQL - returns the number of duplicate groups
        sql = f"""
        SELECT COUNT(*) AS anomaly_count
        FROM (
            {base_sql}
            GROUP BY {column}
            HAVING COUNT(*) > 1
        ) duplicates
        """

        return sql

    async def _generate_unique_sample_data(
        self, rule: RuleSchema, table_name: str, filter_condition: Optional[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate sample data for uniqueness rule

        Get some duplicate records as samples

        Args:
            rule: Rule object
            table_name: Table name
            filter_condition: Filter condition

        Returns:
            List of sample data, or None if no duplicate records
        """
        from shared.database.query_executor import QueryExecutor

        # Check sample data switch
        if not self.sample_data_enabled:
            return None

        try:
            column = self._safe_get_column_name(rule)

            # Build sample SQL: get some duplicate records
            sample_sql = f"""
            SELECT *
            FROM (
                SELECT {column}, COUNT(*) as dup_cnt
                FROM {table_name}
                WHERE {column} IS NOT NULL"""

            if filter_condition:
                sample_sql += f" AND ({filter_condition})"

            sample_sql += f"""
                GROUP BY {column}
                HAVING COUNT(*) > 1
            ) as duplicates
            """

            # Get more records, as there may be multiple duplicate values
            sample_sql += f" ORDER BY {column} LIMIT {self.sample_data_max_records * 2}"

            # Execute sample query
            engine = await self.get_engine()
            query_executor = QueryExecutor(engine)
            sample_result, _ = await query_executor.execute_query(sample_sql)

            # Return sample data
            return sample_result if sample_result else None

        except Exception as e:
            # Log warning if sampling fails but do not affect main flow
            self.logger.warning(
                f"Failed to generate unique sample data for rule {rule.id}: {str(e)}"
            )
            return None
