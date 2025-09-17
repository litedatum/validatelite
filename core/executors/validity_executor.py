"""
Validity rule executor - based on mature existing logic

Ported from mature validation logic in app/models/rule.py
Unified handling: RANGE, ENUM, REGEX and similar rules
"""

from datetime import datetime
from typing import Any, Dict, Optional

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
            # 对于SQLite，尝试使用自定义函数替代REGEX
            if (
                hasattr(self.dialect, "can_use_custom_functions")
                and self.dialect.can_use_custom_functions()
            ):
                return await self._execute_sqlite_custom_regex_rule(rule)
            else:
                raise RuleExecutionError(
                    f"REGEX rule is not supported for {self.dialect.__class__.__name__}"
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

        # Cast column for regex operations if needed (PostgreSQL requires casting
        # for non-text columns)
        regex_column = self.dialect.cast_column_for_regex(column)

        # Generate REGEXP expression using the dialect
        where_clause = f"WHERE {regex_column} {regex_op} '{escaped_pattern}'"

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

    async def _execute_sqlite_custom_regex_rule(
        self, rule: RuleSchema
    ) -> ExecutionResultSchema:
        """使用SQLite自定义函数执行REGEX规则的替代方案"""
        import time

        from shared.database.query_executor import QueryExecutor
        from shared.schema.base import DatasetMetrics

        start_time = time.time()
        table_name = self._safe_get_table_name(rule)

        try:
            # 生成使用自定义函数的SQL
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
        为SQLite生成使用自定义函数的验证SQL - 重构版本

        移除硬编码逻辑，基于规则配置动态确定验证类型
        """
        table = self._safe_get_table_name(rule)
        column = self._safe_get_column_name(rule)
        filter_condition = rule.get_filter_condition()

        # 动态确定验证类型和参数
        validation_info = self._determine_validation_type_from_rule(rule)

        # 根据验证类型生成验证条件
        validation_condition = self._generate_validation_condition_by_type(
            validation_info, column
        )

        # 构建WHERE子句
        where_clause = f"WHERE {validation_condition}"
        if filter_condition:
            where_clause += f" AND ({filter_condition})"

        return f"SELECT COUNT(*) AS anomaly_count FROM {table} {where_clause}"

    def _determine_validation_type_from_rule(self, rule: RuleSchema) -> dict:
        """根据规则配置动态确定验证类型和参数"""
        params = getattr(rule, "parameters", {})
        rule_config = rule.get_rule_config()

        # 优先从规则配置中获取验证类型信息
        validation_info: Dict[str, Any] = {
            "type": None,
            "parameters": {},
        }

        # 1. 检查是否有明确的验证类型配置
        if "validation_type" in params:
            validation_info["type"] = params["validation_type"]
            validation_info["parameters"] = params
        elif "validation_type" in rule_config:
            validation_info["type"] = rule_config["validation_type"]
            validation_info["parameters"] = rule_config

        # 2. 从desired_type字段推断验证类型（这是关键的缺失逻辑）
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

        # 3. 基于pattern推断验证类型
        elif "pattern" in params:
            validation_info = self._infer_validation_from_pattern(params["pattern"])
            # 如果pattern推断失败，尝试description推断
            if validation_info["type"] is None and "description" in params:
                validation_info = self._infer_validation_from_description(
                    params["description"]
                )
            # 合并其他参数
            validation_info["parameters"].update(params)

        # 4. 基于description推断验证类型
        elif "description" in params:
            validation_info = self._infer_validation_from_description(
                params["description"]
            )
            validation_info["parameters"].update(params)

        return validation_info

    def _infer_validation_from_desired_type(self, desired_type: str) -> dict:
        """从desired_type字段推断验证类型（如: 'integer(2)', 'float(4,1)', 'string(10)'）"""
        import re

        # 解析integer(N) 格式
        int_match = re.match(r"integer\((\d+)\)", desired_type)
        if int_match:
            max_digits = int(int_match.group(1))
            return {"type": "integer_digits", "parameters": {"max_digits": max_digits}}

        # 解析float(precision,scale) 格式
        float_match = re.match(r"float\((\d+),(\d+)\)", desired_type)
        if float_match:
            precision = int(float_match.group(1))
            scale = int(float_match.group(2))
            return {
                "type": "float_precision",
                "parameters": {"precision": precision, "scale": scale},
            }

        # 解析string(N) 格式
        string_match = re.match(r"string\((\d+)\)", desired_type)
        if string_match:
            max_length = int(string_match.group(1))
            return {"type": "string_length", "parameters": {"max_length": max_length}}

        # 解析基本类型
        if desired_type == "integer":
            return {"type": "integer_format", "parameters": {}}
        elif desired_type == "float":
            return {"type": "float_format", "parameters": {}}
        elif desired_type == "string":
            return {"type": "string_length", "parameters": {}}

        return {"type": None, "parameters": {}}

    def _infer_validation_from_pattern(self, pattern: str) -> dict:
        """从正则模式推断验证类型"""
        import re

        # 整数位数验证：^-?\\d{1,N}$ 或 ^-?[0-9]{1,N}$
        int_digits_match = re.search(
            r"\\\\d\\{1,(\\d+)\\}|\\[0-9\\]\\{1,(\\d+)\\}", pattern
        )
        if int_digits_match:
            max_digits = int(int_digits_match.group(1) or int_digits_match.group(2))
            return {"type": "integer_digits", "parameters": {"max_digits": max_digits}}

        # 字符串长度验证：^.{0,N}$
        str_length_match = re.search(r"\\.\\{0,(\\d+)\\}", pattern)
        if str_length_match:
            max_length = int(str_length_match.group(1))
            return {"type": "string_length", "parameters": {"max_length": max_length}}

        # 浮点数验证：包含小数点模式
        if r"\\." in pattern and any(x in pattern for x in [r"\\d", "[0-9]"]):
            # 检查是否是float到integer的转换（包含.0*模式）
            if r"\\.0\\*" in pattern or r"\\.0+" in pattern:
                return {"type": "float_to_integer", "parameters": {}}
            return {"type": "float_format", "parameters": {}}

        return {"type": None, "parameters": {}}

    def _infer_validation_from_description(self, description: str) -> dict:
        """从描述推断验证类型"""
        import re

        description_lower = description.lower()

        # Float precision/scale validation - 修复正则表达式
        if "precision/scale validation" in description_lower:
            # 匹配 "Float precision/scale validation for (4,1)" 格式
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
            # 尝试提取位数
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
        """根据验证类型信息生成验证条件"""
        validation_type = validation_info.get("type")
        params = validation_info.get("parameters", {})

        if not validation_type:
            return "1=0"  # 无验证条件

        from typing import cast

        from shared.database.database_dialect import SQLiteDialect

        sqlite_dialect = cast(SQLiteDialect, self.dialect)

        if validation_type == "integer_digits":
            max_digits = params.get("max_digits")
            if not max_digits:
                # 尝试从其他方法提取
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
                # 尝试从其他方法提取
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
            # 特殊情况：float到integer的验证，检查是否为整数
            return (
                f"typeof({column}) NOT IN ('integer', 'real') OR {column} "
                f"!= CAST({column} AS INTEGER)"
            )

        return "1=0"

    def _extract_digits_from_params(self, params: dict) -> Optional[int]:
        """从参数中提取数字位数信息"""
        if "max_digits" in params:
            return int(params["max_digits"])

        # 尝试从pattern参数中提取
        if "pattern" in params:
            pattern = params["pattern"]
            import re

            # 匹配 \\d{1,数字} 格式
            match = re.search(r"\\\\d\\{1,(\\d+)\\}", pattern)
            if match:
                return int(match.group(1))
            # 匹配 [0-9]{1,数字} 格式
            match = re.search(r"\\[0-9\\]\\{1,(\\d+)\\}", pattern)
            if match:
                return int(match.group(1))

        return None

    def _extract_length_from_params(self, params: dict) -> Optional[int]:
        """从参数中提取字符串长度信息"""
        if "max_length" in params:
            return int(params["max_length"])

        # 尝试从pattern参数中提取
        if "pattern" in params:
            pattern = params["pattern"]
            import re

            match = re.search(r"\\.\\{0,(\\d+)\\}", pattern)
            if match:
                return int(match.group(1))

        return None

    def _extract_digits_from_rule(self, rule: RuleSchema) -> Optional[int]:
        """从规则中提取数字位数信息"""
        # 首先尝试从参数中提取
        params = getattr(rule, "parameters", {})
        if "max_digits" in params:
            return int(params["max_digits"])

        # 尝试从pattern参数中提取（适用于REGEX规则）
        if "pattern" in params:
            pattern = params["pattern"]
            # 查找类似 '^-?\\d{1,5}$' 或 '^-?[0-9]{1,2}$' 的模式中的数字
            import re

            # 匹配 \d{1,数字} 格式
            match = re.search(r"\\d\{1,(\d+)\}", pattern)
            if match:
                return int(match.group(1))
            # 匹配 [0-9]{1,数字} 格式
            match = re.search(r"\[0-9\]\{1,(\d+)\}", pattern)
            if match:
                return int(match.group(1))

        # 尝试从规则名称中提取
        if hasattr(rule, "name") and rule.name:
            # 查找类似 "integer(5)" 或 "integer_digits_5" 的模式
            import re

            match = re.search(r"integer.*?(\d+)", rule.name)
            if match:
                return int(match.group(1))

        # 尝试从描述中提取
        description = params.get("description", "")
        if description:
            import re

            # 查找类似 "max 5 digits" 或 "validation for max 5 integer digits" 的模式
            match = re.search(r"max (\d+).*?digit", description)
            if match:
                return int(match.group(1))

        return None

    def _extract_length_from_rule(self, rule: RuleSchema) -> Optional[int]:
        """从规则中提取字符串长度信息"""
        # 首先尝试从参数中提取
        params = getattr(rule, "parameters", {})
        if "max_length" in params:
            return int(params["max_length"])

        # 尝试从pattern参数中提取（适用于REGEX规则）
        if "pattern" in params:
            pattern = params["pattern"]
            # 查找类似 '^.{0,10}$' 的模式中的数字
            import re

            match = re.search(r"\{0,(\d+)\}", pattern)
            if match:
                return int(match.group(1))

        # 尝试从规则名称中提取
        if hasattr(rule, "name") and rule.name:
            # 查找类似 "string(10)" 或 "length_10" 的模式
            import re

            match = re.search(r"(?:string|length).*?(\d+)", rule.name)
            if match:
                return int(match.group(1))

        # 尝试从描述中提取
        description = params.get("description", "")
        if description:
            import re

            # 查找类似 "max 10 characters" 或 "length validation for max 10" 的模式
            match = re.search(r"max (\d+).*?character", description)
            if match:
                return int(match.group(1))

        return None

    def _extract_float_precision_scale_from_description(
        self, description: str
    ) -> tuple[Optional[int], Optional[int]]:
        """从描述中提取float的precision和scale信息"""
        import re

        # 查找类似 "Float precision/scale validation for (4,1)" 的模式
        match = re.search(r"validation for \((\d+),(\d+)\)", description)
        if match:
            precision: Optional[int] = int(match.group(1))
            scale: Optional[int] = int(match.group(2))
            return precision, scale

        # 查找类似 "precision=4, scale=1" 的模式
        precision_match = re.search(
            r"precision[=:]?\s*(\d+)", description, re.IGNORECASE
        )
        scale_match = re.search(r"scale[=:]?\s*(\d+)", description, re.IGNORECASE)

        precision = int(precision_match.group(1)) if precision_match else None
        scale = int(scale_match.group(1)) if scale_match else None

        return precision, scale
