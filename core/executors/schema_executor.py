"""
Schema rule executor - Independent handling of table schema validation

Extracted from ValidityExecutor to provide dedicated schema validation logic.
Handles table-level existence and type checks with prioritization support.
"""

import time
from datetime import datetime
from typing import Optional

from shared.enums.data_types import DataType
from shared.enums.rule_types import RuleType
from shared.exceptions.exception_system import RuleExecutionError
from shared.schema.base import DatasetMetrics
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.result_schema import ExecutionResultSchema
from shared.schema.rule_schema import RuleSchema

from .base_executor import BaseExecutor


class SchemaExecutor(BaseExecutor):
    """
    Schema rule executor

    Dedicated executor for SCHEMA rule type that performs:
    1. Table existence validation
    2. Column existence validation
    3. Data type validation
    4. Strict mode validation (extra columns detection)
    """

    SUPPORTED_TYPES = [RuleType.SCHEMA]

    def __init__(
        self,
        connection: ConnectionSchema,
        test_mode: Optional[bool] = False,
        sample_data_enabled: Optional[bool] = None,
        sample_data_max_records: Optional[int] = None,
    ) -> None:
        """Initialize SchemaExecutor"""
        super().__init__(
            connection, test_mode, sample_data_enabled, sample_data_max_records
        )

    def supports_rule_type(self, rule_type: str) -> bool:
        """Check if the rule type is supported"""
        return rule_type in [t.value for t in self.SUPPORTED_TYPES]

    async def execute_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """Execute schema rule"""
        if rule.type == RuleType.SCHEMA:
            return await self._execute_schema_rule(rule)
        else:
            raise RuleExecutionError(f"Unsupported rule type: {rule.type}")

    async def _execute_schema_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """Execute SCHEMA rule (table-level existence and type checks).

        Additionally attaches per-column details into the execution plan so the
        CLI can apply prioritization/skip semantics:

        execution_plan.schema_details = {
            "field_results": [
                {"column": str, "existence": "PASSED|FAILED", "type": "PASSED|FAILED",
                 "failure_code": "FIELD_MISSING|TYPE_MISMATCH|NONE"}
            ],
            "extras": ["<extra_column>", ...]  # present when strict_mode
        }
        """
        from shared.database.query_executor import QueryExecutor

        start_time = time.time()
        table_name = self._safe_get_table_name(rule)

        try:
            engine = await self.get_engine()
            query_executor = QueryExecutor(engine)

            # Expected columns and switches
            params = rule.get_rule_config()
            columns_cfg = params.get("columns") or {}
            case_insensitive = bool(params.get("case_insensitive", False))
            strict_mode = bool(params.get("strict_mode", False))

            # Fetch actual columns once
            target = rule.get_target_info()
            database = target.get("database")

            actual_columns = await query_executor.get_column_list(
                table_name=table_name,
                database=database,
                entity_name=table_name,
                rule_id=rule.id,
            )

            def key_of(name: str) -> str:
                return name.lower() if case_insensitive else name

            # Standardize actual columns into dict name->type (respecting
            # case-insensitive flag)
            actual_map = {
                key_of(c["name"]): str(c.get("type", "")).upper()
                for c in actual_columns
            }

            # Helper: map vendor-specific type to canonical DataType
            def map_to_datatype(vendor_type: str) -> str | None:
                t = vendor_type.upper().strip()
                # Trim length/precision and extras
                for sep in ["(", " "]:
                    if sep in t:
                        t = t.split(sep, 1)[0]
                        break
                # Common mappings
                string_types = {
                    "CHAR",
                    "CHARACTER",
                    "NCHAR",
                    "NVARCHAR",
                    "VARCHAR",
                    "VARCHAR2",
                    "TEXT",
                    "CLOB",
                }
                integer_types = {
                    "INT",
                    "INTEGER",
                    "BIGINT",
                    "SMALLINT",
                    "MEDIUMINT",
                    "TINYINT",
                }
                float_types = {
                    "FLOAT",
                    "DOUBLE",
                    "REAL",
                    "DECIMAL",
                    "NUMERIC",
                }
                boolean_types = {"BOOLEAN", "BOOL", "BIT"}
                if t in string_types:
                    return DataType.STRING.value
                if t in integer_types:
                    return DataType.INTEGER.value
                if t in float_types:
                    return DataType.FLOAT.value
                if t in boolean_types:
                    return DataType.BOOLEAN.value
                if t == "DATE":
                    return DataType.DATE.value
                if t.startswith("TIMESTAMP") or t in {"DATETIME", "DATETIME2"}:
                    return DataType.DATETIME.value
                return None

            # Count failures across declared columns and strict-mode extras
            total_declared = len(columns_cfg)
            failures = 0
            field_results: list[dict[str, str]] = []

            for declared_name, cfg in columns_cfg.items():
                expected_type_raw = cfg.get("expected_type")
                if expected_type_raw is None:
                    raise RuleExecutionError(
                        "SCHEMA rule requires expected_type for each column"
                    )
                # Validate expected type against DataType
                try:
                    expected_type = DataType(str(expected_type_raw).upper()).value
                except Exception:
                    raise RuleExecutionError(
                        f"Unsupported expected_type for SCHEMA: {expected_type_raw}"
                    )

                lookup_key = key_of(declared_name)
                # Existence check
                if lookup_key not in actual_map:
                    failures += 1
                    field_results.append(
                        {
                            "column": declared_name,
                            "existence": "FAILED",
                            "type": "SKIPPED",
                            "failure_code": "FIELD_MISSING",
                        }
                    )
                    continue

                # Type check
                actual_vendor_type = actual_map[lookup_key]
                actual_canonical = (
                    map_to_datatype(actual_vendor_type) or actual_vendor_type
                )
                if actual_canonical != expected_type:
                    failures += 1
                    field_results.append(
                        {
                            "column": declared_name,
                            "existence": "PASSED",
                            "type": "FAILED",
                            "failure_code": "TYPE_MISMATCH",
                        }
                    )
                else:
                    field_results.append(
                        {
                            "column": declared_name,
                            "existence": "PASSED",
                            "type": "PASSED",
                            "failure_code": "NONE",
                        }
                    )

            if strict_mode:
                # Fail for extra columns not declared
                declared_keys = {key_of(k) for k in columns_cfg.keys()}
                actual_keys = set(actual_map.keys())
                extras = actual_keys - declared_keys
                failures += len(extras)
            else:
                extras = set()

            execution_time = time.time() - start_time

            # For table-level schema rule, interpret total_records as number of
            # declared columns
            dataset_metric = DatasetMetrics(
                entity_name=table_name,
                total_records=total_declared,
                failed_records=failures,
                processing_time=execution_time,
            )

            status = "PASSED" if failures == 0 else "FAILED"

            return ExecutionResultSchema(
                rule_id=rule.id,
                status=status,
                dataset_metrics=[dataset_metric],
                execution_time=execution_time,
                execution_message=(
                    "SCHEMA check passed"
                    if failures == 0
                    else f"SCHEMA check failed: {failures} issues"
                ),
                error_message=None,
                sample_data=None,
                cross_db_metrics=None,
                execution_plan={
                    "execution_type": "metadata",
                    "schema_details": {
                        "field_results": field_results,
                        "extras": sorted(extras) if extras else [],
                    },
                },
                started_at=datetime.fromtimestamp(start_time),
                ended_at=datetime.fromtimestamp(time.time()),
            )

        except Exception as e:
            return await self._handle_execution_error(e, rule, start_time, table_name)