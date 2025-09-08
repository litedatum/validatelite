"""
Schema rule executor - Independent handling of table schema validation

Extracted from ValidityExecutor to provide dedicated schema validation logic.
Handles table-level existence and type checks with prioritization support.
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional

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

    def _extract_type_metadata(self, vendor_type: str) -> Dict[str, Any]:
        """Extract metadata (length, precision, scale) from vendor-specific type string.

        Examples:
        - VARCHAR(255) → {"canonical_type": "STRING", "max_length": 255}
        - DECIMAL(10,2) → {"canonical_type": "FLOAT", "precision": 10, "scale": 2}
        - INTEGER → {"canonical_type": "INTEGER"}
        """
        import re

        vendor_type = vendor_type.upper().strip()
        metadata: Dict[str, Any] = {"canonical_type": None}

        # Extract length/precision pattern: TYPE(length) or TYPE(precision,scale)
        match = re.match(r"^([A-Z_]+)(?:\((\d+)(?:,(\d+))?\))?", vendor_type)
        if not match:
            return metadata

        base_type = match.group(1)
        length_or_precision = match.group(2)
        scale = match.group(3)

        # Map base type to canonical type
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
        integer_types = {"INT", "INTEGER", "BIGINT", "SMALLINT", "MEDIUMINT", "TINYINT"}
        float_types = {"FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC"}
        boolean_types = {"BOOLEAN", "BOOL", "BIT"}

        if base_type in string_types:
            metadata["canonical_type"] = DataType.STRING.value
            if length_or_precision:
                metadata["max_length"] = int(length_or_precision)
        elif base_type in integer_types:
            metadata["canonical_type"] = DataType.INTEGER.value
        elif base_type in float_types:
            metadata["canonical_type"] = DataType.FLOAT.value
            if length_or_precision:
                metadata["precision"] = int(length_or_precision)
            if scale:
                metadata["scale"] = int(scale)
        elif base_type in boolean_types:
            metadata["canonical_type"] = DataType.BOOLEAN.value
        elif base_type == "DATE":
            metadata["canonical_type"] = DataType.DATE.value
        elif base_type.startswith("TIMESTAMP") or base_type in {
            "DATETIME",
            "DATETIME2",
        }:
            metadata["canonical_type"] = DataType.DATETIME.value

        return metadata

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

            try:
                actual_columns = await query_executor.get_column_list(
                    table_name=table_name,
                    database=database,
                    entity_name=table_name,
                    rule_id=rule.id,
                )
            except Exception as table_error:
                # Table doesn't exist or cannot be accessed
                # Return a table-level failure without column-level details
                execution_time = time.time() - start_time
                total_declared = len(columns_cfg)

                dataset_metric = DatasetMetrics(
                    entity_name=table_name,
                    total_records=0,  # No records exist if table doesn't exist
                    failed_records=total_declared,  # All checks fail if no table
                    processing_time=execution_time,
                )

                return ExecutionResultSchema(
                    rule_id=rule.id,
                    status="FAILED",
                    dataset_metrics=[dataset_metric],
                    execution_time=execution_time,
                    execution_message=(
                        f"Table '{table_name}' does not exist or cannot be accessed"
                    ),
                    error_message=str(table_error),
                    sample_data=None,
                    cross_db_metrics=None,
                    execution_plan={
                        "execution_type": "metadata",
                        "schema_details": {
                            "field_results": [],  # No results when table missing
                            "extras": [],
                            "table_exists": False,
                        },
                    },
                    started_at=datetime.fromtimestamp(start_time),
                    ended_at=datetime.fromtimestamp(time.time()),
                )

            def key_of(name: str) -> str:
                return name.lower() if case_insensitive else name

            # Standardize actual columns into dict name->metadata (respecting
            # case-insensitive flag)
            actual_map = {}
            for c in actual_columns:
                col_name = key_of(c["name"])
                col_type = str(c.get("type", "")).upper()
                metadata = self._extract_type_metadata(col_type)

                # Use database metadata if available, fallback to parsed type metadata
                max_length = c.get("character_maximum_length")
                if max_length is None:
                    max_length = metadata.get("max_length")

                precision = c.get("numeric_precision")
                if precision is None:
                    precision = metadata.get("precision")

                scale = c.get("numeric_scale")
                if scale is None:
                    scale = metadata.get("scale")

                actual_map[col_name] = {
                    "type": col_type,
                    "canonical_type": metadata["canonical_type"],
                    "max_length": max_length,
                    "precision": precision,
                    "scale": scale,
                }

            def compare_metadata(
                expected_cfg: Dict[str, Any], actual_meta: Dict[str, Any]
            ) -> Dict[str, Any]:
                """Compare expected metadata with actual metadata.

                Returns dict with validation results and failure details.
                """
                result: Dict[str, Any] = {
                    "type_status": "UNKNOWN",
                    "metadata_status": "UNKNOWN",
                    "failure_details": [],
                }

                # Type validation
                expected_type = expected_cfg.get("expected_type")
                actual_canonical = actual_meta.get("canonical_type")

                if actual_canonical == expected_type:
                    result["type_status"] = "PASSED"
                else:
                    result["type_status"] = "FAILED"
                    result["failure_details"].append(
                        f"Type mismatch: expected {expected_type}, "
                        f"got {actual_canonical}"
                    )

                # Only validate metadata if type matches
                if result["type_status"] == "PASSED":
                    metadata_failures = []

                    # String length validation
                    if (
                        expected_type == DataType.STRING.value
                        and "max_length" in expected_cfg
                    ):
                        expected_length = expected_cfg["max_length"]
                        actual_length = actual_meta.get("max_length")
                        if actual_length is None:
                            metadata_failures.append(
                                f"Expected max_length {expected_length}, "
                                f"but actual type has no length limit"
                            )
                        elif actual_length != expected_length:
                            metadata_failures.append(
                                f"Length mismatch: expected {expected_length}, "
                                f"got {actual_length}"
                            )

                    # Float precision/scale validation
                    if expected_type == DataType.FLOAT.value:
                        if "precision" in expected_cfg:
                            expected_precision = expected_cfg["precision"]
                            actual_precision = actual_meta.get("precision")
                            if actual_precision != expected_precision:
                                metadata_failures.append(
                                    f"Precision mismatch: expected "
                                    f"{expected_precision}, got {actual_precision}"
                                )

                        if "scale" in expected_cfg:
                            expected_scale = expected_cfg["scale"]
                            actual_scale = actual_meta.get("scale")
                            if actual_scale != expected_scale:
                                metadata_failures.append(
                                    f"Scale mismatch: expected {expected_scale}, "
                                    f"got {actual_scale}"
                                )

                    result["metadata_status"] = (
                        "PASSED" if not metadata_failures else "FAILED"
                    )
                    result["failure_details"].extend(metadata_failures)
                else:
                    result["metadata_status"] = "SKIPPED"

                return result

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

                # Enhanced metadata validation
                actual_meta = actual_map[lookup_key]
                expected_cfg = {
                    "expected_type": expected_type,
                    **{
                        k: v
                        for k, v in cfg.items()
                        if k in ["max_length", "precision", "scale"]
                    },
                }

                comparison_result = compare_metadata(expected_cfg, actual_meta)

                if comparison_result["type_status"] == "FAILED":
                    failures += 1
                    field_results.append(
                        {
                            "column": declared_name,
                            "existence": "PASSED",
                            "type": "FAILED",
                            "failure_code": "TYPE_MISMATCH",
                            "failure_details": comparison_result["failure_details"],
                        }
                    )
                elif comparison_result["metadata_status"] == "FAILED":
                    failures += 1
                    field_results.append(
                        {
                            "column": declared_name,
                            "existence": "PASSED",
                            "type": "PASSED",
                            "failure_code": "METADATA_MISMATCH",
                            "failure_details": comparison_result["failure_details"],
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
                        "table_exists": True,
                    },
                },
                started_at=datetime.fromtimestamp(start_time),
                ended_at=datetime.fromtimestamp(time.time()),
            )

        except Exception as e:
            return await self._handle_execution_error(e, rule, start_time, table_name)
