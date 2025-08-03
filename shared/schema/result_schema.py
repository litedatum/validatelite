"""
ExecutionResultSchema - Based on refactored ExecutionResultBase

Inherits from shared.schema.base.ExecutionResultBase, adds rule engine-specific
methods and properties.
Supports unified expression of single-table and multi-table rule results.
"""

from typing import Any, Dict, List, Optional

from shared.enums import ExecutionStatus
from shared.schema.base import DatasetMetrics, ExecutionResultBase
from shared.utils.datetime_utils import format_datetime, now


class ExecutionResultSchema(ExecutionResultBase):
    """
    ExecutionResultSchema - Rule execution result interface

    Based on refactored ExecutionResultBase, provides rule engine-specific features:
    1. Unified single-table and multi-table result expression
    2. Reserved hooks for cross-database metrics
    3. Backward-compatible property access
    """

    # ==================== Convenient methods ====================

    def get_success_rate(self) -> float:
        """Calculate success rate"""
        total = self.total_count
        if total == 0:
            return 1.0
        return (total - self.error_count) / total

    def get_failure_rate(self) -> float:
        """Calculate failure rate"""
        return 1.0 - self.get_success_rate()

    def is_success(self) -> bool:
        """Determine if successful"""
        return self.status == ExecutionStatus.PASSED.value

    def is_failure(self) -> bool:
        """Determine if failed"""
        return self.status == ExecutionStatus.FAILED.value

    def is_error(self) -> bool:
        """Determine if error"""
        return self.status == ExecutionStatus.ERROR.value

    def get_entity_name(self) -> str:
        """
        Get entity name (for CLI error classification support)

        Returns:
            str: Entity name. If there are multiple entities, returns the first one.
            If none, returns "unknown"
        """
        if hasattr(self, "dataset_metrics") and self.dataset_metrics:
            return self.dataset_metrics[0].entity_name
        return "unknown"

    def get_error_classification_hints(self) -> Dict[str, str]:
        """
        Get error classification hints (for CLI to provide better classification basis)

        Based on error message and context information, provide more accurate
        error classification hints.
        Matches error message formats generated in rule_engine.py and
        DatabaseExceptionConverter.
        Note: This method is used for CLI error classification support.

        Returns:
            Dict[str, str]: Classification hint information
        """
        if self.status != "ERROR":
            return {}

        hints = {}
        error_msg = self.error_message.lower() if self.error_message else ""

        # 1. Match validation error message format (_create_validation_error_result)
        if "table" in error_msg and any(
            keyword in error_msg
            for keyword in ["not exist", "does not exist", "not found"]
        ):
            hints["resource_type"] = "table"
            hints["error_type"] = "not_found"
        elif "column" in error_msg and any(
            keyword in error_msg
            for keyword in ["not exist", "does not exist", "not found"]
        ):
            hints["resource_type"] = "column"
            hints["error_type"] = "not_found"

        # 2. Match execution error message format (_create_error_results_for_rules)
        elif "rule execution failed" in error_msg:
            if "syntax" in error_msg or "sql" in error_msg:
                hints["error_type"] = "syntax"
            elif "timeout" in error_msg or "timed out" in error_msg:
                hints["error_type"] = "timeout"
            elif "type" in error_msg or "data type" in error_msg:
                hints["error_type"] = "data_type"
            elif "permission" in error_msg or "access denied" in error_msg:
                hints["error_type"] = "permission"

        # 3. Match merged execution error message format
        elif "merged execution failed" in error_msg:
            if "syntax" in error_msg or "sql" in error_msg:
                hints["error_type"] = "syntax"
            elif "timeout" in error_msg or "timed out" in error_msg:
                hints["error_type"] = "timeout"

        # 4. Match error message format from DatabaseExceptionConverter
        elif "database system error" in error_msg:
            if "timeout" in error_msg:
                hints["error_type"] = "timeout"
            elif "connection" in error_msg:
                hints["error_type"] = "connection"
        elif "sql syntax error" in error_msg:
            hints["error_type"] = "syntax"
        elif "database operation failed" in error_msg:
            hints["error_type"] = "operation"

        # 5. General keyword matching (as fallback)
        else:
            # Resource type determination
            if "table" in error_msg:
                hints["resource_type"] = "table"
            elif "column" in error_msg or "field" in error_msg:
                hints["resource_type"] = "column"

            # Error type determination
            if any(keyword in error_msg for keyword in ["syntax", "sql"]):
                hints["error_type"] = "syntax"
            elif any(
                keyword in error_msg for keyword in ["timeout", "time out", "timed out"]
            ):
                hints["error_type"] = "timeout"
            elif any(keyword in error_msg for keyword in ["type", "data type"]):
                hints["error_type"] = "data_type"
            elif any(
                keyword in error_msg for keyword in ["permission", "access", "denied"]
            ):
                hints["error_type"] = "permission"

        # Add entity name
        entity_name = self.get_entity_name()
        if entity_name != "unknown":
            hints["entity_name"] = entity_name

        return hints

    def to_engine_dict(self) -> Dict[str, Any]:
        """Convert to rule engine-specific dictionary format"""
        return {
            "rule_id": self.rule_id,
            "status": self.status,
            "total_records": self.total_count,
            "failed_records": self.error_count,
            "success_rate": self.get_success_rate(),
            "execution_time": self.execution_time,
            "message": self.execution_message,
            "error_message": self.error_message,
            "sample_data": self.sample_data,
            "executed_at": format_datetime(self.ended_at) if self.ended_at else None,
        }

    @classmethod
    def create_success_result(
        cls,
        rule_id: str,
        entity_name: str,
        total_count: int,
        error_count: int = 0,
        execution_time: float = 0.0,
        message: Optional[str] = None,
    ) -> "ExecutionResultSchema":
        """Create success result"""
        status = (
            ExecutionStatus.PASSED.value
            if error_count == 0
            else ExecutionStatus.FAILED.value
        )

        if message is None:
            success_rate = (
                (total_count - error_count) / total_count if total_count > 0 else 1.0
            )
            message = f"Validation completed, success rate {success_rate:.1%}"

        dataset_metric = DatasetMetrics(
            entity_name=entity_name,
            total_records=total_count,
            failed_records=error_count,
            processing_time=None,
        )

        return cls(
            rule_id=rule_id,
            status=status,
            dataset_metrics=[dataset_metric],
            execution_time=execution_time,
            execution_message=message,
            error_message=None,
            sample_data=None,
            cross_db_metrics=None,
            execution_plan=None,
            started_at=now(),
            ended_at=now(),
        )

    @classmethod
    def create_error_result(
        cls,
        rule_id: str,
        entity_name: str,
        error_message: str,
        execution_time: float = 0.0,
    ) -> "ExecutionResultSchema":
        """Create error result"""
        dataset_metric = DatasetMetrics(
            entity_name=entity_name,
            total_records=0,
            failed_records=0,
            processing_time=None,
        )

        return cls(
            rule_id=rule_id,
            status=ExecutionStatus.ERROR.value,
            dataset_metrics=[dataset_metric],
            execution_time=execution_time,
            execution_message=None,
            error_message=error_message,
            sample_data=None,
            cross_db_metrics=None,
            execution_plan=None,
            started_at=now(),
            ended_at=now(),
        )

    @classmethod
    def create_from_legacy(
        cls,
        rule_id: str,
        status: str,
        total_count: int,
        error_count: int,
        execution_time: float,
        database: str,
        table: str,
        execution_message: Optional[str] = None,
        error_message: Optional[str] = None,
        sample_data: Optional[List[Dict[str, Any]]] = None,
    ) -> "ExecutionResultSchema":
        """Create result from legacy format (backward compatible)"""
        entity_name = f"{database}.{table}"

        dataset_metric = DatasetMetrics(
            entity_name=entity_name,
            total_records=total_count,
            failed_records=error_count,
            processing_time=None,
        )

        return cls(
            rule_id=rule_id,
            status=status,
            dataset_metrics=[dataset_metric],
            execution_time=execution_time,
            execution_message=execution_message,
            error_message=error_message,
            sample_data=sample_data,
            cross_db_metrics=None,
            execution_plan=None,
            started_at=now(),
            ended_at=now(),
        )

    def get_summary(self) -> str:
        """Get result summary"""
        if self.is_error():
            return f"Error: {self.error_message}"
        elif self.is_failure():
            return (
                f"Failure: {self.error_count}/{self.total_count} records do not "
                "meet the rule"
            )
        else:
            return f"Success: {self.total_count} records all passed validation"

    def get_detailed_message(self) -> str:
        """Get detailed message"""
        parts = [self.get_summary()]

        if not self.is_error():
            parts.append(f"Success rate: {self.get_success_rate():.2%}")

        if self.execution_time > 0:
            parts.append(f"Execution time: {self.execution_time:.3f}s")

        return " | ".join(parts)

    # ==================== Multi-dataset support methods ====================

    def get_dataset_by_name(self, entity_name: str) -> Optional[DatasetMetrics]:
        """Get dataset metrics by entity name"""
        for metric in self.dataset_metrics:
            if metric.entity_name == entity_name:
                return metric
        return None

    def add_dataset_metric(self, metric: DatasetMetrics) -> None:
        """Add dataset metric (used for multi-table results)"""
        self.dataset_metrics.append(metric)

    def get_cross_db_summary(self) -> Optional[Dict[str, Any]]:
        """Get cross-database execution summary"""
        if not self.cross_db_metrics:
            return None

        return {
            "strategy": self.cross_db_metrics.strategy_used,
            "transfer_time": self.cross_db_metrics.data_transfer_time,
            "processing_time": self.cross_db_metrics.total_processing_time,
            "data_size_mb": self.cross_db_metrics.temp_data_size_mb,
        }

    # ==================== Hook check methods ====================

    def check_multi_table_support(self) -> Dict[str, Any]:
        """Check multi-table support status"""
        result: Dict[str, Any] = {
            "supported": True,
            "dataset_count": len(self.dataset_metrics),
            "is_cross_database": bool(self.cross_db_metrics),
        }

        if len(self.dataset_metrics) > 1 and not self.cross_db_metrics:
            result["warnings"] = [
                "Multiple dataset results but cross-database metrics not enabled"
            ]

        return result
