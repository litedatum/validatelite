"""
RuleSchema - Based on refactored RuleBase

Inherits from shared.schema.base.RuleBase, adds rule engine-specific methods
and properties.
Supports unified interface for single-table and multi-table rules.
"""

from typing import Any, Dict, Optional

from pydantic import field_validator, model_validator

from shared.enums import RuleAction, RuleCategory, RuleType, SeverityLevel
from shared.enums.data_types import DataType
from shared.exceptions.exception_system import RuleExecutionError
from shared.schema.base import (
    ExecutionStrategy,
    RuleBase,
    RuleTarget,
    TargetEntity,
)


class RuleSchema(RuleBase):
    """
    RuleSchema - Rule engine invocation interface

    Based on refactored RuleBase, provides rule engine-specific features:
    1. Unified single-table and multi-table rule interface
    2. Reserved hooks for cross-database support
    3. Strongly-typed parameter validation

    Note: connection_id is removed from top level.
    The connection context is provided at execution time by the RuleEngine.
    """

    # ==================== Convenient methods ====================

    def get_target_info(self) -> Dict[str, Optional[str]]:
        """
        Get primary target info (backward compatible)

        Returns:
            Dict[str, str]: Dictionary containing primary target info
        """
        primary = self.target.primary_entity
        return {
            "database": primary.database,
            "table": primary.table,
            "column": primary.column,
        }

    def get_full_table_name(self) -> str:
        """Get full table name"""
        primary = self.target.primary_entity
        return f"{primary.database}.{primary.table}"

    def get_target_column(self) -> Optional[str]:
        """Get target column name"""
        return self.target.primary_entity.column

    def get_rule_config(self) -> Dict[str, Any]:
        """
        Get rule configuration

        Filter out system-level parameters, return rule config
        """
        system_params = {
            "filter_condition",
            "database",
            "table",
            "table_name",
            "column",
            "column_name",
        }
        return {k: v for k, v in self.parameters.items() if k not in system_params}

    def get_filter_condition(self) -> Optional[str]:
        """
        Get filter condition

        Extract filter condition from parameters (backward compatible)
        """
        return self.parameters.get("filter_condition")

    def requires_column(self) -> bool:
        """Determine if column name is required"""
        column_required_types = [
            RuleType.NOT_NULL,
            RuleType.UNIQUE,
            RuleType.RANGE,
            RuleType.ENUM,
            RuleType.REGEX,
            RuleType.LENGTH,
            # RuleType.EMAIL, RuleType.PHONE - not supported in current version
        ]
        return self.type in column_required_types

    def is_mergeable_with(self, other: "RuleSchema") -> bool:
        """Determine if can be merged and executed with another rule"""
        # Must be the same table
        if self.get_full_table_name() != other.get_full_table_name():
            return False

        # Both must be single-table rules
        if not (self.target.is_single_table and other.target.is_single_table):
            return False

        # Both must be mergeable rule types
        mergeable_types = [
            RuleType.NOT_NULL,
            RuleType.RANGE,
            RuleType.ENUM,
            RuleType.REGEX,
            RuleType.LENGTH,
        ]
        if self.type not in mergeable_types or other.type not in mergeable_types:
            return False

        # Filter conditions must be the same
        self_filter = self.parameters.get("filter_condition")
        other_filter = other.parameters.get("filter_condition")
        if self_filter != other_filter:
            return False

        return True

    def to_engine_dict(self) -> Dict[str, Any]:
        """Convert to rule engine-specific dictionary format"""
        return {
            "id": self.id,  # id field now always exists
            "name": self.name,
            "type": self.type.value,
            "target": {
                "database": self.target.primary_entity.database,
                "table": self.target.primary_entity.table,
                "column": self.target.primary_entity.column,
            },
            "parameters": self.parameters,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "action": self.action.value,
            "is_active": self.is_active,
            "validation_error": self.validation_error,
        }

    @classmethod
    def from_engine_dict(cls, data: Dict[str, Any]) -> "RuleSchema":
        """
        Create from rule engine dictionary format

        - no longer needs connection_id after refactor
        """
        # Build target
        target_data = data.get("target", {})
        target = RuleTarget(
            entities=[
                TargetEntity(
                    database=target_data.get("database", "main"),
                    table=target_data.get("table", ""),
                    column=target_data.get("column"),
                    connection_id=None,
                    alias=None,
                )
            ],
            relationship_type="single_table",
        )

        return cls(
            name=data["name"],
            description=data.get("description"),
            type=RuleType(data["type"]),
            target=target,
            parameters=data.get("parameters", {}),
            cross_db_config=None,
            threshold=data.get("threshold"),
            category=RuleCategory.COMPLETENESS,  # default category
            severity=SeverityLevel(data.get("severity", "MEDIUM")),
            action=RuleAction(data.get("action", "LOG")),
            is_active=data.get("is_active", True),
            tags=data.get("tags", []),
            template_id=None,
            validation_error=data.get("validation_error"),
        )

    @classmethod
    def from_legacy_params(
        cls, rule_id: str, rule_name: str, rule_type: RuleType, params: Dict[str, Any]
    ) -> "RuleSchema":
        """Create from legacy params format (backward compatible)

        - no longer needs connection_id after refactor
        """
        # Extract target info from params
        database = params.pop("database", "main")
        table = params.pop("table_name", params.pop("table", ""))
        column = params.pop("column_name", params.pop("column", None))

        # Build target
        target = RuleTarget(
            entities=[
                TargetEntity(
                    database=database,
                    table=table,
                    column=column,
                    connection_id=None,
                    alias=None,
                )
            ],
            relationship_type="single_table",
        )

        return cls(
            name=rule_name,
            description=None,
            type=rule_type,
            target=target,
            parameters=params,  # remaining parameters
            cross_db_config=None,
            threshold=None,
            category=RuleCategory.COMPLETENESS,
            severity=SeverityLevel.MEDIUM,
            action=RuleAction.LOG,
            is_active=True,
            tags=None,
            template_id=None,
            validation_error=None,
        )

    # ==================== Rule validation methods ====================

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate rule parameters"""
        return v

    @model_validator(mode="after")
    def validate_rule_consistency(self) -> "RuleSchema":
        """Rule consistency validation"""
        # Commented out FOREIGN_KEY check, not supported in current version
        # if self.type in [RuleType.FOREIGN_KEY] and self.target.is_single_table:
        #     raise ValueError(f"{self.type} requires multiple tables")

        # Validate cross-database config
        if self.cross_db_config and not self.target.is_cross_database:
            # Current version warning, may raise error in future version
            pass

        # Validate parameter completeness
        self._validate_parameters_for_type()

        return self

    def _validate_parameters_for_type(self) -> None:
        """Validate parameters based on rule type"""
        params = self.parameters

        if self.type == RuleType.RANGE:
            # Explicitly retrieve parameter values so that falsy numeric values
            # (e.g., 0) are preserved
            min_value = (
                params.get("min_value") if "min_value" in params else params.get("min")
            )
            max_value = (
                params.get("max_value") if "max_value" in params else params.get("max")
            )

            # At least one boundary must be provided
            if min_value is None and max_value is None:
                raise RuleExecutionError(
                    "RANGE rule requires at least one of min_value/min or "
                    "max_value/max"
                )

            # If both boundaries exist, validate their numeric nature and logical
            # order
            if min_value is not None and max_value is not None:
                # First, ensure both values are numeric
                try:
                    min_val = float(min_value)
                    max_val = float(max_value)
                except (ValueError, TypeError):
                    raise RuleExecutionError(
                        "min_value and max_value must be numeric in RANGE rule"
                    )

                # Logical relationship: min must be <= max (equality allowed)
                if min_val > max_val:
                    raise RuleExecutionError(
                        "min_value must be less than or equal to max_value in "
                        "RANGE rule"
                    )

        elif self.type == RuleType.LENGTH:
            min_length = params.get("min_length")
            max_length = params.get("max_length")
            exact_length = params.get("exact_length")

            if min_length is None and max_length is None and exact_length is None:
                raise RuleExecutionError(
                    "LENGTH rule requires at least one of min_length, max_length, "
                    "or exact_length"
                )

        elif self.type == RuleType.ENUM:
            allowed_values = params.get("allowed_values")
            if not allowed_values:
                raise RuleExecutionError("ENUM rule requires allowed_values parameter")
            if not isinstance(allowed_values, list) or len(allowed_values) == 0:
                raise RuleExecutionError("allowed_values must be a non-empty list")

        elif self.type == RuleType.REGEX:
            import re

            pattern = params.get("pattern")
            if not pattern:
                raise RuleExecutionError("REGEX rule requires pattern parameter")

            try:
                re.compile(pattern)
            except re.error as e:
                raise RuleExecutionError(f"Invalid regex pattern: {e}")

        elif self.type == RuleType.SCHEMA:
            columns_cfg = params.get("columns")
            if not isinstance(columns_cfg, dict) or len(columns_cfg) == 0:
                raise RuleExecutionError(
                    "SCHEMA rule requires non-empty columns mapping"
                )
            for col_name, cfg in columns_cfg.items():
                if not isinstance(cfg, dict) or "expected_type" not in cfg:
                    raise RuleExecutionError(
                        f"Column '{col_name}' must specify expected_type for "
                        "SCHEMA rule"
                    )
                try:
                    DataType(str(cfg["expected_type"]).upper())
                except Exception:
                    raise RuleExecutionError(
                        f"Unsupported expected_type for SCHEMA column '{col_name}': "
                        f"{cfg.get('expected_type')}"
                    )

        # elif self.type == RuleType.CUSTOM_SQL:  # not supported in current version
        #     sql_query = params.get('sql_query') or params.get('custom_sql')
        #     if not sql_query:
        #         raise ValueError("CUSTOM_SQL rule requires sql_query parameter")

    def get_rule_category_name(self) -> str:
        """Get rule category name"""
        category_mapping = {
            RuleType.NOT_NULL: "completeness",
            RuleType.UNIQUE: "uniqueness",
            # RuleType.PRIMARY_KEY: "uniqueness",  # not supported in current version
            RuleType.RANGE: "validity",
            RuleType.LENGTH: "validity",
            RuleType.ENUM: "validity",
            RuleType.REGEX: "format",
            # RuleType.EMAIL: "format",  # not supported in current version
            # RuleType.PHONE: "format",  # not supported in current version
            # RuleType.URL: "format",  # not supported in current version
            RuleType.DATE_FORMAT: "format",
            RuleType.SCHEMA: "validity",
            # RuleType.COUNT: "statistical",  # not supported in current version
            # RuleType.SUM: "statistical",  # not supported in current version
            # RuleType.AVERAGE: "statistical",  # not supported in current version
            # RuleType.CUSTOM_SQL: "custom"  # not supported in current version
        }
        return category_mapping.get(self.type, "unknown")

    # ==================== Hook check methods ====================

    def check_cross_db_support(self) -> Dict[str, Any]:
        """Check cross-database support status"""
        result: Dict[str, Any] = {
            "supported": False,
            "reason": None,
            "suggestions": [],
        }

        if self.target.is_cross_database:
            result["reason"] = (
                "Cross-database rules will be supported in future versions"
            )
            result["suggestions"].append(
                "Please use single-table rules in the current version"
            )

        if self.cross_db_config:
            strategy = self.cross_db_config.execution_strategy
            if strategy != ExecutionStrategy.SQL_NATIVE:
                result["reason"] = (
                    f"Execution strategy {strategy} will be supported in future "
                    "versions"
                )
                result["suggestions"].append(
                    "Current version only supports SQL_NATIVE strategy"
                )

        if not result["reason"]:
            result["supported"] = True

        return result
