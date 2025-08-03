"""
Schema module - core data transfer objects

Provides standardized interface schemas for the rule engine:
1. RuleSchema: Rule definition interface (based on RuleBase)
2. ConnectionSchema: Database connection interface (based on ConnectionBase)
3. ExecutionResultSchema: Execution result interface (based on ExecutionResultBase)

All schemas inherit from tested base classes to maintain compatibility
with existing APIs.
"""

from .base import BaseSchema, ConnectionBase, ExecutionResultBase, RuleBase
from .connection_schema import ConnectionSchema
from .result_schema import ExecutionResultSchema
from .rule_schema import RuleSchema

__all__ = [
    "BaseSchema",
    "RuleBase",
    "ConnectionBase",
    "ExecutionResultBase",
    "RuleSchema",
    "ConnectionSchema",
    "ExecutionResultSchema",
]
