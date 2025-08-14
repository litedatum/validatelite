"""
Shared enums module

Provides unified enum definitions to avoid magic strings, including:
1. Rule type enum
2. Execution status enum
3. Severity level enum
4. Connection type enum
5. Rule category enum
6. Rule action enum
"""

from .connection_types import ConnectionType
from .data_types import DataType
from .execution_status import ExecutionStatus
from .rule_actions import RuleAction
from .rule_categories import RuleCategory
from .rule_types import RuleType
from .severity_levels import SeverityLevel

__all__ = [
    "RuleType",
    "ExecutionStatus",
    "SeverityLevel",
    "ConnectionType",
    "RuleCategory",
    "RuleAction",
    "DataType",
]
