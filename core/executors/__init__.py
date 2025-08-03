"""
Executor registry - based on MVP_CLI_Extraction_Plan.md design

Implements a unified executor architecture as per the design document,
separated from the existing rule_type_registry
"""

from typing import Dict, List, Type

from shared.exceptions.exception_system import RuleExecutionError

from .base_executor import BaseExecutor
from .completeness_executor import CompletenessExecutor
from .uniqueness_executor import UniquenessExecutor
from .validity_executor import ValidityExecutor


class ExecutorRegistry:
    """
    Executor registry - architecture based on the design document

    Separated from the existing RuleTypeRegistry, focused on executor management
    """

    def __init__(self) -> None:
        """Initialize ExecutorRegistry"""
        self._executors: Dict[str, Type[BaseExecutor]] = {}
        self._register_builtin_executors()

    def _register_builtin_executors(self) -> None:
        """Register built-in executors"""
        self.register_executor("completeness", CompletenessExecutor)
        self.register_executor("validity", ValidityExecutor)
        self.register_executor("uniqueness", UniquenessExecutor)

    def register_executor(self, name: str, executor_class: Type[BaseExecutor]) -> None:
        """Register executor"""
        self._executors[name] = executor_class

    def get_executor_for_rule_type(self, rule_type: str) -> Type[BaseExecutor]:
        """
        Get executor based on rule type - as per design doc interface

        Args:
            rule_type: Rule type string, could be "RuleType.NOT_NULL" or "NOT_NULL"

        Returns:
            Type[BaseExecutor]: Executor class

        Raises:
            ValueError: If no corresponding executor is found
        """
        # Standardize rule type string - extract actual enum value
        if rule_type.startswith("RuleType."):
            rule_type_value = rule_type.split(".", 1)[1]  # Extract "NOT_NULL"
        else:
            rule_type_value = rule_type

        for executor_class in self._executors.values():
            if hasattr(executor_class, "SUPPORTED_TYPES"):
                supported_types = [t.value for t in executor_class.SUPPORTED_TYPES]
                if rule_type_value in supported_types:
                    return executor_class

        raise RuleExecutionError(f"No executor found for rule type: {rule_type}")

    def list_supported_types(self) -> List[str]:
        """List all supported rule types"""
        supported = []
        for executor_class in self._executors.values():
            if hasattr(executor_class, "SUPPORTED_TYPES"):
                supported.extend([t.value for t in executor_class.SUPPORTED_TYPES])
        return list(set(supported))


# Global executor registry - as per design document
executor_registry = ExecutorRegistry()

# Export interfaces required by the design document
__all__ = [
    "ExecutorRegistry",
    "executor_registry",
    "BaseExecutor",
    "CompletenessExecutor",
    "ValidityExecutor",
    "UniquenessExecutor",
]

"""
Required updates to existing code:

1. core/engine/rule_engine.py line 239:
   Before: executor_class = rule_type_registry.get_executor_class(rule_type)
   After:
   from core.executors import executor_registry
   executor_class = executor_registry.get_executor_for_rule_type(rule_type)

2. Executor interface adaptation:
   Existing code expects: executor_class(engine, database, table)
   Design doc expects: executor_class(connection)

   Suggestion: add adapter method in BaseExecutor

3. Separation of concerns:
   RuleTypeRegistry focuses on rule type metadata management
   ExecutorRegistry focuses on executor management
"""
