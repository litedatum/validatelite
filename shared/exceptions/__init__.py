"""
Data quality application exception handling system (v2.0)

Unified exception handling system classified by impact scope and nature:
1. System-level exception (EngineError) - affects the entire system operation,
    must stop all operations
2. Operation-level exception (OperationError) - affects a single operation,
    other operations can continue
3. Resource-level exception (RuleExecutionError) - affects a specific resource,
    other resources can continue to be processed
"""

# Base exception class; Three core exception classes (using original names);
# Exception converter; Exception handling strategy
from .exception_system import (
    DatabaseExceptionConverter,
    DataQualityException,
    EngineError,
    ExceptionHandler,
    OperationError,
    RuleExecutionError,
)

__all__ = [
    # Base exception class
    "DataQualityException",
    # Three core exception classes
    "EngineError",
    "OperationError",
    "RuleExecutionError",
    # Exception converter
    "DatabaseExceptionConverter",
    # Exception handling strategy
    "ExceptionHandler",
]
