"""
Data quality application exception handling system (v2.0)
Classified by impact scope and nature, implements a unified exception handling system
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

# ==================== Base Exception Interface ====================


class DataQualityException(Exception, ABC):
    """Base class for data quality exceptions"""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """Initialize DataQualityException"""
        self.message = message
        self.context = context or {}
        self.cause = cause
        super().__init__(message)

        # Automatically add timestamp and exception info
        self.context.setdefault("timestamp", datetime.now().isoformat())
        self.context.setdefault("exception_type", self.__class__.__name__)

    @abstractmethod
    def get_impact_level(self) -> str:
        """Return impact level: SYSTEM, OPERATION, RESOURCE"""
        pass

    @abstractmethod
    def should_stop_execution(self) -> bool:
        """Whether execution should be stopped"""
        pass

    def get_context_info(self) -> Dict[str, Any]:
        """Get rich context information"""
        return {
            "message": self.message,
            "impact_level": self.get_impact_level(),
            "should_stop": self.should_stop_execution(),
            "context": self.context,
            "cause_type": type(self.cause).__name__ if self.cause else None,
            "cause_message": str(self.cause) if self.cause else None,
        }


# ==================== System-level Exception ====================


class EngineError(DataQualityException):
    """System-level exception - affects the entire system operation,
    must stop all operations
    """

    def __init__(
        self,
        message: str,
        connection_id: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize EngineError"""
        context = kwargs.get("context", {})
        if connection_id:
            context["connection_id"] = connection_id
        if operation:
            context["operation"] = operation
        context["severity"] = "CRITICAL"
        kwargs["context"] = context
        super().__init__(message, **kwargs)

    def get_impact_level(self) -> str:
        """Return impact level: SYSTEM, OPERATION, RESOURCE"""
        return "SYSTEM"

    def should_stop_execution(self) -> bool:
        """Whether execution should be stopped"""
        return True


# ==================== Operation-level Exception ====================


class OperationError(DataQualityException):
    """Operation-level exception - affects a single operation,
    other operations can continue
    """

    def __init__(
        self,
        message: str,
        rule_id: Optional[str] = None,
        operation: Optional[str] = None,
        sql: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OperationError"""
        context = kwargs.get("context", {})
        if rule_id:
            context["rule_id"] = rule_id
        if operation:
            context["operation"] = operation
        if sql:
            context["sql"] = sql[:500]  # Limit SQL length
        context["severity"] = "WARNING"
        kwargs["context"] = context
        super().__init__(message, **kwargs)

    def get_impact_level(self) -> str:
        """Return impact level: SYSTEM, OPERATION, RESOURCE"""
        return "OPERATION"

    def should_stop_execution(self) -> bool:
        """Whether execution should be stopped"""
        return False


# ==================== Resource-level Exception ====================


class RuleExecutionError(DataQualityException):
    """Resource-level exception - affects a specific resource,
    other resources can continue to be processed
    """

    def __init__(
        self,
        message: str,
        rule_id: Optional[str] = None,
        entity_name: Optional[str] = None,
        resource_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RuleExecutionError"""
        context = kwargs.get("context", {})
        if rule_id:
            context["rule_id"] = rule_id
        if entity_name:
            context["entity_name"] = entity_name
        if resource_type:
            context["resource_type"] = resource_type  # table, column, etc.
        context["severity"] = "INFO"
        kwargs["context"] = context
        super().__init__(message, **kwargs)

    def get_impact_level(self) -> str:
        """Return impact level: SYSTEM, OPERATION, RESOURCE"""
        return "RESOURCE"

    def should_stop_execution(self) -> bool:
        """Whether execution should be stopped"""
        return False


# ==================== Exception Converter ====================


class DatabaseExceptionConverter:
    """Database exception converter"""

    @staticmethod
    def convert_sqlalchemy_error(
        error: Exception, context: Dict[str, Any]
    ) -> DataQualityException:
        """Convert SQLAlchemy exception to data quality exception"""

        # Dynamically import SQLAlchemy exception types to avoid hard dependency
        try:
            from sqlalchemy.exc import (
                OperationalError,
                ProgrammingError,
                TimeoutError,
            )

            # System-level exception - use EngineError
            if isinstance(error, (OperationalError, TimeoutError)):
                return EngineError(
                    message=f"Database system error: {str(error)[:200]}",
                    connection_id=context.get("connection_id"),
                    operation=context.get("operation", "unknown"),
                    context=context,
                    cause=error,
                )

            # Resource-level exception - use RuleExecutionError
            elif isinstance(error, ProgrammingError):
                error_msg = str(error).lower()
                if any(
                    keyword in error_msg
                    for keyword in ["table", "relation", "not exist"]
                ):
                    return RuleExecutionError(
                        message=f"Table or view not found: {str(error)[:200]}",
                        rule_id=context.get("rule_id"),
                        entity_name=context.get("entity_name"),
                        resource_type="table",
                        context=context,
                        cause=error,
                    )
                elif any(keyword in error_msg for keyword in ["column", "field"]):
                    return RuleExecutionError(
                        message=f"Column not found: {str(error)[:200]}",
                        rule_id=context.get("rule_id"),
                        entity_name=context.get("entity_name"),
                        resource_type="column",
                        context=context,
                        cause=error,
                    )
                else:
                    return OperationError(
                        message=f"SQL syntax error: {str(error)[:200]}",
                        rule_id=context.get("rule_id"),
                        operation=context.get("operation"),
                        sql=context.get("sql"),
                        context=context,
                        cause=error,
                    )

            # Operation-level exception - use OperationError
            else:
                return OperationError(
                    message=f"Database operation failed: {str(error)[:200]}",
                    rule_id=context.get("rule_id"),
                    operation=context.get("operation"),
                    context=context,
                    cause=error,
                )

        except ImportError:
            # If SQLAlchemy is not available, classify based on error message
            error_msg = str(error).lower()

            # System-level exception
            if any(
                keyword in error_msg
                for keyword in ["connection", "timeout", "host", "authentication"]
            ):
                return EngineError(
                    message=f"Database system error: {str(error)[:200]}",
                    connection_id=context.get("connection_id"),
                    operation=context.get("operation", "unknown"),
                    context=context,
                    cause=error,
                )

            # Resource-level exception
            elif any(
                keyword in error_msg
                for keyword in ["table", "column", "not exist", "not found"]
            ):
                return RuleExecutionError(
                    message=f"Resource not found: {str(error)[:200]}",
                    rule_id=context.get("rule_id"),
                    entity_name=context.get("entity_name"),
                    resource_type="unknown",
                    context=context,
                    cause=error,
                )

            # Default to operation-level exception
            else:
                return OperationError(
                    message=f"Database operation failed: {str(error)[:200]}",
                    rule_id=context.get("rule_id"),
                    operation=context.get("operation"),
                    context=context,
                    cause=error,
                )


# ==================== Exception Handling Strategy ====================


class ExceptionHandler:
    """Exception handling strategy"""

    @staticmethod
    def handle_by_impact_level(
        error: DataQualityException,
    ) -> Optional[Dict[str, Any]]:
        """Handle exception according to impact level"""

        impact_level = error.get_impact_level()

        if impact_level == "SYSTEM":
            # System-level exception: propagate directly, stop all execution
            raise error

        elif impact_level == "OPERATION":
            # Operation-level exception: return error info,
            # can continue other operations
            return {
                "status": "ERROR",
                "error_type": "OPERATION",
                "message": error.message,
                "context": error.context,
                "should_continue": True,
            }

        elif impact_level == "RESOURCE":
            # Resource-level exception: return error info,
            # can continue other resources
            return {
                "status": "ERROR",
                "error_type": "RESOURCE",
                "message": error.message,
                "context": error.context,
                "should_continue": True,
            }

        else:
            # Unknown impact level: treat as system-level
            raise error

    @staticmethod
    def create_error_result(
        error: DataQualityException,
        rule_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create error result object"""
        return {
            "rule_id": rule_id or error.context.get("rule_id"),
            "status": "ERROR",
            "error_message": error.message,
            "error_type": error.__class__.__name__,
            "impact_level": error.get_impact_level(),
            "context": error.context,
            "timestamp": error.context.get("timestamp"),
        }
