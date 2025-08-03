"""
CLI Exception System

CLI-specific exceptions that don't belong in shared/exceptions.
These exceptions are designed for CLI-specific error scenarios and
integrate with the CLI error classification system.
"""

from typing import Any, Dict, Optional

from shared.exceptions import DataQualityException


class CliException(DataQualityException):
    """Base exception for all CLI-specific errors"""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize CliException"""
        super().__init__(message)
        self.field = field
        self.value = value
        self.context = context or {}


class ValidationError(CliException):
    """CLI validation error - for parameter validation, rule syntax validation etc."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ValidationError"""
        super().__init__(message, field, value, **kwargs)

    def get_impact_level(self) -> str:
        """Get impact level"""
        return "OPERATION"

    def should_stop_execution(self) -> bool:
        """Should stop execution"""
        return False


class CliConfigError(CliException):
    """CLI configuration error - for CLI-specific config issues"""

    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        config_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CliConfigError"""
        super().__init__(message, **kwargs)
        self.config_file = config_file
        self.config_key = config_key

    def get_impact_level(self) -> str:
        """Get impact level"""
        return "OPERATION"

    def should_stop_execution(self) -> bool:
        """Should stop execution"""
        return False


class CliFileError(CliException):
    """CLI file operation error - for file I/O issues"""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CliFileError"""
        super().__init__(message, **kwargs)
        self.file_path = file_path
        self.operation = operation

    def get_impact_level(self) -> str:
        """Get impact level"""
        return "SYSTEM"

    def should_stop_execution(self) -> bool:
        """Should stop execution"""
        return True


class ConnectionError(CliException):
    """CLI connection error

    - for database connection issues in CLI context
    """

    def __init__(
        self, message: str, connection_url: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Initialize ConnectionError"""
        super().__init__(message, **kwargs)
        self.connection_url = connection_url

    def get_impact_level(self) -> str:
        """Get impact level"""
        return "SYSTEM"

    def should_stop_execution(self) -> bool:
        """Should stop execution"""
        return True


class DatabaseError(CliException):
    """CLI database error - for database operation issues in CLI context"""

    def __init__(
        self,
        message: str,
        database_name: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize DatabaseError"""
        super().__init__(message, **kwargs)
        self.database_name = database_name
        self.operation = operation

    def get_impact_level(self) -> str:
        """Get impact level"""
        return "OPERATION"

    def should_stop_execution(self) -> bool:
        """Should stop execution"""
        return False


class RuleParsingError(CliException):
    """Rule parsing error - for rule syntax and parsing issues"""

    def __init__(
        self,
        message: str,
        rule_expression: Optional[str] = None,
        rule_file: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RuleParsingError"""
        super().__init__(message, **kwargs)
        self.rule_expression = rule_expression
        self.rule_file = rule_file

    def get_impact_level(self) -> str:
        """Get impact level"""
        return "RESOURCE"

    def should_stop_execution(self) -> bool:
        """Should stop execution"""
        return False


class CliProcessingError(CliException):
    """General CLI processing error - for other CLI-specific processing issues"""

    def __init__(
        self, message: str, operation: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Initialize CliProcessingError"""
        super().__init__(message, **kwargs)
        self.operation = operation

    def get_impact_level(self) -> str:
        """Get impact level"""
        return "OPERATION"

    def should_stop_execution(self) -> bool:
        """Should stop execution"""
        return False


# Legacy aliases for backward compatibility (will be deprecated)
CliValidationError = ValidationError
CliConnectionError = ConnectionError
CliDatabaseError = DatabaseError
