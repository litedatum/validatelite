"""
CLI Error Classifier

Classifies errors from different sources into standardized categories
for consistent error handling and user messaging.
"""

from dataclasses import dataclass
from typing import Dict, List

from shared.exceptions import EngineError
from shared.schema.result_schema import ExecutionResultSchema


@dataclass
class CliErrorStrategy:
    """Error handling strategy for a specific error category"""

    category: str
    user_message_template: str
    recovery_actions: List[str]
    exit_code: int
    can_retry: bool
    requires_user_action: bool


class CliErrorClassifier:
    """CLI Error Classifier"""

    def __init__(self) -> None:
        """Initialize CliErrorClassifier"""
        self.strategies = self._load_strategies()

    def classify_native_error(self, error: Exception) -> str:
        """Classify native CLI errors"""

        # error_type = type(error).__name__
        error_message = str(error).lower()

        # Classify based on exception type
        if isinstance(error, FileNotFoundError):
            return "file_not_found"
        elif isinstance(error, PermissionError):
            return "permission_denied"
        elif isinstance(error, (ValueError, TypeError)) and any(
            keyword in error_message for keyword in ["json", "yaml", "parse", "format"]
        ):
            return "file_format_error"
        elif "config" in error_message:
            return "config_file_error"
        elif hasattr(error, "__module__") and "click" in error.__module__:
            return "usage_error"
        elif isinstance(error, OSError):
            return "file_system_error"
        else:
            return "cli_generic"

    def classify_schema_error(self, error: Exception) -> str:
        """Classify Schema creation errors"""

        from shared.exceptions import OperationError, RuleExecutionError

        error_message = str(error).lower()

        # Classify based on exception type and message
        if isinstance(error, OperationError):
            if any(
                keyword in error_message
                for keyword in ["connection", "connect", "invalid"]
            ):
                return "invalid_connection"
            elif any(keyword in error_message for keyword in ["type", "unsupported"]):
                return "unsupported_type"
            elif any(keyword in error_message for keyword in ["file", "path"]):
                return "invalid_file_path"
            else:
                return "connection_config"
        elif isinstance(error, RuleExecutionError):
            if any(
                keyword in error_message for keyword in ["syntax", "parse", "invalid"]
            ):
                return "rule_syntax_error"
            elif any(keyword in error_message for keyword in ["type", "unsupported"]):
                return "unsupported_rule"
            elif any(
                keyword in error_message
                for keyword in ["parameter", "param", "argument"]
            ):
                return "invalid_rule_params"
            else:
                return "rule_config"
        else:
            return "schema_generic"

    def classify_engine_error(self, error: EngineError) -> str:
        """Classify EngineError"""

        message = error.message.lower()
        # context = error.context

        # Classify based on error message and context
        # Prioritize checking for connection-related errors to avoid
        # misclassification as configuration errors
        if any(
            keyword in message
            for keyword in [
                "connection",
                "connect",
                "network",
                "timeout",
                "unable to connect",
            ]
        ):
            return "connectivity"
        elif any(
            keyword in message
            for keyword in ["authentication", "permission", "access", "denied"]
        ):
            return "authorization"
        elif any(
            keyword in message for keyword in ["configuration", "config", "parameter"]
        ):
            return "configuration"
        elif any(keyword in message for keyword in ["memory", "resource", "system"]):
            return "system_resource"
        else:
            return "system_generic"

    def classify_result_error(self, result: ExecutionResultSchema) -> str:
        """Classify result errors"""

        # First, try to use the classification hints provided by the result
        if hasattr(result, "get_error_classification_hints"):
            hints = result.get_error_classification_hints()

            # Classify based on the error type hint
            if "error_type" in hints:
                error_type = hints["error_type"]
                if error_type == "syntax":
                    return "sql_syntax"
                elif error_type == "timeout":
                    return "query_timeout"
                elif error_type == "permission":
                    return "data_access_denied"
                elif error_type == "data_type":
                    return "data_type_mismatch"
                elif error_type == "connection":
                    return "connectivity"
                elif error_type == "not_found" and "resource_type" in hints:
                    if hints["resource_type"] == "table":
                        return "table_not_found"
                    elif hints["resource_type"] == "column":
                        return "column_not_found"
                elif error_type == "operation":
                    return "execution_generic"

            # If only the resource type hint is available, classification is also
            # possible
            elif "resource_type" in hints:
                resource_type = hints["resource_type"]
                if resource_type == "table":
                    return "table_not_found"
                elif resource_type == "column":
                    return "column_not_found"

        # Fallback: Simple classification based on the error message
        error_message = (result.error_message or "").lower()
        # entity_name = (
        #     result.get_entity_name() if hasattr(result, "get_entity_name") else ""
        # )

        if "table" in error_message and any(
            keyword in error_message for keyword in ["not exist", "not found"]
        ):
            return "table_not_found"
        elif "column" in error_message and any(
            keyword in error_message for keyword in ["not exist", "not found"]
        ):
            return "column_not_found"
        elif any(keyword in error_message for keyword in ["syntax", "sql"]):
            return "sql_syntax"
        elif any(keyword in error_message for keyword in ["timeout", "time out"]):
            return "query_timeout"
        elif any(keyword in error_message for keyword in ["type", "data type"]):
            return "data_type_mismatch"
        elif any(keyword in error_message for keyword in ["permission", "access"]):
            return "data_access_denied"

        # Default classification
        return "execution_generic"

    def get_strategy(self, category: str) -> "CliErrorStrategy":
        """Get the error handling strategy"""
        return self.strategies.get(category, self.strategies["generic"])

    def _load_strategies(self) -> Dict[str, "CliErrorStrategy"]:
        """Load error handling strategies"""
        return {
            # Success strategy
            "success": CliErrorStrategy(
                category="success",
                user_message_template="Operation completed successfully",
                recovery_actions=[],
                exit_code=0,
                can_retry=False,
                requires_user_action=False,
            ),
            # CLI native error classification strategies
            "file_not_found": CliErrorStrategy(
                category="file_not_found",
                user_message_template="File not found: {message}",
                recovery_actions=[
                    "Check file path spelling",
                    "Verify file exists",
                    "Check current working directory",
                ],
                exit_code=20,
                can_retry=False,
                requires_user_action=True,
            ),
            "permission_denied": CliErrorStrategy(
                category="permission_denied",
                user_message_template="Permission denied: {message}",
                recovery_actions=[
                    "Check file permissions",
                    "Run with appropriate user privileges",
                    "Contact system administrator",
                ],
                exit_code=21,
                can_retry=False,
                requires_user_action=True,
            ),
            "file_format_error": CliErrorStrategy(
                category="file_format_error",
                user_message_template="File format error: {message}",
                recovery_actions=[
                    "Check file format (JSON/YAML/CSV)",
                    "Verify file content syntax",
                    "Use a text editor to check for formatting issues",
                ],
                exit_code=22,
                can_retry=False,
                requires_user_action=True,
            ),
            "config_file_error": CliErrorStrategy(
                category="config_file_error",
                user_message_template="Configuration file error: {message}",
                recovery_actions=[
                    "Check configuration file syntax",
                    "Verify all required settings are provided",
                    "Use example configuration as reference",
                ],
                exit_code=23,
                can_retry=False,
                requires_user_action=True,
            ),
            "usage_error": CliErrorStrategy(
                category="usage_error",
                user_message_template="Command usage error: {message}",
                recovery_actions=[
                    "Check command syntax",
                    "Use --help to see available options",
                    "Review command documentation",
                ],
                exit_code=24,
                can_retry=False,
                requires_user_action=True,
            ),
            # Schema creation error classification strategies
            "invalid_connection": CliErrorStrategy(
                category="invalid_connection",
                user_message_template="Invalid connection configuration: {message}",
                recovery_actions=[
                    "Check connection parameters",
                    "Verify database URL format",
                    "Ensure all required connection fields are provided",
                ],
                exit_code=30,
                can_retry=False,
                requires_user_action=True,
            ),
            "unsupported_type": CliErrorStrategy(
                category="unsupported_type",
                user_message_template="Unsupported connection type: {message}",
                recovery_actions=[
                    "Check supported connection types",
                    "Verify connection type spelling",
                    "Review documentation for supported databases",
                ],
                exit_code=31,
                can_retry=False,
                requires_user_action=True,
            ),
            "rule_syntax_error": CliErrorStrategy(
                category="rule_syntax_error",
                user_message_template="Rule syntax error: {message}",
                recovery_actions=[
                    "Check rule syntax",
                    "Verify rule parameters",
                    "Use --help rules-help for syntax guide",
                ],
                exit_code=32,
                can_retry=False,
                requires_user_action=True,
            ),
            "unsupported_rule": CliErrorStrategy(
                category="unsupported_rule",
                user_message_template="Unsupported rule type: {message}",
                recovery_actions=[
                    "Check supported rule types",
                    "Verify rule type spelling",
                    "Use --help rules-help for available rules",
                ],
                exit_code=33,
                can_retry=False,
                requires_user_action=True,
            ),
            # Core interface error classification strategies
            "connectivity": CliErrorStrategy(
                category="connectivity",
                user_message_template="Database connection error: {message}",
                recovery_actions=[
                    "Check network connectivity",
                    "Verify database server is running",
                    "Check connection parameters in configuration",
                ],
                exit_code=2,
                can_retry=True,
                requires_user_action=True,
            ),
            "authorization": CliErrorStrategy(
                category="authorization",
                user_message_template="Access denied: {message}",
                recovery_actions=[
                    "Check database credentials",
                    "Verify user permissions",
                    "Contact your database administrator",
                ],
                exit_code=3,
                can_retry=False,
                requires_user_action=True,
            ),
            "configuration": CliErrorStrategy(
                category="configuration",
                user_message_template="Configuration error: {message}",
                recovery_actions=[
                    "Check configuration file syntax",
                    "Verify all required settings are provided",
                    "Use --help to see configuration options",
                ],
                exit_code=4,
                can_retry=False,
                requires_user_action=True,
            ),
            # Result error classification strategies
            "table_not_found": CliErrorStrategy(
                category="table_not_found",
                user_message_template="Table not found in rule '{rule_id}': "
                "{entity_name}",
                recovery_actions=[
                    "Check table name spelling",
                    "Verify database schema",
                    "Ensure table exists in target database",
                ],
                exit_code=7,
                can_retry=False,
                requires_user_action=True,
            ),
            "column_not_found": CliErrorStrategy(
                category="column_not_found",
                user_message_template="Column not found in rule '{rule_id}': "
                "{error_message}",
                recovery_actions=[
                    "Check column name spelling",
                    "Verify table schema",
                    "Update rule configuration",
                ],
                exit_code=8,
                can_retry=False,
                requires_user_action=True,
            ),
            "sql_syntax": CliErrorStrategy(
                category="sql_syntax",
                user_message_template="SQL syntax error in rule '{rule_id}': "
                "{error_message}",
                recovery_actions=[
                    "Check rule definition syntax",
                    "Verify column and table names",
                    "Review SQL query in rule configuration",
                ],
                exit_code=5,
                can_retry=False,
                requires_user_action=True,
            ),
            "execution_generic": CliErrorStrategy(
                category="execution_generic",
                user_message_template="Rule execution error in rule '{rule_id}': "
                "{error_message}",
                recovery_actions=[
                    "Check rule configuration",
                    "Verify rule parameters",
                    "Review error details for specific issues",
                ],
                exit_code=6,
                can_retry=False,
                requires_user_action=True,
            ),
            "query_timeout": CliErrorStrategy(
                category="query_timeout",
                user_message_template="Query timeout in rule '{rule_id}': "
                "{error_message}",
                recovery_actions=[
                    "Check database performance",
                    "Consider optimizing the query",
                    "Increase timeout settings if possible",
                ],
                exit_code=9,
                can_retry=True,
                requires_user_action=True,
            ),
            "data_type_mismatch": CliErrorStrategy(
                category="data_type_mismatch",
                user_message_template="Data type mismatch in rule '{rule_id}': "
                "{error_message}",
                recovery_actions=[
                    "Check column data types",
                    "Verify rule parameter types",
                    "Update rule configuration to match data types",
                ],
                exit_code=10,
                can_retry=False,
                requires_user_action=True,
            ),
            "data_access_denied": CliErrorStrategy(
                category="data_access_denied",
                user_message_template="Data access denied in rule '{rule_id}': "
                "{error_message}",
                recovery_actions=[
                    "Check database permissions",
                    "Verify user has access to the table/column",
                    "Contact database administrator",
                ],
                exit_code=11,
                can_retry=False,
                requires_user_action=True,
            ),
            "file_system_error": CliErrorStrategy(
                category="file_system_error",
                user_message_template="File system error: {message}",
                recovery_actions=[
                    "Check disk space",
                    "Verify file system permissions",
                    "Check for file system issues",
                ],
                exit_code=25,
                can_retry=False,
                requires_user_action=True,
            ),
            "cli_generic": CliErrorStrategy(
                category="cli_generic",
                user_message_template="CLI error: {message}",
                recovery_actions=[
                    "Check command syntax",
                    "Review error details",
                    "Contact support if problem persists",
                ],
                exit_code=26,
                can_retry=False,
                requires_user_action=True,
            ),
            "schema_generic": CliErrorStrategy(
                category="schema_generic",
                user_message_template="Schema error: {message}",
                recovery_actions=[
                    "Check configuration format",
                    "Verify all required fields",
                    "Review schema documentation",
                ],
                exit_code=34,
                can_retry=False,
                requires_user_action=True,
            ),
            "system_generic": CliErrorStrategy(
                category="system_generic",
                user_message_template="System error: {message}",
                recovery_actions=[
                    "Check system resources",
                    "Review system logs",
                    "Contact system administrator",
                ],
                exit_code=12,
                can_retry=False,
                requires_user_action=True,
            ),
            # Default strategy
            "generic": CliErrorStrategy(
                category="generic",
                user_message_template="An error occurred: {message}",
                recovery_actions=[
                    "Review error details",
                    "Check system logs",
                    "Contact support if problem persists",
                ],
                exit_code=1,
                can_retry=False,
                requires_user_action=False,
            ),
        }
