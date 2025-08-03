"""
CLI Exception Handler

Provides unified error handling for the CLI application with:
1. User-friendly error messages
2. Technical error details for logging
3. Recovery suggestions
4. Exit code management
5. Unified handling of different error sources
"""

import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional

from cli.core.error_classifier import CliErrorClassifier
from shared.exceptions import EngineError
from shared.schema.result_schema import ExecutionResultSchema
from shared.utils.logger import get_logger


@dataclass
class CliErrorContext:
    """CLI Error Context"""

    category: str  # Error category
    source: str  # Error source (exception/result)
    user_message: str  # User message
    recovery_actions: List[str]  # Recovery suggestions
    exit_code: int  # Exit code
    technical_details: str  # Technical details
    requires_user_action: bool  # Whether user intervention is required
    can_retry: bool  # Whether it can be retried


class CliExceptionHandler:
    """CLI Exception Handler"""

    def __init__(self, verbose: bool = False):
        """Initialize CliExceptionHandler"""
        self.verbose = verbose
        self.classifier = CliErrorClassifier()
        self.logger = get_logger(__name__)

    def handle_complete_process(
        self,
        cli_error: Optional[Exception] = None,
        schema_error: Optional[Exception] = None,
        engine_error: Optional[EngineError] = None,
        results: Optional[List[ExecutionResultSchema]] = None,
    ) -> CliErrorContext:
        """
        Handle all errors and results from the complete CLI execution process.

        Args:
            cli_error: Native CLI processing error.
            schema_error: Schema creation error (OperationError/RuleExecutionError).
            engine_error: Captured EngineError exception.
            results: List of execution results.

        Returns:
            CliErrorContext: A unified error context.
        """

        # Handle errors by priority
        # 1. Native CLI errors have the highest priority
        if cli_error:
            return self._handle_cli_native_error(cli_error)

        # 2. Schema creation errors
        if schema_error:
            return self._handle_schema_creation_error(schema_error)

        # 3. Core system-level exceptions
        if engine_error:
            return self._handle_engine_error(engine_error)

        # 4. Analyze errors in the results
        if results:
            error_results = self._extract_error_results(results)
            if error_results:
                return self._handle_result_errors(error_results)

        # 5. If there are no errors, return a success context
        return self._create_success_context(results or [])

    def _handle_cli_native_error(self, error: Exception) -> CliErrorContext:
        """Handle native CLI errors"""

        # Classify the error using the classifier
        category = self.classifier.classify_native_error(error)
        strategy = self.classifier.get_strategy(category)

        # Build the user message
        user_message = strategy.user_message_template.format(
            error_type=type(error).__name__, message=str(error)
        )

        # Build the technical details
        technical_details = (
            self._build_native_error_technical_details(error) if self.verbose else ""
        )

        return CliErrorContext(
            category=category,
            source="cli_native",
            user_message=user_message,
            recovery_actions=strategy.recovery_actions,
            exit_code=strategy.exit_code,
            technical_details=technical_details,
            requires_user_action=strategy.requires_user_action,
            can_retry=strategy.can_retry,
        )

    def _handle_schema_creation_error(self, error: Exception) -> CliErrorContext:
        """Handle Schema creation errors"""

        # Classify the error using the classifier
        category = self.classifier.classify_schema_error(error)
        strategy = self.classifier.get_strategy(category)

        # Build the user message
        user_message = strategy.user_message_template.format(
            error_type=type(error).__name__, message=str(error)
        )

        # Build the technical details
        technical_details = (
            self._build_schema_error_technical_details(error) if self.verbose else ""
        )

        return CliErrorContext(
            category=category,
            source="schema_creation",
            user_message=user_message,
            recovery_actions=strategy.recovery_actions,
            exit_code=strategy.exit_code,
            technical_details=technical_details,
            requires_user_action=strategy.requires_user_action,
            can_retry=strategy.can_retry,
        )

    # Retain the original validation_process method for compatibility
    def handle_validation_process(
        self,
        engine_error: Optional[EngineError] = None,
        results: Optional[List[ExecutionResultSchema]] = None,
    ) -> CliErrorContext:
        """
        Handle errors and results of the validation process
        (backward compatible method).

        Args:
            engine_error: Captured EngineError exception.
            results: List of execution results.

        Returns:
            CliErrorContext: A unified error context.
        """
        return self.handle_complete_process(
            cli_error=None,
            schema_error=None,
            engine_error=engine_error,
            results=results,
        )

    def _handle_engine_error(self, error: EngineError) -> CliErrorContext:
        """Handle captured EngineError"""

        # Classify the error using the classifier
        category = self.classifier.classify_engine_error(error)
        strategy = self.classifier.get_strategy(category)

        # Build the user message
        user_message = strategy.user_message_template.format(
            message=error.message, context=error.context
        )

        # Build the technical details
        technical_details = self._build_technical_details(error) if self.verbose else ""

        return CliErrorContext(
            category=category,
            source="exception",
            user_message=user_message,
            recovery_actions=strategy.recovery_actions,
            exit_code=strategy.exit_code,
            technical_details=technical_details,
            requires_user_action=strategy.requires_user_action,
            can_retry=strategy.can_retry,
        )

    def _extract_error_results(
        self, results: List[ExecutionResultSchema]
    ) -> List[ExecutionResultSchema]:
        """Extract error results from the list of results"""
        return [result for result in results if result.status == "ERROR"]

    def _handle_result_errors(
        self, error_results: List[ExecutionResultSchema]
    ) -> CliErrorContext:
        """Handle error information in the results"""

        if not error_results:
            return self._create_success_context([])

        # If there is only one error, handle it in detail
        if len(error_results) == 1:
            return self._handle_single_result_error(error_results[0])

        # If there are multiple errors, classify and summarize them
        return self._handle_multiple_result_errors(error_results)

    def _handle_single_result_error(
        self, result: ExecutionResultSchema
    ) -> CliErrorContext:
        """Handle a single result error"""

        # Classify the error
        category = self.classifier.classify_result_error(result)
        strategy = self.classifier.get_strategy(category)

        # Build the user message
        user_message = strategy.user_message_template.format(
            rule_id=result.rule_id,
            entity_name=self._get_entity_name(result),
            error_message=result.error_message,
            message=result.error_message,  # Add message parameter for template
            # compatibility
        )

        return CliErrorContext(
            category=category,
            source="result",
            user_message=user_message,
            recovery_actions=strategy.recovery_actions,
            exit_code=strategy.exit_code,
            technical_details=(
                self._build_result_technical_details(result) if self.verbose else ""
            ),
            requires_user_action=strategy.requires_user_action,
            can_retry=strategy.can_retry,
        )

    def _handle_multiple_result_errors(
        self, error_results: List[ExecutionResultSchema]
    ) -> CliErrorContext:
        """Handle multiple result errors"""

        # Group errors by type
        error_groups: Dict[str, List[ExecutionResultSchema]] = {}
        for result in error_results:
            category = self.classifier.classify_result_error(result)
            if category not in error_groups:
                error_groups[category] = []
            error_groups[category].append(result)

        # Select the most severe error type as the primary error
        primary_category = self._select_primary_error_category(error_groups)
        strategy = self.classifier.get_strategy(primary_category)

        # Build summary message
        total_errors = len(error_results)
        error_summary = self._build_error_summary(error_groups)

        user_message = (
            f"Multiple validation errors occurred ({total_errors} total):\n"
            f"{error_summary}"
        )

        return CliErrorContext(
            category=primary_category,
            source="result",
            user_message=user_message,
            recovery_actions=strategy.recovery_actions,
            exit_code=strategy.exit_code,
            technical_details=(
                self._build_multiple_errors_technical_details(error_results)
                if self.verbose
                else ""
            ),
            requires_user_action=True,  # Multiple errors usually require user action
            can_retry=False,  # Multiple errors are usually not suitable for
            # automatic retry
        )

    def _create_success_context(
        self, results: List[ExecutionResultSchema]
    ) -> CliErrorContext:
        """Create a success context"""
        strategy = self.classifier.get_strategy("success")
        return CliErrorContext(
            category="success",
            source="none",
            user_message=strategy.user_message_template,
            recovery_actions=strategy.recovery_actions,
            exit_code=strategy.exit_code,
            technical_details="",
            requires_user_action=strategy.requires_user_action,
            can_retry=strategy.can_retry,
        )

    def _build_native_error_technical_details(self, error: Exception) -> str:
        """Build technical details for native CLI errors"""
        details = [
            f"Error Type: {type(error).__name__}",
            f"Error Message: {str(error)}",
            f"Traceback: {traceback.format_exc()}",
        ]
        return "\n".join(details)

    def _build_schema_error_technical_details(self, error: Exception) -> str:
        """Build technical details for Schema creation errors"""
        details = [
            f"Error Type: {type(error).__name__}",
            f"Error Message: {str(error)}",
            f"Traceback: {traceback.format_exc()}",
        ]

        # Add specific Schema error information
        if hasattr(error, "field") and error.field:
            details.append(f"Field: {error.field}")
        if hasattr(error, "value") and error.value:
            details.append(f"Value: {error.value}")
        if hasattr(error, "context") and error.context:
            details.append(f"Context: {error.context}")

        return "\n".join(details)

    def _build_technical_details(self, error: EngineError) -> str:
        """Build technical details for EngineError"""
        details = [
            "Error Type: EngineError",
            f"Error Message: {error.message}",
            f"Context: {error.context}",
        ]
        return "\n".join(details)

    def _build_result_technical_details(self, result: ExecutionResultSchema) -> str:
        """Build technical details for result errors"""
        details = [
            f"Rule ID: {result.rule_id}",
            f"Status: {result.status}",
            f"Error Message: {result.error_message}",
        ]

        # Add entity information
        entity_name = self._get_entity_name(result)
        if entity_name:
            details.append(f"Entity: {entity_name}")

        # Add other possible information
        if hasattr(result, "execution_time"):
            details.append(f"Execution Time: {result.execution_time}")

        return "\n".join(details)

    def _build_multiple_errors_technical_details(
        self, results: List[ExecutionResultSchema]
    ) -> str:
        """Build technical details for multiple result errors"""
        details = [f"Total Errors: {len(results)}"]

        for i, result in enumerate(results, 1):
            details.append(f"\nError {i}:")
            details.append(self._build_result_technical_details(result))

        return "\n".join(details)

    def _build_error_summary(
        self, error_groups: Dict[str, List[ExecutionResultSchema]]
    ) -> str:
        """Build error summary information"""
        summary_lines = []

        for category, results in error_groups.items():
            count = len(results)
            summary_lines.append(f"• {count} {category} errors")

        return "\n".join(summary_lines)

    def _select_primary_error_category(
        self, error_groups: Dict[str, List[ExecutionResultSchema]]
    ) -> str:
        """Select the primary error type"""
        # Priority order (high to low)
        priority_order = [
            "table_not_found",
            "column_not_found",
            "sql_syntax",
            "data_type_mismatch",
            "query_timeout",
            "data_access_denied",
            "execution_generic",
        ]

        # Search by priority
        for category in priority_order:
            if category in error_groups:
                return category

        # If no matching priority, return the first category or a default category
        if error_groups:
            return next(iter(error_groups.keys()))

        return "execution_generic"

    def _get_entity_name(self, result: ExecutionResultSchema) -> str:
        """Get entity name"""
        if hasattr(result, "get_entity_name"):
            return result.get_entity_name()

        # Try to extract from dataset_metrics
        if hasattr(result, "dataset_metrics") and result.dataset_metrics:
            return result.dataset_metrics[0].entity_name

        return "unknown"
