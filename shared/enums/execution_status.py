"""
Execution status enumeration

Defines all possible statuses for rule execution, integrating existing execution
status definitions.
Supports status judgment and status transition validation.
"""

from enum import Enum

from shared.exceptions.exception_system import RuleExecutionError


class ExecutionStatus(str, Enum):
    """
    Execution status enumeration

    Defines all possible statuses during rule execution:
    - PENDING: Waiting for execution
    - RUNNING: Executing
    - PASSED: Execution successful, rule passed
    - FAILED: Execution successful, but rule failed
    - WARNING: Execution successful, but with warnings
    - ERROR: Error occurred during execution
    - CANCELLED: Execution cancelled
    """

    PENDING = "PENDING"  # Waiting for execution
    RUNNING = "RUNNING"  # Executing
    PASSED = "PASSED"  # Execution successful, rule passed
    FAILED = "FAILED"  # Execution successful, but rule failed
    WARNING = "WARNING"  # Execution successful, but with warnings
    ERROR = "ERROR"  # Error occurred during execution
    CANCELLED = "CANCELLED"  # Execution cancelled

    @classmethod
    def is_final_status(cls, status: "ExecutionStatus") -> bool:
        """
        Determine if it is a final status (status that will not change)

        Args:
            status: Execution status

        Returns:
            bool: Whether it is a final status
        """
        final_statuses = [cls.PASSED, cls.FAILED, cls.WARNING, cls.ERROR, cls.CANCELLED]
        return status in final_statuses

    @classmethod
    def is_success_status(cls, status: "ExecutionStatus") -> bool:
        """
        Determine if it is a success status

        Args:
            status: Execution status

        Returns:
            bool: Whether it is a success status
        """
        return status == cls.PASSED

    @classmethod
    def is_failure_status(cls, status: "ExecutionStatus") -> bool:
        """
        Determine if it is a failure status

        Args:
            status: Execution status

        Returns:
            bool: Whether it is a failure status
        """
        failure_statuses = [cls.FAILED, cls.ERROR]
        return status in failure_statuses

    @classmethod
    def is_running_status(cls, status: "ExecutionStatus") -> bool:
        """
        Determine if it is a running status

        Args:
            status: Execution status

        Returns:
            bool: Whether it is a running status
        """
        running_statuses = [cls.PENDING, cls.RUNNING]
        return status in running_statuses

    @classmethod
    def get_priority(cls, status: "ExecutionStatus") -> int:
        """
        Get status priority (for sorting and display)

        Args:
            status: Execution status

        Returns:
            int: Priority value (the larger the value, the higher the priority)
        """
        priorities = {
            cls.ERROR: 5,  # Highest priority for error
            cls.FAILED: 4,  # Next is failure
            cls.WARNING: 3,  # Warning
            cls.RUNNING: 2,  # Running
            cls.PENDING: 1,  # Waiting
            cls.PASSED: 0,  # Passed has lowest priority
            cls.CANCELLED: -1,  # Cancelled has lowest priority
        }
        return priorities.get(status, 0)

    @classmethod
    def can_transition_to(
        cls, from_status: "ExecutionStatus", to_status: "ExecutionStatus"
    ) -> bool:
        """
        Determine if the status transition is valid

        Args:
            from_status: Source status
            to_status: Target status

        Returns:
            bool: Whether the transition is allowed
        """
        # Define valid status transitions
        valid_transitions = {
            cls.PENDING: [cls.RUNNING, cls.CANCELLED],
            cls.RUNNING: [
                cls.PASSED,
                cls.FAILED,
                cls.WARNING,
                cls.ERROR,
                cls.CANCELLED,
            ],
            # Final statuses cannot transition to other statuses
            cls.PASSED: [],
            cls.FAILED: [],
            cls.WARNING: [],
            cls.ERROR: [],
            cls.CANCELLED: [],
        }

        return to_status in valid_transitions.get(from_status, [])

    @classmethod
    def from_string(cls, value: str) -> "ExecutionStatus":
        """
        Create enum value from string, case-insensitive

        Args:
            value: Status string

        Returns:
            ExecutionStatus: Corresponding enum value

        Raises:
            RuleExecutionError: If the string is not a valid execution status
        """
        try:
            return cls(value.upper())
        except ValueError:
            # Check if it is in lowercase form
            for status in cls:
                if status.value.lower() == value.lower():
                    return status
            # If still not found, raise exception
            valid_statuses = ", ".join([s.value for s in cls])
            raise RuleExecutionError(
                f"Invalid execution status: {value}. Valid statuses: {valid_statuses}"
            )
