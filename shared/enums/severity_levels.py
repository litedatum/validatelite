"""
Severity level enumeration

Defines all levels of rule severity, integrating existing severity level definitions.
Supports priority sorting and blocking judgment.
"""

from enum import Enum

from shared.exceptions.exception_system import RuleExecutionError


class SeverityLevel(str, Enum):
    """
    Severity level enumeration

    Defines severity levels when a rule fails:
    - LOW: Low severity, usually only logs
    - MEDIUM: Medium severity, may require attention
    - HIGH: High severity, requires immediate handling
    - CRITICAL: Critical severity, may block data processing
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @classmethod
    def get_priority(cls, severity: "SeverityLevel") -> int:
        """
        Get priority value (the larger the value, the higher the priority)

        Args:
            severity: Severity level

        Returns:
            int: Priority value
        """
        priorities = {cls.LOW: 1, cls.MEDIUM: 2, cls.HIGH: 3, cls.CRITICAL: 4}
        return priorities.get(severity, 0)

    @classmethod
    def should_block_execution(cls, severity: "SeverityLevel") -> bool:
        """
        Determine if execution should be blocked

        Args:
            severity: Severity level

        Returns:
            bool: Whether execution should be blocked
        """
        blocking_severities = [cls.HIGH, cls.CRITICAL]
        return severity in blocking_severities

    @classmethod
    def should_send_alert(cls, severity: "SeverityLevel") -> bool:
        """
        Determine if an alert should be sent

        Args:
            severity: Severity level

        Returns:
            bool: Whether an alert should be sent
        """
        alert_severities = [cls.MEDIUM, cls.HIGH, cls.CRITICAL]
        return severity in alert_severities

    @classmethod
    def get_color_code(cls, severity: "SeverityLevel") -> str:
        """
        Get the color code corresponding to the severity level (for UI display)

        Args:
            severity: Severity level

        Returns:
            str: Color code
        """
        color_codes = {
            cls.LOW: "#28a745",  # Green
            cls.MEDIUM: "#ffc107",  # Yellow
            cls.HIGH: "#fd7e14",  # Orange
            cls.CRITICAL: "#dc3545",  # Red
        }
        return color_codes.get(severity, "#6c757d")  # Default gray

    @classmethod
    def get_icon(cls, severity: "SeverityLevel") -> str:
        """
        Get the icon corresponding to the severity level (for UI display)

        Args:
            severity: Severity level

        Returns:
            str: Icon name
        """
        icons = {
            cls.LOW: "info",
            cls.MEDIUM: "warning",
            cls.HIGH: "alert",
            cls.CRITICAL: "error",
        }
        return icons.get(severity, "info")

    @classmethod
    def from_string(cls, value: str) -> "SeverityLevel":
        """
        Create enum value from string, case-insensitive

        Args:
            value: Severity level string

        Returns:
            SeverityLevel: Corresponding enum value

        Raises:
            RuleExecutionError: If the string is not a valid severity level
        """
        try:
            return cls(value.upper())
        except ValueError:
            # Check if it is in lowercase form
            for severity in cls:
                if severity.value.lower() == value.lower():
                    return severity
            # If still not found, raise exception
            valid_severities = ", ".join([s.value for s in cls])
            raise RuleExecutionError(
                f"Invalid severity level: {value}. Valid severity levels: "
                f"{valid_severities}"
            )

    @classmethod
    def compare(cls, severity1: "SeverityLevel", severity2: "SeverityLevel") -> int:
        """
        Compare two severity levels

        Args:
            severity1: First severity level
            severity2: Second severity level

        Returns:
            int: Comparison result (
                -1: severity1 < severity2,
                0: equal,
                1: severity1 > severity2
            )
        """
        priority1 = cls.get_priority(severity1)
        priority2 = cls.get_priority(severity2)

        if priority1 < priority2:
            return -1
        elif priority1 > priority2:
            return 1
        else:
            return 0
