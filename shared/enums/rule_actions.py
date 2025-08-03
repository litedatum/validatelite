"""
Rule action enumeration

Defines the handling actions when a rule fails, integrating existing rule
action definitions.
Supports action execution and action combination.
"""

from enum import Enum

from shared.exceptions.exception_system import RuleExecutionError


class RuleAction(str, Enum):
    """
    Rule action enumeration

    Defines the handling actions when a rule fails:
    - LOG: Record log, the most basic action
    - ALERT: Send alert, notify relevant personnel
    - BLOCK: Block data writing, prevent erroneous data from entering the system
    - QUARANTINE: Quarantine data, move problematic data to quarantine area
    - CORRECT: Auto-correct, attempt to automatically fix data
    - IGNORE: Ignore errors, continue processing
    """

    LOG = "LOG"  # Record log
    ALERT = "ALERT"  # Send alert
    BLOCK = "BLOCK"  # Block data writing
    QUARANTINE = "QUARANTINE"  # Quarantine data
    CORRECT = "CORRECT"  # Auto-correct
    IGNORE = "IGNORE"  # Ignore error

    @classmethod
    def get_description(cls, action: "RuleAction") -> str:
        """
        Get action description

        Args:
            action: Rule action

        Returns:
            str: Action description
        """
        descriptions = {
            cls.LOG: "Record error log, does not affect data processing flow",
            cls.ALERT: "Send alert notification, remind relevant personnel to handle",
            cls.BLOCK: "Block data writing, prevent erroneous data from entering "
            "the system",
            cls.QUARANTINE: "Move problematic data to a dedicated area",
            cls.CORRECT: "Attempt to automatically fix data errors",
            cls.IGNORE: "Ignore errors, continue normal processing flow",
        }
        return descriptions.get(action, "Unknown action")

    @classmethod
    def get_severity_level(cls, action: "RuleAction") -> int:
        """
        Get action severity level

        Args:
            action: Rule action

        Returns:
            int: Severity level (the larger the value, the more severe)
        """
        severity_levels = {
            cls.IGNORE: 0,  # Ignore, lowest level
            cls.LOG: 1,  # Record log
            cls.CORRECT: 2,  # Auto-correct
            cls.ALERT: 3,  # Send alert
            cls.QUARANTINE: 4,  # Quarantine data
            cls.BLOCK: 5,  # Block writing, highest level
        }
        return severity_levels.get(action, 1)

    @classmethod
    def is_blocking_action(cls, action: "RuleAction") -> bool:
        """
        Determine if it is a blocking action (will interrupt the process)

        Args:
            action: Rule action

        Returns:
            bool: Whether it is a blocking action
        """
        blocking_actions = [cls.BLOCK, cls.QUARANTINE]
        return action in blocking_actions

    @classmethod
    def requires_notification(cls, action: "RuleAction") -> bool:
        """
        Determine if notification is required

        Args:
            action: Rule action

        Returns:
            bool: Whether notification is required
        """
        notification_actions = [cls.ALERT, cls.BLOCK, cls.QUARANTINE]
        return action in notification_actions

    @classmethod
    def can_auto_execute(cls, action: "RuleAction") -> bool:
        """
        Determine if it can be executed automatically (without manual intervention)

        Args:
            action: Rule action

        Returns:
            bool: Whether it can be executed automatically
        """
        auto_actions = [cls.LOG, cls.ALERT, cls.CORRECT, cls.IGNORE]
        return action in auto_actions

    @classmethod
    def get_icon(cls, action: "RuleAction") -> str:
        """
        Get the icon corresponding to the action (for UI display)

        Args:
            action: Rule action

        Returns:
            str: Icon name
        """
        icons = {
            cls.LOG: "file-text",
            cls.ALERT: "bell",
            cls.BLOCK: "shield",
            cls.QUARANTINE: "archive",
            cls.CORRECT: "tool",
            cls.IGNORE: "eye-off",
        }
        return icons.get(action, "help-circle")

    @classmethod
    def get_color_code(cls, action: "RuleAction") -> str:
        """
        Get the color code corresponding to the action (for UI display)

        Args:
            action: Rule action

        Returns:
            str: Color code
        """
        color_codes = {
            cls.LOG: "#6c757d",  # Gray
            cls.ALERT: "#ffc107",  # Yellow
            cls.BLOCK: "#dc3545",  # Red
            cls.QUARANTINE: "#fd7e14",  # Orange
            cls.CORRECT: "#28a745",  # Green
            cls.IGNORE: "#17a2b8",  # Cyan
        }
        return color_codes.get(action, "#6c757d")

    @classmethod
    def from_string(cls, value: str) -> "RuleAction":
        """
        Create enum value from string, case-insensitive

        Args:
            value: Action string

        Returns:
            RuleAction: Corresponding enum value

        Raises:
            RuleExecutionError: If the string is not a valid rule action
        """
        try:
            return cls(value.upper())
        except ValueError:
            # Check if it is in lowercase form
            for action in cls:
                if action.value.lower() == value.lower():
                    return action
            # If still not found, raise exception
            valid_actions = ", ".join([a.value for a in cls])
            raise RuleExecutionError(
                f"Invalid rule action: {value}. Valid actions: {valid_actions}"
            )

    @classmethod
    def get_compatible_actions(cls, action: "RuleAction") -> list["RuleAction"]:
        """
        Get compatible action combinations (actions that can be executed simultaneously)

        Args:
            action: Base action

        Returns:
            list[RuleAction]: List of compatible actions
        """
        # Define action compatibility
        compatibility = {
            cls.LOG: [
                cls.ALERT,
                cls.CORRECT,
            ],  # Log can be combined with alert and correct
            cls.ALERT: [
                cls.LOG,
                cls.QUARANTINE,
            ],  # Alert can be combined with log and quarantine
            cls.BLOCK: [cls.LOG, cls.ALERT],  # Block can be combined with log and alert
            cls.QUARANTINE: [
                cls.LOG,
                cls.ALERT,
            ],  # Quarantine can be combined with log and alert
            cls.CORRECT: [cls.LOG],  # Correct can be combined with log
            cls.IGNORE: [],  # Ignore does not combine with other actions
        }
        return compatibility.get(action, [])
