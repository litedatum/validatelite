"""
Rule category enumeration

Defines the categories of data quality rules, integrating existing rule
category definitions.
Supports category query and rule organization.
"""

from enum import Enum

from shared.exceptions.exception_system import RuleExecutionError


class RuleCategory(str, Enum):
    """
    Rule category enumeration

    Defines the categories of data quality rules:
    - COMPLETENESS: Completeness, checks if data is complete
    - ACCURACY: Accuracy, checks if data is accurate
    - CONSISTENCY: Consistency, checks if data is consistent
    - VALIDITY: Validity, checks if data is valid
    - UNIQUENESS: Uniqueness, checks if data is unique
    - TIMELINESS: Timeliness, checks if data is timely
    - CUSTOM: Custom, user-defined rules
    """

    COMPLETENESS = "COMPLETENESS"  # Completeness
    ACCURACY = "ACCURACY"  # Accuracy
    CONSISTENCY = "CONSISTENCY"  # Consistency
    VALIDITY = "VALIDITY"  # Validity
    UNIQUENESS = "UNIQUENESS"  # Uniqueness
    TIMELINESS = "TIMELINESS"  # Timeliness
    CUSTOM = "CUSTOM"  # Custom

    @classmethod
    def get_description(cls, category: "RuleCategory") -> str:
        """
        Get category description

        Args:
            category: Rule category

        Returns:
            str: Category description
        """
        descriptions = {
            cls.COMPLETENESS: "Check data completeness, e.g., non-null check",
            cls.ACCURACY: "Check data accuracy, e.g., format validation",
            cls.CONSISTENCY: "Check data consistency, e.g., cross-table association",
            cls.VALIDITY: "Check data validity, e.g., range validation",
            cls.UNIQUENESS: "Check data uniqueness, e.g., primary key constraint",
            cls.TIMELINESS: "Check data timeliness, e.g., timestamp validation",
            cls.CUSTOM: "User-defined business rules",
        }
        return descriptions.get(category, "Unknown category")

    @classmethod
    def get_icon(cls, category: "RuleCategory") -> str:
        """
        Get the icon corresponding to the category (for UI display)

        Args:
            category: Rule category

        Returns:
            str: Icon name
        """
        icons = {
            cls.COMPLETENESS: "check-circle",
            cls.ACCURACY: "target",
            cls.CONSISTENCY: "link",
            cls.VALIDITY: "shield-check",
            cls.UNIQUENESS: "key",
            cls.TIMELINESS: "clock",
            cls.CUSTOM: "settings",
        }
        return icons.get(category, "help-circle")

    @classmethod
    def get_color_code(cls, category: "RuleCategory") -> str:
        """
        Get the color code corresponding to the category (for UI display)

        Args:
            category: Rule category

        Returns:
            str: Color code
        """
        color_codes = {
            cls.COMPLETENESS: "#28a745",  # Green
            cls.ACCURACY: "#007bff",  # Blue
            cls.CONSISTENCY: "#6f42c1",  # Purple
            cls.VALIDITY: "#fd7e14",  # Orange
            cls.UNIQUENESS: "#20c997",  # Cyan
            cls.TIMELINESS: "#ffc107",  # Yellow
            cls.CUSTOM: "#6c757d",  # Gray
        }
        return color_codes.get(category, "#6c757d")

    @classmethod
    def get_priority(cls, category: "RuleCategory") -> int:
        """
        Get category priority (for sorting)

        Args:
            category: Rule category

        Returns:
            int: Priority value (the smaller the value, the higher the priority)
        """
        priorities = {
            cls.COMPLETENESS: 1,  # Completeness has the highest priority
            cls.UNIQUENESS: 2,  # Uniqueness is next
            cls.VALIDITY: 3,  # Validity
            cls.ACCURACY: 4,  # Accuracy
            cls.CONSISTENCY: 5,  # Consistency
            cls.TIMELINESS: 6,  # Timeliness
            cls.CUSTOM: 7,  # Custom has the lowest priority
        }
        return priorities.get(category, 999)

    @classmethod
    def is_basic_category(cls, category: "RuleCategory") -> bool:
        """
        Determine if it is a basic category (not custom)

        Args:
            category: Rule category

        Returns:
            bool: Whether it is a basic category
        """
        basic_categories = [
            cls.COMPLETENESS,
            cls.ACCURACY,
            cls.CONSISTENCY,
            cls.VALIDITY,
            cls.UNIQUENESS,
            cls.TIMELINESS,
        ]
        return category in basic_categories

    @classmethod
    def from_string(cls, value: str) -> "RuleCategory":
        """
        Create enum value from string, case-insensitive

        Args:
            value: Category string

        Returns:
            RuleCategory: Corresponding enum value

        Raises:
            RuleExecutionError: If the string is not a valid rule category
        """
        try:
            return cls(value.upper())
        except ValueError:
            # Check if it is in lowercase form
            for category in cls:
                if category.value.lower() == value.lower():
                    return category
            # If still not found, raise exception
            valid_categories = ", ".join([c.value for c in cls])
            raise RuleExecutionError(
                f"Invalid rule category: {value}. Valid categories: {valid_categories}"
            )

    @classmethod
    def get_related_rule_types(cls, category: "RuleCategory") -> list[str]:
        """
        Get related rule types for the category

        Args:
            category: Rule category

        Returns:
            list[str]: List of related rule types
        """
        # Return string list here to avoid circular import
        related_types = {
            cls.COMPLETENESS: ["NOT_NULL"],
            cls.ACCURACY: ["REGEX", "EMAIL", "PHONE", "URL", "DATE_FORMAT"],
            cls.CONSISTENCY: ["FOREIGN_KEY"],
            cls.VALIDITY: ["RANGE", "LENGTH", "ENUM", "MIN_MAX"],
            cls.UNIQUENESS: ["UNIQUE", "PRIMARY_KEY"],
            cls.TIMELINESS: [],  # No specific timeliness rule types for now
            cls.CUSTOM: ["CUSTOM_SQL", "BUSINESS_RULE"],
        }
        return related_types.get(category, [])
