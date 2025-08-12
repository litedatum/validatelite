"""
Rule type enumeration

Defines all types of data quality rules, integrating existing rule type definitions.
Supports category query and type validation.
"""

from enum import Enum

from shared.exceptions.exception_system import RuleExecutionError


class RuleType(str, Enum):
    """
    Rule type enumeration

    Integrates existing rule type definitions, supporting multiple data quality checks:
    - Completeness checks: NOT_NULL, UNIQUE, etc.
    - Format checks: REGEX, EMAIL, etc.
    - Range checks: RANGE, LENGTH, etc.
    - Business rules: CUSTOM_SQL, etc.
    - Statistical checks: COUNT, SUM, etc.
    """

    # Completeness checks
    NOT_NULL = "NOT_NULL"
    UNIQUE = "UNIQUE"
    # PRIMARY_KEY = "PRIMARY_KEY"
    # FOREIGN_KEY = "FOREIGN_KEY"

    # Format checks
    REGEX = "REGEX"
    # EMAIL = "EMAIL"
    # PHONE = "PHONE"
    # URL = "URL"
    DATE_FORMAT = "DATE_FORMAT"

    # Range checks
    RANGE = "RANGE"
    LENGTH = "LENGTH"
    # MIN_MAX = "MIN_MAX"

    # Enum value checks
    ENUM = "ENUM"

    # Schema checks (table-level): existence and type only
    SCHEMA = "SCHEMA"

    # Business rules
    # CUSTOM_SQL = "CUSTOM_SQL"
    # BUSINESS_RULE = "BUSINESS_RULE"

    # Statistical checks
    # COUNT = "COUNT"
    # SUM = "SUM"
    # AVERAGE = "AVERAGE"

    @classmethod
    def get_category(cls, rule_type: "RuleType") -> str:
        """
        Get the category of the rule type

        Args:
            rule_type: Rule type

        Returns:
            str: Rule category
        """
        categories = {
            cls.NOT_NULL: "completeness",
            cls.UNIQUE: "uniqueness",
            # cls.PRIMARY_KEY: "uniqueness",
            # cls.FOREIGN_KEY: "referential_integrity",
            cls.REGEX: "format",
            # cls.EMAIL: "format",
            # cls.PHONE: "format",
            # cls.URL: "format",
            cls.DATE_FORMAT: "format",
            cls.RANGE: "validity",
            cls.LENGTH: "validity",
            # cls.MIN_MAX: "validity",
            cls.ENUM: "validity",
            cls.SCHEMA: "validity",
            # cls.CUSTOM_SQL: "business",
            # cls.BUSINESS_RULE: "business",
            # cls.COUNT: "statistical",
            # cls.SUM: "statistical",
            # cls.AVERAGE: "statistical"
        }
        return categories.get(rule_type, "unknown")

    @classmethod
    def get_completeness_types(cls) -> list["RuleType"]:
        """Get completeness check types"""
        return [cls.NOT_NULL]

    @classmethod
    def get_uniqueness_types(cls) -> list["RuleType"]:
        """Get uniqueness check types"""
        return [cls.UNIQUE]
        # return [cls.UNIQUE, cls.PRIMARY_KEY]

    @classmethod
    def get_validity_types(cls) -> list["RuleType"]:
        """Get validity check types"""
        return [
            cls.RANGE,
            cls.LENGTH,
            # cls.MIN_MAX,
            cls.ENUM,
            cls.REGEX,
            # cls.EMAIL,
            # cls.PHONE,
            # cls.URL,
            cls.DATE_FORMAT,
            cls.SCHEMA,
        ]

    # @classmethod
    # def get_business_types(cls) -> list["RuleType"]:
    #     """
    #     Get business rule types
    #     """
    #     return [cls.CUSTOM_SQL, cls.BUSINESS_RULE]

    # @classmethod
    # def get_statistical_types(cls) -> list["RuleType"]:
    #     """
    #     Get statistical check types
    #     """
    #     return [cls.COUNT, cls.SUM, cls.AVERAGE]

    @classmethod
    def from_string(cls, value: str) -> "RuleType":
        """
        Create enum value from string, case-insensitive

        Args:
            value: Rule type string

        Returns:
            RuleType: Corresponding enum value

        Raises:
            RuleExecutionError: If the string is not a valid rule type
        """
        try:
            return cls(value.upper())
        except ValueError:
            # Check if it is in lowercase form
            for rule_type in cls:
                if rule_type.value.lower() == value.lower():
                    return rule_type
            # If still not found, raise exception
            valid_types = ", ".join([t.value for t in cls])
            raise RuleExecutionError(
                f"Invalid rule type: {value}. Valid types: {valid_types}"
            )

    @classmethod
    def is_mergeable(cls, rule_type: "RuleType") -> bool:
        """
        Determine if the rule type can be merged for execution

        Args:
            rule_type: Rule type

        Returns:
            bool: Whether it can be merged
        """
        mergeable_types = [
            cls.NOT_NULL,
            cls.RANGE,
            cls.ENUM,
            cls.REGEX,
            cls.LENGTH,
            # SCHEMA is table-level and should not be merged with column-level rules
        ]
        return rule_type in mergeable_types
