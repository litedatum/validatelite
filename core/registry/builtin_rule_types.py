"""
Built-in rule types

Registers system built-in rule types, including:
- NOT_NULL: Not null check
- UNIQUE: Uniqueness check
- RANGE: Range check
- ENUM: Enum value check
- REGEX: Regular expression check
- DATE_FORMAT: Date format check
- LENGTH: Length check
- CUSTOM_SQL: Custom SQL check
"""

import logging

from core.registry.rule_type_registry import rule_type_registry

# Configure logger
logger = logging.getLogger(__name__)


def register_builtin_rule_types() -> None:
    """Register all built-in rule types"""
    # Register NOT_NULL rule type
    rule_type_registry.register_rule_type(
        type_id="NOT_NULL",
        name="Not Null Check",
        description="Checks whether a field contains null values, suitable for "
        "required field validation",
        category="completeness",
        icon="check_circle",
        parameters_schema={"type": "object", "properties": {}, "required": []},
        ui_schema={},
        examples=[
            {
                "name": "Customer ID Not Null Check",
                "description": "Checks whether the customer ID field is null",
                "params": {},
            }
        ],
    )

    # Register UNIQUE rule type
    rule_type_registry.register_rule_type(
        type_id="UNIQUE",
        name="Uniqueness Check",
        description="Checks whether field values are unique, suitable for "
        "primary keys, unique indexes, etc.",
        category="uniqueness",
        icon="fingerprint",
        parameters_schema={"type": "object", "properties": {}, "required": []},
        ui_schema={},
        examples=[
            {
                "name": "Username Uniqueness Check",
                "description": "Checks whether the username is unique",
                "params": {},
            }
        ],
    )

    # Register RANGE rule type
    rule_type_registry.register_rule_type(
        type_id="RANGE",
        name="Range Check",
        description="Checks whether a numeric value is within the specified range",
        category="validity",
        icon="trending_up",
        parameters_schema={
            "type": "object",
            "properties": {
                "min": {"type": ["number", "null"], "title": "Minimum Value"},
                "max": {"type": ["number", "null"], "title": "Maximum Value"},
            },
        },
        ui_schema={
            "min": {"ui:placeholder": "Enter minimum value"},
            "max": {"ui:placeholder": "Enter maximum value"},
        },
        examples=[
            {
                "name": "Age Range Check",
                "description": "Checks whether age is between 0 and 120",
                "params": {"min": 0, "max": 120},
            }
        ],
    )

    # Register ENUM rule type
    rule_type_registry.register_rule_type(
        type_id="ENUM",
        name="Enum Value Check",
        description="Checks whether a field value is in the allowed value list",
        category="validity",
        icon="list",
        parameters_schema={
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "title": "Allowed Values",
                    "items": {"type": "string"},
                }
            },
            "required": ["values"],
        },
        ui_schema={
            "values": {
                "ui:widget": "tags",
                "ui:placeholder": "Enter allowed values, press Enter to add",
            }
        },
        examples=[
            {
                "name": "Gender Enum Check",
                "description": "Checks whether the gender field is M or F",
                "params": {"values": ["M", "F"]},
            }
        ],
    )

    # Register REGEX rule type
    rule_type_registry.register_rule_type(
        type_id="REGEX",
        name="Regular Expression Check",
        description="Checks whether a field value matches the specified "
        "regular expression",
        category="format",
        icon="code",
        parameters_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "title": "Regular Expression"}
            },
            "required": ["pattern"],
        },
        ui_schema={
            "pattern": {
                "ui:widget": "textarea",
                "ui:placeholder": "Enter regular expression",
            }
        },
        examples=[
            {
                "name": "Email Format Check",
                "description": "Checks whether the email address matches the "
                "standard format",
                "params": {
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
                },
            }
        ],
    )

    # Register DATE_FORMAT rule type
    rule_type_registry.register_rule_type(
        type_id="DATE_FORMAT",
        name="Date Format Check",
        description="Checks whether a date matches the specified format",
        category="format",
        icon="calendar_today",
        parameters_schema={
            "type": "object",
            "properties": {"format": {"type": "string", "title": "Date Format"}},
            "required": ["format"],
        },
        ui_schema={"format": {"ui:placeholder": "Enter date format, e.g. %Y-%m-%d"}},
        examples=[
            {
                "name": "Standard Date Format Check",
                "description": "Checks whether the date matches the YYYY-MM-DD format",
                "params": {"format": "%Y-%m-%d"},
            }
        ],
    )

    # Register LENGTH rule type
    rule_type_registry.register_rule_type(
        type_id="LENGTH",
        name="Length Check",
        description="Checks whether the string length is within the specified range",
        category="format",
        icon="text_fields",
        parameters_schema={
            "type": "object",
            "properties": {
                "min_length": {
                    "type": ["integer", "null"],
                    "title": "Minimum Length",
                },
                "max_length": {
                    "type": ["integer", "null"],
                    "title": "Maximum Length",
                },
            },
        },
        ui_schema={
            "min_length": {"ui:placeholder": "Enter minimum length"},
            "max_length": {"ui:placeholder": "Enter maximum length"},
        },
        examples=[
            {
                "name": "Phone Number Length Check",
                "description": "Checks whether the phone number is 11 digits",
                "params": {"min_length": 11, "max_length": 11},
            }
        ],
    )

    # Register SCHEMA rule type (table-level)
    rule_type_registry.register_rule_type(
        type_id="SCHEMA",
        name="Schema Check",
        description=(
            "Validates existence and expected data types for declared columns "
            "of a table (table-level rule)."
        ),
        category="validity",
        icon="table_chart",
        parameters_schema={
            "type": "object",
            "properties": {
                "columns": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"expected_type": {"type": "string"}},
                        "required": ["expected_type"],
                    },
                },
                "strict_mode": {"type": ["boolean", "null"]},
                "case_insensitive": {"type": ["boolean", "null"]},
            },
            "required": ["columns"],
        },
        ui_schema={},
        examples=[
            {
                "name": "users table schema",
                "description": "Check id/email/created_at types",
                "params": {
                    "columns": {
                        "id": {"expected_type": "INTEGER"},
                        "email": {"expected_type": "STRING"},
                        "created_at": {"expected_type": "DATETIME"},
                    },
                    "strict_mode": True,
                },
            }
        ],
    )

    # Register CUSTOM_SQL rule type
    rule_type_registry.register_rule_type(
        type_id="CUSTOM_SQL",
        name="Custom SQL Check",
        description="Performs checks using custom SQL statements",
        category="custom",
        icon="code",
        parameters_schema={
            "type": "object",
            "properties": {"sql": {"type": "string", "title": "SQL Statement"}},
            "required": ["sql"],
        },
        ui_schema={
            "sql": {
                "ui:widget": "textarea",
                "ui:placeholder": "Enter SQL statement, should return the number "
                "of abnormal records",
            }
        },
        examples=[
            {
                "name": "Custom Completeness Check",
                "description": "Checks whether there are records in the orders "
                "table without corresponding customers",
                "params": {
                    "sql": "SELECT COUNT(*) FROM orders WHERE customer_id "
                    "NOT IN (SELECT id FROM customers)"
                },
            }
        ],
    )

    logger.info("Successfully registered all built-in rule types")


# Automatically register built-in rule types when the module is imported
register_builtin_rule_types()
