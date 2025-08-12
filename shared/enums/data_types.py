"""
DataType enumeration

Defines the canonical data types used across the system for schema validation
and type comparisons. All CLI and core components should use these enum values
instead of free-form strings.
"""

from enum import Enum


class DataType(str, Enum):
    """Canonical data type enumeration used by Schema rules."""

    STRING = "STRING"
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"
    DATE = "DATE"
    DATETIME = "DATETIME"


__all__ = ["DataType"]
