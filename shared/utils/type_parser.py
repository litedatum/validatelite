"""
Type Definition Parser

Provides reusable parsing logic for syntactic sugar type definitions
while maintaining backward compatibility with detailed JSON format.

Supports formats like:
- string(50) → {"type": "string", "max_length": 50}
- integer(10) → {"type": "integer", "max_digits": 10}
- float(12,2) → {"type": "float", "precision": 12, "scale": 2}
- datetime('yyyymmdd') → {"type": "datetime", "format": "yyyymmdd"}
- date('YYYY-MM-DD') → {"type": "date", "format": "YYYY-MM-DD"}
"""

import re
from typing import Any, Dict, Union

from shared.enums.data_types import DataType


class TypeParseError(Exception):
    """Raised when type definition parsing fails."""

    pass


class TypeParser:
    """
    Parser for type definitions supporting both syntactic sugar and
      detailed JSON formats.
    """

    # Supported base types
    _SUPPORTED_TYPES = {
        "string": DataType.STRING,
        "str": DataType.STRING,  # Allow str as alias for string
        "integer": DataType.INTEGER,
        "int": DataType.INTEGER,  # Allow int as alias for integer
        "float": DataType.FLOAT,
        "boolean": DataType.BOOLEAN,
        "bool": DataType.BOOLEAN,  # Allow bool as alias for boolean
        "date": DataType.DATE,
        "datetime": DataType.DATETIME,
    }

    # Regex patterns for syntactic sugar parsing
    _STRING_PATTERN = re.compile(r"^(string|str)\s*\(\s*(-?\d+)\s*\)$", re.IGNORECASE)
    _INTEGER_PATTERN = re.compile(r"^(integer|int)\s*\(\s*(-?\d+)\s*\)$", re.IGNORECASE)
    _FLOAT_PATTERN = re.compile(
        r"^float\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$", re.IGNORECASE
    )
    _DATETIME_PATTERN = re.compile(
        r'^datetime\s*\(\s*[\'"](.+?)[\'"]\s*\)$', re.IGNORECASE
    )
    _DATE_PATTERN = re.compile(r'^date\s*\(\s*[\'"](.+?)[\'"]\s*\)$', re.IGNORECASE)
    _SIMPLE_TYPE_PATTERN = re.compile(
        r"^(string|str|integer|int|float|boolean|bool|date|datetime)$", re.IGNORECASE
    )

    @classmethod
    def parse_type_definition(
        cls, type_def: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Parse a type definition that can be either:
        1. A string with syntactic sugar (e.g., "string(50)", "float(12,2)")
        2. A detailed JSON object (backward compatibility)

        Args:
            type_def: Type definition as string or dict

        Returns:
            Dict containing parsed type information with keys:
            - type: Canonical type name (STRING, INTEGER, etc.)
            - Additional metadata keys based on type (max_length, precision,
              scale, format)

        Raises:
            TypeParseError: If parsing fails or type is unsupported
        """
        if isinstance(type_def, dict):
            return cls._parse_detailed_format(type_def)
        elif isinstance(type_def, str):
            return cls._parse_syntactic_sugar(type_def.strip())
        else:
            raise TypeParseError(
                f"Type definition must be string or dict, got {type(type_def)}"
            )

    @classmethod
    def _parse_detailed_format(cls, type_def: Dict[str, Any]) -> Dict[str, Any]:
        """Parse detailed JSON format (backward compatibility)."""
        if "type" not in type_def:
            raise TypeParseError("Detailed format must include 'type' field")

        type_name = str(type_def["type"]).lower()
        if type_name not in cls._SUPPORTED_TYPES:
            raise TypeParseError(f"Unsupported type '{type_name}' in detailed format")

        result = {"type": cls._SUPPORTED_TYPES[type_name].value}

        # Copy over additional metadata
        metadata_fields = ["max_length", "precision", "scale", "format"]
        for field in metadata_fields:
            if field in type_def:
                result[field] = type_def[field]

        # Validate metadata consistency
        cls._validate_metadata(result)

        return result

    @classmethod
    def _parse_syntactic_sugar(cls, type_str: str) -> Dict[str, Any]:
        """Parse syntactic sugar format."""
        # Try string(length) pattern
        match = cls._STRING_PATTERN.match(type_str)
        if match:
            length = int(match.group(2))
            if length <= 0:
                raise TypeParseError("String length must be positive")
            return {"type": DataType.STRING.value, "max_length": length}

        # Try integer(digits) pattern
        match = cls._INTEGER_PATTERN.match(type_str)
        if match:
            digits = int(match.group(2))
            if digits <= 0:
                raise TypeParseError("Integer digits must be positive")
            return {"type": DataType.INTEGER.value, "max_digits": digits}

        # Try float(precision,scale) pattern
        match = cls._FLOAT_PATTERN.match(type_str)
        if match:
            precision = int(match.group(1))
            scale = int(match.group(2))
            if precision <= 0:
                raise TypeParseError("Float precision must be positive")
            if scale < 0:
                raise TypeParseError("Float scale cannot be negative")
            if scale > precision:
                raise TypeParseError("Float scale cannot be greater than precision")
            return {
                "type": DataType.FLOAT.value,
                "precision": precision,
                "scale": scale,
            }

        # Try datetime('format') pattern
        match = cls._DATETIME_PATTERN.match(type_str)
        if match:
            format_str = match.group(1)
            return {"type": DataType.DATETIME.value, "format": format_str}

        # Try date('format') pattern
        match = cls._DATE_PATTERN.match(type_str)
        if match:
            format_str = match.group(1)
            return {"type": DataType.DATE.value, "format": format_str}

        # Try simple type names
        match = cls._SIMPLE_TYPE_PATTERN.match(type_str)
        if match:
            type_name = match.group(1).lower()
            return {"type": cls._SUPPORTED_TYPES[type_name].value}

        raise TypeParseError(f"Cannot parse type definition '{type_str}'")

    @classmethod
    def _validate_metadata(cls, parsed_type: Dict[str, Any]) -> None:
        """Validate that metadata is consistent with type."""
        type_value = parsed_type.get("type")

        # Validate max_length is only for strings
        if "max_length" in parsed_type:
            if type_value != DataType.STRING.value:
                raise TypeParseError(
                    "max_length can only be specified for STRING type, "
                    f"not {type_value}"
                )
            if (
                not isinstance(parsed_type["max_length"], int)
                or parsed_type["max_length"] <= 0
            ):
                raise TypeParseError("max_length must be a positive integer")

        # Validate max_digits is only for integers
        if "max_digits" in parsed_type:
            if type_value != DataType.INTEGER.value:
                raise TypeParseError(
                    "max_digits can only be specified for INTEGER type, "
                    f"not {type_value}"
                )
            if (
                not isinstance(parsed_type["max_digits"], int)
                or parsed_type["max_digits"] <= 0
            ):
                raise TypeParseError("max_digits must be a positive integer")

        # Validate precision/scale are only for floats
        if "precision" in parsed_type or "scale" in parsed_type:
            if type_value != DataType.FLOAT.value:
                raise TypeParseError(
                    "precision/scale can only be specified for FLOAT type, "
                    f"not {type_value}"
                )

        if "precision" in parsed_type:
            if (
                not isinstance(parsed_type["precision"], int)
                or parsed_type["precision"] <= 0
            ):
                raise TypeParseError("precision must be a positive integer")

        if "scale" in parsed_type:
            if not isinstance(parsed_type["scale"], int) or parsed_type["scale"] < 0:
                raise TypeParseError("scale must be a non-negative integer")
            if (
                "precision" in parsed_type
                and parsed_type["scale"] > parsed_type["precision"]
            ):
                raise TypeParseError("scale cannot be greater than precision")

        # Validate format is only for datetime and date
        if "format" in parsed_type:
            if type_value not in (DataType.DATETIME.value, DataType.DATE.value):
                raise TypeParseError(
                    f"format can only be specified for DATETIME or DATE type, "
                    f"not {type_value}"
                )

            # For DATE type, validate that format doesn't contain time components
            if type_value == DataType.DATE.value:
                format_str = parsed_type["format"]
                time_indicators = ["h", "H", "m", "M", "s", "S", "a", "A", "p", "P"]
                if any(indicator in format_str for indicator in time_indicators):
                    raise TypeParseError(
                        "format can only be specified for DATETIME type"
                    )

    @classmethod
    def is_syntactic_sugar(cls, type_def: Union[str, Dict[str, Any]]) -> bool:
        """Check if a type definition uses syntactic sugar format."""
        if not isinstance(type_def, str):
            return False

        type_str = type_def.strip()
        return bool(
            cls._STRING_PATTERN.match(type_str)
            or cls._INTEGER_PATTERN.match(type_str)
            or cls._FLOAT_PATTERN.match(type_str)
            or cls._DATETIME_PATTERN.match(type_str)
            or cls._DATE_PATTERN.match(type_str)
            or cls._SIMPLE_TYPE_PATTERN.match(type_str)
        )

    @classmethod
    def normalize_to_detailed_format(
        cls, type_def: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Normalize any type definition to detailed format for backward compatibility.

        Args:
            type_def: Type definition in any supported format

        Returns:
            Dict in detailed format that existing code can use
        """
        parsed = cls.parse_type_definition(type_def)

        # Convert canonical type back to lowercase for existing code compatibility
        if "type" in parsed:
            # Keep the canonical uppercase form for new code, but also provide lowercase
            parsed["desired_type"] = parsed["type"]  # For schema executor
            parsed["type"] = parsed["type"].lower()  # For backward compatibility

        return parsed

    @classmethod
    def parse_desired_type_for_core(
        cls, desired_type_def: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Parse desired_type definition and return fields with desired_ prefix
        for core layer.

        This method handles the CLI-to-core interface naming for desired_type
        fields, ensuring no conflicts with existing type field names.

        Args:
            desired_type_def: Desired type definition in syntactic sugar or
            detailed format

        Returns:
            Dict with desired_ prefixed field names suitable for core layer:
            {
                "desired_type": "STRING",
                "desired_max_length": 50,
                "desired_precision": 10,
                "desired_scale": 2,
                "desired_format": "YYYY-MM-DD"
            }

        Example:
            parse_desired_type_for_core("string(50)")
            → {"desired_type": "STRING", "desired_max_length": 50}

            parse_desired_type_for_core("float(10,2)")
            → {"desired_type": "FLOAT", "desired_precision": 10, "desired_scale": 2}
        """
        # Parse the desired type definition using existing logic
        parsed = cls.parse_type_definition(desired_type_def)

        # Transform to core layer format with desired_ prefix
        core_format = {}

        # Main type field
        if "type" in parsed:
            core_format["desired_type"] = parsed["type"]

        # Metadata fields with desired_ prefix
        metadata_fields = ["max_length", "precision", "scale", "format"]
        for field in metadata_fields:
            if field in parsed:
                core_format[f"desired_{field}"] = parsed[field]

        return core_format


# Convenience functions for common usage patterns
def parse_type(type_def: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Convenience function to parse a type definition."""
    return TypeParser.parse_type_definition(type_def)


def is_syntactic_sugar(type_def: Union[str, Dict[str, Any]]) -> bool:
    """Convenience function to check if type definition uses syntactic sugar."""
    return TypeParser.is_syntactic_sugar(type_def)


def normalize_type(type_def: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Convenience function to normalize type definition to detailed format."""
    return TypeParser.normalize_to_detailed_format(type_def)


def parse_desired_type_for_core(
    desired_type_def: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Convenience function to parse desired_type with proper core layer
    field naming.
    """
    return TypeParser.parse_desired_type_for_core(desired_type_def)
