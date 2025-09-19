"""
Tests for TypeParser utility

Comprehensive test coverage for syntactic sugar type parsing and backward compatibility.
"""

from typing import Any

import pytest

from shared.enums.data_types import DataType
from shared.utils.type_parser import (
    TypeParseError,
    TypeParser,
    is_syntactic_sugar,
    normalize_type,
    parse_type,
)


class TestTypeParser:
    """Test TypeParser class methods"""

    def test_parse_simple_types(self) -> None:
        """Test parsing of simple type names."""
        # Test all supported simple types
        test_cases = [
            ("string", {"type": DataType.STRING.value}),
            ("str", {"type": DataType.STRING.value}),
            ("integer", {"type": DataType.INTEGER.value}),
            ("int", {"type": DataType.INTEGER.value}),
            ("float", {"type": DataType.FLOAT.value}),
            ("boolean", {"type": DataType.BOOLEAN.value}),
            ("bool", {"type": DataType.BOOLEAN.value}),
            ("date", {"type": DataType.DATE.value}),
            ("datetime", {"type": DataType.DATETIME.value}),
        ]

        for input_type, expected in test_cases:
            result = TypeParser.parse_type_definition(input_type)
            assert result == expected

    def test_parse_case_insensitive(self) -> None:
        """Test that parsing is case insensitive."""
        test_cases = ["STRING", "String", "sTrInG", "INTEGER", "Int", "FLOAT", "Float"]

        for input_type in test_cases:
            result = TypeParser.parse_type_definition(input_type)
            assert "type" in result
            assert result["type"] in [dt.value for dt in DataType]

    def test_parse_string_with_length(self) -> None:
        """Test parsing string with length specification."""
        test_cases = [
            ("string(50)", {"type": DataType.STRING.value, "max_length": 50}),
            ("STRING(255)", {"type": DataType.STRING.value, "max_length": 255}),
            ("str(10)", {"type": DataType.STRING.value, "max_length": 10}),
            (
                "string( 100 )",
                {"type": DataType.STRING.value, "max_length": 100},
            ),  # with spaces
        ]

        for input_type, expected in test_cases:
            result = TypeParser.parse_type_definition(input_type)
            assert result == expected

    def test_parse_float_with_precision_scale(self) -> None:
        """Test parsing float with precision and scale."""
        test_cases = [
            (
                "float(10,2)",
                {"type": DataType.FLOAT.value, "precision": 10, "scale": 2},
            ),
            (
                "FLOAT(12,4)",
                {"type": DataType.FLOAT.value, "precision": 12, "scale": 4},
            ),
            (
                "float( 8 , 3 )",
                {"type": DataType.FLOAT.value, "precision": 8, "scale": 3},
            ),  # with spaces
            (
                "float(15,0)",
                {"type": DataType.FLOAT.value, "precision": 15, "scale": 0},
            ),
        ]

        for input_type, expected in test_cases:
            result = TypeParser.parse_type_definition(input_type)
            assert result == expected

    def test_parse_datetime_with_format(self) -> None:
        """Test parsing datetime with format specification."""
        test_cases = [
            (
                "datetime('yyyymmdd')",
                {"type": DataType.DATETIME.value, "format": "yyyymmdd"},
            ),
            (
                'DATETIME("yyyy-mm-dd")',
                {"type": DataType.DATETIME.value, "format": "yyyy-mm-dd"},
            ),
            (
                "datetime( 'dd/mm/yyyy hh:mm:ss' )",
                {"type": DataType.DATETIME.value, "format": "dd/mm/yyyy hh:mm:ss"},
            ),
        ]

        for input_type, expected in test_cases:
            result = TypeParser.parse_type_definition(input_type)
            assert result == expected

    def test_parse_detailed_format_backward_compatibility(self) -> None:
        """Test parsing detailed JSON format for backward compatibility."""
        test_cases: list[tuple[dict, dict]] = [
            ({"type": "string"}, {"type": DataType.STRING.value}),
            (
                {"type": "string", "max_length": 100},
                {"type": DataType.STRING.value, "max_length": 100},
            ),
            (
                {"type": "float", "precision": 10, "scale": 2},
                {"type": DataType.FLOAT.value, "precision": 10, "scale": 2},
            ),
            (
                {"type": "datetime", "format": "yyyy-mm-dd"},
                {"type": DataType.DATETIME.value, "format": "yyyy-mm-dd"},
            ),
        ]

        for input_type, expected in test_cases:
            result = TypeParser.parse_type_definition(input_type)
            assert result == expected

    def test_error_cases(self) -> None:
        """Test error handling for invalid type definitions."""
        error_cases: list[tuple[Any, str]] = [
            ("invalid_type", "Cannot parse type definition"),
            ("string(-1)", "String length must be positive"),
            ("float(0,2)", "Float precision must be positive"),
            ("float(5,-1)", "Float scale cannot be negative"),
            ("float(3,5)", "Float scale cannot be greater than precision"),
            ({"type": "unknown"}, "Unsupported type 'unknown'"),
            ({"missing_type": "value"}, "Detailed format must include 'type' field"),
            (123, "Type definition must be string or dict"),
            (None, "Type definition must be string or dict"),
        ]

        for input_type, expected_error in error_cases:
            with pytest.raises(TypeParseError) as exc_info:
                TypeParser.parse_type_definition(input_type)
            assert expected_error in str(exc_info.value)

    def test_metadata_validation(self) -> None:
        """Test metadata validation for type consistency."""
        # Test invalid metadata combinations in detailed format
        invalid_cases: list[tuple[dict, str]] = [
            (
                {"type": "integer", "max_length": 10},
                "max_length can only be specified for STRING type",
            ),
            (
                {"type": "string", "precision": 5},
                "precision/scale can only be specified for FLOAT type",
            ),
            (
                {"type": "boolean", "scale": 2},
                "precision/scale can only be specified for FLOAT type",
            ),
            (
                {"type": "date", "format": "hh:mi:ss"},
                "format can only be specified for DATETIME type",
            ),
            (
                {"type": "string", "max_length": 0},
                "max_length must be a positive integer",
            ),
            ({"type": "float", "precision": 0}, "precision must be a positive integer"),
            ({"type": "float", "scale": -1}, "scale must be a non-negative integer"),
            (
                {"type": "float", "precision": 3, "scale": 5},
                "scale cannot be greater than precision",
            ),
        ]

        for input_type, expected_error in invalid_cases:
            with pytest.raises(TypeParseError) as exc_info:
                TypeParser.parse_type_definition(input_type)
            assert expected_error in str(exc_info.value)

    def test_is_syntactic_sugar(self) -> None:
        """Test identification of syntactic sugar formats."""
        sugar_cases = [
            "string(50)",
            "float(10,2)",
            "datetime('yyyy-mm-dd')",
            "integer",
            "boolean",
        ]

        detailed_cases = [
            {"type": "string"},
            {"type": "float", "precision": 10},
            123,
            None,
        ]

        case: Any = None
        for case in sugar_cases:
            assert TypeParser.is_syntactic_sugar(case) is True

        for case in detailed_cases:
            assert TypeParser.is_syntactic_sugar(case) is False

    def test_normalize_to_detailed_format(self) -> None:
        """Test normalization to detailed format."""
        test_cases: list[tuple[str | dict, dict]] = [
            (
                "string(50)",
                {"type": "string", "desired_type": "STRING", "max_length": 50},
            ),
            (
                "float(10,2)",
                {"type": "float", "desired_type": "FLOAT", "precision": 10, "scale": 2},
            ),
            ({"type": "boolean"}, {"type": "boolean", "desired_type": "BOOLEAN"}),
        ]

        for input_type, expected_keys in test_cases:
            result = TypeParser.normalize_to_detailed_format(input_type)
            for key, value in expected_keys.items():
                assert result[key] == value


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_parse_type_function(self) -> None:
        """Test parse_type convenience function."""
        result = parse_type("string(100)")
        assert result == {"type": DataType.STRING.value, "max_length": 100}

    def test_is_syntactic_sugar_function(self) -> None:
        """Test is_syntactic_sugar convenience function."""
        assert is_syntactic_sugar("float(10,2)") is True
        assert is_syntactic_sugar({"type": "string"}) is False

    def test_normalize_type_function(self) -> None:
        """Test normalize_type convenience function."""
        result = normalize_type("string(50)")
        assert result["type"] == "string"
        assert result["desired_type"] == "STRING"
        assert result["max_length"] == 50


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_whitespace_handling(self) -> None:
        """Test handling of various whitespace scenarios."""
        test_cases = [
            ("  string  ", {"type": DataType.STRING.value}),
            ("string(  50  )", {"type": DataType.STRING.value, "max_length": 50}),
            (
                "float( 10 , 2 )",
                {"type": DataType.FLOAT.value, "precision": 10, "scale": 2},
            ),
            (
                "datetime( ' format ' )",
                {"type": DataType.DATETIME.value, "format": " format "},
            ),
        ]

        for input_type, expected in test_cases:
            result = TypeParser.parse_type_definition(input_type)
            assert result == expected

    def test_boundary_values(self) -> None:
        """Test boundary values for numeric parameters."""
        # Test valid boundary values
        valid_cases = [
            ("string(1)", {"type": DataType.STRING.value, "max_length": 1}),
            ("float(1,0)", {"type": DataType.FLOAT.value, "precision": 1, "scale": 0}),
            ("float(1,1)", {"type": DataType.FLOAT.value, "precision": 1, "scale": 1}),
        ]

        for input_type, expected in valid_cases:
            result = TypeParser.parse_type_definition(input_type)
            assert result == expected

        # Test invalid boundary values
        invalid_cases = [
            ("string(0)", "String length must be positive"),
            ("float(0,0)", "Float precision must be positive"),
        ]

        for input_type, expected_error in invalid_cases:
            with pytest.raises(TypeParseError) as exc_info:
                TypeParser.parse_type_definition(input_type)
            assert expected_error in str(exc_info.value)

    def test_quote_variations(self) -> None:
        """Test different quote styles for datetime format."""
        test_cases = [
            ("datetime('format')", "format"),
            ('datetime("format")', "format"),
            ("datetime('format with spaces')", "format with spaces"),
            ("datetime(\"format with 'quotes'\")", "format with 'quotes'"),
        ]

        for input_type, expected_format in test_cases:
            result = TypeParser.parse_type_definition(input_type)
            assert result == {
                "type": DataType.DATETIME.value,
                "format": expected_format,
            }

    def test_large_numbers(self) -> None:
        """Test handling of large numeric values."""
        test_cases = [
            ("string(65535)", {"type": DataType.STRING.value, "max_length": 65535}),
            (
                "float(38,10)",
                {"type": DataType.FLOAT.value, "precision": 38, "scale": 10},
            ),
        ]

        for input_type, expected in test_cases:
            result = TypeParser.parse_type_definition(input_type)
            assert result == expected


class TestIntegrationWithDataTypeEnum:
    """Test integration with DataType enum"""

    def test_all_data_types_supported(self) -> None:
        """Test that all DataType enum values are supported."""
        type_mappings = {
            "string": DataType.STRING,
            "integer": DataType.INTEGER,
            "float": DataType.FLOAT,
            "boolean": DataType.BOOLEAN,
            "date": DataType.DATE,
            "datetime": DataType.DATETIME,
        }

        for type_name, expected_enum in type_mappings.items():
            result = TypeParser.parse_type_definition(type_name)
            assert result["type"] == expected_enum.value

    def test_enum_value_consistency(self) -> None:
        """Test that returned type values match DataType enum values."""
        result = TypeParser.parse_type_definition("string")
        assert result["type"] == DataType.STRING.value == "STRING"

        result = TypeParser.parse_type_definition("float(10,2)")
        assert result["type"] == DataType.FLOAT.value == "FLOAT"


@pytest.mark.parametrize(
    "input_type,expected",
    [
        ("string(50)", {"type": "STRING", "max_length": 50}),
        ("float(12,2)", {"type": "FLOAT", "precision": 12, "scale": 2}),
        ("datetime('yyyymmdd')", {"type": "DATETIME", "format": "yyyymmdd"}),
        ("integer", {"type": "INTEGER"}),
        ("boolean", {"type": "BOOLEAN"}),
        ("date", {"type": "DATE"}),
    ],
)
def test_acceptance_criteria_examples(input_type: str, expected: dict) -> None:
    """Test the specific examples from the acceptance criteria."""
    result = parse_type(input_type)
    assert result == expected
