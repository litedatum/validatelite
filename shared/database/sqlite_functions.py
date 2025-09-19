"""
SQLite Custom Validation Functions

Provides numerical precision validation functionality for SQLite,
 replacing REGEX validation
"""

from typing import Any


def validate_integer_digits(value: Any, max_digits: int) -> bool:
    """
    Validate whether integer digits do not exceed the specified number of digits

    Args:
        value: Value to be validated
        max_digits: Maximum allowed digits

    Returns:
        bool: True indicates validation passed, False indicates validation failed

    Examples:
        validate_integer_digits(12345, 5) -> True
        validate_integer_digits(-23456, 5) -> True (negative sign not counted as digit)
        validate_integer_digits(123456, 5) -> False
        validate_integer_digits("abc", 5) -> False
        validate_integer_digits(12.34, 5) -> False (has decimal part)
    """
    if value is None:
        return True  # NULL values skip validation

    try:
        # Try to convert to float then to integer, ensuring it's numerical
        float_val = float(value)
        int_val = int(float_val)

        # Check if there's a decimal part
        if float_val != int_val:
            return False  # Has decimal part, not an integer

        # Calculate digit count (absolute value, remove negative sign)
        digit_count = len(str(abs(int_val)))
        return digit_count <= max_digits

    except (ValueError, TypeError, OverflowError):
        return False  # Invalid values return failure


def validate_string_length(value: Any, max_length: int) -> bool:
    """
    Validate whether string length does not exceed the specified length

    Args:
        value: Value to be validated
        max_length: Maximum allowed length

    Returns:
        bool: True indicates validation passed, False indicates validation failed
    """
    if value is None:
        return True  # NULL values skip validation

    try:
        str_val = str(value)
        return len(str_val) <= max_length
    except Exception:
        return False


def validate_float_precision(value: Any, precision: int, scale: int) -> bool:
    """
    Validate floating point precision and decimal places

    Args:
        value: Value to be validated
        precision: Total precision (integer digits + decimal digits)
        scale: Number of decimal places

    Returns:
        bool: True indicates validation passed, False indicates validation failed

    Examples:
        validate_float_precision(123.45, 5, 2) -> True
        validate_float_precision(1234.56, 5, 2) -> False (total digits exceed 5)
        validate_float_precision(123.456, 5, 2) -> False (decimal places exceed 2)
    """
    if value is None:
        return True  # NULL values skip validation

    try:
        float_val = float(value)
        val_str = str(float_val)

        # Remove negative sign
        if val_str.startswith("-"):
            val_str = val_str[1:]

        if "." in val_str:
            # Case with decimal point
            integer_part, decimal_part = val_str.split(".")

            # Remove trailing zeros
            decimal_part = decimal_part.rstrip("0")

            # Special case: when precision == scale, it means only decimal part,
            #  integer part must be 0
            if precision == scale:
                # Only allow 0.xxxx format, integer part must be 0 and not counted
                #  in precision
                if integer_part != "0":
                    return False
                int_digits = 0  # Integer part 0 is not counted in precision
            else:
                # Normal case: integer part is counted in precision
                int_digits = len(integer_part) if integer_part != "0" else 1

            dec_digits = len(decimal_part)

            # Check integer and decimal digit constraints
            # Integer digits cannot exceed (precision - scale), decimal digits cannot
            #  exceed scale
            max_integer_digits = precision - scale
            return int_digits <= max_integer_digits and dec_digits <= scale
        else:
            # Integer case
            int_digits = len(val_str) if val_str != "0" else 1
            # Integers must also follow precision-scale constraints
            max_integer_digits = precision - scale
            return int_digits <= max_integer_digits

    except (ValueError, TypeError, OverflowError):
        return False


def validate_integer_range_by_digits(value: Any, max_digits: int) -> bool:
    """
    Validate integer digits through range checking (fallback solution)

    Args:
        value: Value to be validated
        max_digits: Maximum allowed digits

    Returns:
        bool: True indicates validation passed, False indicates validation failed
    """
    if value is None:
        return True

    try:
        int_val = int(float(value))
        max_val: int = 10**max_digits - 1  # maximum value for 5 digits is 99999
        min_val: int = -(10**max_digits - 1)  # minimum value for 5 digits is -99999
        return min_val <= int_val <= max_val
    except (ValueError, TypeError, OverflowError):
        return False


# For SQLite registration convenience, provide failure detection versions
def detect_invalid_integer_digits(value: Any, max_digits: int) -> bool:
    """
    Detect values that do not meet integer digit requirements
      (used for COUNT failed records)
    """
    return not validate_integer_digits(value, max_digits)


def detect_invalid_string_length(value: Any, max_length: int) -> bool:
    """Detect values that do not meet string length requirements"""
    return not validate_string_length(value, max_length)


def detect_invalid_float_precision(value: Any, precision: int, scale: int) -> bool:
    """Detect values that do not meet floating point precision requirements"""
    return not validate_float_precision(value, precision, scale)


def validate_date_format(value: Any, format_pattern: str) -> bool:
    """Validate date string format and semantic correctness

    Args:
        value: Date value to be validated (string or integer)
        format_pattern: Date format pattern (YYYY-MM-DD, YYYYMMDD, etc.)

    Returns:
        bool: True indicates validation passed, False indicates validation failed

    Examples:
        validate_date_format("2023-12-25", "YYYY-MM-DD") -> True
        validate_date_format("2023-02-31", "YYYY-MM-DD") -> False (invalid date)
        validate_date_format("not-a-date", "YYYY-MM-DD") -> False (invalid format)
        validate_date_format(20231225, "YYYYMMDD") -> True
        validate_date_format(20230231, "YYYYMMDD") -> False (invalid date)
    """
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return True  # NULL or empty strings are not date format errors

    try:
        from datetime import datetime

        # Convert format pattern to Python datetime format
        python_format = _convert_format_to_python(format_pattern)

        # Convert value to string if it's not already
        date_str = str(value)

        # Parse date using the specified format
        parsed_date = datetime.strptime(date_str, python_format)

        # Round-trip validation to catch semantic errors like 2000-02-31
        return parsed_date.strftime(python_format) == date_str

    except (ValueError, TypeError):
        return False


def _convert_format_to_python(format_pattern: str) -> str:
    """Convert custom format pattern to Python datetime format"""
    # Handle both case variations (YYYY/yyyy, MM/mm, etc.)
    pattern_map = {
        "YYYY": "%Y",
        "yyyy": "%Y",
        "MM": "%m",
        "mm": "%m",
        "DD": "%d",
        "dd": "%d",
        "HH": "%H",
        "hh": "%H",
        "MI": "%M",
        "mi": "%M",
        "SS": "%S",
        "ss": "%S",
    }

    python_format = format_pattern
    # Sort by length (descending) to avoid partial replacements
    for fmt in sorted(pattern_map.keys(), key=len, reverse=True):
        python_format = python_format.replace(fmt, pattern_map[fmt])

    return python_format


def is_valid_date(value: Any, format_pattern: str) -> bool:
    """Alias for validate_date_format for SQLite registration"""
    return validate_date_format(value, format_pattern)
