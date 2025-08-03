"""
Datetime utility module

Provides unified datetime processing functions to ensure consistency of all datetime
operations in the system
"""

from datetime import datetime, timezone
from typing import Optional

# Define UTC timezone constant
UTC = timezone.utc


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Ensure the datetime object is in UTC timezone

    If the input is a naive datetime, add UTC timezone info
    If the input already has timezone info, convert to UTC

    Args:
        dt: Input datetime object

    Returns:
        Datetime object with UTC timezone info, returns None if input is None
    """
    if dt is None:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)

    return dt.astimezone(UTC)


def format_datetime(dt: Optional[datetime]) -> Optional[str]:
    """
    Format datetime as ISO 8601 string (UTC format with Z)

    Args:
        dt: Input datetime object

    Returns:
        Formatted ISO 8601 string, returns None if input is None
    """

    # Ensure UTC timezone
    dt = ensure_utc(dt)
    if dt is None:
        return None

    # Format as ISO 8601, keep 3 digits of milliseconds, add Z for UTC
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """
    Parse ISO 8601 formatted datetime string to datetime object

    Supports multiple common ISO 8601 variants, ensures returned datetime is in UTC

    Args:
        dt_str: ISO 8601 formatted datetime string

    Returns:
        Datetime object with UTC timezone info, returns None if input is None
    """
    if dt_str is None or dt_str.strip() == "":
        return None

    # Handle UTC time with Z
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"

    # Use fromisoformat to parse ISO 8601 string
    try:
        dt = datetime.fromisoformat(dt_str)
    except ValueError:
        # Try other common formats
        try:
            from dateutil import parser

            dt = parser.parse(dt_str)
        except (ImportError, ValueError):
            raise ValueError(f"Unable to parse datetime string: {dt_str}")

    # Ensure UTC timezone
    return ensure_utc(dt)


def now() -> datetime:
    """
    Get current UTC time

    Returns:
        Current UTC time with timezone info
    """
    return datetime.now(UTC)
