"""
test datetime_utils module
"""

from datetime import datetime, timedelta, timezone

import pytest

from shared.utils.datetime_utils import ensure_utc, format_datetime, now, parse_datetime


def test_ensure_utc() -> None:
    # Test for None input.
    assert ensure_utc(None) is None

    # Testing the naive datetime implementation.
    naive_dt = datetime(2023, 6, 15, 8, 30, 45, 123000)
    utc_dt = ensure_utc(naive_dt)
    assert utc_dt is not None
    assert utc_dt.tzinfo == timezone.utc

    # Test non-UTC time zones.
    beijing_tz = timezone(timedelta(hours=8))
    beijing_dt = datetime(2023, 6, 15, 16, 30, 45, 123000, tzinfo=beijing_tz)
    utc_dt = ensure_utc(beijing_dt)
    assert utc_dt is not None
    assert utc_dt.tzinfo == timezone.utc
    assert utc_dt.hour == 8  # Beijing time at 4 PM corresponds to 8 AM UTC.


def test_format_datetime() -> None:
    # Test for None input.
    assert format_datetime(None) is None

    # Test UTC time formatting.
    utc_dt = datetime(2023, 6, 15, 8, 30, 45, 123000, tzinfo=timezone.utc)
    formatted = format_datetime(utc_dt)
    assert formatted == "2023-06-15T08:30:45.123Z"

    # Testing naive datetime formatting.
    naive_dt = datetime(2023, 6, 15, 8, 30, 45, 123000)
    formatted = format_datetime(naive_dt)
    assert formatted == "2023-06-15T08:30:45.123Z"


def test_parse_datetime() -> None:
    # Test for None input.
    assert parse_datetime(None) is None

    # Test with an empty string.
    assert parse_datetime("") is None

    # Test ISO format with a Z (Zulu) time designator.
    dt = parse_datetime("2023-06-15T08:30:45.123Z")
    assert dt is not None
    assert dt.year == 2023
    assert dt.month == 6
    assert dt.day == 15
    assert dt.hour == 8
    assert dt.minute == 30
    assert dt.second == 45
    assert dt.microsecond == 123000
    assert dt.tzinfo == timezone.utc

    # Test ISO format with timezone offset.
    dt = parse_datetime("2023-06-15T16:30:45.123+08:00")
    assert dt is not None
    assert dt.tzinfo == timezone.utc
    assert dt.hour == 8  # Beijing time at 4 PM corresponds to 8 AM UTC.


def test_now() -> None:
    # This code tests that the `now` function returns the current time in the UTC time zone.
    current = now()
    assert current.tzinfo == timezone.utc

    # Verify that the timestamp is close to the current time.
    utc_now = datetime.now(timezone.utc)
    diff = abs((current - utc_now).total_seconds())
    assert diff < 1  # The time difference should be less than one second.


if __name__ == "__main__":
    pytest.main(["-v", "test_rules_new.py"])
