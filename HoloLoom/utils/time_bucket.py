"""Time bucketing utilities for HoloLoom"""

from datetime import datetime, timezone
from typing import Union

TimeInput = Union[str, datetime, float]


def to_utc_datetime(time_input: TimeInput) -> datetime:
    """Convert various time inputs to UTC datetime.

    Args:
        time_input: ISO string, datetime, or Unix timestamp

    Returns:
        UTC-aware datetime object
    """
    if isinstance(time_input, datetime):
        if time_input.tzinfo is None:
            # Naive datetime - assume UTC
            return time_input.replace(tzinfo=timezone.utc)
        else:
            # Convert to UTC
            return time_input.astimezone(timezone.utc)
    elif isinstance(time_input, (int, float)):
        return datetime.fromtimestamp(time_input, tz=timezone.utc)
    elif isinstance(time_input, str):
        # Handle ISO format with Z suffix
        if time_input.endswith('Z'):
            time_input = time_input[:-1] + '+00:00'
        dt = datetime.fromisoformat(time_input)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    else:
        raise ValueError(f"Unsupported time input type: {type(time_input)}")


def _get_time_of_day(hour: int) -> str:
    """Get human-readable time of day from hour.

    Args:
        hour: Hour in 24-hour format (0-23)

    Returns:
        One of: 'night', 'morning', 'afternoon', 'evening'
    """
    if 0 <= hour < 6:
        return "night"
    elif 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 22:
        return "evening"
    else:  # 22-23
        return "night"


def time_bucket(time_input: TimeInput, bucket_size_seconds: int = 3600) -> str:
    """Bucket time into human-readable intervals for temporal memory organization.

    Args:
        time_input: Time as ISO string, datetime, or Unix timestamp
        bucket_size_seconds: Ignored (kept for API compatibility), always uses time-of-day

    Returns:
        Human-readable bucket like '2024-03-10-afternoon'
    """
    dt = to_utc_datetime(time_input)
    date_str = dt.strftime("%Y-%m-%d")
    time_of_day = _get_time_of_day(dt.hour)
    return f"{date_str}-{time_of_day}"
