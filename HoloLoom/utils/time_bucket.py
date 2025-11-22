"""Time bucketing utilities for HoloLoom"""

from datetime import datetime
from typing import Union

TimeInput = Union[str, datetime, float]


def to_utc_datetime(time_input: TimeInput) -> datetime:
    """Convert various time inputs to UTC datetime"""
    if isinstance(time_input, datetime):
        return time_input
    elif isinstance(time_input, (int, float)):
        return datetime.fromtimestamp(time_input)
    elif isinstance(time_input, str):
        return datetime.fromisoformat(time_input)
    else:
        raise ValueError(f"Unsupported time input type: {type(time_input)}")


def time_bucket(time_input: TimeInput, bucket_size_seconds: int = 3600) -> str:
    """Bucket time into intervals for temporal memory organization"""
    dt = to_utc_datetime(time_input)
    timestamp = dt.timestamp()
    bucket_start = int(timestamp // bucket_size_seconds) * bucket_size_seconds
    return datetime.fromtimestamp(bucket_start).isoformat()
