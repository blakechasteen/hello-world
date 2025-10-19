"""Utility helpers for coarse-grained time bucketing.

The knowledge-graph migration scripts stored under ``archive/`` rely on
YYYY-MM-DD-{morning|afternoon|evening|night} buckets to attach events to
"threads" representing temporal continuity.  Several upcoming ingestion
pipelines need the same logic when constructing in-memory graphs, so we
provide a shared implementation here.

``time_bucket`` accepts ``datetime`` objects (naive or timezone aware),
ISO-8601 strings, or UNIX epoch seconds.  All inputs are normalised to
UTC before the bucket label is generated so that backend services agree
on the same coarse time slices regardless of local timezone.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Union

TimeInput = Union[datetime, int, float, str]


def to_utc_datetime(value: TimeInput) -> datetime:
    """Normalise supported inputs to a timezone-aware ``datetime``.

    Args:
        value: A ``datetime`` instance (naive or aware), an ISO-8601 string,
            or a UNIX timestamp expressed as ``int``/``float`` seconds.

    Returns:
        ``datetime`` instance with ``tzinfo`` set to UTC.

    Raises:
        TypeError: If the value type is unsupported.
        ValueError: If a string cannot be parsed as a datetime.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Datetime string is empty")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid datetime string: {value}") from exc
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    raise TypeError(f"Unsupported datetime value type: {type(value)!r}")


def time_bucket(value: TimeInput) -> str:
    """Bucket a timestamp into ``YYYY-MM-DD-{part_of_day}``.

    The day-part boundaries mirror the legacy TypeScript implementation
    shipped with the Neo4j ingestion tooling:

    * ``morning`` – 05:00 ≤ hour < 12:00
    * ``afternoon`` – 12:00 ≤ hour < 17:00
    * ``evening`` – 17:00 ≤ hour < 22:00
    * ``night`` – 22:00 ≤ hour or hour < 05:00

    Args:
        value: Supported timestamp representation.

    Returns:
        Bucket label string.
    """
    dt = to_utc_datetime(value)
    hour = dt.hour

    if 12 <= hour < 17:
        part = "afternoon"
    elif 17 <= hour < 22:
        part = "evening"
    elif hour >= 22 or hour < 5:
        part = "night"
    else:
        part = "morning"

    return f"{dt:%Y-%m-%d}-{part}"


__all__ = ["time_bucket", "to_utc_datetime", "TimeInput"]
