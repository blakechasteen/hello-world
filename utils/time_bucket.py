# utils/time_bucket.py
from datetime import datetime, timezone

def time_bucket(dt) -> str:
    """
    Buckets a datetime (aware or naive) into YYYY-MM-DD-{morning|afternoon|evening|night} in UTC.
    Accepts datetime, ISO string, or epoch seconds.
    """
    if isinstance(dt, (int, float)):
        d = datetime.fromtimestamp(dt, tz=timezone.utc)
    elif isinstance(dt, str):
        # parse ISO-like; fallback naive as UTC
        try:
            d = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except Exception as e:
            raise ValueError(f"Invalid date string: {dt}") from e
    elif isinstance(dt, datetime):
        d = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    else:
        raise TypeError("Unsupported dt type")

    hh = d.hour
    if 12 <= hh < 17:
        part = 'afternoon'
    elif 17 <= hh < 22:
        part = 'evening'
    elif hh >= 22 or hh < 5:
        part = 'night'
    else:
        part = 'morning'
    return f"{d:%Y-%m-%d}-{part}"