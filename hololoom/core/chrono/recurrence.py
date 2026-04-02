"""
Recurrence - Human-readable recurrence patterns for agent calendars.

Syntax designed to be readable in config files and conversation:

    "every:day:9am"           - Daily at 9am
    "every:monday:9am"        - Weekly on Monday at 9am
    "every:weekday:8am"       - Mon-Fri at 8am
    "every:3:days"            - Every 3 days
    "every:2:weeks:monday"    - Every 2 weeks on Monday
    "every:month:15"          - Monthly on the 15th
    "every:month:last"        - Monthly on the last day
    "yearly:mar:15"           - Yearly on March 15
    "yearly:mar:first:monday" - Yearly on first Monday of March
    "once:2026-04-15"         - One-time on a specific date
    "once:2026-04-15:10am"   - One-time with time

Created: 2026-04-02
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date, time
from enum import Enum
from typing import Optional


WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

ORDINALS = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "last": -1,
}

TIME_PATTERN = re.compile(r"^(\d{1,2})(am|pm)?$")


class RecurrenceType(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    INTERVAL_DAYS = "interval_days"
    INTERVAL_WEEKS = "interval_weeks"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    ONCE = "once"


@dataclass(frozen=True)
class Recurrence:
    """Parsed recurrence pattern."""

    type: RecurrenceType
    expression: str  # original string

    # Time of day (None = all-day / midnight)
    at_hour: Optional[int] = None
    at_minute: int = 0

    # Weekly fields
    weekday: Optional[int] = None  # 0=Mon .. 6=Sun

    # Interval fields
    interval: int = 1

    # Monthly fields
    day_of_month: Optional[int] = None  # 1-31, -1 = last
    ordinal: Optional[int] = None  # 1st, 2nd... -1 = last
    ordinal_weekday: Optional[int] = None  # weekday for ordinal

    # Yearly fields
    month: Optional[int] = None  # 1-12

    # Once fields
    once_date: Optional[date] = None

    def next_occurrence(self, after: Optional[datetime] = None) -> datetime:
        """Calculate the next occurrence after a given datetime."""
        now = after or datetime.now()
        at_time = time(self.at_hour or 0, self.at_minute)

        if self.type == RecurrenceType.ONCE:
            if self.once_date is None:
                raise ValueError("once recurrence missing date")
            return datetime.combine(self.once_date, at_time)

        if self.type == RecurrenceType.DAILY:
            candidate = datetime.combine(now.date(), at_time)
            if candidate <= now:
                candidate += timedelta(days=1)
            return candidate

        if self.type == RecurrenceType.WEEKLY:
            if self.weekday is None:
                raise ValueError("weekly recurrence missing weekday")
            days_ahead = self.weekday - now.weekday()
            if days_ahead < 0:
                days_ahead += 7
            candidate = datetime.combine(
                now.date() + timedelta(days=days_ahead), at_time
            )
            if candidate <= now:
                candidate += timedelta(weeks=1)
            return candidate

        if self.type == RecurrenceType.INTERVAL_DAYS:
            candidate = datetime.combine(now.date(), at_time)
            if candidate <= now:
                candidate += timedelta(days=self.interval)
            return candidate

        if self.type == RecurrenceType.INTERVAL_WEEKS:
            wd = self.weekday if self.weekday is not None else now.weekday()
            days_ahead = wd - now.weekday()
            if days_ahead < 0:
                days_ahead += 7
            candidate = datetime.combine(
                now.date() + timedelta(days=days_ahead), at_time
            )
            if candidate <= now:
                candidate += timedelta(weeks=self.interval)
            return candidate

        if self.type == RecurrenceType.MONTHLY:
            return self._next_monthly(now, at_time)

        if self.type == RecurrenceType.YEARLY:
            return self._next_yearly(now, at_time)

        raise ValueError(f"Unknown recurrence type: {self.type}")

    def _next_monthly(self, now: datetime, at_time: time) -> datetime:
        """Next monthly occurrence."""
        import calendar

        if self.ordinal is not None and self.ordinal_weekday is not None:
            # "every:month:first:monday" style
            return _next_ordinal_weekday_in_month(
                now, at_time, self.ordinal, self.ordinal_weekday
            )

        dom = self.day_of_month or 1

        for month_offset in range(0, 13):
            year = now.year + (now.month + month_offset - 1) // 12
            month = (now.month + month_offset - 1) % 12 + 1
            last_day = calendar.monthrange(year, month)[1]

            actual_day = last_day if dom == -1 else min(dom, last_day)
            candidate = datetime.combine(date(year, month, actual_day), at_time)
            if candidate > now:
                return candidate

        raise ValueError("Could not find next monthly occurrence")

    def _next_yearly(self, now: datetime, at_time: time) -> datetime:
        """Next yearly occurrence."""
        import calendar

        target_month = self.month or 1

        if self.ordinal is not None and self.ordinal_weekday is not None:
            # "yearly:mar:first:monday" style
            for year_offset in range(0, 3):
                year = now.year + year_offset
                candidate = _ordinal_weekday_in_month(
                    year, target_month, self.ordinal, self.ordinal_weekday, at_time
                )
                if candidate and candidate > now:
                    return candidate
            raise ValueError("Could not find next yearly occurrence")

        dom = self.day_of_month or 1
        for year_offset in range(0, 3):
            year = now.year + year_offset
            last_day = calendar.monthrange(year, target_month)[1]
            actual_day = last_day if dom == -1 else min(dom, last_day)
            candidate = datetime.combine(
                date(year, target_month, actual_day), at_time
            )
            if candidate > now:
                return candidate

        raise ValueError("Could not find next yearly occurrence")

    def is_due(self, at: Optional[datetime] = None, tolerance_minutes: int = 30) -> bool:
        """Check if this recurrence is due now (within tolerance)."""
        now = at or datetime.now()
        nxt = self.next_occurrence(now - timedelta(minutes=tolerance_minutes))
        return abs((nxt - now).total_seconds()) <= tolerance_minutes * 60

    def __str__(self) -> str:
        return self.expression


def parse_recurrence(expr: str) -> Recurrence:
    """
    Parse a human-readable recurrence expression.

    Examples:
        parse_recurrence("every:day:9am")
        parse_recurrence("every:monday:9am")
        parse_recurrence("every:weekday:8am")
        parse_recurrence("every:3:days")
        parse_recurrence("every:month:15")
        parse_recurrence("yearly:mar:15")
        parse_recurrence("once:2026-04-15")
    """
    parts = [p.strip().lower() for p in expr.split(":")]

    if not parts:
        raise ValueError(f"Empty recurrence expression: {expr!r}")

    head = parts[0]

    if head == "once":
        return _parse_once(parts, expr)
    elif head == "yearly":
        return _parse_yearly(parts, expr)
    elif head == "every":
        return _parse_every(parts, expr)
    else:
        raise ValueError(
            f"Unknown recurrence prefix {head!r} in {expr!r}. "
            "Expected 'every', 'yearly', or 'once'."
        )


def _parse_time(token: str) -> tuple[int, int]:
    """Parse a time token like '9am', '14', '2pm' into (hour, minute)."""
    m = TIME_PATTERN.match(token)
    if not m:
        raise ValueError(f"Invalid time: {token!r}")
    hour = int(m.group(1))
    meridiem = m.group(2)
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    return hour, 0


def _parse_once(parts: list[str], expr: str) -> Recurrence:
    """Parse 'once:2026-04-15' or 'once:2026-04-15:10am'."""
    if len(parts) < 2:
        raise ValueError(f"once requires a date: {expr!r}")
    d = date.fromisoformat(parts[1])
    hour, minute = (0, 0)
    if len(parts) >= 3:
        hour, minute = _parse_time(parts[2])
    return Recurrence(
        type=RecurrenceType.ONCE,
        expression=expr,
        at_hour=hour,
        at_minute=minute,
        once_date=d,
    )


def _parse_yearly(parts: list[str], expr: str) -> Recurrence:
    """Parse 'yearly:mar:15' or 'yearly:mar:first:monday'."""
    if len(parts) < 3:
        raise ValueError(f"yearly requires month and day: {expr!r}")

    month = MONTHS.get(parts[1])
    if month is None:
        raise ValueError(f"Unknown month: {parts[1]!r} in {expr!r}")

    # Check for ordinal weekday: yearly:mar:first:monday
    if parts[2] in ORDINALS and len(parts) >= 4 and parts[3] in WEEKDAYS:
        return Recurrence(
            type=RecurrenceType.YEARLY,
            expression=expr,
            month=month,
            ordinal=ORDINALS[parts[2]],
            ordinal_weekday=WEEKDAYS[parts[3]],
        )

    # Simple day: yearly:mar:15
    try:
        dom = int(parts[2])
    except ValueError:
        raise ValueError(f"Expected day number or ordinal in {expr!r}")

    hour, minute = (0, 0)
    if len(parts) >= 4:
        hour, minute = _parse_time(parts[3])

    return Recurrence(
        type=RecurrenceType.YEARLY,
        expression=expr,
        month=month,
        day_of_month=dom,
        at_hour=hour,
        at_minute=minute,
    )


def _parse_every(parts: list[str], expr: str) -> Recurrence:
    """Parse 'every:...' patterns."""
    if len(parts) < 2:
        raise ValueError(f"every requires a target: {expr!r}")

    target = parts[1]

    # every:day:9am
    if target == "day":
        hour, minute = (0, 0)
        if len(parts) >= 3:
            hour, minute = _parse_time(parts[2])
        return Recurrence(
            type=RecurrenceType.DAILY,
            expression=expr,
            at_hour=hour,
            at_minute=minute,
        )

    # every:weekday:8am (Mon-Fri, stored as daily with weekday flag)
    if target == "weekday":
        hour, minute = (0, 0)
        if len(parts) >= 3:
            hour, minute = _parse_time(parts[2])
        return Recurrence(
            type=RecurrenceType.DAILY,
            expression=expr,
            at_hour=hour,
            at_minute=minute,
            weekday=5,  # sentinel: means "only weekdays" (0-4)
        )

    # every:monday:9am (weekly on a named day)
    if target in WEEKDAYS:
        hour, minute = (0, 0)
        if len(parts) >= 3:
            hour, minute = _parse_time(parts[2])
        return Recurrence(
            type=RecurrenceType.WEEKLY,
            expression=expr,
            weekday=WEEKDAYS[target],
            at_hour=hour,
            at_minute=minute,
        )

    # every:month:15 or every:month:last or every:month:first:monday
    if target == "month":
        return _parse_every_month(parts, expr)

    # every:N:days or every:N:weeks:monday
    try:
        n = int(target)
    except ValueError:
        raise ValueError(f"Unknown target {target!r} in {expr!r}")

    if len(parts) < 3:
        raise ValueError(f"every:{n} requires a unit (days/weeks): {expr!r}")

    unit = parts[2]
    if unit == "days":
        hour, minute = (0, 0)
        if len(parts) >= 4:
            hour, minute = _parse_time(parts[3])
        return Recurrence(
            type=RecurrenceType.INTERVAL_DAYS,
            expression=expr,
            interval=n,
            at_hour=hour,
            at_minute=minute,
        )
    elif unit == "weeks":
        weekday = None
        hour, minute = (0, 0)
        if len(parts) >= 4 and parts[3] in WEEKDAYS:
            weekday = WEEKDAYS[parts[3]]
            if len(parts) >= 5:
                hour, minute = _parse_time(parts[4])
        return Recurrence(
            type=RecurrenceType.INTERVAL_WEEKS,
            expression=expr,
            interval=n,
            weekday=weekday,
            at_hour=hour,
            at_minute=minute,
        )
    else:
        raise ValueError(f"Unknown unit {unit!r} in {expr!r}")


def _parse_every_month(parts: list[str], expr: str) -> Recurrence:
    """Parse 'every:month:...' patterns."""
    if len(parts) < 3:
        raise ValueError(f"every:month requires a day: {expr!r}")

    target = parts[2]

    # every:month:last
    if target == "last":
        return Recurrence(
            type=RecurrenceType.MONTHLY,
            expression=expr,
            day_of_month=-1,
        )

    # every:month:first:monday
    if target in ORDINALS and len(parts) >= 4 and parts[3] in WEEKDAYS:
        return Recurrence(
            type=RecurrenceType.MONTHLY,
            expression=expr,
            ordinal=ORDINALS[target],
            ordinal_weekday=WEEKDAYS[parts[3]],
        )

    # every:month:15
    try:
        dom = int(target)
    except ValueError:
        raise ValueError(f"Expected day number in {expr!r}")

    hour, minute = (0, 0)
    if len(parts) >= 4:
        hour, minute = _parse_time(parts[3])

    return Recurrence(
        type=RecurrenceType.MONTHLY,
        expression=expr,
        day_of_month=dom,
        at_hour=hour,
        at_minute=minute,
    )


# ============================================================================
# Helpers for ordinal weekday calculations
# ============================================================================

def _ordinal_weekday_in_month(
    year: int,
    month: int,
    ordinal: int,
    weekday: int,
    at_time: time,
) -> Optional[datetime]:
    """Find the Nth weekday in a given month (e.g., 1st Monday of March)."""
    import calendar

    last_day = calendar.monthrange(year, month)[1]

    if ordinal == -1:
        # Last occurrence: scan backward
        for day in range(last_day, 0, -1):
            d = date(year, month, day)
            if d.weekday() == weekday:
                return datetime.combine(d, at_time)
        return None

    # Nth occurrence: scan forward
    count = 0
    for day in range(1, last_day + 1):
        d = date(year, month, day)
        if d.weekday() == weekday:
            count += 1
            if count == ordinal:
                return datetime.combine(d, at_time)
    return None


def _next_ordinal_weekday_in_month(
    now: datetime,
    at_time: time,
    ordinal: int,
    weekday: int,
) -> datetime:
    """Find the next ordinal weekday across months."""
    for month_offset in range(0, 13):
        year = now.year + (now.month + month_offset - 1) // 12
        month = (now.month + month_offset - 1) % 12 + 1
        candidate = _ordinal_weekday_in_month(year, month, ordinal, weekday, at_time)
        if candidate and candidate > now:
            return candidate
    raise ValueError("Could not find next ordinal weekday occurrence")
