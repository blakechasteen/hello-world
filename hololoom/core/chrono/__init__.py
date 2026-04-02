"""
Chrono - Temporal Control System
==================================
Manages temporal aspects of the weaving process.

Exports:
- ChronoTrigger: Main temporal controller
- TemporalWindow: Time boundaries for thread activation
- ExecutionLimits: Timeout and halt condition configuration
- AgentCalendar: Scheduled tasks with completion tracking
- CalendarTask: A recurring or one-time task
- CompletionRecord: Historical record of task completion
- Recurrence: Parsed recurrence pattern
- parse_recurrence: Parse human-readable recurrence expressions
"""

from .trigger import ChronoTrigger, ExecutionLimits, TemporalWindow
from .calendar import AgentCalendar, CalendarTask, CompletionRecord, Outcome, Priority
from .recurrence import Recurrence, parse_recurrence

__all__ = [
    # Temporal control
    "ChronoTrigger",
    "TemporalWindow",
    "ExecutionLimits",
    # Agent calendar
    "AgentCalendar",
    "CalendarTask",
    "CompletionRecord",
    "Outcome",
    "Priority",
    # Recurrence
    "Recurrence",
    "parse_recurrence",
]
