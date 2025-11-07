"""
HoloLoom COS - Company Operating System

Elegant, voice-first business management for farmers and makers.

Philosophy:
- Everything is an event (tasks, sales, purchases, notes)
- Single source of truth (event stream database)
- HITL verification for critical data (expenditures, sales, time)
- Views are derived, not stored
- Voice-first, mobile-optimized

Core Components:
- Event stream database (SQLite)
- NLP parser (business/productivity intents)
- HITL verification system
- Query/view layer
- CLI interface

Usage:
    from cos import COSCLI
    cli = COSCLI()
    await cli.log("Baked bread for 3 hours, made 12 loaves")
    await cli.summary_today()
"""

from .core.types import (
    Event, EventType, EventSource, Category, ProductLine,
    ParsedIntent, VerificationRequest,
    DailySummary, WeeklySummary, ProductPerformance
)

from .core.parser import BusinessIntentParser, parse_input
from .core.storage import EventStore, get_store
from .cli import COSCLI

__version__ = "0.1.0"
__all__ = [
    # Types
    'Event', 'EventType', 'EventSource', 'Category', 'ProductLine',
    'ParsedIntent', 'VerificationRequest',
    'DailySummary', 'WeeklySummary', 'ProductPerformance',

    # Core
    'BusinessIntentParser', 'parse_input',
    'EventStore', 'get_store',

    # CLI
    'COSCLI',
]
