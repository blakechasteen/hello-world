"""
Chronos - Simple time tracking for human activities

"The act of keeping time should never cost more time than it saves."
"""

from .core import EventLog, ChronosEvent, ChronosState

__version__ = "0.1.0"
__all__ = ["EventLog", "ChronosEvent", "ChronosState"]
