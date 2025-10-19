"""
Chrono - Temporal Control System
==================================
Manages temporal aspects of the weaving process.

Exports:
- ChronoTrigger: Main temporal controller
- TemporalWindow: Time boundaries for thread activation
- ExecutionLimits: Timeout and halt condition configuration
"""

from .trigger import ChronoTrigger, TemporalWindow, ExecutionLimits

__all__ = ["ChronoTrigger", "TemporalWindow", "ExecutionLimits"]
