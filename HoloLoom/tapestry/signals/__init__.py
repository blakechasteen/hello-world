"""
Tapestry Verification Signals

Pluggable verification signals for holistic quality assessment.

Built-in signals (6 dimensions):
- TestSignal (0.20): Run pytest, check pass/fail
- TroughSignal (0.20): AI slop detection
- AlignmentSignal (0.15): Safety guardrails
- ArchitectureSignal (0.15): Pattern adherence
- DocumentationSignal (0.15): Docstring coverage
- IntegrationSignal (0.15): Cross-system coherence

Usage:
    from HoloLoom.tapestry.signals import SignalRegistry

    # Get all registered signals
    signals = SignalRegistry.get_all()

    # Register custom signal
    @SignalRegistry.register
    class MySignal:
        name = "my_signal"
        weight = 0.10
        async def check(self, thread, context): ...

Created: December 2025
"""

from HoloLoom.tapestry.signals.registry import SignalRegistry

# Import builtins to trigger registration
from HoloLoom.tapestry.signals import builtins

__all__ = [
    "SignalRegistry",
]
