"""
Signal Registry

Registry of verification signals for FabricInspector.

Follows HoloLoom's DepartmentRegistry pattern:
- Plugin discovery via @register decorator
- Multi-dimensional lookup
- Lazy instantiation

Usage:
    @SignalRegistry.register
    class MySignal:
        name = "my_signal"
        weight = 0.10

        async def check(self, thread, context):
            return SignalResult(...)

    # Get all signals
    signals = SignalRegistry.get_all()

    # Get specific signal
    signal = SignalRegistry.get("tests")

Created: December 2025
"""

import logging
from typing import Dict, Type, List, Optional

from HoloLoom.tapestry.protocol import FabricSignal

logger = logging.getLogger(__name__)


class SignalRegistry:
    """
    Registry of verification signals.

    Follows HoloLoom's DepartmentRegistry pattern:
    - Plugin discovery via decorator
    - Multi-dimensional lookup
    - Lazy instantiation

    Signals are registered via the @register decorator and
    instantiated on-demand via get_all() or get().
    """

    _signals: Dict[str, Type[FabricSignal]] = {}

    @classmethod
    def register(cls, signal_cls: Type[FabricSignal]) -> Type[FabricSignal]:
        """
        Decorator for registering signals.

        Usage:
            @SignalRegistry.register
            class TestSignal:
                name = "tests"
                weight = 0.20
                async def check(self, thread, context): ...

        Args:
            signal_cls: Signal class to register

        Returns:
            The same class (allows use as decorator)
        """
        name = getattr(signal_cls, 'name', None)
        if name is None:
            raise ValueError(f"Signal class {signal_cls.__name__} must have 'name' attribute")

        weight = getattr(signal_cls, 'weight', None)
        if weight is None:
            raise ValueError(f"Signal class {signal_cls.__name__} must have 'weight' attribute")

        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Signal weight must be between 0.0 and 1.0, got {weight}")

        cls._signals[name] = signal_cls
        logger.debug(f"Registered signal: {name} (weight={weight:.2f})")
        return signal_cls

    @classmethod
    def get_all(cls) -> List[FabricSignal]:
        """
        Get all registered signals as instances.

        Returns:
            List of instantiated signal objects.
        """
        return [sig() for sig in cls._signals.values()]

    @classmethod
    def get(cls, name: str) -> Optional[FabricSignal]:
        """
        Get signal by name.

        Args:
            name: Signal name (e.g., "tests", "quality")

        Returns:
            Instantiated signal or None if not found.
        """
        sig_cls = cls._signals.get(name)
        return sig_cls() if sig_cls else None

    @classmethod
    def get_names(cls) -> List[str]:
        """Get all registered signal names."""
        return list(cls._signals.keys())

    @classmethod
    def total_weight(cls) -> float:
        """Get sum of all signal weights (should equal ~1.0)."""
        return sum(sig.weight for sig in cls.get_all())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered signals. Useful for testing."""
        cls._signals.clear()
        logger.debug("Cleared all registered signals")

    @classmethod
    def count(cls) -> int:
        """Get number of registered signals."""
        return len(cls._signals)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a signal is registered."""
        return name in cls._signals

    @classmethod
    def get_weights(cls) -> Dict[str, float]:
        """Get mapping of signal names to weights."""
        return {name: sig.weight for name, sig in cls._signals.items()
                for sig in [sig()]}

    @classmethod
    def describe(cls) -> str:
        """Get human-readable description of registered signals."""
        if not cls._signals:
            return "No signals registered"

        lines = ["Registered Signals:"]
        total = 0.0
        for name in sorted(cls._signals.keys()):
            sig = cls._signals[name]()
            lines.append(f"  - {name}: weight={sig.weight:.2f}")
            total += sig.weight
        lines.append(f"  Total weight: {total:.2f}")
        return "\n".join(lines)
