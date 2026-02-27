"""
HoloLoom Safety System

Code-level safety locks for Layer 6 (self-modification) capabilities.
Complements documentation-level safety with runtime enforcement.

Usage:
    from hololoom.safety import SafetyLock, Layer6Capability

    # Check if operation is allowed
    if SafetyLock.is_layer6_enabled():
        # Proceed with Layer 6 operation
        pass
    else:
        # Blocked - Layer 6 not unlocked
        raise SafetyLock.blocked_exception(Layer6Capability.CODE_MODIFICATION)
"""

from hololoom.safety.locks import (
    SafetyLock,
    Layer6Capability,
    Layer6BlockedException,
    SafetyMonitor,
)

__all__ = [
    "SafetyLock",
    "Layer6Capability",
    "Layer6BlockedException",
    "SafetyMonitor",
]
