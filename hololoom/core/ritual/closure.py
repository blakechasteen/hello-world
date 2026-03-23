"""
BindingObligation — mandatory closure for ritual sequences.
=============================================================

A weft thread that must be tied at the selvage. When dawn_prime fires,
a binding opens for dusk_consolidate. The loom refuses to start new
patterns (BLOCKING rituals) while a binding is unresolved.

PARALLEL and DEFERRED still fire — prevents deadlock while enforcing
structural commitment.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OpenBinding:
    """A weft thread that must be tied at the selvage."""
    opener_ritual_id: str
    closer_ritual_id: str
    opened_at: str
    session_id: str
    deadline_seconds: float = 0.0    # 0 = no deadline
    priority: str = "mandatory"      # "mandatory" | "advisory"


class ClosureGate:
    """
    Tracks open bindings per session. Gates BLOCKING dispatches
    until mandatory closures are satisfied.

    Usage:
        gate = ClosureGate()
        gate.open(OpenBinding(opener="dawn_prime", closer="dusk_consolidate", ...))
        assert gate.is_gated("some_other_ritual")  # True
        assert not gate.is_gated("dusk_consolidate")  # False — closers pass through
        gate.close("dusk_consolidate")
        assert not gate.has_mandatory_open()
    """

    def __init__(self) -> None:
        self._bindings: dict[str, OpenBinding] = {}  # keyed by opener_ritual_id

    def open(self, binding: OpenBinding) -> None:
        """Register a pending closure requirement."""
        self._bindings[binding.opener_ritual_id] = binding
        logger.debug(
            "Binding opened: %s → %s (%s)",
            binding.opener_ritual_id,
            binding.closer_ritual_id,
            binding.priority,
        )

    def close(self, closer_ritual_id: str) -> bool:
        """
        Close all bindings that expect this ritual as their closer.
        Returns True if any binding was closed.
        """
        to_remove = [
            key for key, b in self._bindings.items()
            if b.closer_ritual_id == closer_ritual_id
        ]
        for key in to_remove:
            del self._bindings[key]
            logger.debug("Binding closed: %s satisfied by %s", key, closer_ritual_id)
        return len(to_remove) > 0

    def is_gated(self, ritual_id: str) -> bool:
        """
        True if this ritual should be blocked.
        Closer rituals are always allowed through.
        """
        if not self._bindings:
            return False
        # Allow closer rituals through
        closer_ids = {b.closer_ritual_id for b in self._bindings.values()}
        if ritual_id in closer_ids:
            return False
        # Block everything else if there's a mandatory binding
        return self.has_mandatory_open()

    def has_mandatory_open(self) -> bool:
        """True if any mandatory binding is open."""
        return any(b.priority == "mandatory" for b in self._bindings.values())

    def pending(self) -> list[OpenBinding]:
        """List all open bindings."""
        return list(self._bindings.values())

    def force_close_all(self) -> list[str]:
        """
        Emergency stop — close all bindings. Returns list of orphaned opener IDs.
        Used by SnagWatcher on RUNAWAY.
        """
        orphaned = list(self._bindings.keys())
        self._bindings.clear()
        if orphaned:
            logger.warning("Force-closed %d bindings: %s", len(orphaned), orphaned)
        return orphaned

    def overdue(self) -> list[OpenBinding]:
        """Return bindings that have exceeded their deadline."""
        now = datetime.now(timezone.utc)
        result = []
        for b in self._bindings.values():
            if b.deadline_seconds > 0:
                opened = datetime.fromisoformat(b.opened_at)
                elapsed = (now - opened).total_seconds()
                if elapsed > b.deadline_seconds:
                    result.append(b)
        return result
