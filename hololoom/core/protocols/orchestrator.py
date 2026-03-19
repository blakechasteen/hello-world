"""
Weaving Orchestrator Protocol
==============================
Protocol definition for the weaving orchestrator.

Allows server/app code to type-hint against the protocol instead of
importing the concrete WeavingOrchestrator (2,700+ lines, heavy deps).

Usage:
    from hololoom.protocols.orchestrator import WeavingOrchestratorProtocol

    class MyServer:
        def __init__(self, orchestrator: WeavingOrchestratorProtocol):
            self.orchestrator = orchestrator

Author: mythRL Team
Date: 2026-03-12
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from hololoom.protocols.types import ComplexityLevel, Query


@runtime_checkable
class WeavingOrchestratorProtocol(Protocol):
    """
    Protocol for the weaving orchestrator.

    Any object implementing ``weave()`` with the right signature can be
    used as an orchestrator — the concrete ``WeavingOrchestrator``, a
    lightweight stub for testing, or a remote proxy.
    """

    async def weave(
        self,
        query: Query,
        pattern_override: Any | None = None,
        complexity: ComplexityLevel | None = None,
        auto_enhance: bool | None = None,
    ) -> Any:
        """
        Execute the weaving cycle.

        Args:
            query: Query to process.
            pattern_override: Optional PatternCard override.
            complexity: Optional complexity level override.
            auto_enhance: Whether to auto-enhance low-confidence results.

        Returns:
            Spacetime fabric with woven results.
        """
        ...
