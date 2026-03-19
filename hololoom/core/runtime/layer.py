"""
Layer protocol — the contract for all runtime layers.

Each of the 6 layers (Memory, Weaving, Learning, Deliberation, Intent, Mirror)
implements this protocol. The runtime manages their lifecycle.
"""

from enum import Enum
from typing import Any, Protocol, runtime_checkable


class LayerState(str, Enum):
    INIT = "init"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"


@runtime_checkable
class Layer(Protocol):
    """
    Protocol for a runtime layer.

    Layers are independently startable and stoppable.
    A layer that fails to start enters DEGRADED state
    rather than preventing other layers from starting.
    """
    name: str

    async def start(self, runtime: Any) -> None:
        """Initialize and start the layer. Set state to RUNNING or DEGRADED."""
        ...

    async def stop(self) -> None:
        """Clean up and stop the layer. Set state to STOPPED."""
        ...

    def status(self) -> dict[str, Any]:
        """Return diagnostic information about the layer."""
        ...

    def state_snapshot(self) -> dict[str, Any]:
        """Return minimal state for persistence/debugging."""
        ...
