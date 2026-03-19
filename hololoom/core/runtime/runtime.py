"""
HoloLoomRuntime — minimal lifecycle manager for the 6-layer architecture.
=========================================================================

Holds layers + bus. Starts layers in registration order, stops in reverse.
A failed layer enters DEGRADED — other layers continue.

Usage:
    runtime = HoloLoomRuntime()
    runtime.register_layer(RitualLayer())
    async with runtime:
        # runtime is started, layers are running
        print(runtime.status())
    # runtime is stopped, layers are cleaned up
"""

import logging
from collections import OrderedDict
from typing import Any

from .layer import Layer, LayerState

logger = logging.getLogger(__name__)


class HoloLoomRuntime:
    """
    Minimal runtime that manages layers via the Layer protocol.

    Attributes accessible to layers via the runtime reference:
      - bus: UnifiedBus | None
      - journal: RitualJournal | None
      - config: dict
    """

    def __init__(
        self,
        bus: Any = None,
        journal: Any = None,
        config: dict[str, Any] | None = None,
    ):
        self.bus = bus
        self.journal = journal
        self.config = config or {}
        self._layers: OrderedDict[str, Layer] = OrderedDict()
        self._layer_states: dict[str, LayerState] = {}
        self._started = False

    def register_layer(self, layer: Layer) -> None:
        """Register a layer. Must be called before start()."""
        self._layers[layer.name] = layer
        self._layer_states[layer.name] = LayerState.INIT
        logger.info("Layer registered: %s", layer.name)

    async def start(self) -> None:
        """Start all registered layers in registration order."""
        if self._started:
            return
        self._started = True

        for name, layer in self._layers.items():
            try:
                await layer.start(self)
                self._layer_states[name] = LayerState.RUNNING
                logger.info("Layer started: %s", name)
            except Exception as e:
                self._layer_states[name] = LayerState.DEGRADED
                logger.warning("Layer degraded on start: %s — %s", name, e)

    async def stop(self) -> None:
        """Stop all layers in reverse registration order."""
        if not self._started:
            return

        for name in reversed(list(self._layers.keys())):
            layer = self._layers[name]
            try:
                await layer.stop()
                self._layer_states[name] = LayerState.STOPPED
                logger.info("Layer stopped: %s", name)
            except Exception as e:
                logger.warning("Layer failed to stop cleanly: %s — %s", name, e)
                self._layer_states[name] = LayerState.STOPPED

        self._started = False

    def status(self) -> dict[str, Any]:
        """Return status of all layers."""
        layer_status = {}
        for name, layer in self._layers.items():
            try:
                layer_status[name] = {
                    "state": self._layer_states.get(name, LayerState.INIT).value,
                    **layer.status(),
                }
            except Exception as e:
                layer_status[name] = {
                    "state": self._layer_states.get(name, LayerState.INIT).value,
                    "error": str(e),
                }
        return {
            "started": self._started,
            "layers": layer_status,
        }

    def get_layer(self, name: str) -> Layer | None:
        """Get a registered layer by name."""
        return self._layers.get(name)

    async def __aenter__(self) -> "HoloLoomRuntime":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()
