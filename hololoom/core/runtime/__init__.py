"""
HoloLoom Runtime — lifecycle management for the 6-layer architecture.
=====================================================================

Each layer implements the Layer protocol with states:
  INIT → RUNNING → DEGRADED → STOPPED

The runtime starts layers in registration order, stops in reverse.
A failed layer goes DEGRADED — it never prevents other layers from starting.

Usage:
    async with HoloLoomRuntime() as rt:
        rt.register_layer(RitualLayer())
        await rt.start()
"""

from .layer import Layer, LayerState
from .runtime import HoloLoomRuntime

__all__ = [
    "Layer",
    "LayerState",
    "HoloLoomRuntime",
]
