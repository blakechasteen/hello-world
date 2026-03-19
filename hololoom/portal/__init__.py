"""
Portal: Distributed Compute Control Plane for HoloLoom

A 4-component distributed system:
1. Portal Server - Control plane, node registry
2. Node Daemon - Per-device WASM job executor
3. Shuttle Bot - Matrix ChatOps interface
4. WASM Runner - Embedded in Node for job execution

Target: Single user's local network, 1-3 nodes, Matrix room for control.
"""

__version__ = "0.1.0"
