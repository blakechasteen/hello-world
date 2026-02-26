"""
Proto HoloLoom Integration Bridges
===================================

Integration bridges connecting Proto to HoloLoom's core systems:

1. **AgenticBridge** - Wrapper around HoloLoom's AgenticOrchestrator
   - Simplifies reasoning interface for Proto's needs
   - Maps Proto modes to HoloLoom reasoning modes
   - Provides fallback when HoloLoom unavailable

2. **ProtoDepartment** - Implements HoloLoom Department protocol
   - Makes Proto a first-class HoloLoom department
   - Enables department orchestration
   - Provides metrics and verification

Both components designed for graceful degradation and seamless integration.

Status: Production ready (2025-12-02)
"""

from HoloLoom.apps.departments.proto.integration.agentic_bridge import (
    AgenticBridge,
    ProtoReasoningMode,
    AgenticBridgeResult,
)

from HoloLoom.apps.departments.proto.integration.department_bridge import (
    ProtoDepartment,
)

__all__ = [
    "AgenticBridge",
    "ProtoReasoningMode",
    "AgenticBridgeResult",
    "ProtoDepartment",
]
