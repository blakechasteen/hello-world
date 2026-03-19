"""
Server State Management
=======================

Global server state for the HoloLoom API.

The ServerState class holds references to shared resources like the
orchestrator, memory backend, alignment framework, and monitoring.
"""

from typing import Any

from hololoom.protocols.types import MemoryShard

from hololoom.agentic.ml_logic_detector import MLLogicDetector
from hololoom.alignment.audit_trail import AuditTrail
from hololoom.alignment.deception_detection import DeceptionDetector
from hololoom.alignment.safety_guardrails import SafetyGuardrails
from hololoom.config import Config

from .middleware import RateLimiter, ServerStats


class ServerState:
    """
    Global server state.

    Holds references to all shared resources:
    - Orchestrator: Agentic reasoning engine
    - Memory: Persistent memory backend
    - Alignment: Safety guardrails, audit trail, deception detection
    - Monitoring: Agent monitoring, rate limiting, stats

    Attributes:
        orchestrator: Agentic orchestrator instance
        audit_trail: Decision audit logging
        safety_guardrails: Risk-based action gating
        deception_detector: Goal transparency tracking
        config: Server configuration
        shards: In-memory knowledge shards
        memory_backend: Persistent memory backend
        ml_logic_detector: ML-based logic error detector
        rate_limiter: Request rate limiting
        stats: Server statistics
        monitor: Agent monitoring (Nov 2025)
    """

    orchestrator: Any | None = None
    audit_trail: AuditTrail | None = None
    safety_guardrails: SafetyGuardrails | None = None
    deception_detector: DeceptionDetector | None = None
    config: Config | None = None
    shards: list[MemoryShard] = []
    memory_backend: Any | None = None
    ml_logic_detector: MLLogicDetector | None = None
    rate_limiter: RateLimiter | None = None
    stats: ServerStats | None = None
    monitor: Any | None = None


# Global state instance
state = ServerState()
