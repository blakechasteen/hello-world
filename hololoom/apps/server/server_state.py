"""
Agentic API Server State
=========================
Shared server state singleton used by agentic_api.py and all endpoint routers.

Extracted from agentic_api.py (March 2026 Refactor).
"""

from typing import Any

from hololoom.agentic.ml_logic_detector import MLLogicDetector
from hololoom.alignment.audit_trail import AuditTrail
from hololoom.alignment.deception_detection import DeceptionDetector
from hololoom.alignment.safety_guardrails import SafetyGuardrails
from hololoom.apps.server.api.middleware.rate_limiter import RateLimiter, ServerStats
from hololoom.config import Config
from hololoom.protocols.types import MemoryShard


class ServerState:
    """Global server state."""
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
    saas_backend: Any | None = None
    continuous_validator: Any | None = None
    _validator_task: Any | None = None
    governance_engine: Any | None = None
    governance_pipeline: Any | None = None


# Module-level singleton
state = ServerState()
