"""
Agentic API Server State
=========================
Shared server state singleton used by agentic_api.py and all endpoint routers.

Extracted from agentic_api.py (March 2026 Refactor).
"""

from typing import Optional, List, Any

from hololoom.apps.server.api.middleware.rate_limiter import RateLimiter, ServerStats
from hololoom.config import Config
from hololoom.protocols.types import MemoryShard
from hololoom.alignment.audit_trail import AuditTrail
from hololoom.alignment.safety_guardrails import SafetyGuardrails
from hololoom.alignment.deception_detection import DeceptionDetector
from hololoom.agentic.ml_logic_detector import MLLogicDetector


class ServerState:
    """Global server state."""
    orchestrator: Optional[Any] = None
    audit_trail: Optional[AuditTrail] = None
    safety_guardrails: Optional[SafetyGuardrails] = None
    deception_detector: Optional[DeceptionDetector] = None
    config: Optional[Config] = None
    shards: List[MemoryShard] = []
    memory_backend: Optional[Any] = None
    ml_logic_detector: Optional[MLLogicDetector] = None
    rate_limiter: Optional[RateLimiter] = None
    stats: Optional[ServerStats] = None
    monitor: Optional[Any] = None
    saas_backend: Optional[Any] = None
    continuous_validator: Optional[Any] = None
    _validator_task: Optional[Any] = None
    governance_engine: Optional[Any] = None
    governance_pipeline: Optional[Any] = None


# Module-level singleton
state = ServerState()
