"""
SOAR (Security Orchestration, Automation, and Response) System

Automated incident response with playbook-based execution.

**Implemented: 2025-11-16**
**Phase**: 4 - Security Pipeline

Features:
- Automated playbook execution based on security events
- 20+ pre-built actions across 6 categories
- 5 production-ready playbooks
- Dry-run mode for safe testing
- Complete audit logging
- Manual approval workflow
- Playbook versioning

Usage:
    >>> from HoloLoom.security.soar import create_soar_engine, SecurityEvent
    >>> from HoloLoom.security.soar.playbooks import register_all_playbooks
    >>>
    >>> # Create SOAR engine
    >>> soar = create_soar_engine()
    >>>
    >>> # Register playbooks
    >>> register_all_playbooks(soar)
    >>>
    >>> # Process security event
    >>> event = SecurityEvent(
    ...     event_type="security.attack.sql_injection",
    ...     source_ip="192.168.1.100",
    ...     payload="' OR '1'='1",
    ... )
    >>> results = await soar.process_event(event)
"""

from .core import (
    SOAREngine,
    SecurityEvent,
    PlaybookResult,
    PlaybookMetadata,
    PlaybookExecution,
    PlaybookSeverity,
    PlaybookStatus,
    ExecutionMode,
    create_soar_engine,
)

from .actions import (
    ActionExecutor,
    ActionResult,
)

from .playbooks import (
    register_all_playbooks,
    get_playbook_catalog,
)


__all__ = [
    # Core
    "SOAREngine",
    "SecurityEvent",
    "PlaybookResult",
    "PlaybookMetadata",
    "PlaybookExecution",
    "PlaybookSeverity",
    "PlaybookStatus",
    "ExecutionMode",
    "create_soar_engine",

    # Actions
    "ActionExecutor",
    "ActionResult",

    # Playbooks
    "register_all_playbooks",
    "get_playbook_catalog",
]
