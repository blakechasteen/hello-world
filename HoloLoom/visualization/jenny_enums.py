"""
Jenny Enums Module
==================
Consolidated enumeration types for Jenny Generative UI runtime.

Date: December 2025
Status: Phase 2 Elegance Enhancement (Task 2.5)

Philosophy:
> "Enums belong together when they share semantic context."

This module consolidates lifecycle, binding, and status enums that are
used across multiple Jenny modules, providing a single source of truth.

Panel-specific enums (PanelTypeJenny, PanelSizeJenny, LayoutHint)
remain in jenny_spec.py as they are tightly coupled to panel definitions.

Usage:
    from HoloLoom.visualization.jenny_enums import (
        LifecycleStage,
        BindingMode,
        DissolutionTrigger,
        ActionStatus,
    )
"""

from enum import Enum


# ============================================================================
# Lifecycle Enums
# ============================================================================

class LifecycleStage(str, Enum):
    """
    Panel lifecycle states (immutable state machine).

    State transitions:
        compile() → NASCENT → (user pins) → STABLE
                           ↘           ↙
                        (timeout/superseded)
                               ↓
                          DISSOLVING
                               ↓
                           ARCHIVED

    SYSTEM is a special stage for meta-panels (breaks infinite provenance loop).
    """
    NASCENT = "nascent"        # Just compiled, animating in (300ms spawn)
    STABLE = "stable"          # User-pinned, persistent until dismissed
    DISSOLVING = "dissolving"  # Fading out (300ms animation)
    ARCHIVED = "archived"      # In SpecLedger, replayable
    SYSTEM = "system"          # Meta-panel, not logged (breaks strange loop)


class BindingMode(str, Enum):
    """
    Data binding modes for panel content.

    Determines how panel updates when underlying data changes.
    """
    STATIC = "static"          # One-time render, no updates (cheapest)
    REACTIVE = "reactive"      # Re-render on data changes (uses React useEffect)
    STREAMING = "streaming"    # SSE/WebSocket live updates (most expensive)


class DissolutionTrigger(str, Enum):
    """
    What causes panel dissolution.

    Tracked in SpecLedger for understanding user behavior patterns.
    """
    MANUAL = "manual"          # User clicked dismiss button
    TIMEOUT = "timeout"        # Idle timeout (default: 5 minutes)
    CONTEXT_SHIFT = "context"  # Query topic changed significantly
    SUPERSEDED = "superseded"  # New panel replaced this one
    ORPHAN = "orphan"          # Cleanup: parent Spacetime deleted
    MEMORY = "memory"          # Memory pressure forced dissolution


# ============================================================================
# Action Enums
# ============================================================================

class ActionStatus(str, Enum):
    """
    Outcome status of an action execution.

    Used by JennyActionHandler to communicate action results.
    """
    SUCCESS = "success"           # Action completed successfully
    FAILED = "failed"             # Action failed (recoverable)
    BLOCKED = "blocked"           # Action blocked by guardrails
    PENDING = "pending"           # Action requires confirmation
    CANCELLED = "cancelled"       # Action was cancelled by user


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Lifecycle
    "LifecycleStage",
    "BindingMode",
    "DissolutionTrigger",
    # Action
    "ActionStatus",
]
