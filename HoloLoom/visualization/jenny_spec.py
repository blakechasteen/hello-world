"""
Jenny Specification Module
===========================
Core data structures for Jenny Generative UI runtime.

Philosophy:
> "Disposable pixels, durable decisions."
>
> Every UI panel is temporary and should dissolve when no longer needed.
> Every decision that generated it is permanent and fully replayable.

This module provides frozen (immutable) dataclasses for UI specifications,
ensuring full provenance tracking and replay capability.

References:
- VISUAL_QUICK_START.md (HoloLoom visualization patterns)
- CLAUDE.md (Jenny architecture section)
- dashboard.py (PanelSpec, PanelType base patterns)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from uuid import uuid4
from datetime import datetime
import json


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


class PanelTypeJenny(str, Enum):
    """
    Jenny panel types (extends base PanelType).

    Includes all existing HoloLoom visualization types plus Jenny-specific ones.
    """
    # Existing HoloLoom types (from dashboard.py)
    GRAPH = "graph"                # Knowledge graph (KnowledgeGraph2D)
    CONFIDENCE = "confidence"      # Confidence gauge (ConfidenceGauge)
    TIMELINE = "timeline"          # Stage timeline (StageTimeline)
    MEMORY_MAP = "memory_map"      # Memory activation map
    TEXT = "text"                  # Plain text response
    CODE = "code"                  # Code with syntax highlighting
    TABLE = "table"                # Tabular data
    METRIC = "metric"              # Single metric display

    # Jenny-specific types
    REASONING = "reasoning"        # Step-by-step reasoning chain
    SOURCES = "sources"            # Source attribution list
    ACTIONS = "actions"            # Action buttons/forms
    WHY = "why"                    # "Why this UI?" meta-panel
    COMPARISON = "comparison"      # Side-by-side comparison


class PanelSizeJenny(str, Enum):
    """
    Panel size specifications for layout.

    AR mode: Affects physical size in meters
    Web mode: Affects grid column span
    """
    SMALL = "small"        # 1/4 width (4 per row in web, 0.3m in AR)
    MEDIUM = "medium"      # 1/2 width (2 per row in web, 0.5m in AR)
    LARGE = "large"        # 3/4 width (spans 3 of 4 columns)
    XLARGE = "xlarge"      # Full width (spans all columns, 1.0m in AR)


class LayoutHint(str, Enum):
    """
    Layout hints for panel arrangement.

    Passed to JennyCompiler.compile() to suggest layout strategy.
    """
    GRID = "grid"          # Standard grid layout (web default)
    STACK = "stack"        # Vertical stack (mobile, narrow screens)
    RADIAL = "radial"      # Radial around user (AR default)
    FLOW = "flow"          # Auto-flowing responsive layout


# ============================================================================
# Core Spec Dataclass
# ============================================================================

@dataclass(frozen=True)
class JennySpec:
    """
    Frozen UI specification - the "contract" between compiler and renderer.

    Extends PanelSpec from visualization/dashboard.py with:
    - Lifecycle tracking (NASCENT → STABLE → DISSOLVING → ARCHIVED)
    - Binding modes (STATIC, REACTIVE, STREAMING)
    - Full provenance (spec_id, spacetime_id, timestamp)

    Immutability ensures:
    - Provenance integrity (spec cannot be modified after creation)
    - Thread safety (can be passed between threads without locks)
    - Hashability (can be used as dict keys, set members)

    Usage:
        spec = JennySpec(
            panel_type=PanelTypeJenny.GRAPH,
            spacetime_id="spacetime_123",
            content={"nodes": [...], "edges": [...]},
            actions=[{"label": "Pin", "handler": "pin_panel"}]
        )
    """

    # ========================================================================
    # Identity
    # ========================================================================
    spec_id: str = field(default_factory=lambda: str(uuid4()))
    spacetime_id: str = ""  # Links to Spacetime that generated this

    # ========================================================================
    # Panel Type and Content
    # ========================================================================
    panel_type: PanelTypeJenny = PanelTypeJenny.TEXT
    title: Optional[str] = None
    subtitle: Optional[str] = None
    content: Dict[str, Any] = field(default_factory=dict)

    # ========================================================================
    # Lifecycle
    # ========================================================================
    lifecycle: LifecycleStage = LifecycleStage.NASCENT
    dissolution_trigger: Optional[DissolutionTrigger] = None

    # ========================================================================
    # Layout
    # ========================================================================
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z (AR-ready)
    size: PanelSizeJenny = PanelSizeJenny.MEDIUM
    priority: int = 0  # Higher = closer to user (AR), higher z-index (web)

    # ========================================================================
    # Data Binding
    # ========================================================================
    binding_mode: BindingMode = BindingMode.STATIC
    data_source: Optional[str] = None  # Query for reactive/streaming bindings
    refresh_interval_ms: Optional[int] = None  # For reactive mode

    # ========================================================================
    # Actions
    # ========================================================================
    actions: Tuple[Dict[str, Any], ...] = ()  # Immutable tuple of action definitions

    # ========================================================================
    # Provenance
    # ========================================================================
    timestamp: datetime = field(default_factory=datetime.now)
    compiler_version: str = "1.0.0"
    model_checkpoint: Optional[str] = None  # For replay determinism

    # ========================================================================
    # Metadata (extensible)
    # ========================================================================
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate spec after initialization."""
        # Note: Can't modify fields in frozen dataclass, but can validate
        if self.binding_mode == BindingMode.REACTIVE and not self.data_source:
            raise ValueError("REACTIVE binding mode requires data_source")
        if self.binding_mode == BindingMode.STREAMING and not self.data_source:
            raise ValueError("STREAMING binding mode requires data_source")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to serializable dictionary.

        Preserves enum values as strings for JSON compatibility.
        """
        return {
            "spec_id": self.spec_id,
            "spacetime_id": self.spacetime_id,
            "panel_type": self.panel_type.value,
            "title": self.title,
            "subtitle": self.subtitle,
            "content": self.content,
            "lifecycle": self.lifecycle.value,
            "dissolution_trigger": self.dissolution_trigger.value if self.dissolution_trigger else None,
            "position": {"x": self.position[0], "y": self.position[1], "z": self.position[2]},
            "size": self.size.value,
            "priority": self.priority,
            "binding_mode": self.binding_mode.value,
            "data_source": self.data_source,
            "refresh_interval_ms": self.refresh_interval_ms,
            "actions": list(self.actions),
            "timestamp": self.timestamp.isoformat(),
            "compiler_version": self.compiler_version,
            "model_checkpoint": self.model_checkpoint,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JennySpec":
        """Create JennySpec from dictionary."""
        position = data.get("position", {"x": 0, "y": 0, "z": 0})
        if isinstance(position, dict):
            position = (position["x"], position["y"], position["z"])

        return cls(
            spec_id=data["spec_id"],
            spacetime_id=data.get("spacetime_id", ""),
            panel_type=PanelTypeJenny(data["panel_type"]),
            title=data.get("title"),
            subtitle=data.get("subtitle"),
            content=data.get("content", {}),
            lifecycle=LifecycleStage(data.get("lifecycle", "nascent")),
            dissolution_trigger=DissolutionTrigger(data["dissolution_trigger"]) if data.get("dissolution_trigger") else None,
            position=position,
            size=PanelSizeJenny(data.get("size", "medium")),
            priority=data.get("priority", 0),
            binding_mode=BindingMode(data.get("binding_mode", "static")),
            data_source=data.get("data_source"),
            refresh_interval_ms=data.get("refresh_interval_ms"),
            actions=tuple(data.get("actions", [])),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            compiler_version=data.get("compiler_version", "1.0.0"),
            model_checkpoint=data.get("model_checkpoint"),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "JennySpec":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def with_lifecycle(self, new_lifecycle: LifecycleStage, trigger: Optional[DissolutionTrigger] = None) -> "JennySpec":
        """
        Create new spec with updated lifecycle (immutable update pattern).

        Usage:
            stable_spec = nascent_spec.with_lifecycle(LifecycleStage.STABLE)
            dissolving_spec = stable_spec.with_lifecycle(
                LifecycleStage.DISSOLVING,
                trigger=DissolutionTrigger.MANUAL
            )
        """
        return JennySpec(
            spec_id=self.spec_id,
            spacetime_id=self.spacetime_id,
            panel_type=self.panel_type,
            title=self.title,
            subtitle=self.subtitle,
            content=self.content,
            lifecycle=new_lifecycle,
            dissolution_trigger=trigger if new_lifecycle == LifecycleStage.DISSOLVING else self.dissolution_trigger,
            position=self.position,
            size=self.size,
            priority=self.priority,
            binding_mode=self.binding_mode,
            data_source=self.data_source,
            refresh_interval_ms=self.refresh_interval_ms,
            actions=self.actions,
            timestamp=self.timestamp,
            compiler_version=self.compiler_version,
            model_checkpoint=self.model_checkpoint,
            metadata=self.metadata,
        )

    def with_position(self, x: float, y: float, z: float = 0.0) -> "JennySpec":
        """Create new spec with updated position (immutable update pattern)."""
        return JennySpec(
            spec_id=self.spec_id,
            spacetime_id=self.spacetime_id,
            panel_type=self.panel_type,
            title=self.title,
            subtitle=self.subtitle,
            content=self.content,
            lifecycle=self.lifecycle,
            dissolution_trigger=self.dissolution_trigger,
            position=(x, y, z),
            size=self.size,
            priority=self.priority,
            binding_mode=self.binding_mode,
            data_source=self.data_source,
            refresh_interval_ms=self.refresh_interval_ms,
            actions=self.actions,
            timestamp=self.timestamp,
            compiler_version=self.compiler_version,
            model_checkpoint=self.model_checkpoint,
            metadata=self.metadata,
        )


# ============================================================================
# Action Helpers
# ============================================================================

def create_action(
    label: str,
    handler: str,
    action_type: str = "button",
    params: Optional[Dict[str, Any]] = None,
    requires_confirmation: bool = False,
) -> Dict[str, Any]:
    """
    Helper to create action definitions for JennySpec.actions.

    Args:
        label: Display text for the action
        handler: Server-side handler name (e.g., "pin_panel", "dismiss")
        action_type: "button" | "form" | "slider" | "toggle"
        params: Additional parameters for the handler
        requires_confirmation: Whether to show confirmation dialog (SafetyGuardrails)

    Returns:
        Action definition dict

    Usage:
        actions = (
            create_action("Pin", "pin_panel"),
            create_action("Dismiss", "dismiss_panel", requires_confirmation=True),
        )
        spec = JennySpec(actions=actions)
    """
    return {
        "action_id": str(uuid4()),
        "label": label,
        "type": action_type,
        "handler": handler,
        "params": params or {},
        "requires_confirmation": requires_confirmation,
    }


# ============================================================================
# Standard Actions
# ============================================================================

# Pre-defined common actions
ACTION_PIN = create_action("Pin", "pin_panel", action_type="toggle")
ACTION_DISMISS = create_action("Dismiss", "dismiss_panel")
ACTION_WHY = create_action("Why this UI?", "show_why_panel")
ACTION_EXPAND = create_action("Expand", "expand_panel")
ACTION_COLLAPSE = create_action("Collapse", "collapse_panel")
ACTION_COPY = create_action("Copy", "copy_content")
ACTION_EXPORT = create_action("Export", "export_panel", requires_confirmation=True)


# ============================================================================
# Default Actions by Panel Type
# ============================================================================

DEFAULT_ACTIONS: Dict[PanelTypeJenny, Tuple[Dict[str, Any], ...]] = {
    PanelTypeJenny.GRAPH: (ACTION_PIN, ACTION_DISMISS, ACTION_EXPAND, ACTION_WHY),
    PanelTypeJenny.CONFIDENCE: (ACTION_PIN, ACTION_DISMISS, ACTION_WHY),
    PanelTypeJenny.TIMELINE: (ACTION_PIN, ACTION_DISMISS, ACTION_WHY),
    PanelTypeJenny.MEMORY_MAP: (ACTION_PIN, ACTION_DISMISS, ACTION_EXPAND, ACTION_WHY),
    PanelTypeJenny.TEXT: (ACTION_PIN, ACTION_DISMISS, ACTION_COPY, ACTION_WHY),
    PanelTypeJenny.CODE: (ACTION_PIN, ACTION_DISMISS, ACTION_COPY, ACTION_WHY),
    PanelTypeJenny.TABLE: (ACTION_PIN, ACTION_DISMISS, ACTION_EXPORT, ACTION_WHY),
    PanelTypeJenny.METRIC: (ACTION_PIN, ACTION_DISMISS, ACTION_WHY),
    PanelTypeJenny.REASONING: (ACTION_PIN, ACTION_DISMISS, ACTION_EXPAND, ACTION_WHY),
    PanelTypeJenny.SOURCES: (ACTION_PIN, ACTION_DISMISS, ACTION_WHY),
    PanelTypeJenny.ACTIONS: (ACTION_DISMISS,),
    PanelTypeJenny.WHY: (ACTION_DISMISS,),  # Meta-panel, minimal actions
    PanelTypeJenny.COMPARISON: (ACTION_PIN, ACTION_DISMISS, ACTION_EXPORT, ACTION_WHY),
}


def get_default_actions(panel_type: PanelTypeJenny) -> Tuple[Dict[str, Any], ...]:
    """Get default actions for a panel type."""
    return DEFAULT_ACTIONS.get(panel_type, (ACTION_PIN, ACTION_DISMISS, ACTION_WHY))


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    "LifecycleStage",
    "BindingMode",
    "DissolutionTrigger",
    "PanelTypeJenny",
    "PanelSizeJenny",
    "LayoutHint",
    # Core spec
    "JennySpec",
    # Helpers
    "create_action",
    "get_default_actions",
    # Standard actions
    "ACTION_PIN",
    "ACTION_DISMISS",
    "ACTION_WHY",
    "ACTION_EXPAND",
    "ACTION_COLLAPSE",
    "ACTION_COPY",
    "ACTION_EXPORT",
    "DEFAULT_ACTIONS",
]
