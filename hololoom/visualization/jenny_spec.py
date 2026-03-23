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

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

# Import consolidated enums (re-exported for backward compatibility)
from .jenny_enums import BindingMode, DissolutionTrigger, LifecycleStage

# ============================================================================
# Panel-Specific Enums (kept here - tightly coupled to panel definitions)
# ============================================================================

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

    # Visualization primitive types (five primitives grammar)
    PROMOTION = "promotion"        # Trajectory + Distribution (promotion flow + score bars)
    DEPARTMENT = "department"      # Topology + Distribution + Tension (communities + scores + drift)
    CROSS_DEPT = "cross_dept"      # Topology + Tension (bridges + Wasserstein distances)
    FEEDBACK_STATE = "feedback_state"  # Trajectory + Distribution (convergence + bandit arms)


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
    title: str | None = None
    subtitle: str | None = None
    content: dict[str, Any] = field(default_factory=dict)

    # ========================================================================
    # Lifecycle
    # ========================================================================
    lifecycle: LifecycleStage = LifecycleStage.NASCENT
    dissolution_trigger: DissolutionTrigger | None = None

    # ========================================================================
    # Layout
    # ========================================================================
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z (AR-ready)
    size: PanelSizeJenny = PanelSizeJenny.MEDIUM
    priority: int = 0  # Higher = closer to user (AR), higher z-index (web)

    # ========================================================================
    # Data Binding
    # ========================================================================
    binding_mode: BindingMode = BindingMode.STATIC
    data_source: str | None = None  # Query for reactive/streaming bindings
    refresh_interval_ms: int | None = None  # For reactive mode

    # ========================================================================
    # Actions
    # ========================================================================
    actions: tuple[dict[str, Any], ...] = ()  # Immutable tuple of action definitions

    # ========================================================================
    # Provenance
    # ========================================================================
    timestamp: datetime = field(default_factory=datetime.now)
    compiler_version: str = "1.0.0"
    model_checkpoint: str | None = None  # For replay determinism

    # ========================================================================
    # Metadata (extensible)
    # ========================================================================
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate spec after initialization."""
        # Note: Can't modify fields in frozen dataclass, but can validate
        if self.binding_mode == BindingMode.REACTIVE and not self.data_source:
            raise ValueError("REACTIVE binding mode requires data_source")
        if self.binding_mode == BindingMode.STREAMING and not self.data_source:
            raise ValueError("STREAMING binding mode requires data_source")

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> "JennySpec":
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

    def evolve(self, **kwargs) -> "JennySpec":
        """
        Create new spec with updated fields (generic immutable update pattern).

        Uses dataclasses.replace() for elegant field updates on frozen dataclass.

        Usage:
            updated = spec.evolve(title="New Title", priority=2)
            moved = spec.evolve(position=(1.0, 2.0, 0.0))
        """
        return replace(self, **kwargs)

    def with_lifecycle(self, new_lifecycle: LifecycleStage, trigger: DissolutionTrigger | None = None) -> "JennySpec":
        """
        Create new spec with updated lifecycle (immutable update pattern).

        Usage:
            stable_spec = nascent_spec.with_lifecycle(LifecycleStage.STABLE)
            dissolving_spec = stable_spec.with_lifecycle(
                LifecycleStage.DISSOLVING,
                trigger=DissolutionTrigger.MANUAL
            )
        """
        dissolution = trigger if new_lifecycle == LifecycleStage.DISSOLVING else self.dissolution_trigger
        return replace(self, lifecycle=new_lifecycle, dissolution_trigger=dissolution)

    def with_position(self, x: float, y: float, z: float = 0.0) -> "JennySpec":
        """Create new spec with updated position (immutable update pattern)."""
        return replace(self, position=(x, y, z))


# ============================================================================
# Action Helpers
# ============================================================================

def create_action(
    label: str,
    handler: str,
    action_type: str = "button",
    params: dict[str, Any] | None = None,
    requires_confirmation: bool = False,
) -> dict[str, Any]:
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
ACTION_REFRESH = create_action("Refresh", "refresh_panel")
ACTION_DRILL = create_action("Drill Down", "drill_panel")


# ============================================================================
# Default Actions by Panel Type
# ============================================================================

DEFAULT_ACTIONS: dict[PanelTypeJenny, tuple[dict[str, Any], ...]] = {
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
    # Visualization primitives — Pin emits MINUTES feedback, Dismiss emits negative SECONDS
    PanelTypeJenny.PROMOTION: (ACTION_PIN, ACTION_DISMISS, ACTION_REFRESH, ACTION_DRILL, ACTION_WHY),
    PanelTypeJenny.DEPARTMENT: (ACTION_PIN, ACTION_DISMISS, ACTION_REFRESH, ACTION_DRILL, ACTION_EXPAND, ACTION_WHY),
    PanelTypeJenny.CROSS_DEPT: (ACTION_PIN, ACTION_DISMISS, ACTION_REFRESH, ACTION_EXPAND, ACTION_WHY),
    PanelTypeJenny.FEEDBACK_STATE: (ACTION_PIN, ACTION_DISMISS, ACTION_REFRESH, ACTION_WHY),
}


def get_default_actions(panel_type: PanelTypeJenny) -> tuple[dict[str, Any], ...]:
    """Get default actions for a panel type."""
    return DEFAULT_ACTIONS.get(panel_type, (ACTION_PIN, ACTION_DISMISS, ACTION_WHY))


def create_spec(
    spacetime_id: str = "",
    panel_type: PanelTypeJenny = PanelTypeJenny.TEXT,
    title: str | None = None,
    subtitle: str | None = None,
    content: dict[str, Any] | None = None,
    size: PanelSizeJenny = PanelSizeJenny.MEDIUM,
    priority: int = 0,
    lifecycle: LifecycleStage = LifecycleStage.NASCENT,
    binding_mode: BindingMode = BindingMode.STATIC,
    data_source: str | None = None,
    actions: tuple[dict[str, Any], ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> JennySpec:
    """
    Factory function to create JennySpec with sensible defaults.

    This is a convenience wrapper around JennySpec constructor that
    provides cleaner syntax for common use cases in tests and demos.

    Args:
        spacetime_id: Link to Spacetime that generated this
        panel_type: Type of panel (TEXT, GRAPH, etc.)
        title: Panel title
        subtitle: Panel subtitle
        content: Type-specific content dictionary
        size: Panel size (SMALL, MEDIUM, LARGE, XLARGE)
        priority: Priority for layout (higher = closer to user)
        lifecycle: Initial lifecycle stage
        binding_mode: Data binding mode (STATIC, REACTIVE, STREAMING)
        data_source: Query for reactive/streaming bindings
        actions: Tuple of action definitions (uses defaults if None)
        metadata: Additional metadata dictionary

    Returns:
        JennySpec: Configured panel specification

    Usage:
        spec = create_spec(
            spacetime_id="st-001",
            panel_type=PanelTypeJenny.TEXT,
            title="Response",
            content={"text": "Hello, world!"},
            priority=1,
        )
    """
    if content is None:
        content = {}
    if actions is None:
        actions = get_default_actions(panel_type)
    if metadata is None:
        metadata = {}

    return JennySpec(
        spacetime_id=spacetime_id,
        panel_type=panel_type,
        title=title,
        subtitle=subtitle,
        content=content,
        size=size,
        priority=priority,
        lifecycle=lifecycle,
        binding_mode=binding_mode,
        data_source=data_source,
        actions=actions,
        metadata=metadata,
    )


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
    "create_spec",
    "get_default_actions",
    # Standard actions
    "ACTION_PIN",
    "ACTION_DISMISS",
    "ACTION_WHY",
    "ACTION_EXPAND",
    "ACTION_COLLAPSE",
    "ACTION_COPY",
    "ACTION_EXPORT",
    "ACTION_REFRESH",
    "ACTION_DRILL",
    "DEFAULT_ACTIONS",
]
