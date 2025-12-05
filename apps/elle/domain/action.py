"""Actions Elle can take in response to requests."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class ElleMode(Enum):
    """How Elle manifests in AR space."""
    AMBIENT = "ambient"  # Gentle presence, no urgency
    CONSULTING = "consulting"  # Standing beside, advisory
    DIRECTIVE = "directive"  # Clear guidance, front-and-center
    SILENT = "silent"  # Present but not speaking


class VisualStyle(Enum):
    """Visual presentation in AR."""
    SUBTLE_GLOW = "subtle_glow"
    SOFT_HIGHLIGHT = "soft_highlight"
    CLEAR_MARKER = "clear_marker"
    FULL_OVERLAY = "full_overlay"
    INVISIBLE = "invisible"


class Priority(Enum):
    """Task/action urgency."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Symbol(Enum):
    """Mythic symbols that guide Elle's behavior and visual style."""
    CHIMBORAZO = "chimborazo"  # Focus and priority
    PLATO = "plato"            # Clarity and understanding
    PENELOPE = "penelope"      # Patience and weaving


@dataclass(frozen=True)
class Visual:
    """Visual elements to render in AR."""
    
    style: VisualStyle
    target_object_ids: List[str] = field(default_factory=list)  # Objects to highlight
    
    # Optional visual details
    color: Optional[str] = None  # "amber", "blue", "green"
    animation: Optional[str] = None  # "pulse", "drift", "static"
    position: Optional[str] = None  # Where in space: "beside_user", "at_object"
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Followup:
    """Action to dispatch to a service after decision."""
    
    service: str  # "scheduler", "layout_planner", "task_tracker"
    action: str  # "schedule_task", "generate_layout", "create_project"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ElleAction:
    """
    Complete response from Elle to a request.
    
    This is what gets rendered in AR / Matrix / CLI.
    Frozen = can be snapshot-tested, cached, logged immutably.
    """
    
    # How Elle appears
    mode: ElleMode
    
    # What Elle says (if anything)
    utterance: Optional[str] = None
    
    # Visual presentation
    visuals: Optional[Visual] = None
    
    # Suggested task (if any)
    suggested_task_id: Optional[str] = None
    suggested_task_summary: Optional[str] = None
    task_priority: Optional[Priority] = None
    task_estimate_minutes: Optional[int] = None
    
    # Follow-up actions to dispatch
    followups: List[Followup] = field(default_factory=list)
    
    # Reasoning (for debugging/learning)
    reasoning: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_silent(self) -> bool:
        """Is Elle silent in this interaction?"""
        return self.mode == ElleMode.SILENT or self.utterance is None
    
    @property
    def has_task(self) -> bool:
        """Does this action include a task suggestion?"""
        return self.suggested_task_id is not None
    
    @property
    def has_visuals(self) -> bool:
        """Does this action include visual elements?"""
        return self.visuals is not None and self.visuals.style != VisualStyle.INVISIBLE


@dataclass(frozen=True)
class ElleRequest:
    """
    Normalized request to Elle from any adapter.
    
    This is the contract between adapters and the engine.
    """
    
    # Core inputs
    scene: 'SceneSnapshot'  # From domain.scene
    intent: 'UserIntent'  # From domain.intent
    user: 'UserState'  # From domain.intent
    
    # Request metadata
    request_id: str
    timestamp: Any  # datetime
    source_adapter: str  # "ar", "matrix", "cli"
    
    # Extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)
