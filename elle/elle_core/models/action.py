"""
Elle Core - Action Models

Defines the structured output from Elle's decision-making process.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class ActionMode(Enum):
    """How Elle appears/behaves in this moment."""
    IDLE = "idle"                      # Ambient presence, no intervention
    FOCUS_ASSIST = "focus_assist"      # Gentle guidance on current task
    SILENT_INDICATOR = "silent_indicator"  # Visual cue only, no speech
    DEFERRED = "deferred"              # Explicitly choosing to wait


class Priority(Enum):
    """Urgency level of this action."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VisualDirective(Enum):
    """Types of AR visual elements Elle can display."""
    HIGHLIGHT = "highlight"        # Highlight an object
    PATH = "path"                  # Show a path/direction
    INDICATOR = "indicator"        # Show a dot/icon
    AVATAR_MOVE = "avatar_move"    # Move Elle's avatar
    ZONE = "zone"                  # Highlight an area


@dataclass
class Visual:
    """A single visual directive for the AR interface."""
    type: VisualDirective
    targets: List[str]  # Object IDs or coordinates
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.type.value}[{', '.join(self.targets)}]"


@dataclass
class Followup:
    """A followup action to be executed by domain services."""
    action: str  # e.g., "schedule_reminder", "log_observation", "create_task"
    params: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.action}({params_str})"


@dataclass
class ElleAction:
    """
    The structured output from Elle's policy decision.

    This is what Elle decides to do (or not do) in response to a scene.
    """
    mode: ActionMode
    priority: Priority = Priority.LOW
    utterance: Optional[str] = None  # What Elle says (or None for silence)
    visuals: List[Visual] = field(default_factory=list)
    followups: List[Followup] = field(default_factory=list)
    reasoning: Optional[str] = None  # Why Elle chose this action

    def is_silent(self) -> bool:
        """Check if this action has no visible/audible output."""
        return (
            self.utterance is None
            and len(self.visuals) == 0
            and self.mode in [ActionMode.IDLE, ActionMode.DEFERRED]
        )

    def has_intervention(self) -> bool:
        """Check if this action intervenes in the user's flow."""
        return self.utterance is not None or len(self.visuals) > 0

    def __str__(self) -> str:
        parts = [f"Mode: {self.mode.value}"]
        if self.utterance:
            parts.append(f'Says: "{self.utterance}"')
        if self.visuals:
            parts.append(f"Visuals: {', '.join(str(v) for v in self.visuals)}")
        if self.followups:
            parts.append(f"Followups: {', '.join(str(f) for f in self.followups)}")
        return " | ".join(parts)
