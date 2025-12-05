"""User intent and interaction patterns."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class ScanType(Enum):
    """How the user is engaging with the scene."""
    QUICK_GLANCE = "quick_glance"  # Brief look, low engagement
    SLOW_SCAN = "slow_scan"  # Deliberate, thoughtful attention
    FOCUSED = "focused"  # Looking at specific object
    AMBIENT = "ambient"  # Just present, not actively scanning


class InteractionMode(Enum):
    """User's current interaction style."""
    PASSIVE = "passive"  # Just observing
    SEEKING_GUIDANCE = "seeking_guidance"  # Want suggestions
    REQUESTING_INFO = "requesting_info"  # Asking questions
    TASK_EXECUTION = "task_execution"  # Actively doing work


@dataclass(frozen=True)
class UserIntent:
    """
    What the user wants in this moment.
    
    Combination of explicit request + inferred context.
    """
    
    # Primary intent
    mode: InteractionMode
    scan_type: ScanType
    
    # Explicit request (if any)
    explicit_request: Optional[str] = None  # e.g., "help me organize"
    
    # Inferred context
    is_tired: bool = False
    is_rushed: bool = False
    is_exploring: bool = False
    
    # Task context (if relevant)
    active_task_id: Optional[str] = None
    active_project_id: Optional[str] = None
    
    # Extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def wants_action(self) -> bool:
        """Does user want Elle to suggest action?"""
        return self.mode in (
            InteractionMode.SEEKING_GUIDANCE,
            InteractionMode.TASK_EXECUTION
        )
    
    @property
    def high_engagement(self) -> bool:
        """Is user actively engaged?"""
        return self.scan_type in (ScanType.SLOW_SCAN, ScanType.FOCUSED)


@dataclass(frozen=True)
class UserState:
    """
    Current state of the user across sessions.
    
    This is loaded from memory, not inferred per-request.
    """
    
    user_id: str
    name: str
    
    # Energy / capacity
    current_energy_level: str = "medium"  # "high", "medium", "low"
    time_available: Optional[str] = None  # "10_minutes", "hour", "afternoon"
    
    # Preferences
    preferred_pace: str = "moderate"  # "slow", "moderate", "fast"
    preferred_complexity: str = "medium"  # "simple", "medium", "complex"
    
    # Context
    current_projects: List[str] = field(default_factory=list)
    recent_completions: List[str] = field(default_factory=list)
    
    # Extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)
