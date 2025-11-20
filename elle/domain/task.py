"""Task and project domain models."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class TaskStatus(Enum):
    """Lifecycle of a task."""
    SUGGESTED = "suggested"  # Elle suggested it
    ACCEPTED = "accepted"  # User accepted
    IN_PROGRESS = "in_progress"  # Being worked on
    PAUSED = "paused"  # Temporarily stopped
    COMPLETED = "completed"  # Done
    CANCELLED = "cancelled"  # Won't do
    ARCHIVED = "archived"  # Historical record


class TaskType(Enum):
    """Categories of work."""
    MAINTENANCE = "maintenance"  # Ongoing care
    ORGANIZATION = "organization"  # Tidying, arranging
    CULTIVATION = "cultivation"  # Growing, nurturing
    REPAIR = "repair"  # Fixing
    CONSTRUCTION = "construction"  # Building
    PLANNING = "planning"  # Thinking, designing


@dataclass(frozen=True)
class TaskEstimate:
    """Estimated effort for a task."""
    
    time_minutes: int
    difficulty: str = "medium"  # "easy", "medium", "hard"
    
    # Optional resource needs
    tools_needed: List[str] = field(default_factory=list)
    materials_needed: List[str] = field(default_factory=list)
    
    # Conditions
    best_time_of_day: Optional[str] = None  # "morning", "afternoon", "evening"
    best_weather: Optional[str] = None  # "sunny", "cloudy", "any"


@dataclass
class Task:
    """
    A discrete unit of work.
    
    Mutable because status changes over time.
    """
    
    # Identity
    task_id: str
    title: str
    description: str
    
    # Classification
    task_type: TaskType
    status: TaskStatus
    
    # Scheduling
    created_at: datetime
    suggested_by: str = "elle"  # Who suggested it
    
    # Estimation
    estimate: Optional[TaskEstimate] = None
    
    # Relationships
    project_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)  # Other task IDs
    
    # Location
    location: Optional[str] = None  # "shed", "bed_3", etc.
    
    # Completion
    completed_at: Optional[datetime] = None
    actual_duration_minutes: Optional[int] = None
    
    # Notes
    notes: List[str] = field(default_factory=list)
    
    # Extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Project:
    """
    A collection of related tasks toward a larger goal.
    
    Examples: "Shed reorganization", "Cucumber bed overhaul", "Front fence repair"
    """
    
    # Identity
    project_id: str
    title: str
    description: str
    
    # Status
    status: TaskStatus  # Reuse same lifecycle
    
    # Timeframe
    created_at: datetime
    target_completion: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Structure
    task_ids: List[str] = field(default_factory=list)
    milestones: List[str] = field(default_factory=list)
    
    # Context
    location: Optional[str] = None
    
    # Outcome
    success_criteria: List[str] = field(default_factory=list)
    
    # Extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Is this project currently being worked on?"""
        return self.status in (TaskStatus.ACCEPTED, TaskStatus.IN_PROGRESS)
