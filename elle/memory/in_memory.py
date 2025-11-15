"""In-memory implementation of MemoryStore for testing."""

from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

from .store import (
    MemoryStore,
    MemorySnapshot,
    InteractionRecord,
    UserPattern
)
from ..domain import ElleRequest, ElleAction


class InMemoryMemoryStore(MemoryStore):
    """
    Simple in-memory storage for testing.
    
    Not persistent—resets on restart.
    Thread-safe not guaranteed—single-threaded only.
    """
    
    def __init__(self):
        # user_id -> list of interactions
        self._interactions: Dict[str, List[InteractionRecord]] = defaultdict(list)
        
        # user_id -> patterns
        self._patterns: Dict[str, UserPattern] = {}
        
        # task_id -> completion status
        self._task_outcomes: Dict[str, tuple[bool, Optional[datetime]]] = {}
    
    def load_memory(self, user_id: str) -> MemorySnapshot:
        """Load snapshot from in-memory store."""
        
        interactions = self._interactions.get(user_id, [])
        
        # Get recent interactions (last 10)
        recent = interactions[-10:] if len(interactions) > 10 else interactions
        
        # Extract recent locations
        locations = list(dict.fromkeys(  # Dedupe while preserving order
            i.request.scene.location for i in recent
        ))
        
        # Get active tasks (from recent interactions that suggested tasks)
        active_tasks = [
            i.action.suggested_task_id
            for i in recent
            if i.action.suggested_task_id and not i.task_completed
        ]
        
        # Get patterns
        patterns = self._patterns.get(user_id)
        
        return MemorySnapshot(
            user_id=user_id,
            loaded_at=datetime.now(),
            recent_interactions=recent,
            recent_locations=locations,
            patterns=patterns,
            active_tasks=active_tasks,
            active_projects=[],  # TODO: Track projects
            total_interactions=len(interactions),
        )
    
    def record_interaction(
        self,
        request: ElleRequest,
        action: ElleAction
    ) -> None:
        """Record interaction."""
        
        record = InteractionRecord(
            timestamp=datetime.now(),
            request=request,
            action=action,
        )
        
        self._interactions[request.user.user_id].append(record)
    
    def update_task_outcome(
        self,
        task_id: str,
        completed: bool,
        completion_time: Optional[datetime] = None
    ) -> None:
        """Update task completion."""
        
        self._task_outcomes[task_id] = (
            completed,
            completion_time or datetime.now()
        )
    
    def get_user_patterns(self, user_id: str) -> Optional[UserPattern]:
        """Get patterns."""
        return self._patterns.get(user_id)
    
    def set_user_patterns(self, user_id: str, patterns: UserPattern) -> None:
        """Set patterns (for testing)."""
        self._patterns[user_id] = patterns
    
    def clear(self) -> None:
        """Clear all data (for testing)."""
        self._interactions.clear()
        self._patterns.clear()
        self._task_outcomes.clear()
