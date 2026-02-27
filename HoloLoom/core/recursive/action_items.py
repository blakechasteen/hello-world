"""
Phase 6: Action Items System
==============================

Persistent task tracking with intelligent priority scoring and learning.

Philosophy:
-----------
"Great systems don't just respond - they remember what needs to be done."

The action items system extends the recursive learning foundation with:
- Persistent cross-session task tracking
- Intelligent priority scoring with Thompson Sampling
- Auto-extraction from scratchpad observations
- Lifecycle management (pending → in_progress → completed → archived)

This enables HoloLoom to maintain long-term goals and track progress toward them,
learning from completion patterns to improve future scheduling.

Usage:
------
    # Create tracker
    tracker = ActionItemTracker(persist_path="./action_items.json")

    # Add action item
    item = tracker.add_action(
        description="Implement semantic caching for repeated queries",
        priority=0.8,
        context={"query": "How to optimize performance?"}
    )

    # Get high-priority pending items
    pending = tracker.get_pending_actions(min_priority=0.7)

    # Complete action
    tracker.complete_action(item.id, success=True, quality_score=0.92)

    # Learn from completion patterns
    tracker.update_priority_model()

Integration with Scratchpad:
----------------------------
    # Auto-extract from scratchpad observations
    scratchpad_entry = ScratchpadEntry(
        thought="Need better caching",
        action="Research semantic caching approaches",
        observation="TODO: Implement embedding-based cache with similarity threshold",
        score=0.85
    )

    # Extract action items automatically
    actions = extract_action_items_from_scratchpad(scratchpad_entry)
    for action in actions:
        tracker.add_action(**action)
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class ActionStatus(Enum):
    """Lifecycle stages for action items"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    BLOCKED = "blocked"


class ActionCategory(Enum):
    """Categories for automatic classification"""
    BUG_FIX = "bug_fix"
    FEATURE = "feature"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"
    REFACTOR = "refactor"
    TESTING = "testing"
    RESEARCH = "research"
    UNKNOWN = "unknown"


@dataclass
class ActionItem:
    """
    Single action item with metadata and lifecycle tracking.

    Attributes:
        id: Unique identifier (auto-generated)
        description: Clear description of what needs to be done
        priority: 0-1 score (higher = more important)
        status: Current lifecycle stage
        category: Auto-classified category

        created_at: When item was created
        started_at: When work began (optional)
        completed_at: When work finished (optional)
        due_date: Optional deadline

        context: Related data (queries, spacetimes, etc.)
        dependencies: Other action IDs this depends on
        blocks: Other action IDs blocked by this

        success: Whether completed successfully
        quality_score: Quality of completion (0-1)
        completion_time_ms: How long it took
    """
    id: str
    description: str
    priority: float
    status: ActionStatus = ActionStatus.PENDING
    category: ActionCategory = ActionCategory.UNKNOWN

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    due_date: Optional[datetime] = None

    # Relationships
    context: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    blocks: Set[str] = field(default_factory=set)

    # Completion metrics
    success: Optional[bool] = None
    quality_score: Optional[float] = None
    completion_time_ms: Optional[float] = None

    # Learning metadata
    priority_confidence: float = 0.5  # How confident are we in the priority?
    predicted_duration_ms: Optional[float] = None
    actual_vs_predicted_ratio: Optional[float] = None

    def is_overdue(self) -> bool:
        """Check if action is overdue"""
        if self.due_date is None:
            return False
        return datetime.now() > self.due_date and self.status not in [
            ActionStatus.COMPLETED, ActionStatus.ARCHIVED
        ]

    def is_blocked(self) -> bool:
        """Check if action is blocked by dependencies"""
        return self.status == ActionStatus.BLOCKED or len(self.dependencies) > 0

    def get_age_hours(self) -> float:
        """Get age of action item in hours"""
        return (datetime.now() - self.created_at).total_seconds() / 3600

    def get_urgency_score(self) -> float:
        """
        Calculate urgency score based on due date and age.

        Returns:
            0-1 score (higher = more urgent)
        """
        urgency = 0.0

        # Due date urgency
        if self.due_date:
            hours_until_due = (self.due_date - datetime.now()).total_seconds() / 3600
            if hours_until_due < 0:
                urgency = 1.0  # Overdue!
            elif hours_until_due < 24:
                urgency = 0.8  # Due within a day
            elif hours_until_due < 72:
                urgency = 0.5  # Due within 3 days
            else:
                urgency = 0.2  # Due later

        # Age urgency (older items become more urgent)
        age_hours = self.get_age_hours()
        age_urgency = min(1.0, age_hours / (7 * 24))  # Max urgency after 1 week

        # Combine (max of the two)
        return max(urgency, age_urgency * 0.3)

    def get_effective_priority(self) -> float:
        """
        Calculate effective priority combining base priority and urgency.

        Returns:
            0-1 score (higher = should do first)
        """
        urgency = self.get_urgency_score()
        # Weighted combination: 70% base priority, 30% urgency
        return 0.7 * self.priority + 0.3 * urgency


@dataclass
class PriorityModel:
    """
    Thompson Sampling model for learning optimal priorities.

    Learns from completion patterns:
    - Which categories get completed fastest?
    - Which priority ranges are most accurate?
    - What predicts successful completion?
    """
    # Beta distribution parameters per category
    category_priors: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(
        lambda: {"alpha": 1.0, "beta": 1.0}
    ))

    # Completion time predictions per category
    avg_completion_times: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    completion_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Priority calibration
    priority_buckets: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    def update_from_completion(
        self,
        category: ActionCategory,
        priority: float,
        success: bool,
        quality_score: float,
        completion_time_ms: float
    ):
        """Update model after action completion"""
        cat_key = category.value

        # Update Beta priors based on success
        if success and quality_score >= 0.7:
            self.category_priors[cat_key]["alpha"] += quality_score
        else:
            self.category_priors[cat_key]["beta"] += (1.0 - quality_score)

        # Update completion time estimates
        count = self.completion_counts[cat_key]
        avg = self.avg_completion_times[cat_key]

        # Running average
        self.avg_completion_times[cat_key] = (
            (avg * count + completion_time_ms) / (count + 1)
        )
        self.completion_counts[cat_key] += 1

        # Track priority calibration
        priority_bucket = f"{int(priority * 10) / 10:.1f}"
        self.priority_buckets[priority_bucket].append(quality_score)

    def get_expected_success_rate(self, category: ActionCategory) -> float:
        """Get expected success rate for category using Thompson Sampling"""
        cat_key = category.value
        priors = self.category_priors[cat_key]
        alpha = priors["alpha"]
        beta = priors["beta"]

        # Expected value of Beta distribution
        return alpha / (alpha + beta)

    def get_predicted_duration(self, category: ActionCategory) -> float:
        """Get predicted completion duration for category"""
        cat_key = category.value
        return self.avg_completion_times.get(cat_key, 3600000.0)  # Default 1 hour

    def suggest_priority_adjustment(self, priority: float) -> float:
        """
        Suggest adjusted priority based on historical calibration.

        Returns:
            Adjusted priority (0-1)
        """
        priority_bucket = f"{int(priority * 10) / 10:.1f}"
        quality_scores = self.priority_buckets.get(priority_bucket, [])

        if not quality_scores:
            return priority  # No data, keep as is

        # If this priority level consistently over/under-performs, adjust
        avg_quality = sum(quality_scores) / len(quality_scores)

        # If quality is consistently high, this priority level is well-calibrated
        # If quality is low, this priority level might be overestimated
        adjustment = (avg_quality - 0.75) * 0.2  # Small adjustment

        return max(0.0, min(1.0, priority + adjustment))


# ============================================================================
# Action Item Extraction
# ============================================================================

def extract_action_items_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract action items from free text using pattern matching.

    Looks for patterns like:
    - TODO: ...
    - Action: ...
    - Next: ...
    - Fix: ...
    - Implement: ...

    Args:
        text: Input text to extract from

    Returns:
        List of action item dicts with description and auto-detected metadata
    """
    patterns = [
        r"TODO:\s*(.+?)(?:\n|$)",
        r"Action:\s*(.+?)(?:\n|$)",
        r"Next:\s*(.+?)(?:\n|$)",
        r"Fix:\s*(.+?)(?:\n|$)",
        r"Implement:\s*(.+?)(?:\n|$)",
        r"Refactor:\s*(.+?)(?:\n|$)",
        r"Optimize:\s*(.+?)(?:\n|$)",
        r"Research:\s*(.+?)(?:\n|$)",
        r"Test:\s*(.+?)(?:\n|$)",
        r"Document:\s*(.+?)(?:\n|$)",
    ]

    actions = []

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            description = match.group(1).strip()
            category = _classify_action(description, pattern)
            priority = _estimate_priority(description)

            actions.append({
                "description": description,
                "category": category,
                "priority": priority,
                "context": {"extracted_from": text[:100]}
            })

    return actions


def _classify_action(description: str, pattern: str) -> ActionCategory:
    """Auto-classify action based on description and pattern"""
    desc_lower = description.lower()

    if "bug" in desc_lower or "fix" in desc_lower or "error" in desc_lower:
        return ActionCategory.BUG_FIX
    elif "feature" in desc_lower or "add" in desc_lower or "implement" in desc_lower:
        return ActionCategory.FEATURE
    elif "optim" in desc_lower or "speed" in desc_lower or "performance" in desc_lower:
        return ActionCategory.OPTIMIZATION
    elif "document" in desc_lower or "readme" in desc_lower or "doc" in desc_lower:
        return ActionCategory.DOCUMENTATION
    elif "refactor" in desc_lower or "clean" in desc_lower or "reorg" in desc_lower:
        return ActionCategory.REFACTOR
    elif "test" in desc_lower or "verify" in desc_lower:
        return ActionCategory.TESTING
    elif "research" in desc_lower or "investigate" in desc_lower or "explore" in desc_lower:
        return ActionCategory.RESEARCH
    else:
        return ActionCategory.UNKNOWN


def _estimate_priority(description: str) -> float:
    """Estimate priority from description text"""
    desc_lower = description.lower()

    # High priority keywords
    if any(kw in desc_lower for kw in ["urgent", "critical", "asap", "immediately", "blocking"]):
        return 0.9

    # Medium-high priority
    if any(kw in desc_lower for kw in ["important", "needed", "should", "bug"]):
        return 0.7

    # Medium priority
    if any(kw in desc_lower for kw in ["nice to have", "could", "consider"]):
        return 0.5

    # Default
    return 0.6


# ============================================================================
# Action Item Tracker
# ============================================================================

class ActionItemTracker:
    """
    Persistent action item tracking with intelligent priority scoring.

    Features:
    - CRUD operations for action items
    - Priority-based scheduling
    - Thompson Sampling for learning
    - Auto-extraction from text
    - Lifecycle management
    - Statistics and analytics
    """

    def __init__(
        self,
        persist_path: str = "./action_items.json",
        auto_archive_after_days: int = 30
    ):
        """
        Initialize tracker.

        Args:
            persist_path: Path to JSON persistence file
            auto_archive_after_days: Archive completed items after N days
        """
        self.persist_path = Path(persist_path)
        self.auto_archive_after_days = auto_archive_after_days

        self.actions: Dict[str, ActionItem] = {}
        self.priority_model = PriorityModel()

        self._next_id = 1

        # Load from disk if exists
        if self.persist_path.exists():
            self.load()

    def add_action(
        self,
        description: str,
        priority: float = 0.5,
        category: Optional[ActionCategory] = None,
        due_date: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Set[str]] = None
    ) -> ActionItem:
        """
        Add new action item.

        Args:
            description: What needs to be done
            priority: 0-1 importance score
            category: Optional category (auto-detected if None)
            due_date: Optional deadline
            context: Optional context data
            dependencies: Optional set of action IDs this depends on

        Returns:
            Created ActionItem
        """
        # Auto-classify if not specified
        if category is None:
            category = _classify_action(description, "")

        # Adjust priority based on learned model
        adjusted_priority = self.priority_model.suggest_priority_adjustment(priority)

        # Generate ID
        action_id = f"action_{self._next_id:04d}"
        self._next_id += 1

        # Create action
        action = ActionItem(
            id=action_id,
            description=description,
            priority=adjusted_priority,
            category=category,
            due_date=due_date,
            context=context or {},
            dependencies=dependencies or set(),
            predicted_duration_ms=self.priority_model.get_predicted_duration(category)
        )

        self.actions[action_id] = action
        self.save()

        logger.info(f"Added action item: {action_id} - {description[:50]}")

        return action

    def start_action(self, action_id: str) -> ActionItem:
        """Mark action as in progress"""
        action = self.actions.get(action_id)
        if not action:
            raise ValueError(f"Action {action_id} not found")

        action.status = ActionStatus.IN_PROGRESS
        action.started_at = datetime.now()

        self.save()
        logger.info(f"Started action: {action_id}")

        return action

    def complete_action(
        self,
        action_id: str,
        success: bool = True,
        quality_score: float = 0.8
    ) -> ActionItem:
        """
        Mark action as completed and update learning model.

        Args:
            action_id: Action to complete
            success: Whether completed successfully
            quality_score: 0-1 quality assessment

        Returns:
            Updated ActionItem
        """
        action = self.actions.get(action_id)
        if not action:
            raise ValueError(f"Action {action_id} not found")

        # Update action
        action.status = ActionStatus.COMPLETED
        action.completed_at = datetime.now()
        action.success = success
        action.quality_score = quality_score

        # Calculate completion time
        if action.started_at:
            action.completion_time_ms = (
                (action.completed_at - action.started_at).total_seconds() * 1000
            )
        else:
            action.completion_time_ms = (
                (action.completed_at - action.created_at).total_seconds() * 1000
            )

        # Update prediction accuracy
        if action.predicted_duration_ms:
            action.actual_vs_predicted_ratio = (
                action.completion_time_ms / action.predicted_duration_ms
            )

        # Learn from completion
        self.priority_model.update_from_completion(
            category=action.category,
            priority=action.priority,
            success=success,
            quality_score=quality_score,
            completion_time_ms=action.completion_time_ms
        )

        self.save()
        logger.info(
            f"Completed action: {action_id} "
            f"(success={success}, quality={quality_score:.2f})"
        )

        return action

    def get_pending_actions(
        self,
        min_priority: float = 0.0,
        category: Optional[ActionCategory] = None,
        limit: int = 10
    ) -> List[ActionItem]:
        """
        Get pending actions sorted by effective priority.

        Args:
            min_priority: Minimum priority threshold
            category: Optional category filter
            limit: Maximum number to return

        Returns:
            List of ActionItems sorted by effective priority (highest first)
        """
        pending = [
            action for action in self.actions.values()
            if action.status == ActionStatus.PENDING
            and action.get_effective_priority() >= min_priority
            and (category is None or action.category == category)
        ]

        # Sort by effective priority (descending)
        pending.sort(key=lambda a: a.get_effective_priority(), reverse=True)

        return pending[:limit]

    def get_overdue_actions(self) -> List[ActionItem]:
        """Get all overdue actions"""
        return [
            action for action in self.actions.values()
            if action.is_overdue()
        ]

    def auto_archive_completed(self) -> int:
        """
        Archive old completed items.

        Returns:
            Number of items archived
        """
        cutoff = datetime.now() - timedelta(days=self.auto_archive_after_days)
        archived_count = 0

        for action in self.actions.values():
            if (action.status == ActionStatus.COMPLETED
                and action.completed_at
                and action.completed_at < cutoff):

                action.status = ActionStatus.ARCHIVED
                archived_count += 1

        if archived_count > 0:
            self.save()
            logger.info(f"Auto-archived {archived_count} completed items")

        return archived_count

    def extract_from_text(self, text: str) -> List[ActionItem]:
        """
        Extract action items from text and add them.

        Args:
            text: Text to extract from

        Returns:
            List of created ActionItems
        """
        extracted = extract_action_items_from_text(text)
        created = []

        for action_data in extracted:
            action = self.add_action(**action_data)
            created.append(action)

        return created

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about action items.

        Returns:
            Dict with various statistics
        """
        pending = [a for a in self.actions.values() if a.status == ActionStatus.PENDING]
        in_progress = [a for a in self.actions.values() if a.status == ActionStatus.IN_PROGRESS]
        completed = [a for a in self.actions.values() if a.status == ActionStatus.COMPLETED]

        # Category breakdown
        category_counts = defaultdict(int)
        for action in self.actions.values():
            category_counts[action.category.value] += 1

        # Success rate
        success_rate = 0.0
        if completed:
            successful = sum(1 for a in completed if a.success)
            success_rate = successful / len(completed)

        # Average completion time
        avg_completion_time = 0.0
        if completed:
            times = [a.completion_time_ms for a in completed if a.completion_time_ms]
            if times:
                avg_completion_time = sum(times) / len(times)

        return {
            "total_actions": len(self.actions),
            "pending": len(pending),
            "in_progress": len(in_progress),
            "completed": len(completed),
            "overdue": len(self.get_overdue_actions()),
            "category_breakdown": dict(category_counts),
            "success_rate": success_rate,
            "avg_completion_time_ms": avg_completion_time,
            "model_accuracy": self._get_model_accuracy()
        }

    def _get_model_accuracy(self) -> float:
        """Get average prediction accuracy of priority model"""
        completed = [
            a for a in self.actions.values()
            if a.status == ActionStatus.COMPLETED
            and a.actual_vs_predicted_ratio is not None
        ]

        if not completed:
            return 0.0

        # Accuracy = 1 - abs(predicted - actual) / actual
        accuracies = []
        for action in completed:
            ratio = action.actual_vs_predicted_ratio
            # Ratio close to 1.0 is good prediction
            accuracy = 1.0 - min(1.0, abs(1.0 - ratio))
            accuracies.append(accuracy)

        return sum(accuracies) / len(accuracies)

    def save(self):
        """Save to disk"""
        data = {
            "actions": {
                action_id: self._action_to_dict(action)
                for action_id, action in self.actions.items()
            },
            "priority_model": self._model_to_dict(),
            "next_id": self._next_id
        }

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.persist_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load from disk"""
        with open(self.persist_path, 'r') as f:
            data = json.load(f)

        self.actions = {
            action_id: self._action_from_dict(action_dict)
            for action_id, action_dict in data.get("actions", {}).items()
        }

        self._next_id = data.get("next_id", 1)

        model_data = data.get("priority_model", {})
        self.priority_model = self._model_from_dict(model_data)

        logger.info(f"Loaded {len(self.actions)} actions from {self.persist_path}")

    def _action_to_dict(self, action: ActionItem) -> Dict[str, Any]:
        """Convert ActionItem to JSON-serializable dict"""
        d = asdict(action)
        d['status'] = action.status.value
        d['category'] = action.category.value
        d['dependencies'] = list(action.dependencies)
        d['blocks'] = list(action.blocks)
        d['created_at'] = action.created_at.isoformat() if action.created_at else None
        d['started_at'] = action.started_at.isoformat() if action.started_at else None
        d['completed_at'] = action.completed_at.isoformat() if action.completed_at else None
        d['due_date'] = action.due_date.isoformat() if action.due_date else None
        return d

    def _action_from_dict(self, d: Dict[str, Any]) -> ActionItem:
        """Convert dict to ActionItem"""
        return ActionItem(
            id=d['id'],
            description=d['description'],
            priority=d['priority'],
            status=ActionStatus(d['status']),
            category=ActionCategory(d['category']),
            created_at=datetime.fromisoformat(d['created_at']) if d.get('created_at') else datetime.now(),
            started_at=datetime.fromisoformat(d['started_at']) if d.get('started_at') else None,
            completed_at=datetime.fromisoformat(d['completed_at']) if d.get('completed_at') else None,
            due_date=datetime.fromisoformat(d['due_date']) if d.get('due_date') else None,
            context=d.get('context', {}),
            dependencies=set(d.get('dependencies', [])),
            blocks=set(d.get('blocks', [])),
            success=d.get('success'),
            quality_score=d.get('quality_score'),
            completion_time_ms=d.get('completion_time_ms'),
            priority_confidence=d.get('priority_confidence', 0.5),
            predicted_duration_ms=d.get('predicted_duration_ms'),
            actual_vs_predicted_ratio=d.get('actual_vs_predicted_ratio')
        )

    def _model_to_dict(self) -> Dict[str, Any]:
        """Convert PriorityModel to JSON-serializable dict"""
        return {
            "category_priors": dict(self.priority_model.category_priors),
            "avg_completion_times": dict(self.priority_model.avg_completion_times),
            "completion_counts": dict(self.priority_model.completion_counts),
            "priority_buckets": dict(self.priority_model.priority_buckets)
        }

    def _model_from_dict(self, d: Dict[str, Any]) -> PriorityModel:
        """Convert dict to PriorityModel"""
        model = PriorityModel()
        model.category_priors = defaultdict(
            lambda: {"alpha": 1.0, "beta": 1.0},
            d.get("category_priors", {})
        )
        model.avg_completion_times = defaultdict(float, d.get("avg_completion_times", {}))
        model.completion_counts = defaultdict(int, d.get("completion_counts", {}))
        model.priority_buckets = defaultdict(list, d.get("priority_buckets", {}))
        return model


# ============================================================================
# Convenience Functions
# ============================================================================

def create_action_tracker(persist_path: str = "./action_items.json") -> ActionItemTracker:
    """
    Create action item tracker (convenience factory).

    Args:
        persist_path: Path to persistence file

    Returns:
        ActionItemTracker instance
    """
    return ActionItemTracker(persist_path=persist_path)
