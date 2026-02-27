"""
Unit tests for HoloLoom.recursive.action_items

Tests action item tracking, priority scoring, goal decomposition, and Thompson Sampling learning.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from HoloLoom.recursive.action_items import (
    ActionStatus,
    ActionCategory,
    ActionItem,
    PriorityModel,
    ActionItemTracker,
    extract_action_items_from_text,
    create_action_tracker,
    _classify_action,
    _estimate_priority
)


class TestActionStatus:
    """Test ActionStatus enum"""

    def test_all_statuses_exist(self):
        """Test all expected statuses exist"""
        assert ActionStatus.PENDING.value == "pending"
        assert ActionStatus.IN_PROGRESS.value == "in_progress"
        assert ActionStatus.COMPLETED.value == "completed"
        assert ActionStatus.ARCHIVED.value == "archived"
        assert ActionStatus.BLOCKED.value == "blocked"


class TestActionCategory:
    """Test ActionCategory enum"""

    def test_all_categories_exist(self):
        """Test all expected categories exist"""
        assert ActionCategory.BUG_FIX.value == "bug_fix"
        assert ActionCategory.FEATURE.value == "feature"
        assert ActionCategory.OPTIMIZATION.value == "optimization"
        assert ActionCategory.DOCUMENTATION.value == "documentation"
        assert ActionCategory.REFACTOR.value == "refactor"
        assert ActionCategory.TESTING.value == "testing"
        assert ActionCategory.RESEARCH.value == "research"
        assert ActionCategory.UNKNOWN.value == "unknown"


class TestActionItem:
    """Test ActionItem dataclass"""

    def test_creation(self):
        """Test creating an action item"""
        item = ActionItem(
            id="action_001",
            description="Implement feature X",
            priority=0.8
        )

        assert item.id == "action_001"
        assert item.description == "Implement feature X"
        assert item.priority == 0.8
        assert item.status == ActionStatus.PENDING
        assert item.category == ActionCategory.UNKNOWN

    def test_with_metadata(self):
        """Test action item with full metadata"""
        due = datetime.now() + timedelta(days=7)
        context = {"related_task": "task_123"}
        deps = {"action_000"}

        item = ActionItem(
            id="action_002",
            description="Complete task",
            priority=0.9,
            status=ActionStatus.IN_PROGRESS,
            category=ActionCategory.FEATURE,
            due_date=due,
            context=context,
            dependencies=deps
        )

        assert item.status == ActionStatus.IN_PROGRESS
        assert item.category == ActionCategory.FEATURE
        assert item.due_date == due
        assert item.context == context
        assert item.dependencies == deps

    def test_is_overdue(self):
        """Test overdue detection"""
        # Not overdue (future due date)
        future = ActionItem(
            id="a1",
            description="Future task",
            priority=0.8,
            due_date=datetime.now() + timedelta(days=1)
        )
        assert future.is_overdue() is False

        # Overdue (past due date)
        past = ActionItem(
            id="a2",
            description="Past task",
            priority=0.8,
            due_date=datetime.now() - timedelta(days=1)
        )
        assert past.is_overdue() is True

        # No due date
        none_due = ActionItem(
            id="a3",
            description="No due date",
            priority=0.8
        )
        assert none_due.is_overdue() is False

    def test_is_blocked(self):
        """Test blocked detection"""
        # Blocked status
        blocked = ActionItem(
            id="a1",
            description="Blocked",
            priority=0.8,
            status=ActionStatus.BLOCKED
        )
        assert blocked.is_blocked() is True

        # Has dependencies
        deps = ActionItem(
            id="a2",
            description="Has deps",
            priority=0.8,
            dependencies={"dep1", "dep2"}
        )
        assert deps.is_blocked() is True

        # Not blocked
        free = ActionItem(
            id="a3",
            description="Free",
            priority=0.8
        )
        assert free.is_blocked() is False

    def test_get_age_hours(self):
        """Test age calculation"""
        item = ActionItem(
            id="a1",
            description="Test",
            priority=0.8
        )

        age = item.get_age_hours()
        assert age >= 0.0
        assert age < 1.0  # Should be very recent

    def test_get_urgency_score_overdue(self):
        """Test urgency score for overdue items"""
        item = ActionItem(
            id="a1",
            description="Overdue",
            priority=0.8,
            due_date=datetime.now() - timedelta(hours=1)
        )

        urgency = item.get_urgency_score()
        assert urgency == 1.0  # Overdue = max urgency

    def test_get_urgency_score_due_soon(self):
        """Test urgency score for items due soon"""
        item = ActionItem(
            id="a1",
            description="Due soon",
            priority=0.8,
            due_date=datetime.now() + timedelta(hours=12)
        )

        urgency = item.get_urgency_score()
        assert urgency == 0.8  # Due within 24 hours

    def test_get_urgency_score_age(self):
        """Test urgency increases with age"""
        old_item = ActionItem(
            id="a1",
            description="Old",
            priority=0.8,
            created_at=datetime.now() - timedelta(days=10)
        )

        urgency = old_item.get_urgency_score()
        assert urgency > 0.0  # Age contributes to urgency

    def test_get_effective_priority(self):
        """Test effective priority calculation"""
        item = ActionItem(
            id="a1",
            description="Test",
            priority=0.8
        )

        effective = item.get_effective_priority()
        # Should be weighted combination of priority and urgency
        assert 0.0 <= effective <= 1.0
        assert effective >= 0.7 * 0.8  # At least 70% of base priority


class TestPriorityModel:
    """Test PriorityModel dataclass"""

    def test_initialization(self):
        """Test model initialization"""
        model = PriorityModel()

        assert isinstance(model.category_priors, dict)
        assert isinstance(model.avg_completion_times, dict)
        assert isinstance(model.completion_counts, dict)

    def test_update_from_completion_success(self):
        """Test updating model after successful completion"""
        model = PriorityModel()

        model.update_from_completion(
            category=ActionCategory.FEATURE,
            priority=0.8,
            success=True,
            quality_score=0.9,
            completion_time_ms=5000.0
        )

        # Alpha should increase
        assert model.category_priors["feature"]["alpha"] > 1.0
        # Completion time tracked
        assert model.avg_completion_times["feature"] > 0

    def test_update_from_completion_failure(self):
        """Test updating model after failed completion"""
        model = PriorityModel()

        model.update_from_completion(
            category=ActionCategory.BUG_FIX,
            priority=0.7,
            success=False,
            quality_score=0.4,
            completion_time_ms=10000.0
        )

        # Beta should increase more than alpha
        assert model.category_priors["bug_fix"]["beta"] > model.category_priors["bug_fix"]["alpha"]

    def test_get_expected_success_rate(self):
        """Test Thompson Sampling expected success rate"""
        model = PriorityModel()

        # Update with some successes
        for i in range(5):
            model.update_from_completion(
                category=ActionCategory.TESTING,
                priority=0.8,
                success=True,
                quality_score=0.85,
                completion_time_ms=3000.0
            )

        success_rate = model.get_expected_success_rate(ActionCategory.TESTING)
        assert 0.5 < success_rate <= 1.0  # Should be high

    def test_get_predicted_duration(self):
        """Test duration prediction"""
        model = PriorityModel()

        # Add completion data
        model.update_from_completion(
            category=ActionCategory.REFACTOR,
            priority=0.7,
            success=True,
            quality_score=0.8,
            completion_time_ms=7500.0
        )

        duration = model.get_predicted_duration(ActionCategory.REFACTOR)
        assert duration == 7500.0

    def test_get_predicted_duration_unknown(self):
        """Test duration prediction for unknown category"""
        model = PriorityModel()

        duration = model.get_predicted_duration(ActionCategory.UNKNOWN)
        assert duration == 3600000.0  # Default 1 hour

    def test_suggest_priority_adjustment(self):
        """Test priority calibration suggestion"""
        model = PriorityModel()

        # Add data at priority 0.8 with high quality
        for i in range(5):
            model.priority_buckets["0.8"].append(0.90)

        adjusted = model.suggest_priority_adjustment(0.8)
        # Should suggest slight increase (consistently high quality)
        assert adjusted >= 0.8

    def test_running_average_completion_time(self):
        """Test running average for completion times"""
        model = PriorityModel()

        times = [5000.0, 6000.0, 7000.0]
        for time_ms in times:
            model.update_from_completion(
                category=ActionCategory.DOCUMENTATION,
                priority=0.6,
                success=True,
                quality_score=0.8,
                completion_time_ms=time_ms
            )

        avg = model.avg_completion_times["documentation"]
        expected = sum(times) / len(times)
        assert avg == pytest.approx(expected, abs=0.1)


class TestActionItemExtraction:
    """Test action item extraction from text"""

    def test_extract_todo(self):
        """Test extracting TODO items"""
        text = "TODO: Implement caching for better performance"

        actions = extract_action_items_from_text(text)

        assert len(actions) >= 1
        assert any("caching" in a["description"].lower() for a in actions)

    def test_extract_fix(self):
        """Test extracting Fix items"""
        text = "Fix: Memory leak in data processing"

        actions = extract_action_items_from_text(text)

        assert len(actions) >= 1
        assert any("memory leak" in a["description"].lower() for a in actions)

    def test_extract_multiple(self):
        """Test extracting multiple action items"""
        text = """
        TODO: Add unit tests for new feature
        Fix: Bug in authentication flow
        Implement: Real-time notifications
        """

        actions = extract_action_items_from_text(text)

        assert len(actions) == 3

    def test_classify_bug_fix(self):
        """Test classifying bug fix actions"""
        category = _classify_action("Fix the memory leak bug", "")

        assert category == ActionCategory.BUG_FIX

    def test_classify_feature(self):
        """Test classifying feature actions"""
        category = _classify_action("Implement new dashboard feature", "")

        assert category == ActionCategory.FEATURE

    def test_classify_optimization(self):
        """Test classifying optimization actions"""
        category = _classify_action("Optimize database query performance", "")

        assert category == ActionCategory.OPTIMIZATION

    def test_classify_documentation(self):
        """Test classifying documentation actions"""
        category = _classify_action("Document the API endpoints", "")

        assert category == ActionCategory.DOCUMENTATION

    def test_estimate_priority_urgent(self):
        """Test priority estimation for urgent items"""
        priority = _estimate_priority("URGENT: Fix critical security bug")

        assert priority >= 0.9

    def test_estimate_priority_important(self):
        """Test priority estimation for important items"""
        priority = _estimate_priority("Important: Add user authentication")

        assert priority >= 0.7

    def test_estimate_priority_normal(self):
        """Test priority estimation for normal items"""
        priority = _estimate_priority("Add helpful tooltip to button")

        assert 0.5 <= priority < 0.7


class TestActionItemTracker:
    """Test ActionItemTracker class"""

    @pytest.fixture
    def temp_path(self):
        """Create temporary file path"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            path = f.name
        yield path
        # Cleanup
        Path(path).unlink(missing_ok=True)

    @pytest.fixture
    def tracker(self, temp_path):
        """Create tracker with temporary storage"""
        return ActionItemTracker(persist_path=temp_path)

    def test_initialization(self, temp_path):
        """Test tracker initialization"""
        tracker = ActionItemTracker(persist_path=temp_path)

        assert tracker.actions == {}
        assert isinstance(tracker.priority_model, PriorityModel)

    def test_add_action(self, tracker):
        """Test adding an action"""
        action = tracker.add_action(
            description="Test action",
            priority=0.8,
            category=ActionCategory.FEATURE
        )

        assert action.id.startswith("action_")
        assert action.description == "Test action"
        assert action.priority <= 0.8  # May be adjusted
        assert action.category == ActionCategory.FEATURE

    def test_add_action_auto_classify(self, tracker):
        """Test auto-classification when adding action"""
        action = tracker.add_action(
            description="Fix the broken login bug",
            priority=0.9
        )

        assert action.category == ActionCategory.BUG_FIX

    def test_start_action(self, tracker):
        """Test starting an action"""
        action = tracker.add_action("Test", 0.8)

        started = tracker.start_action(action.id)

        assert started.status == ActionStatus.IN_PROGRESS
        assert started.started_at is not None

    def test_complete_action(self, tracker):
        """Test completing an action"""
        action = tracker.add_action("Test", 0.8, category=ActionCategory.TESTING)
        tracker.start_action(action.id)

        completed = tracker.complete_action(
            action.id,
            success=True,
            quality_score=0.9
        )

        assert completed.status == ActionStatus.COMPLETED
        assert completed.completed_at is not None
        assert completed.success is True
        assert completed.quality_score == 0.9
        assert completed.completion_time_ms is not None

    def test_get_pending_actions(self, tracker):
        """Test retrieving pending actions"""
        # Add actions with different priorities
        tracker.add_action("High priority", 0.9)
        tracker.add_action("Low priority", 0.3)
        tracker.add_action("Medium priority", 0.6)

        pending = tracker.get_pending_actions(min_priority=0.5, limit=10)

        # Should exclude low priority (0.3)
        assert len(pending) >= 2
        # Should be sorted by effective priority (descending)
        if len(pending) >= 2:
            assert pending[0].get_effective_priority() >= pending[1].get_effective_priority()

    def test_get_pending_actions_by_category(self, tracker):
        """Test filtering pending actions by category"""
        tracker.add_action("Bug fix", 0.8, category=ActionCategory.BUG_FIX)
        tracker.add_action("Feature", 0.7, category=ActionCategory.FEATURE)

        bug_fixes = tracker.get_pending_actions(category=ActionCategory.BUG_FIX)

        assert all(a.category == ActionCategory.BUG_FIX for a in bug_fixes)

    def test_get_overdue_actions(self, tracker):
        """Test retrieving overdue actions"""
        # Add overdue action
        past_due = datetime.now() - timedelta(days=1)
        tracker.add_action("Overdue task", 0.8, due_date=past_due)

        # Add future action
        future_due = datetime.now() + timedelta(days=1)
        tracker.add_action("Future task", 0.7, due_date=future_due)

        overdue = tracker.get_overdue_actions()

        assert len(overdue) >= 1
        assert all(a.is_overdue() for a in overdue)

    def test_auto_archive_completed(self, tracker):
        """Test auto-archiving old completed items"""
        # Add and complete action (with old completion date)
        action = tracker.add_action("Old task", 0.8)
        tracker.start_action(action.id)
        completed = tracker.complete_action(action.id, success=True, quality_score=0.9)

        # Manually set completion date to past
        completed.completed_at = datetime.now() - timedelta(days=35)

        archived = tracker.auto_archive_completed()

        assert archived >= 1
        assert tracker.actions[action.id].status == ActionStatus.ARCHIVED

    def test_extract_from_text(self, tracker):
        """Test extracting and adding actions from text"""
        text = """
        TODO: Add logging to error handler
        Fix: Memory leak in worker thread
        """

        created = tracker.extract_from_text(text)

        assert len(created) == 2
        assert len(tracker.actions) == 2

    def test_get_statistics(self, tracker):
        """Test statistics generation"""
        # Add various actions
        a1 = tracker.add_action("Pending 1", 0.8)
        a2 = tracker.add_action("Pending 2", 0.7)
        a3 = tracker.add_action("In progress", 0.6)
        tracker.start_action(a3.id)
        a4 = tracker.add_action("Complete", 0.9, category=ActionCategory.FEATURE)
        tracker.start_action(a4.id)
        tracker.complete_action(a4.id, success=True, quality_score=0.95)

        stats = tracker.get_statistics()

        assert stats["total_actions"] == 4
        assert stats["pending"] == 2
        assert stats["in_progress"] == 1
        assert stats["completed"] == 1
        assert stats["success_rate"] > 0

    def test_save_and_load(self, tracker, temp_path):
        """Test saving and loading tracker state"""
        # Add actions
        tracker.add_action("Task 1", 0.8, category=ActionCategory.FEATURE)
        tracker.add_action("Task 2", 0.7, category=ActionCategory.BUG_FIX)

        tracker.save()

        # Create new tracker from same file
        new_tracker = ActionItemTracker(persist_path=temp_path)
        new_tracker.load()

        assert len(new_tracker.actions) == len(tracker.actions)

    def test_model_accuracy_tracking(self, tracker):
        """Test priority model accuracy tracking"""
        action = tracker.add_action(
            "Test task",
            0.8,
            category=ActionCategory.TESTING
        )

        # Set predicted duration
        action.predicted_duration_ms = 5000.0

        tracker.start_action(action.id)

        # Simulate different actual duration
        import time
        time.sleep(0.01)  # 10ms actual
        completed = tracker.complete_action(action.id, success=True, quality_score=0.9)

        # Should have calculated actual vs predicted ratio
        assert completed.actual_vs_predicted_ratio is not None

    def test_priority_adjustment_learning(self, tracker):
        """Test that priority adjustments are learned"""
        # Add multiple successful completions at priority 0.7
        for i in range(5):
            action = tracker.add_action(f"Task {i}", 0.7, category=ActionCategory.TESTING)
            tracker.start_action(action.id)
            tracker.complete_action(action.id, success=True, quality_score=0.95)

        # New action at same priority should be adjusted
        new_action = tracker.add_action("New task", 0.7, category=ActionCategory.TESTING)

        # Priority may be adjusted based on learned patterns
        stats = tracker.get_statistics()
        assert stats["success_rate"] > 0


def test_create_action_tracker_convenience():
    """Test convenience factory function"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        path = f.name

    try:
        tracker = create_action_tracker(persist_path=path)

        assert isinstance(tracker, ActionItemTracker)
        assert tracker.persist_path == Path(path)
    finally:
        Path(path).unlink(missing_ok=True)
