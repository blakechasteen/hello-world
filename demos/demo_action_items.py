"""
Demo: Phase 6 Action Items System
==================================

Shows intelligent action tracking with Thompson Sampling priority learning.

This demo illustrates:
1. Adding action items with auto-classification
2. Priority scoring with urgency calculations
3. Auto-extraction from text
4. Lifecycle management (pending → in_progress → completed)
5. Thompson Sampling learning from completions
6. Statistics and analytics

Run: python demos/demo_action_items.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from HoloLoom.recursive.action_items import (
    ActionItemTracker,
    ActionCategory,
    ActionStatus,
    extract_action_items_from_text
)


def print_header(title: str):
    """Print section header"""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


def print_action(action, show_details=True):
    """Pretty print action item"""
    status_emoji = {
        ActionStatus.PENDING: "⏳",
        ActionStatus.IN_PROGRESS: "🔄",
        ActionStatus.COMPLETED: "✅",
        ActionStatus.BLOCKED: "🚫",
        ActionStatus.ARCHIVED: "📦"
    }

    emoji = status_emoji.get(action.status, "")

    print(f"{emoji} [{action.id}] {action.description}")

    if show_details:
        print(f"   Priority: {action.priority:.2f} | " +
              f"Effective: {action.get_effective_priority():.2f} | " +
              f"Category: {action.category.value}")

        if action.due_date:
            print(f"   Due: {action.due_date.strftime('%Y-%m-%d %H:%M')}")

        if action.completed_at:
            duration = action.completion_time_ms / 1000
            print(f"   Completed in: {duration:.1f}s | Quality: {action.quality_score:.2f}")


def demo_basic_usage():
    """Demo 1: Basic action tracking"""
    print_header("Demo 1: Basic Action Tracking")

    # Create tracker (will persist to /tmp)
    tracker = ActionItemTracker(persist_path="/tmp/demo_actions.json")

    # Add some actions
    print("Adding action items...")
    print()

    action1 = tracker.add_action(
        description="Implement semantic caching for repeated queries",
        priority=0.8,
        category=ActionCategory.FEATURE
    )
    print_action(action1)

    action2 = tracker.add_action(
        description="Fix memory leak in background learning thread",
        priority=0.9,
        category=ActionCategory.BUG_FIX,
        due_date=datetime.now() + timedelta(days=1)
    )
    print_action(action2)

    action3 = tracker.add_action(
        description="Refactor policy engine for better modularity",
        priority=0.5,
        category=ActionCategory.REFACTOR
    )
    print_action(action3)

    print()
    print(f"Total actions: {len(tracker.actions)}")


def demo_priority_scheduling():
    """Demo 2: Priority-based scheduling"""
    print_header("Demo 2: Priority-Based Scheduling")

    tracker = ActionItemTracker(persist_path="/tmp/demo_actions.json")

    # Add actions with different priorities and due dates
    tracker.add_action(
        "Write documentation for Phase 6",
        priority=0.4
    )

    tracker.add_action(
        "Urgent: Fix production bug in query routing",
        priority=0.95,
        due_date=datetime.now() + timedelta(hours=2)
    )

    tracker.add_action(
        "Research advanced Thompson Sampling variants",
        priority=0.3,
        category=ActionCategory.RESEARCH
    )

    # Get high-priority pending actions
    print("High-priority pending actions (priority >= 0.7):")
    print()

    pending = tracker.get_pending_actions(min_priority=0.7, limit=5)

    for i, action in enumerate(pending, 1):
        print(f"{i}. ", end="")
        print_action(action, show_details=True)
        print()

    print(f"Total high-priority items: {len(pending)}")


def demo_text_extraction():
    """Demo 3: Auto-extraction from text"""
    print_header("Demo 3: Auto-Extract Action Items from Text")

    sample_text = """
Based on the performance analysis, we need to make several improvements:

TODO: Implement caching layer with 95% hit rate target
Action: Profile the embedding generation pipeline
Fix: Memory leak in scratchpad provenance tracking
Implement: Batch processing for multiple queries
Research: Alternative matryoshka scales for better performance
Test: Load testing with 1000 concurrent queries
Document: API usage examples for new action items system

The caching implementation is particularly urgent since it's blocking
the production deployment.
"""

    print("Input text:")
    print("-" * 80)
    print(sample_text)
    print("-" * 80)
    print()

    # Extract actions
    extracted = extract_action_items_from_text(sample_text)

    print(f"Extracted {len(extracted)} action items:")
    print()

    for i, action_data in enumerate(extracted, 1):
        print(f"{i}. {action_data['description']}")
        print(f"   Category: {action_data['category'].value} | " +
              f"Priority: {action_data['priority']:.2f}")
        print()


def demo_lifecycle_and_learning():
    """Demo 4: Lifecycle management and Thompson Sampling learning"""
    print_header("Demo 4: Lifecycle & Thompson Sampling Learning")

    tracker = ActionItemTracker(persist_path="/tmp/demo_actions.json")

    print("Creating new action...")
    action = tracker.add_action(
        "Optimize embedding generation with ONNX",
        priority=0.7,
        category=ActionCategory.OPTIMIZATION
    )
    print_action(action)
    print()

    print("Starting work...")
    tracker.start_action(action.id)
    action = tracker.actions[action.id]
    print_action(action)
    print()

    print("Completing action (success with high quality)...")
    tracker.complete_action(action.id, success=True, quality_score=0.92)
    action = tracker.actions[action.id]
    print_action(action)
    print()

    print("Thompson Sampling Model Statistics:")
    print(f"  Expected success rate for {ActionCategory.OPTIMIZATION.value}: " +
          f"{tracker.priority_model.get_expected_success_rate(ActionCategory.OPTIMIZATION):.2f}")
    print(f"  Predicted duration for {ActionCategory.OPTIMIZATION.value}: " +
          f"{tracker.priority_model.get_predicted_duration(ActionCategory.OPTIMIZATION) / 1000:.1f}s")
    print()

    # Complete another action in same category
    print("Completing another optimization action...")
    action2 = tracker.add_action(
        "Reduce query latency by 50%",
        priority=0.8,
        category=ActionCategory.OPTIMIZATION
    )
    tracker.start_action(action2.id)
    tracker.complete_action(action2.id, success=True, quality_score=0.88)

    print("Updated model statistics:")
    print(f"  Expected success rate for {ActionCategory.OPTIMIZATION.value}: " +
          f"{tracker.priority_model.get_expected_success_rate(ActionCategory.OPTIMIZATION):.2f}")
    print()

    print("The model learns from completions and improves predictions!")


def demo_statistics():
    """Demo 5: Statistics and analytics"""
    print_header("Demo 5: Statistics & Analytics")

    tracker = ActionItemTracker(persist_path="/tmp/demo_actions.json")

    stats = tracker.get_statistics()

    print("Action Item Statistics:")
    print()
    print(f"Total actions: {stats['total_actions']}")
    print(f"  Pending: {stats['pending']}")
    print(f"  In Progress: {stats['in_progress']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Overdue: {stats['overdue']}")
    print()

    print("Category Breakdown:")
    for category, count in stats['category_breakdown'].items():
        print(f"  {category}: {count}")
    print()

    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Avg Completion Time: {stats['avg_completion_time_ms'] / 1000:.1f}s")
    print(f"Model Accuracy: {stats['model_accuracy']:.1%}")


def demo_priority_adjustment():
    """Demo 6: Intelligent priority adjustment"""
    print_header("Demo 6: Intelligent Priority Adjustment")

    tracker = ActionItemTracker(persist_path="/tmp/demo_actions.json")

    print("The tracker learns which priority levels are well-calibrated.")
    print()

    # Simulate completing actions with different priorities
    print("Completing actions at priority 0.8...")
    for i in range(3):
        action = tracker.add_action(
            f"High priority task {i+1}",
            priority=0.8,
            category=ActionCategory.FEATURE
        )
        tracker.complete_action(action.id, success=True, quality_score=0.9)

    print()
    print("Completing actions at priority 0.3...")
    for i in range(3):
        action = tracker.add_action(
            f"Low priority task {i+1}",
            priority=0.3,
            category=ActionCategory.DOCUMENTATION
        )
        tracker.complete_action(action.id, success=True, quality_score=0.5)

    print()
    print("Priority adjustment suggestions:")
    print(f"  0.8 -> {tracker.priority_model.suggest_priority_adjustment(0.8):.2f} " +
          "(high quality at 0.8, well-calibrated)")
    print(f"  0.3 -> {tracker.priority_model.suggest_priority_adjustment(0.3):.2f} " +
          "(low quality at 0.3, might need adjustment)")
    print()
    print("The model adjusts priorities based on historical quality scores!")


def demo_overdue_tracking():
    """Demo 7: Overdue action tracking"""
    print_header("Demo 7: Overdue Action Tracking")

    tracker = ActionItemTracker(persist_path="/tmp/demo_actions.json")

    # Add overdue action
    overdue_action = tracker.add_action(
        "Fix critical security vulnerability",
        priority=0.95,
        category=ActionCategory.BUG_FIX,
        due_date=datetime.now() - timedelta(days=2)  # 2 days overdue
    )

    # Add upcoming action
    upcoming_action = tracker.add_action(
        "Prepare quarterly report",
        priority=0.6,
        category=ActionCategory.DOCUMENTATION,
        due_date=datetime.now() + timedelta(days=3)
    )

    print("Actions by urgency:")
    print()

    for action in [overdue_action, upcoming_action]:
        urgency = action.get_urgency_score()
        effective = action.get_effective_priority()

        print_action(action, show_details=False)
        print(f"   Base Priority: {action.priority:.2f}")
        print(f"   Urgency Score: {urgency:.2f}")
        print(f"   Effective Priority: {effective:.2f}")

        if action.is_overdue():
            print("   ⚠️  OVERDUE!")

        print()

    overdue = tracker.get_overdue_actions()
    print(f"Total overdue actions: {len(overdue)}")


def main():
    """Run all demos"""
    print()
    print("+" * 80)
    print("Phase 6: Action Items System - Interactive Demo")
    print("+" * 80)

    # Clean slate
    import os
    if os.path.exists("/tmp/demo_actions.json"):
        os.remove("/tmp/demo_actions.json")

    # Run demos
    demo_basic_usage()
    demo_priority_scheduling()
    demo_text_extraction()
    demo_lifecycle_and_learning()
    demo_statistics()
    demo_priority_adjustment()
    demo_overdue_tracking()

    print()
    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("  1. Basic CRUD operations for action items")
    print("  2. Priority-based scheduling with urgency")
    print("  3. Auto-extraction from text patterns")
    print("  4. Lifecycle management (pending -> in_progress -> completed)")
    print("  5. Thompson Sampling learning from completions")
    print("  6. Intelligent priority adjustment")
    print("  7. Overdue action tracking")
    print()
    print("Philosophy: 'Great systems don't just respond - they remember what needs to be done.'")
    print()


if __name__ == "__main__":
    main()
