#!/usr/bin/env python3
"""
SOUS Multi-User Kitchen Management Demo

Demonstrates Phase 1 features:
- Intelligent task assignment based on skills
- DAG-based dependency management
- Real-time collaboration with presence tracking
- Resource scheduling and conflict detection
- Gamification (points, achievements, streaks)
- Multi-user HoloLoom memory integration

Example: Smith Family Thanksgiving Dinner
- 3 household members with different skills
- 15 interconnected tasks with dependencies
- Critical path optimization (12 hours → 5.5 hours)
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

# Import SOUS components
from sous.models.user import UserProfile, Household, ResourceType, AvailabilityWindow
from sous.models.task import Task, TaskState, TaskPriority
from sous.services.task_manager import TaskManager
from sous.services.collaboration_manager import CollaborationManager, PresenceStatus, EventType
from sous.services.gamification import GamificationEngine
from sous.services.hololoom_bridge import SousHoloLoomBridge


def print_section(title: str):
    """Print section header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70 + "\n")


def load_sample_data():
    """Load sample household and task data"""
    data_dir = Path(__file__).parent / "sous" / "data"

    # Load users
    with open(data_dir / "sample_users.json", "r") as f:
        users_data = json.load(f)

    # Create UserProfile objects
    users = {}
    for user_data in users_data["users"]:
        user = UserProfile(
            user_id=user_data["user_id"],
            name=user_data["name"],
            household_id=user_data["household_id"],
            skills=user_data["skills"],
            dietary_restrictions=user_data.get("dietary_restrictions", []),
            favorite_cuisines=user_data.get("favorite_cuisines", []),
            allergies=user_data.get("allergies", []),
            total_points=user_data.get("total_points", 0),
            level=user_data.get("level", 1),
            tasks_completed=user_data.get("tasks_completed", 0),
            recipes_mastered=set(user_data.get("recipes_mastered", []))
        )

        # Parse availability windows
        for window_data in user_data.get("availability", []):
            window = AvailabilityWindow(
                day_of_week=window_data["day_of_week"],
                start_hour=window_data["start_hour"],
                end_hour=window_data["end_hour"]
            )
            user.availability.append(window)

        users[user.user_id] = user

    # Create Household
    household_data = users_data["household"]
    household = Household(
        household_id=household_data["household_id"],
        name=household_data["name"],
        members=[u.user_id for u in users.values()],
        total_meals_cooked=household_data.get("total_meals_cooked", 0),
        waste_prevented_kg=household_data.get("waste_prevented_kg", 0.0),
        food_shared_kg=household_data.get("food_shared_kg", 0.0),
        composted_kg=household_data.get("composted_kg", 0.0),
        carbon_saved_kg=household_data.get("carbon_saved_kg", 0.0)
    )

    # Add resources
    household.add_resource(ResourceType.OVEN)
    household.add_resource(ResourceType.STOVETOP)
    household.add_resource(ResourceType.COUNTER)

    # Load tasks
    with open(data_dir / "sample_tasks.json", "r") as f:
        tasks_data = json.load(f)

    # Create Task objects
    tasks = []
    for task_data in tasks_data["tasks"]:
        task = Task(
            task_id=task_data["task_id"],
            household_id=task_data["household_id"],
            recipe_id=task_data["recipe_id"],
            description=task_data["description"],
            dependencies=task_data.get("dependencies", []),
            blocking=task_data.get("blocking", []),
            resources_needed=[ResourceType(r) for r in task_data.get("resources_needed", [])],
            required_skill=task_data.get("required_skill"),
            min_skill_level=task_data.get("min_skill_level", 1.0),
            estimated_duration_min=task_data.get("estimated_duration_min", 10),
            priority=TaskPriority(task_data.get("priority", 2)),
            notes=task_data.get("notes", "")
        )
        tasks.append(task)

    return users, household, tasks, tasks_data["recipe_name"]


async def demo_task_assignment(users: Dict[str, UserProfile], household: Household, tasks: List[Task]):
    """Demo 1: Intelligent Task Assignment"""
    print_section("DEMO 1: Intelligent Task Assignment")

    # Create task manager
    manager = TaskManager(household, users)
    manager.add_tasks(tasks)

    # Validate DAG
    print("📊 Validating task dependency graph...")
    errors = manager.graph.validate()
    if errors:
        print(f"   ⚠️  Validation errors: {errors}")
        return
    print("   ✅ DAG validated successfully - no cycles detected\n")

    # Show critical path
    critical_path = manager.get_critical_path()
    total_time = sum(t.estimated_duration_min for t in critical_path)
    print(f"⏱️  Critical Path: {len(critical_path)} tasks, {total_time} minutes ({total_time/60:.1f} hours)")
    print("   Path: " + " → ".join([t.task_id for t in critical_path]))
    print()

    # Assign tasks intelligently
    print("🤖 Running intelligent task assignment algorithm...")
    assignments = await manager.assign_tasks_intelligently()

    print(f"\n✅ Assigned {len(assignments)} tasks:\n")

    # Group assignments by user
    by_user = {}
    for assignment in assignments:
        user_id = assignment.assigned_to
        if user_id not in by_user:
            by_user[user_id] = []
        by_user[user_id].append(assignment)

    # Print assignments
    for user_id, user_assignments in by_user.items():
        user = users[user_id]
        total_minutes = sum(a.scheduled_end.timestamp() - a.scheduled_start.timestamp() for a in user_assignments) / 60

        print(f"👤 {user.name} ({len(user_assignments)} tasks, {total_minutes:.0f} minutes total):")
        for assignment in user_assignments:
            print(f"   • {assignment.task_id}: {assignment.reasoning}")
            print(f"     Confidence: {assignment.confidence:.2f}")
            print(f"     Scheduled: {assignment.scheduled_start.strftime('%H:%M')} - {assignment.scheduled_end.strftime('%H:%M')}")
        print()

    # Show dashboard summary
    summary = manager.get_dashboard_summary()
    print(f"📈 Dashboard Summary:")
    print(f"   Total tasks: {summary['total_tasks']}")
    print(f"   Assigned: {summary['total_tasks'] - summary['pending']}")
    print(f"   Estimated completion: {summary['estimated_completion_min']} minutes ({summary['estimated_completion_min']/60:.1f} hours)")
    print(f"   Progress: {summary['progress_percent']}%")
    print()

    return manager


async def demo_collaboration(manager: TaskManager, users: Dict[str, UserProfile]):
    """Demo 2: Real-Time Collaboration"""
    print_section("DEMO 2: Real-Time Collaboration")

    household_id = list(users.values())[0].household_id
    collab_manager = CollaborationManager(household_id)

    # Simulate users joining
    print("👥 Household members joining cooking session...\n")
    for user_id, user in users.items():
        await collab_manager.update_presence(
            user_id,
            PresenceStatus.ONLINE,
            device_id=f"device_{user_id}"
        )
        print(f"   ✅ {user.name} is online")

    print()

    # Show active users
    active_users = collab_manager.get_active_users()
    print(f"📊 Active Users: {len(active_users)}")
    for presence in active_users:
        user = users[presence.user_id]
        print(f"   • {user.name}: {presence.status.value}")
    print()

    # Simulate starting tasks
    print("🔥 Starting coordinated cooking...\n")

    # Alice starts prep_turkey
    alice_id = "alice"
    task_id = "prep_turkey"
    await manager.start_task(task_id)
    await collab_manager.emit_event(
        EventType.TASK_STARTED,
        alice_id,
        {"task_id": task_id, "task_name": "Prep turkey"}
    )
    await collab_manager.update_presence(
        alice_id,
        PresenceStatus.ACTIVE,
        current_task_id=task_id
    )
    print(f"   👤 {users[alice_id].name} started: {task_id}")

    # Bob starts make_stuffing (parallel task)
    bob_id = "bob"
    task_id = "make_stuffing"
    await manager.start_task(task_id)
    await collab_manager.emit_event(
        EventType.TASK_STARTED,
        bob_id,
        {"task_id": task_id, "task_name": "Make stuffing"}
    )
    await collab_manager.update_presence(
        bob_id,
        PresenceStatus.ACTIVE,
        current_task_id=task_id
    )
    print(f"   👤 {users[bob_id].name} started: {task_id}")

    # Carol starts make_cranberry_sauce (another parallel task)
    carol_id = "carol"
    task_id = "make_cranberry_sauce"
    await manager.start_task(task_id)
    await collab_manager.emit_event(
        EventType.TASK_STARTED,
        carol_id,
        {"task_id": task_id, "task_name": "Make cranberry sauce"}
    )
    await collab_manager.update_presence(
        carol_id,
        PresenceStatus.ACTIVE,
        current_task_id=task_id
    )
    print(f"   👤 {users[carol_id].name} started: {task_id}")
    print()

    # Show activity feed
    activity = collab_manager.get_activity_feed(limit=10)
    print(f"📰 Activity Feed ({len(activity)} recent events):")
    for event in activity[:5]:
        user = users[event.user_id]
        print(f"   {event.timestamp.strftime('%H:%M:%S')} - {user.name}: {event.event_type.value}")
        if 'task_name' in event.data:
            print(f"      Task: {event.data['task_name']}")
    print()

    # Show collaboration statistics
    stats = collab_manager.get_statistics()
    print(f"📊 Collaboration Statistics:")
    print(f"   Total events: {stats['total_events']}")
    print(f"   Active users: {stats['active_users']}")
    print(f"   Event breakdown:")
    for event_type, count in stats['event_breakdown'].items():
        print(f"      {event_type}: {count}")
    print()

    return collab_manager


async def demo_gamification(manager: TaskManager, users: Dict[str, UserProfile]):
    """Demo 3: Gamification System"""
    print_section("DEMO 3: Gamification System")

    gamification = GamificationEngine()

    # Simulate completing tasks and earning points/achievements
    print("🎮 Simulating task completion and gamification...\n")

    # Alice completes prep_turkey
    alice_id = "alice"
    alice = users[alice_id]
    await manager.complete_task("prep_turkey", notes="Turkey prepped perfectly!")

    points = gamification.award_points(alice_id, alice, "task_completion", custom_points=15)
    print(f"   👤 {alice.name} completed 'prep_turkey'")
    print(f"      +{points} points (total: {alice.total_points}, level: {alice.level})")

    # Check for achievements
    user_stats = {
        "tasks_completed": alice.tasks_completed,
        "recipes_mastered": len(alice.recipes_mastered),
        "waste_prevented_kg": 0
    }
    achievements = gamification.check_achievements(alice_id, user_stats)

    if achievements:
        print(f"      🏆 Achievements unlocked:")
        for achievement in achievements:
            print(f"         • {achievement.icon} {achievement.name}: {achievement.description}")
    print()

    # Bob completes make_stuffing
    bob_id = "bob"
    bob = users[bob_id]
    await manager.complete_task("make_stuffing", notes="Stuffing tastes great!")

    points = gamification.award_points(bob_id, bob, "task_completion", custom_points=20)
    print(f"   👤 {bob.name} completed 'make_stuffing'")
    print(f"      +{points} points (total: {bob.total_points}, level: {bob.level})")
    print()

    # Carol completes make_cranberry_sauce
    carol_id = "carol"
    carol = users[carol_id]
    await manager.complete_task("make_cranberry_sauce", notes="Perfect cranberry sauce!")

    points = gamification.award_points(carol_id, carol, "task_completion", custom_points=12)
    print(f"   👤 {carol.name} completed 'make_cranberry_sauce'")
    print(f"      +{points} points (total: {carol.total_points}, level: {carol.level})")
    print()

    # Update streaks
    print("🔥 Updating cooking streaks...\n")
    for user_id, user in users.items():
        streak_achievements = gamification.update_streak(user_id)
        streak = gamification.streaks[user_id]
        print(f"   👤 {user.name}: {streak.current_streak} day streak (longest: {streak.longest_streak})")

        if streak_achievements:
            print(f"      🏆 Streak achievements:")
            for achievement in streak_achievements:
                print(f"         • {achievement.icon} {achievement.name}")
    print()

    # Show user leaderboard
    print("🏆 Household Leaderboard:")
    leaderboard = manager.get_user_leaderboard()
    for i, entry in enumerate(leaderboard, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")
        print(f"   {medal} {entry['name']}: {entry['points']} points (Level {entry['level']})")
        print(f"      Tasks: {entry['tasks_completed']}/{entry['tasks_assigned']} completed")
    print()

    return gamification


async def demo_hololoom_integration(users: Dict[str, UserProfile], household: Household):
    """Demo 4: HoloLoom Memory Integration"""
    print_section("DEMO 4: HoloLoom Memory Integration")

    print("🧠 Initializing HoloLoom bridge...\n")

    # Check if HoloLoom is available
    from sous.services.hololoom_bridge import HOLOLOOM_AVAILABLE

    if not HOLOLOOM_AVAILABLE:
        print("   ⚠️  HoloLoom not available - skipping memory demo")
        print("   (This is okay! SOUS works without HoloLoom)")
        return

    async with SousHoloLoomBridge(household_id=household.household_id) as bridge:
        # Store cooking experiences with user attribution
        print("💾 Storing multi-user cooking memories...\n")

        # Alice shares a substitution tip
        alice_id = "alice"
        result = await bridge.remember_ingredient_substitution(
            original="butter",
            substitute="coconut oil",
            recipe="Thanksgiving stuffing",
            worked_well=True,
            notes="Worked great for gluten-free version",
            user_id=alice_id
        )
        print(f"   ✅ {users[alice_id].name}'s substitution tip stored")
        print(f"      Memory ID: {result['memory_id']}")
        print()

        # Bob shares a cooking session
        bob_id = "bob"
        result = await bridge.remember_cooking_session(
            recipes_made=["Turkey", "Stuffing"],
            success=True,
            notes="Turkey came out perfectly juicy!",
            user_id=bob_id
        )
        print(f"   ✅ {users[bob_id].name}'s cooking session stored")
        print(f"      Memory ID: {result['memory_id']}")
        print()

        # Query for substitution suggestions
        print("🔍 Querying HoloLoom RAG for substitution suggestions...\n")
        result = await bridge.suggest_substitutions(
            missing_ingredient="milk",
            recipe_context="Thanksgiving dinner sides"
        )

        if result['status'] == 'success':
            print(f"   Query: What can substitute for milk?")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Response: {result['response'][:200]}...")
        print()

        # Check bridge status
        status = bridge.get_status()
        print(f"📊 HoloLoom Integration Status:")
        print(f"   Memory active: {status['memory_active']}")
        print(f"   RAG active: {status['rag_active']}")
        print(f"   Agentic active: {status['agentic_active']}")
        print(f"   Household: {household.name}")
        print()


async def main():
    """Run complete multi-user demo"""
    print("\n" + "="*70)
    print(" SOUS Multi-User Kitchen Management System Demo")
    print(" Phase 1: Foundation - Intelligent Coordination & Gamification")
    print("="*70)

    # Load sample data
    print("\n📥 Loading sample data...")
    users, household, tasks, recipe_name = load_sample_data()
    print(f"   ✅ Loaded household: {household.name}")
    print(f"   ✅ Loaded {len(users)} users: {', '.join([u.name for u in users.values()])}")
    print(f"   ✅ Loaded {len(tasks)} tasks for: {recipe_name}")

    # Run demos
    manager = await demo_task_assignment(users, household, tasks)
    await demo_collaboration(manager, users)
    await demo_gamification(manager, users)
    await demo_hololoom_integration(users, household)

    # Final summary
    print_section("SUMMARY")

    summary = manager.get_dashboard_summary()

    print("📊 Session Statistics:")
    print(f"   Recipe: {recipe_name}")
    print(f"   Total tasks: {summary['total_tasks']}")
    print(f"   Completed: {summary['completed']}")
    print(f"   In progress: {summary['in_progress']}")
    print(f"   Pending: {summary['pending']}")
    print(f"   Time estimate: {summary['estimated_completion_min']/60:.1f} hours")
    print(f"   Progress: {summary['progress_percent']}%")
    print()

    print("🏆 Household Achievements:")
    print(f"   Total meals: {household.total_meals_cooked}")
    print(f"   Waste prevented: {household.waste_prevented_kg:.1f} kg")
    print(f"   Food shared: {household.food_shared_kg:.1f} kg")
    print(f"   Composted: {household.composted_kg:.1f} kg")
    print(f"   Carbon saved: {household.carbon_saved_kg:.1f} kg")
    print()

    print("✨ Phase 1 Demo Complete!")
    print()
    print("Key Features Demonstrated:")
    print("   ✅ Skill-based task assignment with confidence scoring")
    print("   ✅ DAG dependency management with critical path analysis")
    print("   ✅ Resource scheduling with conflict detection")
    print("   ✅ Real-time collaboration with presence tracking")
    print("   ✅ Gamification (points, achievements, streaks, leaderboards)")
    print("   ✅ Multi-user HoloLoom memory integration")
    print()
    print("Next Steps:")
    print("   → Phase 2: Voice Assistant & Mobile App")
    print("   → Phase 3: Garden Integration & Harvest Tracking")
    print("   → Phase 4: Local Farm Network & Food Sharing")
    print("   → Phase 5: Analytics & Food System Impact Metrics")
    print()


if __name__ == "__main__":
    asyncio.run(main())
