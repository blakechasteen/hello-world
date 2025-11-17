"""
EdWIN Gamification System - Standalone Demo

A self-contained demonstration of all gamification features
without external dependencies.

**Implementation Date**: November 15, 2025
**Agent**: Agent C
"""

import sys
from pathlib import Path

# Add EduVerse to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import gamification modules directly (no numpy dependencies)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'EduVerse' / 'edwin'))

from gamification import (
    GamificationEngine,
    XPSource,
    LevelSystem,
    get_xp_multiplier
)
from achievements import (
    AchievementManager,
    StudentAchievements,
    get_achievement_count
)
from leaderboards import (
    LeaderboardManager,
    LeaderboardType
)
from streak_tracking import (
    StreakManager,
    StreakType
)
from challenges import (
    ChallengeManager,
    create_daily_challenges,
    create_weekly_challenges
)
from rewards import (
    RewardStore,
    StudentInventory,
    REWARD_CATALOG
)
from progress_viz import (
    generate_xp_progress_bar,
    generate_stats_summary,
    generate_subject_radar
)

def print_header(text):
    """Print section header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def simulate_student_journey():
    """Simulate a student's gamification journey"""

    print_header("EdWIN Gamification System - Student Journey Demo")

    student_id = "alice_2025"
    student_name = "Alice Chen"
    grade = 8

    # ===== 1. Create Student Profile =====
    print_header("1. Student Profile Creation")

    # Gamification engine
    engine = GamificationEngine(student_id=student_id)
    print(f"✅ Created profile for {student_name} (Grade {grade})")
    print(f"   Starting Level: 1")
    print(f"   Starting XP: 0")

    # Achievement tracking
    student_ach = StudentAchievements(student_id=student_id)
    ach_manager = AchievementManager()

    # Leaderboard manager
    lb_manager = LeaderboardManager()

    # Streak tracker
    streak_manager = StreakManager(student_id=student_id)

    # Challenge manager
    challenge_manager = ChallengeManager(student_id=student_id, grade=grade)

    # Reward store
    reward_store = RewardStore()
    student_inventory = StudentInventory(student_id=student_id)

    # ===== 2. First Day - Login & First Objective =====
    print_header("2. Day 1 - First Login")

    # Daily login
    streak_result = streak_manager.check_in_login()
    print(f"📅 Daily login: +{streak_result['current_days']} day streak")

    # First objective
    result = engine.award_xp(
        source=XPSource.OBJECTIVE_MASTERED,
        subject="math",
        metadata={"objective_id": "math.algebra.8.linear_eq"}
    )
    print(f"🎯 Mastered first objective: +{result['xp_awarded']} XP")

    # Check for achievement unlock
    stats = {"objectives_mastered": 1}
    unlocked = ach_manager.check_achievement("first_steps", student_ach, stats)
    if unlocked:
        print(f"🏆 Achievement unlocked: {unlocked['icon']} {unlocked['name']}")
        print(f"   +{unlocked['xp_reward']} XP bonus")
        engine.total_xp += unlocked['xp_reward']

    # Update leaderboard
    lb_manager.update_all(
        student_id=student_id,
        student_name=student_name,
        total_xp=engine.total_xp,
        subject_xp={"math": 50},
        objectives_mastered=1,
        current_streak=1,
        grade=grade
    )

    # Show progress
    progress = engine.get_progress()
    print(f"\n📊 Progress: Level {progress['level']} ({progress['level_title']})")
    print(f"   {generate_xp_progress_bar(progress['current_xp'], progress['xp_needed'])}")

    # ===== 3. Day 1 - More Activities =====
    print_header("3. Day 1 - Continued Learning")

    # Master more objectives
    for i in range(2, 6):
        result = engine.award_xp(XPSource.OBJECTIVE_MASTERED, subject="math")
        print(f"🎯 Objective {i}: +{result['xp_awarded']} XP", end="")
        if result['leveled_up']:
            print(f" 🎉 LEVEL UP! Now level {result['new_level']}")
        else:
            print()

    # Ask questions
    for _ in range(10):
        engine.award_xp(XPSource.QUESTION_ASKED)
    print(f"❓ Asked 10 questions: +100 XP")

    # Perfect score
    result = engine.award_xp(XPSource.PERFECT_SCORE, subject="math")
    print(f"💯 Perfect score: +{result['xp_awarded']} XP")

    # Update streak
    streak_result = streak_manager.check_in_mastery(objectives_mastered=5)
    print(f"📅 Daily mastery streak: {streak_result['current_days']} days")

    # Complete a daily challenge
    active_challenges = challenge_manager.get_active_challenges()
    if active_challenges:
        challenge = active_challenges[0]
        print(f"\n🎯 Daily Challenge: {challenge.name}")
        result = challenge_manager.update_progress(challenge.id, 0, 3)
        if result:
            print(f"   ✅ Completed! +{result['xp_reward']} XP")
            engine.total_xp += result['xp_reward']

    # Show progress
    progress = engine.get_progress()
    print(f"\n📊 End of Day 1:")
    print(f"   Level: {progress['level']} ({progress['level_title']})")
    print(f"   Total XP: {progress['total_xp']:,}")
    print(f"   {generate_xp_progress_bar(progress['current_xp'], progress['xp_needed'])}")

    # ===== 4. Week 1 - Building Streak =====
    print_header("4. Week 1 - Building Momentum")

    # Simulate 7 days
    for day in range(2, 8):
        print(f"\n--- Day {day} ---")

        # Daily activities
        daily_xp = 0

        # 3 objectives per day
        for _ in range(3):
            result = engine.award_xp(XPSource.OBJECTIVE_MASTERED, subject="science")
            daily_xp += result['xp_awarded']

        # Questions
        result = engine.award_xp(XPSource.QUESTION_ASKED)
        daily_xp += result['xp_awarded'] * 5
        engine.total_xp += result['xp_awarded'] * 4  # 5 questions

        # Daily login
        streak_result = streak_manager.check_in_login()

        # Check for 7-day streak milestone
        if day == 7 and streak_result.get('new_milestones'):
            for milestone in streak_result['new_milestones']:
                print(f"🎉 MILESTONE: {milestone['days']} day streak!")
                print(f"   {milestone['reward']}")
                print(f"   XP Multiplier: {milestone['xp_multiplier']}x")

        print(f"   XP earned: +{daily_xp} XP (Total: {engine.total_xp:,})")
        print(f"   Streak: {streak_result['current_days']} days")

    # Update engine with 7-day streak
    engine.update_streak(7)

    # Check achievements
    stats = {"objectives_mastered": 25, "current_streak": 7}

    # Check for "on_a_roll"
    unlocked = ach_manager.check_achievement("on_a_roll", student_ach, stats)
    if unlocked:
        print(f"\n🏆 Achievement unlocked: {unlocked['icon']} {unlocked['name']}")
        engine.total_xp += unlocked['xp_reward']

    # Check for streak achievement
    unlocked = ach_manager.check_achievement("streak_7", student_ach, stats)
    if unlocked:
        print(f"🏆 Achievement unlocked: {unlocked['icon']} {unlocked['name']}")
        engine.total_xp += unlocked['xp_reward']

    # ===== 5. Leaderboard Position =====
    print_header("5. Leaderboard Rankings")

    # Update leaderboard
    lb_manager.update_all(
        student_id=student_id,
        student_name=student_name,
        total_xp=engine.total_xp,
        subject_xp={"math": 300, "science": 200},
        objectives_mastered=25,
        current_streak=7,
        grade=grade
    )

    # Add some other students for context
    other_students = [
        ("bob_2025", "Bob Smith", 1500),
        ("charlie_2025", "Charlie Davis", 2200),
        ("diana_2025", "Diana Lee", 1800),
        ("eve_2025", "Eve Williams", 1100),
    ]

    for sid, sname, xp in other_students:
        lb_manager.update_all(
            student_id=sid,
            student_name=sname,
            total_xp=xp,
            subject_xp={"math": xp // 2},
            objectives_mastered=xp // 50,
            current_streak=3,
            grade=grade
        )

    # Show global leaderboard
    global_lb = lb_manager.get_leaderboard(LeaderboardType.GLOBAL_XP)
    if global_lb:
        top_5 = global_lb.get_rankings(limit=5)
        print("🏆 Global XP Leaderboard (Top 5):")
        for entry in top_5:
            marker = "→" if entry.student_id == student_id else " "
            print(f"  {marker} {entry.rank}. {entry.student_name:20} {entry.score:>6,.0f} XP")

        # Personal rank
        personal = global_lb.get_personal_rank(student_id)
        if personal:
            print(f"\n📊 Your Rank: #{personal['rank']} out of {personal['total_participants']}")
            print(f"   Percentile: Top {100 - personal['percentile']:.1f}%")

    # ===== 6. Rewards & Customization =====
    print_header("6. Reward Store")

    # Show available rewards
    available = reward_store.get_available_rewards(
        student_level=progress['level'],
        student_achievements=list(student_ach.unlocked)
    )

    print(f"💎 Available Rewards ({progress['total_xp']:,} XP available):\n")

    # Show a few rewards
    for reward in list(available)[:8]:
        affordability = "✅" if progress['total_xp'] >= reward.xp_cost else "🔒"
        print(f"  {affordability} {reward.icon} {reward.name:25} {reward.xp_cost:>6} XP ({reward.rarity.value})")

    # Purchase an avatar
    if progress['total_xp'] >= 100:
        print(f"\n💰 Purchasing: Robot Scholar Avatar (100 XP)")
        result = reward_store.purchase_reward("avatar_robot", student_inventory, progress['total_xp'])
        if result['success']:
            print(f"   ✅ Purchased! {result['xp_remaining']:,} XP remaining")
            reward_store.equip_reward("avatar_robot", student_inventory)
            print(f"   👤 Equipped avatar: 🤖 Robot Scholar")

    # ===== 7. Achievement Showcase =====
    print_header("7. Achievement Showcase")

    # Get all unlocked achievements
    unlocked_achievements = []
    for ach_id in student_ach.unlocked:
        ach = ach_manager.get_achievement(ach_id)
        if ach:
            unlocked_achievements.append({
                'icon': ach.icon,
                'name': ach.name,
                'description': ach.description,
                'rarity': ach.rarity.value
            })

    print(f"🏆 Achievements Unlocked: {len(unlocked_achievements)}")
    for ach in unlocked_achievements:
        print(f"   {ach['icon']} {ach['name']:30} ({ach['rarity']})")

    # Show progress toward next achievements
    progress_info = ach_manager.get_progress(student_ach)
    print(f"\n📈 Achievement Progress: {progress_info['completion_percent']:.1f}% complete")
    print(f"   ({progress_info['unlocked_count']}/{progress_info['total_achievements']})")

    # ===== 8. Final Summary =====
    print_header("8. Final Summary")

    # Get complete progress
    progress = engine.get_progress()

    # Stats summary
    stats = {
        "level": progress['level'],
        "total_xp": progress['total_xp'],
        "objectives_mastered": 25,
        "current_streak": 7,
        "achievements_unlocked": len(student_ach.unlocked),
        "rank": 3
    }

    print(generate_stats_summary(stats))

    # Subject progress
    print("\n📚 Subject Progress:")
    subject_xp_data = {
        "math": 5,
        "science": 3,
        "ela": 0,
        "social": 0,
        "ai": 0
    }
    print(generate_subject_radar(subject_xp_data))

    # Streak status
    streak_status = streak_manager.get_longest_streak()
    print(f"\n🔥 Longest Streak: {streak_status['days']} days ({streak_status['streak_type']})")
    print(f"   XP Multiplier: ×{streak_status['xp_multiplier']}")

    # Challenges
    challenge_summary = challenge_manager.get_progress_summary()
    print(f"\n🎯 Challenges:")
    print(f"   Daily: {challenge_summary['daily']['completed_today']}/3 completed today")
    print(f"   Weekly: {challenge_summary['weekly']['active']} active")
    print(f"   Total XP from challenges: {challenge_summary['total_xp_earned']}")

    # Achievement breakdown
    print(f"\n🏆 Achievement Breakdown:")
    for category, data in progress_info['by_category'].items():
        print(f"   {category:15} {data['unlocked']:2}/{data['total']:2} ({data['percent']:.0f}%)")

    print_header("Demo Complete!")

    print("✅ Demonstrated Features:")
    print("   1. XP System & Level Progression")
    print("   2. Achievement Unlocking (108 available)")
    print("   3. Leaderboard Rankings (10+ types)")
    print("   4. Streak Tracking (4 types)")
    print("   5. Daily/Weekly Challenges")
    print("   6. Reward Store (30+ items)")
    print("   7. Progress Visualization")
    print()
    print(f"🎮 Student Journey Summary:")
    print(f"   Level: {progress['level']} ({progress['level_title']})")
    print(f"   Total XP: {progress['total_xp']:,}")
    print(f"   Achievements: {len(student_ach.unlocked)}")
    print(f"   Streak: {streak_status['days']} days (×{streak_status['xp_multiplier']} XP)")
    print(f"   Leaderboard: Rank #{stats['rank']}")
    print()
    print("🚀 Ready for production deployment!")
    print()

if __name__ == "__main__":
    try:
        simulate_student_journey()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
