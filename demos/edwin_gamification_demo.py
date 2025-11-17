"""
EdWIN Gamification System Demo

Demonstrates all gamification features in action.

**Implementation Date**: November 15, 2025
**Agent**: Agent C

Usage:
    PYTHONPATH=. python demos/edwin_gamification_demo.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from EduVerse.edwin.gamification import (
    GamificationEngine, XPSource, LevelSystem
)
from EduVerse.edwin.achievements import (
    AchievementManager, StudentAchievements, get_achievement_count
)
from EduVerse.edwin.leaderboards import LeaderboardManager, LeaderboardType
from EduVerse.edwin.streak_tracking import StreakManager, StreakType
from EduVerse.edwin.challenges import ChallengeManager
from EduVerse.edwin.rewards import RewardStore, StudentInventory
from EduVerse.edwin.progress_viz import (
    generate_xp_progress_bar, generate_subject_radar,
    generate_stats_summary
)


def print_section(title: str):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


async def main():
    """Run comprehensive gamification demo"""

    print_section("EdWIN Gamification System - Complete Demo")

    # ========== Setup ==========
    student_id = "demo_student_001"
    student_name = "Alice Johnson"
    grade = 8

    # Initialize all systems
    print("🎮 Initializing gamification systems...\n")

    # 1. Gamification Engine (XP & Levels)
    gamification = GamificationEngine(student_id=student_id)
    print("✅ Gamification Engine initialized")

    # 2. Achievement System
    achievement_manager = AchievementManager()
    student_achievements = StudentAchievements(student_id=student_id)
    print("✅ Achievement System initialized")

    # 3. Leaderboard System
    leaderboard_manager = LeaderboardManager()
    print("✅ Leaderboard System initialized")

    # 4. Streak Tracking
    streak_manager = StreakManager(student_id=student_id)
    print("✅ Streak Tracking initialized")

    # 5. Challenge System
    challenge_manager = ChallengeManager(student_id=student_id, grade=grade)
    print("✅ Challenge System initialized")

    # 6. Reward Store
    reward_store = RewardStore()
    student_inventory = StudentInventory(student_id=student_id)
    print("✅ Reward Store initialized")

    # ========== Simulate Student Activity ==========
    print_section("Simulating Student Learning Journey")

    # Day 1: First login
    print("📅 Day 1: First Login")
    result = streak_manager.check_in_login()
    print(f"   {result['message']}")

    # Award XP for daily login
    xp_result = gamification.award_xp(XPSource.DAILY_LOGIN)
    print(f"   Earned {xp_result['xp_awarded']} XP for logging in")

    # Check for first login achievement
    stats = {"objectives_mastered": 0, "current_streak": 1}
    unlocked = achievement_manager.check_achievement(
        "first_login", student_achievements, stats
    )
    if unlocked:
        print(f"   🏆 Achievement: {unlocked['name']}")
    print()

    # Day 1: Master first objective
    print("📚 Day 1: Mastering First Objective (Math - Linear Equations)")

    # Award XP
    xp_result = gamification.award_xp(
        XPSource.OBJECTIVE_MASTERED,
        subject="math",
        metadata={"objective_id": "math.algebra.8.linear_eq"}
    )
    print(f"   Earned {xp_result['xp_awarded']} XP")

    if xp_result['leveled_up']:
        print(f"   🎉 Level Up! Now level {xp_result['new_level']}")

    # Check for mastery achievement
    stats["objectives_mastered"] = 1
    unlocked = achievement_manager.check_achievement(
        "first_steps", student_achievements, stats
    )
    if unlocked:
        print(f"   🏆 Achievement: {unlocked['name']} (+{unlocked['xp_reward']} XP)")
        # Award achievement XP
        gamification.award_xp(
            XPSource.CHALLENGE_COMPLETED,
            metadata={"achievement": unlocked['id']}
        )

    # Update challenge progress
    challenge_result = challenge_manager.update_progress(
        challenge_id="daily_master_3",
        requirement_index=0,
        value=1
    )
    print(f"   Daily Challenge: 1/3 objectives mastered")
    print()

    # Day 1: Get perfect score
    print("💯 Day 1: Perfect Score on Quiz")
    xp_result = gamification.award_xp(
        XPSource.PERFECT_SCORE,
        subject="math"
    )
    print(f"   Earned {xp_result['xp_awarded']} XP for perfect score")
    print()

    # Simulate more activity...
    print("📚 Mastering more objectives...")
    for i in range(2, 11):  # Master 9 more objectives
        xp_result = gamification.award_xp(
            XPSource.OBJECTIVE_MASTERED,
            subject=["math", "science", "ela"][i % 3]
        )
        stats["objectives_mastered"] = i

        if i == 5:
            unlocked = achievement_manager.check_achievement(
                "getting_started", student_achievements, stats
            )
            if unlocked:
                print(f"   🏆 Achievement: {unlocked['name']}")

        if i == 10:
            unlocked = achievement_manager.check_achievement(
                "on_a_roll", student_achievements, stats
            )
            if unlocked:
                print(f"   🏆 Achievement: {unlocked['name']}")

    print(f"   Mastered 10 objectives total!")
    print()

    # Simulate 7-day streak
    print("🔥 Simulating 7-Day Streak...")
    for day in range(2, 8):
        streak_manager.check_in_login()
        gamification.update_streak(day)

    print(f"   7-day streak achieved! XP multiplier: ×{gamification.get_progress()['xp_multiplier']}")

    # Check streak achievement
    stats["current_streak"] = 7
    unlocked = achievement_manager.check_achievement(
        "streak_7", student_achievements, stats
    )
    if unlocked:
        print(f"   🏆 Achievement: {unlocked['name']}")
    print()

    # ========== Update Leaderboards ==========
    print("📊 Updating Leaderboards...")

    leaderboard_manager.update_all(
        student_id=student_id,
        student_name=student_name,
        total_xp=gamification.total_xp,
        subject_xp={
            "math": gamification.subject_xp.get("math").total_xp if "math" in gamification.subject_xp else 0,
            "science": gamification.subject_xp.get("science").total_xp if "science" in gamification.subject_xp else 0,
            "ela": gamification.subject_xp.get("ela").total_xp if "ela" in gamification.subject_xp else 0
        },
        objectives_mastered=10,
        current_streak=7,
        grade=grade
    )

    # Add other students for comparison
    leaderboard_manager.update_all(
        student_id="student_002",
        student_name="Bob",
        total_xp=900,
        subject_xp={"math": 450},
        objectives_mastered=8,
        current_streak=5,
        grade=8
    )

    leaderboard_manager.update_all(
        student_id="student_003",
        student_name="Charlie",
        total_xp=1200,
        subject_xp={"math": 600},
        objectives_mastered=12,
        current_streak=10,
        grade=8
    )

    print("   Leaderboards updated!")
    print()

    # ========== Purchase Reward ==========
    print("🛒 Purchasing Reward...")

    result = reward_store.purchase_reward(
        "avatar_robot",
        student_inventory,
        student_xp=gamification.total_xp
    )

    if result["success"]:
        print(f"   ✅ Purchased: {result['reward_name']}")
        print(f"   Cost: {result['xp_spent']} XP")

        # Equip avatar
        reward_store.equip_reward("avatar_robot", student_inventory)
        print(f"   Equipped avatar: {student_inventory.active_avatar}")
    print()

    # ========== Display Progress ==========
    print_section("Current Progress")

    # XP and Level
    progress = gamification.get_progress()
    print(generate_xp_progress_bar(
        progress['current_xp'],
        progress['xp_needed']
    ))
    print(f"\nLevel {progress['level']}: {progress['level_title']}")
    print(f"Total XP: {progress['total_xp']:,}")
    print(f"Streak Multiplier: ×{progress['xp_multiplier']}")
    print()

    # Subject progress
    print("Subject Progress:")
    all_subjects = gamification.get_all_subjects()
    subject_levels = {}
    for subj in all_subjects:
        print(f"  {subj['subject'].upper()}: Level {subj['level']} ({subj['total_xp']} XP)")
        subject_levels[subj['subject']] = subj['level']
    print()

    # Achievements
    ach_progress = achievement_manager.get_progress(student_achievements)
    print(f"Achievements: {ach_progress['unlocked_count']}/{ach_progress['total_achievements']} ({ach_progress['completion_percent']:.1f}%)")
    print()

    # Leaderboard rank
    global_lb = leaderboard_manager.get_leaderboard(LeaderboardType.GLOBAL_XP)
    personal = global_lb.get_personal_rank(student_id)
    if personal:
        print(f"Global Rank: #{personal['rank']} (Top {100 - personal['percentile']:.1f}%)")
    print()

    # Active streaks
    streak_summary = streak_manager.get_summary()
    print("Active Streaks:")
    for streak_type, status in streak_summary['streaks'].items():
        if status['current_days'] > 0:
            print(f"  {streak_type}: {status['current_days']} days")
    print()

    # ========== Show Leaderboard ==========
    print_section("Global Leaderboard (Top 3)")

    if global_lb:
        top_3 = global_lb.get_rankings(limit=3)
        for entry in top_3:
            marker = "👑" if entry.rank == 1 else ("→" if entry.student_id == student_id else " ")
            print(f"{marker} {entry.rank}. {entry.student_name:15} {entry.score:,.0f} XP")
    print()

    # ========== Show Active Challenges ==========
    print_section("Active Challenges")

    active_challenges = challenge_manager.get_active_challenges()
    for challenge in active_challenges[:3]:
        print(f"📋 {challenge.name}")
        print(f"   {challenge.description}")
        print(f"   Reward: {challenge.xp_reward} XP")
        for req in challenge.requirements:
            progress_bar = "█" * int(req.progress_percent / 10) + "░" * (10 - int(req.progress_percent / 10))
            print(f"   [{progress_bar}] {req.current_value}/{req.target_value} {req.unit}")
        print()

    # ========== Summary Stats ==========
    print_section("Statistics Summary")

    stats_data = {
        "level": progress['level'],
        "total_xp": gamification.total_xp,
        "objectives_mastered": 10,
        "current_streak": 7,
        "achievements_unlocked": ach_progress['unlocked_count'],
        "rank": personal['rank'] if personal else "N/A"
    }

    print(generate_stats_summary(stats_data))
    print()

    # ========== Achievement System Info ==========
    print_section("Achievement System")

    counts = get_achievement_count()
    print(f"Total Achievements Available: {counts['total']}")
    print("\nBy Category:")
    for cat, count in counts['by_category'].items():
        print(f"  {cat:12} {count:3} achievements")
    print("\nBy Rarity:")
    for rar, count in counts['by_rarity'].items():
        print(f"  {rar:12} {count:3} achievements")
    print()

    # ========== Next Recommendations ==========
    print_section("Next Recommendations")

    print("Recommended Achievements:")
    recommendations = achievement_manager.get_next_recommendations(
        student_achievements, limit=3
    )
    for rec in recommendations:
        print(f"  {rec['icon']} {rec['name']}")
        print(f"     {rec['requirement']} → +{rec['xp_reward']} XP")
    print()

    print("Available Rewards:")
    available_rewards = reward_store.get_available_rewards(
        student_level=progress['level'],
        student_achievements=list(student_achievements.unlocked)
    )
    for reward in available_rewards[:3]:
        print(f"  {reward.icon} {reward.name} - {reward.xp_cost} XP")
    print()

    print_section("Demo Complete!")
    print("✅ All gamification systems working perfectly!")
    print("\nTry the interactive dashboard at:")
    print("   EduVerse/edwin/static/gamification_dashboard.html")
    print()


if __name__ == "__main__":
    asyncio.run(main())
