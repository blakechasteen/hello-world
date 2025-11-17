"""
EdWIN Gamification System - Simple Demo

Demonstrates gamification features without full EdWIN dependencies.

**Implementation Date**: November 15, 2025
**Agent**: Agent C

Usage:
    python demos/edwin_gamification_simple_demo.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_section(title: str):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    """Run simplified gamification demo"""

    print_section("EdWIN Gamification System - Simple Demo")

    # Import modules individually
    from EduVerse.edwin import gamification, achievements, leaderboards, streak_tracking, challenges, rewards

    # ========== Setup ==========
    student_id = "demo_student_001"
    student_name = "Alice Johnson"
    grade = 8

    print("🎮 Initializing gamification systems...\n")

    # 1. Gamification Engine
    engine = gamification.GamificationEngine(student_id=student_id)
    print("✅ Gamification Engine initialized")

    # 2. Achievement System
    achievement_manager = achievements.AchievementManager()
    student_ach = achievements.StudentAchievements(student_id=student_id)
    print("✅ Achievement System initialized")

    # 3. Leaderboard System
    leaderboard_manager = leaderboards.LeaderboardManager()
    print("✅ Leaderboard System initialized")

    # 4. Streak Tracking
    streak_manager = streak_tracking.StreakManager(student_id=student_id)
    print("✅ Streak Tracking initialized")

    # 5. Challenge System
    challenge_manager = challenges.ChallengeManager(student_id=student_id, grade=grade)
    print("✅ Challenge System initialized")

    # 6. Reward Store
    reward_store = rewards.RewardStore()
    student_inventory = rewards.StudentInventory(student_id=student_id)
    print("✅ Reward Store initialized")

    # ========== Simulate Activity ==========
    print_section("Simulating Student Activity")

    # Day 1: Login
    print("📅 Day 1: First Login")
    streak_result = streak_manager.check_in_login()
    print(f"   {streak_result['message']}")

    xp_result = engine.award_xp(gamification.XPSource.DAILY_LOGIN)
    print(f"   Earned {xp_result['xp_awarded']} XP")
    print()

    # Master objectives
    print("📚 Mastering Objectives...")
    for i in range(1, 11):
        xp_result = engine.award_xp(
            gamification.XPSource.OBJECTIVE_MASTERED,
            subject="math" if i % 2 == 0 else "science"
        )

        if i == 1:
            # Check first achievement
            stats = {"objectives_mastered": 1, "current_streak": 1}
            unlocked = achievement_manager.check_achievement(
                "first_steps", student_ach, stats
            )
            if unlocked:
                print(f"   🏆 Achievement: {unlocked['name']}")

        if i == 5:
            # Check second achievement
            stats = {"objectives_mastered": 5, "current_streak": 1}
            unlocked = achievement_manager.check_achievement(
                "getting_started", student_ach, stats
            )
            if unlocked:
                print(f"   🏆 Achievement: {unlocked['name']}")

        if xp_result['leveled_up']:
            print(f"   🎉 Level Up! Now level {xp_result['new_level']}")

    print(f"   Mastered 10 objectives!")
    print()

    # Perfect score
    print("💯 Getting Perfect Score...")
    xp_result = engine.award_xp(gamification.XPSource.PERFECT_SCORE, subject="math")
    print(f"   Earned {xp_result['xp_awarded']} XP")
    print()

    # Update leaderboard
    print("📊 Updating Leaderboards...")
    leaderboard_manager.update_all(
        student_id=student_id,
        student_name=student_name,
        total_xp=engine.total_xp,
        subject_xp={
            "math": engine.subject_xp.get("math").total_xp if "math" in engine.subject_xp else 0,
            "science": engine.subject_xp.get("science").total_xp if "science" in engine.subject_xp else 0
        },
        objectives_mastered=10,
        current_streak=1,
        grade=grade
    )

    # Add comparison students
    leaderboard_manager.update_all("student_002", "Bob", 900, {"math": 450}, 8, 5, 8)
    leaderboard_manager.update_all("student_003", "Charlie", 1200, {"math": 600}, 12, 10, 8)

    print("   Leaderboards updated!")
    print()

    # Purchase reward
    print("🛒 Purchasing Reward...")
    result = reward_store.purchase_reward(
        "avatar_robot", student_inventory, student_xp=engine.total_xp
    )
    if result["success"]:
        print(f"   ✅ Purchased: {result['reward_name']}")
        reward_store.equip_reward("avatar_robot", student_inventory)
        print(f"   Equipped avatar: {student_inventory.active_avatar}")
    print()

    # ========== Display Progress ==========
    print_section("Current Progress")

    progress = engine.get_progress()
    print(f"Level: {progress['level']} ({progress['level_title']})")
    print(f"Total XP: {progress['total_xp']:,}")
    print(f"XP to next level: {progress['current_xp']}/{progress['xp_needed']}")
    print(f"Progress: {progress['progress_percent']:.1f}%")
    print()

    # Achievements
    ach_progress = achievement_manager.get_progress(student_ach)
    print(f"Achievements: {ach_progress['unlocked_count']}/{ach_progress['total_achievements']} ({ach_progress['completion_percent']:.1f}%)")
    print()

    # Leaderboard
    global_lb = leaderboard_manager.get_leaderboard(leaderboards.LeaderboardType.GLOBAL_XP)
    if global_lb:
        print("Leaderboard (Top 3):")
        top_3 = global_lb.get_rankings(limit=3)
        for entry in top_3:
            marker = "→" if entry.student_id == student_id else " "
            print(f"{marker} {entry.rank}. {entry.student_name:15} {entry.score:,.0f} XP")
        print()

        personal = global_lb.get_personal_rank(student_id)
        if personal:
            print(f"Your Rank: #{personal['rank']} (Top {100 - personal['percentile']:.1f}%)")
    print()

    # Streaks
    streak_summary = streak_manager.get_summary()
    print("Active Streaks:")
    for streak_type, status in streak_summary['streaks'].items():
        if status['current_days'] > 0:
            print(f"  {streak_type}: {status['current_days']} days")
    print()

    # Challenges
    print("Active Challenges:")
    active_challenges = challenge_manager.get_active_challenges()
    for challenge in active_challenges[:3]:
        print(f"  📋 {challenge.name}")
        for req in challenge.requirements:
            print(f"     {req.current_value}/{req.target_value} {req.unit}")
    print()

    # ========== Summary ==========
    print_section("Summary")

    print("✅ All gamification systems working!")
    print()
    print(f"Total Systems: 6")
    print(f"  1. Gamification Engine (XP & Levels)")
    print(f"  2. Achievement System (100+ badges)")
    print(f"  3. Leaderboard System (multiple types)")
    print(f"  4. Streak Tracking (4 types)")
    print(f"  5. Challenge System (daily/weekly)")
    print(f"  6. Reward Store (avatars, themes, power-ups)")
    print()

    stats = {
        "total_xp": engine.total_xp,
        "level": progress['level'],
        "objectives_mastered": 10,
        "achievements": ach_progress['unlocked_count'],
        "leaderboard_rank": personal['rank'] if personal else "N/A",
        "streak": 1
    }

    print("Student Progress:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    print("🎉 Demo complete!")
    print()
    print("Next steps:")
    print("  1. View dashboard: EduVerse/edwin/static/gamification_dashboard.html")
    print("  2. Run tests: pytest EduVerse/edwin/tests/test_gamification.py -v")
    print("  3. Read docs: EduVerse/edwin/GAMIFICATION_SUMMARY.md")


if __name__ == "__main__":
    main()
