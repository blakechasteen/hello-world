"""
Comprehensive Tests for EdWIN Gamification System

**Implementation Date**: November 15, 2025
**Agent**: Agent C
"""

import pytest
from datetime import datetime, timedelta

from EduVerse.edwin.gamification import (
    GamificationEngine, XPSource, LevelSystem, get_xp_multiplier
)
from EduVerse.edwin.achievements import (
    AchievementManager, StudentAchievements, create_all_achievements
)
from EduVerse.edwin.leaderboards import (
    LeaderboardManager, LeaderboardType, Leaderboard, LeaderboardConfig,
    LeaderboardPeriod
)
from EduVerse.edwin.streak_tracking import (
    StreakManager, Streak, StreakType, get_streak_multiplier
)
from EduVerse.edwin.challenges import (
    ChallengeManager, create_daily_challenges, create_weekly_challenges
)
from EduVerse.edwin.rewards import RewardStore, StudentInventory


# ========== Gamification Engine Tests ==========

def test_gamification_engine_initialization():
    """Test gamification engine initialization"""
    engine = GamificationEngine(student_id="test_001")

    assert engine.student_id == "test_001"
    assert engine.total_xp == 0
    assert engine.current_streak == 0
    assert len(engine.subject_xp) == 0


def test_award_xp_basic():
    """Test basic XP awarding"""
    engine = GamificationEngine(student_id="test_001")

    result = engine.award_xp(XPSource.OBJECTIVE_MASTERED)

    assert result['xp_awarded'] == 50
    assert result['new_total'] == 50
    assert result['multiplier'] == 1.0


def test_award_xp_with_streak_multiplier():
    """Test XP with streak multiplier"""
    engine = GamificationEngine(student_id="test_001", starting_streak=7)

    result = engine.award_xp(XPSource.OBJECTIVE_MASTERED)

    # 7-day streak = 1.1x multiplier, 50 XP * 1.1 = 55 XP
    assert result['multiplier'] == 1.1
    assert result['xp_awarded'] == 55


def test_level_progression():
    """Test level progression"""
    engine = GamificationEngine(student_id="test_001")

    # Award enough XP to level up
    result = engine.award_xp(XPSource.PERFECT_SCORE)  # 100 XP
    result = engine.award_xp(XPSource.PERFECT_SCORE)  # 200 XP total

    progress = engine.get_progress()

    assert progress['level'] >= 2  # Should be at least level 2


def test_subject_xp_tracking():
    """Test subject-specific XP tracking"""
    engine = GamificationEngine(student_id="test_001")

    engine.award_xp(XPSource.OBJECTIVE_MASTERED, subject="math")

    assert "math" in engine.subject_xp
    assert engine.subject_xp["math"].total_xp == 50


def test_level_system_xp_calculation():
    """Test level XP calculations"""
    # Level 1 → 2 should require 100 XP
    assert LevelSystem.xp_for_level(1) == 100

    # Total XP to reach level 2 is 100
    assert LevelSystem.total_xp_for_level(2) == 100

    # Level from XP
    assert LevelSystem.level_from_xp(0) == 1
    assert LevelSystem.level_from_xp(100) == 2
    assert LevelSystem.level_from_xp(50) == 1


def test_level_system_progress():
    """Test level progress calculation"""
    current_xp, xp_needed, progress = LevelSystem.progress_to_next_level(50)

    assert current_xp == 50
    assert xp_needed == 100
    assert progress == 0.5


# ========== Achievement System Tests ==========

def test_achievement_creation():
    """Test achievement catalog creation"""
    achievements = create_all_achievements()

    assert len(achievements) >= 100  # At least 100 achievements
    assert "first_steps" in achievements
    assert "streak_7" in achievements


def test_achievement_manager_initialization():
    """Test achievement manager initialization"""
    manager = AchievementManager()

    assert len(manager.achievements) >= 100


def test_achievement_unlock():
    """Test achievement unlocking"""
    manager = AchievementManager()
    student_ach = StudentAchievements(student_id="test_001")

    # Unlock "first_steps" (master 1 objective)
    stats = {"objectives_mastered": 1}
    result = manager.check_achievement("first_steps", student_ach, stats)

    assert result is not None
    assert result['name'] == "First Steps"
    assert "first_steps" in student_ach.unlocked


def test_achievement_already_unlocked():
    """Test that already unlocked achievements return None"""
    manager = AchievementManager()
    student_ach = StudentAchievements(student_id="test_001")

    # Unlock once
    stats = {"objectives_mastered": 1}
    manager.check_achievement("first_steps", student_ach, stats)

    # Try to unlock again
    result = manager.check_achievement("first_steps", student_ach, stats)

    assert result is None  # Already unlocked


def test_achievement_progress():
    """Test achievement progress tracking"""
    manager = AchievementManager()
    student_ach = StudentAchievements(student_id="test_001")

    # Unlock some achievements
    stats = {"objectives_mastered": 5}
    manager.check_achievement("first_steps", student_ach, stats)
    manager.check_achievement("getting_started", student_ach, stats)

    progress = manager.get_progress(student_ach)

    assert progress['unlocked_count'] == 2
    assert progress['completion_percent'] > 0


# ========== Leaderboard System Tests ==========

def test_leaderboard_initialization():
    """Test leaderboard initialization"""
    config = LeaderboardConfig(
        leaderboard_type=LeaderboardType.GLOBAL_XP,
        period=LeaderboardPeriod.ALL_TIME
    )
    leaderboard = Leaderboard(config)

    assert len(leaderboard.entries) == 0


def test_leaderboard_score_update():
    """Test updating leaderboard scores"""
    config = LeaderboardConfig(
        leaderboard_type=LeaderboardType.GLOBAL_XP,
        period=LeaderboardPeriod.ALL_TIME
    )
    leaderboard = Leaderboard(config)

    leaderboard.update_score("student_001", "Alice", 1000, grade=8)

    assert "student_001" in leaderboard.entries
    assert leaderboard.entries["student_001"].score == 1000
    assert leaderboard.entries["student_001"].rank == 1


def test_leaderboard_ranking():
    """Test leaderboard ranking"""
    config = LeaderboardConfig(
        leaderboard_type=LeaderboardType.GLOBAL_XP,
        period=LeaderboardPeriod.ALL_TIME
    )
    leaderboard = Leaderboard(config)

    leaderboard.update_score("student_001", "Alice", 1000)
    leaderboard.update_score("student_002", "Bob", 1500)
    leaderboard.update_score("student_003", "Charlie", 800)

    rankings = leaderboard.get_rankings(limit=10)

    assert rankings[0].student_name == "Bob"  # Highest score
    assert rankings[0].rank == 1
    assert rankings[1].student_name == "Alice"
    assert rankings[1].rank == 2
    assert rankings[2].student_name == "Charlie"
    assert rankings[2].rank == 3


def test_leaderboard_personal_rank():
    """Test personal rank retrieval"""
    config = LeaderboardConfig(
        leaderboard_type=LeaderboardType.GLOBAL_XP,
        period=LeaderboardPeriod.ALL_TIME
    )
    leaderboard = Leaderboard(config)

    leaderboard.update_score("student_001", "Alice", 1000)
    leaderboard.update_score("student_002", "Bob", 1500)

    personal = leaderboard.get_personal_rank("student_001")

    assert personal is not None
    assert personal['rank'] == 2
    assert personal['total_participants'] == 2


def test_leaderboard_manager():
    """Test leaderboard manager"""
    manager = LeaderboardManager()

    manager.update_all(
        student_id="student_001",
        student_name="Alice",
        total_xp=1000,
        subject_xp={"math": 500},
        objectives_mastered=10,
        current_streak=5,
        grade=8
    )

    global_lb = manager.get_leaderboard(LeaderboardType.GLOBAL_XP)
    assert global_lb is not None

    personal = global_lb.get_personal_rank("student_001")
    assert personal is not None


# ========== Streak Tracking Tests ==========

def test_streak_initialization():
    """Test streak initialization"""
    streak = Streak(student_id="test_001", streak_type=StreakType.DAILY_LOGIN)

    assert streak.current_days == 0
    assert streak.best_days == 0


def test_streak_check_in():
    """Test streak check-in"""
    streak = Streak(student_id="test_001", streak_type=StreakType.DAILY_LOGIN)

    result = streak.check_in()

    assert result['success'] is True
    assert result['current_days'] == 1


def test_streak_continuation():
    """Test streak continuation"""
    streak = Streak(student_id="test_001", streak_type=StreakType.DAILY_LOGIN)

    # Day 1
    streak.check_in()

    # Day 2 (simulate time passing)
    streak.last_check_in = datetime.utcnow() - timedelta(days=1)
    result = streak.check_in()

    assert result['current_days'] == 2


def test_streak_expiration():
    """Test streak expiration"""
    streak = Streak(student_id="test_001", streak_type=StreakType.DAILY_LOGIN)

    # Day 1
    streak.check_in()

    # 3 days later (expired)
    streak.last_check_in = datetime.utcnow() - timedelta(days=3)
    result = streak.check_in()

    assert result.get('streak_broken') is True
    assert result['current_days'] == 1  # Reset


def test_streak_freeze():
    """Test streak freeze"""
    streak = Streak(student_id="test_001", streak_type=StreakType.DAILY_LOGIN)

    success = streak.use_freeze()

    assert success is True
    assert streak.freezes_available == 1  # Used 1 of 2


def test_streak_multiplier():
    """Test streak XP multiplier"""
    assert get_streak_multiplier(0) == 1.0
    assert get_streak_multiplier(7) == 1.1
    assert get_streak_multiplier(30) == 1.5
    assert get_streak_multiplier(100) == 2.0


def test_streak_manager():
    """Test streak manager"""
    manager = StreakManager(student_id="test_001")

    result = manager.check_in_login()

    assert result['success'] is True

    summary = manager.get_summary()
    assert summary['student_id'] == "test_001"


# ========== Challenge System Tests ==========

def test_daily_challenges_creation():
    """Test daily challenge creation"""
    challenges = create_daily_challenges()

    assert len(challenges) > 0
    for challenge in challenges:
        assert challenge.expires_at is not None


def test_weekly_challenges_creation():
    """Test weekly challenge creation"""
    challenges = create_weekly_challenges()

    assert len(challenges) > 0
    for challenge in challenges:
        assert challenge.expires_at is not None


def test_challenge_manager_initialization():
    """Test challenge manager initialization"""
    manager = ChallengeManager(student_id="test_001", grade=8)

    active = manager.get_active_challenges()

    assert len(active) >= 3  # At least 3 daily challenges


def test_challenge_progress_update():
    """Test challenge progress update"""
    manager = ChallengeManager(student_id="test_001", grade=8)

    active = manager.get_active_challenges()
    challenge = active[0]

    result = manager.update_progress(
        challenge_id=challenge.id,
        requirement_index=0,
        value=challenge.requirements[0].target_value
    )

    # Challenge should be completed
    assert result is not None
    assert challenge.completed is True


def test_challenge_summary():
    """Test challenge summary"""
    manager = ChallengeManager(student_id="test_001", grade=8)

    summary = manager.get_progress_summary()

    assert summary['student_id'] == "test_001"
    assert 'active_challenges' in summary


# ========== Reward System Tests ==========

def test_reward_store_initialization():
    """Test reward store initialization"""
    store = RewardStore()

    assert len(store.catalog) > 0


def test_reward_purchase():
    """Test reward purchase"""
    store = RewardStore()
    inventory = StudentInventory(student_id="test_001")

    result = store.purchase_reward(
        "avatar_robot",
        inventory,
        student_xp=150
    )

    assert result['success'] is True
    assert "avatar_robot" in inventory.owned_rewards


def test_reward_purchase_insufficient_xp():
    """Test reward purchase with insufficient XP"""
    store = RewardStore()
    inventory = StudentInventory(student_id="test_001")

    result = store.purchase_reward(
        "avatar_robot",
        inventory,
        student_xp=50  # Not enough
    )

    assert result['success'] is False
    assert "Insufficient XP" in result['error']


def test_reward_equip():
    """Test equipping rewards"""
    store = RewardStore()
    inventory = StudentInventory(student_id="test_001")

    # Purchase and equip
    store.purchase_reward("avatar_robot", inventory, student_xp=150)
    success = store.equip_reward("avatar_robot", inventory)

    assert success is True
    assert inventory.active_avatar == "avatar_robot"


# ========== Integration Tests ==========

def test_full_gamification_flow():
    """Test complete gamification flow"""
    # Initialize all systems
    engine = GamificationEngine(student_id="test_001")
    achievement_manager = AchievementManager()
    student_ach = StudentAchievements(student_id="test_001")
    leaderboard_manager = LeaderboardManager()
    streak_manager = StreakManager(student_id="test_001")

    # Day 1: Login and master objective
    streak_manager.check_in_login()
    engine.award_xp(XPSource.DAILY_LOGIN)

    xp_result = engine.award_xp(XPSource.OBJECTIVE_MASTERED, subject="math")

    # Check achievements
    stats = {
        "objectives_mastered": 1,
        "current_streak": 1
    }
    ach_result = achievement_manager.check_achievement(
        "first_steps", student_ach, stats
    )

    # Update leaderboard
    leaderboard_manager.update_all(
        student_id="test_001",
        student_name="Test Student",
        total_xp=engine.total_xp,
        subject_xp={"math": engine.subject_xp.get("math").total_xp if "math" in engine.subject_xp else 0},
        objectives_mastered=1,
        current_streak=1,
        grade=8
    )

    # Verify everything worked
    assert engine.total_xp > 0
    assert ach_result is not None
    assert "first_steps" in student_ach.unlocked

    global_lb = leaderboard_manager.get_leaderboard(LeaderboardType.GLOBAL_XP)
    personal = global_lb.get_personal_rank("test_001")
    assert personal is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
