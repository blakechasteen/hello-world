"""
EdWIN Progress Visualization

Generate visual progress feedback (charts, graphs, calendars).

**Implementation Date**: November 15, 2025
**Agent**: Agent C
"""

from typing import Dict, List
from datetime import datetime, timedelta


def generate_xp_progress_bar(current_xp: int, xp_needed: int, width: int = 40) -> str:
    """Generate ASCII progress bar for XP"""
    progress = min(1.0, current_xp / xp_needed)
    filled = int(width * progress)
    empty = width - filled

    bar = "█" * filled + "░" * empty
    percent = int(progress * 100)

    return f"[{bar}] {percent}% ({current_xp}/{xp_needed} XP)"


def generate_subject_radar(subject_levels: Dict[str, int]) -> str:
    """Generate ASCII radar chart for subject mastery"""
    subjects = ["Math", "Science", "ELA", "Social", "AI"]
    levels = [subject_levels.get(s.lower(), 0) for s in subjects]

    output = ["Subject Mastery Radar:\n"]

    # Simple ASCII representation
    max_level = max(levels) if levels else 1
    for subject, level in zip(subjects, levels):
        bar_length = int((level / max_level) * 20) if max_level > 0 else 0
        bar = "█" * bar_length
        output.append(f"  {subject:8} | {bar} Lvl {level}")

    return "\n".join(output)


def generate_streak_calendar(streak_days: List[datetime], weeks: int = 4) -> str:
    """Generate GitHub-style streak calendar"""
    output = ["Streak Calendar (Last 4 weeks):\n"]
    output.append("  Mon Tue Wed Thu Fri Sat Sun")

    # Create calendar grid
    today = datetime.utcnow().date()
    start_date = today - timedelta(weeks=weeks)

    # Find Monday of start week
    while start_date.weekday() != 0:
        start_date -= timedelta(days=1)

    current = start_date
    week_days = []

    for _ in range(weeks * 7):
        if current in [d.date() for d in streak_days]:
            week_days.append("🟩")
        else:
            week_days.append("⬜")

        if len(week_days) == 7:
            output.append("  " + " ".join(week_days))
            week_days = []

        current += timedelta(days=1)

    return "\n".join(output)


def generate_achievement_showcase(achievements: List[Dict], limit: int = 5) -> str:
    """Generate achievement showcase"""
    output = ["Recent Achievements:\n"]

    for i, ach in enumerate(achievements[:limit], 1):
        output.append(f"  {i}. {ach['icon']} {ach['name']}")
        output.append(f"     {ach['description']}")

    return "\n".join(output)


def generate_leaderboard_viz(entries: List[Dict], my_rank: int) -> str:
    """Generate leaderboard visualization"""
    output = ["Leaderboard:\n"]

    for entry in entries:
        rank = entry['rank']
        name = entry['student_name']
        score = entry['score']

        marker = "→" if rank == my_rank else " "
        output.append(f"{marker} {rank:2}. {name:15} {score:,.0f} XP")

    return "\n".join(output)


def generate_progress_chart(data_points: List[float], width: int = 40) -> str:
    """Generate simple ASCII line chart"""
    if not data_points:
        return "No data"

    output = ["Progress Over Time:\n"]

    # Normalize to chart height
    max_val = max(data_points)
    min_val = min(data_points)
    range_val = max_val - min_val if max_val != min_val else 1

    height = 10
    chart = [[] for _ in range(height)]

    for val in data_points:
        normalized = (val - min_val) / range_val
        row = int(normalized * (height - 1))
        for i in range(height):
            if height - i - 1 == row:
                chart[i].append("●")
            else:
                chart[i].append(" ")

    # Draw chart
    for row in chart:
        output.append("  " + "".join(row))

    # X-axis
    output.append("  " + "─" * len(data_points))

    return "\n".join(output)


def generate_stats_summary(stats: Dict) -> str:
    """Generate stats summary"""
    output = ["Statistics Summary:\n"]

    output.append(f"  Level: {stats.get('level', 0)}")
    output.append(f"  Total XP: {stats.get('total_xp', 0):,}")
    output.append(f"  Objectives Mastered: {stats.get('objectives_mastered', 0)}")
    output.append(f"  Current Streak: {stats.get('current_streak', 0)} days")
    output.append(f"  Achievements: {stats.get('achievements_unlocked', 0)}")
    output.append(f"  Leaderboard Rank: #{stats.get('rank', 'N/A')}")

    return "\n".join(output)


if __name__ == "__main__":
    print("=== EdWIN Progress Visualization Demo ===\n")

    # XP progress bar
    print(generate_xp_progress_bar(750, 1000))
    print()

    # Subject radar
    print(generate_subject_radar({
        "math": 8,
        "science": 6,
        "ela": 5,
        "social": 4,
        "ai": 7
    }))
    print()

    # Streak calendar
    streak_days = [
        datetime.utcnow() - timedelta(days=i)
        for i in range(0, 20, 2)  # Every other day
    ]
    print(generate_streak_calendar(streak_days))
    print()

    # Stats summary
    print(generate_stats_summary({
        "level": 15,
        "total_xp": 7550,
        "objectives_mastered": 32,
        "current_streak": 7,
        "achievements_unlocked": 18,
        "rank": 42
    }))
