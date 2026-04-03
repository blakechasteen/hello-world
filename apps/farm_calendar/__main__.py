#!/usr/bin/env python3
"""
Farm Calendar CLI — view, complete, and track farm tasks.

Usage:
    python -m apps.farm_calendar                    # Show what's due
    python -m apps.farm_calendar upcoming           # Next 7 days
    python -m apps.farm_calendar upcoming 14        # Next 14 days
    python -m apps.farm_calendar all                # All tasks
    python -m apps.farm_calendar complete <task-id> [--notes "..."] [--outcome completed|partial|skipped|deferred|failed]
    python -m apps.farm_calendar history <task-id>  # Completion history
    python -m apps.farm_calendar streak <task-id>   # Streak stats
    python -m apps.farm_calendar tags               # List all tags
    python -m apps.farm_calendar tag <tag>           # Tasks by tag
    python -m apps.farm_calendar add <id> <name> <recurrence> [--tags t1,t2] [--priority low|medium|high|critical]
    python -m apps.farm_calendar log                # Recent completions

Created: 2026-04-02
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from hololoom.core.chrono.calendar import AgentCalendar, CalendarTask, Outcome, Priority
from hololoom.core.chrono.presets.farm import load_farm_calendar

DEFAULT_PATH = Path(__file__).parent.parent / "data" / "calendars" / "farm.json"


def get_calendar() -> AgentCalendar:
    """Load or initialize the farm calendar."""
    if DEFAULT_PATH.exists():
        return AgentCalendar(agent_id="blake-farm", persistence_path=DEFAULT_PATH)
    else:
        cal = load_farm_calendar(agent_id="blake-farm")
        cal.persistence_path = DEFAULT_PATH
        cal.save()
        return cal


def cmd_due(cal: AgentCalendar, args: argparse.Namespace) -> None:
    """Show what's due now + overdue."""
    now = datetime.now()
    print(f"\n  Farm Calendar — {now.strftime('%A, %B %d %Y %I:%M %p')}")
    print(f"  {'=' * 55}\n")

    overdue = cal.overdue(now)
    due_now = cal.due_tasks(now, tolerance_minutes=60)

    if due_now:
        print("  DUE NOW:")
        for t in due_now:
            _print_task(cal, t, now)
        print()

    if overdue:
        # Filter out ones already shown as due
        due_ids = {t.id for t in due_now}
        overdue = [t for t in overdue if t.id not in due_ids]
        if overdue:
            print(f"  NEEDS ATTENTION ({len(overdue)} tasks):")
            for t in overdue[:10]:
                _print_task(cal, t, now)
            if len(overdue) > 10:
                print(f"    ... and {len(overdue) - 10} more")
            print()

    reminders = cal.needs_reminder(now)
    if reminders:
        print("  REMINDERS:")
        for t in reminders:
            nxt = t.next_occurrence(now)
            days = (nxt - now).days
            print(f"    {t.name} in {days} day(s) [{', '.join(t.tags)}]")
        print()

    # Quick stats
    recent = cal.all_completions(since=now - timedelta(days=7))
    print(f"  Completions this week: {len(recent)}")
    print()


def cmd_upcoming(cal: AgentCalendar, args: argparse.Namespace) -> None:
    """Show upcoming tasks."""
    days = args.days or 7
    now = datetime.now()
    upcoming = cal.upcoming(timedelta(days=days), now)

    print(f"\n  Next {days} days ({len(upcoming)} tasks):")
    print(f"  {'=' * 55}\n")

    current_day = None
    for t in upcoming:
        nxt = t.next_occurrence(now)
        day_str = nxt.strftime("%A, %b %d")
        if day_str != current_day:
            current_day = day_str
            print(f"  {day_str}")
        time_str = nxt.strftime("%I:%M %p") if t.schedule.at_hour else "all day"
        priority_marker = {"critical": "!!", "high": "! ", "medium": "  ", "low": "  "}
        marker = priority_marker.get(t.priority.value, "  ")
        print(f"    {marker}{time_str:>8s}  {t.name}  [{', '.join(t.tags)}]")
    print()


def cmd_all(cal: AgentCalendar, args: argparse.Namespace) -> None:
    """List all tasks."""
    print(f"\n  All tasks ({len(cal.tasks)}):\n")
    for t in cal.tasks:
        streak = cal.streak(t.id)
        completions = streak["total_completions"]
        rate = streak["completion_rate"]
        rate_str = f"{rate:.0%}" if completions > 0 else "new"
        print(
            f"  {t.id:30s} {t.priority.value:8s} "
            f"{t.recurrence:25s} {rate_str:>4s} ({completions}x)"
        )
        print(f"    {t.name}")
    print()


def cmd_complete(cal: AgentCalendar, args: argparse.Namespace) -> None:
    """Record a task completion."""
    task = cal.get(args.task_id)
    if not task:
        print(f"  Unknown task: {args.task_id}")
        print(f"  Available: {', '.join(t.id for t in cal.tasks)}")
        sys.exit(1)

    outcome = Outcome(args.outcome) if args.outcome else Outcome.COMPLETED
    record = cal.complete(
        args.task_id,
        outcome=outcome,
        notes=args.notes or "",
    )
    cal.save()

    streak = cal.streak(args.task_id)
    print(f"\n  Recorded: {task.name}")
    print(f"  Outcome:  {record.outcome.value}")
    if record.notes:
        print(f"  Notes:    {record.notes}")
    print(f"  Streak:   {streak['current_streak']} (best: {streak['longest_streak']})")
    print(f"  Rate:     {streak['completion_rate']:.0%} ({streak['total_completions']} total)")
    print()


def cmd_history(cal: AgentCalendar, args: argparse.Namespace) -> None:
    """Show completion history for a task."""
    task = cal.get(args.task_id)
    if not task:
        print(f"  Unknown task: {args.task_id}")
        sys.exit(1)

    records = cal.history(args.task_id, limit=args.limit or 20)
    print(f"\n  History: {task.name} ({len(records)} records)\n")

    for r in records:
        outcome_icon = {
            "completed": "+",
            "partial": "~",
            "skipped": "-",
            "failed": "x",
            "deferred": ">",
        }.get(r.outcome.value, "?")
        print(f"  [{outcome_icon}] {r.completed_at.strftime('%Y-%m-%d %H:%M')}  {r.outcome.value}")
        if r.notes:
            print(f"      {r.notes}")
        if r.duration_seconds:
            print(f"      duration: {r.duration_seconds:.0f}s")
    print()


def cmd_streak(cal: AgentCalendar, args: argparse.Namespace) -> None:
    """Show streak stats for a task."""
    task = cal.get(args.task_id)
    if not task:
        print(f"  Unknown task: {args.task_id}")
        sys.exit(1)

    s = cal.streak(args.task_id)
    print(f"\n  Streak: {task.name}")
    print(f"  Current:    {s['current_streak']}")
    print(f"  Longest:    {s['longest_streak']}")
    print(f"  Rate:       {s['completion_rate']:.0%}")
    print(f"  Total:      {s['total_completions']}")
    print()


def cmd_tags(cal: AgentCalendar, args: argparse.Namespace) -> None:
    """List all tags with task counts."""
    tag_counts: dict[str, int] = {}
    for t in cal.tasks:
        for tag in t.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print(f"\n  Tags ({len(tag_counts)}):\n")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag:20s} {count} task(s)")
    print()


def cmd_tag(cal: AgentCalendar, args: argparse.Namespace) -> None:
    """Show tasks with a specific tag."""
    now = datetime.now()
    tasks = cal.tasks_by_tag(args.tag)
    if not tasks:
        print(f"  No tasks with tag: {args.tag}")
        sys.exit(1)

    print(f"\n  Tasks tagged '{args.tag}' ({len(tasks)}):\n")
    for t in tasks:
        nxt = t.next_occurrence(now)
        last = cal.last_completion(t.id)
        last_str = last.completed_at.strftime("%Y-%m-%d") if last else "never"
        print(f"  {t.id:30s} next: {nxt.strftime('%b %d')}  last: {last_str}")
        print(f"    {t.name}")
    print()


def cmd_add(cal: AgentCalendar, args: argparse.Namespace) -> None:
    """Add a new task."""
    tags = args.tags.split(",") if args.tags else []
    priority = Priority(args.priority) if args.priority else Priority.MEDIUM

    task = CalendarTask(
        id=args.task_id,
        name=args.name,
        recurrence=args.recurrence,
        tags=tags,
        priority=priority,
        description=args.description or "",
    )
    cal.add(task)
    cal.save()
    print(f"\n  Added: {task.name}")
    print(f"  ID:         {task.id}")
    print(f"  Recurrence: {task.recurrence}")
    print(f"  Next:       {task.next_occurrence().strftime('%Y-%m-%d %H:%M')}")
    print()


def cmd_log(cal: AgentCalendar, args: argparse.Namespace) -> None:
    """Show recent completions across all tasks."""
    limit = args.limit or 20
    records = cal.all_completions(limit=limit)

    print(f"\n  Recent completions ({len(records)}):\n")
    for r in records:
        task = cal.get(r.task_id)
        name = task.name if task else r.task_id
        outcome_icon = {
            "completed": "+", "partial": "~", "skipped": "-",
            "failed": "x", "deferred": ">",
        }.get(r.outcome.value, "?")
        print(f"  [{outcome_icon}] {r.completed_at.strftime('%Y-%m-%d %H:%M')}  {name}")
        if r.notes:
            print(f"      {r.notes}")
    print()


def _print_task(cal: AgentCalendar, task: CalendarTask, now: datetime) -> None:
    """Print a single task with context."""
    last = cal.last_completion(task.id)
    last_str = f"last: {last.completed_at.strftime('%b %d')}" if last else "never completed"
    nxt = task.next_occurrence(now)
    priority_marker = {"critical": "!!", "high": "! ", "medium": "  ", "low": "  "}
    marker = priority_marker.get(task.priority.value, "  ")
    print(f"  {marker} {task.name}")
    print(f"      next: {nxt.strftime('%b %d %I:%M %p')}  |  {last_str}  |  {', '.join(task.tags)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Farm Calendar — track and complete farm tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # Default (no subcommand) = due
    subparsers.add_parser("due", help="What's due now (default)")

    p_upcoming = subparsers.add_parser("upcoming", help="Upcoming tasks")
    p_upcoming.add_argument("days", type=int, nargs="?", default=7)

    subparsers.add_parser("all", help="List all tasks")

    p_complete = subparsers.add_parser("complete", help="Record a completion")
    p_complete.add_argument("task_id")
    p_complete.add_argument("--notes", "-n", default="")
    p_complete.add_argument("--outcome", "-o", default="completed",
                            choices=["completed", "partial", "skipped", "deferred", "failed"])

    p_history = subparsers.add_parser("history", help="Task completion history")
    p_history.add_argument("task_id")
    p_history.add_argument("--limit", "-l", type=int, default=20)

    p_streak = subparsers.add_parser("streak", help="Streak stats")
    p_streak.add_argument("task_id")

    subparsers.add_parser("tags", help="List all tags")

    p_tag = subparsers.add_parser("tag", help="Tasks by tag")
    p_tag.add_argument("tag")

    p_add = subparsers.add_parser("add", help="Add a new task")
    p_add.add_argument("task_id")
    p_add.add_argument("name")
    p_add.add_argument("recurrence")
    p_add.add_argument("--tags", "-t", default="")
    p_add.add_argument("--priority", "-p", default="medium",
                        choices=["low", "medium", "high", "critical"])
    p_add.add_argument("--description", "-d", default="")

    p_log = subparsers.add_parser("log", help="Recent completions")
    p_log.add_argument("--limit", "-l", type=int, default=20)

    args = parser.parse_args()
    cal = get_calendar()

    commands = {
        None: cmd_due,
        "due": cmd_due,
        "upcoming": cmd_upcoming,
        "all": cmd_all,
        "complete": cmd_complete,
        "history": cmd_history,
        "streak": cmd_streak,
        "tags": cmd_tags,
        "tag": cmd_tag,
        "add": cmd_add,
        "log": cmd_log,
    }

    handler = commands.get(args.command, cmd_due)
    handler(cal, args)


if __name__ == "__main__":
    main()
