"""
Chronos CLI - Voice of the time tracking core

The 5 verbs: start, stop, log, note, link
"""

import click
from pathlib import Path
from typing import Optional

from .core import ChronosState


# Default log location
DEFAULT_LOG_PATH = Path.home() / ".chronos" / "events.jsonl"


def get_state(log_path: Optional[Path] = None) -> ChronosState:
    """Get Chronos state instance"""
    if log_path is None:
        log_path = DEFAULT_LOG_PATH
    return ChronosState(log_path)


@click.group()
@click.option(
    '--log',
    type=click.Path(),
    help=f'Path to event log (default: {DEFAULT_LOG_PATH})'
)
@click.pass_context
def cli(ctx, log):
    """Chronos - Simple time tracking for human activities"""
    ctx.ensure_object(dict)
    ctx.obj['log_path'] = Path(log) if log else DEFAULT_LOG_PATH


@cli.command()
@click.argument('task')
@click.option('--tag', '-t', multiple=True, help='Tag(s) for this task')
@click.pass_context
def start(ctx, task: str, tag: tuple):
    """
    Start tracking a task

    Examples:
        chronos start "compost_sifting"
        chronos start "email_review" --tag admin --tag urgent
    """
    state = get_state(ctx.obj['log_path'])
    msg = state.start(task, tags=list(tag))
    click.echo(msg)


@cli.command()
@click.argument('task', required=False)
@click.pass_context
def stop(ctx, task: Optional[str]):
    """
    Stop tracking current task (or specified task)

    Examples:
        chronos stop
        chronos stop "compost_sifting"
    """
    state = get_state(ctx.obj['log_path'])
    msg = state.stop(task_name=task)
    click.echo(msg)


@cli.command()
@click.argument('task')
@click.option('--duration', '-d', required=True, help='Duration (e.g., "20m", "1.5h", "90s")')
@click.option('--tag', '-t', multiple=True, help='Tag(s) for this task')
@click.pass_context
def log(ctx, task: str, duration: str, tag: tuple):
    """
    Log a completed task (retroactive entry)

    Examples:
        chronos log "email_review" --duration 20m
        chronos log "planning_meeting" --duration 1.5h --tag admin
    """
    state = get_state(ctx.obj['log_path'])

    # Parse duration string
    duration_sec = parse_duration(duration)
    if duration_sec is None:
        click.echo(f"❌ Invalid duration format: {duration}")
        click.echo("   Use: 30s, 20m, 1.5h")
        return

    msg = state.log(task, duration_sec, tags=list(tag))
    click.echo(msg)


@cli.command()
@click.argument('text')
@click.option('--link-to', help='Event ID to link to (default: current active task)')
@click.pass_context
def note(ctx, text: str, link_to: Optional[str]):
    """
    Add a note to a task

    Examples:
        chronos note "Compost ready for spring beds"
        chronos note "Remember to check moisture" --link-to chr_0042
    """
    state = get_state(ctx.obj['log_path'])
    msg = state.note(text, linked_to=link_to)
    click.echo(msg)


@cli.command()
@click.argument('from_id')
@click.argument('to_id')
@click.option('--relation', '-r', default='related_to', help='Relationship type')
@click.pass_context
def link(ctx, from_id: str, to_id: str, relation: str):
    """
    Link an event to another entity

    Examples:
        chronos link chr_0042 task_garden_prep --relation part_of
        chronos link chr_0015 chr_0020 --relation caused_by
    """
    state = get_state(ctx.obj['log_path'])
    msg = state.link(from_id, to_id, relation)
    click.echo(msg)


@cli.command()
@click.pass_context
def status(ctx):
    """
    Show current status (active task)

    Examples:
        chronos status
    """
    state = get_state(ctx.obj['log_path'])
    msg = state.status()
    click.echo(msg)


@cli.command()
@click.option('--date', help='Show events for specific date (YYYY-MM-DD)')
@click.option('--limit', '-n', type=int, help='Limit number of events shown')
@click.pass_context
def history(ctx, date: Optional[str], limit: Optional[int]):
    """
    Show event history

    Examples:
        chronos history
        chronos history --date 2025-11-06
        chronos history --limit 10
    """
    state = get_state(ctx.obj['log_path'])

    if date:
        events = state.event_log.read_by_date(date)
        click.echo(f"\n📅 Events for {date}:")
    else:
        events = state.event_log.read_all()
        click.echo(f"\n📜 Event history:")

    if not events:
        click.echo("  (no events)")
        return

    # Apply limit
    if limit:
        events = events[-limit:]

    # Display events
    for event in events:
        event_dict = event.to_dict()
        click.echo(f"\n  [{event.id}] {event.event.upper()}")

        for key, value in event_dict.items():
            if key not in ['id', 'event']:
                if key == 'ts':
                    # Format timestamp
                    from datetime import datetime
                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    value = dt.strftime("%Y-%m-%d %H:%M:%S")

                click.echo(f"    {key}: {value}")

    click.echo()


@cli.command()
@click.argument('transcript')
@click.pass_context
def voice(ctx, transcript: str):
    """
    Parse voice command and execute

    Examples:
        chronos voice "start compost sifting"
        chronos voice "done"
        chronos voice "note compost ready for spring beds"
    """
    from .voice import VoiceCommandParser, format_confirmation

    parser = VoiceCommandParser()
    command = parser.parse(transcript)

    if not command:
        click.echo(f"❌ Couldn't understand command: '{transcript}'")
        return

    # Show what we understood (confirmation)
    confirmation = format_confirmation(command)
    click.echo(f"🎤 {confirmation}")

    # Execute the command
    state = get_state(ctx.obj['log_path'])
    verb = command["verb"]
    args = command["args"]

    if verb == "start":
        msg = state.start(args["task"], tags=args.get("tags"))
        click.echo(msg)

    elif verb == "stop":
        msg = state.stop()
        click.echo(msg)

    elif verb == "log":
        msg = state.log(args["task"], args["duration_sec"])
        click.echo(msg)

    elif verb == "note":
        msg = state.note(args["text"])
        click.echo(msg)

    elif verb == "status":
        msg = state.status()
        click.echo(msg)


# ═══════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════

def parse_duration(duration_str: str) -> Optional[float]:
    """
    Parse duration string to seconds

    Examples:
        "30s" → 30
        "20m" → 1200
        "1.5h" → 5400
    """
    duration_str = duration_str.strip().lower()

    try:
        if duration_str.endswith('s'):
            return float(duration_str[:-1])
        elif duration_str.endswith('m'):
            return float(duration_str[:-1]) * 60
        elif duration_str.endswith('h'):
            return float(duration_str[:-1]) * 3600
        else:
            # Try parsing as seconds
            return float(duration_str)
    except ValueError:
        return None


if __name__ == '__main__':
    cli(obj={})
