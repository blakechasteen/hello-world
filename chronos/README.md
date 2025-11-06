# Chronos - Simple Time Tracking

> "The act of keeping time should never cost more time than it saves."

Chronos is a minimal, reliable time tracking system for human activities. It works offline, stores data in plain text, and respects the principle of **simplicity first**.

## Philosophy

- **Append-only**: Never edits, never deletes. Only appends.
- **Plain text**: Human-readable `.jsonl` format
- **5 verbs**: `start`, `stop`, `log`, `note`, `link` - nothing more
- **Offline-first**: Works without network, no dependencies on external services
- **Confirmations**: Every command confirms what it understood
- **Voice-capable**: Simple command grammar works with speech-to-text

## Installation

No installation needed - Chronos is a simple Python package:

```bash
# Clone or copy the chronos directory
# Run directly:
python -m chronos --help
```

**Optional**: Install Click for CLI (not required for core functionality):
```bash
pip install click
```

## Quick Start

### Basic Usage

```bash
# Start tracking a task
python -m chronos start "compost_sifting" --tag farm --tag manual
# ✅ Started compost_sifting at 09:20:15 — tagged #farm #manual
#    (event chr_0001)

# Check status
python -m chronos status
# ⏱ Active: compost_sifting (started 09:20:15, elapsed 25m 10s)

# Stop task
python -m chronos stop
# ✅ Stopped compost_sifting at 09:45:25 — duration 25m 10s
#    (event chr_0002)
```

### Retroactive Logging

```bash
# Log a task you already completed
python -m chronos log "email_review" --duration 20m --tag admin
# ✅ Logged email_review, 20m 0s — tagged #admin
#    (event chr_0003)
```

### Adding Notes

```bash
# Add a note to current task
python -m chronos start "planning"
python -m chronos note "Decided to focus on Phase 1 first"
# ✅ Note added to planning
#    (event chr_0004)

# Add note to specific event
python -m chronos note "This was blocked by weather" --link-to chr_0001
# ✅ Note added to compost_sifting
#    (event chr_0005)
```

### Linking Events

```bash
# Link to external task management system
python -m chronos link chr_0001 task_garden_prep_2025 --relation part_of
# ✅ Linked chr_0001 → task_garden_prep_2025 (part_of)
#    (event chr_0006)
```

### Viewing History

```bash
# View all events
python -m chronos history

# View today's events
python -m chronos history --date 2025-11-06

# View last 5 events
python -m chronos history --limit 5
```

## Voice Commands

Chronos supports simple voice commands via the `voice` subcommand:

```bash
# Start a task via voice
python -m chronos voice "start compost sifting"
# 🎤 Starting compost sifting. Say 'done' when finished.
# ✅ Started compost_sifting at 09:20:15

# Stop via voice
python -m chronos voice "done"
# 🎤 Stopping current task.
# ✅ Stopped compost_sifting at 09:45:25 — duration 25m 10s

# Add note via voice
python -m chronos voice "note: compost ready for spring beds"
# 🎤 Note added.
# ✅ Note added to compost_sifting

# Log completed task via voice
python -m chronos voice "log email review for 20 minutes"
# 🎤 Logging email review for 20 minutes.
# ✅ Logged email_review, 20m 0s
```

### Voice Command Patterns

Chronos recognizes these natural language patterns:

| Intent | Examples |
|--------|----------|
| **Start** | "start task", "begin task", "starting task" |
| **Stop** | "stop", "done", "finished", "complete", "end" |
| **Note** | "note: text", "add note: text", "remember: text" |
| **Log** | "log task for 20 minutes", "log task for 1.5 hours" |
| **Status** | "status", "what's the active task" |

Tags can be included with hashtags:
```bash
python -m chronos voice "start compost sifting #farm #manual"
```

## Data Format

Events are stored as `.jsonl` (JSON Lines) in `~/.chronos/events.jsonl` by default.

Each line is a complete event:

```jsonl
{"event":"start","task":"compost_sifting","tags":["farm","manual"],"ts":"2025-11-06T09:20:15Z","id":"chr_0001"}
{"event":"stop","task":"compost_sifting","ts":"2025-11-06T09:45:25Z","duration_sec":1510,"start_id":"chr_0001","id":"chr_0002"}
{"event":"log","task":"email_review","duration_sec":1200,"tags":["admin"],"ts":"2025-11-06T08:00:00Z","id":"chr_0003"}
{"event":"note","text":"Compost ready for spring beds","linked_to":"chr_0001","ts":"2025-11-06T09:46:00Z","id":"chr_0004"}
{"event":"link","from_id":"chr_0001","to_id":"task_garden_prep_2025","relation":"part_of","ts":"2025-11-06T09:47:00Z","id":"chr_0005"}
```

**Human-readable ✓, Line-by-line ✓, Append-only ✓**

## The 5 Verbs

### 1. START
Begin tracking a task.

```bash
chronos start "task_name" [--tag TAG]
```

**Behavior**:
- Creates a `start` event
- If another task is active, auto-stops it first
- Returns event ID for reference

### 2. STOP
End the current task.

```bash
chronos stop [task_name]
```

**Behavior**:
- Creates a `stop` event with duration
- Links to the corresponding `start` event
- Clears active task

### 3. LOG
Record a completed task retroactively.

```bash
chronos log "task_name" --duration DURATION [--tag TAG]
```

**Duration formats**: `30s`, `20m`, `1.5h`

**Behavior**:
- Creates a `log` event with pre-calculated duration
- Does not affect active task
- Useful for manual time entry

### 4. NOTE
Add context to a task.

```bash
chronos note "text" [--link-to EVENT_ID]
```

**Behavior**:
- Creates a `note` event
- Links to active task by default
- Can link to any event via `--link-to`

### 5. LINK
Connect events to external entities.

```bash
chronos link FROM_ID TO_ID [--relation RELATION]
```

**Behavior**:
- Creates a `link` event
- Connects Chronos events to external systems
- Default relation: `related_to`

## Programmatic API

Use Chronos from Python code:

```python
from pathlib import Path
from chronos.core import ChronosState

# Initialize
log_path = Path.home() / ".chronos" / "events.jsonl"
chronos = ChronosState(log_path)

# Start tracking
msg = chronos.start("coding", tags=["work", "python"])
print(msg)  # ✅ Started coding at 10:00:00 — tagged #work #python

# Check status
status = chronos.status()
print(status)  # ⏱ Active: coding (started 10:00:00, elapsed 15m 30s)

# Stop tracking
msg = chronos.stop()
print(msg)  # ✅ Stopped coding at 10:15:30 — duration 15m 30s

# Log completed task
msg = chronos.log("meeting", duration_sec=3600, tags=["admin"])

# Add note
msg = chronos.note("Made good progress on feature X")

# Link to external system
events = chronos.event_log.read_all()
first_event_id = events[0].id
msg = chronos.link(first_event_id, "github_issue_123", relation="addresses")
```

## iOS Shortcut Integration

Create an iOS Shortcut for voice time tracking:

**Simple Version**:
1. Add "Dictate Text" action
2. Add "Run Script Over SSH" action:
   ```bash
   python -m chronos voice "{dictation}"
   ```
3. Add "Show Notification" with result

**Advanced Version** with menu:
1. Show menu: "Start Task", "Stop Task", "Add Note"
2. For each option, run appropriate `chronos` command
3. Show confirmation notification

## Design Principles

### 1. Separation of Concerns
- **Core**: Measurement (the event log)
- **Speaking Bodies**: Interfaces (CLI, voice, future: Matrix bot)
- **Analysis**: Meaning (summaries, patterns) comes later

### 2. Graceful Degradation
- Core functionality requires only Python standard library
- CLI requires `click` but core works without it
- Voice parsing is simple pattern matching (no ML dependencies)

### 3. Data Integrity
- Append-only log means no data loss
- Each event is self-contained (one JSON object per line)
- No external database required
- Easy to backup (just copy `.jsonl` file)

### 4. Human-Centered Design
- Every command confirms what it understood
- Timestamps in ISO 8601 (sortable, unambiguous)
- Task names normalized automatically ("compost sifting" → "compost_sifting")
- Friendly duration formatting ("1h 25m" instead of "5100 seconds")

## Future Phases

**Phase 0** (Current): Core + CLI + Voice
- ✓ Append-only event log
- ✓ 5 verbs
- ✓ CLI interface
- ✓ Voice command parsing

**Phase 1**: Speaking Bodies
- Matrix bot integration
- Slack/Discord bots
- Mobile apps

**Phase 2**: Reflection
- Daily summaries
- Gap detection
- Gentle nudges

**Phase 3**: Insight
- Cost analysis
- Energy patterns
- Flow metrics
- Integration with HoloLoom (optional)

## Testing

Run the test suite:

```bash
python chronos/tests/test_simple.py
```

All tests should pass:
```
==================================================
Chronos Core Tests
==================================================

Testing event log... ✓ PASS
Testing START/STOP... ✓ PASS
Testing LOG (retroactive)... ✓ PASS
Testing NOTE... ✓ PASS
Testing LINK... ✓ PASS
Testing persistence... ✓ PASS
Testing voice parser... ✓ PASS

==================================================
Results: 7/7 tests passed
✓ All tests passed!
```

## Architecture

```
chronos/
├── core/
│   ├── event_log.py      # Append-only .jsonl writer
│   └── state.py          # 5 verbs implementation
├── cli.py                # Click-based CLI
├── voice.py              # Voice command parser
└── tests/
    ├── test_core.py      # Comprehensive tests (requires pytest)
    └── test_simple.py    # Standalone tests (no dependencies)
```

**Core Principle**: The `core/` module has **zero external dependencies**. It will always work, even if CLI libraries are unavailable.

## FAQ

**Q: Why not use an existing time tracking tool?**
A: Most tools are either too complex (cloud sync, analytics, team features) or too tied to specific platforms. Chronos is intentionally minimal and works everywhere Python runs.

**Q: Why `.jsonl` instead of a database?**
A: JSON Lines is human-readable, version-control friendly, and works with standard Unix tools (`grep`, `awk`, etc.). No database setup, no migrations, no dependencies.

**Q: Can I edit or delete events?**
A: No. Chronos is append-only by design. To "correct" an error, add a note explaining it or log a corrective entry. This preserves complete history.

**Q: How do I backup my data?**
A: Copy `~/.chronos/events.jsonl` anywhere. That's it.

**Q: Can I use custom log locations?**
A: Yes: `python -m chronos --log /path/to/custom.jsonl start "task"`

**Q: Does this work offline?**
A: Yes! Chronos has no network dependencies. All data is local.

## License

MIT License - See LICENSE file for details

---

**Remember**: The act of keeping time should never cost more time than it saves.
