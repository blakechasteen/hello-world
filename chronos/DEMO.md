# Chronos Demo

Quick demonstration of Chronos Phase 0 functionality.

## Manual CLI Demo

```bash
# Start a task
$ python -m chronos start "garden_work" --tag farm --tag physical
✅ Started garden_work at 14:30:00 — tagged #farm #physical
   (event chr_0001)

# Check status
$ python -m chronos status
⏱  Active: garden_work (started 14:30:00, elapsed 15m 30s)

# Add a note while working
$ python -m chronos note "Soil is quite dry, need to water tomorrow"
✅ Note added to garden_work
   (event chr_0002)

# Stop the task
$ python -m chronos stop
✅ Stopped garden_work at 14:45:30 — duration 15m 30s
   (event chr_0003)

# Log something you did earlier
$ python -m chronos log "email_review" --duration 20m --tag admin
✅ Logged email_review, 20m 0s — tagged #admin
   (event chr_0004)

# View history
$ python -m chronos history --limit 5

📜 Event history:

  [chr_0001] START
    task: garden_work
    tags: ['farm', 'physical']
    ts: 2025-11-06 14:30:00

  [chr_0002] NOTE
    text: Soil is quite dry, need to water tomorrow
    linked_to: chr_0001
    ts: 2025-11-06 14:40:15

  [chr_0003] STOP
    task: garden_work
    duration_sec: 930
    start_id: chr_0001
    ts: 2025-11-06 14:45:30

  [chr_0004] LOG
    task: email_review
    duration_sec: 1200
    tags: ['admin']
    ts: 2025-11-06 14:00:00
```

## Voice Command Demo

```bash
# Start via voice
$ python -m chronos voice "start compost sifting"
🎤 Starting compost sifting. Say 'done' when finished.
✅ Started compost_sifting at 15:00:00
   (event chr_0005)

# Add note via voice
$ python -m chronos voice "note: found good earthworms"
🎤 Note added.
✅ Note added to compost_sifting
   (event chr_0006)

# Stop via voice
$ python -m chronos voice "done"
🎤 Stopping current task.
✅ Stopped compost_sifting at 15:25:00 — duration 25m 0s
   (event chr_0007)

# Log via voice
$ python -m chronos voice "log phone call for 15 minutes"
🎤 Logging phone call for 15 minutes.
✅ Logged phone_call, 15m 0s
   (event chr_0008)
```

## Programmatic API Demo

```python
from pathlib import Path
from chronos.core import ChronosState

# Initialize
chronos = ChronosState(Path.home() / ".chronos" / "events.jsonl")

# Start tracking
chronos.start("coding", tags=["work", "python"])
# ✅ Started coding at 16:00:00 — tagged #work #python

# Add context
chronos.note("Working on Chronos Phase 0 implementation")
# ✅ Note added to coding

# Stop
chronos.stop()
# ✅ Stopped coding at 16:45:00 — duration 45m 0s

# Read events
events = chronos.event_log.read_all()
for event in events[-3:]:
    print(f"{event.id}: {event.event} - {event.task}")
# chr_0009: start - coding
# chr_0010: note - None
# chr_0011: stop - coding
```

## Data Format Demo

After running the above commands, `~/.chronos/events.jsonl` contains:

```jsonl
{"event":"start","task":"garden_work","tags":["farm","physical"],"ts":"2025-11-06T14:30:00Z","id":"chr_0001"}
{"event":"note","text":"Soil is quite dry, need to water tomorrow","linked_to":"chr_0001","ts":"2025-11-06T14:40:15Z","id":"chr_0002"}
{"event":"stop","task":"garden_work","ts":"2025-11-06T14:45:30Z","duration_sec":930,"start_id":"chr_0001","id":"chr_0003"}
{"event":"log","task":"email_review","duration_sec":1200,"tags":["admin"],"ts":"2025-11-06T14:00:00Z","id":"chr_0004"}
{"event":"start","task":"compost_sifting","ts":"2025-11-06T15:00:00Z","id":"chr_0005"}
{"event":"note","text":"found good earthworms","linked_to":"chr_0005","ts":"2025-11-06T15:10:00Z","id":"chr_0006"}
{"event":"stop","task":"compost_sifting","ts":"2025-11-06T15:25:00Z","duration_sec":1500,"start_id":"chr_0005","id":"chr_0007"}
{"event":"log","task":"phone_call","duration_sec":900,"ts":"2025-11-06T15:10:00Z","id":"chr_0008"}
{"event":"start","task":"coding","tags":["work","python"],"ts":"2025-11-06T16:00:00Z","id":"chr_0009"}
{"event":"note","text":"Working on Chronos Phase 0 implementation","linked_to":"chr_0009","ts":"2025-11-06T16:20:00Z","id":"chr_0010"}
{"event":"stop","task":"coding","ts":"2025-11-06T16:45:00Z","duration_sec":2700,"start_id":"chr_0009","id":"chr_0011"}
```

**Plain text ✓ Human-readable ✓ Append-only ✓**

## iOS Shortcut Example

Create a Siri Shortcut named "Track Time":

1. **Trigger**: "Hey Siri, track time"
2. **Action 1**: Show Menu
   - Options: "Start Task", "Stop Task", "Add Note"
3. **If "Start Task"**:
   - Dictate Text (store as `taskName`)
   - Run Script Over SSH:
     ```bash
     python -m chronos voice "start {taskName}"
     ```
   - Show notification with result
4. **If "Stop Task"**:
   - Run Script Over SSH:
     ```bash
     python -m chronos voice "done"
     ```
   - Show notification with result
5. **If "Add Note"**:
   - Dictate Text (store as `noteText`)
   - Run Script Over SSH:
     ```bash
     python -m chronos voice "note: {noteText}"
     ```
   - Show notification with result

Usage:
- "Hey Siri, track time" → Select "Start Task" → Say "garden work"
- ✅ Started garden_work at 14:30:00
- (Later) "Hey Siri, track time" → Select "Stop Task"
- ✅ Stopped garden_work at 15:00:00 — duration 30m 0s

## Next Steps

After Phase 0, you can:
- Set up iOS Shortcuts for voice tracking
- Create Matrix bot for team time tracking
- Add daily summary generator
- Build analytics dashboard
- Integrate with HoloLoom (optional)

But the core is complete, simple, and works offline today.
