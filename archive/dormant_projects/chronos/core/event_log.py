"""
Event Log - Append-only time tracking ledger

Never edits. Never deletes. Only appends.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone


@dataclass
class ChronosEvent:
    """One event in the timeline"""

    event: str  # "start" | "stop" | "log" | "note" | "link"
    ts: str  # ISO 8601 timestamp
    id: str  # Unique event ID

    # Event-specific fields
    task: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    duration_sec: Optional[float] = None
    start_id: Optional[str] = None
    text: Optional[str] = None
    linked_to: Optional[str] = None
    from_id: Optional[str] = None
    to_id: Optional[str] = None
    relation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict, omitting None values"""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != []}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChronosEvent":
        """Create event from dict"""
        return cls(**d)


class EventLog:
    """Append-only event log stored as .jsonl"""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._event_counter = self._get_last_event_number()

    def _get_last_event_number(self) -> int:
        """Get the last event number from existing log"""
        if not self.log_path.exists():
            return 0

        with open(self.log_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                return 0

            last_line = lines[-1]
            try:
                event = json.loads(last_line)
                event_id = event.get('id', 'chr_0')
                return int(event_id.split('_')[1])
            except (json.JSONDecodeError, ValueError, IndexError):
                return 0

    def _next_id(self) -> str:
        """Generate next event ID"""
        self._event_counter += 1
        return f"chr_{self._event_counter:04d}"

    def _now_iso(self) -> str:
        """Current time in ISO 8601 format"""
        return datetime.now(timezone.utc).isoformat()

    def append(self, event: ChronosEvent) -> ChronosEvent:
        """
        Append event to log (validation happens before this)

        Returns the event with generated ID and timestamp if not set
        """
        if not event.id:
            event.id = self._next_id()
        if not event.ts:
            event.ts = self._now_iso()

        # Append to log file
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(event.to_dict()) + '\n')

        return event

    def read_all(self) -> List[ChronosEvent]:
        """Read all events from log"""
        if not self.log_path.exists():
            return []

        events = []
        with open(self.log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        event_dict = json.loads(line)
                        events.append(ChronosEvent.from_dict(event_dict))
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue

        return events

    def read_by_date(self, date: str) -> List[ChronosEvent]:
        """Read events for a specific date (YYYY-MM-DD)"""
        events = self.read_all()
        return [
            e for e in events
            if e.ts.startswith(date)
        ]

    def get_event(self, event_id: str) -> Optional[ChronosEvent]:
        """Get a specific event by ID"""
        events = self.read_all()
        for event in events:
            if event.id == event_id:
                return event
        return None
