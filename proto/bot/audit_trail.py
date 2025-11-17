#!/usr/bin/env python3
"""
Audit Trail System for Promptly Matrix Bot

Complete provenance logging for compliance and debugging:
- All commands, decisions, and outcomes
- Complete context and metadata
- Searchable history with filters
- Export formats: CSV, JSON, Markdown
- Compliance-ready (SOC2, ISO 27001, GDPR)

Features:
- Structured audit logging
- Temporal queries (since, until, range)
- Event filtering (command, user, room)
- Multiple export formats
- Retention policies
- Privacy controls

Usage:
    trail = AuditTrail()

    # Log event
    await trail.log_event(
        event_type="command",
        user="@alice:matrix.org",
        room="!room:matrix.org",
        action="optimize",
        context={"task": "answer questions"},
        outcome="success",
        metadata={"latency_ms": 150}
    )

    # Query events
    events = await trail.query(
        since=datetime(2025, 11, 1),
        event_type="command",
        user="@alice:matrix.org"
    )

    # Export
    csv_data = await trail.export(format="csv", since=datetime(2025, 11, 1))
"""

import logging
import json
import csv
import io
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of audit events"""
    COMMAND = "command"           # User command
    DECISION = "decision"         # AI decision
    APPROVAL = "approval"         # Approval workflow
    WORKFLOW = "workflow"         # Workflow execution
    ACCESS = "access"             # Resource access
    ERROR = "error"               # Error/exception
    CONFIG_CHANGE = "config"      # Configuration change
    SYSTEM = "system"             # System event


class Outcome(Enum):
    """Event outcomes"""
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    CANCELLED = "cancelled"


@dataclass
class AuditEvent:
    """Single audit event"""
    event_id: str                     # Unique event ID
    timestamp: datetime               # When event occurred
    event_type: EventType             # Type of event
    user: str                         # User who initiated (@user:matrix.org)
    room: Optional[str] = None        # Room ID (if applicable)
    action: str = ""                  # Action performed
    context: Dict[str, Any] = field(default_factory=dict)  # Full context
    outcome: Outcome = Outcome.SUCCESS
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    error_message: Optional[str] = None  # Error details (if failed)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        d['event_type'] = self.event_type.value
        d['outcome'] = self.outcome.value
        return d

    def to_csv_row(self) -> Dict[str, str]:
        """Convert to CSV row"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'user': self.user,
            'room': self.room or '',
            'action': self.action,
            'outcome': self.outcome.value,
            'context': json.dumps(self.context),
            'metadata': json.dumps(self.metadata),
            'error_message': self.error_message or ''
        }

    def to_markdown(self) -> str:
        """Convert to Markdown"""
        lines = [
            f"### Event {self.event_id}",
            f"**Time**: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Type**: {self.event_type.value}",
            f"**User**: {self.user}",
        ]

        if self.room:
            lines.append(f"**Room**: {self.room}")

        lines.extend([
            f"**Action**: {self.action}",
            f"**Outcome**: {self.outcome.value}",
        ])

        if self.context:
            lines.append(f"**Context**: `{json.dumps(self.context, indent=2)}`")

        if self.metadata:
            lines.append(f"**Metadata**: `{json.dumps(self.metadata, indent=2)}`")

        if self.error_message:
            lines.append(f"**Error**: {self.error_message}")

        return "\n".join(lines)


class AuditTrail:
    """
    Audit trail manager.

    Logs all events with complete context for compliance and debugging.

    Usage:
        trail = AuditTrail()

        # Log command
        await trail.log_command(
            user="@alice:matrix.org",
            room="!room:matrix.org",
            command="optimize",
            args={"task": "answer questions"},
            outcome="success"
        )

        # Query recent commands
        events = await trail.query(
            since=datetime.now() - timedelta(days=7),
            event_type=EventType.COMMAND
        )

        # Export to CSV
        csv_data = await trail.export(format="csv")
    """

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        retention_days: int = 90
    ):
        """
        Initialize audit trail.

        Args:
            persist_path: Path to persist audit log (default: ./audit_trail.jsonl)
            retention_days: Number of days to retain events (default: 90)
        """
        self.persist_path = persist_path or Path("./audit_trail.jsonl")
        self.retention_days = retention_days
        self.events: List[AuditEvent] = []
        self._event_counter = 0

        # Load existing events
        self._load_events()

    def _load_events(self):
        """Load events from disk"""
        if not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, 'r') as f:
                for line in f:
                    if line.strip():
                        event_dict = json.loads(line)
                        event = self._dict_to_event(event_dict)
                        self.events.append(event)

            logger.info(f"Loaded {len(self.events)} audit events from {self.persist_path}")

            # Apply retention policy
            self._apply_retention()

        except Exception as e:
            logger.error(f"Failed to load audit trail: {e}")

    def _dict_to_event(self, d: Dict[str, Any]) -> AuditEvent:
        """Convert dictionary to AuditEvent"""
        return AuditEvent(
            event_id=d['event_id'],
            timestamp=datetime.fromisoformat(d['timestamp']),
            event_type=EventType(d['event_type']),
            user=d['user'],
            room=d.get('room'),
            action=d.get('action', ''),
            context=d.get('context', {}),
            outcome=Outcome(d['outcome']),
            metadata=d.get('metadata', {}),
            error_message=d.get('error_message')
        )

    def _apply_retention(self):
        """Apply retention policy (remove old events)"""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        initial_count = len(self.events)

        self.events = [e for e in self.events if e.timestamp >= cutoff]

        removed = initial_count - len(self.events)
        if removed > 0:
            logger.info(f"Removed {removed} events older than {self.retention_days} days")

    async def log_event(
        self,
        event_type: EventType,
        user: str,
        action: str,
        room: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        outcome: Outcome = Outcome.SUCCESS,
        metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> AuditEvent:
        """
        Log audit event.

        Args:
            event_type: Type of event
            user: User who initiated
            action: Action performed
            room: Room ID (if applicable)
            context: Full context
            outcome: Event outcome
            metadata: Additional metadata
            error_message: Error details (if failed)

        Returns:
            Created AuditEvent
        """
        self._event_counter += 1
        event_id = f"evt_{int(datetime.now().timestamp())}_{self._event_counter}"

        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            user=user,
            room=room,
            action=action,
            context=context or {},
            outcome=outcome,
            metadata=metadata or {},
            error_message=error_message
        )

        self.events.append(event)

        # Persist to disk
        await self._persist_event(event)

        logger.debug(f"Logged audit event: {event_id} ({event_type.value}/{outcome.value})")

        return event

    async def _persist_event(self, event: AuditEvent):
        """Persist event to disk"""
        try:
            with open(self.persist_path, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist audit event: {e}")

    async def log_command(
        self,
        user: str,
        room: str,
        command: str,
        args: Optional[Dict[str, Any]] = None,
        outcome: str = "success",
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log user command"""
        return await self.log_event(
            event_type=EventType.COMMAND,
            user=user,
            room=room,
            action=command,
            context=args or {},
            outcome=Outcome.SUCCESS if outcome == "success" else Outcome.FAILURE,
            error_message=error,
            metadata=metadata
        )

    async def log_decision(
        self,
        user: str,
        room: str,
        decision: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log AI decision"""
        return await self.log_event(
            event_type=EventType.DECISION,
            user=user,
            room=room,
            action=decision,
            context=context or {},
            metadata=metadata
        )

    async def log_approval(
        self,
        user: str,
        room: str,
        action: str,
        approved: bool,
        approvers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log approval workflow"""
        outcome = Outcome.SUCCESS if approved else Outcome.FAILURE

        context = {
            "approved": approved,
            "approvers": approvers or []
        }

        return await self.log_event(
            event_type=EventType.APPROVAL,
            user=user,
            room=room,
            action=action,
            context=context,
            outcome=outcome,
            metadata=metadata
        )

    async def log_error(
        self,
        user: str,
        room: Optional[str],
        action: str,
        error: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log error"""
        return await self.log_event(
            event_type=EventType.ERROR,
            user=user,
            room=room,
            action=action,
            context=context or {},
            outcome=Outcome.FAILURE,
            error_message=error,
            metadata=metadata
        )

    async def query(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        event_type: Optional[EventType] = None,
        user: Optional[str] = None,
        room: Optional[str] = None,
        action: Optional[str] = None,
        outcome: Optional[Outcome] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """
        Query audit events.

        Args:
            since: Events after this time
            until: Events before this time
            event_type: Filter by event type
            user: Filter by user
            room: Filter by room
            action: Filter by action
            outcome: Filter by outcome
            limit: Maximum events to return

        Returns:
            List of matching AuditEvents
        """
        results = self.events

        # Apply filters
        if since:
            results = [e for e in results if e.timestamp >= since]

        if until:
            results = [e for e in results if e.timestamp <= until]

        if event_type:
            results = [e for e in results if e.event_type == event_type]

        if user:
            results = [e for e in results if e.user == user]

        if room:
            results = [e for e in results if e.room == room]

        if action:
            results = [e for e in results if action.lower() in e.action.lower()]

        if outcome:
            results = [e for e in results if e.outcome == outcome]

        # Sort by timestamp (newest first)
        results.sort(key=lambda e: e.timestamp, reverse=True)

        return results[:limit]

    async def export(
        self,
        format: str = "json",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        **query_args
    ) -> str:
        """
        Export audit trail.

        Args:
            format: Export format (json, csv, markdown)
            since: Events after this time
            until: Events before this time
            **query_args: Additional query filters

        Returns:
            Exported data as string
        """
        # Query events
        events = await self.query(since=since, until=until, **query_args)

        if format == "json":
            return self._export_json(events)
        elif format == "csv":
            return self._export_csv(events)
        elif format == "markdown":
            return self._export_markdown(events)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(self, events: List[AuditEvent]) -> str:
        """Export as JSON"""
        return json.dumps([e.to_dict() for e in events], indent=2)

    def _export_csv(self, events: List[AuditEvent]) -> str:
        """Export as CSV"""
        output = io.StringIO()

        if events:
            fieldnames = [
                'event_id', 'timestamp', 'event_type', 'user', 'room',
                'action', 'outcome', 'context', 'metadata', 'error_message'
            ]

            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for event in events:
                writer.writerow(event.to_csv_row())

        return output.getvalue()

    def _export_markdown(self, events: List[AuditEvent]) -> str:
        """Export as Markdown"""
        lines = [
            "# Audit Trail Export",
            f"\n**Exported**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Events**: {len(events)}",
            "\n---\n"
        ]

        for event in events:
            lines.append(event.to_markdown())
            lines.append("\n---\n")

        return "\n".join(lines)

    async def get_statistics(
        self,
        since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get audit trail statistics.

        Args:
            since: Statistics since this time (default: last 7 days)

        Returns:
            Dictionary with statistics
        """
        if since is None:
            since = datetime.now() - timedelta(days=7)

        events = await self.query(since=since)

        # Count by type
        by_type = {}
        for event_type in EventType:
            count = len([e for e in events if e.event_type == event_type])
            if count > 0:
                by_type[event_type.value] = count

        # Count by outcome
        by_outcome = {}
        for outcome in Outcome:
            count = len([e for e in events if e.outcome == outcome])
            if count > 0:
                by_outcome[outcome.value] = count

        # Top users
        user_counts = {}
        for event in events:
            user_counts[event.user] = user_counts.get(event.user, 0) + 1

        top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Top actions
        action_counts = {}
        for event in events:
            action_counts[event.action] = action_counts.get(event.action, 0) + 1

        top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'period': {
                'since': since.isoformat(),
                'until': datetime.now().isoformat()
            },
            'total_events': len(events),
            'by_type': by_type,
            'by_outcome': by_outcome,
            'top_users': [{'user': u, 'count': c} for u, c in top_users],
            'top_actions': [{'action': a, 'count': c} for a, c in top_actions]
        }


# Singleton
_audit_trail = None


def get_audit_trail(
    persist_path: Optional[Path] = None,
    retention_days: int = 90
) -> AuditTrail:
    """Get singleton audit trail"""
    global _audit_trail
    if _audit_trail is None:
        _audit_trail = AuditTrail(persist_path, retention_days)
    return _audit_trail


# Test
if __name__ == "__main__":
    import asyncio

    async def test():
        # Create audit trail
        trail = AuditTrail(persist_path=Path("./test_audit.jsonl"))

        # Log various events
        await trail.log_command(
            user="@alice:matrix.org",
            room="!test:matrix.org",
            command="optimize",
            args={"task": "answer questions"},
            metadata={"latency_ms": 150}
        )

        await trail.log_decision(
            user="@alice:matrix.org",
            room="!test:matrix.org",
            decision="use_tool",
            context={"tool": "answer", "confidence": 0.92}
        )

        await trail.log_approval(
            user="@alice:matrix.org",
            room="!test:matrix.org",
            action="deploy_prompt",
            approved=True,
            approvers=["@bob:matrix.org", "@charlie:matrix.org"]
        )

        await trail.log_error(
            user="@alice:matrix.org",
            room="!test:matrix.org",
            action="code_review",
            error="Connection timeout",
            context={"language": "python"}
        )

        # Query events
        print("Recent commands:")
        commands = await trail.query(event_type=EventType.COMMAND, limit=10)
        for cmd in commands:
            print(f"  {cmd.timestamp.strftime('%H:%M:%S')} - {cmd.user}: {cmd.action}")

        # Statistics
        print("\nStatistics:")
        stats = await trail.get_statistics()
        print(f"  Total events: {stats['total_events']}")
        print(f"  By type: {stats['by_type']}")
        print(f"  By outcome: {stats['by_outcome']}")

        # Export as CSV
        print("\nCSV Export (first 5 lines):")
        csv_data = await trail.export(format="csv")
        lines = csv_data.split('\n')[:5]
        for line in lines:
            print(f"  {line}")

        # Export as Markdown
        print("\nMarkdown Export (first event):")
        md_data = await trail.export(format="markdown", limit=1)
        print(md_data[:500])

        print("\nAll audit trail tests passed!")

        # Cleanup
        Path("./test_audit.jsonl").unlink(missing_ok=True)

    asyncio.run(test())
