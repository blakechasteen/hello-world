"""Pollinator — cross-pollinates findings between active sessions.

When one agent discovers something interesting, the pollinator shares
it with other active agents so they can investigate from their own
perspective. A bug found by the adversarial-tester becomes a
verification task for the power-user.

This turns isolated sessions into a collaborative investigation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import Manager
from .events import Event, EventBus, EventKind


@dataclass
class Finding:
    """A finding worth sharing across sessions."""

    description: str
    source_persona: str
    source_session: str
    kind: str  # "issue" | "confusion"
    shared_with: set[str] = field(default_factory=set)  # session IDs


@dataclass
class PollinatorConfig:
    """Knobs for cross-pollination."""

    enabled: bool = True
    min_agents_for_sharing: int = 2  # need at least this many active
    max_shares_per_finding: int = 3  # don't spam every agent
    share_issues: bool = True
    share_confusion: bool = False  # confusion is often persona-specific
    cooldown_events: int = 2  # don't share back-to-back


class Pollinator(Manager):
    """Cross-pollinates findings between active sessions."""

    def __init__(
        self,
        bus: EventBus,
        config: PollinatorConfig | None = None,
    ):
        super().__init__(bus, backend=None)
        self.config = config or PollinatorConfig()
        self._findings: list[Finding] = []
        self._active_sessions: dict[str, str] = {}  # session_id → persona
        self._events_since_last_share: int = 0

    @property
    def name(self) -> str:
        return "pollinator"

    def attach(self) -> None:
        self.bus.on(EventKind.SESSION_START, self._on_start)
        self.bus.on(EventKind.SESSION_END, self._on_end)
        if self.config.share_issues:
            self.bus.on(EventKind.ISSUE_FOUND, self._on_finding)
        if self.config.share_confusion:
            self.bus.on(EventKind.CONFUSION_FOUND, self._on_finding)
        self._attached = True

    async def _on_start(self, event: Event) -> None:
        self._active_sessions[event.session_id] = event.persona

    async def _on_end(self, event: Event) -> None:
        self._active_sessions.pop(event.session_id, None)

    async def _on_finding(self, event: Event) -> None:
        if not self.config.enabled:
            return

        self._events_since_last_share += 1

        description = event.data.get("description", "")
        if not description:
            return

        # Record the finding.
        finding_kind = "issue" if event.kind == EventKind.ISSUE_FOUND else "confusion"
        finding = Finding(
            description=description,
            source_persona=event.persona,
            source_session=event.session_id,
            kind=finding_kind,
        )
        self._findings.append(finding)

        # Share with other active sessions.
        if (
            len(self._active_sessions) < self.config.min_agents_for_sharing
            or self._events_since_last_share < self.config.cooldown_events
        ):
            return

        self._events_since_last_share = 0
        await self._share(finding)

    async def _share(self, finding: Finding) -> None:
        """Share a finding with other active sessions."""
        targets = [
            sid for sid, persona in self._active_sessions.items()
            if sid != finding.source_session
            and sid not in finding.shared_with
        ]

        # Limit shares.
        targets = targets[:self.config.max_shares_per_finding]

        for sid in targets:
            finding.shared_with.add(sid)
            target_persona = self._active_sessions.get(sid, "unknown")

            message = (
                f"[CROSS-POLLINATION] Another tester ({finding.source_persona}) "
                f"found: {finding.description}\n"
                f"Can you verify or investigate this from your perspective?"
            )

            await self._emit(
                EventKind.INJECT_PROMPT,
                session_id=sid,
                persona=target_persona,
                data={
                    "message": message,
                    "type": "cross_pollination",
                    "source_persona": finding.source_persona,
                    "finding": finding.description,
                },
            )

    @property
    def sharing_stats(self) -> dict[str, Any]:
        """How much cross-pollination has occurred."""
        total_findings = len(self._findings)
        total_shares = sum(len(f.shared_with) for f in self._findings)
        return {
            "total_findings": total_findings,
            "total_shares": total_shares,
            "avg_shares_per_finding": round(total_shares / max(total_findings, 1), 1),
        }
