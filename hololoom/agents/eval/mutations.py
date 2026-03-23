"""
Mutation proposer — the autoresearch loop for agent identity evolution.

Watches rolling metrics and proposes changes to identity files when
evidence crosses thresholds defined in program.md.

Mutation velocity:
    tasks.md      — agent applies immediately (any observation)
    heartbeat.md  — agent applies immediately (metric shift > 0.02)
    tools.md      — flagged for review (tool success crosses boundary)
    program.md    — federation proposal (research question answered)
    soul.md       — federation proposal (red line misfit, N > 50)
    agent.md      — federation proposal (capability change)

The agent can never unilaterally weaken its own red lines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_IDENTITIES_ROOT = Path(__file__).resolve().parent.parent / "identities"


class MutationGate(str, Enum):
    """Who needs to approve this mutation."""
    AUTO = "auto"                # Agent applies (tasks.md, heartbeat.md)
    FLAG = "flag"                # Flagged for human review (tools.md)
    FEDERATION = "federation"    # Federation proposal (soul.md, agent.md, program.md)


class MutationVerdict(str, Enum):
    """Outcome of a mutation evaluation."""
    KEEP = "keep"
    DISCARD = "discard"
    CONTINUE = "continue"        # Ambiguous — need more evidence
    PENDING = "pending"          # Awaiting human review


@dataclass
class Mutation:
    """A proposed change to an identity file."""
    id: str
    agent_name: str
    target_file: str             # "soul.md", "tasks.md", etc.
    section: str                 # Which section to modify
    rationale: str               # Why the agent thinks this helps
    evidence: dict[str, Any]     # Metrics that motivated the change
    gate: MutationGate           # Who approves
    proposed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    verdict: MutationVerdict = MutationVerdict.PENDING
    delta_quality: float = 0.0   # Measured improvement after applying


# File → gate mapping
_FILE_GATES: dict[str, MutationGate] = {
    "tasks.md": MutationGate.AUTO,
    "heartbeat.md": MutationGate.AUTO,
    "tools.md": MutationGate.FLAG,
    "program.md": MutationGate.FEDERATION,
    "soul.md": MutationGate.FEDERATION,
    "agent.md": MutationGate.FEDERATION,
}


@dataclass
class MutationProposer:
    """
    Watches metrics and proposes identity mutations.

    The proposer doesn't apply mutations itself — it generates proposals.
    AUTO-gated mutations are written immediately. FEDERATION proposals
    are written to a proposals/ directory for human review.
    """
    agent_name: str
    root: Path = field(default_factory=lambda: _IDENTITIES_ROOT)
    proposals: list[Mutation] = field(default_factory=list)

    # Thresholds (from program.md defaults)
    metric_shift_threshold: float = 0.02
    tool_success_upper: float = 0.9
    tool_success_lower: float = 0.7
    research_evidence_min: int = 20
    redline_evidence_min: int = 50

    def check_groundedness_drift(
        self,
        current: float,
        peak: float,
        window_size: int,
    ) -> Mutation | None:
        """Propose if groundedness drops significantly from peak."""
        delta = peak - current
        if delta > 0.1 and window_size >= 50:
            mutation = Mutation(
                id=f"drift_groundedness_{int(datetime.now(timezone.utc).timestamp())}",
                agent_name=self.agent_name,
                target_file="heartbeat.md",
                section="Observations",
                rationale=f"Groundedness dropped {delta:.3f} from peak {peak:.3f}",
                evidence={"current": current, "peak": peak, "delta": delta},
                gate=MutationGate.AUTO,
            )
            self.proposals.append(mutation)
            return mutation
        return None

    def check_coherence_rigidity(
        self,
        coherence: float,
    ) -> Mutation | None:
        """Warn if coherence is suspiciously high — possible rigidity."""
        if coherence > 0.95:
            mutation = Mutation(
                id=f"rigidity_warning_{int(datetime.now(timezone.utc).timestamp())}",
                agent_name=self.agent_name,
                target_file="heartbeat.md",
                section="Observations",
                rationale=f"Coherence {coherence:.3f} > 0.95 — possible rigidity",
                evidence={"coherence": coherence},
                gate=MutationGate.AUTO,
            )
            self.proposals.append(mutation)
            return mutation
        return None

    def check_redline_misfit(
        self,
        constraint: str,
        violation_rate: float,
        observations: int,
    ) -> Mutation | None:
        """Propose soul.md change if a red line consistently misfires."""
        if observations >= self.redline_evidence_min and violation_rate > 0.3:
            mutation = Mutation(
                id=f"redline_{constraint[:20]}_{int(datetime.now(timezone.utc).timestamp())}",
                agent_name=self.agent_name,
                target_file="soul.md",
                section="Red Lines",
                rationale=(
                    f"Constraint '{constraint}' violated in {violation_rate:.0%} "
                    f"of {observations} probes — may be too broad or misaligned"
                ),
                evidence={
                    "constraint": constraint,
                    "violation_rate": violation_rate,
                    "observations": observations,
                },
                gate=MutationGate.FEDERATION,
            )
            self.proposals.append(mutation)
            return mutation
        return None

    def check_tool_performance(
        self,
        tool_name: str,
        success_rate: float,
        observations: int,
    ) -> Mutation | None:
        """Flag tools.md update if tool success rate crosses boundary."""
        if observations < 10:
            return None

        if success_rate > self.tool_success_upper or success_rate < self.tool_success_lower:
            mutation = Mutation(
                id=f"tool_{tool_name}_{int(datetime.now(timezone.utc).timestamp())}",
                agent_name=self.agent_name,
                target_file="tools.md",
                section="Performance",
                rationale=(
                    f"Tool '{tool_name}' success rate {success_rate:.2f} "
                    f"crossed boundary ({self.tool_success_lower}-{self.tool_success_upper})"
                ),
                evidence={
                    "tool": tool_name,
                    "success_rate": success_rate,
                    "observations": observations,
                },
                gate=MutationGate.FLAG,
            )
            self.proposals.append(mutation)
            return mutation
        return None

    def check_all_axes_drop(
        self,
        coherence_delta: float,
        differentiation_delta: float,
        groundedness_delta: float,
    ) -> Mutation | None:
        """FREEZE if all three axes drop simultaneously."""
        if all(d < -0.05 for d in [coherence_delta, differentiation_delta, groundedness_delta]):
            mutation = Mutation(
                id=f"freeze_{int(datetime.now(timezone.utc).timestamp())}",
                agent_name=self.agent_name,
                target_file="program.md",
                section="Drift Protection",
                rationale=(
                    "All three axes declining simultaneously - "
                    "freezing mutations, flagging for review"
                ),
                evidence={
                    "coherence_delta": coherence_delta,
                    "differentiation_delta": differentiation_delta,
                    "groundedness_delta": groundedness_delta,
                },
                gate=MutationGate.FEDERATION,
            )
            self.proposals.append(mutation)
            return mutation
        return None

    def write_proposals(self) -> int:
        """
        Write pending proposals to disk.

        AUTO: appended directly to the target file's observations.
        FLAG/FEDERATION: written to proposals/ for human review.
        """
        proposals_dir = self.root / self.agent_name / "proposals"
        written = 0

        for m in self.proposals:
            if m.verdict != MutationVerdict.PENDING:
                continue

            if m.gate == MutationGate.AUTO:
                # Auto-apply: just log it (heartbeat_writer handles actual updates)
                logger.info("AUTO mutation: %s → %s", m.target_file, m.rationale)
                m.verdict = MutationVerdict.KEEP
                written += 1
            else:
                # Federation/flag: write proposal file
                proposals_dir.mkdir(parents=True, exist_ok=True)
                proposal_path = proposals_dir / f"{m.id}.md"
                proposal_path.write_text(
                    f"# Mutation Proposal: {m.id}\n\n"
                    f"**Target:** {m.target_file} → {m.section}\n"
                    f"**Gate:** {m.gate.value}\n"
                    f"**Proposed:** {m.proposed_at}\n\n"
                    f"## Rationale\n{m.rationale}\n\n"
                    f"## Evidence\n```json\n{m.evidence}\n```\n",
                    encoding="utf-8",
                )
                logger.info("PROPOSAL written: %s", proposal_path)
                written += 1

        return written

    def pending_federations(self) -> list[Mutation]:
        """Return proposals awaiting human review."""
        return [
            m for m in self.proposals
            if m.gate in (MutationGate.FLAG, MutationGate.FEDERATION)
            and m.verdict == MutationVerdict.PENDING
        ]
