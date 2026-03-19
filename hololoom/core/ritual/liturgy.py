"""
LiturgySpec — cross-agent rituals requiring coordination.

A liturgy is a multi-agent ritual where MasterWeaver is always the arbitrator.
Participants contribute typed Bobbins, which are merged into a consensus output
via a caller-supplied merge function.

Quorum is a fraction (0.0–1.0) of declared participants.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from .policies import LiturgyFailurePolicy
from .types import RitualContext, RitualResult

logger = logging.getLogger(__name__)


class LiturgyAborted(Exception):
    """Raised when a liturgy's ABORT policy fires."""
    pass


@dataclass
class LiturgySpec:
    """
    A cross-agent ritual requiring coordination.
    MasterWeaver is always the arbitrator.
    """
    name: str
    participants: list[str]        # agent roles that must participate
    quorum: float                  # 0.0–1.0, fraction required to proceed
    sync_point: str                # lifecycle event that triggers sync
    contributions: dict[str, str]  # role → Bobbin type they produce
    consensus_output: str          # Bobbin type of merged result
    failure_policy: LiturgyFailurePolicy
    merge_fn: Callable[[dict[str, RitualResult]], RitualResult]

    async def execute(
        self,
        ctx: RitualContext,
        participant_results: dict[str, RitualResult],
    ) -> RitualResult | None:
        """
        Execute the liturgy merge with quorum enforcement.

        Returns None if the liturgy is deferred (DEFER policy).
        Raises LiturgyAborted if ABORT policy fires.
        """
        quorum_met = len(participant_results) / max(len(self.participants), 1)

        if quorum_met < self.quorum:
            if self.failure_policy == LiturgyFailurePolicy.PARTIAL_QUORUM:
                logger.info(
                    "Liturgy '%s' proceeding with partial quorum: %.1f%% (need %.1f%%)",
                    self.name,
                    quorum_met * 100,
                    self.quorum * 100,
                )
            elif self.failure_policy == LiturgyFailurePolicy.DEFER:
                logger.info(
                    "Liturgy '%s' deferred — insufficient quorum: %.1f%% < %.1f%%",
                    self.name,
                    quorum_met * 100,
                    self.quorum * 100,
                )
                return None
            elif self.failure_policy == LiturgyFailurePolicy.ABORT:
                raise LiturgyAborted(
                    f"Liturgy '{self.name}' aborted — quorum {quorum_met:.1%} "
                    f"< required {self.quorum:.1%}"
                )

        return self.merge_fn(participant_results)
