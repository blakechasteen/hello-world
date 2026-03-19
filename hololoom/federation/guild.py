"""
Guild Management - Trust through specialization.

Guilds are trust groups with domain expertise.
Trust is earned, not given.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from .types import (
    AdmissionPolicy,
    FederationNode,
    Guild,
    GuildError,
    GuildTrustLevel,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  REPUTATION TRACKER
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ReputationRecord:
    """Track a node's reputation in a guild."""

    node_id: str
    guild_id: str

    # Scores
    successes: int = 0
    failures: int = 0
    verifications: int = 0

    # History
    joined_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def score(self) -> float:
        """
        Reputation score (0.0-1.0).

        Uses Wilson score interval for stability with few samples.
        """
        if self.total == 0:
            return 0.5  # Neutral default

        # Wilson score (lower bound of 95% confidence interval)
        n = self.total
        p = self.successes / n
        z = 1.96  # 95% confidence

        denominator = 1 + z * z / n
        center = p + z * z / (2 * n)
        spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)

        return (center - spread) / denominator

    @property
    def days_active(self) -> int:
        """Days since joining."""
        delta = datetime.utcnow() - self.joined_at
        return delta.days


# ═══════════════════════════════════════════════════════════════════════════
#  GUILD MANAGER
# ═══════════════════════════════════════════════════════════════════════════


class GuildManager:
    """
    Manages guilds and reputation.

    Usage:
        manager = GuildManager()

        # Create a medical guild
        guild = await manager.create(
            name="Medical AI",
            domain="medical",
            admission=AdmissionPolicy.VOUCHED
        )

        # Join the guild
        await manager.join(guild.guild_id, my_node_id)

        # Update reputation after verification
        await manager.record_verification(
            guild_id=guild.guild_id,
            node_id=my_node_id,
            success=True
        )
    """

    def __init__(self):
        self._guilds: dict[str, Guild] = {}
        self._reputation: dict[str, dict[str, ReputationRecord]] = defaultdict(dict)
        self._pending_applications: dict[str, list[str]] = defaultdict(list)

    # ───────────────────────────────────────────────────────────────────────
    #  GUILD LIFECYCLE
    # ───────────────────────────────────────────────────────────────────────

    async def create(
        self,
        name: str,
        domain: str,
        *,
        admission: AdmissionPolicy = AdmissionPolicy.OPEN,
        founder: str | None = None,
    ) -> Guild:
        """
        Create a new guild.

        Args:
            name: Human-readable name
            domain: Specialization domain
            admission: How members join
            founder: Founding member node ID

        Returns:
            Created Guild
        """
        import uuid

        guild_id = str(uuid.uuid4())[:8]

        guild = Guild(
            guild_id=guild_id,
            name=name,
            domain=domain,
            admission=admission,
            trust_level=GuildTrustLevel.STARTER,
        )

        self._guilds[guild_id] = guild

        # Add founder as first member
        if founder:
            await self._add_member(guild_id, founder)

        logger.info(f"Created guild '{name}' ({guild_id})")
        return guild

    async def dissolve(self, guild_id: str) -> None:
        """Dissolve a guild."""
        if guild_id not in self._guilds:
            return

        del self._guilds[guild_id]
        self._reputation.pop(guild_id, None)
        self._pending_applications.pop(guild_id, None)

        logger.info(f"Dissolved guild {guild_id}")

    # ───────────────────────────────────────────────────────────────────────
    #  MEMBERSHIP
    # ───────────────────────────────────────────────────────────────────────

    async def join(
        self,
        guild_id: str,
        node_id: str,
        *,
        sponsor: str | None = None,
    ) -> bool:
        """
        Request to join a guild.

        Returns:
            True if joined immediately, False if pending approval
        """
        guild = self._guilds.get(guild_id)
        if not guild:
            raise GuildError(
                f"Guild {guild_id} not found",
                suggestion="Use list_guilds() to find available guilds",
            )

        # Already a member?
        if node_id in guild.members:
            return True

        # Check admission policy
        if guild.admission == AdmissionPolicy.OPEN:
            await self._add_member(guild_id, node_id)
            return True

        elif guild.admission == AdmissionPolicy.VOUCHED:
            if sponsor and sponsor in guild.members:
                await self._add_member(guild_id, node_id)
                return True
            else:
                raise GuildError(
                    f"Guild {guild_id} requires a sponsor",
                    suggestion="Ask an existing member to sponsor you",
                )

        elif guild.admission == AdmissionPolicy.VOTED:
            self._pending_applications[guild_id].append(node_id)
            logger.info(f"Node {node_id[:8]}... applied to guild {guild_id}")
            return False

        else:  # CLOSED
            raise GuildError(
                f"Guild {guild_id} is closed to new members",
            )

    async def leave(self, guild_id: str, node_id: str) -> None:
        """Leave a guild."""
        guild = self._guilds.get(guild_id)
        if not guild:
            return

        guild.members.discard(node_id)

        # Keep reputation for historical record
        logger.info(f"Node {node_id[:8]}... left guild {guild_id}")

    async def _add_member(self, guild_id: str, node_id: str) -> None:
        """Add member to guild."""
        guild = self._guilds.get(guild_id)
        if not guild:
            return

        guild.members.add(node_id)

        # Initialize reputation
        self._reputation[guild_id][node_id] = ReputationRecord(
            node_id=node_id,
            guild_id=guild_id,
        )

        logger.info(f"Node {node_id[:8]}... joined guild {guild_id}")

    async def kick(
        self,
        guild_id: str,
        node_id: str,
        *,
        reason: str = "unspecified",
    ) -> None:
        """Remove a member from guild."""
        await self.leave(guild_id, node_id)
        logger.warning(f"Node {node_id[:8]}... kicked from guild {guild_id}: {reason}")

    # ───────────────────────────────────────────────────────────────────────
    #  REPUTATION
    # ───────────────────────────────────────────────────────────────────────

    async def record_verification(
        self,
        guild_id: str,
        node_id: str,
        *,
        success: bool,
        weight: float = 1.0,
    ) -> float:
        """
        Record a verification outcome.

        Args:
            guild_id: Guild context
            node_id: Node that performed verification
            success: Whether verification was correct
            weight: Importance weight (default: 1.0)

        Returns:
            Updated reputation score
        """
        if guild_id not in self._reputation:
            return 0.5

        if node_id not in self._reputation[guild_id]:
            return 0.5

        record = self._reputation[guild_id][node_id]
        record.verifications += 1
        record.last_active = datetime.utcnow()

        if success:
            record.successes += int(weight)
        else:
            record.failures += int(weight)

        # Update guild in node
        guild = self._guilds.get(guild_id)
        if guild:
            guild.reputation[node_id] = record.score

        return record.score

    async def update_reputation(
        self,
        guild_id: str,
        node_id: str,
        delta: float,
    ) -> float:
        """
        Direct reputation update.

        Returns:
            New reputation score
        """
        if guild_id not in self._reputation:
            return 0.5

        if node_id not in self._reputation[guild_id]:
            return 0.5

        record = self._reputation[guild_id][node_id]

        # Convert delta to success/failure counts
        if delta > 0:
            record.successes += int(delta * 10)
        else:
            record.failures += int(abs(delta) * 10)

        return record.score

    def get_reputation(self, guild_id: str, node_id: str) -> float:
        """Get node's reputation in guild."""
        if guild_id not in self._reputation:
            return 0.5

        if node_id not in self._reputation[guild_id]:
            return 0.5

        return self._reputation[guild_id][node_id].score

    # ───────────────────────────────────────────────────────────────────────
    #  TRUST LEVEL EVOLUTION
    # ───────────────────────────────────────────────────────────────────────

    async def update_trust_level(self, guild_id: str) -> GuildTrustLevel:
        """
        Update guild trust level based on age and verifications.

        Trust levels:
        - STARTER: <30 days, quorum=5
        - ESTABLISHED: 30-180 days AND 100+ verifications, quorum=3
        - VETERAN: >180 days AND 1000+ verifications, quorum=2
        """
        guild = self._guilds.get(guild_id)
        if not guild:
            return GuildTrustLevel.STARTER

        age_days = (datetime.utcnow() - guild.created_at).days

        # Count total verifications
        total_verifications = sum(
            r.verifications for r in self._reputation.get(guild_id, {}).values()
        )

        # Determine level
        if age_days >= 180 and total_verifications >= 1000:
            new_level = GuildTrustLevel.VETERAN
        elif age_days >= 30 and total_verifications >= 100:
            new_level = GuildTrustLevel.ESTABLISHED
        else:
            new_level = GuildTrustLevel.STARTER

        if new_level != guild.trust_level:
            old_level = guild.trust_level
            guild.trust_level = new_level
            logger.info(
                f"Guild {guild_id} trust level: {old_level.name} -> {new_level.name}"
            )

        return new_level

    # ───────────────────────────────────────────────────────────────────────
    #  QUERIES
    # ───────────────────────────────────────────────────────────────────────

    def get_guild(self, guild_id: str) -> Guild | None:
        """Get guild by ID."""
        return self._guilds.get(guild_id)

    async def get_members(self, guild_id: str) -> list[str]:
        """Get guild members."""
        guild = self._guilds.get(guild_id)
        if not guild:
            return []
        return list(guild.members)

    async def get_quorum(self, guild_id: str) -> int:
        """Get required quorum for guild's trust level."""
        guild = self._guilds.get(guild_id)
        if not guild:
            return 5  # Default to strictest

        return guild.quorum

    def list_guilds(
        self,
        *,
        domain: str | None = None,
        min_trust: GuildTrustLevel | None = None,
    ) -> list[Guild]:
        """List guilds with optional filters."""
        guilds = list(self._guilds.values())

        if domain:
            guilds = [g for g in guilds if g.domain == domain]

        if min_trust:
            trust_order = [
                GuildTrustLevel.STARTER,
                GuildTrustLevel.ESTABLISHED,
                GuildTrustLevel.VETERAN,
            ]
            min_idx = trust_order.index(min_trust)
            guilds = [g for g in guilds if trust_order.index(g.trust_level) >= min_idx]

        return guilds

    def find_by_domain(self, domain: str) -> list[Guild]:
        """Find guilds by domain."""
        return [g for g in self._guilds.values() if g.domain == domain]

    # ───────────────────────────────────────────────────────────────────────
    #  VOTING (for VOTED admission)
    # ───────────────────────────────────────────────────────────────────────

    async def vote_admission(
        self,
        guild_id: str,
        applicant: str,
        voter: str,
        approve: bool,
    ) -> bool | None:
        """
        Vote on pending membership application.

        Returns:
            True if approved, False if rejected, None if vote recorded
        """
        guild = self._guilds.get(guild_id)
        if not guild:
            raise GuildError(f"Guild {guild_id} not found")

        if voter not in guild.members:
            raise GuildError("Only members can vote on applications")

        if applicant not in self._pending_applications.get(guild_id, []):
            raise GuildError(f"No pending application from {applicant[:8]}...")

        # TODO: Implement voting logic with vote tracking
        # For now, single sponsor approval is enough
        if approve:
            self._pending_applications[guild_id].remove(applicant)
            await self._add_member(guild_id, applicant)
            return True

        return None


# ═══════════════════════════════════════════════════════════════════════════
#  TRUST CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════


class TrustCalculator:
    """
    Calculate overall trust for a node.

    Combines:
    - Base network trust
    - Guild-specific reputation
    - Historical performance
    """

    def __init__(
        self,
        *,
        guild_weight: float = 0.4,
        network_weight: float = 0.3,
        history_weight: float = 0.3,
    ):
        self._guild_weight = guild_weight
        self._network_weight = network_weight
        self._history_weight = history_weight

    def calculate(
        self,
        node: FederationNode,
        guild_manager: GuildManager,
        *,
        context_guild: str | None = None,
    ) -> float:
        """
        Calculate overall trust score for node.

        Args:
            node: Node to evaluate
            guild_manager: Guild manager for reputation data
            context_guild: Preferred guild context

        Returns:
            Trust score (0.0-1.0)
        """
        # Guild reputation
        guild_score = 0.5
        if context_guild:
            guild_score = guild_manager.get_reputation(context_guild, node.node_id)
        elif node.guilds:
            # Average across all guilds
            scores = [
                guild_manager.get_reputation(g, node.node_id) for g in node.guilds
            ]
            if scores:
                guild_score = sum(scores) / len(scores)

        # Network trust (from node itself)
        network_score = node.trust_score

        # Historical (placeholder - would come from audit trail)
        history_score = 0.5

        # Weighted combination
        total = (
            guild_score * self._guild_weight
            + network_score * self._network_weight
            + history_score * self._history_weight
        )

        return min(max(total, 0.0), 1.0)
