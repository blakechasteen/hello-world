"""
HoloLoom Phase 3: Contribution Tracking
November 2025

Tracks who added what knowledge, enabling attribution
and contribution analytics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Any
import json


class ContributionType(Enum):
    """Types of contributions."""
    ADD_KNOWLEDGE = "add_knowledge"      # Added new knowledge node
    EDIT_KNOWLEDGE = "edit_knowledge"    # Edited existing knowledge
    DELETE_KNOWLEDGE = "delete_knowledge"  # Deleted knowledge
    ADD_EDGE = "add_edge"                # Added relationship
    APPROVE = "approve"                  # Approved draft
    REVIEW = "review"                    # Reviewed content
    QUERY = "query"                      # Query that improved system


class ContributionStatus(Enum):
    """Status of a contribution."""
    DRAFT = "draft"       # Pending review
    REVIEWED = "reviewed"  # Under review
    APPROVED = "approved"  # Published
    REJECTED = "rejected"  # Not accepted


@dataclass
class Contribution:
    """Single contribution record."""
    contribution_id: str
    user_id: str
    contribution_type: ContributionType
    target_id: str  # Node ID, edge ID, etc.
    content: Any    # The actual content
    status: ContributionStatus = ContributionStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    reviewed_by: Optional[str] = None
    review_comment: str = ""
    metadata: Dict = field(default_factory=dict)

    # Quality metrics
    usefulness_score: float = 0.0  # How useful was this contribution
    access_count: int = 0          # How many times accessed

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "contribution_id": self.contribution_id,
            "user_id": self.user_id,
            "contribution_type": self.contribution_type.value,
            "target_id": self.target_id,
            "content": self.content if isinstance(self.content, (str, dict, list)) else str(self.content),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "reviewed_by": self.reviewed_by,
            "review_comment": self.review_comment,
            "usefulness_score": self.usefulness_score,
            "access_count": self.access_count,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Contribution':
        """Deserialize from dictionary."""
        return cls(
            contribution_id=data["contribution_id"],
            user_id=data["user_id"],
            contribution_type=ContributionType(data["contribution_type"]),
            target_id=data["target_id"],
            content=data["content"],
            status=ContributionStatus(data.get("status", "draft")),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
            reviewed_by=data.get("reviewed_by"),
            review_comment=data.get("review_comment", ""),
            usefulness_score=data.get("usefulness_score", 0.0),
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {})
        )


@dataclass
class UserContributionStats:
    """Contribution statistics for a user."""
    user_id: str
    total_contributions: int = 0
    approved_contributions: int = 0
    rejected_contributions: int = 0
    pending_contributions: int = 0
    total_access_count: int = 0
    avg_usefulness: float = 0.0

    # By type
    by_type: Dict[str, int] = field(default_factory=dict)

    @property
    def approval_rate(self) -> float:
        """Get approval rate."""
        total_reviewed = self.approved_contributions + self.rejected_contributions
        if total_reviewed == 0:
            return 0.0
        return self.approved_contributions / total_reviewed


class ContributionTracker:
    """
    Tracks contributions to the knowledge base.

    Features:
    - Record contributions with attribution
    - Track contribution status (draft → reviewed → approved)
    - Calculate contribution statistics
    - Support review workflow
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize tracker.

        Args:
            storage_path: Path to persist data (None for in-memory)
        """
        self.storage_path = storage_path
        self.contributions: Dict[str, Contribution] = {}
        self._by_user: Dict[str, Set[str]] = {}  # user_id -> contribution_ids
        self._by_target: Dict[str, Set[str]] = {}  # target_id -> contribution_ids
        self._counter = 0

        if storage_path:
            self._load()

    def _generate_id(self) -> str:
        """Generate contribution ID."""
        self._counter += 1
        return f"contrib_{self._counter:06d}"

    # === Recording Contributions ===

    def record_contribution(
        self,
        user_id: str,
        contribution_type: ContributionType,
        target_id: str,
        content: Any,
        auto_approve: bool = False,
        metadata: Dict = None
    ) -> Contribution:
        """
        Record a new contribution.

        Args:
            user_id: Who made the contribution
            contribution_type: Type of contribution
            target_id: ID of affected node/edge
            content: The actual content
            auto_approve: Skip review workflow
            metadata: Additional metadata
        """
        contribution_id = self._generate_id()

        status = ContributionStatus.APPROVED if auto_approve else ContributionStatus.DRAFT

        contribution = Contribution(
            contribution_id=contribution_id,
            user_id=user_id,
            contribution_type=contribution_type,
            target_id=target_id,
            content=content,
            status=status,
            metadata=metadata or {}
        )

        self.contributions[contribution_id] = contribution

        # Index by user
        if user_id not in self._by_user:
            self._by_user[user_id] = set()
        self._by_user[user_id].add(contribution_id)

        # Index by target
        if target_id not in self._by_target:
            self._by_target[target_id] = set()
        self._by_target[target_id].add(contribution_id)

        self._save()
        return contribution

    def record_knowledge_add(self, user_id: str, node_id: str, content: str,
                              auto_approve: bool = False) -> Contribution:
        """Convenience: Record adding knowledge."""
        return self.record_contribution(
            user_id=user_id,
            contribution_type=ContributionType.ADD_KNOWLEDGE,
            target_id=node_id,
            content=content,
            auto_approve=auto_approve
        )

    def record_edge_add(self, user_id: str, source: str, target: str,
                        edge_type: str, auto_approve: bool = False) -> Contribution:
        """Convenience: Record adding edge."""
        edge_id = f"{source}__{edge_type}__{target}"
        return self.record_contribution(
            user_id=user_id,
            contribution_type=ContributionType.ADD_EDGE,
            target_id=edge_id,
            content={"source": source, "target": target, "edge_type": edge_type},
            auto_approve=auto_approve
        )

    # === Review Workflow ===

    def submit_for_review(self, contribution_id: str) -> bool:
        """Submit draft for review."""
        contribution = self.contributions.get(contribution_id)
        if not contribution or contribution.status != ContributionStatus.DRAFT:
            return False

        contribution.status = ContributionStatus.REVIEWED
        contribution.updated_at = datetime.now()
        self._save()
        return True

    def approve(self, contribution_id: str, reviewer_id: str,
                comment: str = "") -> bool:
        """Approve a contribution."""
        contribution = self.contributions.get(contribution_id)
        if not contribution:
            return False

        contribution.status = ContributionStatus.APPROVED
        contribution.reviewed_by = reviewer_id
        contribution.review_comment = comment
        contribution.updated_at = datetime.now()
        self._save()
        return True

    def reject(self, contribution_id: str, reviewer_id: str,
               comment: str = "") -> bool:
        """Reject a contribution."""
        contribution = self.contributions.get(contribution_id)
        if not contribution:
            return False

        contribution.status = ContributionStatus.REJECTED
        contribution.reviewed_by = reviewer_id
        contribution.review_comment = comment
        contribution.updated_at = datetime.now()
        self._save()
        return True

    def get_pending_reviews(self) -> List[Contribution]:
        """Get contributions awaiting review."""
        return [
            c for c in self.contributions.values()
            if c.status in (ContributionStatus.DRAFT, ContributionStatus.REVIEWED)
        ]

    # === Querying ===

    def get_contribution(self, contribution_id: str) -> Optional[Contribution]:
        """Get contribution by ID."""
        return self.contributions.get(contribution_id)

    def get_contributions_by_user(self, user_id: str,
                                   status: ContributionStatus = None) -> List[Contribution]:
        """Get contributions by user."""
        contribution_ids = self._by_user.get(user_id, set())
        contributions = [self.contributions[cid] for cid in contribution_ids if cid in self.contributions]

        if status:
            contributions = [c for c in contributions if c.status == status]

        return sorted(contributions, key=lambda c: c.created_at, reverse=True)

    def get_contributions_for_target(self, target_id: str) -> List[Contribution]:
        """Get contributions for a target (node/edge)."""
        contribution_ids = self._by_target.get(target_id, set())
        contributions = [self.contributions[cid] for cid in contribution_ids if cid in self.contributions]
        return sorted(contributions, key=lambda c: c.created_at, reverse=True)

    def get_attribution(self, target_id: str) -> Optional[str]:
        """Get original contributor for a target."""
        contributions = self.get_contributions_for_target(target_id)
        add_contributions = [
            c for c in contributions
            if c.contribution_type == ContributionType.ADD_KNOWLEDGE
               and c.status == ContributionStatus.APPROVED
        ]
        if add_contributions:
            return add_contributions[-1].user_id  # Oldest approved add
        return None

    # === Statistics ===

    def get_user_stats(self, user_id: str) -> UserContributionStats:
        """Get contribution statistics for a user."""
        contributions = self.get_contributions_by_user(user_id)

        stats = UserContributionStats(user_id=user_id)
        stats.total_contributions = len(contributions)

        for c in contributions:
            # By status
            if c.status == ContributionStatus.APPROVED:
                stats.approved_contributions += 1
            elif c.status == ContributionStatus.REJECTED:
                stats.rejected_contributions += 1
            else:
                stats.pending_contributions += 1

            # By type
            type_name = c.contribution_type.value
            stats.by_type[type_name] = stats.by_type.get(type_name, 0) + 1

            # Aggregate metrics
            stats.total_access_count += c.access_count

        # Average usefulness
        if stats.approved_contributions > 0:
            approved = [c for c in contributions if c.status == ContributionStatus.APPROVED]
            stats.avg_usefulness = sum(c.usefulness_score for c in approved) / len(approved)

        return stats

    def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get top contributors."""
        user_stats = {}

        for contribution in self.contributions.values():
            if contribution.status != ContributionStatus.APPROVED:
                continue

            user_id = contribution.user_id
            if user_id not in user_stats:
                user_stats[user_id] = {"user_id": user_id, "contributions": 0, "access_count": 0}

            user_stats[user_id]["contributions"] += 1
            user_stats[user_id]["access_count"] += contribution.access_count

        # Sort by contributions
        leaderboard = sorted(
            user_stats.values(),
            key=lambda x: (x["contributions"], x["access_count"]),
            reverse=True
        )

        return leaderboard[:limit]

    # === Usage Tracking ===

    def record_access(self, target_id: str):
        """Record access to a target (for usefulness tracking)."""
        contribution_ids = self._by_target.get(target_id, set())
        for cid in contribution_ids:
            if cid in self.contributions:
                self.contributions[cid].access_count += 1

    def update_usefulness(self, contribution_id: str, score: float):
        """Update usefulness score for a contribution."""
        contribution = self.contributions.get(contribution_id)
        if contribution:
            # Exponential moving average
            alpha = 0.3
            contribution.usefulness_score = (
                alpha * score + (1 - alpha) * contribution.usefulness_score
            )

    # === Persistence ===

    def _save(self):
        """Save to storage."""
        if not self.storage_path:
            return

        data = {
            "contributions": {cid: c.to_dict() for cid, c in self.contributions.items()},
            "counter": self._counter
        }

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load from storage."""
        if not self.storage_path:
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            self._counter = data.get("counter", 0)

            for cid, cdata in data.get("contributions", {}).items():
                contribution = Contribution.from_dict(cdata)
                self.contributions[cid] = contribution

                # Rebuild indexes
                user_id = contribution.user_id
                if user_id not in self._by_user:
                    self._by_user[user_id] = set()
                self._by_user[user_id].add(cid)

                target_id = contribution.target_id
                if target_id not in self._by_target:
                    self._by_target[target_id] = set()
                self._by_target[target_id].add(cid)

        except FileNotFoundError:
            pass

    # === Export ===

    def export_contributions(self, user_id: Optional[str] = None,
                              format: str = "json") -> str:
        """Export contributions."""
        if user_id:
            contributions = self.get_contributions_by_user(user_id)
        else:
            contributions = list(self.contributions.values())

        if format == "json":
            return json.dumps([c.to_dict() for c in contributions], indent=2)
        elif format == "csv":
            lines = ["id,user_id,type,target,status,created_at"]
            for c in contributions:
                lines.append(f"{c.contribution_id},{c.user_id},{c.contribution_type.value},"
                            f"{c.target_id},{c.status.value},{c.created_at.isoformat()}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unknown format: {format}")


# Quick test
if __name__ == "__main__":
    print("Testing Contribution Tracker...")

    tracker = ContributionTracker()

    # Record some contributions
    c1 = tracker.record_knowledge_add(
        user_id="alice",
        node_id="thompson_sampling",
        content="Thompson Sampling balances exploration and exploitation"
    )

    c2 = tracker.record_knowledge_add(
        user_id="bob",
        node_id="bayesian_methods",
        content="Bayesian methods use prior distributions"
    )

    c3 = tracker.record_edge_add(
        user_id="alice",
        source="thompson_sampling",
        target="bayesian_methods",
        edge_type="IS_A"
    )

    print(f"Recorded {len(tracker.contributions)} contributions")

    # Review workflow
    tracker.submit_for_review(c1.contribution_id)
    tracker.approve(c1.contribution_id, "charlie", "Looks good!")
    print(f"Contribution {c1.contribution_id} status: {c1.status.value}")

    # Get stats
    alice_stats = tracker.get_user_stats("alice")
    print(f"\nAlice's stats:")
    print(f"  Total: {alice_stats.total_contributions}")
    print(f"  Approved: {alice_stats.approved_contributions}")
    print(f"  By type: {alice_stats.by_type}")

    # Leaderboard
    print("\nLeaderboard:")
    for entry in tracker.get_leaderboard():
        print(f"  {entry['user_id']}: {entry['contributions']} contributions")

    # Attribution
    author = tracker.get_attribution("thompson_sampling")
    print(f"\nThompson Sampling attributed to: {author}")
