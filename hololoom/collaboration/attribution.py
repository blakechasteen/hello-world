"""
Contribution Tracking and Attribution System.

Tracks who added/modified what knowledge, enabling accountability and credit.

Phase 3: Multi-User Collaboration - Attribution infrastructure.

Created: November 2025
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, TYPE_CHECKING
from collections import defaultdict
import logging

if TYPE_CHECKING:
    from .annotation_refinement import AnnotationRefiner, RefinementResult, AnnotationType

logger = logging.getLogger(__name__)


class ContributionType(Enum):
    """Types of contributions."""
    CREATE = "create"                       # Created new object/resource
    KNOWLEDGE_ADD = "knowledge_add"         # Added new knowledge
    KNOWLEDGE_EDIT = "knowledge_edit"       # Modified existing
    KNOWLEDGE_DELETE = "knowledge_delete"   # Removed knowledge
    QUERY = "query"                         # Asked a question
    ANNOTATION = "annotation"               # Added annotation/comment
    WHITEBOARD_DRAW = "whiteboard_draw"     # Drew on whiteboard
    WHITEBOARD_EDIT = "whiteboard_edit"     # Edited whiteboard content
    REVIEW = "review"                       # Reviewed content
    APPROVAL = "approval"                   # Approved content
    CORRECTION = "correction"               # Corrected inaccuracy
    LINK = "link"                          # Created relationship link
    TAG = "tag"                            # Added tag/category
    IMPORT = "import"                      # Imported external content


class QualityRating(Enum):
    """Quality ratings for contributions."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"


# Numeric mapping for quality ratings
QUALITY_RATING_SCORES: Dict[QualityRating, float] = {
    QualityRating.EXCELLENT: 1.0,
    QualityRating.GOOD: 0.8,
    QualityRating.ACCEPTABLE: 0.6,
    QualityRating.NEEDS_IMPROVEMENT: 0.4,
    QualityRating.POOR: 0.2,
}


class RaterRole(Enum):
    """Role of the person rating a contribution."""
    OWNER = "owner"           # Scene/project owner - weight 2.0
    EDITOR = "editor"         # Editor - weight 1.5
    CONTRIBUTOR = "contributor"  # Contributor - weight 1.0
    VIEWER = "viewer"         # Viewer - weight 0.5


# Weight multipliers for different rater roles
RATER_WEIGHTS: Dict[RaterRole, float] = {
    RaterRole.OWNER: 2.0,
    RaterRole.EDITOR: 1.5,
    RaterRole.CONTRIBUTOR: 1.0,
    RaterRole.VIEWER: 0.5,
}


@dataclass
class QualityRatingRecord:
    """
    A single quality rating from a rater.

    Supports multiple ratings per contribution with weighted aggregation.
    """
    rater_id: str
    rater_role: RaterRole
    rating: QualityRating
    score: float  # Numeric score (0.0-1.0)
    timestamp: datetime = field(default_factory=datetime.now)
    comment: str = ""  # Optional feedback

    @classmethod
    def from_rating(
        cls,
        rater_id: str,
        rating: QualityRating,
        rater_role: RaterRole = RaterRole.CONTRIBUTOR,
        comment: str = ""
    ) -> 'QualityRatingRecord':
        """Create a rating record from a QualityRating enum."""
        return cls(
            rater_id=rater_id,
            rater_role=rater_role,
            rating=rating,
            score=QUALITY_RATING_SCORES[rating],
            comment=comment
        )

    @classmethod
    def from_score(
        cls,
        rater_id: str,
        score: float,
        rater_role: RaterRole = RaterRole.CONTRIBUTOR,
        comment: str = ""
    ) -> 'QualityRatingRecord':
        """Create a rating record from a numeric score (0.0-1.0)."""
        # Clamp score to valid range
        score = max(0.0, min(1.0, score))

        # Map score to closest QualityRating
        if score >= 0.9:
            rating = QualityRating.EXCELLENT
        elif score >= 0.7:
            rating = QualityRating.GOOD
        elif score >= 0.5:
            rating = QualityRating.ACCEPTABLE
        elif score >= 0.3:
            rating = QualityRating.NEEDS_IMPROVEMENT
        else:
            rating = QualityRating.POOR

        return cls(
            rater_id=rater_id,
            rater_role=rater_role,
            rating=rating,
            score=score,
            comment=comment
        )

    def weighted_score(self) -> float:
        """Get the weighted score based on rater role."""
        return self.score * RATER_WEIGHTS[self.rater_role]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rater_id": self.rater_id,
            "rater_role": self.rater_role.value,
            "rating": self.rating.value,
            "score": self.score,
            "timestamp": self.timestamp.isoformat(),
            "comment": self.comment,
            "weighted_score": self.weighted_score()
        }


@dataclass
class Contribution:
    """A single contribution record."""
    contribution_id: str
    contribution_type: ContributionType
    user_id: str
    target_type: str  # "memory", "node", "edge", "stroke", "annotation", etc.
    target_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Content summary
    content_preview: str = ""  # Brief preview of what was contributed
    content_size: int = 0      # Bytes or token count

    # Context
    session_id: Optional[str] = None
    query_id: Optional[str] = None
    parent_contribution_id: Optional[str] = None  # For edits/reviews

    # Quality metrics (legacy single rating - kept for backward compatibility)
    quality_rating: Optional[QualityRating] = None
    rating_by: Optional[str] = None
    rating_at: Optional[datetime] = None

    # Quality metrics (new multi-rater support)
    ratings: List['QualityRatingRecord'] = field(default_factory=list)

    # Impact metrics
    view_count: int = 0
    use_count: int = 0        # Times used in retrieval
    citation_count: int = 0   # Times referenced
    upvotes: int = 0
    downvotes: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def quality_score(self) -> float:
        """
        Calculate aggregated quality score from all ratings.

        Uses weighted average where owner ratings count 2x, editor ratings 1.5x,
        contributor ratings 1x, and viewer ratings 0.5x.

        Returns:
            float: Weighted average score (0.0-1.0), or 0.0 if no ratings
        """
        if not self.ratings:
            # Fall back to legacy single rating if no multi-ratings
            if self.quality_rating:
                return QUALITY_RATING_SCORES.get(self.quality_rating, 0.5)
            return 0.0

        total_weighted_score = sum(r.weighted_score() for r in self.ratings)
        total_weight = sum(RATER_WEIGHTS[r.rater_role] for r in self.ratings)

        if total_weight == 0:
            return 0.0

        return total_weighted_score / total_weight

    @property
    def rating_count(self) -> int:
        """Get number of ratings for this contribution."""
        return len(self.ratings)

    def add_rating(
        self,
        rater_id: str,
        rating: QualityRating,
        rater_role: RaterRole = RaterRole.CONTRIBUTOR,
        comment: str = ""
    ) -> 'QualityRatingRecord':
        """
        Add a quality rating from a rater.

        If the same rater has already rated, their previous rating is updated.

        Args:
            rater_id: ID of the rater
            rating: Quality rating enum
            rater_role: Role of the rater (affects weight)
            comment: Optional feedback comment

        Returns:
            The created or updated QualityRatingRecord
        """
        # Remove existing rating from same rater
        self.ratings = [r for r in self.ratings if r.rater_id != rater_id]

        # Add new rating
        record = QualityRatingRecord.from_rating(
            rater_id=rater_id,
            rating=rating,
            rater_role=rater_role,
            comment=comment
        )
        self.ratings.append(record)

        # Update legacy fields for backward compatibility
        self.quality_rating = rating
        self.rating_by = rater_id
        self.rating_at = record.timestamp

        return record

    def add_rating_score(
        self,
        rater_id: str,
        score: float,
        rater_role: RaterRole = RaterRole.CONTRIBUTOR,
        comment: str = ""
    ) -> 'QualityRatingRecord':
        """
        Add a numeric quality rating (0.0-1.0).

        If the same rater has already rated, their previous rating is updated.

        Args:
            rater_id: ID of the rater
            score: Numeric score (0.0-1.0)
            rater_role: Role of the rater (affects weight)
            comment: Optional feedback comment

        Returns:
            The created or updated QualityRatingRecord
        """
        # Remove existing rating from same rater
        self.ratings = [r for r in self.ratings if r.rater_id != rater_id]

        # Add new rating
        record = QualityRatingRecord.from_score(
            rater_id=rater_id,
            score=score,
            rater_role=rater_role,
            comment=comment
        )
        self.ratings.append(record)

        # Update legacy fields for backward compatibility
        self.quality_rating = record.rating
        self.rating_by = rater_id
        self.rating_at = record.timestamp

        return record

    def get_rating_by(self, rater_id: str) -> Optional['QualityRatingRecord']:
        """Get a specific rater's rating."""
        for r in self.ratings:
            if r.rater_id == rater_id:
                return r
        return None

    def calculate_impact_score(self) -> float:
        """Calculate contribution impact score (0-1)."""
        # Weighted formula
        usage_score = min(self.use_count / 100, 1.0) * 0.4
        citation_score = min(self.citation_count / 50, 1.0) * 0.3
        vote_score = max(0, (self.upvotes - self.downvotes) / 20) * 0.3
        view_score = min(self.view_count / 500, 1.0) * 0.1

        total = usage_score + citation_score + vote_score + view_score
        return min(total / 1.1, 1.0)  # Normalize

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contribution_id": self.contribution_id,
            "contribution_type": self.contribution_type.value,
            "user_id": self.user_id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "timestamp": self.timestamp.isoformat(),
            "content_preview": self.content_preview,
            "content_size": self.content_size,
            "session_id": self.session_id,
            "query_id": self.query_id,
            "parent_contribution_id": self.parent_contribution_id,
            "quality_rating": self.quality_rating.value if self.quality_rating else None,
            "rating_by": self.rating_by,
            "rating_at": self.rating_at.isoformat() if self.rating_at else None,
            "ratings": [r.to_dict() for r in self.ratings],
            "quality_score": self.quality_score,
            "rating_count": self.rating_count,
            "view_count": self.view_count,
            "use_count": self.use_count,
            "citation_count": self.citation_count,
            "upvotes": self.upvotes,
            "downvotes": self.downvotes,
            "impact_score": self.calculate_impact_score(),
            "metadata": self.metadata
        }


@dataclass
class UserContributionStats:
    """Aggregate contribution statistics for a user."""
    user_id: str
    display_name: str

    # Counts by type
    contribution_counts: Dict[str, int] = field(default_factory=dict)

    # Quality metrics
    avg_quality_score: float = 0.0
    quality_distribution: Dict[str, int] = field(default_factory=dict)

    # Impact metrics
    total_views: int = 0
    total_uses: int = 0
    total_citations: int = 0
    total_upvotes: int = 0
    total_downvotes: int = 0
    avg_impact_score: float = 0.0

    # Activity metrics
    first_contribution: Optional[datetime] = None
    last_contribution: Optional[datetime] = None
    active_days: int = 0
    current_streak: int = 0

    # Content metrics
    total_content_size: int = 0
    unique_targets: int = 0

    def total_contributions(self) -> int:
        """Get total contribution count."""
        return sum(self.contribution_counts.values())

    def net_votes(self) -> int:
        """Get net vote score."""
        return self.total_upvotes - self.total_downvotes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "contribution_counts": self.contribution_counts,
            "total_contributions": self.total_contributions(),
            "avg_quality_score": self.avg_quality_score,
            "quality_distribution": self.quality_distribution,
            "total_views": self.total_views,
            "total_uses": self.total_uses,
            "total_citations": self.total_citations,
            "total_upvotes": self.total_upvotes,
            "total_downvotes": self.total_downvotes,
            "net_votes": self.net_votes(),
            "avg_impact_score": self.avg_impact_score,
            "first_contribution": self.first_contribution.isoformat() if self.first_contribution else None,
            "last_contribution": self.last_contribution.isoformat() if self.last_contribution else None,
            "active_days": self.active_days,
            "current_streak": self.current_streak,
            "total_content_size": self.total_content_size,
            "unique_targets": self.unique_targets
        }


@dataclass
class AttributionContext:
    """Attribution context for displaying credits."""
    target_id: str
    target_type: str
    created_by: str
    created_by_name: str
    created_at: datetime
    last_modified_by: Optional[str] = None
    last_modified_by_name: Optional[str] = None
    last_modified_at: Optional[datetime] = None
    contributor_count: int = 1
    contributors: List[Dict[str, str]] = field(default_factory=list)
    edit_count: int = 0
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_type": self.target_type,
            "created_by": self.created_by,
            "created_by_name": self.created_by_name,
            "created_at": self.created_at.isoformat(),
            "last_modified_by": self.last_modified_by,
            "last_modified_by_name": self.last_modified_by_name,
            "last_modified_at": self.last_modified_at.isoformat() if self.last_modified_at else None,
            "contributor_count": self.contributor_count,
            "contributors": self.contributors,
            "edit_count": self.edit_count,
            "version": self.version
        }

    def to_credit_line(self) -> str:
        """Generate a human-readable credit line."""
        if self.contributor_count == 1:
            return f"Added by {self.created_by_name}"
        elif self.contributor_count == 2:
            return f"Added by {self.created_by_name}, edited by {self.last_modified_by_name}"
        else:
            return f"Added by {self.created_by_name} and {self.contributor_count - 1} others"


class AttributionManager:
    """
    Manages contribution tracking and attribution across collaborative sessions.

    Provides:
    - Contribution recording
    - Attribution lookup
    - User statistics
    - Leaderboards
    - Quality tracking
    """

    def __init__(self, knowledge_base_id: Optional[str] = None):
        self.knowledge_base_id = knowledge_base_id

        # Storage
        self.contributions: Dict[str, Contribution] = {}
        self.target_contributions: Dict[str, List[str]] = defaultdict(list)  # target_id -> contribution_ids
        self.user_contributions: Dict[str, List[str]] = defaultdict(list)   # user_id -> contribution_ids
        self.user_names: Dict[str, str] = {}

        # Statistics cache
        self._user_stats_cache: Dict[str, UserContributionStats] = {}
        self._stats_cache_time: Optional[datetime] = None

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}

    def record_contribution(
        self,
        contribution_type: ContributionType,
        user_id: str,
        user_name: str,
        target_type: str,
        target_id: str,
        content_preview: str = "",
        content_size: int = 0,
        session_id: Optional[str] = None,
        query_id: Optional[str] = None,
        parent_contribution_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Contribution:
        """Record a new contribution."""
        from uuid import uuid4

        contribution_id = f"contrib_{uuid4().hex[:12]}"

        contribution = Contribution(
            contribution_id=contribution_id,
            contribution_type=contribution_type,
            user_id=user_id,
            target_type=target_type,
            target_id=target_id,
            content_preview=content_preview[:200],  # Limit preview length
            content_size=content_size,
            session_id=session_id,
            query_id=query_id,
            parent_contribution_id=parent_contribution_id,
            metadata=metadata or {}
        )

        # Store
        self.contributions[contribution_id] = contribution
        self.target_contributions[target_id].append(contribution_id)
        self.user_contributions[user_id].append(contribution_id)
        self.user_names[user_id] = user_name

        # Invalidate stats cache
        self._stats_cache_time = None

        self._emit_event("contribution_recorded", {
            "contribution": contribution.to_dict()
        })

        logger.info(f"Recorded {contribution_type.value} by {user_id} on {target_id}")
        return contribution

    def rate_contribution(
        self,
        contribution_id: str,
        rating: QualityRating,
        rater_id: str,
        rater_role: RaterRole = RaterRole.CONTRIBUTOR,
        comment: str = ""
    ) -> Optional[QualityRatingRecord]:
        """
        Rate a contribution's quality.

        Supports multiple ratings per contribution with weighted aggregation.
        Owner ratings count 2x, editor ratings 1.5x, contributor 1x, viewer 0.5x.

        Args:
            contribution_id: ID of contribution to rate
            rating: QualityRating enum value
            rater_id: ID of the person rating
            rater_role: Role of the rater (affects weight in aggregation)
            comment: Optional feedback comment

        Returns:
            QualityRatingRecord if successful, None if contribution not found
        """
        contribution = self.contributions.get(contribution_id)
        if not contribution:
            return None

        record = contribution.add_rating(
            rater_id=rater_id,
            rating=rating,
            rater_role=rater_role,
            comment=comment
        )

        self._emit_event("contribution_rated", {
            "contribution_id": contribution_id,
            "rating": rating.value,
            "rater_id": rater_id,
            "rater_role": rater_role.value,
            "quality_score": contribution.quality_score,
            "rating_count": contribution.rating_count
        })

        return record

    def rate_contribution_score(
        self,
        contribution_id: str,
        score: float,
        rater_id: str,
        rater_role: RaterRole = RaterRole.CONTRIBUTOR,
        comment: str = ""
    ) -> Optional[QualityRatingRecord]:
        """
        Rate a contribution with a numeric score (0.0-1.0).

        Supports multiple ratings per contribution with weighted aggregation.

        Args:
            contribution_id: ID of contribution to rate
            score: Numeric score (0.0-1.0)
            rater_id: ID of the person rating
            rater_role: Role of the rater (affects weight in aggregation)
            comment: Optional feedback comment

        Returns:
            QualityRatingRecord if successful, None if contribution not found
        """
        contribution = self.contributions.get(contribution_id)
        if not contribution:
            return None

        record = contribution.add_rating_score(
            rater_id=rater_id,
            score=score,
            rater_role=rater_role,
            comment=comment
        )

        self._emit_event("contribution_rated", {
            "contribution_id": contribution_id,
            "score": score,
            "rating": record.rating.value,
            "rater_id": rater_id,
            "rater_role": rater_role.value,
            "quality_score": contribution.quality_score,
            "rating_count": contribution.rating_count
        })

        return record

    def get_contribution_quality(self, contribution_id: str) -> Optional[float]:
        """
        Get the aggregated quality score for a contribution.

        Returns weighted average of all ratings (0.0-1.0), or None if not found.
        """
        contribution = self.contributions.get(contribution_id)
        if not contribution:
            return None
        return contribution.quality_score

    def get_contribution_ratings(self, contribution_id: str) -> List[QualityRatingRecord]:
        """Get all ratings for a contribution."""
        contribution = self.contributions.get(contribution_id)
        if not contribution:
            return []
        return contribution.ratings.copy()

    def vote_contribution(
        self,
        contribution_id: str,
        user_id: str,
        upvote: bool
    ) -> bool:
        """Vote on a contribution."""
        contribution = self.contributions.get(contribution_id)
        if not contribution:
            return False

        if upvote:
            contribution.upvotes += 1
        else:
            contribution.downvotes += 1

        self._emit_event("contribution_voted", {
            "contribution_id": contribution_id,
            "user_id": user_id,
            "upvote": upvote
        })

        return True

    def increment_usage(self, target_id: str):
        """Increment use count for all contributions to a target."""
        contribution_ids = self.target_contributions.get(target_id, [])
        for cid in contribution_ids:
            contribution = self.contributions.get(cid)
            if contribution:
                contribution.use_count += 1

    def increment_views(self, target_id: str):
        """Increment view count for all contributions to a target."""
        contribution_ids = self.target_contributions.get(target_id, [])
        for cid in contribution_ids:
            contribution = self.contributions.get(cid)
            if contribution:
                contribution.view_count += 1

    def increment_citations(self, target_id: str):
        """Increment citation count when target is referenced."""
        contribution_ids = self.target_contributions.get(target_id, [])
        for cid in contribution_ids:
            contribution = self.contributions.get(cid)
            if contribution:
                contribution.citation_count += 1

    def get_attribution(self, target_id: str) -> Optional[AttributionContext]:
        """Get attribution context for a target."""
        contribution_ids = self.target_contributions.get(target_id, [])
        if not contribution_ids:
            return None

        contributions = [
            self.contributions[cid] for cid in contribution_ids
            if cid in self.contributions
        ]
        if not contributions:
            return None

        # Sort by timestamp
        contributions.sort(key=lambda c: c.timestamp)

        # Creator is first contribution
        creator = contributions[0]

        # Last modifier is last edit/modification
        edits = [c for c in contributions if c.contribution_type in (
            ContributionType.KNOWLEDGE_EDIT,
            ContributionType.CORRECTION,
            ContributionType.WHITEBOARD_EDIT
        )]

        last_modifier = edits[-1] if edits else None

        # Unique contributors
        contributor_ids = set(c.user_id for c in contributions)
        contributors = [
            {"user_id": uid, "display_name": self.user_names.get(uid, uid)}
            for uid in contributor_ids
        ]

        return AttributionContext(
            target_id=target_id,
            target_type=creator.target_type,
            created_by=creator.user_id,
            created_by_name=self.user_names.get(creator.user_id, creator.user_id),
            created_at=creator.timestamp,
            last_modified_by=last_modifier.user_id if last_modifier else None,
            last_modified_by_name=self.user_names.get(last_modifier.user_id, last_modifier.user_id) if last_modifier else None,
            last_modified_at=last_modifier.timestamp if last_modifier else None,
            contributor_count=len(contributor_ids),
            contributors=contributors,
            edit_count=len(edits),
            version=len(contributions)
        )

    def get_user_contributions(
        self,
        user_id: str,
        contribution_type: Optional[ContributionType] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Contribution]:
        """Get contributions by a user."""
        contribution_ids = self.user_contributions.get(user_id, [])
        contributions = [
            self.contributions[cid] for cid in contribution_ids
            if cid in self.contributions
        ]

        if contribution_type:
            contributions = [c for c in contributions if c.contribution_type == contribution_type]

        if since:
            contributions = [c for c in contributions if c.timestamp >= since]

        # Sort by timestamp descending
        contributions.sort(key=lambda c: c.timestamp, reverse=True)

        return contributions[:limit]

    def get_target_history(
        self,
        target_id: str,
        limit: int = 100
    ) -> List[Contribution]:
        """Get contribution history for a target."""
        contribution_ids = self.target_contributions.get(target_id, [])
        contributions = [
            self.contributions[cid] for cid in contribution_ids
            if cid in self.contributions
        ]

        # Sort by timestamp descending
        contributions.sort(key=lambda c: c.timestamp, reverse=True)

        return contributions[:limit]

    def get_user_stats(self, user_id: str, display_name: Optional[str] = None) -> UserContributionStats:
        """Get comprehensive statistics for a user."""
        contribution_ids = self.user_contributions.get(user_id, [])
        contributions = [
            self.contributions[cid] for cid in contribution_ids
            if cid in self.contributions
        ]

        if not contributions:
            return UserContributionStats(
                user_id=user_id,
                display_name=display_name or self.user_names.get(user_id, user_id)
            )

        stats = UserContributionStats(
            user_id=user_id,
            display_name=display_name or self.user_names.get(user_id, user_id)
        )

        # Count by type
        for c in contributions:
            type_key = c.contribution_type.value
            stats.contribution_counts[type_key] = stats.contribution_counts.get(type_key, 0) + 1

        # Quality distribution
        rated = [c for c in contributions if c.quality_rating]
        for c in rated:
            rating_key = c.quality_rating.value
            stats.quality_distribution[rating_key] = stats.quality_distribution.get(rating_key, 0) + 1

        # Average quality (map to numeric)
        quality_map = {
            QualityRating.EXCELLENT: 5,
            QualityRating.GOOD: 4,
            QualityRating.ACCEPTABLE: 3,
            QualityRating.NEEDS_IMPROVEMENT: 2,
            QualityRating.POOR: 1
        }
        if rated:
            total_quality = sum(quality_map[c.quality_rating] for c in rated)
            stats.avg_quality_score = total_quality / len(rated) / 5.0  # Normalize to 0-1

        # Impact metrics
        stats.total_views = sum(c.view_count for c in contributions)
        stats.total_uses = sum(c.use_count for c in contributions)
        stats.total_citations = sum(c.citation_count for c in contributions)
        stats.total_upvotes = sum(c.upvotes for c in contributions)
        stats.total_downvotes = sum(c.downvotes for c in contributions)

        # Average impact
        impact_scores = [c.calculate_impact_score() for c in contributions]
        stats.avg_impact_score = sum(impact_scores) / len(impact_scores)

        # Activity metrics
        sorted_contribs = sorted(contributions, key=lambda c: c.timestamp)
        stats.first_contribution = sorted_contribs[0].timestamp
        stats.last_contribution = sorted_contribs[-1].timestamp

        # Active days
        active_days = set(c.timestamp.date() for c in contributions)
        stats.active_days = len(active_days)

        # Current streak
        stats.current_streak = self._calculate_streak(list(active_days))

        # Content metrics
        stats.total_content_size = sum(c.content_size for c in contributions)
        stats.unique_targets = len(set(c.target_id for c in contributions))

        return stats

    def _calculate_streak(self, active_days: List) -> int:
        """Calculate current contribution streak."""
        if not active_days:
            return 0

        sorted_days = sorted(active_days, reverse=True)
        today = datetime.now().date()

        if sorted_days[0] != today and sorted_days[0] != today - timedelta(days=1):
            return 0

        streak = 1
        for i in range(len(sorted_days) - 1):
            if sorted_days[i] - sorted_days[i + 1] == timedelta(days=1):
                streak += 1
            else:
                break

        return streak

    def get_leaderboard(
        self,
        metric: str = "contributions",  # contributions, impact, quality, uses
        limit: int = 10,
        time_period: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """Get contribution leaderboard."""
        user_ids = list(self.user_contributions.keys())

        entries = []
        for user_id in user_ids:
            stats = self.get_user_stats(user_id)

            # Filter by time period if specified
            if time_period:
                since = datetime.now() - time_period
                contributions = self.get_user_contributions(user_id, since=since, limit=10000)
                if not contributions:
                    continue

            score = 0
            if metric == "contributions":
                score = stats.total_contributions()
            elif metric == "impact":
                score = stats.avg_impact_score
            elif metric == "quality":
                score = stats.avg_quality_score
            elif metric == "uses":
                score = stats.total_uses
            elif metric == "votes":
                score = stats.net_votes()

            entries.append({
                "user_id": user_id,
                "display_name": stats.display_name,
                "score": score,
                "stats": stats.to_dict()
            })

        # Sort by score descending
        entries.sort(key=lambda e: e["score"], reverse=True)

        # Add rank
        for i, entry in enumerate(entries[:limit], 1):
            entry["rank"] = i

        return entries[:limit]

    def on(self, event: str, handler: Callable):
        """Register event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def _emit_event(self, event: str, data: Dict[str, Any]):
        """Emit event to handlers."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                handler(event, data)
            except Exception as e:
                logger.error(f"Attribution event handler error: {e}")

    def to_state(self) -> Dict[str, Any]:
        """Export attribution state for persistence."""
        return {
            "knowledge_base_id": self.knowledge_base_id,
            "total_contributions": len(self.contributions),
            "unique_contributors": len(self.user_contributions),
            "unique_targets": len(self.target_contributions),
            "contributions": {k: v.to_dict() for k, v in self.contributions.items()}
        }


# Factory function
def create_attribution_manager(knowledge_base_id: Optional[str] = None) -> AttributionManager:
    """Create an attribution manager."""
    return AttributionManager(knowledge_base_id=knowledge_base_id)


# ============================================================================
# MRF Refinement Integration (Phase 3 - December 2025)
# ============================================================================

@dataclass
class AnnotationWithRefinement:
    """
    An annotation contribution with optional MRF refinement.

    Extends the standard Contribution with refinement-specific fields.
    """
    contribution: Contribution
    original_text: str
    refined_text: Optional[str] = None
    refinement_applied: bool = False
    refinement_strategy: Optional[str] = None
    quality_before: float = 0.0
    quality_after: float = 0.0
    refinement_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    user_accepted_refinement: Optional[bool] = None
    refinement_timestamp: Optional[datetime] = None

    @property
    def quality_improvement(self) -> float:
        """Calculate quality improvement from refinement."""
        return self.quality_after - self.quality_before

    @property
    def final_text(self) -> str:
        """Get the final text (refined if applied, original otherwise)."""
        if self.refinement_applied and self.refined_text:
            return self.refined_text
        return self.original_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contribution": self.contribution.to_dict(),
            "original_text": self.original_text,
            "refined_text": self.refined_text,
            "refinement_applied": self.refinement_applied,
            "refinement_strategy": self.refinement_strategy,
            "quality_before": self.quality_before,
            "quality_after": self.quality_after,
            "quality_improvement": self.quality_improvement,
            "refinement_suggestions": self.refinement_suggestions,
            "user_accepted_refinement": self.user_accepted_refinement,
            "refinement_timestamp": self.refinement_timestamp.isoformat() if self.refinement_timestamp else None
        }


class RefinementAwareAttributionManager(AttributionManager):
    """
    Attribution manager with MRF refinement integration.

    Extends AttributionManager with:
    - Automatic annotation refinement on creation
    - Tracking of refinement acceptance/rejection
    - Quality metrics for refined annotations
    - Learning from refinement feedback
    """

    def __init__(
        self,
        knowledge_base_id: Optional[str] = None,
        refiner: Optional['AnnotationRefiner'] = None,
        auto_refine: bool = True,
        refinement_threshold: float = 0.7
    ):
        """
        Initialize refinement-aware attribution manager.

        Args:
            knowledge_base_id: Knowledge base identifier
            refiner: AnnotationRefiner instance (created if None)
            auto_refine: Whether to auto-refine annotations
            refinement_threshold: Quality threshold below which to suggest refinement
        """
        super().__init__(knowledge_base_id=knowledge_base_id)

        self.refiner = refiner
        self.auto_refine = auto_refine
        self.refinement_threshold = refinement_threshold

        # Refinement tracking
        self.refined_annotations: Dict[str, AnnotationWithRefinement] = {}
        self.refinement_stats = {
            "total_refined": 0,
            "accepted": 0,
            "rejected": 0,
            "auto_applied": 0,
            "avg_quality_improvement": 0.0
        }

    def _get_refiner(self) -> Optional['AnnotationRefiner']:
        """Get or create the refiner."""
        if self.refiner is None:
            try:
                from .annotation_refinement import create_annotation_refiner
                self.refiner = create_annotation_refiner(enable_learning=True)
            except ImportError:
                logger.warning("AnnotationRefiner not available")
                return None
        return self.refiner

    def record_annotation(
        self,
        user_id: str,
        user_name: str,
        target_id: str,
        annotation_text: str,
        annotation_type: str = "note",
        session_id: Optional[str] = None,
        auto_refine: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Contribution, Optional['RefinementResult']]:
        """
        Record an annotation with optional refinement.

        Args:
            user_id: User ID
            user_name: User display name
            target_id: Target node/object ID
            annotation_text: The annotation content
            annotation_type: Type of annotation (note, question, insight, connection)
            session_id: Session ID
            auto_refine: Override auto-refine setting
            metadata: Additional metadata

        Returns:
            Tuple of (Contribution, optional RefinementResult)
        """
        should_refine = auto_refine if auto_refine is not None else self.auto_refine

        # Record base contribution
        contribution = self.record_contribution(
            contribution_type=ContributionType.ANNOTATION,
            user_id=user_id,
            user_name=user_name,
            target_type="node",
            target_id=target_id,
            content_preview=annotation_text[:200],
            content_size=len(annotation_text),
            session_id=session_id,
            metadata={
                **(metadata or {}),
                "annotation_type": annotation_type,
                "original_text": annotation_text
            }
        )

        refinement_result = None

        # Attempt refinement if enabled
        if should_refine:
            refiner = self._get_refiner()
            if refiner:
                try:
                    from .annotation_refinement import AnnotationType
                    ann_type = AnnotationType(annotation_type)

                    refinement_result = refiner.refine(
                        annotation_id=contribution.contribution_id,
                        text=annotation_text,
                        annotation_type=ann_type,
                        context={
                            "node_id": target_id,
                            "session_id": session_id,
                            "user_id": user_id
                        }
                    )

                    # Track refined annotation
                    self.refined_annotations[contribution.contribution_id] = AnnotationWithRefinement(
                        contribution=contribution,
                        original_text=annotation_text,
                        refined_text=refinement_result.final_text,
                        refinement_applied=False,  # User must accept
                        refinement_strategy=refinement_result.strategy_used.value,
                        quality_before=refinement_result.quality_before,
                        quality_after=refinement_result.quality_after,
                        refinement_suggestions=[s.to_dict() for s in refinement_result.suggestions]
                    )

                    self.refinement_stats["total_refined"] += 1

                    # Emit event
                    self._emit_event("annotation_refined", {
                        "contribution_id": contribution.contribution_id,
                        "quality_before": refinement_result.quality_before,
                        "quality_after": refinement_result.quality_after,
                        "strategy": refinement_result.strategy_used.value,
                        "suggestions_count": len(refinement_result.suggestions)
                    })

                except Exception as e:
                    logger.error(f"Refinement failed: {e}")

        return contribution, refinement_result

    def accept_refinement(
        self,
        contribution_id: str,
        user_id: str
    ) -> bool:
        """
        Accept a refinement suggestion.

        Args:
            contribution_id: The contribution ID
            user_id: The user accepting the refinement

        Returns:
            True if refinement was accepted
        """
        refined = self.refined_annotations.get(contribution_id)
        if not refined:
            return False

        if refined.refinement_applied:
            return False  # Already applied

        refined.refinement_applied = True
        refined.user_accepted_refinement = True
        refined.refinement_timestamp = datetime.now()

        # Update contribution
        contribution = self.contributions.get(contribution_id)
        if contribution:
            contribution.content_preview = refined.refined_text[:200]
            contribution.metadata["refined_text"] = refined.refined_text
            contribution.metadata["refinement_accepted"] = True
            contribution.metadata["refinement_accepted_by"] = user_id

        self.refinement_stats["accepted"] += 1
        self._update_avg_quality_improvement(refined.quality_improvement)

        # Provide positive feedback to refiner
        refiner = self._get_refiner()
        if refiner and refiner.enable_learning:
            refiner.refinement_history.append({
                "annotation_id": contribution_id,
                "accepted": True,
                "quality_improvement": refined.quality_improvement,
                "timestamp": datetime.now().isoformat()
            })

        self._emit_event("refinement_accepted", {
            "contribution_id": contribution_id,
            "user_id": user_id,
            "quality_improvement": refined.quality_improvement
        })

        return True

    def reject_refinement(
        self,
        contribution_id: str,
        user_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Reject a refinement suggestion.

        Args:
            contribution_id: The contribution ID
            user_id: The user rejecting the refinement
            reason: Optional reason for rejection

        Returns:
            True if refinement was rejected
        """
        refined = self.refined_annotations.get(contribution_id)
        if not refined:
            return False

        if refined.refinement_applied:
            return False  # Already applied

        refined.user_accepted_refinement = False
        refined.refinement_timestamp = datetime.now()

        # Update contribution metadata
        contribution = self.contributions.get(contribution_id)
        if contribution:
            contribution.metadata["refinement_rejected"] = True
            contribution.metadata["refinement_rejected_by"] = user_id
            if reason:
                contribution.metadata["refinement_rejection_reason"] = reason

        self.refinement_stats["rejected"] += 1

        # Provide negative feedback to refiner
        refiner = self._get_refiner()
        if refiner and refiner.enable_learning:
            refiner.refinement_history.append({
                "annotation_id": contribution_id,
                "accepted": False,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

        self._emit_event("refinement_rejected", {
            "contribution_id": contribution_id,
            "user_id": user_id,
            "reason": reason
        })

        return True

    def get_refinement_suggestions(
        self,
        contribution_id: str
    ) -> Optional[AnnotationWithRefinement]:
        """Get refinement suggestions for an annotation."""
        return self.refined_annotations.get(contribution_id)

    def get_pending_refinements(
        self,
        user_id: Optional[str] = None
    ) -> List[AnnotationWithRefinement]:
        """
        Get annotations with pending refinement suggestions.

        Args:
            user_id: Optional filter by user

        Returns:
            List of annotations with pending refinements
        """
        pending = []
        for ann_id, refined in self.refined_annotations.items():
            if refined.user_accepted_refinement is None:  # Not yet decided
                if user_id is None or refined.contribution.user_id == user_id:
                    pending.append(refined)

        return pending

    def get_refinement_stats(self) -> Dict[str, Any]:
        """Get refinement statistics."""
        stats = self.refinement_stats.copy()
        stats["pending"] = len(self.get_pending_refinements())
        stats["acceptance_rate"] = (
            stats["accepted"] / stats["total_refined"]
            if stats["total_refined"] > 0 else 0.0
        )
        return stats

    def _update_avg_quality_improvement(self, new_improvement: float):
        """Update running average of quality improvement."""
        accepted = self.refinement_stats["accepted"]
        current_avg = self.refinement_stats["avg_quality_improvement"]

        # Running average
        self.refinement_stats["avg_quality_improvement"] = (
            (current_avg * (accepted - 1) + new_improvement) / accepted
            if accepted > 0 else new_improvement
        )


# Factory for refinement-aware manager
def create_refinement_aware_attribution_manager(
    knowledge_base_id: Optional[str] = None,
    auto_refine: bool = True,
    refinement_threshold: float = 0.7
) -> RefinementAwareAttributionManager:
    """
    Create an attribution manager with MRF refinement integration.

    Args:
        knowledge_base_id: Knowledge base identifier
        auto_refine: Whether to automatically refine annotations
        refinement_threshold: Quality threshold for suggesting refinement

    Returns:
        RefinementAwareAttributionManager instance
    """
    return RefinementAwareAttributionManager(
        knowledge_base_id=knowledge_base_id,
        auto_refine=auto_refine,
        refinement_threshold=refinement_threshold
    )
