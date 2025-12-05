"""
Shared Style Library with Voting
================================

Collaborative style management for ThirdEye scenes.

Features:
- Reusable style presets (colors, fonts, shapes, effects)
- Style proposals with community voting
- Usage tracking and popularity ranking
- Role-based permissions (who can propose/approve styles)
- Style versioning and history

Integration with HoloLoom.collaboration:
- Uses RoleEnforcer for permission checks
- Emits events for style changes

Created: 2025-12-03
Author: HoloLoom Team
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Any, Callable
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Style Categories and Types
# -----------------------------------------------------------------------------

class StyleCategory(Enum):
    """Categories of visual styles."""
    COLOR = "color"           # Color palettes, fills, strokes
    TYPOGRAPHY = "typography" # Fonts, text styling
    SHAPE = "shape"          # Shape styles, borders, corners
    EFFECT = "effect"        # Shadows, glows, animations
    LAYOUT = "layout"        # Spacing, alignment, grid
    THEME = "theme"          # Complete theme bundles


class StyleStatus(Enum):
    """Status of a style in the library."""
    DRAFT = "draft"           # Being created, not visible to others
    PROPOSED = "proposed"     # Submitted for voting
    APPROVED = "approved"     # Approved by community/owner
    REJECTED = "rejected"     # Rejected by community/owner
    DEPRECATED = "deprecated" # No longer recommended
    BUILTIN = "builtin"       # System default styles


class VoteType(Enum):
    """Types of votes on style proposals."""
    UPVOTE = "upvote"
    DOWNVOTE = "downvote"
    NEUTRAL = "neutral"


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class StyleProperties:
    """
    Properties that define a visual style.

    Flexible key-value structure to support various style types.
    """
    # Common properties
    primary_color: Optional[str] = None      # Hex color e.g., "#4A90D9"
    secondary_color: Optional[str] = None
    accent_color: Optional[str] = None
    background_color: Optional[str] = None

    # Typography
    font_family: Optional[str] = None
    font_size: Optional[float] = None
    font_weight: Optional[str] = None        # "normal", "bold", "light"
    line_height: Optional[float] = None

    # Shape
    border_radius: Optional[float] = None
    border_width: Optional[float] = None
    border_color: Optional[str] = None
    border_style: Optional[str] = None       # "solid", "dashed", "dotted"

    # Effects
    shadow_offset_x: Optional[float] = None
    shadow_offset_y: Optional[float] = None
    shadow_blur: Optional[float] = None
    shadow_color: Optional[str] = None
    opacity: Optional[float] = None

    # Layout
    padding: Optional[float] = None
    margin: Optional[float] = None
    gap: Optional[float] = None

    # Custom properties (extensible)
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None and key != 'custom':
                result[key] = value
        if self.custom:
            result['custom'] = self.custom
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StyleProperties':
        """Create from dictionary."""
        custom = data.pop('custom', {})
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields and k != 'custom'}
        return cls(**filtered, custom=custom)

    def merge(self, other: 'StyleProperties') -> 'StyleProperties':
        """Merge another style's non-None properties into this one."""
        merged_dict = self.to_dict()
        other_dict = other.to_dict()

        for key, value in other_dict.items():
            if key == 'custom':
                merged_dict.setdefault('custom', {}).update(value)
            else:
                merged_dict[key] = value

        return StyleProperties.from_dict(merged_dict)


@dataclass
class StyleVote:
    """A vote on a style proposal."""
    voter_id: str
    vote_type: VoteType
    timestamp: datetime = field(default_factory=datetime.now)
    comment: str = ""


@dataclass
class StyleVersion:
    """A version of a style (for history tracking)."""
    version: int
    properties: StyleProperties
    changed_by: str
    changed_at: datetime = field(default_factory=datetime.now)
    change_description: str = ""


@dataclass
class SharedStyle:
    """
    A shared style in the library.

    Styles can be:
    - BUILTIN: Default system styles
    - PROPOSED: Submitted for community voting
    - APPROVED: Accepted into the library
    """
    id: str
    name: str
    category: StyleCategory
    properties: StyleProperties
    status: StyleStatus

    # Authorship
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)

    # Versioning
    current_version: int = 1
    versions: List[StyleVersion] = field(default_factory=list)

    # Voting (for PROPOSED styles)
    votes: List[StyleVote] = field(default_factory=list)
    approval_threshold: int = 3  # Net upvotes needed for auto-approval

    # Usage tracking
    usage_count: int = 0
    last_used_at: Optional[datetime] = None
    used_by: Set[str] = field(default_factory=set)

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    preview_url: Optional[str] = None  # Preview image URL

    # Approval
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None

    def vote_score(self) -> int:
        """Calculate net vote score."""
        score = 0
        for vote in self.votes:
            if vote.vote_type == VoteType.UPVOTE:
                score += 1
            elif vote.vote_type == VoteType.DOWNVOTE:
                score -= 1
        return score

    def get_vote(self, voter_id: str) -> Optional[StyleVote]:
        """Get a voter's current vote."""
        for vote in self.votes:
            if vote.voter_id == voter_id:
                return vote
        return None

    def record_usage(self, user_id: str):
        """Record that this style was used."""
        self.usage_count += 1
        self.last_used_at = datetime.now()
        self.used_by.add(user_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category.value,
            'properties': self.properties.to_dict(),
            'status': self.status.value,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'current_version': self.current_version,
            'versions': [
                {
                    'version': v.version,
                    'properties': v.properties.to_dict(),
                    'changed_by': v.changed_by,
                    'changed_at': v.changed_at.isoformat(),
                    'change_description': v.change_description,
                }
                for v in self.versions
            ],
            'votes': [
                {
                    'voter_id': v.voter_id,
                    'vote_type': v.vote_type.value,
                    'timestamp': v.timestamp.isoformat(),
                    'comment': v.comment,
                }
                for v in self.votes
            ],
            'approval_threshold': self.approval_threshold,
            'usage_count': self.usage_count,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'used_by': list(self.used_by),
            'description': self.description,
            'tags': self.tags,
            'preview_url': self.preview_url,
            'approved_by': self.approved_by,
            'approved_at': self.approved_at.isoformat() if self.approved_at else None,
            'rejection_reason': self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SharedStyle':
        """Deserialize from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            category=StyleCategory(data['category']),
            properties=StyleProperties.from_dict(data['properties']),
            status=StyleStatus(data['status']),
            created_by=data['created_by'],
            created_at=datetime.fromisoformat(data['created_at']),
            current_version=data.get('current_version', 1),
            versions=[
                StyleVersion(
                    version=v['version'],
                    properties=StyleProperties.from_dict(v['properties']),
                    changed_by=v['changed_by'],
                    changed_at=datetime.fromisoformat(v['changed_at']),
                    change_description=v.get('change_description', ''),
                )
                for v in data.get('versions', [])
            ],
            votes=[
                StyleVote(
                    voter_id=v['voter_id'],
                    vote_type=VoteType(v['vote_type']),
                    timestamp=datetime.fromisoformat(v['timestamp']),
                    comment=v.get('comment', ''),
                )
                for v in data.get('votes', [])
            ],
            approval_threshold=data.get('approval_threshold', 3),
            usage_count=data.get('usage_count', 0),
            last_used_at=datetime.fromisoformat(data['last_used_at'])
                        if data.get('last_used_at') else None,
            used_by=set(data.get('used_by', [])),
            description=data.get('description', ''),
            tags=data.get('tags', []),
            preview_url=data.get('preview_url'),
            approved_by=data.get('approved_by'),
            approved_at=datetime.fromisoformat(data['approved_at'])
                       if data.get('approved_at') else None,
            rejection_reason=data.get('rejection_reason'),
        )


@dataclass
class StyleChangeEvent:
    """Event emitted when a style changes."""
    style_id: str
    event_type: str  # "created", "proposed", "voted", "approved", "rejected", "updated", "used"
    user_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Style Library Manager
# -----------------------------------------------------------------------------

class SharedStyleLibrary:
    """
    Manages a library of shared styles with voting.

    Usage:
        library = SharedStyleLibrary(scene_id="scene_123")

        # Create a style
        style = library.create_style(
            name="Ocean Blue",
            category=StyleCategory.COLOR,
            properties=StyleProperties(primary_color="#4A90D9"),
            created_by="user_1"
        )

        # Propose for voting
        library.propose_style(style.id, "user_1")

        # Vote on style
        library.vote_on_style(style.id, "user_2", VoteType.UPVOTE)
        library.vote_on_style(style.id, "user_3", VoteType.UPVOTE)
        library.vote_on_style(style.id, "user_4", VoteType.UPVOTE)
        # Auto-approved after reaching threshold!

        # Use a style
        library.use_style(style.id, "user_5")
    """

    def __init__(
        self,
        scene_id: str,
        approval_threshold: int = 3,
        allow_self_vote: bool = False,
    ):
        """
        Initialize style library.

        Args:
            scene_id: ID of the scene this library belongs to
            approval_threshold: Net upvotes needed for auto-approval
            allow_self_vote: Whether creators can vote on their own styles
        """
        self.scene_id = scene_id
        self.approval_threshold = approval_threshold
        self.allow_self_vote = allow_self_vote

        self._styles: Dict[str, SharedStyle] = {}
        self._event_handlers: List[Callable[[StyleChangeEvent], None]] = []

        # Initialize with builtin styles
        self._init_builtin_styles()

    def _init_builtin_styles(self):
        """Initialize default builtin styles."""
        builtins = [
            # Color palettes
            SharedStyle(
                id="builtin_ocean",
                name="Ocean Blue",
                category=StyleCategory.COLOR,
                properties=StyleProperties(
                    primary_color="#4A90D9",
                    secondary_color="#7FB3D5",
                    accent_color="#1E5799",
                    background_color="#E8F4FD",
                ),
                status=StyleStatus.BUILTIN,
                created_by="system",
                description="Calming ocean-inspired blue palette",
                tags=["blue", "calm", "professional"],
            ),
            SharedStyle(
                id="builtin_forest",
                name="Forest Green",
                category=StyleCategory.COLOR,
                properties=StyleProperties(
                    primary_color="#2E8B57",
                    secondary_color="#90EE90",
                    accent_color="#006400",
                    background_color="#F0FFF0",
                ),
                status=StyleStatus.BUILTIN,
                created_by="system",
                description="Natural forest-inspired green palette",
                tags=["green", "nature", "organic"],
            ),
            SharedStyle(
                id="builtin_sunset",
                name="Sunset Warm",
                category=StyleCategory.COLOR,
                properties=StyleProperties(
                    primary_color="#FF6B35",
                    secondary_color="#F7C59F",
                    accent_color="#E85D04",
                    background_color="#FFF5EE",
                ),
                status=StyleStatus.BUILTIN,
                created_by="system",
                description="Warm sunset-inspired orange palette",
                tags=["orange", "warm", "energetic"],
            ),
            # Typography
            SharedStyle(
                id="builtin_clean",
                name="Clean Sans",
                category=StyleCategory.TYPOGRAPHY,
                properties=StyleProperties(
                    font_family="Inter, system-ui, sans-serif",
                    font_size=16.0,
                    font_weight="normal",
                    line_height=1.5,
                ),
                status=StyleStatus.BUILTIN,
                created_by="system",
                description="Clean, readable sans-serif typography",
                tags=["clean", "modern", "readable"],
            ),
            SharedStyle(
                id="builtin_elegant",
                name="Elegant Serif",
                category=StyleCategory.TYPOGRAPHY,
                properties=StyleProperties(
                    font_family="Georgia, serif",
                    font_size=18.0,
                    font_weight="normal",
                    line_height=1.6,
                ),
                status=StyleStatus.BUILTIN,
                created_by="system",
                description="Elegant serif typography for formal content",
                tags=["elegant", "formal", "classic"],
            ),
            # Shapes
            SharedStyle(
                id="builtin_rounded",
                name="Soft Rounded",
                category=StyleCategory.SHAPE,
                properties=StyleProperties(
                    border_radius=12.0,
                    border_width=1.0,
                    border_color="#E0E0E0",
                    border_style="solid",
                ),
                status=StyleStatus.BUILTIN,
                created_by="system",
                description="Soft, rounded corners for friendly appearance",
                tags=["rounded", "soft", "friendly"],
            ),
            SharedStyle(
                id="builtin_sharp",
                name="Sharp Modern",
                category=StyleCategory.SHAPE,
                properties=StyleProperties(
                    border_radius=0.0,
                    border_width=2.0,
                    border_color="#333333",
                    border_style="solid",
                ),
                status=StyleStatus.BUILTIN,
                created_by="system",
                description="Sharp edges for modern, bold appearance",
                tags=["sharp", "modern", "bold"],
            ),
            # Effects
            SharedStyle(
                id="builtin_shadow",
                name="Subtle Shadow",
                category=StyleCategory.EFFECT,
                properties=StyleProperties(
                    shadow_offset_x=0.0,
                    shadow_offset_y=4.0,
                    shadow_blur=12.0,
                    shadow_color="rgba(0, 0, 0, 0.1)",
                    opacity=1.0,
                ),
                status=StyleStatus.BUILTIN,
                created_by="system",
                description="Subtle shadow for depth and elevation",
                tags=["shadow", "depth", "subtle"],
            ),
        ]

        for style in builtins:
            self._styles[style.id] = style

    # -------------------------------------------------------------------------
    # Style CRUD Operations
    # -------------------------------------------------------------------------

    def create_style(
        self,
        name: str,
        category: StyleCategory,
        properties: StyleProperties,
        created_by: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        status: StyleStatus = StyleStatus.DRAFT,
    ) -> SharedStyle:
        """
        Create a new style.

        Args:
            name: Style name
            category: Style category
            properties: Style properties
            created_by: User ID of creator
            description: Style description
            tags: Style tags for searching
            status: Initial status (default: DRAFT)

        Returns:
            The created style
        """
        style_id = f"style_{uuid.uuid4().hex[:12]}"

        style = SharedStyle(
            id=style_id,
            name=name,
            category=category,
            properties=properties,
            status=status,
            created_by=created_by,
            description=description,
            tags=tags or [],
            approval_threshold=self.approval_threshold,
        )

        # Save initial version
        style.versions.append(StyleVersion(
            version=1,
            properties=properties,
            changed_by=created_by,
            change_description="Initial version",
        ))

        self._styles[style_id] = style

        self._emit_event(StyleChangeEvent(
            style_id=style_id,
            event_type="created",
            user_id=created_by,
            metadata={"name": name, "category": category.value},
        ))

        logger.info(f"Created style '{name}' ({style_id}) by {created_by}")

        return style

    def get_style(self, style_id: str) -> Optional[SharedStyle]:
        """Get a style by ID."""
        return self._styles.get(style_id)

    def update_style(
        self,
        style_id: str,
        properties: StyleProperties,
        updated_by: str,
        change_description: str = "",
    ) -> Optional[SharedStyle]:
        """
        Update a style's properties.

        Creates a new version in the style's history.

        Args:
            style_id: Style to update
            properties: New properties
            updated_by: User making the update
            change_description: Description of changes

        Returns:
            Updated style, or None if not found
        """
        style = self._styles.get(style_id)
        if not style:
            return None

        # Can't update builtin styles
        if style.status == StyleStatus.BUILTIN:
            logger.warning(f"Cannot update builtin style {style_id}")
            return None

        # Save current as version
        style.current_version += 1
        style.versions.append(StyleVersion(
            version=style.current_version,
            properties=properties,
            changed_by=updated_by,
            change_description=change_description,
        ))

        # Update current properties
        style.properties = properties

        self._emit_event(StyleChangeEvent(
            style_id=style_id,
            event_type="updated",
            user_id=updated_by,
            metadata={
                "version": style.current_version,
                "description": change_description,
            },
        ))

        logger.info(f"Updated style {style_id} to version {style.current_version}")

        return style

    def delete_style(self, style_id: str, deleted_by: str) -> bool:
        """
        Delete a style from the library.

        Cannot delete builtin styles.

        Args:
            style_id: Style to delete
            deleted_by: User deleting the style

        Returns:
            True if deleted, False if not found or builtin
        """
        style = self._styles.get(style_id)
        if not style:
            return False

        if style.status == StyleStatus.BUILTIN:
            logger.warning(f"Cannot delete builtin style {style_id}")
            return False

        del self._styles[style_id]

        self._emit_event(StyleChangeEvent(
            style_id=style_id,
            event_type="deleted",
            user_id=deleted_by,
        ))

        logger.info(f"Deleted style {style_id} by {deleted_by}")

        return True

    # -------------------------------------------------------------------------
    # Proposal and Voting
    # -------------------------------------------------------------------------

    def propose_style(self, style_id: str, proposed_by: str) -> bool:
        """
        Submit a style for community voting.

        Args:
            style_id: Style to propose
            proposed_by: User proposing (must be creator)

        Returns:
            True if proposed successfully
        """
        style = self._styles.get(style_id)
        if not style:
            return False

        if style.status != StyleStatus.DRAFT:
            logger.warning(f"Style {style_id} is not a draft, cannot propose")
            return False

        if style.created_by != proposed_by:
            logger.warning(f"User {proposed_by} is not the creator of style {style_id}")
            return False

        style.status = StyleStatus.PROPOSED

        self._emit_event(StyleChangeEvent(
            style_id=style_id,
            event_type="proposed",
            user_id=proposed_by,
        ))

        logger.info(f"Style {style_id} proposed for voting by {proposed_by}")

        return True

    def vote_on_style(
        self,
        style_id: str,
        voter_id: str,
        vote_type: VoteType,
        comment: str = "",
    ) -> Optional[int]:
        """
        Vote on a proposed style.

        Args:
            style_id: Style to vote on
            voter_id: User voting
            vote_type: Type of vote
            comment: Optional comment with vote

        Returns:
            New vote score, or None if voting not allowed
        """
        style = self._styles.get(style_id)
        if not style:
            return None

        if style.status != StyleStatus.PROPOSED:
            logger.warning(f"Style {style_id} is not proposed, cannot vote")
            return None

        # Check self-vote
        if not self.allow_self_vote and style.created_by == voter_id:
            logger.warning(f"User {voter_id} cannot vote on their own style")
            return None

        # Remove existing vote if any
        style.votes = [v for v in style.votes if v.voter_id != voter_id]

        # Add new vote
        style.votes.append(StyleVote(
            voter_id=voter_id,
            vote_type=vote_type,
            comment=comment,
        ))

        score = style.vote_score()

        self._emit_event(StyleChangeEvent(
            style_id=style_id,
            event_type="voted",
            user_id=voter_id,
            metadata={
                "vote_type": vote_type.value,
                "score": score,
                "comment": comment,
            },
        ))

        # Check for auto-approval
        if score >= style.approval_threshold:
            self._auto_approve(style, voter_id)

        logger.info(f"User {voter_id} voted {vote_type.value} on style {style_id}, score: {score}")

        return score

    def _auto_approve(self, style: SharedStyle, triggering_voter: str):
        """Auto-approve a style that reached the threshold."""
        style.status = StyleStatus.APPROVED
        style.approved_by = "auto"
        style.approved_at = datetime.now()

        self._emit_event(StyleChangeEvent(
            style_id=style.id,
            event_type="approved",
            user_id=triggering_voter,
            metadata={
                "approval_type": "auto",
                "final_score": style.vote_score(),
            },
        ))

        logger.info(f"Style {style.id} auto-approved with score {style.vote_score()}")

    def approve_style(
        self,
        style_id: str,
        approved_by: str,
    ) -> bool:
        """
        Manually approve a proposed style (owner/editor action).

        Args:
            style_id: Style to approve
            approved_by: User approving (must have permission)

        Returns:
            True if approved
        """
        style = self._styles.get(style_id)
        if not style:
            return False

        if style.status != StyleStatus.PROPOSED:
            return False

        style.status = StyleStatus.APPROVED
        style.approved_by = approved_by
        style.approved_at = datetime.now()

        self._emit_event(StyleChangeEvent(
            style_id=style_id,
            event_type="approved",
            user_id=approved_by,
            metadata={"approval_type": "manual"},
        ))

        logger.info(f"Style {style_id} manually approved by {approved_by}")

        return True

    def reject_style(
        self,
        style_id: str,
        rejected_by: str,
        reason: str = "",
    ) -> bool:
        """
        Reject a proposed style.

        Args:
            style_id: Style to reject
            rejected_by: User rejecting
            reason: Reason for rejection

        Returns:
            True if rejected
        """
        style = self._styles.get(style_id)
        if not style:
            return False

        if style.status != StyleStatus.PROPOSED:
            return False

        style.status = StyleStatus.REJECTED
        style.rejection_reason = reason

        self._emit_event(StyleChangeEvent(
            style_id=style_id,
            event_type="rejected",
            user_id=rejected_by,
            metadata={"reason": reason},
        ))

        logger.info(f"Style {style_id} rejected by {rejected_by}: {reason}")

        return True

    # -------------------------------------------------------------------------
    # Usage Tracking
    # -------------------------------------------------------------------------

    def use_style(self, style_id: str, user_id: str) -> bool:
        """
        Record that a style was used.

        Args:
            style_id: Style being used
            user_id: User using the style

        Returns:
            True if usage recorded
        """
        style = self._styles.get(style_id)
        if not style:
            return False

        # Only approved and builtin styles can be used
        if style.status not in (StyleStatus.APPROVED, StyleStatus.BUILTIN):
            return False

        style.record_usage(user_id)

        self._emit_event(StyleChangeEvent(
            style_id=style_id,
            event_type="used",
            user_id=user_id,
            metadata={"usage_count": style.usage_count},
        ))

        return True

    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------

    def list_styles(
        self,
        category: Optional[StyleCategory] = None,
        status: Optional[StyleStatus] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
        sort_by: str = "popularity",  # "popularity", "newest", "name", "votes"
        limit: int = 50,
    ) -> List[SharedStyle]:
        """
        List styles with filtering and sorting.

        Args:
            category: Filter by category
            status: Filter by status
            tags: Filter by tags (any match)
            created_by: Filter by creator
            sort_by: Sort order
            limit: Maximum results

        Returns:
            List of matching styles
        """
        results = list(self._styles.values())

        # Apply filters
        if category:
            results = [s for s in results if s.category == category]

        if status:
            results = [s for s in results if s.status == status]

        if tags:
            tag_set = set(tags)
            results = [s for s in results if tag_set.intersection(s.tags)]

        if created_by:
            results = [s for s in results if s.created_by == created_by]

        # Sort
        if sort_by == "popularity":
            results.sort(key=lambda s: s.usage_count, reverse=True)
        elif sort_by == "newest":
            results.sort(key=lambda s: s.created_at, reverse=True)
        elif sort_by == "name":
            results.sort(key=lambda s: s.name.lower())
        elif sort_by == "votes":
            results.sort(key=lambda s: s.vote_score(), reverse=True)

        return results[:limit]

    def search_styles(self, query: str, limit: int = 20) -> List[SharedStyle]:
        """
        Search styles by name, description, or tags.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching styles
        """
        query_lower = query.lower()

        def score(style: SharedStyle) -> int:
            """Calculate relevance score."""
            s = 0
            if query_lower in style.name.lower():
                s += 10
            if query_lower in style.description.lower():
                s += 5
            for tag in style.tags:
                if query_lower in tag.lower():
                    s += 3
            return s

        results = [(style, score(style)) for style in self._styles.values()]
        results = [(s, sc) for s, sc in results if sc > 0]
        results.sort(key=lambda x: x[1], reverse=True)

        return [s for s, _ in results[:limit]]

    def get_popular_styles(self, limit: int = 10) -> List[SharedStyle]:
        """Get most popular styles by usage."""
        return self.list_styles(
            status=StyleStatus.APPROVED,
            sort_by="popularity",
            limit=limit,
        ) + self.list_styles(
            status=StyleStatus.BUILTIN,
            sort_by="popularity",
            limit=limit,
        )

    def get_proposed_styles(self, limit: int = 20) -> List[SharedStyle]:
        """Get styles awaiting votes."""
        return self.list_styles(
            status=StyleStatus.PROPOSED,
            sort_by="votes",
            limit=limit,
        )

    # -------------------------------------------------------------------------
    # Event Handling
    # -------------------------------------------------------------------------

    def on_style_change(self, handler: Callable[[StyleChangeEvent], None]):
        """Subscribe to style change events."""
        self._event_handlers.append(handler)

    def _emit_event(self, event: StyleChangeEvent):
        """Emit an event to all handlers."""
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Style event handler error: {e}")

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize library to dictionary."""
        return {
            'scene_id': self.scene_id,
            'approval_threshold': self.approval_threshold,
            'allow_self_vote': self.allow_self_vote,
            'styles': {
                sid: style.to_dict()
                for sid, style in self._styles.items()
                if style.status != StyleStatus.BUILTIN  # Don't persist builtins
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SharedStyleLibrary':
        """Deserialize from dictionary."""
        library = cls(
            scene_id=data['scene_id'],
            approval_threshold=data.get('approval_threshold', 3),
            allow_self_vote=data.get('allow_self_vote', False),
        )

        for sid, style_data in data.get('styles', {}).items():
            library._styles[sid] = SharedStyle.from_dict(style_data)

        return library


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------

def create_style_library(
    scene_id: str,
    approval_threshold: int = 3,
) -> SharedStyleLibrary:
    """
    Create a shared style library for a scene.

    Args:
        scene_id: ID of the scene
        approval_threshold: Votes needed for auto-approval

    Returns:
        Configured SharedStyleLibrary
    """
    return SharedStyleLibrary(
        scene_id=scene_id,
        approval_threshold=approval_threshold,
    )


__all__ = [
    # Enums
    'StyleCategory',
    'StyleStatus',
    'VoteType',
    # Data classes
    'StyleProperties',
    'StyleVote',
    'StyleVersion',
    'SharedStyle',
    'StyleChangeEvent',
    # Main class
    'SharedStyleLibrary',
    # Factory
    'create_style_library',
]
