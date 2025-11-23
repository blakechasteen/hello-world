"""
Collaboration Engine for Multi-User Editing

Provides real-time collaboration features for nonfiction writing:
- Multi-user draft editing with conflict resolution
- Change tracking (who changed what and when)
- Comment system (inline comments and suggestions)
- Version control integration
- Lock management to prevent simultaneous edits

Features:
- User presence tracking
- Operational transforms for conflict-free editing
- Comment threads with replies
- Suggested edits with accept/reject
- Change history with rollback
- Merge strategies for concurrent edits
"""

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path


class ChangeType(Enum):
    """Type of change made to document"""
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"
    COMMENT = "comment"
    SUGGESTION = "suggestion"


class MergeStrategy(Enum):
    """Strategy for merging conflicting changes"""
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    MANUAL_REVIEW = "manual_review"
    OPERATIONAL_TRANSFORM = "operational_transform"


class ConflictResolution(Enum):
    """Resolution status for conflicts"""
    AUTO_MERGED = "auto_merged"
    MANUAL_REQUIRED = "manual_required"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass
class User:
    """Collaborative user"""
    id: str
    name: str
    email: str
    avatar_url: Optional[str] = None
    is_active: bool = True
    last_seen: Optional[str] = None


@dataclass
class Change:
    """
    A single change to the document

    Uses operational transform model for conflict resolution.
    Each change has a position, type, and content.
    """
    id: str
    user_id: str
    timestamp: str
    change_type: ChangeType
    position: int  # Character position in document
    length: int  # Length of affected text
    content: str  # New content (for insert/replace)
    old_content: str = ""  # Old content (for undo)
    paragraph_id: Optional[str] = None
    merged: bool = False
    conflict_with: Optional[str] = None  # Change ID that conflicts


@dataclass
class Comment:
    """
    Inline comment on document

    Supports threaded discussions with replies.
    """
    id: str
    user_id: str
    timestamp: str
    position: int  # Character position
    length: int  # Selection length
    text: str
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None
    replies: List['Comment'] = field(default_factory=list)
    parent_id: Optional[str] = None


@dataclass
class Suggestion:
    """
    Suggested edit with accept/reject workflow
    """
    id: str
    user_id: str
    timestamp: str
    position: int
    length: int
    suggested_text: str
    original_text: str
    rationale: str
    status: str = "pending"  # pending, accepted, rejected
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None


@dataclass
class Lock:
    """
    Paragraph lock to prevent simultaneous editing
    """
    paragraph_id: str
    user_id: str
    acquired_at: str
    expires_at: str


@dataclass
class Version:
    """
    Document version snapshot
    """
    id: str
    timestamp: str
    user_id: str
    content: str
    changes_since_previous: List[str]  # Change IDs
    message: Optional[str] = None


@dataclass
class CollaborationSession:
    """
    Active collaboration session

    Tracks all users, changes, comments, and locks.
    """
    session_id: str
    document_id: str
    created_at: str
    active_users: List[User] = field(default_factory=list)
    changes: List[Change] = field(default_factory=list)
    comments: List[Comment] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)
    locks: Dict[str, Lock] = field(default_factory=dict)
    versions: List[Version] = field(default_factory=list)


class CollaborationEngine:
    """
    Collaboration engine for multi-user editing

    Provides:
    - User presence tracking
    - Change tracking with operational transforms
    - Comment threads
    - Suggested edits
    - Lock management
    - Version snapshots
    - Conflict resolution

    Usage:
        engine = CollaborationEngine()

        # Join session
        await engine.join_session(session_id, user)

        # Apply change
        change = await engine.apply_change(
            session_id,
            user_id,
            ChangeType.INSERT,
            position=100,
            content="new text"
        )

        # Add comment
        comment = await engine.add_comment(
            session_id,
            user_id,
            position=50,
            length=10,
            text="This needs clarification"
        )

        # Suggest edit
        suggestion = await engine.suggest_edit(
            session_id,
            user_id,
            position=20,
            length=5,
            suggested_text="better wording",
            rationale="More concise"
        )
    """

    def __init__(self, merge_strategy: MergeStrategy = MergeStrategy.OPERATIONAL_TRANSFORM):
        """
        Initialize collaboration engine

        Args:
            merge_strategy: Strategy for conflict resolution
        """
        self.sessions: Dict[str, CollaborationSession] = {}
        self.merge_strategy = merge_strategy

    async def create_session(
        self,
        document_id: str,
        initial_content: str = ""
    ) -> CollaborationSession:
        """
        Create new collaboration session

        Args:
            document_id: Unique document identifier
            initial_content: Initial document content

        Returns:
            New collaboration session
        """
        session_id = self._generate_id(f"{document_id}_{datetime.now().isoformat()}")

        session = CollaborationSession(
            session_id=session_id,
            document_id=document_id,
            created_at=datetime.now().isoformat()
        )

        # Create initial version
        if initial_content:
            version = Version(
                id=self._generate_id(f"v0_{session_id}"),
                timestamp=datetime.now().isoformat(),
                user_id="system",
                content=initial_content,
                changes_since_previous=[],
                message="Initial version"
            )
            session.versions.append(version)

        self.sessions[session_id] = session
        return session

    async def join_session(
        self,
        session_id: str,
        user: User
    ) -> bool:
        """
        User joins collaboration session

        Args:
            session_id: Session to join
            user: User joining

        Returns:
            True if joined successfully
        """
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        # Update or add user
        existing = next((u for u in session.active_users if u.id == user.id), None)
        if existing:
            existing.is_active = True
            existing.last_seen = datetime.now().isoformat()
        else:
            user.is_active = True
            user.last_seen = datetime.now().isoformat()
            session.active_users.append(user)

        return True

    async def leave_session(
        self,
        session_id: str,
        user_id: str
    ) -> bool:
        """
        User leaves collaboration session

        Args:
            session_id: Session to leave
            user_id: User leaving

        Returns:
            True if left successfully
        """
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        user = next((u for u in session.active_users if u.id == user_id), None)
        if user:
            user.is_active = False
            user.last_seen = datetime.now().isoformat()

            # Release all locks held by user
            locks_to_remove = [
                pid for pid, lock in session.locks.items()
                if lock.user_id == user_id
            ]
            for pid in locks_to_remove:
                del session.locks[pid]

        return True

    async def apply_change(
        self,
        session_id: str,
        user_id: str,
        change_type: ChangeType,
        position: int,
        content: str = "",
        length: int = 0,
        paragraph_id: Optional[str] = None
    ) -> Optional[Change]:
        """
        Apply change to document

        Args:
            session_id: Session ID
            user_id: User making change
            change_type: Type of change
            position: Character position
            content: New content (for insert/replace)
            length: Length of affected text
            paragraph_id: Paragraph being edited

        Returns:
            Applied change or None if failed
        """
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # Check lock if paragraph specified
        if paragraph_id and paragraph_id in session.locks:
            lock = session.locks[paragraph_id]
            if lock.user_id != user_id:
                # Paragraph locked by another user
                return None

        # Create change
        change = Change(
            id=self._generate_id(f"{session_id}_{user_id}_{datetime.now().isoformat()}"),
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
            change_type=change_type,
            position=position,
            length=length,
            content=content,
            paragraph_id=paragraph_id
        )

        # Check for conflicts with recent changes
        conflict = await self._detect_conflict(session, change)
        if conflict:
            change.conflict_with = conflict.id

            # Apply merge strategy
            if self.merge_strategy == MergeStrategy.OPERATIONAL_TRANSFORM:
                # Transform change to resolve conflict
                change = await self._operational_transform(change, conflict)
                change.merged = True
            elif self.merge_strategy == MergeStrategy.MANUAL_REVIEW:
                # Mark for manual review
                return change  # Don't add to session yet

        session.changes.append(change)
        return change

    async def add_comment(
        self,
        session_id: str,
        user_id: str,
        position: int,
        length: int,
        text: str,
        parent_id: Optional[str] = None
    ) -> Optional[Comment]:
        """
        Add comment to document

        Args:
            session_id: Session ID
            user_id: User commenting
            position: Character position
            length: Selection length
            text: Comment text
            parent_id: Parent comment (for replies)

        Returns:
            Created comment or None if failed
        """
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        comment = Comment(
            id=self._generate_id(f"comment_{session_id}_{user_id}_{datetime.now().isoformat()}"),
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
            position=position,
            length=length,
            text=text,
            parent_id=parent_id
        )

        # If reply, add to parent
        if parent_id:
            parent = next((c for c in session.comments if c.id == parent_id), None)
            if parent:
                parent.replies.append(comment)

        session.comments.append(comment)
        return comment

    async def resolve_comment(
        self,
        session_id: str,
        comment_id: str,
        user_id: str
    ) -> bool:
        """
        Resolve comment thread

        Args:
            session_id: Session ID
            comment_id: Comment to resolve
            user_id: User resolving

        Returns:
            True if resolved
        """
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        comment = next((c for c in session.comments if c.id == comment_id), None)
        if comment:
            comment.resolved = True
            comment.resolved_by = user_id
            comment.resolved_at = datetime.now().isoformat()
            return True

        return False

    async def suggest_edit(
        self,
        session_id: str,
        user_id: str,
        position: int,
        length: int,
        suggested_text: str,
        original_text: str,
        rationale: str
    ) -> Optional[Suggestion]:
        """
        Suggest edit to document

        Args:
            session_id: Session ID
            user_id: User suggesting
            position: Character position
            length: Selection length
            suggested_text: Suggested replacement
            original_text: Original text
            rationale: Reason for suggestion

        Returns:
            Created suggestion or None if failed
        """
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        suggestion = Suggestion(
            id=self._generate_id(f"suggestion_{session_id}_{user_id}_{datetime.now().isoformat()}"),
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
            position=position,
            length=length,
            suggested_text=suggested_text,
            original_text=original_text,
            rationale=rationale
        )

        session.suggestions.append(suggestion)
        return suggestion

    async def review_suggestion(
        self,
        session_id: str,
        suggestion_id: str,
        user_id: str,
        accept: bool
    ) -> bool:
        """
        Accept or reject suggestion

        Args:
            session_id: Session ID
            suggestion_id: Suggestion to review
            user_id: User reviewing
            accept: True to accept, False to reject

        Returns:
            True if reviewed successfully
        """
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        suggestion = next((s for s in session.suggestions if s.id == suggestion_id), None)
        if suggestion:
            suggestion.status = "accepted" if accept else "rejected"
            suggestion.reviewed_by = user_id
            suggestion.reviewed_at = datetime.now().isoformat()

            # If accepted, apply as change
            if accept:
                await self.apply_change(
                    session_id,
                    user_id,
                    ChangeType.REPLACE,
                    position=suggestion.position,
                    length=suggestion.length,
                    content=suggestion.suggested_text
                )

            return True

        return False

    async def acquire_lock(
        self,
        session_id: str,
        paragraph_id: str,
        user_id: str,
        duration_seconds: int = 300
    ) -> Optional[Lock]:
        """
        Acquire lock on paragraph

        Args:
            session_id: Session ID
            paragraph_id: Paragraph to lock
            user_id: User acquiring lock
            duration_seconds: Lock duration

        Returns:
            Lock if acquired, None if already locked
        """
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # Check existing lock
        if paragraph_id in session.locks:
            existing = session.locks[paragraph_id]
            # Check if expired
            expires = datetime.fromisoformat(existing.expires_at)
            if datetime.now() < expires and existing.user_id != user_id:
                return None  # Still locked by another user

        # Create lock
        lock = Lock(
            paragraph_id=paragraph_id,
            user_id=user_id,
            acquired_at=datetime.now().isoformat(),
            expires_at=(datetime.now().timestamp() + duration_seconds).__str__()
        )

        session.locks[paragraph_id] = lock
        return lock

    async def release_lock(
        self,
        session_id: str,
        paragraph_id: str,
        user_id: str
    ) -> bool:
        """
        Release lock on paragraph

        Args:
            session_id: Session ID
            paragraph_id: Paragraph to unlock
            user_id: User releasing lock

        Returns:
            True if released
        """
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        if paragraph_id in session.locks:
            lock = session.locks[paragraph_id]
            if lock.user_id == user_id:
                del session.locks[paragraph_id]
                return True

        return False

    async def create_version(
        self,
        session_id: str,
        user_id: str,
        content: str,
        message: Optional[str] = None
    ) -> Optional[Version]:
        """
        Create version snapshot

        Args:
            session_id: Session ID
            user_id: User creating version
            content: Current document content
            message: Version message

        Returns:
            Created version or None if failed
        """
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # Get changes since last version
        last_version_time = session.versions[-1].timestamp if session.versions else session.created_at
        changes_since = [
            c.id for c in session.changes
            if c.timestamp > last_version_time
        ]

        version = Version(
            id=self._generate_id(f"v{len(session.versions)}_{session_id}"),
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            content=content,
            changes_since_previous=changes_since,
            message=message
        )

        session.versions.append(version)
        return version

    def get_active_users(self, session_id: str) -> List[User]:
        """Get currently active users in session"""
        if session_id not in self.sessions:
            return []

        session = self.sessions[session_id]
        return [u for u in session.active_users if u.is_active]

    def get_change_history(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Change]:
        """Get recent change history"""
        if session_id not in self.sessions:
            return []

        session = self.sessions[session_id]
        return sorted(session.changes, key=lambda c: c.timestamp, reverse=True)[:limit]

    def get_unresolved_comments(self, session_id: str) -> List[Comment]:
        """Get unresolved comments"""
        if session_id not in self.sessions:
            return []

        session = self.sessions[session_id]
        return [c for c in session.comments if not c.resolved and not c.parent_id]

    def get_pending_suggestions(self, session_id: str) -> List[Suggestion]:
        """Get pending suggestions"""
        if session_id not in self.sessions:
            return []

        session = self.sessions[session_id]
        return [s for s in session.suggestions if s.status == "pending"]

    # ========================================================================
    # Helper methods
    # ========================================================================

    async def _detect_conflict(
        self,
        session: CollaborationSession,
        change: Change
    ) -> Optional[Change]:
        """
        Detect if change conflicts with recent changes

        Two changes conflict if they affect overlapping text regions.
        """
        # Check changes in last 5 minutes
        recent_threshold = (datetime.now().timestamp() - 300).__str__()

        for other in session.changes:
            if other.timestamp < recent_threshold:
                continue

            if other.user_id == change.user_id:
                continue  # Same user, no conflict

            # Check position overlap
            other_end = other.position + other.length
            change_end = change.position + change.length

            if (change.position < other_end and change_end > other.position):
                return other

        return None

    async def _operational_transform(
        self,
        change: Change,
        conflict: Change
    ) -> Change:
        """
        Apply operational transform to resolve conflict

        Adjusts change position based on conflicting change.
        """
        if conflict.change_type == ChangeType.INSERT:
            # Shift position forward
            if change.position >= conflict.position:
                change.position += len(conflict.content)

        elif conflict.change_type == ChangeType.DELETE:
            # Shift position backward
            if change.position > conflict.position:
                change.position -= conflict.length

        elif conflict.change_type == ChangeType.REPLACE:
            # Complex case - adjust for size difference
            if change.position > conflict.position:
                size_diff = len(conflict.content) - conflict.length
                change.position += size_diff

        return change

    def _generate_id(self, text: str) -> str:
        """Generate unique ID from text"""
        return hashlib.md5(text.encode()).hexdigest()[:12]


# Factory function
def create_collaboration_engine(
    merge_strategy: MergeStrategy = MergeStrategy.OPERATIONAL_TRANSFORM
) -> CollaborationEngine:
    """Create collaboration engine"""
    return CollaborationEngine(merge_strategy)
