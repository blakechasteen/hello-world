"""
ChatOps Scratch Pad - Core Types
================================
Dataclass definitions for the persistent artifact storage system.

This module provides:
- ScratchArtifact: Main artifact storage model
- ArtifactScope: Access control levels (USER/ROOM/GLOBAL)
- ScratchPadConfig: Configuration for the scratch pad system

Implements production-grade security with scope-based access control,
audit trail integration, and content-addressed storage.

Created: December 2025
"""

import secrets
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional


# ============================================================================
# Enums
# ============================================================================

class ArtifactScope(str, Enum):
    """
    Access control scope for scratch pad artifacts.

    Determines who can read/modify an artifact:
    - USER: Only the owner in the same room
    - ROOM: All users in the same room
    - GLOBAL: System-wide access (admin only)
    """
    USER = "user"       # Private to owner within room
    ROOM = "room"       # Shared with all room members
    GLOBAL = "global"   # System-wide (admin only)


class ArtifactType(str, Enum):
    """
    Types of artifacts that can be stored.

    Aligned with Spacetime ArtifactType for seamless integration.
    """
    CODE = "code"           # Source code files
    IMAGE = "image"         # Generated images
    DOCUMENT = "document"   # PDF, Markdown, Reports
    ARCHIVE = "archive"     # ZIP, TAR bundles
    AUDIO = "audio"         # Generated speech/audio
    VIDEO = "video"         # Generated video
    DATA = "data"           # CSV, JSON, Binary data
    TEXT = "text"           # Plain text content
    EMBEDDING = "embedding" # Vector embeddings
    CONTEXT = "context"     # Conversation context


class ArtifactStatus(str, Enum):
    """Lifecycle status of an artifact."""
    ACTIVE = "active"       # Normal state
    ARCHIVED = "archived"   # Moved to archive
    EXPIRED = "expired"     # Past TTL, pending cleanup
    DELETED = "deleted"     # Soft deleted


# ============================================================================
# Core Dataclasses
# ============================================================================

@dataclass
class ScratchArtifact:
    """
    A persistent artifact in the scratch pad.

    Artifacts are stored with content-addressed storage for deduplication
    and include full provenance tracking for audit trails.

    Attributes:
        artifact_id: Unique identifier (artifact-{random_hex})
        name: User-friendly name for the artifact
        content_hash: SHA256 hash (CAS key for content storage)
        artifact_type: Type of content stored
        size_bytes: Content size in bytes
        owner_user_id: Matrix user ID who created the artifact
        owner_room_id: Matrix room ID where artifact was created
        scope: Access control level
        created_at: Creation timestamp
        accessed_at: Last access timestamp
        expires_at: Optional expiration (default: 7 days from creation)
        access_count: Number of times accessed
        source_spacetime_id: Link to originating Spacetime (if auto-stored)
        source_session_id: Link to conversation session
        metadata: Additional metadata dictionary
    """
    # Identity
    artifact_id: str
    name: str

    # Content reference (actual content in CAS)
    content_hash: str
    artifact_type: ArtifactType
    size_bytes: int

    # Ownership
    owner_user_id: str
    owner_room_id: str
    scope: ArtifactScope = ArtifactScope.USER

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    status: ArtifactStatus = ArtifactStatus.ACTIVE

    # Provenance
    source_spacetime_id: Optional[str] = None
    source_session_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set default expiration if not provided."""
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(days=7)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique artifact ID."""
        return f"artifact-{secrets.token_hex(12)}"

    @staticmethod
    def compute_content_hash(content: bytes) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content).hexdigest()

    def is_expired(self) -> bool:
        """Check if artifact has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def is_accessible_by(self, user_id: str, room_id: str, is_admin: bool = False) -> bool:
        """
        Check if a user can access this artifact.

        Args:
            user_id: Matrix user ID attempting access
            room_id: Matrix room ID where access is attempted
            is_admin: Whether the user has admin privileges

        Returns:
            True if access is allowed
        """
        if self.status != ArtifactStatus.ACTIVE:
            return False

        if self.is_expired():
            return False

        if self.scope == ArtifactScope.GLOBAL:
            return is_admin

        if self.scope == ArtifactScope.ROOM:
            return room_id == self.owner_room_id

        # USER scope: must match both user and room
        return user_id == self.owner_user_id and room_id == self.owner_room_id

    def can_modify(self, user_id: str, room_id: str, is_admin: bool = False) -> bool:
        """
        Check if a user can modify/delete this artifact.

        Args:
            user_id: Matrix user ID attempting modification
            room_id: Matrix room ID
            is_admin: Whether the user has admin privileges

        Returns:
            True if modification is allowed
        """
        if is_admin:
            return True

        # Only owner can modify
        return user_id == self.owner_user_id and room_id == self.owner_room_id

    def record_access(self) -> None:
        """Record an access event."""
        self.accessed_at = datetime.now()
        self.access_count += 1

    def promote_to_room(self) -> None:
        """Promote from USER scope to ROOM scope (sharing)."""
        if self.scope == ArtifactScope.USER:
            self.scope = ArtifactScope.ROOM

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            'artifact_id': self.artifact_id,
            'name': self.name,
            'content_hash': self.content_hash,
            'artifact_type': self.artifact_type.value,
            'size_bytes': self.size_bytes,
            'owner_user_id': self.owner_user_id,
            'owner_room_id': self.owner_room_id,
            'scope': self.scope.value,
            'created_at': self.created_at.isoformat(),
            'accessed_at': self.accessed_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'access_count': self.access_count,
            'status': self.status.value,
            'source_spacetime_id': self.source_spacetime_id,
            'source_session_id': self.source_session_id,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScratchArtifact':
        """Create from dictionary."""
        return cls(
            artifact_id=data['artifact_id'],
            name=data['name'],
            content_hash=data['content_hash'],
            artifact_type=ArtifactType(data['artifact_type']),
            size_bytes=data['size_bytes'],
            owner_user_id=data['owner_user_id'],
            owner_room_id=data['owner_room_id'],
            scope=ArtifactScope(data.get('scope', 'user')),
            created_at=datetime.fromisoformat(data['created_at']),
            accessed_at=datetime.fromisoformat(data['accessed_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            access_count=data.get('access_count', 0),
            status=ArtifactStatus(data.get('status', 'active')),
            source_spacetime_id=data.get('source_spacetime_id'),
            source_session_id=data.get('source_session_id'),
            metadata=data.get('metadata', {})
        )


@dataclass
class ArtifactReference:
    """
    A lightweight reference to an artifact.

    Used for passing artifact context between commands without
    loading full content. Useful for `!continue` context passing.
    """
    artifact_id: str
    name: str
    artifact_type: ArtifactType
    content_hash: str
    size_bytes: int
    scope: ArtifactScope
    created_at: datetime

    @classmethod
    def from_artifact(cls, artifact: ScratchArtifact) -> 'ArtifactReference':
        """Create a reference from a full artifact."""
        return cls(
            artifact_id=artifact.artifact_id,
            name=artifact.name,
            artifact_type=artifact.artifact_type,
            content_hash=artifact.content_hash,
            size_bytes=artifact.size_bytes,
            scope=artifact.scope,
            created_at=artifact.created_at
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            'artifact_id': self.artifact_id,
            'name': self.name,
            'artifact_type': self.artifact_type.value,
            'content_hash': self.content_hash,
            'size_bytes': self.size_bytes,
            'scope': self.scope.value,
            'created_at': self.created_at.isoformat()
        }


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ScratchPadConfig:
    """
    Configuration for the scratch pad system.

    Controls storage paths, limits, lifecycle, and security settings.
    """
    # Storage paths
    metadata_db_path: str = "./scratchpad/metadata.db"
    cas_storage_path: str = "./scratchpad/content"

    # Size limits
    max_artifact_size_bytes: int = 50 * 1024 * 1024  # 50MB
    max_artifacts_per_user: int = 100
    max_total_storage_bytes: int = 10 * 1024 * 1024 * 1024  # 10GB

    # Lifecycle
    default_ttl_days: int = 7
    cleanup_interval_hours: int = 24
    archive_before_delete: bool = True

    # Security
    enable_audit: bool = True
    audit_log_path: str = "./scratchpad/audit.jsonl"

    # Rate limiting
    rate_limit_requests: int = 10
    rate_limit_window_seconds: int = 60

    # Name validation
    max_name_length: int = 255
    allowed_name_pattern: str = r'^[a-zA-Z0-9_\-\.]+$'

    # Auto-storage from Spacetime
    auto_store_artifacts: bool = True
    auto_store_min_size_bytes: int = 100  # Don't auto-store tiny artifacts

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if self.max_artifact_size_bytes <= 0:
            errors.append("max_artifact_size_bytes must be positive")

        if self.max_artifacts_per_user <= 0:
            errors.append("max_artifacts_per_user must be positive")

        if self.default_ttl_days <= 0:
            errors.append("default_ttl_days must be positive")

        if self.rate_limit_requests <= 0:
            errors.append("rate_limit_requests must be positive")

        if self.rate_limit_window_seconds <= 0:
            errors.append("rate_limit_window_seconds must be positive")

        return errors

    @classmethod
    def production(cls) -> 'ScratchPadConfig':
        """Create production configuration with stricter limits."""
        return cls(
            metadata_db_path="/var/lib/hololoom/scratchpad/metadata.db",
            cas_storage_path="/var/lib/hololoom/scratchpad/content",
            audit_log_path="/var/log/hololoom/scratchpad_audit.jsonl",
            max_artifacts_per_user=50,
            default_ttl_days=30,
            enable_audit=True
        )

    @classmethod
    def development(cls) -> 'ScratchPadConfig':
        """Create development configuration with relaxed limits."""
        return cls(
            metadata_db_path="./dev_scratchpad/metadata.db",
            cas_storage_path="./dev_scratchpad/content",
            audit_log_path="./dev_scratchpad/audit.jsonl",
            max_artifacts_per_user=1000,
            default_ttl_days=7,
            enable_audit=False,
            rate_limit_requests=100
        )


# ============================================================================
# Audit Types
# ============================================================================

class AuditEventType(str, Enum):
    """Types of audit events for scratch pad operations."""
    ARTIFACT_CREATED = "artifact_created"
    ARTIFACT_ACCESSED = "artifact_accessed"
    ARTIFACT_MODIFIED = "artifact_modified"
    ARTIFACT_DELETED = "artifact_deleted"
    ARTIFACT_SHARED = "artifact_shared"
    ARTIFACT_EXPORTED = "artifact_exported"
    QUOTA_EXCEEDED = "quota_exceeded"
    RATE_LIMITED = "rate_limited"
    ACCESS_DENIED = "access_denied"


@dataclass
class AuditEvent:
    """
    An audit event for scratch pad operations.

    All operations are logged for security and compliance.
    """
    event_type: AuditEventType
    timestamp: datetime
    user_id: str
    room_id: str
    artifact_id: Optional[str] = None
    artifact_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary for JSON Lines format."""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'room_id': self.room_id,
            'artifact_id': self.artifact_id,
            'artifact_name': self.artifact_name,
            'details': self.details,
            'success': self.success,
            'error_message': self.error_message
        }


# ============================================================================
# Result Types
# ============================================================================

@dataclass
class StoreResult:
    """Result of a store operation."""
    success: bool
    artifact: Optional[ScratchArtifact] = None
    error: Optional[str] = None
    deduplicated: bool = False  # True if content already existed (CAS hit)


@dataclass
class RetrieveResult:
    """Result of a retrieve operation."""
    success: bool
    artifact: Optional[ScratchArtifact] = None
    content: Optional[bytes] = None
    error: Optional[str] = None


@dataclass
class ListResult:
    """Result of a list operation."""
    success: bool
    artifacts: List[ArtifactReference] = field(default_factory=list)
    total_count: int = 0
    total_size_bytes: int = 0
    error: Optional[str] = None


@dataclass
class DeleteResult:
    """Result of a delete operation."""
    success: bool
    artifact_id: Optional[str] = None
    archived: bool = False
    error: Optional[str] = None


# ============================================================================
# Session Context Types
# ============================================================================

@dataclass
class SessionArtifactContext:
    """
    Artifact context for a conversation session.

    Passed to `!continue` commands to provide artifact references
    from previous interactions in the session.
    """
    session_id: str
    user_id: str
    room_id: str
    artifact_references: List[ArtifactReference] = field(default_factory=list)
    last_spacetime_id: Optional[str] = None
    last_query: Optional[str] = None

    def add_reference(self, ref: ArtifactReference) -> None:
        """Add an artifact reference to the context."""
        # Avoid duplicates
        existing_ids = {r.artifact_id for r in self.artifact_references}
        if ref.artifact_id not in existing_ids:
            self.artifact_references.append(ref)

    def get_references_summary(self) -> str:
        """Get a human-readable summary of available artifacts."""
        if not self.artifact_references:
            return "No artifacts in session context."

        lines = ["Available artifacts:"]
        for ref in self.artifact_references:
            size_kb = ref.size_bytes / 1024
            lines.append(f"  - {ref.name} ({ref.artifact_type.value}, {size_kb:.1f}KB)")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'room_id': self.room_id,
            'artifact_references': [r.to_dict() for r in self.artifact_references],
            'last_spacetime_id': self.last_spacetime_id,
            'last_query': self.last_query
        }
