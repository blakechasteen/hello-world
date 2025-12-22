"""
ChatOps Scratch Pad - Storage Layer
====================================
Dual storage system with SQLite for metadata and CAS for content.

Components:
- MetadataStorage: SQLite database for artifact metadata (searchable)
- ContentStorage: Content-Addressed Storage for deduplication
- ScratchPadStorage: Combined interface

This separation enables:
- Fast metadata queries without loading content
- Content deduplication across artifacts
- Efficient quota management
- Reference counting for cleanup

Created: December 2025
"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .types import (
    ScratchArtifact,
    ArtifactReference,
    ArtifactScope,
    ArtifactType,
    ArtifactStatus,
    ScratchPadConfig,
    AuditEvent,
    AuditEventType,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Content-Addressed Storage
# ============================================================================

@dataclass
class CASEntry:
    """Entry in content-addressed storage."""
    content_hash: str
    size_bytes: int
    path: Path
    reference_count: int = 1
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class ContentStorage:
    """
    Content-Addressed Storage for artifact content.

    Stores content by SHA256 hash, deduplicating identical content.
    Uses directory sharding for filesystem efficiency.

    Directory structure:
        storage_dir/
            ab/cd1234...  # First 2 chars as prefix directory
    """

    def __init__(self, storage_dir: Path, hash_prefix_length: int = 2):
        self.storage_dir = Path(storage_dir)
        self.hash_prefix_length = hash_prefix_length

        self._entries: Dict[str, CASEntry] = {}
        self._lock = threading.RLock()

        # Ensure directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Scan existing storage
        self._scan_existing()

    def _hash_content(self, content: bytes) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content).hexdigest()

    def _object_path(self, content_hash: str) -> Path:
        """Get filesystem path for object by hash."""
        prefix = content_hash[:self.hash_prefix_length]
        suffix = content_hash[self.hash_prefix_length:]
        return self.storage_dir / prefix / suffix

    def _scan_existing(self) -> None:
        """Scan existing storage and populate internal state."""
        try:
            for prefix_dir in self.storage_dir.iterdir():
                if prefix_dir.is_dir() and len(prefix_dir.name) == self.hash_prefix_length:
                    for obj_file in prefix_dir.iterdir():
                        if obj_file.is_file():
                            content_hash = prefix_dir.name + obj_file.name
                            try:
                                size = obj_file.stat().st_size
                                self._entries[content_hash] = CASEntry(
                                    content_hash=content_hash,
                                    size_bytes=size,
                                    path=obj_file,
                                    reference_count=0,  # Updated from metadata
                                )
                            except OSError:
                                pass
        except OSError as e:
            logger.warning(f"Error scanning CAS storage: {e}")

    def store(self, content: bytes) -> Tuple[str, bool]:
        """
        Store content.

        Args:
            content: Binary content to store

        Returns:
            Tuple of (content_hash, was_deduplicated)
        """
        content_hash = self._hash_content(content)
        obj_path = self._object_path(content_hash)

        with self._lock:
            if content_hash in self._entries:
                # Content already exists - increment reference
                self._entries[content_hash].reference_count += 1
                self._entries[content_hash].last_accessed = time.time()
                return content_hash, True

            # Store new content
            obj_path.parent.mkdir(parents=True, exist_ok=True)

            with open(obj_path, "wb") as f:
                f.write(content)

            self._entries[content_hash] = CASEntry(
                content_hash=content_hash,
                size_bytes=len(content),
                path=obj_path,
                reference_count=1,
            )

            return content_hash, False

    def get(self, content_hash: str) -> Optional[bytes]:
        """Get content by hash."""
        with self._lock:
            if content_hash not in self._entries:
                return None

            entry = self._entries[content_hash]
            entry.last_accessed = time.time()

        try:
            with open(entry.path, "rb") as f:
                return f.read()
        except (OSError, IOError) as e:
            logger.error(f"Error reading content {content_hash}: {e}")
            return None

    def exists(self, content_hash: str) -> bool:
        """Check if content exists."""
        with self._lock:
            return content_hash in self._entries

    def add_reference(self, content_hash: str) -> bool:
        """Increment reference count."""
        with self._lock:
            if content_hash in self._entries:
                self._entries[content_hash].reference_count += 1
                return True
            return False

    def remove_reference(self, content_hash: str) -> int:
        """
        Decrement reference count.

        Returns:
            New reference count (-1 if not found)
        """
        with self._lock:
            if content_hash not in self._entries:
                return -1

            self._entries[content_hash].reference_count -= 1
            return self._entries[content_hash].reference_count

    def garbage_collect(self) -> int:
        """Remove unreferenced objects. Returns count deleted."""
        deleted = 0

        with self._lock:
            to_delete = [
                h for h, entry in self._entries.items()
                if entry.reference_count <= 0
            ]

            for content_hash in to_delete:
                entry = self._entries[content_hash]
                try:
                    entry.path.unlink()
                    try:
                        entry.path.parent.rmdir()
                    except OSError:
                        pass
                    del self._entries[content_hash]
                    deleted += 1
                except OSError:
                    pass

        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            total_size = sum(e.size_bytes for e in self._entries.values())
            total_refs = sum(e.reference_count for e in self._entries.values())
            unique = len(self._entries)

            return {
                "unique_objects": unique,
                "total_references": total_refs,
                "total_size_bytes": total_size,
                "deduplication_ratio": total_refs / unique if unique > 0 else 1.0,
            }

    def verify_integrity(self, content_hash: str) -> bool:
        """Verify content matches its hash."""
        content = self.get(content_hash)
        if content is None:
            return False
        return self._hash_content(content) == content_hash


# ============================================================================
# Metadata Storage (SQLite)
# ============================================================================

class MetadataStorage:
    """
    SQLite-based metadata storage for artifacts.

    Provides fast querying of artifact metadata without loading content.
    Supports scope-based access control and efficient listing.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        artifact_type TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        owner_user_id TEXT NOT NULL,
        owner_room_id TEXT NOT NULL,
        scope TEXT NOT NULL DEFAULT 'user',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        accessed_at TEXT NOT NULL,
        expires_at TEXT,
        access_count INTEGER DEFAULT 0,
        source_spacetime_id TEXT,
        source_session_id TEXT,
        metadata TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_artifacts_owner
        ON artifacts(owner_user_id, owner_room_id);

    CREATE INDEX IF NOT EXISTS idx_artifacts_scope_room
        ON artifacts(scope, owner_room_id);

    CREATE INDEX IF NOT EXISTS idx_artifacts_content_hash
        ON artifacts(content_hash);

    CREATE INDEX IF NOT EXISTS idx_artifacts_name
        ON artifacts(name, owner_user_id, owner_room_id);

    CREATE INDEX IF NOT EXISTS idx_artifacts_expires
        ON artifacts(expires_at) WHERE status = 'active';

    CREATE INDEX IF NOT EXISTS idx_artifacts_session
        ON artifacts(source_session_id);
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript(self.SCHEMA)
        conn.commit()

    def store(self, artifact: ScratchArtifact) -> bool:
        """Store artifact metadata."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO artifacts (
                    artifact_id, name, content_hash, artifact_type, size_bytes,
                    owner_user_id, owner_room_id, scope, status,
                    created_at, accessed_at, expires_at, access_count,
                    source_spacetime_id, source_session_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                artifact.artifact_id,
                artifact.name,
                artifact.content_hash,
                artifact.artifact_type.value,
                artifact.size_bytes,
                artifact.owner_user_id,
                artifact.owner_room_id,
                artifact.scope.value,
                artifact.status.value,
                artifact.created_at.isoformat(),
                artifact.accessed_at.isoformat(),
                artifact.expires_at.isoformat() if artifact.expires_at else None,
                artifact.access_count,
                artifact.source_spacetime_id,
                artifact.source_session_id,
                json.dumps(artifact.metadata),
            ))
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error storing artifact metadata: {e}")
            return False

    def get(self, artifact_id: str) -> Optional[ScratchArtifact]:
        """Get artifact by ID."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (artifact_id,)
        ).fetchone()

        if row is None:
            return None

        return self._row_to_artifact(row)

    def get_by_name(
        self,
        name: str,
        user_id: str,
        room_id: str
    ) -> Optional[ScratchArtifact]:
        """Get artifact by name within user/room scope."""
        conn = self._get_connection()
        row = conn.execute("""
            SELECT * FROM artifacts
            WHERE name = ? AND owner_user_id = ? AND owner_room_id = ?
            AND status = 'active'
        """, (name, user_id, room_id)).fetchone()

        if row is None:
            return None

        return self._row_to_artifact(row)

    def list_accessible(
        self,
        user_id: str,
        room_id: str,
        is_admin: bool = False,
        scope_filter: Optional[ArtifactScope] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ArtifactReference]:
        """List artifacts accessible to user."""
        conn = self._get_connection()

        if is_admin and scope_filter == ArtifactScope.GLOBAL:
            # Admin can see global artifacts
            query = """
                SELECT * FROM artifacts
                WHERE scope = 'global' AND status = 'active'
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(query, (limit, offset)).fetchall()
        elif scope_filter == ArtifactScope.ROOM:
            # Room scope - all room members can see
            query = """
                SELECT * FROM artifacts
                WHERE owner_room_id = ? AND scope = 'room' AND status = 'active'
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(query, (room_id, limit, offset)).fetchall()
        elif scope_filter == ArtifactScope.USER:
            # User scope - only owner
            query = """
                SELECT * FROM artifacts
                WHERE owner_user_id = ? AND owner_room_id = ?
                AND scope = 'user' AND status = 'active'
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(query, (user_id, room_id, limit, offset)).fetchall()
        else:
            # No filter - show all accessible
            query = """
                SELECT * FROM artifacts
                WHERE status = 'active' AND (
                    (owner_user_id = ? AND owner_room_id = ?)
                    OR (owner_room_id = ? AND scope = 'room')
                )
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(
                query, (user_id, room_id, room_id, limit, offset)
            ).fetchall()

        return [self._row_to_reference(row) for row in rows]

    def list_by_session(
        self,
        session_id: str,
        user_id: str,
        room_id: str
    ) -> List[ArtifactReference]:
        """List artifacts from a specific session."""
        conn = self._get_connection()
        query = """
            SELECT * FROM artifacts
            WHERE source_session_id = ?
            AND owner_user_id = ? AND owner_room_id = ?
            AND status = 'active'
            ORDER BY created_at DESC
        """
        rows = conn.execute(query, (session_id, user_id, room_id)).fetchall()
        return [self._row_to_reference(row) for row in rows]

    def update(self, artifact: ScratchArtifact) -> bool:
        """Update artifact metadata."""
        return self.store(artifact)

    def update_access(self, artifact_id: str) -> bool:
        """Update access timestamp and count."""
        conn = self._get_connection()
        try:
            conn.execute("""
                UPDATE artifacts
                SET accessed_at = ?, access_count = access_count + 1
                WHERE artifact_id = ?
            """, (datetime.now().isoformat(), artifact_id))
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error updating access: {e}")
            return False

    def update_scope(self, artifact_id: str, new_scope: ArtifactScope) -> bool:
        """Update artifact scope (for sharing)."""
        conn = self._get_connection()
        try:
            conn.execute("""
                UPDATE artifacts SET scope = ? WHERE artifact_id = ?
            """, (new_scope.value, artifact_id))
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error updating scope: {e}")
            return False

    def update_status(self, artifact_id: str, status: ArtifactStatus) -> bool:
        """Update artifact status."""
        conn = self._get_connection()
        try:
            conn.execute("""
                UPDATE artifacts SET status = ? WHERE artifact_id = ?
            """, (status.value, artifact_id))
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error updating status: {e}")
            return False

    def delete(self, artifact_id: str) -> Optional[str]:
        """
        Delete artifact metadata.

        Returns:
            Content hash if deleted, None if not found
        """
        conn = self._get_connection()
        try:
            # Get content hash before delete
            row = conn.execute(
                "SELECT content_hash FROM artifacts WHERE artifact_id = ?",
                (artifact_id,)
            ).fetchone()

            if row is None:
                return None

            conn.execute(
                "DELETE FROM artifacts WHERE artifact_id = ?",
                (artifact_id,)
            )
            conn.commit()
            return row['content_hash']
        except sqlite3.Error as e:
            logger.error(f"Error deleting artifact: {e}")
            return None

    def count_by_user(self, user_id: str, room_id: str) -> int:
        """Count artifacts for a user in a room."""
        conn = self._get_connection()
        row = conn.execute("""
            SELECT COUNT(*) as count FROM artifacts
            WHERE owner_user_id = ? AND owner_room_id = ? AND status = 'active'
        """, (user_id, room_id)).fetchone()
        return row['count'] if row else 0

    def total_size_by_user(self, user_id: str, room_id: str) -> int:
        """Total size of artifacts for a user in a room."""
        conn = self._get_connection()
        row = conn.execute("""
            SELECT COALESCE(SUM(size_bytes), 0) as total FROM artifacts
            WHERE owner_user_id = ? AND owner_room_id = ? AND status = 'active'
        """, (user_id, room_id)).fetchone()
        return row['total'] if row else 0

    def find_expired(self, before: datetime) -> List[str]:
        """Find artifact IDs that have expired."""
        conn = self._get_connection()
        rows = conn.execute("""
            SELECT artifact_id FROM artifacts
            WHERE expires_at IS NOT NULL AND expires_at < ?
            AND status = 'active'
        """, (before.isoformat(),)).fetchall()
        return [row['artifact_id'] for row in rows]

    def get_content_hash_references(self, content_hash: str) -> int:
        """Count how many artifacts reference a content hash."""
        conn = self._get_connection()
        row = conn.execute("""
            SELECT COUNT(*) as count FROM artifacts
            WHERE content_hash = ? AND status = 'active'
        """, (content_hash,)).fetchone()
        return row['count'] if row else 0

    def _row_to_artifact(self, row: sqlite3.Row) -> ScratchArtifact:
        """Convert database row to ScratchArtifact."""
        return ScratchArtifact(
            artifact_id=row['artifact_id'],
            name=row['name'],
            content_hash=row['content_hash'],
            artifact_type=ArtifactType(row['artifact_type']),
            size_bytes=row['size_bytes'],
            owner_user_id=row['owner_user_id'],
            owner_room_id=row['owner_room_id'],
            scope=ArtifactScope(row['scope']),
            status=ArtifactStatus(row['status']),
            created_at=datetime.fromisoformat(row['created_at']),
            accessed_at=datetime.fromisoformat(row['accessed_at']),
            expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
            access_count=row['access_count'],
            source_spacetime_id=row['source_spacetime_id'],
            source_session_id=row['source_session_id'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
        )

    def _row_to_reference(self, row: sqlite3.Row) -> ArtifactReference:
        """Convert database row to ArtifactReference."""
        return ArtifactReference(
            artifact_id=row['artifact_id'],
            name=row['name'],
            artifact_type=ArtifactType(row['artifact_type']),
            content_hash=row['content_hash'],
            size_bytes=row['size_bytes'],
            scope=ArtifactScope(row['scope']),
            created_at=datetime.fromisoformat(row['created_at']),
        )


# ============================================================================
# Audit Log Storage
# ============================================================================

class AuditLogStorage:
    """
    Append-only JSON Lines audit log.

    All operations are logged for security and compliance.
    """

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event: AuditEvent) -> None:
        """Append an audit event."""
        with self._lock:
            try:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(event.to_dict()) + "\n")
            except (OSError, IOError) as e:
                logger.error(f"Failed to write audit log: {e}")

    def query(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Query audit events."""
        events = []

        if not self.log_path.exists():
            return events

        with self._lock:
            try:
                with open(self.log_path, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            event = AuditEvent(
                                event_type=AuditEventType(data['event_type']),
                                timestamp=datetime.fromisoformat(data['timestamp']),
                                user_id=data['user_id'],
                                room_id=data['room_id'],
                                artifact_id=data.get('artifact_id'),
                                artifact_name=data.get('artifact_name'),
                                details=data.get('details', {}),
                                success=data.get('success', True),
                                error_message=data.get('error_message'),
                            )

                            # Apply filters
                            if event_type and event.event_type != event_type:
                                continue
                            if user_id and event.user_id != user_id:
                                continue
                            if artifact_id and event.artifact_id != artifact_id:
                                continue
                            if since and event.timestamp < since:
                                continue

                            events.append(event)

                            if len(events) >= limit:
                                break
                        except (json.JSONDecodeError, KeyError):
                            continue
            except (OSError, IOError) as e:
                logger.error(f"Error reading audit log: {e}")

        return events


# ============================================================================
# Combined Storage Interface
# ============================================================================

class ScratchPadStorage:
    """
    Combined storage interface for scratch pad.

    Coordinates metadata storage (SQLite) and content storage (CAS).
    """

    def __init__(self, config: ScratchPadConfig):
        self.config = config

        # Initialize storage components
        self.metadata = MetadataStorage(Path(config.metadata_db_path))
        self.content = ContentStorage(Path(config.cas_storage_path))

        if config.enable_audit:
            self.audit = AuditLogStorage(Path(config.audit_log_path))
        else:
            self.audit = None

    def store_artifact(
        self,
        name: str,
        content: bytes,
        artifact_type: ArtifactType,
        user_id: str,
        room_id: str,
        scope: ArtifactScope = ArtifactScope.USER,
        source_spacetime_id: Optional[str] = None,
        source_session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[ScratchArtifact], Optional[str]]:
        """
        Store a new artifact.

        Args:
            name: Artifact name
            content: Binary content
            artifact_type: Type of artifact
            user_id: Owner user ID
            room_id: Owner room ID
            scope: Access scope
            source_spacetime_id: Optional link to Spacetime
            source_session_id: Optional link to session
            metadata: Additional metadata

        Returns:
            Tuple of (artifact, error_message)
        """
        # Size check
        if len(content) > self.config.max_artifact_size_bytes:
            return None, f"Content exceeds maximum size ({self.config.max_artifact_size_bytes} bytes)"

        # Quota check
        user_count = self.metadata.count_by_user(user_id, room_id)
        if user_count >= self.config.max_artifacts_per_user:
            return None, f"Quota exceeded ({self.config.max_artifacts_per_user} artifacts)"

        # Store content
        content_hash, deduplicated = self.content.store(content)

        # Create artifact
        artifact = ScratchArtifact(
            artifact_id=ScratchArtifact.generate_id(),
            name=name,
            content_hash=content_hash,
            artifact_type=artifact_type,
            size_bytes=len(content),
            owner_user_id=user_id,
            owner_room_id=room_id,
            scope=scope,
            source_spacetime_id=source_spacetime_id,
            source_session_id=source_session_id,
            metadata=metadata or {},
        )

        # Store metadata
        if not self.metadata.store(artifact):
            # Rollback content reference
            self.content.remove_reference(content_hash)
            return None, "Failed to store artifact metadata"

        # Audit log
        if self.audit:
            self.audit.log(AuditEvent(
                event_type=AuditEventType.ARTIFACT_CREATED,
                timestamp=datetime.now(),
                user_id=user_id,
                room_id=room_id,
                artifact_id=artifact.artifact_id,
                artifact_name=name,
                details={
                    'size_bytes': len(content),
                    'deduplicated': deduplicated,
                    'scope': scope.value,
                },
                success=True,
            ))

        return artifact, None

    def get_artifact(
        self,
        artifact_id: str,
        user_id: str,
        room_id: str,
        is_admin: bool = False
    ) -> Tuple[Optional[ScratchArtifact], Optional[bytes], Optional[str]]:
        """
        Get artifact with content.

        Returns:
            Tuple of (artifact, content, error_message)
        """
        # Get metadata
        artifact = self.metadata.get(artifact_id)
        if artifact is None:
            return None, None, "Artifact not found"

        # Check access
        if not artifact.is_accessible_by(user_id, room_id, is_admin):
            if self.audit:
                self.audit.log(AuditEvent(
                    event_type=AuditEventType.ACCESS_DENIED,
                    timestamp=datetime.now(),
                    user_id=user_id,
                    room_id=room_id,
                    artifact_id=artifact_id,
                    success=False,
                    error_message="Access denied",
                ))
            return None, None, "Access denied"

        # Get content
        content = self.content.get(artifact.content_hash)
        if content is None:
            return None, None, "Content not found"

        # Update access
        artifact.record_access()
        self.metadata.update_access(artifact_id)

        # Audit log
        if self.audit:
            self.audit.log(AuditEvent(
                event_type=AuditEventType.ARTIFACT_ACCESSED,
                timestamp=datetime.now(),
                user_id=user_id,
                room_id=room_id,
                artifact_id=artifact_id,
                artifact_name=artifact.name,
                success=True,
            ))

        return artifact, content, None

    def get_by_name(
        self,
        name: str,
        user_id: str,
        room_id: str
    ) -> Tuple[Optional[ScratchArtifact], Optional[bytes], Optional[str]]:
        """Get artifact by name."""
        artifact = self.metadata.get_by_name(name, user_id, room_id)
        if artifact is None:
            return None, None, f"Artifact '{name}' not found"

        return self.get_artifact(artifact.artifact_id, user_id, room_id)

    def list_artifacts(
        self,
        user_id: str,
        room_id: str,
        is_admin: bool = False,
        scope_filter: Optional[ArtifactScope] = None,
        limit: int = 100
    ) -> List[ArtifactReference]:
        """List accessible artifacts."""
        return self.metadata.list_accessible(
            user_id, room_id, is_admin, scope_filter, limit
        )

    def list_session_artifacts(
        self,
        session_id: str,
        user_id: str,
        room_id: str
    ) -> List[ArtifactReference]:
        """List artifacts from a session."""
        return self.metadata.list_by_session(session_id, user_id, room_id)

    def delete_artifact(
        self,
        artifact_id: str,
        user_id: str,
        room_id: str,
        is_admin: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Delete an artifact."""
        # Get artifact
        artifact = self.metadata.get(artifact_id)
        if artifact is None:
            return False, "Artifact not found"

        # Check permission
        if not artifact.can_modify(user_id, room_id, is_admin):
            return False, "Permission denied"

        # Archive if configured
        if self.config.archive_before_delete:
            self.metadata.update_status(artifact_id, ArtifactStatus.ARCHIVED)

        # Delete metadata
        content_hash = self.metadata.delete(artifact_id)
        if content_hash is None:
            return False, "Failed to delete"

        # Decrement content reference
        ref_count = self.content.remove_reference(content_hash)

        # Audit log
        if self.audit:
            self.audit.log(AuditEvent(
                event_type=AuditEventType.ARTIFACT_DELETED,
                timestamp=datetime.now(),
                user_id=user_id,
                room_id=room_id,
                artifact_id=artifact_id,
                artifact_name=artifact.name,
                details={'remaining_references': ref_count},
                success=True,
            ))

        return True, None

    def share_artifact(
        self,
        artifact_id: str,
        user_id: str,
        room_id: str
    ) -> Tuple[bool, Optional[str]]:
        """Share artifact with room (USER → ROOM scope)."""
        # Get artifact
        artifact = self.metadata.get(artifact_id)
        if artifact is None:
            return False, "Artifact not found"

        # Check ownership
        if not artifact.can_modify(user_id, room_id):
            return False, "Only owner can share"

        if artifact.scope != ArtifactScope.USER:
            return False, "Already shared or global"

        # Update scope
        if not self.metadata.update_scope(artifact_id, ArtifactScope.ROOM):
            return False, "Failed to update scope"

        # Audit log
        if self.audit:
            self.audit.log(AuditEvent(
                event_type=AuditEventType.ARTIFACT_SHARED,
                timestamp=datetime.now(),
                user_id=user_id,
                room_id=room_id,
                artifact_id=artifact_id,
                artifact_name=artifact.name,
                success=True,
            ))

        return True, None

    def cleanup_expired(self) -> int:
        """Clean up expired artifacts. Returns count deleted."""
        expired_ids = self.metadata.find_expired(datetime.now())
        deleted = 0

        for artifact_id in expired_ids:
            artifact = self.metadata.get(artifact_id)
            if artifact:
                self.metadata.update_status(artifact_id, ArtifactStatus.EXPIRED)
                content_hash = self.metadata.delete(artifact_id)
                if content_hash:
                    self.content.remove_reference(content_hash)
                    deleted += 1

        # Garbage collect unreferenced content
        self.content.garbage_collect()

        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "content": self.content.get_stats(),
            "config": {
                "max_artifact_size_bytes": self.config.max_artifact_size_bytes,
                "max_artifacts_per_user": self.config.max_artifacts_per_user,
                "default_ttl_days": self.config.default_ttl_days,
            }
        }
