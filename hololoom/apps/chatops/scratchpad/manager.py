"""
ChatOps Scratch Pad - Manager
=============================
High-level manager for scratch pad operations.

Provides:
- CRUD operations with rate limiting
- Quota enforcement
- Integration with Spacetime artifacts
- Session context management
- Export to Matrix

Created: December 2025
"""

import asyncio
import builtins
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .serialization import ContentSerializer
from .storage import ScratchPadStorage
from .types import (
    ArtifactReference,
    ArtifactScope,
    ArtifactType,
    DeleteResult,
    ListResult,
    RetrieveResult,
    ScratchArtifact,
    ScratchPadConfig,
    SessionArtifactContext,
    StoreResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Rate Limiter
# ============================================================================

@dataclass
class RateLimitState:
    """Rate limit state for a user."""
    requests: list[float] = field(default_factory=list)
    last_cleanup: float = field(default_factory=time.time)


class RateLimiter:
    """
    Sliding window rate limiter.

    Limits requests per user within a time window.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._state: dict[str, RateLimitState] = defaultdict(RateLimitState)

    def check(self, user_id: str) -> tuple[bool, int]:
        """
        Check if request is allowed.

        Args:
            user_id: User identifier

        Returns:
            Tuple of (allowed, remaining_requests)
        """
        now = time.time()
        state = self._state[user_id]

        # Clean up old requests
        cutoff = now - self.window_seconds
        state.requests = [t for t in state.requests if t > cutoff]

        if len(state.requests) >= self.max_requests:
            return False, 0

        return True, self.max_requests - len(state.requests)

    def record(self, user_id: str) -> None:
        """Record a request."""
        self._state[user_id].requests.append(time.time())

    def get_retry_after(self, user_id: str) -> int:
        """Get seconds until next request is allowed."""
        if user_id not in self._state:
            return 0

        state = self._state[user_id]
        if not state.requests:
            return 0

        oldest = min(state.requests)
        return max(0, int(self.window_seconds - (time.time() - oldest)))


# ============================================================================
# Name Validator
# ============================================================================

class NameValidator:
    """Validate artifact names."""

    def __init__(self, max_length: int = 255, pattern: str = r'^[a-zA-Z0-9_\-\.]+$'):
        self.max_length = max_length
        self.pattern = re.compile(pattern)

    def validate(self, name: str) -> tuple[bool, str | None]:
        """
        Validate an artifact name.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not name:
            return False, "Name cannot be empty"

        if len(name) > self.max_length:
            return False, f"Name too long (max {self.max_length} characters)"

        if not self.pattern.match(name):
            return False, "Name can only contain letters, numbers, underscores, hyphens, and dots"

        # Reserved names
        reserved = {'..', '.', 'null', 'undefined', 'none', 'nil'}
        if name.lower() in reserved:
            return False, f"'{name}' is a reserved name"

        return True, None


# ============================================================================
# Scratch Pad Manager
# ============================================================================

class ScratchPadManager:
    """
    High-level manager for scratch pad operations.

    Provides CRUD operations with rate limiting, quota enforcement,
    and Spacetime integration.
    """

    def __init__(self, config: ScratchPadConfig | None = None):
        """
        Initialize the scratch pad manager.

        Args:
            config: Configuration (uses defaults if not provided)
        """
        self.config = config or ScratchPadConfig()

        # Validate config
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {', '.join(errors)}")

        # Initialize storage
        self.storage = ScratchPadStorage(self.config)

        # Rate limiter
        self.rate_limiter = RateLimiter(
            self.config.rate_limit_requests,
            self.config.rate_limit_window_seconds
        )

        # Name validator
        self.name_validator = NameValidator(
            self.config.max_name_length,
            self.config.allowed_name_pattern
        )

        # Serializer
        self.serializer = ContentSerializer()

        # Session contexts
        self._session_contexts: dict[str, SessionArtifactContext] = {}

        # Background cleanup task
        self._cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start background tasks."""
        if self.config.cleanup_interval_hours > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop background tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self) -> None:
        """Background cleanup of expired artifacts."""
        interval = self.config.cleanup_interval_hours * 3600
        while True:
            try:
                await asyncio.sleep(interval)
                deleted = self.storage.cleanup_expired()
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} expired artifacts")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    # =========================================================================
    # Core CRUD Operations
    # =========================================================================

    async def store(
        self,
        name: str,
        content: str | bytes,
        user_id: str,
        room_id: str,
        artifact_type: ArtifactType | None = None,
        scope: ArtifactScope = ArtifactScope.USER,
        source_spacetime_id: str | None = None,
        source_session_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> StoreResult:
        """
        Store an artifact.

        Args:
            name: Artifact name
            content: Content (string or bytes)
            user_id: Owner user ID
            room_id: Owner room ID
            artifact_type: Type (auto-detected if not provided)
            scope: Access scope
            source_spacetime_id: Optional link to Spacetime
            source_session_id: Optional session ID
            metadata: Additional metadata

        Returns:
            StoreResult with artifact or error
        """
        # Rate limit check
        allowed, remaining = self.rate_limiter.check(user_id)
        if not allowed:
            retry_after = self.rate_limiter.get_retry_after(user_id)
            return StoreResult(
                success=False,
                error=f"Rate limited. Try again in {retry_after} seconds."
            )

        # Validate name
        valid, error = self.name_validator.validate(name)
        if not valid:
            return StoreResult(success=False, error=error)

        # Serialize content
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        else:
            content_bytes = content

        # Auto-detect type
        if artifact_type is None:
            artifact_type = self._detect_artifact_type(content_bytes, name)

        # Store
        artifact, error = self.storage.store_artifact(
            name=name,
            content=content_bytes,
            artifact_type=artifact_type,
            user_id=user_id,
            room_id=room_id,
            scope=scope,
            source_spacetime_id=source_spacetime_id,
            source_session_id=source_session_id,
            metadata=metadata
        )

        if error:
            return StoreResult(success=False, error=error)

        # Record rate limit
        self.rate_limiter.record(user_id)

        # Update session context
        if source_session_id:
            self._add_to_session_context(
                source_session_id, user_id, room_id, artifact
            )

        return StoreResult(
            success=True,
            artifact=artifact,
            deduplicated=False  # TODO: Get from storage
        )

    async def get(
        self,
        identifier: str,
        user_id: str,
        room_id: str,
        is_admin: bool = False
    ) -> RetrieveResult:
        """
        Get an artifact by ID or name.

        Args:
            identifier: Artifact ID or name
            user_id: Requesting user ID
            room_id: Requesting room ID
            is_admin: Whether user has admin privileges

        Returns:
            RetrieveResult with artifact and content
        """
        # Rate limit check
        allowed, remaining = self.rate_limiter.check(user_id)
        if not allowed:
            retry_after = self.rate_limiter.get_retry_after(user_id)
            return RetrieveResult(
                success=False,
                error=f"Rate limited. Try again in {retry_after} seconds."
            )

        # Try by ID first
        if identifier.startswith('artifact-'):
            artifact, content, error = self.storage.get_artifact(
                identifier, user_id, room_id, is_admin
            )
        else:
            # Try by name
            artifact, content, error = self.storage.get_by_name(
                identifier, user_id, room_id
            )

        if error:
            return RetrieveResult(success=False, error=error)

        # Record rate limit
        self.rate_limiter.record(user_id)

        return RetrieveResult(
            success=True,
            artifact=artifact,
            content=content
        )

    async def list(
        self,
        user_id: str,
        room_id: str,
        scope_filter: ArtifactScope | None = None,
        is_admin: bool = False,
        limit: int = 100
    ) -> ListResult:
        """
        List accessible artifacts.

        Args:
            user_id: Requesting user ID
            room_id: Requesting room ID
            scope_filter: Optional scope filter
            is_admin: Whether user has admin privileges
            limit: Maximum results

        Returns:
            ListResult with artifact references
        """
        artifacts = self.storage.list_artifacts(
            user_id, room_id, is_admin, scope_filter, limit
        )

        total_size = sum(a.size_bytes for a in artifacts)

        return ListResult(
            success=True,
            artifacts=artifacts,
            total_count=len(artifacts),
            total_size_bytes=total_size
        )

    async def delete(
        self,
        identifier: str,
        user_id: str,
        room_id: str,
        is_admin: bool = False
    ) -> DeleteResult:
        """
        Delete an artifact.

        Args:
            identifier: Artifact ID or name
            user_id: Requesting user ID
            room_id: Requesting room ID
            is_admin: Whether user has admin privileges

        Returns:
            DeleteResult
        """
        # Rate limit check
        allowed, remaining = self.rate_limiter.check(user_id)
        if not allowed:
            retry_after = self.rate_limiter.get_retry_after(user_id)
            return DeleteResult(
                success=False,
                error=f"Rate limited. Try again in {retry_after} seconds."
            )

        # Resolve identifier to ID
        artifact_id = identifier
        if not identifier.startswith('artifact-'):
            artifact = self.storage.metadata.get_by_name(identifier, user_id, room_id)
            if artifact is None:
                return DeleteResult(success=False, error=f"Artifact '{identifier}' not found")
            artifact_id = artifact.artifact_id

        # Delete
        success, error = self.storage.delete_artifact(
            artifact_id, user_id, room_id, is_admin
        )

        if not success:
            return DeleteResult(success=False, error=error)

        # Record rate limit
        self.rate_limiter.record(user_id)

        return DeleteResult(
            success=True,
            artifact_id=artifact_id,
            archived=self.config.archive_before_delete
        )

    async def share(
        self,
        identifier: str,
        user_id: str,
        room_id: str
    ) -> tuple[bool, str | None]:
        """
        Share an artifact with the room.

        Args:
            identifier: Artifact ID or name
            user_id: Owner user ID
            room_id: Room ID

        Returns:
            Tuple of (success, error_message)
        """
        # Resolve identifier
        artifact_id = identifier
        if not identifier.startswith('artifact-'):
            artifact = self.storage.metadata.get_by_name(identifier, user_id, room_id)
            if artifact is None:
                return False, f"Artifact '{identifier}' not found"
            artifact_id = artifact.artifact_id

        return self.storage.share_artifact(artifact_id, user_id, room_id)

    # =========================================================================
    # Spacetime Integration
    # =========================================================================

    async def store_from_spacetime(
        self,
        spacetime: Any,
        user_id: str,
        room_id: str,
        session_id: str | None = None
    ) -> builtins.list[StoreResult]:
        """
        Store artifacts from a Spacetime result.

        Automatically stores artifacts from the Spacetime if configured.

        Args:
            spacetime: Spacetime object with artifacts
            user_id: User ID
            room_id: Room ID
            session_id: Optional session ID

        Returns:
            List of StoreResults for each artifact
        """
        results = []

        if not self.config.auto_store_artifacts:
            return results

        if not hasattr(spacetime, 'artifacts') or not spacetime.artifacts:
            return results

        spacetime_id = getattr(spacetime, 'id', None) or str(id(spacetime))

        for artifact in spacetime.artifacts:
            # Skip tiny artifacts
            content = artifact.content
            if content is None:
                continue

            if isinstance(content, str):
                content_bytes = content.encode('utf-8')
            elif isinstance(content, bytes):
                content_bytes = content
            else:
                continue

            if len(content_bytes) < self.config.auto_store_min_size_bytes:
                continue

            # Map Spacetime artifact type
            artifact_type = self._map_spacetime_type(artifact.type)

            result = await self.store(
                name=artifact.name,
                content=content_bytes,
                user_id=user_id,
                room_id=room_id,
                artifact_type=artifact_type,
                source_spacetime_id=spacetime_id,
                source_session_id=session_id,
                metadata=artifact.metadata
            )

            results.append(result)

        return results

    def _map_spacetime_type(self, spacetime_type: Any) -> ArtifactType:
        """Map Spacetime ArtifactType to scratch pad ArtifactType."""
        type_value = spacetime_type.value if hasattr(spacetime_type, 'value') else str(spacetime_type)

        mapping = {
            'code': ArtifactType.CODE,
            'image': ArtifactType.IMAGE,
            'document': ArtifactType.DOCUMENT,
            'archive': ArtifactType.ARCHIVE,
            'audio': ArtifactType.AUDIO,
            'video': ArtifactType.VIDEO,
            'data': ArtifactType.DATA,
        }

        return mapping.get(type_value.lower(), ArtifactType.DATA)

    # =========================================================================
    # Session Context
    # =========================================================================

    def get_session_context(
        self,
        session_id: str,
        user_id: str,
        room_id: str
    ) -> SessionArtifactContext:
        """
        Get or create session context.

        Args:
            session_id: Session ID
            user_id: User ID
            room_id: Room ID

        Returns:
            SessionArtifactContext with artifact references
        """
        key = f"{room_id}:{user_id}:{session_id}"

        if key not in self._session_contexts:
            # Create new context
            context = SessionArtifactContext(
                session_id=session_id,
                user_id=user_id,
                room_id=room_id
            )

            # Load existing artifacts for session
            refs = self.storage.list_session_artifacts(session_id, user_id, room_id)
            for ref in refs:
                context.artifact_references.append(ref)

            self._session_contexts[key] = context

        return self._session_contexts[key]

    def _add_to_session_context(
        self,
        session_id: str,
        user_id: str,
        room_id: str,
        artifact: ScratchArtifact
    ) -> None:
        """Add artifact to session context."""
        key = f"{room_id}:{user_id}:{session_id}"

        if key not in self._session_contexts:
            self._session_contexts[key] = SessionArtifactContext(
                session_id=session_id,
                user_id=user_id,
                room_id=room_id
            )

        ref = ArtifactReference.from_artifact(artifact)
        self._session_contexts[key].add_reference(ref)

    def clear_session_context(self, session_id: str, user_id: str, room_id: str) -> None:
        """Clear session context."""
        key = f"{room_id}:{user_id}:{session_id}"
        if key in self._session_contexts:
            del self._session_contexts[key]

    # =========================================================================
    # Export
    # =========================================================================

    async def export_to_file(
        self,
        identifier: str,
        user_id: str,
        room_id: str,
        output_path: Path
    ) -> tuple[bool, str | None]:
        """
        Export artifact to a file.

        Args:
            identifier: Artifact ID or name
            user_id: User ID
            room_id: Room ID
            output_path: Path to write file

        Returns:
            Tuple of (success, error_message)
        """
        result = await self.get(identifier, user_id, room_id)
        if not result.success:
            return False, result.error

        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                f.write(result.content)

            return True, None
        except OSError as e:
            return False, str(e)

    async def get_export_data(
        self,
        identifier: str,
        user_id: str,
        room_id: str
    ) -> tuple[bytes | None, str | None, str | None]:
        """
        Get artifact data ready for Matrix export.

        Returns:
            Tuple of (content, filename, mimetype) or (None, None, error)
        """
        result = await self.get(identifier, user_id, room_id)
        if not result.success:
            return None, None, result.error

        # Determine MIME type
        mimetype = self._get_mimetype(result.artifact.artifact_type)

        # Generate filename
        filename = result.artifact.name
        if '.' not in filename:
            ext = self._get_extension(result.artifact.artifact_type)
            filename = f"{filename}.{ext}"

        return result.content, filename, mimetype

    # =========================================================================
    # Utilities
    # =========================================================================

    def _detect_artifact_type(self, content: bytes, name: str) -> ArtifactType:
        """Detect artifact type from content and name."""
        # Check extension
        ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''

        ext_mapping = {
            'py': ArtifactType.CODE,
            'js': ArtifactType.CODE,
            'ts': ArtifactType.CODE,
            'java': ArtifactType.CODE,
            'cpp': ArtifactType.CODE,
            'c': ArtifactType.CODE,
            'h': ArtifactType.CODE,
            'rs': ArtifactType.CODE,
            'go': ArtifactType.CODE,
            'png': ArtifactType.IMAGE,
            'jpg': ArtifactType.IMAGE,
            'jpeg': ArtifactType.IMAGE,
            'gif': ArtifactType.IMAGE,
            'webp': ArtifactType.IMAGE,
            'svg': ArtifactType.IMAGE,
            'pdf': ArtifactType.DOCUMENT,
            'md': ArtifactType.DOCUMENT,
            'txt': ArtifactType.TEXT,
            'json': ArtifactType.DATA,
            'yaml': ArtifactType.DATA,
            'yml': ArtifactType.DATA,
            'csv': ArtifactType.DATA,
            'xml': ArtifactType.DATA,
            'zip': ArtifactType.ARCHIVE,
            'tar': ArtifactType.ARCHIVE,
            'gz': ArtifactType.ARCHIVE,
            'mp3': ArtifactType.AUDIO,
            'wav': ArtifactType.AUDIO,
            'mp4': ArtifactType.VIDEO,
            'webm': ArtifactType.VIDEO,
        }

        if ext in ext_mapping:
            return ext_mapping[ext]

        # Check magic bytes
        if content[:4] == b'\x89PNG':
            return ArtifactType.IMAGE
        if content[:3] == b'\xff\xd8\xff':
            return ArtifactType.IMAGE
        if content[:4] == b'%PDF':
            return ArtifactType.DOCUMENT
        if content[:4] == b'PK\x03\x04':
            return ArtifactType.ARCHIVE

        # Try to decode as text
        try:
            text = content.decode('utf-8')
            if text.strip().startswith(('{', '[')):
                return ArtifactType.DATA
            if any(kw in text for kw in ['def ', 'class ', 'function ', 'import ']):
                return ArtifactType.CODE
            return ArtifactType.TEXT
        except UnicodeDecodeError:
            pass

        return ArtifactType.DATA

    def _get_mimetype(self, artifact_type: ArtifactType) -> str:
        """Get MIME type for artifact type."""
        mapping = {
            ArtifactType.CODE: 'text/plain',
            ArtifactType.IMAGE: 'image/png',
            ArtifactType.DOCUMENT: 'application/octet-stream',
            ArtifactType.ARCHIVE: 'application/zip',
            ArtifactType.AUDIO: 'audio/mpeg',
            ArtifactType.VIDEO: 'video/mp4',
            ArtifactType.DATA: 'application/json',
            ArtifactType.TEXT: 'text/plain',
            ArtifactType.EMBEDDING: 'application/octet-stream',
            ArtifactType.CONTEXT: 'application/json',
        }
        return mapping.get(artifact_type, 'application/octet-stream')

    def _get_extension(self, artifact_type: ArtifactType) -> str:
        """Get file extension for artifact type."""
        mapping = {
            ArtifactType.CODE: 'txt',
            ArtifactType.IMAGE: 'png',
            ArtifactType.DOCUMENT: 'bin',
            ArtifactType.ARCHIVE: 'zip',
            ArtifactType.AUDIO: 'mp3',
            ArtifactType.VIDEO: 'mp4',
            ArtifactType.DATA: 'json',
            ArtifactType.TEXT: 'txt',
            ArtifactType.EMBEDDING: 'bin',
            ArtifactType.CONTEXT: 'json',
        }
        return mapping.get(artifact_type, 'bin')

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics."""
        return self.storage.get_stats()


# ============================================================================
# Factory Functions
# ============================================================================

def create_scratchpad_manager(
    config: ScratchPadConfig | None = None
) -> ScratchPadManager:
    """
    Create a ScratchPadManager instance.

    Args:
        config: Optional configuration

    Returns:
        ScratchPadManager instance
    """
    return ScratchPadManager(config)


async def create_and_start_manager(
    config: ScratchPadConfig | None = None
) -> ScratchPadManager:
    """
    Create and start a ScratchPadManager.

    Args:
        config: Optional configuration

    Returns:
        Started ScratchPadManager instance
    """
    manager = ScratchPadManager(config)
    await manager.start()
    return manager
