"""
ChatOps Scratch Pad
===================
Persistent artifact storage for ChatOps workflows.

This module provides:
- Artifact storage with content-addressed deduplication
- Scope-based access control (USER/ROOM/GLOBAL)
- Session context passing between commands
- Audit trail integration for security

Commands:
    !scratch store <name>   - Store last weaving result
    !scratch get <name|id>  - Retrieve artifact
    !scratch list [scope]   - List artifacts
    !scratch delete <name>  - Delete artifact
    !scratch share <name>   - Share with room
    !scratch export <name>  - Upload to Matrix

Created: December 2025
"""

from .manager import ScratchPadManager
from .serialization import (
    ContentSerializer,
    deserialize_content,
    serialize_content,
)
from .types import (
    ArtifactReference,
    # Enums
    ArtifactScope,
    ArtifactStatus,
    ArtifactType,
    AuditEvent,
    AuditEventType,
    DeleteResult,
    ListResult,
    RetrieveResult,
    # Core dataclasses
    ScratchArtifact,
    ScratchPadConfig,
    # Session context
    SessionArtifactContext,
    # Result types
    StoreResult,
)

__all__ = [
    # Enums
    'ArtifactScope',
    'ArtifactType',
    'ArtifactStatus',
    'AuditEventType',

    # Core dataclasses
    'ScratchArtifact',
    'ArtifactReference',
    'ScratchPadConfig',
    'AuditEvent',

    # Result types
    'StoreResult',
    'RetrieveResult',
    'ListResult',
    'DeleteResult',

    # Session context
    'SessionArtifactContext',

    # Serialization
    'serialize_content',
    'deserialize_content',
    'ContentSerializer',

    # Manager
    'ScratchPadManager',
]

__version__ = '1.0.0'
