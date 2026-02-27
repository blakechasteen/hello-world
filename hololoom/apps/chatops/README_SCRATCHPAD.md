# ChatOps Scratch Pad

**Status**: Production Ready (December 2025)
**Location**: `hololoom/chatops/scratchpad/`
**Total Code**: ~3,500 lines

Persistent artifact storage system for ChatOps workflows that enables context passing between commands, intermediate result storage, and artifact sharing between users.

## Overview

The Scratch Pad system provides:
- **Artifact Storage** with content-addressed deduplication (SHA256)
- **Scope-Based Access Control** (USER/ROOM/GLOBAL)
- **Session Context Passing** between `!continue` commands
- **Audit Trail Integration** for security logging
- **Binary Content Support** with proper serialization
- **Rate Limiting** to prevent abuse

## Quick Start

### Matrix Bot Commands

```
!scratch store <name>   - Store last weaving result
!scratch get <name|id>  - Retrieve artifact
!scratch list [scope]   - List artifacts (user/room/all)
!scratch delete <name>  - Delete artifact
!scratch share <name>   - Share with room (USER→ROOM)
!scratch export <name>  - Export to file
```

### Programmatic Usage

```python
from hololoom.apps.chatops.scratchpad import (
    ScratchPadConfig,
    ScratchPadManager,
    ArtifactScope,
    ArtifactType,
)
from hololoom.apps.chatops.scratchpad.manager import create_and_start_manager

# Create configuration
config = ScratchPadConfig(
    metadata_path="./scratchpad/metadata.db",
    cas_storage_path="./scratchpad/content",
    audit_log_path="./scratchpad/audit.jsonl",
)

# Create and start manager
manager = await create_and_start_manager(config)

# Store an artifact
result = await manager.store(
    name="my_analysis",
    content="Analysis results...",
    user_id="@alice:matrix.org",
    room_id="!room:matrix.org",
    scope=ArtifactScope.USER,
    artifact_type=ArtifactType.TEXT,
)

if result.success:
    print(f"Stored: {result.artifact_id}")

# Retrieve an artifact
result = await manager.get(
    name_or_id="my_analysis",
    user_id="@alice:matrix.org",
    room_id="!room:matrix.org",
)

if result.success:
    print(f"Content: {result.content}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 ChatOps Commands                         │
│  !scratch store/get/list/delete/share/export            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              ScratchPadManager                           │
│  • Artifact CRUD operations                             │
│  • Scope enforcement (USER/ROOM/GLOBAL)                 │
│  • Auto-storage from Spacetime                          │
└─────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────┴──────────────┐
        ↓                              ↓
┌──────────────────┐          ┌────────────────────┐
│ MetadataStorage  │          │ ContentAddressed   │
│   (SQLite)       │          │     Storage        │
│                  │          │   (CAS Layer)      │
│ • Name→Hash map  │          │ • SHA256 hashing   │
│ • Scope control  │          │ • Deduplication    │
│ • Audit trail    │          │ • Ref counting     │
└──────────────────┘          └────────────────────┘
```

## Data Model

### ScratchArtifact

The core artifact type with full provenance tracking:

```python
@dataclass
class ScratchArtifact:
    # Identity
    artifact_id: str              # "artifact-{random_hex}"
    name: str                     # User-friendly name

    # Content
    content_hash: str             # SHA256 (CAS key)
    artifact_type: ArtifactType   # CODE/IMAGE/DOCUMENT/etc.
    size_bytes: int               # Content size

    # Ownership
    owner_user_id: str            # Matrix user ID
    owner_room_id: str            # Matrix room ID
    scope: ArtifactScope          # USER/ROOM/GLOBAL

    # Lifecycle
    created_at: datetime
    accessed_at: datetime
    expires_at: Optional[datetime]  # Default: 7 days
    access_count: int

    # Provenance
    source_spacetime_id: Optional[str]  # Link to Spacetime
    source_session_id: Optional[str]    # Link to conversation

    # Metadata
    metadata: Dict[str, Any]
```

### ArtifactScope

Three access levels for artifacts:

| Scope | Access | Use Case |
|-------|--------|----------|
| **USER** | Owner only in same room | Personal artifacts |
| **ROOM** | All users in same room | Shared team artifacts |
| **GLOBAL** | System-wide (admin only) | System templates |

### ArtifactType

10 content types with automatic detection:

| Type | Extensions | Description |
|------|------------|-------------|
| **CODE** | .py, .js, .ts, .go, .rs | Source code files |
| **IMAGE** | .png, .jpg, .webp, .gif | Image files |
| **DOCUMENT** | .md, .txt, .pdf, .docx | Documents |
| **ARCHIVE** | .zip, .tar, .gz | Archives |
| **AUDIO** | .mp3, .wav, .ogg | Audio files |
| **VIDEO** | .mp4, .webm | Video files |
| **DATA** | .json, .csv, .yaml | Structured data |
| **TEXT** | (default) | Plain text |
| **EMBEDDING** | | Vector embeddings |
| **CONTEXT** | | Session context |

## Commands Reference

### !scratch store

Store an artifact from the last weaving result or provided content.

**Syntax**:
```
!scratch store <name> [--type TYPE] [--scope SCOPE]
```

**Parameters**:
- `name`: Artifact name (alphanumeric, underscores, hyphens)
- `--type`: Artifact type (auto-detected if not specified)
- `--scope`: Access scope (USER by default)

**Examples**:
```
!scratch store my_analysis
!scratch store code_snippet --type code
!scratch store shared_results --scope room
```

### !scratch get

Retrieve an artifact by name or ID.

**Syntax**:
```
!scratch get <name|id>
```

**Examples**:
```
!scratch get my_analysis
!scratch get artifact-a1b2c3d4
```

### !scratch list

List artifacts with optional scope filtering.

**Syntax**:
```
!scratch list [scope] [--limit N]
```

**Parameters**:
- `scope`: Filter by scope (user/room/all)
- `--limit`: Maximum artifacts to return (default: 50)

**Examples**:
```
!scratch list           # List user's artifacts
!scratch list room      # List room artifacts
!scratch list all       # List all accessible
!scratch list --limit 10
```

### !scratch delete

Delete an artifact (owner only).

**Syntax**:
```
!scratch delete <name|id>
```

**Examples**:
```
!scratch delete old_analysis
!scratch delete artifact-a1b2c3d4
```

### !scratch share

Share a USER artifact with the room (changes scope to ROOM).

**Syntax**:
```
!scratch share <name|id>
```

**Examples**:
```
!scratch share my_analysis
```

### !scratch export

Export an artifact to a file.

**Syntax**:
```
!scratch export <name|id> [--path PATH]
```

**Examples**:
```
!scratch export my_analysis
!scratch export code_snippet --path ./exports/
```

## Configuration

### ScratchPadConfig

Complete configuration options:

```python
@dataclass
class ScratchPadConfig:
    # Storage paths
    metadata_db_path: str = "./scratchpad/metadata.db"
    cas_storage_path: str = "./scratchpad/content"
    audit_log_path: str = "./scratchpad/audit.jsonl"

    # Size limits
    max_artifact_size_bytes: int = 50 * 1024 * 1024  # 50MB
    max_artifacts_per_user: int = 100
    max_total_storage_bytes: int = 10 * 1024 * 1024 * 1024  # 10GB

    # Lifecycle
    default_ttl_days: int = 7
    cleanup_interval_hours: int = 24
    archive_before_delete: bool = True

    # Rate limiting
    rate_limit_requests: int = 10
    rate_limit_window_seconds: int = 60

    # Security
    enable_audit: bool = True

    # Name validation
    max_name_length: int = 255
    allowed_name_pattern: str = r'^[a-zA-Z0-9_\-\.]+$'
```

### Environment Variables

```bash
# Storage paths
SCRATCHPAD_METADATA_DB_PATH="./scratchpad/metadata.db"
SCRATCHPAD_CAS_PATH="./scratchpad/content"
SCRATCHPAD_AUDIT_PATH="./scratchpad/audit.jsonl"

# Limits
SCRATCHPAD_MAX_SIZE_BYTES=52428800  # 50MB
SCRATCHPAD_MAX_PER_USER=100
SCRATCHPAD_TTL_DAYS=7

# Rate limiting
SCRATCHPAD_RATE_LIMIT=10
SCRATCHPAD_RATE_WINDOW=60
```

## API Reference

### ScratchPadManager

Main manager class for all scratch pad operations.

#### store()

Store a new artifact.

```python
async def store(
    self,
    name: str,
    content: Union[str, bytes],
    user_id: str,
    room_id: str,
    scope: ArtifactScope = ArtifactScope.USER,
    artifact_type: Optional[ArtifactType] = None,
    source_spacetime_id: Optional[str] = None,
    source_session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    ttl_days: Optional[int] = None,
) -> StoreResult:
    """
    Store a new artifact.

    Returns:
        StoreResult with success status, artifact_id, and any errors
    """
```

#### get()

Retrieve an artifact by name or ID.

```python
async def get(
    self,
    name_or_id: str,
    user_id: str,
    room_id: str,
) -> RetrieveResult:
    """
    Retrieve an artifact.

    Returns:
        RetrieveResult with artifact, content, and access status
    """
```

#### list()

List accessible artifacts.

```python
async def list(
    self,
    user_id: str,
    room_id: str,
    scope_filter: Optional[ArtifactScope] = None,
    limit: int = 50,
    offset: int = 0,
) -> ListResult:
    """
    List artifacts accessible to the user.

    Returns:
        ListResult with list of ArtifactReference objects
    """
```

#### delete()

Delete an artifact (owner only).

```python
async def delete(
    self,
    name_or_id: str,
    user_id: str,
    room_id: str,
) -> DeleteResult:
    """
    Delete an artifact.

    Returns:
        DeleteResult with success status
    """
```

#### share()

Share a USER artifact with the room.

```python
async def share(
    self,
    name_or_id: str,
    user_id: str,
    room_id: str,
) -> StoreResult:
    """
    Share an artifact with the room (USER → ROOM scope).

    Returns:
        StoreResult with updated artifact info
    """
```

#### store_from_spacetime()

Store artifacts from a Spacetime weaving result.

```python
async def store_from_spacetime(
    self,
    spacetime: Any,
    user_id: str,
    room_id: str,
    name_prefix: str = "",
    scope: ArtifactScope = ArtifactScope.USER,
) -> List[StoreResult]:
    """
    Extract and store artifacts from a Spacetime result.

    Returns:
        List of StoreResult for each stored artifact
    """
```

#### get_session_context()

Get session context for `!continue` command.

```python
async def get_session_context(
    self,
    session_id: str,
    user_id: str,
    room_id: str,
) -> SessionArtifactContext:
    """
    Get artifacts linked to a conversation session.

    Returns:
        SessionArtifactContext with artifact references
    """
```

### Result Types

#### StoreResult

```python
@dataclass
class StoreResult:
    success: bool
    artifact_id: Optional[str] = None
    content_hash: Optional[str] = None
    deduplicated: bool = False
    error: Optional[str] = None
    error_code: Optional[str] = None
```

#### RetrieveResult

```python
@dataclass
class RetrieveResult:
    success: bool
    artifact: Optional[ScratchArtifact] = None
    content: Optional[Union[str, bytes]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
```

#### ListResult

```python
@dataclass
class ListResult:
    success: bool
    artifacts: List[ArtifactReference] = field(default_factory=list)
    total_count: int = 0
    error: Optional[str] = None
```

#### DeleteResult

```python
@dataclass
class DeleteResult:
    success: bool
    artifact_id: Optional[str] = None
    content_deleted: bool = False
    error: Optional[str] = None
```

## Security Model

### Scope-Based Access Control

The scratch pad enforces strict access control:

1. **USER Scope**
   - Only the owner can access
   - Must be in the same room
   - Default for new artifacts

2. **ROOM Scope**
   - All users in the room can access
   - Created via `!scratch share` or explicit scope
   - Room membership required

3. **GLOBAL Scope**
   - System-wide access
   - Admin-only creation
   - Used for system templates

### Access Check Logic

```python
def can_access(artifact, user_id, room_id):
    if artifact.scope == ArtifactScope.GLOBAL:
        return True
    if artifact.owner_room_id != room_id:
        return False
    if artifact.scope == ArtifactScope.USER:
        return artifact.owner_user_id == user_id
    if artifact.scope == ArtifactScope.ROOM:
        return True  # Room membership checked by Matrix
    return False
```

### Rate Limiting

Sliding window rate limiting prevents abuse:

```python
# Default: 10 requests per 60 seconds per user
rate_limiter = RateLimiter(
    max_requests=10,
    window_seconds=60
)

# Check before operation
if not rate_limiter.check(user_id):
    raise RateLimitExceededError("Too many requests")
```

### Audit Logging

All operations are logged for security:

```python
@dataclass
class AuditEvent:
    timestamp: datetime
    event_type: AuditEventType
    user_id: str
    room_id: str
    artifact_id: Optional[str]
    details: Dict[str, Any]
```

**Event Types**:
- `STORE`: Artifact created
- `RETRIEVE`: Artifact accessed
- `DELETE`: Artifact deleted
- `SHARE`: Artifact shared
- `EXPORT`: Artifact exported
- `ACCESS_DENIED`: Access attempt blocked

## Integration Examples

### With Weaving Orchestrator

Auto-store artifacts from Spacetime results:

```python
from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.apps.chatops.scratchpad.manager import create_and_start_manager

manager = await create_and_start_manager(config)

async with WeavingOrchestrator(cfg=cfg, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(query)

    # Store all artifacts from the result
    if spacetime.artifacts:
        results = await manager.store_from_spacetime(
            spacetime=spacetime,
            user_id=user_id,
            room_id=room_id,
            name_prefix="weave_",
        )

        for result in results:
            if result.success:
                print(f"Stored: {result.artifact_id}")
```

### With Conversation Handlers

Context passing for multi-turn conversations:

```python
from hololoom.apps.chatops.handlers import ConversationHandlers

class EnhancedHandlers(ConversationHandlers):
    def __init__(self, scratchpad_manager):
        super().__init__()
        self.scratchpad = scratchpad_manager

    async def handle_continue(self, session_id, user_id, room_id, message):
        # Get artifacts from session
        context = await self.scratchpad.get_session_context(
            session_id=session_id,
            user_id=user_id,
            room_id=room_id,
        )

        # Include artifacts in context
        artifact_context = []
        for ref in context.artifact_refs:
            result = await self.scratchpad.get(ref.artifact_id, user_id, room_id)
            if result.success:
                artifact_context.append({
                    "name": ref.name,
                    "type": ref.artifact_type,
                    "content": result.content,
                })

        # Continue with enriched context
        return await self.continue_conversation(
            session_id,
            message,
            extra_context=artifact_context,
        )
```

### With Matrix Bot

Register scratch pad commands:

```python
from nio import AsyncClient

class MatrixBot:
    def __init__(self, client, scratchpad_manager):
        self.client = client
        self.scratchpad = scratchpad_manager

    async def handle_command(self, room_id, user_id, command, args):
        if command == "scratch":
            subcommand = args[0] if args else "help"

            if subcommand == "store":
                name = args[1]
                result = await self.scratchpad.store(
                    name=name,
                    content=self.last_result,
                    user_id=user_id,
                    room_id=room_id,
                )
                return f"Stored as {result.artifact_id}" if result.success else result.error

            elif subcommand == "get":
                name_or_id = args[1]
                result = await self.scratchpad.get(name_or_id, user_id, room_id)
                return result.content if result.success else result.error

            elif subcommand == "list":
                result = await self.scratchpad.list(user_id, room_id)
                return "\n".join(f"- {a.name} ({a.artifact_type})" for a in result.artifacts)
```

## Content-Addressed Storage

### Deduplication

The CAS layer automatically deduplicates content:

```python
# Store same content with different names
result1 = await manager.store("analysis_v1", content, user_id, room_id)
result2 = await manager.store("analysis_v2", content, user_id, room_id)

# Both share the same content hash
assert result1.content_hash == result2.content_hash
assert result2.deduplicated == True  # Second was deduplicated
```

### Reference Counting

Content is only deleted when no artifacts reference it:

```python
# Both artifacts share content
await manager.store("copy1", content, user_id, room_id)
await manager.store("copy2", content, user_id, room_id)

# Delete first - content still exists (ref_count > 0)
await manager.delete("copy1", user_id, room_id)

# Delete second - content deleted (ref_count = 0)
await manager.delete("copy2", user_id, room_id)
```

### Directory Sharding

Content is sharded by hash prefix for filesystem efficiency:

```
content/
├── a1/
│   └── b2c3d4.../  # Full hash
├── b2/
│   └── c3d4e5.../
└── ...
```

## Binary Content Handling

### Serialization

The serialization module handles binary content properly:

```python
from hololoom.apps.chatops.scratchpad import serialize_content, deserialize_content

# Serialize any content
raw_bytes, metadata = serialize_content(content)

# Metadata includes:
# - format: RAW_BYTES, UTF8_TEXT, BASE64, GZIP_BASE64, JSON
# - content_type: Detected MIME type
# - original_size: Pre-compression size
# - compressed: Whether compression was applied
# - encoding: Text encoding if applicable

# Deserialize back
original = deserialize_content(raw_bytes, metadata)
```

### Compression

Large content is automatically compressed:

```python
# Default: Compress if > 1024 bytes and saves > 10%
raw_bytes, metadata = serialize_content(
    large_content,
    compress=True,
    compression_threshold=1024,
)

if metadata.compressed:
    print(f"Compressed: {metadata.original_size} → {len(raw_bytes)}")
```

### Content Type Detection

Automatic detection from magic bytes:

```python
from hololoom.apps.chatops.scratchpad.serialization import detect_content_type, ContentType

content_type = detect_content_type(raw_bytes)
# ContentType.IMAGE_PNG, ContentType.APPLICATION_JSON, etc.
```

## Testing

### Run Tests

```bash
# All scratch pad tests
pytest hololoom/chatops/scratchpad/tests/ -v

# Individual test files
pytest hololoom/chatops/scratchpad/tests/test_manager.py -v
pytest hololoom/chatops/scratchpad/tests/test_storage.py -v
pytest hololoom/chatops/scratchpad/tests/test_integration.py -v
```

### Demo Script

```bash
# Run comprehensive demo
PYTHONPATH=. python demos/demo_chatops_scratchpad.py
```

The demo covers:
1. Store and retrieve operations
2. Listing with scope filtering
3. Sharing artifacts
4. Session context passing
5. Export functionality
6. Content deduplication
7. Type detection
8. Binary content handling
9. Quota enforcement
10. Delete operations

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| **Store** | ~5ms | Plus I/O for large content |
| **Get** | ~2ms | With SQLite cache |
| **List** | ~3ms | With index scan |
| **Delete** | ~3ms | Plus CAS cleanup |
| **Share** | ~2ms | Metadata update only |

**Storage Efficiency**:
- SHA256 deduplication: 0-100% savings depending on content overlap
- Gzip compression: 30-70% savings for text content
- Reference counting: Automatic cleanup of orphaned content

## Troubleshooting

### Common Issues

**Rate limit exceeded**:
```
Error: Too many requests. Try again in 30 seconds.
```
Solution: Wait for rate limit window to reset.

**Artifact not found**:
```
Error: Artifact 'my_analysis' not found
```
Solution: Check name spelling, scope, and room context.

**Access denied**:
```
Error: Access denied to artifact 'private_data'
```
Solution: Artifact may be USER scope owned by another user.

**Quota exceeded**:
```
Error: User quota exceeded (100/100 artifacts)
```
Solution: Delete old artifacts or request quota increase.

**Content too large**:
```
Error: Content exceeds maximum size (50MB)
```
Solution: Split content or use external storage.

### Debug Logging

Enable debug logging:

```python
import logging
logging.getLogger("hololoom.apps.chatops.scratchpad").setLevel(logging.DEBUG)
```

### Audit Trail Review

Review audit log for issues:

```bash
# View recent events
tail -n 100 ./scratchpad/audit.jsonl | jq .

# Filter by event type
cat ./scratchpad/audit.jsonl | jq 'select(.event_type == "ACCESS_DENIED")'

# Filter by user
cat ./scratchpad/audit.jsonl | jq 'select(.user_id == "@alice:matrix.org")'
```

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `types.py` | 518 | Core dataclasses and enums |
| `storage.py` | 612 | SQLite + CAS storage layer |
| `manager.py` | 830 | High-level CRUD operations |
| `serialization.py` | 458 | Binary content handling |
| `__init__.py` | 81 | Public API exports |
| `tests/test_manager.py` | ~300 | Manager unit tests |
| `tests/test_storage.py` | ~250 | Storage layer tests |
| `tests/test_integration.py` | ~200 | End-to-end tests |

**Total**: ~3,500 lines of production code

## See Also

- [ChatOps Overview](../README.md) - Full ChatOps system documentation
- [Conversation Handlers](../handlers/README.md) - Matrix bot command handling
- [Audit Trail](../../alignment/audit_trail.py) - Security logging integration
- [Spacetime Artifacts](../../fabric/spacetime.py) - Artifact data model

---

**Created**: December 2025
**Author**: HoloLoom Team
**License**: MIT
