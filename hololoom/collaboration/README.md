# HoloLoom Collaboration System - Multi-User Knowledge Workspace

**Status**: ✅ Production Ready (December 2025)
**Location**: `hololoom/collaboration/`
**Total Lines**: ~6,492 lines across 12 Python files
**Performance**: <100ms latency for sync, real-time voice via WebRTC
**Test Coverage**: 46+ functional tests, 13 performance benchmarks

Comprehensive multi-user collaborative workspace for HoloLoom with real-time synchronization, role-based access control, contribution attribution, voice/video communication, and intelligent UX adaptation.

---

## Overview

The HoloLoom Collaboration System transforms HoloLoom from a single-user memory system into an enterprise-grade multi-user knowledge workspace. Multiple users can simultaneously:

- **Build knowledge together** - Create, edit, share knowledge with teammates
- **Communicate in real-time** - WebRTC voice/video alongside typed collaboration
- **Track contributions** - Know who said what with multi-rater quality assessment
- **Control access** - Role-based permissions (VIEWER → CONTRIBUTOR → EDITOR → ADMIN → OWNER)
- **Stay in sync** - CRDT-inspired conflict resolution for simultaneous edits
- **See presence** - Real-time cursors, typing indicators, and activity status
- **Learn from interaction** - Thompson Sampling UX personalization based on user behavior

**Key Philosophy**: "Reliable Systems: Safety First" - Every collaboration feature gracefully degrades if components unavailable, with complete audit trails for compliance.

---

## Quick Start

### Basic Multi-User Session

```python
from hololoom.collaboration import (
    create_session_manager,
    UserManager,
    create_presence_manager,
    create_state_synchronizer,
    create_attribution_manager,
    create_voice_manager
)
from hololoom import hololoom

# Step 1: User management
user_manager = UserManager(storage_path="./users.json")
alice = user_manager.create_user("alice", email="alice@example.com", display_name="Alice")
bob = user_manager.create_user("bob", email="bob@example.com", role=UserRole.EDITOR)

# Step 2: Create collaborative session
session_manager = await create_session_manager()
session = session_manager.create_session(
    creator_id=alice.user_id,
    name="Thompson Sampling Research",
    session_type=SessionType.RESEARCH,
    max_participants=10
)

# Step 3: Participants join
alice_join = session_manager.join_session(
    session_id=session.session_id,
    user_id=alice.user_id,
    role=ParticipantRole.OWNER
)
bob_join = session_manager.join_session(
    session_id=session.session_id,
    user_id=bob.user_id,
    role=ParticipantRole.EDITOR
)

# Step 4: Create presence tracking
presence_manager = await create_presence_manager()
await presence_manager.update_presence(
    user_id=alice.user_id,
    session_id=session.session_id,
    status=ActivityStatus.ACTIVE,
    focus=FocusType.KNOWLEDGE_GRAPH
)

# Step 5: Real-time synchronization
state_sync = await create_state_synchronizer()
# When Alice edits a memory
operation = state_sync.create_operation(
    op_type=OperationType.UPDATE,
    node_id="memory_123",
    content="Updated content about Thompson Sampling",
    user_id=alice.user_id
)
await state_sync.apply_operation(operation)

# Bob receives synchronized update automatically
# Conflict resolution is automatic (LAST_WRITER_WINS by default)

# Step 6: Attribution tracking
attribution = await create_attribution_manager()
await attribution.record_contribution(
    user_id=alice.user_id,
    contribution_type=ContributionType.KNOWLEDGE_ADD,
    target_node_id="memory_123",
    content="Added Thompson Sampling explanation",
    context=AttributionContext(
        session_id=session.session_id,
        timestamp=datetime.now()
    )
)

# Step 7: Voice communication
voice_manager = await create_voice_manager()
room = voice_manager.create_voice_room(
    room_id=session.session_id,
    max_participants=10
)
await room.add_participant(
    user_id=alice.user_id,
    media_type=MediaType.AUDIO
)
await room.add_participant(
    user_id=bob.user_id,
    media_type=MediaType.AUDIO
)

# Step 8: Knowledge sharing & export
from hololoom.collaboration import KnowledgeSharing
knowledge_sharing = KnowledgeSharing()
export = await knowledge_sharing.export_session(
    session_id=session.session_id,
    format='markdown',  # json, json-ld, csv, rdf, markdown
    include_annotations=True,
    include_attribution=True
)

print(f"Collaboration active with {len(session.participants)} participants")
print(f"Memory synchronized across all clients")
print(f"Session exported: {export.format}")
```

---

## Key Components

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| **User Management** | `user_manager.py` | 441 | User registration, authentication, team management, session handling |
| **Session Management** | `session.py` | 520+ | Collaborative sessions (KNOWLEDGE_BASE, WHITEBOARD, RESEARCH, REVIEW, PRESENTATION) |
| **Presence Tracking** | `presence.py` | 480+ | Real-time cursor position, focus area, typing indicators, activity status |
| **State Synchronization** | `sync.py` | 750+ | CRDT-inspired sync, conflict resolution (LAST_WRITER_WINS, FIRST_WRITER_WINS, MERGE, MANUAL) |
| **Voice/Video** | `voice.py` | 680+ | WebRTC communication, media tracks, quality adaptation, signaling |
| **Attribution Tracking** | `attribution.py` | 1,335 | Multi-rater contribution quality, 14 contribution types, weighted scoring |
| **Annotation Refinement** | `annotation_refinement.py` | 1,065 | MRF-powered annotation quality, multi-pass refinement strategies |
| **Knowledge Sharing** | `knowledge_sharing.py` | 542 | Export to JSON/JSON-LD/CSV/RDF/Markdown with full provenance |
| **Access Control** | `access_control.py` | 380+ | RBAC permissions, 5-level hierarchy, time-bound access rules |
| **UX Learning** | `ux_learning.py` | 755 | Thompson Sampling adaptation of cursor visibility, voice defaults, presence style, notifications |
| **Contribution Tracker** | `contribution_tracker.py` | 493 | Legacy contribution tracking with review workflow |
| **Exports** | `__init__.py` | 160 | Public API, unified imports |

---

## Contribution Attribution System

### 14 Contribution Types

```python
class ContributionType(Enum):
    CREATE = "create"                 # Created new memory/node
    KNOWLEDGE_ADD = "knowledge_add"   # Added knowledge to existing
    KNOWLEDGE_EDIT = "knowledge_edit" # Modified knowledge
    KNOWLEDGE_DELETE = "knowledge_delete"  # Removed knowledge
    QUERY = "query"                   # Asked a question
    ANNOTATION = "annotation"         # Added annotation
    WHITEBOARD_DRAW = "whiteboard_draw"    # Drew on whiteboard
    WHITEBOARD_EDIT = "whiteboard_edit"    # Modified whiteboard
    REVIEW = "review"                 # Reviewed contribution
    APPROVAL = "approval"             # Approved contribution
    CORRECTION = "correction"         # Fixed error
    LINK = "link"                     # Created relationship
    TAG = "tag"                       # Added semantic tag
    IMPORT = "import"                 # Imported external content
```

### Quality Rating with Multi-Rater Support

```python
class QualityRating(Enum):
    EXCELLENT = "excellent"    # >0.9 quality
    GOOD = "good"             # 0.7-0.9
    ACCEPTABLE = "acceptable"  # 0.5-0.7
    NEEDS_IMPROVEMENT = "needs_improvement"  # 0.3-0.5
    POOR = "poor"             # <0.3

# Multi-rater weighted scoring
RATER_WEIGHTS = {
    RaterRole.OWNER: 2.0,         # Team owner's opinion counts 2x
    RaterRole.EDITOR: 1.5,        # Editors count 1.5x
    RaterRole.CONTRIBUTOR: 1.0,   # Contributors count 1x
    RaterRole.VIEWER: 0.5,        # Viewers count 0.5x
}

# Quality calculation: Weighted average of all rater opinions
weighted_quality = sum(
    (rating * RATER_WEIGHTS[rater_role]) for rating, rater_role in ratings
) / sum(RATER_WEIGHTS.values() for _ in ratings)
```

### Usage Example

```python
from hololoom.collaboration import create_attribution_manager, ContributionType

attribution = await create_attribution_manager()

# Record a contribution
await attribution.record_contribution(
    user_id="usr_alice",
    contribution_type=ContributionType.KNOWLEDGE_ADD,
    target_node_id="memory_123",
    content="Explained Thompson Sampling algorithm",
    context=AttributionContext(
        session_id="session_456",
        timestamp=datetime.now()
    )
)

# Get contribution stats
stats = attribution.get_user_stats("usr_alice")
print(f"Total contributions: {stats.total_contributions}")
print(f"Average quality: {stats.avg_quality:.2f}")

# Rate a contribution
await attribution.rate_contribution(
    contribution_id="contrib_789",
    rater_id="usr_bob",
    rater_role=RaterRole.EDITOR,
    rating=QualityRating.GOOD
)

# Weighted quality updates automatically
```

---

## Role-Based Access Control (RBAC)

### 5-Level Permission Hierarchy

```python
class Permission(Enum):
    READ = "read"          # Can read/view
    WRITE = "write"        # Can create/edit
    DELETE = "delete"      # Can remove
    SHARE = "share"        # Can share with others
    ADMIN = "admin"        # Full control

# Hierarchy (cumulative permissions)
ROLE_HIERARCHY = [
    UserRole.VIEWER,       # READ only
    UserRole.CONTRIBUTOR,  # READ + WRITE
    UserRole.EDITOR,       # READ + WRITE + DELETE
    UserRole.ADMIN,        # READ + WRITE + DELETE + SHARE
    UserRole.OWNER         # All permissions + ADMIN
]

# Check permissions
if alice.has_permission(UserRole.EDITOR):
    # Alice can edit (and read, write)
    pass
```

### Access Levels

```python
class AccessLevel(Enum):
    PRIVATE = "private"           # Owner only
    TEAM = "team"                 # Team members
    ORGANIZATION = "organization" # Entire org
    PUBLIC = "public"             # Anyone

# Time-bound access
class AccessRule:
    resource_id: str
    user_id: str
    permission: Permission
    access_level: AccessLevel
    granted_at: datetime
    expires_at: Optional[datetime]  # Automatic revocation
```

### Usage

```python
from hololoom.collaboration import AccessController, Permission, AccessLevel

controller = AccessController()

# Grant access
controller.grant_access(
    resource_id="memory_123",
    user_id="usr_bob",
    permission=Permission.WRITE,
    access_level=AccessLevel.TEAM,
    expires_at=datetime.now() + timedelta(days=30)  # Expires in 30 days
)

# Check access
can_write = controller.has_access(
    resource_id="memory_123",
    user_id="usr_bob",
    required_permission=Permission.WRITE
)

# Revoke access
controller.revoke_access(
    resource_id="memory_123",
    user_id="usr_bob"
)
```

---

## Real-Time Synchronization (CRDT-Inspired)

### Operation Types

```python
class OperationType(Enum):
    INSERT = "insert"      # Add content
    DELETE = "delete"      # Remove content
    UPDATE = "update"      # Modify content
    MOVE = "move"          # Relocate content
    CREATE = "create"      # Create node
    DESTROY = "destroy"    # Delete node
    BATCH = "batch"        # Multiple operations
```

### Conflict Resolution Strategies

```python
class ConflictResolution(Enum):
    LAST_WRITER_WINS = "lww"      # Latest timestamp wins (default)
    FIRST_WRITER_WINS = "fww"     # Original writer's version kept
    MERGE = "merge"               # Automatic merge of non-conflicting edits
    MANUAL = "manual"             # User chooses
```

### Vector Clocks for Ordering

```python
# Each client maintains vector clock for causality
# Example: Alice makes edit → vector_clock = {alice: 1}
#          Bob reads Alice's edit, makes his own → {alice: 1, bob: 1}
# Causality is tracked, so true concurrent edits detected

# When conflict occurs:
# - LAST_WRITER_WINS: Higher timestamp wins
# - FIRST_WRITER_WINS: Lower timestamp wins
# - MERGE: Concatenate if non-overlapping
# - MANUAL: Both versions sent, user chooses
```

### Usage

```python
from hololoom.collaboration import create_state_synchronizer, OperationType

state_sync = await create_state_synchronizer()

# Alice creates operation
alice_op = state_sync.create_operation(
    op_type=OperationType.UPDATE,
    node_id="memory_123",
    content="Alice's version",
    user_id="usr_alice"
)

# Bob creates concurrent operation on same node
bob_op = state_sync.create_operation(
    op_type=OperationType.UPDATE,
    node_id="memory_123",
    content="Bob's version",
    user_id="usr_bob"
)

# Both operations applied (sync detects conflict)
await state_sync.apply_operation(alice_op)
await state_sync.apply_operation(bob_op)

# Conflict resolution applied based on strategy
result = state_sync.get_conflict_info()
# result.resolution == ConflictResolution.LAST_WRITER_WINS
# Final content: "Bob's version" (later timestamp)
```

---

## Voice and Video Collaboration

### Media Types and Quality

```python
class MediaType(Enum):
    AUDIO = "audio"        # Voice only
    VIDEO = "video"        # Video + audio
    SCREEN = "screen"      # Screen sharing

class StreamQuality(Enum):
    LOW = "low"            # 48 kbps audio, 200p video
    MEDIUM = "medium"      # 128 kbps audio, 720p video (default)
    HIGH = "high"          # 256 kbps audio, 1080p video
    HD = "hd"              # 320+ kbps, 4K video

class ConnectionState(Enum):
    NEW = "new"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    CLOSED = "closed"
```

### WebRTC Signaling

```python
class SignalingMessage(Enum):
    OFFER = "offer"           # SDP offer from initiator
    ANSWER = "answer"         # SDP answer from responder
    ICE_CANDIDATE = "ice"     # ICE candidate for NAT traversal
    READY = "ready"           # Signal ready for media
    ERROR = "error"           # Signal error

class SignalingType(Enum):
    SDP_OFFER = "sdp_offer"
    SDP_ANSWER = "sdp_answer"
    ICE_CANDIDATE = "ice_candidate"
```

### Usage

```python
from hololoom.collaboration import create_voice_manager, MediaType, StreamQuality

voice_manager = await create_voice_manager()

# Create voice room for session
room = voice_manager.create_voice_room(
    room_id="session_123",
    settings=VoiceRoomSettings(
        max_participants=10,
        default_quality=StreamQuality.MEDIUM,
        audio_codec="opus",  # High-quality audio codec
        bitrate_control="adaptive"
    )
)

# Alice joins with audio
alice_connection = await room.add_participant(
    user_id="usr_alice",
    media_type=MediaType.AUDIO
)

# Bob joins with video
bob_connection = await room.add_participant(
    user_id="usr_bob",
    media_type=MediaType.VIDEO,
    quality=StreamQuality.HIGH
)

# Get connection stats
stats = alice_connection.get_stats()
print(f"Audio bitrate: {stats.audio_bitrate} kbps")
print(f"Latency: {stats.latency_ms} ms")
print(f"Packet loss: {stats.packet_loss:.1%}")

# Alice shares screen
screen_track = await alice_connection.add_media_track(
    media_type=MediaType.SCREEN,
    quality=StreamQuality.HIGH
)

# Stop sharing screen
await alice_connection.remove_media_track(screen_track.track_id)
```

---

## Real-Time Presence Tracking

### Activity Status

```python
class ActivityStatus(Enum):
    ONLINE = "online"           # Just connected
    ACTIVE = "active"           # Actively doing something
    IDLE = "idle"               # No activity for 5+ min
    AWAY = "away"               # Left the workspace
    DO_NOT_DISTURB = "dnd"      # Explicitly muted
    OFFLINE = "offline"         # Disconnected

class FocusType(Enum):
    QUERY_INPUT = "query"       # Typing query
    KNOWLEDGE_GRAPH = "graph"   # Viewing/editing graph
    WHITEBOARD = "whiteboard"   # Drawing
    VOICE = "voice"             # In voice chat
    EDITING = "editing"         # Editing content
    REVIEWING = "reviewing"     # Reviewing work
    IDLE = "idle"               # Not active
```

### Cursor and Selection Tracking

```python
@dataclass
class CursorPosition:
    """Real-time cursor position."""
    x: float          # 0.0-1.0 normalized
    y: float
    timestamp: datetime
    node_id: Optional[str] = None  # Hover target

@dataclass
class SelectionState:
    """Currently selected text/element."""
    start_offset: int
    end_offset: int
    node_id: str
    content: str      # Selected text

@dataclass
class TypingIndicator:
    """Show when user is typing."""
    user_id: str
    session_id: str
    is_typing: bool
    node_id: Optional[str] = None
    timestamp: datetime
```

### Usage

```python
from hololoom.collaboration import create_presence_manager

presence = await create_presence_manager()

# Update presence
await presence.update_presence(
    user_id="usr_alice",
    session_id="session_123",
    status=ActivityStatus.ACTIVE,
    focus=FocusType.KNOWLEDGE_GRAPH
)

# Update cursor (broadcast to others)
await presence.update_cursor(
    user_id="usr_alice",
    session_id="session_123",
    cursor=CursorPosition(x=0.5, y=0.3, node_id="memory_456")
)

# Update selection (broadcast to others)
await presence.update_selection(
    user_id="usr_alice",
    session_id="session_123",
    selection=SelectionState(
        start_offset=0,
        end_offset=50,
        node_id="memory_456",
        content="Thompson Sampling algorithm"
    )
)

# Show typing indicator
await presence.set_typing(
    user_id="usr_alice",
    session_id="session_123",
    is_typing=True,
    node_id="memory_456"
)

# Get all active users
active_users = presence.get_active_users("session_123")
for user in active_users:
    print(f"{user.username}: {user.status} (focus: {user.focus})")
```

---

## Adaptive UX Learning (Thompson Sampling)

### 8 Learnable UX Features

The system automatically learns optimal settings for each user:

```python
class UXFeature(Enum):
    CURSOR_VISIBILITY = "cursor_visibility"      # Show/hide other cursors
    VOICE_DEFAULT = "voice_default"              # Mute/unmute on join
    PRESENCE_STYLE = "presence_style"            # Compact/detailed/minimal
    ANNOTATION_TYPE = "annotation_type"          # Default annotation type
    TYPING_INDICATOR = "typing_indicator"        # Show/hide typing indicators
    NOTIFICATION_LEVEL = "notification_level"    # Notification frequency
    COLLABORATION_MODE = "collaboration_mode"    # Real-time/async/hybrid
    WHITEBOARD_TOOLS = "whiteboard_tools"        # Default tool set
```

### Thompson Sampling Learning

Uses Beta(α, β) priors that update based on user actions:

```python
# Success: α ← α + 1 (user liked this setting)
# Failure: β ← β + 1 (user changed it)
# Expected value: E[X] = α / (α + β)
# Confidence: min(1.0, (α + β - 2) / 50)

# System learns which settings work for which users
# Over time, personalization improves
```

### Usage

```python
from hololoom.collaboration import create_ux_learner, UXFeature, LearningContext

ux_learner = create_ux_learner(exploration_rate=0.1)

# Get recommended settings for new session
context = LearningContext.from_session(
    session_id="session_123",
    user_id="usr_alice",
    participant_count=3,
    time_of_day="day"
)

# System recommends based on learned preferences
cursor_setting = ux_learner.select(UXFeature.CURSOR_VISIBILITY, context)[0]
# Returns "show" if Alice usually has cursors enabled

# Provide feedback on selection
ux_learner.feedback(
    feature=UXFeature.CURSOR_VISIBILITY,
    option_id="show",
    success=True,  # Alice kept cursors enabled
    context=context
)

# Get all recommendations
recommendations = ux_learner.get_all_recommendations(context)
print(recommendations)
# Output: {
#   "cursor_visibility": {"top": "show", "confidence": 0.85},
#   "voice_default": {"top": "muted", "confidence": 0.92},
#   "presence_style": {"top": "compact", "confidence": 0.78},
#   ...
# }

# Infer feedback from actions
ux_learner.infer_feedback_from_action("showed_cursors", context)
# Automatically records positive feedback
```

---

## Session Types

```python
class SessionType(Enum):
    KNOWLEDGE_BASE = "kb"      # Shared knowledge repository
    WHITEBOARD = "whiteboard"  # Collaborative drawing
    RESEARCH = "research"      # Multi-agent research session
    REVIEW = "review"          # Code/content review
    PRESENTATION = "present"   # Knowledge presentation

class SessionState(Enum):
    INITIALIZING = "init"      # Setting up
    ACTIVE = "active"          # Live session
    PAUSED = "paused"          # Temporarily paused
    CLOSING = "closing"        # Cleanup
    CLOSED = "closed"           # Finished

class ParticipantRole(Enum):
    OWNER = "owner"            # Created session
    ADMIN = "admin"            # Full control
    EDITOR = "editor"          # Can edit
    COMMENTER = "commenter"    # Can comment only
    VIEWER = "viewer"          # Read-only
```

---

## Knowledge Sharing and Export

### Supported Formats

```python
# JSON: Full structured export with metadata
# JSON-LD: Linked Data for semantic web integration
# Markdown: Human-readable documentation
# CSV: Spreadsheet-compatible format
# RDF: Resource Description Framework for ontologies
```

### Usage

```python
from hololoom.collaboration import KnowledgeSharing

sharing = KnowledgeSharing()

# Export session to Markdown (human-readable)
export = await sharing.export_session(
    session_id="session_123",
    format="markdown",
    include_annotations=True,
    include_attribution=True,
    include_timestamps=True
)

# Access export
with open(f"session_{export.timestamp}.md", "w") as f:
    f.write(export.content)

# Export to JSON for programmatic use
json_export = await sharing.export_session(
    session_id="session_123",
    format="json"
)

# Share with external systems
await sharing.share_knowledge(
    knowledge_id="memory_123",
    scope=ShareScope.ORGANIZATION,  # Share with org members
    include_provenance=True,
    ttl_days=30  # Revoke access after 30 days
)
```

---

## When to Use

### ✅ Use Collaboration System when you need:

- **Multiple users** building knowledge together
- **Real-time synchronization** of edits across clients
- **Access control** with role-based permissions
- **Attribution tracking** (who said what)
- **Voice/video communication** alongside text
- **Activity awareness** (cursors, presence, typing)
- **Contribution quality** rating from multiple reviewers
- **Knowledge export** to share externally
- **Automatic UX adaptation** based on user preferences
- **Audit trails** for compliance and debugging

### 🟡 Consider alternatives when:

- Single-user system (overhead of collaboration features)
- Very simple knowledge (basic CRUD without versioning needed)
- No need for voice/video (can disable WebRTC features)
- Synchronous editing not required (could use async-only)

### ❌ Don't use when:

- Users don't need to interact (separate isolated sessions)
- Extreme latency sensitivity (<10ms, synchronization overhead)
- No compliance/audit requirements (simplified system might be faster)

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│              User Interface (Web/Desktop/AR)                  │
│  - Collaborative editor                                       │
│  - Voice/video controls                                       │
│  - Presence indicators                                        │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│            Collaboration Layer (hololoom/collaboration/)      │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ User Management     │ Session Management               │ │
│  │ - Authentication    │ - Create/join sessions           │ │
│  │ - Profiles          │ - Participant management         │ │
│  │ - Teams             │ - Session lifecycle              │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Real-Time Sync (CRDT)  │ Presence Tracking            │ │
│  │ - Conflict resolution  │ - Cursor position             │ │
│  │ - Vector clocks       │ - Activity status             │ │
│  │ - Operation buffers    │ - Typing indicators           │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Voice/Video (WebRTC)   │ Access Control (RBAC)         │ │
│  │ - Media tracks         │ - Permissions hierarchy        │ │
│  │ - Signaling            │ - Time-bound access           │ │
│  │ - Connection quality   │ - Resource-level rules        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Attribution Tracking   │ UX Learning                   │ │
│  │ - 14 contribution types│ - Thompson Sampling           │ │
│  │ - Multi-rater quality  │ - User preferences            │ │
│  │ - Weighted scoring     │ - Context-aware defaults      │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                                │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│              HoloLoom Core (Weaving, Memory)                  │
│  - Orchestrator                                              │
│  - Knowledge graph (Yarn Graph)                              │
│  - Memory systems (vector, semantic, etc.)                   │
└──────────────────────────────────────────────────────────────┘
```

---

## Performance Characteristics

| Operation | Latency | Throughput | Notes |
|-----------|---------|------------|-------|
| **Join session** | 100-200ms | 10 joins/s | Handshake + presence broadcast |
| **Sync operation** | 10-50ms | 100 ops/s | Apply edit, conflict check |
| **Presence update** | 5-10ms | 1000/s | Cursor, typing, status |
| **Voice latency** | 50-150ms | Per connection | WebRTC jitter + codec |
| **Attribution record** | 5-20ms | 500/s | Quality scoring |
| **UX recommendation** | <1ms | 10000/s | Thompson Sampling sample |
| **Knowledge export** | 100-500ms | Per session | Depends on size |

---

## Testing

```bash
# Run all collaboration tests
pytest hololoom/collaboration/tests/ -v

# Run specific test suites
pytest hololoom/collaboration/tests/test_sync.py -v          # Synchronization
pytest hololoom/collaboration/tests/test_attribution.py -v  # Attribution
pytest hololoom/collaboration/tests/test_voice.py -v         # Voice/video
pytest hololoom/collaboration/tests/test_ux_learning.py -v   # UX learning

# Performance benchmarks
pytest hololoom/collaboration/tests/test_performance.py -v
```

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 160 | Public API exports |
| `user_manager.py` | 441 | User and team management |
| `session.py` | 520+ | Session lifecycle |
| `presence.py` | 480+ | Real-time presence |
| `sync.py` | 750+ | CRDT synchronization |
| `voice.py` | 680+ | WebRTC communication |
| `attribution.py` | 1,335 | Contribution tracking |
| `annotation_refinement.py` | 1,065 | MRF-powered refinement |
| `knowledge_sharing.py` | 542 | Export/share knowledge |
| `access_control.py` | 380+ | RBAC permissions |
| `ux_learning.py` | 755 | Thompson Sampling UX |
| `contribution_tracker.py` | 493 | Legacy tracking |

**Total**: ~6,492 lines of production code

---

## Integration Examples

### Complete Research Session

```python
# Multi-user collaborative research with attribution and voice
async with await create_session_manager() as session_manager:
    # Create research session
    session = session_manager.create_session(
        creator_id=alice.user_id,
        name="ML Research",
        session_type=SessionType.RESEARCH,
        max_participants=5
    )

    # Team joins
    for user in [alice, bob, charlie]:
        session_manager.join_session(session.session_id, user.user_id)

    # Start voice
    voice = await create_voice_manager()
    room = voice.create_voice_room(session.session_id)
    for user in [alice, bob, charlie]:
        await room.add_participant(user.user_id, MediaType.AUDIO)

    # Collaborative editing with sync
    sync = await create_state_synchronizer()

    # Alice adds knowledge
    op1 = sync.create_operation(
        OperationType.CREATE,
        node_id="ml_basics",
        content="Machine Learning Fundamentals",
        user_id=alice.user_id
    )
    await sync.apply_operation(op1)

    # Bob adds to same session (automatically synchronized)
    op2 = sync.create_operation(
        OperationType.UPDATE,
        node_id="ml_basics",
        content="... with examples",
        user_id=bob.user_id
    )
    await sync.apply_operation(op2)

    # Track contributions
    attribution = await create_attribution_manager()
    await attribution.record_contribution(
        alice.user_id,
        ContributionType.KNOWLEDGE_ADD,
        "ml_basics"
    )

    # Rate contributions
    await attribution.rate_contribution(
        contribution_id,
        rater_id=charlie.user_id,
        rater_role=RaterRole.CONTRIBUTOR,
        rating=QualityRating.EXCELLENT
    )

    # Export for sharing
    sharing = KnowledgeSharing()
    export = await sharing.export_session(
        session.session_id,
        format="markdown",
        include_attribution=True
    )
```

---

## Reliability and Safety

The collaboration system follows **"Reliable Systems: Safety First"** principles:

1. **Graceful Degradation**
   - If WebRTC unavailable, falls back to text-only
   - If sync fails, operations queued for later
   - If voice fails, continues with text collaboration

2. **Automatic Fallback**
   - Missing optional components don't crash system
   - Voice optional; collaboration works without it
   - Access control fails safe (deny on error)

3. **Complete Audit Trails**
   - Every operation logged with user, timestamp, content
   - All conflicts recorded for later analysis
   - All access decisions logged
   - All voice sessions recorded (with consent)

4. **Data Persistence**
   - Sessions saved to durable storage
   - Contributions archived (never deleted, only marked)
   - Conflict history retained for replay/resolution
   - Export formats support external tools

---

## Future Enhancements (Roadmap)

**Phase 2** (Planned Q1 2026):
- Offline-first support (local sync, retry on reconnect)
- Advanced whiteboard with gesture recognition
- Natural language intent detection for "Add note about X"
- Asynchronous review workflows with notifications

**Phase 3** (Planned Q2 2026):
- End-to-end encryption for sensitive sessions
- GDPR compliance tools (data export, right to deletion)
- Advanced permission delegation (role templates)
- Analytics dashboard (who contributed what)

**Phase 4** (Planned Q3 2026):
- AI-powered meeting summarization
- Automatic conflict resolution with suggested merges
- Mobile-optimized client
- Integration with calendar/task systems

---

## See Also

- **[HoloLoom Memory System](../memory/)** - Storage and retrieval backend
- **[HoloLoom Agentic Reasoning](../agentic/)** - Multi-query research coordination
- **[HoloLoom Alignment Framework](../alignment/)** - Safety for collaborative actions
- **[Metaprompting Refinement Framework](../prompting/)** - Improve annotation quality
- **[SPRING_DYNAMICS.md](../memory/SPRING_DYNAMICS.md)** - Physics of memory activation
