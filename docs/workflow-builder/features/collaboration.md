# Real-Time Collaboration

Work together on workflows with live cursors, presence awareness, and conflict-free editing.

## Overview

The Workflow Builder supports real-time collaboration, allowing multiple users to edit the same workflow simultaneously. Built on WebSocket connections and CRDT (Conflict-free Replicated Data Types), it ensures smooth collaboration without data loss.

## Getting Started

### Creating a Session

1. Click the **"Collaborate"** button in the toolbar (or press `C`)
2. Click **"Create New Session"**
3. Share the generated link with collaborators

### Joining a Session

**Via Link**:
1. Open the shared session URL
2. Enter your name when prompted
3. Start collaborating

**Via Session ID**:
1. Click **"Collaborate"** in toolbar
2. Click **"Join Session"**
3. Enter the session ID
4. Enter your name

## Collaboration Features

### Remote Cursors

See where others are working in real-time:

```
┌────────────────────────────────────────────┐
│                                            │
│    [Query Node]            ▶ Alice         │
│         │                                  │
│         ↓                                  │
│    [Process Node]  ▶ Bob                   │
│         │                                  │
│         ↓                                  │
│    [Output Node]                           │
│                                            │
└────────────────────────────────────────────┘
```

- Cursors show user name and color
- Throttled updates (50ms) for smooth performance
- Automatically hidden when user is idle

### Presence Panel

Located on the right side, shows:

| Element | Description |
|---------|-------------|
| **Avatar** | User initials with assigned color |
| **Name** | User's display name |
| **Status** | Idle, Editing, or node being edited |
| **Actions** | Follow user, message user |

```
┌─────────────────────┐
│ Collaborators    ×  │
├─────────────────────┤
│ ● Alice             │
│   editing: Query-1  │
│                     │
│ ● Bob               │
│   idle              │
│                     │
│ ○ Carol (away)      │
│   last seen: 5m ago │
├─────────────────────┤
│ Session: abc-123    │
│ [Copy Link]         │
└─────────────────────┘
```

### Node Locking

When editing a node, it's automatically locked:

**Visual Indicators**:
- 🔒 Lock icon on node header
- Striped overlay pattern
- Lock owner name displayed
- Blue border for your locks
- Gray border for others' locks

**Lock Behavior**:
- Auto-acquired when selecting node for edit
- Released when clicking away
- 30-second timeout for abandoned locks
- Ownership transfer on disconnect

### Conflict Resolution

CRDT ensures conflict-free editing:

**Automatic Merging**:
- Node positions merge using "last-write-wins"
- Configuration changes merge field-by-field
- Connections merge based on timestamp

**Rare Conflicts**:
If conflicts can't be auto-resolved:

```
┌─────────────────────────────────────────┐
│ Conflict Detected                       │
├─────────────────────────────────────────┤
│ Both you and Alice edited Query-1       │
│                                         │
│ ┌─────────────┐  ┌─────────────┐       │
│ │ Your Change │  │ Alice's     │       │
│ │             │  │ Change      │       │
│ │ timeout: 30 │  │ timeout: 60 │       │
│ └─────────────┘  └─────────────┘       │
│                                         │
│ [Keep Mine] [Keep Theirs] [Merge Both] │
└─────────────────────────────────────────┘
```

## Session Management

### Session Roles

| Role | Permissions |
|------|-------------|
| **Owner** | Full control, delete session, manage roles |
| **Editor** | Add/edit/delete nodes and connections |
| **Viewer** | Read-only, can observe but not modify |

### Changing Roles

As session owner:
1. Open Presence Panel
2. Click user's name
3. Select **"Change Role"**
4. Choose new role

### Leaving a Session

1. Click **"Collaborate"** in toolbar
2. Click **"Leave Session"**
3. Confirm departure

Your edits are preserved; only your presence is removed.

### Ending a Session

As session owner:
1. Click **"Collaborate"** in toolbar
2. Click **"End Session"**
3. Confirm (all participants will be disconnected)

## Best Practices

### Communication

- **Use node comments** for async communication
- **Lock nodes** before major edits
- **Announce** large changes in chat

### Performance

- **Limit participants** to 5-10 for best performance
- **Close unused panels** to reduce updates
- **Use stable connections** (WiFi > mobile data)

### Workflow Organization

- **Divide work** by canvas regions
- **Use node groups** to organize ownership
- **Establish naming conventions** upfront

## Troubleshooting

### Connection Issues

**Symptoms**: Cursor freezes, "Reconnecting..." message

**Solutions**:
1. Check internet connection
2. Refresh the page (changes are preserved)
3. Clear browser cache
4. Try a different browser

### Sync Problems

**Symptoms**: Changes not appearing for others

**Solutions**:
1. Check WebSocket connection status (bottom right)
2. Click **"Force Sync"** in Collaborate menu
3. Export workflow and reimport

### Lock Stuck

**Symptoms**: Node shows locked but owner left

**Solutions**:
1. Wait 30 seconds for timeout
2. Right-click node → "Force Release Lock"
3. Contact session owner

## Technical Details

### WebSocket Protocol

Messages sent via WebSocket:

```javascript
// Cursor movement (throttled 50ms)
{ type: 'cursor_move', x: 150, y: 200 }

// Presence update
{ type: 'presence_update', status: 'editing', node_id: 'query-1' }

// Node lock
{ type: 'node_lock', node_id: 'query-1', action: 'acquire' }

// Workflow operation (CRDT)
{ type: 'workflow_operation', op: 'update_node', data: {...} }
```

### CRDT Operations

The backend uses a vector clock for ordering:

```python
class WorkflowCRDT:
    vector_clock: Dict[str, int]  # {user_id: sequence}

    def apply_operation(self, op):
        # Merge using vector clock ordering
        # Last-write-wins for concurrent edits
```

### Session Persistence

- Sessions persist for 24 hours of inactivity
- Workflow state saved every 30 seconds
- Full history available for 7 days

---

← [Templates & Presets](templates.md) | [Debugging Tools](debugging.md) →
