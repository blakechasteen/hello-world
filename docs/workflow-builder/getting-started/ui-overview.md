# UI Overview

A comprehensive guide to the Workflow Builder interface components.

## Main Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                           TOOLBAR                                    │
├───────────┬─────────────────────────────────────────┬───────────────┤
│           │                                         │               │
│   AGENT   │              CANVAS                     │  PROPERTIES   │
│  PALETTE  │         (Workflow Area)                 │    PANEL      │
│           │                                         │               │
│           │                                         │               │
├───────────┴─────────────────────────────────────────┴───────────────┤
│                         STATUS BAR                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Toolbar

The toolbar contains primary actions and workflow controls.

### Left Section

| Button | Action | Shortcut |
|--------|--------|----------|
| **New** | Create new workflow | - |
| **Open** | Load workflow from file | - |
| **Save** | Export workflow | Ctrl+S |
| **Undo** | Undo last action | Ctrl+Z |
| **Redo** | Redo undone action | Ctrl+Shift+Z |

### Center Section

| Button | Action | Shortcut |
|--------|--------|----------|
| **Execute** | Run workflow | Ctrl+Enter |
| **Pause** | Pause execution | F6 |
| **Stop** | Stop execution | Shift+F5 |
| **Step** | Execute one node | F10 |

### Right Section

| Button | Action | Shortcut |
|--------|--------|----------|
| **Search** | Search nodes | Ctrl+F |
| **Layout** | Auto-arrange nodes | - |
| **Theme** | Toggle dark/light mode | T |
| **Collaborate** | Open collaboration panel | C |
| **Settings** | Open settings | - |

## Agent Palette

Located on the left side (or bottom sheet on mobile).

### Categories

**Query Agents** (Blue)
- HoloLoom Query
- Memory Search
- Multi-Query

**Processing Agents** (Green)
- Matryoshka Embedder
- Synthesizer
- Recursive Refiner

**Memory Agents** (Purple)
- Memory Store
- Context Retriever
- Knowledge Fusion

**Decision Agents** (Orange)
- Thompson Sampler
- Convergence Engine
- Safety Guardrails

**Output Agents** (Teal)
- Response Generator
- Format Converter

**Control Flow** (Gray)
- Conditional Branch
- Loop Iterator
- Parallel Executor

### Using the Palette

1. **Drag to Add**: Drag any agent onto the canvas
2. **Search**: Type to filter agents by name
3. **Collapse/Expand**: Click category headers
4. **Templates**: Access saved templates at the bottom

## Canvas

The main workspace where you build workflows.

### Navigation

| Action | Mouse | Keyboard |
|--------|-------|----------|
| Pan | Middle-click drag | Arrow keys |
| Zoom | Scroll wheel | +/- keys |
| Fit to view | - | Home |
| Reset zoom | - | 0 |

### Node Interactions

| Action | Method |
|--------|--------|
| Select | Click |
| Multi-select | Shift+Click or drag box |
| Move | Drag selected |
| Delete | Select + Delete key |
| Duplicate | Ctrl+D |
| Copy/Paste | Ctrl+C / Ctrl+V |

### Connection Interactions

| Action | Method |
|--------|--------|
| Create | Drag from output to input port |
| Delete | Select connection + Delete |
| Reroute | Drag connection endpoint |

### Context Menu (Right-Click)

**On Node**:
- Edit Configuration
- Duplicate
- Delete
- Copy
- Add to Group
- Set Breakpoint

**On Canvas**:
- Add Node Here
- Paste
- Select All
- Zoom In/Out
- Fit to View

**On Connection**:
- Delete Connection
- Show Data Flow

## Nodes

### Anatomy of a Node

```
┌────────────────────────────────────┐
│ ○ [Icon] Node Label                │ ← Header (color by category)
├────────────────────────────────────┤
│                                    │
│         Node Preview               │ ← Content area
│                                    │
├────────────────────────────────────┤
│ [input]                   [output] │ ← Ports
└────────────────────────────────────┘
```

### Node States

| State | Visual | Meaning |
|-------|--------|---------|
| Default | White/Dark | Idle |
| Selected | Blue border | Currently selected |
| Running | Blue pulse | Executing |
| Success | Green check | Completed successfully |
| Error | Red X | Failed |
| Breakpoint | Red dot | Will pause here |
| Locked | Lock icon | Being edited by another user |

### Node Groups

Nodes can be grouped for organization:

1. **Select multiple nodes**
2. **Right-click → "Create Group"**
3. **Name the group**

Groups can be:
- Collapsed/expanded
- Moved as a unit
- Converted to composite nodes

## Properties Panel

Located on the right side (or slide-up panel on mobile).

### Sections

**General**
- Label (display name)
- Description
- Tags

**Configuration**
- Node-specific settings
- Input mappings
- Output mappings

**Advanced**
- Timeout
- Retry count
- Error handling

**Debug** (during execution)
- Input data
- Output data
- Execution time
- Error messages

### Input Types

| Type | Control | Example |
|------|---------|---------|
| Text | Input field | Query template |
| Number | Number input | Timeout (seconds) |
| Boolean | Checkbox | Enable caching |
| Select | Dropdown | Complexity mode |
| JSON | Code editor | Custom config |
| Template | Expression editor | `${input.query}` |

## Collaboration Panel

Accessed via "C" key or toolbar button.

### Presence List

Shows online users:
- Avatar with initials
- Status (idle/editing)
- Current selection

### Session Info

- Session ID (copyable)
- Join link
- Participant count

### Actions

- **Leave Session**: Exit collaboration
- **Copy Link**: Share session URL

## Performance Debug Panel

Accessed via "P" key.

Shows real-time metrics:
- **FPS**: Frames per second
- **Frame Time**: Milliseconds per frame
- **Nodes Rendered**: Visible vs total
- **Connections Rendered**: Visible vs total

## Status Bar

Located at the bottom.

Shows:
- Current zoom level
- Node count
- Connection count
- Execution status
- WebSocket connection status
- Last saved time

## Panels and Modals

### Search Panel (Ctrl+F)

- Search by node name, type, or config
- Navigate to results
- Filter by category

### Command Palette (Ctrl+P)

Quick access to all commands:
- Add nodes
- Execute actions
- Navigate panels
- Change settings

### History Panel

Shows undo/redo stack:
- Action descriptions
- Timestamps
- Click to jump to state

### Templates Gallery

Browse and use templates:
- Category filters
- Preview
- One-click apply

### Settings Modal

Configure:
- Theme
- Auto-save
- Grid snap
- Voice control
- Performance options

## Mobile Interface

On smaller screens:

### Layout Changes

- **Palette**: Bottom sheet (swipe up)
- **Properties**: Slide-up panel
- **Toolbar**: Compact mode

### Touch Gestures

| Gesture | Action |
|---------|--------|
| Tap | Select |
| Long press | Context menu |
| Drag | Move node |
| Pinch | Zoom |
| Two-finger drag | Pan |
| Swipe from edge | Open panels |

### Mobile-Specific Features

- Larger touch targets (44px minimum)
- Haptic feedback
- Floating action button
- Simplified toolbar

## Accessibility

### Keyboard Navigation

- Tab through interface elements
- Arrow keys for navigation
- Enter to activate
- Escape to close/cancel

### Screen Reader Support

- ARIA labels on all elements
- Role announcements
- Live regions for updates

### Visual Accessibility

- 4.5:1 contrast ratio minimum
- Focus indicators
- High contrast mode support
- Reduced motion option

## Tips & Tricks

1. **Quick Add**: Double-click canvas to open node picker
2. **Align Nodes**: Select multiple → Layout menu → Align
3. **Zoom to Selection**: Select nodes → Press "." key
4. **Quick Connect**: Alt+Click port to start connection mode
5. **Batch Edit**: Select multiple nodes to edit shared properties
6. **Template Workflow**: Select nodes → Save as Template

---

← [Your First Workflow](first-workflow.md) | [Agent Types](../features/nodes.md) →
