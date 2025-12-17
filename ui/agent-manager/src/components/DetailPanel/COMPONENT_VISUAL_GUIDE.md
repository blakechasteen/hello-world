# MemoryNodes Component - Visual Design Guide

## Component Layout Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ MEMORY NODES COMPONENT                                          │
│                                                                  │
│ ┌──────────────────────────────────────────────────────────────┤
│ │ SEARCH & CONTROLS BAR                                        │
│ │ ┌─────────────────────────────────┐  ┌─────────────────────┤
│ │ │ 🔍 Search by ID or content...   │  │ Sort by: [Rel] [Rec] [Acc] │
│ │ └─────────────────────────────────┘  └─────────────────────┤
│ │                                      Showing 12 of 15 nodes  │
│ │                                      Avg Relevance: 0.82    │
│ └──────────────────────────────────────────────────────────────┤
│                                                                  │
│ ┌──────────────────────────────────────────────────────────────┤
│ │ MEMORY NODE CARDS (2-COLUMN GRID)                            │
│ │                                                              │
│ │ ┌──────────────────────┐  ┌──────────────────────┐         │
│ │ │ HIGH RELEVANCE       │  │ MEDIUM RELEVANCE     │         │
│ │ │ (Emerald Heat Map)   │  │ (Blue Heat Map)      │         │
│ │ │                      │  │                      │         │
│ │ │ [v] [📊 Vector] 95%  │  │ [v] [🔗 Graph] 75%  │         │
│ │ │ [████████████░]      │  │ [████████░░░░░░░]   │         │
│ │ │ [Copy] [⚡5] [1s ago]│  │ [Copy] [⚡3] [15s]   │         │
│ │ │                      │  │                      │         │
│ │ │ Thompson Sampling... │  │ Bayesian methods... │         │
│ │ │ node-0001-abc… │  │ node-0002-def… │         │
│ │ │                      │  │                      │         │
│ │ │ EXPANDED:            │  │ COLLAPSED            │         │
│ │ │ [<] [📊 Vector] 95%  │  │ [>] [🔗 Graph] 75%  │         │
│ │ │ [████████████░]      │  │ [████████░░░░░░░]   │         │
│ │ │ [Copy ✓] [⚡5] [1s]   │  │ [Copy] [⚡3] [15s]   │         │
│ │ │                      │  │                      │         │
│ │ │ Full Content:        │  │ Thompson Sampling... │         │
│ │ │ Thompson Sampling    │  │ node-0002-def…      │         │
│ │ │ balances             │  │                      │         │
│ │ │ exploration...       │  │                      │         │
│ │ │                      │  │                      │         │
│ │ │ Node ID:             │  │                      │         │
│ │ │ node-0001-abc123     │  │                      │         │
│ │ │ [click to copy]      │  │                      │         │
│ │ │                      │  │                      │         │
│ │ │ Metadata:            │  │                      │         │
│ │ │ {                    │  │                      │         │
│ │ │   access_count: 5    │  │                      │         │
│ │ │   confidence: 0.95   │  │                      │         │
│ │ │ }                    │  │                      │         │
│ │ │                      │  │                      │         │
│ │ │ Source: vector       │  │                      │         │
│ │ │ Step: step-01        │  │                      │         │
│ │ │ Relevance: 95.0%     │  │                      │         │
│ │ │ Accessed: 1s ago     │  │                      │         │
│ │ └──────────────────────┘  └──────────────────────┘         │
│ │                                                              │
│ │ ┌──────────────────────┐  ┌──────────────────────┐         │
│ │ │ LOW RELEVANCE        │  │ HOT PATTERN          │         │
│ │ │ (Slate Heat Map)     │  │ (Orange Badge)       │         │
│ │ │                      │  │                      │         │
│ │ │ [v] [⚡ Cache] 55%    │  │ [v] [🔥 Hot] 80%    │         │
│ │ │ [█████░░░░░░░░░]     │  │ [████████░░░░░░]    │         │
│ │ │ [Copy] [⚡1] [1m]    │  │ [Copy] [⚡10] [30s]  │         │
│ │ │                      │  │                      │         │
│ │ │ Memory node...       │  │ Hot pattern node...  │         │
│ │ │ node-0003-ghi…       │  │ node-0004-jkl…       │         │
│ │ └──────────────────────┘  └──────────────────────┘         │
│ │                                                              │
│ └──────────────────────────────────────────────────────────────┤
│                                                                  │
│ When grouped by step:                                          │
│ ┌──────────────────────────────────────────────────────────────┤
│ │ ├─ Step step-01 (4 nodes)                                    │
│ │ │  ├─ [Node Cards...]                                       │
│ │ │  └─ [Node Cards...]                                       │
│ │                                                              │
│ │ └─ Step step-02 (3 nodes)                                    │
│ │    ├─ [Node Cards...]                                       │
│ │    └─ [Node Cards...]                                       │
│ └──────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────┘
```

## Color Scheme Reference

### Heat Map Colors (Relevance-Based)
```
EMERALD (High Relevance ≥ 0.9)
┌─────────────────────────────────┐
│ Background: emerald-950         │
│ Text: slate-100                 │
│ Border: emerald-700             │
│ Progress Bar: emerald-600       │
│ Icon: emerald-500               │
└─────────────────────────────────┘

BLUE (Medium Relevance 0.7-0.9)
┌─────────────────────────────────┐
│ Background: blue-950            │
│ Text: slate-100                 │
│ Border: blue-700                │
│ Progress Bar: blue-600          │
│ Icon: blue-500                  │
└─────────────────────────────────┘

SLATE (Low Relevance < 0.7)
┌─────────────────────────────────┐
│ Background: slate-800           │
│ Text: slate-100                 │
│ Border: slate-700               │
│ Progress Bar: slate-600         │
│ Icon: slate-500                 │
└─────────────────────────────────┘
```

### Source Type Badges
```
GRAPH (Blue Semantic)
🔗 Graph
Text: blue-400 (bright)
Background: blue-900 (dark)
Hover: blue-800

VECTOR (Cyan Semantic)
📊 Vector
Text: cyan-400 (bright)
Background: cyan-900 (dark)
Hover: cyan-800

CACHE (Yellow Semantic)
⚡ Cache
Text: yellow-400 (bright)
Background: yellow-900 (dark)
Hover: yellow-800

HOT PATTERN (Orange Semantic)
🔥 Hot
Text: orange-400 (bright)
Background: orange-900 (dark)
Hover: orange-800
```

### UI Element Colors
```
CONTROLS
Button (inactive): slate-800 background, slate-400 text
Button (active): blue-600 background, white text
Button (hover): slate-700 background
Focus ring: blue-500 (1px)

TEXT HIERARCHY
Primary: slate-100 (headings, titles)
Secondary: slate-300 (descriptions)
Tertiary: slate-400 (metadata, timestamps)
Disabled: slate-600 (inactive states)

BACKGROUND HIERARCHY
Primary: slate-950 (page background)
Secondary: slate-900 (panels, containers)
Tertiary: slate-800 (cards, inputs)
Hover: slate-700 (interactive elements)

FEEDBACK
Success: emerald-500 (checkmark)
Copy feedback: emerald-600
Warning: amber-500
Error: red-500
Info: blue-500
```

## Typography & Spacing

### Font Sizes
```
Header: 14px (uppercase, semibold)
Body: 13px (regular)
Metadata: 12px (small, gray)
Monospace (IDs): 11px (font-mono)
```

### Spacing Scale
```
xs: 4px (0.25rem)
sm: 8px (0.5rem)
md: 12px (0.75rem)
lg: 16px (1rem)
xl: 24px (1.5rem)
2xl: 32px (2rem)

Card padding: 12px (md)
Grid gap: 8px (sm)
Control gap: 8px (sm)
Section gap: 16px (lg)
```

## Component States

### Node Collapsed State
```
┌─────────────────────────────────────┐
│ [>] [🔗 Badge] [Rel%] ║▓▓▓▓░░░░░░║ │
│ [Copy] [⚡Count] [Time ago]         │
│ Preview text truncated to 80 chars… │
└─────────────────────────────────────┘
```

### Node Expanded State
```
┌─────────────────────────────────────┐
│ [v] [🔗 Badge] [Rel%] ║▓▓▓▓░░░░░░║ │
│ [Copy ✓] [⚡Count] [Time ago]       │
│ Preview text...                     │
│ ─────────────────────────────────── │
│ Full Content                        │
│ ┌───────────────────────────────┐  │
│ │ [scrollable content area]    │  │
│ └───────────────────────────────┘  │
│                                     │
│ Node ID                             │
│ ┌───────────────────────────────┐  │
│ │ node-0001-abc123 [click copy] │  │
│ └───────────────────────────────┘  │
│                                     │
│ Metadata                            │
│ ┌───────────────────────────────┐  │
│ │ {                             │  │
│ │   access_count: 5,            │  │
│ │   confidence: 0.95            │  │
│ │ }                             │  │
│ └───────────────────────────────┘  │
│                                     │
│ Source: vector  Relevance: 95.0%   │
│ Step: step-01   Accessed: 1s ago   │
└─────────────────────────────────────┘
```

### Search Active State
```
┌─────────────────────────────────────┐
│ 🔍 thompson [X]                     │
│                                     │
│ Showing 3 of 15 nodes               │
│ Sorted by: [Relevance]              │
└─────────────────────────────────────┘
```

### Sort Button States
```
INACTIVE:
[↑↓ Relevance] - slate-800 bg, slate-400 text

ACTIVE:
[↑↓ Relevance] - blue-600 bg, white text (currently active)

HOVER (inactive):
[↑↓ Relevance] - slate-700 bg
```

### Copy Feedback States
```
INITIAL (waiting):
[Copy Icon] - slate-400 text

COPYING (0.1s):
[Copy Icon] - slate-300 text (animated)

SUCCESS (2s):
[✓ Checkmark] - emerald-500 text (shown for 2 seconds)

RESET:
[Copy Icon] - slate-400 text (after 2s)
```

## Responsive Behavior

### Large Screen (1024px+)
```
Grid: 2 columns
Gap: 8px
Card width: ~50% - 8px
Search: full width
Controls: flex row with space-between
```

### Medium Screen (640px - 1023px)
```
Grid: 2 columns (wraps gracefully)
Gap: 6px
Card width: ~50% - 3px
Search: full width
Controls: flex column or wrapped
```

### Small Screen (<640px)
```
Grid: 1 column
Gap: 4px
Card width: 100%
Search: full width with smaller padding
Controls: stacked vertically
Font size: slightly smaller
Padding: reduced
```

## Animation & Interaction

### Transitions
```
Color: 200ms ease
Border: 200ms ease
Background: 200ms ease
Opacity: 150ms ease
Transform: none (no movements)
```

### Hover Effects
- **Button hover**: Background color change, 200ms ease
- **Card hover**: Border/shadow enhancement (minimal)
- **Search input focus**: Ring highlight, border color change

### Interactive Feedback
- **Copy button**: Immediate icon change to checkmark, 2s duration
- **Expand/collapse**: Chevron rotates (CSS transform not used, icon swaps)
- **Sort button**: Immediate background color change to blue-600

### Progress Bar Animation
```
Bar fill: animated from 0% to relevance%
Animation: transition 300ms ease
Direction: left to right
Color: semantic to heat map color
```

## Accessibility Indicators

### Focus Indicators
```
Default browser focus ring with custom styling:
Color: blue-500
Width: 1px
Offset: 2px
Style: solid
```

### Keyboard Navigation Indicators
```
Tab order: left-to-right, top-to-bottom
Visible focus: 2px blue ring
Tooltip text: Shows via title attribute
```

### Screen Reader Announcements
```
Heading: "Memory Nodes" (implicit via structure)
Buttons: "Expand", "Copy node ID", "Sort by [field]"
Counts: "Showing X of Y nodes"
Status: Color/relevance conveyed via text + aria-label
```

## Empty State Design

```
┌─────────────────────────────────────┐
│                                     │
│                 📚                  │
│                                     │
│       No memory nodes accessed      │
│                                     │
│  (Centered, slate colors)           │
│  (Icon size: 48px)                  │
│  (Text size: 14px)                  │
│                                     │
└─────────────────────────────────────┘
```

## Visual Hierarchy

### Primary Information (High Visual Weight)
- Relevance score with progress bar
- Node content preview
- Source type badge

### Secondary Information (Medium Visual Weight)
- Expanded node ID
- Full content in scrollable area
- Access count and timestamp

### Tertiary Information (Low Visual Weight)
- Metadata (shown only when expanded)
- Step reference
- Exact percentages and technical details

---

## Design System Integration

The MemoryNodes component follows HoloLoom's design system:

**Typography System**: Consistent font sizes and weights
**Color System**: Dark theme with semantic colors (slate, blue, emerald, orange)
**Spacing System**: Consistent 4px base unit
**Interaction System**: Standard hover, focus, and active states
**Accessibility**: WCAG AA compliant with keyboard support
**Responsive Design**: Mobile-first, responsive grid layout

All colors use Tailwind's standard palette for consistency with other UI components.
