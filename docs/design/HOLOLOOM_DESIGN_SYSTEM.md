# HoloLoom Unified Design System

> **Version**: 1.0.0 (December 2025)
> **Status**: Design Specification
> **Philosophy**: "Clarity through craft, trust through transparency"

---

## Brand Identity

### Brand Essence

**HoloLoom** is where **ancient wisdom meets bleeding-edge intelligence**. Like a master weaver at a cosmic loom, we help users spin threads of knowledge into tapestries of understanding.

**Brand Personality Spectrum**:
```
Trustworthy ←——————●——→ Adventurous
     Family Bank        Travel Agency

Conservative ←————●————→ Innovative
                 ↑
            "Relaxed Novel"
```

### Brand Pillars

| Pillar | Expression | Anti-Pattern |
|--------|------------|--------------|
| **Trust** | Transparent decisions, safety-first, audit trails | Hidden complexity, black boxes |
| **Intelligence** | Learns and adapts, multi-perspective reasoning | Static, one-size-fits-all |
| **Craft** | Beautiful data density, every pixel earns its place | Decoration, chartjunk |
| **Flow** | Seamless transitions, anticipatory UI | Jarring interruptions, modal hell |

### Voice & Tone

**Primary Voice**: Knowledgeable guide (not lecturer)
- ✅ "Let me show you what I found..."
- ✅ "Here's why I'm uncertain about this..."
- ❌ "ERROR: Invalid input detected"
- ❌ "Processing your request..."

**Tone Variations by Context**:
| Context | Tone | Example |
|---------|------|---------|
| Success | Warm confidence | "Found 12 related memories" |
| Uncertainty | Honest curiosity | "I'm 65% confident—want me to dig deeper?" |
| Error | Calm helpfulness | "That didn't work. Here's what we can try..." |
| Safety | Clear authority | "⚠️ This action requires review" |

---

## Color System

### Philosophy: Semantic Color

Colors carry **meaning**, not decoration. Every color signals something to the user.

### Base Palette: Cosmic Foundation

The cosmic overlay provides the metaphorical foundation—threads of starlight woven through the interface.

```css
/* Cosmic Foundation - Always Present */
:root {
  /* The Void - Deep space background */
  --cosmic-void: #0a0a0f;
  --cosmic-void-soft: #12121a;

  /* Starlight - Primary accent across all themes */
  --cosmic-starlight: #e8e4ff;
  --cosmic-nebula: #6366f1;      /* Indigo - the "thread" color */
  --cosmic-aurora: #22d3ee;      /* Cyan - activation/energy */

  /* Constellation Lines - Subtle connections */
  --cosmic-thread: rgba(99, 102, 241, 0.3);
  --cosmic-thread-active: rgba(99, 102, 241, 0.8);

  /* Celestial Bodies - Semantic anchors */
  --cosmic-sun: #fbbf24;         /* Warmth, success, confidence */
  --cosmic-mars: #ef4444;        /* Danger, critical, attention */
  --cosmic-earth: #22c55e;       /* Growth, safe, proceed */
  --cosmic-saturn: #a855f7;      /* Mystery, learning, exploration */
}
```

### Theme 1: Tufte (Data-Dense)

**Character**: The astronomer's observatory—every instrument precise, every reading meaningful.

```css
[data-theme="tufte"] {
  /* Backgrounds - Parchment in space */
  --bg-primary: #fafaf9;
  --bg-secondary: #f5f5f4;
  --bg-tertiary: #e7e5e4;
  --bg-elevated: #ffffff;

  /* Text - High contrast, readable */
  --text-primary: #1c1917;
  --text-secondary: #44403c;
  --text-tertiary: #78716c;
  --text-muted: #a8a29e;

  /* Data Ink - The only decoration that matters */
  --data-ink: #1c1917;
  --data-ink-secondary: #57534e;
  --grid-line: #e7e5e4;

  /* Accent - Cosmic thread through parchment */
  --accent-primary: var(--cosmic-nebula);
  --accent-hover: #4f46e5;

  /* Semantic */
  --semantic-success: #15803d;
  --semantic-warning: #a16207;
  --semantic-error: #b91c1c;
  --semantic-info: #1d4ed8;

  /* Special: Sparklines and micro-viz */
  --sparkline-stroke: var(--cosmic-nebula);
  --sparkline-fill: rgba(99, 102, 241, 0.1);
}
```

### Theme 2: Modern (Clean SaaS)

**Character**: The luxury starship bridge—spacious, calm, purposeful.

```css
[data-theme="modern"] {
  /* Backgrounds - Soft depth */
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-tertiary: #f1f5f9;
  --bg-elevated: #ffffff;
  --bg-overlay: rgba(15, 23, 42, 0.5);

  /* Text - Balanced hierarchy */
  --text-primary: #0f172a;
  --text-secondary: #475569;
  --text-tertiary: #94a3b8;
  --text-muted: #cbd5e1;

  /* Accent - Cosmic nebula */
  --accent-primary: var(--cosmic-nebula);
  --accent-secondary: var(--cosmic-aurora);
  --accent-hover: #4f46e5;
  --accent-subtle: rgba(99, 102, 241, 0.1);

  /* Surfaces */
  --surface-card: #ffffff;
  --surface-card-hover: #f8fafc;
  --border-subtle: #e2e8f0;
  --border-default: #cbd5e1;

  /* Shadows - Soft elevation */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
  --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.15);
}
```

### Theme 3: Terminal (Hacker)

**Character**: The starship engine room—raw power, direct control, no pretense.

```css
[data-theme="terminal"] {
  /* Backgrounds - Deep void */
  --bg-primary: var(--cosmic-void);
  --bg-secondary: var(--cosmic-void-soft);
  --bg-tertiary: #1a1a24;
  --bg-elevated: #1e1e2a;

  /* Text - Phosphor glow */
  --text-primary: #e4e4e7;
  --text-secondary: #a1a1aa;
  --text-tertiary: #71717a;
  --text-muted: #52525b;

  /* Code-specific */
  --text-code: #22d3ee;
  --text-string: #a5f3fc;
  --text-number: #fbbf24;
  --text-keyword: #c084fc;
  --text-comment: #6b7280;

  /* Accent - Neon cosmic */
  --accent-primary: var(--cosmic-aurora);
  --accent-secondary: var(--cosmic-nebula);
  --accent-hover: #06b6d4;

  /* Terminal specific */
  --cursor-color: var(--cosmic-aurora);
  --selection-bg: rgba(34, 211, 238, 0.3);
  --prompt-char: #22c55e;

  /* Borders - Subtle grid */
  --border-subtle: #27272a;
  --border-default: #3f3f46;
  --border-glow: rgba(34, 211, 238, 0.5);
}
```

### Semantic Color Mapping (All Themes)

```css
/* Universal semantic tokens */
:root {
  /* Confidence Spectrum */
  --confidence-high: #22c55e;      /* ≥0.8 - Green */
  --confidence-good: #84cc16;      /* 0.6-0.8 - Lime */
  --confidence-moderate: #eab308;  /* 0.4-0.6 - Yellow */
  --confidence-low: #f97316;       /* 0.2-0.4 - Orange */
  --confidence-very-low: #ef4444;  /* <0.2 - Red */

  /* Safety Levels */
  --safety-safe: #22c55e;
  --safety-caution: #eab308;
  --safety-warning: #f97316;
  --safety-danger: #ef4444;
  --safety-critical: #dc2626;

  /* Memory/Knowledge States */
  --memory-hot: #ef4444;           /* Frequently accessed */
  --memory-warm: #f97316;
  --memory-neutral: #6b7280;
  --memory-cool: #3b82f6;
  --memory-cold: #6366f1;          /* Rarely accessed */

  /* Edge Types (Knowledge Graph) */
  --edge-is-a: #3b82f6;            /* Blue - taxonomy */
  --edge-uses: #22c55e;            /* Green - functional */
  --edge-mentions: #9ca3af;        /* Gray - reference */
  --edge-leads-to: #f97316;        /* Orange - causal */
  --edge-part-of: #a855f7;         /* Purple - composition */

  /* Reasoning Modes */
  --mode-direct: #6b7280;
  --mode-verify: #3b82f6;
  --mode-research: #a855f7;
  --mode-plan: #22c55e;
}
```

---

## Typography

### Philosophy: Purposeful Hierarchy

Every text element has exactly one job. Typography creates scannable structure.

### Type Scale (8pt Base)

```css
:root {
  /* Scale: 1.25 ratio (Major Third) */
  --font-size-xs: 0.64rem;    /* 10.24px - Micro labels */
  --font-size-sm: 0.8rem;     /* 12.8px - Captions, metadata */
  --font-size-base: 1rem;     /* 16px - Body text */
  --font-size-md: 1.25rem;    /* 20px - Subheadings */
  --font-size-lg: 1.563rem;   /* 25px - Section headers */
  --font-size-xl: 1.953rem;   /* 31.25px - Page titles */
  --font-size-2xl: 2.441rem;  /* 39px - Hero text */
  --font-size-3xl: 3.052rem;  /* 48.8px - Display */

  /* Line Heights */
  --line-height-tight: 1.2;
  --line-height-snug: 1.375;
  --line-height-normal: 1.5;
  --line-height-relaxed: 1.625;
  --line-height-loose: 2;

  /* Letter Spacing */
  --tracking-tighter: -0.05em;
  --tracking-tight: -0.025em;
  --tracking-normal: 0;
  --tracking-wide: 0.025em;
  --tracking-wider: 0.05em;
  --tracking-widest: 0.1em;
}
```

### Font Families

```css
:root {
  /* Primary - UI and Body */
  --font-sans: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont,
               'Segoe UI', Roboto, sans-serif;

  /* Mono - Code, Data, Terminal */
  --font-mono: 'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code',
               Consolas, monospace;

  /* Display - Headlines, Marketing */
  --font-display: 'Satoshi', 'Inter', var(--font-sans);

  /* Special - Cosmic/Mythological accents */
  --font-cosmic: 'Cinzel', 'Cormorant Garamond', Georgia, serif;
}
```

### Typography Tokens

```css
/* Headings */
.heading-display {
  font-family: var(--font-display);
  font-size: var(--font-size-3xl);
  font-weight: 700;
  line-height: var(--line-height-tight);
  letter-spacing: var(--tracking-tight);
}

.heading-1 {
  font-family: var(--font-display);
  font-size: var(--font-size-2xl);
  font-weight: 600;
  line-height: var(--line-height-tight);
}

.heading-2 {
  font-family: var(--font-sans);
  font-size: var(--font-size-xl);
  font-weight: 600;
  line-height: var(--line-height-snug);
}

.heading-3 {
  font-family: var(--font-sans);
  font-size: var(--font-size-lg);
  font-weight: 600;
  line-height: var(--line-height-snug);
}

/* Body */
.body-large {
  font-family: var(--font-sans);
  font-size: var(--font-size-md);
  font-weight: 400;
  line-height: var(--line-height-relaxed);
}

.body-default {
  font-family: var(--font-sans);
  font-size: var(--font-size-base);
  font-weight: 400;
  line-height: var(--line-height-normal);
}

.body-small {
  font-family: var(--font-sans);
  font-size: var(--font-size-sm);
  font-weight: 400;
  line-height: var(--line-height-normal);
}

/* Data/Code */
.mono-data {
  font-family: var(--font-mono);
  font-size: var(--font-size-sm);
  font-weight: 500;
  font-feature-settings: 'tnum' 1; /* Tabular numbers */
}

.mono-code {
  font-family: var(--font-mono);
  font-size: var(--font-size-sm);
  font-weight: 400;
  line-height: var(--line-height-relaxed);
}

/* Labels */
.label-default {
  font-family: var(--font-sans);
  font-size: var(--font-size-sm);
  font-weight: 500;
  letter-spacing: var(--tracking-wide);
  text-transform: uppercase;
}

.label-cosmic {
  font-family: var(--font-cosmic);
  font-size: var(--font-size-sm);
  font-weight: 400;
  letter-spacing: var(--tracking-widest);
  text-transform: uppercase;
}
```

---

## Spacing System

### Philosophy: Consistent Rhythm

Spacing creates visual rhythm. We use an 8pt grid with intentional exceptions.

### Space Scale

```css
:root {
  /* Base unit: 4px */
  --space-0: 0;
  --space-px: 1px;
  --space-0.5: 0.125rem;  /* 2px */
  --space-1: 0.25rem;     /* 4px */
  --space-1.5: 0.375rem;  /* 6px */
  --space-2: 0.5rem;      /* 8px - Base rhythm */
  --space-2.5: 0.625rem;  /* 10px */
  --space-3: 0.75rem;     /* 12px */
  --space-4: 1rem;        /* 16px */
  --space-5: 1.25rem;     /* 20px */
  --space-6: 1.5rem;      /* 24px */
  --space-8: 2rem;        /* 32px */
  --space-10: 2.5rem;     /* 40px */
  --space-12: 3rem;       /* 48px */
  --space-16: 4rem;       /* 64px */
  --space-20: 5rem;       /* 80px */
  --space-24: 6rem;       /* 96px */
  --space-32: 8rem;       /* 128px */
}
```

### Semantic Spacing

```css
:root {
  /* Component internal spacing */
  --spacing-component-xs: var(--space-1);
  --spacing-component-sm: var(--space-2);
  --spacing-component-md: var(--space-3);
  --spacing-component-lg: var(--space-4);

  /* Between components */
  --spacing-stack-xs: var(--space-2);
  --spacing-stack-sm: var(--space-4);
  --spacing-stack-md: var(--space-6);
  --spacing-stack-lg: var(--space-8);
  --spacing-stack-xl: var(--space-12);

  /* Section spacing */
  --spacing-section-sm: var(--space-8);
  --spacing-section-md: var(--space-12);
  --spacing-section-lg: var(--space-16);
  --spacing-section-xl: var(--space-24);

  /* Page margins */
  --spacing-page-x: var(--space-6);
  --spacing-page-y: var(--space-8);

  /* Data-dense layouts (Tufte) */
  --spacing-dense-x: var(--space-2);
  --spacing-dense-y: var(--space-1);
}
```

### Layout Grid

```css
:root {
  /* Container widths */
  --container-sm: 640px;
  --container-md: 768px;
  --container-lg: 1024px;
  --container-xl: 1280px;
  --container-2xl: 1536px;
  --container-full: 100%;

  /* Column system */
  --grid-columns: 12;
  --grid-gutter: var(--space-6);
  --grid-gutter-dense: var(--space-3);

  /* Sidebar widths */
  --sidebar-collapsed: 64px;
  --sidebar-default: 280px;
  --sidebar-expanded: 360px;

  /* Panel widths */
  --panel-sm: 320px;
  --panel-md: 400px;
  --panel-lg: 480px;
  --panel-xl: 640px;
}
```

---

## Component Library Foundation

### Core Components

#### 1. Thread (The Fundamental Unit)

The **Thread** is our atomic building block—representing a single piece of woven knowledge.

```typescript
interface ThreadProps {
  id: string;
  type: 'memory' | 'query' | 'response' | 'reasoning' | 'action';
  content: string;
  confidence?: number;        // 0-1, shows confidence indicator
  timestamp?: Date;
  connections?: string[];     // Related thread IDs
  metadata?: Record<string, unknown>;

  // Visual variants
  variant?: 'default' | 'compact' | 'expanded' | 'card';
  glow?: boolean;            // Cosmic activation glow

  // Interaction
  onPin?: () => void;
  onDismiss?: () => void;
  onExpand?: () => void;
  onNavigate?: (direction: 'forward' | 'backward' | 'sideways' | 'deep') => void;
}
```

#### 2. Loom Panel

Container for related threads, with lifecycle management.

```typescript
interface LoomPanelProps {
  id: string;
  title: string;
  subtitle?: string;

  // Jenny lifecycle
  lifecycle: 'nascent' | 'stable' | 'dissolving' | 'archived';

  // Layout
  size: 'compact' | 'default' | 'large' | 'full';
  collapsible?: boolean;
  pinned?: boolean;

  // Content
  children: React.ReactNode;

  // Header actions
  actions?: Array<{
    icon: string;
    label: string;
    onClick: () => void;
  }>;

  // Safety
  safetyLevel?: 'safe' | 'caution' | 'warning' | 'danger';
}
```

#### 3. Confidence Indicator

Visual representation of certainty across the system.

```typescript
interface ConfidenceIndicatorProps {
  value: number;              // 0-1
  showLabel?: boolean;
  showEpistemic?: boolean;    // Show "I'm uncertain about my uncertainty"

  // Variants
  variant: 'badge' | 'bar' | 'gauge' | 'ring' | 'sparkline';
  size?: 'sm' | 'md' | 'lg';

  // Interaction
  onClick?: () => void;       // Opens confidence explanation
}
```

#### 4. Safety Gate

Visual indicator and interaction point for alignment system.

```typescript
interface SafetyGateProps {
  level: 'low' | 'medium' | 'high' | 'critical';
  action: string;
  reason?: string;

  // State
  status: 'pending' | 'approved' | 'blocked' | 'escalated';

  // Human-in-the-loop
  requiresApproval?: boolean;
  onApprove?: () => void;
  onBlock?: () => void;
  onEscalate?: () => void;

  // Audit
  auditTrailId?: string;
}
```

#### 5. Memory Navigator

Spatial navigation through knowledge graph.

```typescript
interface MemoryNavigatorProps {
  currentNode: string;
  graph: KnowledgeGraph;

  // Navigation state
  path: string[];
  direction?: 'forward' | 'backward' | 'sideways' | 'deep';

  // Visualization
  view: 'graph' | 'list' | 'timeline' | 'constellation';
  showConnections?: boolean;
  highlightPath?: string[];

  // Interaction
  onNavigate: (nodeId: string, direction: Direction) => void;
  onTimeTravel?: (timestamp: Date) => void;
}
```

#### 6. Reasoning Chain

Visualizes multi-step agentic reasoning.

```typescript
interface ReasoningChainProps {
  steps: Array<{
    type: 'query' | 'retrieve' | 'reason' | 'verify' | 'synthesize';
    content: string;
    confidence: number;
    duration?: number;
    sources?: string[];
  }>;

  mode: 'direct' | 'verify' | 'research' | 'plan_execute';
  status: 'thinking' | 'complete' | 'failed';

  // Streaming
  streaming?: boolean;
  currentStep?: number;

  // Interaction
  onStepClick?: (index: number) => void;
  onSourceClick?: (sourceId: string) => void;
}
```

### Composite Components

#### 7. Chat Interface

Full-featured chat with RAG and agentic modes.

```typescript
interface ChatInterfaceProps {
  // Conversation
  messages: Message[];

  // Input
  inputMode: 'text' | 'voice' | 'multimodal';
  placeholder?: string;

  // Reasoning mode selector
  reasoningMode: 'direct' | 'verify' | 'research' | 'plan_execute';
  onModeChange: (mode: ReasoningMode) => void;

  // Safety integration
  safetyEnabled?: boolean;
  onSafetyEvent?: (event: SafetyEvent) => void;

  // Memory integration
  showMemoryPanel?: boolean;
  onMemoryNavigate?: (nodeId: string) => void;

  // Streaming
  streamingEnabled?: boolean;
}
```

#### 8. Workflow Canvas

Drag-and-drop pipeline builder.

```typescript
interface WorkflowCanvasProps {
  // Graph
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];

  // Available agents
  agentPalette: AgentType[];

  // State
  execution?: {
    status: 'idle' | 'running' | 'paused' | 'complete' | 'error';
    currentNode?: string;
    progress?: number;
  };

  // Interaction
  onNodeAdd: (type: AgentType, position: Position) => void;
  onNodeConnect: (from: string, to: string) => void;
  onExecute: () => void;
  onExport: (format: 'json' | 'python' | 'yaml') => void;
}
```

#### 9. Analytics Dashboard

Real-time system monitoring.

```typescript
interface AnalyticsDashboardProps {
  // Panels to show
  panels: Array<{
    type: 'confidence_trajectory' | 'cache_gauge' | 'stage_waterfall' |
          'knowledge_graph' | 'safety_monitor' | 'learning_stats';
    size: 'sm' | 'md' | 'lg' | 'full';
  }>;

  // Time range
  timeRange: 'live' | '1h' | '24h' | '7d' | '30d';

  // Refresh
  autoRefresh?: boolean;
  refreshInterval?: number;

  // Layout
  layout: 'grid' | 'flow' | 'stack';
}
```

---

## Interaction Patterns

### Philosophy: Anticipatory, Never Blocking

The UI should feel like a skilled assistant—ready before you ask, never in your way.

### Core Patterns

#### 1. Progressive Disclosure

Information reveals itself as needed, not all at once.

```
Level 0: Glanceable summary (confidence badge, status icon)
    ↓ hover/focus
Level 1: Contextual preview (tooltip with key details)
    ↓ click/tap
Level 2: Full detail (panel expansion or navigation)
    ↓ explicit request
Level 3: Deep dive (separate view, full audit trail)
```

#### 2. Streaming-First

All operations that take >100ms should stream their progress.

```typescript
// Pattern: Stream with progressive enhancement
interface StreamingState<T> {
  status: 'idle' | 'streaming' | 'complete' | 'error';
  chunks: T[];
  progress?: number;  // 0-1 if deterministic

  // For non-deterministic operations
  stage?: string;     // "Retrieving memories...", "Reasoning...", etc.
  confidence?: number; // Current confidence
}
```

#### 3. Optimistic Updates with Safety Rails

Actions feel instant, but safety checks happen.

```typescript
// Pattern: Optimistic with rollback
async function executeAction(action: Action) {
  // 1. Immediate visual feedback
  setOptimisticState(action);

  // 2. Safety gate check (parallel)
  const safetyResult = await checkSafety(action);

  if (safetyResult.level === 'critical') {
    // 3a. Rollback and show gate
    rollbackOptimistic();
    showSafetyGate(safetyResult);
  } else {
    // 3b. Commit and maybe show info
    commitAction(action);
    if (safetyResult.level !== 'low') {
      showSafetyInfo(safetyResult);
    }
  }
}
```

#### 4. Spatial Memory

Users build mental models of where things are. Respect that.

```typescript
// Pattern: Consistent spatial anchors
const SPATIAL_ANCHORS = {
  safety: 'top-right',      // Always visible, always same place
  navigation: 'left',       // Sidebar or rail
  content: 'center',        // Main interaction area
  details: 'right',         // Context panel
  timeline: 'bottom',       // Temporal navigation
  actions: 'bottom-right',  // Primary actions
};
```

#### 5. Keyboard-First, Touch-Complete

Every action has a keyboard shortcut; every action works with touch.

```typescript
// Universal shortcuts
const GLOBAL_SHORTCUTS = {
  'Cmd+K': 'command_palette',        // Universal command
  'Cmd+/': 'help',                   // Contextual help
  'Cmd+.': 'toggle_theme',           // Cycle themes
  'Escape': 'dismiss_or_back',       // Context-aware escape
  'Cmd+Shift+S': 'safety_dashboard', // Jump to safety
  'Cmd+M': 'memory_navigator',       // Open memory
  'Cmd+Enter': 'execute',            // Submit/run

  // Navigation
  'j/k': 'navigate_list',
  'h/l': 'navigate_horizontal',
  'g g': 'go_to_top',
  'G': 'go_to_bottom',
  '/': 'search',
};
```

### Micro-Interactions

#### Confidence Pulse

When confidence changes significantly, pulse the indicator.

```css
@keyframes confidence-pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.1); opacity: 0.8; }
}

.confidence-changed {
  animation: confidence-pulse 0.6s ease-in-out;
}
```

#### Thread Weaving

New connections in the graph animate like threads being woven.

```css
@keyframes thread-weave {
  0% {
    stroke-dashoffset: 100%;
    opacity: 0;
  }
  50% {
    opacity: 1;
  }
  100% {
    stroke-dashoffset: 0;
    opacity: 1;
  }
}

.edge-new {
  animation: thread-weave 0.8s ease-out forwards;
}
```

#### Safety Gate Appearance

Safety gates slide in with authority, not aggression.

```css
@keyframes safety-gate-enter {
  0% {
    transform: translateY(-8px);
    opacity: 0;
    backdrop-filter: blur(0px);
  }
  100% {
    transform: translateY(0);
    opacity: 1;
    backdrop-filter: blur(8px);
  }
}

.safety-gate {
  animation: safety-gate-enter 0.3s ease-out;
}
```

---

## Animation Philosophy

### Principles

1. **Purposeful**: Every animation communicates something
2. **Swift**: 150-300ms for most transitions (never >500ms for UI)
3. **Interruptible**: User action always wins
4. **Reduced Motion**: Respect `prefers-reduced-motion`

### Timing Functions

```css
:root {
  /* Standard easings */
  --ease-in: cubic-bezier(0.4, 0, 1, 1);
  --ease-out: cubic-bezier(0, 0, 0.2, 1);
  --ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);

  /* Expressive easings */
  --ease-bounce: cubic-bezier(0.34, 1.56, 0.64, 1);
  --ease-smooth: cubic-bezier(0.4, 0, 0, 1);

  /* Cosmic/mystical */
  --ease-cosmic: cubic-bezier(0.22, 1, 0.36, 1);

  /* Durations */
  --duration-instant: 50ms;
  --duration-fast: 150ms;
  --duration-normal: 250ms;
  --duration-slow: 350ms;
  --duration-slower: 500ms;
}
```

### Reduced Motion

```css
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

---

## Accessibility Standards

### Compliance: WCAG 2.1 AA (AAA where feasible)

### Requirements

#### Color Contrast
- **Normal text**: 4.5:1 minimum
- **Large text** (≥18px or ≥14px bold): 3:1 minimum
- **UI components**: 3:1 minimum
- **Focus indicators**: 3:1 minimum

#### Keyboard Navigation
- All interactive elements focusable
- Logical tab order
- Visible focus indicators
- Skip links for main content
- No keyboard traps

#### Screen Readers
- Semantic HTML (landmarks, headings, lists)
- ARIA labels where needed
- Live regions for dynamic content
- Meaningful alt text

#### Motion
- Respect `prefers-reduced-motion`
- No auto-playing animations >5s
- Pause/stop controls for animations

### ARIA Patterns

```typescript
// Pattern: Accessible panel
<div
  role="region"
  aria-labelledby="panel-title"
  aria-describedby="panel-description"
  aria-live="polite"
  aria-relevant="additions text"
>
  <h2 id="panel-title">Safety Dashboard</h2>
  <p id="panel-description">Real-time safety monitoring</p>
  {/* content */}
</div>

// Pattern: Confidence indicator
<div
  role="meter"
  aria-valuenow={0.85}
  aria-valuemin={0}
  aria-valuemax={1}
  aria-valuetext="85% confidence"
  aria-label="Response confidence"
>
  {/* visual representation */}
</div>

// Pattern: Navigation
<nav aria-label="Memory navigation">
  <button aria-label="Navigate forward">→</button>
  <button aria-label="Navigate backward">←</button>
  <button aria-label="Navigate sideways">↔</button>
  <button aria-label="Navigate deep">↓</button>
</nav>
```

---

## Platform Adaptations

### Responsive Breakpoints

```css
:root {
  --breakpoint-sm: 640px;   /* Mobile landscape */
  --breakpoint-md: 768px;   /* Tablet portrait */
  --breakpoint-lg: 1024px;  /* Tablet landscape / small desktop */
  --breakpoint-xl: 1280px;  /* Desktop */
  --breakpoint-2xl: 1536px; /* Large desktop */
}
```

### Platform-Specific Considerations

#### Web (Primary)
- Full feature set
- All themes available
- Keyboard + mouse + touch
- PWA capabilities

#### Desktop (Electron)
- Native window chrome
- System tray integration
- File system access
- Offline support
- Background processes

#### VS Code Extension
- Follow VS Code theming
- Webview panels
- TreeView for navigation
- Status bar integration
- Command palette commands

#### Mobile (PWA)
- Touch-optimized hit targets (44px minimum)
- Gesture navigation
- Reduced complexity mode
- Adaptive layout
- Pull-to-refresh

#### AR (Future)
- Spatial panel placement
- Gaze-based interaction
- Hand gesture support
- World-anchored content

---

## Design Tokens Export

### Format: Style Dictionary

All tokens are exportable to:
- CSS Custom Properties
- Tailwind config
- React Native StyleSheet
- Figma Variables
- iOS/Android native

```json
{
  "color": {
    "cosmic": {
      "void": { "value": "#0a0a0f" },
      "nebula": { "value": "#6366f1" },
      "aurora": { "value": "#22d3ee" }
    },
    "semantic": {
      "confidence": {
        "high": { "value": "#22c55e" },
        "good": { "value": "#84cc16" },
        "moderate": { "value": "#eab308" },
        "low": { "value": "#f97316" },
        "very-low": { "value": "#ef4444" }
      }
    }
  },
  "typography": {
    "font": {
      "sans": { "value": "'Inter', sans-serif" },
      "mono": { "value": "'JetBrains Mono', monospace" }
    }
  },
  "spacing": {
    "base": { "value": "8px" },
    "scale": {
      "1": { "value": "4px" },
      "2": { "value": "8px" },
      "4": { "value": "16px" }
    }
  }
}
```

---

## Next Steps: Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
1. ✅ Design system specification (this document)
2. Set up design token pipeline
3. Create Figma component library
4. Build CSS foundation with all themes
5. Create core React components (Thread, Panel, Confidence)

### Phase 2: Safety First (Week 3-4) - Priority #1
1. Safety Dashboard full implementation
2. Safety Gate component
3. Audit Trail viewer
4. Risk visualization components
5. Human-in-the-loop flows

### Phase 3: Chat + RAG (Week 5-6) - Priority #2
1. Chat interface with streaming
2. RAG mode integration
3. Source attribution UI
4. Reasoning chain visualization
5. Multi-modal input

### Phase 4: Memory Exploration (Week 7-8) - Priority #3
1. Memory Navigator component
2. Knowledge Graph visualization
3. Time-travel interface
4. Pattern discovery UI
5. Navigation directions

### Phase 5: Multi-Device (Week 9-10) - Priority #4
1. Cross-device sync UI
2. Device pairing flow
3. Sync status indicators
4. Conflict resolution UI
5. Offline mode indicators

### Phase 6: Workflow Builder (Week 11-12) - Priority #5
1. Workflow canvas
2. Agent palette
3. Connection drawing
4. Execution visualization
5. Export capabilities

### Phase 7: Voice + Analytics (Week 13-14) - Priority #6, #7
1. Voice input integration
2. Voice correction UI
3. Analytics dashboard
4. Real-time metrics
5. Historical analysis

### Phase 8: Polish + AR (Week 15-16) - Priority #8
1. AR visualization foundation
2. Cross-platform polish
3. Performance optimization
4. Accessibility audit
5. Documentation

---

## Appendix: Cosmic Glossary

| Term | Meaning in HoloLoom |
|------|---------------------|
| **Thread** | A single piece of knowledge or reasoning step |
| **Loom** | The system that weaves threads together |
| **Weave** | The process of combining threads into understanding |
| **Yarn** | Connected threads forming a knowledge graph |
| **Warp** | The continuous mathematical space for reasoning |
| **Shuttle** | The orchestrator moving through the loom |
| **Fabric** | The final woven output (Spacetime) |
| **Constellation** | A cluster of related memories |
| **Nebula** | Primary accent color (indigo) - the color of threads |
| **Aurora** | Secondary accent (cyan) - the color of activation |
| **Void** | Background space - the cosmic canvas |

---

*"In the cosmic loom of knowledge, every thread matters, every weave tells a story, and every user is a master weaver."*
