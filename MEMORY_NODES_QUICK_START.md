# MemoryNodes Component - Quick Start Guide

## Installation & Import

```typescript
import { MemoryNodes, MemoryNode } from '@/components/DetailPanel';
```

## Basic Usage

### Simple Display
```typescript
function MyComponent() {
  const nodes: MemoryNode[] = [
    {
      id: 'node-0001-abc123',
      content: 'Thompson Sampling balances exploration and exploitation',
      relevance: 0.92,
      sourceType: 'vector',
      accessedAt: '2025-12-11T10:30:00Z',
      stepId: 'step-01',
      metadata: { access_count: 3 },
    },
    // ... more nodes
  ];

  return <MemoryNodes nodes={nodes} />;
}
```

### With All Features
```typescript
function ThreadDetailsPanel() {
  const [memoryNodes, setMemoryNodes] = useState<MemoryNode[]>([]);
  const [groupByStep, setGroupByStep] = useState(false);

  const handleNodeClick = (nodeId: string) => {
    console.log('Copied node ID:', nodeId);
  };

  return (
    <div className="space-y-4">
      {/* Config toggle */}
      <label className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={groupByStep}
          onChange={(e) => setGroupByStep(e.target.checked)}
        />
        <span>Group by Step</span>
      </label>

      {/* Component */}
      <MemoryNodes
        nodes={memoryNodes}
        groupByStep={groupByStep}
        onNodeClick={handleNodeClick}
        className="custom-styling"
      />
    </div>
  );
}
```

## Data Model

### MemoryNode Interface
```typescript
interface MemoryNode {
  // Unique identifier (shown in compact form: first 16 chars)
  id: string;

  // Node content (truncated to 80 chars in preview, full in expansion)
  content: string;

  // Relevance score 0.0-1.0
  // - 0.9+: High (emerald heat map)
  // - 0.7-0.9: Medium (blue heat map)
  // - <0.7: Low (slate heat map)
  relevance: number;

  // Source type determines badge display
  // - 'graph': Knowledge graph (🔗 blue)
  // - 'vector': Vector database (📊 cyan)
  // - 'cache': Query cache (⚡ yellow)
  // - 'hot_pattern': Hot pattern (🔥 orange)
  sourceType: 'graph' | 'vector' | 'cache' | 'hot_pattern';

  // ISO 8601 timestamp (e.g., '2025-12-11T10:30:00Z')
  // Displayed as relative time (e.g., '42s ago')
  accessedAt: string;

  // Step ID (e.g., 'step-01', 'step-02')
  // Used for optional grouping
  stepId: string;

  // Optional metadata object
  // Commonly includes:
  // - access_count: How many times accessed
  // - confidence: Confidence score
  // - Custom fields as needed
  metadata?: Record<string, unknown>;
}
```

### MemoryNodesProps Interface
```typescript
interface MemoryNodesProps {
  // Required: array of nodes to display
  nodes: MemoryNode[];

  // Optional: group nodes by step (default: false)
  groupByStep?: boolean;

  // Optional: callback when node ID is copied
  onNodeClick?: (nodeId: string) => void;

  // Optional: CSS class for container
  className?: string;
}
```

## Features & How to Use

### 1. Heat Map Coloring
**Automatic** - Based on relevance score:
- Emerald background: relevance ≥ 0.9 (high)
- Blue background: relevance 0.7-0.9 (medium)
- Slate background: relevance < 0.7 (low)

Visual progress bar shows exact percentage.

### 2. Source Type Badges
**Automatic** - Color-coded by source:
```
🔗 Graph   → Blue (knowledge graph symbolic)
📊 Vector  → Cyan (vector DB semantic)
⚡ Cache   → Yellow (fast retrieval)
🔥 Hot     → Orange (frequently accessed)
```

### 3. Sorting
**UI buttons** - Click to sort:
```
[↑↓ Relevance] [↑↓ Recency] [↑↓ Access Count]
```
Active button highlighted in blue.

### 4. Search/Filter
**Search box** - Type to filter:
- Searches node ID and content
- Case-insensitive substring match
- Real-time (updates as you type)
- Maintains sort order
- Shows "Showing X of Y nodes"

### 5. Expand Node
**Click chevron** (< or v) to expand and see:
- Full content (scrollable)
- Full node ID (clickable to copy)
- Metadata (if present)
- Details row (source, step, relevance, accessed)

### 6. Copy Node ID
**Two ways**:
1. Click copy button (left of node)
2. Click on full node ID in expanded view

Visual feedback: button shows ✓ for 2 seconds after copy.

Calls `onNodeClick(nodeId)` if callback provided.

### 7. Group by Step
**Enable option** to group nodes:
```
Step step-01
├─ node-0001 (high relevance)
└─ node-0002 (medium relevance)

Step step-02
├─ node-0003 (low relevance)
└─ node-0004 (high relevance)
```

Useful for understanding memory retrieval timeline.

### 8. Summary Statistics
**Bottom of controls** shows:
```
Showing 12 of 15 nodes • Avg Relevance: 0.82
```

Updates when filtering or grouping.

### 9. Responsive Layout
**Automatic**:
- Mobile: Single column grid
- Large screens: Two column grid
- All controls responsive

### 10. Dark Theme
**Default** dark theme:
- Slate backgrounds (950/900/800)
- High contrast text
- Professional styling
- WCAG AA compliant

## Common Patterns

### Display in Thread Details
```typescript
function ThreadDetailsPanel({ thread }: { thread: AgentThread }) {
  return (
    <div className="space-y-4">
      <h2>Memory Nodes</h2>
      {thread.memoryNodes && thread.memoryNodes.length > 0 ? (
        <MemoryNodes nodes={thread.memoryNodes} />
      ) : (
        <p className="text-slate-400">No memory nodes accessed</p>
      )}
    </div>
  );
}
```

### With Step Grouping & Config
```typescript
function MemoryBrowser() {
  const [groupByStep, setGroupByStep] = useState(false);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3>Memory Nodes</h3>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={groupByStep}
            onChange={(e) => setGroupByStep(e.target.checked)}
          />
          <span>Group by Step</span>
        </label>
      </div>

      <MemoryNodes
        nodes={memoryNodes}
        groupByStep={groupByStep}
      />
    </div>
  );
}
```

### Parallel with File Tree
```typescript
function DetailsPanel() {
  const [selectedStep, setSelectedStep] = useState<Step | null>(null);

  return (
    <div className="grid grid-cols-2 gap-4">
      <FileTreeViewer
        files={selectedStep?.files || []}
        onSelect={handleFileSelect}
      />
      <MemoryNodes
        nodes={selectedStep?.memoryNodes || []}
        groupByStep={false}
      />
    </div>
  );
}
```

### With Logging
```typescript
function ThreadDetail({ thread }: { thread: AgentThread }) {
  const handleNodeClick = (nodeId: string) => {
    console.log(`[AUDIT] User copied node: ${nodeId}`);
    analytics.track('memory_node_copied', { nodeId, threadId: thread.id });
  };

  return (
    <MemoryNodes
      nodes={thread.memoryNodes}
      onNodeClick={handleNodeClick}
    />
  );
}
```

## Data Preparation

### From API Response
```typescript
async function fetchThreadDetails(threadId: string) {
  const response = await api.get(`/threads/${threadId}`);

  // Transform API response to MemoryNode array
  const memoryNodes: MemoryNode[] = response.memory_nodes.map((node: any) => ({
    id: node.node_id,
    content: node.text_content,
    relevance: node.relevance_score, // 0-1
    sourceType: node.source, // 'graph' | 'vector' | 'cache' | 'hot_pattern'
    accessedAt: new Date(node.accessed_timestamp).toISOString(),
    stepId: node.step_reference,
    metadata: node.extra_data,
  }));

  return memoryNodes;
}
```

### Mock Data for Development
```typescript
function generateMockNodes(count: number = 10): MemoryNode[] {
  const sources = ['graph', 'vector', 'cache', 'hot_pattern'] as const;
  const contents = [
    'Thompson Sampling balances exploration and exploitation',
    'UCB algorithm provides theoretical guarantees',
    'Multi-armed bandit problem formulation',
    // ... more
  ];

  return Array.from({ length: count }, (_, i) => ({
    id: `node-${String(i + 1).padStart(4, '0')}-${Math.random().toString(36).substring(7)}`,
    content: contents[i % contents.length],
    relevance: 0.5 + Math.random() * 0.5, // 0.5-1.0
    sourceType: sources[Math.floor(Math.random() * sources.length)],
    accessedAt: new Date(Date.now() - Math.random() * 60000).toISOString(),
    stepId: `step-${String(Math.floor(i / 2) + 1).padStart(2, '0')}`,
    metadata: {
      access_count: Math.floor(Math.random() * 10) + 1,
      confidence: (0.5 + Math.random() * 0.5).toFixed(2),
    },
  }));
}
```

## Styling Customization

### Default Dark Theme
Component uses Tailwind dark classes. No configuration needed.

### Custom CSS Class
```typescript
<MemoryNodes
  nodes={nodes}
  className="max-w-4xl mx-auto"
/>
```

### Override Tailwind Colors
Edit `tailwind.config.js`:
```javascript
module.exports = {
  theme: {
    extend: {
      colors: {
        slate: { /* ... */ },
        emerald: { /* ... */ },
        blue: { /* ... */ },
      },
    },
  },
};
```

## Performance Tips

### For Large Datasets (100+ nodes)
1. Component handles memoization internally
2. Search filters before rendering
3. Use `groupByStep` for organization
4. Pagination could be added in Phase 5

### For Small Datasets (<20 nodes)
1. No special optimization needed
2. All features work smoothly
3. Minimal memory footprint

### Handling Real-Time Updates
```typescript
function ThreadPanel({ thread }: { thread: AgentThread }) {
  const [memoryNodes, setMemoryNodes] = useState(thread.memoryNodes);

  // Watch for updates
  useEffect(() => {
    const unsubscribe = threadService.onMemoryNodesUpdated((nodes) => {
      setMemoryNodes(nodes);
    });

    return unsubscribe;
  }, []);

  return <MemoryNodes nodes={memoryNodes} />;
}
```

## Accessibility

### Keyboard Navigation
- **Tab**: Navigate interactive elements
- **Space/Enter**: Toggle expand, click buttons
- **Ctrl+F**: Browser search still works in content

### Screen Reader
Component uses semantic HTML and ARIA attributes. Screen readers will announce:
- Node headings and summaries
- Button purposes
- Expanded content

### High Contrast
Dark theme provides WCAG AA compliant contrast (4.5:1 minimum).

## Troubleshooting

### Nodes Not Showing
```typescript
// Check 1: Nodes array is not empty
console.log(nodes.length); // Should be > 0

// Check 2: Required fields present
nodes.forEach(n => {
  console.assert(n.id, 'Missing id');
  console.assert(n.content, 'Missing content');
  console.assert(n.relevance, 'Missing relevance');
  console.assert(n.sourceType, 'Missing sourceType');
  console.assert(n.accessedAt, 'Missing accessedAt');
  console.assert(n.stepId, 'Missing stepId');
});

// Check 3: Valid data types
console.log(typeof nodes[0].relevance); // Should be 'number'
console.log(nodes[0].relevance >= 0 && nodes[0].relevance <= 1); // Should be true
```

### Styles Not Applying
```typescript
// Check 1: Tailwind CSS imported in main.css
// @tailwind base;
// @tailwind components;
// @tailwind utilities;

// Check 2: Parent has dark class (if needed)
<div className="dark">
  <MemoryNodes nodes={nodes} />
</div>

// Check 3: Build process includes Tailwind
// tailwind.config.js should exist and reference your files
```

### Copy Not Working
```typescript
// Check 1: Clipboard API available
if (navigator.clipboard) {
  navigator.clipboard.writeText('test');
}

// Check 2: HTTPS context (required for clipboard)
// Localhost OK, HTTP not OK for clipboard

// Check 3: Check onNodeClick callback
<MemoryNodes
  nodes={nodes}
  onNodeClick={(id) => console.log('Copied:', id)}
/>
```

## Files Reference

| File | Purpose |
|------|---------|
| `MemoryNodes.tsx` | Main component (538 lines) |
| `MemoryNodes.demo.tsx` | Interactive demo (280 lines) |
| `MemoryNodes.test.tsx` | Test suite (490 lines, 80+ tests) |
| `MemoryNodes.md` | Complete documentation (470 lines) |
| `index.ts` | TypeScript exports |

## Next Steps

1. **Import & Use**: Copy code from "Basic Usage" section
2. **Customize Data**: Prepare MemoryNode[] from your API
3. **Test**: Run demo or integrate into your component
4. **Tune**: Adjust props and styling as needed
5. **Monitor**: Use `onNodeClick` for analytics if needed

## Support

For more information:
- See `MemoryNodes.md` for complete documentation
- Run `MemoryNodes.demo.tsx` for interactive examples
- Check `MemoryNodes.test.tsx` for usage patterns
- Review `MEMORY_NODES_DELIVERY.md` for delivery details

---

**Version**: 1.0.0
**Status**: Production Ready
**Date**: December 11, 2025
