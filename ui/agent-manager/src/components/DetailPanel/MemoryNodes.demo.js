import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { MemoryNodes } from './MemoryNodes';
/**
 * Mock data generator for demo purposes
 */
const generateMockMemoryNodes = (count = 12) => {
    const sourceTypes = ['graph', 'vector', 'cache', 'hot_pattern'];
    const sampleContents = [
        'Thompson Sampling balances exploration and exploitation in bandit problems',
        'UCB algorithm provides theoretical guarantees for regret bounds',
        'Multi-armed bandit problem: maximize expected cumulative reward',
        'Posterior distribution in Bayesian inference represents parameter uncertainty',
        'Exploitation vs exploration tradeoff in reinforcement learning',
        'Contextual bandits extend standard bandits with feature vectors',
        'Regret analysis bounds suboptimal decisions compared to optimal policy',
        'Thompson Sampling samples from posterior, empirically optimal exploration',
        'Softmax temperature controls exploration in neural networks',
        'Linear bandits assume reward linearly related to context features',
        'Dirichlet distribution conjugate prior for categorical distributions',
        'Beta distribution conjugate prior for binomial experiments',
    ];
    return Array.from({ length: count }, (_, i) => {
        const now = new Date();
        const accessedAgo = Math.random() * 60000; // 0-60 seconds ago
        return {
            id: `node-${String(i + 1).padStart(4, '0')}-${Math.random().toString(36).substring(7)}`,
            content: sampleContents[i % sampleContents.length],
            relevance: 0.5 + Math.random() * 0.5, // 0.5 - 1.0
            sourceType: sourceTypes[Math.floor(Math.random() * sourceTypes.length)],
            accessedAt: new Date(now.getTime() - accessedAgo).toISOString(),
            stepId: `step-${String(Math.floor(i / 3) + 1).padStart(2, '0')}`,
            metadata: {
                access_count: Math.floor(Math.random() * 10) + 1,
                confidence: (0.5 + Math.random() * 0.5).toFixed(2),
                embedding_dim: 384,
            },
        };
    });
};
/**
 * Demo Component for MemoryNodes
 * Showcases all features and configurations
 */
export const MemoryNodesDemo = () => {
    const [mockNodes] = useState(() => generateMockMemoryNodes(15));
    const [groupByStep, setGroupByStep] = useState(false);
    const [lastCopiedId, setLastCopiedId] = useState(null);
    const handleNodeClick = (nodeId) => {
        setLastCopiedId(nodeId);
        setTimeout(() => setLastCopiedId(null), 2000);
    };
    return (_jsxs("div", { className: "min-h-screen bg-slate-950 p-8 space-y-8", children: [_jsxs("div", { className: "space-y-2", children: [_jsx("h1", { className: "text-3xl font-bold text-white", children: "MemoryNodes Component Demo" }), _jsx("p", { className: "text-slate-400", children: "Comprehensive demonstration of the MemoryNodes component with all features" })] }), _jsxs("div", { className: "bg-slate-900 border border-slate-800 rounded-lg p-6 space-y-4", children: [_jsx("h2", { className: "text-lg font-semibold text-white", children: "Configuration" }), _jsxs("div", { className: "space-y-3", children: [_jsxs("label", { className: "flex items-center gap-3 cursor-pointer", children: [_jsx("input", { type: "checkbox", checked: groupByStep, onChange: (e) => setGroupByStep(e.target.checked), className: "w-4 h-4 rounded border-slate-600 text-blue-600 focus:ring-blue-600 bg-slate-800" }), _jsx("span", { className: "text-slate-300", children: "Group by Step" }), _jsxs("span", { className: "text-slate-500 text-sm", children: ["(Currently ", groupByStep ? 'enabled' : 'disabled', ")"] })] }), lastCopiedId && (_jsxs("div", { className: "p-3 bg-emerald-900 border border-emerald-700 rounded text-emerald-100 text-sm", children: ["\u2713 Copied: ", lastCopiedId] }))] }), _jsxs("div", { className: "pt-3 border-t border-slate-800 space-y-2", children: [_jsx("h3", { className: "text-sm font-semibold text-slate-400 uppercase", children: "Features Demonstrated" }), _jsxs("ul", { className: "space-y-1 text-sm text-slate-400", children: [_jsx("li", { children: "\u2713 Heat map coloring based on relevance (emerald/blue/slate)" }), _jsx("li", { children: "\u2713 Source type badges (\uD83D\uDD17 graph, \uD83D\uDCCA vector, \u26A1 cache, \uD83D\uDD25 hot)" }), _jsx("li", { children: "\u2713 Expandable nodes with full content and metadata" }), _jsx("li", { children: "\u2713 Sortable by relevance, recency, or access count" }), _jsx("li", { children: "\u2713 Search/filter by node ID or content" }), _jsx("li", { children: "\u2713 Click-to-copy node IDs with feedback" }), _jsx("li", { children: "\u2713 Grouping by execution step" }), _jsx("li", { children: "\u2713 Access count and recency tracking" }), _jsx("li", { children: "\u2713 Relevance score visualization with progress bar" })] })] })] }), _jsxs("div", { className: "bg-slate-900 border border-slate-800 rounded-lg p-6 space-y-4", children: [_jsx("h2", { className: "text-lg font-semibold text-white", children: "Memory Nodes" }), _jsxs("p", { className: "text-slate-400 text-sm", children: [mockNodes.length, " memory nodes accessed during thread execution", groupByStep && ' (grouped by step)'] }), _jsx(MemoryNodes, { nodes: mockNodes, groupByStep: groupByStep, onNodeClick: handleNodeClick, className: "mt-4" })] }), _jsxs("div", { className: "bg-slate-900 border border-slate-800 rounded-lg p-6 space-y-4", children: [_jsx("h2", { className: "text-lg font-semibold text-white", children: "Component Documentation" }), _jsxs("div", { className: "space-y-6 text-sm text-slate-300", children: [_jsxs("div", { className: "space-y-2", children: [_jsx("h3", { className: "font-semibold text-slate-200", children: "Interface" }), _jsx("pre", { className: "bg-slate-950 p-4 rounded border border-slate-800 overflow-x-auto text-xs font-mono", children: `interface MemoryNode {
  id: string;                              // Unique node identifier
  content: string;                         // Node content/text
  relevance: number;                       // 0.0-1.0 relevance score
  sourceType: 'graph' | 'vector'           // Where memory came from
           | 'cache' | 'hot_pattern';
  accessedAt: string;                      // ISO timestamp
  stepId: string;                          // Execution step reference
  metadata?: Record<string, unknown>;      // Optional extra data
}

interface MemoryNodesProps {
  nodes: MemoryNode[];                     // Array of nodes to display
  groupByStep?: boolean;                   // Group by step (default: false)
  onNodeClick?: (nodeId: string) => void;  // Callback when copying ID
  className?: string;                      // CSS class
}` })] }), _jsxs("div", { className: "space-y-2", children: [_jsx("h3", { className: "font-semibold text-slate-200", children: "Features" }), _jsxs("div", { className: "space-y-2", children: [_jsxs("div", { children: [_jsx("h4", { className: "font-medium text-slate-300", children: "Heat Map Coloring" }), _jsx("p", { className: "text-slate-400", children: "Relevance-based coloring: emerald (0.9+), blue (0.7-0.9), slate (<0.7)" })] }), _jsxs("div", { children: [_jsx("h4", { className: "font-medium text-slate-300", children: "Source Badges" }), _jsx("p", { className: "text-slate-400", children: "Visual indicators for memory source: \uD83D\uDD17 Graph, \uD83D\uDCCA Vector, \u26A1 Cache, \uD83D\uDD25 Hot Pattern" })] }), _jsxs("div", { children: [_jsx("h4", { className: "font-medium text-slate-300", children: "Sorting Options" }), _jsx("p", { className: "text-slate-400", children: "Sort by relevance (descending), recency (newest first), or access count (highest first)" })] }), _jsxs("div", { children: [_jsx("h4", { className: "font-medium text-slate-300", children: "Search/Filter" }), _jsx("p", { className: "text-slate-400", children: "Real-time search by node ID or content substring" })] }), _jsxs("div", { children: [_jsx("h4", { className: "font-medium text-slate-300", children: "Expandable Nodes" }), _jsx("p", { className: "text-slate-400", children: "Click to expand for full content, metadata, and node details" })] }), _jsxs("div", { children: [_jsx("h4", { className: "font-medium text-slate-300", children: "Click-to-Copy" }), _jsx("p", { className: "text-slate-400", children: "Copy node ID to clipboard with visual feedback confirmation" })] }), _jsxs("div", { children: [_jsx("h4", { className: "font-medium text-slate-300", children: "Grouping" }), _jsx("p", { className: "text-slate-400", children: "Optional grouping by execution step for organization" })] })] })] }), _jsxs("div", { className: "space-y-2", children: [_jsx("h3", { className: "font-semibold text-slate-200", children: "Basic Usage" }), _jsx("pre", { className: "bg-slate-950 p-4 rounded border border-slate-800 overflow-x-auto text-xs font-mono", children: `import { MemoryNodes } from '@/components/DetailPanel';

function MyComponent() {
  const nodes: MemoryNode[] = [
    {
      id: 'node-0001-abc123',
      content: 'Thompson Sampling balances...',
      relevance: 0.92,
      sourceType: 'vector',
      accessedAt: '2025-12-11T10:30:00Z',
      stepId: 'step-01',
      metadata: { access_count: 3 },
    },
    // ... more nodes
  ];

  const handleNodeClick = (nodeId: string) => {
    console.log('Copied:', nodeId);
  };

  return (
    <MemoryNodes
      nodes={nodes}
      groupByStep={true}
      onNodeClick={handleNodeClick}
    />
  );
}` })] }), _jsxs("div", { className: "space-y-2", children: [_jsx("h3", { className: "font-semibold text-slate-200", children: "Styling" }), _jsxs("ul", { className: "space-y-1 text-slate-400 list-disc list-inside", children: [_jsx("li", { children: "Dark theme (slate/blue/emerald/orange color palette)" }), _jsx("li", { children: "Responsive grid layout (1 column on mobile, 2 on large screens)" }), _jsx("li", { children: "Hover effects and transitions for interactivity" }), _jsx("li", { children: "Monospace fonts for IDs and code content" }), _jsx("li", { children: "Compact card design with visual hierarchy" }), _jsx("li", { children: "Progress bar for relevance visualization" })] })] }), _jsxs("div", { className: "space-y-2", children: [_jsx("h3", { className: "font-semibold text-slate-200", children: "Accessibility" }), _jsxs("ul", { className: "space-y-1 text-slate-400 list-disc list-inside", children: [_jsx("li", { children: "Semantic HTML structure" }), _jsx("li", { children: "Proper button and input elements" }), _jsx("li", { children: "Title attributes for tooltips" }), _jsx("li", { children: "Keyboard navigation support" }), _jsx("li", { children: "Color-independent information (not relying on color alone)" }), _jsx("li", { children: "High contrast dark theme" })] })] })] })] }), _jsxs("div", { className: "bg-slate-900 border border-slate-800 rounded-lg p-6 space-y-4", children: [_jsx("h2", { className: "text-lg font-semibold text-white", children: "Generated Sample Data" }), _jsxs("div", { className: "grid grid-cols-2 gap-4 text-sm text-slate-400", children: [_jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Total Nodes:" }), _jsx("span", { className: "ml-2 text-slate-300 font-mono", children: mockNodes.length })] }), _jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Average Relevance:" }), _jsx("span", { className: "ml-2 text-slate-300 font-mono", children: (mockNodes.reduce((sum, n) => sum + n.relevance, 0) / mockNodes.length).toFixed(2) })] }), _jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Unique Steps:" }), _jsx("span", { className: "ml-2 text-slate-300 font-mono", children: new Set(mockNodes.map((n) => n.stepId)).size })] }), _jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Source Types:" }), _jsx("span", { className: "ml-2 text-slate-300 font-mono", children: new Set(mockNodes.map((n) => n.sourceType)).size })] })] })] })] }));
};
export default MemoryNodesDemo;
//# sourceMappingURL=MemoryNodes.demo.js.map