import React, { useState } from 'react';
import { MemoryNodes } from './MemoryNodes';

/**
 * Mock data generator for demo purposes
 */
const generateMockMemoryNodes = (count: number = 12) => {
  const sourceTypes = ['graph', 'vector', 'cache', 'hot_pattern'] as const;
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
export const MemoryNodesDemo: React.FC = () => {
  const [mockNodes] = useState(() => generateMockMemoryNodes(15));
  const [groupByStep, setGroupByStep] = useState(false);
  const [lastCopiedId, setLastCopiedId] = useState<string | null>(null);

  const handleNodeClick = (nodeId: string) => {
    setLastCopiedId(nodeId);
    setTimeout(() => setLastCopiedId(null), 2000);
  };

  return (
    <div className="min-h-screen bg-slate-950 p-8 space-y-8">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-bold text-white">MemoryNodes Component Demo</h1>
        <p className="text-slate-400">
          Comprehensive demonstration of the MemoryNodes component with all features
        </p>
      </div>

      {/* Configuration Panel */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 space-y-4">
        <h2 className="text-lg font-semibold text-white">Configuration</h2>

        <div className="space-y-3">
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={groupByStep}
              onChange={(e) => setGroupByStep(e.target.checked)}
              className="w-4 h-4 rounded border-slate-600 text-blue-600 focus:ring-blue-600 bg-slate-800"
            />
            <span className="text-slate-300">Group by Step</span>
            <span className="text-slate-500 text-sm">
              (Currently {groupByStep ? 'enabled' : 'disabled'})
            </span>
          </label>

          {lastCopiedId && (
            <div className="p-3 bg-emerald-900 border border-emerald-700 rounded text-emerald-100 text-sm">
              ✓ Copied: {lastCopiedId}
            </div>
          )}
        </div>

        <div className="pt-3 border-t border-slate-800 space-y-2">
          <h3 className="text-sm font-semibold text-slate-400 uppercase">
            Features Demonstrated
          </h3>
          <ul className="space-y-1 text-sm text-slate-400">
            <li>✓ Heat map coloring based on relevance (emerald/blue/slate)</li>
            <li>✓ Source type badges (🔗 graph, 📊 vector, ⚡ cache, 🔥 hot)</li>
            <li>✓ Expandable nodes with full content and metadata</li>
            <li>✓ Sortable by relevance, recency, or access count</li>
            <li>✓ Search/filter by node ID or content</li>
            <li>✓ Click-to-copy node IDs with feedback</li>
            <li>✓ Grouping by execution step</li>
            <li>✓ Access count and recency tracking</li>
            <li>✓ Relevance score visualization with progress bar</li>
          </ul>
        </div>
      </div>

      {/* Main Demo */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 space-y-4">
        <h2 className="text-lg font-semibold text-white">Memory Nodes</h2>
        <p className="text-slate-400 text-sm">
          {mockNodes.length} memory nodes accessed during thread execution
          {groupByStep && ' (grouped by step)'}
        </p>

        {/* Component */}
        <MemoryNodes
          nodes={mockNodes}
          groupByStep={groupByStep}
          onNodeClick={handleNodeClick}
          className="mt-4"
        />
      </div>

      {/* Documentation */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 space-y-4">
        <h2 className="text-lg font-semibold text-white">Component Documentation</h2>

        <div className="space-y-6 text-sm text-slate-300">
          {/* Interface Section */}
          <div className="space-y-2">
            <h3 className="font-semibold text-slate-200">Interface</h3>
            <pre className="bg-slate-950 p-4 rounded border border-slate-800 overflow-x-auto text-xs font-mono">
{`interface MemoryNode {
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
}`}
            </pre>
          </div>

          {/* Features Section */}
          <div className="space-y-2">
            <h3 className="font-semibold text-slate-200">Features</h3>
            <div className="space-y-2">
              <div>
                <h4 className="font-medium text-slate-300">Heat Map Coloring</h4>
                <p className="text-slate-400">
                  Relevance-based coloring: emerald (0.9+), blue (0.7-0.9), slate (&lt;0.7)
                </p>
              </div>
              <div>
                <h4 className="font-medium text-slate-300">Source Badges</h4>
                <p className="text-slate-400">
                  Visual indicators for memory source: 🔗 Graph, 📊 Vector, ⚡ Cache, 🔥 Hot Pattern
                </p>
              </div>
              <div>
                <h4 className="font-medium text-slate-300">Sorting Options</h4>
                <p className="text-slate-400">
                  Sort by relevance (descending), recency (newest first), or access count (highest first)
                </p>
              </div>
              <div>
                <h4 className="font-medium text-slate-300">Search/Filter</h4>
                <p className="text-slate-400">
                  Real-time search by node ID or content substring
                </p>
              </div>
              <div>
                <h4 className="font-medium text-slate-300">Expandable Nodes</h4>
                <p className="text-slate-400">
                  Click to expand for full content, metadata, and node details
                </p>
              </div>
              <div>
                <h4 className="font-medium text-slate-300">Click-to-Copy</h4>
                <p className="text-slate-400">
                  Copy node ID to clipboard with visual feedback confirmation
                </p>
              </div>
              <div>
                <h4 className="font-medium text-slate-300">Grouping</h4>
                <p className="text-slate-400">
                  Optional grouping by execution step for organization
                </p>
              </div>
            </div>
          </div>

          {/* Usage Section */}
          <div className="space-y-2">
            <h3 className="font-semibold text-slate-200">Basic Usage</h3>
            <pre className="bg-slate-950 p-4 rounded border border-slate-800 overflow-x-auto text-xs font-mono">
{`import { MemoryNodes } from '@/components/DetailPanel';

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
}`}
            </pre>
          </div>

          {/* Styling Section */}
          <div className="space-y-2">
            <h3 className="font-semibold text-slate-200">Styling</h3>
            <ul className="space-y-1 text-slate-400 list-disc list-inside">
              <li>Dark theme (slate/blue/emerald/orange color palette)</li>
              <li>Responsive grid layout (1 column on mobile, 2 on large screens)</li>
              <li>Hover effects and transitions for interactivity</li>
              <li>Monospace fonts for IDs and code content</li>
              <li>Compact card design with visual hierarchy</li>
              <li>Progress bar for relevance visualization</li>
            </ul>
          </div>

          {/* Accessibility Section */}
          <div className="space-y-2">
            <h3 className="font-semibold text-slate-200">Accessibility</h3>
            <ul className="space-y-1 text-slate-400 list-disc list-inside">
              <li>Semantic HTML structure</li>
              <li>Proper button and input elements</li>
              <li>Title attributes for tooltips</li>
              <li>Keyboard navigation support</li>
              <li>Color-independent information (not relying on color alone)</li>
              <li>High contrast dark theme</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Sample Data Info */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 space-y-4">
        <h2 className="text-lg font-semibold text-white">Generated Sample Data</h2>
        <div className="grid grid-cols-2 gap-4 text-sm text-slate-400">
          <div>
            <span className="text-slate-500">Total Nodes:</span>
            <span className="ml-2 text-slate-300 font-mono">{mockNodes.length}</span>
          </div>
          <div>
            <span className="text-slate-500">Average Relevance:</span>
            <span className="ml-2 text-slate-300 font-mono">
              {(
                mockNodes.reduce((sum, n) => sum + n.relevance, 0) / mockNodes.length
              ).toFixed(2)}
            </span>
          </div>
          <div>
            <span className="text-slate-500">Unique Steps:</span>
            <span className="ml-2 text-slate-300 font-mono">
              {new Set(mockNodes.map((n) => n.stepId)).size}
            </span>
          </div>
          <div>
            <span className="text-slate-500">Source Types:</span>
            <span className="ml-2 text-slate-300 font-mono">
              {new Set(mockNodes.map((n) => n.sourceType)).size}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MemoryNodesDemo;
