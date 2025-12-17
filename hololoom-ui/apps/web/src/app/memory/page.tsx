'use client';

import { useState } from 'react';
import { Navigation } from '../../components/Navigation';
import { Button, Badge } from '@hololoom/design-system';
import {
  MemorySearch,
  MemoryGraph,
  MemoryTimeline,
  MemoryStats,
  MemoryDetail,
} from '../../components/memory';

type ViewMode = 'graph' | 'timeline' | 'list';

export default function MemoryPage() {
  const [viewMode, setViewMode] = useState<ViewMode>('graph');
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  return (
    <div className="min-h-screen bg-bg-primary">
      <Navigation />

      <main className="main-content py-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold text-fg-primary mb-2">
              Memory Explorer
            </h1>
            <p className="text-fg-secondary">
              Navigate and explore your knowledge graph with multi-scale memory retrieval
            </p>
          </div>

          {/* View Mode Toggles */}
          <div className="flex items-center gap-2 bg-bg-secondary rounded-lg p-1">
            <ViewModeButton
              mode="graph"
              currentMode={viewMode}
              onClick={() => setViewMode('graph')}
              icon={<GraphIcon />}
              label="Graph"
            />
            <ViewModeButton
              mode="timeline"
              currentMode={viewMode}
              onClick={() => setViewMode('timeline')}
              icon={<TimelineIcon />}
              label="Timeline"
            />
            <ViewModeButton
              mode="list"
              currentMode={viewMode}
              onClick={() => setViewMode('list')}
              icon={<ListIcon />}
              label="List"
            />
          </div>
        </div>

        {/* Search Bar */}
        <MemorySearch
          value={searchQuery}
          onChange={setSearchQuery}
          onSearch={(q) => console.log('Search:', q)}
        />

        {/* Stats Overview */}
        <MemoryStats className="my-6" />

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main View (2/3 width on large screens) */}
          <div className="lg:col-span-2">
            {viewMode === 'graph' && (
              <MemoryGraph
                searchQuery={searchQuery}
                onNodeSelect={setSelectedMemoryId}
                selectedNodeId={selectedMemoryId}
              />
            )}
            {viewMode === 'timeline' && (
              <MemoryTimeline
                searchQuery={searchQuery}
                onMemorySelect={setSelectedMemoryId}
                selectedMemoryId={selectedMemoryId}
              />
            )}
            {viewMode === 'list' && (
              <MemoryListView
                searchQuery={searchQuery}
                onMemorySelect={setSelectedMemoryId}
                selectedMemoryId={selectedMemoryId}
              />
            )}
          </div>

          {/* Detail Panel (1/3 width on large screens) */}
          <div className="lg:col-span-1">
            <MemoryDetail
              memoryId={selectedMemoryId}
              onClose={() => setSelectedMemoryId(null)}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

// View mode toggle button
function ViewModeButton({
  mode,
  currentMode,
  onClick,
  icon,
  label,
}: {
  mode: ViewMode;
  currentMode: ViewMode;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  const isActive = mode === currentMode;

  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
        isActive
          ? 'bg-cosmic-nebula/20 text-cosmic-nebula'
          : 'text-fg-tertiary hover:text-fg-secondary hover:bg-bg-tertiary'
      }`}
    >
      {icon}
      <span className="hidden sm:inline">{label}</span>
    </button>
  );
}

// Simple list view component (inline for now)
function MemoryListView({
  searchQuery,
  onMemorySelect,
  selectedMemoryId,
}: {
  searchQuery: string;
  onMemorySelect: (id: string) => void;
  selectedMemoryId: string | null;
}) {
  // Mock memories for demonstration
  const memories = generateMockMemories(20).filter(
    (m) =>
      !searchQuery ||
      m.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
      m.entities.some((e) => e.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="bg-bg-primary border border-border-primary rounded-xl">
      <div className="p-4 border-b border-border-primary">
        <h3 className="text-lg font-semibold text-fg-primary">
          Memory List
        </h3>
        <p className="text-sm text-fg-tertiary">
          {memories.length} memories {searchQuery && `matching "${searchQuery}"`}
        </p>
      </div>
      <div className="divide-y divide-border-primary max-h-[600px] overflow-y-auto">
        {memories.map((memory) => (
          <button
            key={memory.id}
            onClick={() => onMemorySelect(memory.id)}
            className={`w-full text-left p-4 hover:bg-bg-secondary/50 transition-colors ${
              selectedMemoryId === memory.id ? 'bg-cosmic-nebula/10' : ''
            }`}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 min-w-0">
                <p className="text-sm text-fg-primary line-clamp-2">
                  {memory.content}
                </p>
                <div className="flex flex-wrap gap-1 mt-2">
                  {memory.entities.slice(0, 3).map((entity) => (
                    <Badge key={entity} variant="secondary" size="sm">
                      {entity}
                    </Badge>
                  ))}
                  {memory.entities.length > 3 && (
                    <Badge variant="secondary" size="sm">
                      +{memory.entities.length - 3}
                    </Badge>
                  )}
                </div>
              </div>
              <div className="text-right flex-shrink-0">
                <p className="text-xs text-fg-tertiary">
                  {formatRelativeTime(memory.timestamp)}
                </p>
                <p className="text-xs text-fg-tertiary mt-1">
                  {(memory.relevance * 100).toFixed(0)}% relevant
                </p>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

// Mock data generator
function generateMockMemories(count: number) {
  const contents = [
    'Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration and exploitation.',
    'Neural networks learn hierarchical representations through backpropagation of gradients.',
    'The attention mechanism allows models to focus on relevant parts of the input sequence.',
    'Matryoshka embeddings provide multi-scale semantic representations at different dimensionalities.',
    'Knowledge graphs represent entities and relationships as nodes and edges.',
    'Reinforcement learning agents learn through trial and error with reward signals.',
    'The transformer architecture revolutionized NLP with self-attention mechanisms.',
    'PPO is a policy gradient method that uses clipped objective functions for stable training.',
    'Retrieval-augmented generation combines LLMs with external knowledge retrieval.',
    'Graph neural networks propagate information across graph structures.',
  ];

  const entityPools = [
    ['Thompson Sampling', 'Bayesian', 'Multi-Armed Bandit', 'Exploration'],
    ['Neural Networks', 'Backpropagation', 'Gradients', 'Deep Learning'],
    ['Attention', 'Transformer', 'Sequence', 'NLP'],
    ['Matryoshka', 'Embeddings', 'Multi-Scale', 'Semantic'],
    ['Knowledge Graph', 'Entities', 'Relationships', 'Graph'],
    ['Reinforcement Learning', 'Agent', 'Reward', 'Policy'],
    ['Transformer', 'Self-Attention', 'NLP', 'Architecture'],
    ['PPO', 'Policy Gradient', 'Clipping', 'Training'],
    ['RAG', 'LLM', 'Retrieval', 'Generation'],
    ['GNN', 'Graph', 'Propagation', 'Neural'],
  ];

  return Array.from({ length: count }, (_, i) => ({
    id: `memory-${i}`,
    content: contents[i % contents.length],
    entities: entityPools[i % entityPools.length],
    timestamp: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000),
    relevance: 0.5 + Math.random() * 0.5,
    type: (['episodic', 'semantic', 'procedural'] as const)[i % 3],
  }));
}

function formatRelativeTime(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  return `${diffDays}d ago`;
}

// Icons
function GraphIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="3" />
      <circle cx="19" cy="5" r="2" />
      <circle cx="5" cy="19" r="2" />
      <circle cx="5" cy="5" r="2" />
      <circle cx="19" cy="19" r="2" />
      <path d="M12 9V4.5" />
      <path d="m7.5 7.5 2.5 2.5" />
      <path d="M9 12H4.5" />
      <path d="m7.5 16.5 2.5-2.5" />
      <path d="M12 15v4.5" />
      <path d="m16.5 16.5-2.5-2.5" />
      <path d="M15 12h4.5" />
      <path d="m16.5 7.5-2.5 2.5" />
    </svg>
  );
}

function TimelineIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 2v20" />
      <path d="m19 15-7 7-7-7" />
      <path d="M5 9h14" />
    </svg>
  );
}

function ListIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="8" x2="21" y1="6" y2="6" />
      <line x1="8" x2="21" y1="12" y2="12" />
      <line x1="8" x2="21" y1="18" y2="18" />
      <line x1="3" x2="3.01" y1="6" y2="6" />
      <line x1="3" x2="3.01" y1="12" y2="12" />
      <line x1="3" x2="3.01" y1="18" y2="18" />
    </svg>
  );
}
