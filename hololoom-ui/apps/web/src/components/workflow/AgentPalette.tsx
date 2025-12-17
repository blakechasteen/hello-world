'use client';

import { useState } from 'react';
import { Badge } from '@hololoom/design-system';
import type { AgentType } from '../../app/workflow/page';

interface AgentPaletteProps {
  onDragStart: (type: AgentType, x: number, y: number) => void;
}

interface AgentCategory {
  name: string;
  color: string;
  agents: Array<{
    type: AgentType;
    label: string;
    description: string;
    icon: React.ReactNode;
  }>;
}

const agentCategories: AgentCategory[] = [
  {
    name: 'Query',
    color: 'cosmic-nebula',
    agents: [
      {
        type: 'hololoom_query',
        label: 'HoloLoom Query',
        description: 'Full weaving cycle query',
        icon: <QueryIcon />,
      },
      {
        type: 'memory_search',
        label: 'Memory Search',
        description: 'Knowledge graph search',
        icon: <SearchIcon />,
      },
      {
        type: 'multi_query',
        label: 'Multi-Query',
        description: 'Break into sub-questions',
        icon: <MultiQueryIcon />,
      },
    ],
  },
  {
    name: 'Processing',
    color: 'cosmic-aurora',
    agents: [
      {
        type: 'matryoshka_embedder',
        label: 'Matryoshka Embedder',
        description: 'Multi-scale embeddings',
        icon: <EmbedIcon />,
      },
      {
        type: 'synthesizer',
        label: 'Synthesizer',
        description: 'Extract entities/motifs',
        icon: <SynthIcon />,
      },
      {
        type: 'recursive_refiner',
        label: 'Recursive Refiner',
        description: 'Quality refinement',
        icon: <RefineIcon />,
      },
    ],
  },
  {
    name: 'Memory',
    color: 'cosmic-starlight',
    agents: [
      {
        type: 'memory_store',
        label: 'Memory Store',
        description: 'Persist to graph+vector',
        icon: <StoreIcon />,
      },
      {
        type: 'context_retriever',
        label: 'Context Retriever',
        description: 'Retrieve context',
        icon: <RetrieveIcon />,
      },
      {
        type: 'knowledge_fusion',
        label: 'Knowledge Fusion',
        description: 'Multi-hop traversal',
        icon: <FusionIcon />,
      },
    ],
  },
  {
    name: 'Decision',
    color: 'confidence-high',
    agents: [
      {
        type: 'thompson_sampler',
        label: 'Thompson Sampler',
        description: 'Bayesian exploration',
        icon: <ThompsonIcon />,
      },
      {
        type: 'convergence_engine',
        label: 'Convergence Engine',
        description: 'Decision collapse',
        icon: <ConvergeIcon />,
      },
      {
        type: 'safety_guardrails',
        label: 'Safety Guardrails',
        description: 'Risk gating',
        icon: <SafetyIcon />,
      },
    ],
  },
  {
    name: 'Output',
    color: 'confidence-medium',
    agents: [
      {
        type: 'response_generator',
        label: 'Response Generator',
        description: 'Generate response',
        icon: <ResponseIcon />,
      },
      {
        type: 'format_converter',
        label: 'Format Converter',
        description: 'JSON/Markdown/HTML',
        icon: <FormatIcon />,
      },
    ],
  },
  {
    name: 'Control Flow',
    color: 'fg-tertiary',
    agents: [
      {
        type: 'conditional_branch',
        label: 'Conditional',
        description: 'If/else logic',
        icon: <BranchIcon />,
      },
      {
        type: 'loop_iterator',
        label: 'Loop',
        description: 'Repeat until condition',
        icon: <LoopIcon />,
      },
      {
        type: 'parallel_executor',
        label: 'Parallel',
        description: 'Concurrent execution',
        icon: <ParallelIcon />,
      },
      {
        type: 'aggregator',
        label: 'Aggregator',
        description: 'Merge multiple inputs',
        icon: <AggregateIcon />,
      },
    ],
  },
];

export function AgentPalette({ onDragStart }: AgentPaletteProps) {
  const [expandedCategory, setExpandedCategory] = useState<string | null>('Query');
  const [searchQuery, setSearchQuery] = useState('');

  const filteredCategories = searchQuery
    ? agentCategories.map((cat) => ({
        ...cat,
        agents: cat.agents.filter(
          (a) =>
            a.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
            a.description.toLowerCase().includes(searchQuery.toLowerCase())
        ),
      })).filter((cat) => cat.agents.length > 0)
    : agentCategories;

  const handleDragStart = (e: React.DragEvent, type: AgentType) => {
    e.dataTransfer.setData('agentType', type);
    e.dataTransfer.effectAllowed = 'copy';
  };

  return (
    <div className="w-64 flex-shrink-0 border-r border-border-primary bg-bg-primary overflow-y-auto">
      <div className="p-3 border-b border-border-primary">
        <h3 className="text-sm font-medium text-fg-primary mb-2">Agent Library</h3>
        <div className="relative">
          <SearchInputIcon className="absolute left-2.5 top-1/2 -translate-y-1/2 text-fg-tertiary" />
          <input
            type="text"
            placeholder="Search agents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 text-sm bg-bg-secondary border border-border-primary rounded-lg text-fg-primary placeholder:text-fg-tertiary focus:outline-none focus:ring-1 focus:ring-cosmic-nebula"
          />
        </div>
      </div>

      <div className="p-2">
        {filteredCategories.map((category) => (
          <div key={category.name} className="mb-1">
            <button
              onClick={() =>
                setExpandedCategory(
                  expandedCategory === category.name ? null : category.name
                )
              }
              className="w-full flex items-center justify-between px-2 py-1.5 text-sm font-medium text-fg-secondary hover:text-fg-primary hover:bg-bg-secondary rounded-lg transition-colors"
            >
              <span className="flex items-center gap-2">
                <span
                  className={`w-2 h-2 rounded-full bg-${category.color}`}
                />
                {category.name}
              </span>
              <ChevronIcon
                className={`w-4 h-4 transition-transform ${
                  expandedCategory === category.name ? 'rotate-90' : ''
                }`}
              />
            </button>

            {(expandedCategory === category.name || searchQuery) && (
              <div className="mt-1 space-y-1 pl-2">
                {category.agents.map((agent) => (
                  <div
                    key={agent.type}
                    draggable
                    onDragStart={(e) => handleDragStart(e, agent.type)}
                    className="flex items-start gap-2 p-2 bg-bg-secondary hover:bg-bg-tertiary rounded-lg cursor-grab active:cursor-grabbing transition-colors group"
                  >
                    <div className={`flex-shrink-0 w-8 h-8 rounded-lg bg-${category.color}/10 text-${category.color} flex items-center justify-center`}>
                      {agent.icon}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-fg-primary truncate">
                        {agent.label}
                      </p>
                      <p className="text-xs text-fg-tertiary truncate">
                        {agent.description}
                      </p>
                    </div>
                    <div className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                      <DragHandleIcon className="w-4 h-4 text-fg-tertiary" />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="p-3 border-t border-border-primary">
        <p className="text-xs text-fg-tertiary">
          Drag agents onto the canvas to build your workflow
        </p>
      </div>
    </div>
  );
}

// Icons
function SearchInputIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="m9 18 6-6-6-6" />
    </svg>
  );
}

function DragHandleIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <circle cx="9" cy="5" r="1" />
      <circle cx="9" cy="12" r="1" />
      <circle cx="9" cy="19" r="1" />
      <circle cx="15" cy="5" r="1" />
      <circle cx="15" cy="12" r="1" />
      <circle cx="15" cy="19" r="1" />
    </svg>
  );
}

// Agent Icons
function QueryIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
      <path d="M12 17h.01" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

function MultiQueryIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 3H5a2 2 0 0 0-2 2v3" />
      <path d="M21 8V5a2 2 0 0 0-2-2h-3" />
      <path d="M3 16v3a2 2 0 0 0 2 2h3" />
      <path d="M16 21h3a2 2 0 0 0 2-2v-3" />
    </svg>
  );
}

function EmbedIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <circle cx="12" cy="12" r="7" />
      <circle cx="12" cy="12" r="10" />
    </svg>
  );
}

function SynthIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2" />
      <path d="M8.5 2h7" />
    </svg>
  );
}

function RefineIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
      <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
      <path d="M16 16h5v5" />
    </svg>
  );
}

function StoreIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5V19A9 3 0 0 0 21 19V5" />
      <path d="M3 12A9 3 0 0 0 21 12" />
    </svg>
  );
}

function RetrieveIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" x2="12" y1="3" y2="15" />
    </svg>
  );
}

function FusionIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v4" />
      <path d="M12 18v4" />
      <path d="m4.93 4.93 2.83 2.83" />
      <path d="m16.24 16.24 2.83 2.83" />
      <path d="M2 12h4" />
      <path d="M18 12h4" />
      <path d="m4.93 19.07 2.83-2.83" />
      <path d="m16.24 7.76 2.83-2.83" />
    </svg>
  );
}

function ThompsonIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 12h6" />
      <path d="M22 12h-6" />
      <path d="M12 2v6" />
      <path d="M12 22v-6" />
      <circle cx="12" cy="12" r="4" />
    </svg>
  );
}

function ConvergeIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m5 12 7-7 7 7" />
      <path d="m5 19 7-7 7 7" />
    </svg>
  );
}

function SafetyIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
}

function ResponseIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function FormatIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 7V4h16v3" />
      <path d="M9 20h6" />
      <path d="M12 4v16" />
    </svg>
  );
}

function BranchIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="6" x2="6" y1="3" y2="15" />
      <circle cx="18" cy="6" r="3" />
      <circle cx="6" cy="18" r="3" />
      <path d="M18 9a9 9 0 0 1-9 9" />
    </svg>
  );
}

function LoopIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m17 2 4 4-4 4" />
      <path d="M3 11v-1a4 4 0 0 1 4-4h14" />
      <path d="m7 22-4-4 4-4" />
      <path d="M21 13v1a4 4 0 0 1-4 4H3" />
    </svg>
  );
}

function ParallelIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect width="6" height="14" x="2" y="5" rx="2" />
      <rect width="6" height="10" x="16" y="7" rx="2" />
      <rect width="6" height="6" x="9" y="9" rx="2" />
    </svg>
  );
}

function AggregateIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 8V5a2 2 0 0 1 2-2h4" />
      <path d="M3 16v3a2 2 0 0 0 2 2h4" />
      <path d="M21 8V5a2 2 0 0 0-2-2h-4" />
      <path d="M21 16v3a2 2 0 0 1-2 2h-4" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}
