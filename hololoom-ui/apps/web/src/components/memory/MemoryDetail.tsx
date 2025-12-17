'use client';

import { useState, useEffect } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Button,
  Badge,
  ConfidenceIndicator,
  SafetyBadge,
} from '@hololoom/design-system';
import type { SafetyLevel } from '@hololoom/api-client';

interface MemoryDetailProps {
  memoryId: string | null;
  onClose: () => void;
}

interface MemoryInfo {
  id: string;
  content: string;
  type: 'episodic' | 'semantic' | 'procedural';
  timestamp: Date;
  lastAccessed: Date;
  accessCount: number;
  activation: number;
  heat: number;
  entities: Array<{ name: string; type: string }>;
  relationships: Array<{ type: string; target: string; weight: number }>;
  consolidation: {
    stage: 'recent' | 'consolidating' | 'consolidated';
    progress: number;
  };
  provenance: {
    source: string;
    confidence: number;
    safetyLevel: SafetyLevel;
  };
  embedding: {
    dimensions: number;
    scale: '384D' | '256D' | '128D';
  };
}

export function MemoryDetail({ memoryId, onClose }: MemoryDetailProps) {
  const [memory, setMemory] = useState<MemoryInfo | null>(null);
  const [activeTab, setActiveTab] = useState<'details' | 'connections' | 'provenance'>('details');

  // Load memory details
  useEffect(() => {
    if (!memoryId) {
      setMemory(null);
      return;
    }

    // Mock data - would fetch from API
    const mockMemory: MemoryInfo = {
      id: memoryId,
      content: 'Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration and exploitation by sampling from posterior distributions.',
      type: 'semantic',
      timestamp: new Date(Date.now() - 3600000 * 24),
      lastAccessed: new Date(Date.now() - 3600000 * 2),
      accessCount: 47,
      activation: 0.85,
      heat: 0.72,
      entities: [
        { name: 'Thompson Sampling', type: 'Algorithm' },
        { name: 'Bayesian', type: 'Paradigm' },
        { name: 'Multi-Armed Bandit', type: 'Problem' },
        { name: 'Exploration', type: 'Concept' },
        { name: 'Exploitation', type: 'Concept' },
      ],
      relationships: [
        { type: 'IS_A', target: 'Bandit Algorithm', weight: 0.95 },
        { type: 'USES', target: 'Beta Distribution', weight: 0.88 },
        { type: 'SOLVES', target: 'Exploration-Exploitation Tradeoff', weight: 0.92 },
        { type: 'RELATED_TO', target: 'Reinforcement Learning', weight: 0.75 },
      ],
      consolidation: {
        stage: 'consolidated',
        progress: 1.0,
      },
      provenance: {
        source: 'RAG Ingestion',
        confidence: 0.92,
        safetyLevel: 'safe',
      },
      embedding: {
        dimensions: 384,
        scale: '384D',
      },
    };

    setMemory(mockMemory);
    setActiveTab('details');
  }, [memoryId]);

  if (!memoryId || !memory) {
    return (
      <Card className="h-full">
        <CardBody className="h-full flex items-center justify-center">
          <div className="text-center">
            <EmptyStateIcon className="w-16 h-16 mx-auto text-fg-tertiary/50 mb-4" />
            <p className="text-fg-tertiary">Select a memory to view details</p>
          </div>
        </CardBody>
      </Card>
    );
  }

  const tabs = [
    { id: 'details', label: 'Details' },
    { id: 'connections', label: 'Connections' },
    { id: 'provenance', label: 'Provenance' },
  ] as const;

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="flex-shrink-0">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <Badge
                variant={
                  memory.type === 'episodic'
                    ? 'default'
                    : memory.type === 'semantic'
                    ? 'secondary'
                    : 'accent'
                }
              >
                {memory.type}
              </Badge>
              <SafetyBadge level={memory.provenance.safetyLevel} size="sm" />
            </div>
            <p className="text-xs text-fg-tertiary truncate">ID: {memory.id}</p>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <CloseIcon />
          </Button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mt-4 p-1 bg-bg-secondary rounded-lg">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                activeTab === tab.id
                  ? 'bg-bg-primary text-fg-primary shadow-sm'
                  : 'text-fg-tertiary hover:text-fg-secondary'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </CardHeader>

      <CardBody className="flex-1 overflow-y-auto">
        {activeTab === 'details' && (
          <div className="space-y-6">
            {/* Content */}
            <div>
              <h4 className="text-xs font-medium text-fg-tertiary uppercase tracking-wider mb-2">
                Content
              </h4>
              <p className="text-sm text-fg-primary leading-relaxed">
                {memory.content}
              </p>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 gap-4">
              <MetricCard
                label="Activation"
                value={<ConfidenceIndicator value={memory.activation} showLabel />}
              />
              <MetricCard
                label="Heat Score"
                value={
                  <div className="flex items-center gap-2">
                    <span className="text-lg font-semibold text-fg-primary">
                      {(memory.heat * 100).toFixed(0)}%
                    </span>
                    <FireIcon className={memory.heat > 0.5 ? 'text-cosmic-starlight' : 'text-fg-tertiary'} />
                  </div>
                }
              />
              <MetricCard
                label="Access Count"
                value={
                  <span className="text-lg font-semibold text-fg-primary">
                    {memory.accessCount}
                  </span>
                }
              />
              <MetricCard
                label="Embedding"
                value={
                  <Badge variant="secondary">{memory.embedding.scale}</Badge>
                }
              />
            </div>

            {/* Entities */}
            <div>
              <h4 className="text-xs font-medium text-fg-tertiary uppercase tracking-wider mb-2">
                Entities ({memory.entities.length})
              </h4>
              <div className="flex flex-wrap gap-2">
                {memory.entities.map((entity) => (
                  <div
                    key={entity.name}
                    className="flex items-center gap-1.5 px-2 py-1 bg-bg-secondary rounded-lg"
                  >
                    <span className="text-xs text-cosmic-nebula">{entity.type}</span>
                    <span className="text-sm text-fg-primary">{entity.name}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Consolidation */}
            <div>
              <h4 className="text-xs font-medium text-fg-tertiary uppercase tracking-wider mb-2">
                Consolidation
              </h4>
              <div className="flex items-center gap-3">
                <Badge
                  variant={
                    memory.consolidation.stage === 'recent'
                      ? 'success'
                      : memory.consolidation.stage === 'consolidating'
                      ? 'warning'
                      : 'secondary'
                  }
                >
                  {memory.consolidation.stage}
                </Badge>
                <div className="flex-1 h-2 bg-bg-secondary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-cosmic-nebula to-cosmic-aurora transition-all"
                    style={{ width: `${memory.consolidation.progress * 100}%` }}
                  />
                </div>
                <span className="text-xs text-fg-tertiary">
                  {(memory.consolidation.progress * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            {/* Timestamps */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-xs text-fg-tertiary">Created</p>
                <p className="text-fg-secondary">{memory.timestamp.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-xs text-fg-tertiary">Last Accessed</p>
                <p className="text-fg-secondary">{memory.lastAccessed.toLocaleString()}</p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'connections' && (
          <div className="space-y-4">
            <p className="text-sm text-fg-tertiary">
              {memory.relationships.length} connections found
            </p>
            {memory.relationships.map((rel, i) => (
              <div
                key={i}
                className="p-3 bg-bg-secondary rounded-lg hover:bg-bg-tertiary transition-colors cursor-pointer"
              >
                <div className="flex items-center justify-between mb-2">
                  <Badge variant="secondary" size="sm">{rel.type}</Badge>
                  <span className="text-xs text-fg-tertiary">
                    {(rel.weight * 100).toFixed(0)}% weight
                  </span>
                </div>
                <p className="text-sm text-fg-primary">{rel.target}</p>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'provenance' && (
          <div className="space-y-6">
            <div>
              <h4 className="text-xs font-medium text-fg-tertiary uppercase tracking-wider mb-2">
                Source
              </h4>
              <p className="text-sm text-fg-primary">{memory.provenance.source}</p>
            </div>

            <div>
              <h4 className="text-xs font-medium text-fg-tertiary uppercase tracking-wider mb-2">
                Confidence
              </h4>
              <ConfidenceIndicator value={memory.provenance.confidence} showLabel size="lg" />
            </div>

            <div>
              <h4 className="text-xs font-medium text-fg-tertiary uppercase tracking-wider mb-2">
                Safety Assessment
              </h4>
              <SafetyBadge level={memory.provenance.safetyLevel} />
            </div>

            <div>
              <h4 className="text-xs font-medium text-fg-tertiary uppercase tracking-wider mb-2">
                Trace
              </h4>
              <div className="space-y-2 text-xs font-mono bg-bg-secondary p-3 rounded-lg">
                <p className="text-fg-tertiary">Created: {memory.timestamp.toISOString()}</p>
                <p className="text-fg-tertiary">Last accessed: {memory.lastAccessed.toISOString()}</p>
                <p className="text-fg-tertiary">Access count: {memory.accessCount}</p>
                <p className="text-fg-tertiary">Memory ID: {memory.id}</p>
              </div>
            </div>
          </div>
        )}
      </CardBody>

      {/* Actions */}
      <div className="flex-shrink-0 p-4 border-t border-border-primary">
        <div className="flex gap-2">
          <Button variant="secondary" size="sm" className="flex-1">
            <NavigateIcon className="mr-1" />
            Navigate
          </Button>
          <Button variant="secondary" size="sm" className="flex-1">
            <ExportIcon className="mr-1" />
            Export
          </Button>
        </div>
      </div>
    </Card>
  );
}

function MetricCard({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="p-3 bg-bg-secondary rounded-lg">
      <p className="text-xs text-fg-tertiary mb-1">{label}</p>
      {value}
    </div>
  );
}

function EmptyStateIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4" />
      <path d="M12 8h.01" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </svg>
  );
}

function FireIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z" />
    </svg>
  );
}

function NavigateIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <polygon points="3 11 22 2 13 21 11 13 3 11" />
    </svg>
  );
}

function ExportIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" x2="12" y1="15" y2="3" />
    </svg>
  );
}
