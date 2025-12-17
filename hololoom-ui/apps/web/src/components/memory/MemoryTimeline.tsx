'use client';

import { useState, useEffect } from 'react';
import { Card, CardHeader, CardBody, Button, Badge, ConfidenceIndicator } from '@hololoom/design-system';

interface MemoryTimelineProps {
  searchQuery: string;
  onMemorySelect: (memoryId: string) => void;
  selectedMemoryId: string | null;
}

interface TimelineMemory {
  id: string;
  content: string;
  timestamp: Date;
  type: 'episodic' | 'semantic' | 'procedural';
  entities: string[];
  activation: number;
  consolidation: 'recent' | 'consolidating' | 'consolidated';
}

type TimeScale = 'hours' | 'days' | 'weeks';

export function MemoryTimeline({
  searchQuery,
  onMemorySelect,
  selectedMemoryId,
}: MemoryTimelineProps) {
  const [memories, setMemories] = useState<TimelineMemory[]>([]);
  const [timeScale, setTimeScale] = useState<TimeScale>('days');
  const [showConsolidationStages, setShowConsolidationStages] = useState(true);

  // Generate mock timeline data
  useEffect(() => {
    const now = Date.now();
    const mockMemories: TimelineMemory[] = Array.from({ length: 30 }, (_, i) => {
      const hoursAgo = Math.random() * 168; // Up to 7 days
      return {
        id: `mem-${i}`,
        content: [
          'Thompson Sampling uses posterior distributions to balance exploration and exploitation.',
          'Neural networks learn through gradient descent optimization.',
          'The attention mechanism enables models to focus on relevant input parts.',
          'Knowledge graphs store entities and relationships as nodes and edges.',
          'Reinforcement learning agents improve through reward-based feedback.',
          'Matryoshka embeddings provide multi-scale semantic representations.',
        ][i % 6],
        timestamp: new Date(now - hoursAgo * 3600000),
        type: (['episodic', 'semantic', 'procedural'] as const)[i % 3],
        entities: [
          ['Thompson Sampling', 'Bayesian'],
          ['Neural Network', 'Gradient'],
          ['Attention', 'Transformer'],
          ['Knowledge Graph', 'Entity'],
          ['Reinforcement Learning', 'Reward'],
          ['Matryoshka', 'Embedding'],
        ][i % 6],
        activation: Math.random(),
        consolidation: hoursAgo < 1 ? 'recent' : hoursAgo < 24 ? 'consolidating' : 'consolidated',
      };
    }).sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

    // Filter by search
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      setMemories(
        mockMemories.filter(
          (m) =>
            m.content.toLowerCase().includes(query) ||
            m.entities.some((e) => e.toLowerCase().includes(query))
        )
      );
    } else {
      setMemories(mockMemories);
    }
  }, [searchQuery]);

  // Group memories by time period
  const groupedMemories = groupByTime(memories, timeScale);

  return (
    <Card className="h-[600px] flex flex-col">
      <CardHeader className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 flex-shrink-0">
        <div>
          <h3 className="text-lg font-semibold text-fg-primary">Memory Timeline</h3>
          <p className="text-sm text-fg-tertiary">
            {memories.length} memories over time
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Time Scale Selector */}
          <div className="flex items-center gap-1 bg-bg-secondary rounded-lg p-1">
            {(['hours', 'days', 'weeks'] as TimeScale[]).map((scale) => (
              <button
                key={scale}
                onClick={() => setTimeScale(scale)}
                className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                  timeScale === scale
                    ? 'bg-cosmic-nebula/20 text-cosmic-nebula'
                    : 'text-fg-tertiary hover:text-fg-secondary'
                }`}
              >
                {scale.charAt(0).toUpperCase() + scale.slice(1)}
              </button>
            ))}
          </div>

          {/* Consolidation Toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowConsolidationStages(!showConsolidationStages)}
          >
            <ConsolidateIcon />
            <span className="hidden sm:inline ml-1">Stages</span>
          </Button>
        </div>
      </CardHeader>

      <CardBody className="flex-1 overflow-y-auto p-0">
        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gradient-to-b from-cosmic-nebula/50 via-cosmic-aurora/50 to-cosmic-starlight/50" />

          {/* Grouped sections */}
          {Object.entries(groupedMemories).map(([period, periodMemories]) => (
            <div key={period} className="relative">
              {/* Period header */}
              <div className="sticky top-0 z-10 bg-bg-primary/95 backdrop-blur-sm px-4 py-2 border-b border-border-primary">
                <div className="flex items-center gap-3 ml-12">
                  <span className="text-sm font-medium text-fg-secondary">{period}</span>
                  <Badge variant="secondary" size="sm">
                    {periodMemories.length}
                  </Badge>
                </div>
              </div>

              {/* Memories in this period */}
              <div className="py-2">
                {periodMemories.map((memory) => (
                  <button
                    key={memory.id}
                    onClick={() => onMemorySelect(memory.id)}
                    className={`w-full text-left relative pl-16 pr-4 py-3 hover:bg-bg-secondary/50 transition-colors ${
                      selectedMemoryId === memory.id ? 'bg-cosmic-nebula/10' : ''
                    }`}
                  >
                    {/* Timeline node */}
                    <div className="absolute left-6 top-1/2 -translate-y-1/2 w-5 h-5 rounded-full border-2 border-bg-primary bg-gradient-to-br from-cosmic-nebula to-cosmic-aurora flex items-center justify-center">
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{
                          backgroundColor:
                            memory.consolidation === 'recent'
                              ? '#22c55e'
                              : memory.consolidation === 'consolidating'
                              ? '#f59e0b'
                              : '#6366f1',
                        }}
                      />
                    </div>

                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge
                            variant={
                              memory.type === 'episodic'
                                ? 'default'
                                : memory.type === 'semantic'
                                ? 'secondary'
                                : 'accent'
                            }
                            size="sm"
                          >
                            {memory.type}
                          </Badge>
                          {showConsolidationStages && (
                            <Badge
                              variant={
                                memory.consolidation === 'recent'
                                  ? 'success'
                                  : memory.consolidation === 'consolidating'
                                  ? 'warning'
                                  : 'secondary'
                              }
                              size="sm"
                            >
                              {memory.consolidation}
                            </Badge>
                          )}
                          <span className="text-xs text-fg-tertiary">
                            {formatTime(memory.timestamp)}
                          </span>
                        </div>

                        <p className="text-sm text-fg-primary line-clamp-2">
                          {memory.content}
                        </p>

                        <div className="flex flex-wrap gap-1 mt-2">
                          {memory.entities.map((entity) => (
                            <span
                              key={entity}
                              className="text-xs px-1.5 py-0.5 bg-bg-tertiary text-fg-tertiary rounded"
                            >
                              {entity}
                            </span>
                          ))}
                        </div>
                      </div>

                      <div className="flex-shrink-0">
                        <ConfidenceIndicator
                          value={memory.activation}
                          size="sm"
                        />
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </CardBody>
    </Card>
  );
}

function groupByTime(
  memories: TimelineMemory[],
  scale: TimeScale
): Record<string, TimelineMemory[]> {
  const groups: Record<string, TimelineMemory[]> = {};
  const now = new Date();

  memories.forEach((memory) => {
    let key: string;

    if (scale === 'hours') {
      const hoursAgo = Math.floor((now.getTime() - memory.timestamp.getTime()) / 3600000);
      if (hoursAgo === 0) key = 'This hour';
      else if (hoursAgo === 1) key = '1 hour ago';
      else if (hoursAgo < 24) key = `${hoursAgo} hours ago`;
      else key = 'Earlier';
    } else if (scale === 'days') {
      const daysAgo = Math.floor((now.getTime() - memory.timestamp.getTime()) / 86400000);
      if (daysAgo === 0) key = 'Today';
      else if (daysAgo === 1) key = 'Yesterday';
      else if (daysAgo < 7) key = `${daysAgo} days ago`;
      else key = 'Last week';
    } else {
      const weeksAgo = Math.floor((now.getTime() - memory.timestamp.getTime()) / 604800000);
      if (weeksAgo === 0) key = 'This week';
      else if (weeksAgo === 1) key = 'Last week';
      else key = `${weeksAgo} weeks ago`;
    }

    if (!groups[key]) groups[key] = [];
    groups[key].push(memory);
  });

  return groups;
}

function formatTime(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);

  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;

  return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function ConsolidateIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 2v4" />
      <path d="m16 6-4 4-4-4" />
      <rect width="20" height="8" x="2" y="14" rx="2" />
      <path d="M6 18h.01" />
      <path d="M10 18h.01" />
    </svg>
  );
}
