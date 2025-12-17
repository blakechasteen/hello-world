'use client';

import { useState, useEffect } from 'react';
import { Card, CardBody, ConfidenceIndicator } from '@hololoom/design-system';

interface MemoryStatsProps {
  className?: string;
}

interface Stats {
  totalMemories: number;
  activeMemories: number;
  entities: number;
  relationships: number;
  avgActivation: number;
  coherence: number;
  hotPatterns: number;
  cacheHitRate: number;
}

export function MemoryStats({ className }: MemoryStatsProps) {
  const [stats, setStats] = useState<Stats>({
    totalMemories: 0,
    activeMemories: 0,
    entities: 0,
    relationships: 0,
    avgActivation: 0,
    coherence: 0,
    hotPatterns: 0,
    cacheHitRate: 0,
  });

  useEffect(() => {
    // Mock stats - would connect to HoloLoom API
    setStats({
      totalMemories: 1523,
      activeMemories: 342,
      entities: 847,
      relationships: 2156,
      avgActivation: 0.67,
      coherence: 0.82,
      hotPatterns: 24,
      cacheHitRate: 0.89,
    });

    // Simulate real-time updates
    const interval = setInterval(() => {
      setStats((prev) => ({
        ...prev,
        activeMemories: prev.activeMemories + Math.floor(Math.random() * 5 - 2),
        avgActivation: Math.min(1, Math.max(0, prev.avgActivation + (Math.random() - 0.5) * 0.02)),
        coherence: Math.min(1, Math.max(0, prev.coherence + (Math.random() - 0.5) * 0.01)),
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const statCards = [
    {
      label: 'Total Memories',
      value: stats.totalMemories.toLocaleString(),
      subValue: `${stats.activeMemories} active`,
      icon: <MemoryIcon />,
      color: 'cosmic-nebula',
    },
    {
      label: 'Knowledge Graph',
      value: stats.entities.toLocaleString(),
      subValue: `${stats.relationships.toLocaleString()} edges`,
      icon: <GraphIcon />,
      color: 'cosmic-aurora',
    },
    {
      label: 'Avg Activation',
      value: (stats.avgActivation * 100).toFixed(1) + '%',
      subValue: `${stats.coherence >= 0.7 ? 'High' : stats.coherence >= 0.4 ? 'Medium' : 'Low'} coherence`,
      icon: <WaveIcon />,
      color: stats.avgActivation >= 0.7 ? 'confidence-high' : stats.avgActivation >= 0.4 ? 'confidence-medium' : 'confidence-low',
    },
    {
      label: 'Hot Patterns',
      value: stats.hotPatterns.toString(),
      subValue: `${(stats.cacheHitRate * 100).toFixed(0)}% cache hit`,
      icon: <FireIcon />,
      color: 'cosmic-starlight',
    },
  ];

  return (
    <div className={`grid grid-cols-2 lg:grid-cols-4 gap-4 ${className}`}>
      {statCards.map((card) => (
        <Card key={card.label} className="relative overflow-hidden">
          <CardBody className="p-4">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs font-medium text-fg-tertiary uppercase tracking-wider">
                  {card.label}
                </p>
                <p className="text-2xl font-bold text-fg-primary mt-1">
                  {card.value}
                </p>
                <p className="text-sm text-fg-tertiary mt-0.5">
                  {card.subValue}
                </p>
              </div>
              <div className={`w-10 h-10 rounded-lg bg-${card.color}/10 flex items-center justify-center text-${card.color}`}>
                {card.icon}
              </div>
            </div>

            {/* Activation indicator for memory stats */}
            {card.label === 'Avg Activation' && (
              <div className="mt-3 pt-3 border-t border-border-primary">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-fg-tertiary">Coherence</span>
                  <ConfidenceIndicator value={stats.coherence} size="sm" />
                </div>
              </div>
            )}
          </CardBody>

          {/* Decorative gradient */}
          <div
            className={`absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-${card.color}/50 to-transparent`}
          />
        </Card>
      ))}
    </div>
  );
}

function MemoryIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 2a4 4 0 0 0-4 4v12a4 4 0 0 0 8 0V6a4 4 0 0 0-4-4z" />
      <path d="M12 2v20" />
      <path d="M4 12h2" />
      <path d="M18 12h2" />
      <path d="M6 6h2" />
      <path d="M16 6h2" />
      <path d="M6 18h2" />
      <path d="M16 18h2" />
    </svg>
  );
}

function GraphIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
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
      <path d="m14.4 9.6 3-3" />
      <path d="m6.6 16.6 3-3" />
    </svg>
  );
}

function WaveIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M2 12h2" />
      <path d="M6 8v8" />
      <path d="M10 4v16" />
      <path d="M14 6v12" />
      <path d="M18 10v4" />
      <path d="M22 12h-2" />
    </svg>
  );
}

function FireIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z" />
    </svg>
  );
}
