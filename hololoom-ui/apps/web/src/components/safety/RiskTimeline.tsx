'use client';

import { useState, useEffect } from 'react';
import { Card, CardHeader, CardBody, Badge, SafetyBadge } from '@hololoom/design-system';
import type { SafetyLevel } from '@hololoom/api-client';

interface RiskTimelineProps {
  timeRange: '1h' | '24h' | '7d' | '30d';
}

interface RiskEvent {
  id: string;
  timestamp: Date;
  level: SafetyLevel;
  title: string;
  description: string;
  action: string;
  resolved: boolean;
}

// Generate mock events
function generateMockEvents(count: number): RiskEvent[] {
  const events: RiskEvent[] = [];
  const levels: SafetyLevel[] = ['safe', 'caution', 'warning', 'danger'];
  const titles = [
    'High-risk action gated',
    'Confidence threshold not met',
    'Unusual resource access pattern',
    'Action escalated for review',
    'Safety guardrail triggered',
  ];

  for (let i = 0; i < count; i++) {
    const level = levels[Math.floor(Math.random() * levels.length)];
    events.push({
      id: `event-${i}`,
      timestamp: new Date(Date.now() - Math.random() * 86400000 * 7),
      level,
      title: titles[Math.floor(Math.random() * titles.length)],
      description: `Action "${['execute_code', 'file_write', 'api_call', 'memory_modify'][Math.floor(Math.random() * 4)]}" was processed`,
      action: level === 'danger' ? 'Blocked' : level === 'warning' ? 'Escalated' : 'Allowed',
      resolved: Math.random() > 0.3,
    });
  }

  return events.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
}

export function RiskTimeline({ timeRange }: RiskTimelineProps) {
  const [events, setEvents] = useState<RiskEvent[]>(() => generateMockEvents(15));
  const [filter, setFilter] = useState<SafetyLevel | 'all'>('all');

  const filteredEvents = filter === 'all'
    ? events
    : events.filter((e) => e.level === filter);

  const levelCounts = {
    safe: events.filter((e) => e.level === 'safe').length,
    caution: events.filter((e) => e.level === 'caution').length,
    warning: events.filter((e) => e.level === 'warning').length,
    danger: events.filter((e) => e.level === 'danger').length,
  };

  return (
    <Card>
      <CardHeader className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-fg-primary">Risk Timeline</h2>
          <p className="text-sm text-fg-tertiary">
            {events.length} events in selected period
          </p>
        </div>

        {/* Filter buttons */}
        <div className="flex flex-wrap gap-2">
          <FilterButton
            active={filter === 'all'}
            onClick={() => setFilter('all')}
            count={events.length}
          >
            All
          </FilterButton>
          <FilterButton
            active={filter === 'danger'}
            onClick={() => setFilter('danger')}
            count={levelCounts.danger}
            variant="danger"
          >
            Critical
          </FilterButton>
          <FilterButton
            active={filter === 'warning'}
            onClick={() => setFilter('warning')}
            count={levelCounts.warning}
            variant="warning"
          >
            Warning
          </FilterButton>
          <FilterButton
            active={filter === 'caution'}
            onClick={() => setFilter('caution')}
            count={levelCounts.caution}
            variant="secondary"
          >
            Caution
          </FilterButton>
        </div>
      </CardHeader>
      <CardBody>
        {filteredEvents.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-fg-tertiary">No events matching filter</p>
          </div>
        ) : (
          <div className="relative">
            {/* Timeline line */}
            <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-border-primary" />

            {/* Events */}
            <div className="space-y-4">
              {filteredEvents.slice(0, 10).map((event) => (
                <div key={event.id} className="relative pl-10">
                  {/* Timeline dot */}
                  <div
                    className={`absolute left-2.5 w-3 h-3 rounded-full border-2 border-bg-primary ${
                      event.level === 'danger'
                        ? 'bg-safety-danger'
                        : event.level === 'warning'
                        ? 'bg-safety-warning'
                        : event.level === 'caution'
                        ? 'bg-safety-caution'
                        : 'bg-confidence-high'
                    }`}
                  />

                  {/* Event card */}
                  <div className="bg-bg-secondary/50 rounded-lg p-4">
                    <div className="flex items-start justify-between gap-4 mb-2">
                      <div className="flex items-center gap-2">
                        <SafetyBadge level={event.level} size="sm" />
                        <span className="font-medium text-fg-primary">
                          {event.title}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge
                          variant={
                            event.action === 'Blocked'
                              ? 'danger'
                              : event.action === 'Escalated'
                              ? 'warning'
                              : 'success'
                          }
                          size="sm"
                        >
                          {event.action}
                        </Badge>
                        {event.resolved && (
                          <Badge variant="secondary" size="sm">
                            Resolved
                          </Badge>
                        )}
                      </div>
                    </div>
                    <p className="text-sm text-fg-tertiary mb-2">
                      {event.description}
                    </p>
                    <p className="text-xs text-fg-tertiary">
                      {formatRelativeTime(event.timestamp)}
                    </p>
                  </div>
                </div>
              ))}
            </div>

            {filteredEvents.length > 10 && (
              <div className="mt-4 text-center">
                <button className="text-sm text-cosmic-nebula hover:text-cosmic-aurora transition-colors">
                  View {filteredEvents.length - 10} more events
                </button>
              </div>
            )}
          </div>
        )}
      </CardBody>
    </Card>
  );
}

interface FilterButtonProps {
  active: boolean;
  onClick: () => void;
  count: number;
  variant?: 'danger' | 'warning' | 'secondary';
  children: React.ReactNode;
}

function FilterButton({ active, onClick, count, variant, children }: FilterButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 text-sm rounded-lg transition-colors flex items-center gap-2 ${
        active
          ? 'bg-cosmic-nebula text-white'
          : 'bg-bg-secondary text-fg-secondary hover:text-fg-primary'
      }`}
    >
      {children}
      <span
        className={`text-xs px-1.5 py-0.5 rounded ${
          active ? 'bg-white/20' : 'bg-bg-tertiary'
        }`}
      >
        {count}
      </span>
    </button>
  );
}

function formatRelativeTime(date: Date): string {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);

  if (seconds < 60) return 'Just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)} minutes ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} hours ago`;
  return `${Math.floor(seconds / 86400)} days ago`;
}
