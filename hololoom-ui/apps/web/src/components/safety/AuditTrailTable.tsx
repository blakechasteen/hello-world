'use client';

import { useState, useEffect } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Button,
  Badge,
  SafetyBadge,
  Input,
  ConfidenceIndicator,
} from '@hololoom/design-system';
import type { SafetyLevel } from '@hololoom/api-client';

interface AuditTrailTableProps {
  timeRange: '1h' | '24h' | '7d' | '30d';
  autoRefresh: boolean;
}

interface AuditEntry {
  id: string;
  timestamp: Date;
  query: string;
  action: string;
  tool: string;
  safetyLevel: SafetyLevel;
  confidence: number;
  latencyMs: number;
  outcome: 'success' | 'blocked' | 'escalated';
}

// Generate mock entries
function generateMockEntries(count: number): AuditEntry[] {
  const tools = ['answer', 'research', 'code_execute', 'memory_store', 'file_read'];
  const actions = ['query', 'store', 'execute', 'retrieve', 'analyze'];
  const queries = [
    'What is Thompson Sampling?',
    'Explain the memory system',
    'Run the test suite',
    'Store conversation context',
    'Analyze code patterns',
  ];

  return Array.from({ length: count }, (_, i) => ({
    id: `audit-${i}`,
    timestamp: new Date(Date.now() - Math.random() * 86400000),
    query: queries[Math.floor(Math.random() * queries.length)],
    action: actions[Math.floor(Math.random() * actions.length)],
    tool: tools[Math.floor(Math.random() * tools.length)],
    safetyLevel: (['safe', 'safe', 'safe', 'caution', 'warning'] as SafetyLevel[])[
      Math.floor(Math.random() * 5)
    ],
    confidence: 0.7 + Math.random() * 0.3,
    latencyMs: 50 + Math.random() * 200,
    outcome: (['success', 'success', 'success', 'blocked', 'escalated'] as const)[
      Math.floor(Math.random() * 5)
    ],
  })).sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
}

export function AuditTrailTable({ timeRange, autoRefresh }: AuditTrailTableProps) {
  const [entries, setEntries] = useState<AuditEntry[]>(() => generateMockEntries(50));
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(0);
  const [selectedEntry, setSelectedEntry] = useState<AuditEntry | null>(null);
  const pageSize = 10;

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      // Add new entry at the top
      const newEntry = generateMockEntries(1)[0];
      newEntry.timestamp = new Date();
      setEntries((prev) => [newEntry, ...prev.slice(0, 99)]);
    }, 3000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  const filteredEntries = entries.filter(
    (entry) =>
      entry.query.toLowerCase().includes(search.toLowerCase()) ||
      entry.action.toLowerCase().includes(search.toLowerCase()) ||
      entry.tool.toLowerCase().includes(search.toLowerCase())
  );

  const paginatedEntries = filteredEntries.slice(
    page * pageSize,
    (page + 1) * pageSize
  );

  const totalPages = Math.ceil(filteredEntries.length / pageSize);

  return (
    <Card>
      <CardHeader className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-fg-primary">Audit Trail</h2>
          <p className="text-sm text-fg-tertiary">
            {filteredEntries.length} entries • Complete decision provenance
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Input
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(0);
            }}
            placeholder="Search entries..."
            className="w-64"
          />
          <Button variant="secondary" size="sm">
            Export
          </Button>
        </div>
      </CardHeader>
      <CardBody className="p-0">
        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border-primary bg-bg-secondary/50">
                <th className="px-4 py-3 text-left text-xs font-medium text-fg-tertiary uppercase tracking-wider">
                  Time
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-fg-tertiary uppercase tracking-wider">
                  Query / Action
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-fg-tertiary uppercase tracking-wider">
                  Tool
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-fg-tertiary uppercase tracking-wider">
                  Safety
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-fg-tertiary uppercase tracking-wider">
                  Confidence
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-fg-tertiary uppercase tracking-wider">
                  Outcome
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-fg-tertiary uppercase tracking-wider">
                  Latency
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-primary">
              {paginatedEntries.map((entry) => (
                <tr
                  key={entry.id}
                  onClick={() => setSelectedEntry(entry)}
                  className="hover:bg-bg-secondary/50 cursor-pointer transition-colors"
                >
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-fg-tertiary">
                    {formatTime(entry.timestamp)}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex flex-col">
                      <span className="text-sm text-fg-primary truncate max-w-xs">
                        {entry.query}
                      </span>
                      <span className="text-xs text-fg-tertiary">{entry.action}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <Badge variant="secondary" size="sm">
                      {entry.tool}
                    </Badge>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <SafetyBadge level={entry.safetyLevel} size="sm" />
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <ConfidenceIndicator value={entry.confidence} size="sm" />
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <Badge
                      variant={
                        entry.outcome === 'success'
                          ? 'success'
                          : entry.outcome === 'blocked'
                          ? 'danger'
                          : 'warning'
                      }
                      size="sm"
                    >
                      {entry.outcome}
                    </Badge>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-fg-tertiary">
                    {entry.latencyMs.toFixed(0)}ms
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-border-primary">
          <p className="text-sm text-fg-tertiary">
            Showing {page * pageSize + 1} to{' '}
            {Math.min((page + 1) * pageSize, filteredEntries.length)} of{' '}
            {filteredEntries.length} entries
          </p>
          <div className="flex gap-2">
            <Button
              variant="secondary"
              size="sm"
              disabled={page === 0}
              onClick={() => setPage((p) => p - 1)}
            >
              Previous
            </Button>
            <Button
              variant="secondary"
              size="sm"
              disabled={page >= totalPages - 1}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </Button>
          </div>
        </div>
      </CardBody>

      {/* Detail modal */}
      {selectedEntry && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-bg-primary/80 backdrop-blur-sm"
          onClick={() => setSelectedEntry(null)}
        >
          <div
            className="bg-bg-primary border border-border-primary rounded-lg shadow-xl max-w-lg w-full m-4 p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between mb-4">
              <h3 className="text-lg font-semibold text-fg-primary">
                Audit Entry Details
              </h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelectedEntry(null)}
              >
                <CloseIcon />
              </Button>
            </div>

            <div className="space-y-4">
              <DetailRow label="ID" value={selectedEntry.id} />
              <DetailRow
                label="Timestamp"
                value={selectedEntry.timestamp.toISOString()}
              />
              <DetailRow label="Query" value={selectedEntry.query} />
              <DetailRow label="Action" value={selectedEntry.action} />
              <DetailRow
                label="Tool"
                value={<Badge variant="secondary">{selectedEntry.tool}</Badge>}
              />
              <DetailRow
                label="Safety Level"
                value={<SafetyBadge level={selectedEntry.safetyLevel} />}
              />
              <DetailRow
                label="Confidence"
                value={
                  <div className="flex items-center gap-2">
                    <span>{(selectedEntry.confidence * 100).toFixed(1)}%</span>
                    <ConfidenceIndicator value={selectedEntry.confidence} />
                  </div>
                }
              />
              <DetailRow
                label="Outcome"
                value={
                  <Badge
                    variant={
                      selectedEntry.outcome === 'success'
                        ? 'success'
                        : selectedEntry.outcome === 'blocked'
                        ? 'danger'
                        : 'warning'
                    }
                  >
                    {selectedEntry.outcome}
                  </Badge>
                }
              />
              <DetailRow
                label="Latency"
                value={`${selectedEntry.latencyMs.toFixed(2)}ms`}
              />
            </div>

            <div className="mt-6 flex gap-2">
              <Button variant="secondary" size="sm" className="flex-1">
                View Full Trace
              </Button>
              <Button variant="secondary" size="sm" className="flex-1">
                Export JSON
              </Button>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}

function DetailRow({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="flex justify-between items-start">
      <span className="text-sm text-fg-tertiary">{label}</span>
      <span className="text-sm text-fg-primary text-right">{value}</span>
    </div>
  );
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function CloseIcon() {
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
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </svg>
  );
}
