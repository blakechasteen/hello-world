'use client';

import { useState } from 'react';
import { Card, Badge } from '@hololoom/design-system';
import type { Message, Source } from '../../app/chat/page';

interface SourcePanelProps {
  message: Message;
  onClose: () => void;
}

export function SourcePanel({ message, onClose }: SourcePanelProps) {
  const [activeTab, setActiveTab] = useState<'sources' | 'reasoning'>('sources');

  return (
    <div className="fixed right-0 top-0 bottom-0 w-96 bg-bg-primary border-l border-border-primary shadow-xl flex flex-col z-50">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-primary">
        <h3 className="font-semibold text-fg-primary">Response Details</h3>
        <button
          onClick={onClose}
          className="p-1 text-fg-tertiary hover:text-fg-primary transition-colors rounded"
        >
          <CloseIcon />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-border-primary">
        <button
          onClick={() => setActiveTab('sources')}
          className={`flex-1 px-4 py-2.5 text-sm font-medium transition-colors ${
            activeTab === 'sources'
              ? 'text-cosmic-nebula border-b-2 border-cosmic-nebula'
              : 'text-fg-tertiary hover:text-fg-secondary'
          }`}
        >
          Sources ({message.sources?.length || 0})
        </button>
        <button
          onClick={() => setActiveTab('reasoning')}
          className={`flex-1 px-4 py-2.5 text-sm font-medium transition-colors ${
            activeTab === 'reasoning'
              ? 'text-cosmic-nebula border-b-2 border-cosmic-nebula'
              : 'text-fg-tertiary hover:text-fg-secondary'
          }`}
        >
          Reasoning
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'sources' ? (
          <SourcesTab sources={message.sources || []} />
        ) : (
          <ReasoningTab message={message} />
        )}
      </div>

      {/* Footer - Confidence Summary */}
      <div className="border-t border-border-primary px-4 py-3 bg-bg-secondary/50">
        <div className="flex items-center justify-between">
          <span className="text-sm text-fg-tertiary">Overall Confidence</span>
          <div className="flex items-center gap-2">
            <div className="w-24 h-2 bg-bg-tertiary rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${getConfidenceColor(message.confidence || 0)}`}
                style={{ width: `${(message.confidence || 0) * 100}%` }}
              />
            </div>
            <span className="text-sm font-semibold text-fg-primary">
              {Math.round((message.confidence || 0) * 100)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function SourcesTab({ sources }: { sources: Source[] }) {
  if (sources.length === 0) {
    return (
      <div className="text-center py-8 text-fg-tertiary">
        <p>No sources available for this response.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {sources.map((source, index) => (
        <SourceCard key={source.id} source={source} index={index + 1} />
      ))}
    </div>
  );
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  const typeColors: Record<Source['type'], string> = {
    memory: 'bg-cosmic-aurora/10 text-cosmic-aurora border-cosmic-aurora/20',
    knowledge_graph: 'bg-cosmic-nebula/10 text-cosmic-nebula border-cosmic-nebula/20',
    cache: 'bg-safety-safe/10 text-safety-safe border-safety-safe/20',
  };

  const typeLabels: Record<Source['type'], string> = {
    memory: 'Memory',
    knowledge_graph: 'Graph',
    cache: 'Cache',
  };

  return (
    <Card className="p-3">
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="w-5 h-5 rounded-full bg-bg-tertiary flex items-center justify-center text-xs font-medium text-fg-secondary">
            {index}
          </span>
          <span className={`px-2 py-0.5 text-xs font-medium rounded border ${typeColors[source.type]}`}>
            {typeLabels[source.type]}
          </span>
        </div>
        <RelevanceBar relevance={source.relevance} />
      </div>

      <p className="text-sm text-fg-secondary leading-relaxed mb-2">
        {source.content}
      </p>

      {source.metadata && (
        <div className="flex items-center gap-3 text-xs text-fg-tertiary">
          {source.metadata.timestamp && (
            <span>
              {formatRelativeTime(source.metadata.timestamp as number)}
            </span>
          )}
          {source.metadata.accessCount && (
            <span>
              {source.metadata.accessCount} accesses
            </span>
          )}
        </div>
      )}
    </Card>
  );
}

function RelevanceBar({ relevance }: { relevance: number }) {
  const percentage = Math.round(relevance * 100);

  return (
    <div className="flex items-center gap-1.5">
      <div className="w-16 h-1.5 bg-bg-tertiary rounded-full overflow-hidden">
        <div
          className="h-full bg-cosmic-nebula rounded-full"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="text-xs text-fg-tertiary">{percentage}%</span>
    </div>
  );
}

function ReasoningTab({ message }: { message: Message }) {
  return (
    <div className="space-y-4">
      {/* Mode Info */}
      <div className="p-3 bg-bg-secondary rounded-lg">
        <div className="flex items-center gap-2 mb-2">
          <ModeIcon mode={message.reasoningMode} />
          <span className="font-medium text-fg-primary">
            {formatModeName(message.reasoningMode)}
          </span>
        </div>
        <p className="text-sm text-fg-tertiary">
          {getModeDescription(message.reasoningMode)}
        </p>
      </div>

      {/* Steps */}
      {message.steps && message.steps.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-fg-primary mb-2">
            Reasoning Steps
          </h4>
          <div className="space-y-2">
            {message.steps.map((step, index) => (
              <div
                key={index}
                className="flex items-start gap-3 p-2 rounded-lg bg-bg-secondary/50"
              >
                <span className="w-5 h-5 rounded-full bg-safety-safe/20 text-safety-safe flex items-center justify-center text-xs font-medium flex-shrink-0">
                  ✓
                </span>
                <span className="text-sm text-fg-secondary">{step}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Verification Status */}
      {message.verified !== undefined && (
        <div className="p-3 bg-bg-secondary rounded-lg">
          <div className="flex items-center gap-2">
            {message.verified ? (
              <>
                <div className="w-6 h-6 rounded-full bg-safety-safe/20 text-safety-safe flex items-center justify-center">
                  <VerifiedIcon />
                </div>
                <span className="font-medium text-safety-safe">Verified</span>
              </>
            ) : (
              <>
                <div className="w-6 h-6 rounded-full bg-safety-risky/20 text-safety-risky flex items-center justify-center">
                  <WarningIcon />
                </div>
                <span className="font-medium text-safety-risky">Unverified</span>
              </>
            )}
          </div>
        </div>
      )}

      {/* Metadata */}
      <div>
        <h4 className="text-sm font-medium text-fg-primary mb-2">Metadata</h4>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="p-2 bg-bg-secondary rounded">
            <span className="text-fg-tertiary block text-xs">Timestamp</span>
            <span className="text-fg-secondary">
              {new Date(message.timestamp).toLocaleString()}
            </span>
          </div>
          <div className="p-2 bg-bg-secondary rounded">
            <span className="text-fg-tertiary block text-xs">Sources Used</span>
            <span className="text-fg-secondary">
              {message.sources?.length || 0}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function ModeIcon({ mode }: { mode?: string }) {
  const icons: Record<string, React.ReactNode> = {
    direct: <DirectIcon />,
    verify: <VerifyIcon />,
    research: <ResearchIcon />,
    plan_execute: <PlanIcon />,
  };

  return (
    <div className="w-6 h-6 rounded-full bg-cosmic-nebula/20 text-cosmic-nebula flex items-center justify-center">
      {icons[mode || 'direct']}
    </div>
  );
}

function formatModeName(mode?: string): string {
  const names: Record<string, string> = {
    direct: 'Direct Answer',
    verify: 'Verify & Confirm',
    research: 'Deep Research',
    plan_execute: 'Plan & Execute',
  };
  return names[mode || 'direct'];
}

function getModeDescription(mode?: string): string {
  const descriptions: Record<string, string> = {
    direct: 'Single-pass answer generation for straightforward queries.',
    verify: 'Cross-referenced answer with verification against multiple sources.',
    research: 'Multi-query exploration synthesizing information from various angles.',
    plan_execute: 'Task decomposition with step-by-step execution and synthesis.',
  };
  return descriptions[mode || 'direct'];
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.7) return 'bg-safety-safe';
  if (confidence >= 0.5) return 'bg-safety-risky';
  return 'bg-safety-critical';
}

function formatRelativeTime(timestamp: number): string {
  const now = Date.now();
  const diff = now - timestamp;
  const days = Math.floor(diff / 86400000);

  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days} days ago`;
  if (days < 30) return `${Math.floor(days / 7)} weeks ago`;
  return `${Math.floor(days / 30)} months ago`;
}

function CloseIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function DirectIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  );
}

function VerifyIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  );
}

function ResearchIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function PlanIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="8" y1="6" x2="21" y2="6" />
      <line x1="8" y1="12" x2="21" y2="12" />
      <line x1="8" y1="18" x2="21" y2="18" />
      <line x1="3" y1="6" x2="3.01" y2="6" />
      <line x1="3" y1="12" x2="3.01" y2="12" />
      <line x1="3" y1="18" x2="3.01" y2="18" />
    </svg>
  );
}

function VerifiedIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function WarningIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}
