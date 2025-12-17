/**
 * InjectionHistoryLog Component
 *
 * Displays history of MRF/MCTS injections for a thread with:
 * - Chronological list with timestamps
 * - Strategy/config details for each injection
 * - Step context (which step was injected)
 * - Filter by injection type (MRF/MCTS)
 * - Success/failure status indicators
 *
 * HoloLoom Agent Manager UI - Phase 5
 * Date: 2025-12-15
 */

import React, { useState, useMemo } from 'react';
import {
  Filter,
  RefreshCw,
  CheckCircle,
  XCircle,
  Zap,
  Brain,
  ChevronDown,
  ChevronRight,
  Clock,
  Hash,
} from 'lucide-react';
import { InjectionHistoryEntry, MCTSConfig } from '../../lib/api';
import { useInjectionHistory } from '../../stores/stepStore';

// ============================================================================
// Types
// ============================================================================

type InjectionFilter = 'all' | 'mrf' | 'mcts';

interface InjectionHistoryLogProps {
  /** Thread ID to display history for */
  threadId: string;
  /** Maximum entries to show (default: 50) */
  maxEntries?: number;
  /** Callback when step ID is clicked */
  onStepClick?: (stepId: string) => void;
  /** Optional CSS class */
  className?: string;
}

// ============================================================================
// Config
// ============================================================================

/**
 * MRF Strategy display configuration
 */
const MRF_STRATEGY_CONFIG: Record<string, { label: string; description: string; color: string }> = {
  auto: {
    label: 'AUTO',
    description: 'Automatic strategy selection',
    color: 'text-slate-400',
  },
  verify: {
    label: 'VERIFY',
    description: 'Verify accuracy and correctness',
    color: 'text-blue-400',
  },
  elegance: {
    label: 'ELEGANCE',
    description: 'Optimize for clarity and simplicity',
    color: 'text-purple-400',
  },
  critique: {
    label: 'CRITIQUE',
    description: 'Critical analysis and improvement',
    color: 'text-amber-400',
  },
  refine: {
    label: 'REFINE',
    description: 'Iterative refinement',
    color: 'text-emerald-400',
  },
  hofstadter: {
    label: 'HOFSTADTER',
    description: 'Recursive self-reference',
    color: 'text-pink-400',
  },
};

/**
 * Injection type badge configuration
 */
const INJECTION_TYPE_CONFIG: Record<'mrf' | 'mcts', { icon: React.ReactNode; label: string; bgColor: string; textColor: string }> = {
  mrf: {
    icon: <Brain className="w-3 h-3" />,
    label: 'MRF',
    bgColor: 'bg-purple-900/50',
    textColor: 'text-purple-300',
  },
  mcts: {
    icon: <Zap className="w-3 h-3" />,
    label: 'MCTS',
    bgColor: 'bg-amber-900/50',
    textColor: 'text-amber-300',
  },
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Format timestamp for display
 */
function formatTimestamp(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return timestamp;
  }
}

/**
 * Format relative time
 */
function getRelativeTime(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHour = Math.floor(diffMin / 60);

    if (diffSec < 60) return `${diffSec}s ago`;
    if (diffMin < 60) return `${diffMin}m ago`;
    if (diffHour < 24) return `${diffHour}h ago`;
    return formatTimestamp(timestamp);
  } catch {
    return timestamp;
  }
}

/**
 * Format MCTS config for display
 */
function formatMCTSConfig(config?: MCTSConfig): string {
  if (!config) return 'N/A';
  return `Budget: ${config.budget}, Exploration: ${config.exploration.toFixed(2)}`;
}

// ============================================================================
// Sub-Components
// ============================================================================

/**
 * Injection type badge
 */
const InjectionTypeBadge: React.FC<{ type: 'mrf' | 'mcts' }> = ({ type }) => {
  const config = INJECTION_TYPE_CONFIG[type];
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${config.bgColor} ${config.textColor}`}
    >
      {config.icon}
      {config.label}
    </span>
  );
};

/**
 * Success/failure status badge
 */
const StatusBadge: React.FC<{ success: boolean }> = ({ success }) => {
  if (success) {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-emerald-400">
        <CheckCircle className="w-3 h-3" />
        Success
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-xs text-red-400">
      <XCircle className="w-3 h-3" />
      Failed
    </span>
  );
};

/**
 * MRF strategy details
 */
const MRFStrategyDetails: React.FC<{ strategy?: string }> = ({ strategy }) => {
  const strategyKey = strategy?.toLowerCase() || 'auto';
  const config = MRF_STRATEGY_CONFIG[strategyKey] || MRF_STRATEGY_CONFIG.auto;

  return (
    <div className="text-xs">
      <span className={`font-medium ${config.color}`}>{config.label}</span>
      <span className="text-slate-500 ml-2">{config.description}</span>
    </div>
  );
};

/**
 * MCTS config details
 */
const MCTSConfigDetails: React.FC<{ config?: MCTSConfig }> = ({ config }) => {
  if (!config) return <span className="text-xs text-slate-500">No config</span>;

  return (
    <div className="text-xs flex items-center gap-3">
      <span className="text-slate-400">
        Budget: <span className="text-amber-300 font-medium">{config.budget}</span>
      </span>
      <span className="text-slate-400">
        Exploration: <span className="text-amber-300 font-medium">{config.exploration.toFixed(2)}</span>
      </span>
    </div>
  );
};

/**
 * Single injection entry row
 */
const InjectionEntry: React.FC<{
  entry: InjectionHistoryEntry;
  isExpanded: boolean;
  onToggle: () => void;
  onStepClick?: (stepId: string) => void;
}> = ({ entry, isExpanded, onToggle, onStepClick }) => {
  return (
    <div
      className={`border rounded-lg transition-colors ${
        entry.success
          ? 'border-slate-700 bg-slate-800/50'
          : 'border-red-900/50 bg-red-950/20'
      }`}
    >
      {/* Header row */}
      <div
        className="flex items-center gap-3 px-3 py-2 cursor-pointer hover:bg-slate-700/30"
        onClick={onToggle}
      >
        {/* Expand/collapse icon */}
        <button className="text-slate-500 hover:text-slate-300">
          {isExpanded ? (
            <ChevronDown className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
        </button>

        {/* Type badge */}
        <InjectionTypeBadge type={entry.injectionType} />

        {/* Strategy/Config summary */}
        <div className="flex-1 min-w-0 truncate text-sm text-slate-300">
          {entry.injectionType === 'mrf' ? (
            <span>{entry.strategy?.toUpperCase() || 'AUTO'}</span>
          ) : (
            <span>{formatMCTSConfig(entry.config)}</span>
          )}
        </div>

        {/* Status */}
        <StatusBadge success={entry.success} />

        {/* Timestamp */}
        <span className="text-xs text-slate-500 whitespace-nowrap">
          {getRelativeTime(entry.timestamp)}
        </span>
      </div>

      {/* Expanded details */}
      {isExpanded && (
        <div className="px-3 pb-3 pt-1 border-t border-slate-700/50">
          <div className="space-y-2 text-sm">
            {/* Step reference */}
            <div className="flex items-center gap-2">
              <Hash className="w-3 h-3 text-slate-500" />
              <span className="text-slate-400">Step:</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onStepClick?.(entry.stepId);
                }}
                className="text-blue-400 hover:text-blue-300 hover:underline font-mono text-xs"
              >
                {entry.stepId}
              </button>
            </div>

            {/* Timestamp */}
            <div className="flex items-center gap-2">
              <Clock className="w-3 h-3 text-slate-500" />
              <span className="text-slate-400">Time:</span>
              <span className="text-slate-300">{formatTimestamp(entry.timestamp)}</span>
            </div>

            {/* Details based on type */}
            {entry.injectionType === 'mrf' ? (
              <div className="mt-2">
                <span className="text-slate-400 text-xs">Strategy:</span>
                <div className="mt-1">
                  <MRFStrategyDetails strategy={entry.strategy} />
                </div>
              </div>
            ) : (
              <div className="mt-2">
                <span className="text-slate-400 text-xs">Configuration:</span>
                <div className="mt-1">
                  <MCTSConfigDetails config={entry.config} />
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

/**
 * InjectionHistoryLog Component
 *
 * Displays injection history for a thread with filtering and details.
 */
export const InjectionHistoryLog: React.FC<InjectionHistoryLogProps> = ({
  threadId,
  maxEntries = 50,
  onStepClick,
  className = '',
}) => {
  // State
  const [filter, setFilter] = useState<InjectionFilter>('all');
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Get history from store
  const { history, reload } = useInjectionHistory(threadId);

  // Filter and limit entries
  const filteredEntries = useMemo(() => {
    let entries = history || [];

    // Apply type filter
    if (filter !== 'all') {
      entries = entries.filter((e) => e.injectionType === filter);
    }

    // Limit entries
    return entries.slice(0, maxEntries);
  }, [history, filter, maxEntries]);

  // Stats
  const stats = useMemo(() => {
    const all = history || [];
    return {
      total: all.length,
      mrf: all.filter((e) => e.injectionType === 'mrf').length,
      mcts: all.filter((e) => e.injectionType === 'mcts').length,
      successful: all.filter((e) => e.success).length,
    };
  }, [history]);

  // Handlers
  const handleRefresh = async () => {
    setIsRefreshing(true);
    await reload();
    setIsRefreshing(false);
  };

  const toggleExpanded = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium text-slate-200">Injection History</h3>
          <span className="text-xs text-slate-500">
            ({stats.total} total)
          </span>
        </div>

        {/* Refresh button */}
        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="p-1 text-slate-400 hover:text-slate-200 disabled:opacity-50"
          title="Refresh history"
        >
          <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Filter bar */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-slate-700/50">
        <Filter className="w-3 h-3 text-slate-500" />
        <div className="flex items-center gap-1">
          <button
            onClick={() => setFilter('all')}
            className={`px-2 py-0.5 text-xs rounded transition-colors ${
              filter === 'all'
                ? 'bg-slate-600 text-slate-100'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            All ({stats.total})
          </button>
          <button
            onClick={() => setFilter('mrf')}
            className={`px-2 py-0.5 text-xs rounded transition-colors ${
              filter === 'mrf'
                ? 'bg-purple-800/50 text-purple-200'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            MRF ({stats.mrf})
          </button>
          <button
            onClick={() => setFilter('mcts')}
            className={`px-2 py-0.5 text-xs rounded transition-colors ${
              filter === 'mcts'
                ? 'bg-amber-800/50 text-amber-200'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            MCTS ({stats.mcts})
          </button>
        </div>

        {/* Success rate */}
        <div className="ml-auto text-xs text-slate-500">
          <span className="text-emerald-400">{stats.successful}</span>
          <span> / {stats.total} successful</span>
        </div>
      </div>

      {/* Entry list */}
      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {filteredEntries.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-slate-500">
            <Zap className="w-8 h-8 mb-2 opacity-50" />
            <p className="text-sm">No injection history</p>
            <p className="text-xs mt-1">
              {filter !== 'all'
                ? `No ${filter.toUpperCase()} injections found`
                : 'Injections will appear here'}
            </p>
          </div>
        ) : (
          filteredEntries.map((entry) => (
            <InjectionEntry
              key={entry.id}
              entry={entry}
              isExpanded={expandedIds.has(entry.id)}
              onToggle={() => toggleExpanded(entry.id)}
              onStepClick={onStepClick}
            />
          ))
        )}
      </div>
    </div>
  );
};

export default InjectionHistoryLog;
