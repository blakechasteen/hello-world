import React, { useState, useMemo } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Copy,
  CheckCircle,
  Filter,
  ArrowUpDown,
  Zap,
} from 'lucide-react';

/**
 * MemoryNode represents a memory element accessed during thread execution
 * Source types indicate where the memory came from:
 * - graph: Knowledge graph (symbolic relationships)
 * - vector: Vector database (semantic similarity)
 * - cache: Query cache (fast retrieval)
 * - hot_pattern: Hot pattern feedback (frequently accessed)
 */
interface MemoryNode {
  id: string;
  content: string;
  relevance: number;
  sourceType: 'graph' | 'vector' | 'cache' | 'hot_pattern';
  accessedAt: string;
  stepId: string;
  metadata?: Record<string, unknown>;
}

/**
 * Sorting options for memory nodes
 */
type SortField = 'relevance' | 'recency' | 'access_count';

/**
 * Source type badge configuration
 */
interface SourceTypeConfig {
  icon: string;
  label: string;
  color: string;
  bgColor: string;
}

interface MemoryNodesProps {
  /** Array of memory nodes to display */
  nodes: MemoryNode[];
  /** Group nodes by step (if true) or show all together */
  groupByStep?: boolean;
  /** Callback when a node ID is clicked (for copying) */
  onNodeClick?: (nodeId: string) => void;
  /** Optional CSS class for styling */
  className?: string;
}

/**
 * MemoryNodes Component
 *
 * Displays memory nodes accessed during thread execution with:
 * - Relevance-based heat map coloring
 * - Sortable by relevance, recency, or access count
 * - Expandable nodes with full content
 * - Search/filter by content
 * - Click-to-copy node IDs
 * - Visual source type badges
 */
export const MemoryNodes: React.FC<MemoryNodesProps> = ({
  nodes,
  groupByStep = false,
  onNodeClick,
  className = '',
}) => {
  // State management
  const [expandedNodeIds, setExpandedNodeIds] = useState<Set<string>>(
    new Set()
  );
  const [sortField, setSortField] = useState<SortField>('relevance');
  const [searchQuery, setSearchQuery] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);

  /**
   * Source type badge configuration mapping
   */
  const sourceTypeConfig: Record<MemoryNode['sourceType'], SourceTypeConfig> =
    {
      graph: {
        icon: '🔗',
        label: 'Graph',
        color: 'text-blue-400',
        bgColor: 'bg-blue-900',
      },
      vector: {
        icon: '📊',
        label: 'Vector',
        color: 'text-cyan-400',
        bgColor: 'bg-cyan-900',
      },
      cache: {
        icon: '⚡',
        label: 'Cache',
        color: 'text-yellow-400',
        bgColor: 'bg-yellow-900',
      },
      hot_pattern: {
        icon: '🔥',
        label: 'Hot',
        color: 'text-orange-400',
        bgColor: 'bg-orange-900',
      },
    };

  /**
   * Get relevance-based color for heat map effect
   * 0.9+ = emerald (high relevance)
   * 0.7-0.9 = blue (medium relevance)
   * <0.7 = slate (low relevance)
   */
  const getRelevanceColor = (
    relevance: number
  ): {
    bgColor: string;
    borderColor: string;
    barColor: string;
  } => {
    if (relevance >= 0.9) {
      return {
        bgColor: 'bg-emerald-950',
        borderColor: 'border-emerald-700',
        barColor: 'bg-emerald-600',
      };
    } else if (relevance >= 0.7) {
      return {
        bgColor: 'bg-blue-950',
        borderColor: 'border-blue-700',
        barColor: 'bg-blue-600',
      };
    } else {
      return {
        bgColor: 'bg-slate-800',
        borderColor: 'border-slate-700',
        barColor: 'bg-slate-600',
      };
    }
  };

  /**
   * Get access count from metadata or return 1 as default
   */
  const getAccessCount = (node: MemoryNode): number => {
    return (node.metadata?.access_count as number) ?? 1;
  };

  /**
   * Parse timestamp to Date object
   */
  const parseTimestamp = (timestamp: string): Date => {
    return new Date(timestamp);
  };

  /**
   * Filter nodes by search query
   */
  const filteredNodes = useMemo(() => {
    if (!searchQuery) return nodes;
    const query = searchQuery.toLowerCase();
    return nodes.filter(
      (node) =>
        node.id.toLowerCase().includes(query) ||
        node.content.toLowerCase().includes(query)
    );
  }, [nodes, searchQuery]);

  /**
   * Sort nodes based on selected field
   */
  const sortedNodes = useMemo(() => {
    const sorted = [...filteredNodes];
    sorted.sort((a, b) => {
      switch (sortField) {
        case 'relevance':
          return b.relevance - a.relevance;
        case 'recency':
          return (
            parseTimestamp(b.accessedAt).getTime() -
            parseTimestamp(a.accessedAt).getTime()
          );
        case 'access_count':
          return getAccessCount(b) - getAccessCount(a);
        default:
          return 0;
      }
    });
    return sorted;
  }, [filteredNodes, sortField]);

  /**
   * Group nodes by step if requested
   */
  const groupedNodes = useMemo(() => {
    if (!groupByStep) {
      return { all: sortedNodes };
    }

    const groups: Record<string, MemoryNode[]> = {};
    sortedNodes.forEach((node) => {
      const step = node.stepId || 'unknown';
      if (!groups[step]) {
        groups[step] = [];
      }
      groups[step].push(node);
    });

    return groups;
  }, [sortedNodes, groupByStep]);

  /**
   * Toggle node expansion state
   */
  const toggleNodeExpanded = (nodeId: string) => {
    const newExpanded = new Set(expandedNodeIds);
    if (newExpanded.has(nodeId)) {
      newExpanded.delete(nodeId);
    } else {
      newExpanded.add(nodeId);
    }
    setExpandedNodeIds(newExpanded);
  };

  /**
   * Copy node ID to clipboard and show feedback
   */
  const handleCopyNodeId = (nodeId: string) => {
    navigator.clipboard.writeText(nodeId);
    setCopiedId(nodeId);
    setTimeout(() => setCopiedId(null), 2000);
    onNodeClick?.(nodeId);
  };

  /**
   * Truncate text to specified length
   */
  const truncateText = (text: string, maxLength: number = 60): string => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '…';
  };

  /**
   * Format timestamp for display
   */
  const formatTimestamp = (timestamp: string): string => {
    const date = parseTimestamp(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSecs = Math.floor(diffMs / 1000);

    if (diffSecs < 60) return `${diffSecs}s ago`;
    const diffMins = Math.floor(diffSecs / 60);
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  // Render a single memory node card
  const renderNodeCard = (node: MemoryNode) => {
    const isExpanded = expandedNodeIds.has(node.id);
    const relevanceColors = getRelevanceColor(node.relevance);
    const sourceConfig = sourceTypeConfig[node.sourceType];
    const accessCount = getAccessCount(node);

    return (
      <div
        key={node.id}
        className={`
          border rounded-lg transition-all duration-200
          ${relevanceColors.bgColor}
          ${relevanceColors.borderColor}
          border
          hover:shadow-lg hover:shadow-slate-900
        `}
      >
        {/* Node header - always visible */}
        <div className="p-3 space-y-2">
          {/* Top row: expand, source badge, relevance bar, copy button */}
          <div className="flex items-center justify-between gap-2">
            {/* Expand/collapse button */}
            <button
              onClick={() => toggleNodeExpanded(node.id)}
              className="p-1 text-slate-400 hover:text-slate-300 hover:bg-slate-700 rounded transition-colors flex-shrink-0"
              title={isExpanded ? 'Collapse' : 'Expand'}
            >
              {isExpanded ? (
                <ChevronDown size={14} />
              ) : (
                <ChevronRight size={14} />
              )}
            </button>

            {/* Source type badge */}
            <span
              className={`
                px-2 py-0.5 rounded text-xs font-semibold flex-shrink-0
                ${sourceConfig.color}
                ${sourceConfig.bgColor}
              `}
              title={sourceConfig.label}
            >
              {sourceConfig.icon} {sourceConfig.label}
            </span>

            {/* Relevance score and bar */}
            <div className="flex-1 flex items-center gap-2 min-w-0">
              <div className="text-xs font-mono text-slate-300 flex-shrink-0">
                {(node.relevance * 100).toFixed(0)}%
              </div>
              <div className="flex-1 h-1 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-300 ${relevanceColors.barColor}`}
                  style={{ width: `${node.relevance * 100}%` }}
                />
              </div>
            </div>

            {/* Copy button */}
            <button
              onClick={() => handleCopyNodeId(node.id)}
              className="p-1 text-slate-400 hover:text-slate-300 hover:bg-slate-700 rounded transition-colors flex-shrink-0"
              title="Copy node ID"
            >
              {copiedId === node.id ? (
                <CheckCircle size={14} className="text-emerald-500" />
              ) : (
                <Copy size={14} />
              )}
            </button>
          </div>

          {/* Node ID and metadata row */}
          <div className="flex items-center justify-between gap-2">
            <div className="font-mono text-xs text-slate-400 truncate flex-1">
              {node.id.substring(0, 16)}
              {node.id.length > 16 ? '…' : ''}
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-500 flex-shrink-0">
              <span title="Access count">
                <Zap size={12} className="inline" /> {accessCount}
              </span>
              <span>{formatTimestamp(node.accessedAt)}</span>
            </div>
          </div>

          {/* Content preview */}
          <div className="text-xs text-slate-300 leading-relaxed">
            {truncateText(node.content, 80)}
          </div>
        </div>

        {/* Expanded content */}
        {isExpanded && (
          <div className="border-t border-slate-700 bg-slate-900 px-3 py-2 space-y-2">
            {/* Full content */}
            <div className="space-y-1">
              <h4 className="text-xs font-semibold text-slate-400 uppercase">
                Full Content
              </h4>
              <div className="text-xs text-slate-300 leading-relaxed max-h-32 overflow-y-auto bg-slate-950 p-2 rounded border border-slate-800 font-mono">
                {node.content}
              </div>
            </div>

            {/* Full node ID for easy copying */}
            <div className="space-y-1">
              <h4 className="text-xs font-semibold text-slate-400 uppercase">
                Node ID
              </h4>
              <div
                className="text-xs text-slate-300 font-mono bg-slate-950 p-2 rounded border border-slate-800 cursor-copy hover:border-blue-600 transition-colors break-all"
                onClick={() => handleCopyNodeId(node.id)}
                title="Click to copy"
              >
                {node.id}
              </div>
            </div>

            {/* Metadata if available */}
            {node.metadata && Object.keys(node.metadata).length > 0 && (
              <div className="space-y-1">
                <h4 className="text-xs font-semibold text-slate-400 uppercase">
                  Metadata
                </h4>
                <div className="text-xs text-slate-400 bg-slate-950 p-2 rounded border border-slate-800 max-h-24 overflow-y-auto">
                  <pre className="whitespace-pre-wrap break-words">
                    {JSON.stringify(node.metadata, null, 2)}
                  </pre>
                </div>
              </div>
            )}

            {/* Details row */}
            <div className="pt-2 border-t border-slate-800 grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-slate-500">Source:</span>
                <span className="text-slate-300 ml-1">{node.sourceType}</span>
              </div>
              <div>
                <span className="text-slate-500">Step:</span>
                <span className="text-slate-300 ml-1">{node.stepId}</span>
              </div>
              <div>
                <span className="text-slate-500">Relevance:</span>
                <span className="text-slate-300 ml-1">
                  {(node.relevance * 100).toFixed(1)}%
                </span>
              </div>
              <div>
                <span className="text-slate-500">Accessed:</span>
                <span className="text-slate-300 ml-1">
                  {formatTimestamp(node.accessedAt)}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Empty state
  if (nodes.length === 0) {
    return (
      <div
        className={`
          flex items-center justify-center p-8 rounded-lg
          bg-slate-900 border border-slate-800 text-slate-400
          ${className}
        `}
      >
        <div className="text-center">
          <div className="text-3xl mb-2">📚</div>
          <div className="text-sm">No memory nodes accessed</div>
        </div>
      </div>
    );
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Controls: Search, Sort, Summary */}
      <div className="space-y-3">
        {/* Search input */}
        <div className="relative">
          <Filter
            size={16}
            className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-500"
          />
          <input
            type="text"
            placeholder="Search nodes by ID or content…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-2 bg-slate-900 border border-slate-700 rounded text-slate-100 text-sm placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 transition-colors"
          />
        </div>

        {/* Sort controls and summary */}
        <div className="flex items-center justify-between gap-2">
          {/* Sort buttons */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 font-semibold uppercase">
              Sort by:
            </span>
            {(['relevance', 'recency', 'access_count'] as const).map(
              (field) => (
                <button
                  key={field}
                  onClick={() => setSortField(field)}
                  className={`
                    px-2 py-1 text-xs font-medium rounded transition-colors
                    inline-flex items-center gap-1
                    ${
                      sortField === field
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                    }
                  `}
                  title={`Sort by ${field}`}
                >
                  <ArrowUpDown size={12} />
                  {field === 'relevance'
                    ? 'Relevance'
                    : field === 'recency'
                    ? 'Recency'
                    : 'Access Count'}
                </button>
              )
            )}
          </div>

          {/* Summary stats */}
          <div className="text-xs text-slate-400 space-x-3">
            <span>
              Showing {sortedNodes.length} of {nodes.length} nodes
            </span>
            <span className="text-slate-600">•</span>
            <span>
              Avg Relevance:{' '}
              {(
                nodes.reduce((sum, n) => sum + n.relevance, 0) / nodes.length
              ).toFixed(2)}
            </span>
          </div>
        </div>
      </div>

      {/* Nodes container */}
      {Object.entries(groupedNodes).map(([groupKey, groupNodes]) => (
        <div key={groupKey} className="space-y-2">
          {/* Group header if grouping by step */}
          {groupByStep && groupKey !== 'all' && (
            <div className="px-3 py-1 bg-slate-900 border-l-2 border-blue-600 rounded">
              <h3 className="text-xs font-semibold text-slate-400 uppercase">
                Step {groupKey}
                <span className="text-slate-600 ml-2">
                  ({groupNodes.length} nodes)
                </span>
              </h3>
            </div>
          )}

          {/* Node cards grid */}
          <div className="grid gap-2 grid-cols-1 lg:grid-cols-2">
            {groupNodes.map((node) => renderNodeCard(node))}
          </div>
        </div>
      ))}
    </div>
  );
};

/**
 * Default export
 */
export default MemoryNodes;
