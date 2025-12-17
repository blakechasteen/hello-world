import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useMemo } from 'react';
import { ChevronDown, ChevronRight, Copy, CheckCircle, Filter, ArrowUpDown, Zap, } from 'lucide-react';
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
export const MemoryNodes = ({ nodes, groupByStep = false, onNodeClick, className = '', }) => {
    // State management
    const [expandedNodeIds, setExpandedNodeIds] = useState(new Set());
    const [sortField, setSortField] = useState('relevance');
    const [searchQuery, setSearchQuery] = useState('');
    const [copiedId, setCopiedId] = useState(null);
    /**
     * Source type badge configuration mapping
     */
    const sourceTypeConfig = {
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
    const getRelevanceColor = (relevance) => {
        if (relevance >= 0.9) {
            return {
                bgColor: 'bg-emerald-950',
                borderColor: 'border-emerald-700',
                barColor: 'bg-emerald-600',
            };
        }
        else if (relevance >= 0.7) {
            return {
                bgColor: 'bg-blue-950',
                borderColor: 'border-blue-700',
                barColor: 'bg-blue-600',
            };
        }
        else {
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
    const getAccessCount = (node) => {
        return node.metadata?.access_count ?? 1;
    };
    /**
     * Parse timestamp to Date object
     */
    const parseTimestamp = (timestamp) => {
        return new Date(timestamp);
    };
    /**
     * Filter nodes by search query
     */
    const filteredNodes = useMemo(() => {
        if (!searchQuery)
            return nodes;
        const query = searchQuery.toLowerCase();
        return nodes.filter((node) => node.id.toLowerCase().includes(query) ||
            node.content.toLowerCase().includes(query));
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
                    return (parseTimestamp(b.accessedAt).getTime() -
                        parseTimestamp(a.accessedAt).getTime());
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
        const groups = {};
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
    const toggleNodeExpanded = (nodeId) => {
        const newExpanded = new Set(expandedNodeIds);
        if (newExpanded.has(nodeId)) {
            newExpanded.delete(nodeId);
        }
        else {
            newExpanded.add(nodeId);
        }
        setExpandedNodeIds(newExpanded);
    };
    /**
     * Copy node ID to clipboard and show feedback
     */
    const handleCopyNodeId = (nodeId) => {
        navigator.clipboard.writeText(nodeId);
        setCopiedId(nodeId);
        setTimeout(() => setCopiedId(null), 2000);
        onNodeClick?.(nodeId);
    };
    /**
     * Truncate text to specified length
     */
    const truncateText = (text, maxLength = 60) => {
        if (text.length <= maxLength)
            return text;
        return text.substring(0, maxLength) + '…';
    };
    /**
     * Format timestamp for display
     */
    const formatTimestamp = (timestamp) => {
        const date = parseTimestamp(timestamp);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffSecs = Math.floor(diffMs / 1000);
        if (diffSecs < 60)
            return `${diffSecs}s ago`;
        const diffMins = Math.floor(diffSecs / 60);
        if (diffMins < 60)
            return `${diffMins}m ago`;
        const diffHours = Math.floor(diffMins / 60);
        if (diffHours < 24)
            return `${diffHours}h ago`;
        const diffDays = Math.floor(diffHours / 24);
        return `${diffDays}d ago`;
    };
    // Render a single memory node card
    const renderNodeCard = (node) => {
        const isExpanded = expandedNodeIds.has(node.id);
        const relevanceColors = getRelevanceColor(node.relevance);
        const sourceConfig = sourceTypeConfig[node.sourceType];
        const accessCount = getAccessCount(node);
        return (_jsxs("div", { className: `
          border rounded-lg transition-all duration-200
          ${relevanceColors.bgColor}
          ${relevanceColors.borderColor}
          border
          hover:shadow-lg hover:shadow-slate-900
        `, children: [_jsxs("div", { className: "p-3 space-y-2", children: [_jsxs("div", { className: "flex items-center justify-between gap-2", children: [_jsx("button", { onClick: () => toggleNodeExpanded(node.id), className: "p-1 text-slate-400 hover:text-slate-300 hover:bg-slate-700 rounded transition-colors flex-shrink-0", title: isExpanded ? 'Collapse' : 'Expand', children: isExpanded ? (_jsx(ChevronDown, { size: 14 })) : (_jsx(ChevronRight, { size: 14 })) }), _jsxs("span", { className: `
                px-2 py-0.5 rounded text-xs font-semibold flex-shrink-0
                ${sourceConfig.color}
                ${sourceConfig.bgColor}
              `, title: sourceConfig.label, children: [sourceConfig.icon, " ", sourceConfig.label] }), _jsxs("div", { className: "flex-1 flex items-center gap-2 min-w-0", children: [_jsxs("div", { className: "text-xs font-mono text-slate-300 flex-shrink-0", children: [(node.relevance * 100).toFixed(0), "%"] }), _jsx("div", { className: "flex-1 h-1 bg-slate-700 rounded-full overflow-hidden", children: _jsx("div", { className: `h-full transition-all duration-300 ${relevanceColors.barColor}`, style: { width: `${node.relevance * 100}%` } }) })] }), _jsx("button", { onClick: () => handleCopyNodeId(node.id), className: "p-1 text-slate-400 hover:text-slate-300 hover:bg-slate-700 rounded transition-colors flex-shrink-0", title: "Copy node ID", children: copiedId === node.id ? (_jsx(CheckCircle, { size: 14, className: "text-emerald-500" })) : (_jsx(Copy, { size: 14 })) })] }), _jsxs("div", { className: "flex items-center justify-between gap-2", children: [_jsxs("div", { className: "font-mono text-xs text-slate-400 truncate flex-1", children: [node.id.substring(0, 16), node.id.length > 16 ? '…' : ''] }), _jsxs("div", { className: "flex items-center gap-2 text-xs text-slate-500 flex-shrink-0", children: [_jsxs("span", { title: "Access count", children: [_jsx(Zap, { size: 12, className: "inline" }), " ", accessCount] }), _jsx("span", { children: formatTimestamp(node.accessedAt) })] })] }), _jsx("div", { className: "text-xs text-slate-300 leading-relaxed", children: truncateText(node.content, 80) })] }), isExpanded && (_jsxs("div", { className: "border-t border-slate-700 bg-slate-900 px-3 py-2 space-y-2", children: [_jsxs("div", { className: "space-y-1", children: [_jsx("h4", { className: "text-xs font-semibold text-slate-400 uppercase", children: "Full Content" }), _jsx("div", { className: "text-xs text-slate-300 leading-relaxed max-h-32 overflow-y-auto bg-slate-950 p-2 rounded border border-slate-800 font-mono", children: node.content })] }), _jsxs("div", { className: "space-y-1", children: [_jsx("h4", { className: "text-xs font-semibold text-slate-400 uppercase", children: "Node ID" }), _jsx("div", { className: "text-xs text-slate-300 font-mono bg-slate-950 p-2 rounded border border-slate-800 cursor-copy hover:border-blue-600 transition-colors break-all", onClick: () => handleCopyNodeId(node.id), title: "Click to copy", children: node.id })] }), node.metadata && Object.keys(node.metadata).length > 0 && (_jsxs("div", { className: "space-y-1", children: [_jsx("h4", { className: "text-xs font-semibold text-slate-400 uppercase", children: "Metadata" }), _jsx("div", { className: "text-xs text-slate-400 bg-slate-950 p-2 rounded border border-slate-800 max-h-24 overflow-y-auto", children: _jsx("pre", { className: "whitespace-pre-wrap break-words", children: JSON.stringify(node.metadata, null, 2) }) })] })), _jsxs("div", { className: "pt-2 border-t border-slate-800 grid grid-cols-2 gap-2 text-xs", children: [_jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Source:" }), _jsx("span", { className: "text-slate-300 ml-1", children: node.sourceType })] }), _jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Step:" }), _jsx("span", { className: "text-slate-300 ml-1", children: node.stepId })] }), _jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Relevance:" }), _jsxs("span", { className: "text-slate-300 ml-1", children: [(node.relevance * 100).toFixed(1), "%"] })] }), _jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Accessed:" }), _jsx("span", { className: "text-slate-300 ml-1", children: formatTimestamp(node.accessedAt) })] })] })] }))] }, node.id));
    };
    // Empty state
    if (nodes.length === 0) {
        return (_jsx("div", { className: `
          flex items-center justify-center p-8 rounded-lg
          bg-slate-900 border border-slate-800 text-slate-400
          ${className}
        `, children: _jsxs("div", { className: "text-center", children: [_jsx("div", { className: "text-3xl mb-2", children: "\uD83D\uDCDA" }), _jsx("div", { className: "text-sm", children: "No memory nodes accessed" })] }) }));
    }
    return (_jsxs("div", { className: `space-y-4 ${className}`, children: [_jsxs("div", { className: "space-y-3", children: [_jsxs("div", { className: "relative", children: [_jsx(Filter, { size: 16, className: "absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-500" }), _jsx("input", { type: "text", placeholder: "Search nodes by ID or content\u2026", value: searchQuery, onChange: (e) => setSearchQuery(e.target.value), className: "w-full pl-9 pr-3 py-2 bg-slate-900 border border-slate-700 rounded text-slate-100 text-sm placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 transition-colors" })] }), _jsxs("div", { className: "flex items-center justify-between gap-2", children: [_jsxs("div", { className: "flex items-center gap-2", children: [_jsx("span", { className: "text-xs text-slate-500 font-semibold uppercase", children: "Sort by:" }), ['relevance', 'recency', 'access_count'].map((field) => (_jsxs("button", { onClick: () => setSortField(field), className: `
                    px-2 py-1 text-xs font-medium rounded transition-colors
                    inline-flex items-center gap-1
                    ${sortField === field
                                            ? 'bg-blue-600 text-white'
                                            : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}
                  `, title: `Sort by ${field}`, children: [_jsx(ArrowUpDown, { size: 12 }), field === 'relevance'
                                                ? 'Relevance'
                                                : field === 'recency'
                                                    ? 'Recency'
                                                    : 'Access Count'] }, field)))] }), _jsxs("div", { className: "text-xs text-slate-400 space-x-3", children: [_jsxs("span", { children: ["Showing ", sortedNodes.length, " of ", nodes.length, " nodes"] }), _jsx("span", { className: "text-slate-600", children: "\u2022" }), _jsxs("span", { children: ["Avg Relevance:", ' ', (nodes.reduce((sum, n) => sum + n.relevance, 0) / nodes.length).toFixed(2)] })] })] })] }), Object.entries(groupedNodes).map(([groupKey, groupNodes]) => (_jsxs("div", { className: "space-y-2", children: [groupByStep && groupKey !== 'all' && (_jsx("div", { className: "px-3 py-1 bg-slate-900 border-l-2 border-blue-600 rounded", children: _jsxs("h3", { className: "text-xs font-semibold text-slate-400 uppercase", children: ["Step ", groupKey, _jsxs("span", { className: "text-slate-600 ml-2", children: ["(", groupNodes.length, " nodes)"] })] }) })), _jsx("div", { className: "grid gap-2 grid-cols-1 lg:grid-cols-2", children: groupNodes.map((node) => renderNodeCard(node)) })] }, groupKey)))] }));
};
/**
 * Default export
 */
export default MemoryNodes;
//# sourceMappingURL=MemoryNodes.js.map