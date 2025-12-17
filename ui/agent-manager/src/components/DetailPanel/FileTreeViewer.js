import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * FileTreeViewer Component
 *
 * Color-coded file tree showing files being modified by agents.
 * Features hierarchical navigation, agent tracking, status indicators,
 * search/filter functionality, and animated interactions.
 *
 * @component
 * @example
 * ```tsx
 * <FileTreeViewer
 *   files={fileTree}
 *   activeAgentId="agent-1"
 *   onFileSelect={(path) => console.log(path)}
 *   agentColorMap={colorMap}
 * />
 * ```
 */
import { useState, useMemo, useCallback, useRef } from 'react';
import './FileTreeViewer.css';
// ============================================================================
// Constants
// ============================================================================
const DEFAULT_AGENT_COLORS = [
    '#3B82F6', // blue
    '#10B981', // emerald
    '#F59E0B', // amber
    '#EF4444', // red
    '#8B5CF6', // violet
    '#EC4899', // pink
    '#06B6D4', // cyan
    '#84CC16', // lime
];
const STATUS_INDICATORS = {
    modified: '●', // filled circle
    read: '○', // empty circle
    created: '+', // plus
    deleted: '−', // minus
};
const STATUS_LABELS = {
    modified: 'Modified',
    read: 'Read-only',
    created: 'Created',
    deleted: 'Deleted',
};
// ============================================================================
// Component
// ============================================================================
const FileTreeViewer = ({ files, activeAgentId, onFileSelect, agentColorMap = {}, }) => {
    // State Management
    const [expandedState, setExpandedState] = useState({});
    const [selectedPath, setSelectedPath] = useState(null);
    const [filterOptions, setFilterOptions] = useState({
        searchQuery: '',
        filterStatus: 'all',
    });
    const [showDetails, setShowDetails] = useState(false);
    const detailsRef = useRef(null);
    // Memoized color mapping
    const getAgentColor = useCallback((agentId) => {
        if (!agentId)
            return '#6B7280'; // gray default
        if (agentColorMap[agentId])
            return agentColorMap[agentId];
        // Fallback to default palette based on hash
        const hash = agentId.charCodeAt(0);
        return DEFAULT_AGENT_COLORS[hash % DEFAULT_AGENT_COLORS.length];
    }, [agentColorMap]);
    // Filter and search logic
    const filteredFiles = useMemo(() => {
        const filterNode = (node) => {
            // Apply status filter
            if (filterOptions.filterStatus !== 'all' &&
                node.status !== filterOptions.filterStatus) {
                // Still check children if directory
                if (node.isDirectory && node.children) {
                    const filteredChildren = node.children
                        .map(child => filterNode(child))
                        .filter((child) => child !== null);
                    if (filteredChildren.length === 0)
                        return null;
                    return { ...node, children: filteredChildren };
                }
                return null;
            }
            // Apply agent filter
            if (filterOptions.filterAgent && node.modifiedBy !== filterOptions.filterAgent) {
                if (node.isDirectory && node.children) {
                    const filteredChildren = node.children
                        .map(child => filterNode(child))
                        .filter((child) => child !== null);
                    if (filteredChildren.length === 0)
                        return null;
                    return { ...node, children: filteredChildren };
                }
                return null;
            }
            // Apply search filter
            if (filterOptions.searchQuery) {
                const query = filterOptions.searchQuery.toLowerCase();
                const matches = node.name.toLowerCase().includes(query) ||
                    node.path.toLowerCase().includes(query);
                if (node.isDirectory && node.children) {
                    const filteredChildren = node.children
                        .map(child => filterNode(child))
                        .filter((child) => child !== null);
                    if (matches || filteredChildren.length > 0) {
                        return { ...node, children: filteredChildren };
                    }
                    return null;
                }
                return matches ? node : null;
            }
            // No filters applied, process children if directory
            if (node.isDirectory && node.children) {
                const filteredChildren = node.children
                    .map(child => filterNode(child))
                    .filter((child) => child !== null);
                return { ...node, children: filteredChildren };
            }
            return node;
        };
        return files
            .map(file => filterNode(file))
            .filter((file) => file !== null);
    }, [files, filterOptions]);
    // Get all unique agents for filter dropdown
    const uniqueAgents = useMemo(() => {
        const agents = new Set();
        const collectAgents = (nodes) => {
            nodes.forEach(node => {
                if (node.modifiedBy)
                    agents.add(node.modifiedBy);
                if (node.children)
                    collectAgents(node.children);
            });
        };
        collectAgents(files);
        return Array.from(agents).sort();
    }, [files]);
    // Toggle folder expansion
    const toggleExpanded = useCallback((path) => {
        setExpandedState(prev => ({
            ...prev,
            [path]: !prev[path],
        }));
    }, []);
    // Expand/collapse all
    const expandAll = useCallback(() => {
        const allPaths = [];
        const collectPaths = (nodes) => {
            nodes.forEach(node => {
                if (node.isDirectory) {
                    allPaths.push(node.path);
                    if (node.children)
                        collectPaths(node.children);
                }
            });
        };
        collectPaths(filteredFiles);
        setExpandedState(Object.fromEntries(allPaths.map(path => [path, true])));
    }, [filteredFiles]);
    const collapseAll = useCallback(() => {
        setExpandedState({});
    }, []);
    // Handle file selection
    const handleFileSelect = useCallback((path, isDirectory) => {
        if (isDirectory) {
            toggleExpanded(path);
        }
        else {
            setSelectedPath(path);
            onFileSelect?.(path);
        }
    }, [toggleExpanded, onFileSelect]);
    // Find file node by path for details panel
    const findNodeByPath = useCallback((path, nodes = files) => {
        for (const node of nodes) {
            if (node.path === path)
                return node;
            if (node.children) {
                const found = findNodeByPath(path, node.children);
                if (found)
                    return found;
            }
        }
        return null;
    }, [files]);
    const selectedNode = selectedPath ? findNodeByPath(selectedPath) : null;
    // Render file tree item
    const renderTreeItem = (node, depth = 0) => {
        const isDirectory = node.isDirectory;
        const isExpanded = expandedState[node.path];
        const isSelected = selectedPath === node.path;
        const agentColor = getAgentColor(node.modifiedBy);
        const statusIndicator = STATUS_INDICATORS[node.status];
        return (_jsxs("div", { className: "file-tree-item", children: [_jsxs("div", { className: `tree-row ${isSelected ? 'selected' : ''} status-${node.status}`, style: { paddingLeft: `${depth * 16 + 8}px` }, onClick: () => handleFileSelect(node.path, isDirectory), children: [depth > 0 && (_jsx("div", { className: "tree-connector", style: { left: `${depth * 16}px` } })), isDirectory && (_jsx("button", { className: `expand-button ${isExpanded ? 'expanded' : ''}`, onClick: (e) => {
                                e.stopPropagation();
                                toggleExpanded(node.path);
                            }, "aria-label": isExpanded ? 'Collapse' : 'Expand', title: isExpanded ? 'Collapse folder' : 'Expand folder', children: "\u25B6" })), !isDirectory && _jsx("div", { className: "expand-spacer" }), _jsx("div", { className: "agent-indicator", style: { backgroundColor: agentColor }, title: node.modifiedBy || 'Unknown agent' }), _jsx("span", { className: "file-icon", children: isDirectory ? '📁' : '📄' }), _jsx("span", { className: "file-name", children: node.name }), _jsx("span", { className: `status-indicator status-${node.status}`, title: STATUS_LABELS[node.status], children: statusIndicator }), node.modifiedBy && (_jsx("span", { className: "agent-label", style: { color: agentColor }, children: node.modifiedBy }))] }), isDirectory && isExpanded && node.children && (_jsx("div", { className: "tree-children", children: node.children.map(child => renderTreeItem(child, depth + 1)) }))] }, node.path));
    };
    // Render details panel for selected file
    const renderDetailsPanel = () => {
        if (!selectedNode)
            return null;
        const agentColor = getAgentColor(selectedNode.modifiedBy);
        const statusLabel = STATUS_LABELS[selectedNode.status];
        return (_jsxs("div", { className: "details-panel", ref: detailsRef, children: [_jsxs("div", { className: "details-header", children: [_jsx("h3", { children: "File Details" }), _jsx("button", { className: "close-button", onClick: () => {
                                setSelectedPath(null);
                                setShowDetails(false);
                            }, "aria-label": "Close details", children: "\u2715" })] }), _jsxs("div", { className: "details-content", children: [_jsxs("div", { className: "detail-row", children: [_jsx("label", { children: "Path:" }), _jsx("code", { className: "file-path", children: selectedNode.path })] }), _jsxs("div", { className: "detail-row", children: [_jsx("label", { children: "Name:" }), _jsx("span", { children: selectedNode.name })] }), _jsxs("div", { className: "detail-row", children: [_jsx("label", { children: "Type:" }), _jsx("span", { children: selectedNode.isDirectory ? 'Directory' : 'File' })] }), _jsxs("div", { className: "detail-row", children: [_jsx("label", { children: "Status:" }), _jsxs("span", { className: `status-badge status-${selectedNode.status}`, children: [_jsx("span", { className: "status-dot", children: STATUS_INDICATORS[selectedNode.status] }), statusLabel] })] }), selectedNode.modifiedBy && (_jsxs("div", { className: "detail-row", children: [_jsx("label", { children: "Modified By:" }), _jsxs("div", { className: "agent-info", children: [_jsx("div", { className: "agent-color-box", style: { backgroundColor: agentColor } }), _jsx("span", { children: selectedNode.modifiedBy })] })] })), _jsxs("div", { className: "detail-row", children: [_jsx("label", { children: "Last Modified:" }), _jsx("time", { children: new Date(selectedNode.lastModified).toLocaleString() })] }), selectedNode.isDirectory && selectedNode.children && (_jsxs("div", { className: "detail-row", children: [_jsx("label", { children: "Items:" }), _jsxs("span", { children: [selectedNode.children.length, " file(s)/folder(s)"] })] }))] })] }));
    };
    return (_jsxs("div", { className: "file-tree-viewer", children: [_jsxs("div", { className: "toolbar", children: [_jsx("input", { type: "text", className: "search-input", placeholder: "Search files...", value: filterOptions.searchQuery, onChange: (e) => setFilterOptions(prev => ({
                            ...prev,
                            searchQuery: e.target.value,
                        })), "aria-label": "Search files" }), _jsxs("select", { className: "filter-select", value: filterOptions.filterStatus || 'all', onChange: (e) => setFilterOptions(prev => ({
                            ...prev,
                            filterStatus: e.target.value || 'all',
                        })), "aria-label": "Filter by status", children: [_jsx("option", { value: "all", children: "All Status" }), _jsx("option", { value: "modified", children: "Modified" }), _jsx("option", { value: "read", children: "Read-only" }), _jsx("option", { value: "created", children: "Created" }), _jsx("option", { value: "deleted", children: "Deleted" })] }), uniqueAgents.length > 0 && (_jsxs("select", { className: "filter-select", value: filterOptions.filterAgent || '', onChange: (e) => setFilterOptions(prev => ({
                            ...prev,
                            filterAgent: e.target.value || undefined,
                        })), "aria-label": "Filter by agent", children: [_jsx("option", { value: "", children: "All Agents" }), uniqueAgents.map(agent => (_jsx("option", { value: agent, children: agent }, agent)))] })), _jsxs("div", { className: "button-group", children: [_jsx("button", { className: "icon-button", onClick: expandAll, title: "Expand all folders", "aria-label": "Expand all", children: "\u25BC" }), _jsx("button", { className: "icon-button", onClick: collapseAll, title: "Collapse all folders", "aria-label": "Collapse all", children: "\u25B6" })] })] }), _jsxs("div", { className: "content-area", children: [_jsx("div", { className: "tree-container", children: filteredFiles.length === 0 ? (_jsx("div", { className: "empty-state", children: _jsx("p", { children: "No files match your filters" }) })) : (_jsx("div", { className: "tree-root", children: filteredFiles.map(file => renderTreeItem(file)) })) }), selectedNode && showDetails && renderDetailsPanel()] }), selectedNode && (_jsx("button", { className: `details-toggle ${showDetails ? 'active' : ''}`, onClick: () => setShowDetails(!showDetails), title: showDetails ? 'Hide details' : 'Show details', "aria-label": showDetails ? 'Hide details' : 'Show details', children: "\u2139" }))] }));
};
export default FileTreeViewer;
//# sourceMappingURL=FileTreeViewer.js.map