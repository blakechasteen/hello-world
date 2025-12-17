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

import React, { useState, useMemo, useCallback, useRef } from 'react';
import './FileTreeViewer.css';

// ============================================================================
// Type Definitions
// ============================================================================

interface FileNode {
  path: string;
  name: string;
  isDirectory: boolean;
  children?: FileNode[];
  modifiedBy?: string;  // agent ID
  status: 'modified' | 'read' | 'created' | 'deleted';
  lastModified: string;
}

interface FileTreeViewerProps {
  files: FileNode[];
  activeAgentId?: string;
  onFileSelect?: (path: string) => void;
  agentColorMap?: Record<string, string>;
}

interface ExpandedState {
  [path: string]: boolean;
}

interface FilterOptions {
  searchQuery: string;
  filterAgent?: string;
  filterStatus?: 'modified' | 'read' | 'created' | 'deleted' | 'all';
}

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
  modified: '●',  // filled circle
  read: '○',      // empty circle
  created: '+',   // plus
  deleted: '−',   // minus
} as const;

const STATUS_LABELS = {
  modified: 'Modified',
  read: 'Read-only',
  created: 'Created',
  deleted: 'Deleted',
} as const;

// ============================================================================
// Component
// ============================================================================

const FileTreeViewer: React.FC<FileTreeViewerProps> = ({
  files,
  activeAgentId,
  onFileSelect,
  agentColorMap = {},
}) => {
  // State Management
  const [expandedState, setExpandedState] = useState<ExpandedState>({});
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [filterOptions, setFilterOptions] = useState<FilterOptions>({
    searchQuery: '',
    filterStatus: 'all',
  });
  const [showDetails, setShowDetails] = useState(false);
  const detailsRef = useRef<HTMLDivElement>(null);

  // Memoized color mapping
  const getAgentColor = useCallback(
    (agentId?: string): string => {
      if (!agentId) return '#6B7280'; // gray default
      if (agentColorMap[agentId]) return agentColorMap[agentId];

      // Fallback to default palette based on hash
      const hash = agentId.charCodeAt(0);
      return DEFAULT_AGENT_COLORS[hash % DEFAULT_AGENT_COLORS.length];
    },
    [agentColorMap]
  );

  // Filter and search logic
  const filteredFiles = useMemo(() => {
    const filterNode = (node: FileNode): FileNode | null => {
      // Apply status filter
      if (
        filterOptions.filterStatus !== 'all' &&
        node.status !== filterOptions.filterStatus
      ) {
        // Still check children if directory
        if (node.isDirectory && node.children) {
          const filteredChildren = node.children
            .map(child => filterNode(child))
            .filter((child): child is FileNode => child !== null);
          if (filteredChildren.length === 0) return null;
          return { ...node, children: filteredChildren };
        }
        return null;
      }

      // Apply agent filter
      if (filterOptions.filterAgent && node.modifiedBy !== filterOptions.filterAgent) {
        if (node.isDirectory && node.children) {
          const filteredChildren = node.children
            .map(child => filterNode(child))
            .filter((child): child is FileNode => child !== null);
          if (filteredChildren.length === 0) return null;
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
            .filter((child): child is FileNode => child !== null);
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
          .filter((child): child is FileNode => child !== null);
        return { ...node, children: filteredChildren };
      }

      return node;
    };

    return files
      .map(file => filterNode(file))
      .filter((file): file is FileNode => file !== null);
  }, [files, filterOptions]);

  // Get all unique agents for filter dropdown
  const uniqueAgents = useMemo(() => {
    const agents = new Set<string>();
    const collectAgents = (nodes: FileNode[]) => {
      nodes.forEach(node => {
        if (node.modifiedBy) agents.add(node.modifiedBy);
        if (node.children) collectAgents(node.children);
      });
    };
    collectAgents(files);
    return Array.from(agents).sort();
  }, [files]);

  // Toggle folder expansion
  const toggleExpanded = useCallback((path: string) => {
    setExpandedState(prev => ({
      ...prev,
      [path]: !prev[path],
    }));
  }, []);

  // Expand/collapse all
  const expandAll = useCallback(() => {
    const allPaths: string[] = [];
    const collectPaths = (nodes: FileNode[]) => {
      nodes.forEach(node => {
        if (node.isDirectory) {
          allPaths.push(node.path);
          if (node.children) collectPaths(node.children);
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
  const handleFileSelect = useCallback((path: string, isDirectory: boolean) => {
    if (isDirectory) {
      toggleExpanded(path);
    } else {
      setSelectedPath(path);
      onFileSelect?.(path);
    }
  }, [toggleExpanded, onFileSelect]);

  // Find file node by path for details panel
  const findNodeByPath = useCallback((path: string, nodes = files): FileNode | null => {
    for (const node of nodes) {
      if (node.path === path) return node;
      if (node.children) {
        const found = findNodeByPath(path, node.children);
        if (found) return found;
      }
    }
    return null;
  }, [files]);

  const selectedNode = selectedPath ? findNodeByPath(selectedPath) : null;

  // Render file tree item
  const renderTreeItem = (node: FileNode, depth: number = 0): React.ReactNode => {
    const isDirectory = node.isDirectory;
    const isExpanded = expandedState[node.path];
    const isSelected = selectedPath === node.path;
    const agentColor = getAgentColor(node.modifiedBy);
    const statusIndicator = STATUS_INDICATORS[node.status];

    return (
      <div key={node.path} className="file-tree-item">
        {/* Tree item row */}
        <div
          className={`tree-row ${isSelected ? 'selected' : ''} status-${node.status}`}
          style={{ paddingLeft: `${depth * 16 + 8}px` }}
          onClick={() => handleFileSelect(node.path, isDirectory)}
        >
          {/* Hierarchy connector lines */}
          {depth > 0 && (
            <div
              className="tree-connector"
              style={{ left: `${depth * 16}px` }}
            />
          )}

          {/* Expand/collapse chevron */}
          {isDirectory && (
            <button
              className={`expand-button ${isExpanded ? 'expanded' : ''}`}
              onClick={(e) => {
                e.stopPropagation();
                toggleExpanded(node.path);
              }}
              aria-label={isExpanded ? 'Collapse' : 'Expand'}
              title={isExpanded ? 'Collapse folder' : 'Expand folder'}
            >
              ▶
            </button>
          )}
          {!isDirectory && <div className="expand-spacer" />}

          {/* Agent color indicator */}
          <div
            className="agent-indicator"
            style={{ backgroundColor: agentColor }}
            title={node.modifiedBy || 'Unknown agent'}
          />

          {/* File/folder icon */}
          <span className="file-icon">
            {isDirectory ? '📁' : '📄'}
          </span>

          {/* Filename */}
          <span className="file-name">{node.name}</span>

          {/* Status indicator */}
          <span
            className={`status-indicator status-${node.status}`}
            title={STATUS_LABELS[node.status]}
          >
            {statusIndicator}
          </span>

          {/* Modified by agent */}
          {node.modifiedBy && (
            <span className="agent-label" style={{ color: agentColor }}>
              {node.modifiedBy}
            </span>
          )}
        </div>

        {/* Children (if expanded) */}
        {isDirectory && isExpanded && node.children && (
          <div className="tree-children">
            {node.children.map(child => renderTreeItem(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  // Render details panel for selected file
  const renderDetailsPanel = (): React.ReactNode => {
    if (!selectedNode) return null;

    const agentColor = getAgentColor(selectedNode.modifiedBy);
    const statusLabel = STATUS_LABELS[selectedNode.status];

    return (
      <div className="details-panel" ref={detailsRef}>
        <div className="details-header">
          <h3>File Details</h3>
          <button
            className="close-button"
            onClick={() => {
              setSelectedPath(null);
              setShowDetails(false);
            }}
            aria-label="Close details"
          >
            ✕
          </button>
        </div>

        <div className="details-content">
          {/* File path */}
          <div className="detail-row">
            <label>Path:</label>
            <code className="file-path">{selectedNode.path}</code>
          </div>

          {/* Filename */}
          <div className="detail-row">
            <label>Name:</label>
            <span>{selectedNode.name}</span>
          </div>

          {/* Type */}
          <div className="detail-row">
            <label>Type:</label>
            <span>{selectedNode.isDirectory ? 'Directory' : 'File'}</span>
          </div>

          {/* Status */}
          <div className="detail-row">
            <label>Status:</label>
            <span className={`status-badge status-${selectedNode.status}`}>
              <span className="status-dot">
                {STATUS_INDICATORS[selectedNode.status]}
              </span>
              {statusLabel}
            </span>
          </div>

          {/* Modified by agent */}
          {selectedNode.modifiedBy && (
            <div className="detail-row">
              <label>Modified By:</label>
              <div className="agent-info">
                <div
                  className="agent-color-box"
                  style={{ backgroundColor: agentColor }}
                />
                <span>{selectedNode.modifiedBy}</span>
              </div>
            </div>
          )}

          {/* Last modified timestamp */}
          <div className="detail-row">
            <label>Last Modified:</label>
            <time>{new Date(selectedNode.lastModified).toLocaleString()}</time>
          </div>

          {/* Child count for directories */}
          {selectedNode.isDirectory && selectedNode.children && (
            <div className="detail-row">
              <label>Items:</label>
              <span>{selectedNode.children.length} file(s)/folder(s)</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="file-tree-viewer">
      {/* Toolbar */}
      <div className="toolbar">
        {/* Search input */}
        <input
          type="text"
          className="search-input"
          placeholder="Search files..."
          value={filterOptions.searchQuery}
          onChange={(e) =>
            setFilterOptions(prev => ({
              ...prev,
              searchQuery: e.target.value,
            }))
          }
          aria-label="Search files"
        />

        {/* Status filter */}
        <select
          className="filter-select"
          value={filterOptions.filterStatus || 'all'}
          onChange={(e) =>
            setFilterOptions(prev => ({
              ...prev,
              filterStatus: (e.target.value as any) || 'all',
            }))
          }
          aria-label="Filter by status"
        >
          <option value="all">All Status</option>
          <option value="modified">Modified</option>
          <option value="read">Read-only</option>
          <option value="created">Created</option>
          <option value="deleted">Deleted</option>
        </select>

        {/* Agent filter */}
        {uniqueAgents.length > 0 && (
          <select
            className="filter-select"
            value={filterOptions.filterAgent || ''}
            onChange={(e) =>
              setFilterOptions(prev => ({
                ...prev,
                filterAgent: e.target.value || undefined,
              }))
            }
            aria-label="Filter by agent"
          >
            <option value="">All Agents</option>
            {uniqueAgents.map(agent => (
              <option key={agent} value={agent}>
                {agent}
              </option>
            ))}
          </select>
        )}

        {/* Expand/collapse all buttons */}
        <div className="button-group">
          <button
            className="icon-button"
            onClick={expandAll}
            title="Expand all folders"
            aria-label="Expand all"
          >
            ▼
          </button>
          <button
            className="icon-button"
            onClick={collapseAll}
            title="Collapse all folders"
            aria-label="Collapse all"
          >
            ▶
          </button>
        </div>
      </div>

      {/* Main content area */}
      <div className="content-area">
        {/* File tree */}
        <div className="tree-container">
          {filteredFiles.length === 0 ? (
            <div className="empty-state">
              <p>No files match your filters</p>
            </div>
          ) : (
            <div className="tree-root">
              {filteredFiles.map(file => renderTreeItem(file))}
            </div>
          )}
        </div>

        {/* Details panel */}
        {selectedNode && showDetails && renderDetailsPanel()}
      </div>

      {/* Toggle details panel button */}
      {selectedNode && (
        <button
          className={`details-toggle ${showDetails ? 'active' : ''}`}
          onClick={() => setShowDetails(!showDetails)}
          title={showDetails ? 'Hide details' : 'Show details'}
          aria-label={showDetails ? 'Hide details' : 'Show details'}
        >
          ℹ
        </button>
      )}
    </div>
  );
};

export default FileTreeViewer;
