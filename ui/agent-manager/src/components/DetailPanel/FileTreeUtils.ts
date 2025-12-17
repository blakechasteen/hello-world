/**
 * FileTreeViewer Utilities
 *
 * Helper functions for building, transforming, and working with file trees.
 */

// ============================================================================
// Type Definitions
// ============================================================================

export interface FileNode {
  path: string;
  name: string;
  isDirectory: boolean;
  children?: FileNode[];
  modifiedBy?: string;
  status: 'modified' | 'read' | 'created' | 'deleted';
  lastModified: string;
}

export interface FileTreeStats {
  totalFiles: number;
  totalDirs: number;
  filesByStatus: Record<string, number>;
  filesByAgent: Record<string, number>;
  maxDepth: number;
  averageDepth: number;
}

export interface PathInfo {
  directory: string;
  filename: string;
  extension: string;
  segments: string[];
}

// ============================================================================
// Tree Building & Transformation
// ============================================================================

/**
 * Convert a flat list of file paths to a hierarchical tree structure
 *
 * @example
 * ```ts
 * const paths = [
 *   { path: '/src/components/Header.tsx', status: 'modified', modifiedBy: 'agent-1' },
 *   { path: '/src/utils/format.ts', status: 'created', modifiedBy: 'agent-2' }
 * ];
 * const tree = buildTreeFromPaths(paths);
 * ```
 */
export function buildTreeFromPaths(
  paths: Array<{
    path: string;
    status: FileNode['status'];
    modifiedBy?: string;
    lastModified?: string;
  }>
): FileNode[] {
  const nodeMap = new Map<string, FileNode>();

  // Create all nodes
  paths.forEach(item => {
    const segments = item.path.split('/').filter(Boolean);
    let currentPath = '';

    segments.forEach((segment, index) => {
      currentPath += `/${segment}`;
      const isDirectory = index < segments.length - 1;

      if (!nodeMap.has(currentPath)) {
        nodeMap.set(currentPath, {
          path: currentPath,
          name: segment,
          isDirectory,
          modifiedBy: item.modifiedBy,
          status: item.status,
          lastModified: item.lastModified || new Date().toISOString(),
        });
      }

      // Update parent directory if we're at the leaf
      if (index === segments.length - 1) {
        const node = nodeMap.get(currentPath)!;
        node.modifiedBy = item.modifiedBy;
        node.status = item.status;
        node.lastModified = item.lastModified || new Date().toISOString();
      }
    });
  });

  // Build tree structure
  const rootPaths = new Set<string>();
  nodeMap.forEach((node, path) => {
    if (path.split('/').filter(Boolean).length === 1) {
      rootPaths.add(path);
    }
  });

  // Add children to directories
  nodeMap.forEach((node, path) => {
    if (node.isDirectory) {
      node.children = [];
    }
  });

  nodeMap.forEach((node, path) => {
    if (!node.isDirectory) {
      const parentPath = path.substring(0, path.lastIndexOf('/'));
      const parent = nodeMap.get(parentPath);
      if (parent && parent.children) {
        parent.children.push(node);
      }
    }
  });

  // Extract roots
  return Array.from(rootPaths)
    .map(path => nodeMap.get(path)!)
    .sort((a, b) => a.name.localeCompare(b.name));
}

/**
 * Find a node in the tree by path
 */
export function findNodeByPath(
  path: string,
  nodes: FileNode[]
): FileNode | null {
  for (const node of nodes) {
    if (node.path === path) return node;
    if (node.children) {
      const found = findNodeByPath(path, node.children);
      if (found) return found;
    }
  }
  return null;
}

/**
 * Get all paths in the tree (flat list)
 */
export function getAllPaths(nodes: FileNode[]): string[] {
  const paths: string[] = [];

  const collect = (node: FileNode) => {
    paths.push(node.path);
    if (node.children) {
      node.children.forEach(collect);
    }
  };

  nodes.forEach(collect);
  return paths;
}

/**
 * Get all files (non-directories) in the tree
 */
export function getAllFiles(nodes: FileNode[]): FileNode[] {
  const files: FileNode[] = [];

  const collect = (node: FileNode) => {
    if (!node.isDirectory) {
      files.push(node);
    }
    if (node.children) {
      node.children.forEach(collect);
    }
  };

  nodes.forEach(collect);
  return files;
}

/**
 * Get all directories in the tree
 */
export function getAllDirectories(nodes: FileNode[]): FileNode[] {
  const dirs: FileNode[] = [];

  const collect = (node: FileNode) => {
    if (node.isDirectory) {
      dirs.push(node);
    }
    if (node.children) {
      node.children.forEach(collect);
    }
  };

  nodes.forEach(collect);
  return dirs;
}

// ============================================================================
// Tree Filtering & Searching
// ============================================================================

/**
 * Filter tree by status
 */
export function filterByStatus(
  nodes: FileNode[],
  status: FileNode['status']
): FileNode[] {
  const result: FileNode[] = [];

  const process = (node: FileNode): FileNode | null => {
    if (node.status === status) {
      const copy = { ...node };
      if (node.children) {
        copy.children = node.children
          .map(process)
          .filter((child): child is FileNode => child !== null);
      }
      return copy;
    }

    if (node.isDirectory && node.children) {
      const filteredChildren = node.children
        .map(process)
        .filter((child): child is FileNode => child !== null);
      if (filteredChildren.length > 0) {
        return { ...node, children: filteredChildren };
      }
    }

    return null;
  };

  nodes.forEach(node => {
    const filtered = process(node);
    if (filtered) result.push(filtered);
  });

  return result;
}

/**
 * Filter tree by agent
 */
export function filterByAgent(
  nodes: FileNode[],
  agentId: string
): FileNode[] {
  const result: FileNode[] = [];

  const process = (node: FileNode): FileNode | null => {
    if (node.modifiedBy === agentId) {
      const copy = { ...node };
      if (node.children) {
        copy.children = node.children
          .map(process)
          .filter((child): child is FileNode => child !== null);
      }
      return copy;
    }

    if (node.isDirectory && node.children) {
      const filteredChildren = node.children
        .map(process)
        .filter((child): child is FileNode => child !== null);
      if (filteredChildren.length > 0) {
        return { ...node, children: filteredChildren };
      }
    }

    return null;
  };

  nodes.forEach(node => {
    const filtered = process(node);
    if (filtered) result.push(filtered);
  });

  return result;
}

/**
 * Search tree by filename or path (case-insensitive)
 */
export function searchTree(
  nodes: FileNode[],
  query: string
): FileNode[] {
  const lowerQuery = query.toLowerCase();
  const result: FileNode[] = [];

  const process = (node: FileNode): FileNode | null => {
    const matches =
      node.name.toLowerCase().includes(lowerQuery) ||
      node.path.toLowerCase().includes(lowerQuery);

    const copy = { ...node };

    if (node.children) {
      const filteredChildren = node.children
        .map(process)
        .filter((child): child is FileNode => child !== null);
      copy.children = filteredChildren;
    }

    return matches || (copy.children && copy.children.length > 0) ? copy : null;
  };

  nodes.forEach(node => {
    const filtered = process(node);
    if (filtered) result.push(filtered);
  });

  return result;
}

// ============================================================================
// Tree Analysis & Statistics
// ============================================================================

/**
 * Calculate statistics about the file tree
 */
export function calculateTreeStats(nodes: FileNode[]): FileTreeStats {
  const stats: FileTreeStats = {
    totalFiles: 0,
    totalDirs: 0,
    filesByStatus: {},
    filesByAgent: {},
    maxDepth: 0,
    averageDepth: 0,
  };

  const depths: number[] = [];

  const traverse = (node: FileNode, depth: number) => {
    depths.push(depth);
    stats.maxDepth = Math.max(stats.maxDepth, depth);

    if (node.isDirectory) {
      stats.totalDirs++;
    } else {
      stats.totalFiles++;
    }

    // Count by status
    stats.filesByStatus[node.status] = (stats.filesByStatus[node.status] || 0) + 1;

    // Count by agent
    if (node.modifiedBy) {
      stats.filesByAgent[node.modifiedBy] = (stats.filesByAgent[node.modifiedBy] || 0) + 1;
    }

    if (node.children) {
      node.children.forEach(child => traverse(child, depth + 1));
    }
  };

  nodes.forEach(node => traverse(node, 0));

  // Calculate average depth
  if (depths.length > 0) {
    stats.averageDepth = depths.reduce((a, b) => a + b, 0) / depths.length;
  }

  return stats;
}

/**
 * Get files modified after a certain timestamp
 */
export function getRecentFiles(
  nodes: FileNode[],
  since: Date
): FileNode[] {
  const files = getAllFiles(nodes);
  return files.filter(
    file => new Date(file.lastModified).getTime() > since.getTime()
  );
}

/**
 * Group files by agent
 */
export function groupFilesByAgent(nodes: FileNode[]): Record<string, FileNode[]> {
  const files = getAllFiles(nodes);
  const grouped: Record<string, FileNode[]> = {};

  files.forEach(file => {
    const agent = file.modifiedBy || 'unknown';
    if (!grouped[agent]) grouped[agent] = [];
    grouped[agent].push(file);
  });

  return grouped;
}

/**
 * Group files by status
 */
export function groupFilesByStatus(nodes: FileNode[]): Record<string, FileNode[]> {
  const files = getAllFiles(nodes);
  const grouped: Record<string, FileNode[]> = {};

  files.forEach(file => {
    if (!grouped[file.status]) grouped[file.status] = [];
    grouped[file.status].push(file);
  });

  return grouped;
}

// ============================================================================
// Path Utilities
// ============================================================================

/**
 * Parse a file path into components
 */
export function parsePath(path: string): PathInfo {
  const normalized = path.startsWith('/') ? path.slice(1) : path;
  const segments = normalized.split('/');
  const filename = segments[segments.length - 1];
  const directory = segments.slice(0, -1).join('/');
  const dotIndex = filename.lastIndexOf('.');
  const extension = dotIndex > 0 ? filename.slice(dotIndex + 1) : '';

  return {
    directory: directory ? `/${directory}` : '/',
    filename,
    extension,
    segments,
  };
}

/**
 * Get file extension
 */
export function getExtension(filename: string): string {
  const dotIndex = filename.lastIndexOf('.');
  return dotIndex > 0 ? filename.slice(dotIndex + 1).toLowerCase() : '';
}

/**
 * Get file icon emoji based on extension
 */
export function getFileIcon(filename: string): string {
  const ext = getExtension(filename).toLowerCase();

  const iconMap: Record<string, string> = {
    // Code
    ts: '📘',
    tsx: '⚛️',
    js: '📙',
    jsx: '⚛️',
    py: '🐍',
    java: '☕',
    go: '🐹',
    rs: '🦀',
    cpp: '⚙️',
    cs: '#️⃣',
    rb: '💎',
    php: '🐘',

    // Markup
    html: '🌐',
    xml: '📋',
    json: '📊',
    yaml: '📄',
    yml: '📄',
    toml: '📄',

    // Styles
    css: '🎨',
    scss: '🎨',
    less: '🎨',

    // Images
    png: '🖼️',
    jpg: '🖼️',
    jpeg: '🖼️',
    gif: '🖼️',
    svg: '🎨',

    // Documents
    pdf: '📕',
    doc: '📘',
    docx: '📘',
    xlsx: '📊',
    pptx: '📽️',
    md: '📝',

    // Config
    env: '⚙️',
    gitignore: '🚫',
    npmrc: '📦',

    // Default
    default: '📄',
  };

  return iconMap[ext] || iconMap['default'];
}

/**
 * Check if a path is under a directory
 */
export function isUnderDirectory(
  filePath: string,
  directoryPath: string
): boolean {
  return filePath.startsWith(directoryPath + '/');
}

/**
 * Get the parent path of a file
 */
export function getParentPath(path: string): string | null {
  const lastSlash = path.lastIndexOf('/');
  return lastSlash > 0 ? path.slice(0, lastSlash) : null;
}

/**
 * Get the relative path from a base path
 */
export function getRelativePath(filePath: string, basePath: string): string {
  if (!filePath.startsWith(basePath)) return filePath;
  const relative = filePath.slice(basePath.length);
  return relative.startsWith('/') ? relative.slice(1) : relative;
}

// ============================================================================
// Tree Sorting
// ============================================================================

/**
 * Sort tree by name (recursively)
 */
export function sortTreeByName(nodes: FileNode[]): FileNode[] {
  const sorted = [...nodes].sort((a, b) => a.name.localeCompare(b.name));
  sorted.forEach(node => {
    if (node.children) {
      node.children = sortTreeByName(node.children);
    }
  });
  return sorted;
}

/**
 * Sort tree by modification time (newest first)
 */
export function sortTreeByTime(nodes: FileNode[], newest = true): FileNode[] {
  const sorted = [...nodes].sort((a, b) => {
    const timeA = new Date(a.lastModified).getTime();
    const timeB = new Date(b.lastModified).getTime();
    return newest ? timeB - timeA : timeA - timeB;
  });
  sorted.forEach(node => {
    if (node.children) {
      node.children = sortTreeByTime(node.children, newest);
    }
  });
  return sorted;
}

/**
 * Sort tree: directories first, then files, both by name
 */
export function sortTreeHierarchical(nodes: FileNode[]): FileNode[] {
  const sorted = [...nodes].sort((a, b) => {
    if (a.isDirectory !== b.isDirectory) {
      return a.isDirectory ? -1 : 1;
    }
    return a.name.localeCompare(b.name);
  });
  sorted.forEach(node => {
    if (node.children) {
      node.children = sortTreeHierarchical(node.children);
    }
  });
  return sorted;
}

// ============================================================================
// Color Utilities
// ============================================================================

/**
 * Default agent colors (matches component)
 */
export const DEFAULT_AGENT_COLORS = [
  '#3B82F6', // blue
  '#10B981', // emerald
  '#F59E0B', // amber
  '#EF4444', // red
  '#8B5CF6', // violet
  '#EC4899', // pink
  '#06B6D4', // cyan
  '#84CC16', // lime
];

/**
 * Get color for agent by ID
 */
export function getAgentColor(
  agentId: string,
  colorMap?: Record<string, string>
): string {
  if (colorMap && colorMap[agentId]) {
    return colorMap[agentId];
  }

  // Fallback to palette based on character codes
  let hash = 0;
  for (let i = 0; i < agentId.length; i++) {
    hash = agentId.charCodeAt(i) + ((hash << 5) - hash);
  }

  return DEFAULT_AGENT_COLORS[Math.abs(hash) % DEFAULT_AGENT_COLORS.length];
}

/**
 * Get color for status
 */
export function getStatusColor(status: FileNode['status']): string {
  const colorMap = {
    modified: '#F59E0B',  // amber
    read: '#6B7280',      // gray
    created: '#10B981',   // green
    deleted: '#EF4444',   // red
  };
  return colorMap[status];
}

// ============================================================================
// Export utilities for common patterns
// ============================================================================

export default {
  buildTreeFromPaths,
  findNodeByPath,
  getAllPaths,
  getAllFiles,
  getAllDirectories,
  filterByStatus,
  filterByAgent,
  searchTree,
  calculateTreeStats,
  getRecentFiles,
  groupFilesByAgent,
  groupFilesByStatus,
  parsePath,
  getExtension,
  getFileIcon,
  isUnderDirectory,
  getParentPath,
  getRelativePath,
  sortTreeByName,
  sortTreeByTime,
  sortTreeHierarchical,
  getAgentColor,
  getStatusColor,
};
