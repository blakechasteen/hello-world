/**
 * FileTreeViewer Utilities
 *
 * Helper functions for building, transforming, and working with file trees.
 */
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
export function buildTreeFromPaths(paths) {
    const nodeMap = new Map();
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
                const node = nodeMap.get(currentPath);
                node.modifiedBy = item.modifiedBy;
                node.status = item.status;
                node.lastModified = item.lastModified || new Date().toISOString();
            }
        });
    });
    // Build tree structure
    const rootPaths = new Set();
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
        .map(path => nodeMap.get(path))
        .sort((a, b) => a.name.localeCompare(b.name));
}
/**
 * Find a node in the tree by path
 */
export function findNodeByPath(path, nodes) {
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
}
/**
 * Get all paths in the tree (flat list)
 */
export function getAllPaths(nodes) {
    const paths = [];
    const collect = (node) => {
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
export function getAllFiles(nodes) {
    const files = [];
    const collect = (node) => {
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
export function getAllDirectories(nodes) {
    const dirs = [];
    const collect = (node) => {
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
export function filterByStatus(nodes, status) {
    const result = [];
    const process = (node) => {
        if (node.status === status) {
            const copy = { ...node };
            if (node.children) {
                copy.children = node.children
                    .map(process)
                    .filter((child) => child !== null);
            }
            return copy;
        }
        if (node.isDirectory && node.children) {
            const filteredChildren = node.children
                .map(process)
                .filter((child) => child !== null);
            if (filteredChildren.length > 0) {
                return { ...node, children: filteredChildren };
            }
        }
        return null;
    };
    nodes.forEach(node => {
        const filtered = process(node);
        if (filtered)
            result.push(filtered);
    });
    return result;
}
/**
 * Filter tree by agent
 */
export function filterByAgent(nodes, agentId) {
    const result = [];
    const process = (node) => {
        if (node.modifiedBy === agentId) {
            const copy = { ...node };
            if (node.children) {
                copy.children = node.children
                    .map(process)
                    .filter((child) => child !== null);
            }
            return copy;
        }
        if (node.isDirectory && node.children) {
            const filteredChildren = node.children
                .map(process)
                .filter((child) => child !== null);
            if (filteredChildren.length > 0) {
                return { ...node, children: filteredChildren };
            }
        }
        return null;
    };
    nodes.forEach(node => {
        const filtered = process(node);
        if (filtered)
            result.push(filtered);
    });
    return result;
}
/**
 * Search tree by filename or path (case-insensitive)
 */
export function searchTree(nodes, query) {
    const lowerQuery = query.toLowerCase();
    const result = [];
    const process = (node) => {
        const matches = node.name.toLowerCase().includes(lowerQuery) ||
            node.path.toLowerCase().includes(lowerQuery);
        const copy = { ...node };
        if (node.children) {
            const filteredChildren = node.children
                .map(process)
                .filter((child) => child !== null);
            copy.children = filteredChildren;
        }
        return matches || (copy.children && copy.children.length > 0) ? copy : null;
    };
    nodes.forEach(node => {
        const filtered = process(node);
        if (filtered)
            result.push(filtered);
    });
    return result;
}
// ============================================================================
// Tree Analysis & Statistics
// ============================================================================
/**
 * Calculate statistics about the file tree
 */
export function calculateTreeStats(nodes) {
    const stats = {
        totalFiles: 0,
        totalDirs: 0,
        filesByStatus: {},
        filesByAgent: {},
        maxDepth: 0,
        averageDepth: 0,
    };
    const depths = [];
    const traverse = (node, depth) => {
        depths.push(depth);
        stats.maxDepth = Math.max(stats.maxDepth, depth);
        if (node.isDirectory) {
            stats.totalDirs++;
        }
        else {
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
export function getRecentFiles(nodes, since) {
    const files = getAllFiles(nodes);
    return files.filter(file => new Date(file.lastModified).getTime() > since.getTime());
}
/**
 * Group files by agent
 */
export function groupFilesByAgent(nodes) {
    const files = getAllFiles(nodes);
    const grouped = {};
    files.forEach(file => {
        const agent = file.modifiedBy || 'unknown';
        if (!grouped[agent])
            grouped[agent] = [];
        grouped[agent].push(file);
    });
    return grouped;
}
/**
 * Group files by status
 */
export function groupFilesByStatus(nodes) {
    const files = getAllFiles(nodes);
    const grouped = {};
    files.forEach(file => {
        if (!grouped[file.status])
            grouped[file.status] = [];
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
export function parsePath(path) {
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
export function getExtension(filename) {
    const dotIndex = filename.lastIndexOf('.');
    return dotIndex > 0 ? filename.slice(dotIndex + 1).toLowerCase() : '';
}
/**
 * Get file icon emoji based on extension
 */
export function getFileIcon(filename) {
    const ext = getExtension(filename).toLowerCase();
    const iconMap = {
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
export function isUnderDirectory(filePath, directoryPath) {
    return filePath.startsWith(directoryPath + '/');
}
/**
 * Get the parent path of a file
 */
export function getParentPath(path) {
    const lastSlash = path.lastIndexOf('/');
    return lastSlash > 0 ? path.slice(0, lastSlash) : null;
}
/**
 * Get the relative path from a base path
 */
export function getRelativePath(filePath, basePath) {
    if (!filePath.startsWith(basePath))
        return filePath;
    const relative = filePath.slice(basePath.length);
    return relative.startsWith('/') ? relative.slice(1) : relative;
}
// ============================================================================
// Tree Sorting
// ============================================================================
/**
 * Sort tree by name (recursively)
 */
export function sortTreeByName(nodes) {
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
export function sortTreeByTime(nodes, newest = true) {
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
export function sortTreeHierarchical(nodes) {
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
export function getAgentColor(agentId, colorMap) {
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
export function getStatusColor(status) {
    const colorMap = {
        modified: '#F59E0B', // amber
        read: '#6B7280', // gray
        created: '#10B981', // green
        deleted: '#EF4444', // red
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
//# sourceMappingURL=FileTreeUtils.js.map