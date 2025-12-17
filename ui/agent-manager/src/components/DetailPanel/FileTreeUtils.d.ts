/**
 * FileTreeViewer Utilities
 *
 * Helper functions for building, transforming, and working with file trees.
 */
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
export declare function buildTreeFromPaths(paths: Array<{
    path: string;
    status: FileNode['status'];
    modifiedBy?: string;
    lastModified?: string;
}>): FileNode[];
/**
 * Find a node in the tree by path
 */
export declare function findNodeByPath(path: string, nodes: FileNode[]): FileNode | null;
/**
 * Get all paths in the tree (flat list)
 */
export declare function getAllPaths(nodes: FileNode[]): string[];
/**
 * Get all files (non-directories) in the tree
 */
export declare function getAllFiles(nodes: FileNode[]): FileNode[];
/**
 * Get all directories in the tree
 */
export declare function getAllDirectories(nodes: FileNode[]): FileNode[];
/**
 * Filter tree by status
 */
export declare function filterByStatus(nodes: FileNode[], status: FileNode['status']): FileNode[];
/**
 * Filter tree by agent
 */
export declare function filterByAgent(nodes: FileNode[], agentId: string): FileNode[];
/**
 * Search tree by filename or path (case-insensitive)
 */
export declare function searchTree(nodes: FileNode[], query: string): FileNode[];
/**
 * Calculate statistics about the file tree
 */
export declare function calculateTreeStats(nodes: FileNode[]): FileTreeStats;
/**
 * Get files modified after a certain timestamp
 */
export declare function getRecentFiles(nodes: FileNode[], since: Date): FileNode[];
/**
 * Group files by agent
 */
export declare function groupFilesByAgent(nodes: FileNode[]): Record<string, FileNode[]>;
/**
 * Group files by status
 */
export declare function groupFilesByStatus(nodes: FileNode[]): Record<string, FileNode[]>;
/**
 * Parse a file path into components
 */
export declare function parsePath(path: string): PathInfo;
/**
 * Get file extension
 */
export declare function getExtension(filename: string): string;
/**
 * Get file icon emoji based on extension
 */
export declare function getFileIcon(filename: string): string;
/**
 * Check if a path is under a directory
 */
export declare function isUnderDirectory(filePath: string, directoryPath: string): boolean;
/**
 * Get the parent path of a file
 */
export declare function getParentPath(path: string): string | null;
/**
 * Get the relative path from a base path
 */
export declare function getRelativePath(filePath: string, basePath: string): string;
/**
 * Sort tree by name (recursively)
 */
export declare function sortTreeByName(nodes: FileNode[]): FileNode[];
/**
 * Sort tree by modification time (newest first)
 */
export declare function sortTreeByTime(nodes: FileNode[], newest?: boolean): FileNode[];
/**
 * Sort tree: directories first, then files, both by name
 */
export declare function sortTreeHierarchical(nodes: FileNode[]): FileNode[];
/**
 * Default agent colors (matches component)
 */
export declare const DEFAULT_AGENT_COLORS: string[];
/**
 * Get color for agent by ID
 */
export declare function getAgentColor(agentId: string, colorMap?: Record<string, string>): string;
/**
 * Get color for status
 */
export declare function getStatusColor(status: FileNode['status']): string;
declare const _default: {
    buildTreeFromPaths: typeof buildTreeFromPaths;
    findNodeByPath: typeof findNodeByPath;
    getAllPaths: typeof getAllPaths;
    getAllFiles: typeof getAllFiles;
    getAllDirectories: typeof getAllDirectories;
    filterByStatus: typeof filterByStatus;
    filterByAgent: typeof filterByAgent;
    searchTree: typeof searchTree;
    calculateTreeStats: typeof calculateTreeStats;
    getRecentFiles: typeof getRecentFiles;
    groupFilesByAgent: typeof groupFilesByAgent;
    groupFilesByStatus: typeof groupFilesByStatus;
    parsePath: typeof parsePath;
    getExtension: typeof getExtension;
    getFileIcon: typeof getFileIcon;
    isUnderDirectory: typeof isUnderDirectory;
    getParentPath: typeof getParentPath;
    getRelativePath: typeof getRelativePath;
    sortTreeByName: typeof sortTreeByName;
    sortTreeByTime: typeof sortTreeByTime;
    sortTreeHierarchical: typeof sortTreeHierarchical;
    getAgentColor: typeof getAgentColor;
    getStatusColor: typeof getStatusColor;
};
export default _default;
//# sourceMappingURL=FileTreeUtils.d.ts.map