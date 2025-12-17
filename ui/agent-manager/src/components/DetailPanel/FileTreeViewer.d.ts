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
import React from 'react';
import './FileTreeViewer.css';
interface FileNode {
    path: string;
    name: string;
    isDirectory: boolean;
    children?: FileNode[];
    modifiedBy?: string;
    status: 'modified' | 'read' | 'created' | 'deleted';
    lastModified: string;
}
interface FileTreeViewerProps {
    files: FileNode[];
    activeAgentId?: string;
    onFileSelect?: (path: string) => void;
    agentColorMap?: Record<string, string>;
}
declare const FileTreeViewer: React.FC<FileTreeViewerProps>;
export default FileTreeViewer;
//# sourceMappingURL=FileTreeViewer.d.ts.map