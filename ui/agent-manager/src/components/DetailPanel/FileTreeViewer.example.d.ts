/**
 * FileTreeViewer Component - Usage Examples
 *
 * Demonstrates how to use the FileTreeViewer component with various
 * configurations, data structures, and integration patterns.
 */
import React from 'react';
export declare const BasicExample: React.FC;
export declare const CustomColorsExample: React.FC;
export declare const ActiveAgentExample: React.FC;
export declare const LargeFileTreeExample: React.FC;
/**
 * Example of connecting FileTreeViewer to real agent activity data
 */
interface AgentActivity {
    agentId: string;
    filePath: string;
    action: 'read' | 'write' | 'create' | 'delete';
    timestamp: string;
}
export declare const RealDataIntegrationExample: React.FC<{
    activities?: AgentActivity[];
}>;
export declare const ControlledExample: React.FC;
export declare const FileTreeViewerDemo: React.FC;
export default FileTreeViewerDemo;
//# sourceMappingURL=FileTreeViewer.example.d.ts.map