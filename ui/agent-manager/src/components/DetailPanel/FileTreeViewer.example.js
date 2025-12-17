import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * FileTreeViewer Component - Usage Examples
 *
 * Demonstrates how to use the FileTreeViewer component with various
 * configurations, data structures, and integration patterns.
 */
import { useState } from 'react';
import FileTreeViewer from './FileTreeViewer';
// ============================================================================
// Example Data
// ============================================================================
/**
 * Sample file tree with multiple agents making modifications
 */
const SAMPLE_FILE_TREE = [
    {
        path: '/src',
        name: 'src',
        isDirectory: true,
        modifiedBy: 'agent-1',
        status: 'modified',
        lastModified: new Date(Date.now() - 3600000).toISOString(),
        children: [
            {
                path: '/src/components',
                name: 'components',
                isDirectory: true,
                modifiedBy: 'agent-2',
                status: 'modified',
                lastModified: new Date(Date.now() - 1800000).toISOString(),
                children: [
                    {
                        path: '/src/components/Header.tsx',
                        name: 'Header.tsx',
                        isDirectory: false,
                        modifiedBy: 'agent-1',
                        status: 'modified',
                        lastModified: new Date(Date.now() - 600000).toISOString(),
                    },
                    {
                        path: '/src/components/Footer.tsx',
                        name: 'Footer.tsx',
                        isDirectory: false,
                        modifiedBy: 'agent-2',
                        status: 'created',
                        lastModified: new Date(Date.now() - 300000).toISOString(),
                    },
                    {
                        path: '/src/components/Sidebar.tsx',
                        name: 'Sidebar.tsx',
                        isDirectory: false,
                        modifiedBy: 'agent-3',
                        status: 'modified',
                        lastModified: new Date(Date.now() - 900000).toISOString(),
                    },
                ],
            },
            {
                path: '/src/utils',
                name: 'utils',
                isDirectory: true,
                modifiedBy: 'agent-4',
                status: 'modified',
                lastModified: new Date(Date.now() - 7200000).toISOString(),
                children: [
                    {
                        path: '/src/utils/formatting.ts',
                        name: 'formatting.ts',
                        isDirectory: false,
                        modifiedBy: 'agent-4',
                        status: 'modified',
                        lastModified: new Date(Date.now() - 7200000).toISOString(),
                    },
                    {
                        path: '/src/utils/validation.ts',
                        name: 'validation.ts',
                        isDirectory: false,
                        modifiedBy: 'agent-1',
                        status: 'read',
                        lastModified: new Date(Date.now() - 86400000).toISOString(),
                    },
                    {
                        path: '/src/utils/deprecated.ts',
                        name: 'deprecated.ts',
                        isDirectory: false,
                        modifiedBy: 'agent-5',
                        status: 'deleted',
                        lastModified: new Date(Date.now() - 172800000).toISOString(),
                    },
                ],
            },
            {
                path: '/src/api.ts',
                name: 'api.ts',
                isDirectory: false,
                modifiedBy: 'agent-2',
                status: 'modified',
                lastModified: new Date(Date.now() - 1200000).toISOString(),
            },
            {
                path: '/src/types.ts',
                name: 'types.ts',
                isDirectory: false,
                modifiedBy: 'agent-3',
                status: 'read',
                lastModified: new Date(Date.now() - 2592000000).toISOString(),
            },
        ],
    },
    {
        path: '/public',
        name: 'public',
        isDirectory: true,
        modifiedBy: 'agent-6',
        status: 'modified',
        lastModified: new Date(Date.now() - 10800000).toISOString(),
        children: [
            {
                path: '/public/index.html',
                name: 'index.html',
                isDirectory: false,
                modifiedBy: 'agent-6',
                status: 'modified',
                lastModified: new Date(Date.now() - 10800000).toISOString(),
            },
            {
                path: '/public/favicon.ico',
                name: 'favicon.ico',
                isDirectory: false,
                modifiedBy: 'agent-1',
                status: 'created',
                lastModified: new Date(Date.now() - 604800000).toISOString(),
            },
        ],
    },
    {
        path: '/tests',
        name: 'tests',
        isDirectory: true,
        modifiedBy: 'agent-7',
        status: 'modified',
        lastModified: new Date(Date.now() - 5400000).toISOString(),
        children: [
            {
                path: '/tests/components.test.ts',
                name: 'components.test.ts',
                isDirectory: false,
                modifiedBy: 'agent-7',
                status: 'modified',
                lastModified: new Date(Date.now() - 5400000).toISOString(),
            },
            {
                path: '/tests/utils.test.ts',
                name: 'utils.test.ts',
                isDirectory: false,
                modifiedBy: 'agent-8',
                status: 'created',
                lastModified: new Date(Date.now() - 1800000).toISOString(),
            },
        ],
    },
    {
        path: '/package.json',
        name: 'package.json',
        isDirectory: false,
        modifiedBy: 'agent-1',
        status: 'modified',
        lastModified: new Date(Date.now() - 18000000).toISOString(),
    },
    {
        path: '/.gitignore',
        name: '.gitignore',
        isDirectory: false,
        modifiedBy: 'agent-2',
        status: 'read',
        lastModified: new Date(Date.now() - 2592000000).toISOString(),
    },
];
/**
 * Custom agent color map
 */
const CUSTOM_COLOR_MAP = {
    'agent-1': '#3B82F6', // blue
    'agent-2': '#10B981', // emerald
    'agent-3': '#F59E0B', // amber
    'agent-4': '#EF4444', // red
    'agent-5': '#8B5CF6', // violet
    'agent-6': '#EC4899', // pink
    'agent-7': '#06B6D4', // cyan
    'agent-8': '#84CC16', // lime
};
// ============================================================================
// Example 1: Basic Usage
// ============================================================================
export const BasicExample = () => {
    const [selectedPath, setSelectedPath] = useState(null);
    return (_jsxs("div", { style: { height: '500px', width: '100%' }, children: [_jsx("h2", { children: "Basic FileTreeViewer" }), _jsx(FileTreeViewer, { files: SAMPLE_FILE_TREE, onFileSelect: (path) => {
                    console.log('Selected:', path);
                    setSelectedPath(path);
                } }), selectedPath && _jsxs("p", { children: ["Selected: ", selectedPath] })] }));
};
// ============================================================================
// Example 2: With Custom Colors
// ============================================================================
export const CustomColorsExample = () => {
    return (_jsxs("div", { style: { height: '500px', width: '100%' }, children: [_jsx("h2", { children: "FileTreeViewer with Custom Agent Colors" }), _jsx(FileTreeViewer, { files: SAMPLE_FILE_TREE, agentColorMap: CUSTOM_COLOR_MAP, onFileSelect: (path) => console.log('Selected:', path) })] }));
};
// ============================================================================
// Example 3: With Active Agent Highlight
// ============================================================================
export const ActiveAgentExample = () => {
    const [activeAgent, setActiveAgent] = useState('agent-1');
    return (_jsxs("div", { style: { height: '600px', width: '100%' }, children: [_jsx("h2", { children: "FileTreeViewer with Active Agent" }), _jsx("div", { style: { marginBottom: '16px' }, children: _jsxs("label", { children: ["Active Agent:", ' ', _jsx("select", { value: activeAgent, onChange: (e) => setActiveAgent(e.target.value), children: ['agent-1', 'agent-2', 'agent-3', 'agent-4', 'agent-5', 'agent-6', 'agent-7', 'agent-8'].map(agent => (_jsx("option", { value: agent, children: agent }, agent))) })] }) }), _jsx(FileTreeViewer, { files: SAMPLE_FILE_TREE, activeAgentId: activeAgent, agentColorMap: CUSTOM_COLOR_MAP, onFileSelect: (path) => console.log('Selected by', activeAgent, ':', path) })] }));
};
// ============================================================================
// Example 4: Large File Tree
// ============================================================================
/**
 * Generate a large file tree for performance testing
 */
const generateLargeFileTree = (depth = 3, filesPerDir = 10) => {
    const agents = ['agent-1', 'agent-2', 'agent-3', 'agent-4', 'agent-5'];
    const statuses = [
        'modified',
        'read',
        'created',
        'deleted',
    ];
    const generateNode = (path, name, currentDepth) => {
        const isDirectory = currentDepth < depth;
        const agent = agents[Math.floor(Math.random() * agents.length)];
        const status = statuses[Math.floor(Math.random() * statuses.length)];
        const node = {
            path,
            name,
            isDirectory,
            modifiedBy: agent,
            status,
            lastModified: new Date(Date.now() - Math.random() * 2592000000).toISOString(),
        };
        if (isDirectory) {
            node.children = Array.from({ length: filesPerDir }, (_, i) => {
                const childName = Math.random() > 0.3 ? `file${i}.ts` : `folder${i}`;
                const childPath = `${path}/${childName}`;
                return generateNode(childPath, childName, currentDepth + 1);
            });
        }
        return node;
    };
    return Array.from({ length: 3 }, (_, i) => generateNode(`/root${i}`, `root${i}`, 0));
};
export const LargeFileTreeExample = () => {
    const [largeTree] = useState(() => generateLargeFileTree(4, 8));
    const [stats, setStats] = useState('');
    const handleFileSelect = (path) => {
        const timestamp = new Date().toLocaleTimeString();
        setStats(`Selected ${path} at ${timestamp}`);
    };
    return (_jsxs("div", { style: { height: '600px', width: '100%' }, children: [_jsx("h2", { children: "Large FileTreeViewer (Performance Test)" }), _jsx("p", { children: "File tree with 4 levels and ~8 files per directory" }), _jsx(FileTreeViewer, { files: largeTree, agentColorMap: CUSTOM_COLOR_MAP, onFileSelect: handleFileSelect }), stats && _jsx("p", { style: { marginTop: '16px', color: '#6B7280' }, children: stats })] }));
};
export const RealDataIntegrationExample = ({ activities = [
    { agentId: 'agent-1', filePath: '/src/components/Header.tsx', action: 'write', timestamp: new Date().toISOString() },
    { agentId: 'agent-2', filePath: '/src/api.ts', action: 'read', timestamp: new Date().toISOString() },
    { agentId: 'agent-3', filePath: '/src/utils/validation.ts', action: 'write', timestamp: new Date().toISOString() },
], }) => {
    // Convert activities to file tree
    const buildTreeFromActivities = (acts) => {
        const nodes = new Map();
        acts.forEach(activity => {
            const parts = activity.filePath.split('/').filter(Boolean);
            let currentPath = '';
            parts.forEach((part, index) => {
                currentPath += `/${part}`;
                if (!nodes.has(currentPath)) {
                    const isDirectory = index < parts.length - 1;
                    const lastActivity = acts.filter(a => a.filePath.startsWith(currentPath)).pop();
                    nodes.set(currentPath, {
                        path: currentPath,
                        name: part,
                        isDirectory,
                        modifiedBy: lastActivity?.agentId,
                        status: lastActivity?.action === 'write' ? 'modified' :
                            lastActivity?.action === 'read' ? 'read' :
                                lastActivity?.action === 'create' ? 'created' :
                                    'deleted',
                        lastModified: lastActivity?.timestamp || new Date().toISOString(),
                    });
                }
            });
        });
        // Build tree structure
        const roots = [];
        const rootPaths = new Set();
        nodes.forEach((node, path) => {
            if (path.split('/').filter(Boolean).length === 1) {
                roots.push(node);
                rootPaths.add(path);
            }
        });
        // Add children
        nodes.forEach((node, path) => {
            const parts = path.split('/').filter(Boolean);
            if (parts.length > 1) {
                const parentPath = '/' + parts.slice(0, -1).join('/');
                const parent = nodes.get(parentPath);
                if (parent && parent.isDirectory) {
                    if (!parent.children)
                        parent.children = [];
                    parent.children.push(node);
                }
            }
        });
        return roots;
    };
    const fileTree = buildTreeFromActivities(activities);
    return (_jsxs("div", { style: { height: '500px', width: '100%' }, children: [_jsx("h2", { children: "Real Data Integration Example" }), _jsxs("p", { children: ["Built from ", activities.length, " agent activities"] }), _jsx(FileTreeViewer, { files: fileTree, agentColorMap: CUSTOM_COLOR_MAP, onFileSelect: (path) => console.log('Selected:', path) })] }));
};
// ============================================================================
// Example 6: Controlled Component
// ============================================================================
export const ControlledExample = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [expandedDirs, setExpandedDirs] = useState(new Set(['/src', '/src/components']));
    return (_jsxs("div", { style: { height: '600px', width: '100%', display: 'flex', gap: '16px' }, children: [_jsxs("div", { style: { flex: 1 }, children: [_jsx("h2", { children: "Controlled FileTreeViewer" }), _jsx(FileTreeViewer, { files: SAMPLE_FILE_TREE, agentColorMap: CUSTOM_COLOR_MAP, onFileSelect: setSelectedFile })] }), _jsxs("div", { style: { flex: 1, padding: '16px', backgroundColor: '#f5f5f5', borderRadius: '8px' }, children: [_jsx("h3", { children: "Component State" }), _jsxs("p", { children: [_jsx("strong", { children: "Selected File:" }), _jsx("br", {}), selectedFile ? _jsx("code", { children: selectedFile }) : _jsx("em", { children: "None" })] }), _jsxs("p", { children: [_jsx("strong", { children: "Expanded Directories:" }), _jsx("br", {}), expandedDirs.size === 0 ? (_jsx("em", { children: "None" })) : (_jsx("ul", { children: Array.from(expandedDirs).map(dir => (_jsx("li", { children: _jsx("code", { children: dir }) }, dir))) }))] })] })] }));
};
// ============================================================================
// Demo Component
// ============================================================================
export const FileTreeViewerDemo = () => {
    const [activeExample, setActiveExample] = useState('basic');
    const examples = {
        basic: { component: BasicExample, label: 'Basic Usage' },
        colors: { component: CustomColorsExample, label: 'Custom Colors' },
        active: { component: ActiveAgentExample, label: 'Active Agent' },
        large: { component: LargeFileTreeExample, label: 'Large Tree' },
        real: { component: RealDataIntegrationExample, label: 'Real Data' },
        controlled: { component: ControlledExample, label: 'Controlled Component' },
    };
    const Example = examples[activeExample]?.component || BasicExample;
    return (_jsxs("div", { style: { padding: '24px' }, children: [_jsx("h1", { children: "FileTreeViewer Component Demo" }), _jsx("div", { style: { marginBottom: '24px', display: 'flex', gap: '8px', flexWrap: 'wrap' }, children: Object.entries(examples).map(([key, { label }]) => (_jsx("button", { onClick: () => setActiveExample(key), style: {
                        padding: '8px 16px',
                        backgroundColor: activeExample === key ? '#3B82F6' : '#E5E7EB',
                        color: activeExample === key ? 'white' : 'black',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontWeight: activeExample === key ? 'bold' : 'normal',
                    }, children: label }, key))) }), _jsx(Example, {})] }));
};
export default FileTreeViewerDemo;
//# sourceMappingURL=FileTreeViewer.example.js.map