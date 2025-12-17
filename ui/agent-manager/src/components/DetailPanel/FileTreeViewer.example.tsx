/**
 * FileTreeViewer Component - Usage Examples
 *
 * Demonstrates how to use the FileTreeViewer component with various
 * configurations, data structures, and integration patterns.
 */

import React, { useState } from 'react';
import FileTreeViewer from './FileTreeViewer';

// ============================================================================
// Type Definitions
// ============================================================================

interface FileNode {
  path: string;
  name: string;
  isDirectory: boolean;
  children?: FileNode[];
  modifiedBy?: string;
  status: 'modified' | 'read' | 'created' | 'deleted';
  lastModified: string;
}

// ============================================================================
// Example Data
// ============================================================================

/**
 * Sample file tree with multiple agents making modifications
 */
const SAMPLE_FILE_TREE: FileNode[] = [
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
const CUSTOM_COLOR_MAP: Record<string, string> = {
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

export const BasicExample: React.FC = () => {
  const [selectedPath, setSelectedPath] = useState<string | null>(null);

  return (
    <div style={{ height: '500px', width: '100%' }}>
      <h2>Basic FileTreeViewer</h2>
      <FileTreeViewer
        files={SAMPLE_FILE_TREE}
        onFileSelect={(path) => {
          console.log('Selected:', path);
          setSelectedPath(path);
        }}
      />
      {selectedPath && <p>Selected: {selectedPath}</p>}
    </div>
  );
};

// ============================================================================
// Example 2: With Custom Colors
// ============================================================================

export const CustomColorsExample: React.FC = () => {
  return (
    <div style={{ height: '500px', width: '100%' }}>
      <h2>FileTreeViewer with Custom Agent Colors</h2>
      <FileTreeViewer
        files={SAMPLE_FILE_TREE}
        agentColorMap={CUSTOM_COLOR_MAP}
        onFileSelect={(path) => console.log('Selected:', path)}
      />
    </div>
  );
};

// ============================================================================
// Example 3: With Active Agent Highlight
// ============================================================================

export const ActiveAgentExample: React.FC = () => {
  const [activeAgent, setActiveAgent] = useState('agent-1');

  return (
    <div style={{ height: '600px', width: '100%' }}>
      <h2>FileTreeViewer with Active Agent</h2>
      <div style={{ marginBottom: '16px' }}>
        <label>
          Active Agent:{' '}
          <select value={activeAgent} onChange={(e) => setActiveAgent(e.target.value)}>
            {['agent-1', 'agent-2', 'agent-3', 'agent-4', 'agent-5', 'agent-6', 'agent-7', 'agent-8'].map(
              agent => (
                <option key={agent} value={agent}>
                  {agent}
                </option>
              )
            )}
          </select>
        </label>
      </div>
      <FileTreeViewer
        files={SAMPLE_FILE_TREE}
        activeAgentId={activeAgent}
        agentColorMap={CUSTOM_COLOR_MAP}
        onFileSelect={(path) => console.log('Selected by', activeAgent, ':', path)}
      />
    </div>
  );
};

// ============================================================================
// Example 4: Large File Tree
// ============================================================================

/**
 * Generate a large file tree for performance testing
 */
const generateLargeFileTree = (depth: number = 3, filesPerDir: number = 10): FileNode[] => {
  const agents = ['agent-1', 'agent-2', 'agent-3', 'agent-4', 'agent-5'];
  const statuses: Array<'modified' | 'read' | 'created' | 'deleted'> = [
    'modified',
    'read',
    'created',
    'deleted',
  ];

  const generateNode = (path: string, name: string, currentDepth: number): FileNode => {
    const isDirectory = currentDepth < depth;
    const agent = agents[Math.floor(Math.random() * agents.length)];
    const status = statuses[Math.floor(Math.random() * statuses.length)];

    const node: FileNode = {
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

export const LargeFileTreeExample: React.FC = () => {
  const [largeTree] = useState(() => generateLargeFileTree(4, 8));
  const [stats, setStats] = useState<string>('');

  const handleFileSelect = (path: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setStats(`Selected ${path} at ${timestamp}`);
  };

  return (
    <div style={{ height: '600px', width: '100%' }}>
      <h2>Large FileTreeViewer (Performance Test)</h2>
      <p>File tree with 4 levels and ~8 files per directory</p>
      <FileTreeViewer
        files={largeTree}
        agentColorMap={CUSTOM_COLOR_MAP}
        onFileSelect={handleFileSelect}
      />
      {stats && <p style={{ marginTop: '16px', color: '#6B7280' }}>{stats}</p>}
    </div>
  );
};

// ============================================================================
// Example 5: Integration with Real Data
// ============================================================================

/**
 * Example of connecting FileTreeViewer to real agent activity data
 */
interface AgentActivity {
  agentId: string;
  filePath: string;
  action: 'read' | 'write' | 'create' | 'delete';
  timestamp: string;
}

export const RealDataIntegrationExample: React.FC<{ activities?: AgentActivity[] }> = ({
  activities = [
    { agentId: 'agent-1', filePath: '/src/components/Header.tsx', action: 'write', timestamp: new Date().toISOString() },
    { agentId: 'agent-2', filePath: '/src/api.ts', action: 'read', timestamp: new Date().toISOString() },
    { agentId: 'agent-3', filePath: '/src/utils/validation.ts', action: 'write', timestamp: new Date().toISOString() },
  ],
}) => {
  // Convert activities to file tree
  const buildTreeFromActivities = (acts: AgentActivity[]): FileNode[] => {
    const nodes = new Map<string, FileNode>();

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
            status:
              lastActivity?.action === 'write' ? 'modified' :
              lastActivity?.action === 'read' ? 'read' :
              lastActivity?.action === 'create' ? 'created' :
              'deleted',
            lastModified: lastActivity?.timestamp || new Date().toISOString(),
          });
        }
      });
    });

    // Build tree structure
    const roots: FileNode[] = [];
    const rootPaths = new Set<string>();

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
          if (!parent.children) parent.children = [];
          parent.children.push(node);
        }
      }
    });

    return roots;
  };

  const fileTree = buildTreeFromActivities(activities);

  return (
    <div style={{ height: '500px', width: '100%' }}>
      <h2>Real Data Integration Example</h2>
      <p>Built from {activities.length} agent activities</p>
      <FileTreeViewer
        files={fileTree}
        agentColorMap={CUSTOM_COLOR_MAP}
        onFileSelect={(path) => console.log('Selected:', path)}
      />
    </div>
  );
};

// ============================================================================
// Example 6: Controlled Component
// ============================================================================

export const ControlledExample: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set(['/src', '/src/components']));

  return (
    <div style={{ height: '600px', width: '100%', display: 'flex', gap: '16px' }}>
      <div style={{ flex: 1 }}>
        <h2>Controlled FileTreeViewer</h2>
        <FileTreeViewer
          files={SAMPLE_FILE_TREE}
          agentColorMap={CUSTOM_COLOR_MAP}
          onFileSelect={setSelectedFile}
        />
      </div>
      <div style={{ flex: 1, padding: '16px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
        <h3>Component State</h3>
        <p>
          <strong>Selected File:</strong>
          <br />
          {selectedFile ? <code>{selectedFile}</code> : <em>None</em>}
        </p>
        <p>
          <strong>Expanded Directories:</strong>
          <br />
          {expandedDirs.size === 0 ? (
            <em>None</em>
          ) : (
            <ul>
              {Array.from(expandedDirs).map(dir => (
                <li key={dir}>
                  <code>{dir}</code>
                </li>
              ))}
            </ul>
          )}
        </p>
      </div>
    </div>
  );
};

// ============================================================================
// Demo Component
// ============================================================================

export const FileTreeViewerDemo: React.FC = () => {
  const [activeExample, setActiveExample] = useState<string>('basic');

  const examples = {
    basic: { component: BasicExample, label: 'Basic Usage' },
    colors: { component: CustomColorsExample, label: 'Custom Colors' },
    active: { component: ActiveAgentExample, label: 'Active Agent' },
    large: { component: LargeFileTreeExample, label: 'Large Tree' },
    real: { component: RealDataIntegrationExample, label: 'Real Data' },
    controlled: { component: ControlledExample, label: 'Controlled Component' },
  };

  const Example = examples[activeExample as keyof typeof examples]?.component || BasicExample;

  return (
    <div style={{ padding: '24px' }}>
      <h1>FileTreeViewer Component Demo</h1>
      <div style={{ marginBottom: '24px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        {Object.entries(examples).map(([key, { label }]) => (
          <button
            key={key}
            onClick={() => setActiveExample(key)}
            style={{
              padding: '8px 16px',
              backgroundColor: activeExample === key ? '#3B82F6' : '#E5E7EB',
              color: activeExample === key ? 'white' : 'black',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: activeExample === key ? 'bold' : 'normal',
            }}
          >
            {label}
          </button>
        ))}
      </div>
      <Example />
    </div>
  );
};

export default FileTreeViewerDemo;
