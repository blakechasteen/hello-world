# FileTreeViewer - Integration Guide

Quick guide for integrating the FileTreeViewer component into the HoloLoom Agent Manager UI.

## Installation

### 1. Copy Component Files

Copy these files to `src/components/DetailPanel/`:
```bash
cp FileTreeViewer.tsx src/components/DetailPanel/
cp FileTreeViewer.css src/components/DetailPanel/
cp FileTreeUtils.ts src/components/DetailPanel/
```

### 2. Import in Your Component

```tsx
import FileTreeViewer from './components/DetailPanel/FileTreeViewer';
import type { FileNode } from './components/DetailPanel/FileTreeUtils';
```

## Basic Integration

### Simple Implementation

```tsx
import React, { useCallback } from 'react';
import FileTreeViewer from './components/DetailPanel/FileTreeViewer';

export const AgentManagerPanel: React.FC = () => {
  const handleFileSelect = useCallback((path: string) => {
    console.log('Selected file:', path);
    // TODO: Show file diff, history, etc.
  }, []);

  // TODO: Replace with real file tree from agent data
  const fileTree = [
    {
      path: '/src',
      name: 'src',
      isDirectory: true,
      modifiedBy: 'agent-1',
      status: 'modified' as const,
      lastModified: new Date().toISOString(),
      children: [
        {
          path: '/src/index.ts',
          name: 'index.ts',
          isDirectory: false,
          modifiedBy: 'agent-1',
          status: 'modified' as const,
          lastModified: new Date().toISOString(),
        },
      ],
    },
  ];

  return (
    <div style={{ height: '600px' }}>
      <FileTreeViewer
        files={fileTree}
        onFileSelect={handleFileSelect}
      />
    </div>
  );
};
```

## Integration with Agent Data

### Converting Agent Activities to File Tree

```tsx
import { buildTreeFromPaths } from './components/DetailPanel/FileTreeUtils';
import type { FileNode } from './components/DetailPanel/FileTreeUtils';

interface AgentActivity {
  agentId: string;
  filePath: string;
  action: 'read' | 'write' | 'create' | 'delete';
  timestamp: string;
}

export const AgentManagerPanel: React.FC<{
  activities: AgentActivity[];
}> = ({ activities }) => {
  // Convert agent activities to file tree
  const fileTree = buildTreeFromPaths(
    activities.map(activity => ({
      path: activity.filePath,
      status:
        activity.action === 'write'
          ? 'modified'
          : activity.action === 'create'
            ? 'created'
            : activity.action === 'delete'
              ? 'deleted'
              : 'read',
      modifiedBy: activity.agentId,
      lastModified: activity.timestamp,
    }))
  );

  return (
    <div style={{ height: '600px' }}>
      <FileTreeViewer
        files={fileTree}
        onFileSelect={(path) => {
          showFileDetails(path);
        }}
      />
    </div>
  );
};
```

## Advanced: Redux/Zustand Integration

### With Redux

```tsx
import { useSelector, useDispatch } from 'react-redux';
import FileTreeViewer from './components/DetailPanel/FileTreeViewer';

export const AgentManagerPanel: React.FC = () => {
  const dispatch = useDispatch();
  const fileTree = useSelector(state => state.agents.fileTree);
  const selectedAgent = useSelector(state => state.agents.selectedAgent);

  const handleFileSelect = (path: string) => {
    dispatch({
      type: 'SELECT_FILE',
      payload: { path, agent: selectedAgent },
    });
  };

  return (
    <FileTreeViewer
      files={fileTree}
      activeAgentId={selectedAgent}
      onFileSelect={handleFileSelect}
      agentColorMap={{
        'agent-1': '#3B82F6',
        'agent-2': '#10B981',
        'agent-3': '#F59E0B',
      }}
    />
  );
};
```

### With Zustand

```tsx
import { useAgentStore } from './store/agentStore';
import FileTreeViewer from './components/DetailPanel/FileTreeViewer';

export const AgentManagerPanel: React.FC = () => {
  const fileTree = useAgentStore(state => state.fileTree);
  const selectedAgent = useAgentStore(state => state.selectedAgent);
  const selectFile = useAgentStore(state => state.selectFile);
  const colorMap = useAgentStore(state => state.agentColorMap);

  return (
    <FileTreeViewer
      files={fileTree}
      activeAgentId={selectedAgent}
      onFileSelect={selectFile}
      agentColorMap={colorMap}
    />
  );
};
```

## Real-Time Updates with WebSocket

### Using React Hooks

```tsx
import React, { useEffect, useState } from 'react';
import FileTreeViewer from './components/DetailPanel/FileTreeViewer';
import { buildTreeFromPaths } from './components/DetailPanel/FileTreeUtils';

export const LiveAgentMonitor: React.FC = () => {
  const [fileTree, setFileTree] = useState([]);
  const [activities, setActivities] = useState([]);

  useEffect(() => {
    // Connect to WebSocket
    const ws = new WebSocket('ws://localhost:8000/agent-activity');

    ws.onmessage = (event) => {
      const activity = JSON.parse(event.data);

      // Update activities list
      setActivities(prev => [...prev, activity]);

      // Rebuild file tree
      const newTree = buildTreeFromPaths(
        activities.map(a => ({
          path: a.filePath,
          status: getStatusFromAction(a.action),
          modifiedBy: a.agentId,
          lastModified: a.timestamp,
        }))
      );
      setFileTree(newTree);
    };

    return () => ws.close();
  }, [activities]);

  return (
    <FileTreeViewer
      files={fileTree}
      onFileSelect={(path) => {
        // Handle file selection
        showFileDetails(path);
      }}
    />
  );
};

function getStatusFromAction(
  action: string
): 'modified' | 'read' | 'created' | 'deleted' {
  switch (action) {
    case 'write':
      return 'modified';
    case 'create':
      return 'created';
    case 'delete':
      return 'deleted';
    default:
      return 'read';
  }
}
```

## Styling Integration

### Dark Theme (Default - No Changes Needed)

The component comes with a dark theme that matches HoloLoom's design system. No additional styling required.

### Custom Theme

Override CSS variables to customize:

```css
:root {
  --color-bg: #1f2937;
  --color-bg-dark: #111827;
  --color-border: #374151;
  --color-text: #d1d5db;
  --color-accent: #3b82f6;
}
```

### Container Sizing

```tsx
// Full height container
<div style={{ height: '600px', width: '100%' }}>
  <FileTreeViewer files={fileTree} />
</div>

// Flexible container
<div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
  <FileTreeViewer files={fileTree} />
</div>

// Responsive container
<div
  style={{
    height: 'calc(100vh - 200px)',
    width: '100%',
    '@media (max-width: 768px)': {
      height: '50vh',
    },
  }}
>
  <FileTreeViewer files={fileTree} />
</div>
```

## Showing File Details

### Basic Implementation

```tsx
export const AgentManagerPanel: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  const handleFileSelect = (path: string) => {
    setSelectedFile(path);
    showFileDetailsPanel(path);
  };

  return (
    <div style={{ display: 'flex', gap: '16px' }}>
      <div style={{ flex: 1 }}>
        <FileTreeViewer
          files={fileTree}
          onFileSelect={handleFileSelect}
        />
      </div>
      {selectedFile && (
        <div style={{ flex: 1 }}>
          <FileDetailsPanel filePath={selectedFile} />
        </div>
      )}
    </div>
  );
};
```

### With Diff Viewer

```tsx
import React, { useState } from 'react';
import FileTreeViewer from './components/DetailPanel/FileTreeViewer';
import DiffViewer from './components/DiffViewer'; // Your diff component

export const AgentFileMonitor: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  return (
    <div style={{ display: 'flex', gap: '16px', height: '100%' }}>
      <div style={{ flex: 0.4 }}>
        <FileTreeViewer
          files={fileTree}
          onFileSelect={setSelectedFile}
        />
      </div>
      {selectedFile && (
        <div style={{ flex: 0.6 }}>
          <DiffViewer
            filePath={selectedFile}
            before={getFileBefore(selectedFile)}
            after={getFileAfter(selectedFile)}
          />
        </div>
      )}
    </div>
  );
};
```

## Filtering Agent Activity

### Filter by Specific Agent

```tsx
import {
  filterByAgent,
  calculateTreeStats,
} from './components/DetailPanel/FileTreeUtils';

export const AgentActivityFilter: React.FC<{
  agentId: string;
}> = ({ agentId }) => {
  const allFiles = useSelector(state => state.agents.fileTree);

  // Filter to only show files modified by selected agent
  const agentFiles = filterByAgent(allFiles, agentId);

  // Calculate statistics
  const stats = calculateTreeStats(agentFiles);

  return (
    <div>
      <h3>
        {agentId} - {stats.totalFiles} files modified
      </h3>
      <FileTreeViewer files={agentFiles} />
    </div>
  );
};
```

### Filter by Recent Changes

```tsx
import {
  getRecentFiles,
  buildTreeFromPaths,
} from './components/DetailPanel/FileTreeUtils';

export const RecentChangesPanel: React.FC = () => {
  const allFiles = useSelector(state => state.agents.fileTree);

  // Get files modified in last hour
  const oneHourAgo = new Date(Date.now() - 3600000);
  const recentFiles = getRecentFiles(allFiles, oneHourAgo);

  // Rebuild tree with only recent files
  const recentTree = buildTreeFromPaths(
    recentFiles.map(f => ({
      path: f.path,
      status: f.status,
      modifiedBy: f.modifiedBy,
      lastModified: f.lastModified,
    }))
  );

  return (
    <div>
      <h3>Files Modified (Last Hour)</h3>
      <FileTreeViewer files={recentTree} />
    </div>
  );
};
```

## Testing Integration

### With Testing Library

```tsx
import { render, screen, fireEvent } from '@testing-library/react';
import AgentManagerPanel from './AgentManagerPanel';

test('displays file tree and handles selection', () => {
  const mockOnFileSelect = jest.fn();

  render(
    <AgentManagerPanel
      files={mockFiles}
      onFileSelect={mockOnFileSelect}
    />
  );

  // Search for a file
  const searchInput = screen.getByPlaceholderText('Search files...');
  fireEvent.change(searchInput, { target: { value: 'Header' } });

  // Select the file
  const headerFile = screen.getByText('Header.tsx');
  fireEvent.click(headerFile);

  expect(mockOnFileSelect).toHaveBeenCalledWith('/src/components/Header.tsx');
});
```

## Performance Optimization

### Memoization

```tsx
import { useMemo } from 'react';
import { buildTreeFromPaths } from './components/DetailPanel/FileTreeUtils';

export const AgentManagerPanel: React.FC<{
  activities: AgentActivity[];
}> = ({ activities }) => {
  // Memoize file tree building
  const fileTree = useMemo(
    () =>
      buildTreeFromPaths(
        activities.map(a => ({
          path: a.filePath,
          status: getStatusFromAction(a.action),
          modifiedBy: a.agentId,
          lastModified: a.timestamp,
        }))
      ),
    [activities]
  );

  return <FileTreeViewer files={fileTree} />;
};
```

### Lazy Loading Children

For very large trees (>5000 files), consider loading children on demand:

```tsx
const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set());

const handleExpand = async (path: string) => {
  // Fetch children if not already loaded
  if (!expandedPaths.has(path)) {
    const children = await fetchFileTreeChildren(path);
    updateTreeWithChildren(path, children);
    setExpandedPaths(prev => new Set([...prev, path]));
  }
};
```

## Common Integration Patterns

### Pattern 1: Multi-Agent Dashboard

```tsx
export const MultiAgentDashboard: React.FC = () => {
  const agents = useSelector(state => state.agents.list);
  const [selectedAgent, setSelectedAgent] = useState(agents[0].id);

  const agentFileTree = useSelector(
    state => state.agents.fileTreeByAgent[selectedAgent]
  );

  return (
    <div>
      <select value={selectedAgent} onChange={e => setSelectedAgent(e.target.value)}>
        {agents.map(agent => (
          <option key={agent.id} value={agent.id}>
            {agent.name}
          </option>
        ))}
      </select>
      <FileTreeViewer
        files={agentFileTree}
        activeAgentId={selectedAgent}
        agentColorMap={createColorMap(agents)}
      />
    </div>
  );
};
```

### Pattern 2: File Diff View

```tsx
export const FileDiffPanel: React.FC = () => {
  const [selectedPath, setSelectedPath] = useState<string | null>(null);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '16px' }}>
      <FileTreeViewer
        files={fileTree}
        onFileSelect={setSelectedPath}
      />
      {selectedPath && (
        <FileDiffViewer filePath={selectedPath} />
      )}
    </div>
  );
};
```

### Pattern 3: Real-Time Monitoring

```tsx
export const LiveAgentMonitor: React.FC = () => {
  const activities$ = useWebSocket('ws://localhost/activities');
  const [fileTree, setFileTree] = useState([]);

  useEffect(() => {
    const subscription = activities$.subscribe(activity => {
      setFileTree(prev => updateTreeWithActivity(prev, activity));
    });

    return () => subscription.unsubscribe();
  }, [activities$]);

  return (
    <FileTreeViewer
      files={fileTree}
      onFileSelect={path => showFileDiff(path)}
    />
  );
};
```

## Troubleshooting Integration

### Files Not Showing

1. Verify FileNode structure matches interface
2. Check `path` property is unique for each node
3. Ensure `children` are properly nested
4. Verify CSS file is imported

### Colors Not Appearing

1. Check `agentColorMap` keys match `modifiedBy` values
2. Verify hex color codes are valid
3. Check CSS file is loaded in browser
4. Clear browser cache

### Filters Not Working

1. Verify file statuses are valid: 'modified', 'read', 'created', 'deleted'
2. Check agent IDs in data match dropdown options
3. Verify search query is in correct format
4. Check for case sensitivity in search

### Performance Issues

1. Profile with DevTools Performance tab
2. Check tree size (>5000 files may need optimization)
3. Verify no N+1 queries in data fetching
4. Consider virtual scrolling for large trees

## Next Steps

1. ✅ Copy component files to project
2. ✅ Import FileTreeViewer in your component
3. ✅ Pass real file tree data
4. ✅ Hook up onFileSelect callback
5. ✅ Style container to match layout
6. ✅ Integrate with agent data stream
7. ✅ Test with keyboard and screen reader
8. ✅ Deploy to production

## Support & Documentation

- **README**: `FileTreeViewer.README.md` - Complete API reference
- **Examples**: `FileTreeViewer.example.tsx` - 6 working examples
- **Tests**: `FileTreeViewer.test.tsx` - 40+ test cases
- **Utils**: `FileTreeUtils.ts` - 25+ utility functions

## Questions?

Refer to the comprehensive documentation included with the component.
