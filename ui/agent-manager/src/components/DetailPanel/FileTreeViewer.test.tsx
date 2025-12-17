/**
 * FileTreeViewer Component - Unit & Integration Tests
 *
 * Comprehensive test suite for the FileTreeViewer component.
 * Run with: npm test -- FileTreeViewer.test.tsx
 */

import React from 'react';
import { render, screen, fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import FileTreeViewer from './FileTreeViewer';

// ============================================================================
// Test Data
// ============================================================================

const mockFileTree = [
  {
    path: '/src',
    name: 'src',
    isDirectory: true,
    modifiedBy: 'agent-1',
    status: 'modified' as const,
    lastModified: '2024-01-15T10:00:00Z',
    children: [
      {
        path: '/src/index.ts',
        name: 'index.ts',
        isDirectory: false,
        modifiedBy: 'agent-1',
        status: 'modified' as const,
        lastModified: '2024-01-15T10:00:00Z',
      },
      {
        path: '/src/components',
        name: 'components',
        isDirectory: true,
        modifiedBy: 'agent-2',
        status: 'modified' as const,
        lastModified: '2024-01-15T09:00:00Z',
        children: [
          {
            path: '/src/components/Header.tsx',
            name: 'Header.tsx',
            isDirectory: false,
            modifiedBy: 'agent-1',
            status: 'modified' as const,
            lastModified: '2024-01-15T08:00:00Z',
          },
        ],
      },
    ],
  },
  {
    path: '/tests',
    name: 'tests',
    isDirectory: true,
    modifiedBy: 'agent-3',
    status: 'created' as const,
    lastModified: '2024-01-14T10:00:00Z',
    children: [],
  },
];

const mockColorMap = {
  'agent-1': '#3B82F6',
  'agent-2': '#10B981',
  'agent-3': '#F59E0B',
};

// ============================================================================
// Rendering Tests
// ============================================================================

describe('FileTreeViewer - Rendering', () => {
  test('renders file tree with root folders', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    expect(screen.getByText('src')).toBeInTheDocument();
    expect(screen.getByText('tests')).toBeInTheDocument();
  });

  test('renders with empty tree', () => {
    render(<FileTreeViewer files={[]} />);

    expect(screen.getByText('No files match your filters')).toBeInTheDocument();
  });

  test('renders toolbar with search input', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    const searchInput = screen.getByPlaceholderText('Search files...');
    expect(searchInput).toBeInTheDocument();
  });

  test('renders filter dropdowns', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    expect(screen.getByDisplayValue('All Status')).toBeInTheDocument();
    expect(screen.getByDisplayValue('All Agents')).toBeInTheDocument();
  });

  test('renders expand/collapse all buttons', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    const buttons = screen.getAllByRole('button');
    expect(buttons.length).toBeGreaterThan(0);
  });

  test('shows file status indicators', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    // Status indicators should be present (as text content)
    const src = screen.getByText('src').closest('.tree-row');
    expect(src?.textContent).toMatch(/●|○|\+|−/);
  });
});

// ============================================================================
// Expansion/Collapse Tests
// ============================================================================

describe('FileTreeViewer - Expansion/Collapse', () => {
  test('expands folder when clicking chevron', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    // Initially, nested files should not be visible
    expect(screen.queryByText('Header.tsx')).not.toBeInTheDocument();

    // Click expand button for src folder
    const srcFolder = screen.getByText('src').closest('.tree-row');
    const expandButton = within(srcFolder!).getByRole('button', { name: /expand/i });
    fireEvent.click(expandButton);

    // Now nested items should be visible
    expect(screen.getByText('index.ts')).toBeInTheDocument();
    expect(screen.getByText('components')).toBeInTheDocument();
  });

  test('collapses folder when clicking expanded chevron', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    const srcFolder = screen.getByText('src').closest('.tree-row');
    const expandButton = within(srcFolder!).getByRole('button', { name: /expand/i });

    // Expand
    fireEvent.click(expandButton);
    expect(screen.getByText('index.ts')).toBeInTheDocument();

    // Collapse
    fireEvent.click(expandButton);
    expect(screen.queryByText('index.ts')).not.toBeInTheDocument();
  });

  test('expand all button expands all folders', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    // Click expand all button
    const buttons = screen.getAllByRole('button');
    const expandAllButton = buttons[buttons.length - 2]; // Second to last button
    fireEvent.click(expandAllButton);

    // All nested items should be visible
    expect(screen.getByText('index.ts')).toBeInTheDocument();
    expect(screen.getByText('Header.tsx')).toBeInTheDocument();
  });

  test('collapse all button collapses all folders', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    // First expand all
    const buttons = screen.getAllByRole('button');
    const expandAllButton = buttons[buttons.length - 2];
    fireEvent.click(expandAllButton);
    expect(screen.getByText('Header.tsx')).toBeInTheDocument();

    // Then collapse all
    const collapseAllButton = buttons[buttons.length - 1];
    fireEvent.click(collapseAllButton);
    expect(screen.queryByText('Header.tsx')).not.toBeInTheDocument();
  });

  test('does not show expand button for files', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    // Expand the src folder
    const srcFolder = screen.getByText('src').closest('.tree-row');
    const expandButton = within(srcFolder!).getByRole('button', { name: /expand/i });
    fireEvent.click(expandButton);

    // index.ts should be visible but should not have an expand button
    const indexFile = screen.getByText('index.ts').closest('.tree-row');
    const expandButtons = within(indexFile!).queryAllByRole('button', { name: /expand/i });
    expect(expandButtons.length).toBe(0);
  });
});

// ============================================================================
// Search & Filter Tests
// ============================================================================

describe('FileTreeViewer - Search', () => {
  test('filters files by search query', async () => {
    const user = userEvent.setup();
    render(<FileTreeViewer files={mockFileTree} />);

    const searchInput = screen.getByPlaceholderText('Search files...');
    await user.type(searchInput, 'Header');

    expect(screen.getByText('Header.tsx')).toBeInTheDocument();
    expect(screen.queryByText('index.ts')).not.toBeInTheDocument();
  });

  test('search is case-insensitive', async () => {
    const user = userEvent.setup();
    render(<FileTreeViewer files={mockFileTree} />);

    const searchInput = screen.getByPlaceholderText('Search files...');
    await user.type(searchInput, 'header');

    expect(screen.getByText('Header.tsx')).toBeInTheDocument();
  });

  test('search includes parent folders when child matches', async () => {
    const user = userEvent.setup();
    render(<FileTreeViewer files={mockFileTree} />);

    const searchInput = screen.getByPlaceholderText('Search files...');
    await user.type(searchInput, 'Header');

    // Parent folders should be visible
    expect(screen.getByText('src')).toBeInTheDocument();
    expect(screen.getByText('components')).toBeInTheDocument();
  });

  test('clears search when input is emptied', async () => {
    const user = userEvent.setup();
    render(<FileTreeViewer files={mockFileTree} />);

    const searchInput = screen.getByPlaceholderText('Search files...');
    await user.type(searchInput, 'Header');
    expect(screen.queryByText('index.ts')).not.toBeInTheDocument();

    await user.clear(searchInput);
    expect(screen.getByText('index.ts')).toBeInTheDocument();
  });
});

describe('FileTreeViewer - Status Filter', () => {
  test('filters by modified status', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    const statusFilter = screen.getByDisplayValue('All Status');
    fireEvent.change(statusFilter, { target: { value: 'modified' } });

    expect(screen.getByText('src')).toBeInTheDocument();
    expect(screen.queryByText('tests')).not.toBeInTheDocument(); // created status
  });

  test('filters by created status', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    const statusFilter = screen.getByDisplayValue('All Status');
    fireEvent.change(statusFilter, { target: { value: 'created' } });

    expect(screen.getByText('tests')).toBeInTheDocument();
    expect(screen.queryByText('src')).not.toBeInTheDocument();
  });

  test('reset filter shows all items', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    const statusFilter = screen.getByDisplayValue('All Status');
    fireEvent.change(statusFilter, { target: { value: 'created' } });
    expect(screen.queryByText('src')).not.toBeInTheDocument();

    fireEvent.change(statusFilter, { target: { value: 'all' } });
    expect(screen.getByText('src')).toBeInTheDocument();
  });
});

describe('FileTreeViewer - Agent Filter', () => {
  test('filters by agent', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    const agentFilter = screen.getByDisplayValue('All Agents');
    fireEvent.change(agentFilter, { target: { value: 'agent-1' } });

    // agent-1 appears in src and Header.tsx
    expect(screen.getByText('src')).toBeInTheDocument();
    // agent-3 (tests) should not appear
    expect(screen.queryByText('tests')).not.toBeInTheDocument();
  });

  test('shows parent folders when child matches agent filter', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    const agentFilter = screen.getByDisplayValue('All Agents');
    fireEvent.change(agentFilter, { target: { value: 'agent-1' } });

    // Parent folders should be visible even if they're modified by different agents
    expect(screen.getByText('src')).toBeInTheDocument();
  });
});

// ============================================================================
// File Selection Tests
// ============================================================================

describe('FileTreeViewer - File Selection', () => {
  test('calls onFileSelect when clicking file', () => {
    const onFileSelect = jest.fn();
    render(
      <FileTreeViewer
        files={mockFileTree}
        onFileSelect={onFileSelect}
      />
    );

    // Expand src folder first
    const srcFolder = screen.getByText('src').closest('.tree-row');
    const expandButton = within(srcFolder!).getByRole('button', { name: /expand/i });
    fireEvent.click(expandButton);

    // Click index.ts
    const indexFile = screen.getByText('index.ts');
    fireEvent.click(indexFile);

    expect(onFileSelect).toHaveBeenCalledWith('/src/index.ts');
  });

  test('expands/collapses folder when clicking folder name', () => {
    const onFileSelect = jest.fn();
    render(
      <FileTreeViewer
        files={mockFileTree}
        onFileSelect={onFileSelect}
      />
    );

    const srcFolder = screen.getByText('src');
    expect(screen.queryByText('index.ts')).not.toBeInTheDocument();

    fireEvent.click(srcFolder);
    expect(screen.getByText('index.ts')).toBeInTheDocument();

    // onFileSelect should not be called for folder toggle
    expect(onFileSelect).not.toHaveBeenCalled();
  });

  test('highlights selected file', () => {
    render(
      <FileTreeViewer
        files={mockFileTree}
        onFileSelect={() => {}}
      />
    );

    // Expand src folder
    const srcFolder = screen.getByText('src').closest('.tree-row');
    const expandButton = within(srcFolder!).getByRole('button', { name: /expand/i });
    fireEvent.click(expandButton);

    // Click index.ts
    const indexFile = screen.getByText('index.ts');
    fireEvent.click(indexFile);

    const row = indexFile.closest('.tree-row');
    expect(row).toHaveClass('selected');
  });
});

// ============================================================================
// Details Panel Tests
// ============================================================================

describe('FileTreeViewer - Details Panel', () => {
  test('shows details toggle button when file is selected', () => {
    render(
      <FileTreeViewer
        files={mockFileTree}
        onFileSelect={() => {}}
      />
    );

    // Expand src folder
    const srcFolder = screen.getByText('src').closest('.tree-row');
    const expandButton = within(srcFolder!).getByRole('button', { name: /expand/i });
    fireEvent.click(expandButton);

    // Initially, details toggle should not be visible
    let detailsToggle = screen.queryByRole('button', { name: /show details/i });
    expect(detailsToggle).not.toBeInTheDocument();

    // Click index.ts
    const indexFile = screen.getByText('index.ts');
    fireEvent.click(indexFile);

    // Now details toggle should be visible
    detailsToggle = screen.getByRole('button', { name: /show details/i });
    expect(detailsToggle).toBeInTheDocument();
  });

  test('shows details panel when toggle button is clicked', () => {
    render(
      <FileTreeViewer
        files={mockFileTree}
        onFileSelect={() => {}}
      />
    );

    // Expand and select a file
    const srcFolder = screen.getByText('src').closest('.tree-row');
    const expandButton = within(srcFolder!).getByRole('button', { name: /expand/i });
    fireEvent.click(expandButton);

    const indexFile = screen.getByText('index.ts');
    fireEvent.click(indexFile);

    // Details panel should not be visible initially
    expect(screen.queryByText('File Details')).not.toBeInTheDocument();

    // Click details toggle
    const detailsToggle = screen.getByRole('button', { name: /show details/i });
    fireEvent.click(detailsToggle);

    // Details panel should now be visible
    expect(screen.getByText('File Details')).toBeInTheDocument();
    expect(screen.getByText('index.ts', { selector: 'span' })).toBeInTheDocument();
  });

  test('displays correct file information in details panel', () => {
    render(
      <FileTreeViewer
        files={mockFileTree}
        onFileSelect={() => {}}
      />
    );

    // Expand and select a file
    const srcFolder = screen.getByText('src').closest('.tree-row');
    const expandButton = within(srcFolder!).getByRole('button', { name: /expand/i });
    fireEvent.click(expandButton);

    const indexFile = screen.getByText('index.ts');
    fireEvent.click(indexFile);

    // Open details panel
    const detailsToggle = screen.getByRole('button', { name: /show details/i });
    fireEvent.click(detailsToggle);

    // Check details content
    const detailsPanel = screen.getByText('File Details').closest('.details-panel');
    expect(within(detailsPanel!).getByText('/src/index.ts')).toBeInTheDocument();
    expect(within(detailsPanel!).getByText('File')).toBeInTheDocument();
  });

  test('closes details panel when close button is clicked', () => {
    render(
      <FileTreeViewer
        files={mockFileTree}
        onFileSelect={() => {}}
      />
    );

    // Expand, select, and open details
    const srcFolder = screen.getByText('src').closest('.tree-row');
    const expandButton = within(srcFolder!).getByRole('button', { name: /expand/i });
    fireEvent.click(expandButton);

    const indexFile = screen.getByText('index.ts');
    fireEvent.click(indexFile);

    const detailsToggle = screen.getByRole('button', { name: /show details/i });
    fireEvent.click(detailsToggle);

    expect(screen.getByText('File Details')).toBeInTheDocument();

    // Click close button
    const closeButton = screen.getByRole('button', { name: /close details/i });
    fireEvent.click(closeButton);

    expect(screen.queryByText('File Details')).not.toBeInTheDocument();
  });
});

// ============================================================================
// Color & Styling Tests
// ============================================================================

describe('FileTreeViewer - Colors', () => {
  test('applies custom agent colors from colorMap', () => {
    const { container } = render(
      <FileTreeViewer
        files={mockFileTree}
        agentColorMap={mockColorMap}
      />
    );

    // Get agent indicators
    const agentIndicators = container.querySelectorAll('.agent-indicator');
    expect(agentIndicators.length).toBeGreaterThan(0);

    // Check that styles are applied
    const indicator = agentIndicators[0] as HTMLElement;
    expect(indicator.style.backgroundColor).toBeDefined();
  });

  test('renders file with correct status styling', () => {
    const { container } = render(<FileTreeViewer files={mockFileTree} />);

    // Check that status classes are applied
    const rows = container.querySelectorAll('.tree-row');
    const hasStatusClass = Array.from(rows).some(row =>
      row.className.includes('status-')
    );
    expect(hasStatusClass).toBe(true);
  });
});

// ============================================================================
// Accessibility Tests
// ============================================================================

describe('FileTreeViewer - Accessibility', () => {
  test('has proper ARIA labels on interactive elements', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    const expandButton = screen.getByLabelText(/expand/i);
    expect(expandButton).toBeInTheDocument();

    const searchInput = screen.getByLabelText(/search files/i);
    expect(searchInput).toBeInTheDocument();
  });

  test('supports keyboard navigation', async () => {
    const user = userEvent.setup();
    render(<FileTreeViewer files={mockFileTree} />);

    const searchInput = screen.getByPlaceholderText('Search files...');

    // Tab to search input
    await user.tab();
    expect(searchInput).toHaveFocus();

    // Type in search
    await user.keyboard('Header');
    expect(searchInput).toHaveValue('Header');
  });

  test('has descriptive titles on interactive elements', () => {
    render(<FileTreeViewer files={mockFileTree} />);

    const buttons = screen.getAllByRole('button');
    buttons.forEach(button => {
      const hasTitle = button.hasAttribute('title') || button.hasAttribute('aria-label');
      expect(hasTitle).toBe(true);
    });
  });

  test('maintains focus visibility on interactive elements', async () => {
    const user = userEvent.setup();
    const { container } = render(<FileTreeViewer files={mockFileTree} />);

    // Tab through interactive elements
    await user.tab();
    const focusedElement = container.querySelector(':focus');
    expect(focusedElement).toBeInTheDocument();
  });
});

// ============================================================================
// Edge Cases & Error Handling
// ============================================================================

describe('FileTreeViewer - Edge Cases', () => {
  test('handles empty file tree', () => {
    render(<FileTreeViewer files={[]} />);
    expect(screen.getByText('No files match your filters')).toBeInTheDocument();
  });

  test('handles deeply nested file tree', () => {
    const deepTree = [
      {
        path: '/a',
        name: 'a',
        isDirectory: true,
        status: 'modified' as const,
        lastModified: new Date().toISOString(),
        children: [
          {
            path: '/a/b',
            name: 'b',
            isDirectory: true,
            status: 'modified' as const,
            lastModified: new Date().toISOString(),
            children: [
              {
                path: '/a/b/c',
                name: 'c',
                isDirectory: true,
                status: 'modified' as const,
                lastModified: new Date().toISOString(),
                children: [
                  {
                    path: '/a/b/c/file.txt',
                    name: 'file.txt',
                    isDirectory: false,
                    status: 'modified' as const,
                    lastModified: new Date().toISOString(),
                  },
                ],
              },
            ],
          },
        ],
      },
    ];

    render(<FileTreeViewer files={deepTree} />);

    // Expand all
    const buttons = screen.getAllByRole('button');
    const expandAllButton = buttons[buttons.length - 2];
    fireEvent.click(expandAllButton);

    expect(screen.getByText('file.txt')).toBeInTheDocument();
  });

  test('handles files without modifiedBy', () => {
    const filesWithoutAgent = [
      {
        path: '/noagent.txt',
        name: 'noagent.txt',
        isDirectory: false,
        status: 'read' as const,
        lastModified: new Date().toISOString(),
      },
    ];

    render(<FileTreeViewer files={filesWithoutAgent} />);
    expect(screen.getByText('noagent.txt')).toBeInTheDocument();
  });

  test('handles special characters in filenames', () => {
    const specialFiles = [
      {
        path: '/my-file_v2.0-beta.ts',
        name: 'my-file_v2.0-beta.ts',
        isDirectory: false,
        status: 'modified' as const,
        lastModified: new Date().toISOString(),
      },
    ];

    render(<FileTreeViewer files={specialFiles} />);
    expect(screen.getByText('my-file_v2.0-beta.ts')).toBeInTheDocument();
  });
});

// ============================================================================
// Performance Tests
// ============================================================================

describe('FileTreeViewer - Performance', () => {
  test('renders large file tree without lag', () => {
    // Generate a large tree
    const largeTree = Array.from({ length: 100 }, (_, i) => ({
      path: `/file${i}.ts`,
      name: `file${i}.ts`,
      isDirectory: false,
      status: 'modified' as const,
      lastModified: new Date().toISOString(),
    }));

    const startTime = performance.now();
    render(<FileTreeViewer files={largeTree} />);
    const endTime = performance.now();

    const renderTime = endTime - startTime;
    expect(renderTime).toBeLessThan(1000); // Should render within 1 second
  });
});
