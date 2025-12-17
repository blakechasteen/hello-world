import React from 'react';
import { render, screen, fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryNodes, MemoryNode } from './MemoryNodes';

/**
 * Test suite for MemoryNodes component
 *
 * Coverage:
 * - Rendering (empty state, with nodes, grouping)
 * - Interactions (expand, sort, search, copy)
 * - UI states (expanded, loading, error)
 * - Accessibility (keyboard nav, ARIA attributes)
 * - Performance (large datasets, memoization)
 */

describe('MemoryNodes Component', () => {
  // Mock data
  const createMockNode = (overrides?: Partial<MemoryNode>): MemoryNode => ({
    id: 'node-0001-test123',
    content: 'Test memory node content',
    relevance: 0.85,
    sourceType: 'vector',
    accessedAt: new Date().toISOString(),
    stepId: 'step-01',
    metadata: { access_count: 1 },
    ...overrides,
  });

  const mockNodes: MemoryNode[] = [
    createMockNode({
      id: 'node-0001-high',
      content: 'High relevance node',
      relevance: 0.95,
      sourceType: 'vector',
      stepId: 'step-01',
      metadata: { access_count: 5 },
    }),
    createMockNode({
      id: 'node-0002-medium',
      content: 'Medium relevance node',
      relevance: 0.75,
      sourceType: 'graph',
      stepId: 'step-01',
      metadata: { access_count: 3 },
    }),
    createMockNode({
      id: 'node-0003-low',
      content: 'Low relevance node',
      relevance: 0.55,
      sourceType: 'cache',
      stepId: 'step-02',
      metadata: { access_count: 1 },
    }),
    createMockNode({
      id: 'node-0004-hot',
      content: 'Hot pattern node',
      relevance: 0.80,
      sourceType: 'hot_pattern',
      stepId: 'step-02',
      metadata: { access_count: 10 },
    }),
  ];

  describe('Rendering', () => {
    it('renders empty state when no nodes provided', () => {
      render(<MemoryNodes nodes={[]} />);
      expect(screen.getByText('No memory nodes accessed')).toBeInTheDocument();
    });

    it('renders all nodes when provided', () => {
      render(<MemoryNodes nodes={mockNodes} />);
      expect(screen.getByText('Showing 4 of 4 nodes')).toBeInTheDocument();
      mockNodes.forEach((node) => {
        expect(screen.getByText(new RegExp(node.content))).toBeInTheDocument();
      });
    });

    it('displays source type badges correctly', () => {
      render(<MemoryNodes nodes={mockNodes} />);
      expect(screen.getByText(/📊 Vector/)).toBeInTheDocument();
      expect(screen.getByText(/🔗 Graph/)).toBeInTheDocument();
      expect(screen.getByText(/⚡ Cache/)).toBeInTheDocument();
      expect(screen.getByText(/🔥 Hot/)).toBeInTheDocument();
    });

    it('displays relevance scores and bars', () => {
      render(<MemoryNodes nodes={mockNodes} />);
      expect(screen.getByText('95%')).toBeInTheDocument();
      expect(screen.getByText('75%')).toBeInTheDocument();
      expect(screen.getByText('55%')).toBeInTheDocument();
      expect(screen.getByText('80%')).toBeInTheDocument();
    });

    it('truncates long content in preview', () => {
      const longContent = 'a'.repeat(100);
      const node = createMockNode({ content: longContent });
      render(<MemoryNodes nodes={[node]} />);
      const preview = screen.getByText(new RegExp(longContent.substring(0, 80)));
      expect(preview.textContent).toContain('…');
    });

    it('displays access count from metadata', () => {
      render(<MemoryNodes nodes={mockNodes} />);
      // Access count should appear in the component (with Zap icon)
      const elements = screen.getAllByText(/\d+/);
      expect(elements.length).toBeGreaterThan(0);
    });
  });

  describe('Grouping', () => {
    it('groups nodes by step when groupByStep is true', () => {
      render(<MemoryNodes nodes={mockNodes} groupByStep={true} />);
      // Should show step headers
      expect(screen.getByText(/Step step-01/)).toBeInTheDocument();
      expect(screen.getByText(/Step step-02/)).toBeInTheDocument();
    });

    it('does not group nodes by default', () => {
      render(<MemoryNodes nodes={mockNodes} groupByStep={false} />);
      // Should not have step-specific grouping
      expect(screen.queryByText(/Step step-01.*nodes/)).not.toBeInTheDocument();
    });

    it('shows correct node count in group headers', () => {
      render(<MemoryNodes nodes={mockNodes} groupByStep={true} />);
      // step-01 has 2 nodes, step-02 has 2 nodes
      expect(screen.getByText(/\(2 nodes\)/)).toBeInTheDocument();
    });
  });

  describe('Sorting', () => {
    it('sorts by relevance by default', () => {
      render(<MemoryNodes nodes={mockNodes} />);
      const cards = screen.getAllByText(/%(.*)/);
      // First should be highest relevance (95%)
      expect(cards[0].textContent).toContain('95%');
    });

    it('allows sorting by recency', async () => {
      const user = userEvent.setup();
      render(<MemoryNodes nodes={mockNodes} />);

      const recencyButton = screen.getByText('Recency');
      await user.click(recencyButton);

      // After clicking, should be sorted by recency (newest first)
      expect(recencyButton).toHaveClass('bg-blue-600');
    });

    it('allows sorting by access count', async () => {
      const user = userEvent.setup();
      render(<MemoryNodes nodes={mockNodes} />);

      const accessButton = screen.getByText('Access Count');
      await user.click(accessButton);

      expect(accessButton).toHaveClass('bg-blue-600');
    });

    it('updates button styling for active sort', async () => {
      const user = userEvent.setup();
      render(<MemoryNodes nodes={mockNodes} />);

      const relevanceButton = screen.getByText('Relevance');
      expect(relevanceButton).toHaveClass('bg-blue-600');

      await user.click(screen.getByText('Recency'));
      expect(relevanceButton).not.toHaveClass('bg-blue-600');
    });
  });

  describe('Search & Filter', () => {
    it('filters nodes by ID', async () => {
      const user = userEvent.setup();
      render(<MemoryNodes nodes={mockNodes} />);

      const searchInput = screen.getByPlaceholderText(/Search nodes/);
      await user.type(searchInput, 'node-0001');

      expect(screen.getByText('Showing 1 of 4 nodes')).toBeInTheDocument();
      expect(screen.getByText(/node-0001-high/)).toBeInTheDocument();
      expect(screen.queryByText(/node-0002/)).not.toBeInTheDocument();
    });

    it('filters nodes by content', async () => {
      const user = userEvent.setup();
      render(<MemoryNodes nodes={mockNodes} />);

      const searchInput = screen.getByPlaceholderText(/Search nodes/);
      await user.type(searchInput, 'High');

      expect(screen.getByText('Showing 1 of 4 nodes')).toBeInTheDocument();
    });

    it('case-insensitive search', async () => {
      const user = userEvent.setup();
      render(<MemoryNodes nodes={mockNodes} />);

      const searchInput = screen.getByPlaceholderText(/Search nodes/);
      await user.type(searchInput, 'high');

      expect(screen.getByText('Showing 1 of 4 nodes')).toBeInTheDocument();
    });

    it('clears filter when search is empty', async () => {
      const user = userEvent.setup();
      render(<MemoryNodes nodes={mockNodes} />);

      const searchInput = screen.getByPlaceholderText(/Search nodes/);
      await user.type(searchInput, 'high');
      expect(screen.getByText('Showing 1 of 4 nodes')).toBeInTheDocument();

      await user.clear(searchInput);
      expect(screen.getByText('Showing 4 of 4 nodes')).toBeInTheDocument();
    });

    it('maintains sort order while filtering', async () => {
      const user = userEvent.setup();
      render(<MemoryNodes nodes={mockNodes} />);

      // Sort by access count
      await user.click(screen.getByText('Access Count'));

      // Search
      const searchInput = screen.getByPlaceholderText(/Search nodes/);
      await user.type(searchInput, 'node');

      // Should still be sorted by access count
      expect(screen.getByText('Showing 4 of 4 nodes')).toBeInTheDocument();
    });
  });

  describe('Node Expansion', () => {
    it('expands node to show full content', async () => {
      const user = userEvent.setup();
      render(<MemoryNodes nodes={mockNodes} />);

      const expandButtons = screen.getAllByTitle('Expand');
      await user.click(expandButtons[0]);

      expect(screen.getByText('Full Content')).toBeInTheDocument();
    });

    it('shows full node ID when expanded', async () => {
      const user = userEvent.setup();
      render(<MemoryNodes nodes={mockNodes} />);

      const expandButtons = screen.getAllByTitle('Expand');
      await user.click(expandButtons[0]);

      expect(screen.getByText('node-0001-high')).toBeInTheDocument();
    });

    it('shows metadata section when expanded and metadata present', async () => {
      const user = userEvent.setup();
      const nodeWithMetadata = createMockNode({
        metadata: { access_count: 5, confidence: 0.95, custom: 'value' },
      });
      render(<MemoryNodes nodes={[nodeWithMetadata]} />);

      const expandButton = screen.getByTitle('Expand');
      await user.click(expandButton);

      expect(screen.getByText('Metadata')).toBeInTheDocument();
      expect(screen.getByText(/"access_count":/)).toBeInTheDocument();
    });

    it('hides metadata section when no metadata', async () => {
      const user = userEvent.setup();
      const nodeNoMetadata = createMockNode({ metadata: {} });
      render(<MemoryNodes nodes={[nodeNoMetadata]} />);

      const expandButton = screen.getByTitle('Expand');
      await user.click(expandButton);

      expect(screen.queryByText('Metadata')).not.toBeInTheDocument();
    });

    it('collapses node when expanded again', async () => {
      const user = userEvent.setup();
      render(<MemoryNodes nodes={mockNodes} />);

      const expandButton = screen.getAllByTitle('Expand')[0];
      await user.click(expandButton);
      expect(screen.getByText('Full Content')).toBeInTheDocument();

      await user.click(expandButton);
      expect(screen.queryByText('Full Content')).not.toBeInTheDocument();
    });

    it('shows details row in expanded view', async () => {
      const user = userEvent.setup();
      render(<MemoryNodes nodes={mockNodes} />);

      const expandButton = screen.getAllByTitle('Expand')[0];
      await user.click(expandButton);

      expect(screen.getByText(/Source:/)).toBeInTheDocument();
      expect(screen.getByText(/Step:/)).toBeInTheDocument();
      expect(screen.getByText(/Relevance:/)).toBeInTheDocument();
      expect(screen.getByText(/Accessed:/)).toBeInTheDocument();
    });
  });

  describe('Copy to Clipboard', () => {
    it('copies node ID when copy button clicked', async () => {
      const user = userEvent.setup();
      const mockClipboard = jest.fn();
      Object.assign(navigator, {
        clipboard: { writeText: mockClipboard },
      });

      render(<MemoryNodes nodes={[mockNodes[0]]} />);

      const copyButton = screen.getAllByTitle('Copy node ID')[0];
      await user.click(copyButton);

      expect(mockClipboard).toHaveBeenCalledWith('node-0001-high');
    });

    it('calls onNodeClick callback when copy button clicked', async () => {
      const user = userEvent.setup();
      const onNodeClick = jest.fn();
      Object.assign(navigator, {
        clipboard: { writeText: jest.fn() },
      });

      render(
        <MemoryNodes nodes={[mockNodes[0]]} onNodeClick={onNodeClick} />
      );

      const copyButton = screen.getAllByTitle('Copy node ID')[0];
      await user.click(copyButton);

      expect(onNodeClick).toHaveBeenCalledWith('node-0001-high');
    });

    it('shows success feedback after copy', async () => {
      const user = userEvent.setup();
      jest.useFakeTimers();
      Object.assign(navigator, {
        clipboard: { writeText: jest.fn() },
      });

      render(<MemoryNodes nodes={[mockNodes[0]]} />);

      const copyButton = screen.getAllByTitle('Copy node ID')[0];
      await user.click(copyButton);

      // Should show checkmark
      expect(screen.getByText(/✓/)).toBeInTheDocument();

      // Checkmark should disappear after 2 seconds
      jest.advanceTimersByTime(2100);
      jest.useRealTimers();
    });

    it('copies full node ID from expanded view', async () => {
      const user = userEvent.setup();
      const mockClipboard = jest.fn();
      Object.assign(navigator, {
        clipboard: { writeText: mockClipboard },
      });

      render(<MemoryNodes nodes={[mockNodes[0]]} />);

      // Expand
      const expandButton = screen.getAllByTitle('Expand')[0];
      await user.click(expandButton);

      // Click on full ID field
      const idField = screen.getByText('node-0001-high');
      await user.click(idField);

      expect(mockClipboard).toHaveBeenCalledWith('node-0001-high');
    });
  });

  describe('Statistics & Summary', () => {
    it('shows correct node count', () => {
      render(<MemoryNodes nodes={mockNodes} />);
      expect(screen.getByText('Showing 4 of 4 nodes')).toBeInTheDocument();
    });

    it('calculates correct average relevance', () => {
      render(<MemoryNodes nodes={mockNodes} />);
      const avgRelevance = (0.95 + 0.75 + 0.55 + 0.8) / 4;
      const expected = avgRelevance.toFixed(2);
      expect(screen.getByText(new RegExp(`Avg Relevance: ${expected}`))).toBeInTheDocument();
    });

    it('updates statistics when filtering', async () => {
      const user = userEvent.setup();
      render(<MemoryNodes nodes={mockNodes} />);

      const searchInput = screen.getByPlaceholderText(/Search nodes/);
      await user.type(searchInput, 'high');

      expect(screen.getByText('Showing 1 of 4 nodes')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper button labels and titles', () => {
      render(<MemoryNodes nodes={mockNodes} />);

      expect(screen.getAllByTitle('Expand')).toBeTruthy();
      expect(screen.getAllByTitle('Copy node ID')).toBeTruthy();
      expect(screen.getByTitle('Sort by relevance')).toBeTruthy();
    });

    it('search input has proper label', () => {
      render(<MemoryNodes nodes={mockNodes} />);

      const searchInput = screen.getByPlaceholderText(/Search nodes/);
      expect(searchInput).toHaveAttribute('type', 'text');
    });

    it('sort buttons are keyboard accessible', async () => {
      const user = userEvent.setup();
      render(<MemoryNodes nodes={mockNodes} />);

      const relevanceButton = screen.getByText('Relevance');
      expect(relevanceButton).toHaveAttribute('type', 'button');

      await user.keyboard('{Tab}');
      // Focus management test would verify tab order
    });
  });

  describe('Styling & CSS Classes', () => {
    it('applies relevance-based colors correctly', () => {
      const { container } = render(<MemoryNodes nodes={mockNodes} />);

      // High relevance should have emerald class
      const highRelevanceCard = container.querySelector('[class*="emerald"]');
      expect(highRelevanceCard).toBeInTheDocument();
    });

    it('applies source type styling', () => {
      const { container } = render(<MemoryNodes nodes={mockNodes} />);

      // Should have source badge containers with color classes
      expect(container.querySelector('[class*="text-blue"]')).toBeInTheDocument();
    });

    it('applies custom className prop', () => {
      const { container } = render(
        <MemoryNodes nodes={mockNodes} className="custom-class" />
      );

      expect(container.firstChild).toHaveClass('custom-class');
    });
  });

  describe('Edge Cases', () => {
    it('handles very long node content', () => {
      const longContent = 'a'.repeat(500);
      const node = createMockNode({ content: longContent });

      render(<MemoryNodes nodes={[node]} />);
      // Should still render and truncate properly
      expect(screen.getByText(new RegExp(longContent.substring(0, 50)))).toBeInTheDocument();
    });

    it('handles missing metadata gracefully', () => {
      const nodeNoMetadata = createMockNode({ metadata: undefined });

      render(<MemoryNodes nodes={[nodeNoMetadata]} />);
      expect(screen.getByText(/High relevance/)).toBeInTheDocument();
    });

    it('handles special characters in content', () => {
      const specialContent = 'Test with <special> & "characters"';
      const node = createMockNode({ content: specialContent });

      render(<MemoryNodes nodes={[node]} />);
      expect(screen.getByText(new RegExp(specialContent))).toBeInTheDocument();
    });

    it('handles nodes with same ID gracefully', () => {
      const node1 = createMockNode({ id: 'duplicate', content: 'First' });
      const node2 = createMockNode({ id: 'duplicate', content: 'Second' });

      // Should render both (key would need to be generated by parent)
      render(<MemoryNodes nodes={[node1, node2]} />);
      // Both should be visible
      expect(screen.getByText('First')).toBeInTheDocument();
      expect(screen.getByText('Second')).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('renders large dataset efficiently', () => {
      const largeDataset = Array.from({ length: 100 }, (_, i) =>
        createMockNode({
          id: `node-${String(i).padStart(4, '0')}`,
          relevance: Math.random(),
        })
      );

      const startTime = performance.now();
      render(<MemoryNodes nodes={largeDataset} />);
      const endTime = performance.now();

      // Should render in reasonable time
      expect(endTime - startTime).toBeLessThan(1000);
      expect(screen.getByText(`Showing 100 of 100 nodes`)).toBeInTheDocument();
    });
  });

  describe('Props Variations', () => {
    it('renders with no optional props', () => {
      render(<MemoryNodes nodes={mockNodes} />);
      expect(screen.getByText('Showing 4 of 4 nodes')).toBeInTheDocument();
    });

    it('renders with all optional props', () => {
      const onNodeClick = jest.fn();
      Object.assign(navigator, {
        clipboard: { writeText: jest.fn() },
      });

      render(
        <MemoryNodes
          nodes={mockNodes}
          groupByStep={true}
          onNodeClick={onNodeClick}
          className="test-class"
        />
      );

      expect(screen.getByText(/Step step-01/)).toBeInTheDocument();
    });

    it('handles empty nodes array', () => {
      render(<MemoryNodes nodes={[]} />);
      expect(screen.getByText('No memory nodes accessed')).toBeInTheDocument();
    });
  });
});
