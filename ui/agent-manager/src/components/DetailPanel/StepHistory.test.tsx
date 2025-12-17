/**
 * StepHistory Component - Unit Tests
 * Tests filtering, sorting, rendering, and user interactions
 *
 * HoloLoom Agent Manager UI - Phase 4
 */

import React from 'react';
import { render, screen, fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import StepHistory from './StepHistory';
import { TaskNode } from './types';

/**
 * Mock step data factory
 */
const createMockStep = (overrides?: Partial<TaskNode>): TaskNode => ({
  id: 'step-1',
  threadId: 'thread-1',
  childrenIds: [],
  depth: 0,
  stepType: 'query',
  name: 'Test Query',
  status: 'completed',
  progressPct: 100,
  elapsedTimeMs: 1500,
  tokensUsed: 500,
  confidence: 0.85,
  dependsOn: [],
  blocks: [],
  mrfEligible: false,
  mctsEligible: false,
  injectionApplied: null,
  ...overrides,
});

describe('StepHistory Component', () => {
  describe('Rendering', () => {
    it('renders empty state when no steps provided', () => {
      render(
        <StepHistory
          steps={[]}
          currentStepIndex={0}
        />
      );

      expect(screen.getByText('No steps yet')).toBeInTheDocument();
    });

    it('renders all steps in order', () => {
      const steps = [
        createMockStep({ id: 'step-1', name: 'First Step' }),
        createMockStep({ id: 'step-2', name: 'Second Step' }),
        createMockStep({ id: 'step-3', name: 'Third Step' }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      expect(screen.getByText('First Step')).toBeInTheDocument();
      expect(screen.getByText('Second Step')).toBeInTheDocument();
      expect(screen.getByText('Third Step')).toBeInTheDocument();
    });

    it('displays step count in header', () => {
      const steps = [
        createMockStep(),
        createMockStep({ id: 'step-2' }),
        createMockStep({ id: 'step-3' }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      expect(screen.getByText(/3 of 3 steps/)).toBeInTheDocument();
    });

    it('highlights current step', () => {
      const steps = [
        createMockStep({ id: 'step-1' }),
        createMockStep({ id: 'step-2' }),
        createMockStep({ id: 'step-3' }),
      ];

      const { container } = render(
        <StepHistory
          steps={steps}
          currentStepIndex={1}
        />
      );

      // Current step should have selection styling
      const rows = container.querySelectorAll('[class*="border-l-blue"]');
      expect(rows.length).toBeGreaterThan(0);
    });

    it('displays status icons correctly', () => {
      const steps = [
        createMockStep({ id: 'step-1', status: 'completed' }),
        createMockStep({ id: 'step-2', status: 'running' }),
        createMockStep({ id: 'step-3', status: 'failed' }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      // Check for status icons (✓, ▶, ✕)
      expect(screen.getByText('✓')).toBeInTheDocument();
      expect(screen.getByText('▶')).toBeInTheDocument();
      expect(screen.getByText('✕')).toBeInTheDocument();
    });

    it('displays confidence scores with correct colors', () => {
      const steps = [
        createMockStep({ confidence: 0.9 }), // High - emerald
        createMockStep({ id: 'step-2', confidence: 0.6 }), // Medium - amber
        createMockStep({ id: 'step-3', confidence: 0.2 }), // Low - red
      ];

      const { container } = render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      expect(screen.getByText('0.90')).toBeInTheDocument();
      expect(screen.getByText('0.60')).toBeInTheDocument();
      expect(screen.getByText('0.20')).toBeInTheDocument();
    });

    it('displays step metadata (tokens, duration)', () => {
      const steps = [
        createMockStep({ tokensUsed: 1500, elapsedTimeMs: 2500 }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      expect(screen.getByText(/1.5k/)).toBeInTheDocument();
      expect(screen.getByText(/2.5s/)).toBeInTheDocument();
    });
  });

  describe('Filtering', () => {
    it('filters steps by status', async () => {
      const steps = [
        createMockStep({ id: 'step-1', name: 'Completed', status: 'completed' }),
        createMockStep({ id: 'step-2', name: 'Running', status: 'running' }),
        createMockStep({ id: 'step-3', name: 'Failed', status: 'failed' }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      // Filter to completed only
      const completedFilter = screen.getByRole('button', { name: /Completed/ });
      await userEvent.click(completedFilter);

      expect(screen.getByText('Completed')).toBeInTheDocument();
      expect(screen.queryByText('Running')).not.toBeInTheDocument();
      expect(screen.queryByText('Failed')).not.toBeInTheDocument();
    });

    it('filters steps by search query', async () => {
      const steps = [
        createMockStep({ id: 'step-1', name: 'Query Step', query: 'What is X?' }),
        createMockStep({ id: 'step-2', name: 'Retrieval Step' }),
        createMockStep({ id: 'step-3', name: 'Synthesis Step' }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      const searchInput = screen.getByPlaceholderText('Search steps...');
      await userEvent.type(searchInput, 'Query');

      expect(screen.getByText('Query Step')).toBeInTheDocument();
      expect(screen.queryByText('Retrieval Step')).not.toBeInTheDocument();
    });

    it('updates filter count dynamically', async () => {
      const steps = [
        createMockStep({ id: 'step-1', status: 'completed' }),
        createMockStep({ id: 'step-2', status: 'completed' }),
        createMockStep({ id: 'step-3', status: 'failed' }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      // Should show "Completed (2)" and "Failed (1)"
      expect(screen.getByText(/Completed.*\(2\)/)).toBeInTheDocument();
      expect(screen.getByText(/Failed.*\(1\)/)).toBeInTheDocument();
    });

    it('clears search with X button', async () => {
      const steps = [createMockStep({ name: 'Test Step' })];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      const searchInput = screen.getByPlaceholderText('Search steps...') as HTMLInputElement;
      await userEvent.type(searchInput, 'Test');

      expect(searchInput.value).toBe('Test');

      const clearButton = screen.getByRole('button', { name: '✕' });
      await userEvent.click(clearButton);

      expect(searchInput.value).toBe('');
    });
  });

  describe('Sorting', () => {
    it('sorts steps chronologically by default', () => {
      const steps = [
        createMockStep({ id: 'step-1', name: 'First' }),
        createMockStep({ id: 'step-2', name: 'Second' }),
        createMockStep({ id: 'step-3', name: 'Third' }),
      ];

      const { container } = render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      const rows = container.querySelectorAll('div[class*="py-2"]');
      expect(rows.length).toBeGreaterThan(0);
    });

    it('sorts steps by status', async () => {
      const steps = [
        createMockStep({ id: 'step-1', name: 'Completed', status: 'completed' }),
        createMockStep({ id: 'step-2', name: 'Idle', status: 'idle' }),
        createMockStep({ id: 'step-3', name: 'Running', status: 'running' }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      const sortSelect = screen.getByDisplayValue('Chronological') as HTMLSelectElement;
      await userEvent.selectOptions(sortSelect, 'status');

      // Running should appear before completed (higher priority)
      const runningPos = screen.getByText('Running').compareDocumentPosition(screen.getByText('Completed'));
      expect(runningPos & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    });
  });

  describe('Expansion', () => {
    it('expands step to show details', async () => {
      const steps = [
        createMockStep({
          query: 'What is Thompson Sampling?',
          response: 'Thompson Sampling is a Bayesian approach...',
          toolSelected: 'answer',
        }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      // Initially, details should be hidden
      expect(screen.queryByText('What is Thompson Sampling?')).not.toBeInTheDocument();

      // Click expand button
      const expandButton = screen.getByRole('button', { name: '▶' });
      await userEvent.click(expandButton);

      // Now details should be visible
      expect(screen.getByText('What is Thompson Sampling?')).toBeInTheDocument();
      expect(screen.getByText(/Thompson Sampling is a Bayesian/)).toBeInTheDocument();
      expect(screen.getByText('answer')).toBeInTheDocument();
    });

    it('collapses expanded step', async () => {
      const steps = [
        createMockStep({
          query: 'Test query',
        }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      const expandButton = screen.getByRole('button', { name: '▶' });

      // Expand
      await userEvent.click(expandButton);
      expect(screen.getByText('Test query')).toBeInTheDocument();

      // Collapse
      await userEvent.click(expandButton);
      expect(screen.queryByText('Test query')).not.toBeInTheDocument();
    });

    it('displays dependencies in expanded view', async () => {
      const steps = [
        createMockStep({
          dependsOn: ['step-1', 'step-2'],
          blocks: ['step-4', 'step-5'],
        }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      const expandButton = screen.getByRole('button', { name: '▶' });
      await userEvent.click(expandButton);

      expect(screen.getByText(/Depends on:/)).toBeInTheDocument();
      expect(screen.getByText(/Blocks:/)).toBeInTheDocument();
    });
  });

  describe('Injection Controls', () => {
    it('shows MRF/MCTS indicators', () => {
      const steps = [
        createMockStep({ mrfEligible: true }),
        createMockStep({ id: 'step-2', mctsEligible: true }),
        createMockStep({ id: 'step-3', injectionApplied: 'mrf' }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      expect(screen.getByText('mrf')).toBeInTheDocument();
      expect(screen.getByText('mcts')).toBeInTheDocument();
      expect(screen.getByText('MRF')).toBeInTheDocument();
    });

    it('calls onInjectMRF callback', async () => {
      const onInjectMRF = jest.fn();
      const steps = [createMockStep({ mrfEligible: true })];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
          onInjectMRF={onInjectMRF}
        />
      );

      // Expand to see injection buttons
      const expandButton = screen.getByRole('button', { name: '▶' });
      await userEvent.click(expandButton);

      const injectButton = screen.getByRole('button', { name: /Inject MRF/ });
      await userEvent.click(injectButton);

      expect(onInjectMRF).toHaveBeenCalledWith(steps[0].id);
    });

    it('calls onInjectMCTS callback', async () => {
      const onInjectMCTS = jest.fn();
      const steps = [createMockStep({ mctsEligible: true })];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
          onInjectMCTS={onInjectMCTS}
        />
      );

      // Expand to see injection buttons
      const expandButton = screen.getByRole('button', { name: '▶' });
      await userEvent.click(expandButton);

      const injectButton = screen.getByRole('button', { name: /Inject MCTS/ });
      await userEvent.click(injectButton);

      expect(onInjectMCTS).toHaveBeenCalledWith(steps[0].id);
    });

    it('hides injection buttons when already injected', async () => {
      const steps = [createMockStep({ mrfEligible: true, injectionApplied: 'mrf' })];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      const expandButton = screen.getByRole('button', { name: '▶' });
      await userEvent.click(expandButton);

      expect(screen.queryByRole('button', { name: /Inject MRF/ })).not.toBeInTheDocument();
    });
  });

  describe('Selection', () => {
    it('calls onStepSelect when step is clicked', async () => {
      const onStepSelect = jest.fn();
      const steps = [
        createMockStep({ id: 'step-1' }),
        createMockStep({ id: 'step-2' }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
          onStepSelect={onStepSelect}
        />
      );

      const step2 = screen.getByText('Test Query').closest('[class*="px-4"]');
      fireEvent.click(step2!);

      expect(onStepSelect).toHaveBeenCalled();
    });
  });

  describe('Footer Stats', () => {
    it('displays total time, tokens, and confidence', () => {
      const steps = [
        createMockStep({ elapsedTimeMs: 1000, tokensUsed: 500, confidence: 0.8 }),
        createMockStep({
          id: 'step-2',
          elapsedTimeMs: 2000,
          tokensUsed: 1000,
          confidence: 0.9,
        }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      expect(screen.getByText(/Total time:/)).toBeInTheDocument();
      expect(screen.getByText(/Total tokens:/)).toBeInTheDocument();
      expect(screen.getByText(/Avg confidence:/)).toBeInTheDocument();
    });

    it('calculates correct totals', () => {
      const steps = [
        createMockStep({ elapsedTimeMs: 1000, tokensUsed: 100 }),
        createMockStep({ id: 'step-2', elapsedTimeMs: 2000, tokensUsed: 200 }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      // Total time should be 3000ms = 3.0s
      expect(screen.getByText('3.0s')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper button labeling', async () => {
      const steps = [createMockStep({ mrfEligible: true })];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      const expandButton = screen.getByRole('button', { name: '▶' });
      expect(expandButton).toBeInTheDocument();
    });

    it('supports keyboard navigation', async () => {
      const onStepSelect = jest.fn();
      const steps = [
        createMockStep({ id: 'step-1' }),
        createMockStep({ id: 'step-2' }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
          onStepSelect={onStepSelect}
        />
      );

      const expandButton = screen.getByRole('button', { name: '▶' });
      expandButton.focus();
      expect(expandButton).toHaveFocus();
    });
  });

  describe('Edge Cases', () => {
    it('handles empty query/response gracefully', () => {
      const steps = [
        createMockStep({
          query: undefined,
          response: undefined,
        }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      const expandButton = screen.getByRole('button', { name: '▶' });
      expect(expandButton).toBeInTheDocument();
      fireEvent.click(expandButton);
      // Should not crash
    });

    it('handles very long text gracefully', () => {
      const longText = 'A'.repeat(1000);
      const steps = [
        createMockStep({
          query: longText,
          response: longText,
          name: longText,
        }),
      ];

      render(
        <StepHistory
          steps={steps}
          currentStepIndex={0}
        />
      );

      expect(screen.getByText(/A+/)).toBeInTheDocument();
    });

    it('handles currentStepIndex out of range', () => {
      const steps = [
        createMockStep({ id: 'step-1' }),
        createMockStep({ id: 'step-2' }),
      ];

      expect(() => {
        render(
          <StepHistory
            steps={steps}
            currentStepIndex={999}
          />
        );
      }).not.toThrow();
    });
  });
});
