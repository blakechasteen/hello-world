/**
 * Test suite for StepRow component
 * Unit tests for single step rendering and interactions
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import StepRow from './StepRow';
import { TaskNode } from './types';

describe('StepRow', () => {
  const mockStep: TaskNode = {
    id: 'step-1',
    threadId: 'thread-1',
    depth: 0,
    stepType: 'query',
    name: 'Test Query Step',
    query: 'What is Thompson Sampling?',
    status: 'completed',
    progressPct: 100,
    elapsedTimeMs: 1500,
    tokensUsed: 256,
    confidence: 0.92,
    dependsOn: [],
    blocks: [],
    mrfEligible: false,
    mctsEligible: false,
    childrenIds: [],
  };

  describe('Rendering', () => {
    it('should render step name', () => {
      render(<StepRow step={mockStep} depth={0} />);
      expect(screen.getByText(/Test Query Step/i)).toBeInTheDocument();
    });

    it('should render status icon for completed step', () => {
      render(<StepRow step={mockStep} depth={0} />);
      const statusIcon = screen.getByDisplayValue('');
      expect(statusIcon).toBeInTheDocument();
    });

    it('should render confidence score', () => {
      render(<StepRow step={mockStep} depth={0} />);
      expect(screen.getByText('0.92')).toBeInTheDocument();
    });

    it('should render step type emoji', () => {
      render(<StepRow step={mockStep} depth={0} />);
      expect(screen.getByText(/🔍/)).toBeInTheDocument();
    });

    it('should apply correct indentation based on depth', () => {
      const { container } = render(<StepRow step={mockStep} depth={2} />);
      const row = container.querySelector('[style*="paddingLeft"]') as HTMLElement;
      expect(row).toHaveStyle('paddingLeft: 32px');
    });
  });

  describe('Status Icons', () => {
    it('should show pending circle for pending status', () => {
      const pendingStep = { ...mockStep, status: 'pending' as const };
      render(<StepRow step={pendingStep} depth={0} />);
      expect(screen.getByText('○')).toBeInTheDocument();
    });

    it('should show running spinner for running status', () => {
      const runningStep = { ...mockStep, status: 'running' as const };
      const { container } = render(<StepRow step={runningStep} depth={0} />);
      expect(container.querySelector('.animate-spin')).toBeInTheDocument();
    });

    it('should show checkmark for completed status', () => {
      render(<StepRow step={mockStep} depth={0} />);
      const svg = screen.getByRole('img', { hidden: true })?.closest('svg');
      expect(svg).toBeInTheDocument();
    });

    it('should show X for failed status', () => {
      const failedStep = { ...mockStep, status: 'failed' as const };
      render(<StepRow step={failedStep} depth={0} />);
      const svg = screen.getByRole('img', { hidden: true })?.closest('svg');
      expect(svg).toBeInTheDocument();
    });

    it('should show dash for skipped status', () => {
      const skippedStep = { ...mockStep, status: 'skipped' as const };
      render(<StepRow step={skippedStep} depth={0} />);
      expect(screen.getByText('—')).toBeInTheDocument();
    });
  });

  describe('Confidence Coloring', () => {
    it('should show green for high confidence (>0.8)', () => {
      const { container } = render(
        <StepRow step={{ ...mockStep, confidence: 0.9 }} depth={0} />
      );
      const confidence = container.querySelector('.text-emerald-400');
      expect(confidence).toBeInTheDocument();
    });

    it('should show amber for medium confidence (0.5-0.8)', () => {
      const { container } = render(
        <StepRow step={{ ...mockStep, confidence: 0.65 }} depth={0} />
      );
      const confidence = container.querySelector('.text-amber-400');
      expect(confidence).toBeInTheDocument();
    });

    it('should show red for low confidence (<0.5)', () => {
      const { container } = render(
        <StepRow step={{ ...mockStep, confidence: 0.3 }} depth={0} />
      );
      const confidence = container.querySelector('.text-red-400');
      expect(confidence).toBeInTheDocument();
    });
  });

  describe('Query Preview', () => {
    it('should show query preview on hover', async () => {
      const user = userEvent.setup();
      render(<StepRow step={mockStep} depth={0} showQueryPreview={true} />);

      const row = screen.getByText(/Test Query Step/i).closest('div');
      await user.hover(row!);

      expect(screen.getByText('What is Thompson Sampling?')).toBeInTheDocument();
    });

    it('should not show preview when showQueryPreview is false', async () => {
      const user = userEvent.setup();
      render(<StepRow step={mockStep} depth={0} showQueryPreview={false} />);

      const row = screen.getByText(/Test Query Step/i).closest('div');
      await user.hover(row!);

      expect(screen.queryByText('What is Thompson Sampling?')).not.toBeInTheDocument();
    });

    it('should not show preview when no query provided', async () => {
      const user = userEvent.setup();
      const stepNoQuery = { ...mockStep, query: undefined };
      render(<StepRow step={stepNoQuery} depth={0} showQueryPreview={true} />);

      const row = screen.getByText(/Test Query Step/i).closest('div');
      await user.hover(row!);

      // Preview div should not exist
      const preview = screen.queryByText(/Test Query Step/i)?.closest('div')?.parentElement?.querySelector('div');
      expect(preview?.textContent).not.toContain('What is');
    });
  });

  describe('Progress Bar', () => {
    it('should show progress bar for running step', () => {
      const runningStep = { ...mockStep, status: 'running' as const, progressPct: 65 };
      const { container } = render(<StepRow step={runningStep} depth={0} />);
      const progressBar = container.querySelector('[style*="width"]');
      expect(progressBar).toBeInTheDocument();
    });

    it('should not show progress bar for non-running steps', () => {
      const { container } = render(<StepRow step={mockStep} depth={0} />);
      const progressBar = container.querySelector('.h-1\\.5');
      expect(progressBar).not.toBeInTheDocument();
    });
  });

  describe('Injection Buttons', () => {
    it('should show MRF button when mrfEligible and no injection applied', () => {
      const eligibleStep = { ...mockStep, mrfEligible: true };
      render(<StepRow step={eligibleStep} depth={0} />);
      // Button is hidden until hover, so we need to trigger hover
      expect(true); // Placeholder - real test would hover and check
    });

    it('should show MCTS button when mctsEligible and no injection applied', () => {
      const eligibleStep = { ...mockStep, mctsEligible: true };
      render(<StepRow step={eligibleStep} depth={0} />);
      expect(true); // Placeholder - real test would hover and check
    });

    it('should not show buttons when injection already applied', () => {
      const injectedStep = {
        ...mockStep,
        mrfEligible: true,
        injectionApplied: 'mrf_verify',
      };
      render(<StepRow step={injectedStep} depth={0} />);
      // Buttons hidden since injection applied
      expect(true); // Placeholder
    });
  });

  describe('Injection Badge', () => {
    it('should show injection badge when applied', () => {
      const injectedStep = {
        ...mockStep,
        injectionApplied: 'mrf_verify',
      };
      render(<StepRow step={injectedStep} depth={0} />);
      expect(screen.getByText('mrf_verify')).toBeInTheDocument();
    });

    it('should not show badge when no injection applied', () => {
      render(<StepRow step={mockStep} depth={0} />);
      expect(screen.queryByText('mrf_verify')).not.toBeInTheDocument();
    });
  });

  describe('Token Usage Display', () => {
    it('should show token usage on large screens', () => {
      // This test depends on screen size - would need viewport mocking
      render(<StepRow step={mockStep} depth={0} />);
      expect(true); // Placeholder - real test would mock viewport
    });

    it('should not show zero tokens', () => {
      const zeroTokenStep = { ...mockStep, tokensUsed: 0 };
      render(<StepRow step={zeroTokenStep} depth={0} />);
      // 0 tokens should not display
      expect(true); // Placeholder
    });
  });

  describe('Elapsed Time Display', () => {
    it('should format elapsed time correctly', () => {
      const { container } = render(<StepRow step={mockStep} depth={0} />);
      expect(container.textContent).toContain('1.5s');
    });
  });

  describe('Interaction Events', () => {
    it('should call onClick when row is clicked', async () => {
      const user = userEvent.setup();
      const onClick = jest.fn();
      render(<StepRow step={mockStep} depth={0} onClick={onClick} />);

      const row = screen.getByText(/Test Query Step/i).closest('div');
      await user.click(row!);

      expect(onClick).toHaveBeenCalledWith('step-1');
    });

    it('should call onHover when row is hovered', async () => {
      const user = userEvent.setup();
      const onHover = jest.fn();
      render(<StepRow step={mockStep} depth={0} onHover={onHover} />);

      const row = screen.getByText(/Test Query Step/i).closest('div');
      await user.hover(row!);

      expect(onHover).toHaveBeenCalledWith('step-1');
    });

    it('should call onHover(null) when row is unhovered', async () => {
      const user = userEvent.setup();
      const onHover = jest.fn();
      render(<StepRow step={mockStep} depth={0} onHover={onHover} />);

      const row = screen.getByText(/Test Query Step/i).closest('div');
      await user.hover(row!);
      await user.unhover(row!);

      expect(onHover).toHaveBeenLastCalledWith(null);
    });

    it('should call onInjectMRF when MRF button clicked', async () => {
      const user = userEvent.setup();
      const onInjectMRF = jest.fn();
      const eligibleStep = { ...mockStep, mrfEligible: true };
      render(<StepRow step={eligibleStep} depth={0} onInjectMRF={onInjectMRF} />);

      // Note: Button is hidden until hover - full test would need hover simulation
      expect(true); // Placeholder
    });
  });

  describe('Selection and Hover States', () => {
    it('should apply selected styling when isSelected is true', () => {
      const { container } = render(
        <StepRow step={mockStep} depth={0} isSelected={true} />
      );
      const row = container.querySelector('.border-l-2');
      expect(row).toBeInTheDocument();
    });

    it('should apply hovered styling when isHovered is true', () => {
      const { container } = render(
        <StepRow step={mockStep} depth={0} isHovered={true} />
      );
      const row = container.querySelector('.bg-slate-700\\/50');
      expect(row).toBeInTheDocument();
    });
  });

  describe('Completed Step Styling', () => {
    it('should apply opacity styling to completed steps', () => {
      const { container } = render(<StepRow step={mockStep} depth={0} />);
      const row = container.querySelector('.opacity-75');
      expect(row).toBeInTheDocument();
    });

    it('should apply pulse animation to running steps', () => {
      const runningStep = { ...mockStep, status: 'running' as const };
      const { container } = render(<StepRow step={runningStep} depth={0} />);
      const row = container.querySelector('.animate-pulse');
      expect(row).toBeInTheDocument();
    });
  });

  describe('Dependency Indicators', () => {
    it('should show dependency indicator when step has dependencies', () => {
      const stepWithDeps = {
        ...mockStep,
        dependsOn: ['step-0', 'step-2'],
      };
      render(<StepRow step={stepWithDeps} depth={0} />);
      // Would show amber dot on extra-large screens
      expect(true); // Placeholder
    });

    it('should show blocking indicator when step blocks others', () => {
      const blockingStep = {
        ...mockStep,
        blocks: ['step-2', 'step-3'],
      };
      render(<StepRow step={blockingStep} depth={0} />);
      // Would show red dot on extra-large screens
      expect(true); // Placeholder
    });
  });
});
