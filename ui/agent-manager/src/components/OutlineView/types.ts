/**
 * Type definitions for Outline View components
 * TaskNode interface and related types for HoloLoom Agent Manager UI Phase 3
 */

export type StepType = 'query' | 'research' | 'verify' | 'synthesize' | 'execute';
export type StepStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

/**
 * Represents a single task node within an execution thread
 * Corresponds to a step in the agent execution pipeline
 */
export interface TaskNode {
  /** Unique identifier for this task node */
  id: string;

  /** ID of the parent thread containing this task */
  threadId: string;

  /** ID of parent task node (for hierarchical steps) */
  parentId?: string;

  /** IDs of child task nodes (for hierarchical decomposition) */
  childrenIds: string[];

  /** Nesting depth (0 = top-level step, 1+ = sub-steps) */
  depth: number;

  /** Type of step being executed */
  stepType: StepType;

  /** Human-readable name for the step */
  name: string;

  /** Optional query text associated with this step */
  query?: string;

  /** Current execution status */
  status: StepStatus;

  /** Progress percentage (0-100) */
  progressPct: number;

  /** Elapsed time in milliseconds */
  elapsedTimeMs: number;

  /** Number of tokens used in this step */
  tokensUsed: number;

  /** Confidence score (0.0-1.0) */
  confidence: number;

  /** IDs of tasks this step depends on */
  dependsOn: string[];

  /** IDs of tasks blocked by this step */
  blocks: string[];

  /** Whether this step is eligible for MRF injection */
  mrfEligible: boolean;

  /** Whether this step is eligible for MCTS injection */
  mctsEligible: boolean;

  /** Name of injection applied if any (e.g., 'mrf_verify', 'mcts_research') */
  injectionApplied?: string;
}

/**
 * Extended TaskNode with derived properties for rendering
 */
export interface TaskNodeRenderProps extends TaskNode {
  /** Whether this step is currently being hovered */
  isHovered?: boolean;

  /** Whether this step is selected */
  isSelected?: boolean;

  /** Whether this step is the active/focused step */
  isActive?: boolean;
}

/**
 * Props for status icon display
 */
export interface StatusIconProps {
  status: StepStatus;
  className?: string;
}

/**
 * Props for confidence display
 */
export interface ConfidenceProps {
  confidence: number;
  className?: string;
}

/**
 * Props for progress bar (single bar)
 */
export interface ProgressBarProps {
  progress: number;
  className?: string;
}

/**
 * Props for multi-dimensional progress bars
 */
export interface ProgressBarsProps {
  currentStep: number;
  totalSteps: number;
  elapsedTimeMs: number;
  timeBudgetMs?: number;
  tokensUsed: number;
  tokenBudget?: number;
  variant?: 'stacked' | 'inline' | 'detailed';
  size?: 'sm' | 'md' | 'lg';
  showPercentages?: boolean;
  showValues?: boolean;
  className?: string;
}

/**
 * Color type for progress bars
 */
export type ProgressColor = 'blue' | 'amber' | 'purple' | 'green' | 'red';
export type ProgressSize = 'sm' | 'md' | 'lg';
export type ProgressVariant = 'stacked' | 'inline' | 'detailed';
