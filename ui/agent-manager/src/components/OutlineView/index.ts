/**
 * OutlineView Components
 * Exports for HoloLoom Agent Manager UI Phase 3 - Outline View
 */

// Main Components
export { OutlineView, default } from './OutlineView';
export { default as ThreadCard } from './ThreadCard';

// Sub-Components
export { default as StepRow } from './StepRow';
export { default as StepList } from './StepList';
export { PriorityControls } from './PriorityControls';
export { ThreadControls } from './ThreadControls';
export { InjectMenu } from './InjectMenu';
export { default as ProgressBar } from './ProgressBar';
export { default as ProgressBars } from './ProgressBars';

// Types (from types.ts - shared types)
export type {
  TaskNode,
  TaskNodeRenderProps,
  StatusIconProps,
  ConfidenceProps,
  ProgressBarProps,
  ProgressBarsProps,
  ProgressColor,
  ProgressSize,
  ProgressVariant,
  StepType,
  StepStatus,
} from './types';
