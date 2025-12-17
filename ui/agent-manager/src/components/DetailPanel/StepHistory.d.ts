/**
 * StepHistory Component
 * Scrollable list showing all steps executed in a thread with full details
 * Features: Virtualized scrolling, filtering, sorting, detail expansion
 *
 * HoloLoom Agent Manager UI - Phase 4
 */
import React from 'react';
import { TaskNode } from './types';
/**
 * Props for StepHistory component
 */
interface StepHistoryProps {
    /** All steps in chronological order */
    steps: TaskNode[];
    /** Currently selected/executing step index */
    currentStepIndex: number;
    /** Callback when step is selected */
    onStepSelect?: (stepId: string) => void;
    /** Optional callback for injecting MRF */
    onInjectMRF?: (stepId: string) => void;
    /** Optional callback for injecting MCTS */
    onInjectMCTS?: (stepId: string) => void;
    /** Custom CSS class */
    className?: string;
}
/**
 * StepHistory - Main component
 * Displays all steps in a virtualized scrollable list with filtering and sorting
 */
export declare const StepHistory: React.FC<StepHistoryProps>;
export default StepHistory;
//# sourceMappingURL=StepHistory.d.ts.map