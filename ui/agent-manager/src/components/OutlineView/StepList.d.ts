/**
 * StepList Component
 * Container for a list of task steps within a thread
 * Manages step hierarchy, visual connectors, and interaction state
 */
import React from 'react';
import { TaskNode } from './types';
interface StepListProps {
    /** Array of task nodes to display */
    steps: TaskNode[];
    /** ID of the parent thread */
    threadId: string;
    /** Root task node (optional, for header display) */
    rootTask?: TaskNode;
    /** Currently hovered step ID */
    hoveredStepId?: string | null;
    /** Currently selected step ID */
    selectedStepId?: string | null;
    /** Callback when hovering over a step */
    onStepHover?: (stepId: string | null) => void;
    /** Callback when clicking a step */
    onStepSelect?: (stepId: string) => void;
    /** Callback when injecting MRF */
    onInjectMRF?: (stepId: string) => void;
    /** Callback when injecting MCTS */
    onInjectMCTS?: (stepId: string) => void;
    /** Whether to show query preview on hover */
    showQueryPreview?: boolean;
    /** Custom className for the container */
    className?: string;
}
/**
 * StepList Component
 */
export declare const StepList: React.FC<StepListProps>;
export default StepList;
//# sourceMappingURL=StepList.d.ts.map