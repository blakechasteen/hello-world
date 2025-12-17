/**
 * StepRow Component
 * Represents a single task step in the execution outline
 * Shows status, progress, confidence, and action buttons
 */
import React from 'react';
import { TaskNode } from './types';
interface StepRowProps {
    /** The task node to display */
    step: TaskNode;
    /** Depth for indentation (in levels, 12px per level) */
    depth: number;
    /** Whether this row is currently hovered */
    isHovered?: boolean;
    /** Whether this row is selected */
    isSelected?: boolean;
    /** Callback when hovering */
    onHover?: (stepId: string | null) => void;
    /** Callback when clicking */
    onClick?: (stepId: string) => void;
    /** Callback when injecting MRF */
    onInjectMRF?: (stepId: string) => void;
    /** Callback when injecting MCTS */
    onInjectMCTS?: (stepId: string) => void;
    /** Whether to show detailed query on hover */
    showQueryPreview?: boolean;
}
/**
 * StepRow Component
 */
export declare const StepRow: React.FC<StepRowProps>;
export default StepRow;
//# sourceMappingURL=StepRow.d.ts.map