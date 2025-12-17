/**
 * PriorityControls Component
 * Upvote/downvote controls for thread priority management
 * Supports both vertical and horizontal layouts with keyboard accessibility
 */
import React from 'react';
interface PriorityControlsProps {
    /** Thread ID to control */
    threadId: string;
    /** Current priority value (0-100) */
    priority: number;
    /** Size of the controls */
    size?: 'sm' | 'md';
    /** Layout orientation */
    orientation?: 'horizontal' | 'vertical';
    /** Custom CSS class */
    className?: string;
    /** Optional callback when priority changes */
    onChange?: (newPriority: number) => void;
}
export declare const PriorityControls: React.FC<PriorityControlsProps>;
export default PriorityControls;
//# sourceMappingURL=PriorityControls.d.ts.map