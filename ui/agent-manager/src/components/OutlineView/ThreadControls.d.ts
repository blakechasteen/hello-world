/**
 * ThreadControls Component
 * Pause/Resume/Cancel controls for thread lifecycle management
 * Shows contextual buttons based on thread status
 */
import React from 'react';
interface ThreadControlsProps {
    /** Thread ID to control */
    threadId: string;
    /** Current thread status */
    status: 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
    /** Size of the controls */
    size?: 'sm' | 'md';
    /** Custom CSS class */
    className?: string;
    /** Show confirmation dialog before cancelling */
    showCancelConfirm?: boolean;
    /** Optional callback when action is performed */
    onAction?: (action: 'pause' | 'resume' | 'cancel' | 'retry') => void;
}
export declare const ThreadControls: React.FC<ThreadControlsProps>;
export default ThreadControls;
//# sourceMappingURL=ThreadControls.d.ts.map