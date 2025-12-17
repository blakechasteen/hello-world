import React from 'react';
import { AgentThread } from '../../stores/agentManagerStore';
/**
 * ThreadCard Component
 * Displays a collapsible card for an agent thread with summary and detailed step list
 *
 * Features:
 * - Status indicator with color-coding
 * - Editable thread name (click to edit)
 * - Priority controls (up/down vote buttons)
 * - Expanded view showing step list with MRF/MCTS injection buttons
 * - Visual indicators for dependencies
 * - Status-based border colors with animations
 */
interface ThreadCardProps {
    thread: AgentThread;
    isActive?: boolean;
    onSelect?: (threadId: string) => void;
}
export declare const ThreadCard: React.FC<ThreadCardProps>;
export default ThreadCard;
//# sourceMappingURL=ThreadCard.d.ts.map