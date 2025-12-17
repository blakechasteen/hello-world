/**
 * DetailPanel Component
 * Main expanded view container that shows detailed information about a selected thread
 *
 * Features:
 * - Displays selected thread's full details (name, status, agent type, reasoning mode)
 * - Multi-dimensional progress tracking (steps, time, tokens) with ProgressBars component
 * - Confidence and epistemic confidence displays
 * - Tabbed interface for History, Memory, and Files views
 * - Shows dependencies (depends_on, blocks) with visual indicators
 * - Smooth slide-in animation from right
 * - Editable thread name with inline editing
 * - Responsive layout with min/max widths
 *
 * Theme:
 * - Dark theme: bg-slate-900 for panel, bg-slate-800 for cards
 * - Borders: border-slate-700
 * - Text: text-slate-100 for headings, text-slate-400 for secondary
 * - Smooth animations and transitions
 *
 * Dependencies:
 * - useAgentManagerStore: Zustand store for agent state
 * - StatusBadge: Component for status display
 * - ProgressBars: Multi-dimensional progress visualization
 *
 * Tab Layout:
 * - History: Shows step-by-step reasoning trace (StepHistory component)
 * - Memory: Shows memory nodes and knowledge graph access (MemoryNodes component)
 * - Files: Shows files accessed/modified during execution (FileTreeViewer component)
 *
 * Usage:
 * ```tsx
 * const [selectedThreadId, setSelectedThreadId] = useState<string | null>(null);
 *
 * {selectedThreadId && (
 *   <DetailPanel
 *     threadId={selectedThreadId}
 *     onClose={() => setSelectedThreadId(null)}
 *   />
 * )}
 * ```
 */
import React from 'react';
/**
 * DetailPanelProps
 * Props for DetailPanel component
 */
export interface DetailPanelProps {
    /** ID of the thread to display details for */
    threadId: string;
    /** Callback when user closes the detail panel */
    onClose: () => void;
}
/**
 * DetailPanel Component
 * Shows comprehensive details for a selected agent thread
 *
 * Component structure:
 * 1. Header - Title, status badge, close button
 * 2. Confidence section - Confidence and epistemic confidence bars
 * 3. Progress section - Multi-dimensional progress (steps, time, tokens)
 * 4. Dependencies - Shows depends_on and blocks relationships
 * 5. Tabs - History, Memory, Files navigation
 * 6. Tab content - Context-specific content for each tab
 * 7. Footer - Metadata timestamps
 */
export declare const DetailPanel: React.FC<DetailPanelProps>;
export default DetailPanel;
//# sourceMappingURL=DetailPanel.d.ts.map