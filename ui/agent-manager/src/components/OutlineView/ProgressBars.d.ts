/**
 * ProgressBars Component
 * Multi-dimensional progress tracking showing step, time, and token progress simultaneously
 *
 * Displays three progress dimensions:
 * 1. Step progress (currentStep / totalSteps) - Blue
 * 2. Time progress (elapsedTimeMs / timeBudgetMs) - Amber
 * 3. Token progress (tokensUsed / tokenBudget) - Purple
 *
 * Features:
 * - Three layout variants: stacked (vertical), inline (horizontal), detailed (with labels)
 * - Three size presets: sm, md, lg
 * - Graceful handling of undefined budgets
 * - Tufte-style: Maximum data, minimum ink
 * - Color-blind friendly palette
 * - Accessibility support (ARIA roles and attributes)
 *
 * Design Philosophy:
 * - Show data prominently, minimize decoration
 * - Meaningful colors (blue=steps, amber=time, purple=tokens)
 * - Responsive to container width
 * - Smooth animations for value changes
 */
import React from 'react';
import { ProgressSize } from './ProgressBar';
export type ProgressVariant = 'stacked' | 'inline' | 'detailed';
export interface ProgressBarsProps {
    /** Current step number (0-based or 1-based, should align with totalSteps) */
    currentStep: number;
    /** Total number of steps */
    totalSteps: number;
    /** Elapsed time in milliseconds */
    elapsedTimeMs: number;
    /** Time budget in milliseconds (optional, if not set time progress is hidden) */
    timeBudgetMs?: number;
    /** Tokens used */
    tokensUsed: number;
    /** Token budget (optional, if not set token progress is hidden) */
    tokenBudget?: number;
    /** Layout variant */
    variant?: ProgressVariant;
    /** Size preset (sm=2px, md=4px, lg=8px bars) */
    size?: ProgressSize;
    /** Whether to show percentage labels (for detailed variant) */
    showPercentages?: boolean;
    /** Whether to show actual values (for detailed variant) */
    showValues?: boolean;
    /** CSS class name for the container */
    className?: string;
}
/**
 * ProgressBars Component
 */
export declare const ProgressBars: React.FC<ProgressBarsProps>;
export default ProgressBars;
//# sourceMappingURL=ProgressBars.d.ts.map