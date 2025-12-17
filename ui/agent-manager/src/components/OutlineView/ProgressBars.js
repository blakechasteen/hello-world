import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
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
import { useMemo, useCallback } from 'react';
import ProgressBar from './ProgressBar';
/**
 * Format milliseconds as human-readable time string
 */
const formatTime = (ms) => {
    if (ms < 1000) {
        return `${Math.round(ms)}ms`;
    }
    const seconds = (ms / 1000).toFixed(1);
    return `${seconds}s`;
};
/**
 * Format token count with K suffix for large numbers
 */
const formatTokens = (tokens) => {
    if (tokens >= 1000) {
        return `${(tokens / 1000).toFixed(1)}k`;
    }
    return tokens.toLocaleString();
};
/**
 * ProgressBars Component
 */
export const ProgressBars = ({ currentStep, totalSteps, elapsedTimeMs, timeBudgetMs, tokensUsed, tokenBudget, variant = 'stacked', size = 'md', showPercentages = false, showValues = false, className = '', }) => {
    // Calculate step progress
    const stepProgress = useMemo(() => {
        if (totalSteps === 0)
            return 0;
        return (currentStep / totalSteps) * 100;
    }, [currentStep, totalSteps]);
    // Calculate time progress
    const timeProgress = useMemo(() => {
        if (!timeBudgetMs || timeBudgetMs === 0)
            return undefined;
        return Math.min((elapsedTimeMs / timeBudgetMs) * 100, 100);
    }, [elapsedTimeMs, timeBudgetMs]);
    // Calculate token progress
    const tokenProgress = useMemo(() => {
        if (!tokenBudget || tokenBudget === 0)
            return undefined;
        return Math.min((tokensUsed / tokenBudget) * 100, 100);
    }, [tokensUsed, tokenBudget]);
    // Determine if budgets are exceeded
    const isTimeOverBudget = useMemo(() => {
        return timeBudgetMs ? elapsedTimeMs > timeBudgetMs : false;
    }, [elapsedTimeMs, timeBudgetMs]);
    const isTokensOverBudget = useMemo(() => {
        return tokenBudget ? tokensUsed > tokenBudget : false;
    }, [tokensUsed, tokenBudget]);
    // Format label for detailed variant
    const formatStepLabel = useCallback(() => {
        const pct = Math.min(Math.round(stepProgress), 100);
        if (showValues) {
            return `${currentStep}/${totalSteps}`;
        }
        if (showPercentages) {
            return `${pct}%`;
        }
        return `Step ${currentStep}/${totalSteps}`;
    }, [currentStep, totalSteps, stepProgress, showValues, showPercentages]);
    const formatTimeLabel = useCallback(() => {
        const pct = timeProgress ? Math.round(timeProgress) : 0;
        if (showValues) {
            return `${formatTime(elapsedTimeMs)} / ${formatTime(timeBudgetMs || 0)}`;
        }
        if (showPercentages) {
            return `${pct}%`;
        }
        return formatTime(elapsedTimeMs);
    }, [elapsedTimeMs, timeBudgetMs, timeProgress, showValues, showPercentages]);
    const formatTokenLabel = useCallback(() => {
        const pct = tokenProgress ? Math.round(tokenProgress) : 0;
        if (showValues) {
            return `${formatTokens(tokensUsed)} / ${formatTokens(tokenBudget || 0)}`;
        }
        if (showPercentages) {
            return `${pct}%`;
        }
        return formatTokens(tokensUsed);
    }, [tokensUsed, tokenBudget, tokenProgress, showValues, showPercentages]);
    // Render based on variant
    if (variant === 'inline') {
        return (_jsxs("div", { className: `flex items-center gap-3 ${className}`, role: "region", "aria-label": "Multi-dimensional progress tracking", children: [_jsx("div", { className: "flex-1 min-w-0", children: _jsx(ProgressBar, { value: currentStep, max: totalSteps, color: "blue", size: size, animated: currentStep < totalSteps }) }), timeBudgetMs && (_jsx("div", { className: "flex-1 min-w-0", children: _jsx(ProgressBar, { value: elapsedTimeMs, max: timeBudgetMs, color: isTimeOverBudget ? 'red' : 'amber', size: size, animated: elapsedTimeMs < timeBudgetMs }) })), tokenBudget && (_jsx("div", { className: "flex-1 min-w-0", children: _jsx(ProgressBar, { value: tokensUsed, max: tokenBudget, color: isTokensOverBudget ? 'red' : 'purple', size: size, animated: tokensUsed < tokenBudget }) }))] }));
    }
    if (variant === 'detailed') {
        return (_jsxs("div", { className: `flex flex-col gap-2 ${className}`, role: "region", "aria-label": "Multi-dimensional progress tracking", children: [_jsxs("div", { className: "space-y-0.5", children: [_jsx("div", { className: "text-xs text-slate-400 font-medium uppercase tracking-wide", children: "Steps" }), _jsx(ProgressBar, { value: currentStep, max: totalSteps, color: "blue", size: size, showLabel: true, label: formatStepLabel(), animated: currentStep < totalSteps })] }), timeBudgetMs && (_jsxs("div", { className: "space-y-0.5", children: [_jsx("div", { className: "text-xs text-slate-400 font-medium uppercase tracking-wide", children: "Time" }), _jsx(ProgressBar, { value: elapsedTimeMs, max: timeBudgetMs, color: isTimeOverBudget ? 'red' : 'amber', size: size, showLabel: true, label: formatTimeLabel(), animated: elapsedTimeMs < timeBudgetMs })] })), tokenBudget && (_jsxs("div", { className: "space-y-0.5", children: [_jsx("div", { className: "text-xs text-slate-400 font-medium uppercase tracking-wide", children: "Tokens" }), _jsx(ProgressBar, { value: tokensUsed, max: tokenBudget, color: isTokensOverBudget ? 'red' : 'purple', size: size, showLabel: true, label: formatTokenLabel(), animated: tokensUsed < tokenBudget })] }))] }));
    }
    // Default: stacked variant
    return (_jsxs("div", { className: `flex flex-col gap-1 ${className}`, role: "region", "aria-label": "Multi-dimensional progress tracking", children: [_jsx(ProgressBar, { value: currentStep, max: totalSteps, color: "blue", size: size, showLabel: showPercentages || showValues, label: showValues
                    ? `${currentStep}/${totalSteps}`
                    : showPercentages
                        ? `${Math.round(stepProgress)}%`
                        : undefined, animated: currentStep < totalSteps }), timeBudgetMs && (_jsx(ProgressBar, { value: elapsedTimeMs, max: timeBudgetMs, color: isTimeOverBudget ? 'red' : 'amber', size: size, showLabel: showPercentages || showValues, label: showValues
                    ? `${formatTime(elapsedTimeMs)} / ${formatTime(timeBudgetMs)}`
                    : showPercentages
                        ? `${Math.round(timeProgress || 0)}%`
                        : undefined, animated: elapsedTimeMs < timeBudgetMs })), tokenBudget && (_jsx(ProgressBar, { value: tokensUsed, max: tokenBudget, color: isTokensOverBudget ? 'red' : 'purple', size: size, showLabel: showPercentages || showValues, label: showValues
                    ? `${formatTokens(tokensUsed)} / ${formatTokens(tokenBudget)}`
                    : showPercentages
                        ? `${Math.round(tokenProgress || 0)}%`
                        : undefined, animated: tokensUsed < tokenBudget }))] }));
};
ProgressBars.displayName = 'ProgressBars';
export default ProgressBars;
//# sourceMappingURL=ProgressBars.js.map