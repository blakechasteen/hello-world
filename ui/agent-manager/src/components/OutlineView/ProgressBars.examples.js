import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * ProgressBars Component Examples
 * Demonstrates all variants, sizes, and configuration options
 *
 * Usage:
 * - Copy examples into your component or documentation
 * - Use as reference for integrating ProgressBars into your UI
 */
import { useState, useEffect } from 'react';
import ProgressBars from './ProgressBars';
/**
 * Example 1: Stacked Variant (Default)
 * Vertical layout with three bars stacked on top of each other
 */
export const StackedExample = () => {
    const [currentStep, setCurrentStep] = useState(3);
    useEffect(() => {
        const interval = setInterval(() => {
            setCurrentStep((prev) => (prev < 7 ? prev + 1 : 0));
        }, 2000);
        return () => clearInterval(interval);
    }, []);
    return (_jsxs("div", { className: "p-4 bg-slate-900 rounded border border-slate-700", children: [_jsx("h3", { className: "text-sm font-semibold text-slate-100 mb-3", children: "Stacked Variant (Default)" }), _jsx(ProgressBars, { currentStep: currentStep, totalSteps: 7, elapsedTimeMs: currentStep * 1500, timeBudgetMs: 10500, tokensUsed: currentStep * 250, tokenBudget: 1750, variant: "stacked", size: "md" })] }));
};
/**
 * Example 2: Inline Variant
 * Horizontal layout with three bars side by side
 */
export const InlineExample = () => {
    const [currentStep, setCurrentStep] = useState(5);
    useEffect(() => {
        const interval = setInterval(() => {
            setCurrentStep((prev) => (prev < 10 ? prev + 1 : 0));
        }, 1500);
        return () => clearInterval(interval);
    }, []);
    return (_jsxs("div", { className: "p-4 bg-slate-900 rounded border border-slate-700", children: [_jsx("h3", { className: "text-sm font-semibold text-slate-100 mb-3", children: "Inline Variant (Side by Side)" }), _jsx(ProgressBars, { currentStep: currentStep, totalSteps: 10, elapsedTimeMs: currentStep * 1200, timeBudgetMs: 12000, tokensUsed: currentStep * 350, tokenBudget: 3500, variant: "inline", size: "sm" })] }));
};
/**
 * Example 3: Detailed Variant
 * Stacked with labels, headers, and value display
 */
export const DetailedExample = () => {
    const [currentStep, setCurrentStep] = useState(2);
    useEffect(() => {
        const interval = setInterval(() => {
            setCurrentStep((prev) => (prev < 5 ? prev + 1 : 0));
        }, 2500);
        return () => clearInterval(interval);
    }, []);
    return (_jsxs("div", { className: "p-4 bg-slate-900 rounded border border-slate-700", children: [_jsx("h3", { className: "text-sm font-semibold text-slate-100 mb-3", children: "Detailed Variant (With Labels)" }), _jsx(ProgressBars, { currentStep: currentStep, totalSteps: 5, elapsedTimeMs: currentStep * 2000, timeBudgetMs: 10000, tokensUsed: currentStep * 500, tokenBudget: 2500, variant: "detailed", size: "lg", showValues: true })] }));
};
/**
 * Example 4: Size Variants
 * Shows all three size options (sm, md, lg)
 */
export const SizeVariantsExample = () => {
    return (_jsxs("div", { className: "space-y-4", children: [_jsxs("div", { className: "p-4 bg-slate-900 rounded border border-slate-700", children: [_jsx("h3", { className: "text-sm font-semibold text-slate-100 mb-3", children: "Small Size" }), _jsx(ProgressBars, { currentStep: 4, totalSteps: 8, elapsedTimeMs: 4000, timeBudgetMs: 8000, tokensUsed: 1200, tokenBudget: 2000, variant: "stacked", size: "sm" })] }), _jsxs("div", { className: "p-4 bg-slate-900 rounded border border-slate-700", children: [_jsx("h3", { className: "text-sm font-semibold text-slate-100 mb-3", children: "Medium Size (Default)" }), _jsx(ProgressBars, { currentStep: 4, totalSteps: 8, elapsedTimeMs: 4000, timeBudgetMs: 8000, tokensUsed: 1200, tokenBudget: 2000, variant: "stacked", size: "md" })] }), _jsxs("div", { className: "p-4 bg-slate-900 rounded border border-slate-700", children: [_jsx("h3", { className: "text-sm font-semibold text-slate-100 mb-3", children: "Large Size" }), _jsx(ProgressBars, { currentStep: 4, totalSteps: 8, elapsedTimeMs: 4000, timeBudgetMs: 8000, tokensUsed: 1200, tokenBudget: 2000, variant: "stacked", size: "lg" })] })] }));
};
/**
 * Example 5: Over-Budget States
 * Demonstrates what happens when values exceed budgets
 */
export const OverBudgetExample = () => {
    return (_jsxs("div", { className: "space-y-4", children: [_jsxs("div", { className: "p-4 bg-slate-900 rounded border border-slate-700", children: [_jsx("h3", { className: "text-sm font-semibold text-slate-100 mb-3", children: "Time Over Budget" }), _jsx(ProgressBars, { currentStep: 7, totalSteps: 7, elapsedTimeMs: 12000, timeBudgetMs: 10000, tokensUsed: 1500, tokenBudget: 2000, variant: "detailed", size: "md", showValues: true })] }), _jsxs("div", { className: "p-4 bg-slate-900 rounded border border-slate-700", children: [_jsx("h3", { className: "text-sm font-semibold text-slate-100 mb-3", children: "Tokens Over Budget" }), _jsx(ProgressBars, { currentStep: 7, totalSteps: 7, elapsedTimeMs: 7000, timeBudgetMs: 10000, tokensUsed: 2500, tokenBudget: 2000, variant: "detailed", size: "md", showValues: true })] }), _jsxs("div", { className: "p-4 bg-slate-900 rounded border border-slate-700", children: [_jsx("h3", { className: "text-sm font-semibold text-slate-100 mb-3", children: "Both Over Budget" }), _jsx(ProgressBars, { currentStep: 7, totalSteps: 7, elapsedTimeMs: 12000, timeBudgetMs: 10000, tokensUsed: 2800, tokenBudget: 2000, variant: "detailed", size: "md", showValues: true })] })] }));
};
/**
 * Example 6: No Budgets
 * Shows behavior when time and token budgets are not set
 */
export const NoBudgetsExample = () => {
    const [currentStep, setCurrentStep] = useState(2);
    useEffect(() => {
        const interval = setInterval(() => {
            setCurrentStep((prev) => (prev < 6 ? prev + 1 : 0));
        }, 2000);
        return () => clearInterval(interval);
    }, []);
    return (_jsxs("div", { className: "space-y-4", children: [_jsxs("div", { className: "p-4 bg-slate-900 rounded border border-slate-700", children: [_jsx("h3", { className: "text-sm font-semibold text-slate-100 mb-3", children: "Only Step Progress (No Time/Token Budgets)" }), _jsx(ProgressBars, { currentStep: currentStep, totalSteps: 6, elapsedTimeMs: currentStep * 1000, tokensUsed: currentStep * 200, variant: "stacked", size: "md" })] }), _jsxs("div", { className: "p-4 bg-slate-900 rounded border border-slate-700", children: [_jsx("h3", { className: "text-sm font-semibold text-slate-100 mb-3", children: "Steps + Time (No Token Budget)" }), _jsx(ProgressBars, { currentStep: currentStep, totalSteps: 6, elapsedTimeMs: currentStep * 1000, timeBudgetMs: 6000, tokensUsed: currentStep * 200, variant: "detailed", size: "md" })] })] }));
};
/**
 * Example 7: Display Options
 * Shows percentages and values in labels
 */
export const DisplayOptionsExample = () => {
    return (_jsxs("div", { className: "space-y-4", children: [_jsxs("div", { className: "p-4 bg-slate-900 rounded border border-slate-700", children: [_jsx("h3", { className: "text-sm font-semibold text-slate-100 mb-3", children: "No Additional Labels (Minimal)" }), _jsx(ProgressBars, { currentStep: 4, totalSteps: 8, elapsedTimeMs: 4000, timeBudgetMs: 8000, tokensUsed: 1200, tokenBudget: 2000, variant: "stacked", size: "md", showPercentages: false, showValues: false })] }), _jsxs("div", { className: "p-4 bg-slate-900 rounded border border-slate-700", children: [_jsx("h3", { className: "text-sm font-semibold text-slate-100 mb-3", children: "With Percentages" }), _jsx(ProgressBars, { currentStep: 4, totalSteps: 8, elapsedTimeMs: 4000, timeBudgetMs: 8000, tokensUsed: 1200, tokenBudget: 2000, variant: "stacked", size: "md", showPercentages: true, showValues: false })] }), _jsxs("div", { className: "p-4 bg-slate-900 rounded border border-slate-700", children: [_jsx("h3", { className: "text-sm font-semibold text-slate-100 mb-3", children: "With Actual Values" }), _jsx(ProgressBars, { currentStep: 4, totalSteps: 8, elapsedTimeMs: 4000, timeBudgetMs: 8000, tokensUsed: 1200, tokenBudget: 2000, variant: "stacked", size: "md", showPercentages: false, showValues: true })] })] }));
};
/**
 * Complete Demo Component
 * Shows all examples together
 */
export const ProgressBarsDemo = () => {
    return (_jsxs("div", { className: "space-y-8 p-6 bg-slate-800 min-h-screen", children: [_jsxs("div", { className: "mb-6", children: [_jsx("h1", { className: "text-2xl font-bold text-slate-100 mb-2", children: "ProgressBars Component Demo" }), _jsx("p", { className: "text-slate-400 text-sm", children: "Multi-dimensional progress tracking showing step, time, and token progress simultaneously" })] }), _jsxs("section", { children: [_jsx("h2", { className: "text-lg font-semibold text-slate-100 mb-4", children: "Variants" }), _jsxs("div", { className: "space-y-4", children: [_jsx(StackedExample, {}), _jsx(InlineExample, {}), _jsx(DetailedExample, {})] })] }), _jsxs("section", { children: [_jsx("h2", { className: "text-lg font-semibold text-slate-100 mb-4", children: "Size Options" }), _jsx(SizeVariantsExample, {})] }), _jsxs("section", { children: [_jsx("h2", { className: "text-lg font-semibold text-slate-100 mb-4", children: "Budget States" }), _jsx(OverBudgetExample, {})] }), _jsxs("section", { children: [_jsx("h2", { className: "text-lg font-semibold text-slate-100 mb-4", children: "Optional Budgets" }), _jsx(NoBudgetsExample, {})] }), _jsxs("section", { children: [_jsx("h2", { className: "text-lg font-semibold text-slate-100 mb-4", children: "Display Options" }), _jsx(DisplayOptionsExample, {})] })] }));
};
export default ProgressBarsDemo;
//# sourceMappingURL=ProgressBars.examples.js.map