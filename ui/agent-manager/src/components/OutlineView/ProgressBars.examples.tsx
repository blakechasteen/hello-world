/**
 * ProgressBars Component Examples
 * Demonstrates all variants, sizes, and configuration options
 *
 * Usage:
 * - Copy examples into your component or documentation
 * - Use as reference for integrating ProgressBars into your UI
 */

import React, { useState, useEffect } from 'react';
import ProgressBars, { ProgressBarsProps } from './ProgressBars';

/**
 * Example 1: Stacked Variant (Default)
 * Vertical layout with three bars stacked on top of each other
 */
export const StackedExample: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(3);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStep((prev) => (prev < 7 ? prev + 1 : 0));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-4 bg-slate-900 rounded border border-slate-700">
      <h3 className="text-sm font-semibold text-slate-100 mb-3">
        Stacked Variant (Default)
      </h3>
      <ProgressBars
        currentStep={currentStep}
        totalSteps={7}
        elapsedTimeMs={currentStep * 1500}
        timeBudgetMs={10500}
        tokensUsed={currentStep * 250}
        tokenBudget={1750}
        variant="stacked"
        size="md"
      />
    </div>
  );
};

/**
 * Example 2: Inline Variant
 * Horizontal layout with three bars side by side
 */
export const InlineExample: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(5);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStep((prev) => (prev < 10 ? prev + 1 : 0));
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-4 bg-slate-900 rounded border border-slate-700">
      <h3 className="text-sm font-semibold text-slate-100 mb-3">
        Inline Variant (Side by Side)
      </h3>
      <ProgressBars
        currentStep={currentStep}
        totalSteps={10}
        elapsedTimeMs={currentStep * 1200}
        timeBudgetMs={12000}
        tokensUsed={currentStep * 350}
        tokenBudget={3500}
        variant="inline"
        size="sm"
      />
    </div>
  );
};

/**
 * Example 3: Detailed Variant
 * Stacked with labels, headers, and value display
 */
export const DetailedExample: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(2);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStep((prev) => (prev < 5 ? prev + 1 : 0));
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-4 bg-slate-900 rounded border border-slate-700">
      <h3 className="text-sm font-semibold text-slate-100 mb-3">
        Detailed Variant (With Labels)
      </h3>
      <ProgressBars
        currentStep={currentStep}
        totalSteps={5}
        elapsedTimeMs={currentStep * 2000}
        timeBudgetMs={10000}
        tokensUsed={currentStep * 500}
        tokenBudget={2500}
        variant="detailed"
        size="lg"
        showValues={true}
      />
    </div>
  );
};

/**
 * Example 4: Size Variants
 * Shows all three size options (sm, md, lg)
 */
export const SizeVariantsExample: React.FC = () => {
  return (
    <div className="space-y-4">
      <div className="p-4 bg-slate-900 rounded border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">
          Small Size
        </h3>
        <ProgressBars
          currentStep={4}
          totalSteps={8}
          elapsedTimeMs={4000}
          timeBudgetMs={8000}
          tokensUsed={1200}
          tokenBudget={2000}
          variant="stacked"
          size="sm"
        />
      </div>

      <div className="p-4 bg-slate-900 rounded border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">
          Medium Size (Default)
        </h3>
        <ProgressBars
          currentStep={4}
          totalSteps={8}
          elapsedTimeMs={4000}
          timeBudgetMs={8000}
          tokensUsed={1200}
          tokenBudget={2000}
          variant="stacked"
          size="md"
        />
      </div>

      <div className="p-4 bg-slate-900 rounded border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">
          Large Size
        </h3>
        <ProgressBars
          currentStep={4}
          totalSteps={8}
          elapsedTimeMs={4000}
          timeBudgetMs={8000}
          tokensUsed={1200}
          tokenBudget={2000}
          variant="stacked"
          size="lg"
        />
      </div>
    </div>
  );
};

/**
 * Example 5: Over-Budget States
 * Demonstrates what happens when values exceed budgets
 */
export const OverBudgetExample: React.FC = () => {
  return (
    <div className="space-y-4">
      <div className="p-4 bg-slate-900 rounded border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">
          Time Over Budget
        </h3>
        <ProgressBars
          currentStep={7}
          totalSteps={7}
          elapsedTimeMs={12000}
          timeBudgetMs={10000}
          tokensUsed={1500}
          tokenBudget={2000}
          variant="detailed"
          size="md"
          showValues={true}
        />
      </div>

      <div className="p-4 bg-slate-900 rounded border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">
          Tokens Over Budget
        </h3>
        <ProgressBars
          currentStep={7}
          totalSteps={7}
          elapsedTimeMs={7000}
          timeBudgetMs={10000}
          tokensUsed={2500}
          tokenBudget={2000}
          variant="detailed"
          size="md"
          showValues={true}
        />
      </div>

      <div className="p-4 bg-slate-900 rounded border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">
          Both Over Budget
        </h3>
        <ProgressBars
          currentStep={7}
          totalSteps={7}
          elapsedTimeMs={12000}
          timeBudgetMs={10000}
          tokensUsed={2800}
          tokenBudget={2000}
          variant="detailed"
          size="md"
          showValues={true}
        />
      </div>
    </div>
  );
};

/**
 * Example 6: No Budgets
 * Shows behavior when time and token budgets are not set
 */
export const NoBudgetsExample: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(2);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStep((prev) => (prev < 6 ? prev + 1 : 0));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-4">
      <div className="p-4 bg-slate-900 rounded border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">
          Only Step Progress (No Time/Token Budgets)
        </h3>
        <ProgressBars
          currentStep={currentStep}
          totalSteps={6}
          elapsedTimeMs={currentStep * 1000}
          tokensUsed={currentStep * 200}
          variant="stacked"
          size="md"
        />
      </div>

      <div className="p-4 bg-slate-900 rounded border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">
          Steps + Time (No Token Budget)
        </h3>
        <ProgressBars
          currentStep={currentStep}
          totalSteps={6}
          elapsedTimeMs={currentStep * 1000}
          timeBudgetMs={6000}
          tokensUsed={currentStep * 200}
          variant="detailed"
          size="md"
        />
      </div>
    </div>
  );
};

/**
 * Example 7: Display Options
 * Shows percentages and values in labels
 */
export const DisplayOptionsExample: React.FC = () => {
  return (
    <div className="space-y-4">
      <div className="p-4 bg-slate-900 rounded border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">
          No Additional Labels (Minimal)
        </h3>
        <ProgressBars
          currentStep={4}
          totalSteps={8}
          elapsedTimeMs={4000}
          timeBudgetMs={8000}
          tokensUsed={1200}
          tokenBudget={2000}
          variant="stacked"
          size="md"
          showPercentages={false}
          showValues={false}
        />
      </div>

      <div className="p-4 bg-slate-900 rounded border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">
          With Percentages
        </h3>
        <ProgressBars
          currentStep={4}
          totalSteps={8}
          elapsedTimeMs={4000}
          timeBudgetMs={8000}
          tokensUsed={1200}
          tokenBudget={2000}
          variant="stacked"
          size="md"
          showPercentages={true}
          showValues={false}
        />
      </div>

      <div className="p-4 bg-slate-900 rounded border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-100 mb-3">
          With Actual Values
        </h3>
        <ProgressBars
          currentStep={4}
          totalSteps={8}
          elapsedTimeMs={4000}
          timeBudgetMs={8000}
          tokensUsed={1200}
          tokenBudget={2000}
          variant="stacked"
          size="md"
          showPercentages={false}
          showValues={true}
        />
      </div>
    </div>
  );
};

/**
 * Complete Demo Component
 * Shows all examples together
 */
export const ProgressBarsDemo: React.FC = () => {
  return (
    <div className="space-y-8 p-6 bg-slate-800 min-h-screen">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-100 mb-2">
          ProgressBars Component Demo
        </h1>
        <p className="text-slate-400 text-sm">
          Multi-dimensional progress tracking showing step, time, and token
          progress simultaneously
        </p>
      </div>

      <section>
        <h2 className="text-lg font-semibold text-slate-100 mb-4">
          Variants
        </h2>
        <div className="space-y-4">
          <StackedExample />
          <InlineExample />
          <DetailedExample />
        </div>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-slate-100 mb-4">
          Size Options
        </h2>
        <SizeVariantsExample />
      </section>

      <section>
        <h2 className="text-lg font-semibold text-slate-100 mb-4">
          Budget States
        </h2>
        <OverBudgetExample />
      </section>

      <section>
        <h2 className="text-lg font-semibold text-slate-100 mb-4">
          Optional Budgets
        </h2>
        <NoBudgetsExample />
      </section>

      <section>
        <h2 className="text-lg font-semibold text-slate-100 mb-4">
          Display Options
        </h2>
        <DisplayOptionsExample />
      </section>
    </div>
  );
};

export default ProgressBarsDemo;
