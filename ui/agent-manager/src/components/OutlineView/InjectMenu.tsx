/**
 * InjectMenu Component
 * Dropdown menu for MRF/MCTS injection controls
 * Allows users to inject optimization strategies at specific steps
 */

import React, { useCallback, useState, useRef, useEffect } from 'react';

interface MRFStrategy {
  id: string;
  label: string;
  description: string;
}

interface MCTSConfig {
  budget: number;
  exploration: number;
}

interface InjectMenuProps {
  /** Thread ID for context */
  threadId: string;

  /** Step ID to inject at */
  stepId: string;

  /** Whether MRF injection is eligible for this step */
  mrfEligible: boolean;

  /** Whether MCTS injection is eligible for this step */
  mctsEligible: boolean;

  /** Callback when MRF strategy is selected */
  onInjectMRF?: (strategy: string) => void;

  /** Callback when MCTS config is selected */
  onInjectMCTS?: (config: MCTSConfig) => void;

  /** Already applied injection type */
  injected?: 'mrf' | 'mcts' | null;

  /** Already applied MRF strategy (if injected) */
  appliedStrategy?: string;

  /** Size of the button */
  size?: 'sm' | 'md';

  /** Custom CSS class */
  className?: string;
}

export const InjectMenu: React.FC<InjectMenuProps> = ({
  threadId,
  stepId,
  mrfEligible,
  mctsEligible,
  onInjectMRF,
  onInjectMCTS,
  injected = null,
  appliedStrategy,
  size = 'md',
  className = '',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeSubmenu, setActiveSubmenu] = useState<'mrf' | 'mcts' | null>(null);
  const [selectedBudget, setSelectedBudget] = useState<number>(100);
  const [selectedExploration, setSelectedExploration] = useState<number>(1.0);
  const menuRef = useRef<HTMLDivElement>(null);

  // MRF strategies available
  const mrfStrategies: MRFStrategy[] = [
    { id: 'auto', label: 'AUTO', description: 'Automatic strategy selection' },
    { id: 'verify', label: 'VERIFY', description: 'Verify accuracy' },
    { id: 'elegance', label: 'ELEGANCE', description: 'Improve clarity' },
    { id: 'critique', label: 'CRITIQUE', description: 'Critical analysis' },
    { id: 'refine', label: 'REFINE', description: 'Iterative refinement' },
    { id: 'hofstadter', label: 'HOFSTADTER', description: 'Recursive self-reference' },
  ];

  // MCTS budget options
  const budgetOptions = [50, 100, 200, 500];

  // MCTS exploration options
  const explorationOptions = [0.5, 1.0, 1.4, 2.0];

  // Compute button size classes
  const buttonSizeClasses = {
    sm: 'w-7 h-7 text-xs px-1',
    md: 'w-8 h-8 text-sm px-1',
  };

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setActiveSubmenu(null);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
    return undefined;
  }, [isOpen]);

  // Handle MRF strategy selection
  const handleSelectMRFStrategy = useCallback(
    (strategyId: string) => {
      onInjectMRF?.(strategyId);
      setIsOpen(false);
      setActiveSubmenu(null);
    },
    [onInjectMRF]
  );

  // Handle MCTS injection
  const handleSelectMCTSConfig = useCallback(
    (budget: number, exploration: number) => {
      onInjectMCTS?.({ budget, exploration });
      setIsOpen(false);
      setActiveSubmenu(null);
    },
    [onInjectMCTS]
  );

  // Check if any injection is available
  const anyEligible = mrfEligible || mctsEligible;

  if (!anyEligible) {
    return (
      <div
        className={`relative group ${className}`}
        title="Injection not available for this step"
      >
        <button
          disabled
          className={`
            ${buttonSizeClasses[size]}
            flex items-center justify-center rounded
            transition-all duration-150 ease-out
            bg-slate-700 text-slate-600 cursor-not-allowed opacity-50
            font-bold
          `}
          aria-disabled="true"
        >
          ⚡
        </button>
        <div
          className={`
            absolute bottom-full left-1/2 -translate-x-1/2 mb-2
            bg-slate-900 text-white rounded px-2 py-1 text-xs
            opacity-0 group-hover:opacity-100 transition-opacity
            pointer-events-none whitespace-nowrap
            border border-slate-700
          `}
        >
          Not available
        </div>
      </div>
    );
  }

  return (
    <div
      ref={menuRef}
      className={`relative ${className}`}
    >
      {/* Menu trigger button */}
      <div className="relative group">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className={`
            ${buttonSizeClasses[size]}
            flex items-center justify-center rounded
            transition-all duration-150 ease-out
            ${
              injected
                ? 'bg-purple-700 text-purple-200 hover:bg-purple-600 hover:text-white'
                : 'bg-slate-700 text-slate-200 hover:bg-slate-600 hover:text-white'
            }
            active:scale-95
            focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-1 focus:ring-offset-slate-800
            font-bold
          `}
          title={injected ? `Injected: ${injected}` : 'Inject optimization'}
          aria-label="Inject MRF or MCTS"
          aria-expanded={isOpen}
          aria-haspopup="true"
        >
          ⚡
          {injected && (
            <span
              className="absolute -top-1 -right-1 w-2 h-2 bg-green-400 rounded-full"
              aria-label={`${injected} injected`}
            />
          )}
        </button>

        {/* Tooltip */}
        <div
          className={`
            absolute bottom-full left-1/2 -translate-x-1/2 mb-2
            bg-slate-900 text-white rounded px-2 py-1 text-xs
            opacity-0 group-hover:opacity-100 transition-opacity
            pointer-events-none whitespace-nowrap
            border border-slate-700
          `}
        >
          {injected ? `${injected.toUpperCase()} injected` : 'Inject strategy'}
        </div>
      </div>

      {/* Dropdown menu */}
      {isOpen && (
        <div
          className={`
            absolute top-full mt-1 right-0
            bg-slate-800 border border-slate-700 rounded-lg
            shadow-xl z-50
            min-w-max
            overflow-hidden
          `}
          role="menu"
          aria-orientation="vertical"
        >
          {/* MRF Section */}
          {mrfEligible && (
            <>
              <div className="border-b border-slate-700 px-3 py-2">
                <div className="text-xs font-semibold text-slate-300 uppercase tracking-wide">
                  MRF Refinement
                </div>
              </div>

              {mrfStrategies.map((strategy) => (
                <button
                  key={strategy.id}
                  onClick={() => handleSelectMRFStrategy(strategy.id)}
                  className={`
                    w-full px-3 py-2 text-left
                    transition-colors duration-100
                    ${
                      injected === 'mrf' && appliedStrategy === strategy.id
                        ? 'bg-purple-700 text-purple-200'
                        : 'text-slate-200 hover:bg-slate-700 hover:text-white'
                    }
                    text-sm
                    flex items-center justify-between
                  `}
                  role="menuitem"
                >
                  <div>
                    <div className="font-medium">{strategy.label}</div>
                    <div className="text-xs text-slate-400">{strategy.description}</div>
                  </div>
                  {injected === 'mrf' && appliedStrategy === strategy.id && (
                    <span className="text-green-400 font-bold">✓</span>
                  )}
                </button>
              ))}
            </>
          )}

          {/* MCTS Section */}
          {mctsEligible && (
            <>
              <div
                className={`
                  border-t border-slate-700 px-3 py-2
                  ${mrfEligible ? 'border-t' : ''}
                `}
              >
                <button
                  onClick={() => setActiveSubmenu(activeSubmenu === 'mcts' ? null : 'mcts')}
                  className="w-full text-left text-xs font-semibold text-slate-300 uppercase tracking-wide hover:text-slate-200 transition-colors"
                  aria-haspopup="true"
                  aria-expanded={activeSubmenu === 'mcts'}
                >
                  MCTS Planning {activeSubmenu === 'mcts' ? '▼' : '▶'}
                </button>
              </div>

              {/* MCTS Configuration */}
              {activeSubmenu === 'mcts' && (
                <div className="border-t border-slate-700 p-3 space-y-3 bg-slate-900/50">
                  {/* Budget selection */}
                  <div>
                    <label className="text-xs font-medium text-slate-300 block mb-1">
                      Iterations (Budget)
                    </label>
                    <div className="flex flex-wrap gap-1">
                      {budgetOptions.map((budget) => (
                        <button
                          key={budget}
                          onClick={() => setSelectedBudget(budget)}
                          className={`
                            px-2 py-1 text-xs rounded
                            transition-colors duration-100
                            ${
                              selectedBudget === budget
                                ? 'bg-blue-600 text-white'
                                : 'bg-slate-700 text-slate-300 hover:text-slate-200'
                            }
                          `}
                        >
                          {budget}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Exploration selection */}
                  <div>
                    <label className="text-xs font-medium text-slate-300 block mb-1">
                      Exploration (c)
                    </label>
                    <div className="flex flex-wrap gap-1">
                      {explorationOptions.map((exp) => (
                        <button
                          key={exp}
                          onClick={() => setSelectedExploration(exp)}
                          className={`
                            px-2 py-1 text-xs rounded
                            transition-colors duration-100
                            ${
                              selectedExploration === exp
                                ? 'bg-blue-600 text-white'
                                : 'bg-slate-700 text-slate-300 hover:text-slate-200'
                            }
                          `}
                        >
                          {exp.toFixed(1)}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Apply button */}
                  <button
                    onClick={() => handleSelectMCTSConfig(selectedBudget, selectedExploration)}
                    className={`
                      w-full px-3 py-1.5 rounded
                      bg-blue-700 text-blue-200 hover:bg-blue-600 hover:text-white
                      transition-colors duration-100
                      text-sm font-medium
                      focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 focus:ring-offset-slate-800
                    `}
                  >
                    Apply MCTS
                  </button>

                  {injected === 'mcts' && (
                    <div className="text-xs text-green-400 text-center font-medium">
                      ✓ Applied
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default InjectMenu;
