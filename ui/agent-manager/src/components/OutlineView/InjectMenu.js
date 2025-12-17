import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
/**
 * InjectMenu Component
 * Dropdown menu for MRF/MCTS injection controls
 * Allows users to inject optimization strategies at specific steps
 */
import { useCallback, useState, useRef, useEffect } from 'react';
export const InjectMenu = ({ threadId, stepId, mrfEligible, mctsEligible, onInjectMRF, onInjectMCTS, injected = null, appliedStrategy, size = 'md', className = '', }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [activeSubmenu, setActiveSubmenu] = useState(null);
    const [selectedBudget, setSelectedBudget] = useState(100);
    const [selectedExploration, setSelectedExploration] = useState(1.0);
    const menuRef = useRef(null);
    // MRF strategies available
    const mrfStrategies = [
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
        const handleClickOutside = (event) => {
            if (menuRef.current && !menuRef.current.contains(event.target)) {
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
    const handleSelectMRFStrategy = useCallback((strategyId) => {
        onInjectMRF?.(strategyId);
        setIsOpen(false);
        setActiveSubmenu(null);
    }, [onInjectMRF]);
    // Handle MCTS injection
    const handleSelectMCTSConfig = useCallback((budget, exploration) => {
        onInjectMCTS?.({ budget, exploration });
        setIsOpen(false);
        setActiveSubmenu(null);
    }, [onInjectMCTS]);
    // Check if any injection is available
    const anyEligible = mrfEligible || mctsEligible;
    if (!anyEligible) {
        return (_jsxs("div", { className: `relative group ${className}`, title: "Injection not available for this step", children: [_jsx("button", { disabled: true, className: `
            ${buttonSizeClasses[size]}
            flex items-center justify-center rounded
            transition-all duration-150 ease-out
            bg-slate-700 text-slate-600 cursor-not-allowed opacity-50
            font-bold
          `, "aria-disabled": "true", children: "\u26A1" }), _jsx("div", { className: `
            absolute bottom-full left-1/2 -translate-x-1/2 mb-2
            bg-slate-900 text-white rounded px-2 py-1 text-xs
            opacity-0 group-hover:opacity-100 transition-opacity
            pointer-events-none whitespace-nowrap
            border border-slate-700
          `, children: "Not available" })] }));
    }
    return (_jsxs("div", { ref: menuRef, className: `relative ${className}`, children: [_jsxs("div", { className: "relative group", children: [_jsxs("button", { onClick: () => setIsOpen(!isOpen), className: `
            ${buttonSizeClasses[size]}
            flex items-center justify-center rounded
            transition-all duration-150 ease-out
            ${injected
                            ? 'bg-purple-700 text-purple-200 hover:bg-purple-600 hover:text-white'
                            : 'bg-slate-700 text-slate-200 hover:bg-slate-600 hover:text-white'}
            active:scale-95
            focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-1 focus:ring-offset-slate-800
            font-bold
          `, title: injected ? `Injected: ${injected}` : 'Inject optimization', "aria-label": "Inject MRF or MCTS", "aria-expanded": isOpen, "aria-haspopup": "true", children: ["\u26A1", injected && (_jsx("span", { className: "absolute -top-1 -right-1 w-2 h-2 bg-green-400 rounded-full", "aria-label": `${injected} injected` }))] }), _jsx("div", { className: `
            absolute bottom-full left-1/2 -translate-x-1/2 mb-2
            bg-slate-900 text-white rounded px-2 py-1 text-xs
            opacity-0 group-hover:opacity-100 transition-opacity
            pointer-events-none whitespace-nowrap
            border border-slate-700
          `, children: injected ? `${injected.toUpperCase()} injected` : 'Inject strategy' })] }), isOpen && (_jsxs("div", { className: `
            absolute top-full mt-1 right-0
            bg-slate-800 border border-slate-700 rounded-lg
            shadow-xl z-50
            min-w-max
            overflow-hidden
          `, role: "menu", "aria-orientation": "vertical", children: [mrfEligible && (_jsxs(_Fragment, { children: [_jsx("div", { className: "border-b border-slate-700 px-3 py-2", children: _jsx("div", { className: "text-xs font-semibold text-slate-300 uppercase tracking-wide", children: "MRF Refinement" }) }), mrfStrategies.map((strategy) => (_jsxs("button", { onClick: () => handleSelectMRFStrategy(strategy.id), className: `
                    w-full px-3 py-2 text-left
                    transition-colors duration-100
                    ${injected === 'mrf' && appliedStrategy === strategy.id
                                    ? 'bg-purple-700 text-purple-200'
                                    : 'text-slate-200 hover:bg-slate-700 hover:text-white'}
                    text-sm
                    flex items-center justify-between
                  `, role: "menuitem", children: [_jsxs("div", { children: [_jsx("div", { className: "font-medium", children: strategy.label }), _jsx("div", { className: "text-xs text-slate-400", children: strategy.description })] }), injected === 'mrf' && appliedStrategy === strategy.id && (_jsx("span", { className: "text-green-400 font-bold", children: "\u2713" }))] }, strategy.id)))] })), mctsEligible && (_jsxs(_Fragment, { children: [_jsx("div", { className: `
                  border-t border-slate-700 px-3 py-2
                  ${mrfEligible ? 'border-t' : ''}
                `, children: _jsxs("button", { onClick: () => setActiveSubmenu(activeSubmenu === 'mcts' ? null : 'mcts'), className: "w-full text-left text-xs font-semibold text-slate-300 uppercase tracking-wide hover:text-slate-200 transition-colors", "aria-haspopup": "true", "aria-expanded": activeSubmenu === 'mcts', children: ["MCTS Planning ", activeSubmenu === 'mcts' ? '▼' : '▶'] }) }), activeSubmenu === 'mcts' && (_jsxs("div", { className: "border-t border-slate-700 p-3 space-y-3 bg-slate-900/50", children: [_jsxs("div", { children: [_jsx("label", { className: "text-xs font-medium text-slate-300 block mb-1", children: "Iterations (Budget)" }), _jsx("div", { className: "flex flex-wrap gap-1", children: budgetOptions.map((budget) => (_jsx("button", { onClick: () => setSelectedBudget(budget), className: `
                            px-2 py-1 text-xs rounded
                            transition-colors duration-100
                            ${selectedBudget === budget
                                                        ? 'bg-blue-600 text-white'
                                                        : 'bg-slate-700 text-slate-300 hover:text-slate-200'}
                          `, children: budget }, budget))) })] }), _jsxs("div", { children: [_jsx("label", { className: "text-xs font-medium text-slate-300 block mb-1", children: "Exploration (c)" }), _jsx("div", { className: "flex flex-wrap gap-1", children: explorationOptions.map((exp) => (_jsx("button", { onClick: () => setSelectedExploration(exp), className: `
                            px-2 py-1 text-xs rounded
                            transition-colors duration-100
                            ${selectedExploration === exp
                                                        ? 'bg-blue-600 text-white'
                                                        : 'bg-slate-700 text-slate-300 hover:text-slate-200'}
                          `, children: exp.toFixed(1) }, exp))) })] }), _jsx("button", { onClick: () => handleSelectMCTSConfig(selectedBudget, selectedExploration), className: `
                      w-full px-3 py-1.5 rounded
                      bg-blue-700 text-blue-200 hover:bg-blue-600 hover:text-white
                      transition-colors duration-100
                      text-sm font-medium
                      focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 focus:ring-offset-slate-800
                    `, children: "Apply MCTS" }), injected === 'mcts' && (_jsx("div", { className: "text-xs text-green-400 text-center font-medium", children: "\u2713 Applied" }))] }))] }))] }))] }));
};
export default InjectMenu;
//# sourceMappingURL=InjectMenu.js.map