import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React from 'react';
import { useAgentManagerStore } from '../../stores/agentManagerStore';
export const Header = ({ onToggleSidebar }) => {
    const { filter, setFilter, viewMode, setViewMode, isConnected } = useAgentManagerStore();
    const [showFilterMenu, setShowFilterMenu] = React.useState(false);
    const filterMenuRef = React.useRef(null);
    // Close filter menu on outside click
    React.useEffect(() => {
        const handleClickOutside = (e) => {
            if (filterMenuRef.current &&
                !filterMenuRef.current.contains(e.target)) {
                setShowFilterMenu(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);
    const filterOptions = [
        { value: 'all', label: 'All Threads' },
        { value: 'active', label: 'Active' },
        { value: 'completed', label: 'Completed' },
        { value: 'failed', label: 'Failed' },
    ];
    const viewOptions = [
        { value: 'outline', label: 'Outline', icon: '≡' },
        { value: 'tree', label: 'Tree', icon: '⊢' },
        { value: 'swarm', label: 'Swarm', icon: '◆' },
    ];
    return (_jsxs("header", { className: "bg-slate-900 border-b border-slate-800 px-6 py-3 flex items-center justify-between gap-6 sticky top-0 z-40", children: [_jsxs("div", { className: "flex items-center gap-3", children: [_jsx("button", { onClick: onToggleSidebar, className: "p-2 hover:bg-slate-800 rounded-md transition-colors text-slate-400 hover:text-slate-200", title: "Toggle sidebar", children: _jsx("svg", { className: "w-5 h-5", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: _jsx("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M4 6h16M4 12h16M4 18h16" }) }) }), _jsxs("button", { className: "px-3 py-1.5 bg-emerald-600 hover:bg-emerald-700 rounded-md text-white font-medium text-sm transition-colors flex items-center gap-2", children: [_jsx("span", { children: "+" }), _jsx("span", { children: "New Thread" })] })] }), _jsx("div", { className: "flex-1 flex justify-center", children: _jsxs("div", { className: "relative", ref: filterMenuRef, children: [_jsxs("button", { onClick: () => setShowFilterMenu(!showFilterMenu), className: "px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-md text-slate-100 text-sm font-medium transition-colors flex items-center gap-2", children: [_jsx("span", { children: "Filter:" }), _jsx("span", { className: "font-semibold", children: filterOptions.find((o) => o.value === filter)?.label }), _jsx("svg", { className: `w-4 h-4 transition-transform ${showFilterMenu ? 'rotate-180' : ''}`, fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: _jsx("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M19 14l-7 7m0 0l-7-7m7 7V3" }) })] }), showFilterMenu && (_jsx("div", { className: "absolute top-full left-0 mt-2 bg-slate-800 border border-slate-700 rounded-md shadow-lg overflow-hidden min-w-max", children: filterOptions.map((option) => (_jsx("button", { onClick: () => {
                                    setFilter(option.value);
                                    setShowFilterMenu(false);
                                }, className: `block w-full text-left px-4 py-2 text-sm transition-colors ${filter === option.value
                                    ? 'bg-emerald-600 text-white font-medium'
                                    : 'text-slate-300 hover:bg-slate-700'}`, children: option.label }, option.value))) }))] }) }), _jsxs("div", { className: "flex items-center gap-4", children: [_jsx("div", { className: "flex gap-1 bg-slate-800 rounded-md p-1", children: viewOptions.map((option) => (_jsx("button", { onClick: () => setViewMode(option.value), className: `px-2.5 py-1.5 rounded transition-colors font-mono text-xs font-semibold ${viewMode === option.value
                                ? 'bg-emerald-600 text-white'
                                : 'text-slate-400 hover:text-slate-200'}`, title: option.label, children: option.icon }, option.value))) }), _jsxs("div", { className: "flex items-center gap-2 px-3 py-1.5 bg-slate-800 rounded-md", children: [_jsx("div", { className: `w-2.5 h-2.5 rounded-full transition-colors ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`, title: isConnected ? 'Connected' : 'Disconnected' }), _jsx("span", { className: "text-xs text-slate-400 font-medium", children: isConnected ? 'Connected' : 'Offline' })] })] })] }));
};
export default Header;
//# sourceMappingURL=Header.js.map