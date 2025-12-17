import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React from 'react';
import { Sidebar } from '../Sidebar/Sidebar';
import { Header } from '../Header/Header';
import { MainPanel } from '../MainPanel/MainPanel';
/**
 * Main Layout Component
 * Provides the overall structure with sidebar, header, and main panel
 * Uses CSS Grid for responsive layout
 *
 * Layout structure:
 * - Header: Full width at top
 * - Sidebar: Fixed 240px on left (collapses on mobile)
 * - MainPanel: Flexible content area
 */
export const Layout = () => {
    const [sidebarOpen, setSidebarOpen] = React.useState(true);
    return (_jsxs("div", { className: "flex flex-col h-screen bg-slate-950 text-slate-100", children: [_jsx(Header, { onToggleSidebar: () => setSidebarOpen(!sidebarOpen) }), _jsxs("div", { className: "flex flex-1 overflow-hidden", children: [_jsx("div", { className: `${sidebarOpen ? 'w-60' : 'w-0'} transition-all duration-300 ease-in-out overflow-hidden border-r border-slate-800 bg-slate-900`, children: _jsx(Sidebar, {}) }), _jsx("div", { className: "flex-1 overflow-auto", children: _jsx(MainPanel, {}) })] })] }));
};
export default Layout;
//# sourceMappingURL=Layout.js.map