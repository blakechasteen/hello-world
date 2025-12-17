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
export const Layout: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = React.useState(true);

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100">
      {/* Header */}
      <Header onToggleSidebar={() => setSidebarOpen(!sidebarOpen)} />

      {/* Main Content Area with Sidebar */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar - Responsive */}
        <div
          className={`${
            sidebarOpen ? 'w-60' : 'w-0'
          } transition-all duration-300 ease-in-out overflow-hidden border-r border-slate-800 bg-slate-900`}
        >
          <Sidebar />
        </div>

        {/* Main Panel - Flexible */}
        <div className="flex-1 overflow-auto">
          <MainPanel />
        </div>
      </div>
    </div>
  );
};

export default Layout;
