import React from 'react';
import { useAgentManagerStore } from '../../stores/agentManagerStore';

/**
 * Header Component
 * Top navigation bar with:
 * - New Thread button (left)
 * - Filter dropdown (center)
 * - View toggle buttons (right)
 * - Connection status indicator
 *
 * Inspired by GitHub dark theme (#0d1117 equivalent in slate-950)
 */
interface HeaderProps {
  onToggleSidebar?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onToggleSidebar }) => {
  const { filter, setFilter, viewMode, setViewMode, isConnected } =
    useAgentManagerStore();

  const [showFilterMenu, setShowFilterMenu] = React.useState(false);
  const filterMenuRef = React.useRef<HTMLDivElement>(null);

  // Close filter menu on outside click
  React.useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        filterMenuRef.current &&
        !filterMenuRef.current.contains(e.target as Node)
      ) {
        setShowFilterMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const filterOptions = [
    { value: 'all' as const, label: 'All Threads' },
    { value: 'active' as const, label: 'Active' },
    { value: 'completed' as const, label: 'Completed' },
    { value: 'failed' as const, label: 'Failed' },
  ];

  const viewOptions = [
    { value: 'outline' as const, label: 'Outline', icon: '≡' },
    { value: 'tree' as const, label: 'Tree', icon: '⊢' },
    { value: 'swarm' as const, label: 'Swarm', icon: '◆' },
  ];

  return (
    <header className="bg-slate-900 border-b border-slate-800 px-6 py-3 flex items-center justify-between gap-6 sticky top-0 z-40">
      {/* Left: Menu toggle + New Thread button */}
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="p-2 hover:bg-slate-800 rounded-md transition-colors text-slate-400 hover:text-slate-200"
          title="Toggle sidebar"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 6h16M4 12h16M4 18h16"
            />
          </svg>
        </button>

        <button className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-700 rounded-md text-white font-medium text-sm transition-colors flex items-center gap-2">
          <span>+</span>
          <span>New Thread</span>
        </button>
      </div>

      {/* Center: Filter Dropdown */}
      <div className="flex-1 flex justify-center">
        <div className="relative" ref={filterMenuRef}>
          <button
            onClick={() => setShowFilterMenu(!showFilterMenu)}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-md text-slate-100 text-sm font-medium transition-colors flex items-center gap-2"
          >
            <span>Filter:</span>
            <span className="font-semibold">
              {filterOptions.find((o) => o.value === filter)?.label}
            </span>
            <svg
              className={`w-4 h-4 transition-transform ${
                showFilterMenu ? 'rotate-180' : ''
              }`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 14l-7 7m0 0l-7-7m7 7V3"
              />
            </svg>
          </button>

          {/* Filter Menu Dropdown */}
          {showFilterMenu && (
            <div className="absolute top-full left-0 mt-2 bg-slate-800 border border-slate-700 rounded-md shadow-lg overflow-hidden min-w-max">
              {filterOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => {
                    setFilter(option.value);
                    setShowFilterMenu(false);
                  }}
                  className={`block w-full text-left px-4 py-2 text-sm transition-colors ${
                    filter === option.value
                      ? 'bg-emerald-600 text-white font-medium'
                      : 'text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Right: View toggle + Connection status */}
      <div className="flex items-center gap-4">
        {/* View Toggle Buttons */}
        <div className="flex gap-1 bg-slate-800 rounded-md p-1">
          {viewOptions.map((option) => (
            <button
              key={option.value}
              onClick={() => setViewMode(option.value)}
              className={`px-2.5 py-1.5 rounded transition-colors font-mono text-xs font-semibold ${
                viewMode === option.value
                  ? 'bg-emerald-600 text-white'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title={option.label}
            >
              {option.icon}
            </button>
          ))}
        </div>

        {/* Connection Status Indicator */}
        <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 rounded-md">
          <div
            className={`w-2.5 h-2.5 rounded-full transition-colors ${
              isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
            }`}
            title={isConnected ? 'Connected' : 'Disconnected'}
          />
          <span className="text-xs text-slate-400 font-medium">
            {isConnected ? 'Connected' : 'Offline'}
          </span>
        </div>
      </div>
    </header>
  );
};

export default Header;
