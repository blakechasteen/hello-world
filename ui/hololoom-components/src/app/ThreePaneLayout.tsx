/**
 * ThreePaneLayout - Unified Consciousness Shell
 *
 * Three-pane UI that visualizes HoloLoom's thinking process in real-time.
 * Layout: Memory Palace (Graph) | Active Thinking (Timeline) | Awareness (Metrics)
 *
 * Features:
 * - Resizable panes with drag handles
 * - Persist pane sizes to localStorage
 * - Collapse/expand panes on mobile
 * - Dark/light theme toggle
 *
 * Part of HoloLoom Phase 1 Frontend (November 2025)
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { clsx } from 'clsx';

// ============================================================================
// Types
// ============================================================================

export interface PaneConfig {
  id: string;
  title: string;
  minWidth: number;
  defaultWidth: number;
  collapsible?: boolean;
}

export interface ThreePaneLayoutProps {
  /** Left pane content (Memory Palace / Knowledge Graph) */
  leftPane?: React.ReactNode;
  /** Center pane content (Active Thinking / Timeline) */
  centerPane?: React.ReactNode;
  /** Right pane content (Awareness / Metrics) */
  rightPane?: React.ReactNode;
  /** Pane configuration */
  paneConfig?: {
    left: Partial<PaneConfig>;
    center: Partial<PaneConfig>;
    right: Partial<PaneConfig>;
  };
  /** Storage key for persisting sizes */
  storageKey?: string;
  /** Additional CSS classes */
  className?: string;
  /** Header content */
  header?: React.ReactNode;
  /** Footer content */
  footer?: React.ReactNode;
}

interface PaneSizes {
  left: number;
  right: number;
}

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_PANE_CONFIG: Record<'left' | 'center' | 'right', PaneConfig> = {
  left: { id: 'memory-palace', title: 'Memory Palace', minWidth: 200, defaultWidth: 300, collapsible: true },
  center: { id: 'active-thinking', title: 'Active Thinking', minWidth: 400, defaultWidth: 500, collapsible: false },
  right: { id: 'awareness', title: 'Awareness', minWidth: 200, defaultWidth: 280, collapsible: true },
};

const MOBILE_BREAKPOINT = 768;
const STORAGE_KEY_DEFAULT = 'hololoom-pane-sizes';
const DRAG_HANDLE_WIDTH = 8;

// ============================================================================
// Hook: Persisted Pane Sizes
// ============================================================================

function usePersistedSizes(storageKey: string, defaultSizes: PaneSizes): [PaneSizes, (sizes: PaneSizes) => void] {
  const [sizes, setSizes] = useState<PaneSizes>(() => {
    if (typeof window === 'undefined') return defaultSizes;
    try {
      const stored = localStorage.getItem(storageKey);
      return stored ? JSON.parse(stored) : defaultSizes;
    } catch {
      return defaultSizes;
    }
  });

  const persistSizes = useCallback((newSizes: PaneSizes) => {
    setSizes(newSizes);
    try {
      localStorage.setItem(storageKey, JSON.stringify(newSizes));
    } catch {
      // localStorage unavailable
    }
  }, [storageKey]);

  return [sizes, persistSizes];
}

// ============================================================================
// Hook: Window Size
// ============================================================================

function useWindowSize() {
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    function handleResize() {
      setSize({ width: window.innerWidth, height: window.innerHeight });
    }
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return size;
}

// ============================================================================
// Drag Handle Component
// ============================================================================

interface DragHandleProps {
  onDrag: (deltaX: number) => void;
  position: 'left' | 'right';
  disabled?: boolean;
}

const DragHandle: React.FC<DragHandleProps> = ({ onDrag, position, disabled }) => {
  const isDragging = useRef(false);
  const lastX = useRef(0);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (disabled) return;
    e.preventDefault();
    isDragging.current = true;
    lastX.current = e.clientX;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, [disabled]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      const deltaX = e.clientX - lastX.current;
      lastX.current = e.clientX;
      onDrag(deltaX);
    };

    const handleMouseUp = () => {
      isDragging.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [onDrag]);

  if (disabled) return null;

  return (
    <div
      className={clsx(
        'flex-shrink-0 cursor-col-resize group relative',
        'hover:bg-blue-500/20 active:bg-blue-500/30',
        'transition-colors duration-150'
      )}
      style={{ width: DRAG_HANDLE_WIDTH }}
      onMouseDown={handleMouseDown}
      role="separator"
      aria-orientation="vertical"
      aria-label={`Resize ${position} pane`}
    >
      {/* Visual indicator */}
      <div
        className={clsx(
          'absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2',
          'w-1 h-8 rounded-full',
          'bg-gray-300 dark:bg-gray-600',
          'group-hover:bg-blue-500 group-active:bg-blue-600',
          'transition-colors duration-150'
        )}
      />
    </div>
  );
};

// ============================================================================
// Pane Component
// ============================================================================

interface PaneProps {
  config: PaneConfig;
  width: number | 'flex';
  collapsed: boolean;
  onToggleCollapse?: () => void;
  children?: React.ReactNode;
  className?: string;
}

const Pane: React.FC<PaneProps> = ({
  config,
  width,
  collapsed,
  onToggleCollapse,
  children,
  className,
}) => {
  if (collapsed) {
    return (
      <div
        className={clsx(
          'flex flex-col items-center py-4',
          'bg-gray-50 dark:bg-gray-800',
          'border-r border-gray-200 dark:border-gray-700',
          className
        )}
        style={{ width: 40 }}
      >
        <button
          onClick={onToggleCollapse}
          className={clsx(
            'p-2 rounded-lg',
            'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200',
            'hover:bg-gray-200 dark:hover:bg-gray-700',
            'transition-colors duration-150'
          )}
          title={`Expand ${config.title}`}
          aria-label={`Expand ${config.title}`}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>
        <span
          className="writing-mode-vertical text-xs text-gray-500 dark:text-gray-400 mt-2"
          style={{ writingMode: 'vertical-rl', textOrientation: 'mixed' }}
        >
          {config.title}
        </span>
      </div>
    );
  }

  return (
    <div
      className={clsx(
        'flex flex-col overflow-hidden',
        'bg-white dark:bg-gray-900',
        className
      )}
      style={{ width: width === 'flex' ? undefined : width, flex: width === 'flex' ? 1 : undefined }}
    >
      {/* Pane Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200 dark:border-gray-700">
        <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          {config.title}
        </h2>
        {config.collapsible && onToggleCollapse && (
          <button
            onClick={onToggleCollapse}
            className={clsx(
              'p-1 rounded',
              'text-gray-400 hover:text-gray-600 dark:hover:text-gray-200',
              'hover:bg-gray-100 dark:hover:bg-gray-800',
              'transition-colors duration-150'
            )}
            title={`Collapse ${config.title}`}
            aria-label={`Collapse ${config.title}`}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Pane Content */}
      <div className="flex-1 overflow-auto">
        {children}
      </div>
    </div>
  );
};

// ============================================================================
// Mobile View Component
// ============================================================================

interface MobileViewProps {
  panes: { id: string; title: string; content: React.ReactNode }[];
  className?: string;
}

const MobileView: React.FC<MobileViewProps> = ({ panes, className }) => {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <div className={clsx('flex flex-col h-full', className)}>
      {/* Tab Bar */}
      <div className="flex border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
        {panes.map((pane, index) => (
          <button
            key={pane.id}
            onClick={() => setActiveTab(index)}
            className={clsx(
              'flex-1 px-4 py-3 text-sm font-medium',
              'transition-colors duration-150',
              activeTab === index
                ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400 bg-white dark:bg-gray-900'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
            )}
          >
            {pane.title}
          </button>
        ))}
      </div>

      {/* Active Pane Content */}
      <div className="flex-1 overflow-auto bg-white dark:bg-gray-900">
        {panes[activeTab]?.content}
      </div>
    </div>
  );
};

// ============================================================================
// Theme Toggle Component
// ============================================================================

interface ThemeToggleProps {
  className?: string;
}

const ThemeToggle: React.FC<ThemeToggleProps> = ({ className }) => {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    // Check initial theme
    const dark = document.documentElement.classList.contains('dark');
    setIsDark(dark);
  }, []);

  const toggleTheme = useCallback(() => {
    const newDark = !isDark;
    setIsDark(newDark);
    document.documentElement.classList.toggle('dark', newDark);
    try {
      localStorage.setItem('theme', newDark ? 'dark' : 'light');
    } catch {
      // localStorage unavailable
    }
  }, [isDark]);

  return (
    <button
      onClick={toggleTheme}
      className={clsx(
        'p-2 rounded-lg',
        'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200',
        'hover:bg-gray-100 dark:hover:bg-gray-800',
        'transition-colors duration-150',
        className
      )}
      title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      {isDark ? (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
        </svg>
      ) : (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
        </svg>
      )}
    </button>
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const ThreePaneLayout: React.FC<ThreePaneLayoutProps> = ({
  leftPane,
  centerPane,
  rightPane,
  paneConfig,
  storageKey = STORAGE_KEY_DEFAULT,
  className,
  header,
  footer,
}) => {
  // Merge pane configs
  const config = {
    left: { ...DEFAULT_PANE_CONFIG.left, ...paneConfig?.left },
    center: { ...DEFAULT_PANE_CONFIG.center, ...paneConfig?.center },
    right: { ...DEFAULT_PANE_CONFIG.right, ...paneConfig?.right },
  };

  // Persisted sizes
  const [sizes, setSizes] = usePersistedSizes(storageKey, {
    left: config.left.defaultWidth,
    right: config.right.defaultWidth,
  });

  // Collapsed state
  const [collapsed, setCollapsed] = useState({ left: false, right: false });

  // Window size for responsive
  const windowSize = useWindowSize();
  const isMobile = windowSize.width > 0 && windowSize.width < MOBILE_BREAKPOINT;

  // Container ref for bounds checking
  const containerRef = useRef<HTMLDivElement>(null);

  // Handle left drag
  const handleLeftDrag = useCallback((deltaX: number) => {
    setSizes(prev => {
      const newLeft = Math.max(config.left.minWidth, prev.left + deltaX);
      const containerWidth = containerRef.current?.clientWidth || window.innerWidth;
      const maxLeft = containerWidth - config.center.minWidth - prev.right - DRAG_HANDLE_WIDTH * 2;
      return { ...prev, left: Math.min(newLeft, maxLeft) };
    });
  }, [config.left.minWidth, config.center.minWidth, setSizes]);

  // Handle right drag
  const handleRightDrag = useCallback((deltaX: number) => {
    setSizes(prev => {
      const newRight = Math.max(config.right.minWidth, prev.right - deltaX);
      const containerWidth = containerRef.current?.clientWidth || window.innerWidth;
      const maxRight = containerWidth - config.center.minWidth - prev.left - DRAG_HANDLE_WIDTH * 2;
      return { ...prev, right: Math.min(newRight, maxRight) };
    });
  }, [config.right.minWidth, config.center.minWidth, setSizes]);

  // Toggle collapse
  const toggleLeftCollapse = useCallback(() => setCollapsed(prev => ({ ...prev, left: !prev.left })), []);
  const toggleRightCollapse = useCallback(() => setCollapsed(prev => ({ ...prev, right: !prev.right })), []);

  // Mobile view
  if (isMobile) {
    return (
      <div className={clsx('flex flex-col h-screen bg-gray-100 dark:bg-gray-950', className)}>
        {/* Header */}
        {header && (
          <div className="flex-shrink-0 border-b border-gray-200 dark:border-gray-700">
            {header}
          </div>
        )}

        {/* Mobile Tabs */}
        <MobileView
          panes={[
            { id: config.left.id, title: config.left.title, content: leftPane },
            { id: config.center.id, title: config.center.title, content: centerPane },
            { id: config.right.id, title: config.right.title, content: rightPane },
          ]}
          className="flex-1"
        />

        {/* Footer */}
        {footer && (
          <div className="flex-shrink-0 border-t border-gray-200 dark:border-gray-700">
            {footer}
          </div>
        )}
      </div>
    );
  }

  // Desktop view
  return (
    <div
      ref={containerRef}
      className={clsx('flex flex-col h-screen bg-gray-100 dark:bg-gray-950', className)}
    >
      {/* Header */}
      {header ? (
        <div className="flex-shrink-0 border-b border-gray-200 dark:border-gray-700">
          {header}
        </div>
      ) : (
        <div className="flex-shrink-0 flex items-center justify-between px-4 py-2 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700">
          <h1 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
            HoloLoom
          </h1>
          <ThemeToggle />
        </div>
      )}

      {/* Three-Pane Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Pane */}
        <Pane
          config={config.left}
          width={collapsed.left ? 40 : sizes.left}
          collapsed={collapsed.left}
          onToggleCollapse={config.left.collapsible ? toggleLeftCollapse : undefined}
          className="border-r border-gray-200 dark:border-gray-700"
        >
          {leftPane}
        </Pane>

        {/* Left Drag Handle */}
        <DragHandle onDrag={handleLeftDrag} position="left" disabled={collapsed.left} />

        {/* Center Pane */}
        <Pane
          config={config.center}
          width="flex"
          collapsed={false}
        >
          {centerPane}
        </Pane>

        {/* Right Drag Handle */}
        <DragHandle onDrag={handleRightDrag} position="right" disabled={collapsed.right} />

        {/* Right Pane */}
        <Pane
          config={config.right}
          width={collapsed.right ? 40 : sizes.right}
          collapsed={collapsed.right}
          onToggleCollapse={config.right.collapsible ? toggleRightCollapse : undefined}
          className="border-l border-gray-200 dark:border-gray-700"
        >
          {rightPane}
        </Pane>
      </div>

      {/* Footer */}
      {footer && (
        <div className="flex-shrink-0 border-t border-gray-200 dark:border-gray-700">
          {footer}
        </div>
      )}
    </div>
  );
};

ThreePaneLayout.displayName = 'ThreePaneLayout';

export default ThreePaneLayout;
