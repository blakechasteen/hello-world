/**
 * StageTimeline Component
 *
 * Real-time visualization of the 9-stage HoloLoom weaving cycle.
 * Uses Canvas 2D rendering for high performance (<16ms per frame).
 *
 * Features:
 * - Horizontal timeline with stage nodes
 * - Color transitions: pending (gray) → active (blue pulse) → complete (green)
 * - Duration labels when complete
 * - Sparkline for token rate
 * - Keyboard navigation and screen reader support
 * - Respects prefers-reduced-motion
 *
 * @module components/timeline/StageTimeline
 */

import React, { useRef, useEffect, useCallback, useMemo, useState } from 'react';
import { clsx } from 'clsx';
import { StageTimelineRenderer, type StageTimelineConfig, type RenderState } from './StageTimelineRenderer';
import { useAnimationFrame } from '../../hooks/useAnimationFrame';
import {
  WEAVING_STAGES,
  STAGE_DISPLAY_NAMES,
  type StageTimingData,
} from '../../store/slices/timelineSlice';
import type { WeavingStage } from '../../types';

// ============================================================================
// Types
// ============================================================================

export interface StageTimelineProps {
  /** Timing data for each stage */
  stages: StageTimingData[];
  /** Currently active stage */
  activeStage: WeavingStage | null;
  /** Current token generation rate (tokens/second) */
  tokenRate?: number;
  /** Token rate history for sparkline */
  tokenRateHistory?: number[];
  /** Component width in pixels */
  width?: number;
  /** Component height in pixels */
  height?: number;
  /** Show duration labels */
  showDurations?: boolean;
  /** Show stage names */
  showNames?: boolean;
  /** Enable animations */
  animated?: boolean;
  /** Dark mode */
  darkMode?: boolean;
  /** Callback when stage is clicked */
  onStageClick?: (stage: WeavingStage) => void;
  /** Callback when stage is hovered */
  onStageHover?: (stage: WeavingStage | null) => void;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Component
// ============================================================================

export const StageTimeline: React.FC<StageTimelineProps> = ({
  stages,
  activeStage,
  tokenRate = 0,
  tokenRateHistory = [],
  width = 800,
  height = 120,
  showDurations = true,
  showNames = true,
  animated = true,
  darkMode = false,
  onStageClick,
  onStageHover,
  className,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<StageTimelineRenderer | null>(null);
  const [focusedStage, setFocusedStage] = useState<WeavingStage | null>(null);
  const [hoveredStage, setHoveredStage] = useState<WeavingStage | null>(null);

  // Check for reduced motion preference
  const prefersReducedMotion = useMemo(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  }, []);

  const shouldAnimate = animated && !prefersReducedMotion;

  // Initialize renderer
  useEffect(() => {
    if (!canvasRef.current) return;

    const config: Partial<StageTimelineConfig> = {
      width,
      height,
      showDurations,
      showNames,
      animate: shouldAnimate,
      darkMode,
      dpr: window.devicePixelRatio || 1,
    };

    rendererRef.current = new StageTimelineRenderer(canvasRef.current, config);

    return () => {
      rendererRef.current?.dispose();
      rendererRef.current = null;
    };
  }, []);

  // Update renderer config when props change
  useEffect(() => {
    rendererRef.current?.updateConfig({
      width,
      height,
      showDurations,
      showNames,
      animate: shouldAnimate,
      darkMode,
    });
  }, [width, height, showDurations, showNames, shouldAnimate, darkMode]);

  // Render callback for animation frame
  const renderCallback = useCallback((timestamp: number) => {
    if (!rendererRef.current) return;

    const state: RenderState = {
      stages,
      activeStage,
      tokenRate,
      tokenRateHistory,
      timestamp,
    };

    rendererRef.current.render(state);
  }, [stages, activeStage, tokenRate, tokenRateHistory]);

  // Use animation frame for smooth updates
  useAnimationFrame(renderCallback, {
    enabled: shouldAnimate,
    targetFps: 60,
    respectReducedMotion: true,
  });

  // Render once for non-animated mode
  useEffect(() => {
    if (!shouldAnimate) {
      renderCallback(performance.now());
    }
  }, [shouldAnimate, stages, activeStage, tokenRate, tokenRateHistory]);

  // Handle mouse move for hover detection
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!rendererRef.current) return;

    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const stage = rendererRef.current.getNodeAtPoint(x, y);
    setHoveredStage(stage);
    onStageHover?.(stage);

    // Update cursor
    if (canvasRef.current) {
      canvasRef.current.style.cursor = stage ? 'pointer' : 'default';
    }
  }, [onStageHover]);

  // Handle mouse leave
  const handleMouseLeave = useCallback(() => {
    setHoveredStage(null);
    onStageHover?.(null);
    if (canvasRef.current) {
      canvasRef.current.style.cursor = 'default';
    }
  }, [onStageHover]);

  // Handle click
  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!rendererRef.current || !onStageClick) return;

    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const stage = rendererRef.current.getNodeAtPoint(x, y);
    if (stage) {
      onStageClick(stage);
    }
  }, [onStageClick]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (!focusedStage) {
      // Focus first stage if none focused
      if (e.key === 'Tab' && !e.shiftKey) {
        setFocusedStage(WEAVING_STAGES[0]);
        e.preventDefault();
      }
      return;
    }

    const currentIndex = WEAVING_STAGES.indexOf(focusedStage);

    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        if (currentIndex < WEAVING_STAGES.length - 1) {
          setFocusedStage(WEAVING_STAGES[currentIndex + 1]);
        }
        e.preventDefault();
        break;

      case 'ArrowLeft':
      case 'ArrowUp':
        if (currentIndex > 0) {
          setFocusedStage(WEAVING_STAGES[currentIndex - 1]);
        }
        e.preventDefault();
        break;

      case 'Home':
        setFocusedStage(WEAVING_STAGES[0]);
        e.preventDefault();
        break;

      case 'End':
        setFocusedStage(WEAVING_STAGES[WEAVING_STAGES.length - 1]);
        e.preventDefault();
        break;

      case 'Enter':
      case ' ':
        if (onStageClick && focusedStage) {
          onStageClick(focusedStage);
          e.preventDefault();
        }
        break;

      case 'Escape':
        setFocusedStage(null);
        e.preventDefault();
        break;
    }
  }, [focusedStage, onStageClick]);

  // Calculate progress for screen readers
  const completedCount = stages.filter(s => s.status === 'complete').length;
  const progress = Math.round((completedCount / WEAVING_STAGES.length) * 100);

  // Get active stage name for screen reader
  const activeStageDisplay = activeStage ? STAGE_DISPLAY_NAMES[activeStage] : 'None';

  return (
    <div
      className={clsx('stage-timeline relative', className)}
      role="region"
      aria-label="HoloLoom Weaving Timeline"
    >
      {/* Main canvas */}
      <canvas
        ref={canvasRef}
        className="timeline-canvas"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        tabIndex={0}
        role="listbox"
        aria-label={`Weaving stages: ${completedCount} of ${WEAVING_STAGES.length} complete`}
        aria-activedescendant={focusedStage ? `stage-${focusedStage}` : undefined}
      />

      {/* Screen reader accessible list (hidden visually) */}
      <div className="sr-only" aria-live="polite">
        <ul>
          {stages.map((stage) => (
            <li
              key={stage.stage}
              id={`stage-${stage.stage}`}
              role="option"
              aria-selected={stage.stage === focusedStage}
            >
              {STAGE_DISPLAY_NAMES[stage.stage]}:
              {stage.status === 'active' && ' Currently processing'}
              {stage.status === 'complete' && ` Complete (${stage.duration?.toFixed(0) || 0}ms)`}
              {stage.status === 'pending' && ' Pending'}
              {stage.status === 'error' && ' Error'}
            </li>
          ))}
        </ul>
      </div>

      {/* Live region for stage announcements */}
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {activeStage && `Now processing: ${activeStageDisplay}`}
      </div>

      {/* Progress indicator for screen readers */}
      <div className="sr-only" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
        {progress}% complete
      </div>

      {/* Tooltip for hovered stage */}
      {hoveredStage && (
        <StageTooltip
          stage={hoveredStage}
          stageTiming={stages.find(s => s.stage === hoveredStage)}
          renderer={rendererRef.current}
          darkMode={darkMode}
        />
      )}
    </div>
  );
};

// ============================================================================
// Tooltip Component
// ============================================================================

interface StageTooltipProps {
  stage: WeavingStage;
  stageTiming?: StageTimingData;
  renderer: StageTimelineRenderer | null;
  darkMode?: boolean;
}

const StageTooltip: React.FC<StageTooltipProps> = ({
  stage,
  stageTiming,
  renderer,
  darkMode = false,
}) => {
  const position = renderer?.getNodePosition(stage);
  if (!position) return null;

  const name = STAGE_DISPLAY_NAMES[stage];
  const status = stageTiming?.status || 'pending';
  const duration = stageTiming?.duration;
  const avgDuration = stageTiming?.avgDuration;

  return (
    <div
      className={clsx(
        'absolute pointer-events-none z-10 px-2 py-1 rounded shadow-lg text-xs',
        'transform -translate-x-1/2 -translate-y-full',
        darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900 border border-gray-200'
      )}
      style={{
        left: position.x,
        top: position.y - 30,
      }}
      role="tooltip"
    >
      <div className="font-medium">{name}</div>
      <div className="text-gray-500 dark:text-gray-400">
        Status: {status.charAt(0).toUpperCase() + status.slice(1)}
      </div>
      {duration !== null && duration !== undefined && (
        <div className="text-gray-500 dark:text-gray-400">
          Duration: {duration.toFixed(1)}ms
          {avgDuration && ` (avg: ${avgDuration.toFixed(1)}ms)`}
        </div>
      )}
      {stageTiming?.metrics?.cacheHit && (
        <div className="text-green-500">Cache hit</div>
      )}
    </div>
  );
};

// ============================================================================
// Connected Component (uses store)
// ============================================================================

// This will be added when integrating with the main store

export default StageTimeline;
