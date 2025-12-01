/**
 * ConfidenceLandscape Component
 *
 * Interactive 2D visualization of confidence grid as heatmap with contours.
 * Supports cell selection, tooltips, and multiple view modes.
 *
 * @module components/landscape/ConfidenceLandscape
 */

import React, {
  useRef,
  useEffect,
  useCallback,
  useState,
  useMemo,
} from 'react';
import {
  ConfidenceLandscapeRenderer,
  type LandscapeRenderConfig,
  type LandscapeRenderState,
} from './ConfidenceLandscapeRenderer';
import { useAnimationFrame } from '../../hooks/useAnimationFrame';
import type {
  ColorScale,
  GridCell,
  HighlightedCell,
  LandscapeViewMode,
} from '../../store/slices/landscapeSlice';

// ============================================================================
// Types
// ============================================================================

export interface ConfidenceLandscapeProps {
  /** 2D grid of confidence values */
  grid: number[][];
  /** Grid width */
  width: number;
  /** Grid height */
  height: number;
  /** X-axis labels */
  xLabels: string[];
  /** Y-axis labels */
  yLabels: string[];
  /** Minimum value */
  minValue?: number;
  /** Maximum value */
  maxValue?: number;
  /** Highlighted cells */
  highlightedCells?: HighlightedCell[];
  /** View mode */
  viewMode?: LandscapeViewMode;
  /** Color scale */
  colorScale?: ColorScale;
  /** Show contour lines */
  showContourLines?: boolean;
  /** Contour interval */
  contourInterval?: number;
  /** Show values in cells */
  showValues?: boolean;
  /** Dark mode */
  darkMode?: boolean;
  /** Whether to animate transitions */
  animated?: boolean;
  /** Callback when a cell is clicked */
  onCellClick?: (cell: GridCell) => void;
  /** Callback when a cell is hovered */
  onCellHover?: (cell: GridCell | null) => void;
  /** Additional CSS class */
  className?: string;
  /** Container width */
  containerWidth?: number;
  /** Container height */
  containerHeight?: number;
}

// ============================================================================
// Tooltip Component
// ============================================================================

interface TooltipProps {
  cell: GridCell;
  x: number;
  y: number;
  darkMode: boolean;
}

const Tooltip: React.FC<TooltipProps> = ({ cell, x, y, darkMode }) => {
  const bgColor = darkMode ? 'bg-gray-800' : 'bg-white';
  const textColor = darkMode ? 'text-gray-100' : 'text-gray-900';
  const borderColor = darkMode ? 'border-gray-700' : 'border-gray-200';

  // Get confidence level description
  const getConfidenceLevel = (value: number): { label: string; color: string } => {
    if (value >= 0.8) return { label: 'High', color: 'text-green-500' };
    if (value >= 0.6) return { label: 'Good', color: 'text-blue-500' };
    if (value >= 0.4) return { label: 'Moderate', color: 'text-yellow-500' };
    if (value >= 0.2) return { label: 'Low', color: 'text-orange-500' };
    return { label: 'Very Low', color: 'text-red-500' };
  };

  const level = getConfidenceLevel(cell.value);

  return (
    <div
      className={`absolute z-50 pointer-events-none ${bgColor} ${textColor} ${borderColor} border rounded-lg shadow-lg p-3 text-sm min-w-[180px]`}
      style={{
        left: x + 15,
        top: y + 15,
        transform: 'translateY(-50%)',
      }}
    >
      <div className="font-semibold mb-1">{cell.xLabel} × {cell.yLabel}</div>
      <div className="flex justify-between items-center">
        <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>Confidence:</span>
        <span className={`font-mono font-medium ${level.color}`}>
          {(cell.value * 100).toFixed(1)}%
        </span>
      </div>
      <div className="flex justify-between items-center mt-1">
        <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>Level:</span>
        <span className={level.color}>{level.label}</span>
      </div>
      {cell.highlighted && (
        <div className={`mt-2 pt-2 border-t ${borderColor}`}>
          <span className="text-indigo-500">✓ Highlighted</span>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Legend Component
// ============================================================================

interface LegendProps {
  viewMode: LandscapeViewMode;
  showContourLines: boolean;
  onViewModeChange?: (mode: LandscapeViewMode) => void;
  onToggleContours?: () => void;
  darkMode: boolean;
}

const ViewControls: React.FC<LegendProps> = ({
  viewMode,
  showContourLines,
  onViewModeChange,
  onToggleContours,
  darkMode,
}) => {
  const bgColor = darkMode ? 'bg-gray-800/80' : 'bg-white/80';
  const textColor = darkMode ? 'text-gray-300' : 'text-gray-600';
  const buttonBg = darkMode ? 'bg-gray-700' : 'bg-gray-100';
  const buttonActiveBg = darkMode ? 'bg-indigo-600' : 'bg-indigo-500';

  return (
    <div className={`absolute top-2 right-2 ${bgColor} backdrop-blur rounded-lg p-2 ${textColor} text-xs flex gap-2`}>
      {/* View mode buttons */}
      <div className="flex gap-1">
        <button
          onClick={() => onViewModeChange?.('heatmap')}
          className={`px-2 py-1 rounded ${
            viewMode === 'heatmap'
              ? `${buttonActiveBg} text-white`
              : buttonBg
          }`}
          title="Heatmap view"
        >
          ▦
        </button>
        <button
          onClick={() => onViewModeChange?.('contour')}
          className={`px-2 py-1 rounded ${
            viewMode === 'contour'
              ? `${buttonActiveBg} text-white`
              : buttonBg
          }`}
          title="Contour view"
        >
          ◎
        </button>
        <button
          onClick={() => onViewModeChange?.('table')}
          className={`px-2 py-1 rounded ${
            viewMode === 'table'
              ? `${buttonActiveBg} text-white`
              : buttonBg
          }`}
          title="Table view"
        >
          ⊞
        </button>
      </div>

      {/* Contour toggle */}
      {viewMode !== 'table' && (
        <button
          onClick={onToggleContours}
          className={`px-2 py-1 rounded ${
            showContourLines
              ? `${buttonActiveBg} text-white`
              : buttonBg
          }`}
          title="Toggle contour lines"
        >
          ≋
        </button>
      )}
    </div>
  );
};

// ============================================================================
// Statistics Component
// ============================================================================

interface StatisticsProps {
  grid: number[][];
  darkMode: boolean;
}

const Statistics: React.FC<StatisticsProps> = ({ grid, darkMode }) => {
  const stats = useMemo(() => {
    const values: number[] = [];
    for (const row of grid) {
      for (const val of row) {
        if (val !== undefined && !isNaN(val)) {
          values.push(val);
        }
      }
    }

    if (values.length === 0) {
      return { mean: 0, min: 0, max: 0, total: 0 };
    }

    const sum = values.reduce((a, b) => a + b, 0);
    const mean = sum / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);

    return { mean, min, max, total: values.length };
  }, [grid]);

  const bgColor = darkMode ? 'bg-gray-800/80' : 'bg-white/80';
  const textColor = darkMode ? 'text-gray-300' : 'text-gray-600';

  return (
    <div className={`absolute bottom-2 left-2 ${bgColor} backdrop-blur rounded-lg p-2 ${textColor} text-xs`}>
      <div className="flex gap-4">
        <div>
          <span className="opacity-60">Avg:</span>{' '}
          <span className="font-mono">{(stats.mean * 100).toFixed(1)}%</span>
        </div>
        <div>
          <span className="opacity-60">Range:</span>{' '}
          <span className="font-mono">
            {(stats.min * 100).toFixed(0)}-{(stats.max * 100).toFixed(0)}%
          </span>
        </div>
        <div>
          <span className="opacity-60">Cells:</span>{' '}
          <span className="font-mono">{stats.total}</span>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const ConfidenceLandscape: React.FC<ConfidenceLandscapeProps> = ({
  grid,
  width,
  height,
  xLabels,
  yLabels,
  minValue = 0,
  maxValue = 1,
  highlightedCells = [],
  viewMode = 'heatmap',
  colorScale = 'viridis',
  showContourLines = true,
  contourInterval = 0.1,
  showValues = true,
  darkMode = false,
  animated = true,
  onCellClick,
  onCellHover,
  className = '',
  containerWidth = 600,
  containerHeight = 400,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<ConfidenceLandscapeRenderer | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Tooltip state
  const [tooltipCell, setTooltipCell] = useState<GridCell | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });

  // Selected/hovered cell state
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number } | null>(null);

  // Animation state
  const animationProgressRef = useRef(1);
  const previousGridRef = useRef<number[][] | null>(null);

  // Local view state (can be controlled externally if needed)
  const [localViewMode, setLocalViewMode] = useState(viewMode);
  const [localShowContours, setLocalShowContours] = useState(showContourLines);

  // Check for reduced motion preference
  const prefersReducedMotion = useMemo(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }, []);

  const shouldAnimate = animated && !prefersReducedMotion;

  // Initialize renderer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const config: Partial<LandscapeRenderConfig> = {
      colorScale,
      showContourLines: localShowContours,
      contourInterval,
      showValues,
      darkMode,
    };

    rendererRef.current = new ConfidenceLandscapeRenderer(canvas, config);

    // Initial render
    const state: LandscapeRenderState = {
      grid,
      width,
      height,
      xLabels,
      yLabels,
      minValue,
      maxValue,
      highlightedCells,
      hoveredCell: null,
      selectedCell: null,
      viewMode: localViewMode,
      animationProgress: 1,
      previousGrid: null,
    };
    rendererRef.current.render(state);

    return () => {
      rendererRef.current?.dispose();
      rendererRef.current = null;
    };
  }, []);

  // Update renderer config when props change
  useEffect(() => {
    rendererRef.current?.updateConfig({
      colorScale,
      showContourLines: localShowContours,
      contourInterval,
      showValues,
      darkMode,
    });
  }, [colorScale, localShowContours, contourInterval, showValues, darkMode]);

  // Handle grid changes with animation
  useEffect(() => {
    if (shouldAnimate && rendererRef.current) {
      previousGridRef.current = grid;
      animationProgressRef.current = 0;
    }
  }, [grid, shouldAnimate]);

  // Handle resize
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const resizeObserver = new ResizeObserver(() => {
      rendererRef.current?.resize();
    });

    resizeObserver.observe(container);
    return () => resizeObserver.disconnect();
  }, []);

  // Animation frame callback
  const onFrame = useCallback(
    (deltaTime: number) => {
      if (!rendererRef.current) return;

      // Update animation progress
      if (shouldAnimate && animationProgressRef.current < 1) {
        animationProgressRef.current = Math.min(
          1,
          animationProgressRef.current + deltaTime / 300
        );
      }

      const state: LandscapeRenderState = {
        grid,
        width,
        height,
        xLabels,
        yLabels,
        minValue,
        maxValue,
        highlightedCells,
        hoveredCell,
        selectedCell,
        viewMode: localViewMode,
        animationProgress: animationProgressRef.current,
        previousGrid: previousGridRef.current,
      };

      rendererRef.current.render(state);
    },
    [
      grid,
      width,
      height,
      xLabels,
      yLabels,
      minValue,
      maxValue,
      highlightedCells,
      hoveredCell,
      selectedCell,
      localViewMode,
      shouldAnimate,
    ]
  );

  // Use animation frame hook
  useAnimationFrame(onFrame, {
    enabled: shouldAnimate,
    priority: 1,
  });

  // Render once when not animating
  useEffect(() => {
    if (!shouldAnimate) {
      onFrame(0);
    }
  }, [shouldAnimate, onFrame]);

  // Mouse handlers
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || !rendererRef.current) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const state: LandscapeRenderState = {
        grid,
        width,
        height,
        xLabels,
        yLabels,
        minValue,
        maxValue,
        highlightedCells,
        hoveredCell,
        selectedCell,
        viewMode: localViewMode,
        animationProgress: 1,
        previousGrid: null,
      };

      const cell = rendererRef.current.getCellAtPoint(e.clientX, e.clientY, state);

      if (cell) {
        setHoveredCell(cell);
        const value = grid[cell.row]?.[cell.col] ?? 0;
        const isHighlighted = highlightedCells.some(
          (h) => h.row === cell.row && h.col === cell.col
        );

        const gridCell: GridCell = {
          row: cell.row,
          col: cell.col,
          value,
          xLabel: xLabels[cell.col] ?? `Col ${cell.col}`,
          yLabel: yLabels[cell.row] ?? `Row ${cell.row}`,
          highlighted: isHighlighted,
        };

        setTooltipCell(gridCell);
        setTooltipPosition({ x, y });
        onCellHover?.(gridCell);
      } else {
        setHoveredCell(null);
        setTooltipCell(null);
        onCellHover?.(null);
      }
    },
    [grid, width, height, xLabels, yLabels, highlightedCells, localViewMode, onCellHover]
  );

  const handleMouseLeave = useCallback(() => {
    setHoveredCell(null);
    setTooltipCell(null);
    onCellHover?.(null);
  }, [onCellHover]);

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || !rendererRef.current) return;

      const state: LandscapeRenderState = {
        grid,
        width,
        height,
        xLabels,
        yLabels,
        minValue,
        maxValue,
        highlightedCells,
        hoveredCell,
        selectedCell,
        viewMode: localViewMode,
        animationProgress: 1,
        previousGrid: null,
      };

      const cell = rendererRef.current.getCellAtPoint(e.clientX, e.clientY, state);

      if (cell) {
        setSelectedCell(cell);
        const value = grid[cell.row]?.[cell.col] ?? 0;
        const isHighlighted = highlightedCells.some(
          (h) => h.row === cell.row && h.col === cell.col
        );

        const gridCell: GridCell = {
          row: cell.row,
          col: cell.col,
          value,
          xLabel: xLabels[cell.col] ?? `Col ${cell.col}`,
          yLabel: yLabels[cell.row] ?? `Row ${cell.row}`,
          highlighted: isHighlighted,
        };

        onCellClick?.(gridCell);
      }
    },
    [grid, width, height, xLabels, yLabels, highlightedCells, localViewMode, onCellClick]
  );

  // Keyboard handlers for accessibility
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLCanvasElement>) => {
      if (!selectedCell) return;

      let newRow = selectedCell.row;
      let newCol = selectedCell.col;

      switch (e.key) {
        case 'ArrowUp':
          e.preventDefault();
          newRow = Math.max(0, selectedCell.row - 1);
          break;
        case 'ArrowDown':
          e.preventDefault();
          newRow = Math.min(height - 1, selectedCell.row + 1);
          break;
        case 'ArrowLeft':
          e.preventDefault();
          newCol = Math.max(0, selectedCell.col - 1);
          break;
        case 'ArrowRight':
          e.preventDefault();
          newCol = Math.min(width - 1, selectedCell.col + 1);
          break;
        case 'Enter':
        case ' ':
          e.preventDefault();
          if (selectedCell) {
            const value = grid[selectedCell.row]?.[selectedCell.col] ?? 0;
            const isHighlighted = highlightedCells.some(
              (h) => h.row === selectedCell.row && h.col === selectedCell.col
            );
            const gridCell: GridCell = {
              row: selectedCell.row,
              col: selectedCell.col,
              value,
              xLabel: xLabels[selectedCell.col] ?? `Col ${selectedCell.col}`,
              yLabel: yLabels[selectedCell.row] ?? `Row ${selectedCell.row}`,
              highlighted: isHighlighted,
            };
            onCellClick?.(gridCell);
          }
          return;
        default:
          return;
      }

      setSelectedCell({ row: newRow, col: newCol });
    },
    [selectedCell, width, height, grid, xLabels, yLabels, highlightedCells, onCellClick]
  );

  // ARIA description
  const ariaDescription = useMemo(() => {
    const highConfCount = grid.flat().filter((v) => v >= 0.8).length;
    const lowConfCount = grid.flat().filter((v) => v <= 0.3).length;
    return `Confidence landscape with ${width} columns and ${height} rows. ${highConfCount} high confidence cells, ${lowConfCount} low confidence cells.`;
  }, [grid, width, height]);

  return (
    <div
      ref={containerRef}
      className={`relative overflow-hidden ${className}`}
      style={{ width: containerWidth, height: containerHeight }}
    >
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-crosshair"
        tabIndex={0}
        role="img"
        aria-label="Confidence landscape visualization"
        aria-description={ariaDescription}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
      />

      {/* View controls */}
      <ViewControls
        viewMode={localViewMode}
        showContourLines={localShowContours}
        onViewModeChange={setLocalViewMode}
        onToggleContours={() => setLocalShowContours(!localShowContours)}
        darkMode={darkMode}
      />

      {/* Statistics */}
      <Statistics grid={grid} darkMode={darkMode} />

      {/* Tooltip */}
      {tooltipCell && (
        <Tooltip
          cell={tooltipCell}
          x={tooltipPosition.x}
          y={tooltipPosition.y}
          darkMode={darkMode}
        />
      )}

      {/* Screen reader summary */}
      <div className="sr-only" aria-live="polite">
        {selectedCell &&
          `Selected cell at row ${yLabels[selectedCell.row] ?? selectedCell.row}, column ${xLabels[selectedCell.col] ?? selectedCell.col}, confidence ${((grid[selectedCell.row]?.[selectedCell.col] ?? 0) * 100).toFixed(1)}%`}
      </div>

      {/* Controls hint */}
      <div
        className={`absolute bottom-2 right-2 text-xs ${
          darkMode ? 'text-gray-500' : 'text-gray-400'
        }`}
      >
        Click to select • Arrow keys to navigate
      </div>
    </div>
  );
};

export default ConfidenceLandscape;
