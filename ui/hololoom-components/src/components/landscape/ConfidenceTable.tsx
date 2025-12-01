/**
 * ConfidenceTable Component
 *
 * Accessible table-based fallback for the Confidence Landscape visualization.
 * Provides full keyboard navigation and screen reader support.
 *
 * @module components/landscape/ConfidenceTable
 */

import React, { useCallback, useMemo, useState, useRef, useEffect } from 'react';
import type { GridCell, HighlightedCell, HighlightReason } from '../../store/slices/landscapeSlice';

// ============================================================================
// Types
// ============================================================================

export interface ConfidenceTableProps {
  /** 2D grid of confidence values */
  grid: number[][];
  /** Grid width (columns) */
  width: number;
  /** Grid height (rows) */
  height: number;
  /** X-axis labels (column headers) */
  xLabels: string[];
  /** Y-axis labels (row headers) */
  yLabels: string[];
  /** Minimum value for color scaling */
  minValue?: number;
  /** Maximum value for color scaling */
  maxValue?: number;
  /** Highlighted cells */
  highlightedCells?: HighlightedCell[];
  /** Show percentage values */
  showPercentage?: boolean;
  /** Dark mode */
  darkMode?: boolean;
  /** Compact mode (smaller cells) */
  compact?: boolean;
  /** Callback when a cell is clicked */
  onCellClick?: (cell: GridCell) => void;
  /** Callback when a cell is focused */
  onCellFocus?: (cell: GridCell | null) => void;
  /** Additional CSS class */
  className?: string;
  /** Caption for the table (accessibility) */
  caption?: string;
  /** Show row/column indices */
  showIndices?: boolean;
  /** Enable sorting by column */
  sortable?: boolean;
}

type SortDirection = 'asc' | 'desc' | null;

interface SortState {
  column: number | null;
  direction: SortDirection;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get background color for a confidence value
 */
function getConfidenceColor(
  value: number,
  min: number,
  max: number,
  darkMode: boolean
): string {
  const t = max > min ? (value - min) / (max - min) : 0.5;
  const clampedT = Math.max(0, Math.min(1, t));

  if (darkMode) {
    // Dark mode: Low = dark red, High = dark green
    if (clampedT < 0.3) return `rgba(220, 38, 38, ${0.3 + clampedT})`;
    if (clampedT < 0.5) return `rgba(234, 179, 8, ${0.3 + clampedT * 0.5})`;
    if (clampedT < 0.7) return `rgba(34, 197, 94, ${0.2 + clampedT * 0.3})`;
    return `rgba(34, 197, 94, ${0.4 + clampedT * 0.4})`;
  } else {
    // Light mode: Low = light red, High = light green
    if (clampedT < 0.3) return `rgba(239, 68, 68, ${0.2 + clampedT * 0.5})`;
    if (clampedT < 0.5) return `rgba(234, 179, 8, ${0.2 + clampedT * 0.4})`;
    if (clampedT < 0.7) return `rgba(34, 197, 94, ${0.1 + clampedT * 0.3})`;
    return `rgba(34, 197, 94, ${0.2 + clampedT * 0.5})`;
  }
}

/**
 * Get text color that contrasts with background
 */
function getTextColor(value: number, darkMode: boolean): string {
  if (darkMode) {
    return value < 0.5 ? '#f9fafb' : '#111827';
  }
  return value > 0.6 ? '#111827' : '#1f2937';
}

/**
 * Get highlight border color
 */
function getHighlightBorderColor(reason?: HighlightReason): string {
  switch (reason) {
    case 'retrieval_path':
      return '#ef4444';
    case 'high_confidence':
      return '#22c55e';
    case 'low_confidence':
      return '#f59e0b';
    case 'anomaly':
      return '#ec4899';
    case 'related':
      return '#8b5cf6';
    case 'selected':
    default:
      return '#3b82f6';
  }
}

/**
 * Get confidence level description for screen readers
 */
function getConfidenceLevel(value: number): string {
  if (value >= 0.8) return 'very high';
  if (value >= 0.6) return 'high';
  if (value >= 0.4) return 'moderate';
  if (value >= 0.2) return 'low';
  return 'very low';
}

// ============================================================================
// Main Component
// ============================================================================

export const ConfidenceTable: React.FC<ConfidenceTableProps> = ({
  grid,
  width,
  height,
  xLabels,
  yLabels,
  minValue = 0,
  maxValue = 1,
  highlightedCells = [],
  showPercentage = true,
  darkMode = false,
  compact = false,
  onCellClick,
  onCellFocus,
  className = '',
  caption = 'Confidence landscape data',
  showIndices = false,
  sortable = false,
}) => {
  const tableRef = useRef<HTMLTableElement>(null);
  const [focusedCell, setFocusedCell] = useState<{ row: number; col: number } | null>(null);
  const [sortState, setSortState] = useState<SortState>({ column: null, direction: null });

  // Calculate sorted row indices
  const sortedRowIndices = useMemo(() => {
    const indices = Array.from({ length: height }, (_, i) => i);

    if (sortState.column === null || sortState.direction === null) {
      return indices;
    }

    return indices.sort((a, b) => {
      const valA = grid[a]?.[sortState.column!] ?? 0;
      const valB = grid[b]?.[sortState.column!] ?? 0;
      return sortState.direction === 'asc' ? valA - valB : valB - valA;
    });
  }, [grid, height, sortState]);

  // Check if a cell is highlighted
  const getCellHighlight = useCallback(
    (row: number, col: number): HighlightedCell | null => {
      return highlightedCells.find((h) => h.row === row && h.col === col) ?? null;
    },
    [highlightedCells]
  );

  // Create GridCell object
  const createGridCell = useCallback(
    (row: number, col: number): GridCell => {
      const value = grid[row]?.[col] ?? 0;
      const highlight = getCellHighlight(row, col);

      return {
        row,
        col,
        value,
        xLabel: xLabels[col] ?? `Column ${col + 1}`,
        yLabel: yLabels[row] ?? `Row ${row + 1}`,
        highlighted: highlight !== null,
      };
    },
    [grid, xLabels, yLabels, getCellHighlight]
  );

  // Handle cell click
  const handleCellClick = useCallback(
    (row: number, col: number) => {
      const cell = createGridCell(row, col);
      onCellClick?.(cell);
    },
    [createGridCell, onCellClick]
  );

  // Handle cell focus
  const handleCellFocus = useCallback(
    (row: number, col: number) => {
      setFocusedCell({ row, col });
      const cell = createGridCell(row, col);
      onCellFocus?.(cell);
    },
    [createGridCell, onCellFocus]
  );

  // Handle cell blur
  const handleCellBlur = useCallback(() => {
    setFocusedCell(null);
    onCellFocus?.(null);
  }, [onCellFocus]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent, row: number, col: number) => {
      let newRow = row;
      let newCol = col;

      switch (e.key) {
        case 'ArrowUp':
          e.preventDefault();
          newRow = Math.max(0, row - 1);
          break;
        case 'ArrowDown':
          e.preventDefault();
          newRow = Math.min(height - 1, row + 1);
          break;
        case 'ArrowLeft':
          e.preventDefault();
          newCol = Math.max(0, col - 1);
          break;
        case 'ArrowRight':
          e.preventDefault();
          newCol = Math.min(width - 1, col + 1);
          break;
        case 'Home':
          e.preventDefault();
          if (e.ctrlKey) {
            newRow = 0;
            newCol = 0;
          } else {
            newCol = 0;
          }
          break;
        case 'End':
          e.preventDefault();
          if (e.ctrlKey) {
            newRow = height - 1;
            newCol = width - 1;
          } else {
            newCol = width - 1;
          }
          break;
        case 'Enter':
        case ' ':
          e.preventDefault();
          handleCellClick(row, col);
          return;
        default:
          return;
      }

      // Focus the new cell
      const cellId = `cell-${newRow}-${newCol}`;
      const cellElement = document.getElementById(cellId);
      cellElement?.focus();
    },
    [height, width, handleCellClick]
  );

  // Handle column header click for sorting
  const handleHeaderClick = useCallback(
    (col: number) => {
      if (!sortable) return;

      setSortState((prev) => {
        if (prev.column !== col) {
          return { column: col, direction: 'desc' };
        }
        if (prev.direction === 'desc') {
          return { column: col, direction: 'asc' };
        }
        return { column: null, direction: null };
      });
    },
    [sortable]
  );

  // Calculate statistics
  const stats = useMemo(() => {
    const values = grid.flat().filter((v) => !isNaN(v));
    if (values.length === 0) {
      return { mean: 0, min: 0, max: 0, count: 0 };
    }

    const sum = values.reduce((a, b) => a + b, 0);
    return {
      mean: sum / values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      count: values.length,
    };
  }, [grid]);

  // Base styles
  const cellSize = compact ? 'px-2 py-1 text-xs' : 'px-3 py-2 text-sm';
  const headerBg = darkMode ? 'bg-gray-800' : 'bg-gray-100';
  const tableBorder = darkMode ? 'border-gray-700' : 'border-gray-200';
  const headerText = darkMode ? 'text-gray-200' : 'text-gray-700';

  if (width === 0 || height === 0 || grid.length === 0) {
    return (
      <div
        className={`${className} ${darkMode ? 'text-gray-400' : 'text-gray-500'} p-4 text-center`}
        role="status"
      >
        No data available
      </div>
    );
  }

  return (
    <div className={`${className} overflow-auto`}>
      {/* Summary for screen readers */}
      <div className="sr-only" role="status" aria-live="polite">
        {focusedCell && (
          <>
            {`${yLabels[focusedCell.row] ?? `Row ${focusedCell.row + 1}`}, ${xLabels[focusedCell.col] ?? `Column ${focusedCell.col + 1}`}: ${showPercentage ? `${((grid[focusedCell.row]?.[focusedCell.col] ?? 0) * 100).toFixed(1)}%` : (grid[focusedCell.row]?.[focusedCell.col] ?? 0).toFixed(3)}, ${getConfidenceLevel(grid[focusedCell.row]?.[focusedCell.col] ?? 0)} confidence`}
          </>
        )}
      </div>

      {/* Statistics summary */}
      <div
        className={`mb-2 text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}
        aria-hidden="true"
      >
        <span className="mr-4">
          Avg: {showPercentage ? `${(stats.mean * 100).toFixed(1)}%` : stats.mean.toFixed(3)}
        </span>
        <span className="mr-4">
          Range: {showPercentage ? `${(stats.min * 100).toFixed(0)}-${(stats.max * 100).toFixed(0)}%` : `${stats.min.toFixed(2)}-${stats.max.toFixed(2)}`}
        </span>
        <span>Cells: {stats.count}</span>
      </div>

      <table
        ref={tableRef}
        className={`border-collapse border ${tableBorder} w-full`}
        role="grid"
        aria-label={caption}
        aria-rowcount={height + 1}
        aria-colcount={width + 1}
      >
        <caption className="sr-only">{caption}</caption>

        {/* Column headers */}
        <thead>
          <tr>
            {/* Corner cell */}
            <th
              className={`${cellSize} ${headerBg} ${headerText} border ${tableBorder} font-medium sticky left-0 top-0 z-20`}
              scope="col"
            >
              {showIndices ? '#' : ''}
            </th>

            {/* Column headers */}
            {Array.from({ length: width }, (_, col) => (
              <th
                key={col}
                className={`${cellSize} ${headerBg} ${headerText} border ${tableBorder} font-medium sticky top-0 z-10 ${sortable ? 'cursor-pointer hover:bg-opacity-80' : ''}`}
                scope="col"
                onClick={() => handleHeaderClick(col)}
                onKeyDown={(e) => {
                  if (sortable && (e.key === 'Enter' || e.key === ' ')) {
                    e.preventDefault();
                    handleHeaderClick(col);
                  }
                }}
                tabIndex={sortable ? 0 : -1}
                aria-sort={
                  sortState.column === col
                    ? sortState.direction === 'asc'
                      ? 'ascending'
                      : 'descending'
                    : undefined
                }
              >
                <div className="flex items-center justify-center gap-1">
                  <span>{xLabels[col] ?? (showIndices ? col + 1 : `Col ${col + 1}`)}</span>
                  {sortable && sortState.column === col && (
                    <span aria-hidden="true">
                      {sortState.direction === 'asc' ? '↑' : '↓'}
                    </span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>

        {/* Data rows */}
        <tbody>
          {sortedRowIndices.map((originalRow, displayIndex) => (
            <tr key={originalRow}>
              {/* Row header */}
              <th
                className={`${cellSize} ${headerBg} ${headerText} border ${tableBorder} font-medium sticky left-0 z-10`}
                scope="row"
              >
                {yLabels[originalRow] ?? (showIndices ? originalRow + 1 : `Row ${originalRow + 1}`)}
              </th>

              {/* Data cells */}
              {Array.from({ length: width }, (_, col) => {
                const value = grid[originalRow]?.[col] ?? 0;
                const highlight = getCellHighlight(originalRow, col);
                const isFocused =
                  focusedCell?.row === originalRow && focusedCell?.col === col;

                const bgColor = getConfidenceColor(value, minValue, maxValue, darkMode);
                const textColor = getTextColor(value, darkMode);

                return (
                  <td
                    key={col}
                    id={`cell-${originalRow}-${col}`}
                    className={`${cellSize} border ${tableBorder} text-center font-mono tabular-nums cursor-pointer transition-shadow duration-150`}
                    style={{
                      backgroundColor: bgColor,
                      color: textColor,
                      boxShadow: highlight
                        ? `inset 0 0 0 2px ${getHighlightBorderColor(highlight.reason)}`
                        : isFocused
                          ? `inset 0 0 0 2px ${darkMode ? '#60a5fa' : '#3b82f6'}`
                          : undefined,
                    }}
                    tabIndex={0}
                    role="gridcell"
                    aria-rowindex={displayIndex + 2}
                    aria-colindex={col + 2}
                    aria-selected={highlight !== null}
                    aria-label={`${yLabels[originalRow] ?? `Row ${originalRow + 1}`}, ${xLabels[col] ?? `Column ${col + 1}`}: ${showPercentage ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)}, ${getConfidenceLevel(value)} confidence${highlight ? ', highlighted' : ''}`}
                    onClick={() => handleCellClick(originalRow, col)}
                    onFocus={() => handleCellFocus(originalRow, col)}
                    onBlur={handleCellBlur}
                    onKeyDown={(e) => handleKeyDown(e, originalRow, col)}
                  >
                    {showPercentage ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>

      {/* Keyboard instructions for screen readers */}
      <div className="sr-only">
        Use arrow keys to navigate between cells. Press Enter or Space to select a cell.
        {sortable && ' Click column headers to sort.'}
      </div>

      {/* Visual keyboard hint */}
      <div
        className={`mt-2 text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}
        aria-hidden="true"
      >
        Arrow keys to navigate • Enter to select
        {sortable && ' • Click headers to sort'}
      </div>
    </div>
  );
};

export default ConfidenceTable;
