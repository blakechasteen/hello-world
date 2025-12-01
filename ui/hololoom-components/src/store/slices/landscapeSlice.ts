/**
 * Landscape Slice
 *
 * Zustand store slice for Confidence Landscape state management.
 * Manages 2D confidence grid data, cell highlighting, and view modes.
 *
 * @module store/slices/landscapeSlice
 */

import type { StateCreator } from 'zustand';

// ============================================================================
// Types
// ============================================================================

/**
 * A single cell in the confidence grid
 */
export interface GridCell {
  /** Row index */
  row: number;
  /** Column index */
  col: number;
  /** Confidence value (0.0-1.0) */
  value: number;
  /** X-axis label for this column */
  xLabel: string;
  /** Y-axis label for this row */
  yLabel: string;
  /** Whether this cell is highlighted (e.g., on retrieval path) */
  highlighted: boolean;
  /** Optional metadata about this cell */
  metadata?: Record<string, unknown>;
}

/**
 * Highlight reasons for cells
 */
export type HighlightReason =
  | 'retrieval_path'
  | 'high_confidence'
  | 'low_confidence'
  | 'anomaly'
  | 'selected'
  | 'related';

/**
 * Highlighted cell with reason
 */
export interface HighlightedCell {
  row: number;
  col: number;
  reason: HighlightReason;
  /** Strength of highlight (0.0-1.0) */
  strength: number;
}

/**
 * View mode for the confidence landscape
 */
export type LandscapeViewMode = 'contour' | 'table' | 'heatmap';

/**
 * Color scale for the visualization
 */
export type ColorScale = 'viridis' | 'plasma' | 'inferno' | 'magma' | 'cividis' | 'turbo';

/**
 * Landscape state
 */
export interface LandscapeState {
  /** 2D grid of confidence values */
  grid: number[][];
  /** Grid dimensions */
  width: number;
  height: number;
  /** X-axis labels (columns) */
  xLabels: string[];
  /** Y-axis labels (rows) */
  yLabels: string[];
  /** Minimum value in grid */
  minValue: number;
  /** Maximum value in grid */
  maxValue: number;
  /** Highlighted cells */
  highlightedCells: HighlightedCell[];
  /** Currently hovered cell */
  hoveredCell: { row: number; col: number } | null;
  /** Currently selected cell */
  selectedCell: { row: number; col: number } | null;
  /** Current view mode */
  viewMode: LandscapeViewMode;
  /** Color scale */
  colorScale: ColorScale;
  /** Whether to show contour lines in heatmap mode */
  showContourLines: boolean;
  /** Contour line interval (0.0-1.0) */
  contourInterval: number;
  /** Whether to show values in cells (table mode) */
  showValues: boolean;
  /** Whether landscape is loading */
  isLoading: boolean;
  /** Error message if any */
  error: string | null;
  /** Animation progress (0.0-1.0) for transitions */
  animationProgress: number;
  /** Previous grid for smooth transitions */
  previousGrid: number[][] | null;
}

/**
 * Landscape actions
 */
export interface LandscapeActions {
  /** Set confidence grid data */
  setConfidenceGrid: (data: {
    grid: number[][];
    width: number;
    height: number;
    xLabels: string[];
    yLabels: string[];
    minValue?: number;
    maxValue?: number;
    highlightCells?: Array<{ row: number; col: number; reason?: HighlightReason; strength?: number }>;
  }) => void;
  /** Set hovered cell */
  setHoveredCell: (cell: { row: number; col: number } | null) => void;
  /** Set selected cell */
  setSelectedCell: (cell: { row: number; col: number } | null) => void;
  /** Add highlighted cells */
  addHighlightedCells: (cells: HighlightedCell[]) => void;
  /** Clear highlighted cells */
  clearHighlightedCells: () => void;
  /** Set view mode */
  setViewMode: (mode: LandscapeViewMode) => void;
  /** Set color scale */
  setColorScale: (scale: ColorScale) => void;
  /** Toggle contour lines */
  toggleContourLines: () => void;
  /** Set contour interval */
  setContourInterval: (interval: number) => void;
  /** Toggle value display */
  toggleValues: () => void;
  /** Set loading state */
  setLoading: (loading: boolean) => void;
  /** Set error */
  setError: (error: string | null) => void;
  /** Update animation progress */
  setAnimationProgress: (progress: number) => void;
  /** Reset landscape state */
  resetLandscape: () => void;
  /** Get cell at position */
  getCellAt: (row: number, col: number) => GridCell | null;
  /** Get interpolated value at fractional position */
  getInterpolatedValue: (x: number, y: number) => number;
}

/**
 * Combined landscape slice type
 */
export type LandscapeSlice = LandscapeState & LandscapeActions;

// ============================================================================
// Initial State
// ============================================================================

const initialState: LandscapeState = {
  grid: [],
  width: 0,
  height: 0,
  xLabels: [],
  yLabels: [],
  minValue: 0,
  maxValue: 1,
  highlightedCells: [],
  hoveredCell: null,
  selectedCell: null,
  viewMode: 'heatmap',
  colorScale: 'viridis',
  showContourLines: true,
  contourInterval: 0.1,
  showValues: true,
  isLoading: false,
  error: null,
  animationProgress: 1,
  previousGrid: null,
};

// ============================================================================
// Slice Creator
// ============================================================================

/**
 * Create the landscape slice
 */
export const createLandscapeSlice: StateCreator<LandscapeSlice, [], [], LandscapeSlice> = (
  set,
  get
) => ({
  ...initialState,

  setConfidenceGrid: (data) => {
    const { grid, width, height, xLabels, yLabels, highlightCells } = data;

    // Calculate min/max if not provided
    let minValue = data.minValue ?? Infinity;
    let maxValue = data.maxValue ?? -Infinity;

    if (data.minValue === undefined || data.maxValue === undefined) {
      for (const row of grid) {
        for (const val of row) {
          if (val < minValue) minValue = val;
          if (val > maxValue) maxValue = val;
        }
      }
    }

    // Ensure we have valid bounds
    if (!isFinite(minValue)) minValue = 0;
    if (!isFinite(maxValue)) maxValue = 1;
    if (minValue === maxValue) {
      minValue = 0;
      maxValue = 1;
    }

    // Process highlighted cells
    const highlightedCells: HighlightedCell[] = (highlightCells ?? []).map((cell) => ({
      row: cell.row,
      col: cell.col,
      reason: cell.reason ?? 'selected',
      strength: cell.strength ?? 1,
    }));

    set((state) => ({
      previousGrid: state.grid.length > 0 ? state.grid : null,
      grid,
      width,
      height,
      xLabels,
      yLabels,
      minValue,
      maxValue,
      highlightedCells,
      animationProgress: 0, // Reset animation for smooth transition
      isLoading: false,
      error: null,
    }));
  },

  setHoveredCell: (cell) => {
    set({ hoveredCell: cell });
  },

  setSelectedCell: (cell) => {
    set({ selectedCell: cell });
  },

  addHighlightedCells: (cells) => {
    set((state) => ({
      highlightedCells: [
        ...state.highlightedCells.filter(
          (existing) => !cells.some((c) => c.row === existing.row && c.col === existing.col)
        ),
        ...cells,
      ],
    }));
  },

  clearHighlightedCells: () => {
    set({ highlightedCells: [] });
  },

  setViewMode: (mode) => {
    set({ viewMode: mode });
  },

  setColorScale: (scale) => {
    set({ colorScale: scale });
  },

  toggleContourLines: () => {
    set((state) => ({ showContourLines: !state.showContourLines }));
  },

  setContourInterval: (interval) => {
    set({ contourInterval: Math.max(0.01, Math.min(0.5, interval)) });
  },

  toggleValues: () => {
    set((state) => ({ showValues: !state.showValues }));
  },

  setLoading: (loading) => {
    set({ isLoading: loading });
  },

  setError: (error) => {
    set({ error, isLoading: false });
  },

  setAnimationProgress: (progress) => {
    set({ animationProgress: Math.max(0, Math.min(1, progress)) });
  },

  resetLandscape: () => {
    set(initialState);
  },

  getCellAt: (row, col) => {
    const state = get();
    if (row < 0 || row >= state.height || col < 0 || col >= state.width) {
      return null;
    }

    const value = state.grid[row]?.[col];
    if (value === undefined) return null;

    const isHighlighted = state.highlightedCells.some(
      (h) => h.row === row && h.col === col
    );

    return {
      row,
      col,
      value,
      xLabel: state.xLabels[col] ?? `Col ${col}`,
      yLabel: state.yLabels[row] ?? `Row ${row}`,
      highlighted: isHighlighted,
    };
  },

  getInterpolatedValue: (x, y) => {
    const state = get();
    const { grid, width, height } = state;

    if (width === 0 || height === 0 || grid.length === 0) return 0;

    // Clamp to grid bounds
    const clampedX = Math.max(0, Math.min(width - 1, x));
    const clampedY = Math.max(0, Math.min(height - 1, y));

    // Get the four surrounding cells
    const x0 = Math.floor(clampedX);
    const x1 = Math.min(x0 + 1, width - 1);
    const y0 = Math.floor(clampedY);
    const y1 = Math.min(y0 + 1, height - 1);

    // Get the fractional parts
    const xFrac = clampedX - x0;
    const yFrac = clampedY - y0;

    // Bilinear interpolation
    const v00 = grid[y0]?.[x0] ?? 0;
    const v10 = grid[y0]?.[x1] ?? 0;
    const v01 = grid[y1]?.[x0] ?? 0;
    const v11 = grid[y1]?.[x1] ?? 0;

    const v0 = v00 * (1 - xFrac) + v10 * xFrac;
    const v1 = v01 * (1 - xFrac) + v11 * xFrac;

    return v0 * (1 - yFrac) + v1 * yFrac;
  },
});

// ============================================================================
// Selectors
// ============================================================================

/**
 * Select the confidence grid
 */
export const selectGrid = (state: LandscapeSlice): number[][] => state.grid;

/**
 * Select grid dimensions
 */
export const selectGridDimensions = (
  state: LandscapeSlice
): { width: number; height: number } => ({
  width: state.width,
  height: state.height,
});

/**
 * Select axis labels
 */
export const selectAxisLabels = (
  state: LandscapeSlice
): { xLabels: string[]; yLabels: string[] } => ({
  xLabels: state.xLabels,
  yLabels: state.yLabels,
});

/**
 * Select value range
 */
export const selectValueRange = (
  state: LandscapeSlice
): { min: number; max: number } => ({
  min: state.minValue,
  max: state.maxValue,
});

/**
 * Select highlighted cells
 */
export const selectHighlightedCells = (state: LandscapeSlice): HighlightedCell[] =>
  state.highlightedCells;

/**
 * Select hovered cell info
 */
export const selectHoveredCellInfo = (state: LandscapeSlice): GridCell | null => {
  if (!state.hoveredCell) return null;
  return state.getCellAt(state.hoveredCell.row, state.hoveredCell.col);
};

/**
 * Select selected cell info
 */
export const selectSelectedCellInfo = (state: LandscapeSlice): GridCell | null => {
  if (!state.selectedCell) return null;
  return state.getCellAt(state.selectedCell.row, state.selectedCell.col);
};

/**
 * Select view settings
 */
export const selectViewSettings = (
  state: LandscapeSlice
): {
  viewMode: LandscapeViewMode;
  colorScale: ColorScale;
  showContourLines: boolean;
  contourInterval: number;
  showValues: boolean;
} => ({
  viewMode: state.viewMode,
  colorScale: state.colorScale,
  showContourLines: state.showContourLines,
  contourInterval: state.contourInterval,
  showValues: state.showValues,
});

/**
 * Select loading state
 */
export const selectIsLandscapeLoading = (state: LandscapeSlice): boolean => state.isLoading;

/**
 * Select error
 */
export const selectLandscapeError = (state: LandscapeSlice): string | null => state.error;

/**
 * Select animation progress
 */
export const selectAnimationProgress = (state: LandscapeSlice): number =>
  state.animationProgress;

/**
 * Check if a cell is highlighted
 */
export const selectIsCellHighlighted = (
  state: LandscapeSlice,
  row: number,
  col: number
): boolean => state.highlightedCells.some((h) => h.row === row && h.col === col);

/**
 * Get highlight info for a cell
 */
export const selectCellHighlight = (
  state: LandscapeSlice,
  row: number,
  col: number
): HighlightedCell | null =>
  state.highlightedCells.find((h) => h.row === row && h.col === col) ?? null;

/**
 * Select high confidence cells (above threshold)
 */
export const selectHighConfidenceCells = (
  state: LandscapeSlice,
  threshold: number = 0.8
): GridCell[] => {
  const cells: GridCell[] = [];

  for (let row = 0; row < state.height; row++) {
    for (let col = 0; col < state.width; col++) {
      const value = state.grid[row]?.[col];
      if (value !== undefined && value >= threshold) {
        const cell = state.getCellAt(row, col);
        if (cell) cells.push(cell);
      }
    }
  }

  return cells.sort((a, b) => b.value - a.value);
};

/**
 * Select low confidence cells (below threshold)
 */
export const selectLowConfidenceCells = (
  state: LandscapeSlice,
  threshold: number = 0.3
): GridCell[] => {
  const cells: GridCell[] = [];

  for (let row = 0; row < state.height; row++) {
    for (let col = 0; col < state.width; col++) {
      const value = state.grid[row]?.[col];
      if (value !== undefined && value <= threshold) {
        const cell = state.getCellAt(row, col);
        if (cell) cells.push(cell);
      }
    }
  }

  return cells.sort((a, b) => a.value - b.value);
};

/**
 * Calculate statistics for the grid
 */
export const selectGridStatistics = (
  state: LandscapeSlice
): {
  mean: number;
  median: number;
  stdDev: number;
  totalCells: number;
} => {
  const values: number[] = [];

  for (const row of state.grid) {
    for (const val of row) {
      if (val !== undefined && !isNaN(val)) {
        values.push(val);
      }
    }
  }

  if (values.length === 0) {
    return { mean: 0, median: 0, stdDev: 0, totalCells: 0 };
  }

  // Mean
  const sum = values.reduce((a, b) => a + b, 0);
  const mean = sum / values.length;

  // Median
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];

  // Standard deviation
  const squaredDiffs = values.map((v) => Math.pow(v - mean, 2));
  const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
  const stdDev = Math.sqrt(avgSquaredDiff);

  return {
    mean,
    median,
    stdDev,
    totalCells: values.length,
  };
};
