/**
 * Confidence Landscape Canvas 2D Renderer
 *
 * High-performance Canvas 2D rendering for confidence grid visualization.
 * Supports heatmap, contour lines, and cell highlighting.
 *
 * @module components/landscape/ConfidenceLandscapeRenderer
 */

import type { ColorScale, HighlightedCell, LandscapeViewMode } from '../../store/slices/landscapeSlice';

// ============================================================================
// Types
// ============================================================================

export interface LandscapeRenderConfig {
  /** Color scale for the heatmap */
  colorScale: ColorScale;
  /** Whether to show contour lines */
  showContourLines: boolean;
  /** Contour line interval (0.0-1.0) */
  contourInterval: number;
  /** Whether to show values in cells */
  showValues: boolean;
  /** Font size for values */
  valueFontSize: number;
  /** Dark mode */
  darkMode: boolean;
  /** Cell padding in pixels */
  cellPadding: number;
  /** Border radius for cells */
  cellBorderRadius: number;
  /** Highlight border width */
  highlightBorderWidth: number;
  /** Animation duration for transitions */
  animationDuration: number;
}

export interface LandscapeRenderState {
  /** 2D grid of confidence values */
  grid: number[][];
  /** Grid dimensions */
  width: number;
  height: number;
  /** X-axis labels */
  xLabels: string[];
  /** Y-axis labels */
  yLabels: string[];
  /** Minimum value */
  minValue: number;
  /** Maximum value */
  maxValue: number;
  /** Highlighted cells */
  highlightedCells: HighlightedCell[];
  /** Hovered cell */
  hoveredCell: { row: number; col: number } | null;
  /** Selected cell */
  selectedCell: { row: number; col: number } | null;
  /** View mode */
  viewMode: LandscapeViewMode;
  /** Animation progress (0.0-1.0) */
  animationProgress: number;
  /** Previous grid for interpolation */
  previousGrid: number[][] | null;
}

// ============================================================================
// Color Scales
// ============================================================================

/**
 * Color scale definitions (Matplotlib-inspired)
 */
const COLOR_SCALES: Record<ColorScale, Array<[number, number, number]>> = {
  viridis: [
    [68, 1, 84],
    [72, 40, 120],
    [62, 73, 137],
    [49, 104, 142],
    [38, 130, 142],
    [31, 158, 137],
    [53, 183, 121],
    [109, 205, 89],
    [180, 222, 44],
    [253, 231, 37],
  ],
  plasma: [
    [13, 8, 135],
    [75, 3, 161],
    [125, 3, 168],
    [168, 34, 150],
    [203, 70, 121],
    [229, 107, 93],
    [248, 148, 65],
    [253, 195, 40],
    [240, 249, 33],
  ],
  inferno: [
    [0, 0, 4],
    [40, 11, 84],
    [101, 21, 110],
    [159, 42, 99],
    [212, 72, 66],
    [245, 125, 21],
    [250, 193, 39],
    [252, 255, 164],
  ],
  magma: [
    [0, 0, 4],
    [28, 16, 68],
    [79, 18, 123],
    [129, 37, 129],
    [181, 54, 122],
    [229, 80, 100],
    [251, 135, 97],
    [254, 194, 135],
    [252, 253, 191],
  ],
  cividis: [
    [0, 32, 77],
    [0, 67, 106],
    [53, 92, 109],
    [90, 115, 106],
    [126, 138, 104],
    [164, 162, 101],
    [204, 186, 92],
    [248, 213, 75],
    [254, 232, 56],
  ],
  turbo: [
    [48, 18, 59],
    [86, 59, 215],
    [40, 127, 255],
    [18, 185, 205],
    [65, 220, 118],
    [163, 230, 53],
    [239, 202, 47],
    [255, 130, 30],
    [217, 55, 19],
    [122, 4, 3],
  ],
};

// ============================================================================
// Renderer Class
// ============================================================================

/**
 * High-performance Canvas 2D renderer for confidence landscape
 */
export class ConfidenceLandscapeRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private config: LandscapeRenderConfig;
  private dpr: number;

  constructor(canvas: HTMLCanvasElement, config?: Partial<LandscapeRenderConfig>) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get 2D rendering context');
    }
    this.ctx = ctx;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);

    this.config = {
      colorScale: 'viridis',
      showContourLines: true,
      contourInterval: 0.1,
      showValues: true,
      valueFontSize: 10,
      darkMode: false,
      cellPadding: 2,
      cellBorderRadius: 4,
      highlightBorderWidth: 2,
      animationDuration: 300,
      ...config,
    };

    this.resize();
  }

  /**
   * Update renderer configuration
   */
  updateConfig(config: Partial<LandscapeRenderConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Resize canvas to match container
   */
  resize(): void {
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * this.dpr;
    this.canvas.height = rect.height * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);
  }

  /**
   * Get color for a value using the current color scale
   */
  private getColor(value: number, min: number, max: number): string {
    const scale = COLOR_SCALES[this.config.colorScale];
    const t = max > min ? (value - min) / (max - min) : 0.5;
    const clampedT = Math.max(0, Math.min(1, t));

    // Interpolate between color scale stops
    const idx = clampedT * (scale.length - 1);
    const lowIdx = Math.floor(idx);
    const highIdx = Math.min(lowIdx + 1, scale.length - 1);
    const frac = idx - lowIdx;

    const lowColor = scale[lowIdx];
    const highColor = scale[highIdx];

    const r = Math.round(lowColor[0] + (highColor[0] - lowColor[0]) * frac);
    const g = Math.round(lowColor[1] + (highColor[1] - lowColor[1]) * frac);
    const b = Math.round(lowColor[2] + (highColor[2] - lowColor[2]) * frac);

    return `rgb(${r}, ${g}, ${b})`;
  }

  /**
   * Get text color that contrasts with background
   */
  private getTextColor(bgValue: number, min: number, max: number): string {
    const t = max > min ? (bgValue - min) / (max - min) : 0.5;
    // Use dark text on light backgrounds (higher values in viridis)
    return t > 0.6 ? '#000000' : '#ffffff';
  }

  /**
   * Draw rounded rectangle
   */
  private roundRect(
    x: number,
    y: number,
    width: number,
    height: number,
    radius: number
  ): void {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  /**
   * Draw contour lines at specified intervals
   */
  private drawContourLines(
    state: LandscapeRenderState,
    cellWidth: number,
    cellHeight: number,
    offsetX: number,
    offsetY: number
  ): void {
    const { grid, minValue, maxValue } = state;
    const ctx = this.ctx;
    const interval = this.config.contourInterval;

    ctx.strokeStyle = this.config.darkMode
      ? 'rgba(255, 255, 255, 0.3)'
      : 'rgba(0, 0, 0, 0.2)';
    ctx.lineWidth = 1;

    // Calculate contour levels
    const levels: number[] = [];
    for (let v = Math.ceil(minValue / interval) * interval; v <= maxValue; v += interval) {
      levels.push(v);
    }

    // Simple marching squares for each level
    for (const level of levels) {
      ctx.beginPath();

      for (let row = 0; row < state.height - 1; row++) {
        for (let col = 0; col < state.width - 1; col++) {
          const v00 = grid[row]?.[col] ?? 0;
          const v10 = grid[row]?.[col + 1] ?? 0;
          const v01 = grid[row + 1]?.[col] ?? 0;
          const v11 = grid[row + 1]?.[col + 1] ?? 0;

          // Calculate cell corners
          const x0 = offsetX + col * cellWidth;
          const x1 = offsetX + (col + 1) * cellWidth;
          const y0 = offsetY + row * cellHeight;
          const y1 = offsetY + (row + 1) * cellHeight;

          // Determine marching squares case
          const case_idx =
            (v00 >= level ? 8 : 0) |
            (v10 >= level ? 4 : 0) |
            (v11 >= level ? 2 : 0) |
            (v01 >= level ? 1 : 0);

          // Linear interpolation helper
          const lerp = (a: number, b: number, va: number, vb: number): number => {
            const t = va === vb ? 0.5 : (level - va) / (vb - va);
            return a + t * (b - a);
          };

          // Draw line segments based on case
          switch (case_idx) {
            case 1:
            case 14:
              ctx.moveTo(x0, lerp(y0, y1, v00, v01));
              ctx.lineTo(lerp(x0, x1, v01, v11), y1);
              break;
            case 2:
            case 13:
              ctx.moveTo(lerp(x0, x1, v01, v11), y1);
              ctx.lineTo(x1, lerp(y0, y1, v10, v11));
              break;
            case 3:
            case 12:
              ctx.moveTo(x0, lerp(y0, y1, v00, v01));
              ctx.lineTo(x1, lerp(y0, y1, v10, v11));
              break;
            case 4:
            case 11:
              ctx.moveTo(lerp(x0, x1, v00, v10), y0);
              ctx.lineTo(x1, lerp(y0, y1, v10, v11));
              break;
            case 5:
              ctx.moveTo(x0, lerp(y0, y1, v00, v01));
              ctx.lineTo(lerp(x0, x1, v00, v10), y0);
              ctx.moveTo(lerp(x0, x1, v01, v11), y1);
              ctx.lineTo(x1, lerp(y0, y1, v10, v11));
              break;
            case 6:
            case 9:
              ctx.moveTo(lerp(x0, x1, v00, v10), y0);
              ctx.lineTo(lerp(x0, x1, v01, v11), y1);
              break;
            case 7:
            case 8:
              ctx.moveTo(x0, lerp(y0, y1, v00, v01));
              ctx.lineTo(lerp(x0, x1, v00, v10), y0);
              break;
            case 10:
              ctx.moveTo(lerp(x0, x1, v00, v10), y0);
              ctx.lineTo(x1, lerp(y0, y1, v10, v11));
              ctx.moveTo(x0, lerp(y0, y1, v00, v01));
              ctx.lineTo(lerp(x0, x1, v01, v11), y1);
              break;
          }
        }
      }

      ctx.stroke();
    }
  }

  /**
   * Main render function
   */
  render(state: LandscapeRenderState): void {
    const ctx = this.ctx;
    const rect = this.canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Background
    ctx.fillStyle = this.config.darkMode ? '#1e1e1e' : '#ffffff';
    ctx.fillRect(0, 0, width, height);

    if (state.width === 0 || state.height === 0 || state.grid.length === 0) {
      // Draw empty state message
      ctx.fillStyle = this.config.darkMode ? '#888888' : '#666666';
      ctx.font = '14px system-ui, -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('No data available', width / 2, height / 2);
      return;
    }

    // Calculate layout
    const labelMarginY = 60; // Space for Y-axis labels
    const labelMarginX = 40; // Space for X-axis labels
    const legendWidth = 60; // Space for color legend

    const gridWidth = width - labelMarginY - legendWidth - 20;
    const gridHeight = height - labelMarginX - 20;

    const cellWidth = gridWidth / state.width;
    const cellHeight = gridHeight / state.height;

    const offsetX = labelMarginY + 10;
    const offsetY = 10;

    // Interpolate values if animating
    const getInterpolatedValue = (row: number, col: number): number => {
      const currentVal = state.grid[row]?.[col] ?? 0;
      if (!state.previousGrid || state.animationProgress >= 1) {
        return currentVal;
      }
      const prevVal = state.previousGrid[row]?.[col] ?? currentVal;
      return prevVal + (currentVal - prevVal) * state.animationProgress;
    };

    // Draw cells
    const { padding } = { padding: this.config.cellPadding };

    for (let row = 0; row < state.height; row++) {
      for (let col = 0; col < state.width; col++) {
        const value = getInterpolatedValue(row, col);
        const x = offsetX + col * cellWidth + padding;
        const y = offsetY + row * cellHeight + padding;
        const w = cellWidth - padding * 2;
        const h = cellHeight - padding * 2;

        // Cell background
        ctx.fillStyle = this.getColor(value, state.minValue, state.maxValue);
        this.roundRect(x, y, w, h, this.config.cellBorderRadius);
        ctx.fill();

        // Check for highlighting
        const highlight = state.highlightedCells.find(
          (c) => c.row === row && c.col === col
        );
        const isHovered =
          state.hoveredCell?.row === row && state.hoveredCell?.col === col;
        const isSelected =
          state.selectedCell?.row === row && state.selectedCell?.col === col;

        if (highlight || isHovered || isSelected) {
          ctx.strokeStyle = isSelected
            ? '#3b82f6'
            : isHovered
              ? this.config.darkMode
                ? '#ffffff'
                : '#000000'
              : this.getHighlightColor(highlight?.reason);
          ctx.lineWidth = this.config.highlightBorderWidth;
          this.roundRect(x, y, w, h, this.config.cellBorderRadius);
          ctx.stroke();
        }

        // Draw value text
        if (this.config.showValues && w > 30 && h > 20) {
          ctx.fillStyle = this.getTextColor(value, state.minValue, state.maxValue);
          ctx.font = `${this.config.valueFontSize}px system-ui, -apple-system, sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(value.toFixed(2), x + w / 2, y + h / 2);
        }
      }
    }

    // Draw contour lines if enabled
    if (this.config.showContourLines && state.viewMode !== 'table') {
      this.drawContourLines(state, cellWidth, cellHeight, offsetX, offsetY);
    }

    // Draw Y-axis labels
    ctx.fillStyle = this.config.darkMode ? '#cccccc' : '#333333';
    ctx.font = '11px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let row = 0; row < state.height; row++) {
      const label = state.yLabels[row] ?? `Row ${row}`;
      const y = offsetY + row * cellHeight + cellHeight / 2;
      const displayLabel = label.length > 8 ? label.substring(0, 7) + '…' : label;
      ctx.fillText(displayLabel, offsetX - 5, y);
    }

    // Draw X-axis labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    for (let col = 0; col < state.width; col++) {
      const label = state.xLabels[col] ?? `Col ${col}`;
      const x = offsetX + col * cellWidth + cellWidth / 2;
      const displayLabel = label.length > 8 ? label.substring(0, 7) + '…' : label;

      ctx.save();
      ctx.translate(x, offsetY + gridHeight + 5);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(displayLabel, 0, 0);
      ctx.restore();
    }

    // Draw color legend
    this.drawLegend(state, width - legendWidth - 5, offsetY, legendWidth - 10, gridHeight);
  }

  /**
   * Draw color scale legend
   */
  private drawLegend(
    state: LandscapeRenderState,
    x: number,
    y: number,
    width: number,
    height: number
  ): void {
    const ctx = this.ctx;
    const steps = 50;

    // Draw gradient bar
    for (let i = 0; i < steps; i++) {
      const t = 1 - i / (steps - 1); // Reverse so high values are at top
      const value = state.minValue + t * (state.maxValue - state.minValue);
      const cy = y + (i / steps) * height;
      const ch = height / steps + 1;

      ctx.fillStyle = this.getColor(value, state.minValue, state.maxValue);
      ctx.fillRect(x, cy, width, ch);
    }

    // Draw border
    ctx.strokeStyle = this.config.darkMode ? '#444444' : '#cccccc';
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);

    // Draw tick marks and labels
    ctx.fillStyle = this.config.darkMode ? '#cccccc' : '#333333';
    ctx.font = '10px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    const ticks = [0, 0.25, 0.5, 0.75, 1];
    for (const t of ticks) {
      const value = state.minValue + t * (state.maxValue - state.minValue);
      const ty = y + (1 - t) * height;

      ctx.beginPath();
      ctx.moveTo(x + width, ty);
      ctx.lineTo(x + width + 3, ty);
      ctx.stroke();

      ctx.fillText(value.toFixed(2), x + width + 5, ty);
    }
  }

  /**
   * Get highlight border color based on reason
   */
  private getHighlightColor(reason?: string): string {
    switch (reason) {
      case 'retrieval_path':
        return '#ef4444'; // Red
      case 'high_confidence':
        return '#22c55e'; // Green
      case 'low_confidence':
        return '#f59e0b'; // Amber
      case 'anomaly':
        return '#ec4899'; // Pink
      case 'related':
        return '#8b5cf6'; // Purple
      case 'selected':
      default:
        return '#3b82f6'; // Blue
    }
  }

  /**
   * Get cell at screen coordinates
   */
  getCellAtPoint(
    screenX: number,
    screenY: number,
    state: LandscapeRenderState
  ): { row: number; col: number } | null {
    if (state.width === 0 || state.height === 0) return null;

    const rect = this.canvas.getBoundingClientRect();
    const x = screenX - rect.left;
    const y = screenY - rect.top;

    // Calculate layout (same as render)
    const labelMarginY = 60;
    const labelMarginX = 40;
    const legendWidth = 60;

    const gridWidth = rect.width - labelMarginY - legendWidth - 20;
    const gridHeight = rect.height - labelMarginX - 20;

    const cellWidth = gridWidth / state.width;
    const cellHeight = gridHeight / state.height;

    const offsetX = labelMarginY + 10;
    const offsetY = 10;

    // Check if within grid bounds
    if (x < offsetX || x > offsetX + gridWidth) return null;
    if (y < offsetY || y > offsetY + gridHeight) return null;

    const col = Math.floor((x - offsetX) / cellWidth);
    const row = Math.floor((y - offsetY) / cellHeight);

    if (col < 0 || col >= state.width || row < 0 || row >= state.height) {
      return null;
    }

    return { row, col };
  }

  /**
   * Dispose of resources
   */
  dispose(): void {
    // No resources to clean up for 2D context
  }
}
