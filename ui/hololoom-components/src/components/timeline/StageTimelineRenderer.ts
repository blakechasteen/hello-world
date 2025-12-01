/**
 * StageTimelineRenderer
 *
 * High-performance Canvas 2D renderer for the 9-stage weaving cycle timeline.
 * Renders stage nodes, connections, and animations at 60fps.
 *
 * Design principles:
 * - Canvas 2D for performance (not DOM elements)
 * - Object pooling for minimal allocations
 * - Double-buffering for smooth updates
 * - Respects prefers-reduced-motion
 *
 * @module components/timeline/StageTimelineRenderer
 */

import type { StageTimingData, StageStatus } from '../../store/slices/timelineSlice';
import { WEAVING_STAGES, STAGE_DISPLAY_NAMES } from '../../store/slices/timelineSlice';
import type { WeavingStage } from '../../types';

// ============================================================================
// Types
// ============================================================================

export interface StageTimelineConfig {
  /** Canvas width */
  width: number;
  /** Canvas height */
  height: number;
  /** Node radius in pixels */
  nodeRadius: number;
  /** Spacing between nodes */
  nodeSpacing: number;
  /** Vertical padding */
  paddingY: number;
  /** Horizontal padding */
  paddingX: number;
  /** Whether to animate (respects reduced motion) */
  animate: boolean;
  /** Animation pulse speed (ms per cycle) */
  pulseSpeed: number;
  /** Show duration labels */
  showDurations: boolean;
  /** Show stage names */
  showNames: boolean;
  /** Dark mode */
  darkMode: boolean;
  /** Device pixel ratio for crisp rendering */
  dpr: number;
}

export interface StageNodePosition {
  x: number;
  y: number;
  stage: WeavingStage;
}

export interface RenderState {
  stages: StageTimingData[];
  activeStage: WeavingStage | null;
  tokenRate: number;
  tokenRateHistory: number[];
  timestamp: number;
}

// ============================================================================
// Color Palette
// ============================================================================

const COLORS = {
  light: {
    pending: '#e5e7eb',      // gray-200
    pendingStroke: '#d1d5db', // gray-300
    active: '#3b82f6',       // blue-500
    activeGlow: 'rgba(59, 130, 246, 0.3)',
    complete: '#22c55e',     // green-500
    completeGlow: 'rgba(34, 197, 94, 0.2)',
    error: '#ef4444',        // red-500
    connector: '#d1d5db',    // gray-300
    connectorComplete: '#86efac', // green-300
    text: '#1f2937',         // gray-800
    textMuted: '#6b7280',    // gray-500
    background: '#ffffff',
    sparkline: '#3b82f6',
    sparklineFill: 'rgba(59, 130, 246, 0.1)',
  },
  dark: {
    pending: '#374151',      // gray-700
    pendingStroke: '#4b5563', // gray-600
    active: '#60a5fa',       // blue-400
    activeGlow: 'rgba(96, 165, 250, 0.4)',
    complete: '#4ade80',     // green-400
    completeGlow: 'rgba(74, 222, 128, 0.2)',
    error: '#f87171',        // red-400
    connector: '#4b5563',    // gray-600
    connectorComplete: '#22c55e', // green-500
    text: '#f9fafb',         // gray-50
    textMuted: '#9ca3af',    // gray-400
    background: '#1f2937',
    sparkline: '#60a5fa',
    sparklineFill: 'rgba(96, 165, 250, 0.1)',
  },
};

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: StageTimelineConfig = {
  width: 800,
  height: 120,
  nodeRadius: 16,
  nodeSpacing: 80,
  paddingY: 40,
  paddingX: 40,
  animate: true,
  pulseSpeed: 1500,
  showDurations: true,
  showNames: true,
  darkMode: false,
  dpr: typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1,
};

// ============================================================================
// Renderer Class
// ============================================================================

export class StageTimelineRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private config: StageTimelineConfig;
  private nodePositions: StageNodePosition[] = [];
  private animationPhase = 0;
  private offscreenCanvas: HTMLCanvasElement | null = null;
  private offscreenCtx: CanvasRenderingContext2D | null = null;

  constructor(canvas: HTMLCanvasElement, config: Partial<StageTimelineConfig> = {}) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to get 2D context from canvas');
    }
    this.ctx = ctx;
    this.config = { ...DEFAULT_CONFIG, ...config };

    this.setupCanvas();
    this.calculateNodePositions();
    this.setupOffscreenCanvas();
  }

  /**
   * Setup canvas with proper DPR scaling for crisp rendering
   */
  private setupCanvas(): void {
    const { width, height, dpr } = this.config;

    // Set display size
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;

    // Set actual size in memory (scaled for DPR)
    this.canvas.width = width * dpr;
    this.canvas.height = height * dpr;

    // Scale context to match
    this.ctx.scale(dpr, dpr);
  }

  /**
   * Setup offscreen canvas for double-buffering
   */
  private setupOffscreenCanvas(): void {
    if (typeof OffscreenCanvas !== 'undefined') {
      const { width, height, dpr } = this.config;
      this.offscreenCanvas = document.createElement('canvas');
      this.offscreenCanvas.width = width * dpr;
      this.offscreenCanvas.height = height * dpr;
      this.offscreenCtx = this.offscreenCanvas.getContext('2d');
      if (this.offscreenCtx) {
        this.offscreenCtx.scale(dpr, dpr);
      }
    }
  }

  /**
   * Calculate positions for all stage nodes
   */
  private calculateNodePositions(): void {
    const { nodeSpacing, paddingX, height } = this.config;
    const centerY = height / 2;

    this.nodePositions = WEAVING_STAGES.map((stage, index) => ({
      x: paddingX + index * nodeSpacing,
      y: centerY,
      stage,
    }));
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<StageTimelineConfig>): void {
    const needsResize = config.width !== undefined || config.height !== undefined || config.dpr !== undefined;

    this.config = { ...this.config, ...config };

    if (needsResize) {
      this.setupCanvas();
      this.setupOffscreenCanvas();
    }

    if (config.nodeSpacing !== undefined || config.paddingX !== undefined) {
      this.calculateNodePositions();
    }
  }

  /**
   * Get colors based on dark mode
   */
  private getColors() {
    return this.config.darkMode ? COLORS.dark : COLORS.light;
  }

  /**
   * Render the complete timeline
   */
  render(state: RenderState): void {
    const ctx = this.offscreenCtx || this.ctx;
    const colors = this.getColors();
    const { width, height, animate, pulseSpeed } = this.config;

    // Update animation phase
    if (animate) {
      this.animationPhase = (state.timestamp % pulseSpeed) / pulseSpeed;
    }

    // Clear canvas
    ctx.fillStyle = colors.background;
    ctx.fillRect(0, 0, width, height);

    // Draw connectors first (behind nodes)
    this.drawConnectors(ctx, state, colors);

    // Draw nodes
    for (let i = 0; i < this.nodePositions.length; i++) {
      const pos = this.nodePositions[i];
      const stageTiming = state.stages.find(s => s.stage === pos.stage);
      const isActive = pos.stage === state.activeStage;

      this.drawNode(ctx, pos, stageTiming?.status || 'pending', isActive, colors);
    }

    // Draw sparkline for token rate
    if (state.tokenRateHistory.length > 1) {
      this.drawSparkline(ctx, state.tokenRateHistory, colors);
    }

    // Draw duration labels
    if (this.config.showDurations) {
      this.drawDurations(ctx, state, colors);
    }

    // Draw stage names
    if (this.config.showNames) {
      this.drawStageNames(ctx, colors);
    }

    // Copy from offscreen canvas if using double-buffering
    if (this.offscreenCanvas && this.offscreenCtx) {
      this.ctx.drawImage(
        this.offscreenCanvas,
        0, 0,
        this.offscreenCanvas.width,
        this.offscreenCanvas.height,
        0, 0,
        this.config.width,
        this.config.height
      );
    }
  }

  /**
   * Draw connector lines between nodes
   */
  private drawConnectors(
    ctx: CanvasRenderingContext2D,
    state: RenderState,
    colors: typeof COLORS.light
  ): void {
    const { nodeRadius } = this.config;

    for (let i = 0; i < this.nodePositions.length - 1; i++) {
      const from = this.nodePositions[i];
      const to = this.nodePositions[i + 1];
      const fromStage = state.stages.find(s => s.stage === from.stage);
      const isComplete = fromStage?.status === 'complete';

      ctx.beginPath();
      ctx.moveTo(from.x + nodeRadius + 4, from.y);
      ctx.lineTo(to.x - nodeRadius - 4, to.y);

      ctx.strokeStyle = isComplete ? colors.connectorComplete : colors.connector;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  /**
   * Draw a single stage node
   */
  private drawNode(
    ctx: CanvasRenderingContext2D,
    pos: StageNodePosition,
    status: StageStatus,
    isActive: boolean,
    colors: typeof COLORS.light
  ): void {
    const { nodeRadius, animate } = this.config;
    const { x, y } = pos;

    // Draw glow for active/complete states
    if (isActive && animate) {
      const pulseScale = 1 + Math.sin(this.animationPhase * Math.PI * 2) * 0.15;
      const glowRadius = nodeRadius * pulseScale + 8;

      ctx.beginPath();
      ctx.arc(x, y, glowRadius, 0, Math.PI * 2);
      ctx.fillStyle = colors.activeGlow;
      ctx.fill();
    } else if (status === 'complete') {
      ctx.beginPath();
      ctx.arc(x, y, nodeRadius + 4, 0, Math.PI * 2);
      ctx.fillStyle = colors.completeGlow;
      ctx.fill();
    }

    // Draw node circle
    ctx.beginPath();
    ctx.arc(x, y, nodeRadius, 0, Math.PI * 2);

    // Fill based on status
    switch (status) {
      case 'active':
        ctx.fillStyle = colors.active;
        break;
      case 'complete':
        ctx.fillStyle = colors.complete;
        break;
      case 'error':
        ctx.fillStyle = colors.error;
        break;
      default:
        ctx.fillStyle = colors.pending;
    }
    ctx.fill();

    // Draw border
    ctx.strokeStyle = status === 'pending' ? colors.pendingStroke : 'transparent';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw icon/number inside node
    ctx.fillStyle = status === 'pending' ? colors.textMuted : '#ffffff';
    ctx.font = `bold ${nodeRadius * 0.8}px system-ui, -apple-system, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Use checkmark for complete, number for others
    if (status === 'complete') {
      ctx.fillText('✓', x, y);
    } else if (status === 'error') {
      ctx.fillText('!', x, y);
    } else {
      const stageIndex = WEAVING_STAGES.indexOf(pos.stage) + 1;
      ctx.fillText(stageIndex.toString(), x, y);
    }
  }

  /**
   * Draw sparkline for token rate
   */
  private drawSparkline(
    ctx: CanvasRenderingContext2D,
    history: number[],
    colors: typeof COLORS.light
  ): void {
    const { width, height, paddingX } = this.config;
    const sparklineY = height - 20;
    const sparklineWidth = width - paddingX * 2;
    const sparklineHeight = 15;

    if (history.length < 2) return;

    const max = Math.max(...history, 1);
    const min = Math.min(...history, 0);
    const range = max - min || 1;

    // Draw fill
    ctx.beginPath();
    ctx.moveTo(paddingX, sparklineY);

    for (let i = 0; i < history.length; i++) {
      const x = paddingX + (i / (history.length - 1)) * sparklineWidth;
      const y = sparklineY - ((history[i] - min) / range) * sparklineHeight;
      ctx.lineTo(x, y);
    }

    ctx.lineTo(paddingX + sparklineWidth, sparklineY);
    ctx.closePath();
    ctx.fillStyle = colors.sparklineFill;
    ctx.fill();

    // Draw line
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const x = paddingX + (i / (history.length - 1)) * sparklineWidth;
      const y = sparklineY - ((history[i] - min) / range) * sparklineHeight;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.strokeStyle = colors.sparkline;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Draw endpoint
    const lastX = paddingX + sparklineWidth;
    const lastY = sparklineY - ((history[history.length - 1] - min) / range) * sparklineHeight;
    ctx.beginPath();
    ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
    ctx.fillStyle = colors.sparkline;
    ctx.fill();
  }

  /**
   * Draw duration labels below nodes
   */
  private drawDurations(
    ctx: CanvasRenderingContext2D,
    state: RenderState,
    colors: typeof COLORS.light
  ): void {
    const { nodeRadius } = this.config;

    ctx.font = '10px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = colors.textMuted;

    for (const pos of this.nodePositions) {
      const stageTiming = state.stages.find(s => s.stage === pos.stage);
      if (stageTiming?.duration !== null && stageTiming?.duration !== undefined) {
        const label = `${stageTiming.duration.toFixed(0)}ms`;
        ctx.fillText(label, pos.x, pos.y + nodeRadius + 14);
      }
    }
  }

  /**
   * Draw stage names above nodes
   */
  private drawStageNames(
    ctx: CanvasRenderingContext2D,
    colors: typeof COLORS.light
  ): void {
    const { nodeRadius } = this.config;

    ctx.font = '9px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = colors.textMuted;

    for (const pos of this.nodePositions) {
      const name = STAGE_DISPLAY_NAMES[pos.stage];
      // Truncate long names
      const shortName = name.length > 10 ? name.slice(0, 9) + '…' : name;
      ctx.fillText(shortName, pos.x, pos.y - nodeRadius - 8);
    }
  }

  /**
   * Get node at a given point (for hit testing)
   */
  getNodeAtPoint(x: number, y: number): WeavingStage | null {
    const { nodeRadius } = this.config;

    for (const pos of this.nodePositions) {
      const dx = x - pos.x;
      const dy = y - pos.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance <= nodeRadius) {
        return pos.stage;
      }
    }

    return null;
  }

  /**
   * Get node position for a stage
   */
  getNodePosition(stage: WeavingStage): StageNodePosition | null {
    return this.nodePositions.find(p => p.stage === stage) || null;
  }

  /**
   * Get all node positions
   */
  getAllNodePositions(): StageNodePosition[] {
    return [...this.nodePositions];
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    this.offscreenCanvas = null;
    this.offscreenCtx = null;
  }
}

export default StageTimelineRenderer;
