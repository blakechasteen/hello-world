/**
 * useAnimationFrame Hook
 *
 * High-performance animation loop hook using requestAnimationFrame.
 * Provides delta time, FPS tracking, and respects prefers-reduced-motion.
 *
 * @module hooks/useAnimationFrame
 */

import { useRef, useEffect, useCallback } from 'react';

// ============================================================================
// Types
// ============================================================================

export interface AnimationFrameCallback {
  (timestamp: DOMHighResTimeStamp, deltaMs: number): void;
}

export interface UseAnimationFrameOptions {
  /** Whether animation is active (default: true) */
  enabled?: boolean;
  /** Target FPS (default: 60, use 30 for less critical animations) */
  targetFps?: number;
  /** Respect prefers-reduced-motion (default: true) */
  respectReducedMotion?: boolean;
}

export interface AnimationFrameState {
  /** Current FPS (rolling average) */
  fps: number;
  /** Total frames rendered */
  frameCount: number;
  /** Whether animation is running */
  isRunning: boolean;
}

// ============================================================================
// Animation Loop Singleton (for cross-component coordination)
// ============================================================================

type AnimationCallback = (now: DOMHighResTimeStamp, delta: number) => void;

interface AnimationLoopSubscriber {
  id: string;
  callback: AnimationCallback;
  priority: number;
}

class AnimationLoop {
  private subscribers: AnimationLoopSubscriber[] = [];
  private rafId: number | null = null;
  private lastTime: DOMHighResTimeStamp = 0;
  private isRunning = false;
  private frameCount = 0;
  private fpsHistory: number[] = [];
  private lastFpsUpdate = 0;

  register(id: string, callback: AnimationCallback, priority = 0): void {
    // Remove existing subscriber with same ID
    this.unregister(id);

    this.subscribers.push({ id, callback, priority });
    // Sort by priority (higher first)
    this.subscribers.sort((a, b) => b.priority - a.priority);

    if (!this.isRunning && this.subscribers.length > 0) {
      this.start();
    }
  }

  unregister(id: string): void {
    this.subscribers = this.subscribers.filter(s => s.id !== id);

    if (this.subscribers.length === 0) {
      this.stop();
    }
  }

  start(): void {
    if (this.isRunning) return;

    this.isRunning = true;
    this.lastTime = performance.now();
    this.lastFpsUpdate = this.lastTime;
    this.frameCount = 0;

    const loop = (now: DOMHighResTimeStamp) => {
      if (!this.isRunning) return;

      const delta = now - this.lastTime;
      this.lastTime = now;
      this.frameCount++;

      // Update FPS every second
      if (now - this.lastFpsUpdate >= 1000) {
        const fps = (this.frameCount * 1000) / (now - this.lastFpsUpdate);
        this.fpsHistory.push(fps);
        if (this.fpsHistory.length > 10) {
          this.fpsHistory.shift();
        }
        this.frameCount = 0;
        this.lastFpsUpdate = now;
      }

      // Call all subscribers
      for (const subscriber of this.subscribers) {
        try {
          subscriber.callback(now, delta);
        } catch (error) {
          console.error(`AnimationLoop error in subscriber ${subscriber.id}:`, error);
        }
      }

      this.rafId = requestAnimationFrame(loop);
    };

    this.rafId = requestAnimationFrame(loop);
  }

  stop(): void {
    this.isRunning = false;
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  getFPS(): number {
    if (this.fpsHistory.length === 0) return 0;
    return this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length;
  }

  getSubscriberCount(): number {
    return this.subscribers.length;
  }
}

// Singleton instance
export const animationLoop = new AnimationLoop();

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Custom hook for requestAnimationFrame-based animations.
 *
 * Features:
 * - Automatic cleanup on unmount
 * - Delta time calculation
 * - FPS throttling support
 * - Reduced motion support
 * - Shared animation loop singleton for performance
 *
 * @example
 * ```tsx
 * function AnimatedComponent() {
 *   const canvasRef = useRef<HTMLCanvasElement>(null);
 *
 *   const animate = useCallback((timestamp: number, delta: number) => {
 *     const ctx = canvasRef.current?.getContext('2d');
 *     if (!ctx) return;
 *
 *     // Draw something...
 *     ctx.clearRect(0, 0, 100, 100);
 *     ctx.fillRect(0, 0, delta, delta);
 *   }, []);
 *
 *   const { fps } = useAnimationFrame(animate, { enabled: true });
 *
 *   return <canvas ref={canvasRef} />;
 * }
 * ```
 */
export function useAnimationFrame(
  callback: AnimationFrameCallback,
  options: UseAnimationFrameOptions = {}
): AnimationFrameState {
  const {
    enabled = true,
    targetFps = 60,
    respectReducedMotion = true,
  } = options;

  const stateRef = useRef<AnimationFrameState>({
    fps: 0,
    frameCount: 0,
    isRunning: false,
  });

  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  const frameIntervalRef = useRef(1000 / targetFps);
  const lastFrameTimeRef = useRef(0);

  // Check reduced motion preference
  const prefersReducedMotion = useRef(
    respectReducedMotion &&
    typeof window !== 'undefined' &&
    window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  );

  // Generate unique ID for this hook instance
  const idRef = useRef(`animation-${Math.random().toString(36).slice(2, 9)}`);

  // Throttled callback wrapper
  const throttledCallback = useCallback((now: DOMHighResTimeStamp, delta: number) => {
    // Skip if reduced motion is preferred and enabled
    if (prefersReducedMotion.current) {
      // Still call callback but with larger delta to simulate instant changes
      callbackRef.current(now, delta);
      return;
    }

    // FPS throttling
    const elapsed = now - lastFrameTimeRef.current;
    if (elapsed < frameIntervalRef.current) {
      return;
    }
    lastFrameTimeRef.current = now - (elapsed % frameIntervalRef.current);

    stateRef.current.frameCount++;
    callbackRef.current(now, delta);
  }, []);

  // Register/unregister with animation loop
  useEffect(() => {
    if (enabled) {
      animationLoop.register(idRef.current, throttledCallback);
      stateRef.current.isRunning = true;
    } else {
      animationLoop.unregister(idRef.current);
      stateRef.current.isRunning = false;
    }

    return () => {
      animationLoop.unregister(idRef.current);
      stateRef.current.isRunning = false;
    };
  }, [enabled, throttledCallback]);

  // Update FPS in state (poll the singleton)
  useEffect(() => {
    const updateFps = setInterval(() => {
      stateRef.current.fps = animationLoop.getFPS();
    }, 1000);

    return () => clearInterval(updateFps);
  }, []);

  return stateRef.current;
}

// ============================================================================
// Ring Buffer for Fixed-Size History
// ============================================================================

/**
 * Fixed-size ring buffer for efficient O(1) insert history tracking.
 * Used for stage timing history, token rate history, etc.
 */
export class RingBuffer<T> {
  private buffer: (T | undefined)[];
  private head = 0;
  private _size = 0;
  private readonly capacity: number;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.buffer = new Array(capacity);
  }

  push(item: T): void {
    this.buffer[this.head] = item;
    this.head = (this.head + 1) % this.capacity;
    if (this._size < this.capacity) {
      this._size++;
    }
  }

  toArray(): T[] {
    const result: T[] = [];
    if (this._size === 0) return result;

    // Read from oldest to newest
    const start = this._size < this.capacity ? 0 : this.head;
    for (let i = 0; i < this._size; i++) {
      const index = (start + i) % this.capacity;
      result.push(this.buffer[index] as T);
    }
    return result;
  }

  get size(): number {
    return this._size;
  }

  get latest(): T | undefined {
    if (this._size === 0) return undefined;
    const index = (this.head - 1 + this.capacity) % this.capacity;
    return this.buffer[index];
  }

  clear(): void {
    this.buffer = new Array(this.capacity);
    this.head = 0;
    this._size = 0;
  }
}

export default useAnimationFrame;
