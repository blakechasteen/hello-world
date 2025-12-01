/**
 * Web Workers
 *
 * Background workers for computationally intensive tasks.
 */

// Export worker types for TypeScript
export type {
  WorkerNode,
  WorkerEdge,
  LayoutConfig,
  InitMessage,
  StepMessage,
  StopMessage,
  UpdateConfigMessage,
  UpdateNodeMessage,
  IncomingMessage,
  PositionUpdate,
  ProgressMessage,
  CompleteMessage,
  ErrorMessage,
  OutgoingMessage,
} from './forceLayout.worker';

/**
 * Create a force layout worker instance.
 *
 * Usage:
 * ```typescript
 * import { createForceLayoutWorker } from './workers';
 *
 * const worker = createForceLayoutWorker();
 * worker.postMessage({ type: 'init', nodes: [...], edges: [...], config: {...} });
 * worker.onmessage = (e) => {
 *   if (e.data.type === 'progress') {
 *     // Update node positions
 *   }
 * };
 * ```
 */
export function createForceLayoutWorker(): Worker {
  // Create worker from the worker file
  // Note: In production, this would be bundled by the build tool (Vite, webpack)
  return new Worker(new URL('./forceLayout.worker.ts', import.meta.url), {
    type: 'module',
  });
}
