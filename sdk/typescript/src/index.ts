/**
 * HoloLoom TypeScript SDK
 * Main entry point - exports client and all types
 */

import { HoloLoomClient } from './client.js';
import type { ClientConfig } from './types/index.js';

// Export client class
export { HoloLoomClient };
export default HoloLoomClient;

// Re-export all types
export * from './types/index.js';

// Convenience factory function
export function createHoloLoomClient(config?: ClientConfig): HoloLoomClient {
  return new HoloLoomClient(config);
}
