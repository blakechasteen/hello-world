/**
 * DetailPanel Components
 * Exports for HoloLoom Agent Manager UI Phase 4 - Detail Panel & File Tree
 */

// Main Components
export { DetailPanel, default } from './DetailPanel';
export { default as StepHistory } from './StepHistory';
export { default as MemoryNodes } from './MemoryNodes';
export { default as FileTreeViewer } from './FileTreeViewer';
export { InjectionHistoryLog } from './InjectionHistoryLog';

// Types
export type {
  TaskNode,
  StepType,
  StepStatus,
  MemoryNode,
  MemorySourceType,
  FileNode,
  FileStatus,
  DetailTab,
  StepSortOption,
  MemorySortOption,
  StepFilterOption,
} from './types';

// Constants
export {
  MEMORY_SOURCE_CONFIG,
  FILE_STATUS_CONFIG,
  AGENT_COLORS,
  DETAIL_TABS,
  getAgentColor,
} from './types';
