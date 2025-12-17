/**
 * Utility exports
 */

export {
  type HoloLoomConfig,
  getConfig,
  updateConfig,
  onConfigChange,
} from './config';

export {
  type EditorContext,
  getEditorContext,
  getSelectedText,
  getWordAtCursor,
  getCurrentLine,
  formatContextForPrompt,
  getContextSummary,
} from './context';
