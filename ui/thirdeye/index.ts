/**
 * ThirdEye - Chat-integrated 3D Concept Visualization
 * =====================================================
 *
 * TypeScript client for the ThirdEye visualization system.
 * Connects to the Python backend and renders concepts using Three.js.
 *
 * Usage:
 *   import { createThoughtspace } from './thirdeye';
 *
 *   const container = document.getElementById('thirdeye-container');
 *   const thoughtspace = createThoughtspace({ container });
 *   await thoughtspace.start();
 *
 * Created: 2025-11-30
 * Author: HoloLoom Team
 */

// Chat bridge
export {
  ChatBridge,
  getChatBridge,
  type Concept,
  type Transition,
  type ConceptExtraction,
  type RenderCommand,
  type RenderFrame,
  type ConceptCallback,
  type FrameCallback,
  type CommandCallback,
  type ErrorCallback,
  type ConnectionCallback,
} from './ChatBridge';

// Transition engine
export {
  TransitionEngine,
  type ConceptNode,
} from './TransitionEngine';

// Thoughtspace
export {
  Thoughtspace,
  createThoughtspace,
  type ThoughtspaceMode,
  type ThoughtspaceConfig,
} from './Thoughtspace';
