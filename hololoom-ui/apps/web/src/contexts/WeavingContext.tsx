'use client';

import {
  createContext,
  useContext,
  useReducer,
  type ReactNode,
  type Dispatch,
} from 'react';

// --- State ---

export interface WeavingState {
  activeJobId: string | null;
  activeQuery: string | null;
  currentStage: number;
  currentStageName: string;
  stageProgress: number;
  stageDurations: Record<number, number>;
  selectedNodeId: string | null;
  highlightedNodeIds: Set<string>;
  pendingReference: { nodeId: string; content: string } | null;
  confidence: number | null;
  toolUsed: string | null;
  latencyMs: number | null;
  activatedMemoryIds: string[];
  activeRefinementId: string | null;
  degradedEndpoints: Map<string, string>;
  panes: { memory: boolean; chat: boolean; metrics: boolean };
}

const initialState: WeavingState = {
  activeJobId: null,
  activeQuery: null,
  currentStage: 0,
  currentStageName: '',
  stageProgress: 0,
  stageDurations: {},
  selectedNodeId: null,
  highlightedNodeIds: new Set(),
  pendingReference: null,
  confidence: null,
  toolUsed: null,
  latencyMs: null,
  activatedMemoryIds: [],
  activeRefinementId: null,
  degradedEndpoints: new Map(),
  panes: { memory: true, chat: true, metrics: true },
};

// --- Actions ---

export type WeavingAction =
  // Core weaving lifecycle
  | { type: 'QUERY_START'; jobId: string; query: string }
  | { type: 'STAGE_UPDATE'; stage: number; stageName: string; progress: number }
  | { type: 'STAGE_COMPLETE'; stage: number; durationMs: number }
  | {
      type: 'QUERY_COMPLETE';
      confidence: number;
      toolUsed: string;
      latencyMs: number;
      memoryIds: string[];
    }
  // Cross-pane signals
  | { type: 'SELECT_NODE'; nodeId: string | null }
  | { type: 'HIGHLIGHT_NODES'; nodeIds: string[] }
  | { type: 'REFERENCE_NODE'; nodeId: string; content: string }
  | { type: 'CLEAR_REFERENCE' }
  | { type: 'TOGGLE_PANE'; pane: 'memory' | 'chat' | 'metrics' }
  // Refinement signals (chat → awareness)
  | { type: 'REFINEMENT_START'; refinementId: string }
  | { type: 'REFINEMENT_COMPLETE'; refinedResponse: string; confidence: number }
  // Data source degradation signals
  | { type: 'DATA_SOURCE_CHANGED'; endpoint: string; tier: string }
  | { type: 'RESET' };

// --- Reducer ---

function weavingReducer(state: WeavingState, action: WeavingAction): WeavingState {
  switch (action.type) {
    case 'QUERY_START':
      return {
        ...initialState,
        panes: state.panes,
        activeJobId: action.jobId,
        activeQuery: action.query,
      };

    case 'STAGE_UPDATE':
      return {
        ...state,
        currentStage: action.stage,
        currentStageName: action.stageName,
        stageProgress: action.progress,
      };

    case 'STAGE_COMPLETE':
      return {
        ...state,
        stageDurations: {
          ...state.stageDurations,
          [action.stage]: action.durationMs,
        },
      };

    case 'QUERY_COMPLETE':
      return {
        ...state,
        confidence: action.confidence,
        toolUsed: action.toolUsed,
        latencyMs: action.latencyMs,
        activatedMemoryIds: action.memoryIds,
        highlightedNodeIds: new Set(action.memoryIds),
        activeJobId: null,
        currentStage: 0,
        stageProgress: 0,
      };

    case 'SELECT_NODE':
      return { ...state, selectedNodeId: action.nodeId };

    case 'HIGHLIGHT_NODES':
      return { ...state, highlightedNodeIds: new Set(action.nodeIds) };

    case 'REFERENCE_NODE':
      return {
        ...state,
        pendingReference: { nodeId: action.nodeId, content: action.content },
      };

    case 'CLEAR_REFERENCE':
      return { ...state, pendingReference: null };

    case 'TOGGLE_PANE':
      return {
        ...state,
        panes: { ...state.panes, [action.pane]: !state.panes[action.pane] },
      };

    case 'REFINEMENT_START':
      return { ...state, activeRefinementId: action.refinementId };

    case 'REFINEMENT_COMPLETE':
      return {
        ...state,
        activeRefinementId: null,
        confidence: action.confidence,
      };

    case 'DATA_SOURCE_CHANGED': {
      const endpoints = new Map(state.degradedEndpoints);
      if (action.tier === 'live') {
        endpoints.delete(action.endpoint);
      } else {
        endpoints.set(action.endpoint, action.tier);
      }
      return { ...state, degradedEndpoints: endpoints };
    }

    case 'RESET':
      return { ...initialState, panes: state.panes };

    default:
      return state;
  }
}

// --- Context ---

const WeavingStateContext = createContext<WeavingState | null>(null);
const WeavingDispatchContext = createContext<Dispatch<WeavingAction> | null>(null);

export function WeavingProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(weavingReducer, initialState);

  return (
    <WeavingStateContext.Provider value={state}>
      <WeavingDispatchContext.Provider value={dispatch}>
        {children}
      </WeavingDispatchContext.Provider>
    </WeavingStateContext.Provider>
  );
}

export function useWeaving(): WeavingState {
  const ctx = useContext(WeavingStateContext);
  if (!ctx) throw new Error('useWeaving must be used within WeavingProvider');
  return ctx;
}

export function useWeavingDispatch(): Dispatch<WeavingAction> {
  const ctx = useContext(WeavingDispatchContext);
  if (!ctx) throw new Error('useWeavingDispatch must be used within WeavingProvider');
  return ctx;
}
