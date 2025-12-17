/**
 * @hololoom/api-client
 *
 * TypeScript API client for HoloLoom backend.
 *
 * @example
 * ```tsx
 * import { HoloLoomProvider, useQuery, useConversation } from '@hololoom/api-client';
 *
 * // Wrap your app with the provider
 * function App() {
 *   return (
 *     <HoloLoomProvider config={{ baseUrl: 'http://localhost:8000' }}>
 *       <Chat />
 *     </HoloLoomProvider>
 *   );
 * }
 *
 * // Use hooks in your components
 * function Chat() {
 *   const { messages, sendMessage, isLoading } = useConversation();
 *
 *   return (
 *     <div>
 *       {messages.map((msg) => (
 *         <Message key={msg.id} {...msg} />
 *       ))}
 *       <Input onSubmit={(text) => sendMessage(text, { mode: 'verify' })} />
 *     </div>
 *   );
 * }
 * ```
 */

// Client
export {
  HoloLoomClient,
  createHoloLoomClient,
  HoloLoomApiError,
  HoloLoomRateLimitError,
  HoloLoomConnectionError,
} from './client';

// React Hooks
export {
  HoloLoomProvider,
  useHoloLoom,
  useQuery,
  useRecall,
  useMemoryGraph,
  useStats,
  useProgressSubscription,
  useConversation,
  type HoloLoomProviderProps,
} from './hooks';

// Types
export type {
  // Core types
  ReasoningMode,
  SafetyLevel,
  ThreadStatus,
  MemoryState,
  ConfidenceLevel,

  // Query types
  CodeContext,
  QueryRequest,
  QueryResponse,
  VerificationResult,
  ReasoningStep,

  // Memory types
  MemorySource,
  MemoryNode,
  MemoryEdge,
  MemoryGraph,
  ExperienceRequest,
  ExperienceResponse,
  RecallRequest,
  RecallResponse,

  // Safety types
  SafetyAssessment,
  AuditEntry,

  // Workflow types
  WorkflowNode,
  WorkflowConnection,
  WorkflowDefinition,
  WorkflowExecuteRequest,
  WorkflowStatus,

  // Stats types
  HealthStatus,
  SystemStats,

  // WebSocket types
  WSMessageType,
  WSMessage,
  ProgressEvent,

  // Config types
  HoloLoomClientConfig,

  // Error types
  ApiError,
  RateLimitError,
} from './types';
