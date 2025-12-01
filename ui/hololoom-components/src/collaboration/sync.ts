/**
 * Y.js CRDT Sync Module - Phase 3: Collaborative Intelligence
 *
 * Provides conflict-free replicated data type (CRDT) synchronization
 * for real-time collaborative editing of knowledge graphs.
 *
 * Uses Y.js for:
 * - Automatic conflict resolution
 * - Offline-first support
 * - Efficient delta synchronization
 *
 * @module collaboration/sync
 */

// Note: Y.js types are provided when y-websocket is installed
// For now, we use interface definitions compatible with Y.js

// ============================================================================
// Types
// ============================================================================

/**
 * Y.js document interface (compatible with y-js)
 */
export interface YDoc {
  getMap: <T>(name: string) => YMap<T>;
  getArray: <T>(name: string) => YArray<T>;
  getText: (name: string) => YText;
  on: (event: string, handler: (...args: unknown[]) => void) => void;
  off: (event: string, handler: (...args: unknown[]) => void) => void;
  transact: (fn: () => void) => void;
  destroy: () => void;
  clientID: number;
}

export interface YMap<T> {
  get: (key: string) => T | undefined;
  set: (key: string, value: T) => void;
  delete: (key: string) => void;
  has: (key: string) => boolean;
  keys: () => IterableIterator<string>;
  values: () => IterableIterator<T>;
  entries: () => IterableIterator<[string, T]>;
  forEach: (fn: (value: T, key: string) => void) => void;
  toJSON: () => Record<string, T>;
  observe: (fn: (event: YMapEvent<T>) => void) => void;
  unobserve: (fn: (event: YMapEvent<T>) => void) => void;
  size: number;
}

export interface YArray<T> {
  get: (index: number) => T | undefined;
  push: (items: T[]) => void;
  insert: (index: number, items: T[]) => void;
  delete: (index: number, length?: number) => void;
  toArray: () => T[];
  toJSON: () => T[];
  length: number;
  observe: (fn: (event: YArrayEvent<T>) => void) => void;
  unobserve: (fn: (event: YArrayEvent<T>) => void) => void;
}

export interface YText {
  insert: (index: number, text: string) => void;
  delete: (index: number, length: number) => void;
  toString: () => string;
  toJSON: () => string;
  length: number;
  observe: (fn: (event: YTextEvent) => void) => void;
  unobserve: (fn: (event: YTextEvent) => void) => void;
}

export interface YMapEvent<T> {
  target: YMap<T>;
  keysChanged: Set<string>;
  changes: {
    keys: Map<string, { action: 'add' | 'update' | 'delete'; oldValue?: T }>;
  };
}

export interface YArrayEvent<T> {
  target: YArray<T>;
  changes: {
    added: Set<{ insert: T[] }>;
    deleted: Set<{ delete: number }>;
  };
}

export interface YTextEvent {
  target: YText;
  delta: Array<{ insert?: string; delete?: number; retain?: number }>;
}

/**
 * WebSocket provider interface
 */
export interface WebSocketProvider {
  connect: () => void;
  disconnect: () => void;
  destroy: () => void;
  connected: boolean;
  synced: boolean;
  on: (event: string, handler: (...args: unknown[]) => void) => void;
  off: (event: string, handler: (...args: unknown[]) => void) => void;
  awareness: AwarenessProtocol;
}

/**
 * Awareness protocol for presence
 */
export interface AwarenessProtocol {
  setLocalState: (state: Record<string, unknown> | null) => void;
  getLocalState: () => Record<string, unknown> | null;
  getStates: () => Map<number, Record<string, unknown>>;
  on: (event: string, handler: (...args: unknown[]) => void) => void;
  off: (event: string, handler: (...args: unknown[]) => void) => void;
  clientID: number;
}

/**
 * Sync configuration
 */
export interface SyncConfig {
  /** WebSocket server URL */
  serverUrl: string;
  /** Room/session name */
  roomName: string;
  /** Authentication token */
  authToken?: string;
  /** Reconnection settings */
  reconnect?: {
    enabled: boolean;
    maxAttempts: number;
    delayMs: number;
    maxDelayMs: number;
  };
  /** Awareness broadcast interval (ms) */
  awarenessUpdateInterval?: number;
}

/**
 * Shared state structure stored in Y.js doc
 */
export interface SharedGraphState {
  nodes: Record<string, SharedNode>;
  edges: Record<string, SharedEdge>;
  annotations: Record<string, SharedAnnotation>;
  viewport: SharedViewport;
}

export interface SharedNode {
  id: string;
  label: string;
  type: string;
  x: number;
  y: number;
  importance: number;
  metadata?: Record<string, unknown>;
}

export interface SharedEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  weight: number;
}

export interface SharedAnnotation {
  id: string;
  targetId: string;
  targetType: 'node' | 'edge';
  content: string;
  authorId: string;
  authorName: string;
  type: 'note' | 'question' | 'insight' | 'warning';
  resolved: boolean;
  createdAt: number;
  updatedAt: number;
}

export interface SharedViewport {
  zoom: number;
  panX: number;
  panY: number;
  followUserId?: string;
}

/**
 * User awareness state
 */
export interface UserAwareness {
  userId: string;
  name: string;
  color: string;
  cursor?: { x: number; y: number };
  focus?: { type: string; targetId?: string };
  selection?: string[];
}

// ============================================================================
// Sync Manager Class
// ============================================================================

/**
 * Event types emitted by SyncManager
 */
export type SyncEvent =
  | { type: 'connected' }
  | { type: 'disconnected' }
  | { type: 'synced' }
  | { type: 'error'; error: Error }
  | { type: 'nodesChanged'; changes: NodeChange[] }
  | { type: 'edgesChanged'; changes: EdgeChange[] }
  | { type: 'annotationsChanged'; changes: AnnotationChange[] }
  | { type: 'viewportChanged'; viewport: SharedViewport }
  | { type: 'awarenessChanged'; states: Map<number, UserAwareness> };

export interface NodeChange {
  action: 'add' | 'update' | 'delete';
  nodeId: string;
  node?: SharedNode;
}

export interface EdgeChange {
  action: 'add' | 'update' | 'delete';
  edgeId: string;
  edge?: SharedEdge;
}

export interface AnnotationChange {
  action: 'add' | 'update' | 'delete';
  annotationId: string;
  annotation?: SharedAnnotation;
}

type SyncEventHandler = (event: SyncEvent) => void;

/**
 * CRDT Sync Manager
 *
 * Manages Y.js document synchronization with the collaboration server.
 * Provides a clean API for reading and writing shared state.
 */
export class CRDTSyncManager {
  private doc: YDoc | null = null;
  private provider: WebSocketProvider | null = null;
  private config: SyncConfig;
  private handlers: Set<SyncEventHandler> = new Set();
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  // Cached Y.js shared types
  private nodesMap: YMap<SharedNode> | null = null;
  private edgesMap: YMap<SharedEdge> | null = null;
  private annotationsMap: YMap<SharedAnnotation> | null = null;
  private viewportMap: YMap<unknown> | null = null;

  constructor(config: SyncConfig) {
    this.config = {
      ...config,
      reconnect: config.reconnect ?? {
        enabled: true,
        maxAttempts: 10,
        delayMs: 1000,
        maxDelayMs: 30000,
      },
      awarenessUpdateInterval: config.awarenessUpdateInterval ?? 100,
    };
  }

  /**
   * Connect to the sync server
   */
  async connect(): Promise<void> {
    // In a real implementation, we'd import Y.js and y-websocket
    // For now, we provide a mock implementation that can be swapped
    try {
      // Create Y.js document
      this.doc = this.createYDoc();

      // Get shared types
      this.nodesMap = this.doc.getMap<SharedNode>('nodes');
      this.edgesMap = this.doc.getMap<SharedEdge>('edges');
      this.annotationsMap = this.doc.getMap<SharedAnnotation>('annotations');
      this.viewportMap = this.doc.getMap<unknown>('viewport');

      // Set up observers
      this.setupObservers();

      // Create WebSocket provider
      this.provider = this.createWebSocketProvider(this.doc);

      // Set up provider event handlers
      this.setupProviderHandlers();

      // Connect
      this.provider.connect();

      this.emit({ type: 'connected' });
    } catch (error) {
      this.emit({ type: 'error', error: error as Error });
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from the sync server
   */
  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.provider) {
      this.provider.disconnect();
      this.provider.destroy();
      this.provider = null;
    }

    if (this.doc) {
      this.doc.destroy();
      this.doc = null;
    }

    this.nodesMap = null;
    this.edgesMap = null;
    this.annotationsMap = null;
    this.viewportMap = null;

    this.emit({ type: 'disconnected' });
  }

  /**
   * Check if connected
   */
  get isConnected(): boolean {
    return this.provider?.connected ?? false;
  }

  /**
   * Check if synced with server
   */
  get isSynced(): boolean {
    return this.provider?.synced ?? false;
  }

  /**
   * Get client ID
   */
  get clientId(): number {
    return this.doc?.clientID ?? 0;
  }

  // ============================================================================
  // Node Operations
  // ============================================================================

  /**
   * Get all nodes
   */
  getNodes(): SharedNode[] {
    if (!this.nodesMap) return [];
    return Array.from(this.nodesMap.values());
  }

  /**
   * Get node by ID
   */
  getNode(nodeId: string): SharedNode | undefined {
    return this.nodesMap?.get(nodeId);
  }

  /**
   * Add or update a node
   */
  setNode(node: SharedNode): void {
    if (!this.nodesMap || !this.doc) return;

    this.doc.transact(() => {
      this.nodesMap!.set(node.id, node);
    });
  }

  /**
   * Delete a node
   */
  deleteNode(nodeId: string): void {
    if (!this.nodesMap || !this.doc) return;

    this.doc.transact(() => {
      this.nodesMap!.delete(nodeId);
    });
  }

  /**
   * Update node position (optimized for frequent updates)
   */
  updateNodePosition(nodeId: string, x: number, y: number): void {
    if (!this.nodesMap || !this.doc) return;

    const node = this.nodesMap.get(nodeId);
    if (!node) return;

    this.doc.transact(() => {
      this.nodesMap!.set(nodeId, { ...node, x, y });
    });
  }

  // ============================================================================
  // Edge Operations
  // ============================================================================

  /**
   * Get all edges
   */
  getEdges(): SharedEdge[] {
    if (!this.edgesMap) return [];
    return Array.from(this.edgesMap.values());
  }

  /**
   * Get edge by ID
   */
  getEdge(edgeId: string): SharedEdge | undefined {
    return this.edgesMap?.get(edgeId);
  }

  /**
   * Add or update an edge
   */
  setEdge(edge: SharedEdge): void {
    if (!this.edgesMap || !this.doc) return;

    this.doc.transact(() => {
      this.edgesMap!.set(edge.id, edge);
    });
  }

  /**
   * Delete an edge
   */
  deleteEdge(edgeId: string): void {
    if (!this.edgesMap || !this.doc) return;

    this.doc.transact(() => {
      this.edgesMap!.delete(edgeId);
    });
  }

  // ============================================================================
  // Annotation Operations
  // ============================================================================

  /**
   * Get all annotations
   */
  getAnnotations(): SharedAnnotation[] {
    if (!this.annotationsMap) return [];
    return Array.from(this.annotationsMap.values());
  }

  /**
   * Get annotations for a target
   */
  getAnnotationsForTarget(targetId: string): SharedAnnotation[] {
    return this.getAnnotations().filter((a) => a.targetId === targetId);
  }

  /**
   * Add or update an annotation
   */
  setAnnotation(annotation: SharedAnnotation): void {
    if (!this.annotationsMap || !this.doc) return;

    this.doc.transact(() => {
      this.annotationsMap!.set(annotation.id, annotation);
    });
  }

  /**
   * Delete an annotation
   */
  deleteAnnotation(annotationId: string): void {
    if (!this.annotationsMap || !this.doc) return;

    this.doc.transact(() => {
      this.annotationsMap!.delete(annotationId);
    });
  }

  // ============================================================================
  // Viewport Operations
  // ============================================================================

  /**
   * Get shared viewport
   */
  getViewport(): SharedViewport | null {
    if (!this.viewportMap) return null;

    return {
      zoom: (this.viewportMap.get('zoom') as number) ?? 1,
      panX: (this.viewportMap.get('panX') as number) ?? 0,
      panY: (this.viewportMap.get('panY') as number) ?? 0,
      followUserId: this.viewportMap.get('followUserId') as string | undefined,
    };
  }

  /**
   * Update shared viewport
   */
  setViewport(viewport: Partial<SharedViewport>): void {
    if (!this.viewportMap || !this.doc) return;

    this.doc.transact(() => {
      if (viewport.zoom !== undefined) this.viewportMap!.set('zoom', viewport.zoom);
      if (viewport.panX !== undefined) this.viewportMap!.set('panX', viewport.panX);
      if (viewport.panY !== undefined) this.viewportMap!.set('panY', viewport.panY);
      if (viewport.followUserId !== undefined)
        this.viewportMap!.set('followUserId', viewport.followUserId);
    });
  }

  // ============================================================================
  // Awareness (Presence)
  // ============================================================================

  /**
   * Set local user awareness state
   */
  setLocalAwareness(state: UserAwareness): void {
    this.provider?.awareness.setLocalState(state as Record<string, unknown>);
  }

  /**
   * Update local cursor position
   */
  setLocalCursor(x: number, y: number): void {
    const currentState = this.provider?.awareness.getLocalState() as UserAwareness | null;
    if (!currentState) return;

    this.provider?.awareness.setLocalState({
      ...currentState,
      cursor: { x, y },
    });
  }

  /**
   * Update local focus
   */
  setLocalFocus(focus: { type: string; targetId?: string }): void {
    const currentState = this.provider?.awareness.getLocalState() as UserAwareness | null;
    if (!currentState) return;

    this.provider?.awareness.setLocalState({
      ...currentState,
      focus,
    });
  }

  /**
   * Get all awareness states
   */
  getAwarenessStates(): Map<number, UserAwareness> {
    const states = this.provider?.awareness.getStates();
    if (!states) return new Map();

    const result = new Map<number, UserAwareness>();
    states.forEach((state, clientId) => {
      result.set(clientId, state as UserAwareness);
    });

    return result;
  }

  /**
   * Clear local awareness (e.g., when leaving)
   */
  clearLocalAwareness(): void {
    this.provider?.awareness.setLocalState(null);
  }

  // ============================================================================
  // Event Subscription
  // ============================================================================

  /**
   * Subscribe to sync events
   */
  subscribe(handler: SyncEventHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  /**
   * Emit event to all handlers
   */
  private emit(event: SyncEvent): void {
    this.handlers.forEach((handler) => {
      try {
        handler(event);
      } catch (error) {
        console.error('Sync event handler error:', error);
      }
    });
  }

  // ============================================================================
  // Internal Methods
  // ============================================================================

  /**
   * Create Y.js document (mock implementation)
   */
  private createYDoc(): YDoc {
    // This would be replaced with actual Y.js:
    // import * as Y from 'yjs';
    // return new Y.Doc();

    // Mock implementation for structure
    const maps: Map<string, MockYMap> = new Map();

    const doc: YDoc = {
      clientID: Math.floor(Math.random() * 1000000),
      getMap: <T>(name: string) => {
        if (!maps.has(name)) {
          maps.set(name, new MockYMap<T>());
        }
        return maps.get(name) as unknown as YMap<T>;
      },
      getArray: <T>(_name: string) => new MockYArray<T>() as unknown as YArray<T>,
      getText: (_name: string) => new MockYText() as unknown as YText,
      on: () => {},
      off: () => {},
      transact: (fn) => fn(),
      destroy: () => {},
    };

    return doc;
  }

  /**
   * Create WebSocket provider (mock implementation)
   */
  private createWebSocketProvider(_doc: YDoc): WebSocketProvider {
    // This would be replaced with actual y-websocket:
    // import { WebsocketProvider } from 'y-websocket';
    // return new WebsocketProvider(this.config.serverUrl, this.config.roomName, doc);

    // Mock implementation
    const awareness: AwarenessProtocol = {
      clientID: _doc.clientID,
      setLocalState: () => {},
      getLocalState: () => ({}),
      getStates: () => new Map(),
      on: () => {},
      off: () => {},
    };

    return {
      connected: false,
      synced: false,
      connect: () => {
        // Simulate connection
        setTimeout(() => {
          (this.provider as { connected: boolean }).connected = true;
          (this.provider as { synced: boolean }).synced = true;
          this.emit({ type: 'synced' });
        }, 100);
      },
      disconnect: () => {
        (this.provider as { connected: boolean }).connected = false;
        (this.provider as { synced: boolean }).synced = false;
      },
      destroy: () => {},
      on: () => {},
      off: () => {},
      awareness,
    };
  }

  /**
   * Set up Y.js observers
   */
  private setupObservers(): void {
    if (!this.nodesMap || !this.edgesMap || !this.annotationsMap || !this.viewportMap) {
      return;
    }

    // Node changes
    this.nodesMap.observe((event) => {
      const changes: NodeChange[] = [];
      event.changes.keys.forEach((change, key) => {
        changes.push({
          action: change.action,
          nodeId: key,
          node: change.action !== 'delete' ? this.nodesMap?.get(key) : undefined,
        });
      });
      if (changes.length > 0) {
        this.emit({ type: 'nodesChanged', changes });
      }
    });

    // Edge changes
    this.edgesMap.observe((event) => {
      const changes: EdgeChange[] = [];
      event.changes.keys.forEach((change, key) => {
        changes.push({
          action: change.action,
          edgeId: key,
          edge: change.action !== 'delete' ? this.edgesMap?.get(key) : undefined,
        });
      });
      if (changes.length > 0) {
        this.emit({ type: 'edgesChanged', changes });
      }
    });

    // Annotation changes
    this.annotationsMap.observe((event) => {
      const changes: AnnotationChange[] = [];
      event.changes.keys.forEach((change, key) => {
        changes.push({
          action: change.action,
          annotationId: key,
          annotation:
            change.action !== 'delete' ? this.annotationsMap?.get(key) : undefined,
        });
      });
      if (changes.length > 0) {
        this.emit({ type: 'annotationsChanged', changes });
      }
    });

    // Viewport changes
    this.viewportMap.observe(() => {
      const viewport = this.getViewport();
      if (viewport) {
        this.emit({ type: 'viewportChanged', viewport });
      }
    });
  }

  /**
   * Set up provider event handlers
   */
  private setupProviderHandlers(): void {
    if (!this.provider) return;

    this.provider.on('status', (status: { status: string }) => {
      if (status.status === 'connected') {
        this.reconnectAttempts = 0;
        this.emit({ type: 'connected' });
      } else if (status.status === 'disconnected') {
        this.emit({ type: 'disconnected' });
        this.scheduleReconnect();
      }
    });

    this.provider.on('sync', (synced: boolean) => {
      if (synced) {
        this.emit({ type: 'synced' });
      }
    });

    // Awareness changes
    this.provider.awareness.on('change', () => {
      const states = this.getAwarenessStates();
      this.emit({ type: 'awarenessChanged', states });
    });
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    const reconnectConfig = this.config.reconnect;
    if (!reconnectConfig?.enabled) return;

    if (this.reconnectAttempts >= reconnectConfig.maxAttempts) {
      this.emit({
        type: 'error',
        error: new Error('Max reconnection attempts reached'),
      });
      return;
    }

    const delay = Math.min(
      reconnectConfig.delayMs * Math.pow(2, this.reconnectAttempts),
      reconnectConfig.maxDelayMs
    );

    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++;
      this.connect();
    }, delay);
  }
}

// ============================================================================
// Mock Implementations (for testing/development)
// ============================================================================

class MockYMap<T> {
  private data = new Map<string, T>();
  private observers: Array<(event: YMapEvent<T>) => void> = [];

  get(key: string): T | undefined {
    return this.data.get(key);
  }

  set(key: string, value: T): void {
    const action = this.data.has(key) ? 'update' : 'add';
    const oldValue = this.data.get(key);
    this.data.set(key, value);

    this.notifyObservers({
      target: this as unknown as YMap<T>,
      keysChanged: new Set([key]),
      changes: {
        keys: new Map([[key, { action, oldValue }]]),
      },
    });
  }

  delete(key: string): void {
    const oldValue = this.data.get(key);
    this.data.delete(key);

    this.notifyObservers({
      target: this as unknown as YMap<T>,
      keysChanged: new Set([key]),
      changes: {
        keys: new Map([[key, { action: 'delete', oldValue }]]),
      },
    });
  }

  has(key: string): boolean {
    return this.data.has(key);
  }

  keys(): IterableIterator<string> {
    return this.data.keys();
  }

  values(): IterableIterator<T> {
    return this.data.values();
  }

  entries(): IterableIterator<[string, T]> {
    return this.data.entries();
  }

  forEach(fn: (value: T, key: string) => void): void {
    this.data.forEach(fn);
  }

  toJSON(): Record<string, T> {
    return Object.fromEntries(this.data);
  }

  observe(fn: (event: YMapEvent<T>) => void): void {
    this.observers.push(fn);
  }

  unobserve(fn: (event: YMapEvent<T>) => void): void {
    const idx = this.observers.indexOf(fn);
    if (idx !== -1) this.observers.splice(idx, 1);
  }

  get size(): number {
    return this.data.size;
  }

  private notifyObservers(event: YMapEvent<T>): void {
    this.observers.forEach((fn) => fn(event));
  }
}

class MockYArray<T> {
  private data: T[] = [];

  get(index: number): T | undefined {
    return this.data[index];
  }

  push(items: T[]): void {
    this.data.push(...items);
  }

  insert(index: number, items: T[]): void {
    this.data.splice(index, 0, ...items);
  }

  delete(index: number, length = 1): void {
    this.data.splice(index, length);
  }

  toArray(): T[] {
    return [...this.data];
  }

  toJSON(): T[] {
    return this.toArray();
  }

  get length(): number {
    return this.data.length;
  }

  observe(): void {}
  unobserve(): void {}
}

class MockYText {
  private text = '';

  insert(index: number, str: string): void {
    this.text = this.text.slice(0, index) + str + this.text.slice(index);
  }

  delete(index: number, length: number): void {
    this.text = this.text.slice(0, index) + this.text.slice(index + length);
  }

  toString(): string {
    return this.text;
  }

  toJSON(): string {
    return this.text;
  }

  get length(): number {
    return this.text.length;
  }

  observe(): void {}
  unobserve(): void {}
}

// ============================================================================
// Exports
// ============================================================================

export { MockYMap, MockYArray, MockYText };
