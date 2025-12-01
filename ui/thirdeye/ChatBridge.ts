/**
 * ChatBridge - Connect to chat WebSocket and receive concept updates
 *
 * Listens to the ThirdEye Python server for concept extractions
 * and transitions, forwarding them to the visualization engine.
 *
 * Created: 2025-11-30
 * Author: HoloLoom Team
 */

// Types
export interface Concept {
  id: string;
  label: string;
  type: string;
  position: { x: number; y: number; z: number };
  importance: number;
  confidence: number;
  style: string;
  connections: string[];
  parent_id: string | null;
  description: string | null;
  metadata: Record<string, unknown>;
  created_at: number;
  last_active: number;
}

export interface Transition {
  type: string;
  duration_ms: number;
  easing: string;
  delay_ms: number;
  stagger_ms: number;
  from_concept_id: string | null;
  to_concept_id: string;
  semantic_distance: number;
  triggered_by: string;
  visual: {
    fade_overlap: number;
    scale_factor: number;
    blur_peak: number;
  };
}

export interface ConceptExtraction {
  concepts: Concept[];
  transition: Transition | null;
  primary_concept: Concept | null;
}

export interface RenderCommand {
  type: string;
  target: string | null;
  data: Record<string, unknown>;
}

export interface RenderFrame {
  frame_id: number;
  commands: RenderCommand[];
  metadata: Record<string, unknown>;
}

// Event callbacks
export type ConceptCallback = (extraction: ConceptExtraction) => void;
export type FrameCallback = (frame: RenderFrame) => void;
export type CommandCallback = (command: RenderCommand) => void;
export type ErrorCallback = (error: Error) => void;
export type ConnectionCallback = (connected: boolean) => void;

/**
 * ChatBridge connects to the ThirdEye Python server
 * and receives concept visualizations.
 */
export class ChatBridge {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectInterval: number = 3000;
  private maxReconnectAttempts: number = 5;
  private reconnectAttempts: number = 0;
  private autoReconnect: boolean = true;

  // Callbacks
  private onConcept: ConceptCallback | null = null;
  private onFrame: FrameCallback | null = null;
  private onCommand: CommandCallback | null = null;
  private onError: ErrorCallback | null = null;
  private onConnection: ConnectionCallback | null = null;

  constructor(serverUrl: string = 'ws://localhost:8003/ws/thirdeye') {
    this.url = serverUrl;
  }

  /**
   * Connect to the ThirdEye server
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          console.log('[ChatBridge] Connected to ThirdEye server');
          this.reconnectAttempts = 0;
          this.onConnection?.(true);
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event.data);
        };

        this.ws.onerror = (error) => {
          console.error('[ChatBridge] WebSocket error:', error);
          this.onError?.(new Error('WebSocket error'));
        };

        this.ws.onclose = () => {
          console.log('[ChatBridge] Connection closed');
          this.onConnection?.(false);
          this.handleReconnect();
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Disconnect from the server
   */
  disconnect(): void {
    this.autoReconnect = false;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Handle incoming messages
   */
  private handleMessage(data: string): void {
    try {
      const message = JSON.parse(data);

      switch (message.type) {
        case 'concept_extraction':
          this.onConcept?.(message.data as ConceptExtraction);
          break;

        case 'render_frame':
          this.onFrame?.(message.data as RenderFrame);
          break;

        case 'render_command':
          this.onCommand?.(message.data as RenderCommand);
          break;

        case 'error':
          this.onError?.(new Error(message.message));
          break;

        default:
          console.log('[ChatBridge] Unknown message type:', message.type);
      }
    } catch (error) {
      console.error('[ChatBridge] Failed to parse message:', error);
    }
  }

  /**
   * Handle reconnection attempts
   */
  private handleReconnect(): void {
    if (!this.autoReconnect) return;
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[ChatBridge] Max reconnect attempts reached');
      this.onError?.(new Error('Max reconnect attempts reached'));
      return;
    }

    this.reconnectAttempts++;
    console.log(`[ChatBridge] Reconnecting in ${this.reconnectInterval}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      this.connect().catch((error) => {
        console.error('[ChatBridge] Reconnect failed:', error);
      });
    }, this.reconnectInterval);
  }

  /**
   * Send a message to the server
   */
  send(message: object): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('[ChatBridge] Cannot send: WebSocket not connected');
    }
  }

  /**
   * Request focus on a specific concept
   */
  focusConcept(conceptId: string): void {
    this.send({
      type: 'focus_concept',
      concept_id: conceptId,
    });
  }

  /**
   * Toggle Thoughtspace mode
   */
  toggleMode(): void {
    this.send({ type: 'toggle_mode' });
  }

  /**
   * Request camera orbit
   */
  orbit(angleDelta: number): void {
    this.send({
      type: 'orbit',
      angle_delta: angleDelta,
    });
  }

  /**
   * Request camera zoom
   */
  zoom(factor: number): void {
    this.send({
      type: 'zoom',
      factor: factor,
    });
  }

  // Event listeners

  onConceptExtraction(callback: ConceptCallback): void {
    this.onConcept = callback;
  }

  onRenderFrame(callback: FrameCallback): void {
    this.onFrame = callback;
  }

  onRenderCommand(callback: CommandCallback): void {
    this.onCommand = callback;
  }

  onErrorEvent(callback: ErrorCallback): void {
    this.onError = callback;
  }

  onConnectionChange(callback: ConnectionCallback): void {
    this.onConnection = callback;
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
let bridgeInstance: ChatBridge | null = null;

/**
 * Get or create the ChatBridge singleton
 */
export function getChatBridge(serverUrl?: string): ChatBridge {
  if (!bridgeInstance) {
    bridgeInstance = new ChatBridge(serverUrl);
  }
  return bridgeInstance;
}
