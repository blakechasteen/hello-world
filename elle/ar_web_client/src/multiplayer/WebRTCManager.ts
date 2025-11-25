/**
 * WebRTCManager - Multi-User Avatar Synchronization
 *
 * Handles WebRTC peer-to-peer connections for real-time avatar state sync.
 * Supports 2-4 simultaneous users with <50ms latency target.
 *
 * Features:
 * - WebRTC data channels for avatar pose/position sync
 * - Signaling server integration (WebSocket)
 * - Connection state management
 * - Automatic reconnection
 * - Latency optimization
 *
 * Architecture:
 * - Mesh topology (each peer connects to all others)
 * - Data channels for pose updates (unreliable, low latency)
 * - Heartbeat for connection monitoring
 *
 * Created: 2025-11-22 (Phase 6.2)
 */

import * as THREE from 'three';
import { BodyPose } from '../services/poseEstimation';

/**
 * Avatar state update message
 */
export interface AvatarStateUpdate {
  userId: string;
  timestamp: number;
  pose: BodyPose | null;
  position: [number, number, number];
  rotation: [number, number, number, number];  // Quaternion
}

/**
 * Peer connection state
 */
export enum PeerConnectionState {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  FAILED = 'failed',
}

/**
 * Peer info
 */
export interface PeerInfo {
  userId: string;
  state: PeerConnectionState;
  latency: number;  // ms
  lastUpdate: number;  // timestamp
}

/**
 * WebRTC configuration
 */
export interface WebRTCConfig {
  /**
   * Signaling server URL (WebSocket)
   */
  signalingServerUrl: string;

  /**
   * STUN/TURN servers for NAT traversal
   */
  iceServers?: RTCIceServer[];

  /**
   * Update rate in Hz (default: 30)
   * Higher = smoother but more bandwidth
   */
  updateRate?: number;

  /**
   * Enable automatic reconnection (default: true)
   */
  enableReconnection?: boolean;

  /**
   * Reconnection timeout in ms (default: 5000)
   */
  reconnectionTimeout?: number;

  /**
   * Heartbeat interval in ms (default: 1000)
   */
  heartbeatInterval?: number;
}

/**
 * Signaling message types
 */
enum SignalingMessageType {
  OFFER = 'offer',
  ANSWER = 'answer',
  ICE_CANDIDATE = 'ice-candidate',
  JOIN = 'join',
  LEAVE = 'leave',
  PEER_LIST = 'peer-list',
}

/**
 * Signaling message
 */
interface SignalingMessage {
  type: SignalingMessageType;
  from?: string;
  to?: string;
  data?: any;
}

/**
 * WebRTCManager - Manages WebRTC connections for multi-user avatars
 *
 * Usage:
 * ```typescript
 * const manager = new WebRTCManager({
 *   signalingServerUrl: 'ws://localhost:8080',
 *   iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
 * });
 *
 * // Join session
 * await manager.connect('my-user-id');
 *
 * // Send avatar state
 * manager.sendAvatarState({
 *   userId: 'my-user-id',
 *   timestamp: Date.now(),
 *   pose: currentPose,
 *   position: [0, 0, 0],
 *   rotation: [0, 0, 0, 1],
 * });
 *
 * // Receive updates from other users
 * manager.on('avatar-update', (update: AvatarStateUpdate) => {
 *   console.log('Update from', update.userId);
 * });
 * ```
 */
export class WebRTCManager {
  private config: Required<WebRTCConfig>;
  private userId: string | null = null;

  // Signaling
  private signalingSocket: WebSocket | null = null;
  private signalingConnected: boolean = false;

  // Peer connections
  private peers: Map<string, RTCPeerConnection> = new Map();
  private dataChannels: Map<string, RTCDataChannel> = new Map();
  private peerStates: Map<string, PeerInfo> = new Map();

  // Event handlers
  private eventHandlers: Map<string, Function[]> = new Map();

  // Update scheduling
  private updateInterval: number | null = null;
  private heartbeatInterval: number | null = null;

  // State tracking
  private lastState: AvatarStateUpdate | null = null;

  constructor(config: WebRTCConfig) {
    this.config = {
      signalingServerUrl: config.signalingServerUrl,
      iceServers: config.iceServers ?? [
        { urls: 'stun:stun.l.google.com:19302' },
      ],
      updateRate: config.updateRate ?? 30,
      enableReconnection: config.enableReconnection ?? true,
      reconnectionTimeout: config.reconnectionTimeout ?? 5000,
      heartbeatInterval: config.heartbeatInterval ?? 1000,
    };
  }

  /**
   * Connect to signaling server and join session
   */
  async connect(userId: string): Promise<void> {
    this.userId = userId;

    return new Promise((resolve, reject) => {
      try {
        this.signalingSocket = new WebSocket(this.config.signalingServerUrl);

        this.signalingSocket.onopen = () => {
          console.log('[WebRTCManager] Connected to signaling server');
          this.signalingConnected = true;

          // Send join message
          this.sendSignalingMessage({
            type: SignalingMessageType.JOIN,
            from: userId,
          });

          // Start heartbeat
          this.startHeartbeat();

          resolve();
        };

        this.signalingSocket.onmessage = (event) => {
          this.handleSignalingMessage(JSON.parse(event.data));
        };

        this.signalingSocket.onerror = (error) => {
          console.error('[WebRTCManager] Signaling error:', error);
          reject(error);
        };

        this.signalingSocket.onclose = () => {
          console.log('[WebRTCManager] Signaling connection closed');
          this.signalingConnected = false;

          if (this.config.enableReconnection) {
            setTimeout(() => this.reconnect(), this.config.reconnectionTimeout);
          }
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Reconnect to signaling server
   */
  private async reconnect(): Promise<void> {
    if (!this.userId || this.signalingConnected) return;

    console.log('[WebRTCManager] Attempting reconnection...');

    try {
      await this.connect(this.userId);
      console.log('[WebRTCManager] Reconnected successfully');
    } catch (error) {
      console.error('[WebRTCManager] Reconnection failed:', error);

      if (this.config.enableReconnection) {
        setTimeout(() => this.reconnect(), this.config.reconnectionTimeout);
      }
    }
  }

  /**
   * Handle signaling messages
   */
  private async handleSignalingMessage(message: SignalingMessage): Promise<void> {
    switch (message.type) {
      case SignalingMessageType.PEER_LIST:
        // New peer list from server
        const peers: string[] = message.data.peers;
        console.log('[WebRTCManager] Received peer list:', peers);

        // Create connections to all peers
        for (const peerId of peers) {
          if (peerId !== this.userId && !this.peers.has(peerId)) {
            await this.createPeerConnection(peerId, true);
          }
        }
        break;

      case SignalingMessageType.OFFER:
        // Received offer from peer
        await this.handleOffer(message.from!, message.data);
        break;

      case SignalingMessageType.ANSWER:
        // Received answer from peer
        await this.handleAnswer(message.from!, message.data);
        break;

      case SignalingMessageType.ICE_CANDIDATE:
        // Received ICE candidate
        await this.handleIceCandidate(message.from!, message.data);
        break;

      case SignalingMessageType.LEAVE:
        // Peer left
        this.removePeer(message.from!);
        break;
    }
  }

  /**
   * Create peer connection
   */
  private async createPeerConnection(
    peerId: string,
    initiator: boolean
  ): Promise<void> {
    console.log(`[WebRTCManager] Creating connection to ${peerId} (initiator: ${initiator})`);

    const pc = new RTCPeerConnection({
      iceServers: this.config.iceServers,
    });

    // ICE candidate handler
    pc.onicecandidate = (event) => {
      if (event.candidate && this.signalingSocket) {
        this.sendSignalingMessage({
          type: SignalingMessageType.ICE_CANDIDATE,
          from: this.userId!,
          to: peerId,
          data: event.candidate,
        });
      }
    };

    // Connection state handler
    pc.onconnectionstatechange = () => {
      console.log(`[WebRTCManager] Connection to ${peerId}: ${pc.connectionState}`);

      const state = this.mapConnectionState(pc.connectionState);
      this.updatePeerState(peerId, { state });

      if (pc.connectionState === 'failed' && this.config.enableReconnection) {
        setTimeout(() => this.createPeerConnection(peerId, initiator), 1000);
      }
    };

    this.peers.set(peerId, pc);

    // Create data channel
    if (initiator) {
      const channel = pc.createDataChannel('avatar-state', {
        ordered: false,  // Unreliable for low latency
        maxRetransmits: 0,
      });

      this.setupDataChannel(peerId, channel);

      // Create and send offer
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      this.sendSignalingMessage({
        type: SignalingMessageType.OFFER,
        from: this.userId!,
        to: peerId,
        data: offer,
      });
    } else {
      // Wait for data channel from peer
      pc.ondatachannel = (event) => {
        this.setupDataChannel(peerId, event.channel);
      };
    }
  }

  /**
   * Setup data channel handlers
   */
  private setupDataChannel(peerId: string, channel: RTCDataChannel): void {
    console.log(`[WebRTCManager] Data channel opened with ${peerId}`);

    channel.onopen = () => {
      console.log(`[WebRTCManager] Data channel ${peerId} ready`);
      this.dataChannels.set(peerId, channel);
      this.updatePeerState(peerId, { state: PeerConnectionState.CONNECTED });
    };

    channel.onmessage = (event) => {
      const update: AvatarStateUpdate = JSON.parse(event.data);
      this.handleAvatarUpdate(update);
    };

    channel.onclose = () => {
      console.log(`[WebRTCManager] Data channel ${peerId} closed`);
      this.dataChannels.delete(peerId);
    };

    channel.onerror = (error) => {
      console.error(`[WebRTCManager] Data channel error ${peerId}:`, error);
    };
  }

  /**
   * Handle offer from peer
   */
  private async handleOffer(peerId: string, offer: RTCSessionDescriptionInit): Promise<void> {
    // Create connection if doesn't exist
    if (!this.peers.has(peerId)) {
      await this.createPeerConnection(peerId, false);
    }

    const pc = this.peers.get(peerId)!;

    await pc.setRemoteDescription(new RTCSessionDescription(offer));

    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);

    this.sendSignalingMessage({
      type: SignalingMessageType.ANSWER,
      from: this.userId!,
      to: peerId,
      data: answer,
    });
  }

  /**
   * Handle answer from peer
   */
  private async handleAnswer(peerId: string, answer: RTCSessionDescriptionInit): Promise<void> {
    const pc = this.peers.get(peerId);
    if (!pc) return;

    await pc.setRemoteDescription(new RTCSessionDescription(answer));
  }

  /**
   * Handle ICE candidate from peer
   */
  private async handleIceCandidate(peerId: string, candidate: RTCIceCandidateInit): Promise<void> {
    const pc = this.peers.get(peerId);
    if (!pc) return;

    await pc.addIceCandidate(new RTCIceCandidate(candidate));
  }

  /**
   * Send avatar state to all peers
   */
  sendAvatarState(state: AvatarStateUpdate): void {
    this.lastState = state;

    // Send to all connected peers
    this.dataChannels.forEach((channel, peerId) => {
      if (channel.readyState === 'open') {
        try {
          channel.send(JSON.stringify(state));
        } catch (error) {
          console.error(`[WebRTCManager] Failed to send to ${peerId}:`, error);
        }
      }
    });
  }

  /**
   * Handle avatar update from peer
   */
  private handleAvatarUpdate(update: AvatarStateUpdate): void {
    // Update peer state
    this.updatePeerState(update.userId, {
      lastUpdate: update.timestamp,
      latency: Date.now() - update.timestamp,
    });

    // Emit event
    this.emit('avatar-update', update);
  }

  /**
   * Update peer state
   */
  private updatePeerState(peerId: string, updates: Partial<PeerInfo>): void {
    const existing = this.peerStates.get(peerId) || {
      userId: peerId,
      state: PeerConnectionState.DISCONNECTED,
      latency: 0,
      lastUpdate: 0,
    };

    this.peerStates.set(peerId, { ...existing, ...updates });
  }

  /**
   * Remove peer
   */
  private removePeer(peerId: string): void {
    console.log(`[WebRTCManager] Removing peer ${peerId}`);

    const pc = this.peers.get(peerId);
    if (pc) {
      pc.close();
      this.peers.delete(peerId);
    }

    this.dataChannels.delete(peerId);
    this.peerStates.delete(peerId);

    this.emit('peer-disconnected', peerId);
  }

  /**
   * Send signaling message
   */
  private sendSignalingMessage(message: SignalingMessage): void {
    if (this.signalingSocket && this.signalingConnected) {
      this.signalingSocket.send(JSON.stringify(message));
    }
  }

  /**
   * Map WebRTC connection state to PeerConnectionState
   */
  private mapConnectionState(state: RTCPeerConnectionState): PeerConnectionState {
    switch (state) {
      case 'connecting':
      case 'new':
        return PeerConnectionState.CONNECTING;
      case 'connected':
        return PeerConnectionState.CONNECTED;
      case 'disconnected':
      case 'closed':
        return PeerConnectionState.DISCONNECTED;
      case 'failed':
        return PeerConnectionState.FAILED;
      default:
        return PeerConnectionState.DISCONNECTED;
    }
  }

  /**
   * Start heartbeat
   */
  private startHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }

    this.heartbeatInterval = window.setInterval(() => {
      // Check peer connections
      this.peerStates.forEach((peer, peerId) => {
        const timeSinceUpdate = Date.now() - peer.lastUpdate;

        if (timeSinceUpdate > 5000 && peer.state === PeerConnectionState.CONNECTED) {
          console.warn(`[WebRTCManager] Peer ${peerId} timeout (${timeSinceUpdate}ms)`);
          this.updatePeerState(peerId, { state: PeerConnectionState.DISCONNECTED });
        }
      });
    }, this.config.heartbeatInterval);
  }

  /**
   * Get connected peers
   */
  getConnectedPeers(): PeerInfo[] {
    return Array.from(this.peerStates.values()).filter(
      (peer) => peer.state === PeerConnectionState.CONNECTED
    );
  }

  /**
   * Get peer count
   */
  getPeerCount(): number {
    return this.getConnectedPeers().length;
  }

  /**
   * Get average latency
   */
  getAverageLatency(): number {
    const peers = this.getConnectedPeers();
    if (peers.length === 0) return 0;

    const total = peers.reduce((sum, peer) => sum + peer.latency, 0);
    return total / peers.length;
  }

  /**
   * Event emitter
   */
  on(event: string, handler: Function): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event)!.push(handler);
  }

  private emit(event: string, ...args: any[]): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach((handler) => handler(...args));
    }
  }

  /**
   * Disconnect and cleanup
   */
  disconnect(): void {
    // Close all peer connections
    this.peers.forEach((pc) => pc.close());
    this.peers.clear();
    this.dataChannels.clear();
    this.peerStates.clear();

    // Close signaling connection
    if (this.signalingSocket) {
      this.signalingSocket.close();
      this.signalingSocket = null;
    }

    // Clear intervals
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }

    console.log('[WebRTCManager] Disconnected');
  }
}

/**
 * Create WebRTC manager
 */
export function createWebRTCManager(config: WebRTCConfig): WebRTCManager {
  return new WebRTCManager(config);
}
