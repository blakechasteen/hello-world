/**
 * Voice Communication - Phase 3: Collaborative Intelligence
 *
 * WebRTC-based voice communication for collaborative sessions.
 * Enables real-time voice chat between session participants.
 *
 * Features:
 * - Peer-to-peer voice communication
 * - Push-to-talk and open mic modes
 * - Voice activity detection
 * - Spatial audio (experimental)
 *
 * @module collaboration/voice
 */

// ============================================================================
// Types
// ============================================================================

/**
 * Voice channel participant
 */
export interface VoiceParticipant {
  peerId: string;
  userId: string;
  name: string;
  isSpeaking: boolean;
  isMuted: boolean;
  isDeafened: boolean;
  volume: number;
  audioLevel: number;
}

/**
 * Voice channel configuration
 */
export interface VoiceConfig {
  /** Signaling server URL */
  signalingUrl: string;
  /** Room/channel ID */
  channelId: string;
  /** User information */
  user: {
    peerId: string;
    userId: string;
    name: string;
  };
  /** Audio settings */
  audio?: {
    echoCancellation?: boolean;
    noiseSuppression?: boolean;
    autoGainControl?: boolean;
    sampleRate?: number;
  };
  /** Voice activity detection */
  vad?: {
    enabled: boolean;
    threshold: number;
    smoothing: number;
  };
}

/**
 * Voice channel state
 */
export interface VoiceState {
  status: 'disconnected' | 'connecting' | 'connected';
  isMuted: boolean;
  isDeafened: boolean;
  isPushToTalk: boolean;
  isSpeaking: boolean;
  participants: Map<string, VoiceParticipant>;
  error: string | null;
}

/**
 * Voice events
 */
export type VoiceEvent =
  | { type: 'connected' }
  | { type: 'disconnected' }
  | { type: 'participantJoined'; participant: VoiceParticipant }
  | { type: 'participantLeft'; peerId: string }
  | { type: 'participantSpeaking'; peerId: string; speaking: boolean }
  | { type: 'participantMuted'; peerId: string; muted: boolean }
  | { type: 'localMuted'; muted: boolean }
  | { type: 'localDeafened'; deafened: boolean }
  | { type: 'error'; error: string };

type VoiceEventHandler = (event: VoiceEvent) => void;

// ============================================================================
// Voice Channel Manager
// ============================================================================

/**
 * Manages voice communication in a collaborative session
 */
export class VoiceChannelManager {
  private config: VoiceConfig;
  private state: VoiceState;
  private handlers: Set<VoiceEventHandler> = new Set();

  // WebRTC
  private localStream: MediaStream | null = null;
  private peerConnections: Map<string, RTCPeerConnection> = new Map();
  private audioElements: Map<string, HTMLAudioElement> = new Map();

  // Signaling
  private ws: WebSocket | null = null;

  // Voice Activity Detection
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private vadInterval: ReturnType<typeof setInterval> | null = null;

  constructor(config: VoiceConfig) {
    this.config = {
      ...config,
      audio: {
        echoCancellation: config.audio?.echoCancellation ?? true,
        noiseSuppression: config.audio?.noiseSuppression ?? true,
        autoGainControl: config.audio?.autoGainControl ?? true,
        sampleRate: config.audio?.sampleRate ?? 48000,
      },
      vad: {
        enabled: config.vad?.enabled ?? true,
        threshold: config.vad?.threshold ?? 0.01,
        smoothing: config.vad?.smoothing ?? 0.8,
      },
    };

    this.state = {
      status: 'disconnected',
      isMuted: false,
      isDeafened: false,
      isPushToTalk: false,
      isSpeaking: false,
      participants: new Map(),
      error: null,
    };
  }

  // ============================================================================
  // Connection
  // ============================================================================

  /**
   * Connect to voice channel
   */
  async connect(): Promise<void> {
    if (this.state.status !== 'disconnected') return;

    this.state.status = 'connecting';

    try {
      // Get microphone access
      this.localStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: this.config.audio!.echoCancellation,
          noiseSuppression: this.config.audio!.noiseSuppression,
          autoGainControl: this.config.audio!.autoGainControl,
          sampleRate: this.config.audio!.sampleRate,
        },
      });

      // Set up voice activity detection
      if (this.config.vad!.enabled) {
        this.setupVAD();
      }

      // Connect to signaling server
      await this.connectSignaling();

      this.state.status = 'connected';
      this.emit({ type: 'connected' });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to connect';
      this.state.error = errorMessage;
      this.state.status = 'disconnected';
      this.emit({ type: 'error', error: errorMessage });
      throw error;
    }
  }

  /**
   * Disconnect from voice channel
   */
  disconnect(): void {
    // Stop VAD
    if (this.vadInterval) {
      clearInterval(this.vadInterval);
      this.vadInterval = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
      this.analyser = null;
    }

    // Stop local stream
    if (this.localStream) {
      this.localStream.getTracks().forEach((track) => track.stop());
      this.localStream = null;
    }

    // Close peer connections
    this.peerConnections.forEach((pc) => pc.close());
    this.peerConnections.clear();

    // Remove audio elements
    this.audioElements.forEach((audio) => {
      audio.srcObject = null;
      audio.remove();
    });
    this.audioElements.clear();

    // Close signaling
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    // Reset state
    this.state.status = 'disconnected';
    this.state.participants.clear();

    this.emit({ type: 'disconnected' });
  }

  // ============================================================================
  // Audio Controls
  // ============================================================================

  /**
   * Mute/unmute local microphone
   */
  setMuted(muted: boolean): void {
    if (!this.localStream) return;

    this.localStream.getAudioTracks().forEach((track) => {
      track.enabled = !muted;
    });

    this.state.isMuted = muted;
    this.emit({ type: 'localMuted', muted });

    // Broadcast mute state
    this.sendSignal({ type: 'mute', muted });
  }

  /**
   * Toggle mute
   */
  toggleMute(): void {
    this.setMuted(!this.state.isMuted);
  }

  /**
   * Deafen/undeafen (mute all incoming audio)
   */
  setDeafened(deafened: boolean): void {
    this.audioElements.forEach((audio) => {
      audio.muted = deafened;
    });

    this.state.isDeafened = deafened;
    this.emit({ type: 'localDeafened', deafened });
  }

  /**
   * Toggle deafen
   */
  toggleDeafen(): void {
    this.setDeafened(!this.state.isDeafened);
  }

  /**
   * Set push-to-talk mode
   */
  setPushToTalk(enabled: boolean): void {
    this.state.isPushToTalk = enabled;

    if (enabled) {
      // Start muted in PTT mode
      this.setMuted(true);
    }
  }

  /**
   * Push-to-talk key down (unmute while held)
   */
  pttKeyDown(): void {
    if (this.state.isPushToTalk) {
      this.setMuted(false);
    }
  }

  /**
   * Push-to-talk key up (mute when released)
   */
  pttKeyUp(): void {
    if (this.state.isPushToTalk) {
      this.setMuted(true);
    }
  }

  /**
   * Set participant volume
   */
  setParticipantVolume(peerId: string, volume: number): void {
    const audio = this.audioElements.get(peerId);
    if (audio) {
      audio.volume = Math.max(0, Math.min(1, volume));
    }

    const participant = this.state.participants.get(peerId);
    if (participant) {
      participant.volume = volume;
    }
  }

  // ============================================================================
  // State Access
  // ============================================================================

  /**
   * Get current state
   */
  getState(): Readonly<VoiceState> {
    return this.state;
  }

  /**
   * Get all participants
   */
  getParticipants(): VoiceParticipant[] {
    return Array.from(this.state.participants.values());
  }

  /**
   * Check if connected
   */
  get isConnected(): boolean {
    return this.state.status === 'connected';
  }

  // ============================================================================
  // Event Subscription
  // ============================================================================

  /**
   * Subscribe to voice events
   */
  subscribe(handler: VoiceEventHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  private emit(event: VoiceEvent): void {
    this.handlers.forEach((handler) => {
      try {
        handler(event);
      } catch (error) {
        console.error('Voice event handler error:', error);
      }
    });
  }

  // ============================================================================
  // WebRTC
  // ============================================================================

  private async createPeerConnection(peerId: string): Promise<RTCPeerConnection> {
    const pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
    });

    // Add local stream
    if (this.localStream) {
      this.localStream.getTracks().forEach((track) => {
        pc.addTrack(track, this.localStream!);
      });
    }

    // Handle incoming stream
    pc.ontrack = (event) => {
      const stream = event.streams[0];
      if (stream) {
        this.setupRemoteAudio(peerId, stream);
      }
    };

    // Handle ICE candidates
    pc.onicecandidate = (event) => {
      if (event.candidate) {
        this.sendSignal({
          type: 'ice-candidate',
          candidate: event.candidate,
          peerId,
        });
      }
    };

    // Handle connection state
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
        this.handlePeerDisconnected(peerId);
      }
    };

    this.peerConnections.set(peerId, pc);
    return pc;
  }

  private setupRemoteAudio(peerId: string, stream: MediaStream): void {
    // Create audio element for playback
    const audio = document.createElement('audio');
    audio.srcObject = stream;
    audio.autoplay = true;
    audio.muted = this.state.isDeafened;
    audio.volume = 1.0;

    // Keep reference
    this.audioElements.set(peerId, audio);

    // Set up audio level monitoring
    this.monitorAudioLevel(peerId, stream);
  }

  private monitorAudioLevel(peerId: string, stream: MediaStream): void {
    if (!this.audioContext) return;

    const source = this.audioContext.createMediaStreamSource(stream);
    const analyser = this.audioContext.createAnalyser();
    analyser.fftSize = 256;

    source.connect(analyser);

    const dataArray = new Float32Array(analyser.frequencyBinCount);
    let smoothedLevel = 0;

    const check = () => {
      if (!this.state.participants.has(peerId)) return;

      analyser.getFloatTimeDomainData(dataArray);
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i] * dataArray[i];
      }
      const rms = Math.sqrt(sum / dataArray.length);

      // Smooth the level
      smoothedLevel = this.config.vad!.smoothing * smoothedLevel + (1 - this.config.vad!.smoothing) * rms;

      const participant = this.state.participants.get(peerId);
      if (participant) {
        participant.audioLevel = smoothedLevel;

        const wasSpeaking = participant.isSpeaking;
        participant.isSpeaking = smoothedLevel > this.config.vad!.threshold;

        if (participant.isSpeaking !== wasSpeaking) {
          this.emit({
            type: 'participantSpeaking',
            peerId,
            speaking: participant.isSpeaking,
          });
        }
      }

      requestAnimationFrame(check);
    };

    requestAnimationFrame(check);
  }

  private handlePeerDisconnected(peerId: string): void {
    // Clean up peer connection
    const pc = this.peerConnections.get(peerId);
    if (pc) {
      pc.close();
      this.peerConnections.delete(peerId);
    }

    // Clean up audio element
    const audio = this.audioElements.get(peerId);
    if (audio) {
      audio.srcObject = null;
      audio.remove();
      this.audioElements.delete(peerId);
    }

    // Remove from participants
    this.state.participants.delete(peerId);
    this.emit({ type: 'participantLeft', peerId });
  }

  // ============================================================================
  // Voice Activity Detection
  // ============================================================================

  private setupVAD(): void {
    if (!this.localStream) return;

    this.audioContext = new AudioContext();
    const source = this.audioContext.createMediaStreamSource(this.localStream);
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 256;

    source.connect(this.analyser);

    const dataArray = new Float32Array(this.analyser.frequencyBinCount);
    let smoothedLevel = 0;

    this.vadInterval = setInterval(() => {
      if (!this.analyser) return;

      this.analyser.getFloatTimeDomainData(dataArray);
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i] * dataArray[i];
      }
      const rms = Math.sqrt(sum / dataArray.length);

      // Smooth the level
      smoothedLevel = this.config.vad!.smoothing * smoothedLevel + (1 - this.config.vad!.smoothing) * rms;

      const wasSpeaking = this.state.isSpeaking;
      this.state.isSpeaking = smoothedLevel > this.config.vad!.threshold && !this.state.isMuted;

      if (this.state.isSpeaking !== wasSpeaking) {
        this.sendSignal({ type: 'speaking', speaking: this.state.isSpeaking });
      }
    }, 50);
  }

  // ============================================================================
  // Signaling
  // ============================================================================

  private async connectSignaling(): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = `${this.config.signalingUrl}?channel=${this.config.channelId}&peerId=${this.config.user.peerId}`;
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        // Announce presence
        this.sendSignal({
          type: 'join',
          user: this.config.user,
        });
        resolve();
      };

      this.ws.onerror = (error) => {
        reject(new Error('Signaling connection failed'));
      };

      this.ws.onmessage = (event) => {
        this.handleSignal(JSON.parse(event.data));
      };

      this.ws.onclose = () => {
        if (this.state.status === 'connected') {
          this.disconnect();
        }
      };
    });
  }

  private sendSignal(message: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  private async handleSignal(signal: Record<string, unknown>): Promise<void> {
    switch (signal.type) {
      case 'join':
        await this.handlePeerJoined(signal as { user: VoiceConfig['user'] });
        break;

      case 'leave':
        this.handlePeerDisconnected(signal.peerId as string);
        break;

      case 'offer':
        await this.handleOffer(signal as { peerId: string; sdp: RTCSessionDescriptionInit });
        break;

      case 'answer':
        await this.handleAnswer(signal as { peerId: string; sdp: RTCSessionDescriptionInit });
        break;

      case 'ice-candidate':
        await this.handleIceCandidate(signal as { peerId: string; candidate: RTCIceCandidateInit });
        break;

      case 'mute':
        this.handlePeerMuted(signal as { peerId: string; muted: boolean });
        break;

      case 'speaking':
        this.handlePeerSpeaking(signal as { peerId: string; speaking: boolean });
        break;
    }
  }

  private async handlePeerJoined(signal: { user: VoiceConfig['user'] }): Promise<void> {
    const participant: VoiceParticipant = {
      peerId: signal.user.peerId,
      userId: signal.user.userId,
      name: signal.user.name,
      isSpeaking: false,
      isMuted: false,
      isDeafened: false,
      volume: 1.0,
      audioLevel: 0,
    };

    this.state.participants.set(participant.peerId, participant);
    this.emit({ type: 'participantJoined', participant });

    // Initiate WebRTC connection
    const pc = await this.createPeerConnection(participant.peerId);
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    this.sendSignal({
      type: 'offer',
      peerId: participant.peerId,
      sdp: offer,
    });
  }

  private async handleOffer(signal: { peerId: string; sdp: RTCSessionDescriptionInit }): Promise<void> {
    let pc = this.peerConnections.get(signal.peerId);
    if (!pc) {
      pc = await this.createPeerConnection(signal.peerId);
    }

    await pc.setRemoteDescription(signal.sdp);
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);

    this.sendSignal({
      type: 'answer',
      peerId: signal.peerId,
      sdp: answer,
    });
  }

  private async handleAnswer(signal: { peerId: string; sdp: RTCSessionDescriptionInit }): Promise<void> {
    const pc = this.peerConnections.get(signal.peerId);
    if (pc) {
      await pc.setRemoteDescription(signal.sdp);
    }
  }

  private async handleIceCandidate(signal: { peerId: string; candidate: RTCIceCandidateInit }): Promise<void> {
    const pc = this.peerConnections.get(signal.peerId);
    if (pc) {
      await pc.addIceCandidate(signal.candidate);
    }
  }

  private handlePeerMuted(signal: { peerId: string; muted: boolean }): void {
    const participant = this.state.participants.get(signal.peerId);
    if (participant) {
      participant.isMuted = signal.muted;
      this.emit({ type: 'participantMuted', peerId: signal.peerId, muted: signal.muted });
    }
  }

  private handlePeerSpeaking(signal: { peerId: string; speaking: boolean }): void {
    const participant = this.state.participants.get(signal.peerId);
    if (participant) {
      participant.isSpeaking = signal.speaking;
      this.emit({
        type: 'participantSpeaking',
        peerId: signal.peerId,
        speaking: signal.speaking,
      });
    }
  }
}

// ============================================================================
// Factory
// ============================================================================

/**
 * Create voice channel manager
 */
export function createVoiceChannel(config: VoiceConfig): VoiceChannelManager {
  return new VoiceChannelManager(config);
}
