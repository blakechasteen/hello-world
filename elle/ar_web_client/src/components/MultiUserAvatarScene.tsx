/**
 * MultiUserAvatarScene - Complete Multi-User AR Avatar System
 *
 * Integrates all Phase 6.2 systems for multi-user AR avatars:
 * - Local avatar with person segmentation (AvatarCompositor)
 * - Remote avatars via WebRTC (WebRTCManager)
 * - Coordinate alignment via spatial anchors (SpatialAnchorManager)
 * - Real-time pose synchronization (<50ms latency)
 *
 * Features:
 * - 2-4 simultaneous users (mesh topology)
 * - Clean background removal via person segmentation
 * - QR code-based coordinate alignment
 * - Automatic reconnection and error recovery
 * - Connection status UI
 *
 * Architecture:
 * - Canvas: Composited output (video + avatars, segmented)
 * - Three.js: 3D avatar rendering
 * - WebRTC: P2P avatar state sync
 * - Spatial Anchors: Coordinate system alignment
 *
 * Created: 2025-11-22 (Phase 6.2)
 */

import React, { useEffect, useRef, useState } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import * as THREE from 'three';

// Phase 6.1: Avatar rendering
import { Avatar } from './Avatar';

// Phase 6.2: Multi-user systems
import { AvatarCompositor, createAvatarCompositor } from '../compositing/AvatarCompositor';
import { WebRTCManager, createWebRTCManager, AvatarStateUpdate, PeerInfo } from '../multiplayer/WebRTCManager';
import { SpatialAnchorManager, createSpatialAnchorManager, SpatialAnchor } from '../multiplayer/SpatialAnchorManager';

// Phase 6.3: LOD system
import { AvatarLOD, LODMetricsDisplay } from './AvatarLOD';
import { LODManager, createLODManager } from '../optimization/LODManager';
import { LODLevel } from '../optimization/LODLevel';
import { FrustumCuller, createFrustumCuller } from '../optimization/FrustumCuller';

// Phase 5: Pose estimation
import { getPoseEstimationService, BodyPose } from '../services/poseEstimation';

/**
 * Configuration for multi-user avatar scene
 */
export interface MultiUserAvatarSceneConfig {
  /**
   * Current user ID
   */
  userId: string;

  /**
   * Local avatar URL (VRM file)
   */
  avatarUrl: string;

  /**
   * Signaling server URL for WebRTC
   */
  signalingServerUrl: string;

  /**
   * STUN/TURN servers for NAT traversal
   */
  iceServers?: RTCIceServer[];

  /**
   * Enable person segmentation (default: true)
   */
  enableSegmentation?: boolean;

  /**
   * Enable spatial anchors (default: true)
   */
  enableSpatialAnchors?: boolean;

  /**
   * Avatar update rate in Hz (default: 30)
   */
  updateRate?: number;

  /**
   * Show debug overlay (default: false)
   */
  showDebugOverlay?: boolean;

  /**
   * Enable LOD system (default: true)
   */
  enableLOD?: boolean;

  /**
   * Show LOD metrics (default: false)
   */
  showLODMetrics?: boolean;

  /**
   * Enable occlusion culling (default: true)
   */
  enableOcclusionCulling?: boolean;
}

/**
 * Remote avatar state
 */
interface RemoteAvatar {
  userId: string;
  avatarUrl: string;
  pose: BodyPose | null;
  position: THREE.Vector3;
  rotation: THREE.Quaternion;
  lastUpdate: number;
}

/**
 * MultiUserAvatarScene - Complete multi-user AR avatar system
 *
 * Usage:
 * ```tsx
 * <MultiUserAvatarScene
 *   userId="user-123"
 *   avatarUrl="/avatars/my-avatar.vrm"
 *   signalingServerUrl="ws://localhost:8080"
 * />
 * ```
 */
export const MultiUserAvatarScene: React.FC<MultiUserAvatarSceneConfig> = ({
  userId,
  avatarUrl,
  signalingServerUrl,
  iceServers = [{ urls: 'stun:stun.l.google.com:19302' }],
  enableSegmentation = true,
  enableSpatialAnchors = true,
  updateRate = 30,
  showDebugOverlay = false,
  enableLOD = true,
  showLODMetrics = false,
  enableOcclusionCulling = true,
}) => {
  // Video and canvas refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const avatarCanvasRef = useRef<HTMLCanvasElement>(null);

  // Local state
  const [localPose, setLocalPose] = useState<BodyPose | null>(null);
  const [localPosition, setLocalPosition] = useState<THREE.Vector3>(new THREE.Vector3(0, 0, 0));
  const [localRotation, setLocalRotation] = useState<THREE.Quaternion>(new THREE.Quaternion());

  // Remote avatars
  const [remoteAvatars, setRemoteAvatars] = useState<Map<string, RemoteAvatar>>(new Map());

  // Connection state
  const [isConnected, setIsConnected] = useState(false);
  const [peers, setPeers] = useState<PeerInfo[]>([]);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  // Spatial anchor state
  const [activeAnchor, setActiveAnchor] = useState<SpatialAnchor | null>(null);
  const [isScanning, setIsScanning] = useState(false);

  // Managers (refs to persist across renders)
  const compositorRef = useRef<AvatarCompositor | null>(null);
  const webrtcRef = useRef<WebRTCManager | null>(null);
  const anchorManagerRef = useRef<SpatialAnchorManager | null>(null);
  const lodManagerRef = useRef<LODManager | null>(null);
  const frustumCullerRef = useRef<FrustumCuller | null>(null);
  const poseServiceRef = useRef<any>(null);

  // Animation frame ID
  const animationFrameRef = useRef<number | null>(null);

  /**
   * Initialize webcam
   */
  useEffect(() => {
    const initWebcam = async () => {
      if (!videoRef.current) return;

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
          audio: false,
        });

        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        console.log('[MultiUserAvatarScene] Webcam initialized');
      } catch (error) {
        console.error('[MultiUserAvatarScene] Webcam initialization failed:', error);
        setConnectionError('Failed to access webcam');
      }
    };

    initWebcam();

    // Cleanup
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  /**
   * Initialize all managers
   */
  useEffect(() => {
    const initManagers = async () => {
      try {
        // 1. Initialize compositor (if segmentation enabled)
        if (enableSegmentation) {
          compositorRef.current = createAvatarCompositor({
            enableEdgeRefinement: true,
            edgeBlurRadius: 3,
            enableTemporalSmoothing: true,
            temporalSmoothingFactor: 0.3,
            enableWebGL: true,
          });
          console.log('[MultiUserAvatarScene] Compositor initialized');
        }

        // 2. Initialize WebRTC manager
        webrtcRef.current = createWebRTCManager({
          signalingServerUrl,
          iceServers,
          updateRate,
          enableReconnection: true,
        });

        // Connect to signaling server
        await webrtcRef.current.connect(userId);
        setIsConnected(true);
        console.log('[MultiUserAvatarScene] WebRTC connected');

        // Listen for avatar updates
        webrtcRef.current.on('avatar-update', handleRemoteAvatarUpdate);
        webrtcRef.current.on('peer-disconnected', handlePeerDisconnected);

        // 3. Initialize spatial anchor manager (if enabled)
        if (enableSpatialAnchors) {
          anchorManagerRef.current = createSpatialAnchorManager({
            enablePersistence: true,
            enableAutoDetection: false,
          });
          console.log('[MultiUserAvatarScene] Spatial anchor manager initialized');
        }

        // 4. Initialize pose estimation service
        poseServiceRef.current = getPoseEstimationService();
        console.log('[MultiUserAvatarScene] Pose estimation service initialized');
      } catch (error) {
        console.error('[MultiUserAvatarScene] Manager initialization failed:', error);
        setConnectionError('Failed to initialize managers');
      }
    };

    initManagers();

    // Cleanup
    return () => {
      if (webrtcRef.current) {
        webrtcRef.current.disconnect();
      }
      if (compositorRef.current) {
        compositorRef.current.dispose();
      }
    };
  }, [userId, signalingServerUrl, iceServers, updateRate, enableSegmentation, enableSpatialAnchors]);

  /**
   * Handle remote avatar update
   */
  const handleRemoteAvatarUpdate = (update: AvatarStateUpdate) => {
    setRemoteAvatars((prev) => {
      const newMap = new Map(prev);

      const position = new THREE.Vector3(...update.position);
      const rotation = new THREE.Quaternion(...update.rotation);

      // Transform to anchor space if anchor is active
      const transformedPosition = anchorManagerRef.current?.getActiveAnchor()
        ? anchorManagerRef.current.anchorToWorld(position)
        : position;

      newMap.set(update.userId, {
        userId: update.userId,
        avatarUrl: avatarUrl, // TODO: Support different avatars per user
        pose: update.pose,
        position: transformedPosition,
        rotation,
        lastUpdate: update.timestamp,
      });

      return newMap;
    });
  };

  /**
   * Handle peer disconnection
   */
  const handlePeerDisconnected = (peerId: string) => {
    setRemoteAvatars((prev) => {
      const newMap = new Map(prev);
      newMap.delete(peerId);
      return newMap;
    });
  };

  /**
   * Main render loop
   */
  useEffect(() => {
    const renderLoop = async () => {
      if (!videoRef.current || !canvasRef.current || !avatarCanvasRef.current) {
        animationFrameRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      try {
        // 1. Get current pose from video
        if (poseServiceRef.current) {
          const pose = await poseServiceRef.current.processFrame(videoRef.current);
          setLocalPose(pose);
        }

        // 2. Composite avatar with video (if segmentation enabled)
        if (enableSegmentation && compositorRef.current) {
          const result = await compositorRef.current.composite(
            videoRef.current,
            avatarCanvasRef.current
          );

          // Draw composited result to main canvas
          const ctx = canvasRef.current.getContext('2d');
          if (ctx) {
            ctx.drawImage(result.canvas, 0, 0);
          }
        }

        // 3. Send avatar state to peers
        if (webrtcRef.current && isConnected) {
          const state: AvatarStateUpdate = {
            userId,
            timestamp: Date.now(),
            pose: localPose,
            position: localPosition.toArray() as [number, number, number],
            rotation: localRotation.toArray() as [number, number, number, number],
          };

          webrtcRef.current.sendAvatarState(state);
        }

        // 4. Update peer list
        if (webrtcRef.current) {
          setPeers(webrtcRef.current.getConnectedPeers());
        }
      } catch (error) {
        console.error('[MultiUserAvatarScene] Render loop error:', error);
      }

      animationFrameRef.current = requestAnimationFrame(renderLoop);
    };

    renderLoop();

    // Cleanup
    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [userId, localPose, localPosition, localRotation, isConnected, enableSegmentation]);

  /**
   * Scan for spatial anchor (QR code)
   */
  const scanForAnchor = async () => {
    if (!anchorManagerRef.current || !videoRef.current) return;

    setIsScanning(true);

    const detected = await anchorManagerRef.current.detectAnchor(videoRef.current);

    if (detected) {
      await anchorManagerRef.current.alignToAnchor(detected.id);
      setActiveAnchor(detected);
      console.log('[MultiUserAvatarScene] Aligned to anchor:', detected.id);
    } else {
      console.warn('[MultiUserAvatarScene] No anchor detected');
    }

    setIsScanning(false);
  };

  /**
   * Create new spatial anchor
   */
  const createAnchor = async () => {
    if (!anchorManagerRef.current) return;

    const anchor = await anchorManagerRef.current.createAnchor({
      type: 'manual',
      position: localPosition,
      rotation: localRotation,
      data: `anchor-${Date.now()}`,
      createdBy: userId,
    });

    await anchorManagerRef.current.alignToAnchor(anchor.id);
    setActiveAnchor(anchor);
    console.log('[MultiUserAvatarScene] Created anchor:', anchor.id);
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100vh' }}>
      {/* Hidden video element (webcam feed) */}
      <video
        ref={videoRef}
        style={{ display: 'none' }}
        playsInline
        muted
      />

      {/* Main composited output canvas */}
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          zIndex: 1,
        }}
      />

      {/* Hidden canvas for avatar rendering (input to compositor) */}
      <canvas
        ref={avatarCanvasRef}
        style={{ display: 'none' }}
      />

      {/* Three.js scene for 3D avatars */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 2,
        pointerEvents: 'none',
      }}>
        <Canvas>
          {/* Lighting */}
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 5]} intensity={1} />

          {/* Avatar rendering with LOD system */}
          <AvatarSceneContent
            enableLOD={enableLOD}
            enableOcclusionCulling={enableOcclusionCulling}
            userId={userId}
            avatarUrl={avatarUrl}
            localPose={localPose}
            localPosition={localPosition}
            localRotation={localRotation}
            remoteAvatars={remoteAvatars}
            lodManagerRef={lodManagerRef}
            frustumCullerRef={frustumCullerRef}
            showLODMetrics={showLODMetrics}
          />
        </Canvas>
      </div>

      {/* Debug overlay */}
      {showDebugOverlay && (
        <div style={{
          position: 'absolute',
          top: 10,
          left: 10,
          zIndex: 3,
          background: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          padding: '10px',
          borderRadius: '5px',
          fontFamily: 'monospace',
          fontSize: '12px',
        }}>
          <div><strong>Multi-User Avatar Scene</strong></div>
          <div>User ID: {userId}</div>
          <div>Connected: {isConnected ? '✓' : '✗'}</div>
          <div>Peers: {peers.length}</div>
          <div>Anchor: {activeAnchor ? activeAnchor.id : 'None'}</div>
          <div>Pose: {localPose ? '✓' : '✗'}</div>
          <div>Segmentation: {enableSegmentation ? 'ON' : 'OFF'}</div>
          {connectionError && <div style={{ color: 'red' }}>Error: {connectionError}</div>}
        </div>
      )}

      {/* Connection status UI */}
      {!isConnected && !connectionError && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          zIndex: 4,
          background: 'rgba(0, 0, 0, 0.8)',
          color: 'white',
          padding: '20px',
          borderRadius: '10px',
          textAlign: 'center',
        }}>
          <div>Connecting to signaling server...</div>
        </div>
      )}

      {/* Error UI */}
      {connectionError && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          zIndex: 4,
          background: 'rgba(255, 0, 0, 0.8)',
          color: 'white',
          padding: '20px',
          borderRadius: '10px',
          textAlign: 'center',
        }}>
          <div><strong>Error</strong></div>
          <div>{connectionError}</div>
        </div>
      )}

      {/* Spatial anchor controls */}
      {enableSpatialAnchors && (
        <div style={{
          position: 'absolute',
          bottom: 20,
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 3,
          display: 'flex',
          gap: '10px',
        }}>
          <button
            onClick={scanForAnchor}
            disabled={isScanning}
            style={{
              padding: '10px 20px',
              background: activeAnchor ? '#4CAF50' : '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: isScanning ? 'not-allowed' : 'pointer',
              fontSize: '14px',
            }}
          >
            {isScanning ? 'Scanning...' : activeAnchor ? `Aligned: ${activeAnchor.id.slice(0, 8)}` : 'Scan QR Code'}
          </button>

          <button
            onClick={createAnchor}
            style={{
              padding: '10px 20px',
              background: '#FF9800',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '14px',
            }}
          >
            Create Anchor
          </button>
        </div>
      )}

      {/* Peer list UI */}
      {peers.length > 0 && (
        <div style={{
          position: 'absolute',
          top: 10,
          right: 10,
          zIndex: 3,
          background: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          padding: '10px',
          borderRadius: '5px',
          fontFamily: 'monospace',
          fontSize: '12px',
        }}>
          <div><strong>Connected Peers ({peers.length})</strong></div>
          {peers.map((peer) => (
            <div key={peer.userId} style={{ marginTop: '5px' }}>
              {peer.userId.slice(0, 8)} - {peer.latency}ms
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

/**
 * AvatarSceneContent - Renders avatars with optional LOD and occlusion culling
 *
 * Internal component that handles avatar rendering inside Canvas.
 * Conditionally uses AvatarLOD or Avatar based on enableLOD flag.
 * Manages frustum culling for occlusion optimization.
 */
interface AvatarSceneContentProps {
  enableLOD: boolean;
  enableOcclusionCulling: boolean;
  userId: string;
  avatarUrl: string;
  localPose: BodyPose | null;
  localPosition: THREE.Vector3;
  localRotation: THREE.Quaternion;
  remoteAvatars: Map<string, RemoteAvatar>;
  lodManagerRef: React.MutableRefObject<LODManager | null>;
  frustumCullerRef: React.MutableRefObject<FrustumCuller | null>;
  showLODMetrics?: boolean;
}

const AvatarSceneContent: React.FC<AvatarSceneContentProps> = ({
  enableLOD,
  enableOcclusionCulling,
  userId,
  avatarUrl,
  localPose,
  localPosition,
  localRotation,
  remoteAvatars,
  lodManagerRef,
  frustumCullerRef,
  showLODMetrics = false,
}) => {
  const { camera } = useThree();

  /**
   * Initialize LOD manager
   */
  useEffect(() => {
    if (enableLOD && !lodManagerRef.current) {
      lodManagerRef.current = createLODManager(camera);
      console.log('[AvatarSceneContent] LOD manager initialized');
    }
  }, [enableLOD, camera, lodManagerRef]);

  /**
   * Initialize frustum culler
   */
  useEffect(() => {
    if (enableOcclusionCulling && !frustumCullerRef.current) {
      frustumCullerRef.current = createFrustumCuller(camera, { margin: 0.1 });
      console.log('[AvatarSceneContent] Frustum culler initialized');
    }
  }, [enableOcclusionCulling, camera, frustumCullerRef]);

  /**
   * Update camera reference when camera changes
   */
  useEffect(() => {
    if (enableLOD && lodManagerRef.current) {
      lodManagerRef.current.setCamera(camera);
    }
    if (enableOcclusionCulling && frustumCullerRef.current) {
      frustumCullerRef.current.setCamera(camera);
    }
  }, [camera, enableLOD, enableOcclusionCulling, lodManagerRef, frustumCullerRef]);

  /**
   * Update frustum each frame (before avatar rendering)
   */
  useFrame(() => {
    if (enableOcclusionCulling && frustumCullerRef.current) {
      frustumCullerRef.current.updateFrustum();
    }
  });

  return (
    <>
      {/* Local avatar */}
      {enableLOD && lodManagerRef.current ? (
        <AvatarLOD
          avatarId={userId}
          lodManager={lodManagerRef.current}
          url={avatarUrl}
          pose={localPose}
          position={[localPosition.x, localPosition.y, localPosition.z]}
          rotation={[
            localRotation.x,
            localRotation.y,
            localRotation.z,
            localRotation.w,
          ]}
          userId={userId}
          frustumCuller={enableOcclusionCulling ? frustumCullerRef.current ?? undefined : undefined}
        />
      ) : (
        <Avatar
          url={avatarUrl}
          pose={localPose}
          position={[localPosition.x, localPosition.y, localPosition.z]}
          rotation={[
            localRotation.x,
            localRotation.y,
            localRotation.z,
            localRotation.w,
          ]}
          userId={userId}
        />
      )}

      {/* Remote avatars */}
      {Array.from(remoteAvatars.values()).map((remote) =>
        enableLOD && lodManagerRef.current ? (
          <AvatarLOD
            key={remote.userId}
            avatarId={remote.userId}
            lodManager={lodManagerRef.current}
            url={remote.avatarUrl}
            pose={remote.pose}
            position={[remote.position.x, remote.position.y, remote.position.z]}
            rotation={[
              remote.rotation.x,
              remote.rotation.y,
              remote.rotation.z,
              remote.rotation.w,
            ]}
            userId={remote.userId}
            frustumCuller={enableOcclusionCulling ? frustumCullerRef.current ?? undefined : undefined}
          />
        ) : (
          <Avatar
            key={remote.userId}
            url={remote.avatarUrl}
            pose={remote.pose}
            position={[remote.position.x, remote.position.y, remote.position.z]}
            rotation={[
              remote.rotation.x,
              remote.rotation.y,
              remote.rotation.z,
              remote.rotation.w,
            ]}
            userId={remote.userId}
          />
        )
      )}

      {/* LOD metrics display */}
      {showLODMetrics && enableLOD && lodManagerRef.current && (
        <LODMetricsDisplay lodManager={lodManagerRef.current} position={[10, 60]} />
      )}
    </>
  );
};

/**
 * Default export
 */
export default MultiUserAvatarScene;
