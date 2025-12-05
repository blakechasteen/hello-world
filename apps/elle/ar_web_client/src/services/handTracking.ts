/**
 * Hand Tracking Service - MediaPipe Hands integration for browser
 *
 * Provides real-time hand pose estimation and gesture recognition using
 * MediaPipe Hands. Detects 21 landmarks per hand and classifies gestures.
 *
 * Gestures supported:
 * - point (index finger extended)
 * - grab/fist (all fingers closed)
 * - open_palm (all fingers extended)
 * - pinch (thumb and index touching)
 * - thumbs_up
 * - peace (index and middle extended)
 *
 * Created: 2025-11-22
 */

import { Hands, Results, NormalizedLandmark } from '@mediapipe/hands'
import { Camera } from '@mediapipe/camera_utils'

// ============================================================================
// Types
// ============================================================================

export interface HandLandmark {
  x: number
  y: number
  z: number
  visibility?: number
}

export interface HandPose {
  handId: string // "left" or "right"
  landmarks: HandLandmark[]
  confidence: number
  gesture?: string
  worldLandmarks?: HandLandmark[] // 3D world coordinates
}

export enum Gesture {
  POINT = 'point',
  GRAB = 'grab',
  OPEN_PALM = 'open_palm',
  PINCH = 'pinch',
  THUMBS_UP = 'thumbs_up',
  PEACE = 'peace',
  UNKNOWN = 'unknown',
}

export interface HandTrackingConfig {
  maxHands?: number
  minDetectionConfidence?: number
  minTrackingConfidence?: number
  modelComplexity?: 0 | 1 // 0 = lite (faster), 1 = full (more accurate)
}

export interface PointingDirection {
  x: number
  y: number
  z: number
}

// ============================================================================
// MediaPipe Landmarks
// ============================================================================

const HAND_LANDMARKS = [
  'WRIST',
  'THUMB_CMC',
  'THUMB_MCP',
  'THUMB_IP',
  'THUMB_TIP',
  'INDEX_FINGER_MCP',
  'INDEX_FINGER_PIP',
  'INDEX_FINGER_DIP',
  'INDEX_FINGER_TIP',
  'MIDDLE_FINGER_MCP',
  'MIDDLE_FINGER_PIP',
  'MIDDLE_FINGER_DIP',
  'MIDDLE_FINGER_TIP',
  'RING_FINGER_MCP',
  'RING_FINGER_PIP',
  'RING_FINGER_DIP',
  'RING_FINGER_TIP',
  'PINKY_MCP',
  'PINKY_PIP',
  'PINKY_DIP',
  'PINKY_TIP',
]

const FINGERTIP_INDICES = {
  thumb: 4,
  index: 8,
  middle: 12,
  ring: 16,
  pinky: 20,
}

// ============================================================================
// Hand Tracking Service
// ============================================================================

export class HandTrackingService {
  private hands: Hands | null = null
  private camera: Camera | null = null
  private initialized = false
  private currentResults: Results | null = null
  private onResultsCallback: ((results: HandPose[]) => void) | null = null

  constructor(private config: HandTrackingConfig = {}) {
    this.config = {
      maxHands: config.maxHands ?? 2,
      minDetectionConfidence: config.minDetectionConfidence ?? 0.7,
      minTrackingConfidence: config.minTrackingConfidence ?? 0.5,
      modelComplexity: config.modelComplexity ?? 0, // lite by default
    }
  }

  /**
   * Initialize MediaPipe Hands
   */
  async initialize(): Promise<boolean> {
    if (this.initialized) return true

    try {
      // Create MediaPipe Hands instance
      this.hands = new Hands({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        },
      })

      // Configure
      this.hands.setOptions({
        maxNumHands: this.config.maxHands!,
        modelComplexity: this.config.modelComplexity!,
        minDetectionConfidence: this.config.minDetectionConfidence!,
        minTrackingConfidence: this.config.minTrackingConfidence!,
      })

      // Set up results callback
      this.hands.onResults((results) => {
        this.currentResults = results
        if (this.onResultsCallback) {
          const handPoses = this.processResults(results)
          this.onResultsCallback(handPoses)
        }
      })

      this.initialized = true
      console.log('✅ MediaPipe Hands initialized')
      return true
    } catch (error) {
      console.error('❌ Failed to initialize MediaPipe Hands:', error)
      return false
    }
  }

  /**
   * Start tracking from video element
   */
  async startTracking(
    videoElement: HTMLVideoElement,
    onResults: (results: HandPose[]) => void
  ): Promise<boolean> {
    if (!this.initialized) {
      const success = await this.initialize()
      if (!success) return false
    }

    this.onResultsCallback = onResults

    try {
      // Create camera
      this.camera = new Camera(videoElement, {
        onFrame: async () => {
          if (this.hands) {
            await this.hands.send({ image: videoElement })
          }
        },
        width: 1280,
        height: 720,
      })

      await this.camera.start()
      console.log('✅ Hand tracking started')
      return true
    } catch (error) {
      console.error('❌ Failed to start hand tracking:', error)
      return false
    }
  }

  /**
   * Process single frame (for non-camera use cases)
   */
  async processFrame(
    input: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement
  ): Promise<HandPose[]> {
    if (!this.initialized) {
      const success = await this.initialize()
      if (!success) return []
    }

    if (!this.hands) return []

    await this.hands.send({ image: input })

    // Wait for results (with timeout)
    return new Promise((resolve) => {
      const timeout = setTimeout(() => resolve([]), 1000)

      const checkResults = () => {
        if (this.currentResults) {
          clearTimeout(timeout)
          const handPoses = this.processResults(this.currentResults)
          this.currentResults = null
          resolve(handPoses)
        } else {
          requestAnimationFrame(checkResults)
        }
      }

      checkResults()
    })
  }

  /**
   * Stop tracking
   */
  stop(): void {
    if (this.camera) {
      this.camera.stop()
      this.camera = null
    }
    this.onResultsCallback = null
    console.log('🛑 Hand tracking stopped')
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    this.stop()
    if (this.hands) {
      this.hands.close()
      this.hands = null
    }
    this.initialized = false
    console.log('🧹 Hand tracking cleaned up')
  }

  /**
   * Process MediaPipe results into HandPose objects
   */
  private processResults(results: Results): HandPose[] {
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      return []
    }

    const handPoses: HandPose[] = []

    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      const landmarks = results.multiHandLandmarks[i]
      const worldLandmarks = results.multiHandWorldLandmarks?.[i]
      const handedness = results.multiHandedness?.[i]

      // Determine hand ID (left/right)
      const handId =
        handedness?.label?.toLowerCase() || `hand_${i}`

      // Convert landmarks
      const handLandmarks: HandLandmark[] = landmarks.map((lm) => ({
        x: lm.x,
        y: lm.y,
        z: lm.z,
        visibility: (lm as any).visibility || 1.0,
      }))

      const worldHandLandmarks: HandLandmark[] | undefined =
        worldLandmarks?.map((lm) => ({
          x: lm.x,
          y: lm.y,
          z: lm.z,
        }))

      // Create hand pose
      const handPose: HandPose = {
        handId,
        landmarks: handLandmarks,
        confidence: 0.9, // MediaPipe doesn't provide per-hand confidence
        worldLandmarks: worldHandLandmarks,
      }

      // Recognize gesture
      handPose.gesture = this.recognizeGesture(handPose)

      handPoses.push(handPose)
    }

    return handPoses
  }

  /**
   * Recognize gesture from hand landmarks
   */
  private recognizeGesture(handPose: HandPose): string {
    if (handPose.landmarks.length !== 21) {
      return Gesture.UNKNOWN
    }

    const fingerStates = this.getFingerStates(handPose.landmarks)

    // Check gestures in priority order
    if (this.isPoint(fingerStates)) {
      return Gesture.POINT
    } else if (this.isGrab(fingerStates)) {
      return Gesture.GRAB
    } else if (this.isOpenPalm(fingerStates)) {
      return Gesture.OPEN_PALM
    } else if (this.isPinch(handPose.landmarks)) {
      return Gesture.PINCH
    } else if (this.isThumbsUp(handPose.landmarks, fingerStates)) {
      return Gesture.THUMBS_UP
    } else if (this.isPeace(fingerStates)) {
      return Gesture.PEACE
    }

    return Gesture.UNKNOWN
  }

  /**
   * Determine which fingers are extended
   */
  private getFingerStates(landmarks: HandLandmark[]): Record<string, boolean> {
    const states: Record<string, boolean> = {}

    // Thumb (horizontal distance from wrist)
    const thumbTip = landmarks[FINGERTIP_INDICES.thumb]
    const thumbMcp = landmarks[2]
    const wrist = landmarks[0]
    states.thumb = Math.abs(thumbTip.x - wrist.x) > Math.abs(thumbMcp.x - wrist.x)

    // Other fingers (tip above MCP)
    for (const [finger, tipIdx] of Object.entries(FINGERTIP_INDICES)) {
      if (finger === 'thumb') continue

      const tip = landmarks[tipIdx]
      const mcp = landmarks[tipIdx - 3] // MCP is 3 indices before tip

      // Extended if tip is above MCP (lower y value in screen coords)
      states[finger] = tip.y < mcp.y
    }

    return states
  }

  private isPoint(states: Record<string, boolean>): boolean {
    return states.index && !states.middle && !states.ring && !states.pinky
  }

  private isGrab(states: Record<string, boolean>): boolean {
    return !states.thumb && !states.index && !states.middle && !states.ring && !states.pinky
  }

  private isOpenPalm(states: Record<string, boolean>): boolean {
    return states.thumb && states.index && states.middle && states.ring && states.pinky
  }

  private isPinch(landmarks: HandLandmark[]): boolean {
    const thumbTip = landmarks[FINGERTIP_INDICES.thumb]
    const indexTip = landmarks[FINGERTIP_INDICES.index]

    const distance = Math.sqrt(
      Math.pow(thumbTip.x - indexTip.x, 2) +
        Math.pow(thumbTip.y - indexTip.y, 2) +
        Math.pow(thumbTip.z - indexTip.z, 2)
    )

    return distance < 0.05 // Threshold for "touching"
  }

  private isThumbsUp(
    landmarks: HandLandmark[],
    states: Record<string, boolean>
  ): boolean {
    const thumbTip = landmarks[FINGERTIP_INDICES.thumb]
    const wrist = landmarks[0]

    // Thumb pointing upward (tip above wrist)
    const thumbUp = thumbTip.y < wrist.y

    // Other fingers closed
    const othersClosed = !states.index && !states.middle && !states.ring && !states.pinky

    return thumbUp && othersClosed
  }

  private isPeace(states: Record<string, boolean>): boolean {
    return states.index && states.middle && !states.ring && !states.pinky
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get pointing direction from hand pose (for point gesture)
 */
export function getPointingDirection(handPose: HandPose): PointingDirection | null {
  if (handPose.gesture !== Gesture.POINT || handPose.landmarks.length !== 21) {
    return null
  }

  const indexMcp = handPose.landmarks[5]
  const indexTip = handPose.landmarks[8]

  const direction = {
    x: indexTip.x - indexMcp.x,
    y: indexTip.y - indexMcp.y,
    z: indexTip.z - indexMcp.z,
  }

  // Normalize
  const norm = Math.sqrt(direction.x ** 2 + direction.y ** 2 + direction.z ** 2)
  if (norm > 0) {
    direction.x /= norm
    direction.y /= norm
    direction.z /= norm
  }

  return direction
}

/**
 * Get pinch strength (0.0 = open, 1.0 = fully pinched)
 */
export function getPinchStrength(handPose: HandPose): number {
  if (handPose.landmarks.length !== 21) return 0

  const thumbTip = handPose.landmarks[FINGERTIP_INDICES.thumb]
  const indexTip = handPose.landmarks[FINGERTIP_INDICES.index]

  const distance = Math.sqrt(
    Math.pow(thumbTip.x - indexTip.x, 2) +
      Math.pow(thumbTip.y - indexTip.y, 2) +
      Math.pow(thumbTip.z - indexTip.z, 2)
  )

  // Normalize to 0-1 (assuming max distance ~0.2)
  const strength = 1.0 - Math.min(distance / 0.2, 1.0)

  return strength
}

/**
 * Check if hand is in frame center (for UI interactions)
 */
export function isHandInCenter(
  handPose: HandPose,
  centerThreshold: number = 0.2
): boolean {
  // Get wrist position
  const wrist = handPose.landmarks[0]

  // Check if within center region (0.4-0.6 on both axes)
  const centerX = Math.abs(wrist.x - 0.5)
  const centerY = Math.abs(wrist.y - 0.5)

  return centerX < centerThreshold && centerY < centerThreshold
}

// ============================================================================
// Singleton Instance
// ============================================================================

let handTrackingInstance: HandTrackingService | null = null

/**
 * Get singleton hand tracking service
 */
export function getHandTrackingService(
  config?: HandTrackingConfig
): HandTrackingService {
  if (!handTrackingInstance) {
    handTrackingInstance = new HandTrackingService(config)
  }
  return handTrackingInstance
}

/**
 * Reset singleton (for testing)
 */
export function resetHandTrackingService(): void {
  if (handTrackingInstance) {
    handTrackingInstance.cleanup()
    handTrackingInstance = null
  }
}
