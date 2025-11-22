/**
 * ARScene Component - Main 3D AR scene with object detection and visualization
 *
 * Handles:
 * - Camera feed and world tracking
 * - Real-time object detection (TensorFlow.js COCO-SSD)
 * - Hand tracking and gesture recognition (MediaPipe Hands)
 * - AR visualization rendering (overlays, highlights, paths)
 * - Spatial context updates
 *
 * Created: 2025-11-22
 * Updated: 2025-11-22 (Phase 2 - Vision Integration)
 */

import { useRef, useEffect, useState } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useXR, useHitTest } from '@react-three/xr'
import * as THREE from 'three'
import AROverlay from './AROverlay'
import ARHighlight from './ARHighlight'
import ARPath from './ARPath'
import {
  getObjectDetectionService,
  getHandTrackingService,
  type DetectedObject,
  type HandPose,
} from '../services'

interface ARSceneProps {
  visualizations: any[]
  onContextUpdate: (context: any) => void
  enableVision?: boolean // Enable vision processing (default: true)
  visionUpdateInterval?: number // Vision update interval in ms (default: 100)
}

export default function ARScene({
  visualizations,
  onContextUpdate,
  enableVision = true,
  visionUpdateInterval = 100,
}: ARSceneProps) {
  const { camera, gl } = useThree()
  const { session, player } = useXR()
  const sceneRef = useRef<THREE.Group>(null)

  // Vision state
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([])
  const [handPoses, setHandPoses] = useState<HandPose[]>([])
  const [visionInitialized, setVisionInitialized] = useState(false)

  // Vision services
  const objectDetectionService = useRef(getObjectDetectionService())
  const handTrackingService = useRef(getHandTrackingService())

  // Canvas for camera feed processing
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const lastVisionUpdate = useRef<number>(0)

  // Detect plane for object placement
  const hitTestRef = useRef<THREE.Mesh>(null)
  useHitTest((hitMatrix) => {
    if (hitTestRef.current) {
      hitTestRef.current.matrix.copy(hitMatrix)
      hitTestRef.current.matrix.decompose(
        hitTestRef.current.position,
        hitTestRef.current.quaternion,
        hitTestRef.current.scale
      )
    }
  })

  // Initialize vision services
  useEffect(() => {
    if (!enableVision) return

    const initVision = async () => {
      try {
        // Initialize object detection
        const objDetInit = await objectDetectionService.current.initialize()
        console.log('Object detection initialized:', objDetInit)

        // Initialize hand tracking
        const handInit = await handTrackingService.current.initialize()
        console.log('Hand tracking initialized:', handInit)

        setVisionInitialized(objDetInit && handInit)
      } catch (error) {
        console.error('Failed to initialize vision services:', error)
        setVisionInitialized(false)
      }
    }

    initVision()

    // Cleanup
    return () => {
      objectDetectionService.current.cleanup()
      handTrackingService.current.cleanup()
    }
  }, [enableVision])

  // Create canvas for camera feed processing
  useEffect(() => {
    if (!enableVision || !visionInitialized) return

    // Create off-screen canvas
    const canvas = document.createElement('canvas')
    canvas.width = 640
    canvas.height = 480
    canvasRef.current = canvas

    console.log('Vision canvas created')
  }, [enableVision, visionInitialized])

  // Update AR context every frame (with vision processing)
  useFrame(async () => {
    if (!session || !camera) return

    // Extract user position and orientation
    const position = camera.position
    const quaternion = camera.quaternion
    const gazeDirection = new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion)

    // Process vision data (throttled)
    const now = performance.now()
    if (
      enableVision &&
      visionInitialized &&
      canvasRef.current &&
      now - lastVisionUpdate.current > visionUpdateInterval
    ) {
      lastVisionUpdate.current = now

      try {
        // Capture camera frame to canvas
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        if (ctx && gl.domElement) {
          // Draw WebGL canvas to 2D canvas
          ctx.drawImage(gl.domElement, 0, 0, canvas.width, canvas.height)

          // Run object detection
          const objects = await objectDetectionService.current.detectObjects(
            canvas,
            20, // maxDetections
            0.5 // minConfidence
          )
          setDetectedObjects(objects)

          // Run hand tracking
          const hands = await handTrackingService.current.processFrame(canvas)
          setHandPoses(hands)
        }
      } catch (error) {
        console.error('Vision processing error:', error)
      }
    }

    // Build context update with vision data
    const context = {
      userPosition: {
        x: position.x,
        y: position.y,
        z: position.z,
      },
      userRotation: {
        x: quaternion.x,
        y: quaternion.y,
        z: quaternion.z,
        w: quaternion.w,
      },
      gazeDirection: {
        x: gazeDirection.x,
        y: gazeDirection.y,
        z: gazeDirection.z,
      },
      visibleObjects: convertDetectedObjectsToARObjects(detectedObjects),
      handGestures: handPoses.map((hand) => ({
        handId: hand.handId,
        gesture: hand.gesture,
        confidence: hand.confidence,
      })),
      sessionId: session.mode || 'unknown',
      platform: 'webxr',
    }

    // Send context update (throttled in hook)
    onContextUpdate(context)
  })

  return (
    <group ref={sceneRef}>
      {/* Ambient lighting */}
      <ambientLight intensity={0.5} />

      {/* Directional light following camera */}
      <directional Light
        position={[5, 10, 5]}
        intensity={1.0}
        castShadow
      />

      {/* Ground plane indicator (invisible, for hit testing) */}
      <mesh
        ref={hitTestRef}
        rotation-x={-Math.PI / 2}
        visible={false}
      >
        <planeGeometry args={[100, 100]} />
        <meshBasicMaterial transparent opacity={0} />
      </mesh>

      {/* Render detected object labels */}
      {enableVision &&
        detectedObjects.map((obj) => (
          <AROverlay
            key={obj.id}
            id={obj.id}
            content={`${obj.label} (${(obj.confidence * 100).toFixed(0)}%)`}
            position={{
              x: (obj.bbox.xMin + obj.bbox.xMax) / 2 - 0.5,
              y: 0.5,
              z: -1.5,
            }}
            style={{
              color: '#00ff00',
              opacity: 0.8,
              size: 0.8,
            }}
          />
        ))}

      {/* Render hand gesture indicators */}
      {enableVision &&
        handPoses.map((hand, idx) => (
          <AROverlay
            key={`hand_${hand.handId}_${idx}`}
            id={`hand_${hand.handId}_${idx}`}
            content={`${hand.handId} ✋ ${hand.gesture || 'unknown'}`}
            position={{ x: idx === 0 ? -0.5 : 0.5, y: 1.2, z: -1.0 }}
            style={{
              color: '#ffaa00',
              opacity: 0.9,
              size: 1.0,
            }}
          />
        ))}

      {/* Render visualizations from Elle */}
      {visualizations.map((viz) => {
        switch (viz.type) {
          case 'overlay':
            return <AROverlay key={viz.id} {...viz} />
          case 'highlight':
            return <ARHighlight key={viz.id} {...viz} />
          case 'path':
            return <ARPath key={viz.id} {...viz} />
          default:
            return null
        }
      })}

      {/* Development: Show coordinate axes */}
      {process.env.NODE_ENV === 'development' && (
        <axesHelper args={[1]} />
      )}
    </group>
  )
}

/**
 * Convert TensorFlow.js detected objects to AR context format
 */
function convertDetectedObjectsToARObjects(objects: DetectedObject[]): any[] {
  return objects.map((obj) => ({
    id: obj.id,
    label: obj.label,
    confidence: obj.confidence,
    bbox: obj.bbox,
    // Estimate 3D position from 2D bbox (simplified)
    position: {
      x: (obj.bbox.xMin + obj.bbox.xMax) / 2 - 0.5,
      y: 0.5,
      z: -1.5,
    },
    objectType: inferObjectType(obj.label),
  }))
}

/**
 * Infer object type from label (for Elle decision-making)
 */
function inferObjectType(label: string): string {
  const typeMap: Record<string, string> = {
    person: 'person',
    chair: 'furniture',
    couch: 'furniture',
    bed: 'furniture',
    dining_table: 'furniture',
    laptop: 'electronics',
    cell_phone: 'electronics',
    tv: 'electronics',
    bottle: 'container',
    cup: 'container',
    bowl: 'container',
    book: 'document',
    scissors: 'tool',
    knife: 'tool',
    spoon: 'utensil',
    fork: 'utensil',
  }
  return typeMap[label] || 'object'
}
