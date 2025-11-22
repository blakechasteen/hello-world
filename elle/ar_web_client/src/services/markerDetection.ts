/**
 * Marker Detection Service - ArUco and QR code detection for browser
 *
 * Provides fiducial marker detection for precise AR tracking and anchoring.
 * Uses opencv.js for ArUco markers and jsQR for QR code detection.
 *
 * Features:
 * - ArUco marker detection (4x4_50, 5x5_100, 6x6_250 dictionaries)
 * - QR code detection and decoding
 * - 6-DOF pose estimation (with camera calibration)
 * - Multiple marker tracking
 *
 * Created: 2025-11-22 (Phase 4)
 */

import jsQR from 'jsqr'

// ============================================================================
// Types
// ============================================================================

export interface Marker {
  id: string // Marker ID or decoded QR data
  markerType: 'aruco' | 'qr_code'
  corners: [number, number][] // 4 corner points [[x, y], ...]
  center: [number, number] // Center point [x, y]
  position?: [number, number, number] // 3D position [x, y, z] in meters
  rotation?: [number, number, number] // Rotation vector [rx, ry, rz]
  data?: string // Decoded data (for QR codes)
  confidence: number
  timestamp: number
}

export interface MarkerDetectionConfig {
  markerType?: 'aruco' | 'qr_code' // Default: aruco
  arucoDictionary?: 'DICT_4X4_50' | 'DICT_5X5_100' | 'DICT_6X6_250' // Default: DICT_4X4_50
  markerSizeMm?: number // Physical marker size in mm (default: 100)
  enablePose?: boolean // Estimate 6-DOF pose (default: false, requires camera calibration)
  minMarkerPerimeter?: number // Minimum marker perimeter in pixels (default: 80)
}

export interface CameraCalibration {
  fx: number // Focal length X
  fy: number // Focal length Y
  cx: number // Principal point X
  cy: number // Principal point Y
  distortion?: number[] // Distortion coefficients
}

// ============================================================================
// Marker Detection Service
// ============================================================================

export class MarkerDetectionService {
  private cv: any = null
  private initialized = false
  private markerType: 'aruco' | 'qr_code'
  private arucoDictionary: string
  private markerSizeMm: number
  private enablePose: boolean
  private minMarkerPerimeter: number
  private cameraCalibration: CameraCalibration | null = null

  constructor(private config: MarkerDetectionConfig = {}) {
    this.markerType = config.markerType ?? 'aruco'
    this.arucoDictionary = config.arucoDictionary ?? 'DICT_4X4_50'
    this.markerSizeMm = config.markerSizeMm ?? 100
    this.enablePose = config.enablePose ?? false
    this.minMarkerPerimeter = config.minMarkerPerimeter ?? 80
  }

  /**
   * Initialize marker detection (loads opencv.js)
   */
  async initialize(): Promise<boolean> {
    if (this.initialized) return true

    try {
      if (this.markerType === 'aruco') {
        // Load opencv.js from CDN
        await this.loadOpenCV()
        console.log('✅ ArUco marker detection initialized')
      } else {
        // QR code detection uses jsQR (no initialization needed)
        console.log('✅ QR code detection initialized')
      }

      this.initialized = true
      return true
    } catch (error) {
      console.error('❌ Failed to initialize marker detection:', error)
      return false
    }
  }

  /**
   * Set camera calibration for pose estimation
   */
  setCameraCalibration(calibration: CameraCalibration): void {
    this.cameraCalibration = calibration
    console.log('Camera calibration set for pose estimation')
  }

  /**
   * Detect markers in image or video frame
   */
  async detectMarkers(
    input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement
  ): Promise<Marker[]> {
    if (!this.initialized) {
      console.warn('Marker detection not initialized')
      return []
    }

    if (this.markerType === 'aruco') {
      return this.detectArUcoMarkers(input)
    } else {
      return this.detectQRCodes(input)
    }
  }

  /**
   * Detect ArUco markers using opencv.js
   */
  private detectArUcoMarkers(
    input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement
  ): Marker[] {
    if (!this.cv) return []

    const markers: Marker[] = []

    try {
      // Convert input to cv.Mat
      const canvas = this.toCanvas(input)
      const src = this.cv.imread(canvas)
      const gray = new this.cv.Mat()

      // Convert to grayscale
      this.cv.cvtColor(src, gray, this.cv.COLOR_RGBA2GRAY)

      // Create ArUco dictionary
      const dictionary = this.cv.aruco.getPredefinedDictionary(
        this.cv.aruco[this.arucoDictionary]
      )

      // Detect markers
      const corners = new this.cv.MatVector()
      const ids = new this.cv.Mat()
      const rejected = new this.cv.MatVector()

      const detectorParams = new this.cv.aruco.DetectorParameters()
      detectorParams.minMarkerPerimeterRate = this.minMarkerPerimeter / 1000.0

      this.cv.aruco.detectMarkers(gray, dictionary, corners, ids, detectorParams, rejected)

      // Process detected markers
      if (ids.rows > 0) {
        for (let i = 0; i < ids.rows; i++) {
          const markerId = ids.intAt(i, 0)
          const cornerMat = corners.get(i)

          // Extract corner points (4 corners per marker)
          const markerCorners: [number, number][] = []
          for (let j = 0; j < 4; j++) {
            markerCorners.push([cornerMat.floatAt(0, j * 2), cornerMat.floatAt(0, j * 2 + 1)])
          }

          // Calculate center
          const centerX =
            markerCorners.reduce((sum, corner) => sum + corner[0], 0) / 4
          const centerY =
            markerCorners.reduce((sum, corner) => sum + corner[1], 0) / 4

          // Estimate pose if camera calibration available
          let position: [number, number, number] | undefined
          let rotation: [number, number, number] | undefined

          if (this.enablePose && this.cameraCalibration) {
            const pose = this.estimateArUcoPose(markerCorners, this.cameraCalibration)
            if (pose) {
              position = pose.position
              rotation = pose.rotation
            }
          }

          markers.push({
            id: markerId.toString(),
            markerType: 'aruco',
            corners: markerCorners,
            center: [centerX, centerY],
            position,
            rotation,
            confidence: 1.0, // ArUco doesn't provide confidence
            timestamp: Date.now(),
          })
        }
      }

      // Cleanup
      src.delete()
      gray.delete()
      corners.delete()
      ids.delete()
      rejected.delete()
    } catch (error) {
      console.error('ArUco detection failed:', error)
    }

    return markers
  }

  /**
   * Detect QR codes using jsQR
   */
  private detectQRCodes(
    input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement
  ): Marker[] {
    const markers: Marker[] = []

    try {
      // Convert to canvas
      const canvas = this.toCanvas(input)
      const ctx = canvas.getContext('2d')!
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

      // Detect QR code
      const code = jsQR(imageData.data, imageData.width, imageData.height)

      if (code) {
        // Extract corners
        const corners: [number, number][] = [
          [code.location.topLeftCorner.x, code.location.topLeftCorner.y],
          [code.location.topRightCorner.x, code.location.topRightCorner.y],
          [code.location.bottomRightCorner.x, code.location.bottomRightCorner.y],
          [code.location.bottomLeftCorner.x, code.location.bottomLeftCorner.y],
        ]

        // Calculate center
        const centerX = corners.reduce((sum, corner) => sum + corner[0], 0) / 4
        const centerY = corners.reduce((sum, corner) => sum + corner[1], 0) / 4

        markers.push({
          id: code.data,
          markerType: 'qr_code',
          corners,
          center: [centerX, centerY],
          data: code.data,
          confidence: 1.0,
          timestamp: Date.now(),
        })
      }
    } catch (error) {
      console.error('QR code detection failed:', error)
    }

    return markers
  }

  /**
   * Estimate 6-DOF pose from ArUco marker corners
   * (Simplified implementation - full pose estimation requires opencv.js solvePnP)
   */
  private estimateArUcoPose(
    corners: [number, number][],
    calibration: CameraCalibration
  ): { position: [number, number, number]; rotation: [number, number, number] } | null {
    // This is a simplified estimation
    // Full implementation would use cv.solvePnP for accurate pose

    try {
      // Estimate distance based on marker size
      const markerSizeMeters = this.markerSizeMm / 1000.0
      const pixelWidth = Math.abs(corners[1][0] - corners[0][0])
      const estimatedDistance = (calibration.fx * markerSizeMeters) / pixelWidth

      // Estimate position (simplified - assumes marker is parallel to camera)
      const centerX = corners.reduce((sum, corner) => sum + corner[0], 0) / 4
      const centerY = corners.reduce((sum, corner) => sum + corner[1], 0) / 4

      const x = ((centerX - calibration.cx) * estimatedDistance) / calibration.fx
      const y = ((centerY - calibration.cy) * estimatedDistance) / calibration.fy
      const z = estimatedDistance

      // Simplified rotation (assumes no rotation for now)
      const rotation: [number, number, number] = [0, 0, 0]

      return {
        position: [x, y, z],
        rotation,
      }
    } catch (error) {
      console.error('Pose estimation failed:', error)
      return null
    }
  }

  /**
   * Convert input to canvas
   */
  private toCanvas(
    input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement
  ): HTMLCanvasElement {
    if (input instanceof HTMLCanvasElement) {
      return input
    }

    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')!

    if (input instanceof HTMLVideoElement) {
      canvas.width = input.videoWidth
      canvas.height = input.videoHeight
    } else {
      canvas.width = input.width
      canvas.height = input.height
    }

    ctx.drawImage(input, 0, 0)
    return canvas
  }

  /**
   * Load opencv.js from CDN
   */
  private async loadOpenCV(): Promise<void> {
    return new Promise((resolve, reject) => {
      if ((window as any).cv) {
        this.cv = (window as any).cv
        resolve()
        return
      }

      const script = document.createElement('script')
      script.src = 'https://docs.opencv.org/4.5.2/opencv.js'
      script.async = true

      script.onload = () => {
        // Wait for opencv.js to initialize
        const checkOpenCV = () => {
          if ((window as any).cv && (window as any).cv.aruco) {
            this.cv = (window as any).cv
            resolve()
          } else {
            setTimeout(checkOpenCV, 100)
          }
        }
        checkOpenCV()
      }

      script.onerror = () => {
        reject(new Error('Failed to load opencv.js'))
      }

      document.head.appendChild(script)
    })
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    this.cv = null
    this.initialized = false
    console.log('🧹 Marker detection cleaned up')
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Create default camera calibration (simplified pinhole model)
 */
export function createDefaultCameraCalibration(
  width: number,
  height: number
): CameraCalibration {
  // Assume focal length = width (typical for many cameras)
  const fx = width
  const fy = width
  const cx = width / 2
  const cy = height / 2

  return { fx, fy, cx, cy }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let markerDetectionInstance: MarkerDetectionService | null = null

/**
 * Get singleton marker detection service
 */
export function getMarkerDetectionService(
  config?: MarkerDetectionConfig
): MarkerDetectionService {
  if (!markerDetectionInstance) {
    markerDetectionInstance = new MarkerDetectionService(config)
  }
  return markerDetectionInstance
}

/**
 * Reset singleton (for testing)
 */
export function resetMarkerDetectionService(): void {
  if (markerDetectionInstance) {
    markerDetectionInstance.cleanup()
    markerDetectionInstance = null
  }
}
