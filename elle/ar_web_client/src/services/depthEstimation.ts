/**
 * Depth Estimation Service - Browser-based monocular depth estimation
 *
 * Provides depth estimation for AR applications using browser-compatible models.
 * Uses ONNX Runtime Web to run MiDaS models in the browser.
 *
 * Features:
 * - MiDaS Small (fast, ~10MB, 30ms inference)
 * - Hardware acceleration (WebGL, WebGPU)
 * - Point cloud generation
 * - Depth-to-3D coordinate conversion
 *
 * Created: 2025-11-22 (Phase 4)
 */

import * as ort from 'onnxruntime-web'

// ============================================================================
// Types
// ============================================================================

export interface DepthMap {
  depth: Float32Array
  width: number
  height: number
  minDepth: number
  maxDepth: number
  unit: 'normalized' | 'meters' | 'disparity'
  timestamp: number
}

export interface Point3D {
  x: number
  y: number
  z: number
}

export interface DepthEstimationConfig {
  modelPath?: string // Path to ONNX model
  inputSize?: number // Input resolution (default: 256)
  normalize?: boolean // Normalize output to 0-1 (default: true)
  backend?: 'webgl' | 'webgpu' | 'wasm' // Execution provider
}

// ============================================================================
// Depth Estimation Service
// ============================================================================

export class DepthEstimationService {
  private session: ort.InferenceSession | null = null
  private initialized = false
  private inputSize: number
  private normalize: boolean
  private backend: 'webgl' | 'webgpu' | 'wasm'

  constructor(private config: DepthEstimationConfig = {}) {
    this.inputSize = config.inputSize ?? 256
    this.normalize = config.normalize ?? true
    this.backend = config.backend ?? 'webgl'
  }

  /**
   * Initialize ONNX Runtime and load MiDaS model
   */
  async initialize(): Promise<boolean> {
    if (this.initialized) return true

    try {
      // Configure ONNX Runtime
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/'

      // Set execution provider
      const executionProviders: ort.InferenceSession.ExecutionProviderConfig[] =
        this.backend === 'webgl'
          ? ['webgl']
          : this.backend === 'webgpu'
          ? ['webgpu']
          : ['wasm']

      // Load MiDaS model (using small variant for browser performance)
      const modelPath =
        this.config.modelPath ||
        'https://huggingface.co/isl-org/MiDaS/resolve/main/midas_v21_small_256.onnx'

      this.session = await ort.InferenceSession.create(modelPath, {
        executionProviders,
      })

      this.initialized = true
      console.log('✅ Depth estimation initialized (MiDaS Small)')
      return true
    } catch (error) {
      console.error('❌ Failed to initialize depth estimation:', error)
      return false
    }
  }

  /**
   * Estimate depth from image or video frame
   */
  async estimateDepth(
    input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement
  ): Promise<DepthMap | null> {
    if (!this.initialized || !this.session) {
      console.warn('Depth estimation not initialized')
      return null
    }

    try {
      // Prepare input tensor
      const inputTensor = await this.preprocessInput(input)

      // Run inference
      const feeds = { input: inputTensor }
      const results = await this.session.run(feeds)

      // Extract depth map
      const outputTensor = results.output
      const depthData = outputTensor.data as Float32Array

      // Get output dimensions
      const [batch, height, width] = outputTensor.dims

      // Normalize depth values
      let minDepth = Number.POSITIVE_INFINITY
      let maxDepth = Number.NEGATIVE_INFINITY

      for (let i = 0; i < depthData.length; i++) {
        minDepth = Math.min(minDepth, depthData[i])
        maxDepth = Math.max(maxDepth, depthData[i])
      }

      // Normalize to 0-1 if requested
      const processedDepth = new Float32Array(depthData.length)
      if (this.normalize) {
        const range = maxDepth - minDepth
        for (let i = 0; i < depthData.length; i++) {
          processedDepth[i] = (depthData[i] - minDepth) / range
        }
        minDepth = 0.0
        maxDepth = 1.0
      } else {
        processedDepth.set(depthData)
      }

      return {
        depth: processedDepth,
        width: width as number,
        height: height as number,
        minDepth,
        maxDepth,
        unit: this.normalize ? 'normalized' : 'disparity',
        timestamp: Date.now(),
      }
    } catch (error) {
      console.error('Depth estimation failed:', error)
      return null
    }
  }

  /**
   * Get depth value at specific pixel coordinates
   */
  getDepthAtPoint(depthMap: DepthMap, x: number, y: number): number | null {
    if (x < 0 || x >= depthMap.width || y < 0 || y >= depthMap.height) {
      return null
    }

    const index = y * depthMap.width + x
    return depthMap.depth[index]
  }

  /**
   * Convert 2D pixel + depth to 3D point using camera intrinsics
   *
   * Camera intrinsics:
   * fx, fy: Focal lengths (pixels)
   * cx, cy: Principal point (image center)
   */
  depthTo3DPoint(
    depthMap: DepthMap,
    x: number,
    y: number,
    fx: number,
    fy: number,
    cx: number,
    cy: number
  ): Point3D | null {
    const depth = this.getDepthAtPoint(depthMap, x, y)
    if (depth === null) return null

    // Back-project to 3D
    const Z = depth
    const X = ((x - cx) * Z) / fx
    const Y = ((y - cy) * Z) / fy

    return { x: X, y: Y, z: Z }
  }

  /**
   * Create 3D point cloud from depth map
   */
  createPointCloud(
    depthMap: DepthMap,
    fx: number,
    fy: number,
    cx?: number,
    cy?: number,
    stride: number = 4 // Sample every Nth pixel for performance
  ): Point3D[] {
    const centerX = cx ?? depthMap.width / 2
    const centerY = cy ?? depthMap.height / 2

    const points: Point3D[] = []

    for (let y = 0; y < depthMap.height; y += stride) {
      for (let x = 0; x < depthMap.width; x += stride) {
        const point = this.depthTo3DPoint(depthMap, x, y, fx, fy, centerX, centerY)
        if (point) {
          points.push(point)
        }
      }
    }

    return points
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    if (this.session) {
      // ONNX Runtime doesn't require explicit cleanup
      this.session = null
    }
    this.initialized = false
    console.log('🧹 Depth estimation cleaned up')
  }

  /**
   * Preprocess input for MiDaS model
   */
  private async preprocessInput(
    input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement
  ): Promise<ort.Tensor> {
    // Create canvas for preprocessing
    const canvas = document.createElement('canvas')
    canvas.width = this.inputSize
    canvas.height = this.inputSize
    const ctx = canvas.getContext('2d')!

    // Draw input to canvas (resized)
    ctx.drawImage(input, 0, 0, this.inputSize, this.inputSize)

    // Get image data
    const imageData = ctx.getImageData(0, 0, this.inputSize, this.inputSize)
    const { data } = imageData

    // Convert to RGB float32 tensor (NCHW format)
    const size = this.inputSize * this.inputSize
    const float32Data = new Float32Array(3 * size)

    // Normalize to [0, 1] and convert to NCHW
    for (let i = 0; i < size; i++) {
      float32Data[i] = data[i * 4] / 255 // R
      float32Data[size + i] = data[i * 4 + 1] / 255 // G
      float32Data[2 * size + i] = data[i * 4 + 2] / 255 // B
    }

    // Create tensor (batch=1, channels=3, height=inputSize, width=inputSize)
    return new ort.Tensor('float32', float32Data, [1, 3, this.inputSize, this.inputSize])
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let depthEstimationInstance: DepthEstimationService | null = null

/**
 * Get singleton depth estimation service
 */
export function getDepthEstimationService(
  config?: DepthEstimationConfig
): DepthEstimationService {
  if (!depthEstimationInstance) {
    depthEstimationInstance = new DepthEstimationService(config)
  }
  return depthEstimationInstance
}

/**
 * Reset singleton (for testing)
 */
export function resetDepthEstimationService(): void {
  if (depthEstimationInstance) {
    depthEstimationInstance.cleanup()
    depthEstimationInstance = null
  }
}
