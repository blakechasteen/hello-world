# HoloLoom Vision Tools - Comprehensive Computer Vision for AR

**Status**: ✅ Production Ready (Phase 2, 4, 5 - December 2025)
**Location**: `HoloLoom/vision/`
**Total Lines**: ~4,515 lines across 10 Python files
**Date Created**: November 22, 2025 (Phase 2)
**Last Updated**: December 2025 (Phases 4 & 5 integrated)

## Overview

HoloLoom Vision Tools provides a comprehensive, production-grade computer vision system for augmented reality applications. The system implements a phased architecture spanning object detection, scene understanding, hand tracking, depth estimation, marker detection, semantic segmentation, pose estimation, and visual SLAM.

The vision system is designed with **graceful degradation** as a core principle: every processor has fallback implementations (mock backends) that work even when optional dependencies (YOLOv8, MediaPipe, MiDaS) are unavailable. This ensures your AR applications never crash due to missing vision libraries - they degrade to mock detection gracefully.

### Key Features

**Phase 2 (Core Vision)**:
- **Object Detection**: YOLO (80 COCO classes) + TensorFlow.js COCO-SSD (frontend)
- **Scene Analysis**: Spatial relationships (8 types), lighting, dominant colors
- **Hand Tracking**: MediaPipe Hands (21 landmarks) with 6 gesture recognition

**Phase 4 (Depth & Markers)**:
- **Depth Estimation**: MiDaS/ZoeDepth monocular depth (32-512px resolution)
- **Marker Detection**: ArUco/QR/AprilTag with 6-DOF pose estimation

**Phase 5 (Advanced Vision)**:
- **Semantic Segmentation**: DeepLabV3/SegFormer (21-150 semantic classes)
- **Pose Estimation**: MediaPipe Pose (33 keypoints) full-body tracking
- **Visual SLAM**: ORB feature-based camera tracking with 6-DOF odometry

**Architecture**:
- **Protocol-Based**: All processors follow VisionProcessor protocol for swappable implementations
- **Dual Deployment**: Python backend + JavaScript frontend support
- **Real-Time**: Optimized for AR with <100ms latency
- **Graceful Degradation**: Mock backends ensure zero crashes on missing dependencies

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Vision Pipeline                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Camera Frame (RGB numpy array or Canvas)                    │
│         ↓                                                     │
│  ┌──────────────────────────────────────────────┐            │
│  │  1. Object Detection                          │            │
│  │     - YOLO (backend): 80 COCO classes         │            │
│  │     - COCO-SSD (frontend): lite_mobilenet_v2  │            │
│  │     - Output: DetectedObject[]                │            │
│  └──────────────────────────────────────────────┘            │
│         ↓                                                     │
│  ┌──────────────────────────────────────────────┐            │
│  │  2. Scene Analysis                            │            │
│  │     - Spatial relationships (8 types)         │            │
│  │     - Scene type classification               │            │
│  │     - Lighting analysis                       │            │
│  │     - Dominant color extraction               │            │
│  │     - Output: SceneUnderstanding              │            │
│  └──────────────────────────────────────────────┘            │
│         ↓                                                     │
│  ┌──────────────────────────────────────────────┐            │
│  │  3. Hand Tracking                             │            │
│  │     - MediaPipe Hands (21 landmarks)          │            │
│  │     - Gesture recognition (6 patterns)        │            │
│  │     - Left/right hand detection               │            │
│  │     - Output: HandPose[]                      │            │
│  └──────────────────────────────────────────────┘            │
│         ↓                                                     │
│  AR Context Update                                            │
│  {visibleObjects, handGestures, sceneType}                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Python Backend (HoloLoom/vision/)

### Installation

```bash
# Core dependencies
pip install numpy opencv-python pillow

# Object detection (YOLO)
pip install ultralytics torch torchvision

# Hand tracking (MediaPipe)
pip install mediapipe

# Scene analysis (optional)
pip install scikit-learn
```

### Quick Start

```python
from HoloLoom.vision import create_object_detector, create_scene_analyzer, create_hand_tracker
import numpy as np
import cv2

# Initialize vision processors
object_detector = create_object_detector(backend="yolo")
await object_detector.initialize()

scene_analyzer = create_scene_analyzer()
await scene_analyzer.initialize()

hand_tracker = create_hand_tracker(backend="mediapipe")
await hand_tracker.initialize()

# Process camera frame
frame = cv2.imread("workshop.jpg")  # RGB numpy array

# Detect objects
objects = await object_detector.detect_objects(frame, confidence_threshold=0.5)
for obj in objects:
    print(f"Found {obj.label} at ({obj.bbox.x_min}, {obj.bbox.y_min}) with {obj.confidence:.2f} confidence")

# Analyze scene
scene = await scene_analyzer.analyze_scene(frame, objects)
print(f"Scene type: {scene.scene_type}")
print(f"Lighting: {scene.lighting}")
print(f"Relationships: {len(scene.relationships)}")

# Track hands
hands = await hand_tracker.track_hands(frame)
for hand in hands:
    print(f"{hand.hand_id} hand: {hand.gesture} gesture")
```

### Object Detection

**Backends**:
- `yolo`: YOLOv8n (6.3MB model, GPU-accelerated)
- `mock`: Testing backend (no dependencies)

**COCO Classes** (80 total):
```python
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
```

**Performance**:
- YOLO inference: ~50ms per frame (GPU)
- YOLO inference: ~200ms per frame (CPU)
- Non-maximum suppression: ~5ms

### Scene Analysis

**Scene Types**:
- workshop, kitchen, office, living_room, bedroom, bathroom, garage, outdoor, warehouse, unknown

**Spatial Relationships** (8 types):
```python
RelationshipTypes = [
    "left_of",      # Object A is to the left of B
    "right_of",     # Object A is to the right of B
    "above",        # Object A is above B
    "below",        # Object A is below B
    "near",         # Object A is near B (Euclidean distance < threshold)
    "contains",     # Object A contains B (B's bbox inside A's bbox)
    "on_top_of",    # Object A is on top of B (vertical proximity + contact)
    "inside",       # Object A is inside B (inverse of contains)
]
```

**Lighting Conditions**:
- bright, dim, backlit, normal

**Dominant Colors**:
- K-means clustering (3 clusters)
- Returns RGB values as `List[Tuple[int, int, int]]`

### Hand Tracking

**Gestures Recognized** (6 patterns):
```python
Gestures = {
    "point":       # Index finger extended, others closed
    "grab":        # All fingers closed (fist)
    "open_palm":   # All fingers extended
    "pinch":       # Thumb and index touching
    "thumbs_up":   # Thumb pointing up, others closed
    "peace":       # Index and middle extended, others closed
}
```

**Landmarks** (21 per hand):
```
WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP,
INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_DIP, INDEX_FINGER_TIP,
MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP, MIDDLE_FINGER_TIP,
RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_DIP, RING_FINGER_TIP,
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP
```

**Helper Functions**:
```python
from HoloLoom.vision.hand_tracker import get_pointing_direction, get_pinch_strength

# Get 3D pointing direction for navigation
direction = get_pointing_direction(hand_pose)  # Returns Vector3 or None

# Get pinch strength (0.0 = open, 1.0 = fully pinched)
strength = get_pinch_strength(hand_pose)  # Returns float 0.0-1.0
```

---

## TypeScript Frontend (elle/ar_web_client/)

### Installation

```bash
cd elle/ar_web_client
npm install

# Dependencies automatically installed:
# - @tensorflow/tfjs
# - @tensorflow/tfjs-backend-webgl
# - @tensorflow-models/coco-ssd
# - @mediapipe/hands
# - @mediapipe/camera_utils
```

### Object Detection (TensorFlow.js)

```typescript
import { getObjectDetectionService } from './services'

// Initialize (one-time, loads 5.4MB model)
const detector = getObjectDetectionService()
await detector.initialize()

// Detect from canvas (e.g., WebXR camera frame)
const canvas = document.querySelector('canvas')
const objects = await detector.detectObjects(
  canvas,
  20,    // maxDetections
  0.5    // minConfidence
)

// Results
objects.forEach(obj => {
  console.log(`${obj.label} (${(obj.confidence * 100).toFixed(0)}%)`)
  console.log(`  bbox: ${obj.bbox.xMin}, ${obj.bbox.yMin} → ${obj.bbox.xMax}, ${obj.bbox.yMax}`)
})

// Cleanup
await detector.cleanup()
```

**Performance**:
- Model load: ~2-3s (cached after first load)
- Inference: ~30-50ms per frame (WebGL GPU)
- Model: lite_mobilenet_v2 (5.4MB)

### Hand Tracking (MediaPipe)

```typescript
import { getHandTrackingService, Gesture, getPointingDirection } from './services'

// Initialize
const handTracker = getHandTrackingService({
  maxHands: 2,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.5,
  modelComplexity: 0,  // 0 = lite (faster), 1 = full (more accurate)
})
await handTracker.initialize()

// Start tracking from video element
const video = document.querySelector('video')
await handTracker.startTracking(video, (hands) => {
  hands.forEach(hand => {
    console.log(`${hand.handId} hand: ${hand.gesture}`)

    if (hand.gesture === Gesture.POINT) {
      const direction = getPointingDirection(hand)
      console.log(`Pointing at: ${direction.x}, ${direction.y}, ${direction.z}`)
    }
  })
})

// Stop tracking
handTracker.stop()

// Cleanup
await handTracker.cleanup()
```

**Performance**:
- Model load: ~1-2s (CDN cached)
- Inference: ~20-30ms per frame (WebGL)
- Hand tracking latency: <50ms total

### AR Scene Integration

The `ARScene` component automatically integrates vision services:

```typescript
<ARScene
  visualizations={visualizations}
  onContextUpdate={handleContextUpdate}
  enableVision={true}              // Enable vision processing
  visionUpdateInterval={100}       // Process every 100ms (10 FPS)
/>
```

**Vision Output**:
```typescript
interface ARContext {
  // ... position, rotation, gaze
  visibleObjects: Array<{
    id: string
    label: string
    confidence: number
    bbox: BoundingBox
    position: Vector3  // Estimated 3D position
  }>
  handGestures: Array<{
    handId: string     // "left" or "right"
    gesture: string    // "point", "grab", etc.
    confidence: number
  }>
}
```

---

## Backend API Endpoints

The FastAPI server exposes vision processing endpoints:

### POST /ar/vision/detect_objects

Detect objects in uploaded image.

**Request**:
```bash
curl -X POST http://localhost:8000/ar/vision/detect_objects \
  -F "file=@workshop.jpg"
```

**Response**:
```json
{
  "objects": [
    {
      "id": "obj_1_1732234567890",
      "label": "scissors",
      "confidence": 0.87,
      "bbox": {"xMin": 0.34, "yMin": 0.56, "xMax": 0.45, "yMax": 0.78},
      "classId": 76
    }
  ],
  "count": 1,
  "processing_time_ms": 52.3
}
```

### POST /ar/vision/analyze_scene

Analyze scene for spatial understanding.

**Request**:
```bash
curl -X POST http://localhost:8000/ar/vision/analyze_scene \
  -F "file=@workshop.jpg"
```

**Response**:
```json
{
  "scene_type": "workshop",
  "objects": [...],
  "relationships": [
    {
      "object1": "obj_1",
      "object2": "obj_2",
      "relationship": "near"
    }
  ],
  "spatial_layout": {
    "object_count": 5,
    "density": 0.42,
    "distribution": "clustered"
  },
  "lighting": "bright",
  "dominant_colors": [[180, 120, 90], [220, 210, 200], [45, 35, 25]],
  "processing_time_ms": 125.7
}
```

### POST /ar/vision/track_hands

Track hands and recognize gestures.

**Request**:
```bash
curl -X POST http://localhost:8000/ar/vision/track_hands \
  -F "file=@hand_gesture.jpg"
```

**Response**:
```json
{
  "hands": [
    {
      "handId": "right",
      "gesture": "point",
      "confidence": 0.9,
      "landmarks": [
        {"x": 0.5, "y": 0.6, "z": -0.1},
        ...  // 21 landmarks total
      ]
    }
  ],
  "count": 1,
  "processing_time_ms": 43.2
}
```

---

## Key Components

| Component | Lines | Purpose | Phase |
|-----------|-------|---------|-------|
| **protocol.py** | 448 | Core data models, protocols, vision pipeline | 2-5 |
| **object_detector.py** | 341 | YOLO/COCO-SSD object detection (80 classes) | 2 |
| **scene_analyzer.py** | 440 | Scene understanding, spatial relationships | 2 |
| **hand_tracker.py** | 481 | MediaPipe hand tracking, 21 landmarks | 2 |
| **depth_estimator.py** | 395 | MiDaS/ZoeDepth depth estimation | 4 |
| **marker_detector.py** | 478 | ArUco/QR/AprilTag detection, 6-DOF pose | 4 |
| **semantic_segmenter.py** | 559 | DeepLabV3/SegFormer segmentation | 5 |
| **pose_estimator.py** | 590 | MediaPipe pose estimation (33 keypoints) | 5 |
| **slam_processor.py** | 604 | Visual SLAM, ORB features, odometry | 5 |
| **__init__.py** | 179 | Module exports, public API | 2-5 |

**Total**: ~4,515 lines of production code

## Phase 4: Depth Estimation & Marker Detection

### Depth Estimation (depth_estimator.py)

**Backends**:
- `midas` (default): MiDaS v2.1 (~100ms GPU), lightweight, robust
- `zoedepth`: ZoeDepth (~150ms GPU), higher accuracy

**Quick Start**:
```python
from HoloLoom.vision import create_depth_estimator, depth_to_3d_point, create_point_cloud

estimator = create_depth_estimator(backend="midas")
await estimator.initialize()

# Estimate depth
depth_map = await estimator.estimate_depth(frame)

# Convert to 3D point
point_3d = depth_to_3d_point(
    x_pixel=320, y_pixel=240,
    depth_value=depth_map.depth[240, 320],
    camera_matrix=camera_matrix
)

# Create point cloud
point_cloud = create_point_cloud(depth_map)
```

**Performance**:
- MiDaS: ~50-100ms per frame (GPU), ~1-2s (CPU)
- ZoeDepth: ~100-150ms per frame (GPU), ~3-5s (CPU)

### Marker Detection (marker_detector.py)

**Supported Markers**:
- **ArUco**: 4x4, 5x5, 6x6, OpenCV dictionaries
- **QR Codes**: Any size, variable data capacity
- **AprilTags**: 16h5, 25h9, 36h11

**Features**:
- 6-DOF pose estimation (position + rotation in 3D)
- Multiple marker types in single image
- Decoded data extraction (QR codes)

**Quick Start**:
```python
from HoloLoom.vision import create_marker_detector

detector = create_marker_detector(
    marker_types=["aruco", "qr_code", "apriltag"]
)
await detector.initialize()

markers = await detector.detect_markers(frame)

for marker in markers:
    print(f"Marker: {marker.id} ({marker.marker_type})")
    if marker.position:
        print(f"  3D Position: ({marker.position[0]:.2f}m, "
              f"{marker.position[1]:.2f}m, {marker.position[2]:.2f}m)")
```

**6-DOF Pose**: Each marker provides:
- **Position**: (x, y, z) in meters relative to camera
- **Rotation**: (rx, ry, rz) rotation vector in radians

## Phase 5: Advanced Vision (Segmentation, Pose, SLAM)

### Semantic Segmentation (semantic_segmenter.py)

**Models**:
- **DeepLabV3-ResNet50**: ~80ms GPU, 21 COCO classes
- **DeepLabV3-ResNet101**: ~120ms GPU, 150 ADE20K classes
- **SegFormer-B0**: ~40ms GPU, lightweight
- **SegFormer-B5**: ~200ms GPU, high accuracy

**Datasets**:
- COCO (21 classes): person, car, dog, etc.
- ADE20K (150 classes): fine-grained scene categories
- Cityscapes (19 classes): street scenes

**Quick Start**:
```python
from HoloLoom.vision import create_semantic_segmenter, visualize_segmentation

segmenter = create_semantic_segmenter(
    model="deeplabv3_resnet50",
    dataset="coco"
)
await segmenter.initialize()

mask = await segmenter.segment_image(frame)

# Get specific class mask
person_mask = mask.get_class_mask(class_id=15)

# Visualize
visualization = visualize_segmentation(frame, mask)
```

### Pose Estimation (pose_estimator.py)

**Features**:
- 33 keypoints covering full body
- 3 complexity levels (LITE/FULL/HEAVY)
- World coordinates available
- Gesture classification

**Keypoints** (33 total):
- 0: Nose
- 1-10: Face (eyes, ears, mouth)
- 11-14: Torso
- 15-22: Arms
- 23-28: Legs
- 29-32: Advanced facial features

**Quick Start**:
```python
from HoloLoom.vision import create_pose_estimator, get_joint_angle, detect_gesture

estimator = create_pose_estimator(model_complexity=1)
await estimator.initialize()

pose = await estimator.estimate_pose(frame)

# Get joint angle
angle = get_joint_angle(
    pose.keypoints[11],  # Shoulder
    pose.keypoints[13],  # Elbow
    pose.keypoints[15]   # Wrist
)

# Detect gesture
gesture = detect_gesture(pose)  # "standing", "sitting", "running"
```

### Visual SLAM (slam_processor.py)

**Capabilities**:
- Camera pose tracking (6-DOF)
- Feature detection and matching (ORB features)
- Essential matrix estimation
- Map point triangulation
- Loop closure detection

**Quick Start**:
```python
from HoloLoom.vision import create_slam_processor, create_camera_matrix

slam = create_slam_processor(
    camera_matrix=create_camera_matrix(fx=525, fy=525, cx=320, cy=240),
    tracking_quality="balanced"
)
await slam.initialize()

# Process frames
pose1 = await slam.track_frame(frame1)
pose2 = await slam.track_frame(frame2)

# Result: SLAMPose with 6-DOF camera position + orientation
```

**Output** (SLAMPose):
- **position**: (x, y, z) in meters
- **orientation**: (x, y, z, w) quaternion
- **tracking_quality**: 0.0-1.0 confidence
- **num_features**: tracked feature count

### TypeScript Frontend

| File | Lines | Purpose |
|------|-------|---------|
| [objectDetection.ts](../../elle/ar_web_client/src/services/objectDetection.ts) | 120 | TensorFlow.js COCO-SSD service |
| [handTracking.ts](../../elle/ar_web_client/src/services/handTracking.ts) | 380 | MediaPipe Hands service |
| [index.ts](../../elle/ar_web_client/src/services/index.ts) | 30 | Service exports |

**Total**: ~530 lines

### Integration

| File | Lines | Purpose |
|------|-------|---------|
| [ARScene.tsx](../../elle/ar_web_client/src/components/ARScene.tsx) | 280 | Vision-integrated AR scene (updated) |
| [ar_api.py](../server/ar_api.py) | +160 | Vision endpoints added to FastAPI server |

**Total Phase 2**: ~2,240 lines of production code

---

## Performance Characteristics

### Phase 2 (Core Vision)

| Component | Latency | GPU Memory | Notes |
|-----------|---------|-----------|-------|
| **Object Detection (YOLO)** | 30-100ms | 1-4GB | Real-time capable |
| **Object Detection (COCO-SSD)** | 40-150ms | 200-500MB | Faster on CPU, web-capable |
| **Scene Analysis** | 20-50ms | 100-200MB | Fast spatial reasoning |
| **Hand Tracking** | 5-20ms | 100-200MB | Real-time, both hands |

### Phase 4 (Depth & Markers)

| Component | Latency | GPU Memory | Notes |
|-----------|---------|-----------|-------|
| **Depth Estimation (MiDaS)** | 50-100ms | 2-4GB | MiDaS small ~50ms, v2.1 ~100ms |
| **Depth Estimation (ZoeDepth)** | 100-150ms | 4-6GB | Higher accuracy |
| **Marker Detection** | 10-30ms | 50-100MB | Fast, real-time |

### Phase 5 (Advanced Vision)

| Component | Latency | GPU Memory | Notes |
|-----------|---------|-----------|-------|
| **Semantic Segmentation** | 40-200ms | 1-8GB | Depends on model and resolution |
| **Pose Estimation** | 25-150ms | 500MB-2GB | Model complexity dependent |
| **Visual SLAM** | 30-100ms | 500MB-1GB | Feature tracking, pose estimation |

### Complete Pipeline

| Configuration | Latency | GPU Memory |
|---------------|---------|-----------|
| **Phase 2 Only** | 100-150ms | 2-5GB |
| **Phase 2 + Phase 4** | 150-250ms | 6-10GB |
| **All Phases (2+4+5)** | 250-400ms | 8-16GB |

**Optimization Tips**:
- Use LITE/FAST models for real-time (>30fps)
- Reduce input resolution (320x240) for faster inference
- Enable batch processing for multiple frames
- Use GPU for significant speedup (10-20x vs CPU)
- YOLOv8 nano (n) is fastest, v8x is highest accuracy

---

## Testing

### Python Tests

```bash
# Run all vision tests
pytest HoloLoom/vision/tests/ -v

# Test object detection
pytest HoloLoom/vision/tests/test_object_detector.py -v

# Test scene analysis
pytest HoloLoom/vision/tests/test_scene_analyzer.py -v

# Test hand tracking
pytest HoloLoom/vision/tests/test_hand_tracker.py -v
```

### Frontend Tests

```bash
cd elle/ar_web_client
npm test

# Test object detection service
npm test -- objectDetection

# Test hand tracking service
npm test -- handTracking
```

---

## Troubleshooting

### Backend Issues

**YOLO not loading**:
```python
# Check if model file exists
ls ~/.ultralytics/models/yolov8n.pt  # Should be 6.3MB

# Force re-download
rm -rf ~/.ultralytics/models/yolov8n.pt
# Model will auto-download on next run
```

**MediaPipe not found**:
```bash
pip install mediapipe
# If fails, try:
pip install mediapipe --no-cache-dir
```

**Slow CPU inference**:
- YOLO defaults to CPU if GPU unavailable
- Install CUDA-enabled PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Frontend Issues

**TensorFlow.js not loading models**:
- Check network connection (models downloaded from CDN)
- Clear browser cache
- Check browser console for CORS errors

**MediaPipe hands not initializing**:
- Ensure HTTPS connection (required for WebXR)
- Check `locateFile` CDN path in console
- Verify WebGL support: chrome://gpu

**Poor frame rate**:
- Reduce `visionUpdateInterval` (default: 100ms = 10 FPS)
- Lower MediaPipe `modelComplexity` to 0 (lite)
- Reduce COCO-SSD `maxDetections` to 10

---

## Integration with HoloLoom

The vision system integrates seamlessly with HoloLoom's weaving orchestrator and memory systems:

```python
from HoloLoom import HoloLoom
from HoloLoom.vision import VisionPipeline, create_object_detector, create_scene_analyzer

# Create HoloLoom instance with vision
async with HoloLoom() as loom:
    # Create vision pipeline
    vision = VisionPipeline(
        object_detector=create_object_detector(),
        scene_analyzer=create_scene_analyzer()
    )

    # Process frame
    result = await vision.process(frame)

    # Store vision results in memory
    for obj in result.objects:
        await loom.experience(f"Detected {obj.label} at ({obj.bbox.center()})")

    # Query based on vision
    memories = await loom.recall(f"What objects did I see?")
```

**Integration Points**:
- **Memory Storage**: Vision results can be persisted to knowledge graph
- **Scene Understanding**: Integrate with spatial reasoning
- **Temporal Tracking**: SLAM provides camera trajectory for temporal context
- **Gesture Control**: Hand tracking enables gesture-based interaction
- **Semantic Search**: Segmentation enables semantic scene queries

## When to Use

**✅ Use HoloLoom Vision when you need**:
- Real-time object detection in AR (YOLOv8 backend)
- Hand gesture recognition for interaction
- 6-DOF marker tracking for spatial anchoring
- Scene understanding for context-aware AR
- Depth estimation for spatial layout
- Full-body pose estimation for avatar control
- Semantic scene understanding with segmentation
- Visual SLAM for camera tracking and mapping
- Complete vision pipeline with multiple processors

**✅ Use Specific Processors when**:
- **ObjectDetector**: General object recognition (people, objects, vehicles)
- **HandTracker**: Gesture-based UI, hand pose capture
- **DepthEstimator**: 3D reconstruction, spatial layout, proximity detection
- **MarkerDetector**: Spatial anchoring, fiducial-based AR, marker-following
- **SemanticSegmenter**: Scene parsing, semantic understanding, region extraction
- **PoseEstimator**: Full-body tracking, animation, gesture analysis
- **SLAMProcessor**: Camera tracking, visual odometry, loop closure

**🟡 Consider alternatives when**:
- Only need simple classification (use lightweight models like MobileNet)
- Real-time latency is <16ms requirement (use simplified detection)
- Running on embedded devices (use quantized/pruned models)
- Only need 2D tracking without 3D (skip depth estimation)
- No hand tracking needed (disable HandTracker)

**❌ Don't use Vision when**:
- Latency <10ms is critical (vision pipelines inherently slow)
- No camera input available
- Privacy prevents video processing
- Static images only (SLAM and tracking need sequences)
- Thermal or infrared imaging (requires specialized models)

## Roadmap

**Completed** (November-December 2025):
- ✅ Phase 2: Object detection, hand tracking, scene analysis
- ✅ Phase 4: Depth estimation, marker detection
- ✅ Phase 5: Semantic segmentation, pose estimation, SLAM

**Planned** (2026+):
- 🔵 Real-time face recognition (Phase 6)
- 🔵 3D pose reconstruction (Phase 7)
- 🔵 Optical flow for motion analysis (Phase 8)
- 🔵 Instance segmentation (Phase 9)
- 🔵 Panoptic segmentation (Phase 10)

---

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [MediaPipe Hands Guide](https://google.github.io/mediapipe/solutions/hands.html)
- [TensorFlow.js Models](https://github.com/tensorflow/tfjs-models)
- [COCO Dataset Classes](https://cocodataset.org/#explore)
- [WebXR Hand Input](https://immersive-web.github.io/webxr-hand-input/)

---

**Documentation Status**:
- ✅ Phase 2 (Core Vision): Complete - Object Detection, Scene Analysis, Hand Tracking
- ✅ Phase 4 (Depth & Markers): Complete - Depth Estimation, Marker Detection
- ✅ Phase 5 (Advanced Vision): Complete - Segmentation, Pose Estimation, SLAM

**Created**: 2025-11-22 (Phase 2 - Vision Tools)
**Updated**: December 2025 (Phases 4 & 5 integrated)
**Total Code**: ~4,515 lines across 10 Python files
**Author**: HoloLoom AR Team
**License**: MIT
