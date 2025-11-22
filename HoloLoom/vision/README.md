# HoloLoom Vision Tools - Phase 2

**Status**: ✅ Complete (2025-11-22)
**Integration**: AR UX/UI System for Elle + HoloLoom
**Performance**: Real-time processing at 10 FPS (100ms intervals)

## Overview

The HoloLoom Vision Tools provide real-time computer vision capabilities for AR applications, enabling spatial understanding, object detection, and gesture recognition. The system integrates seamlessly with the Elle AR adapter and supports both Python (backend) and TypeScript (frontend) implementations.

### Key Features

- **Object Detection**: YOLO (backend) + TensorFlow.js COCO-SSD (frontend)
- **Scene Analysis**: Spatial relationships, lighting, dominant colors
- **Hand Tracking**: MediaPipe Hands with 6 gesture recognition patterns
- **Protocol-Based**: Swappable implementations with graceful degradation
- **Dual Deployment**: Python backend + JavaScript frontend
- **Real-Time**: Optimized for AR with <100ms latency

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

## Files

### Python Backend

| File | Lines | Purpose |
|------|-------|---------|
| [protocol.py](protocol.py) | 340 | Protocol definitions and data models |
| [object_detector.py](object_detector.py) | 250 | YOLO + mock object detection |
| [scene_analyzer.py](scene_analyzer.py) | 280 | Scene analysis and spatial relationships |
| [hand_tracker.py](hand_tracker.py) | 350 | MediaPipe hand tracking + gestures |
| [\_\_init\_\_.py](__init__.py) | 50 | Factory functions and exports |

**Total**: ~1,270 lines

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

| Component | Backend (Python) | Frontend (TypeScript) |
|-----------|------------------|----------------------|
| **Object Detection** | ~50ms (GPU), ~200ms (CPU) | ~30-50ms (WebGL) |
| **Scene Analysis** | ~80ms | N/A (backend only) |
| **Hand Tracking** | ~40ms | ~20-30ms (WebGL) |
| **Model Size** | YOLO: 6.3MB | COCO-SSD: 5.4MB, MediaPipe Hands: ~1.5MB |
| **Initialization** | ~2-3s | ~2-4s (model download + WebGL compile) |

**Total Latency** (AR scene at 10 FPS):
- Vision processing: ~100ms per iteration (throttled)
- AR context update: <10ms
- Total: ~110ms per cycle

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

## Future Enhancements

**Phase 3 Roadmap**:
- Depth estimation (MiDaS, ZoeDepth)
- Marker detection (ArUco, QR codes)
- Semantic segmentation (Mask R-CNN)
- Pose estimation (MediaPipe Pose)
- Spatial mapping integration

---

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [MediaPipe Hands Guide](https://google.github.io/mediapipe/solutions/hands.html)
- [TensorFlow.js Models](https://github.com/tensorflow/tfjs-models)
- [COCO Dataset Classes](https://cocodataset.org/#explore)
- [WebXR Hand Input](https://immersive-web.github.io/webxr-hand-input/)

---

**Created**: 2025-11-22 (Phase 2 - Vision Tools)
**Author**: HoloLoom AR Team
**License**: MIT
