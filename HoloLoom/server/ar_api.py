"""
AR API Server - FastAPI endpoints for AR client integration

Provides WebSocket connection for real-time AR interaction and REST endpoints
for AR-specific operations.

Architecture:
    AR Client (React/WebXR) ←WebSocket→ ar_api.py ←→ ARAdapter ←→ Elle Core ←→ HoloLoom

Endpoints:
    - WebSocket: /ws/ar - Real-time bidirectional AR communication
    - POST /ar/query - Synchronous AR query
    - POST /ar/context - Update AR spatial context
    - GET /ar/session/{session_id} - Get session state
    - POST /ar/vision/detect_objects - Object detection (Phase 2)
    - POST /ar/vision/analyze_scene - Scene analysis (Phase 2)
    - POST /ar/vision/track_hands - Hand tracking (Phase 2)

Created: 2025-11-22 (Phase 1 Prototype)
Updated: 2025-11-22 (Phase 2 - Vision Endpoints)
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel
import logging
import json
import asyncio
import numpy as np
from io import BytesIO
from PIL import Image

# HoloLoom imports
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.Documentation.types import Query, MemoryShard

# Vision tools imports (Phase 2)
from HoloLoom.vision import (
    create_object_detector,
    create_scene_analyzer,
    create_hand_tracker,
    DetectedObject,
    SceneUnderstanding,
    HandPose,
)

# Elle imports
from elle.core.policy import EllePolicy
from elle.core.llm_client import create_llm_client
from elle.domain.scene import Intent

# AR Adapter imports
from elle.adapters.ar_adapter import ARAdapter
from elle.adapters.ar_adapter.ar_events import (
    AREvent,
    AREventType,
    ARContext,
    ARObject,
    Vector3,
    Quaternion,
    VoiceEvent,
)
from elle.adapters.ar_adapter.platform_bridge import WebXRBridge


logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class ARQueryRequest(BaseModel):
    """AR query request"""
    text: str
    context: Dict[str, Any]  # ARContext serialized
    session_id: str
    intent: Optional[str] = None


class ARQueryResponse(BaseModel):
    """AR query response"""
    response: str
    visualizations: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]


class ARContextUpdate(BaseModel):
    """AR context update"""
    session_id: str
    user_position: Dict[str, float]  # {x, y, z}
    user_rotation: Dict[str, float]  # {x, y, z, w}
    gaze_direction: Dict[str, float]
    visible_objects: List[Dict[str, Any]]


# ============================================================================
# Vision Request/Response Models (Phase 2)
# ============================================================================

class VisionDetectionResponse(BaseModel):
    """Object detection response"""
    objects: List[Dict[str, Any]]  # DetectedObject serialized
    count: int
    processing_time_ms: float


class VisionSceneResponse(BaseModel):
    """Scene analysis response"""
    scene_type: str
    objects: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    spatial_layout: Dict[str, Any]
    lighting: str
    dominant_colors: List[List[int]]
    processing_time_ms: float


class VisionHandsResponse(BaseModel):
    """Hand tracking response"""
    hands: List[Dict[str, Any]]  # HandPose serialized
    count: int
    processing_time_ms: float


# ============================================================================
# Session Management
# ============================================================================

class ARSession:
    """AR session state"""

    def __init__(self, session_id: str, orchestrator: WeavingOrchestrator):
        self.session_id = session_id
        self.orchestrator = orchestrator
        self.ar_adapter = ARAdapter()
        self.context: Optional[ARContext] = None
        self.created_at = datetime.now()
        self.last_activity = datetime.now()

        # Initialize session
        self.ar_adapter.start_session(session_id)

    async def process_ar_event(self, event: AREvent) -> Dict[str, Any]:
        """Process AR event through Elle → HoloLoom pipeline"""
        self.last_activity = datetime.now()
        self.context = event.context

        # Step 1: AR Event → Elle Request
        elle_request = self.ar_adapter.ar_event_to_elle_request(event)

        # Step 2: Query HoloLoom with Elle request
        # Convert scene to text query for now
        # In full implementation, Elle Core would handle this
        if isinstance(event, VoiceEvent):
            query_text = event.transcript
        else:
            # Generate query from scene
            query_text = f"User looking at {event.context.gaze_target or 'unknown object'}"

        # Create HoloLoom query
        query = Query(text=query_text)

        # Weave through HoloLoom
        spacetime = await self.orchestrator.weave(query)

        # Step 3: Convert response to AR visualizations
        # For prototype, create simple visualization
        from elle.domain.action import ElleAction, Symbol

        elle_action = ElleAction(
            visual_guidance=spacetime.response,
            symbol=Symbol.PLATO,
        )

        viz_set = self.ar_adapter.elle_action_to_visualizations(
            elle_action,
            event.context,
        )

        # Step 4: Return response with visualizations
        return {
            "response": spacetime.response,
            "visualizations": [
                self._serialize_visualization(viz)
                for viz in viz_set.visualizations
            ],
            "confidence": spacetime.confidence,
            "metadata": {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
            },
        }

    def _serialize_visualization(self, viz) -> Dict[str, Any]:
        """Serialize visualization for JSON transmission"""
        return {
            "id": viz.id,
            "type": viz.viz_type.value,
            "style": {
                "color": viz.style.color,
                "opacity": viz.style.opacity,
                "size": viz.style.size,
            },
        }

    def cleanup(self):
        """Clean up session resources"""
        self.ar_adapter.end_session()


# ============================================================================
# AR API Application
# ============================================================================

class ARAPI:
    """AR API server"""

    def __init__(self):
        self.app = FastAPI(title="HoloLoom AR API", version="1.0.0")
        self.sessions: Dict[str, ARSession] = {}
        self.orchestrator: Optional[WeavingOrchestrator] = None

        # Vision processors (Phase 2)
        self.object_detector = None
        self.scene_analyzer = None
        self.hand_tracker = None
        self.vision_initialized = False

        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self._register_routes()

    async def initialize(self):
        """Initialize HoloLoom orchestrator and vision processors"""
        # Initialize orchestrator
        config = Config.fast()
        shards = []  # Load from storage in production

        self.orchestrator = WeavingOrchestrator(cfg=config, shards=shards)
        await self.orchestrator.__aenter__()

        # Initialize vision processors (Phase 2)
        try:
            self.object_detector = create_object_detector(backend="yolo")
            await self.object_detector.initialize()

            self.scene_analyzer = create_scene_analyzer()
            await self.scene_analyzer.initialize()

            self.hand_tracker = create_hand_tracker(backend="mediapipe")
            await self.hand_tracker.initialize()

            self.vision_initialized = True
            logger.info("Vision processors initialized successfully")
        except Exception as e:
            logger.warning(f"Vision initialization failed (will use mock): {e}")
            # Fallback to mock backends
            self.object_detector = create_object_detector(backend="mock")
            self.scene_analyzer = create_scene_analyzer()
            self.hand_tracker = create_hand_tracker(backend="mock")
            self.vision_initialized = False

        logger.info("AR API initialized")

    def _register_routes(self):
        """Register API routes"""

        @self.app.on_event("startup")
        async def startup():
            await self.initialize()

        @self.app.on_event("shutdown")
        async def shutdown():
            # Clean up all sessions
            for session in self.sessions.values():
                session.cleanup()

            if self.orchestrator:
                await self.orchestrator.__aexit__(None, None, None)

        @self.app.get("/")
        async def root():
            return {
                "service": "HoloLoom AR API",
                "version": "1.0.0",
                "status": "running",
                "active_sessions": len(self.sessions),
            }

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "active_sessions": len(self.sessions),
                "timestamp": datetime.now().isoformat(),
            }

        @self.app.post("/ar/query", response_model=ARQueryResponse)
        async def ar_query(request: ARQueryRequest):
            """
            Synchronous AR query endpoint.

            For REST clients that don't use WebSocket.
            """
            try:
                # Get or create session
                session = self.sessions.get(request.session_id)
                if not session:
                    session = ARSession(request.session_id, self.orchestrator)
                    self.sessions[request.session_id] = session

                # Create voice event from request
                context = self._deserialize_context(request.context)
                event = VoiceEvent(
                    transcript=request.text,
                    intent=request.intent,
                    context=context,
                )

                # Process event
                result = await session.process_ar_event(event)

                return ARQueryResponse(**result)

            except Exception as e:
                logger.error(f"AR query error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.websocket("/ws/ar")
        async def websocket_ar(websocket: WebSocket):
            """
            WebSocket endpoint for real-time AR communication.

            Protocol:
                Client → Server:
                    - {"type": "start_session", "config": {...}}
                    - {"type": "ar_event", "event": {...}}
                    - {"type": "context_update", "context": {...}}
                    - {"type": "end_session"}

                Server → Client:
                    - {"type": "session_started", "session_id": "..."}
                    - {"type": "response", "data": {...}}
                    - {"type": "visualizations", "visualizations": [...]}
                    - {"type": "error", "error": "..."}
            """
            await websocket.accept()
            session: Optional[ARSession] = None

            try:
                while True:
                    # Receive message
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    msg_type = message.get("type")

                    if msg_type == "start_session":
                        # Create new session
                        import uuid
                        session_id = str(uuid.uuid4())
                        session = ARSession(session_id, self.orchestrator)
                        self.sessions[session_id] = session

                        await websocket.send_json({
                            "type": "session_started",
                            "session_id": session_id,
                        })

                    elif msg_type == "ar_event":
                        if not session:
                            await websocket.send_json({
                                "type": "error",
                                "error": "No active session",
                            })
                            continue

                        # Deserialize and process AR event
                        event = self._deserialize_event(message["event"])
                        result = await session.process_ar_event(event)

                        await websocket.send_json({
                            "type": "response",
                            "data": result,
                        })

                    elif msg_type == "context_update":
                        if not session:
                            await websocket.send_json({
                                "type": "error",
                                "error": "No active session",
                            })
                            continue

                        # Update context
                        context = self._deserialize_context(message["context"])
                        session.context = context

                    elif msg_type == "end_session":
                        if session:
                            session.cleanup()
                            if session.session_id in self.sessions:
                                del self.sessions[session.session_id]
                        break

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                })
            finally:
                if session:
                    session.cleanup()

        # ====================================================================
        # Vision Processing Endpoints (Phase 2)
        # ====================================================================

        @self.app.post("/ar/vision/detect_objects", response_model=VisionDetectionResponse)
        async def detect_objects_endpoint(file: UploadFile = File(...)):
            """
            Object detection endpoint.

            Detects objects in uploaded image using YOLO/COCO-SSD.
            Returns list of detected objects with bboxes and confidence scores.
            """
            if not self.object_detector:
                raise HTTPException(status_code=503, detail="Vision not initialized")

            try:
                import time
                start_time = time.time()

                # Read image file
                contents = await file.read()
                image = Image.open(BytesIO(contents))
                frame = np.array(image)

                # Detect objects
                detected_objects = await self.object_detector.detect_objects(
                    frame,
                    confidence_threshold=0.5
                )

                processing_time = (time.time() - start_time) * 1000

                return VisionDetectionResponse(
                    objects=[
                        {
                            "id": obj.id,
                            "label": obj.label,
                            "confidence": obj.confidence,
                            "bbox": {
                                "xMin": obj.bbox.x_min,
                                "yMin": obj.bbox.y_min,
                                "xMax": obj.bbox.x_max,
                                "yMax": obj.bbox.y_max,
                            },
                            "classId": obj.class_id,
                        }
                        for obj in detected_objects
                    ],
                    count=len(detected_objects),
                    processing_time_ms=processing_time,
                )

            except Exception as e:
                logger.error(f"Object detection error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ar/vision/analyze_scene", response_model=VisionSceneResponse)
        async def analyze_scene_endpoint(file: UploadFile = File(...)):
            """
            Scene analysis endpoint.

            Analyzes image to determine scene type, objects, spatial relationships,
            lighting conditions, and dominant colors.
            """
            if not self.scene_analyzer or not self.object_detector:
                raise HTTPException(status_code=503, detail="Vision not initialized")

            try:
                import time
                start_time = time.time()

                # Read image file
                contents = await file.read()
                image = Image.open(BytesIO(contents))
                frame = np.array(image)

                # Detect objects first
                detected_objects = await self.object_detector.detect_objects(frame)

                # Analyze scene
                scene_understanding = await self.scene_analyzer.analyze_scene(
                    frame,
                    detected_objects
                )

                processing_time = (time.time() - start_time) * 1000

                return VisionSceneResponse(
                    scene_type=scene_understanding.scene_type,
                    objects=[
                        {
                            "id": obj.id,
                            "label": obj.label,
                            "confidence": obj.confidence,
                        }
                        for obj in scene_understanding.objects
                    ],
                    relationships=[
                        {
                            "object1": rel.object1_id,
                            "object2": rel.object2_id,
                            "relationship": rel.relationship_type,
                        }
                        for rel in scene_understanding.relationships
                    ],
                    spatial_layout=scene_understanding.spatial_layout,
                    lighting=scene_understanding.lighting,
                    dominant_colors=scene_understanding.dominant_colors,
                    processing_time_ms=processing_time,
                )

            except Exception as e:
                logger.error(f"Scene analysis error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ar/vision/track_hands", response_model=VisionHandsResponse)
        async def track_hands_endpoint(file: UploadFile = File(...)):
            """
            Hand tracking endpoint.

            Detects hands and recognizes gestures in uploaded image using MediaPipe Hands.
            Returns hand poses with 21 landmarks and gesture classification.
            """
            if not self.hand_tracker:
                raise HTTPException(status_code=503, detail="Vision not initialized")

            try:
                import time
                start_time = time.time()

                # Read image file
                contents = await file.read()
                image = Image.open(BytesIO(contents))
                frame = np.array(image)

                # Track hands
                hand_poses = await self.hand_tracker.track_hands(frame)

                processing_time = (time.time() - start_time) * 1000

                return VisionHandsResponse(
                    hands=[
                        {
                            "handId": hand.hand_id,
                            "gesture": hand.gesture,
                            "confidence": hand.confidence,
                            "landmarks": [
                                {"x": lm.x, "y": lm.y, "z": lm.z}
                                for lm in hand.landmarks
                            ],
                        }
                        for hand in hand_poses
                    ],
                    count=len(hand_poses),
                    processing_time_ms=processing_time,
                )

            except Exception as e:
                logger.error(f"Hand tracking error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

    def _deserialize_context(self, data: Dict[str, Any]) -> ARContext:
        """Deserialize AR context from JSON"""
        return ARContext(
            user_position=Vector3(**data["userPosition"]),
            user_rotation=Quaternion(**data["userRotation"]),
            gaze_direction=Vector3(**data["gazeDirection"]),
            gaze_target=data.get("gazeTarget"),
            visible_objects=[
                ARObject(
                    id=obj["id"],
                    label=obj["label"],
                    position=Vector3(**obj["position"]),
                    object_type=obj.get("objectType", "unknown"),
                    confidence=obj.get("confidence", 1.0),
                )
                for obj in data.get("visibleObjects", [])
            ],
            session_id=data.get("sessionId", ""),
            platform=data.get("platform", "webxr"),
        )

    def _deserialize_event(self, data: Dict[str, Any]) -> AREvent:
        """Deserialize AR event from JSON"""
        event_type = AREventType(data["eventType"])
        context = self._deserialize_context(data["context"])

        if event_type == AREventType.VOICE:
            return VoiceEvent(
                transcript=data["transcript"],
                intent=data.get("intent"),
                entities=data.get("entities", {}),
                confidence=data.get("confidence", 1.0),
                context=context,
            )

        # Add other event types as needed
        return AREvent(
            event_type=event_type,
            context=context,
        )


# ============================================================================
# Application Instance
# ============================================================================

# Create global app instance
ar_api = ARAPI()
app = ar_api.app


# ============================================================================
# Development Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ar_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
