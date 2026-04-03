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
    - POST /ar/vision/estimate_depth - Depth estimation (Phase 4)
    - POST /ar/vision/detect_markers - Marker detection (Phase 4)
    - POST /ar/vision/segment_image - Semantic segmentation (Phase 5)
    - POST /ar/vision/estimate_pose - Full-body pose estimation (Phase 5)
    - POST /ar/vision/track_camera - SLAM camera tracking (Phase 5)

Created: 2025-11-22 (Phase 1 Prototype)
Updated: 2025-11-22 (Phase 2 - Vision Endpoints)
Updated: 2025-11-22 (Phase 5 - Advanced Vision)
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime
from io import BytesIO
from typing import Any

import numpy as np

# AR Adapter imports
from elle.adapters.ar_adapter import ARAdapter
from elle.adapters.ar_adapter.ar_events import (
    ARContext,
    AREvent,
    AREventType,
    ARObject,
    Quaternion,
    Vector3,
    VoiceEvent,
)

# Elle imports
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

# HoloLoom imports
from hololoom.config import Config
from hololoom.protocols.types import Query

# Vision tools imports (Phase 2)
from hololoom.vision import (
    create_hand_tracker,
    create_object_detector,
    create_scene_analyzer,
)
from hololoom.core.orchestrator.weaving_orchestrator import WeavingOrchestrator

# Import the new Redis-based rate limiter
try:
    from hololoom.apps.server.redis_rate_limiter import (
        EndpointRateLimiter,
        cleanup_rate_limiter,
        get_rate_limiter,
        init_rate_limiter,
    )
    REDIS_RATE_LIMITER_AVAILABLE = True
except ImportError:
    logger.warning("Redis rate limiter not available, falling back to in-memory rate limiting")
    REDIS_RATE_LIMITER_AVAILABLE = False


logger = logging.getLogger(__name__)


# ============================================================================
# Security Configuration
# ============================================================================

# File upload limits (prevent DoS attacks)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size
ALLOWED_IMAGE_FORMATS = {'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'}

async def validate_image_upload(file: UploadFile) -> None:
    """
    Validate uploaded image file for security.

    Prevents DoS attacks via:
    - File size limits (10MB max)
    - Format validation (only images)
    - Content-type verification

    Args:
        file: Uploaded file to validate

    Raises:
        HTTPException: If validation fails
    """
    # Check content type
    if file.content_type not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed: {', '.join(ALLOWED_IMAGE_FORMATS)}"
        )

    # Check file size (read in chunks to avoid loading entire file)
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
        )

    # Validate it's actually an image by trying to open it
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image.verify()  # Verify it's a valid image
        # Reset file pointer for actual use
        await file.seek(0)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )


# Helper function for secure client IP extraction
def _get_client_ip(request: Request) -> str:
    """
    Securely extract client IP address.

    SECURITY: Only trust X-Forwarded-For when BEHIND_PROXY=true.
    This prevents IP spoofing via forged X-Forwarded-For headers.
    """
    import os
    behind_proxy = os.environ.get("BEHIND_PROXY", "false").lower() == "true"

    if behind_proxy:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            ips = [ip.strip() for ip in forwarded_for.split(",") if ip.strip()]
            if ips:
                return ips[0]

    return request.client.host if request.client else "unknown"


class RateLimiter:
    """
    Sliding window rate limiter for vision endpoints.

    Prevents DoS attacks on computationally expensive vision operations.
    Uses per-IP tracking with configurable rate limits.
    Uses asyncio.Lock for proper async concurrency control.
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()  # Async-safe access

    async def check_rate_limit(self, request: Request) -> dict[str, Any]:
        """
        Check if request exceeds rate limit and return rate limit info.

        Args:
            request: FastAPI request object

        Returns:
            Dict containing rate limit headers:
                - limit: Maximum requests per window
                - remaining: Remaining requests in current window
                - reset: Timestamp when window resets

        Raises:
            HTTPException: If rate limit exceeded (status 429)
        """
        # Get client IP securely (handles X-Forwarded-For when behind proxy)
        client_ip = _get_client_ip(request)

        # Async-safe critical section for rate limit checking
        async with self._lock:
            # Get current timestamp
            now = time.time()

            # Get request queue for this IP
            queue = self.requests[client_ip]

            # Remove old requests outside the window
            while queue and queue[0] < now - self.window_seconds:
                queue.popleft()

            # Check if limit exceeded
            if len(queue) >= self.max_requests:
                # Calculate when the oldest request will expire
                reset_time = int(queue[0] + self.window_seconds) if queue else int(now + self.window_seconds)
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds}s",
                    headers={
                        "X-RateLimit-Limit": str(self.max_requests),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(reset_time),
                        "Retry-After": str(reset_time - int(now))  # Seconds to wait
                    }
                )

            # Add current request
            queue.append(now)

            # Calculate reset time (when the current window ends)
            reset_time = int(now + self.window_seconds)

            # Return rate limit info for headers
            return {
                "limit": self.max_requests,
                "remaining": self.max_requests - len(queue),
                "reset": reset_time
            }


# Create global rate limiter for vision endpoints
# Vision processing is computationally expensive, so use conservative limits:
# - 10 requests per 60 seconds (1 request every 6 seconds on average)
# Use Redis-based rate limiter if available for distributed deployment support
if REDIS_RATE_LIMITER_AVAILABLE:
    # Redis rate limiter will be initialized in FastAPI lifespan
    vision_rate_limiter = None  # Will be set during startup
else:
    # Fallback to in-memory rate limiter for single-instance deployments
    vision_rate_limiter = RateLimiter(max_requests=10, window_seconds=60)


# ============================================================================
# Request/Response Models
# ============================================================================

class ARQueryRequest(BaseModel):
    """AR query request"""
    text: str
    context: dict[str, Any]  # ARContext serialized
    session_id: str
    intent: str | None = None


class ARQueryResponse(BaseModel):
    """AR query response"""
    response: str
    visualizations: list[dict[str, Any]]
    confidence: float
    metadata: dict[str, Any]


class ARContextUpdate(BaseModel):
    """AR context update"""
    session_id: str
    user_position: dict[str, float]  # {x, y, z}
    user_rotation: dict[str, float]  # {x, y, z, w}
    gaze_direction: dict[str, float]
    visible_objects: list[dict[str, Any]]


# ============================================================================
# Vision Request/Response Models (Phase 2)
# ============================================================================

class VisionDetectionResponse(BaseModel):
    """Object detection response"""
    objects: list[dict[str, Any]]  # DetectedObject serialized
    count: int
    processing_time_ms: float


class VisionSceneResponse(BaseModel):
    """Scene analysis response"""
    scene_type: str
    objects: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    spatial_layout: dict[str, Any]
    lighting: str
    dominant_colors: list[list[int]]
    processing_time_ms: float


class VisionHandsResponse(BaseModel):
    """Hand tracking response"""
    hands: list[dict[str, Any]]  # HandPose serialized
    count: int
    processing_time_ms: float


class VisionDepthResponse(BaseModel):
    """Depth estimation response (Phase 4)"""
    depth_available: bool
    width: int
    height: int
    min_depth: float
    max_depth: float
    unit: str  # normalized, meters, disparity
    processing_time_ms: float


class VisionMarkersResponse(BaseModel):
    """Marker detection response (Phase 4)"""
    markers: list[dict[str, Any]]  # Marker serialized
    count: int
    processing_time_ms: float


# ============================================================================
# Vision Request/Response Models (Phase 5)
# ============================================================================

class VisionSegmentationResponse(BaseModel):
    """Semantic segmentation response (Phase 5)"""
    width: int
    height: int
    num_classes: int
    class_names: list[str]
    class_distribution: dict[str, float]  # Percentage per class
    processing_time_ms: float


class VisionPoseResponse(BaseModel):
    """Pose estimation response (Phase 5)"""
    poses: list[dict[str, Any]]  # BodyPose serialized
    count: int
    processing_time_ms: float


class VisionSLAMResponse(BaseModel):
    """SLAM camera tracking response (Phase 5)"""
    position: list[float]  # [x, y, z]
    orientation: list[float]  # Quaternion [x, y, z, w]
    tracking_quality: float
    num_features: int
    map_points: int
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
        self.context: ARContext | None = None
        self.created_at = datetime.now()
        self.last_activity = datetime.now()

        # Initialize session
        self.ar_adapter.start_session(session_id)

    async def process_ar_event(self, event: AREvent) -> dict[str, Any]:
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

    def _serialize_visualization(self, viz) -> dict[str, Any]:
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
        self.sessions: dict[str, ARSession] = {}
        self.orchestrator: WeavingOrchestrator | None = None

        # Vision processors (Phase 2)
        self.object_detector = None
        self.scene_analyzer = None
        self.hand_tracker = None
        self.vision_initialized = False

        # Phase 4 processors
        self.depth_estimator = None
        self.marker_detector = None

        # Phase 5 processors
        self.semantic_segmenter = None
        self.pose_estimator = None
        self.slam_processor = None

        # Setup CORS
        # SECURITY: Use environment variables for allowed origins (no wildcard in production)
        import os as _cors_os
        _cors_origins = _cors_os.environ.get(
            "CORS_ORIGINS",
            "http://localhost:3000,http://localhost:8080"
        ).split(",")
        _cors_allow_credentials = _cors_os.environ.get(
            "CORS_ALLOW_CREDENTIALS", "false"
        ).lower() == "true"

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=_cors_origins,
            allow_credentials=_cors_allow_credentials,  # SECURITY: Only enable if explicitly configured
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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

        # Initialize Phase 4 processors
        try:
            from hololoom.vision import create_depth_estimator, create_marker_detector

            self.depth_estimator = create_depth_estimator(model="midas_small")
            await self.depth_estimator.initialize()

            self.marker_detector = create_marker_detector(marker_type="aruco")
            await self.marker_detector.initialize()

            logger.info("Phase 4 vision processors initialized successfully")
        except Exception as e:
            logger.warning(f"Phase 4 vision initialization failed: {e}")
            self.depth_estimator = None
            self.marker_detector = None

        # Initialize Phase 5 processors
        try:
            from hololoom.vision import (
                create_pose_estimator,
                create_semantic_segmenter,
                create_slam_processor,
            )

            self.semantic_segmenter = create_semantic_segmenter(
                model="deeplabv3_resnet50",
                dataset="coco"
            )
            await self.semantic_segmenter.initialize()

            self.pose_estimator = create_pose_estimator(
                model_complexity=1,  # Full model
                enable_segmentation=False
            )
            await self.pose_estimator.initialize()

            self.slam_processor = create_slam_processor(
                feature_detector="orb",
                max_features=500
            )
            await self.slam_processor.initialize()

            logger.info("Phase 5 vision processors initialized successfully")
        except Exception as e:
            logger.warning(f"Phase 5 vision initialization failed: {e}")
            self.semantic_segmenter = None
            self.pose_estimator = None
            self.slam_processor = None

        logger.info("AR API initialized")

    async def _check_rate_limit(self, request: Request, endpoint: str) -> dict[str, Any]:
        """
        Check rate limit using either Redis or in-memory limiter.

        Args:
            request: FastAPI request
            endpoint: Endpoint name for rate limiting

        Returns:
            Rate limit info dict with limit, remaining, reset

        Raises:
            HTTPException: If rate limit exceeded
        """
        import time

        if REDIS_RATE_LIMITER_AVAILABLE and isinstance(vision_rate_limiter, EndpointRateLimiter):
            # Use Redis-based rate limiter for distributed deployments
            await vision_rate_limiter.check_endpoint_limit(request, endpoint)
            # Set rate limit info for headers (Redis limiter handles headers internally)
            return {
                "limit": 10,
                "remaining": "N/A",  # Redis limiter handles this
                "reset": int(time.time() + 60)
            }
        else:
            # Use in-memory rate limiter for single-instance deployments
            return await vision_rate_limiter.check_rate_limit(request)

    def _create_rate_limited_response(self, response_data, rate_limit_info: dict[str, Any]):
        """
        Helper to create a JSON response with rate limit headers.

        Args:
            response_data: The Pydantic model response data
            rate_limit_info: Rate limit info from check_rate_limit()

        Returns:
            JSONResponse with rate limit headers
        """
        return JSONResponse(
            content=response_data.dict(),
            headers={
                "X-RateLimit-Limit": str(rate_limit_info["limit"]),
                "X-RateLimit-Remaining": str(rate_limit_info["remaining"]),
                "X-RateLimit-Reset": str(rate_limit_info["reset"])
            }
        )

    def _register_routes(self):
        """Register API routes"""

        @self.app.on_event("startup")
        async def startup():
            global vision_rate_limiter

            # Initialize Redis rate limiter if available
            if REDIS_RATE_LIMITER_AVAILABLE:
                try:
                    await init_rate_limiter()
                    # Use the global rate limiter which supports distributed rate limiting
                    vision_rate_limiter = get_rate_limiter()
                    logger.info("Redis-based distributed rate limiter initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Redis rate limiter: {e}")
                    # Fall back to in-memory rate limiter
                    vision_rate_limiter = RateLimiter(max_requests=10, window_seconds=60)
                    logger.info("Falling back to in-memory rate limiter")

            await self.initialize()

        @self.app.on_event("shutdown")
        async def shutdown():
            # Clean up Redis rate limiter if used
            if REDIS_RATE_LIMITER_AVAILABLE:
                try:
                    await cleanup_rate_limiter()
                    logger.info("Redis rate limiter cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up Redis rate limiter: {e}")

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

            except HTTPException:
                # Re-raise HTTP exceptions (including rate limit)
                raise
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
            session: ARSession | None = None

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
            except HTTPException:
                # Re-raise HTTP exceptions (including rate limit)
                raise
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
        async def detect_objects_endpoint(request: Request, file: UploadFile = File(...)):
            """
            Object detection endpoint.

            Detects objects in uploaded image using YOLO/COCO-SSD.
            Returns list of detected objects with bboxes and confidence scores.

            Rate Limits:
                - 10 requests per 60 seconds per IP address
                - Returns X-RateLimit-* headers with request limits

            Security:
                - Max file size: 10MB
                - Allowed formats: JPEG, PNG, WebP, GIF
                - File content validation

            Returns:
                VisionDetectionResponse with rate limit headers
                429 status if rate limit exceeded
            """
            if not self.object_detector:
                raise HTTPException(status_code=503, detail="Vision not initialized")

            try:
                import time
                start_time = time.time()

                # Rate limiting (prevent DoS) - returns rate limit info
                rate_limit_info = await self._check_rate_limit(request, "vision/detect_objects")

                # Validate file upload (security)
                await validate_image_upload(file)

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

                response_data = VisionDetectionResponse(
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

                # Return with rate limit headers
                return self._create_rate_limited_response(response_data, rate_limit_info)

            except HTTPException:
                # Re-raise HTTP exceptions (including rate limit)
                raise
            except HTTPException:
                # Re-raise HTTP exceptions (including rate limit)
                raise
            except Exception as e:
                logger.error(f"Object detection error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ar/vision/analyze_scene", response_model=VisionSceneResponse)
        async def analyze_scene_endpoint(request: Request, file: UploadFile = File(...)):
            """
            Scene analysis endpoint.

            Analyzes image to determine scene type, objects, spatial relationships,
            lighting conditions, and dominant colors.

            Rate Limits:
                - 10 requests per 60 seconds per IP address
                - Returns X-RateLimit-* headers with request limits

            Security:
                - Max file size: 10MB
                - Allowed formats: JPEG, PNG, WebP, GIF
                - File content validation

            Returns:
                VisionSceneResponse with rate limit headers
                429 status if rate limit exceeded
            """
            if not self.scene_analyzer or not self.object_detector:
                raise HTTPException(status_code=503, detail="Vision not initialized")

            try:
                import time
                start_time = time.time()

                # Rate limiting (prevent DoS) - returns rate limit info
                rate_limit_info = await self._check_rate_limit(request, "vision/analyze_scene")

                # Validate file upload (security)
                await validate_image_upload(file)

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

                response_data = VisionSceneResponse(
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

                # Return with rate limit headers
                return self._create_rate_limited_response(response_data, rate_limit_info)

            except HTTPException:
                # Re-raise HTTP exceptions (including rate limit)
                raise
            except HTTPException:
                # Re-raise HTTP exceptions (including rate limit)
                raise
            except Exception as e:
                logger.error(f"Scene analysis error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ar/vision/track_hands", response_model=VisionHandsResponse)
        async def track_hands_endpoint(request: Request, file: UploadFile = File(...)):
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

                # Rate limiting (prevent DoS) - returns rate limit info
                rate_limit_info = await self._check_rate_limit(request, "vision/track_hands")

                # Validate file upload (security)
                await validate_image_upload(file)

                # Read image file
                contents = await file.read()
                image = Image.open(BytesIO(contents))
                frame = np.array(image)

                # Track hands
                hand_poses = await self.hand_tracker.track_hands(frame)

                processing_time = (time.time() - start_time) * 1000

                response_data = VisionHandsResponse(
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

                # Return with rate limit headers
                return self._create_rate_limited_response(response_data, rate_limit_info)

            except HTTPException:
                # Re-raise HTTP exceptions (including rate limit)
                raise
            except Exception as e:
                logger.error(f"Hand tracking error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ar/vision/estimate_depth", response_model=VisionDepthResponse)
        async def estimate_depth_endpoint(request: Request, file: UploadFile = File(...)):
            """
            Depth estimation endpoint (Phase 4).

            Estimates depth map from uploaded image using MiDaS or ZoeDepth.
            Returns depth map metadata and processing time.
            """
            if not self.depth_estimator:
                raise HTTPException(status_code=503, detail="Depth estimator not initialized")

            try:
                import time
                start_time = time.time()

                # Rate limiting (prevent DoS) - returns rate limit info
                rate_limit_info = await self._check_rate_limit(request, "vision/estimate_depth")

                # Validate file upload (security)
                await validate_image_upload(file)

                # Read image file
                contents = await file.read()
                image = Image.open(BytesIO(contents))
                frame = np.array(image)

                # Estimate depth
                depth_map = await self.depth_estimator.estimate_depth(frame)

                processing_time = (time.time() - start_time) * 1000

                response_data = VisionDepthResponse(
                    depth_available=True,
                    width=depth_map.width,
                    height=depth_map.height,
                    min_depth=float(depth_map.min_depth),
                    max_depth=float(depth_map.max_depth),
                    unit=depth_map.unit,
                    processing_time_ms=processing_time,
                )

                # Return with rate limit headers
                return self._create_rate_limited_response(response_data, rate_limit_info)

            except HTTPException:
                # Re-raise HTTP exceptions (including rate limit)
                raise
            except Exception as e:
                logger.error(f"Depth estimation error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ar/vision/detect_markers", response_model=VisionMarkersResponse)
        async def detect_markers_endpoint(request: Request, file: UploadFile = File(...)):
            """
            Marker detection endpoint (Phase 4).

            Detects ArUco markers, QR codes, or AprilTags in uploaded image.
            Returns markers with 2D corners and optional 6-DOF pose if camera calibrated.
            """
            if not self.marker_detector:
                raise HTTPException(status_code=503, detail="Marker detector not initialized")

            try:
                import time
                start_time = time.time()

                # Rate limiting (prevent DoS) - returns rate limit info
                rate_limit_info = await self._check_rate_limit(request, "vision/detect_markers")

                # Validate file upload (security)
                await validate_image_upload(file)

                # Read image file
                contents = await file.read()
                image = Image.open(BytesIO(contents))
                frame = np.array(image)

                # Detect markers
                markers = await self.marker_detector.detect_markers(frame)

                processing_time = (time.time() - start_time) * 1000

                response_data = VisionMarkersResponse(
                    markers=[
                        {
                            "id": marker.id,
                            "markerType": marker.marker_type,
                            "corners": marker.corners,
                            "center": marker.center,
                            "position": marker.position,
                            "rotation": marker.rotation,
                            "data": marker.data,
                            "confidence": marker.confidence,
                        }
                        for marker in markers
                    ],
                    count=len(markers),
                    processing_time_ms=processing_time,
                )

                # Return with rate limit headers
                return self._create_rate_limited_response(response_data, rate_limit_info)

            except HTTPException:
                # Re-raise HTTP exceptions (including rate limit)
                raise
            except Exception as e:
                logger.error(f"Marker detection error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        # ====================================================================
        # Vision Processing Endpoints (Phase 5)
        # ====================================================================

        @self.app.post("/ar/vision/segment_image", response_model=VisionSegmentationResponse)
        async def segment_image_endpoint(request: Request, file: UploadFile = File(...)):
            """
            Semantic segmentation endpoint (Phase 5).

            Performs pixel-level semantic segmentation using DeepLabV3 or SegFormer.
            Returns segmentation mask, class distribution, and processing time.
            """
            if not self.semantic_segmenter:
                raise HTTPException(status_code=503, detail="Semantic segmenter not initialized")

            try:
                import time
                start_time = time.time()

                # Rate limiting (prevent DoS) - returns rate limit info
                rate_limit_info = await self._check_rate_limit(request, "vision/segment_image")

                # Validate file upload (security)
                await validate_image_upload(file)

                # Read image file
                contents = await file.read()
                image = Image.open(BytesIO(contents))
                frame = np.array(image)

                # Segment image
                segmentation = await self.semantic_segmenter.process_frame(frame)

                # Calculate class distribution
                from hololoom.vision import get_class_distribution
                class_dist = get_class_distribution(segmentation)

                processing_time = (time.time() - start_time) * 1000

                response_data = VisionSegmentationResponse(
                    width=segmentation.width,
                    height=segmentation.height,
                    num_classes=segmentation.num_classes,
                    class_names=segmentation.class_names,
                    class_distribution=class_dist,
                    processing_time_ms=processing_time,
                )

                # Return with rate limit headers
                return self._create_rate_limited_response(response_data, rate_limit_info)

            except HTTPException:
                # Re-raise HTTP exceptions (including rate limit)
                raise
            except Exception as e:
                logger.error(f"Semantic segmentation error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ar/vision/estimate_pose", response_model=VisionPoseResponse)
        async def estimate_pose_endpoint(request: Request, file: UploadFile = File(...)):
            """
            Pose estimation endpoint (Phase 5).

            Estimates full-body pose using MediaPipe Pose with 33 keypoints.
            Returns body poses with 3D world coordinates, visibility scores, and gesture detection.
            """
            if not self.pose_estimator:
                raise HTTPException(status_code=503, detail="Pose estimator not initialized")

            try:
                import time
                start_time = time.time()

                # Rate limiting (prevent DoS) - returns rate limit info
                rate_limit_info = await self._check_rate_limit(request, "vision/estimate_pose")

                # Validate file upload (security)
                await validate_image_upload(file)

                # Read image file
                contents = await file.read()
                image = Image.open(BytesIO(contents))
                frame = np.array(image)

                # Estimate pose
                pose = await self.pose_estimator.process_frame(frame)

                processing_time = (time.time() - start_time) * 1000

                # Serialize pose
                if pose:
                    from hololoom.vision import detect_gesture

                    gesture = detect_gesture(pose)

                    poses_data = [
                        {
                            "keypoints": [
                                {
                                    "x": kp.x,
                                    "y": kp.y,
                                    "z": kp.z,
                                    "visibility": kp.visibility,
                                    "presence": kp.presence,
                                }
                                for kp in pose.keypoints
                            ],
                            "confidence": pose.confidence,
                            "gesture": gesture,
                        }
                    ]
                else:
                    poses_data = []

                response_data = VisionPoseResponse(
                    poses=poses_data,
                    count=len(poses_data),
                    processing_time_ms=processing_time,
                )

                # Return with rate limit headers
                return self._create_rate_limited_response(response_data, rate_limit_info)

            except HTTPException:
                # Re-raise HTTP exceptions (including rate limit)
                raise
            except Exception as e:
                logger.error(f"Pose estimation error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ar/vision/track_camera", response_model=VisionSLAMResponse)
        async def track_camera_endpoint(request: Request, file: UploadFile = File(...)):
            """
            SLAM camera tracking endpoint (Phase 5).

            Estimates camera pose using visual SLAM with ORB features.
            Returns 6-DOF camera pose (position + orientation), tracking quality,
            and number of tracked features.

            Note: This endpoint processes a single frame. For real-time tracking,
            use WebSocket with sequential frames to maintain SLAM state.
            """
            if not self.slam_processor:
                raise HTTPException(status_code=503, detail="SLAM processor not initialized")

            try:
                import time
                start_time = time.time()

                # Rate limiting (prevent DoS) - returns rate limit info
                rate_limit_info = await self._check_rate_limit(request, "vision/track_camera")

                # Validate file upload (security)
                await validate_image_upload(file)

                # Read image file
                contents = await file.read()
                image = Image.open(BytesIO(contents))
                frame = np.array(image)

                # Track camera pose
                slam_pose = await self.slam_processor.process_frame(frame)

                processing_time = (time.time() - start_time) * 1000

                # Get map points
                map_points = self.slam_processor.get_map_points()

                response_data = VisionSLAMResponse(
                    position=list(slam_pose.position),
                    orientation=list(slam_pose.orientation),
                    tracking_quality=slam_pose.tracking_quality,
                    num_features=slam_pose.num_features,
                    map_points=len(map_points),
                    processing_time_ms=processing_time,
                )

                # Return with rate limit headers
                return self._create_rate_limited_response(response_data, rate_limit_info)

            except HTTPException:
                # Re-raise HTTP exceptions (including rate limit)
                raise
            except Exception as e:
                logger.error(f"SLAM tracking error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

    def _deserialize_context(self, data: dict[str, Any]) -> ARContext:
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

    def _deserialize_event(self, data: dict[str, Any]) -> AREvent:
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
