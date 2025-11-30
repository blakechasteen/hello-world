"""
AR API Server with Comprehensive Security

Enhanced version of ar_api.py with integrated security features:
- WAF (Web Application Firewall) protection
- Request signing authentication
- Security monitoring and alerting
- Rate limiting
- Input validation

Created: 2025-11-26
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Request, Depends
from fastapi.responses import JSONResponse
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
import time
from collections import defaultdict, deque
import os

# Import security modules
from HoloLoom.server.waf_middleware import WAFMiddleware, WAFRule
from HoloLoom.server.request_signing import SignatureValidator, APIKey, validate_api_request
from HoloLoom.server.security_monitor import (
    SecurityMonitor,
    SecurityEventType,
    AlertSeverity
)

# HoloLoom imports
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.protocols.types import Query, MemoryShard

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
# Security Configuration
# ============================================================================

# File upload limits (prevent DoS attacks)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size
ALLOWED_IMAGE_FORMATS = {'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'}

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 100  # requests
RATE_LIMIT_WINDOW = 60  # seconds

# Security monitoring configuration
ALERT_AUTH_FAILURES = 5  # failures before alert
ALERT_RATE_LIMITS = 10  # rate limit violations before alert
ALERT_WAF_VIOLATIONS = 3  # WAF violations before alert


# ============================================================================
# Pydantic Models
# ============================================================================

class ARQueryRequest(BaseModel):
    """AR query request model with validation"""
    text: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "text": "What am I looking at?",
                "context": {
                    "location": "workshop",
                    "objects_detected": ["hammer", "screwdriver"]
                },
                "session_id": "ar-session-123"
            }
        }


class ARContextUpdate(BaseModel):
    """AR context update model"""
    session_id: str
    location: Optional[Dict[str, float]] = None  # lat, lng, alt
    rotation: Optional[Dict[str, float]] = None  # x, y, z, w
    objects: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class SecurityConfig(BaseModel):
    """Security configuration"""
    enable_waf: bool = True
    enable_signing: bool = True
    enable_monitoring: bool = True
    enable_rate_limiting: bool = True
    waf_block_on_violation: bool = True
    smtp_config: Optional[Dict] = None
    slack_webhook_url: Optional[str] = None


# ============================================================================
# Rate Limiter
# ============================================================================

class EnhancedRateLimiter:
    """
    Enhanced sliding window rate limiter with per-IP and per-endpoint tracking
    """

    def __init__(self, requests: int = 100, window: int = 60):
        self.requests = requests
        self.window = window
        self.ip_requests: Dict[str, deque] = defaultdict(deque)
        self.endpoint_requests: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        self.blocked_ips: set = set()

    async def check_rate_limit(
        self,
        client_ip: str,
        endpoint: str,
        security_monitor: Optional[SecurityMonitor] = None
    ) -> bool:
        """
        Check if request should be rate limited

        Args:
            client_ip: Client IP address
            endpoint: API endpoint being accessed
            security_monitor: Optional security monitor for logging

        Returns:
            True if request is allowed, False if rate limited
        """
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            return False

        now = time.time()
        cutoff = now - self.window

        # Clean old requests for this IP
        ip_queue = self.ip_requests[client_ip]
        while ip_queue and ip_queue[0] < cutoff:
            ip_queue.popleft()

        # Clean old requests for this endpoint
        endpoint_queue = self.endpoint_requests[endpoint][client_ip]
        while endpoint_queue and endpoint_queue[0] < cutoff:
            endpoint_queue.popleft()

        # Check limits
        if len(ip_queue) >= self.requests:
            # Log rate limit violation
            if security_monitor:
                await security_monitor.log_rate_limit(
                    client_ip=client_ip,
                    endpoint=endpoint,
                    limit=self.requests,
                    window=f"{self.window}s"
                )

            # Auto-block after repeated violations
            if len(ip_queue) >= self.requests * 2:
                self.blocked_ips.add(client_ip)
                logger.warning(f"Auto-blocked IP for excessive rate limiting: {client_ip}")

            return False

        # Add current request
        ip_queue.append(now)
        endpoint_queue.append(now)

        return True

    def unblock_ip(self, client_ip: str):
        """Unblock an IP address"""
        self.blocked_ips.discard(client_ip)


# ============================================================================
# Custom WAF Rules for AR
# ============================================================================

class ARSpecificWAFRule(WAFRule):
    """Custom WAF rule for AR-specific attacks"""

    def __init__(self):
        super().__init__(
            "WAF-AR-001",
            "AR-Specific Attack Detection",
            "HIGH"
        )

    def check(self, request: Request, body: bytes = b"") -> Optional[str]:
        """Check for AR-specific attack patterns"""

        # Check for coordinate manipulation attacks
        if body:
            try:
                body_str = body.decode('utf-8', errors='ignore')

                # Check for unrealistic coordinates
                if '"location"' in body_str:
                    # Check for coordinates outside valid ranges
                    import re
                    lat_pattern = r'"lat[itude]*"\s*:\s*(-?\d+\.?\d*)'
                    lng_pattern = r'"l[o]?ng[itude]*"\s*:\s*(-?\d+\.?\d*)'

                    lat_matches = re.findall(lat_pattern, body_str)
                    lng_matches = re.findall(lng_pattern, body_str)

                    for lat in lat_matches:
                        try:
                            lat_val = float(lat)
                            if abs(lat_val) > 90:
                                return f"Invalid latitude value: {lat_val}"
                        except:
                            pass

                    for lng in lng_matches:
                        try:
                            lng_val = float(lng)
                            if abs(lng_val) > 180:
                                return f"Invalid longitude value: {lng_val}"
                        except:
                            pass

                # Check for object injection attacks (trying to inject malicious 3D objects)
                if '"model_url"' in body_str or '"texture_url"' in body_str:
                    # Check for suspicious URLs
                    url_pattern = r'https?://[^\s"\']+'
                    urls = re.findall(url_pattern, body_str)

                    for url in urls:
                        # Check for known malicious patterns
                        if any(pattern in url.lower() for pattern in [
                            'javascript:', 'data:', 'vbscript:', 'file://',
                            '.exe', '.dll', '.bat', '.cmd', '.scr'
                        ]):
                            return f"Suspicious URL detected: {url}"

            except Exception as e:
                logger.error(f"Error in AR WAF rule: {e}")

        return None


# ============================================================================
# Application Factory
# ============================================================================

def create_secured_app(security_config: SecurityConfig = None) -> FastAPI:
    """
    Create FastAPI application with comprehensive security

    Args:
        security_config: Security configuration

    Returns:
        Configured FastAPI application
    """

    if security_config is None:
        security_config = SecurityConfig()

    # Create FastAPI app
    app = FastAPI(
        title="HoloLoom AR API (Secured)",
        description="AR API with comprehensive security features",
        version="2.0.0"
    )

    # ========================================================================
    # Initialize Security Components
    # ========================================================================

    # Initialize security monitor
    security_monitor = SecurityMonitor(
        alert_threshold_auth_failures=ALERT_AUTH_FAILURES,
        alert_threshold_rate_limit=ALERT_RATE_LIMITS,
        alert_threshold_waf_violations=ALERT_WAF_VIOLATIONS,
        enable_email_alerts=security_config.smtp_config is not None,
        enable_slack_alerts=security_config.slack_webhook_url is not None,
        smtp_config=security_config.smtp_config,
        slack_webhook_url=security_config.slack_webhook_url
    )

    # Initialize signature validator
    signature_validator = SignatureValidator(
        max_timestamp_age_seconds=300,
        require_nonce=True,
        enforce_replay_protection=True
    )

    # Initialize rate limiter
    rate_limiter = EnhancedRateLimiter(
        requests=RATE_LIMIT_REQUESTS,
        window=RATE_LIMIT_WINDOW
    )

    # Store in app state for access in endpoints
    app.state.security_monitor = security_monitor
    app.state.signature_validator = signature_validator
    app.state.rate_limiter = rate_limiter

    # ========================================================================
    # Add Middleware
    # ========================================================================

    # Add CORS middleware (configure for your domains)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://ar.hololoom.io"],  # Configure for your domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add WAF middleware
    if security_config.enable_waf:
        # Create custom WAF rules
        custom_rules = [ARSpecificWAFRule()]

        app.add_middleware(
            WAFMiddleware,
            enabled=True,
            custom_rules=custom_rules,
            log_violations=True,
            block_on_violation=security_config.waf_block_on_violation,
            security_monitor=security_monitor
        )

    # ========================================================================
    # Startup/Shutdown Events
    # ========================================================================

    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup"""
        await security_monitor.start()
        logger.info("Security monitoring started")

        # Generate initial API keys for testing (in production, use proper key management)
        if os.getenv("GENERATE_TEST_KEYS") == "true":
            key_id, secret = signature_validator.generate_api_key(
                client_name="AR Test Client",
                expires_in_days=30,
                rate_limit_qps=10.0
            )
            logger.info(f"Test API Key generated - Key ID: {key_id}, Secret: {secret}")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up on shutdown"""
        await security_monitor.stop()
        logger.info("Security monitoring stopped")

    # ========================================================================
    # Health & Monitoring Endpoints
    # ========================================================================

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "security": {
                "waf_enabled": security_config.enable_waf,
                "signing_enabled": security_config.enable_signing,
                "monitoring_enabled": security_config.enable_monitoring,
                "rate_limiting_enabled": security_config.enable_rate_limiting
            }
        }

    @app.get("/security/metrics")
    async def security_metrics(api_key: APIKey = Depends(validate_api_request)):
        """
        Get security metrics (requires authentication)

        Returns Prometheus-formatted metrics
        """
        # Check permission
        if "admin" not in api_key.permissions:
            raise HTTPException(status_code=403, detail="Admin permission required")

        metrics = security_monitor.get_prometheus_metrics()
        return JSONResponse(content=metrics, media_type="text/plain")

    @app.get("/security/dashboard")
    async def security_dashboard(api_key: APIKey = Depends(validate_api_request)):
        """
        Get security dashboard data (requires authentication)

        Returns comprehensive security statistics
        """
        # Check permission
        if "admin" not in api_key.permissions:
            raise HTTPException(status_code=403, detail="Admin permission required")

        dashboard = security_monitor.get_dashboard_data()
        return JSONResponse(content=dashboard)

    @app.get("/security/waf/stats")
    async def waf_statistics(api_key: APIKey = Depends(validate_api_request)):
        """Get WAF statistics (requires authentication)"""
        # Check permission
        if "admin" not in api_key.permissions:
            raise HTTPException(status_code=403, detail="Admin permission required")

        # Get WAF middleware from app
        for middleware in app.middleware:
            if isinstance(middleware, WAFMiddleware):
                return middleware.get_statistics()

        return {"error": "WAF not enabled"}

    # ========================================================================
    # WebSocket Endpoint for Real-Time Security Monitoring
    # ========================================================================

    @app.websocket("/ws/security")
    async def security_websocket(
        websocket: WebSocket,
        api_key: str = None
    ):
        """WebSocket for real-time security monitoring (requires admin key)"""

        # Validate API key
        if not api_key or api_key not in ["admin_key"]:  # Replace with proper validation
            await websocket.close(code=1008, reason="Unauthorized")
            return

        await websocket.accept()
        await security_monitor.add_websocket_client(websocket)

        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except WebSocketDisconnect:
            await security_monitor.remove_websocket_client(websocket)

    # ========================================================================
    # AR Query Endpoints (Protected)
    # ========================================================================

    @app.post("/ar/query")
    async def ar_query(
        request: Request,
        query: ARQueryRequest,
        api_key: APIKey = Depends(validate_api_request)
    ):
        """
        Process AR query with full security

        Security features:
        - API key authentication
        - Rate limiting
        - Input validation
        - Security monitoring
        """

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Check rate limit
        if security_config.enable_rate_limiting:
            if not await rate_limiter.check_rate_limit(
                client_ip,
                "/ar/query",
                security_monitor
            ):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Log successful authentication
        logger.info(f"AR query from {api_key.client_name} ({client_ip}): {query.text[:50]}...")

        try:
            # Process AR query (implement your logic here)
            # This is where you'd integrate with Elle/HoloLoom

            # Example response
            response = {
                "response": f"Processing AR query: {query.text}",
                "confidence": 0.95,
                "session_id": query.session_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            return JSONResponse(content=response)

        except Exception as e:
            logger.error(f"Error processing AR query: {e}")

            # Log security event
            await security_monitor.log_event(
                event_type=SecurityEventType.ANOMALY,
                client_ip=client_ip,
                severity=AlertSeverity.WARNING,
                details={"error": str(e)},
                user_id=api_key.client_name,
                request_path="/ar/query"
            )

            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/ar/vision/detect_objects")
    async def detect_objects(
        request: Request,
        file: UploadFile = File(...),
        api_key: APIKey = Depends(validate_api_request)
    ):
        """
        Detect objects in uploaded image with security validation

        Security features:
        - File size validation
        - Format validation
        - Content verification
        - Rate limiting
        """

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Check rate limit (stricter for vision endpoints)
        if security_config.enable_rate_limiting:
            if not await rate_limiter.check_rate_limit(
                client_ip,
                "/ar/vision/detect_objects",
                security_monitor
            ):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Validate uploaded file
        await validate_image_upload(file)

        try:
            # Read image
            contents = await file.read()
            image = Image.open(BytesIO(contents))

            # Convert to numpy array
            image_array = np.array(image)

            # Process with object detector (implement your logic)
            # detector = create_object_detector()
            # objects = await detector.detect(image_array)

            # Example response
            response = {
                "objects": [
                    {
                        "label": "example",
                        "confidence": 0.95,
                        "bbox": [100, 100, 200, 200]
                    }
                ],
                "timestamp": datetime.utcnow().isoformat()
            }

            return JSONResponse(content=response)

        except Exception as e:
            logger.error(f"Error in object detection: {e}")

            # Log security event
            await security_monitor.log_event(
                event_type=SecurityEventType.ANOMALY,
                client_ip=client_ip,
                severity=AlertSeverity.WARNING,
                details={"error": str(e), "endpoint": "detect_objects"},
                user_id=api_key.client_name,
                request_path="/ar/vision/detect_objects"
            )

            raise HTTPException(status_code=500, detail="Object detection failed")

    # ========================================================================
    # API Key Management Endpoints
    # ========================================================================

    @app.post("/admin/api-keys/generate")
    async def generate_api_key(
        client_name: str,
        expires_in_days: Optional[int] = 30,
        rate_limit_qps: Optional[float] = 100.0,
        admin_key: str = None
    ):
        """Generate new API key (requires admin key)"""

        # Simple admin check (replace with proper admin authentication)
        if admin_key != os.getenv("ADMIN_KEY", "admin_secret"):
            raise HTTPException(status_code=403, detail="Invalid admin key")

        key_id, secret = signature_validator.generate_api_key(
            client_name=client_name,
            expires_in_days=expires_in_days,
            rate_limit_qps=rate_limit_qps
        )

        return {
            "key_id": key_id,
            "secret": secret,
            "client_name": client_name,
            "expires_in_days": expires_in_days,
            "rate_limit_qps": rate_limit_qps
        }

    @app.post("/admin/api-keys/{key_id}/revoke")
    async def revoke_api_key(
        key_id: str,
        admin_key: str = None
    ):
        """Revoke API key (requires admin key)"""

        # Simple admin check
        if admin_key != os.getenv("ADMIN_KEY", "admin_secret"):
            raise HTTPException(status_code=403, detail="Invalid admin key")

        success = signature_validator.revoke_api_key(key_id)

        if success:
            return {"message": f"API key {key_id} revoked successfully"}
        else:
            raise HTTPException(status_code=404, detail="API key not found")

    return app


# ============================================================================
# Utility Functions
# ============================================================================

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


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Load security configuration from environment
    security_config = SecurityConfig(
        enable_waf=os.getenv("ENABLE_WAF", "true").lower() == "true",
        enable_signing=os.getenv("ENABLE_SIGNING", "true").lower() == "true",
        enable_monitoring=os.getenv("ENABLE_MONITORING", "true").lower() == "true",
        enable_rate_limiting=os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true",
        waf_block_on_violation=os.getenv("WAF_BLOCK", "true").lower() == "true",
        smtp_config={
            "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
            "port": int(os.getenv("SMTP_PORT", "587")),
            "username": os.getenv("SMTP_USERNAME"),
            "password": os.getenv("SMTP_PASSWORD"),
            "from_address": os.getenv("SMTP_FROM", "security@hololoom.io"),
            "to_address": os.getenv("SMTP_TO", "admin@hololoom.io"),
            "use_tls": True
        } if os.getenv("SMTP_HOST") else None,
        slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL")
    )

    # Create secured application
    app = create_secured_app(security_config)

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )