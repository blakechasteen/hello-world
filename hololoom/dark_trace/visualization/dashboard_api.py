"""
Dark Trace Visualization: Dashboard API
========================================

FastAPI-based REST and WebSocket API for Dark Trace dashboard.
Provides endpoints for historical data, real-time streaming,
and dashboard configuration.

Endpoints:
- GET /api/dark-trace/status - Server status
- GET /api/dark-trace/features - Feature list with labels
- GET /api/dark-trace/layers - Layer information
- GET /api/dark-trace/statistics - Current statistics
- GET /api/dark-trace/history - Activation history
- POST /api/dark-trace/steering - Apply steering vector
- WS /ws/dark-trace/stream - Real-time activation streaming

Usage:
    from hololoom.dark_trace.visualization.dashboard_api import (
        DarkTraceDashboardAPI,
        create_dashboard_api,
    )

    api = create_dashboard_api(dark_trace_engine)
    app.include_router(api.router, prefix="/api/dark-trace")

Created: 2025-12-28
"""

from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Any,
    Optional,
    TYPE_CHECKING,
)
from datetime import datetime
import asyncio
import logging
import time

# Try to import FastAPI, gracefully degrade if not available
try:
    from fastapi import (
        APIRouter,
        WebSocket,
        WebSocketDisconnect,
        HTTPException,
        Query,
        Body,
    )
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = Any  # Type stub
    WebSocket = Any
    BaseModel = object

if TYPE_CHECKING:
    from hololoom.dark_trace.engine import DarkTraceEngine
    from hololoom.dark_trace.visualization.streaming_server import (
        DarkTraceStreamingServer,
    )


logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models (for FastAPI validation)
# =============================================================================


if FASTAPI_AVAILABLE:
    class FeatureInfo(BaseModel):
        """Feature information model."""
        index: int = Field(..., description="Feature index")
        label: Optional[str] = Field(None, description="Feature label")
        namespace: str = Field(..., description="Feature namespace (sae/semantic)")
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class LayerInfo(BaseModel):
        """Layer information model."""
        index: int = Field(..., description="Layer index")
        n_features: int = Field(..., description="Number of features")
        feature_dim: int = Field(..., description="Feature dimension")
        has_sae: bool = Field(False, description="Whether layer has SAE")

    class ActivationEntry(BaseModel):
        """Activation history entry."""
        timestamp: float = Field(..., description="Unix timestamp")
        layer: int = Field(..., description="Layer index")
        feature_index: int = Field(..., description="Feature index")
        activation: float = Field(..., description="Activation value")
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class SteeringRequest(BaseModel):
        """Steering vector request."""
        features: Dict[str, float] = Field(
            ...,
            description="Feature name/index to steering coefficient mapping"
        )
        scale: float = Field(1.0, description="Overall scaling factor")

    class SteeringResponse(BaseModel):
        """Steering vector response."""
        success: bool
        features_applied: int
        features_blocked: int
        message: str

    class StatusResponse(BaseModel):
        """Server status response."""
        status: str
        timestamp: str
        engine_available: bool
        streaming_available: bool
        client_count: int
        statistics: Dict[str, Any]

else:
    # Stub classes when FastAPI not available
    class FeatureInfo:
        pass

    class LayerInfo:
        pass

    class ActivationEntry:
        pass

    class SteeringRequest:
        pass

    class SteeringResponse:
        pass

    class StatusResponse:
        pass


# =============================================================================
# Dashboard API Configuration
# =============================================================================


@dataclass
class DashboardAPIConfig:
    """Configuration for dashboard API."""
    prefix: str = "/api/dark-trace"
    enable_streaming: bool = True
    enable_steering: bool = True
    max_history_entries: int = 1000
    history_retention_seconds: float = 3600.0  # 1 hour
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


# =============================================================================
# Dashboard API
# =============================================================================


class DarkTraceDashboardAPI:
    """
    FastAPI-based dashboard API for Dark Trace.

    Provides REST endpoints for:
    - Feature information and labels
    - Layer configuration
    - Statistics and metrics
    - Activation history
    - Steering vector application

    Plus WebSocket endpoint for real-time streaming.

    Usage:
        api = DarkTraceDashboardAPI(engine, config)
        app.include_router(api.router)
    """

    def __init__(
        self,
        engine: Optional["DarkTraceEngine"] = None,
        streaming_server: Optional["DarkTraceStreamingServer"] = None,
        config: Optional[DashboardAPIConfig] = None,
    ):
        """
        Initialize dashboard API.

        Args:
            engine: Dark Trace engine instance
            streaming_server: Optional streaming server
            config: API configuration
        """
        if not FASTAPI_AVAILABLE:
            raise RuntimeError(
                "FastAPI not installed. Install with: pip install fastapi"
            )

        self.engine = engine
        self.streaming_server = streaming_server
        self.config = config or DashboardAPIConfig()

        # Activation history buffer
        self._activation_history: List[Dict[str, Any]] = []
        self._history_lock = asyncio.Lock()

        # WebSocket clients
        self._ws_clients: Dict[str, WebSocket] = {}

        # Create router
        self.router = APIRouter(tags=["Dark Trace"])
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Set up API routes."""
        # Status endpoint
        @self.router.get("/status", response_model=StatusResponse)
        async def get_status():
            """Get server and engine status."""
            return StatusResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                engine_available=self.engine is not None,
                streaming_available=self.streaming_server is not None,
                client_count=len(self._ws_clients),
                statistics=self._get_statistics(),
            )

        # Features endpoint
        @self.router.get("/features", response_model=List[FeatureInfo])
        async def get_features(
            namespace: Optional[str] = Query(None, description="Filter by namespace"),
            limit: int = Query(100, description="Maximum features to return"),
            offset: int = Query(0, description="Offset for pagination"),
        ):
            """Get list of features with labels."""
            if not self.engine:
                raise HTTPException(status_code=503, detail="Engine not available")

            features = self._get_features(namespace, limit, offset)
            return features

        # Layers endpoint
        @self.router.get("/layers", response_model=List[LayerInfo])
        async def get_layers():
            """Get layer information."""
            if not self.engine:
                raise HTTPException(status_code=503, detail="Engine not available")

            return self._get_layers()

        # Statistics endpoint
        @self.router.get("/statistics")
        async def get_statistics():
            """Get current statistics."""
            return self._get_statistics()

        # History endpoint
        @self.router.get("/history", response_model=List[ActivationEntry])
        async def get_history(
            layer: Optional[int] = Query(None, description="Filter by layer"),
            feature: Optional[int] = Query(None, description="Filter by feature"),
            limit: int = Query(100, description="Maximum entries"),
            since: Optional[float] = Query(None, description="Unix timestamp cutoff"),
        ):
            """Get activation history."""
            return await self._get_history(layer, feature, limit, since)

        # Steering endpoint
        if self.config.enable_steering:
            @self.router.post("/steering", response_model=SteeringResponse)
            async def apply_steering(request: SteeringRequest):
                """Apply steering vector."""
                if not self.engine:
                    raise HTTPException(status_code=503, detail="Engine not available")

                return await self._apply_steering(request)

        # Clear steering endpoint
        if self.config.enable_steering:
            @self.router.delete("/steering")
            async def clear_steering():
                """Clear all steering vectors."""
                if not self.engine:
                    raise HTTPException(status_code=503, detail="Engine not available")

                self.engine.clear_steering()
                return {"success": True, "message": "Steering cleared"}

        # Feature details endpoint
        @self.router.get("/features/{feature_id}")
        async def get_feature_details(feature_id: str):
            """Get detailed information about a specific feature."""
            if not self.engine:
                raise HTTPException(status_code=503, detail="Engine not available")

            details = self._get_feature_details(feature_id)
            if not details:
                raise HTTPException(status_code=404, detail="Feature not found")
            return details

        # WebSocket endpoint
        if self.config.enable_streaming:
            @self.router.websocket("/stream")
            async def websocket_endpoint(websocket: WebSocket):
                """WebSocket endpoint for real-time streaming."""
                await self._handle_websocket(websocket)

    def _get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = {
            "timestamp": time.time(),
            "history_size": len(self._activation_history),
            "ws_clients": len(self._ws_clients),
        }

        if self.engine:
            try:
                engine_stats = self.engine.get_statistics()
                stats.update(engine_stats)
            except Exception as e:
                stats["engine_error"] = str(e)

        if self.streaming_server:
            stats["streaming"] = {
                "running": self.streaming_server.is_running,
                "clients": self.streaming_server.client_count,
            }

        return stats

    def _get_features(
        self,
        namespace: Optional[str],
        limit: int,
        offset: int,
    ) -> List[Dict[str, Any]]:
        """Get feature list."""
        features = []

        if not self.engine:
            return features

        # Try to get features from registry
        try:
            registry = getattr(self.engine, 'registry', None)
            if registry:
                all_features = list(registry.get_all_features().values())

                # Filter by namespace
                if namespace:
                    all_features = [
                        f for f in all_features
                        if f.namespace == namespace
                    ]

                # Paginate
                paginated = all_features[offset:offset + limit]

                for f in paginated:
                    features.append({
                        "index": f.index,
                        "label": f.label,
                        "namespace": f.namespace,
                        "metadata": f.metadata or {},
                    })
        except Exception as e:
            logger.warning(f"Error getting features: {e}")

        return features

    def _get_layers(self) -> List[Dict[str, Any]]:
        """Get layer information."""
        layers = []

        if not self.engine:
            return layers

        try:
            config = getattr(self.engine, 'config', None)
            if config:
                # Get from config
                n_layers = getattr(config, 'n_layers', 1)
                feature_dim = getattr(config, 'input_dim', 384)
                sae_dim = getattr(config, 'sae_dim', 4096)

                for i in range(n_layers):
                    layers.append({
                        "index": i,
                        "n_features": sae_dim,
                        "feature_dim": feature_dim,
                        "has_sae": True,
                    })
        except Exception as e:
            logger.warning(f"Error getting layers: {e}")

        return layers

    async def _get_history(
        self,
        layer: Optional[int],
        feature: Optional[int],
        limit: int,
        since: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Get activation history with filters."""
        async with self._history_lock:
            filtered = self._activation_history.copy()

        # Apply filters
        if layer is not None:
            filtered = [e for e in filtered if e.get("layer") == layer]

        if feature is not None:
            filtered = [e for e in filtered if e.get("feature_index") == feature]

        if since is not None:
            filtered = [e for e in filtered if e.get("timestamp", 0) >= since]

        # Sort by timestamp descending
        filtered.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        # Limit
        return filtered[:limit]

    async def _apply_steering(self, request: SteeringRequest) -> Dict[str, Any]:
        """Apply steering vector."""
        if not self.engine:
            return {
                "success": False,
                "features_applied": 0,
                "features_blocked": 0,
                "message": "Engine not available",
            }

        try:
            result = self.engine.steer(request.features)

            return {
                "success": True,
                "features_applied": len(result.features_applied),
                "features_blocked": len(result.features_blocked),
                "message": f"Applied steering to {len(result.features_applied)} features",
            }
        except Exception as e:
            return {
                "success": False,
                "features_applied": 0,
                "features_blocked": 0,
                "message": str(e),
            }

    def _get_feature_details(self, feature_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed feature information."""
        if not self.engine:
            return None

        try:
            registry = getattr(self.engine, 'registry', None)
            if registry:
                feature = registry.get_feature(feature_id)
                if feature:
                    return {
                        "id": feature_id,
                        "index": feature.index,
                        "label": feature.label,
                        "namespace": feature.namespace,
                        "metadata": feature.metadata or {},
                        "correlations": [],  # Could be populated from registry
                    }
        except Exception as e:
            logger.warning(f"Error getting feature details: {e}")

        return None

    async def _handle_websocket(self, websocket: WebSocket) -> None:
        """Handle WebSocket connection."""
        await websocket.accept()

        client_id = f"ws_{id(websocket)}_{int(time.time())}"
        self._ws_clients[client_id] = websocket

        logger.info(f"WebSocket client connected: {client_id}")

        try:
            # Send initial state
            await websocket.send_json({
                "type": "connected",
                "client_id": client_id,
                "timestamp": time.time(),
            })

            # Listen for messages
            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=30.0,
                    )

                    # Handle client messages
                    await self._handle_ws_message(client_id, data)

                except asyncio.TimeoutError:
                    # Send heartbeat
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": time.time(),
                    })

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning(f"WebSocket error for {client_id}: {e}")
        finally:
            self._ws_clients.pop(client_id, None)
            logger.info(f"WebSocket client disconnected: {client_id}")

    async def _handle_ws_message(
        self,
        client_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Handle incoming WebSocket message."""
        msg_type = data.get("type", "")

        if msg_type == "subscribe":
            # Handle subscription request
            logger.debug(f"Client {client_id} subscribed: {data}")

        elif msg_type == "request_history":
            # Send recent history
            history = await self._get_history(None, None, 50, None)
            ws = self._ws_clients.get(client_id)
            if ws:
                await ws.send_json({
                    "type": "history",
                    "data": history,
                })

    # -------------------------------------------------------------------------
    # Public API for recording activations
    # -------------------------------------------------------------------------

    async def record_activation(
        self,
        layer: int,
        feature_index: int,
        activation: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an activation for history and streaming.

        Args:
            layer: Layer index
            feature_index: Feature index
            activation: Activation value
            metadata: Optional metadata
        """
        entry = {
            "timestamp": time.time(),
            "layer": layer,
            "feature_index": feature_index,
            "activation": activation,
            "metadata": metadata or {},
        }

        # Add to history
        async with self._history_lock:
            self._activation_history.append(entry)

            # Cleanup old entries
            cutoff = time.time() - self.config.history_retention_seconds
            self._activation_history = [
                e for e in self._activation_history
                if e.get("timestamp", 0) > cutoff
            ]

            # Limit size
            if len(self._activation_history) > self.config.max_history_entries:
                self._activation_history = self._activation_history[
                    -self.config.max_history_entries:
                ]

        # Broadcast to WebSocket clients
        await self._broadcast_ws({
            "type": "activation",
            "data": entry,
        })

    async def record_activations_batch(
        self,
        layer: int,
        feature_indices: List[int],
        activations: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a batch of activations.

        Args:
            layer: Layer index
            feature_indices: Feature indices
            activations: Activation values
            metadata: Optional metadata
        """
        timestamp = time.time()
        entries = []

        for idx, val in zip(feature_indices, activations):
            entries.append({
                "timestamp": timestamp,
                "layer": layer,
                "feature_index": idx,
                "activation": val,
                "metadata": metadata or {},
            })

        # Add to history
        async with self._history_lock:
            self._activation_history.extend(entries)

            # Cleanup
            cutoff = timestamp - self.config.history_retention_seconds
            self._activation_history = [
                e for e in self._activation_history
                if e.get("timestamp", 0) > cutoff
            ][-self.config.max_history_entries:]

        # Broadcast
        await self._broadcast_ws({
            "type": "activations_batch",
            "data": {
                "layer": layer,
                "features": [
                    {"index": idx, "activation": val}
                    for idx, val in zip(feature_indices, activations)
                ],
                "timestamp": timestamp,
            },
        })

    async def _broadcast_ws(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all WebSocket clients."""
        disconnected = []

        for client_id, ws in self._ws_clients.items():
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(client_id)

        for client_id in disconnected:
            self._ws_clients.pop(client_id, None)


# =============================================================================
# Factory Functions
# =============================================================================


def create_dashboard_api(
    engine: Optional["DarkTraceEngine"] = None,
    streaming_server: Optional["DarkTraceStreamingServer"] = None,
    config: Optional[DashboardAPIConfig] = None,
) -> DarkTraceDashboardAPI:
    """
    Factory function for dashboard API.

    Args:
        engine: Dark Trace engine
        streaming_server: Streaming server
        config: API configuration

    Returns:
        DarkTraceDashboardAPI instance
    """
    return DarkTraceDashboardAPI(engine, streaming_server, config)


def create_dashboard_router(
    engine: Optional["DarkTraceEngine"] = None,
    streaming_server: Optional["DarkTraceStreamingServer"] = None,
    prefix: str = "/api/dark-trace",
) -> APIRouter:
    """
    Create a FastAPI router for Dark Trace dashboard.

    Args:
        engine: Dark Trace engine
        streaming_server: Streaming server
        prefix: URL prefix

    Returns:
        FastAPI APIRouter
    """
    config = DashboardAPIConfig(prefix=prefix)
    api = DarkTraceDashboardAPI(engine, streaming_server, config)
    return api.router
