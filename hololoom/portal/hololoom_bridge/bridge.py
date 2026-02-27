"""
HoloLoom Bridge: Async HTTP client for HoloLoom server.

Connects Portal's distributed compute to HoloLoom's intelligence:
- Recall: Search semantic memory
- Experience: Store content to knowledge graph
- Weave: Execute full reasoning cycles
- Status: Monitor system health

Graceful fallback if HoloLoom unavailable (returns error responses, never crashes).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import httpx
import json
from pydantic import BaseModel, Field


# ============================================================================
# Pydantic Models
# ============================================================================

class LoomQuery(BaseModel):
    """Query to HoloLoom memory system."""

    text: str = Field(description="Query text")
    k: int = Field(default=5, ge=1, le=100, description="Number of results")
    mode: str = Field(default="fast", description="Query mode: fast, balanced, deep")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "What is Thompson Sampling?",
                "k": 5,
                "mode": "fast",
                "context": {"source": "portal"}
            }
        }


class LoomResult(BaseModel):
    """Result from hololoom."""

    success: bool = Field(description="Query succeeded")
    data: Any = Field(description="Response data")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence 0-1")
    latency_ms: float = Field(default=0.0, description="Query latency")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": [{"text": "Thompson Sampling...", "score": 0.92}],
                "confidence": 0.92,
                "latency_ms": 125.5,
                "error": None,
                "timestamp": "2025-12-03T10:30:00Z"
            }
        }


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BridgeConfig:
    """HoloLoom Bridge configuration."""

    hololoom_url: str = "http://localhost:8000"
    timeout_seconds: int = 30
    retries: int = 2
    fallback_on_error: bool = True
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hololoom_url": self.hololoom_url,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
            "fallback_on_error": self.fallback_on_error,
            "verbose": self.verbose,
        }


# ============================================================================
# HoloLoom Bridge
# ============================================================================

class HoloLoomBridge:
    """
    Async HTTP bridge to HoloLoom server.

    Provides clean interface to HoloLoom's memory and reasoning systems,
    with graceful fallback if server unavailable.

    Example:
        >>> bridge = HoloLoomBridge()
        >>> result = await bridge.recall("Thompson Sampling", k=5)
        >>> print(f"Found {len(result.data)} memories")
        >>> await bridge.experience("Learned about Thompson Sampling")
        >>> weave = await bridge.weave("Explain TS", mode="verify")
        >>> print(weave.data)
    """

    def __init__(self, config: Optional[BridgeConfig] = None):
        """Initialize bridge with optional config."""
        self.config = config or BridgeConfig()
        self.client: Optional[httpx.AsyncClient] = None
        self._available = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if not self.client:
            self.client = httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
                limits=httpx.Limits(max_keepalive_connections=5)
            )

    async def _check_availability(self) -> bool:
        """Check if HoloLoom server is available."""
        try:
            await self._ensure_client()
            response = await self.client.get(
                f"{self.config.hololoom_url}/health",
                timeout=5.0
            )
            self._available = response.status_code == 200
            return self._available
        except Exception as e:
            if self.config.verbose:
                print(f"HoloLoom unavailable: {e}")
            self._available = False
            return False

    async def recall(
        self,
        query: str,
        k: int = 5,
        mode: str = "fast",
        context: Optional[Dict[str, Any]] = None
    ) -> LoomResult:
        """
        Search HoloLoom semantic memory.

        Args:
            query: Search query text
            k: Number of results (1-100)
            mode: Query mode (fast, balanced, deep)
            context: Optional additional context

        Returns:
            LoomResult with matching memories
        """
        loom_query = LoomQuery(text=query, k=k, mode=mode, context=context)

        try:
            await self._ensure_client()

            response = await self.client.post(
                f"{self.config.hololoom_url}/query",
                json=loom_query.model_dump(),
            )

            if response.status_code == 200:
                data = response.json()
                return LoomResult(
                    success=True,
                    data=data.get("results", []),
                    confidence=data.get("confidence", 0.0),
                    latency_ms=data.get("latency_ms", 0.0),
                )
            else:
                return LoomResult(
                    success=False,
                    data=[],
                    error=f"HTTP {response.status_code}",
                )

        except Exception as e:
            error_msg = str(e)
            if self.config.verbose:
                print(f"Recall error: {error_msg}")

            if self.config.fallback_on_error:
                return LoomResult(
                    success=False,
                    data=[],
                    error=error_msg,
                )
            raise

    async def experience(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store content to HoloLoom memory.

        Args:
            content: Content to store
            metadata: Optional metadata

        Returns:
            Memory ID if successful, empty string if failed
        """
        try:
            await self._ensure_client()

            payload = {
                "content": content,
                "metadata": metadata or {},
            }

            response = await self.client.post(
                f"{self.config.hololoom_url}/experience",
                json=payload,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("memory_id", "")
            else:
                if self.config.verbose:
                    print(f"Experience failed: HTTP {response.status_code}")
                return ""

        except Exception as e:
            if self.config.verbose:
                print(f"Experience error: {e}")
            return ""

    async def weave(
        self,
        query: str,
        mode: str = "fast"
    ) -> LoomResult:
        """
        Execute full HoloLoom weaving cycle (reasoning).

        Args:
            query: Query to reason about
            mode: Complexity mode (fast, balanced, deep, research)

        Returns:
            LoomResult with reasoning outcome
        """
        try:
            await self._ensure_client()

            payload = {
                "query": query,
                "mode": mode,
            }

            response = await self.client.post(
                f"{self.config.hololoom_url}/weave",
                json=payload,
            )

            if response.status_code == 200:
                data = response.json()
                return LoomResult(
                    success=True,
                    data=data.get("response", ""),
                    confidence=data.get("confidence", 0.0),
                    latency_ms=data.get("latency_ms", 0.0),
                )
            else:
                return LoomResult(
                    success=False,
                    data="",
                    error=f"HTTP {response.status_code}",
                )

        except Exception as e:
            error_msg = str(e)
            if self.config.verbose:
                print(f"Weave error: {error_msg}")

            if self.config.fallback_on_error:
                return LoomResult(
                    success=False,
                    data="",
                    error=error_msg,
                )
            raise

    async def status(self) -> Dict[str, Any]:
        """
        Get HoloLoom system status.

        Returns:
            Status dictionary with health, metrics, etc.
        """
        try:
            await self._ensure_client()

            response = await self.client.get(
                f"{self.config.hololoom_url}/health",
                timeout=5.0
            )

            if response.status_code == 200:
                status = response.json()
                return {
                    "available": True,
                    "status": status,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            else:
                return {
                    "available": False,
                    "error": f"HTTP {response.status_code}",
                    "timestamp": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def close(self):
        """Close HTTP client connection."""
        if self.client:
            await self.client.aclose()
            self.client = None
