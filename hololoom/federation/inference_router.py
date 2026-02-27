#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference Router - Capability-Based Distributed Inference
==========================================================

Part of Phase 4: Open & Safe Agent Federation (December 2025)
Day 19: Distributed Inference

Routes inference requests to capable federation nodes using:
1. Capability matching (which nodes have the required model?)
2. Load balancing (weighted least-connections)
3. Safety gating (signature, trust, rate limit checks)
4. Streaming proxy for token-by-token forwarding

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                 INFERENCE ROUTING                            │
    │                                                               │
    │  Request: generate("Explain...", model="llama3")             │
    │                     │                                         │
    │                     ▼                                         │
    │         ┌─────────────────────┐                              │
    │         │  Capability Router  │ ← Find nodes with llama3     │
    │         └─────────────────────┘                              │
    │                     │                                         │
    │    ┌────────────────┼────────────────┐                       │
    │    ▼                ▼                ▼                        │
    │ [Node A]        [Node B]        [Node C]                     │
    │ llama3 ✓        llama3 ✓        mistral ✗                    │
    │ load: 30%       load: 80%       load: 10%                    │
    │    │                                                          │
    │    └── Selected (lowest load + has capability)               │
    │                     │                                         │
    │                     ▼                                         │
    │         Execute on Node A, stream response                   │
    └─────────────────────────────────────────────────────────────┘

Author: Claude Code (Phase 4 - December 2025)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from hololoom.federation.types import FederationNode, GuildTrustLevel
from hololoom.federation.load_balancer import (
    LoadBalancer,
    SelectionResult,
    LoadBalanceStrategy,
)
from hololoom.federation.safety import (
    FederationSafetyGate,
    SignedRequest,
    SafetyCheckResult,
    FederationSafetyResult,
)
from hololoom.federation.rate_limiter import (
    FederatedRateLimiter,
    RateLimitTier,
    get_tier_for_trust_level,
)
from hololoom.federation.wire_protocol import (
    JSONRPCBuilder,
    RPCRequest,
    RPCResponse,
    RPCError,
    ErrorCode,
)

logger = logging.getLogger(__name__)


# ============================================================================
#  Constants
# ============================================================================

# Default inference method names
INFERENCE_GENERATE_METHOD = "hololoom.inference.generate"
INFERENCE_CAPABILITIES_METHOD = "hololoom.inference.capabilities"

# Default timeouts
DEFAULT_INFERENCE_TIMEOUT_SECONDS = 60.0
DEFAULT_STREAM_TIMEOUT_SECONDS = 120.0

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 0.5


# ============================================================================
#  Data Classes
# ============================================================================

class InferenceStatus(Enum):
    """Status of an inference request."""
    PENDING = auto()
    ROUTING = auto()
    EXECUTING = auto()
    STREAMING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    SAFETY_BLOCKED = auto()
    NO_CAPABLE_NODE = auto()


@dataclass
class InferenceRequest:
    """Request for inference generation."""

    prompt: str
    model: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7
    stream: bool = False
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_params(self) -> Dict[str, Any]:
        """Convert to RPC params dict."""
        return {
            "prompt": self.prompt,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": self.stream,
            **self.metadata,
        }


@dataclass
class InferenceToken:
    """Single token from streaming inference."""

    token: str
    token_index: int
    is_final: bool = False
    cumulative_text: str = ""
    logprobs: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result of an inference request."""

    request_id: str
    status: InferenceStatus
    response_text: str = ""
    model_used: Optional[str] = None
    node_id: Optional[str] = None
    tokens_generated: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == InferenceStatus.COMPLETED

    @property
    def failed(self) -> bool:
        return self.status in (
            InferenceStatus.FAILED,
            InferenceStatus.TIMEOUT,
            InferenceStatus.SAFETY_BLOCKED,
            InferenceStatus.NO_CAPABLE_NODE,
        )


@dataclass
class NodeCapabilities:
    """Inference capabilities of a federation node."""

    node_id: str
    models: Set[str]
    max_tokens: int = 4096
    supports_streaming: bool = True
    current_load: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def supports_model(self, model: str) -> bool:
        """Check if node supports a model."""
        if not model:
            return True  # No specific model required
        return model.lower() in {m.lower() for m in self.models}


# ============================================================================
#  Streaming Proxy
# ============================================================================

class StreamingProxy:
    """
    Proxy for streaming inference responses token-by-token.

    Handles:
    - Token buffering and forwarding
    - Timeout management
    - Error handling and recovery
    - Metrics collection

    Example:
        >>> async with StreamingProxy(node, request) as proxy:
        ...     async for token in proxy.stream():
        ...         print(token.token, end="", flush=True)
    """

    def __init__(
        self,
        node: FederationNode,
        request: InferenceRequest,
        timeout_seconds: float = DEFAULT_STREAM_TIMEOUT_SECONDS,
        on_token: Optional[Callable[[InferenceToken], None]] = None,
    ):
        """
        Initialize streaming proxy.

        Args:
            node: Target federation node
            request: Inference request to stream
            timeout_seconds: Maximum time for entire stream
            on_token: Optional callback for each token
        """
        self._node = node
        self._request = request
        self._timeout = timeout_seconds
        self._on_token = on_token

        self._started_at: Optional[float] = None
        self._token_count = 0
        self._cumulative_text = ""
        self._is_active = False
        self._error: Optional[str] = None

    async def __aenter__(self) -> 'StreamingProxy':
        """Start streaming session."""
        self._started_at = time.time()
        self._is_active = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """End streaming session."""
        self._is_active = False

    async def stream(self) -> AsyncGenerator[InferenceToken, None]:
        """
        Stream tokens from the inference node.

        Yields:
            InferenceToken objects as they arrive

        Raises:
            asyncio.TimeoutError: If stream exceeds timeout
            RuntimeError: If stream encounters an error
        """
        if not self._is_active:
            raise RuntimeError("StreamingProxy not active. Use 'async with' context.")

        try:
            # In production, this would connect to the actual node
            # For now, we simulate token streaming
            async for token in self._simulate_stream():
                self._token_count += 1
                self._cumulative_text += token.token
                token.cumulative_text = self._cumulative_text
                token.token_index = self._token_count - 1

                # Check timeout
                if time.time() - self._started_at > self._timeout:
                    raise asyncio.TimeoutError("Streaming timeout exceeded")

                # Invoke callback if provided
                if self._on_token:
                    self._on_token(token)

                yield token

                if token.is_final:
                    break

        except asyncio.TimeoutError:
            self._error = "Streaming timeout"
            raise
        except Exception as e:
            self._error = str(e)
            raise RuntimeError(f"Streaming error: {e}") from e

    async def _simulate_stream(self) -> AsyncGenerator[InferenceToken, None]:
        """
        Simulate token streaming for testing.

        In production, this would be replaced with actual RPC streaming.
        """
        # Simulated response tokens
        tokens = [
            "This", " is", " a", " simulated", " streaming",
            " response", " from", " the", " federation", " node", "."
        ]

        for i, token_text in enumerate(tokens):
            await asyncio.sleep(0.05)  # Simulate network latency
            yield InferenceToken(
                token=token_text,
                token_index=i,
                is_final=(i == len(tokens) - 1),
            )

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def cumulative_text(self) -> str:
        return self._cumulative_text

    @property
    def elapsed_ms(self) -> float:
        if not self._started_at:
            return 0.0
        return (time.time() - self._started_at) * 1000

    @property
    def error(self) -> Optional[str]:
        return self._error


# ============================================================================
#  Inference Router
# ============================================================================

class InferenceRouter:
    """
    Routes inference requests to capable federation nodes.

    Combines capability matching with load balancing and safety gating
    to select the optimal node for each inference request.

    Example:
        >>> router = InferenceRouter(
        ...     load_balancer=load_balancer,
        ...     safety_gate=safety_gate,
        ...     rate_limiter=rate_limiter,
        ... )
        >>> result = await router.generate(
        ...     InferenceRequest(prompt="Explain quantum computing", model="llama3")
        ... )
        >>> print(result.response_text)
    """

    def __init__(
        self,
        load_balancer: Optional[LoadBalancer] = None,
        safety_gate: Optional[FederationSafetyGate] = None,
        rate_limiter: Optional[FederatedRateLimiter] = None,
        rpc_builder: Optional[JSONRPCBuilder] = None,
        default_model: str = "llama3",
        inference_timeout: float = DEFAULT_INFERENCE_TIMEOUT_SECONDS,
        enable_retries: bool = True,
    ):
        """
        Initialize inference router.

        Args:
            load_balancer: LoadBalancer for node selection
            safety_gate: FederationSafetyGate for request validation
            rate_limiter: FederatedRateLimiter for rate limiting
            rpc_builder: JSONRPCBuilder for RPC message construction
            default_model: Default model if none specified
            inference_timeout: Timeout for inference requests
            enable_retries: Whether to retry on transient failures
        """
        self._load_balancer = load_balancer or LoadBalancer()
        self._safety_gate = safety_gate
        self._rate_limiter = rate_limiter or FederatedRateLimiter()
        self._rpc_builder = rpc_builder or JSONRPCBuilder()

        self._default_model = default_model
        self._timeout = inference_timeout
        self._enable_retries = enable_retries

        # Metrics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_latency_ms = 0.0

    async def generate(
        self,
        request: InferenceRequest,
        sender_node: Optional[FederationNode] = None,
        trust_level: Optional[GuildTrustLevel] = None,
    ) -> InferenceResult:
        """
        Generate text using a capable federation node.

        Args:
            request: Inference request to process
            sender_node: Node making the request (for safety checks)
            trust_level: Trust level for safety/rate limit checks

        Returns:
            InferenceResult with generated text or error
        """
        start_time = time.time()
        self._total_requests += 1

        # Set default model if not specified
        if not request.model:
            request.model = self._default_model

        # Safety gate check (if configured)
        if self._safety_gate and sender_node:
            safety_result = await self._check_safety(request, sender_node, trust_level)
            if not safety_result.allowed:
                self._failed_requests += 1
                return InferenceResult(
                    request_id=request.request_id,
                    status=InferenceStatus.SAFETY_BLOCKED,
                    error=safety_result.denied_reason,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        # Rate limit check
        if sender_node:
            rate_limit_result = self._check_rate_limit(sender_node.node_id, trust_level)
            if not rate_limit_result.allowed:
                self._failed_requests += 1
                return InferenceResult(
                    request_id=request.request_id,
                    status=InferenceStatus.SAFETY_BLOCKED,
                    error=rate_limit_result.reason,
                    metadata=rate_limit_result.details,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        # Find capable node
        selection = self._select_node(request.model, trust_level)
        if selection.node is None:
            self._failed_requests += 1
            return InferenceResult(
                request_id=request.request_id,
                status=InferenceStatus.NO_CAPABLE_NODE,
                error=f"No capable node found for model: {request.model}",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Execute inference
        result = await self._execute_inference(request, selection)

        # Update metrics
        latency_ms = (time.time() - start_time) * 1000
        result.latency_ms = latency_ms
        self._total_latency_ms += latency_ms

        if result.success:
            self._successful_requests += 1
            # Record successful rate limit
            if sender_node:
                self._rate_limiter.record(sender_node.node_id)
        else:
            self._failed_requests += 1

        return result

    async def generate_stream(
        self,
        request: InferenceRequest,
        sender_node: Optional[FederationNode] = None,
        trust_level: Optional[GuildTrustLevel] = None,
    ) -> AsyncGenerator[InferenceToken, None]:
        """
        Generate text with streaming token output.

        Args:
            request: Inference request (stream=True is forced)
            sender_node: Node making the request
            trust_level: Trust level for safety checks

        Yields:
            InferenceToken objects as they are generated
        """
        # Force streaming mode
        request.stream = True

        # Safety checks
        if self._safety_gate and sender_node:
            safety_result = await self._check_safety(request, sender_node, trust_level)
            if not safety_result.allowed:
                yield InferenceToken(
                    token="",
                    token_index=0,
                    is_final=True,
                    metadata={"error": safety_result.denied_reason},
                )
                return

        # Rate limit check
        if sender_node:
            rate_limit_result = self._check_rate_limit(sender_node.node_id, trust_level)
            if not rate_limit_result.allowed:
                yield InferenceToken(
                    token="",
                    token_index=0,
                    is_final=True,
                    metadata={"error": rate_limit_result.reason},
                )
                return

        # Find capable node
        selection = self._select_node(request.model or self._default_model, trust_level)
        if selection.node is None:
            yield InferenceToken(
                token="",
                token_index=0,
                is_final=True,
                metadata={"error": f"No capable node for model: {request.model}"},
            )
            return

        # Stream from node
        async with StreamingProxy(
            node=selection.node,
            request=request,
            timeout_seconds=DEFAULT_STREAM_TIMEOUT_SECONDS,
        ) as proxy:
            async for token in proxy.stream():
                yield token

        # Record successful request
        if sender_node:
            self._rate_limiter.record(sender_node.node_id)

    async def get_capabilities(
        self,
        node_id: Optional[str] = None,
    ) -> Union[NodeCapabilities, List[NodeCapabilities]]:
        """
        Get inference capabilities of federation nodes.

        Args:
            node_id: Specific node ID (returns all if None)

        Returns:
            NodeCapabilities or list of NodeCapabilities
        """
        if node_id:
            # Get specific node capabilities
            models = self._load_balancer.get_models()
            node_ids = self._load_balancer.get_available_nodes(model=None)

            if node_id not in node_ids:
                return NodeCapabilities(
                    node_id=node_id,
                    models=set(),
                    metadata={"error": "Node not found"},
                )

            # Get models this node supports
            node_models = set()
            for model in models:
                available = self._load_balancer.get_available_nodes(model=model)
                if node_id in available:
                    node_models.add(model)

            return NodeCapabilities(
                node_id=node_id,
                models=node_models,
            )

        # Get all node capabilities
        all_capabilities = []
        node_ids = self._load_balancer.get_available_nodes(model=None)

        for nid in node_ids:
            caps = await self.get_capabilities(node_id=nid)
            all_capabilities.append(caps)

        return all_capabilities

    def register_node(
        self,
        node: FederationNode,
        models: Set[str],
        initial_load: float = 0.0,
    ) -> None:
        """
        Register a node with its inference capabilities.

        Args:
            node: Federation node to register
            models: Set of model names supported
            initial_load: Initial load percentage (0.0-1.0)
        """
        self._load_balancer.register_node(
            node=node,
            models=models,
            initial_load=initial_load,
        )

    def unregister_node(self, node_id: str) -> None:
        """Unregister a node from the router."""
        self._load_balancer.unregister_node(node_id)

    def get_available_models(self) -> Set[str]:
        """Get set of all available models across all nodes."""
        return self._load_balancer.get_models()

    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (
                self._successful_requests / self._total_requests
                if self._total_requests > 0 else 0.0
            ),
            "avg_latency_ms": (
                self._total_latency_ms / self._total_requests
                if self._total_requests > 0 else 0.0
            ),
            "available_models": list(self.get_available_models()),
            "registered_nodes": len(self._load_balancer.get_available_nodes()),
        }

    # =========================================================================
    #  Private Methods
    # =========================================================================

    async def _check_safety(
        self,
        request: InferenceRequest,
        sender_node: FederationNode,
        trust_level: Optional[GuildTrustLevel],
    ) -> FederationSafetyResult:
        """Run safety gate checks on request."""
        # Build signed request for safety check
        signed_request = SignedRequest(
            method=INFERENCE_GENERATE_METHOD,
            params=request.to_params(),
            sender_id=sender_node.node_id,
            timestamp=time.time(),
            nonce=str(uuid.uuid4()),
            signature=b"",  # Would be signed in production
            request_id=request.request_id,
        )

        # Run safety checks
        return await self._safety_gate.check_request(
            request=signed_request,
            node=sender_node,
            guild_trust_level=trust_level,
        )

    def _check_rate_limit(
        self,
        identity_id: str,
        trust_level: Optional[GuildTrustLevel],
    ) -> SafetyCheckResult:
        """Check rate limits for identity."""
        tier = RateLimitTier.OBSERVER
        if trust_level:
            tier = get_tier_for_trust_level(trust_level)

        return self._rate_limiter.check(identity_id, tier)

    def _select_node(
        self,
        model: Optional[str],
        required_trust: Optional[GuildTrustLevel],
    ) -> SelectionResult:
        """Select best node for inference request."""
        return self._load_balancer.select_node(
            model=model,
            required_trust=required_trust,
        )

    async def _execute_inference(
        self,
        request: InferenceRequest,
        selection: SelectionResult,
    ) -> InferenceResult:
        """Execute inference on selected node."""
        node = selection.node
        retries = MAX_RETRIES if self._enable_retries else 1
        last_error: Optional[str] = None

        for attempt in range(retries):
            try:
                # Acquire connection
                if not self._load_balancer.acquire_connection(node.node_id):
                    last_error = "Failed to acquire connection"
                    continue

                try:
                    # Build RPC request
                    rpc_request = self._rpc_builder.build_request(
                        method=INFERENCE_GENERATE_METHOD,
                        params=request.to_params(),
                        request_id=request.request_id,
                    )

                    # Execute inference (simulated for now)
                    response_text = await self._call_inference_rpc(
                        node, rpc_request
                    )

                    # Release connection with success
                    self._load_balancer.release_connection(
                        node.node_id,
                        success=True,
                        latency_ms=0.0,  # Would be actual latency
                    )

                    return InferenceResult(
                        request_id=request.request_id,
                        status=InferenceStatus.COMPLETED,
                        response_text=response_text,
                        model_used=request.model,
                        node_id=node.node_id,
                        tokens_generated=len(response_text.split()),
                        metadata={
                            "selection_reason": selection.reason,
                            "alternatives": selection.alternatives,
                        },
                    )

                except asyncio.TimeoutError:
                    self._load_balancer.release_connection(
                        node.node_id,
                        success=False,
                        latency_ms=self._timeout * 1000,
                    )
                    last_error = "Inference timeout"

                except Exception as e:
                    self._load_balancer.release_connection(
                        node.node_id,
                        success=False,
                        latency_ms=0.0,
                    )
                    last_error = str(e)

            except Exception as e:
                last_error = str(e)

            # Wait before retry
            if attempt < retries - 1:
                await asyncio.sleep(RETRY_DELAY_SECONDS * (attempt + 1))

        # All retries failed
        return InferenceResult(
            request_id=request.request_id,
            status=InferenceStatus.FAILED,
            error=last_error,
            node_id=node.node_id if node else None,
        )

    async def _call_inference_rpc(
        self,
        node: FederationNode,
        rpc_request: RPCRequest,
    ) -> str:
        """
        Call inference RPC on node.

        In production, this would make actual HTTP/gRPC calls.
        For now, simulates a response.
        """
        # Simulate network latency
        await asyncio.sleep(0.1)

        # Simulated response
        prompt = rpc_request.params.get("prompt", "")
        return f"[Simulated response from {node.node_id}] This is a generated response for: {prompt[:50]}..."


# ============================================================================
#  Factory Functions
# ============================================================================

def create_inference_router(
    load_balancer: Optional[LoadBalancer] = None,
    safety_gate: Optional[FederationSafetyGate] = None,
    rate_limiter: Optional[FederatedRateLimiter] = None,
    default_model: str = "llama3",
    **kwargs,
) -> InferenceRouter:
    """
    Create a configured InferenceRouter.

    Args:
        load_balancer: LoadBalancer instance (creates default if None)
        safety_gate: FederationSafetyGate instance (creates default if None)
        rate_limiter: FederatedRateLimiter instance (creates default if None)
        default_model: Default model for inference
        **kwargs: Additional arguments passed to InferenceRouter

    Returns:
        Configured InferenceRouter
    """
    if load_balancer is None:
        load_balancer = LoadBalancer(
            strategy=LoadBalanceStrategy.WEIGHTED_LEAST_CONN
        )

    if safety_gate is None:
        try:
            from hololoom.federation.safety import create_federation_safety_gate
            safety_gate = create_federation_safety_gate()
        except ImportError:
            pass

    if rate_limiter is None:
        rate_limiter = FederatedRateLimiter()

    return InferenceRouter(
        load_balancer=load_balancer,
        safety_gate=safety_gate,
        rate_limiter=rate_limiter,
        default_model=default_model,
        **kwargs,
    )


# ============================================================================
#  Exports
# ============================================================================

__all__ = [
    # Enums
    "InferenceStatus",

    # Data classes
    "InferenceRequest",
    "InferenceToken",
    "InferenceResult",
    "NodeCapabilities",

    # Classes
    "StreamingProxy",
    "InferenceRouter",

    # Factory functions
    "create_inference_router",

    # Constants
    "INFERENCE_GENERATE_METHOD",
    "INFERENCE_CAPABILITIES_METHOD",
    "DEFAULT_INFERENCE_TIMEOUT_SECONDS",
    "DEFAULT_STREAM_TIMEOUT_SECONDS",
]
