#!/usr/bin/env python3
"""
Federation Wire Protocol
========================

Part of Phase 4: Open & Safe Agent Federation (December 2025)
Day 16: Wire Protocol Specification

JSON-RPC 2.0 wire protocol for HoloLoom federation communication.
All federation messages follow this format for interoperability.

Wire Format:
    {
        "jsonrpc": "2.0",
        "method": "hololoom.rag.recall",
        "params": {...},
        "id": "req-001",
        "meta": {
            "sender_id": "abc123...",
            "timestamp": 1702300800,
            "nonce": "unique-id",
            "signature": "base64..."
        }
    }

Author: Claude Code (Phase 4 - December 2025)
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

# ============================================================================
#  JSON-RPC 2.0 Error Codes
# ============================================================================

class ErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 error codes plus HoloLoom extensions."""

    # Standard JSON-RPC errors (-32768 to -32000 reserved)
    PARSE_ERROR = -32700       # Invalid JSON
    INVALID_REQUEST = -32600   # Not a valid Request object
    METHOD_NOT_FOUND = -32601  # Method does not exist
    INVALID_PARAMS = -32602    # Invalid method parameters
    INTERNAL_ERROR = -32603    # Internal JSON-RPC error

    # HoloLoom Federation errors (-32000 to -32099)
    SIGNATURE_INVALID = -32001     # Ed25519 signature verification failed
    SIGNATURE_EXPIRED = -32002     # Request timestamp too old
    RATE_LIMITED = -32003          # Rate limit exceeded
    TRUST_INSUFFICIENT = -32004    # Guild trust level too low
    SAFETY_BLOCKED = -32005        # SafetyGuardrails blocked action
    NODE_UNAVAILABLE = -32006      # Target node not reachable
    CAPABILITY_MISSING = -32007    # Node lacks required capability
    NONCE_REUSED = -32008          # Nonce already used (replay attack)
    AUDIT_REQUIRED = -32009        # Action requires audit approval


# Error messages for each code
ERROR_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.PARSE_ERROR: "Parse error",
    ErrorCode.INVALID_REQUEST: "Invalid Request",
    ErrorCode.METHOD_NOT_FOUND: "Method not found",
    ErrorCode.INVALID_PARAMS: "Invalid params",
    ErrorCode.INTERNAL_ERROR: "Internal error",
    ErrorCode.SIGNATURE_INVALID: "Signature verification failed",
    ErrorCode.SIGNATURE_EXPIRED: "Request signature expired",
    ErrorCode.RATE_LIMITED: "Rate limit exceeded",
    ErrorCode.TRUST_INSUFFICIENT: "Insufficient trust level",
    ErrorCode.SAFETY_BLOCKED: "Action blocked by safety guardrails",
    ErrorCode.NODE_UNAVAILABLE: "Target node unavailable",
    ErrorCode.CAPABILITY_MISSING: "Required capability not available",
    ErrorCode.NONCE_REUSED: "Nonce already used",
    ErrorCode.AUDIT_REQUIRED: "Action requires audit approval",
}


# ============================================================================
#  Data Classes
# ============================================================================

@dataclass(frozen=True, slots=True)
class RPCError:
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {"code": self.code, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        return result

    @classmethod
    def from_code(
        cls,
        code: ErrorCode,
        data: dict[str, Any] | None = None,
    ) -> RPCError:
        """Create error from standard code."""
        return cls(
            code=code.value,
            message=ERROR_MESSAGES.get(code, "Unknown error"),
            data=data,
        )


@dataclass(slots=True)
class RequestMeta:
    """Metadata for signed federation requests."""

    sender_id: str                    # Node ID (40 hex chars)
    timestamp: float                  # Unix timestamp
    nonce: str                        # Unique request ID
    signature: bytes = field(default=b"")  # Ed25519 signature
    sender_public_key: bytes = field(default=b"")  # 32-byte public key

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "signature": base64.b64encode(self.signature).decode() if self.signature else "",
            "sender_public_key": base64.b64encode(self.sender_public_key).decode() if self.sender_public_key else "",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequestMeta:
        """Create from dict (e.g., parsed JSON)."""
        return cls(
            sender_id=data.get("sender_id", ""),
            timestamp=data.get("timestamp", 0.0),
            nonce=data.get("nonce", ""),
            signature=base64.b64decode(data.get("signature", "")) if data.get("signature") else b"",
            sender_public_key=base64.b64decode(data.get("sender_public_key", "")) if data.get("sender_public_key") else b"",
        )


@dataclass(slots=True)
class RPCRequest:
    """JSON-RPC 2.0 request with federation metadata."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str | None = None          # None for notifications
    meta: RequestMeta | None = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
        }
        if self.params:
            result["params"] = self.params
        if self.id is not None:
            result["id"] = self.id
        if self.meta is not None:
            result["meta"] = self.meta.to_dict()
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def get_signing_payload(self) -> bytes:
        """Get bytes to sign (method + params + id + timestamp + nonce)."""
        payload_dict = {
            "method": self.method,
            "params": self.params,
            "id": self.id,
        }
        if self.meta:
            payload_dict["timestamp"] = self.meta.timestamp
            payload_dict["nonce"] = self.meta.nonce
        return json.dumps(payload_dict, sort_keys=True, separators=(",", ":")).encode()


@dataclass(slots=True)
class RPCResponse:
    """JSON-RPC 2.0 response."""

    id: str | None
    result: Any | None = None
    error: RPCError | None = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        response: dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
        }
        if self.error is not None:
            response["error"] = self.error.to_dict()
        else:
            response["result"] = self.result
        return response

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @property
    def is_error(self) -> bool:
        """Check if this is an error response."""
        return self.error is not None


# ============================================================================
#  JSON-RPC Builder
# ============================================================================

class JSONRPCBuilder:
    """
    Builder for JSON-RPC 2.0 messages with HoloLoom federation extensions.

    Example:
        >>> builder = JSONRPCBuilder(sender_id="abc123")
        >>> request = builder.build_request(
        ...     method="hololoom.rag.recall",
        ...     params={"query": "What is Thompson Sampling?", "k": 10}
        ... )
        >>> print(request.to_json())
    """

    def __init__(
        self,
        sender_id: str = "",
        default_timeout_seconds: float = 30.0,
    ):
        """
        Initialize builder.

        Args:
            sender_id: Node ID for request metadata
            default_timeout_seconds: Default request timeout
        """
        self._sender_id = sender_id
        self._default_timeout = default_timeout_seconds
        self._request_counter = 0

    def build_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        request_id: str | None = None,
        include_meta: bool = True,
    ) -> RPCRequest:
        """
        Build a JSON-RPC 2.0 request.

        Args:
            method: RPC method name (e.g., "hololoom.rag.recall")
            params: Method parameters
            request_id: Optional request ID (auto-generated if None)
            include_meta: Include federation metadata

        Returns:
            RPCRequest ready for signing and sending
        """
        if request_id is None:
            self._request_counter += 1
            request_id = f"req-{self._request_counter:06d}"

        meta = None
        if include_meta and self._sender_id:
            meta = RequestMeta(
                sender_id=self._sender_id,
                timestamp=time.time(),
                nonce=str(uuid.uuid4()),
            )

        return RPCRequest(
            method=method,
            params=params or {},
            id=request_id,
            meta=meta,
        )

    def build_notification(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> RPCRequest:
        """
        Build a JSON-RPC 2.0 notification (no response expected).

        Args:
            method: RPC method name
            params: Method parameters

        Returns:
            RPCRequest without id (notification)
        """
        return RPCRequest(
            method=method,
            params=params or {},
            id=None,  # Notifications have no id
            meta=None,
        )

    def build_response(
        self,
        request_id: str | None,
        result: Any = None,
        error: RPCError | None = None,
    ) -> RPCResponse:
        """
        Build a JSON-RPC 2.0 response.

        Args:
            request_id: ID from original request
            result: Success result (mutually exclusive with error)
            error: Error object (mutually exclusive with result)

        Returns:
            RPCResponse
        """
        return RPCResponse(
            id=request_id,
            result=result,
            error=error,
        )

    def build_error_response(
        self,
        request_id: str | None,
        code: ErrorCode,
        data: dict[str, Any] | None = None,
    ) -> RPCResponse:
        """
        Build an error response from standard error code.

        Args:
            request_id: ID from original request
            code: Standard error code
            data: Additional error data

        Returns:
            RPCResponse with error
        """
        return RPCResponse(
            id=request_id,
            error=RPCError.from_code(code, data),
        )

    def build_batch(
        self,
        requests: Sequence[RPCRequest],
    ) -> list[dict[str, Any]]:
        """
        Build a batch request (array of requests).

        Args:
            requests: Sequence of RPCRequest objects

        Returns:
            List of request dicts (JSON-serializable)
        """
        return [req.to_dict() for req in requests]


# ============================================================================
#  Request Validator
# ============================================================================

class RequestValidator:
    """
    Validates JSON-RPC 2.0 requests for correctness and security.

    Example:
        >>> validator = RequestValidator()
        >>> result = validator.validate(raw_json)
        >>> if result.valid:
        ...     request = result.request
        ... else:
        ...     print(result.error)
    """

    # Maximum allowed timestamp age (5 minutes)
    MAX_TIMESTAMP_AGE_SECONDS = 300.0

    # Maximum request size (1MB)
    MAX_REQUEST_SIZE_BYTES = 1024 * 1024

    # Valid HoloLoom method prefixes
    VALID_METHOD_PREFIXES = (
        "hololoom.rag.",
        "hololoom.inference.",
        "hololoom.pattern.",
        "hololoom.threads.",
        "hololoom.features.",
        "hololoom.convergence.",
        "hololoom.tool.",
        "hololoom.spacetime.",
        "hololoom.node.",
    )

    def __init__(
        self,
        max_timestamp_age: float = MAX_TIMESTAMP_AGE_SECONDS,
        max_request_size: int = MAX_REQUEST_SIZE_BYTES,
        require_signature: bool = True,
    ):
        """
        Initialize validator.

        Args:
            max_timestamp_age: Maximum age of request timestamp (seconds)
            max_request_size: Maximum request size in bytes
            require_signature: Whether to require signed requests
        """
        self._max_timestamp_age = max_timestamp_age
        self._max_request_size = max_request_size
        self._require_signature = require_signature
        self._used_nonces: set[str] = set()

    def validate_json(self, raw_json: str) -> ValidationResult:
        """
        Validate raw JSON and parse into RPCRequest.

        Args:
            raw_json: Raw JSON string

        Returns:
            ValidationResult with request or error
        """
        # Check size
        if len(raw_json.encode()) > self._max_request_size:
            return ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.INVALID_REQUEST,
                    {"reason": f"Request exceeds maximum size ({self._max_request_size} bytes)"},
                ),
            )

        # Parse JSON
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            return ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.PARSE_ERROR,
                    {"reason": str(e)},
                ),
            )

        return self.validate_dict(data)

    def validate_dict(self, data: dict[str, Any]) -> ValidationResult:
        """
        Validate parsed JSON dict.

        Args:
            data: Parsed JSON dict

        Returns:
            ValidationResult with request or error
        """
        # Check required fields
        if not isinstance(data, dict):
            return ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.INVALID_REQUEST,
                    {"reason": "Request must be an object"},
                ),
            )

        if data.get("jsonrpc") != "2.0":
            return ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.INVALID_REQUEST,
                    {"reason": "jsonrpc must be '2.0'"},
                ),
            )

        method = data.get("method")
        if not isinstance(method, str) or not method:
            return ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.INVALID_REQUEST,
                    {"reason": "method must be a non-empty string"},
                ),
            )

        # Validate method name
        if not any(method.startswith(prefix) for prefix in self.VALID_METHOD_PREFIXES):
            return ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.METHOD_NOT_FOUND,
                    {"reason": f"Unknown method prefix: {method}"},
                ),
            )

        # Validate params if present
        params = data.get("params", {})
        if not isinstance(params, dict):
            return ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.INVALID_PARAMS,
                    {"reason": "params must be an object"},
                ),
            )

        # Parse metadata if present
        meta = None
        if "meta" in data:
            meta_result = self._validate_meta(data["meta"])
            if not meta_result.valid:
                return meta_result
            meta = meta_result.meta

        # Build request
        request = RPCRequest(
            method=method,
            params=params,
            id=data.get("id"),
            meta=meta,
            jsonrpc="2.0",
        )

        return ValidationResult(valid=True, request=request)

    def _validate_meta(self, meta_data: Any) -> ValidationResult:
        """Validate request metadata."""
        if not isinstance(meta_data, dict):
            return ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.INVALID_REQUEST,
                    {"reason": "meta must be an object"},
                ),
            )

        # Required meta fields
        sender_id = meta_data.get("sender_id")
        if not isinstance(sender_id, str) or len(sender_id) != 40:
            return ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.INVALID_REQUEST,
                    {"reason": "sender_id must be 40 hex characters"},
                ),
            )

        timestamp = meta_data.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            return ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.INVALID_REQUEST,
                    {"reason": "timestamp must be a number"},
                ),
            )

        # Check timestamp age
        now = time.time()
        age = abs(now - timestamp)
        if age > self._max_timestamp_age:
            return ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.SIGNATURE_EXPIRED,
                    {"reason": f"Request too old ({age:.1f}s > {self._max_timestamp_age}s)"},
                ),
            )

        nonce = meta_data.get("nonce")
        if not isinstance(nonce, str) or not nonce:
            return ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.INVALID_REQUEST,
                    {"reason": "nonce must be a non-empty string"},
                ),
            )

        # Check nonce reuse (replay attack prevention)
        if nonce in self._used_nonces:
            return ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.NONCE_REUSED,
                    {"reason": "Nonce already used"},
                ),
            )
        self._used_nonces.add(nonce)

        # Check signature if required
        signature = meta_data.get("signature", "")
        if self._require_signature and not signature:
            return ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.SIGNATURE_INVALID,
                    {"reason": "Signature required"},
                ),
            )

        # Parse metadata
        meta = RequestMeta.from_dict(meta_data)

        return ValidationResult(valid=True, meta=meta)

    def validate_batch(
        self,
        raw_json: str,
    ) -> list[ValidationResult]:
        """
        Validate a batch request (array of requests).

        Args:
            raw_json: Raw JSON string containing array

        Returns:
            List of ValidationResults (one per request)
        """
        # Check size
        if len(raw_json.encode()) > self._max_request_size:
            return [ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.INVALID_REQUEST,
                    {"reason": "Batch exceeds maximum size"},
                ),
            )]

        # Parse JSON
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            return [ValidationResult(
                valid=False,
                error=RPCError.from_code(ErrorCode.PARSE_ERROR, {"reason": str(e)}),
            )]

        if not isinstance(data, list):
            return [ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.INVALID_REQUEST,
                    {"reason": "Batch must be an array"},
                ),
            )]

        if len(data) == 0:
            return [ValidationResult(
                valid=False,
                error=RPCError.from_code(
                    ErrorCode.INVALID_REQUEST,
                    {"reason": "Batch cannot be empty"},
                ),
            )]

        return [self.validate_dict(item) for item in data]

    def clear_nonce_cache(self) -> None:
        """Clear used nonces (call periodically to prevent memory growth)."""
        self._used_nonces.clear()


@dataclass
class ValidationResult:
    """Result of request validation."""

    valid: bool
    request: RPCRequest | None = None
    meta: RequestMeta | None = None
    error: RPCError | None = None


# ============================================================================
#  Method Registry
# ============================================================================

# HoloLoom RPC method namespace
HOLOLOOM_METHODS: dict[str, dict[str, Any]] = {
    # RAG methods
    "hololoom.rag.recall": {
        "description": "Retrieve relevant memories for a query",
        "params": ["query", "k", "min_confidence", "include_sources"],
        "required": ["query"],
    },
    "hololoom.rag.store": {
        "description": "Store new memory",
        "params": ["content", "metadata", "embedding"],
        "required": ["content"],
    },

    # Inference methods
    "hololoom.inference.generate": {
        "description": "Generate response using model",
        "params": ["prompt", "model", "max_tokens", "stream"],
        "required": ["prompt"],
    },
    "hololoom.inference.capabilities": {
        "description": "Get node inference capabilities",
        "params": [],
        "required": [],
    },

    # Pattern methods
    "hololoom.pattern.select": {
        "description": "Select processing pattern",
        "params": ["query_text", "user_preference"],
        "required": ["query_text"],
    },
    "hololoom.pattern.complexity": {
        "description": "Detect query complexity",
        "params": ["query_text"],
        "required": ["query_text"],
    },

    # Thread methods
    "hololoom.threads.select": {
        "description": "Select relevant threads",
        "params": ["query", "context_ids", "max_threads"],
        "required": ["query"],
    },
    "hololoom.threads.expand": {
        "description": "Expand thread context",
        "params": ["thread_ids", "depth"],
        "required": ["thread_ids"],
    },

    # Feature methods
    "hololoom.features.extract": {
        "description": "Extract features from content",
        "params": ["content", "feature_types"],
        "required": ["content"],
    },
    "hololoom.features.dim": {
        "description": "Get feature dimensions",
        "params": ["feature_type"],
        "required": [],
    },

    # Convergence methods
    "hololoom.convergence.collapse": {
        "description": "Collapse probability to decision",
        "params": ["probabilities", "strategy"],
        "required": ["probabilities"],
    },
    "hololoom.convergence.update": {
        "description": "Update Thompson priors",
        "params": ["tool_id", "success", "confidence"],
        "required": ["tool_id", "success"],
    },

    # Tool methods
    "hololoom.tool.execute": {
        "description": "Execute a tool",
        "params": ["tool_name", "args"],
        "required": ["tool_name"],
    },
    "hololoom.tool.list": {
        "description": "List available tools",
        "params": [],
        "required": [],
    },

    # Spacetime methods
    "hololoom.spacetime.assemble": {
        "description": "Assemble spacetime artifact",
        "params": ["features", "context", "metadata"],
        "required": ["features"],
    },

    # Node methods
    "hololoom.node.ping": {
        "description": "Ping node for liveness",
        "params": [],
        "required": [],
    },
    "hololoom.node.info": {
        "description": "Get node information",
        "params": [],
        "required": [],
    },
    "hololoom.node.capabilities": {
        "description": "Get node capabilities",
        "params": [],
        "required": [],
    },
}


# ============================================================================
#  Convenience Functions
# ============================================================================

def create_builder(sender_id: str = "") -> JSONRPCBuilder:
    """Create a JSON-RPC builder with default settings."""
    return JSONRPCBuilder(sender_id=sender_id)


def create_validator(require_signature: bool = True) -> RequestValidator:
    """Create a request validator with default settings."""
    return RequestValidator(require_signature=require_signature)


def parse_request(raw_json: str, require_signature: bool = True) -> ValidationResult:
    """Parse and validate a single JSON-RPC request."""
    validator = RequestValidator(require_signature=require_signature)
    return validator.validate_json(raw_json)


def parse_batch(raw_json: str, require_signature: bool = True) -> list[ValidationResult]:
    """Parse and validate a batch of JSON-RPC requests."""
    validator = RequestValidator(require_signature=require_signature)
    return validator.validate_batch(raw_json)


# ============================================================================
#  Exports
# ============================================================================

__all__ = [
    # Enums
    "ErrorCode",

    # Data classes
    "RPCError",
    "RequestMeta",
    "RPCRequest",
    "RPCResponse",
    "ValidationResult",

    # Classes
    "JSONRPCBuilder",
    "RequestValidator",

    # Constants
    "ERROR_MESSAGES",
    "HOLOLOOM_METHODS",

    # Functions
    "create_builder",
    "create_validator",
    "parse_request",
    "parse_batch",
]
