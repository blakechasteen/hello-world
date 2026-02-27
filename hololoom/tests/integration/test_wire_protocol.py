#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wire Protocol Tests
===================

Comprehensive tests for the JSON-RPC 2.0 wire protocol implementation.

Tests cover:
- JSONRPCBuilder: Request/response construction
- RequestValidator: Validation and security checks
- Error handling: All error codes
- Batch requests: Array handling
- Security: Nonce reuse, timestamp expiry, signature requirements

Author: Claude Code (Phase 4 - December 2025)
"""

import json
import time
import pytest

from HoloLoom.federation.wire_protocol import (
    # Enums
    ErrorCode,
    # Data classes
    RPCError,
    RequestMeta,
    RPCRequest,
    RPCResponse,
    ValidationResult,
    # Classes
    JSONRPCBuilder,
    RequestValidator,
    # Constants
    ERROR_MESSAGES,
    HOLOLOOM_METHODS,
    # Functions
    create_builder,
    create_validator,
    parse_request,
    parse_batch,
)


# ============================================================================
#  Test Fixtures
# ============================================================================

@pytest.fixture
def builder():
    """Create a JSONRPCBuilder with test sender ID."""
    return JSONRPCBuilder(sender_id="a" * 40)


@pytest.fixture
def validator():
    """Create a RequestValidator with relaxed settings for testing."""
    return RequestValidator(
        max_timestamp_age=300.0,
        require_signature=False,  # Relax for most tests
    )


@pytest.fixture
def strict_validator():
    """Create a RequestValidator with strict settings."""
    return RequestValidator(
        max_timestamp_age=300.0,
        require_signature=True,
    )


# ============================================================================
#  RPCError Tests
# ============================================================================

class TestRPCError:
    """Tests for RPCError data class."""

    def test_create_error(self):
        """Test basic error creation."""
        error = RPCError(code=-32600, message="Invalid Request")
        assert error.code == -32600
        assert error.message == "Invalid Request"
        assert error.data is None

    def test_create_error_with_data(self):
        """Test error creation with additional data."""
        error = RPCError(
            code=-32602,
            message="Invalid params",
            data={"param": "query", "reason": "required"},
        )
        assert error.data == {"param": "query", "reason": "required"}

    def test_to_dict(self):
        """Test error serialization."""
        error = RPCError(code=-32600, message="Invalid Request")
        result = error.to_dict()
        assert result == {"code": -32600, "message": "Invalid Request"}

    def test_to_dict_with_data(self):
        """Test error serialization with data."""
        error = RPCError(code=-32600, message="Test", data={"key": "value"})
        result = error.to_dict()
        assert result["data"] == {"key": "value"}

    def test_from_code(self):
        """Test error creation from standard code."""
        error = RPCError.from_code(ErrorCode.PARSE_ERROR)
        assert error.code == -32700
        assert error.message == "Parse error"

    def test_from_code_with_data(self):
        """Test error creation from code with data."""
        error = RPCError.from_code(
            ErrorCode.RATE_LIMITED,
            data={"retry_after": 60},
        )
        assert error.code == -32003
        assert error.data == {"retry_after": 60}


# ============================================================================
#  RequestMeta Tests
# ============================================================================

class TestRequestMeta:
    """Tests for RequestMeta data class."""

    def test_create_meta(self):
        """Test basic metadata creation."""
        meta = RequestMeta(
            sender_id="a" * 40,
            timestamp=1702300800.0,
            nonce="test-nonce",
        )
        assert meta.sender_id == "a" * 40
        assert meta.timestamp == 1702300800.0
        assert meta.nonce == "test-nonce"

    def test_to_dict(self):
        """Test metadata serialization."""
        meta = RequestMeta(
            sender_id="a" * 40,
            timestamp=1702300800.0,
            nonce="test-nonce",
        )
        result = meta.to_dict()
        assert result["sender_id"] == "a" * 40
        assert result["timestamp"] == 1702300800.0
        assert result["nonce"] == "test-nonce"

    def test_to_dict_with_signature(self):
        """Test metadata serialization with signature."""
        meta = RequestMeta(
            sender_id="a" * 40,
            timestamp=1702300800.0,
            nonce="test-nonce",
            signature=b"test-signature",
            sender_public_key=b"x" * 32,
        )
        result = meta.to_dict()
        assert result["signature"] != ""
        assert result["sender_public_key"] != ""

    def test_from_dict(self):
        """Test metadata deserialization."""
        data = {
            "sender_id": "b" * 40,
            "timestamp": 1702300900.0,
            "nonce": "another-nonce",
        }
        meta = RequestMeta.from_dict(data)
        assert meta.sender_id == "b" * 40
        assert meta.timestamp == 1702300900.0
        assert meta.nonce == "another-nonce"


# ============================================================================
#  RPCRequest Tests
# ============================================================================

class TestRPCRequest:
    """Tests for RPCRequest data class."""

    def test_create_request(self):
        """Test basic request creation."""
        request = RPCRequest(
            method="hololoom.rag.recall",
            params={"query": "test"},
            id="req-001",
        )
        assert request.method == "hololoom.rag.recall"
        assert request.params == {"query": "test"}
        assert request.id == "req-001"
        assert request.jsonrpc == "2.0"

    def test_to_dict(self):
        """Test request serialization."""
        request = RPCRequest(
            method="hololoom.rag.recall",
            params={"query": "test"},
            id="req-001",
        )
        result = request.to_dict()
        assert result["jsonrpc"] == "2.0"
        assert result["method"] == "hololoom.rag.recall"
        assert result["params"] == {"query": "test"}
        assert result["id"] == "req-001"

    def test_to_json(self):
        """Test JSON serialization."""
        request = RPCRequest(
            method="hololoom.rag.recall",
            params={"query": "test"},
            id="req-001",
        )
        json_str = request.to_json()
        parsed = json.loads(json_str)
        assert parsed["method"] == "hololoom.rag.recall"

    def test_notification_no_id(self):
        """Test notification (no id field)."""
        request = RPCRequest(
            method="hololoom.node.ping",
            id=None,
        )
        result = request.to_dict()
        assert "id" not in result

    def test_get_signing_payload(self):
        """Test signing payload generation."""
        meta = RequestMeta(
            sender_id="a" * 40,
            timestamp=1702300800.0,
            nonce="test-nonce",
        )
        request = RPCRequest(
            method="hololoom.rag.recall",
            params={"query": "test"},
            id="req-001",
            meta=meta,
        )
        payload = request.get_signing_payload()
        assert isinstance(payload, bytes)
        # Payload should be deterministic JSON
        parsed = json.loads(payload.decode())
        assert parsed["method"] == "hololoom.rag.recall"
        assert parsed["timestamp"] == 1702300800.0


# ============================================================================
#  RPCResponse Tests
# ============================================================================

class TestRPCResponse:
    """Tests for RPCResponse data class."""

    def test_success_response(self):
        """Test successful response."""
        response = RPCResponse(
            id="req-001",
            result={"data": "value"},
        )
        assert response.id == "req-001"
        assert response.result == {"data": "value"}
        assert response.error is None
        assert not response.is_error

    def test_error_response(self):
        """Test error response."""
        response = RPCResponse(
            id="req-001",
            error=RPCError(code=-32600, message="Invalid Request"),
        )
        assert response.is_error
        assert response.error.code == -32600

    def test_to_dict_success(self):
        """Test success response serialization."""
        response = RPCResponse(id="req-001", result={"key": "value"})
        result = response.to_dict()
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == "req-001"
        assert result["result"] == {"key": "value"}
        assert "error" not in result

    def test_to_dict_error(self):
        """Test error response serialization."""
        response = RPCResponse(
            id="req-001",
            error=RPCError(code=-32600, message="Test"),
        )
        result = response.to_dict()
        assert "error" in result
        assert "result" not in result


# ============================================================================
#  JSONRPCBuilder Tests
# ============================================================================

class TestJSONRPCBuilder:
    """Tests for JSONRPCBuilder class."""

    def test_build_request(self, builder):
        """Test basic request building."""
        request = builder.build_request(
            method="hololoom.rag.recall",
            params={"query": "test", "k": 10},
        )
        assert request.method == "hololoom.rag.recall"
        assert request.params == {"query": "test", "k": 10}
        assert request.id is not None
        assert request.meta is not None

    def test_build_request_custom_id(self, builder):
        """Test request with custom ID."""
        request = builder.build_request(
            method="hololoom.rag.recall",
            request_id="custom-id",
        )
        assert request.id == "custom-id"

    def test_build_request_auto_increment_id(self, builder):
        """Test auto-incrementing request IDs."""
        req1 = builder.build_request(method="hololoom.node.ping")
        req2 = builder.build_request(method="hololoom.node.ping")
        assert req1.id != req2.id

    def test_build_request_no_meta(self):
        """Test request without metadata."""
        builder = JSONRPCBuilder()  # No sender_id
        request = builder.build_request(
            method="hololoom.rag.recall",
            include_meta=False,
        )
        assert request.meta is None

    def test_build_notification(self, builder):
        """Test notification building."""
        notification = builder.build_notification(
            method="hololoom.node.ping",
        )
        assert notification.id is None
        assert notification.meta is None

    def test_build_response(self, builder):
        """Test response building."""
        response = builder.build_response(
            request_id="req-001",
            result={"success": True},
        )
        assert response.id == "req-001"
        assert response.result == {"success": True}
        assert not response.is_error

    def test_build_error_response(self, builder):
        """Test error response building."""
        response = builder.build_error_response(
            request_id="req-001",
            code=ErrorCode.RATE_LIMITED,
            data={"retry_after": 30},
        )
        assert response.is_error
        assert response.error.code == -32003
        assert response.error.data == {"retry_after": 30}

    def test_build_batch(self, builder):
        """Test batch request building."""
        requests = [
            builder.build_request(method="hololoom.node.ping"),
            builder.build_request(method="hololoom.node.info"),
        ]
        batch = builder.build_batch(requests)
        assert len(batch) == 2
        assert all(isinstance(r, dict) for r in batch)


# ============================================================================
#  RequestValidator Tests
# ============================================================================

class TestRequestValidator:
    """Tests for RequestValidator class."""

    def test_validate_valid_request(self, validator):
        """Test validation of valid request."""
        request_json = json.dumps({
            "jsonrpc": "2.0",
            "method": "hololoom.rag.recall",
            "params": {"query": "test"},
            "id": "req-001",
        })
        result = validator.validate_json(request_json)
        assert result.valid
        assert result.request is not None
        assert result.request.method == "hololoom.rag.recall"

    def test_validate_invalid_json(self, validator):
        """Test validation of invalid JSON."""
        result = validator.validate_json("not valid json")
        assert not result.valid
        assert result.error.code == ErrorCode.PARSE_ERROR

    def test_validate_missing_jsonrpc(self, validator):
        """Test validation with missing jsonrpc field."""
        result = validator.validate_json(json.dumps({
            "method": "hololoom.rag.recall",
            "id": "req-001",
        }))
        assert not result.valid
        assert result.error.code == ErrorCode.INVALID_REQUEST

    def test_validate_wrong_jsonrpc_version(self, validator):
        """Test validation with wrong jsonrpc version."""
        result = validator.validate_json(json.dumps({
            "jsonrpc": "1.0",
            "method": "hololoom.rag.recall",
            "id": "req-001",
        }))
        assert not result.valid

    def test_validate_missing_method(self, validator):
        """Test validation with missing method."""
        result = validator.validate_json(json.dumps({
            "jsonrpc": "2.0",
            "id": "req-001",
        }))
        assert not result.valid
        assert result.error.code == ErrorCode.INVALID_REQUEST

    def test_validate_unknown_method(self, validator):
        """Test validation with unknown method prefix."""
        result = validator.validate_json(json.dumps({
            "jsonrpc": "2.0",
            "method": "unknown.method",
            "id": "req-001",
        }))
        assert not result.valid
        assert result.error.code == ErrorCode.METHOD_NOT_FOUND

    def test_validate_invalid_params_type(self, validator):
        """Test validation with non-object params."""
        result = validator.validate_json(json.dumps({
            "jsonrpc": "2.0",
            "method": "hololoom.rag.recall",
            "params": ["array", "not", "object"],
            "id": "req-001",
        }))
        assert not result.valid
        assert result.error.code == ErrorCode.INVALID_PARAMS

    def test_validate_request_too_large(self):
        """Test validation of oversized request."""
        validator = RequestValidator(max_request_size=100)
        large_request = json.dumps({
            "jsonrpc": "2.0",
            "method": "hololoom.rag.recall",
            "params": {"query": "x" * 1000},
            "id": "req-001",
        })
        result = validator.validate_json(large_request)
        assert not result.valid
        assert result.error.code == ErrorCode.INVALID_REQUEST

    def test_validate_with_metadata(self, validator):
        """Test validation with metadata."""
        now = time.time()
        result = validator.validate_json(json.dumps({
            "jsonrpc": "2.0",
            "method": "hololoom.rag.recall",
            "params": {"query": "test"},
            "id": "req-001",
            "meta": {
                "sender_id": "a" * 40,
                "timestamp": now,
                "nonce": "test-nonce-123",
            },
        }))
        assert result.valid
        assert result.request.meta is not None
        assert result.request.meta.sender_id == "a" * 40

    def test_validate_invalid_sender_id(self, validator):
        """Test validation with invalid sender_id."""
        now = time.time()
        result = validator.validate_json(json.dumps({
            "jsonrpc": "2.0",
            "method": "hololoom.rag.recall",
            "id": "req-001",
            "meta": {
                "sender_id": "too-short",
                "timestamp": now,
                "nonce": "test-nonce",
            },
        }))
        assert not result.valid

    def test_validate_expired_timestamp(self):
        """Test validation with expired timestamp."""
        validator = RequestValidator(
            max_timestamp_age=60.0,
            require_signature=False,
        )
        old_time = time.time() - 120  # 2 minutes ago
        result = validator.validate_json(json.dumps({
            "jsonrpc": "2.0",
            "method": "hololoom.rag.recall",
            "id": "req-001",
            "meta": {
                "sender_id": "a" * 40,
                "timestamp": old_time,
                "nonce": "test-nonce",
            },
        }))
        assert not result.valid
        assert result.error.code == ErrorCode.SIGNATURE_EXPIRED

    def test_validate_nonce_reuse(self, validator):
        """Test that nonce reuse is detected."""
        now = time.time()
        request_json = json.dumps({
            "jsonrpc": "2.0",
            "method": "hololoom.rag.recall",
            "id": "req-001",
            "meta": {
                "sender_id": "a" * 40,
                "timestamp": now,
                "nonce": "same-nonce",
            },
        })

        # First request should succeed
        result1 = validator.validate_json(request_json)
        assert result1.valid

        # Second request with same nonce should fail
        result2 = validator.validate_json(request_json)
        assert not result2.valid
        assert result2.error.code == ErrorCode.NONCE_REUSED

    def test_validate_signature_required(self, strict_validator):
        """Test that signature is required when configured."""
        now = time.time()
        result = strict_validator.validate_json(json.dumps({
            "jsonrpc": "2.0",
            "method": "hololoom.rag.recall",
            "id": "req-001",
            "meta": {
                "sender_id": "a" * 40,
                "timestamp": now,
                "nonce": "test-nonce",
                # No signature
            },
        }))
        assert not result.valid
        assert result.error.code == ErrorCode.SIGNATURE_INVALID

    def test_clear_nonce_cache(self, validator):
        """Test nonce cache clearing."""
        now = time.time()

        # Use a nonce
        validator.validate_json(json.dumps({
            "jsonrpc": "2.0",
            "method": "hololoom.rag.recall",
            "id": "req-001",
            "meta": {
                "sender_id": "a" * 40,
                "timestamp": now,
                "nonce": "clear-test-nonce",
            },
        }))

        # Clear cache
        validator.clear_nonce_cache()

        # Same nonce should now work again
        result = validator.validate_json(json.dumps({
            "jsonrpc": "2.0",
            "method": "hololoom.rag.recall",
            "id": "req-002",
            "meta": {
                "sender_id": "a" * 40,
                "timestamp": now,
                "nonce": "clear-test-nonce",
            },
        }))
        assert result.valid


# ============================================================================
#  Batch Request Tests
# ============================================================================

class TestBatchRequests:
    """Tests for batch request handling."""

    def test_validate_batch(self, validator):
        """Test batch request validation."""
        batch_json = json.dumps([
            {
                "jsonrpc": "2.0",
                "method": "hololoom.node.ping",
                "id": "req-001",
            },
            {
                "jsonrpc": "2.0",
                "method": "hololoom.node.info",
                "id": "req-002",
            },
        ])
        results = validator.validate_batch(batch_json)
        assert len(results) == 2
        assert all(r.valid for r in results)

    def test_validate_batch_with_errors(self, validator):
        """Test batch with some invalid requests."""
        batch_json = json.dumps([
            {
                "jsonrpc": "2.0",
                "method": "hololoom.node.ping",
                "id": "req-001",
            },
            {
                "jsonrpc": "2.0",
                "method": "invalid.method",  # Invalid
                "id": "req-002",
            },
        ])
        results = validator.validate_batch(batch_json)
        assert len(results) == 2
        assert results[0].valid
        assert not results[1].valid

    def test_validate_empty_batch(self, validator):
        """Test empty batch."""
        results = validator.validate_batch("[]")
        assert len(results) == 1
        assert not results[0].valid

    def test_validate_non_array_batch(self, validator):
        """Test batch that's not an array."""
        results = validator.validate_batch('{"not": "array"}')
        assert len(results) == 1
        assert not results[0].valid


# ============================================================================
#  Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_create_builder(self):
        """Test create_builder function."""
        builder = create_builder(sender_id="b" * 40)
        request = builder.build_request(method="hololoom.node.ping")
        assert request.meta.sender_id == "b" * 40

    def test_create_validator(self):
        """Test create_validator function."""
        validator = create_validator(require_signature=False)
        result = validator.validate_json(json.dumps({
            "jsonrpc": "2.0",
            "method": "hololoom.node.ping",
            "id": "req-001",
        }))
        assert result.valid

    def test_parse_request(self):
        """Test parse_request function."""
        result = parse_request(
            json.dumps({
                "jsonrpc": "2.0",
                "method": "hololoom.node.ping",
                "id": "req-001",
            }),
            require_signature=False,
        )
        assert result.valid

    def test_parse_batch(self):
        """Test parse_batch function."""
        results = parse_batch(
            json.dumps([
                {"jsonrpc": "2.0", "method": "hololoom.node.ping", "id": "1"},
                {"jsonrpc": "2.0", "method": "hololoom.node.info", "id": "2"},
            ]),
            require_signature=False,
        )
        assert len(results) == 2
        assert all(r.valid for r in results)


# ============================================================================
#  Error Code Tests
# ============================================================================

class TestErrorCodes:
    """Tests for error codes and messages."""

    def test_all_error_codes_have_messages(self):
        """Test that all error codes have corresponding messages."""
        for code in ErrorCode:
            assert code in ERROR_MESSAGES, f"Missing message for {code}"

    def test_standard_json_rpc_codes(self):
        """Test standard JSON-RPC error codes."""
        assert ErrorCode.PARSE_ERROR == -32700
        assert ErrorCode.INVALID_REQUEST == -32600
        assert ErrorCode.METHOD_NOT_FOUND == -32601
        assert ErrorCode.INVALID_PARAMS == -32602
        assert ErrorCode.INTERNAL_ERROR == -32603

    def test_hololoom_error_codes(self):
        """Test HoloLoom-specific error codes."""
        assert ErrorCode.SIGNATURE_INVALID == -32001
        assert ErrorCode.RATE_LIMITED == -32003
        assert ErrorCode.TRUST_INSUFFICIENT == -32004
        assert ErrorCode.SAFETY_BLOCKED == -32005


# ============================================================================
#  Method Registry Tests
# ============================================================================

class TestMethodRegistry:
    """Tests for the method registry."""

    def test_rag_methods_registered(self):
        """Test RAG methods are registered."""
        assert "hololoom.rag.recall" in HOLOLOOM_METHODS
        assert "hololoom.rag.store" in HOLOLOOM_METHODS

    def test_inference_methods_registered(self):
        """Test inference methods are registered."""
        assert "hololoom.inference.generate" in HOLOLOOM_METHODS
        assert "hololoom.inference.capabilities" in HOLOLOOM_METHODS

    def test_node_methods_registered(self):
        """Test node methods are registered."""
        assert "hololoom.node.ping" in HOLOLOOM_METHODS
        assert "hololoom.node.info" in HOLOLOOM_METHODS

    def test_method_has_required_fields(self):
        """Test method definitions have required fields."""
        for method_name, method_def in HOLOLOOM_METHODS.items():
            assert "description" in method_def, f"{method_name} missing description"
            assert "params" in method_def, f"{method_name} missing params"
            assert "required" in method_def, f"{method_name} missing required"


# ============================================================================
#  Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the wire protocol."""

    def test_full_request_response_cycle(self, builder, validator):
        """Test complete request-response cycle."""
        # Build request
        request = builder.build_request(
            method="hololoom.rag.recall",
            params={"query": "What is Thompson Sampling?", "k": 10},
        )

        # Serialize
        request_json = request.to_json()

        # Validate (would normally happen on receiving node)
        result = validator.validate_json(request_json)
        assert result.valid

        # Build response
        response = builder.build_response(
            request_id=result.request.id,
            result={
                "memories": [
                    {"text": "Thompson Sampling balances...", "confidence": 0.92}
                ],
                "total": 1,
            },
        )

        # Serialize and verify
        response_json = response.to_json()
        parsed = json.loads(response_json)
        assert parsed["result"]["total"] == 1

    def test_error_response_cycle(self, builder, validator):
        """Test error response cycle."""
        # Build request
        request = builder.build_request(
            method="hololoom.rag.recall",
            params={"query": "test"},
        )

        # Simulate rate limit error
        response = builder.build_error_response(
            request_id=request.id,
            code=ErrorCode.RATE_LIMITED,
            data={"retry_after": 30, "limit": 60},
        )

        # Verify error response structure
        result = response.to_dict()
        assert result["error"]["code"] == -32003
        assert result["error"]["data"]["retry_after"] == 30

    def test_batch_request_response(self, builder, validator):
        """Test batch request and response handling."""
        # Build batch of requests
        requests = [
            builder.build_request(
                method="hololoom.node.ping",
                request_id="ping-001",
            ),
            builder.build_request(
                method="hololoom.node.capabilities",
                request_id="cap-001",
            ),
        ]

        # Serialize as batch
        batch = builder.build_batch(requests)
        batch_json = json.dumps(batch)

        # Validate batch
        results = validator.validate_batch(batch_json)
        assert len(results) == 2
        assert all(r.valid for r in results)

        # Build batch responses
        responses = [
            builder.build_response(
                request_id="ping-001",
                result={"pong": True},
            ),
            builder.build_response(
                request_id="cap-001",
                result={"models": ["llama3"], "load": 0.3},
            ),
        ]

        # Serialize responses
        response_batch = [r.to_dict() for r in responses]
        response_json = json.dumps(response_batch)

        # Verify structure
        parsed = json.loads(response_json)
        assert len(parsed) == 2
        assert parsed[0]["result"]["pong"] is True
        assert parsed[1]["result"]["load"] == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
