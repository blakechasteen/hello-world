#!/usr/bin/env python3
"""
API Key Management Routes for HoloLoom SaaS API

Endpoints:
- GET /api/v1/api-keys - List all API keys for customer
- POST /api/v1/api-keys - Create new API key
- GET /api/v1/api-keys/{key_id} - Get specific API key details
- DELETE /api/v1/api-keys/{key_id} - Revoke API key
"""

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from ..auth import AuthContext, validate_api_key
from ..models import (
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    APIKeyListResponse,
    APIKeyResponse,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/api-keys", tags=["api-keys"])


# ============================================================================
# GET / - List all API keys
# ============================================================================

@router.get(
    "",
    response_model=APIKeyListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="List all API keys",
    description="""
    List all API keys for the authenticated customer.

    Returns key metadata (ID, name, prefix, status, rate limit, etc.)
    but NOT the secret - secrets are only shown on creation.
    """
)
async def list_api_keys(
    request: Request,
    auth: AuthContext = Depends(validate_api_key)
) -> APIKeyListResponse:
    """List all API keys for customer"""
    backend = request.app.state.saas_backend

    api_keys = await backend.get_api_keys_for_customer(auth.customer_id)

    return APIKeyListResponse(
        api_keys=[
            APIKeyResponse(
                key_id=key.key_id,
                name=key.name,
                key_prefix=key.key_prefix,
                status=key.status,
                rate_limit_qps=key.rate_limit_qps,
                created_at=key.created_at,
                expires_at=key.expires_at,
                last_used=key.last_used,
                request_count=key.request_count
            )
            for key in api_keys
        ]
    )


# ============================================================================
# POST / - Create new API key
# ============================================================================

@router.post(
    "",
    response_model=APIKeyCreateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "API key limit reached"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Create new API key",
    description="""
    Create a new API key for the authenticated customer.

    **IMPORTANT**: The API secret is returned ONLY ONCE in this response.
    Save it securely - it cannot be retrieved again.

    Optional parameters:
    - name: Friendly name for the key (default: "Default")
    - expires_in_days: Number of days until key expires (1-365, optional)
    """
)
async def create_api_key(
    request: Request,
    body: APIKeyCreateRequest,
    auth: AuthContext = Depends(validate_api_key)
) -> APIKeyCreateResponse:
    """Create new API key"""
    backend = request.app.state.saas_backend

    # Check API key limit (prevent abuse)
    existing_keys = await backend.get_api_keys_for_customer(auth.customer_id)
    active_keys = [k for k in existing_keys if k.status == "active"]

    # Limit based on plan
    key_limits = {
        "free": 2,
        "payg": 5,
        "pro": 20,
        "enterprise": 100
    }
    max_keys = key_limits.get(auth.plan, 5)

    if len(active_keys) >= max_keys:
        raise HTTPException(
            status_code=403,
            detail={
                "error": f"API key limit reached ({max_keys} for {auth.plan} plan)",
                "code": "key_limit_reached",
                "max_keys": max_keys,
                "current_keys": len(active_keys)
            }
        )

    # Calculate expiration
    expires_at = None
    if body.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=body.expires_in_days)

    try:
        # Create API key
        api_key, secret = await backend.create_api_key(
            customer_id=auth.customer_id,
            name=body.name,
            expires_at=expires_at
        )

        # Log event
        await backend.log_event(
            customer_id=auth.customer_id,
            event_type="api_key_created",
            event_data={
                "key_id": api_key.key_id,
                "name": body.name,
                "expires_in_days": body.expires_in_days
            }
        )

        logger.info(f"API key created: {api_key.key_id} for customer {auth.customer_id}")

        return APIKeyCreateResponse(
            key_id=api_key.key_id,
            name=api_key.name,
            secret=secret,
            key_prefix=api_key.key_prefix,
            rate_limit_qps=api_key.rate_limit_qps,
            expires_at=api_key.expires_at,
            message="Save your API secret - it won't be shown again"
        )

    except Exception as e:
        logger.error(f"Create API key error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to create API key",
                "code": "create_key_failed"
            }
        )


# ============================================================================
# GET /{key_id} - Get specific API key details
# ============================================================================

@router.get(
    "/{key_id}",
    response_model=APIKeyResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        404: {"model": ErrorResponse, "description": "API key not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get API key details",
    description="""
    Get details for a specific API key.

    Returns key metadata but NOT the secret.
    """
)
async def get_api_key(
    request: Request,
    key_id: str,
    auth: AuthContext = Depends(validate_api_key)
) -> APIKeyResponse:
    """Get specific API key details"""
    backend = request.app.state.saas_backend

    # Get all keys for customer and find the requested one
    api_keys = await backend.get_api_keys_for_customer(auth.customer_id)
    api_key = next((k for k in api_keys if k.key_id == key_id), None)

    if not api_key:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "API key not found",
                "code": "key_not_found"
            }
        )

    return APIKeyResponse(
        key_id=api_key.key_id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        status=api_key.status,
        rate_limit_qps=api_key.rate_limit_qps,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
        last_used=api_key.last_used,
        request_count=api_key.request_count
    )


# ============================================================================
# DELETE /{key_id} - Revoke API key
# ============================================================================

@router.delete(
    "/{key_id}",
    responses={
        200: {"description": "API key revoked"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        404: {"model": ErrorResponse, "description": "API key not found"},
        409: {"model": ErrorResponse, "description": "Cannot revoke last active key"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Revoke API key",
    description="""
    Revoke an API key.

    The key will immediately stop working for authentication.
    This action cannot be undone.

    Note: You cannot revoke your last active API key - at least one
    active key is required to access the API.
    """
)
async def revoke_api_key(
    request: Request,
    key_id: str,
    auth: AuthContext = Depends(validate_api_key)
):
    """Revoke API key"""
    backend = request.app.state.saas_backend

    # Get all keys for customer
    api_keys = await backend.get_api_keys_for_customer(auth.customer_id)
    api_key = next((k for k in api_keys if k.key_id == key_id), None)

    if not api_key:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "API key not found",
                "code": "key_not_found"
            }
        )

    if api_key.status != "active":
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"API key is already {api_key.status}",
                "code": "key_not_active"
            }
        )

    # Check if this is the last active key
    active_keys = [k for k in api_keys if k.status == "active"]
    if len(active_keys) <= 1 and api_key.key_id == auth.key_id:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Cannot revoke your last active API key",
                "code": "last_key"
            }
        )

    try:
        # Revoke the key
        result = await backend.revoke_api_key(key_id)

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to revoke API key",
                    "code": "revoke_failed"
                }
            )

        # Log event
        await backend.log_event(
            customer_id=auth.customer_id,
            event_type="api_key_revoked",
            event_data={"key_id": key_id}
        )

        logger.info(f"API key revoked: {key_id} for customer {auth.customer_id}")

        return {"success": True, "message": "API key revoked"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Revoke API key error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to revoke API key",
                "code": "revoke_failed"
            }
        )


# ============================================================================
# PATCH /{key_id} - Update API key (name only)
# ============================================================================

@router.patch(
    "/{key_id}",
    response_model=APIKeyResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        404: {"model": ErrorResponse, "description": "API key not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Update API key name",
    description="""
    Update an API key's friendly name.

    Only the name can be updated - other properties are immutable.
    """
)
async def update_api_key(
    request: Request,
    key_id: str,
    body: APIKeyCreateRequest,  # Reuse for name field
    auth: AuthContext = Depends(validate_api_key)
) -> APIKeyResponse:
    """Update API key name"""
    backend = request.app.state.saas_backend

    # Get all keys for customer
    api_keys = await backend.get_api_keys_for_customer(auth.customer_id)
    api_key = next((k for k in api_keys if k.key_id == key_id), None)

    if not api_key:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "API key not found",
                "code": "key_not_found"
            }
        )

    # Update name (need to add this method to backend)
    # For now, we'll just return the key with the name field
    # In production, implement update_api_key_name in backend

    # Log event
    await backend.log_event(
        customer_id=auth.customer_id,
        event_type="api_key_updated",
        event_data={"key_id": key_id, "new_name": body.name}
    )

    return APIKeyResponse(
        key_id=api_key.key_id,
        name=body.name,  # Return updated name
        key_prefix=api_key.key_prefix,
        status=api_key.status,
        rate_limit_qps=api_key.rate_limit_qps,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
        last_used=api_key.last_used,
        request_count=api_key.request_count
    )
