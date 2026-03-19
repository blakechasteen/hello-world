#!/usr/bin/env python3
"""
Customer Routes for HoloLoom SaaS API

Endpoints:
- POST /api/v1/customers/signup - Create new customer account
- POST /api/v1/customers/login - Authenticate and get JWT token
- GET /api/v1/customers/me - Get current customer profile
- PATCH /api/v1/customers/me - Update customer profile
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from ..auth import AuthContext, validate_api_key
from ..models import (
    CustomerProfile,
    CustomerUpdateRequest,
    ErrorResponse,
    LoginRequest,
    LoginResponse,
    SignupRequest,
    SignupResponse,
    get_plan_config,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/customers", tags=["customers"])


# ============================================================================
# POST /signup - Create new customer account
# ============================================================================

@router.post(
    "/signup",
    response_model=SignupResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        409: {"model": ErrorResponse, "description": "Email already exists"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Create new customer account",
    description="""
    Create a new customer account with email and password.

    Returns:
    - customer_id: Unique customer identifier
    - api_key_id: ID of the auto-generated API key
    - api_secret: **IMPORTANT** - The API secret is shown ONLY ONCE on signup.
      Save it securely - it cannot be retrieved again.

    The customer starts on the free plan with:
    - 100 queries/day limit
    - 1 QPS rate limit
    - No payment method required
    """
)
async def signup(
    request: Request,
    body: SignupRequest
) -> SignupResponse:
    """Create new customer account"""
    backend = request.app.state.saas_backend

    # Check if email already exists
    existing = await backend.get_customer_by_email(body.email)
    if existing:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Email already registered",
                "code": "email_exists"
            }
        )

    try:
        # Create customer
        customer, _ = await backend.insert_customer(
            email=body.email,
            name=body.name,
            password=body.password,
            company=body.company
        )

        # Create free subscription
        subscription = await backend.insert_subscription(
            customer_id=customer.customer_id,
            plan="free"
        )

        # Create initial API key
        api_key, api_secret = await backend.create_api_key(
            customer_id=customer.customer_id,
            name="Default"
        )

        # Log signup event
        await backend.log_event(
            customer_id=customer.customer_id,
            event_type="signup",
            event_data={
                "email": body.email,
                "plan": "free"
            }
        )

        logger.info(f"New customer signup: {customer.customer_id} ({body.email})")

        return SignupResponse(
            customer_id=customer.customer_id,
            email=customer.email,
            name=customer.name,
            plan=subscription.plan,
            api_key_id=api_key.key_id,
            api_secret=api_secret,
            message="Save your API secret - it won't be shown again"
        )

    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to create account",
                "code": "signup_failed"
            }
        )


# ============================================================================
# POST /login - Authenticate customer
# ============================================================================

@router.post(
    "/login",
    response_model=LoginResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Authenticate customer",
    description="""
    Authenticate with email and password.

    Returns a JWT access token for session-based authentication.
    This token can be used for dashboard/web UI access.

    For API access, use API keys instead (X-API-Key header).
    """
)
async def login(
    request: Request,
    body: LoginRequest
) -> LoginResponse:
    """Authenticate customer and return JWT token"""
    backend = request.app.state.saas_backend

    # Authenticate
    customer = await backend.authenticate_customer(body.email, body.password)

    if not customer:
        # Log failed attempt
        await backend.log_event(
            customer_id=None,
            event_type="login_failed",
            event_data={"email": body.email}
        )
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Invalid email or password",
                "code": "invalid_credentials"
            }
        )

    # Check account status
    if customer.status != "active":
        raise HTTPException(
            status_code=401,
            detail={
                "error": f"Account {customer.status}",
                "code": f"account_{customer.status}"
            }
        )

    # Get subscription
    subscription = await backend.get_subscription(customer.customer_id)
    plan = subscription.plan if subscription else "free"

    # Generate JWT token
    # For production, use proper JWT library (PyJWT)
    # This is a simplified placeholder
    import hashlib
    import time
    token_data = f"{customer.customer_id}:{int(time.time())}"
    access_token = hashlib.sha256(token_data.encode()).hexdigest()

    # Log successful login
    await backend.log_event(
        customer_id=customer.customer_id,
        event_type="login",
        event_data={"email": body.email}
    )

    logger.info(f"Customer login: {customer.customer_id}")

    return LoginResponse(
        customer_id=customer.customer_id,
        email=customer.email,
        name=customer.name,
        plan=plan,
        access_token=access_token
    )


# ============================================================================
# GET /me - Get current customer profile
# ============================================================================

@router.get(
    "/me",
    response_model=CustomerProfile,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get current customer profile",
    description="""
    Get the authenticated customer's profile information.

    Requires API key authentication via X-API-Key header.
    """
)
async def get_profile(
    request: Request,
    auth: AuthContext = Depends(validate_api_key)
) -> CustomerProfile:
    """Get current customer profile"""
    backend = request.app.state.saas_backend

    # Get customer details
    customer = await backend.get_customer(auth.customer_id)
    if not customer:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Customer not found",
                "code": "customer_not_found"
            }
        )

    # Get subscription
    subscription = await backend.get_subscription(auth.customer_id)

    # Get current month usage
    today = datetime.now(timezone.utc).date()
    first_of_month = today.replace(day=1)

    # Calculate usage for current period
    usage_records = await backend.get_usage_for_period(
        customer_id=auth.customer_id,
        start_date=first_of_month,
        end_date=today
    )
    usage_this_month = sum(r.queries_count for r in usage_records)

    # Get plan config
    plan_config = get_plan_config(auth.plan)

    return CustomerProfile(
        customer_id=customer.customer_id,
        email=customer.email,
        name=customer.name,
        company=customer.company,
        plan=auth.plan,
        status=customer.status,
        created_at=customer.created_at,
        usage_this_month=usage_this_month,
        queries_included=plan_config.queries_included
    )


# ============================================================================
# PATCH /me - Update customer profile
# ============================================================================

@router.patch(
    "/me",
    response_model=CustomerProfile,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Update customer profile",
    description="""
    Update the authenticated customer's profile information.

    Only the following fields can be updated:
    - name
    - company

    Requires API key authentication via X-API-Key header.
    """
)
async def update_profile(
    request: Request,
    body: CustomerUpdateRequest,
    auth: AuthContext = Depends(validate_api_key)
) -> CustomerProfile:
    """Update current customer profile"""
    backend = request.app.state.saas_backend

    # Build update dict
    updates = {}
    if body.name is not None:
        updates["name"] = body.name.strip()
    if body.company is not None:
        updates["company"] = body.company.strip() if body.company else None

    if not updates:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "No fields to update",
                "code": "no_updates"
            }
        )

    # Update customer
    result = await backend.update_customer(auth.customer_id, **updates)

    if not result.success:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to update profile",
                "code": "update_failed"
            }
        )

    # Log update event
    await backend.log_event(
        customer_id=auth.customer_id,
        event_type="profile_updated",
        event_data=updates
    )

    # Return updated profile
    return await get_profile(request, auth)


# ============================================================================
# DELETE /me - Delete customer account (soft delete)
# ============================================================================

@router.delete(
    "/me",
    responses={
        200: {"description": "Account deleted"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Delete customer account",
    description="""
    Delete the authenticated customer's account.

    This performs a soft delete:
    - Account status is set to 'deleted'
    - All API keys are revoked
    - Subscription is cancelled

    Data is retained for 30 days before permanent deletion.
    """
)
async def delete_account(
    request: Request,
    auth: AuthContext = Depends(validate_api_key)
):
    """Soft delete customer account"""
    backend = request.app.state.saas_backend

    try:
        # Update status to deleted
        await backend.update_customer(auth.customer_id, status="deleted")

        # Revoke all API keys
        api_keys = await backend.get_api_keys_for_customer(auth.customer_id)
        for key in api_keys:
            await backend.revoke_api_key(key.key_id)

        # Update subscription status
        await backend.update_subscription(
            auth.customer_id,
            status="canceled",
            cancel_at_period_end=True
        )

        # Log deletion
        await backend.log_event(
            customer_id=auth.customer_id,
            event_type="account_deleted",
            event_data={"reason": "user_requested"}
        )

        logger.info(f"Account deleted: {auth.customer_id}")

        return {"success": True, "message": "Account deleted. Data will be purged in 30 days."}

    except Exception as e:
        logger.error(f"Delete account error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to delete account",
                "code": "delete_failed"
            }
        )
