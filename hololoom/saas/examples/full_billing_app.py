#!/usr/bin/env python3
"""
Full Billing Example App

Demonstrates complete SaaS toolkit with Stripe integration.
This is the full production setup for monetized apps.

Requirements:
    pip install stripe

Environment Variables:
    STRIPE_API_KEY - Your Stripe secret key (sk_...)
    STRIPE_WEBHOOK_SECRET - Webhook signing secret (whsec_...)
    DATABASE_URL - PostgreSQL connection string (optional)

Run with:
    STRIPE_API_KEY=sk_test_... PYTHONPATH=. uvicorn HoloLoom.saas.examples.full_billing_app:app --reload
"""

import os
from datetime import date, timedelta
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel

from HoloLoom.saas import (
    SaaSBackend,
    SaaSConfig,
    create_saas_backend,
    PLAN_CONFIGS,
)
from HoloLoom.saas.auth import validate_api_key, AuthContext
from HoloLoom.saas.routes import customers_router, api_keys_router, health_router


# ============================================================================
# Configuration
# ============================================================================

STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
DATABASE_URL = os.getenv("DATABASE_URL")

# Validate Stripe is configured
if not STRIPE_API_KEY:
    print("WARNING: STRIPE_API_KEY not set. Billing features disabled.")
    print("Set STRIPE_API_KEY=sk_test_... for full functionality.")

# Try to import Stripe
try:
    import stripe
    stripe.api_key = STRIPE_API_KEY
    STRIPE_AVAILABLE = bool(STRIPE_API_KEY)
except ImportError:
    STRIPE_AVAILABLE = False
    print("WARNING: stripe package not installed. Run: pip install stripe")


# Create config based on environment
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    import urllib.parse
    parsed = urllib.parse.urlparse(DATABASE_URL)
    config = SaaSConfig(
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        database=parsed.path.lstrip("/") or "hololoom_billing",
        user=parsed.username or "hololoom",
        password=parsed.password or "",
        enable_billing=STRIPE_AVAILABLE,
        enable_usage_tracking=True,
        enable_audit_log=True,
        stripe_api_key=STRIPE_API_KEY,
        stripe_webhook_secret=STRIPE_WEBHOOK_SECRET,
        fallback_to_sqlite=True,
    )
else:
    config = SaaSConfig(
        sqlite_path="./data/full_billing_example.db",
        fallback_to_sqlite=True,
        enable_billing=STRIPE_AVAILABLE,
        enable_usage_tracking=True,
        enable_audit_log=True,
        stripe_api_key=STRIPE_API_KEY,
        stripe_webhook_secret=STRIPE_WEBHOOK_SECRET,
    )

backend = create_saas_backend(config)


# ============================================================================
# App Setup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage backend lifecycle"""
    await backend.connect()
    app.state.saas_backend = backend
    print(f"Backend connected (billing={'enabled' if STRIPE_AVAILABLE else 'disabled'})")
    yield
    await backend.close()
    print("Backend closed")


app = FastAPI(
    title="Full Billing Example",
    description="HoloLoom SaaS with complete Stripe billing",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Mount SaaS Routes
# ============================================================================

app.include_router(customers_router, tags=["customers"])
app.include_router(api_keys_router, tags=["api-keys"])
app.include_router(health_router, tags=["health"])


# ============================================================================
# Billing Endpoints
# ============================================================================

class UpgradeRequest(BaseModel):
    plan: str  # "payg", "pro", "enterprise"


@app.get("/api/v1/billing/plans")
async def list_plans():
    """List available subscription plans."""
    return {
        "plans": [
            {
                "id": plan_id,
                "name": plan_id.title(),
                "rate_limit_qps": config.rate_limit_qps,
                "queries_included": config.queries_included,
                "price_per_query_cents": config.price_per_query_cents,
            }
            for plan_id, config in PLAN_CONFIGS.items()
        ]
    }


@app.get("/api/v1/billing/subscription")
async def get_subscription(auth: AuthContext = Depends(validate_api_key)):
    """Get current subscription status."""
    subscription = await backend.get_subscription(auth.customer_id)

    if not subscription:
        return {
            "status": "no_subscription",
            "plan": "free",
        }

    return {
        "subscription_id": subscription.subscription_id,
        "plan": subscription.plan,
        "status": subscription.status,
        "current_period_start": subscription.current_period_start.isoformat() if subscription.current_period_start else None,
        "current_period_end": subscription.current_period_end.isoformat() if subscription.current_period_end else None,
    }


@app.post("/api/v1/billing/upgrade")
async def upgrade_plan(
    request: UpgradeRequest,
    auth: AuthContext = Depends(validate_api_key),
):
    """
    Upgrade subscription plan.

    In production, this would:
    1. Create/update Stripe subscription
    2. Handle payment method if needed
    3. Update local subscription record
    """
    if request.plan not in PLAN_CONFIGS:
        raise HTTPException(400, f"Invalid plan: {request.plan}")

    if not STRIPE_AVAILABLE:
        # Demo mode - just update locally
        subscription = await backend.get_subscription(auth.customer_id)
        if subscription:
            await backend.update_subscription(
                subscription.subscription_id,
                plan=request.plan,
                status="active",
            )
        return {
            "status": "upgraded",
            "plan": request.plan,
            "note": "Demo mode - no Stripe charge",
        }

    # Production: Create Stripe checkout session
    # customer = await backend.get_customer(auth.customer_id)
    # checkout_session = stripe.checkout.Session.create(...)
    # return {"checkout_url": checkout_session.url}

    return {
        "status": "upgraded",
        "plan": request.plan,
    }


@app.get("/api/v1/billing/usage")
async def get_billing_usage(auth: AuthContext = Depends(validate_api_key)):
    """
    Get usage and estimated billing for current period.
    """
    # Get subscription
    subscription = await backend.get_subscription(auth.customer_id)
    plan_config = PLAN_CONFIGS.get(subscription.plan if subscription else "free")

    # Get usage for current period
    start = date.today().replace(day=1)  # Start of month
    usage = await backend.get_usage_for_period(auth.customer_id, start)
    total_queries = sum(u.queries_count for u in usage)

    # Calculate overage
    included = plan_config.queries_included
    overage = max(0, total_queries - included)
    overage_cost = overage * plan_config.price_per_query_cents / 100

    return {
        "period": {
            "start": start.isoformat(),
            "end": date.today().isoformat(),
        },
        "plan": subscription.plan if subscription else "free",
        "usage": {
            "queries": total_queries,
            "queries_included": included,
            "overage_queries": overage,
        },
        "estimated_cost": {
            "overage_usd": round(overage_cost, 2),
            "price_per_query_cents": plan_config.price_per_query_cents,
        },
    }


# ============================================================================
# Stripe Webhook Handler
# ============================================================================

@app.post("/webhooks/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="Stripe-Signature"),
):
    """
    Handle Stripe webhook events.

    Common events to handle:
    - checkout.session.completed
    - customer.subscription.updated
    - customer.subscription.deleted
    - invoice.paid
    - invoice.payment_failed
    """
    if not STRIPE_AVAILABLE:
        return JSONResponse({"status": "billing_disabled"}, status_code=200)

    payload = await request.body()

    # Verify webhook signature
    if STRIPE_WEBHOOK_SECRET and stripe_signature:
        try:
            event = stripe.Webhook.construct_event(
                payload, stripe_signature, STRIPE_WEBHOOK_SECRET
            )
        except stripe.error.SignatureVerificationError:
            raise HTTPException(400, "Invalid signature")
    else:
        # No signature verification (not recommended for production)
        import json
        event = json.loads(payload)

    event_type = event.get("type", "")
    data = event.get("data", {}).get("object", {})

    # Handle events
    if event_type == "checkout.session.completed":
        # Customer completed checkout - activate subscription
        customer_email = data.get("customer_email")
        subscription_id = data.get("subscription")
        # await backend.activate_subscription(customer_email, subscription_id)
        print(f"Checkout completed: {customer_email}")

    elif event_type == "customer.subscription.updated":
        # Subscription changed (upgrade/downgrade)
        subscription_id = data.get("id")
        status = data.get("status")
        # await backend.update_subscription_status(subscription_id, status)
        print(f"Subscription updated: {subscription_id} -> {status}")

    elif event_type == "customer.subscription.deleted":
        # Subscription cancelled
        subscription_id = data.get("id")
        # await backend.cancel_subscription(subscription_id)
        print(f"Subscription cancelled: {subscription_id}")

    elif event_type == "invoice.payment_failed":
        # Payment failed - may need to notify customer
        customer_email = data.get("customer_email")
        print(f"Payment failed for: {customer_email}")

    # Log webhook event
    await backend.log_event(
        event_type=f"stripe_webhook_{event_type}",
        customer_id=None,
        event_data={"stripe_event_id": event.get("id")},
    )

    return {"status": "received"}


# ============================================================================
# Protected API Endpoints
# ============================================================================

@app.post("/api/v1/query")
async def query_endpoint(
    request: Request,
    data: dict,
    auth: AuthContext = Depends(validate_api_key),
):
    """
    Protected endpoint with usage tracking and billing.

    Tracks queries for billing purposes.
    """
    # Check if customer has exceeded limits (for free tier)
    if auth.plan == "free":
        today_usage = await backend.get_usage_for_date(auth.customer_id, date.today())
        if today_usage and today_usage.queries_count >= 100:  # Free tier: 100/day
            raise HTTPException(
                429,
                {
                    "error": "daily_limit_exceeded",
                    "message": "Free tier limit (100 queries/day) exceeded",
                    "upgrade_url": "/api/v1/billing/plans",
                }
            )

    # Track usage
    await backend.record_usage(
        customer_id=auth.customer_id,
        queries_delta=1,
        tokens_delta=len(str(data)) * 4,
    )

    # Your API logic here
    return {
        "status": "success",
        "customer_id": auth.customer_id,
        "plan": auth.plan,
        "data": data,
    }


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with app info"""
    return {
        "name": "Full Billing Example",
        "features": {
            "authentication": True,
            "usage_tracking": True,
            "billing": STRIPE_AVAILABLE,
            "audit_log": True,
        },
        "billing_endpoints": {
            "plans": "GET /api/v1/billing/plans",
            "subscription": "GET /api/v1/billing/subscription",
            "upgrade": "POST /api/v1/billing/upgrade",
            "usage": "GET /api/v1/billing/usage",
            "webhook": "POST /webhooks/stripe",
        },
        "docs": "/docs",
        "stripe_configured": STRIPE_AVAILABLE,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("""
Full Billing Example App

To run with Stripe:
    export STRIPE_API_KEY=sk_test_...
    export STRIPE_WEBHOOK_SECRET=whsec_...
    PYTHONPATH=. uvicorn HoloLoom.saas.examples.full_billing_app:app --reload

To run without Stripe (demo mode):
    PYTHONPATH=. uvicorn HoloLoom.saas.examples.full_billing_app:app --reload

For Stripe webhooks in development, use Stripe CLI:
    stripe listen --forward-to localhost:8000/webhooks/stripe
""")
