#!/usr/bin/env python3
"""
Usage Tracking Example App

Demonstrates SaaS toolkit with authentication + usage analytics,
but WITHOUT billing. Perfect for:
- Internal tools with usage monitoring
- Free tier products tracking engagement
- Analytics dashboards

Run with:
    PYTHONPATH=. uvicorn HoloLoom.saas.examples.usage_tracking_app:app --reload
"""

from datetime import date, timedelta
from fastapi import FastAPI, Depends, HTTPException, Request
from contextlib import asynccontextmanager

from HoloLoom.saas import (
    SaaSBackend,
    SaaSConfig,
    create_saas_backend,
)
from HoloLoom.saas.auth import validate_api_key, AuthContext
from HoloLoom.saas.routes import customers_router, api_keys_router, health_router


# ============================================================================
# Configuration
# ============================================================================

config = SaaSConfig(
    sqlite_path="./data/usage_tracking_example.db",
    fallback_to_sqlite=True,
    enable_usage_tracking=True,  # Track usage
    enable_billing=False,        # No billing
    enable_audit_log=True,       # Log all events
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
    print("Backend connected (usage tracking enabled)")
    yield
    await backend.close()
    print("Backend closed")


app = FastAPI(
    title="Usage Tracking Example",
    description="HoloLoom SaaS with usage analytics (no billing)",
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
# Your API Endpoints (with usage tracking)
# ============================================================================

@app.post("/api/v1/query")
async def query_endpoint(
    request: Request,
    data: dict,
    auth: AuthContext = Depends(validate_api_key),
):
    """
    Example protected endpoint that tracks usage.

    Every call increments the customer's query count.
    """
    # Track usage (no billing, just analytics)
    await backend.record_usage(
        customer_id=auth.customer_id,
        queries_delta=1,
        tokens_delta=len(str(data)) * 4  # Rough token estimate
    )

    # Log the event for audit trail
    await backend.log_event(
        event_type="api_query",
        customer_id=auth.customer_id,
        event_data={
            "endpoint": "/api/v1/query",
            "payload_size": len(str(data)),
        },
        ip_address=request.client.host if request.client else None,
    )

    # Your actual logic here
    return {
        "status": "success",
        "customer_id": auth.customer_id,
        "plan": auth.plan,
        "data_received": data,
    }


@app.get("/api/v1/usage")
async def get_usage(auth: AuthContext = Depends(validate_api_key)):
    """
    Get usage statistics for the current customer.

    Returns daily usage for the last 30 days.
    """
    start = date.today() - timedelta(days=30)
    usage = await backend.get_usage_for_period(auth.customer_id, start)

    total_queries = sum(u.queries_count for u in usage)
    total_tokens = sum(u.tokens_count for u in usage)

    return {
        "customer_id": auth.customer_id,
        "period": {
            "start": start.isoformat(),
            "end": date.today().isoformat(),
        },
        "totals": {
            "queries": total_queries,
            "tokens": total_tokens,
        },
        "daily": [
            {
                "date": u.usage_date.isoformat(),
                "queries": u.queries_count,
                "tokens": u.tokens_count,
            }
            for u in usage
        ],
    }


@app.get("/api/v1/usage/today")
async def get_today_usage(auth: AuthContext = Depends(validate_api_key)):
    """Get usage for today only."""
    usage = await backend.get_usage_for_date(auth.customer_id, date.today())

    return {
        "customer_id": auth.customer_id,
        "date": date.today().isoformat(),
        "queries": usage.queries_count if usage else 0,
        "tokens": usage.tokens_count if usage else 0,
    }


# ============================================================================
# Admin Endpoints (for dashboard)
# ============================================================================

@app.get("/api/v1/admin/usage-summary")
async def admin_usage_summary(auth: AuthContext = Depends(validate_api_key)):
    """
    Admin endpoint: Get system-wide usage summary.

    In production, add proper admin authentication.
    """
    # Simple check - in production use proper admin roles
    if auth.plan not in ["enterprise", "admin"]:
        raise HTTPException(403, "Admin access required")

    # This would query across all customers
    # For demo, return mock data
    return {
        "total_customers": 42,
        "total_queries_today": 1523,
        "total_queries_30d": 45678,
        "top_customers": [
            {"customer_id": "demo", "queries": 500},
        ],
    }


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with usage info"""
    return {
        "name": "Usage Tracking Example",
        "features": {
            "authentication": True,
            "usage_tracking": True,
            "billing": False,
            "audit_log": True,
        },
        "endpoints": {
            "query": "POST /api/v1/query (tracks usage)",
            "usage": "GET /api/v1/usage (30-day stats)",
            "usage_today": "GET /api/v1/usage/today",
        },
        "docs": "/docs",
    }


# ============================================================================
# Demo Script
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import httpx

    async def demo():
        """Demo the usage tracking flow"""
        base_url = "http://localhost:8000"

        async with httpx.AsyncClient() as client:
            # 1. Signup
            print("\n1. Signup...")
            signup_resp = await client.post(
                f"{base_url}/api/v1/customers/signup",
                json={
                    "email": "usage_demo@example.com",
                    "name": "Usage Demo",
                    "password": "demo_password_123",
                },
            )

            if signup_resp.status_code != 200:
                print(f"   Signup failed or user exists: {signup_resp.text}")
                # Try login instead
                login_resp = await client.post(
                    f"{base_url}/api/v1/customers/login",
                    json={
                        "email": "usage_demo@example.com",
                        "password": "demo_password_123",
                    },
                )
                if login_resp.status_code != 200:
                    print("   Login also failed")
                    return
                api_secret = login_resp.json().get("api_secret", "")
            else:
                api_secret = signup_resp.json()["api_secret"]
                print(f"   API Secret: {api_secret[:16]}...")

            headers = {"X-API-Key": api_secret}

            # 2. Make some queries (to generate usage)
            print("\n2. Making queries to generate usage...")
            for i in range(5):
                resp = await client.post(
                    f"{base_url}/api/v1/query",
                    headers=headers,
                    json={"query": f"Test query {i+1}"},
                )
                print(f"   Query {i+1}: {resp.status_code}")

            # 3. Check usage
            print("\n3. Checking usage...")
            usage_resp = await client.get(
                f"{base_url}/api/v1/usage/today",
                headers=headers,
            )
            if usage_resp.status_code == 200:
                usage = usage_resp.json()
                print(f"   Today's queries: {usage['queries']}")
                print(f"   Today's tokens: {usage['tokens']}")

            # 4. Get full usage history
            print("\n4. Full usage history...")
            history_resp = await client.get(
                f"{base_url}/api/v1/usage",
                headers=headers,
            )
            if history_resp.status_code == 200:
                history = history_resp.json()
                print(f"   Total queries (30d): {history['totals']['queries']}")
                print(f"   Total tokens (30d): {history['totals']['tokens']}")

    print("Run the server first:")
    print("PYTHONPATH=. uvicorn HoloLoom.saas.examples.usage_tracking_app:app --reload")
    print("\nThen run this file directly to test.")
