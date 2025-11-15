#!/usr/bin/env python3
"""
RBAC System Demo

Comprehensive demonstration of HoloLoom's Role-Based Access Control system:
- Role hierarchy (admin > write > read > guest)
- Permission checking
- Policy engine with complex rules
- FastAPI endpoint protection
- Resource-based access control

Run:
    python demos/demo_rbac.py

Then test endpoints:
    # Health check (guest access)
    curl http://localhost:8080/health

    # Query endpoint (requires read role)
    curl -H "Authorization: Bearer user_read_123" http://localhost:8080/query?q=test

    # Ingest endpoint (requires write role)
    curl -X POST -H "Authorization: Bearer user_write_456" \
         -H "Content-Type: application/json" \
         -d '{"text": "test data"}' \
         http://localhost:8080/ingest

    # Admin endpoint (requires admin role)
    curl -H "Authorization: Bearer user_admin_789" http://localhost:8080/admin/users

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime, time

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# RBAC imports
from HoloLoom.security.rbac import (
    Role,
    Permission,
    RBACManager,
    create_rbac_manager,
    PolicyEngine,
    PolicyContext,
    PolicyDecision,
    CommonPolicies,
    create_policy_engine,
)

from HoloLoom.security.rbac.decorators import (
    set_rbac_manager,
    require_role,
    require_permission,
    require_any_role,
    get_current_user_id,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="HoloLoom RBAC Demo",
    description="Demonstration of Role-Based Access Control",
    version="1.0.0"
)


# Request/Response models
class QueryRequest(BaseModel):
    text: str


class IngestRequest(BaseModel):
    text: str


# ============================================================================
# RBAC Setup
# ============================================================================

# Global instances
rbac_manager: Optional[RBACManager] = None
policy_engine: Optional[PolicyEngine] = None


async def setup_rbac():
    """
    Initialize RBAC system.

    Creates:
    - RBAC manager with in-memory storage
    - Policy engine with common policies
    - Test users with different roles
    """
    global rbac_manager, policy_engine

    logger.info("=" * 70)
    logger.info("Setting up RBAC system...")
    logger.info("=" * 70)

    # Create RBAC manager
    rbac_manager = create_rbac_manager(storage_type="memory", enable_caching=True)
    set_rbac_manager(rbac_manager)

    # Create policy engine
    policy_engine = create_policy_engine(rbac=rbac_manager, add_common_policies=True)

    # Add business hours policy (9 AM - 5 PM)
    policy_engine.add_rule(CommonPolicies.business_hours_only(
        start=time(9, 0),
        end=time(17, 0)
    ))

    # Add IP whitelist (for demo, allow all)
    # In production, restrict to trusted IPs
    # policy_engine.add_rule(CommonPolicies.ip_whitelist(["192.168.1.1"]))

    # Create test users
    logger.info("\nCreating test users...")
    await rbac_manager.assign_role("user_admin_789", Role.ADMIN, assigned_by="system")
    await rbac_manager.assign_role("user_write_456", Role.WRITE, assigned_by="system")
    await rbac_manager.assign_role("user_read_123", Role.READ, assigned_by="system")
    await rbac_manager.assign_role("user_guest_000", Role.GUEST, assigned_by="system")

    logger.info("✅ Test users created:")
    logger.info("   - user_admin_789 (ADMIN)")
    logger.info("   - user_write_456 (WRITE)")
    logger.info("   - user_read_123 (READ)")
    logger.info("   - user_guest_000 (GUEST)")

    logger.info("\n✅ RBAC system ready!\n")


@app.on_event("startup")
async def startup_event():
    """Initialize RBAC on app startup."""
    await setup_rbac()


# ============================================================================
# Public Endpoints (No Auth Required)
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "name": "HoloLoom RBAC Demo",
        "version": "1.0.0",
        "docs": "/docs",
        "test_users": {
            "admin": "Bearer user_admin_789",
            "write": "Bearer user_write_456",
            "read": "Bearer user_read_123",
            "guest": "Bearer user_guest_000"
        },
        "endpoints": {
            "health": "GET /health (guest)",
            "query": "GET /query (read)",
            "ingest": "POST /ingest (write)",
            "admin_users": "GET /admin/users (admin)",
            "admin_stats": "GET /admin/stats (admin)"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint (accessible to all, including guest).

    No authentication required.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Role-Protected Endpoints
# ============================================================================

@app.get("/query")
async def query_endpoint(
    q: str,
    user_id: str = require_role(Role.READ)
):
    """
    Query endpoint (requires READ role or higher).

    Accessible to: READ, WRITE, ADMIN
    """
    logger.info(f"✅ Query authorized for user {user_id}")

    return {
        "user_id": user_id,
        "query": q,
        "result": f"Results for: {q}",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/ingest")
async def ingest_endpoint(
    request: IngestRequest,
    user_id: str = require_role(Role.WRITE)
):
    """
    Data ingestion endpoint (requires WRITE role or higher).

    Accessible to: WRITE, ADMIN
    """
    logger.info(f"✅ Ingest authorized for user {user_id}")

    return {
        "user_id": user_id,
        "ingested": request.text,
        "status": "success",
        "timestamp": datetime.now().isoformat()
    }


@app.delete("/data/{data_id}")
async def delete_endpoint(
    data_id: str,
    user_id: str = require_role(Role.ADMIN)
):
    """
    Delete endpoint (requires ADMIN role).

    Accessible to: ADMIN only
    """
    logger.info(f"✅ Delete authorized for user {user_id}")

    return {
        "user_id": user_id,
        "deleted": data_id,
        "status": "success",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Permission-Protected Endpoints
# ============================================================================

@app.get("/stats")
async def stats_endpoint(
    user_id: str = require_permission(Permission.SYSTEM_METRICS)
):
    """
    Statistics endpoint (requires SYSTEM_METRICS permission).

    Accessible to: READ, WRITE, ADMIN (anyone with metrics permission)
    """
    logger.info(f"✅ Stats access authorized for user {user_id}")

    return {
        "user_id": user_id,
        "rbac_stats": rbac_manager.get_statistics(),
        "policy_stats": policy_engine.get_statistics(),
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Admin Endpoints
# ============================================================================

@app.get("/admin/users")
async def list_users(
    user_id: str = require_role(Role.ADMIN)
):
    """
    List all users (admin only).

    Accessible to: ADMIN only
    """
    logger.info(f"✅ Admin access authorized for user {user_id}")

    # Get all roles
    users = {}
    for role in Role:
        role_users = await rbac_manager.list_users_with_role(role)
        users[role.value] = [
            {
                "user_id": u.user_id,
                "assigned_at": u.assigned_at.isoformat(),
                "assigned_by": u.assigned_by
            }
            for u in role_users
        ]

    return {
        "admin_user_id": user_id,
        "users": users,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/admin/assign-role")
async def assign_role_endpoint(
    user_id: str,
    role: Role,
    admin_id: str = require_role(Role.ADMIN)
):
    """
    Assign role to user (admin only).

    Accessible to: ADMIN only
    """
    logger.info(f"✅ Role assignment authorized by admin {admin_id}")

    user_role = await rbac_manager.assign_role(
        user_id=user_id,
        role=role,
        assigned_by=admin_id
    )

    return {
        "admin_id": admin_id,
        "assigned": {
            "user_id": user_role.user_id,
            "role": user_role.role.value,
            "assigned_at": user_role.assigned_at.isoformat()
        },
        "timestamp": datetime.now().isoformat()
    }


@app.delete("/admin/revoke-role/{user_id}")
async def revoke_role_endpoint(
    user_id: str,
    admin_id: str = require_role(Role.ADMIN)
):
    """
    Revoke user's role (admin only).

    Accessible to: ADMIN only
    """
    logger.info(f"✅ Role revocation authorized by admin {admin_id}")

    success = await rbac_manager.revoke_role(user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No role found for user {user_id}"
        )

    return {
        "admin_id": admin_id,
        "revoked": user_id,
        "status": "success",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Policy Engine Endpoint
# ============================================================================

@app.post("/policy/evaluate")
async def evaluate_policy(
    action: str,
    resource_owner_id: Optional[str] = None,
    user_id: str = Depends(get_current_user_id)
):
    """
    Evaluate policy for action.

    Shows how policy engine evaluates complex rules.
    """
    context = PolicyContext(
        user_id=user_id,
        resource_owner_id=resource_owner_id,
        action=action,
        timestamp=datetime.now()
    )

    result = await policy_engine.evaluate(context)

    return {
        "user_id": user_id,
        "action": action,
        "decision": result.decision.value,
        "reason": result.reason,
        "matched_rules": result.matched_rules,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Demo CLI
# ============================================================================

async def run_demo():
    """
    Run interactive demo showing RBAC features.
    """
    print("=" * 70)
    print("HoloLoom RBAC Demo")
    print("=" * 70)

    # Setup
    await setup_rbac()

    print("\n" + "=" * 70)
    print("Role Hierarchy Demo")
    print("=" * 70)

    # Test role hierarchy
    roles = [Role.GUEST, Role.READ, Role.WRITE, Role.ADMIN]
    for role in roles:
        permissions = await rbac_manager.get_user_permissions(f"test_{role.value}")
        # Temporarily assign role for demo
        await rbac_manager.assign_role(f"test_{role.value}", role)
        permissions = await rbac_manager.get_user_permissions(f"test_{role.value}")

        print(f"\n{role.value.upper()} role:")
        print(f"  Total permissions: {len(permissions)}")
        print(f"  Sample permissions:")
        for perm in list(permissions)[:5]:
            print(f"    - {perm.value}")

    print("\n" + "=" * 70)
    print("Permission Checking Demo")
    print("=" * 70)

    # Test permission checking
    test_cases = [
        ("user_read_123", Permission.QUERY_READ, True),
        ("user_read_123", Permission.QUERY_WRITE, False),
        ("user_write_456", Permission.QUERY_WRITE, True),
        ("user_admin_789", Permission.SYSTEM_CONFIG, True),
    ]

    for user_id, permission, expected in test_cases:
        has_perm = await rbac_manager.check_permission(user_id, permission)
        status = "✅" if has_perm == expected else "❌"
        print(f"\n{status} {user_id} has {permission.value}: {has_perm}")

    print("\n" + "=" * 70)
    print("Policy Engine Demo")
    print("=" * 70)

    # Test policy engine
    policy_tests = [
        ("admin_123", Role.ADMIN, None, "delete_user"),
        ("user_123", Role.WRITE, "user_123", "edit_profile"),
        ("user_123", Role.WRITE, "user_456", "edit_profile"),
    ]

    for user_id, role, owner_id, action in policy_tests:
        # Assign role temporarily
        await rbac_manager.assign_role(user_id, role)

        context = PolicyContext(
            user_id=user_id,
            resource_owner_id=owner_id,
            action=action
        )

        result = await policy_engine.evaluate(context)
        print(f"\n  User: {user_id} ({role.value})")
        print(f"  Action: {action}")
        print(f"  Resource owner: {owner_id or 'N/A'}")
        print(f"  Decision: {result.decision.value}")
        print(f"  Reason: {result.reason}")

    print("\n" + "=" * 70)
    print("RBAC Statistics")
    print("=" * 70)

    rbac_stats = rbac_manager.get_statistics()
    policy_stats = policy_engine.get_statistics()

    print(f"\nRBAC Manager:")
    print(f"  Role assignments: {rbac_stats['role_assignments']}")
    print(f"  Permission checks: {rbac_stats['permission_checks']}")
    print(f"  Cache hits: {rbac_stats['cache_hits']}")
    print(f"  Cache hit rate: {rbac_stats['cache_hit_rate']:.1%}")

    print(f"\nPolicy Engine:")
    print(f"  Total evaluations: {policy_stats['evaluations']}")
    print(f"  Allows: {policy_stats['allows']}")
    print(f"  Denies: {policy_stats['denies']}")
    print(f"  Total rules: {policy_stats['total_rules']}")

    print("\n" + "=" * 70)
    print("\n✅ Demo complete!\n")
    print("To run the FastAPI server:")
    print("  uvicorn demos.demo_rbac:app --reload --port 8080")
    print("\nTest with curl:")
    print("  curl http://localhost:8080/")
    print("  curl -H 'Authorization: Bearer user_read_123' http://localhost:8080/query?q=test")
    print("  curl -H 'Authorization: Bearer user_admin_789' http://localhost:8080/admin/users")
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Run demo
    asyncio.run(run_demo())
