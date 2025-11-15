#!/usr/bin/env python3
"""
Basic RBAC Tests (No External Dependencies)

Simplified test suite that doesn't require pytest or other packages.
Tests core RBAC functionality using asyncio.

Run: python HoloLoom/security/tests/test_rbac_basic.py
"""

import asyncio
from datetime import datetime, timedelta, time

# Import directly from submodules to avoid cryptography dependency
from HoloLoom.security.rbac.core import (
    Role,
    Permission,
    RBACManager,
    UserRole,
    create_rbac_manager,
    ROLE_HIERARCHY,
    PERMISSION_MATRIX,
    get_role_level,
    role_inherits_from,
)

from HoloLoom.security.rbac.policy_engine import (
    PolicyEngine,
    PolicyRule,
    PolicyContext,
    PolicyDecision,
    CommonPolicies,
    create_policy_engine,
)


class TestRunner:
    """Simple test runner."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def test(self, name):
        """Decorator for test functions."""
        def decorator(func):
            self.tests.append((name, func))
            return func
        return decorator

    async def run(self):
        """Run all tests."""
        print("=" * 70)
        print("RBAC Basic Tests")
        print("=" * 70)

        for name, test_func in self.tests:
            try:
                print(f"\n▶ {name}...", end=" ")
                if asyncio.iscoroutinefunction(test_func):
                    await test_func()
                else:
                    test_func()
                print("✅ PASS")
                self.passed += 1
            except AssertionError as e:
                print(f"❌ FAIL: {e}")
                self.failed += 1
            except Exception as e:
                print(f"❌ ERROR: {e}")
                self.failed += 1

        print("\n" + "=" * 70)
        print(f"Results: {self.passed} passed, {self.failed} failed")
        print("=" * 70)

        return self.failed == 0


runner = TestRunner()


# ============================================================================
# Role Hierarchy Tests
# ============================================================================

@runner.test("Role hierarchy structure")
def test_role_hierarchy():
    """Test role hierarchy is correctly defined."""
    assert Role.ADMIN in ROLE_HIERARCHY
    assert Role.WRITE in ROLE_HIERARCHY[Role.ADMIN]
    assert Role.READ in ROLE_HIERARCHY[Role.ADMIN]
    assert len(ROLE_HIERARCHY[Role.GUEST]) == 0


@runner.test("Role level ordering")
def test_role_levels():
    """Test role level ordering."""
    assert get_role_level(Role.ADMIN) > get_role_level(Role.WRITE)
    assert get_role_level(Role.WRITE) > get_role_level(Role.READ)
    assert get_role_level(Role.READ) > get_role_level(Role.GUEST)


@runner.test("Role inheritance")
def test_role_inheritance():
    """Test role inheritance checks."""
    assert role_inherits_from(Role.ADMIN, Role.WRITE)
    assert role_inherits_from(Role.ADMIN, Role.READ)
    assert not role_inherits_from(Role.READ, Role.ADMIN)


# ============================================================================
# Permission Matrix Tests
# ============================================================================

@runner.test("Admin has all permissions")
def test_admin_permissions():
    """Admin should have all permissions."""
    admin_perms = PERMISSION_MATRIX[Role.ADMIN]
    assert Permission.QUERY_READ in admin_perms
    assert Permission.QUERY_WRITE in admin_perms
    assert Permission.QUERY_DELETE in admin_perms
    assert Permission.SYSTEM_CONFIG in admin_perms


@runner.test("Write has subset of admin")
def test_write_permissions():
    """Write should have subset of admin permissions."""
    admin_perms = PERMISSION_MATRIX[Role.ADMIN]
    write_perms = PERMISSION_MATRIX[Role.WRITE]
    assert write_perms.issubset(admin_perms)
    assert Permission.QUERY_READ in write_perms
    assert Permission.QUERY_WRITE in write_perms
    assert Permission.QUERY_DELETE not in write_perms


@runner.test("Read has read-only permissions")
def test_read_permissions():
    """Read should have only read permissions."""
    read_perms = PERMISSION_MATRIX[Role.READ]
    assert Permission.QUERY_READ in read_perms
    assert Permission.QUERY_WRITE not in read_perms
    assert Permission.QUERY_DELETE not in read_perms


@runner.test("Guest has minimal permissions")
def test_guest_permissions():
    """Guest should have minimal permissions."""
    guest_perms = PERMISSION_MATRIX[Role.GUEST]
    assert Permission.SYSTEM_HEALTH in guest_perms
    assert Permission.QUERY_READ not in guest_perms


# ============================================================================
# UserRole Tests
# ============================================================================

@runner.test("UserRole creation")
def test_user_role_creation():
    """Test creating UserRole."""
    user_role = UserRole(
        user_id="user_123",
        role=Role.WRITE,
        assigned_at=datetime.now(),
        assigned_by="admin_456"
    )
    assert user_role.user_id == "user_123"
    assert user_role.role == Role.WRITE
    assert not user_role.is_expired()


@runner.test("UserRole expiration")
def test_user_role_expiration():
    """Test UserRole expiration."""
    expired_role = UserRole(
        user_id="user_123",
        role=Role.READ,
        assigned_at=datetime.now() - timedelta(days=2),
        assigned_by="system",
        expires_at=datetime.now() - timedelta(days=1)
    )
    assert expired_role.is_expired()

    valid_role = UserRole(
        user_id="user_456",
        role=Role.WRITE,
        assigned_at=datetime.now(),
        assigned_by="system",
        expires_at=datetime.now() + timedelta(days=30)
    )
    assert not valid_role.is_expired()


@runner.test("UserRole serialization")
def test_user_role_serialization():
    """Test UserRole to_dict/from_dict."""
    user_role = UserRole(
        user_id="user_123",
        role=Role.ADMIN,
        assigned_at=datetime.now(),
        assigned_by="system",
        metadata={"environment": "production"}
    )

    data = user_role.to_dict()
    assert data["user_id"] == "user_123"
    assert data["role"] == "admin"

    restored = UserRole.from_dict(data)
    assert restored.user_id == user_role.user_id
    assert restored.role == user_role.role


# ============================================================================
# RBAC Manager Tests
# ============================================================================

@runner.test("Assign and get role")
async def test_assign_get_role():
    """Test assigning and retrieving roles."""
    rbac = create_rbac_manager(storage_type="memory")

    user_role = await rbac.assign_role("user_123", Role.WRITE)
    assert user_role.user_id == "user_123"
    assert user_role.role == Role.WRITE

    role = await rbac.get_user_role("user_123")
    assert role == Role.WRITE


@runner.test("Get user permissions")
async def test_get_permissions():
    """Test getting user permissions (including inherited)."""
    rbac = create_rbac_manager(storage_type="memory")
    await rbac.assign_role("user_123", Role.WRITE)

    permissions = await rbac.get_user_permissions("user_123")
    assert Permission.QUERY_READ in permissions
    assert Permission.QUERY_WRITE in permissions
    assert Permission.QUERY_DELETE not in permissions


@runner.test("Admin inherits all permissions")
async def test_admin_inherits():
    """Test admin role has all permissions."""
    rbac = create_rbac_manager(storage_type="memory")
    await rbac.assign_role("admin_123", Role.ADMIN)

    permissions = await rbac.get_user_permissions("admin_123")
    assert Permission.QUERY_READ in permissions
    assert Permission.QUERY_WRITE in permissions
    assert Permission.QUERY_DELETE in permissions
    assert Permission.SYSTEM_CONFIG in permissions


@runner.test("Check permission")
async def test_check_permission():
    """Test permission checking."""
    rbac = create_rbac_manager(storage_type="memory")
    await rbac.assign_role("user_123", Role.READ)

    assert await rbac.check_permission("user_123", Permission.QUERY_READ)
    assert not await rbac.check_permission("user_123", Permission.QUERY_WRITE)


@runner.test("Revoke role")
async def test_revoke_role():
    """Test revoking role."""
    rbac = create_rbac_manager(storage_type="memory")

    await rbac.assign_role("user_123", Role.WRITE)
    assert await rbac.get_user_role("user_123") == Role.WRITE

    success = await rbac.revoke_role("user_123")
    assert success
    assert await rbac.get_user_role("user_123") is None


@runner.test("Permission caching")
async def test_permission_caching():
    """Test permission caching."""
    rbac = create_rbac_manager(storage_type="memory", enable_caching=True)
    await rbac.assign_role("user_123", Role.READ)

    # First call (cache miss)
    permissions1 = await rbac.get_user_permissions("user_123")

    # Second call (cache hit)
    permissions2 = await rbac.get_user_permissions("user_123")

    assert permissions1 == permissions2

    stats = rbac.get_statistics()
    assert stats["cache_hits"] > 0


# ============================================================================
# Policy Engine Tests
# ============================================================================

@runner.test("Admin full access policy")
async def test_admin_policy():
    """Test admin full access policy."""
    engine = create_policy_engine(add_common_policies=True)

    context = PolicyContext(
        user_id="admin_123",
        role=Role.ADMIN,
        action="delete_user"
    )

    result = await engine.evaluate(context)
    assert result.decision == PolicyDecision.ALLOW


@runner.test("Owner can edit policy")
async def test_owner_edit():
    """Test owner can edit own resources."""
    engine = create_policy_engine(add_common_policies=True)

    context = PolicyContext(
        user_id="user_123",
        role=Role.WRITE,
        resource_owner_id="user_123",
        action="edit_profile"
    )

    result = await engine.evaluate(context)
    assert result.decision == PolicyDecision.ALLOW


@runner.test("Non-owner cannot edit")
async def test_non_owner_edit():
    """Test non-owner cannot edit others' resources."""
    engine = create_policy_engine(add_common_policies=True)

    context = PolicyContext(
        user_id="user_123",
        role=Role.WRITE,
        resource_owner_id="user_456",
        action="edit_profile"
    )

    result = await engine.evaluate(context)
    assert result.decision == PolicyDecision.DENY


@runner.test("Policy priority ordering")
async def test_policy_priority():
    """Test policies are evaluated in priority order."""
    engine = PolicyEngine()

    async def always_true(ctx):
        return True

    # High priority ALLOW
    engine.add_rule(PolicyRule(
        name="high_allow",
        condition=always_true,
        effect=PolicyDecision.ALLOW,
        priority=100
    ))

    # Low priority DENY
    engine.add_rule(PolicyRule(
        name="low_deny",
        condition=always_true,
        effect=PolicyDecision.DENY,
        priority=1
    ))

    context = PolicyContext(user_id="user_123")
    result = await engine.evaluate(context)

    # High priority ALLOW should win
    assert result.decision == PolicyDecision.ALLOW


# ============================================================================
# Integration Tests
# ============================================================================

@runner.test("RBAC with policy engine integration")
async def test_rbac_policy_integration():
    """Test RBAC manager integrated with policy engine."""
    rbac = create_rbac_manager(storage_type="memory")
    engine = create_policy_engine(rbac=rbac, add_common_policies=True)

    await rbac.assign_role("user_123", Role.WRITE)

    context = PolicyContext(user_id="user_123")
    result = await engine.evaluate(context)

    # Should have role enriched from RBAC
    assert context.role == Role.WRITE


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all tests."""
    success = await runner.run()

    if success:
        print("\n✅ All tests passed!\n")
        return 0
    else:
        print("\n❌ Some tests failed!\n")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
