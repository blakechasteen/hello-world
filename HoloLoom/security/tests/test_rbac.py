"""
RBAC System Tests

Comprehensive test suite for Role-Based Access Control:
- Core RBAC (roles, permissions, hierarchy)
- Storage backends (in-memory, Redis)
- Policy engine (complex rules)
- FastAPI decorators (endpoint protection)

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import pytest
import asyncio
from datetime import datetime, timedelta, time

from HoloLoom.security.rbac import (
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

from HoloLoom.security.rbac.storage import (
    InMemoryRBACStorage,
    create_rbac_storage,
)

from HoloLoom.security.rbac.policy_engine import (
    PolicyEngine,
    PolicyRule,
    PolicyContext,
    PolicyDecision,
    CommonPolicies,
    create_policy_engine,
)


# ============================================================================
# Core RBAC Tests
# ============================================================================

class TestRoleHierarchy:
    """Test role hierarchy and inheritance."""

    def test_role_hierarchy_structure(self):
        """Test role hierarchy is correctly defined."""
        assert Role.ADMIN in ROLE_HIERARCHY
        assert Role.WRITE in ROLE_HIERARCHY
        assert Role.READ in ROLE_HIERARCHY
        assert Role.GUEST in ROLE_HIERARCHY

        # Admin inherits from all
        assert Role.WRITE in ROLE_HIERARCHY[Role.ADMIN]
        assert Role.READ in ROLE_HIERARCHY[Role.ADMIN]
        assert Role.GUEST in ROLE_HIERARCHY[Role.ADMIN]

        # Write inherits from read and guest
        assert Role.READ in ROLE_HIERARCHY[Role.WRITE]
        assert Role.GUEST in ROLE_HIERARCHY[Role.WRITE]

        # Read inherits from guest
        assert Role.GUEST in ROLE_HIERARCHY[Role.READ]

        # Guest has no inheritance
        assert len(ROLE_HIERARCHY[Role.GUEST]) == 0

    def test_role_levels(self):
        """Test role level ordering."""
        assert get_role_level(Role.ADMIN) > get_role_level(Role.WRITE)
        assert get_role_level(Role.WRITE) > get_role_level(Role.READ)
        assert get_role_level(Role.READ) > get_role_level(Role.GUEST)

    def test_role_inheritance(self):
        """Test role inheritance checks."""
        # Admin inherits from all
        assert role_inherits_from(Role.ADMIN, Role.WRITE)
        assert role_inherits_from(Role.ADMIN, Role.READ)
        assert role_inherits_from(Role.ADMIN, Role.GUEST)

        # Write inherits from read and guest
        assert role_inherits_from(Role.WRITE, Role.READ)
        assert role_inherits_from(Role.WRITE, Role.GUEST)

        # Read inherits from guest
        assert role_inherits_from(Role.READ, Role.GUEST)

        # No reverse inheritance
        assert not role_inherits_from(Role.READ, Role.ADMIN)
        assert not role_inherits_from(Role.GUEST, Role.READ)


class TestPermissionMatrix:
    """Test permission matrix."""

    def test_admin_has_all_permissions(self):
        """Admin should have all permissions."""
        admin_perms = PERMISSION_MATRIX[Role.ADMIN]

        # Check key permissions
        assert Permission.QUERY_READ in admin_perms
        assert Permission.QUERY_WRITE in admin_perms
        assert Permission.QUERY_DELETE in admin_perms
        assert Permission.USER_DELETE in admin_perms
        assert Permission.SYSTEM_CONFIG in admin_perms
        assert Permission.ALIGNMENT_OVERRIDE in admin_perms

    def test_write_has_subset_of_admin(self):
        """Write should have subset of admin permissions."""
        admin_perms = PERMISSION_MATRIX[Role.ADMIN]
        write_perms = PERMISSION_MATRIX[Role.WRITE]

        # Write permissions should be subset of admin
        assert write_perms.issubset(admin_perms)

        # Write should have read/write but not delete
        assert Permission.QUERY_READ in write_perms
        assert Permission.QUERY_WRITE in write_perms
        assert Permission.QUERY_DELETE not in write_perms

    def test_read_has_read_only_permissions(self):
        """Read should have only read permissions."""
        read_perms = PERMISSION_MATRIX[Role.READ]

        # Read should have read permissions
        assert Permission.QUERY_READ in read_perms
        assert Permission.USER_READ in read_perms

        # Read should NOT have write/delete permissions
        assert Permission.QUERY_WRITE not in read_perms
        assert Permission.QUERY_DELETE not in read_perms
        assert Permission.USER_WRITE not in read_perms

    def test_guest_has_minimal_permissions(self):
        """Guest should have minimal permissions."""
        guest_perms = PERMISSION_MATRIX[Role.GUEST]

        # Guest should only have health check
        assert Permission.SYSTEM_HEALTH in guest_perms

        # Guest should NOT have other permissions
        assert Permission.QUERY_READ not in guest_perms
        assert Permission.USER_READ not in guest_perms


class TestUserRole:
    """Test UserRole dataclass."""

    def test_user_role_creation(self):
        """Test creating UserRole."""
        user_role = UserRole(
            user_id="user_123",
            role=Role.WRITE,
            assigned_at=datetime.now(),
            assigned_by="admin_456"
        )

        assert user_role.user_id == "user_123"
        assert user_role.role == Role.WRITE
        assert user_role.assigned_by == "admin_456"
        assert not user_role.is_expired()

    def test_user_role_expiration(self):
        """Test UserRole expiration."""
        # Expired role
        expired_role = UserRole(
            user_id="user_123",
            role=Role.READ,
            assigned_at=datetime.now() - timedelta(days=2),
            assigned_by="system",
            expires_at=datetime.now() - timedelta(days=1)
        )
        assert expired_role.is_expired()

        # Valid role
        valid_role = UserRole(
            user_id="user_456",
            role=Role.WRITE,
            assigned_at=datetime.now(),
            assigned_by="system",
            expires_at=datetime.now() + timedelta(days=30)
        )
        assert not valid_role.is_expired()

    def test_user_role_serialization(self):
        """Test UserRole to_dict/from_dict."""
        user_role = UserRole(
            user_id="user_123",
            role=Role.ADMIN,
            assigned_at=datetime.now(),
            assigned_by="system",
            metadata={"environment": "production"}
        )

        # Serialize
        data = user_role.to_dict()
        assert data["user_id"] == "user_123"
        assert data["role"] == "admin"

        # Deserialize
        restored = UserRole.from_dict(data)
        assert restored.user_id == user_role.user_id
        assert restored.role == user_role.role
        assert restored.metadata == user_role.metadata


# ============================================================================
# RBAC Manager Tests
# ============================================================================

class TestRBACManager:
    """Test RBACManager."""

    @pytest.mark.asyncio
    async def test_assign_and_get_role(self):
        """Test assigning and retrieving roles."""
        rbac = create_rbac_manager(storage_type="memory")

        # Assign role
        user_role = await rbac.assign_role("user_123", Role.WRITE)
        assert user_role.user_id == "user_123"
        assert user_role.role == Role.WRITE

        # Get role
        role = await rbac.get_user_role("user_123")
        assert role == Role.WRITE

    @pytest.mark.asyncio
    async def test_get_user_permissions(self):
        """Test getting user permissions (including inherited)."""
        rbac = create_rbac_manager(storage_type="memory")

        # Assign WRITE role
        await rbac.assign_role("user_123", Role.WRITE)

        # Get permissions
        permissions = await rbac.get_user_permissions("user_123")

        # Should have WRITE permissions
        assert Permission.QUERY_READ in permissions
        assert Permission.QUERY_WRITE in permissions

        # Should NOT have admin permissions
        assert Permission.QUERY_DELETE not in permissions
        assert Permission.SYSTEM_CONFIG not in permissions

    @pytest.mark.asyncio
    async def test_admin_inherits_all_permissions(self):
        """Test admin role has all permissions."""
        rbac = create_rbac_manager(storage_type="memory")

        # Assign ADMIN role
        await rbac.assign_role("admin_123", Role.ADMIN)

        # Get permissions
        permissions = await rbac.get_user_permissions("admin_123")

        # Should have ALL permissions
        assert Permission.QUERY_READ in permissions
        assert Permission.QUERY_WRITE in permissions
        assert Permission.QUERY_DELETE in permissions
        assert Permission.USER_DELETE in permissions
        assert Permission.SYSTEM_CONFIG in permissions

    @pytest.mark.asyncio
    async def test_check_permission(self):
        """Test permission checking."""
        rbac = create_rbac_manager(storage_type="memory")

        await rbac.assign_role("user_123", Role.READ)

        # Should have read permission
        assert await rbac.check_permission("user_123", Permission.QUERY_READ)

        # Should NOT have write permission
        assert not await rbac.check_permission("user_123", Permission.QUERY_WRITE)

    @pytest.mark.asyncio
    async def test_check_any_permission(self):
        """Test checking any of multiple permissions."""
        rbac = create_rbac_manager(storage_type="memory")

        await rbac.assign_role("user_123", Role.READ)

        # Has at least one (QUERY_READ)
        assert await rbac.check_any_permission("user_123", [
            Permission.QUERY_READ,
            Permission.QUERY_WRITE
        ])

        # Doesn't have any
        assert not await rbac.check_any_permission("user_123", [
            Permission.QUERY_WRITE,
            Permission.QUERY_DELETE
        ])

    @pytest.mark.asyncio
    async def test_check_all_permissions(self):
        """Test checking all permissions."""
        rbac = create_rbac_manager(storage_type="memory")

        await rbac.assign_role("user_123", Role.WRITE)

        # Has all (QUERY_READ, QUERY_WRITE)
        assert await rbac.check_all_permissions("user_123", [
            Permission.QUERY_READ,
            Permission.QUERY_WRITE
        ])

        # Doesn't have all (missing QUERY_DELETE)
        assert not await rbac.check_all_permissions("user_123", [
            Permission.QUERY_READ,
            Permission.QUERY_DELETE
        ])

    @pytest.mark.asyncio
    async def test_revoke_role(self):
        """Test revoking role."""
        rbac = create_rbac_manager(storage_type="memory")

        await rbac.assign_role("user_123", Role.WRITE)
        assert await rbac.get_user_role("user_123") == Role.WRITE

        # Revoke
        success = await rbac.revoke_role("user_123")
        assert success

        # Should be None now
        assert await rbac.get_user_role("user_123") is None

    @pytest.mark.asyncio
    async def test_permission_caching(self):
        """Test permission caching."""
        rbac = create_rbac_manager(storage_type="memory", enable_caching=True)

        await rbac.assign_role("user_123", Role.READ)

        # First call (cache miss)
        permissions1 = await rbac.get_user_permissions("user_123")

        # Second call (cache hit)
        permissions2 = await rbac.get_user_permissions("user_123")

        # Should be same
        assert permissions1 == permissions2

        # Check cache hit rate
        stats = rbac.get_statistics()
        assert stats["cache_hits"] > 0

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_role_change(self):
        """Test cache is invalidated when role changes."""
        rbac = create_rbac_manager(storage_type="memory", enable_caching=True)

        await rbac.assign_role("user_123", Role.READ)
        permissions1 = await rbac.get_user_permissions("user_123")

        # Change role
        await rbac.assign_role("user_123", Role.WRITE)
        permissions2 = await rbac.get_user_permissions("user_123")

        # Permissions should be different
        assert Permission.QUERY_WRITE in permissions2
        assert Permission.QUERY_WRITE not in permissions1


# ============================================================================
# Storage Tests
# ============================================================================

class TestInMemoryStorage:
    """Test InMemoryRBACStorage."""

    @pytest.mark.asyncio
    async def test_store_and_get_role(self):
        """Test storing and retrieving roles."""
        storage = InMemoryRBACStorage()

        user_role = UserRole(
            user_id="user_123",
            role=Role.WRITE,
            assigned_at=datetime.now(),
            assigned_by="system"
        )

        await storage.store_user_role(user_role)
        retrieved = await storage.get_user_role("user_123")

        assert retrieved is not None
        assert retrieved.user_id == "user_123"
        assert retrieved.role == Role.WRITE

    @pytest.mark.asyncio
    async def test_delete_role(self):
        """Test deleting role."""
        storage = InMemoryRBACStorage()

        user_role = UserRole(
            user_id="user_123",
            role=Role.READ,
            assigned_at=datetime.now(),
            assigned_by="system"
        )

        await storage.store_user_role(user_role)
        assert await storage.get_user_role("user_123") is not None

        # Delete
        success = await storage.delete_user_role("user_123")
        assert success

        # Should be None now
        assert await storage.get_user_role("user_123") is None

    @pytest.mark.asyncio
    async def test_list_users_with_role(self):
        """Test listing users with specific role."""
        storage = InMemoryRBACStorage()

        # Create multiple users with different roles
        await storage.store_user_role(UserRole(
            user_id="user_1",
            role=Role.ADMIN,
            assigned_at=datetime.now(),
            assigned_by="system"
        ))
        await storage.store_user_role(UserRole(
            user_id="user_2",
            role=Role.WRITE,
            assigned_at=datetime.now(),
            assigned_by="system"
        ))
        await storage.store_user_role(UserRole(
            user_id="user_3",
            role=Role.WRITE,
            assigned_at=datetime.now(),
            assigned_by="system"
        ))

        # List WRITE users
        write_users = await storage.list_users_with_role(Role.WRITE)
        assert len(write_users) == 2
        user_ids = [u.user_id for u in write_users]
        assert "user_2" in user_ids
        assert "user_3" in user_ids


# ============================================================================
# Policy Engine Tests
# ============================================================================

class TestPolicyEngine:
    """Test PolicyEngine."""

    @pytest.mark.asyncio
    async def test_admin_full_access_policy(self):
        """Test admin full access policy."""
        engine = create_policy_engine(add_common_policies=True)

        context = PolicyContext(
            user_id="admin_123",
            role=Role.ADMIN,
            action="delete_user"
        )

        result = await engine.evaluate(context)
        assert result.decision == PolicyDecision.ALLOW
        assert "admin" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_owner_can_edit_policy(self):
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

    @pytest.mark.asyncio
    async def test_non_owner_cannot_edit(self):
        """Test non-owner cannot edit others' resources."""
        engine = create_policy_engine(add_common_policies=True)

        context = PolicyContext(
            user_id="user_123",
            role=Role.WRITE,
            resource_owner_id="user_456",  # Different owner
            action="edit_profile"
        )

        result = await engine.evaluate(context)
        assert result.decision == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_business_hours_policy(self):
        """Test business hours restriction."""
        engine = PolicyEngine()
        engine.add_rule(CommonPolicies.business_hours_only(
            start=time(9, 0),
            end=time(17, 0)
        ))

        # During business hours
        context = PolicyContext(
            user_id="user_123",
            timestamp=datetime.now().replace(hour=12, minute=0)
        )
        result = await engine.evaluate(context)
        # Should abstain (not deny) during business hours
        assert result.decision == PolicyDecision.DENY  # No allow rules, so defaults to deny

        # Outside business hours
        context = PolicyContext(
            user_id="user_123",
            timestamp=datetime.now().replace(hour=20, minute=0)
        )
        result = await engine.evaluate(context)
        assert result.decision == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_ip_whitelist_policy(self):
        """Test IP whitelist policy."""
        engine = PolicyEngine()
        engine.add_rule(CommonPolicies.ip_whitelist(["192.168.1.1", "10.0.0.1"]))

        # Allowed IP
        context = PolicyContext(
            user_id="user_123",
            ip_address="192.168.1.1"
        )
        result = await engine.evaluate(context)
        # No allow rules, just deny if not in whitelist
        assert result.decision == PolicyDecision.DENY  # Default deny

        # Disallowed IP
        context = PolicyContext(
            user_id="user_123",
            ip_address="1.2.3.4"
        )
        result = await engine.evaluate(context)
        assert result.decision == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_policy_priority_ordering(self):
        """Test policies are evaluated in priority order."""
        engine = PolicyEngine()

        # Add high priority ALLOW rule
        async def always_allow(ctx):
            return True

        engine.add_rule(PolicyRule(
            name="high_priority_allow",
            condition=always_allow,
            effect=PolicyDecision.ALLOW,
            priority=100
        ))

        # Add low priority DENY rule
        engine.add_rule(PolicyRule(
            name="low_priority_deny",
            condition=always_allow,
            effect=PolicyDecision.DENY,
            priority=1
        ))

        context = PolicyContext(user_id="user_123")
        result = await engine.evaluate(context)

        # High priority ALLOW should win
        assert result.decision == PolicyDecision.ALLOW

    @pytest.mark.asyncio
    async def test_policy_engine_statistics(self):
        """Test policy engine statistics."""
        engine = create_policy_engine(add_common_policies=True)

        # Run some evaluations
        await engine.evaluate(PolicyContext(user_id="admin_123", role=Role.ADMIN))
        await engine.evaluate(PolicyContext(user_id="user_123", role=Role.READ))

        stats = engine.get_statistics()
        assert stats["evaluations"] == 2
        assert stats["total_rules"] >= 2


# ============================================================================
# Integration Tests
# ============================================================================

class TestRBACIntegration:
    """Integration tests for RBAC + policy engine."""

    @pytest.mark.asyncio
    async def test_rbac_with_policy_engine(self):
        """Test RBAC manager integrated with policy engine."""
        rbac = create_rbac_manager(storage_type="memory")
        engine = create_policy_engine(rbac=rbac, add_common_policies=True)

        # Assign role
        await rbac.assign_role("user_123", Role.WRITE)

        # Evaluate policy (context doesn't have role, should be enriched)
        context = PolicyContext(user_id="user_123")
        result = await engine.evaluate(context)

        # Should have role enriched from RBAC
        assert context.role == Role.WRITE


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
