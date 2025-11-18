"""
Tenant Isolation Layer for Multi-Tenant Memory Systems
=======================================================

**Purpose**: Provides strong tenant isolation guarantees across all HoloLoom
memory backends (NetworkX, Neo4j, Qdrant, in-memory).

**Date**: 2025-11-17
**Phase**: Tenant Isolation and PII Flows
**Key Features**:
- Tenant-scoped memory namespacing
- Row-level security enforcement
- Cross-tenant access prevention
- Tenant data lifecycle management (delete, export)
- Audit logging for all tenant operations

**Philosophy**:
"Zero-trust multi-tenancy: A tenant should never be able to access,
modify, or infer the existence of another tenant's data."

**Compliance**:
- SOC 2 CC6.1: Logical access controls
- GDPR Article 32: Security of processing
- HIPAA § 164.308: Administrative safeguards

Author: HoloLoom Security Team
Timestamp: 2025-11-17
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Set, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import asyncio
import re
from uuid import UUID, uuid4


logger = logging.getLogger(__name__)


# ============================================================================
# Input Validation
# ============================================================================

# Tenant ID must be alphanumeric with underscores and hyphens only
# No path traversal characters (/, \, ..), no colons (used as delimiter)
TENANT_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')


def _validate_tenant_id(tenant_id: str) -> None:
    """
    Validate tenant ID to prevent security vulnerabilities.

    Security checks:
    - Not empty
    - Only alphanumeric, underscore, hyphen (no path traversal)
    - No colons (used as delimiter in scoped keys)
    - Length limit (1-64 chars)

    Args:
        tenant_id: Tenant identifier to validate

    Raises:
        ValueError: If tenant ID is invalid

    Examples:
        _validate_tenant_id("acme_corp")        # OK
        _validate_tenant_id("acme-corp-123")    # OK
        _validate_tenant_id("../other_tenant")  # FAIL - path traversal
        _validate_tenant_id("tenant:fake")      # FAIL - colon injection
        _validate_tenant_id("")                 # FAIL - empty
        _validate_tenant_id("x" * 100)          # FAIL - too long
    """
    if not tenant_id:
        raise ValueError("Tenant ID cannot be empty")

    if not TENANT_ID_PATTERN.match(tenant_id):
        raise ValueError(
            f"Invalid tenant ID: '{tenant_id}'. "
            "Must be 1-64 alphanumeric characters with underscores or hyphens only. "
            "No path traversal (/, \\, ..) or delimiter (:) characters allowed."
        )


# ============================================================================
# Tenant Types and Enums
# ============================================================================

class TenantTier(Enum):
    """
    Tenant subscription tier.

    Determines quotas, features, and SLA guarantees.
    """
    FREE = "free"              # 100 memories, 1 user, basic support
    STARTER = "starter"        # 10K memories, 5 users, email support
    PROFESSIONAL = "professional"  # 1M memories, 50 users, priority support
    ENTERPRISE = "enterprise"  # Unlimited, custom SLA, dedicated support


class TenantStatus(Enum):
    """Tenant account status."""
    ACTIVE = "active"          # Normal operation
    SUSPENDED = "suspended"    # Temporarily disabled
    DELETED = "deleted"        # Soft-deleted (data retained for 30 days)
    ARCHIVED = "archived"      # Permanently archived (read-only)


@dataclass
class TenantMetadata:
    """
    Metadata for a tenant.

    Contains:
    - Identification (tenant_id, name)
    - Status and tier
    - Quotas and usage
    - Compliance settings
    - Timestamps

    Example:
        tenant = TenantMetadata(
            tenant_id="acme_corp",
            name="Acme Corporation",
            tier=TenantTier.ENTERPRISE,
            compliance_mode="hipaa",
            data_residency="us-west-2"
        )
    """
    tenant_id: str                              # Unique tenant identifier
    name: str                                   # Human-readable name
    tier: TenantTier = TenantTier.FREE          # Subscription tier
    status: TenantStatus = TenantStatus.ACTIVE  # Account status

    # Quotas (tier-dependent)
    max_memories: int = 100                     # Max memories allowed
    max_users: int = 1                          # Max users per tenant
    max_api_calls_per_hour: int = 1000          # Rate limit

    # Compliance settings
    compliance_mode: Optional[str] = None       # "hipaa", "gdpr", "soc2", None
    data_residency: Optional[str] = None        # "us", "eu", "us-west-2", etc.
    encryption_required: bool = False           # Encrypt data at rest?
    audit_all_access: bool = False              # Log every access?

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    deleted_at: Optional[datetime] = None       # Soft delete timestamp

    # Usage tracking
    current_memories: int = 0                   # Current memory count
    current_users: int = 0                      # Current user count
    api_calls_this_hour: int = 0                # Rate limit tracking

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'tenant_id': self.tenant_id,
            'name': self.name,
            'tier': self.tier.value,
            'status': self.status.value,
            'max_memories': self.max_memories,
            'max_users': self.max_users,
            'max_api_calls_per_hour': self.max_api_calls_per_hour,
            'compliance_mode': self.compliance_mode,
            'data_residency': self.data_residency,
            'encryption_required': self.encryption_required,
            'audit_all_access': self.audit_all_access,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'deleted_at': self.deleted_at.isoformat() if self.deleted_at else None,
            'current_memories': self.current_memories,
            'current_users': self.current_users,
            'api_calls_this_hour': self.api_calls_this_hour,
        }


@dataclass
class TenantContext:
    """
    Execution context for a tenant operation.

    Every memory operation MUST include a TenantContext to enforce isolation.

    Contains:
    - tenant_id: Which tenant is making the request
    - user_id: Which user within the tenant
    - request_id: Unique request ID for auditing
    - ip_address: Client IP (for audit trail)
    - permissions: What this user can do

    Example:
        context = TenantContext(
            tenant_id="acme_corp",
            user_id="john.doe@acme.com",
            request_id=uuid4(),
            ip_address="203.0.113.42",
            permissions={"read", "write"}
        )

        # Use in memory operations
        memories = await memory_store.recall(query, context=context)
    """
    tenant_id: str                              # Which tenant
    user_id: str                                # Which user
    request_id: UUID = field(default_factory=uuid4)  # Request ID
    ip_address: Optional[str] = None            # Client IP
    permissions: Set[str] = field(default_factory=lambda: {"read"})  # What can they do?
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def can_read(self) -> bool:
        """Check if user has read permission."""
        return "read" in self.permissions

    def can_write(self) -> bool:
        """Check if user has write permission."""
        return "write" in self.permissions

    def can_delete(self) -> bool:
        """Check if user has delete permission."""
        return "delete" in self.permissions

    def can_admin(self) -> bool:
        """Check if user has admin permission."""
        return "admin" in self.permissions


# ============================================================================
# Tenant Isolation Exceptions
# ============================================================================

class TenantIsolationError(Exception):
    """Base exception for tenant isolation violations."""
    pass


class CrossTenantAccessError(TenantIsolationError):
    """Raised when attempting to access another tenant's data."""
    pass


class TenantSuspendedError(TenantIsolationError):
    """Raised when tenant account is suspended."""
    pass


class TenantQuotaExceededError(TenantIsolationError):
    """Raised when tenant exceeds quota."""
    pass


class InsufficientPermissionsError(TenantIsolationError):
    """Raised when user lacks required permissions."""
    pass


# ============================================================================
# Tenant Isolation Layer
# ============================================================================

class TenantIsolationLayer:
    """
    Enforces tenant isolation across all memory operations.

    Features:
    - Automatic tenant ID prefixing for all keys/IDs
    - Permission checks before every operation
    - Quota enforcement
    - Cross-tenant access prevention
    - Audit logging

    Usage:
        isolation = TenantIsolationLayer(registry)

        # Create tenant
        tenant = await isolation.create_tenant(
            tenant_id="acme_corp",
            name="Acme Corporation",
            tier=TenantTier.ENTERPRISE
        )

        # Create context for operation
        context = TenantContext(
            tenant_id="acme_corp",
            user_id="john.doe@acme.com",
            permissions={"read", "write"}
        )

        # Validate before operation
        await isolation.validate_operation(context, operation="write")

        # Scope memory key to tenant
        scoped_key = isolation.scope_key("memory_123", context.tenant_id)
        # Result: "tenant:acme_corp:memory_123"

        # Store memory with tenant isolation
        await memory_store.store(scoped_key, data)
    """

    def __init__(
        self,
        tenant_registry: Optional['TenantRegistry'] = None,
        enable_audit_logging: bool = True,
        strict_mode: bool = True
    ):
        """
        Initialize tenant isolation layer.

        Args:
            tenant_registry: Registry of all tenants
            enable_audit_logging: Log all tenant operations
            strict_mode: Fail on any isolation violation (recommended: True)
        """
        self.tenant_registry = tenant_registry or TenantRegistry()
        self.enable_audit_logging = enable_audit_logging
        self.strict_mode = strict_mode

        # Audit log buffer
        self._audit_log: List[Dict[str, Any]] = []

        logger.info("TenantIsolationLayer initialized (strict_mode=%s)", strict_mode)

    async def validate_operation(
        self,
        context: TenantContext,
        operation: str
    ) -> None:
        """
        Validate that operation is allowed for this tenant.

        Checks:
        1. Tenant exists and is active
        2. User has required permissions
        3. Quotas not exceeded (for write operations)

        Args:
            context: Tenant context
            operation: Operation type ("read", "write", "delete", "admin")

        Raises:
            TenantSuspendedError: If tenant is suspended/deleted
            InsufficientPermissionsError: If user lacks permission
            TenantQuotaExceededError: If quota exceeded
        """
        # Get tenant metadata
        tenant = await self.tenant_registry.get_tenant(context.tenant_id)

        if not tenant:
            raise TenantIsolationError(f"Tenant not found: {context.tenant_id}")

        # Check tenant status
        if tenant.status == TenantStatus.SUSPENDED:
            raise TenantSuspendedError(
                f"Tenant {context.tenant_id} is suspended"
            )

        if tenant.status == TenantStatus.DELETED:
            raise TenantSuspendedError(
                f"Tenant {context.tenant_id} is deleted"
            )

        # Check permissions
        if operation == "read" and not context.can_read():
            raise InsufficientPermissionsError(
                f"User {context.user_id} lacks read permission"
            )

        if operation == "write" and not context.can_write():
            raise InsufficientPermissionsError(
                f"User {context.user_id} lacks write permission"
            )

        if operation == "delete" and not context.can_delete():
            raise InsufficientPermissionsError(
                f"User {context.user_id} lacks delete permission"
            )

        if operation == "admin" and not context.can_admin():
            raise InsufficientPermissionsError(
                f"User {context.user_id} lacks admin permission"
            )

        # Check quotas for write operations
        if operation == "write":
            if tenant.current_memories >= tenant.max_memories:
                raise TenantQuotaExceededError(
                    f"Tenant {context.tenant_id} exceeded memory quota "
                    f"({tenant.current_memories}/{tenant.max_memories})"
                )

        # Audit log
        if self.enable_audit_logging or tenant.audit_all_access:
            await self._log_operation(context, operation, "allowed")

    def scope_key(self, key: str, tenant_id: str) -> str:
        """
        Scope a key/ID to a tenant namespace.

        Args:
            key: Original key
            tenant_id: Tenant ID

        Returns:
            Scoped key: "tenant:{tenant_id}:{key}"

        Raises:
            ValueError: If tenant_id is invalid (security check)

        Example:
            scoped = scope_key("memory_123", "acme_corp")
            # Result: "tenant:acme_corp:memory_123"
        """
        # SECURITY: Validate tenant ID to prevent path traversal
        _validate_tenant_id(tenant_id)
        return f"tenant:{tenant_id}:{key}"

    def unscope_key(self, scoped_key: str) -> tuple[str, str]:
        """
        Extract tenant ID and original key from scoped key.

        Args:
            scoped_key: Scoped key "tenant:{tenant_id}:{key}"

        Returns:
            (tenant_id, original_key)

        Raises:
            ValueError: If key format is invalid or tenant_id is malicious

        Example:
            tenant_id, key = unscope_key("tenant:acme_corp:memory_123")
            # tenant_id = "acme_corp", key = "memory_123"
        """
        parts = scoped_key.split(":", 2)
        if len(parts) != 3 or parts[0] != "tenant":
            raise ValueError(f"Invalid scoped key format: {scoped_key}")

        tenant_id = parts[1]
        original_key = parts[2]

        # SECURITY: Validate extracted tenant ID to prevent path traversal
        _validate_tenant_id(tenant_id)

        return tenant_id, original_key

    def validate_scoped_key(
        self,
        scoped_key: str,
        expected_tenant_id: str
    ) -> bool:
        """
        Validate that scoped key belongs to expected tenant.

        Args:
            scoped_key: Scoped key to validate
            expected_tenant_id: Expected tenant ID

        Returns:
            True if valid, False otherwise

        Example:
            valid = validate_scoped_key(
                "tenant:acme_corp:memory_123",
                "acme_corp"
            )  # True

            valid = validate_scoped_key(
                "tenant:other_corp:memory_123",
                "acme_corp"
            )  # False - cross-tenant access attempt!
        """
        try:
            tenant_id, _ = self.unscope_key(scoped_key)
            return tenant_id == expected_tenant_id
        except ValueError:
            return False

    async def filter_cross_tenant_data(
        self,
        items: List[Dict[str, Any]],
        tenant_id: str,
        key_field: str = "id"
    ) -> List[Dict[str, Any]]:
        """
        Filter out items that don't belong to tenant.

        Used as defense-in-depth to prevent leaks from underlying storage.

        Args:
            items: List of items to filter
            tenant_id: Expected tenant ID
            key_field: Field name containing the scoped key

        Returns:
            Filtered list containing only tenant's data

        Example:
            results = await storage.query(...)
            safe_results = await isolation.filter_cross_tenant_data(
                results,
                tenant_id="acme_corp",
                key_field="memory_id"
            )
        """
        filtered = []

        for item in items:
            scoped_key = item.get(key_field)
            if not scoped_key:
                continue

            # Check if key belongs to this tenant
            if self.validate_scoped_key(scoped_key, tenant_id):
                filtered.append(item)
            else:
                # Log potential cross-tenant leak
                logger.warning(
                    "Cross-tenant data leak prevented: %s tried to access %s",
                    tenant_id,
                    scoped_key
                )

                if self.strict_mode:
                    raise CrossTenantAccessError(
                        f"Tenant {tenant_id} attempted to access {scoped_key}"
                    )

        return filtered

    async def _log_operation(
        self,
        context: TenantContext,
        operation: str,
        status: str
    ) -> None:
        """Log operation to audit trail."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "request_id": str(context.request_id),
            "operation": operation,
            "status": status,
            "ip_address": context.ip_address,
        }

        self._audit_log.append(log_entry)

        # Keep only recent logs (last 10,000)
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]

    async def get_audit_log(
        self,
        tenant_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get audit log for tenant.

        Args:
            tenant_id: Tenant to get logs for
            limit: Maximum number of logs to return

        Returns:
            List of audit log entries
        """
        logs = [
            log for log in self._audit_log
            if log["tenant_id"] == tenant_id
        ]

        return logs[-limit:]


# ============================================================================
# Tenant Registry
# ============================================================================

class TenantRegistry:
    """
    Registry of all tenants in the system.

    Provides CRUD operations for tenant metadata.

    Usage:
        registry = TenantRegistry()

        # Create tenant
        tenant = await registry.create_tenant(
            tenant_id="acme_corp",
            name="Acme Corporation",
            tier=TenantTier.ENTERPRISE
        )

        # Get tenant
        tenant = await registry.get_tenant("acme_corp")

        # List all tenants
        all_tenants = await registry.list_tenants()

        # Delete tenant (soft delete)
        await registry.delete_tenant("acme_corp")
    """

    def __init__(self):
        """Initialize tenant registry."""
        # In-memory storage (replace with database in production)
        self._tenants: Dict[str, TenantMetadata] = {}
        logger.info("TenantRegistry initialized")

    async def create_tenant(
        self,
        tenant_id: str,
        name: str,
        tier: TenantTier = TenantTier.FREE,
        **kwargs
    ) -> TenantMetadata:
        """
        Create a new tenant.

        Args:
            tenant_id: Unique tenant identifier
            name: Human-readable name
            tier: Subscription tier
            **kwargs: Additional metadata fields

        Returns:
            TenantMetadata for new tenant

        Raises:
            ValueError: If tenant_id already exists or is invalid
        """
        # SECURITY: Validate tenant ID to prevent path traversal
        _validate_tenant_id(tenant_id)

        if tenant_id in self._tenants:
            raise ValueError(f"Tenant already exists: {tenant_id}")

        # Set tier-based quotas
        quotas = self._get_tier_quotas(tier)

        tenant = TenantMetadata(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            **quotas,
            **kwargs
        )

        self._tenants[tenant_id] = tenant

        logger.info("Created tenant: %s (tier=%s)", tenant_id, tier.value)

        return tenant

    async def get_tenant(self, tenant_id: str) -> Optional[TenantMetadata]:
        """
        Get tenant metadata.

        Args:
            tenant_id: Tenant to retrieve

        Returns:
            TenantMetadata or None if not found
        """
        return self._tenants.get(tenant_id)

    async def update_tenant(
        self,
        tenant_id: str,
        **updates
    ) -> TenantMetadata:
        """
        Update tenant metadata.

        Args:
            tenant_id: Tenant to update
            **updates: Fields to update

        Returns:
            Updated TenantMetadata

        Raises:
            ValueError: If tenant not found
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant not found: {tenant_id}")

        # Update fields
        for key, value in updates.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)

        tenant.updated_at = datetime.utcnow()

        logger.info("Updated tenant: %s", tenant_id)

        return tenant

    async def delete_tenant(
        self,
        tenant_id: str,
        hard_delete: bool = False
    ) -> None:
        """
        Delete tenant (soft delete by default).

        Args:
            tenant_id: Tenant to delete
            hard_delete: Permanently delete (True) or soft delete (False)

        Raises:
            ValueError: If tenant not found
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant not found: {tenant_id}")

        if hard_delete:
            # Permanent deletion
            del self._tenants[tenant_id]
            logger.warning("Hard deleted tenant: %s", tenant_id)
        else:
            # Soft delete
            tenant.status = TenantStatus.DELETED
            tenant.deleted_at = datetime.utcnow()
            logger.info("Soft deleted tenant: %s", tenant_id)

    async def list_tenants(
        self,
        status: Optional[TenantStatus] = None
    ) -> List[TenantMetadata]:
        """
        List all tenants.

        Args:
            status: Filter by status (None = all)

        Returns:
            List of TenantMetadata
        """
        tenants = list(self._tenants.values())

        if status:
            tenants = [t for t in tenants if t.status == status]

        return tenants

    def _get_tier_quotas(self, tier: TenantTier) -> Dict[str, int]:
        """Get quota limits for tier."""
        tier_quotas = {
            TenantTier.FREE: {
                "max_memories": 100,
                "max_users": 1,
                "max_api_calls_per_hour": 100,
            },
            TenantTier.STARTER: {
                "max_memories": 10000,
                "max_users": 5,
                "max_api_calls_per_hour": 1000,
            },
            TenantTier.PROFESSIONAL: {
                "max_memories": 1000000,
                "max_users": 50,
                "max_api_calls_per_hour": 10000,
            },
            TenantTier.ENTERPRISE: {
                "max_memories": -1,  # Unlimited
                "max_users": -1,
                "max_api_calls_per_hour": -1,
            },
        }

        return tier_quotas.get(tier, tier_quotas[TenantTier.FREE])


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'TenantTier',
    'TenantStatus',
    'TenantMetadata',
    'TenantContext',
    'TenantIsolationError',
    'CrossTenantAccessError',
    'TenantSuspendedError',
    'TenantQuotaExceededError',
    'InsufficientPermissionsError',
    'TenantIsolationLayer',
    'TenantRegistry',
]
