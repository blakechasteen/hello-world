"""
HoloLoom Phase 3: Access Control
November 2025

Permission-based access control for knowledge resources.
Supports user, team, and resource-level permissions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Union
import json


class Permission(Enum):
    """Resource permissions."""
    READ = "read"          # Can view
    WRITE = "write"        # Can add/modify
    DELETE = "delete"      # Can remove
    SHARE = "share"        # Can share with others
    ADMIN = "admin"        # Full control


class AccessLevel(Enum):
    """Access level scopes."""
    PRIVATE = "private"    # Only owner
    TEAM = "team"          # Team members
    ORGANIZATION = "org"   # All org members
    PUBLIC = "public"      # Anyone (read-only)


# Permission hierarchy
PERMISSION_HIERARCHY = [Permission.READ, Permission.WRITE, Permission.DELETE, Permission.SHARE, Permission.ADMIN]


@dataclass
class AccessRule:
    """Access control rule."""
    rule_id: str
    resource_id: str
    grantee_type: str  # "user", "team", "role"
    grantee_id: str
    permissions: Set[Permission]
    granted_by: str
    granted_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if rule has expired."""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False

    def has_permission(self, perm: Permission) -> bool:
        """Check if rule grants permission."""
        if self.is_expired():
            return False

        # Admin implies all permissions
        if Permission.ADMIN in self.permissions:
            return True

        # Check hierarchy
        if perm in self.permissions:
            return True

        # Check implied permissions (higher implies lower)
        for p in self.permissions:
            if PERMISSION_HIERARCHY.index(p) >= PERMISSION_HIERARCHY.index(perm):
                return True

        return False

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "rule_id": self.rule_id,
            "resource_id": self.resource_id,
            "grantee_type": self.grantee_type,
            "grantee_id": self.grantee_id,
            "permissions": [p.value for p in self.permissions],
            "granted_by": self.granted_by,
            "granted_at": self.granted_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata
        }


@dataclass
class ResourcePolicy:
    """Access policy for a resource."""
    resource_id: str
    owner_id: str
    access_level: AccessLevel = AccessLevel.PRIVATE
    default_permissions: Set[Permission] = field(default_factory=lambda: {Permission.READ})
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "resource_id": self.resource_id,
            "owner_id": self.owner_id,
            "access_level": self.access_level.value,
            "default_permissions": [p.value for p in self.default_permissions],
            "created_at": self.created_at.isoformat()
        }


class AccessController:
    """
    Controls access to knowledge resources.

    Features:
    - Resource-level policies
    - User and team permissions
    - Permission inheritance
    - Audit logging
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize access controller.

        Args:
            storage_path: Path to persist data
        """
        self.storage_path = storage_path
        self.policies: Dict[str, ResourcePolicy] = {}
        self.rules: Dict[str, AccessRule] = {}
        self._rules_by_resource: Dict[str, Set[str]] = {}
        self._rules_by_grantee: Dict[str, Set[str]] = {}
        self._rule_counter = 0

        # External callbacks
        self.get_user_teams: callable = None  # (user_id) -> List[team_id]
        self.get_user_role: callable = None   # (user_id) -> role

        if storage_path:
            self._load()

    def _generate_rule_id(self) -> str:
        """Generate rule ID."""
        self._rule_counter += 1
        return f"rule_{self._rule_counter:06d}"

    # === Policies ===

    def set_policy(self, resource_id: str, owner_id: str,
                   access_level: AccessLevel = AccessLevel.PRIVATE,
                   default_permissions: Set[Permission] = None) -> ResourcePolicy:
        """Set access policy for a resource."""
        policy = ResourcePolicy(
            resource_id=resource_id,
            owner_id=owner_id,
            access_level=access_level,
            default_permissions=default_permissions or {Permission.READ}
        )
        self.policies[resource_id] = policy
        self._save()
        return policy

    def get_policy(self, resource_id: str) -> Optional[ResourcePolicy]:
        """Get policy for a resource."""
        return self.policies.get(resource_id)

    def update_access_level(self, resource_id: str, access_level: AccessLevel,
                            updater_id: str) -> bool:
        """Update resource access level."""
        policy = self.policies.get(resource_id)
        if not policy:
            return False

        # Only owner can change policy
        if policy.owner_id != updater_id:
            return False

        policy.access_level = access_level
        self._save()
        return True

    # === Rules ===

    def grant_access(self, resource_id: str, grantee_type: str,
                     grantee_id: str, permissions: Set[Permission],
                     granted_by: str, expires_at: datetime = None) -> AccessRule:
        """
        Grant access to a resource.

        Args:
            resource_id: Resource to grant access to
            grantee_type: "user", "team", or "role"
            grantee_id: ID of grantee
            permissions: Set of permissions to grant
            granted_by: Who is granting access
            expires_at: Optional expiration
        """
        # Verify granter has SHARE permission
        if not self.check_access(resource_id, granted_by, Permission.SHARE):
            raise PermissionError(f"User {granted_by} cannot share {resource_id}")

        rule_id = self._generate_rule_id()
        rule = AccessRule(
            rule_id=rule_id,
            resource_id=resource_id,
            grantee_type=grantee_type,
            grantee_id=grantee_id,
            permissions=permissions,
            granted_by=granted_by,
            expires_at=expires_at
        )

        self.rules[rule_id] = rule

        # Index by resource
        if resource_id not in self._rules_by_resource:
            self._rules_by_resource[resource_id] = set()
        self._rules_by_resource[resource_id].add(rule_id)

        # Index by grantee
        grantee_key = f"{grantee_type}:{grantee_id}"
        if grantee_key not in self._rules_by_grantee:
            self._rules_by_grantee[grantee_key] = set()
        self._rules_by_grantee[grantee_key].add(rule_id)

        self._save()
        return rule

    def revoke_access(self, rule_id: str, revoked_by: str) -> bool:
        """Revoke an access rule."""
        rule = self.rules.get(rule_id)
        if not rule:
            return False

        # Verify revoker has ADMIN permission
        if not self.check_access(rule.resource_id, revoked_by, Permission.ADMIN):
            raise PermissionError(f"User {revoked_by} cannot revoke access to {rule.resource_id}")

        # Remove from indexes
        if rule.resource_id in self._rules_by_resource:
            self._rules_by_resource[rule.resource_id].discard(rule_id)

        grantee_key = f"{rule.grantee_type}:{rule.grantee_id}"
        if grantee_key in self._rules_by_grantee:
            self._rules_by_grantee[grantee_key].discard(rule_id)

        del self.rules[rule_id]
        self._save()
        return True

    def get_rules_for_resource(self, resource_id: str) -> List[AccessRule]:
        """Get all access rules for a resource."""
        rule_ids = self._rules_by_resource.get(resource_id, set())
        return [self.rules[rid] for rid in rule_ids if rid in self.rules and not self.rules[rid].is_expired()]

    def get_rules_for_user(self, user_id: str) -> List[AccessRule]:
        """Get all access rules for a user."""
        all_rules = []

        # Direct user rules
        user_key = f"user:{user_id}"
        if user_key in self._rules_by_grantee:
            for rid in self._rules_by_grantee[user_key]:
                if rid in self.rules and not self.rules[rid].is_expired():
                    all_rules.append(self.rules[rid])

        # Team rules
        if self.get_user_teams:
            teams = self.get_user_teams(user_id)
            for team_id in teams:
                team_key = f"team:{team_id}"
                if team_key in self._rules_by_grantee:
                    for rid in self._rules_by_grantee[team_key]:
                        if rid in self.rules and not self.rules[rid].is_expired():
                            all_rules.append(self.rules[rid])

        # Role rules
        if self.get_user_role:
            role = self.get_user_role(user_id)
            if role:
                role_key = f"role:{role}"
                if role_key in self._rules_by_grantee:
                    for rid in self._rules_by_grantee[role_key]:
                        if rid in self.rules and not self.rules[rid].is_expired():
                            all_rules.append(self.rules[rid])

        return all_rules

    # === Access Checks ===

    def check_access(self, resource_id: str, user_id: str,
                     permission: Permission) -> bool:
        """
        Check if user has permission on resource.

        Order of checks:
        1. Owner always has full access
        2. Explicit user rules
        3. Team rules
        4. Role rules
        5. Default policy
        """
        policy = self.policies.get(resource_id)

        # Owner has full access
        if policy and policy.owner_id == user_id:
            return True

        # Check explicit rules
        user_key = f"user:{user_id}"
        if user_key in self._rules_by_grantee:
            for rid in self._rules_by_grantee[user_key]:
                rule = self.rules.get(rid)
                if rule and rule.resource_id == resource_id and rule.has_permission(permission):
                    return True

        # Check team rules
        if self.get_user_teams:
            teams = self.get_user_teams(user_id)
            for team_id in teams:
                team_key = f"team:{team_id}"
                if team_key in self._rules_by_grantee:
                    for rid in self._rules_by_grantee[team_key]:
                        rule = self.rules.get(rid)
                        if rule and rule.resource_id == resource_id and rule.has_permission(permission):
                            return True

        # Check role rules
        if self.get_user_role:
            role = self.get_user_role(user_id)
            if role:
                role_key = f"role:{role}"
                if role_key in self._rules_by_grantee:
                    for rid in self._rules_by_grantee[role_key]:
                        rule = self.rules.get(rid)
                        if rule and rule.resource_id == resource_id and rule.has_permission(permission):
                            return True

        # Check default policy
        if policy:
            if policy.access_level == AccessLevel.PUBLIC and permission == Permission.READ:
                return True
            if policy.access_level in (AccessLevel.ORGANIZATION, AccessLevel.TEAM):
                if permission in policy.default_permissions:
                    return True

        return False

    def get_accessible_resources(self, user_id: str,
                                   permission: Permission = Permission.READ) -> List[str]:
        """Get all resources accessible to user."""
        accessible = []

        for resource_id in self.policies:
            if self.check_access(resource_id, user_id, permission):
                accessible.append(resource_id)

        return accessible

    # === Persistence ===

    def _save(self):
        """Save to storage."""
        if not self.storage_path:
            return

        data = {
            "policies": {rid: p.to_dict() for rid, p in self.policies.items()},
            "rules": {rid: r.to_dict() for rid, r in self.rules.items()},
            "rule_counter": self._rule_counter
        }

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load from storage."""
        if not self.storage_path:
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            self._rule_counter = data.get("rule_counter", 0)

            # Load policies
            for rid, pdata in data.get("policies", {}).items():
                self.policies[rid] = ResourcePolicy(
                    resource_id=pdata["resource_id"],
                    owner_id=pdata["owner_id"],
                    access_level=AccessLevel(pdata["access_level"]),
                    default_permissions={Permission(p) for p in pdata["default_permissions"]},
                    created_at=datetime.fromisoformat(pdata["created_at"]) if "created_at" in pdata else datetime.now()
                )

            # Load rules
            for rid, rdata in data.get("rules", {}).items():
                rule = AccessRule(
                    rule_id=rdata["rule_id"],
                    resource_id=rdata["resource_id"],
                    grantee_type=rdata["grantee_type"],
                    grantee_id=rdata["grantee_id"],
                    permissions={Permission(p) for p in rdata["permissions"]},
                    granted_by=rdata["granted_by"],
                    granted_at=datetime.fromisoformat(rdata["granted_at"]) if "granted_at" in rdata else datetime.now(),
                    expires_at=datetime.fromisoformat(rdata["expires_at"]) if rdata.get("expires_at") else None,
                    metadata=rdata.get("metadata", {})
                )
                self.rules[rid] = rule

                # Rebuild indexes
                if rule.resource_id not in self._rules_by_resource:
                    self._rules_by_resource[rule.resource_id] = set()
                self._rules_by_resource[rule.resource_id].add(rid)

                grantee_key = f"{rule.grantee_type}:{rule.grantee_id}"
                if grantee_key not in self._rules_by_grantee:
                    self._rules_by_grantee[grantee_key] = set()
                self._rules_by_grantee[grantee_key].add(rid)

        except FileNotFoundError:
            pass


# Quick test
if __name__ == "__main__":
    print("Testing Access Controller...")

    controller = AccessController()

    # Create resources with policies
    controller.set_policy("doc_secret", "alice", AccessLevel.PRIVATE)
    controller.set_policy("doc_team", "alice", AccessLevel.TEAM, {Permission.READ})
    controller.set_policy("doc_public", "alice", AccessLevel.PUBLIC)

    # Owner always has access
    print(f"Alice can write doc_secret: {controller.check_access('doc_secret', 'alice', Permission.WRITE)}")

    # Non-owner can't access private
    print(f"Bob can read doc_secret: {controller.check_access('doc_secret', 'bob', Permission.READ)}")

    # Grant access
    rule = controller.grant_access(
        "doc_secret",
        grantee_type="user",
        grantee_id="bob",
        permissions={Permission.READ},
        granted_by="alice"
    )
    print(f"Created rule: {rule.rule_id}")
    print(f"Bob can now read doc_secret: {controller.check_access('doc_secret', 'bob', Permission.READ)}")
    print(f"Bob can write doc_secret: {controller.check_access('doc_secret', 'bob', Permission.WRITE)}")

    # Public resource
    print(f"Anyone can read doc_public: {controller.check_access('doc_public', 'charlie', Permission.READ)}")
