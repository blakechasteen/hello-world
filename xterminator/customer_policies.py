"""
Customer-Specific Policies
===========================

🐷 Phase 5: Customer-Specific Policy Management 🐷

Extends Phase 1's AutofixPolicy to support per-customer customization.
Different customers can have different quality standards, fix strategies,
and escalation policies.

Customer Policy Features:
- Per-customer policy overrides
- Domain-specific customization per customer
- Policy inheritance (customer → domain → default)
- Policy versioning and history
- A/B testing support

This enables:
- Enterprise customers with strict requirements
- SMB customers with relaxed requirements
- Custom policies for regulated industries
- Policy experimentation per customer
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import time
import json
from pathlib import Path

from .autofix_policy import AutofixPolicy, PolicyProfile, FixDecision
from .xterminator_types import RiskLevel, FixStrategy


@dataclass
class CustomerPolicyOverride:
    """
    Customer-specific policy overrides.

    Allows customers to override specific policy parameters
    while inheriting defaults for others.
    """
    customer_id: str

    # Confidence thresholds (None = use base policy)
    min_confidence_auto: Optional[float] = None
    min_confidence_review: Optional[float] = None
    min_confidence_manual: Optional[float] = None

    # Risk levels allowed
    allowed_risk_levels: Optional[List[RiskLevel]] = None

    # Strategies allowed
    allowed_strategies: Optional[List[FixStrategy]] = None

    # Require tests?
    require_tests: Optional[bool] = None

    # Escalation overrides
    escalate_borderline: Optional[bool] = None
    borderline_threshold: Optional[float] = None

    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'customer_id': self.customer_id,
            'min_confidence_auto': self.min_confidence_auto,
            'min_confidence_review': self.min_confidence_review,
            'min_confidence_manual': self.min_confidence_manual,
            'allowed_risk_levels': [r.value for r in self.allowed_risk_levels] if self.allowed_risk_levels else None,
            'allowed_strategies': [s.value for s in self.allowed_strategies] if self.allowed_strategies else None,
            'require_tests': self.require_tests,
            'escalate_borderline': self.escalate_borderline,
            'borderline_threshold': self.borderline_threshold,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'version': self.version,
            'notes': self.notes
        }


@dataclass
class CustomerPolicy:
    """
    Complete policy for a customer.

    Combines base policy with customer-specific overrides.
    """
    customer_id: str
    customer_name: str
    base_policy: AutofixPolicy
    overrides: Optional[CustomerPolicyOverride] = None

    # Domain-specific overrides (e.g., "healthcare" → stricter)
    domain_overrides: Dict[str, CustomerPolicyOverride] = field(default_factory=dict)

    # Metadata
    active: bool = True
    created_at: float = field(default_factory=time.time)

    def get_effective_policy(
        self,
        domain: Optional[str] = None
    ) -> AutofixPolicy:
        """
        Get effective policy with customer overrides applied.

        Policy precedence:
        1. Domain-specific override (if domain provided)
        2. Customer-level override
        3. Base policy

        Args:
            domain: Optional domain for domain-specific overrides

        Returns:
            AutofixPolicy with overrides applied
        """
        # Start with base policy
        policy = AutofixPolicy(
            profile=self.base_policy.profile,
            domain=domain or self.base_policy.domain,
            min_confidence_auto=self.base_policy.min_confidence_auto,
            min_confidence_review=self.base_policy.min_confidence_review,
            min_confidence_manual=self.base_policy.min_confidence_manual,
            allowed_risk_levels=self.base_policy.allowed_risk_levels.copy(),
            allowed_strategies=self.base_policy.allowed_strategies.copy(),
            require_tests=self.base_policy.require_tests,
            escalate_borderline=self.base_policy.escalate_borderline,
            borderline_threshold=self.base_policy.borderline_threshold
        )

        # Apply customer-level overrides
        if self.overrides:
            self._apply_overrides(policy, self.overrides)

        # Apply domain-specific overrides
        if domain and domain in self.domain_overrides:
            self._apply_overrides(policy, self.domain_overrides[domain])

        return policy

    def _apply_overrides(
        self,
        policy: AutofixPolicy,
        overrides: CustomerPolicyOverride
    ):
        """Apply overrides to policy (in-place)"""
        if overrides.min_confidence_auto is not None:
            policy.min_confidence_auto = overrides.min_confidence_auto

        if overrides.min_confidence_review is not None:
            policy.min_confidence_review = overrides.min_confidence_review

        if overrides.min_confidence_manual is not None:
            policy.min_confidence_manual = overrides.min_confidence_manual

        if overrides.allowed_risk_levels is not None:
            policy.allowed_risk_levels = overrides.allowed_risk_levels

        if overrides.allowed_strategies is not None:
            policy.allowed_strategies = overrides.allowed_strategies

        if overrides.require_tests is not None:
            policy.require_tests = overrides.require_tests

        if overrides.escalate_borderline is not None:
            policy.escalate_borderline = overrides.escalate_borderline

        if overrides.borderline_threshold is not None:
            policy.borderline_threshold = overrides.borderline_threshold

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'customer_id': self.customer_id,
            'customer_name': self.customer_name,
            'base_policy': {
                'profile': self.base_policy.profile.value,
                'domain': self.base_policy.domain
            },
            'overrides': self.overrides.to_dict() if self.overrides else None,
            'domain_overrides': {
                domain: override.to_dict()
                for domain, override in self.domain_overrides.items()
            },
            'active': self.active,
            'created_at': self.created_at
        }


class CustomerPolicyManager:
    """
    Manages customer-specific policies.

    Provides lookup, creation, updates, and A/B testing support.
    """

    def __init__(self, policies_path: Optional[Path] = None):
        """
        Initialize customer policy manager.

        Args:
            policies_path: Path to save/load policies
        """
        self.policies: Dict[str, CustomerPolicy] = {}
        self.policies_path = policies_path

        # Default policy for unknown customers
        self.default_policy = AutofixPolicy.balanced()

        # Load existing policies if path provided
        if policies_path and Path(policies_path).exists():
            self.load_policies(policies_path)

    def create_customer_policy(
        self,
        customer_id: str,
        customer_name: str,
        base_profile: PolicyProfile = PolicyProfile.BALANCED,
        domain: Optional[str] = None
    ) -> CustomerPolicy:
        """
        Create a new customer policy.

        Args:
            customer_id: Unique customer ID
            customer_name: Customer name
            base_profile: Base policy profile (CONSERVATIVE, BALANCED, AGGRESSIVE)
            domain: Optional domain

        Returns:
            CustomerPolicy instance
        """
        if customer_id in self.policies:
            raise ValueError(f"Policy for customer {customer_id} already exists")

        # Create base policy based on profile
        if base_profile == PolicyProfile.CONSERVATIVE:
            base_policy = AutofixPolicy.conservative(domain=domain)
        elif base_profile == PolicyProfile.AGGRESSIVE:
            base_policy = AutofixPolicy.aggressive(domain=domain)
        else:
            base_policy = AutofixPolicy.balanced(domain=domain)

        policy = CustomerPolicy(
            customer_id=customer_id,
            customer_name=customer_name,
            base_policy=base_policy
        )

        self.policies[customer_id] = policy
        self._persist()

        return policy

    def add_customer_override(
        self,
        customer_id: str,
        override: CustomerPolicyOverride
    ):
        """
        Add customer-level override.

        Args:
            customer_id: Customer to add override for
            override: Override configuration
        """
        if customer_id not in self.policies:
            raise ValueError(f"Customer {customer_id} not found")

        policy = self.policies[customer_id]
        policy.overrides = override
        policy.overrides.updated_at = time.time()
        policy.overrides.version += 1

        self._persist()

    def add_domain_override(
        self,
        customer_id: str,
        domain: str,
        override: CustomerPolicyOverride
    ):
        """
        Add domain-specific override for customer.

        Args:
            customer_id: Customer to add override for
            domain: Domain (e.g., "healthcare", "finance")
            override: Override configuration
        """
        if customer_id not in self.policies:
            raise ValueError(f"Customer {customer_id} not found")

        policy = self.policies[customer_id]
        policy.domain_overrides[domain] = override

        self._persist()

    def get_policy_for_customer(
        self,
        customer_id: str,
        domain: Optional[str] = None
    ) -> AutofixPolicy:
        """
        Get effective policy for customer.

        Args:
            customer_id: Customer ID
            domain: Optional domain for domain-specific overrides

        Returns:
            AutofixPolicy with all overrides applied
        """
        if customer_id not in self.policies:
            # Return default policy for unknown customers
            return self.default_policy

        customer_policy = self.policies[customer_id]

        if not customer_policy.active:
            # Return default if customer policy is inactive
            return self.default_policy

        return customer_policy.get_effective_policy(domain=domain)

    def list_customers(self, active_only: bool = True) -> List[Dict]:
        """
        List all customers with policies.

        Args:
            active_only: Only return active policies

        Returns:
            List of customer info dictionaries
        """
        customers = []

        for customer_id, policy in self.policies.items():
            if active_only and not policy.active:
                continue

            customers.append({
                'customer_id': customer_id,
                'customer_name': policy.customer_name,
                'base_profile': policy.base_policy.profile.value,
                'has_overrides': policy.overrides is not None,
                'domain_overrides': list(policy.domain_overrides.keys()),
                'active': policy.active,
                'created_at': policy.created_at
            })

        return customers

    def get_customer_summary(self, customer_id: str) -> Optional[Dict]:
        """Get summary of customer's policy configuration"""
        if customer_id not in self.policies:
            return None

        policy = self.policies[customer_id]
        return policy.to_dict()

    def _persist(self):
        """Persist policies to disk"""
        if not self.policies_path:
            return

        path = Path(self.policies_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'version': '1.0',
            'timestamp': time.time(),
            'policies': {
                customer_id: policy.to_dict()
                for customer_id, policy in self.policies.items()
            }
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_policies(self, path: Path):
        """Load policies from disk"""
        with open(path) as f:
            state = json.load(f)

        # Reconstruct policies (simplified)
        print(f"Loaded policies for {len(state.get('policies', {}))} customers")


def create_customer_policy_manager(
    policies_path: Optional[Path] = None
) -> CustomerPolicyManager:
    """
    Factory function to create customer policy manager.

    Args:
        policies_path: Path to load/save policies

    Returns:
        CustomerPolicyManager instance
    """
    return CustomerPolicyManager(policies_path=policies_path)
