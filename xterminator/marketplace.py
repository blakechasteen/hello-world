"""
Marketplace Quality Enforcement
================================

🐷 Phase 5: Marketplace Quality Enforcement 🐷

Enables third-party departments to integrate with xTerminator while
maintaining quality standards through certification and continuous monitoring.

Marketplace Features:
- Department registry with metadata
- Quality certification system (Bronze, Silver, Gold, Platinum)
- Continuous monitoring and quality tracking
- Automatic revocation on quality degradation
- Customer trust through verified quality

This enables:
- Third-party department marketplace (Moonshot Idea #6)
- Trust through quality enforcement (Moonshot Idea #4)
- Revenue growth through ecosystem expansion
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
from pathlib import Path
import time
import json
import uuid


class QualityTier(Enum):
    """Quality tier for marketplace departments"""
    UNVERIFIED = "unverified"  # Not yet certified
    BRONZE = "bronze"          # Basic quality (60-70%)
    SILVER = "silver"          # Good quality (70-85%)
    GOLD = "gold"              # Excellent quality (85-95%)
    PLATINUM = "platinum"      # Outstanding quality (95%+)
    REVOKED = "revoked"        # Certification revoked


@dataclass
class QualityCertification:
    """
    Quality certification for a department.

    Tracks certification level, expiration, and renewal history.
    """
    department_name: str
    tier: QualityTier
    certified_at: float  # Unix timestamp
    expires_at: float    # Unix timestamp

    # Certification criteria met
    success_rate: float  # 0.0-1.0
    avg_confidence: float  # 0.0-1.0
    total_fixes: int

    # Certification metadata
    certification_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    previous_tier: Optional[QualityTier] = None
    renewal_count: int = 0

    # Revocation details (if revoked)
    revoked: bool = False
    revoked_at: Optional[float] = None
    revocation_reason: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Check if certification is expired"""
        return time.time() > self.expires_at

    @property
    def days_until_expiration(self) -> int:
        """Days until certification expires"""
        seconds_left = self.expires_at - time.time()
        return max(0, int(seconds_left / 86400))

    @property
    def is_valid(self) -> bool:
        """Check if certification is valid (not expired, not revoked)"""
        return not self.is_expired and not self.revoked

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'department_name': self.department_name,
            'tier': self.tier.value,
            'certification_id': self.certification_id,
            'certified_at': self.certified_at,
            'expires_at': self.expires_at,
            'days_until_expiration': self.days_until_expiration,
            'success_rate': self.success_rate,
            'avg_confidence': self.avg_confidence,
            'total_fixes': self.total_fixes,
            'previous_tier': self.previous_tier.value if self.previous_tier else None,
            'renewal_count': self.renewal_count,
            'revoked': self.revoked,
            'revoked_at': self.revoked_at,
            'revocation_reason': self.revocation_reason,
            'is_valid': self.is_valid
        }


@dataclass
class MarketplaceDepartment:
    """
    A department registered in the marketplace.

    Contains metadata, certification, and quality tracking.
    """
    department_name: str
    provider: str  # Who provides this department
    description: str
    version: str

    # Certification
    certification: Optional[QualityCertification] = None

    # Quality tracking
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0

    # Reputation
    customer_ratings: List[float] = field(default_factory=list)  # 1.0-5.0
    quality_scores: List[float] = field(default_factory=list)     # 0.0-1.0

    # Metadata
    registered_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    tags: Set[str] = field(default_factory=set)

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def avg_rating(self) -> float:
        """Calculate average customer rating"""
        if not self.customer_ratings:
            return 0.0
        return sum(self.customer_ratings) / len(self.customer_ratings)

    @property
    def avg_quality(self) -> float:
        """Calculate average quality score"""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores) / len(self.quality_scores)

    @property
    def is_certified(self) -> bool:
        """Check if department is certified"""
        return self.certification is not None and self.certification.is_valid

    @property
    def certification_tier(self) -> Optional[QualityTier]:
        """Get certification tier"""
        if self.certification and self.certification.is_valid:
            return self.certification.tier
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'department_name': self.department_name,
            'provider': self.provider,
            'description': self.description,
            'version': self.version,
            'certification': self.certification.to_dict() if self.certification else None,
            'total_requests': self.total_requests,
            'success_rate': self.success_rate,
            'avg_response_time_ms': self.avg_response_time_ms,
            'avg_rating': self.avg_rating,
            'avg_quality': self.avg_quality,
            'is_certified': self.is_certified,
            'certification_tier': self.certification_tier.value if self.certification_tier else None,
            'registered_at': self.registered_at,
            'last_used': self.last_used,
            'tags': list(self.tags)
        }


class MarketplaceRegistry:
    """
    Registry of marketplace departments with quality enforcement.

    Maintains catalog of third-party departments, tracks quality,
    issues certifications, and enforces quality standards.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize marketplace registry.

        Args:
            registry_path: Path to save/load registry state
        """
        self.departments: Dict[str, MarketplaceDepartment] = {}
        self.registry_path = registry_path

        # Load existing registry if path provided
        if registry_path and Path(registry_path).exists():
            self.load_registry(registry_path)

    def register_department(
        self,
        department_name: str,
        provider: str,
        description: str,
        version: str = "1.0.0",
        tags: Optional[Set[str]] = None
    ) -> MarketplaceDepartment:
        """
        Register a new department in marketplace.

        Args:
            department_name: Unique department name
            provider: Provider/vendor name
            description: Department description
            version: Version string
            tags: Optional tags (e.g., "qa", "security", "performance")

        Returns:
            MarketplaceDepartment instance
        """
        if department_name in self.departments:
            raise ValueError(f"Department {department_name} already registered")

        dept = MarketplaceDepartment(
            department_name=department_name,
            provider=provider,
            description=description,
            version=version,
            tags=tags or set()
        )

        self.departments[department_name] = dept
        self._persist()

        return dept

    def certify_department(
        self,
        department_name: str,
        success_rate: float,
        avg_confidence: float,
        total_fixes: int,
        validity_days: int = 90
    ) -> QualityCertification:
        """
        Certify a department based on quality metrics.

        Args:
            department_name: Department to certify
            success_rate: Success rate (0.0-1.0)
            avg_confidence: Average confidence (0.0-1.0)
            total_fixes: Total fixes performed
            validity_days: Certification validity in days

        Returns:
            QualityCertification

        Raises:
            ValueError: If department not found or doesn't meet minimum criteria
        """
        if department_name not in self.departments:
            raise ValueError(f"Department {department_name} not registered")

        dept = self.departments[department_name]

        # Determine tier based on success rate
        tier = self._determine_tier(success_rate, avg_confidence, total_fixes)

        # Minimum criteria check
        if tier == QualityTier.UNVERIFIED:
            raise ValueError(f"Department does not meet minimum certification criteria")

        # Create certification
        now = time.time()
        cert = QualityCertification(
            department_name=department_name,
            tier=tier,
            certified_at=now,
            expires_at=now + (validity_days * 86400),
            success_rate=success_rate,
            avg_confidence=avg_confidence,
            total_fixes=total_fixes,
            previous_tier=dept.certification.tier if dept.certification else None,
            renewal_count=dept.certification.renewal_count + 1 if dept.certification else 0
        )

        dept.certification = cert
        self._persist()

        return cert

    def _determine_tier(
        self,
        success_rate: float,
        avg_confidence: float,
        total_fixes: int
    ) -> QualityTier:
        """
        Determine quality tier based on metrics.

        Tiers:
        - PLATINUM: 95%+ success, 90%+ confidence, 100+ fixes
        - GOLD: 85-95% success, 80%+ confidence, 50+ fixes
        - SILVER: 70-85% success, 70%+ confidence, 25+ fixes
        - BRONZE: 60-70% success, 60%+ confidence, 10+ fixes
        - UNVERIFIED: Below minimums
        """
        # Minimum fix count required
        if total_fixes < 10:
            return QualityTier.UNVERIFIED

        # Platinum: Outstanding
        if success_rate >= 0.95 and avg_confidence >= 0.90 and total_fixes >= 100:
            return QualityTier.PLATINUM

        # Gold: Excellent
        if success_rate >= 0.85 and avg_confidence >= 0.80 and total_fixes >= 50:
            return QualityTier.GOLD

        # Silver: Good
        if success_rate >= 0.70 and avg_confidence >= 0.70 and total_fixes >= 25:
            return QualityTier.SILVER

        # Bronze: Basic
        if success_rate >= 0.60 and avg_confidence >= 0.60 and total_fixes >= 10:
            return QualityTier.BRONZE

        # Below minimums
        return QualityTier.UNVERIFIED

    def revoke_certification(
        self,
        department_name: str,
        reason: str
    ):
        """
        Revoke a department's certification.

        Args:
            department_name: Department to revoke
            reason: Revocation reason
        """
        if department_name not in self.departments:
            raise ValueError(f"Department {department_name} not registered")

        dept = self.departments[department_name]

        if not dept.certification:
            raise ValueError(f"Department {department_name} is not certified")

        dept.certification.revoked = True
        dept.certification.revoked_at = time.time()
        dept.certification.revocation_reason = reason
        dept.certification.tier = QualityTier.REVOKED

        self._persist()

    def check_quality_degradation(
        self,
        department_name: str,
        current_success_rate: float,
        threshold: float = 0.10
    ) -> bool:
        """
        Check if department quality has degraded significantly.

        Args:
            department_name: Department to check
            current_success_rate: Current success rate
            threshold: Degradation threshold (default 10%)

        Returns:
            True if degraded beyond threshold
        """
        if department_name not in self.departments:
            return False

        dept = self.departments[department_name]

        if not dept.certification or not dept.certification.is_valid:
            return False

        # Check if current success rate is significantly lower
        certified_rate = dept.certification.success_rate
        degradation = certified_rate - current_success_rate

        return degradation > threshold

    def get_certified_departments(
        self,
        min_tier: Optional[QualityTier] = None
    ) -> List[MarketplaceDepartment]:
        """
        Get list of certified departments.

        Args:
            min_tier: Minimum tier to include (optional)

        Returns:
            List of certified departments
        """
        certified = [
            dept for dept in self.departments.values()
            if dept.is_certified
        ]

        if min_tier:
            tier_order = [
                QualityTier.BRONZE,
                QualityTier.SILVER,
                QualityTier.GOLD,
                QualityTier.PLATINUM
            ]
            min_index = tier_order.index(min_tier)
            certified = [
                dept for dept in certified
                if tier_order.index(dept.certification_tier) >= min_index
            ]

        return certified

    def get_department_info(self, department_name: str) -> Optional[Dict]:
        """Get department information"""
        if department_name not in self.departments:
            return None

        return self.departments[department_name].to_dict()

    def get_marketplace_stats(self) -> Dict:
        """Get marketplace statistics"""
        total_depts = len(self.departments)
        certified_depts = len([d for d in self.departments.values() if d.is_certified])

        tier_counts = {
            'bronze': 0,
            'silver': 0,
            'gold': 0,
            'platinum': 0
        }

        for dept in self.departments.values():
            if dept.certification and dept.certification.is_valid:
                tier = dept.certification.tier
                if tier in [QualityTier.BRONZE, QualityTier.SILVER, QualityTier.GOLD, QualityTier.PLATINUM]:
                    tier_counts[tier.value] += 1

        return {
            'total_departments': total_depts,
            'certified_departments': certified_depts,
            'certification_rate': certified_depts / total_depts if total_depts > 0 else 0.0,
            'tier_distribution': tier_counts,
            'total_requests': sum(d.total_requests for d in self.departments.values()),
            'avg_success_rate': sum(d.success_rate for d in self.departments.values()) / total_depts if total_depts > 0 else 0.0
        }

    def _persist(self):
        """Persist registry to disk"""
        if not self.registry_path:
            return

        path = Path(self.registry_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'version': '1.0',
            'timestamp': time.time(),
            'departments': {
                name: dept.to_dict()
                for name, dept in self.departments.items()
            }
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_registry(self, path: Path):
        """Load registry from disk"""
        with open(path) as f:
            state = json.load(f)

        # Reconstruct departments (simplified - would need full reconstruction)
        # For demo purposes, just track that we loaded
        print(f"Loaded registry with {len(state.get('departments', {}))} departments")


def create_marketplace_registry(
    registry_path: Optional[Path] = None
) -> MarketplaceRegistry:
    """
    Factory function to create marketplace registry.

    Args:
        registry_path: Path to load/save registry state

    Returns:
        MarketplaceRegistry instance
    """
    return MarketplaceRegistry(registry_path=registry_path)
