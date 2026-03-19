"""
Dark Trace Research: Adversarial Catalog
=========================================

Persistence and management of discovered adversarial vulnerabilities.
Enables tracking, comparison, and regression detection across model versions.

Key Capabilities:
- Vulnerability persistence (JSON, SQLite)
- Cross-version comparison
- Regression detection
- Vulnerability clustering
- Export for reporting

Created: 2025-12-28
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# =============================================================================
# Data Structures
# =============================================================================


class VulnerabilityStatus(Enum):
    """Status of a cataloged vulnerability."""
    ACTIVE = "active"           # Currently exploitable
    MITIGATED = "mitigated"     # Fixed in current version
    REGRESSED = "regressed"     # Re-appeared after fix
    IGNORED = "ignored"         # Accepted risk
    INVESTIGATING = "investigating"  # Under review


@dataclass
class CatalogedVulnerability:
    """
    A vulnerability entry in the catalog.

    Stores complete information about a discovered vulnerability
    for tracking and regression detection.
    """
    # Identity
    id: str                          # Unique identifier (hash of key fields)
    feature_idx: int                 # SAE feature index
    model_version: str               # Model version when discovered
    discovered_at: str               # ISO timestamp

    # Vulnerability details
    vulnerability_score: float       # 0.0 to 1.0
    vulnerability_level: str         # ROBUST/LOW/MEDIUM/HIGH/CRITICAL
    attack_method: str               # Method that found it (FGSM/PGD/GENETIC/etc)
    epsilon: float                   # Perturbation bound used

    # Evidence
    best_perturbation: list[float] | None = None  # Best adversarial perturbation
    activation_before: float = 0.0
    activation_after: float = 0.0
    activation_change: float = 0.0

    # Tracking
    status: VulnerabilityStatus = VulnerabilityStatus.ACTIVE
    last_verified: str | None = None  # ISO timestamp
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    # History
    version_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CatalogedVulnerability":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("status"), str):
            data["status"] = VulnerabilityStatus(data["status"])
        return cls(**data)

    @staticmethod
    def generate_id(
        feature_idx: int,
        model_version: str,
        attack_method: str
    ) -> str:
        """Generate unique ID for vulnerability."""
        key = f"{feature_idx}:{model_version}:{attack_method}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class CatalogSummary:
    """Summary statistics for the catalog."""
    total_vulnerabilities: int
    active_count: int
    mitigated_count: int
    regressed_count: int
    by_level: dict[str, int]
    by_method: dict[str, int]
    most_vulnerable_features: list[int]
    avg_vulnerability_score: float
    last_updated: str


@dataclass
class RegressionReport:
    """Report on vulnerability regressions between versions."""
    from_version: str
    to_version: str
    new_vulnerabilities: list[CatalogedVulnerability]
    fixed_vulnerabilities: list[CatalogedVulnerability]
    regressed_vulnerabilities: list[CatalogedVulnerability]
    unchanged_vulnerabilities: list[CatalogedVulnerability]
    summary: dict[str, Any]


# =============================================================================
# Catalog Manager
# =============================================================================


class AdversarialCatalog:
    """
    Manager for adversarial vulnerability catalog.

    Provides persistence, querying, and comparison capabilities.

    Usage:
        catalog = AdversarialCatalog("./vulnerabilities")

        # Add vulnerability
        vuln = catalog.add_vulnerability(
            feature_idx=42,
            model_version="v1.0",
            vulnerability_score=0.75,
            vulnerability_level="HIGH",
            attack_method="FGSM",
            epsilon=0.1
        )

        # Query
        high_vulns = catalog.get_by_level("HIGH")

        # Compare versions
        report = catalog.compare_versions("v1.0", "v1.1")
    """

    def __init__(self, storage_path: str | None = None):
        """
        Initialize catalog.

        Args:
            storage_path: Path for persistence (None for in-memory only)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._vulnerabilities: dict[str, CatalogedVulnerability] = {}

        # Load existing if path provided
        if self.storage_path and self.storage_path.exists():
            self._load()

    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------

    def add_vulnerability(
        self,
        feature_idx: int,
        model_version: str,
        vulnerability_score: float,
        vulnerability_level: str,
        attack_method: str,
        epsilon: float,
        best_perturbation: list[float] | None = None,
        activation_before: float = 0.0,
        activation_after: float = 0.0,
        notes: str = "",
        tags: list[str] | None = None
    ) -> CatalogedVulnerability:
        """
        Add a new vulnerability to the catalog.

        Returns:
            Created CatalogedVulnerability
        """
        vuln_id = CatalogedVulnerability.generate_id(
            feature_idx, model_version, attack_method
        )

        # Check if exists
        if vuln_id in self._vulnerabilities:
            # Update existing
            existing = self._vulnerabilities[vuln_id]
            existing.vulnerability_score = vulnerability_score
            existing.vulnerability_level = vulnerability_level
            existing.last_verified = datetime.now().isoformat()
            existing.version_history.append({
                "timestamp": datetime.now().isoformat(),
                "score": vulnerability_score,
                "level": vulnerability_level
            })
            self._save()
            return existing

        # Create new
        vuln = CatalogedVulnerability(
            id=vuln_id,
            feature_idx=feature_idx,
            model_version=model_version,
            discovered_at=datetime.now().isoformat(),
            vulnerability_score=vulnerability_score,
            vulnerability_level=vulnerability_level,
            attack_method=attack_method,
            epsilon=epsilon,
            best_perturbation=best_perturbation,
            activation_before=activation_before,
            activation_after=activation_after,
            activation_change=activation_after - activation_before,
            notes=notes,
            tags=tags or []
        )

        self._vulnerabilities[vuln_id] = vuln
        self._save()
        return vuln

    def get_vulnerability(self, vuln_id: str) -> CatalogedVulnerability | None:
        """Get vulnerability by ID."""
        return self._vulnerabilities.get(vuln_id)

    def update_status(
        self,
        vuln_id: str,
        status: VulnerabilityStatus,
        notes: str = ""
    ) -> bool:
        """Update vulnerability status."""
        if vuln_id not in self._vulnerabilities:
            return False

        vuln = self._vulnerabilities[vuln_id]
        old_status = vuln.status
        vuln.status = status
        vuln.last_verified = datetime.now().isoformat()

        if notes:
            vuln.notes = notes

        vuln.version_history.append({
            "timestamp": datetime.now().isoformat(),
            "status_change": f"{old_status.value} -> {status.value}",
            "notes": notes
        })

        self._save()
        return True

    def remove_vulnerability(self, vuln_id: str) -> bool:
        """Remove vulnerability from catalog."""
        if vuln_id in self._vulnerabilities:
            del self._vulnerabilities[vuln_id]
            self._save()
            return True
        return False

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_all(self) -> list[CatalogedVulnerability]:
        """Get all vulnerabilities."""
        return list(self._vulnerabilities.values())

    def get_by_feature(self, feature_idx: int) -> list[CatalogedVulnerability]:
        """Get all vulnerabilities for a feature."""
        return [v for v in self._vulnerabilities.values()
                if v.feature_idx == feature_idx]

    def get_by_level(self, level: str) -> list[CatalogedVulnerability]:
        """Get vulnerabilities by level."""
        return [v for v in self._vulnerabilities.values()
                if v.vulnerability_level == level]

    def get_by_status(self, status: VulnerabilityStatus) -> list[CatalogedVulnerability]:
        """Get vulnerabilities by status."""
        return [v for v in self._vulnerabilities.values()
                if v.status == status]

    def get_by_version(self, model_version: str) -> list[CatalogedVulnerability]:
        """Get vulnerabilities for a specific model version."""
        return [v for v in self._vulnerabilities.values()
                if v.model_version == model_version]

    def get_active_critical(self) -> list[CatalogedVulnerability]:
        """Get active CRITICAL and HIGH vulnerabilities."""
        return [v for v in self._vulnerabilities.values()
                if v.status == VulnerabilityStatus.ACTIVE
                and v.vulnerability_level in ("CRITICAL", "HIGH")]

    def search(
        self,
        feature_idx: int | None = None,
        model_version: str | None = None,
        min_score: float | None = None,
        status: VulnerabilityStatus | None = None,
        tags: list[str] | None = None
    ) -> list[CatalogedVulnerability]:
        """
        Search vulnerabilities with filters.

        All filters are ANDed together.
        """
        results = list(self._vulnerabilities.values())

        if feature_idx is not None:
            results = [v for v in results if v.feature_idx == feature_idx]

        if model_version is not None:
            results = [v for v in results if v.model_version == model_version]

        if min_score is not None:
            results = [v for v in results if v.vulnerability_score >= min_score]

        if status is not None:
            results = [v for v in results if v.status == status]

        if tags:
            tag_set = set(tags)
            results = [v for v in results if tag_set.intersection(set(v.tags))]

        return results

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------

    def get_summary(self) -> CatalogSummary:
        """Get summary statistics for the catalog."""
        vulns = list(self._vulnerabilities.values())

        if not vulns:
            return CatalogSummary(
                total_vulnerabilities=0,
                active_count=0,
                mitigated_count=0,
                regressed_count=0,
                by_level={},
                by_method={},
                most_vulnerable_features=[],
                avg_vulnerability_score=0.0,
                last_updated=datetime.now().isoformat()
            )

        # Count by status
        active = sum(1 for v in vulns if v.status == VulnerabilityStatus.ACTIVE)
        mitigated = sum(1 for v in vulns if v.status == VulnerabilityStatus.MITIGATED)
        regressed = sum(1 for v in vulns if v.status == VulnerabilityStatus.REGRESSED)

        # Count by level
        by_level: dict[str, int] = {}
        for v in vulns:
            by_level[v.vulnerability_level] = by_level.get(v.vulnerability_level, 0) + 1

        # Count by method
        by_method: dict[str, int] = {}
        for v in vulns:
            by_method[v.attack_method] = by_method.get(v.attack_method, 0) + 1

        # Most vulnerable features
        feature_scores: dict[int, float] = {}
        for v in vulns:
            if v.feature_idx not in feature_scores:
                feature_scores[v.feature_idx] = 0.0
            feature_scores[v.feature_idx] = max(
                feature_scores[v.feature_idx],
                v.vulnerability_score
            )
        most_vulnerable = sorted(
            feature_scores.keys(),
            key=lambda f: feature_scores[f],
            reverse=True
        )[:10]

        # Average score
        avg_score = sum(v.vulnerability_score for v in vulns) / len(vulns)

        return CatalogSummary(
            total_vulnerabilities=len(vulns),
            active_count=active,
            mitigated_count=mitigated,
            regressed_count=regressed,
            by_level=by_level,
            by_method=by_method,
            most_vulnerable_features=most_vulnerable,
            avg_vulnerability_score=avg_score,
            last_updated=datetime.now().isoformat()
        )

    def compare_versions(
        self,
        version1: str,
        version2: str
    ) -> RegressionReport:
        """
        Compare vulnerabilities between two model versions.

        Identifies new, fixed, and regressed vulnerabilities.
        """
        v1_vulns = {v.feature_idx: v for v in self.get_by_version(version1)}
        v2_vulns = {v.feature_idx: v for v in self.get_by_version(version2)}

        v1_features = set(v1_vulns.keys())
        v2_features = set(v2_vulns.keys())

        # New in v2
        new_features = v2_features - v1_features
        new_vulns = [v2_vulns[f] for f in new_features]

        # Fixed (in v1 but not v2, or score dropped significantly)
        fixed_features = v1_features - v2_features
        fixed_vulns = [v1_vulns[f] for f in fixed_features]

        # Check for regressions (score increased significantly)
        regressed_vulns = []
        unchanged_vulns = []

        for feature in v1_features.intersection(v2_features):
            v1_score = v1_vulns[feature].vulnerability_score
            v2_score = v2_vulns[feature].vulnerability_score

            if v2_score > v1_score + 0.1:  # 10% increase = regression
                regressed_vulns.append(v2_vulns[feature])
            else:
                unchanged_vulns.append(v2_vulns[feature])

        return RegressionReport(
            from_version=version1,
            to_version=version2,
            new_vulnerabilities=new_vulns,
            fixed_vulnerabilities=fixed_vulns,
            regressed_vulnerabilities=regressed_vulns,
            unchanged_vulnerabilities=unchanged_vulns,
            summary={
                "new_count": len(new_vulns),
                "fixed_count": len(fixed_vulns),
                "regressed_count": len(regressed_vulns),
                "unchanged_count": len(unchanged_vulns),
                "net_change": len(new_vulns) - len(fixed_vulns),
                "regression_detected": len(regressed_vulns) > 0
            }
        )

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _save(self):
        """Save catalog to disk."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        catalog_file = self.storage_path / "catalog.json"

        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "vulnerabilities": [v.to_dict() for v in self._vulnerabilities.values()]
        }

        with open(catalog_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load catalog from disk."""
        if not self.storage_path:
            return

        catalog_file = self.storage_path / "catalog.json"
        if not catalog_file.exists():
            return

        with open(catalog_file) as f:
            data = json.load(f)

        for vuln_data in data.get("vulnerabilities", []):
            vuln = CatalogedVulnerability.from_dict(vuln_data)
            self._vulnerabilities[vuln.id] = vuln

    def export_json(self, filepath: str):
        """Export catalog to JSON file."""
        data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "summary": asdict(self.get_summary()),
            "vulnerabilities": [v.to_dict() for v in self._vulnerabilities.values()]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def import_json(self, filepath: str, merge: bool = True):
        """
        Import vulnerabilities from JSON file.

        Args:
            filepath: Path to JSON file
            merge: If True, merge with existing. If False, replace.
        """
        with open(filepath) as f:
            data = json.load(f)

        if not merge:
            self._vulnerabilities.clear()

        for vuln_data in data.get("vulnerabilities", []):
            vuln = CatalogedVulnerability.from_dict(vuln_data)
            self._vulnerabilities[vuln.id] = vuln

        self._save()


# =============================================================================
# Factory Functions
# =============================================================================


def create_catalog(
    storage_path: str | None = None
) -> AdversarialCatalog:
    """
    Factory function for catalog.

    Args:
        storage_path: Path for persistence (None for in-memory)

    Returns:
        AdversarialCatalog instance
    """
    return AdversarialCatalog(storage_path)
