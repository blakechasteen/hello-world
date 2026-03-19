"""
Math Catalog — Auto-generated operation catalog from the math library.

Scans core/warp/math/ at startup and builds a MathOperation entry for each
module. 151 modules become 151 selectable operations — no manual maintenance.

Cost table provides three tiers:
  Tier 1: domain defaults (analysis=5, algebra=10, geometry=12, ...)
  Tier 2: filename overrides (spectral_clustering=30, ricci_flow=50, ...)
  Tier 3: fallback (MODERATE, cost=10)

Step masks filter operations per weaving step — Step 4 (Resonance) gets
feature extraction math, Step 7 (Convergence) gets decision math.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Root of the math library
MATH_ROOT = Path(__file__).parent


# ============================================================================
# Extended Domain Enum (8 → 12)
# ============================================================================

class MathDomain(Enum):
    """Mathematical domain categories — covers all 12 subdirectories."""
    ANALYSIS = "analysis"
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    DECISION = "decision"
    EXTENSIONS = "extensions"
    COMPUTATIONAL = "computational"
    GRAPH = "graph"
    INFORMATION = "information"
    LOGIC = "logic"
    CAUSAL = "causal"
    SELF_ASSEMBLY = "self_assembly"
    ADVANCED = "advanced"
    ROOT = "root"  # modules at math/ root level


class OperationLevel(Enum):
    """Computational complexity level."""
    BASIC = "basic"         # O(n), <1ms
    MODERATE = "moderate"   # O(n²), 1-5ms
    ADVANCED = "advanced"   # O(n³), 5-30ms
    EXPENSIVE = "expensive" # O(n⁴+), 30-200ms


class QueryIntent(Enum):
    """High-level query intent categories."""
    SIMILARITY = "similarity"
    OPTIMIZATION = "optimization"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    DECISION = "decision"
    VERIFICATION = "verification"
    TRANSFORMATION = "transformation"


# ============================================================================
# Operation Specification
# ============================================================================

@dataclass
class MathOperation:
    """Specification for a selectable mathematical operation."""
    name: str
    domain: MathDomain
    level: OperationLevel
    description: str
    use_cases: list[QueryIntent]
    prerequisites: list[str] = field(default_factory=list)
    module_path: str = ""
    function_name: str = ""
    estimated_cost: int = 10
    _module: object = field(default=None, repr=False)

    def is_applicable(self, intent: QueryIntent) -> bool:
        return intent in self.use_cases

    def load_module(self):
        """Lazy-import the module. Returns the module object or None."""
        if self._module is not None:
            return self._module
        try:
            import importlib
            self._module = importlib.import_module(self.module_path)
            return self._module
        except Exception as e:
            logger.debug("Failed to import %s: %s", self.module_path, e)
            return None


# ============================================================================
# Step Masks
# ============================================================================

@dataclass
class StepMask:
    """Per-step filter for which operations are relevant."""
    step_id: int
    step_name: str
    allowed_domains: set[MathDomain]
    max_level: OperationLevel
    budget_fraction: float  # fraction of total query budget this step can use
    priority_ops: list[str] = field(default_factory=list)


# Default step masks
STEP_MASKS = {
    4: StepMask(
        step_id=4, step_name="Resonance",
        allowed_domains={MathDomain.ANALYSIS, MathDomain.INFORMATION, MathDomain.GRAPH},
        max_level=OperationLevel.MODERATE,
        budget_fraction=0.15,
    ),
    5: StepMask(
        step_id=5, step_name="Warp",
        allowed_domains={MathDomain.ALGEBRA, MathDomain.GEOMETRY, MathDomain.ANALYSIS},
        max_level=OperationLevel.ADVANCED,
        budget_fraction=0.20,
    ),
    55: StepMask(
        step_id=55, step_name="Compute",
        allowed_domains={MathDomain.ANALYSIS, MathDomain.COMPUTATIONAL, MathDomain.ALGEBRA},
        max_level=OperationLevel.ADVANCED,
        budget_fraction=0.15,
    ),
    # Step 5.5b — unrestricted, primary injection point
    56: StepMask(
        step_id=56, step_name="MathPipeline",
        allowed_domains=set(MathDomain),
        max_level=OperationLevel.EXPENSIVE,
        budget_fraction=0.30,
    ),
    65: StepMask(
        step_id=65, step_name="Packing",
        allowed_domains={MathDomain.ANALYSIS, MathDomain.COMPUTATIONAL},
        max_level=OperationLevel.MODERATE,
        budget_fraction=0.10,
    ),
    7: StepMask(
        step_id=7, step_name="Convergence",
        allowed_domains={MathDomain.DECISION, MathDomain.INFORMATION, MathDomain.SELF_ASSEMBLY},
        max_level=OperationLevel.ADVANCED,
        budget_fraction=0.10,
    ),
}

LEVEL_ORDER = {
    OperationLevel.BASIC: 0,
    OperationLevel.MODERATE: 1,
    OperationLevel.ADVANCED: 2,
    OperationLevel.EXPENSIVE: 3,
}


# ============================================================================
# Cost Table
# ============================================================================

# Tier 1: domain defaults
DOMAIN_DEFAULTS = {
    MathDomain.ANALYSIS: (OperationLevel.MODERATE, 5),
    MathDomain.ALGEBRA: (OperationLevel.MODERATE, 10),
    MathDomain.GEOMETRY: (OperationLevel.ADVANCED, 12),
    MathDomain.DECISION: (OperationLevel.MODERATE, 8),
    MathDomain.EXTENSIONS: (OperationLevel.MODERATE, 8),
    MathDomain.COMPUTATIONAL: (OperationLevel.MODERATE, 8),
    MathDomain.GRAPH: (OperationLevel.MODERATE, 10),
    MathDomain.INFORMATION: (OperationLevel.BASIC, 3),
    MathDomain.LOGIC: (OperationLevel.MODERATE, 5),
    MathDomain.CAUSAL: (OperationLevel.ADVANCED, 15),
    MathDomain.SELF_ASSEMBLY: (OperationLevel.ADVANCED, 15),
    MathDomain.ADVANCED: (OperationLevel.ADVANCED, 12),
    MathDomain.ROOT: (OperationLevel.MODERATE, 8),
}

# Tier 2: filename-specific overrides (preserve existing 19 costs)
COST_OVERRIDES = {
    # BASIC (cost 1-3)
    "information_theory": (OperationLevel.BASIC, 2),
    "renyi_entropy": (OperationLevel.BASIC, 2),
    "description_length": (OperationLevel.BASIC, 3),
    "order_statistics": (OperationLevel.BASIC, 1),
    "streaming": (OperationLevel.BASIC, 2),
    # MODERATE (cost 5-10)
    "kalman_filtering": (OperationLevel.MODERATE, 5),
    "bootstrap": (OperationLevel.MODERATE, 5),
    "changepoint_detection": (OperationLevel.MODERATE, 8),
    "hidden_markov_models": (OperationLevel.MODERATE, 10),
    "time_series_models": (OperationLevel.MODERATE, 8),
    "thompson_sampling": (OperationLevel.MODERATE, 3),
    "contextual_bandit": (OperationLevel.MODERATE, 5),
    "game_theory": (OperationLevel.MODERATE, 8),
    "graph_coloring": (OperationLevel.MODERATE, 8),
    # ADVANCED (cost 12-25)
    "stability_analysis": (OperationLevel.ADVANCED, 15),
    "bifurcation_analysis": (OperationLevel.ADVANCED, 18),
    "chaos_dynamical_systems": (OperationLevel.ADVANCED, 20),
    "fokker_planck": (OperationLevel.ADVANCED, 15),
    "fourier_harmonic": (OperationLevel.ADVANCED, 10),
    "spectral_clustering": (OperationLevel.ADVANCED, 25),
    "manifold_learning": (OperationLevel.ADVANCED, 20),
    "counterfactual_regret": (OperationLevel.ADVANCED, 15),
    "lie_algebras": (OperationLevel.ADVANCED, 15),
    "representation_theory": (OperationLevel.ADVANCED, 18),
    # EXPENSIVE (cost 30+)
    "riemannian_geometry": (OperationLevel.EXPENSIVE, 40),
    "advanced_curvature": (OperationLevel.EXPENSIVE, 50),
    "algebraic_geometry": (OperationLevel.EXPENSIVE, 35),
    "toposes": (OperationLevel.EXPENSIVE, 30),
    "extended_persistence": (OperationLevel.EXPENSIVE, 35),
    "alpha_cech_complexes": (OperationLevel.EXPENSIVE, 40),
}

# Domain → default applicable intents
DOMAIN_INTENTS = {
    MathDomain.ANALYSIS: [QueryIntent.ANALYSIS, QueryIntent.VERIFICATION, QueryIntent.SIMILARITY],
    MathDomain.ALGEBRA: [QueryIntent.TRANSFORMATION, QueryIntent.ANALYSIS],
    MathDomain.GEOMETRY: [QueryIntent.SIMILARITY, QueryIntent.ANALYSIS, QueryIntent.TRANSFORMATION],
    MathDomain.DECISION: [QueryIntent.DECISION, QueryIntent.OPTIMIZATION],
    MathDomain.EXTENSIONS: [QueryIntent.ANALYSIS, QueryIntent.TRANSFORMATION],
    MathDomain.COMPUTATIONAL: [QueryIntent.TRANSFORMATION, QueryIntent.OPTIMIZATION],
    MathDomain.GRAPH: [QueryIntent.ANALYSIS, QueryIntent.SIMILARITY],
    MathDomain.INFORMATION: [QueryIntent.ANALYSIS, QueryIntent.DECISION],
    MathDomain.LOGIC: [QueryIntent.VERIFICATION, QueryIntent.ANALYSIS],
    MathDomain.CAUSAL: [QueryIntent.ANALYSIS, QueryIntent.VERIFICATION],
    MathDomain.SELF_ASSEMBLY: [QueryIntent.OPTIMIZATION, QueryIntent.GENERATION],
    MathDomain.ADVANCED: [QueryIntent.ANALYSIS, QueryIntent.TRANSFORMATION],
    MathDomain.ROOT: [QueryIntent.ANALYSIS, QueryIntent.DECISION],
}

# Directory name → MathDomain
DIR_TO_DOMAIN = {
    "analysis": MathDomain.ANALYSIS,
    "algebra": MathDomain.ALGEBRA,
    "geometry": MathDomain.GEOMETRY,
    "decision": MathDomain.DECISION,
    "extensions": MathDomain.EXTENSIONS,
    "computational": MathDomain.COMPUTATIONAL,
    "graph": MathDomain.GRAPH,
    "information": MathDomain.INFORMATION,
    "logic": MathDomain.LOGIC,
    "causal": MathDomain.CAUSAL,
    "self_assembly": MathDomain.SELF_ASSEMBLY,
    "advanced": MathDomain.ADVANCED,
}


# ============================================================================
# Catalog Builder
# ============================================================================

class MathCatalogBuilder:
    """Scans core/warp/math/ and builds MathOperation entries for all modules."""

    @staticmethod
    def build(root: Optional[Path] = None) -> dict[str, MathOperation]:
        """Scan the math library and return a catalog of all operations.

        Returns dict mapping operation name to MathOperation.
        """
        root = root or MATH_ROOT
        catalog = {}

        # Scan subdirectories
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and entry.name in DIR_TO_DOMAIN and entry.name != "__pycache__":
                domain = DIR_TO_DOMAIN[entry.name]
                for py_file in sorted(entry.glob("*.py")):
                    if py_file.name.startswith("__") or py_file.name.startswith("test_"):
                        continue
                    op = MathCatalogBuilder._build_operation(py_file, domain, root)
                    if op:
                        catalog[op.name] = op

        # Scan root-level modules
        for py_file in sorted(root.glob("*.py")):
            if py_file.name.startswith("__") or py_file.name.startswith("test_"):
                continue
            if py_file.name in ("catalog.py", "budget.py", "loom.py"):
                continue  # Skip infrastructure files
            op = MathCatalogBuilder._build_operation(py_file, MathDomain.ROOT, root)
            if op:
                catalog[op.name] = op

        logger.info("MathCatalog: %d operations across %d domains",
                     len(catalog), len(set(op.domain for op in catalog.values())))
        return catalog

    @staticmethod
    def _build_operation(py_file: Path, domain: MathDomain, root: Path) -> Optional[MathOperation]:
        """Build a MathOperation from a Python file."""
        name = py_file.stem

        # Compute module path: hololoom.core.warp.math.{subdir}.{module}
        # Use the absolute path parts starting from 'hololoom'
        parts = py_file.with_suffix("").parts
        # Find 'hololoom' in the path and take everything from there
        try:
            hololoom_idx = parts.index("hololoom")
            module_path = ".".join(parts[hololoom_idx:])
        except ValueError:
            module_path = name  # fallback

        # Get cost/level from override or domain default
        if name in COST_OVERRIDES:
            level, cost = COST_OVERRIDES[name]
        elif domain in DOMAIN_DEFAULTS:
            level, cost = DOMAIN_DEFAULTS[domain]
        else:
            level, cost = OperationLevel.MODERATE, 10

        # Get applicable intents from domain
        use_cases = DOMAIN_INTENTS.get(domain, [QueryIntent.ANALYSIS])

        # Description from filename
        description = name.replace("_", " ").title()

        return MathOperation(
            name=name,
            domain=domain,
            level=level,
            description=description,
            use_cases=use_cases,
            module_path=module_path,
            estimated_cost=cost,
        )

    @staticmethod
    def filter(
        catalog: dict[str, MathOperation],
        allowed_domains: Optional[set[MathDomain]] = None,
        max_level: Optional[OperationLevel] = None,
        max_cost: Optional[int] = None,
        intent: Optional[QueryIntent] = None,
    ) -> dict[str, MathOperation]:
        """Filter catalog by step mask constraints."""
        filtered = {}
        for name, op in catalog.items():
            if allowed_domains and op.domain not in allowed_domains:
                continue
            if max_level and LEVEL_ORDER.get(op.level, 0) > LEVEL_ORDER.get(max_level, 3):
                continue
            if max_cost and op.estimated_cost > max_cost:
                continue
            if intent and not op.is_applicable(intent):
                continue
            filtered[name] = op
        return filtered


# Singleton catalog
_catalog: Optional[dict[str, MathOperation]] = None


def get_catalog() -> dict[str, MathOperation]:
    """Get or build the global math catalog."""
    global _catalog
    if _catalog is None:
        _catalog = MathCatalogBuilder.build()
    return _catalog
