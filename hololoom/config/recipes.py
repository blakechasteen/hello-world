"""
Recipe System
=============

Named stage sets that determine WHAT stages run in a pipeline.
Composable via set union. Recipes replace 55+ boolean flags.

Core tiers (BARE through RESEARCH) are standalone.
Add-ons (JENNY, METRICS, etc.) compose on top via union.

Usage:
    Config(tier=FAST)                       # 5-stage fast pipeline
    Config(tier=FULL, addons=[JENNY])       # full + Jenny panels
    Config(tier=RESEARCH, addons=[METRICS]) # research + monitoring
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Recipe:
    """A named set of stages. Composable via union."""
    name: str
    stages: frozenset[str]

    def __or__(self, other: 'Recipe') -> 'Recipe':
        """Compose two recipes: ``FAST | JENNY``."""
        return Recipe(
            name=f"{self.name}+{other.name}",
            stages=self.stages | other.stages,
        )

    def __contains__(self, stage_name: str) -> bool:
        return stage_name in self.stages

    def __len__(self) -> int:
        return len(self.stages)

    def __repr__(self) -> str:
        return f"Recipe({self.name!r}, {len(self.stages)} stages)"


# =========================================================================
# Core Tiers
# =========================================================================

BARE = Recipe('bare', frozenset({
    'pattern_selection',
    'feature_extraction',
    'decision_collapse',
}))

LITE = Recipe('lite', frozenset({
    'feature_extraction',
    'quick_decision',
    'fast_execution',
}))

FAST = Recipe('fast', frozenset({
    'pattern_selection',
    'temporal_control',
    'feature_extraction',
    'memory_retrieval',
    'quick_decision',
}))

FULL = Recipe('full', frozenset({
    'pattern_selection',
    'temporal_control',
    'thread_selection',
    'feature_extraction',
    'warp_tensioning',
    'warp_compute',
    'memory_retrieval',
    'beta_wave_packing',
    'decision_collapse',
    'tool_execution',
    'fabric_weaving',
}))

RESEARCH = Recipe('research', FULL.stages | frozenset({
    'meta_prompt_enhancement',
    'smart_routing',
    'conscience_gating',
    'recursive_learning',
    'reflection',
}))


# =========================================================================
# Add-ons (composable via union)
# =========================================================================

JENNY = Recipe('jenny', frozenset({'jenny_compilation'}))
METRICS = Recipe('metrics', frozenset({'production_metrics'}))
CONSCIENCE = Recipe('conscience', frozenset({'conscience_gating'}))
AWARENESS = Recipe('awareness', frozenset({'consciousness_perception'}))


# =========================================================================
# Registry (for lookup by name/complexity level)
# =========================================================================

TIER_RECIPES = {
    'bare': BARE,
    'lite': LITE,
    'fast': FAST,
    'full': FULL,
    'research': RESEARCH,
}

ADDON_RECIPES = {
    'jenny': JENNY,
    'metrics': METRICS,
    'conscience': CONSCIENCE,
    'awareness': AWARENESS,
}


def merge_recipes(recipes: list[Recipe]) -> frozenset[str]:
    """Union all recipe stage sets."""
    if not recipes:
        return frozenset()
    return frozenset().union(*(r.stages for r in recipes))


def recipe_from_name(name: str) -> Recipe:
    """Look up a recipe by name (tier or addon)."""
    name_lower = name.lower()
    if name_lower in TIER_RECIPES:
        return TIER_RECIPES[name_lower]
    if name_lower in ADDON_RECIPES:
        return ADDON_RECIPES[name_lower]
    raise ValueError(
        f"Unknown recipe: {name!r}. "
        f"Available tiers: {list(TIER_RECIPES)}. "
        f"Available addons: {list(ADDON_RECIPES)}."
    )
