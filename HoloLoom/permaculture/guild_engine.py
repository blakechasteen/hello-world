"""
Guild Recommendation Engine - RAG-powered companion planting

This module uses HoloLoom's RAG system to recommend optimal plant guilds
based on permaculture principles.

**Created: 2025-11-21**
"""

import asyncio
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass

from HoloLoom.permaculture.plant_database import (
    Plant,
    PlantDatabase,
    Guild,
    GuildRole,
    Layer,
    Function,
)
from HoloLoom import HoloLoom
from HoloLoom.config import Config


@dataclass
class GuildRecommendation:
    """A recommended guild with scoring"""
    guild: Guild
    score: float
    layer_diversity_score: float
    function_diversity_score: float
    compatibility_score: float
    zone_compatibility: bool
    reasoning: str


class GuildRecommendationEngine:
    """
    RAG-powered guild recommendation engine

    Uses HoloLoom's memory and semantic search to recommend
    optimal plant guilds based on:
    - Central plant requirements
    - Layer diversity (vertical stacking)
    - Function diversity (nitrogen fixers, pest deterrents, etc.)
    - Companion compatibility
    - Climate zone compatibility
    """

    def __init__(
        self,
        plant_db: PlantDatabase,
        config: Optional[Config] = None
    ):
        """
        Initialize guild engine

        Args:
            plant_db: PlantDatabase with all available plants
            config: HoloLoom config (defaults to Config.fast())
        """
        self.plant_db = plant_db
        self.config = config or Config.fast()
        self.hololoom: Optional[HoloLoom] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.hololoom = HoloLoom(cfg=self.config)
        await self.hololoom.__aenter__()

        # Ingest all plant knowledge into HoloLoom
        await self._ingest_plant_knowledge()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.hololoom:
            await self.hololoom.__aexit__(exc_type, exc_val, exc_tb)

    async def _ingest_plant_knowledge(self) -> None:
        """Ingest all plants into HoloLoom's memory system"""
        for plant in self.plant_db.plants.values():
            # Create rich text description for semantic embedding
            plant_text = self._plant_to_text(plant)
            await self.hololoom.experience(plant_text)

    def _plant_to_text(self, plant: Plant) -> str:
        """Convert plant to rich text description for RAG"""
        layers_str = ", ".join([layer.value for layer in plant.layers])
        functions_str = ", ".join([func.value for func in plant.functions])
        companions_str = ", ".join(plant.companions) if plant.companions else "none listed"

        text = f"""
Plant: {plant.common_name} ({plant.scientific_name})
Family: {plant.family}
Layers: {layers_str}
Functions: {functions_str}
Climate Zones: {plant.hardiness_zones}
Sun: {plant.sun_requirements}, Water: {plant.water_needs}
Height: {plant.height_ft}ft, Spread: {plant.spread_ft}ft
Root Depth: {plant.root_depth}
Lifecycle: {plant.lifecycle}
Description: {plant.description}
Yield: {plant.yield_info}
Known Companions: {companions_str}
"""
        return text.strip()

    async def recommend_guild(
        self,
        central_plant_name: str,
        climate_zone: int,
        target_layers: Optional[Set[Layer]] = None,
        target_functions: Optional[Set[Function]] = None,
        max_plants: int = 8,
    ) -> GuildRecommendation:
        """
        Recommend a guild around a central plant

        Args:
            central_plant_name: Common name of central plant (e.g., "Apple")
            climate_zone: USDA hardiness zone
            target_layers: Desired layers to include (defaults to all)
            target_functions: Desired functions to include (defaults to key functions)
            max_plants: Maximum guild members (excluding central plant)

        Returns:
            GuildRecommendation with scored guild
        """
        central_plant = self.plant_db.get_plant(central_plant_name)
        if not central_plant:
            raise ValueError(f"Central plant '{central_plant_name}' not found")

        # Check zone compatibility
        zone_compatible = climate_zone in central_plant.hardiness_zones
        if not zone_compatible:
            print(f"⚠ Warning: {central_plant_name} not rated for zone {climate_zone}")

        # Default target layers (all except central plant's layers)
        if target_layers is None:
            target_layers = {
                Layer.SHRUB,
                Layer.HERBACEOUS,
                Layer.GROUND_COVER,
                Layer.VINE,
                Layer.ROOT,
            }
            # Remove layers already occupied by central plant
            for layer in central_plant.layers:
                target_layers.discard(layer)

        # Default target functions
        if target_functions is None:
            target_functions = {
                Function.NITROGEN_FIXER,
                Function.DYNAMIC_ACCUMULATOR,
                Function.POLLINATOR_ATTRACTOR,
                Function.PEST_DETERRENT,
                Function.GROUNDCOVER,
            }

        # Find candidate plants
        candidates = await self._find_candidates(
            central_plant,
            climate_zone,
            target_layers,
            target_functions,
        )

        # Score and select optimal guild members
        guild_members = await self._select_optimal_members(
            central_plant,
            candidates,
            target_layers,
            target_functions,
            max_plants,
        )

        # Create guild
        guild = Guild(
            name=f"{central_plant.common_name} Guild",
            central_plant=central_plant,
            members=guild_members,
            description=f"Permaculture guild centered around {central_plant.common_name}",
            climate_zones=[climate_zone],
        )

        # Score the guild
        recommendation = self._score_guild(guild, target_layers, target_functions, zone_compatible)

        return recommendation

    async def _find_candidates(
        self,
        central_plant: Plant,
        climate_zone: int,
        target_layers: Set[Layer],
        target_functions: Set[Function],
    ) -> List[Plant]:
        """Find candidate plants for the guild"""
        candidates = []

        # 1. Get known companions (highest priority)
        for companion_name in central_plant.companions:
            companion = self.plant_db.get_plant(companion_name)
            if companion and climate_zone in companion.hardiness_zones:
                candidates.append(companion)

        # 2. Search by target functions
        for function in target_functions:
            function_plants = self.plant_db.search_by_function(function)
            for plant in function_plants:
                if (
                    plant.common_name != central_plant.common_name
                    and climate_zone in plant.hardiness_zones
                    and plant not in candidates
                ):
                    candidates.append(plant)

        # 3. Search by target layers
        for layer in target_layers:
            layer_plants = self.plant_db.search_by_layer(layer)
            for plant in layer_plants:
                if (
                    plant.common_name != central_plant.common_name
                    and climate_zone in plant.hardiness_zones
                    and plant not in candidates
                ):
                    candidates.append(plant)

        # 4. Remove antagonists
        antagonist_names = {name.lower() for name in central_plant.antagonists}
        candidates = [
            plant for plant in candidates
            if plant.common_name.lower() not in antagonist_names
        ]

        return candidates

    async def _select_optimal_members(
        self,
        central_plant: Plant,
        candidates: List[Plant],
        target_layers: Set[Layer],
        target_functions: Set[Function],
        max_plants: int,
    ) -> List[GuildRole]:
        """Select optimal guild members using greedy algorithm"""
        selected: List[GuildRole] = []
        covered_layers: Set[Layer] = set(central_plant.layers)
        covered_functions: Set[Function] = set(central_plant.functions)

        # Priority 1: Known companions
        companions = [
            p for p in candidates
            if p.common_name in central_plant.companions
        ]

        for companion in companions[:max_plants]:
            primary_func = self._select_primary_function(
                companion,
                target_functions,
                covered_functions,
            )
            selected.append(GuildRole(
                plant=companion,
                primary_function=primary_func,
                notes="Known companion plant"
            ))
            covered_layers.update(companion.layers)
            covered_functions.update(companion.functions)

        # Priority 2: Fill missing layers
        remaining_layers = target_layers - covered_layers
        for layer in remaining_layers:
            if len(selected) >= max_plants:
                break

            layer_candidates = [
                p for p in candidates
                if layer in p.layers and p not in [r.plant for r in selected]
            ]

            if layer_candidates:
                # Prefer plants with multiple target functions
                best = max(
                    layer_candidates,
                    key=lambda p: len(set(p.functions) & target_functions)
                )

                primary_func = self._select_primary_function(
                    best,
                    target_functions,
                    covered_functions,
                )

                selected.append(GuildRole(
                    plant=best,
                    primary_function=primary_func,
                    notes=f"Fills {layer.value} layer"
                ))
                covered_layers.update(best.layers)
                covered_functions.update(best.functions)

        # Priority 3: Fill missing functions
        remaining_functions = target_functions - covered_functions
        for function in remaining_functions:
            if len(selected) >= max_plants:
                break

            func_candidates = [
                p for p in candidates
                if function in p.functions and p not in [r.plant for r in selected]
            ]

            if func_candidates:
                # Prefer plants in uncovered layers
                best = max(
                    func_candidates,
                    key=lambda p: len(set(p.layers) - covered_layers)
                )

                selected.append(GuildRole(
                    plant=best,
                    primary_function=function,
                    notes=f"Provides {function.value} function"
                ))
                covered_layers.update(best.layers)
                covered_functions.update(best.functions)

        return selected

    def _select_primary_function(
        self,
        plant: Plant,
        target_functions: Set[Function],
        covered_functions: Set[Function],
    ) -> Function:
        """Select the most valuable primary function for a plant"""
        # Prefer functions in target that aren't covered yet
        uncovered = set(plant.functions) & target_functions - covered_functions
        if uncovered:
            return next(iter(uncovered))

        # Otherwise, prefer functions in target
        in_target = set(plant.functions) & target_functions
        if in_target:
            return next(iter(in_target))

        # Otherwise, just use first function
        return plant.functions[0]

    def _score_guild(
        self,
        guild: Guild,
        target_layers: Set[Layer],
        target_functions: Set[Function],
        zone_compatible: bool,
    ) -> GuildRecommendation:
        """Score a guild based on permaculture principles"""
        # Layer diversity (want good vertical stacking)
        layer_counts = guild.get_layer_diversity()
        covered_layers = set(layer_counts.keys())
        layer_diversity = len(covered_layers & target_layers) / len(target_layers)

        # Function diversity (want key functions covered)
        function_counts = guild.get_function_diversity()
        covered_functions = set(function_counts.keys())
        function_diversity = len(covered_functions & target_functions) / len(target_functions)

        # Companion compatibility (known companions are good)
        companions_used = sum(
            1 for role in guild.members
            if role.plant.common_name in guild.central_plant.companions
        )
        compatibility = companions_used / len(guild.members) if guild.members else 0

        # Overall score (weighted average)
        score = (
            0.3 * layer_diversity +
            0.4 * function_diversity +
            0.3 * compatibility
        )

        # Zone penalty
        if not zone_compatible:
            score *= 0.8

        # Generate reasoning
        reasoning = self._generate_reasoning(
            guild,
            layer_diversity,
            function_diversity,
            compatibility,
            covered_layers,
            covered_functions,
        )

        return GuildRecommendation(
            guild=guild,
            score=score,
            layer_diversity_score=layer_diversity,
            function_diversity_score=function_diversity,
            compatibility_score=compatibility,
            zone_compatibility=zone_compatible,
            reasoning=reasoning,
        )

    def _generate_reasoning(
        self,
        guild: Guild,
        layer_div: float,
        func_div: float,
        compat: float,
        covered_layers: Set[Layer],
        covered_functions: Set[Function],
    ) -> str:
        """Generate human-readable reasoning for recommendation"""
        lines = [
            f"Guild Score: {guild.name}",
            f"",
            f"Layer Diversity: {layer_div:.1%}",
            f"  Covered: {', '.join([l.value for l in sorted(covered_layers, key=lambda x: x.value)])}",
            f"",
            f"Function Diversity: {func_div:.1%}",
            f"  Covered: {', '.join([f.value for f in sorted(covered_functions, key=lambda x: x.value)])}",
            f"",
            f"Companion Compatibility: {compat:.1%}",
            f"",
            f"Members ({len(guild.members)}):",
        ]

        for role in guild.members:
            layers_str = ", ".join([l.value for l in role.plant.layers])
            lines.append(
                f"  • {role.plant.common_name} ({layers_str}) - {role.primary_function.value}"
            )
            if role.notes:
                lines.append(f"    └─ {role.notes}")

        return "\n".join(lines)

    async def query_plants(self, query: str, limit: int = 5) -> List[Plant]:
        """
        Semantic search for plants using RAG

        Args:
            query: Natural language query (e.g., "nitrogen fixing ground covers")
            limit: Maximum results

        Returns:
            List of matching plants
        """
        if not self.hololoom:
            raise RuntimeError("GuildEngine not initialized (use async context manager)")

        # Use HoloLoom's recall for semantic search
        memories = await self.hololoom.recall(query, k=limit)

        # Extract plant names from memory texts
        plants = []
        for memory in memories:
            # Parse plant name from memory text (format: "Plant: Name (Scientific)")
            lines = memory.text.split('\n')
            if lines and lines[0].startswith("Plant:"):
                plant_line = lines[0].replace("Plant:", "").strip()
                common_name = plant_line.split("(")[0].strip()
                plant = self.plant_db.get_plant(common_name)
                if plant:
                    plants.append(plant)

        return plants[:limit]
