"""
Activation Adapter — ACT-R base-level activation for PPR seed weighting.

Bridges the existing ActivationField (spreading activation with decay)
into the v2 navigator's seed extraction pipeline.

ACT-R theory: items accessed more recently and more frequently get higher
base-level activation, making them easier to retrieve. We model this as
a weight multiplier on PPR seeds — recently-attended nodes get boosted
seed weight, pulling PPR results toward them.

This gives cross-turn continuity: if you asked about Thompson Sampling
last turn, its graph nodes retain activation, so this turn's PPR starts
with a gentle bias toward that neighborhood. The bias decays naturally.

No LLM calls. No external dependencies beyond ActivationField.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ActivationConfig:
    """Configuration for activation-based seed boosting."""
    retrieval_boost: float = 0.3    # Activation boost on retrieval
    query_boost: float = 0.5       # Activation boost for query-mentioned entities
    decay_rate: float = 0.1        # Slow decay between turns (ACT-R forgetting)
    max_activation: float = 1.0    # Ceiling on activation level
    min_activation: float = 0.05   # Prune threshold
    weight_amplifier: float = 1.0  # How much activation amplifies seed weight
    spread_decay: float = 0.5      # Decay factor for graph-based spreading


class ActivationAdapter:
    """Bridges ActivationField to v2 Navigator seed weights.

    ACT-R base-level activation: B_i = ln(sum(t_j^-d))
    Simplified: activation = recency * frequency, with decay.

    The adapter maintains a lightweight activation state (dict of node_id → float)
    that can optionally be backed by the full ActivationField with spatial indexing
    and graph-based spreading.

    Usage in the navigator pipeline:
        1. extract_seeds() finds seed nodes from query
        2. adapter.boost_seeds(seeds) amplifies weights by activation
        3. PPR runs with activation-adjusted seed weights
        4. adapter.on_retrieval(retrieved_ids) boosts retrieved nodes
        5. adapter.decay_step() runs between turns
    """

    def __init__(
        self,
        config: Optional[ActivationConfig] = None,
        activation_field=None,  # Optional[ActivationField] — avoid hard import
    ):
        self.config = config or ActivationConfig()
        self._field = activation_field  # Optional backing ActivationField

        # Lightweight activation levels (always maintained, even without ActivationField)
        self._levels: Dict[str, float] = {}

        # Sync from ActivationField if provided
        if self._field is not None:
            self._levels = self._field.levels

    @property
    def levels(self) -> Dict[str, float]:
        """Current activation levels for all tracked nodes."""
        if self._field is not None:
            return self._field.levels
        return self._levels

    def boost_seeds(self, seeds: list) -> list:
        """Apply activation-based weight boost to PPR seeds.

        ACT-R pattern: higher activation → higher retrieval probability.
        Activation amplifies seed weight multiplicatively:
            new_weight = weight * (1.0 + activation * amplifier)

        A node with activation=0.5 and amplifier=1.0 gets 1.5x weight.
        A node with activation=0.0 gets no change (1.0x weight).
        """
        levels = self.levels
        if not levels:
            return seeds

        boosted = 0
        for seed in seeds:
            activation = levels.get(seed.node_id, 0.0)
            if activation > self.config.min_activation:
                multiplier = 1.0 + activation * self.config.weight_amplifier
                seed.weight *= multiplier
                boosted += 1

        if boosted > 0:
            logger.debug("Activation boosted %d/%d seeds", boosted, len(seeds))

        return seeds

    def on_retrieval(self, node_ids: List[str]):
        """Boost activation of retrieved nodes (ACT-R recency effect).

        Every time a node appears in PPR results, its activation increases.
        This creates a recency bias: recently-retrieved nodes are more
        likely to appear again, modeling the "tip of the tongue" effect.
        """
        levels = self.levels
        cfg = self.config

        for nid in node_ids:
            current = levels.get(nid, 0.0)
            levels[nid] = min(cfg.max_activation, current + cfg.retrieval_boost)

    def on_query_entities(self, entity_ids: List[str]):
        """Boost activation for entities mentioned in the query.

        Higher boost than retrieval — explicit mention signals strong relevance.
        """
        levels = self.levels
        cfg = self.config

        for eid in entity_ids:
            current = levels.get(eid, 0.0)
            levels[eid] = min(cfg.max_activation, current + cfg.query_boost)

    def decay_step(self):
        """Decay all activations (ACT-R forgetting between turns).

        Uses the configured decay_rate. If backed by ActivationField,
        delegates to its decay method. Otherwise does simple multiplicative decay.
        """
        if self._field is not None:
            self._field.decay(rate=self.config.decay_rate)
            return

        # Simple multiplicative decay
        factor = 1.0 - self.config.decay_rate
        pruned: Dict[str, float] = {}
        for nid, level in self._levels.items():
            new_level = level * factor
            if new_level > self.config.min_activation:
                pruned[nid] = new_level
        self._levels = pruned

    def spread(self, graph=None):
        """Spread activation through graph connections.

        Only works when backed by a real ActivationField with a graph.
        Delegates to ActivationField.spread_via_graph().
        """
        if self._field is not None and graph is not None:
            self._field.spread_via_graph(
                graph=graph,
                iterations=2,
                decay_factor=self.config.spread_decay,
            )

    def active_node_ids(self, min_activation: float = 0.1) -> List[str]:
        """Get node IDs with activation above threshold, sorted by level."""
        levels = self.levels
        active = [
            (nid, level) for nid, level in levels.items()
            if level >= min_activation
        ]
        active.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in active]

    def get_activation(self, node_id: str) -> float:
        """Get activation level for a specific node."""
        return self.levels.get(node_id, 0.0)

    def clear(self):
        """Reset all activations."""
        if self._field is not None:
            self._field.clear()
        else:
            self._levels.clear()

    def stats(self) -> Dict[str, float]:
        """Activation field statistics."""
        levels = self.levels
        active = [v for v in levels.values() if v > self.config.min_activation]
        return {
            "n_tracked": len(levels),
            "n_active": len(active),
            "total_activation": sum(levels.values()),
            "mean_activation": sum(active) / len(active) if active else 0.0,
            "max_activation": max(active) if active else 0.0,
        }
