"""
ScoredObservation — Domain-agnostic output of any additive scoring pipeline.

Not a new type. A convention on MemoryItem enforced by pure functions.
Any domain (hive, kitchen, system) defines scoring terms; this module
composes them into storable, queryable, learnable observations.

Convention mapping:
    memory_type  = domain string ("hive", "kitchen", ...)
    entity_type  = category within domain ("pest", "stores", ...)
    entity_ids   = what this observation is about
    importance   = normalized composite score (0.0-1.0)
    properties   = {band, status, composite, breakdown, mode, observed_at}

Usage:
    from hololoom.memory.scored_observation import (
        ScoringTerm, scored_observation, advance_observation
    )

    terms = [ScoringTerm("w_varroa", 30, varroa_score), ...]
    obs = scored_observation("Varroa high", "hive", "pest", terms, ctx)
    obs_id = await bus.store(obs)

    # Later:
    res_id = await advance_observation(bus, obs_id, "resolved", "Treated")
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .bus import MemoryItem

logger = logging.getLogger(__name__)


# =============================================================================
# Scoring Primitives
# =============================================================================

@dataclass(frozen=True)
class ScoringTerm:
    """A single named scoring term. Pure function, fixed weight.

    The fn should accept a context dict and return a float in [0, 1].
    Weight determines the term's maximum contribution to the composite.
    """
    name: str
    weight: float
    fn: Callable[..., float]


def band_from_score(composite: float) -> str:
    """Map a 0-100 composite score to a priority band.

    Thresholds match SOUS GPS/MFS banding:
        P0 (≥80): emergency / immediate action
        P1 (≥35): needs action soon
        P2 (≥18): monitor
        P3 (<18): informational
    """
    if composite >= 80:
        return "P0"
    elif composite >= 35:
        return "P1"
    elif composite >= 18:
        return "P2"
    return "P3"


def score_terms(
    terms: List[ScoringTerm],
    context: Dict[str, Any],
    mode_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute additive score from terms.

    Args:
        terms: Scoring terms to evaluate.
        context: Domain-specific context dict passed to each term's fn.
        mode_weights: Optional multipliers per term name. Values >1.0
                      amplify, <1.0 dampen. This is where the
                      ObservationReflector's learned weights feed in.

    Returns:
        (composite, breakdown) where composite is the sum of weighted
        contributions and breakdown is {term_name: weighted_contribution}.
    """
    breakdown = {}
    total = 0.0
    for term in terms:
        raw = term.fn(context)
        weight = term.weight
        if mode_weights and term.name in mode_weights:
            weight *= mode_weights[term.name]
        contribution = weight * raw
        breakdown[term.name] = round(contribution, 4)
        total += contribution
    return round(total, 4), breakdown


# =============================================================================
# Observation Factory
# =============================================================================

def scored_observation(
    content: str,
    domain: str,
    category: str,
    terms: List[ScoringTerm],
    context: Dict[str, Any],
    entity_ids: Optional[List[str]] = None,
    mode: str = "default",
    mode_weights: Optional[Dict[str, float]] = None,
) -> MemoryItem:
    """Produce a MemoryItem from an additive scoring pipeline.

    Domain-agnostic. The domain module defines the terms;
    this function composes them into a storable observation.
    """
    composite, breakdown = score_terms(terms, context, mode_weights)
    band = band_from_score(composite)
    importance = min(1.0, max(0.0, composite / 100.0))

    return MemoryItem(
        content=content,
        memory_type=domain,
        entity_ids=entity_ids or [],
        properties={
            "band": band,
            "status": "open",
            "composite": composite,
            "breakdown": breakdown,
            "mode": mode,
            "observed_at": datetime.now(timezone.utc).isoformat(),
        },
        importance=importance,
    )


# =============================================================================
# Lifecycle
# =============================================================================

async def advance_observation(
    bus: Any,  # LiteMemoryBus (duck-typed to avoid circular import)
    observation_id: str,
    new_status: str,
    notes: Optional[str] = None,
) -> str:
    """Advance an observation's lifecycle status.

    Updates the original's status property, creates a resolution
    MemoryItem with the new status, and links them with a FOLLOWED_BY
    edge for temporal chaining.

    Args:
        bus: A LiteMemoryBus (or anything with get_item, store, store_edge).
        observation_id: ID of the observation to advance.
        new_status: New status string ("acted", "resolved", "escalated", "reopened").
        notes: Optional notes about the transition.

    Returns:
        The resolution item's ID.

    Raises:
        KeyError: If observation_id not found in the bus.
    """
    original = bus.get_item(observation_id)
    if original is None:
        raise KeyError(f"Observation {observation_id} not found")

    # Update original's status in place
    original.properties["status"] = new_status

    # Create resolution item
    resolution_content = notes or f"Status changed to {new_status}"
    resolution = MemoryItem(
        content=resolution_content,
        memory_type=original.memory_type,
        entity_ids=original.entity_ids,
        properties={
            "band": original.properties.get("band", "P3"),
            "status": new_status,
            "composite": original.properties.get("composite", 0.0),
            "original_id": observation_id,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        },
        importance=original.importance * (0.5 if new_status == "resolved" else 1.0),
    )
    resolution_id = await bus.store(resolution)

    # Temporal chain
    await bus.store_edge(observation_id, resolution_id, "FOLLOWED_BY")

    logger.debug(
        "Advanced observation %s → %s (status=%s)",
        observation_id, resolution_id, new_status,
    )
    return resolution_id
