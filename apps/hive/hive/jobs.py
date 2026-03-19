"""Hive job definitions — what each cron task does.

Each job is an async function that runs a specific check over hive data,
scores it via the additive scoring pipeline, and stores findings in the bus.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from hololoom.core.memory.scored_observation import ScoringTerm, scored_observation, advance_observation
from hololoom.core.memory.observation_reflector import ObservationReflector
from hololoom.core.memory.bus import MemoryQuery

from .bus_client import HiveBusClient
from .models import Hive, HiveInspection, load_hives
from .terms import HIVE_TERMS, inspection_to_context
from .modes import detect_mode, get_mode_weights

logger = logging.getLogger(__name__)


async def job_inspect(
    client: HiveBusClient,
    reflector: ObservationReflector,
    hives: List[Hive],
    mode: Optional[str] = None,
    hive_id: Optional[str] = None,
    dry_run: bool = False,
) -> List[Dict]:
    """Score all hives (or one) and store findings.

    This is the primary job — called by `hive check`.
    Returns list of {hive_id, band, composite, content, node_id}.
    """
    mode = mode or detect_mode()
    mode_weights = get_mode_weights(mode)

    # Blend learned adjustments from reflector
    learned = reflector.suggest_weight_adjustments(mode, HIVE_TERMS)
    combined_weights = {**learned}
    for k, v in mode_weights.items():
        combined_weights[k] = combined_weights.get(k, 1.0) * v

    results = []
    targets = [h for h in hives if hive_id is None or h.id == hive_id]

    for hive in targets:
        inspection = hive.latest_inspection
        if inspection is None:
            logger.warning("Hive %s has no inspections, skipping", hive.id)
            continue

        ctx = inspection_to_context(inspection)
        obs = scored_observation(
            content=_build_content(hive, inspection),
            domain="hive",
            category="inspection",
            terms=HIVE_TERMS,
            context=ctx,
            entity_ids=[hive.id],
            mode=mode,
            mode_weights=combined_weights,
        )

        node_id = None
        if not dry_run:
            node_id = await client.store(obs)

        results.append({
            "hive_id": hive.id,
            "hive_name": hive.name,
            "band": obs.properties["band"],
            "composite": obs.properties["composite"],
            "content": obs.content,
            "node_id": node_id,
            "mode": mode,
        })

    return results


async def job_varroa(
    client: HiveBusClient,
    reflector: ObservationReflector,
    hives: List[Hive],
) -> List[Dict]:
    """Monthly varroa monitoring — flags hives due for sugar roll/alcohol wash."""
    results = []
    for hive in hives:
        inspection = hive.latest_inspection
        if inspection is None:
            continue
        ctx = inspection_to_context(inspection)
        # Varroa-focused scoring: only varroa term matters
        from hololoom.core.memory.scored_observation import score_terms
        composite, breakdown = score_terms(
            [t for t in HIVE_TERMS if t.name == "w_varroa"],
            ctx,
        )
        if composite > 0:
            obs = scored_observation(
                content=f"Varroa check: {hive.name} ({hive.id}) — count {inspection.varroa_count}/100",
                domain="hive",
                category="varroa",
                terms=[t for t in HIVE_TERMS if t.name == "w_varroa"],
                context=ctx,
                entity_ids=[hive.id],
                mode="varroa_check",
            )
            node_id = await client.store(obs)
            results.append({"hive_id": hive.id, "band": obs.properties["band"], "node_id": node_id})
    return results


async def job_stores(
    client: HiveBusClient,
    reflector: ObservationReflector,
    hives: List[Hive],
) -> List[Dict]:
    """Check honey/pollen stores against seasonal needs."""
    mode = detect_mode()
    results = []
    for hive in hives:
        inspection = hive.latest_inspection
        if inspection is None:
            continue
        ctx = inspection_to_context(inspection)
        stores_terms = [t for t in HIVE_TERMS if t.name == "w_stores"]
        mode_weights = get_mode_weights(mode)
        obs = scored_observation(
            content=f"Stores check: {hive.name} ({hive.id}) — {inspection.honey_stores}",
            domain="hive",
            category="stores",
            terms=stores_terms,
            context=ctx,
            entity_ids=[hive.id],
            mode=mode,
            mode_weights=mode_weights,
        )
        if obs.properties["composite"] > 0:
            node_id = await client.store(obs)
            results.append({"hive_id": hive.id, "band": obs.properties["band"], "node_id": node_id})
    return results


async def job_treatment_window(
    client: HiveBusClient,
    reflector: ObservationReflector,
    hives: List[Hive],
) -> List[Dict]:
    """Check if temperature conditions allow varroa treatment.

    Stub: returns a reminder. Future: integrate weather API for temp check.
    """
    # Fixed-score term: always fires at a set level (reminder pattern)
    reminder_term = ScoringTerm("w_treatment_reminder", 20, lambda ctx: 1.0)
    obs = scored_observation(
        content="Treatment window check — verify outdoor temps 35-85F for oxalic acid",
        domain="hive",
        category="treatment",
        terms=[reminder_term],
        context={},
        mode="treatment",
    )
    node_id = await client.store(obs)
    return [{"type": "treatment_reminder", "band": obs.properties["band"], "node_id": node_id}]


async def job_harvest(
    client: HiveBusClient,
    reflector: ObservationReflector,
    hives: List[Hive],
) -> List[Dict]:
    """Check harvest readiness (heavy stores = ready to extract)."""
    harvest_term = ScoringTerm("w_harvest_ready", 25, lambda ctx: 1.0 if ctx.get("heavy") else 0.0)
    results = []
    for hive in hives:
        inspection = hive.latest_inspection
        if inspection is None:
            continue
        if inspection.honey_stores == "heavy":
            obs = scored_observation(
                content=f"Harvest ready: {hive.name} ({hive.id}) — stores heavy",
                domain="hive",
                category="harvest",
                terms=[harvest_term],
                context={"heavy": True},
                entity_ids=[hive.id],
                mode="harvest",
            )
            node_id = await client.store(obs)
            results.append({"hive_id": hive.id, "band": obs.properties["band"], "node_id": node_id})
    return results


async def job_equipment(
    client: HiveBusClient,
    reflector: ObservationReflector,
    hives: List[Hive],
) -> List[Dict]:
    """Seasonal equipment prep reminders."""
    mode = detect_mode()
    equipment_map = {
        "spring": "Prep: assemble frames, check bottom boards, order packages/nucs",
        "nectar_flow": "Prep: supers ready, queen excluders, extraction equipment",
        "fall": "Prep: entrance reducers, mouse guards, winter wraps/insulation",
        "winter": "Prep: check ventilation, emergency fondant, repair equipment",
    }
    content = equipment_map.get(mode, "Check equipment for current season")
    equipment_term = ScoringTerm("w_equipment_reminder", 10, lambda ctx: 1.0)
    obs = scored_observation(
        content=f"Equipment ({mode}): {content}",
        domain="hive",
        category="equipment",
        terms=[equipment_term],
        context={},
        mode=mode,
    )
    node_id = await client.store(obs)
    return [{"type": "equipment_reminder", "mode": mode, "band": obs.properties["band"], "node_id": node_id}]


# =============================================================================
# Helpers
# =============================================================================

def _build_content(hive: Hive, inspection: HiveInspection) -> str:
    """Build human-readable content string for a scored observation."""
    parts = [f"{hive.name} ({hive.id})"]
    if inspection.varroa_count > 0:
        parts.append(f"varroa {inspection.varroa_count}/100")
    if not inspection.queen_seen:
        parts.append("queen NOT seen")
    if not inspection.eggs_seen:
        parts.append("no eggs")
    if inspection.brood_pattern != "solid":
        parts.append(f"brood {inspection.brood_pattern}")
    parts.append(f"stores {inspection.honey_stores}")
    if inspection.queen_cells > 0:
        parts.append(f"{inspection.queen_cells} queen cells")
    if inspection.disease_signs:
        parts.append(f"disease: {', '.join(inspection.disease_signs)}")
    return " | ".join(parts)


# Job registry for CLI dispatch
JOB_REGISTRY = {
    "inspect": job_inspect,
    "varroa": job_varroa,
    "stores": job_stores,
    "treatment": job_treatment_window,
    "harvest": job_harvest,
    "equipment": job_equipment,
}
