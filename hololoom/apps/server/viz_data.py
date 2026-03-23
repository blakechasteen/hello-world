"""
Visualization Data Layer — five primitives for structured AI cognition.

Every visualization composes from: Trajectory, Topology, Distribution, Tension.
(Attention is the fifth primitive — it lives in the interaction layer, not the data layer.)

Same data shapes, three renderers (Jenny HTML, React, Glyph).
Renderers don't know the domain — they render primitives.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from hololoom.apps.server import vault_bridge
from hololoom.apps.server.vault_bridge import VAULT_ROOT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/viz", tags=["visualization"])


# ============================================================================
# Helpers
# ============================================================================

def _safe_import(fn):
    """Run fn, return None on any import/runtime failure."""
    try:
        return fn()
    except Exception:
        return None


# ============================================================================
# Department Visualization
# ============================================================================

@router.get("/department/{dept}")
async def department_viz(dept: str):
    """Department knowledge state — all five primitives."""

    # Trajectory: promotion flow
    trajectory = _safe_import(lambda: _dept_trajectory(dept))

    # Topology: belief communities
    topology = _safe_import(lambda: _dept_topology(dept))

    # Distribution: score histogram + rubric weights
    distribution = _safe_import(lambda: _dept_distribution(dept))

    # Tension: drift + phase state
    tension = _safe_import(lambda: _dept_tension(dept))

    # Meta
    meta = _dept_meta(dept)

    return {
        "department": dept,
        "trajectory": trajectory,
        "topology": topology,
        "distribution": distribution,
        "tension": tension,
        "meta": meta,
    }


def _dept_meta(dept: str) -> dict:
    """Basic department metadata."""
    try:
        from hololoom.apps.server.department_head import get_rubric, list_departments
        rubric = get_rubric(dept)
        return {
            "name": dept,
            "n_rubric_terms": len(rubric),
            "rubric_terms": [{"name": t.name, "weight": t.weight, "mechanical": t.mechanical} for t in rubric],
            "exists": dept in list_departments(),
        }
    except Exception:
        return {"name": dept, "exists": False}


def _dept_trajectory(dept: str) -> Optional[dict]:
    """Promotion trajectory — flow of notes through pipeline stages."""
    para_dir = VAULT_ROOT / "departments" / dept / "para"
    inbox_dir = VAULT_ROOT / "departments" / dept / "inbox"

    n_inbox = 0
    if inbox_dir.is_dir():
        for worker_dir in inbox_dir.iterdir():
            if worker_dir.is_dir():
                n_inbox += len(list(worker_dir.glob("*.md")))

    n_para = 0
    if para_dir.is_dir():
        n_para = len(list(para_dir.rglob("*.md")))

    vault_inbox = VAULT_ROOT / "00_Inbox" / f"{dept}-head"
    n_vault = len(list(vault_inbox.glob("*.md"))) if vault_inbox.is_dir() else 0

    return {
        "stages": [
            {"name": "inbox", "count": n_inbox},
            {"name": "para", "count": n_para},
            {"name": "vault_pending", "count": n_vault},
        ],
        "flow": {
            "inbox_to_para": n_para,
            "para_to_vault": n_vault,
            "total_processed": n_inbox + n_para + n_vault,
        },
    }


def _dept_topology(dept: str) -> Optional[dict]:
    """Belief communities and connections."""
    try:
        from hololoom.apps.server.belief_communities import detect_communities
        result = detect_communities(dept)
        if not result:
            return None
        return {
            "n_communities": result.n_communities,
            "algebraic_connectivity": result.algebraic_connectivity,
            "inter_links": result.inter_community_links,
            "intra_links": result.intra_community_links,
            "communities": {
                str(k): {"paths": v, "titles": result.community_titles[k]}
                for k, v in result.communities.items()
            },
        }
    except Exception:
        return None


def _dept_distribution(dept: str) -> Optional[dict]:
    """Score and rubric weight distributions."""
    try:
        from hololoom.apps.server.department_head import get_rubric, get_head_bandit
        rubric = get_rubric(dept)
        bandit = get_head_bandit()
        stats = bandit.stats(dept)

        return {
            "rubric_weights": {t.name: t.weight for t in rubric},
            "bandit_means": {term: s.get("mean", 0.5) for term, s in stats.items()},
            "total_weight": sum(t.weight for t in rubric),
        }
    except Exception:
        return None


def _dept_tension(dept: str) -> Optional[dict]:
    """Drift from vault + emergent phase."""
    result = {}

    # KL drift
    try:
        from hololoom.apps.server.oversight import compute_dept_vault_drift
        drift = compute_dept_vault_drift(dept)
        if drift:
            result["drift"] = drift
    except Exception:
        pass

    # Emergence phase
    try:
        from hololoom.apps.server.physics_integration import analyze_department_emergence
        # Would need embeddings — skip if not available
        result["phase"] = "unknown"
    except Exception:
        pass

    return result if result else None


# ============================================================================
# Cross-Department Visualization
# ============================================================================

@router.get("/cross-departments")
async def cross_dept_viz():
    """Cross-department relationships — topology + distribution + tension."""
    try:
        from hololoom.apps.server.cross_department import cross_department_matrix, unified_knowledge_graph
        matrix = cross_department_matrix()
        graph = unified_knowledge_graph()
        return {
            "topology": {
                "departments": matrix.get("departments", []),
                "comparisons": matrix.get("comparisons", []),
                "unified_graph": graph,
            },
            "distribution": {
                "n_departments": len(matrix.get("departments", [])),
            },
            "tension": {
                "most_connected": min(matrix.get("comparisons", [{}]), key=lambda c: c.get("distance", 1), default={}).get("dept_a", "none") if matrix.get("comparisons") else "none",
                "most_isolated": max(matrix.get("comparisons", [{}]), key=lambda c: c.get("distance", 0), default={}).get("dept_b", "none") if matrix.get("comparisons") else "none",
            },
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Feedback Visualization
# ============================================================================

@router.get("/feedback")
async def feedback_viz():
    """Learning state — temporal layers."""
    result = {"trajectory": {}, "distribution": {}, "tension": {}, "temporal_layers": {}}

    # Bandit arms distribution
    try:
        from hololoom.apps.server.department_head import get_head_bandit, list_departments
        bandit = get_head_bandit()
        all_stats = {}
        for dept in list_departments():
            stats = bandit.stats(dept)
            if stats:
                all_stats[dept] = stats
        result["distribution"]["bandit_arms"] = all_stats
    except Exception:
        pass

    # Loom learner stats
    try:
        from hololoom.core.warp.math.loom import get_loom_learner
        learner = get_loom_learner()
        result["distribution"]["loom_arms"] = learner.stats_summary()
        result["trajectory"]["n_loom_arms"] = len(learner.arms)
    except Exception:
        pass

    # FeedbackHub summary
    try:
        from hololoom.core.reflection.feedback_hub import FeedbackHub
        # Hub is a singleton per server — would need to access the running instance
        result["temporal_layers"] = {
            "sub_second": 0.05,
            "seconds": 0.15,
            "minutes": 0.30,
            "hours": 0.60,
            "days": 0.85,
            "weeks": 1.00,
        }
    except Exception:
        pass

    return result


# ============================================================================
# Math Activity Visualization
# ============================================================================

@router.get("/math-activity")
async def math_activity_viz():
    """Math module activity — which fired, what helped."""
    result = {"distribution": {}, "trajectory": {}, "budget": {}}

    # Catalog stats
    try:
        from hololoom.core.warp.math.catalog import get_catalog
        catalog = get_catalog()
        from collections import Counter
        domains = Counter(op.domain.value for op in catalog.values())
        levels = Counter(op.level.value for op in catalog.values())
        result["distribution"] = {
            "by_domain": dict(domains),
            "by_level": dict(levels),
            "total_operations": len(catalog),
        }
    except Exception:
        pass

    # Budget allocation
    try:
        from hololoom.core.warp.math.budget import BudgetTier
        result["budget"] = {
            "tiers": {t.name: t.value for t in BudgetTier},
        }
    except Exception:
        pass

    # Math status across all integration layers
    try:
        from hololoom.apps.server.math_extensions_v5 import math_extensions_v5_status
        v5 = math_extensions_v5_status()
        available = sum(1 for v in v5.values() if v)
        result["trajectory"]["v5_available"] = available
        result["trajectory"]["v5_total"] = len(v5)
    except Exception:
        pass

    return result


# ============================================================================
# Full System Overview
# ============================================================================

@router.get("/overview")
async def system_overview():
    """Complete system overview — all primitives for all domains."""
    from hololoom.apps.server.department_head import list_departments

    depts = list_departments()
    return {
        "departments": depts,
        "n_departments": len(depts),
        "endpoints": {
            "per_department": f"/viz/department/{{dept}}",
            "cross_department": "/viz/cross-departments",
            "feedback": "/viz/feedback",
            "math": "/viz/math-activity",
        },
        "primitives": ["trajectory", "topology", "distribution", "tension", "attention"],
        "renderers": ["jenny_html", "react", "glyph"],
    }


# ============================================================================
# Feedback Signal Endpoint (Attention = fifth primitive)
# ============================================================================

class FeedbackSignalRequest(BaseModel):
    """Interaction feedback from visualization panels."""
    subject: str
    dimension: str
    timescale: str  # SUB_SECOND, SECONDS, MINUTES, HOURS, DAYS, WEEKS
    value: float
    weight: float = 0.15
    timestamp: Optional[str] = None


@router.post("/feedback/signal")
async def receive_feedback_signal(signal: FeedbackSignalRequest):
    """Receive a feedback signal from visualization interaction.

    Every click, hover, pin, and dismiss IS a feedback signal.
    The visualization IS the feedback collection surface.
    """
    try:
        from hololoom.core.protocols.feedback import FeedbackSignal, Timescale
        from hololoom.core.reflection.feedback_hub import FeedbackHub

        timescale_map = {
            "SUB_SECOND": Timescale.SUB_SECOND,
            "SECONDS": Timescale.SECONDS,
            "MINUTES": Timescale.MINUTES,
            "HOURS": Timescale.HOURS,
            "DAYS": Timescale.DAYS,
            "WEEKS": Timescale.WEEKS,
        }
        ts = timescale_map.get(signal.timescale, Timescale.SECONDS)

        fb_signal = FeedbackSignal(
            source="visualization",
            subject=signal.subject,
            dimension=signal.dimension,
            value=signal.value,
            confidence=abs(signal.value),
            timescale=ts,
        )

        hub = FeedbackHub.get_instance()
        hub.submit(fb_signal)

        logger.info(f"Feedback: {signal.subject}/{signal.dimension} = {signal.value} ({signal.timescale})")
        return {"status": "accepted", "timescale": signal.timescale, "effective_weight": fb_signal.effective_weight}
    except ImportError:
        # FeedbackHub not available — log and accept
        logger.info(f"Feedback (no hub): {signal.subject}/{signal.dimension} = {signal.value}")
        return {"status": "logged", "timescale": signal.timescale}


# ============================================================================
# Physics Coupling Visualization
# ============================================================================

@router.get("/physics")
async def physics_viz():
    """Coupled physics state — trajectory + distribution + tension."""
    try:
        from hololoom.physics.coupling import get_coupler, WaveHierarchy
        coupler = get_coupler()
        status = coupler.status()

        return {
            "trajectory": {
                "energy_trend": status["energy_trend"],
                "order_trend": status["order_trend"],
                "total_steps": status["total_steps"],
                "avg_step_ms": status["avg_step_ms"],
                "render_as": "sparkline",
            },
            "distribution": {
                "meta_thermo": status["meta_thermo"],
                "phases": status["phases"],
                "n_phases": status["n_phases"],
                "render_as": "bar",
            },
            "tension": {
                "temperature": status["state"]["temperature"],
                "entropy": status["state"]["entropy"],
                "free_energy": status["state"]["free_energy"],
                "order_parameter": status["state"]["order_parameter"],
                "phase_type": status["state"]["phase_type"],
                "render_as": "gauge",
            },
            "state": status["state"],
        }
    except ImportError:
        return {"error": "Physics coupling not available"}
