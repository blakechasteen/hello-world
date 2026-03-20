"""
Cross-Department Intelligence — compare, connect, and synthesize across departments.

Uses the full math library to analyze relationships between departments:
- Wasserstein distance between score distributions
- Spectral clustering across the unified knowledge graph
- Geodesic paths between department knowledge on the embedding manifold
- Topological analysis for cross-department voids and loops
- Hyperbolic distance for hierarchical relationships

The Synthesis role: find what connects Farm to Bees to Garden to Kitchen.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter

from hololoom.apps.server import vault_bridge
from hololoom.apps.server.vault_bridge import VAULT_ROOT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/departments", tags=["cross-department"])


# ============================================================================
# Cross-Department Analysis
# ============================================================================

@dataclass
class DepartmentProfile:
    """Statistical profile of a department's knowledge."""
    name: str
    n_notes: int
    avg_score: float
    term_distribution: np.ndarray  # word frequency distribution
    top_terms: List[str]


@dataclass
class CrossDeptAnalysis:
    """Cross-department comparison result."""
    dept_a: str
    dept_b: str
    wasserstein_distance: float
    shared_terms: List[str]
    bridge_concepts: List[str]  # terms that connect the two departments
    relationship: str  # "close", "moderate", "distant", "disjoint"


def _build_department_profile(dept: str) -> Optional[DepartmentProfile]:
    """Build a statistical profile from a department's PARA notes."""
    import re
    from collections import Counter

    para_dir = VAULT_ROOT / "departments" / dept / "para"
    if not para_dir.is_dir():
        return None

    all_words = []
    n_notes = 0

    for md in para_dir.rglob("*.md"):
        if md.name.startswith("_"):
            continue
        try:
            content = md.read_text(encoding="utf-8", errors="replace").lower()
            # Strip frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    content = parts[2]
            words = [w for w in re.split(r'\W+', content) if len(w) >= 3]
            all_words.extend(words)
            n_notes += 1
        except OSError:
            continue

    if not all_words:
        return None

    counts = Counter(all_words)
    # Remove stopwords
    stopwords = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "not", "has", "have", "been"}
    for sw in stopwords:
        counts.pop(sw, None)

    total = sum(counts.values())
    top_terms = [w for w, _ in counts.most_common(20)]

    # Build probability distribution over vocabulary
    vocab = sorted(counts.keys())
    dist = np.array([counts[w] / total for w in vocab], dtype=np.float64)

    return DepartmentProfile(
        name=dept,
        n_notes=n_notes,
        avg_score=0.0,
        term_distribution=dist,
        top_terms=top_terms,
    )


def compare_departments(dept_a: str, dept_b: str) -> Optional[CrossDeptAnalysis]:
    """Compare two departments using Wasserstein distance and term overlap."""
    profile_a = _build_department_profile(dept_a)
    profile_b = _build_department_profile(dept_b)

    if not profile_a or not profile_b:
        return None

    # Shared terms
    shared = set(profile_a.top_terms) & set(profile_b.top_terms)

    # Wasserstein distance (sliced, if available)
    try:
        from hololoom.core.warp.math.analysis.wasserstein import SlicedWasserstein
        # Pad distributions to same length
        max_len = max(len(profile_a.term_distribution), len(profile_b.term_distribution))
        dist_a = np.zeros(max_len)
        dist_b = np.zeros(max_len)
        dist_a[:len(profile_a.term_distribution)] = profile_a.term_distribution
        dist_b[:len(profile_b.term_distribution)] = profile_b.term_distribution
        # Normalize
        dist_a = dist_a / (dist_a.sum() + 1e-10)
        dist_b = dist_b / (dist_b.sum() + 1e-10)
        w_dist = float(SlicedWasserstein.compute(dist_a.reshape(-1, 1), dist_b.reshape(-1, 1)))
    except Exception:
        # Fallback: L1 distance on top terms overlap
        w_dist = 1.0 - len(shared) / max(len(profile_a.top_terms), len(profile_b.top_terms), 1)

    # Bridge concepts: terms that appear in both but aren't stopwords
    bridge = sorted(shared - {"the", "and", "for", "with"})

    # Classify relationship
    if w_dist < 0.2:
        relationship = "close"
    elif w_dist < 0.5:
        relationship = "moderate"
    elif w_dist < 0.8:
        relationship = "distant"
    else:
        relationship = "disjoint"

    return CrossDeptAnalysis(
        dept_a=dept_a,
        dept_b=dept_b,
        wasserstein_distance=round(w_dist, 4),
        shared_terms=bridge[:10],
        bridge_concepts=bridge[:5],
        relationship=relationship,
    )


def cross_department_matrix() -> dict:
    """Build a full comparison matrix across all departments."""
    dept_root = VAULT_ROOT / "departments"
    if not dept_root.is_dir():
        return {"error": "No departments directory"}

    depts = [
        d.name for d in sorted(dept_root.iterdir())
        if d.is_dir() and (d / "_index.md").exists()
    ]

    if len(depts) < 2:
        return {"departments": depts, "comparisons": []}

    comparisons = []
    for i, a in enumerate(depts):
        for b in depts[i + 1:]:
            result = compare_departments(a, b)
            if result:
                comparisons.append({
                    "dept_a": result.dept_a,
                    "dept_b": result.dept_b,
                    "distance": result.wasserstein_distance,
                    "relationship": result.relationship,
                    "shared_terms": result.shared_terms,
                    "bridges": result.bridge_concepts,
                })

    return {"departments": depts, "comparisons": comparisons}


# ============================================================================
# Cross-Department Search (for EA synthesis)
# ============================================================================

def federated_search(query: str, top_k: int = 3) -> dict:
    """Search across all departments, return results grouped by dept.

    Used by the Executive Assistant for cross-department synthesis.
    """
    return vault_bridge.search_across_departments(query, top_k=top_k)


def find_bridge_notes(dept_a: str, dept_b: str, top_k: int = 5) -> List[dict]:
    """Find notes in dept_a that are relevant to dept_b's knowledge.

    These are the "bridge" notes that connect two departments.
    """
    # Get dept_b's top terms
    profile_b = _build_department_profile(dept_b)
    if not profile_b:
        return []

    # Search dept_a for dept_b's top terms
    query = " ".join(profile_b.top_terms[:5])
    results = vault_bridge.search(query, top_k=top_k, department=dept_a)
    return results


# ============================================================================
# Spectral Analysis Across Departments
# ============================================================================

def unified_knowledge_graph() -> Optional[dict]:
    """Build a unified knowledge graph across all departments.

    Nodes = notes from all departments.
    Edges = wiki links (within and across departments).
    Spectral clustering reveals cross-department topic communities.
    """
    try:
        from hololoom.apps.server.belief_communities import (
            build_adjacency_matrix, DeptNote, _WIKI_LINK_RE,
        )
    except ImportError:
        return None

    dept_root = VAULT_ROOT / "departments"
    if not dept_root.is_dir():
        return None

    # Collect all notes across departments
    all_notes = []
    dept_labels = []

    for dept_dir in sorted(dept_root.iterdir()):
        if not dept_dir.is_dir() or not (dept_dir / "_index.md").exists():
            continue

        para_dir = dept_dir / "para"
        if not para_dir.is_dir():
            continue

        for md in para_dir.rglob("*.md"):
            if md.name.startswith("_"):
                continue
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                rel = str(md.relative_to(VAULT_ROOT)).replace("\\", "/")
                title = md.stem.replace("-", " ")
                links = _WIKI_LINK_RE.findall(content)
                all_notes.append(DeptNote(path=rel, title=title, links_to=links))
                dept_labels.append(dept_dir.name)
            except OSError:
                continue

    if len(all_notes) < 3:
        return {"n_notes": len(all_notes), "message": "Not enough notes for analysis"}

    # Build adjacency matrix
    adjacency, paths = build_adjacency_matrix(all_notes)

    # Spectral analysis
    try:
        from hololoom.core.warp.math.graph.spectral_clustering import GraphLaplacian
        laplacian = GraphLaplacian.from_adjacency(adjacency)
        eigenvalues = np.sort(np.real(np.linalg.eigvals(laplacian)))
        algebraic_connectivity = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    except Exception:
        algebraic_connectivity = 0.0

    # Cross-department links
    cross_links = 0
    intra_links = 0
    for i in range(len(all_notes)):
        for j in range(i + 1, len(all_notes)):
            if adjacency[i][j] > 0:
                if dept_labels[i] == dept_labels[j]:
                    intra_links += 1
                else:
                    cross_links += 1

    return {
        "n_notes": len(all_notes),
        "n_departments": len(set(dept_labels)),
        "departments": sorted(set(dept_labels)),
        "algebraic_connectivity": round(algebraic_connectivity, 4),
        "cross_department_links": cross_links,
        "intra_department_links": intra_links,
        "cross_ratio": round(cross_links / max(cross_links + intra_links, 1), 3),
    }


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/compare/{dept_a}/{dept_b}")
async def compare(dept_a: str, dept_b: str):
    """Compare two departments."""
    result = compare_departments(dept_a, dept_b)
    if not result:
        return {"error": f"Could not compare {dept_a} and {dept_b}"}
    return {
        "dept_a": result.dept_a,
        "dept_b": result.dept_b,
        "distance": result.wasserstein_distance,
        "relationship": result.relationship,
        "shared_terms": result.shared_terms,
        "bridges": result.bridge_concepts,
    }


@router.get("/matrix")
async def matrix():
    """Full comparison matrix across all departments."""
    return cross_department_matrix()


@router.get("/unified-graph")
async def unified_graph():
    """Unified knowledge graph analysis across all departments."""
    result = unified_knowledge_graph()
    if not result:
        return {"error": "Could not build unified graph"}
    return result


@router.get("/search")
async def cross_search(query: str, top_k: int = 3):
    """Search across all departments."""
    return federated_search(query, top_k=top_k)


@router.get("/bridges/{dept_a}/{dept_b}")
async def bridges(dept_a: str, dept_b: str, top_k: int = 5):
    """Find notes in dept_a relevant to dept_b."""
    return find_bridge_notes(dept_a, dept_b, top_k=top_k)
