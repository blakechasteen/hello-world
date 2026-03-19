"""
Belief Communities — spectral clustering on department knowledge graphs.

Builds a graph from department PARA notes (nodes = notes, edges = wiki links),
then uses spectral clustering to detect coherent topic clusters.

Reports community health via algebraic connectivity (Fiedler value):
  High connectivity = well-connected, integrated knowledge
  Low connectivity = isolated clusters forming

Math: Graph Laplacian L = D - A, eigenvalues reveal structure.
Fiedler vector (eigenvector of lambda_2) gives optimal 2-way partition.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import APIRouter

from hololoom.apps.server.vault_bridge import VAULT_ROOT

logger = logging.getLogger(__name__)

# Graceful import of spectral clustering
try:
    from hololoom.core.warp.math.graph.spectral_clustering import (
        GraphLaplacian,
        SpectralEmbedding,
    )
    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False
    logger.info("Spectral clustering unavailable — belief communities disabled")

router = APIRouter(prefix="/department", tags=["belief-communities"])

_WIKI_LINK_RE = re.compile(r'\[\[([^\]]+)\]\]')


# ============================================================================
# Graph Construction
# ============================================================================

@dataclass
class DeptNote:
    """A note in the department knowledge graph."""
    path: str           # relative to VAULT_ROOT
    title: str          # extracted from heading or filename
    links_to: list[str] # wiki link targets found in content


def _extract_notes(dept: str) -> list[DeptNote]:
    """Scan department PARA for notes with their wiki links."""
    para_dir = VAULT_ROOT / "departments" / dept / "para"
    if not para_dir.is_dir():
        return []

    notes = []
    for md in para_dir.rglob("*.md"):
        if md.name.startswith("_"):
            continue
        try:
            content = md.read_text(encoding="utf-8", errors="replace")
            rel = str(md.relative_to(VAULT_ROOT)).replace("\\", "/")

            # Extract title
            title = md.stem.replace("-", " ")
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("# ") and not line.startswith("##"):
                    title = line[2:].strip()
                    break

            # Extract wiki links
            links = _WIKI_LINK_RE.findall(content)

            notes.append(DeptNote(path=rel, title=title, links_to=links))
        except OSError:
            continue

    return notes


def build_adjacency_matrix(notes: list[DeptNote]) -> tuple:
    """Build adjacency matrix from wiki links between notes.

    Returns (adjacency_matrix, note_paths) where adjacency[i][j] > 0
    if note i links to note j.
    """
    paths = [n.path for n in notes]
    n = len(paths)
    if n == 0:
        return np.zeros((0, 0)), []

    # Build lookup: various forms of note reference → index
    path_to_idx = {}
    for i, note in enumerate(notes):
        path_to_idx[note.path] = i
        path_to_idx[note.title.lower()] = i
        # Also index by filename stem
        stem = Path(note.path).stem.replace("-", " ").lower()
        path_to_idx[stem] = i

    adjacency = np.zeros((n, n), dtype=np.float64)

    for i, note in enumerate(notes):
        for link in note.links_to:
            link_lower = link.lower().strip()
            # Try exact path, then stem, then title
            j = path_to_idx.get(link_lower)
            if j is None:
                # Try matching by stem
                link_stem = link_lower.split("/")[-1].replace("-", " ")
                j = path_to_idx.get(link_stem)
            if j is not None and j != i:
                adjacency[i][j] += 1.0
                adjacency[j][i] += 1.0  # undirected

    return adjacency, paths


# ============================================================================
# Community Detection
# ============================================================================

@dataclass
class CommunityResult:
    """Result of belief community detection."""
    n_communities: int
    communities: dict[int, list[str]]  # cluster_id → [note_paths]
    community_titles: dict[int, list[str]]  # cluster_id → [note_titles]
    algebraic_connectivity: float  # Fiedler value (graph health)
    inter_community_links: int
    intra_community_links: int
    note_count: int


def detect_communities(dept: str, n_clusters: int | None = None) -> CommunityResult | None:
    """Detect belief communities in a department's knowledge graph.

    Uses spectral clustering on the wiki-link adjacency matrix.
    If n_clusters is None, auto-detects from eigenvalue gap.
    """
    if not HAS_SPECTRAL:
        return None

    notes = _extract_notes(dept)
    if len(notes) < 2:
        return CommunityResult(
            n_communities=1 if notes else 0,
            communities={0: [n.path for n in notes]} if notes else {},
            community_titles={0: [n.title for n in notes]} if notes else {},
            algebraic_connectivity=0.0,
            inter_community_links=0,
            intra_community_links=0,
            note_count=len(notes),
        )

    adjacency, paths = build_adjacency_matrix(notes)

    # Compute Laplacian and spectral embedding
    laplacian = GraphLaplacian.from_adjacency(adjacency)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

    # Algebraic connectivity = second smallest eigenvalue
    alg_connectivity = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

    # Auto-detect number of clusters from eigenvalue gap
    if n_clusters is None:
        # Find largest gap in sorted eigenvalues (up to 5 clusters)
        max_k = min(5, len(eigenvalues) - 1)
        if max_k < 1:
            n_clusters = 1
        else:
            gaps = [eigenvalues[i + 1] - eigenvalues[i] for i in range(max_k)]
            n_clusters = int(np.argmax(gaps)) + 1
            n_clusters = max(1, min(n_clusters, len(notes) // 2))

    # Simple k-means on spectral embedding
    if n_clusters <= 1:
        labels = np.zeros(len(notes), dtype=int)
    else:
        # Use first k eigenvectors as embedding
        embedding = eigenvectors[:, :n_clusters]
        labels = _simple_kmeans(embedding, n_clusters)

    # Build community result
    communities = {}
    community_titles = {}
    title_lookup = {n.path: n.title for n in notes}

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        member_paths = [paths[i] for i in range(len(paths)) if mask[i]]
        communities[cluster_id] = member_paths
        community_titles[cluster_id] = [title_lookup.get(p, p) for p in member_paths]

    # Count inter vs intra community links
    inter = 0
    intra = 0
    for i in range(len(notes)):
        for j in range(i + 1, len(notes)):
            if adjacency[i][j] > 0:
                if labels[i] == labels[j]:
                    intra += int(adjacency[i][j])
                else:
                    inter += int(adjacency[i][j])

    return CommunityResult(
        n_communities=n_clusters,
        communities=communities,
        community_titles=community_titles,
        algebraic_connectivity=round(alg_connectivity, 4),
        inter_community_links=inter,
        intra_community_links=intra,
        note_count=len(notes),
    )


def _simple_kmeans(data: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
    """Simple k-means clustering. Returns labels array."""
    n = data.shape[0]
    if n <= k:
        return np.arange(n)

    # Initialize centroids with k-means++
    centroids = [data[np.random.randint(n)]]
    for _ in range(1, k):
        dists = np.array([min(np.linalg.norm(x - c) for c in centroids) for x in data])
        probs = dists ** 2
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n) / n
        centroids.append(data[np.random.choice(n, p=probs)])
    centroids = np.array(centroids)

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        new_labels = np.array([
            np.argmin([np.linalg.norm(x - c) for c in centroids])
            for x in data
        ])
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update centroids
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = data[mask].mean(axis=0)

    return labels


def community_distances(result: CommunityResult, dept: str) -> dict[str, float]:
    """Compute hyperbolic distances between belief communities.

    Uses Poincare ball distance if available, Euclidean otherwise.
    """
    try:
        from hololoom.apps.server.math_extensions import (
            hyperbolic_community_distance,
            hyperbolic_hierarchy_depth,
        )
    except ImportError:
        return {}

    # Get embeddings for each community
    try:
        import sys
        sys.path.insert(0, str(VAULT_ROOT))
        from services.data_service import data_service
        if not data_service.vector_store:
            return {}

        community_embeddings = {}
        for cid, paths in result.communities.items():
            embs = []
            for p in paths:
                for store_path, emb in data_service.vector_store.items():
                    if p in str(store_path).replace("\\", "/"):
                        embs.append(emb)
                        break
            if embs:
                community_embeddings[cid] = np.array(embs)

        if len(community_embeddings) < 2:
            return {}

        distances = {}
        cids = sorted(community_embeddings.keys())
        for i, c1 in enumerate(cids):
            for c2 in cids[i + 1:]:
                d = hyperbolic_community_distance(
                    community_embeddings[c1], community_embeddings[c2],
                )
                distances[f"c{c1}_c{c2}"] = round(d, 4)

        # Add hierarchy depth
        all_embs = np.vstack(list(community_embeddings.values()))
        distances["hierarchy_depth"] = round(hyperbolic_hierarchy_depth(all_embs), 4)

        return distances

    except Exception:
        return {}


def community_context(dept: str) -> str:
    """Generate community context string for Head/EA evaluation prompts."""
    result = detect_communities(dept)
    if result is None or result.note_count < 2:
        return ""

    lines = [
        f"\n## Belief Communities ({result.note_count} notes, {result.n_communities} communities)\n",
        f"Algebraic connectivity: {result.algebraic_connectivity} "
        f"({'well-connected' if result.algebraic_connectivity > 0.5 else 'weakly-connected' if result.algebraic_connectivity > 0.1 else 'fragmented'})\n",
        f"Links: {result.intra_community_links} within communities, {result.inter_community_links} between communities\n",
    ]
    for cid, titles in result.community_titles.items():
        lines.append(f"- Community {cid}: {', '.join(titles[:5])}" +
                      (f" (+{len(titles)-5} more)" if len(titles) > 5 else ""))

    return "\n".join(lines)


# ============================================================================
# Endpoint
# ============================================================================

@router.get("/{dept}/topology")
async def topology_audit(dept: str):
    """Run topological analysis on department embedding space."""
    try:
        from hololoom.apps.server.math_extensions import audit_department_topology
        result = audit_department_topology(dept)
        if result is None:
            return {"error": "Topology audit unavailable (not enough embeddings or TDA not installed)"}
        return {
            "dept": dept,
            "n_components": result.n_components,
            "n_loops": result.n_loops,
            "n_voids": result.n_voids,
            "max_persistence_h0": result.max_persistence_h0,
            "max_persistence_h1": result.max_persistence_h1,
            "health": result.health,
        }
    except ImportError:
        return {"error": "Math extensions not available"}


@router.get("/{dept}/communities")
async def get_communities(dept: str, n_clusters: int | None = None):
    """Detect belief communities in a department's knowledge graph."""
    if not HAS_SPECTRAL:
        return {"error": "Spectral clustering unavailable (missing dependency)"}

    result = detect_communities(dept, n_clusters)
    if result is None:
        return {"error": "Could not detect communities"}

    return {
        "dept": dept,
        "n_communities": result.n_communities,
        "note_count": result.note_count,
        "algebraic_connectivity": result.algebraic_connectivity,
        "inter_community_links": result.inter_community_links,
        "intra_community_links": result.intra_community_links,
        "communities": {
            str(k): {"paths": v, "titles": result.community_titles[k]}
            for k, v in result.communities.items()
        },
    }
