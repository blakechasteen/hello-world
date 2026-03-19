"""
Hierarchy Manager — Multi-resolution memory with community detection.

Builds hierarchical summaries over the knowledge graph:
  L0: Raw nodes (entity, factual, episodic) — already in the graph
  L1: Cluster summaries (5-10 nodes merged) — created by this module
  L2: Topic summaries (multiple L1 clusters merged) — created by this module

Summaries are graph nodes with SUMMARIZES edges connecting them to sources.
PPR naturally reaches summaries through these edges, so they participate
in retrieval without special handling.

Community detection uses lightweight pure-graph approaches:
  - Connected components via BFS
  - Within components, group by shared entity connections (co-occurrence)
  - No external dependencies (no Louvain/Leiden)

Extractive summarization (no LLM needed):
  - Key sentence extraction by TF-IDF-like scoring
  - Preserves factual content without hallucination
  - Optional LLM summarization when available

No external dependencies beyond the graph backend.
"""

import logging
import math
import uuid
from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HierarchyConfig:
    """Configuration for hierarchy building."""
    min_cluster_size: int = 3          # Minimum nodes to form a cluster
    max_cluster_size: int = 15         # Maximum nodes in a cluster
    l1_max_clusters: int = 50          # Max L1 clusters to create
    l2_min_l1_clusters: int = 3        # Min L1 clusters to form an L2
    extractive_sentences: int = 3      # Sentences per extractive summary
    summary_importance_boost: float = 1.2  # Importance multiplier for summaries


@dataclass
class ClusterInfo:
    """Information about a detected cluster."""
    cluster_id: str
    node_ids: list[str]
    level: int  # 0=raw nodes, 1=L1 cluster, 2=L2 topic
    summary_node_id: str | None = None
    summary_text: str | None = None


class HierarchyManager:
    """Multi-resolution memory hierarchy.

    Builds summary layers over the knowledge graph. Summaries are stored
    as graph nodes with SUMMARIZES edges, so PPR traversal naturally
    includes them at appropriate resolution.

    Usage:
        mgr = HierarchyManager(graph=LiteBusGraph(bus))
        clusters = mgr.detect_communities()
        summaries = await mgr.build_hierarchy(bus)
        # Now PPR can reach summary nodes via SUMMARIZES edges
    """

    def __init__(
        self,
        graph,  # GraphBackend
        config: HierarchyConfig | None = None,
    ):
        self.graph = graph
        self.config = config or HierarchyConfig()

    # =========================================================================
    # Community Detection (pure graph, no external deps)
    # =========================================================================

    def detect_communities(self) -> list[ClusterInfo]:
        """Detect communities using connected components + co-occurrence clustering.

        Algorithm:
          1. Find connected components via BFS
          2. Within large components, split by entity co-occurrence
          3. Return clusters of appropriate size
        """
        all_nodes = set(self.graph.nodes())
        visited: set[str] = set()
        components: list[set[str]] = []

        # Step 1: Connected components via BFS
        for start in all_nodes:
            if start in visited:
                continue
            component = self._bfs_component(start, all_nodes)
            visited |= component
            if len(component) >= self.config.min_cluster_size:
                components.append(component)

        logger.info(
            "HierarchyManager: found %d connected components (min_size=%d)",
            len(components), self.config.min_cluster_size,
        )

        # Step 2: Split large components by co-occurrence
        clusters: list[ClusterInfo] = []
        for component in components:
            if len(component) <= self.config.max_cluster_size:
                clusters.append(ClusterInfo(
                    cluster_id=uuid.uuid4().hex[:10],
                    node_ids=list(component),
                    level=1,
                ))
            else:
                sub_clusters = self._split_by_cooccurrence(component)
                clusters.extend(sub_clusters)

        # Limit total clusters
        clusters = clusters[:self.config.l1_max_clusters]

        logger.info("HierarchyManager: %d L1 clusters detected", len(clusters))
        return clusters

    def _bfs_component(self, start: str, all_nodes: set[str]) -> set[str]:
        """BFS from start node to find connected component."""
        visited = {start}
        queue = [start]

        while queue:
            node = queue.pop(0)
            for neighbor_id, _, _ in self.graph.neighbors(node):
                if neighbor_id in all_nodes and neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)

        return visited

    def _split_by_cooccurrence(self, component: set[str]) -> list[ClusterInfo]:
        """Split a large component into sub-clusters by shared neighbors.

        Nodes that share many neighbors are likely related. We group them
        using a greedy assignment to the cluster with highest overlap.
        """
        nodes = list(component)
        max_size = self.config.max_cluster_size

        # Build neighbor sets for each node
        neighbor_sets: dict[str, set[str]] = {}
        for nid in nodes:
            neighbors = {n for n, _, _ in self.graph.neighbors(nid)}
            neighbor_sets[nid] = neighbors & component  # Only within-component

        # Greedy clustering: assign each node to best-matching cluster
        clusters: list[list[str]] = []
        cluster_neighbors: list[set[str]] = []
        assigned: set[str] = set()

        for nid in nodes:
            if nid in assigned:
                continue

            best_cluster = -1
            best_overlap = 0

            for i, cn in enumerate(cluster_neighbors):
                if len(clusters[i]) >= max_size:
                    continue
                overlap = len(neighbor_sets[nid] & cn)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cluster = i

            if best_cluster >= 0 and best_overlap > 0:
                clusters[best_cluster].append(nid)
                cluster_neighbors[best_cluster] |= neighbor_sets[nid]
            else:
                # Start new cluster
                clusters.append([nid])
                cluster_neighbors.append(neighbor_sets[nid])

            assigned.add(nid)

        return [
            ClusterInfo(
                cluster_id=uuid.uuid4().hex[:10],
                node_ids=c,
                level=1,
            )
            for c in clusters
            if len(c) >= self.config.min_cluster_size
        ]

    # =========================================================================
    # Hierarchy Building
    # =========================================================================

    async def build_hierarchy(
        self,
        bus: Any,
        llm_fn: Callable[[str, int], Coroutine[Any, Any, str]] | None = None,
    ) -> list[ClusterInfo]:
        """Build full summary hierarchy over current graph state.

        Steps:
          1. Detect communities (L1 clusters)
          2. For each cluster: generate summary (extractive or LLM)
          3. Store summary nodes with SUMMARIZES edges
          4. Group L1 clusters into L2 topics (if enough clusters)
          5. Generate L2 summaries

        Args:
            bus: LiteMemoryBus for storing summary nodes
            llm_fn: Optional LLM function for abstractive summaries

        Returns:
            List of ClusterInfo with summary_node_id populated
        """
        clusters = self.detect_communities()

        if not clusters:
            return []

        # Build L1 summaries
        l1_clusters = []
        for cluster in clusters:
            contents = self._get_cluster_contents(cluster.node_ids)
            if not contents:
                continue

            if llm_fn is not None:
                summary = await self._llm_summary(contents, llm_fn)
            else:
                summary = self._extractive_summary(contents)

            if summary:
                cluster.summary_text = summary
                summary_id = await self._store_summary(
                    bus, summary, cluster.node_ids, level=1,
                )
                cluster.summary_node_id = summary_id
                l1_clusters.append(cluster)

        logger.info("HierarchyManager: built %d L1 summaries", len(l1_clusters))

        # Build L2 summaries (group L1 clusters)
        if len(l1_clusters) >= self.config.l2_min_l1_clusters:
            l2_clusters = self._group_l1_into_l2(l1_clusters)
            for l2_cluster in l2_clusters:
                l1_summaries = [
                    c.summary_text for c in l1_clusters
                    if c.cluster_id in set(l2_cluster.node_ids)
                    and c.summary_text
                ]
                if not l1_summaries:
                    continue

                if llm_fn is not None:
                    summary = await self._llm_summary(l1_summaries, llm_fn)
                else:
                    summary = self._extractive_summary(l1_summaries)

                if summary:
                    l2_cluster.summary_text = summary
                    # L2 SUMMARIZES the L1 summary nodes
                    l1_summary_ids = [
                        c.summary_node_id for c in l1_clusters
                        if c.cluster_id in set(l2_cluster.node_ids)
                        and c.summary_node_id
                    ]
                    summary_id = await self._store_summary(
                        bus, summary, l1_summary_ids, level=2,
                    )
                    l2_cluster.summary_node_id = summary_id
                    l1_clusters.append(l2_cluster)

            logger.info("HierarchyManager: built %d L2 topic summaries", len(l2_clusters))

        return l1_clusters

    def _group_l1_into_l2(self, l1_clusters: list[ClusterInfo]) -> list[ClusterInfo]:
        """Group L1 clusters into L2 topic clusters by node overlap.

        Two L1 clusters are related if they share entity connections.
        We use greedy grouping similar to _split_by_cooccurrence.
        """
        # Build node sets for each L1 cluster
        cluster_nodes = {c.cluster_id: set(c.node_ids) for c in l1_clusters}

        # Group clusters that share nodes
        l2_groups: list[list[str]] = []  # Each group is a list of cluster_ids
        assigned: set[str] = set()

        for cluster in l1_clusters:
            cid = cluster.cluster_id
            if cid in assigned:
                continue

            best_group = -1
            best_overlap = 0

            for i, group in enumerate(l2_groups):
                group_nodes = set()
                for gid in group:
                    group_nodes |= cluster_nodes.get(gid, set())
                overlap = len(cluster_nodes[cid] & group_nodes)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_group = i

            if best_group >= 0 and best_overlap > 0:
                l2_groups[best_group].append(cid)
            else:
                l2_groups.append([cid])

            assigned.add(cid)

        return [
            ClusterInfo(
                cluster_id=uuid.uuid4().hex[:10],
                node_ids=group,  # These are L1 cluster_ids, not raw node_ids
                level=2,
            )
            for group in l2_groups
            if len(group) >= self.config.l2_min_l1_clusters
        ]

    # =========================================================================
    # Summarization
    # =========================================================================

    def _extractive_summary(self, contents: list[str]) -> str:
        """No-LLM summary: key sentence extraction by TF-IDF-like scoring.

        Selects the most informative sentences across all content pieces.
        No hallucination — every word comes from the source material.
        """
        # Split all content into sentences
        sentences: list[tuple[str, float]] = []

        # Word frequency for TF-IDF-like scoring
        word_freq: dict[str, int] = defaultdict(int)
        for content in contents:
            for word in content.lower().split():
                word = word.strip(".,;:!?()[]\"'")
                if len(word) > 2:
                    word_freq[word] += 1

        # Score each sentence by sum of inverse-document-frequency-like weights
        doc_count = max(1, len(contents))
        for content in contents:
            for sentence in self._split_sentences(content):
                if len(sentence.split()) < 4:
                    continue

                score = 0.0
                words = sentence.lower().split()
                for word in words:
                    word = word.strip(".,;:!?()[]\"'")
                    freq = word_freq.get(word, 0)
                    if freq > 0:
                        # TF-IDF-like: rare-across-docs words score higher
                        idf = math.log(1 + doc_count / freq)
                        score += idf

                # Normalize by sentence length
                score /= max(1, len(words))
                sentences.append((sentence, score))

        # Select top-k sentences by score
        sentences.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in sentences[:self.config.extractive_sentences]]

        return " ".join(selected) if selected else ""

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Simple sentence splitting (no nltk dependency)."""
        sentences = []
        current = []
        for char in text:
            current.append(char)
            if char in ".!?" and len(current) > 10:
                sentences.append("".join(current).strip())
                current = []
        if current:
            remainder = "".join(current).strip()
            if len(remainder) > 10:
                sentences.append(remainder)
        return sentences

    async def _llm_summary(
        self,
        contents: list[str],
        llm_fn: Callable[[str, int], Coroutine[Any, Any, str]],
    ) -> str:
        """LLM-generated summary of a cluster's contents."""
        combined = "\n---\n".join(c[:500] for c in contents[:10])
        prompt = (
            "Summarize the following related knowledge items into a single "
            "concise paragraph. Preserve key facts and relationships. "
            "Do not add information not present in the sources.\n\n"
            f"{combined}\n\n"
            "Summary:"
        )
        try:
            return await llm_fn(prompt, 256)
        except Exception as e:
            logger.warning("HierarchyManager: LLM summary failed: %s", e)
            return self._extractive_summary(contents)

    # =========================================================================
    # Storage
    # =========================================================================

    def _get_cluster_contents(self, node_ids: list[str]) -> list[str]:
        """Extract text content from cluster nodes."""
        contents = []
        for nid in node_ids:
            data = self.graph.node_data(nid)
            content = data.get("content", "")
            if content:
                contents.append(content)
        return contents

    async def _store_summary(
        self,
        bus: Any,
        summary: str,
        source_ids: list[str],
        level: int,
    ) -> str:
        """Store a summary node in the bus with SUMMARIZES edges.

        The summary node:
          - memory_type = "summary"
          - importance = avg_source_importance * boost
          - Has SUMMARIZES edges to each source node
        """
        from hololoom.memory.lite_bus import MemoryItem

        # Compute importance from sources
        importances = []
        for sid in source_ids:
            data = self.graph.node_data(sid)
            importances.append(data.get("importance", 0.5))
        avg_importance = sum(importances) / max(1, len(importances))

        summary_item = MemoryItem(
            content=f"[L{level} summary] {summary}",
            memory_type="summary",
            importance=min(1.0, avg_importance * self.config.summary_importance_boost),
        )
        summary_id = await bus.store(summary_item)

        # Add SUMMARIZES edges (summary → source)
        for sid in source_ids:
            if hasattr(bus, '_edges'):
                if summary_id not in bus._edges:
                    bus._edges[summary_id] = []
                bus._edges[summary_id].append((sid, "SUMMARIZES"))

        logger.debug(
            "Stored L%d summary %s → %d sources",
            level, summary_id[:10], len(source_ids),
        )
        return summary_id
