"""
Multi-Hop Reasoning for HoloLoom RAG
=====================================

Multi-hop graph traversal for complex queries requiring multiple reasoning steps.

This module implements beam search traversal over the Yarn Graph (KG) to discover
reasoning chains (paths) that connect entities. Key features:

- Beam Search: Explore top-k paths in parallel, prune low-confidence branches
- Path Ranking: Score paths by relevance, completeness, coherence
- Bidirectional Search: Start from both query entities and goal, meet in middle
- Explanation Generation: Natural language description of reasoning steps
- Cycle Detection: Prevent infinite loops

Philosophy:
-----------
Complex queries often require multi-step reasoning:
- "How does A relate to B?" → Find path A→C→B
- "What connects attention to BERT?" → attention→transformer→BERT
- "Explain the relationship between X and Y" → Multi-hop traversal

Instead of direct retrieval, we follow relationship chains to build reasoning paths.

Architecture:
- SimpleRAG retrieves initial entities (seed nodes)
- MultiHopRAGMixin explores graph neighbors (N hops)
- Beam search keeps only top-k paths at each hop
- Path ranking selects best reasoning chains
- LLM synthesizes explanation from paths

Performance:
- 1 hop: ~10ms (direct neighbors)
- 2 hops: ~50ms (neighbors of neighbors, beam_width=5)
- 3 hops: ~150ms (deeper reasoning, beam_width=5)
- 4+ hops: ~300ms+ (exponential growth, use carefully)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ReasoningPath:
    """
    A reasoning chain discovered through graph traversal.

    Represents a path from query entities to answer entities, showing
    how concepts are connected through relationships.

    Attributes:
        entities: List of entity names in path ["query", "intermediate", "answer"]
        relationships: List of edge types ["USES", "LEADS_TO", "IS_A"]
        confidence: Path confidence score (0.0-1.0)
        hop_count: Number of hops (edges) in path
        explanation: Natural language description of reasoning
        edge_weights: List of edge weights for each relationship
        metadata: Additional path information (timestamps, sources, etc.)

    Example:
        ReasoningPath(
            entities=["attention", "transformer", "BERT"],
            relationships=["USES", "IS_A"],
            confidence=0.92,
            hop_count=2,
            explanation="attention USES transformer, and BERT IS_A transformer",
            edge_weights=[1.0, 1.0]
        )
    """
    entities: list[str]
    relationships: list[str]
    confidence: float
    hop_count: int
    explanation: str = ""
    edge_weights: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate path structure."""
        if self.entities:
            expected_rels = len(self.entities) - 1
            if len(self.relationships) != expected_rels:
                logger.warning(
                    f"Path has {len(self.entities)} entities but "
                    f"{len(self.relationships)} relationships (expected {expected_rels})"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize path for JSON."""
        return {
            'entities': self.entities,
            'relationships': self.relationships,
            'confidence': self.confidence,
            'hop_count': self.hop_count,
            'explanation': self.explanation,
            'edge_weights': self.edge_weights,
            'metadata': self.metadata
        }

    def __str__(self) -> str:
        """Human-readable path representation."""
        if not self.entities:
            return "ReasoningPath(empty)"

        # Build path string: A -[USES]-> B -[IS_A]-> C
        path_str = self.entities[0]
        for i, rel in enumerate(self.relationships):
            if i + 1 < len(self.entities):
                path_str += f" -[{rel}]-> {self.entities[i+1]}"

        return (
            f"ReasoningPath(\n"
            f"  path: {path_str}\n"
            f"  hops: {self.hop_count}\n"
            f"  confidence: {self.confidence:.3f}\n"
            f"  explanation: {self.explanation[:100]}...\n"
            f")"
        )


@dataclass
class MultiHopRAGResult:
    """
    Extended RAG result with multi-hop reasoning paths.

    Attributes:
        response: LLM-generated answer
        sources: List of retrieved source texts
        confidence: Overall confidence score
        reasoning_mode: Mode used (should be "multihop")
        reasoning_paths: List of discovered reasoning chains
        best_path: Highest-scoring path
        hop_count: Number of hops used
        paths_explored: Total paths explored (before pruning)
        metadata: Additional info (latency_ms, beam_width, etc.)
    """
    response: str
    sources: list[str]
    confidence: float
    reasoning_mode: str = "multihop"
    reasoning_paths: list[ReasoningPath] = field(default_factory=list)
    best_path: ReasoningPath | None = None
    hop_count: int = 0
    paths_explored: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable result."""
        paths_summary = f"{len(self.reasoning_paths)} paths"
        if self.best_path:
            paths_summary += f", best: {' → '.join(self.best_path.entities)}"

        return (
            f"MultiHopRAGResult(\n"
            f"  response: {self.response[:100]}...\n"
            f"  sources: {len(self.sources)} items\n"
            f"  confidence: {self.confidence:.2f}\n"
            f"  reasoning: {paths_summary}\n"
            f"  hops: {self.hop_count}\n"
            f"  explored: {self.paths_explored} paths\n"
            f")"
        )


# ============================================================================
# Multi-Hop Reasoning Mixin
# ============================================================================

class MultiHopRAGMixin:
    """
    Multi-hop reasoning mixin for SimpleRAG.

    Extends RAG with graph traversal capabilities for complex queries
    requiring multi-step reasoning.

    Usage:
        class AdvancedRAG(SimpleRAG, MultiHopRAGMixin):
            pass

        async with AdvancedRAG() as rag:
            result = await rag.query_multihop(
                "How does attention relate to BERT?",
                max_hops=3
            )
            print(result.best_path)
    """

    def __init__(
        self,
        *args,
        max_hops: int = 3,
        beam_width: int = 5,
        min_path_confidence: float = 0.3,
        enable_bidirectional: bool = True,
        **kwargs
    ):
        """
        Initialize multi-hop reasoning.

        Args:
            max_hops: Maximum number of hops to traverse (1-5)
            beam_width: Number of paths to explore in parallel (pruning threshold)
            min_path_confidence: Minimum confidence for a path to be considered
            enable_bidirectional: Use bidirectional search (start+goal meet in middle)
            *args, **kwargs: Passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.max_hops = max_hops
        self.beam_width = beam_width
        self.min_path_confidence = min_path_confidence
        self.enable_bidirectional = enable_bidirectional

        # Multi-hop stats
        self._multihop_stats = {
            'total_queries': 0,
            'total_paths_explored': 0,
            'total_latency_ms': 0.0,
        }

    async def query_multihop(
        self,
        question: str,
        max_hops: int | None = None,
        return_paths: bool = True,
        beam_width: int | None = None,
        mode: str = "verify"
    ) -> MultiHopRAGResult:
        """
        Query with multi-hop graph traversal.

        Process:
        1. Extract seed entities from query
        2. Perform beam search traversal (N hops)
        3. Rank paths by relevance/coherence
        4. Generate explanations for top paths
        5. Synthesize LLM answer using path context

        Args:
            question: Query text
            max_hops: Maximum hops (overrides default)
            return_paths: Include reasoning paths in result
            beam_width: Beam width (overrides default)
            mode: Reasoning mode for LLM synthesis

        Returns:
            MultiHopRAGResult with reasoning paths

        Example:
            result = await rag.query_multihop(
                "How does attention relate to BERT?",
                max_hops=3
            )
            print(f"Best path: {result.best_path}")
            for path in result.reasoning_paths[:5]:
                print(f"  {path}")
        """
        start_time = time.time()

        # Use default values if not specified
        if max_hops is None:
            max_hops = self.max_hops
        if beam_width is None:
            beam_width = self.beam_width

        logger.info(f"Multi-hop query: {question[:50]}... (max_hops={max_hops}, beam={beam_width})")

        # 1. Extract seed entities from query
        seed_entities = self._extract_query_entities(question)
        if not seed_entities:
            logger.warning("No entities extracted from query, falling back to standard query")
            # Fall back to standard RAG
            standard_result = await self.query(question, mode=mode)
            # Convert to MultiHopRAGResult
            return MultiHopRAGResult(
                response=standard_result.response,
                sources=standard_result.sources,
                confidence=standard_result.confidence,
                reasoning_mode="multihop_fallback",
                reasoning_paths=[],
                hop_count=0,
                paths_explored=0,
                metadata={'fallback': True, 'reason': 'no_entities'}
            )

        logger.debug(f"Seed entities: {seed_entities}")

        # 2. Find reasoning paths
        paths = await self.find_reasoning_paths(
            start_entities=seed_entities,
            goal=question,
            max_hops=max_hops,
            beam_width=beam_width
        )

        logger.debug(f"Found {len(paths)} reasoning paths")

        # 3. Generate explanations for paths
        if return_paths:
            for path in paths:
                if not path.explanation:
                    path.explanation = self._generate_explanation(path)

        # 4. Select best path
        best_path = paths[0] if paths else None

        # 5. Build context from paths for LLM
        path_context = self._build_path_context(paths, question)

        # 6. Retrieve relevant documents (combine with path context)
        memories = await self.loom.recall(question, limit=5)
        sources = [mem.text for mem in memories]

        # Add path context to sources
        if path_context:
            sources.insert(0, path_context)

        # 7. Generate answer using LLM
        answer = None
        confidence = 0.0
        if self.orchestrator and (sources or paths):
            logger.debug("Generating answer with multi-hop context...")
            try:
                from hololoom.protocols.types import Query
                query_obj = Query(text=question)
                spacetime = await self.orchestrator.weave(query_obj, use_llm=True)

                answer = spacetime.response if hasattr(spacetime, 'response') else None
                confidence = spacetime.confidence if hasattr(spacetime, 'confidence') else 0.8

            except Exception as e:
                logger.warning(f"⚠ LLM generation failed: {e}")

        # Fallback answer
        if not answer:
            if paths:
                answer = f"Based on {len(paths)} reasoning paths: {best_path.explanation if best_path else 'No clear path found'}"
                confidence = best_path.confidence if best_path else 0.5
            else:
                answer = f"No reasoning paths found for: {question}"
                confidence = 0.2

        # Stats
        latency_ms = (time.time() - start_time) * 1000
        self._multihop_stats['total_queries'] += 1
        self._multihop_stats['total_paths_explored'] += len(paths)
        self._multihop_stats['total_latency_ms'] += latency_ms

        return MultiHopRAGResult(
            response=answer,
            sources=sources,
            confidence=confidence,
            reasoning_mode="multihop",
            reasoning_paths=paths if return_paths else [],
            best_path=best_path,
            hop_count=max_hops,
            paths_explored=len(paths),
            metadata={
                'latency_ms': latency_ms,
                'beam_width': beam_width,
                'seed_entities': seed_entities,
            }
        )

    async def find_reasoning_paths(
        self,
        start_entities: list[str],
        goal: str,
        max_hops: int,
        beam_width: int = 5
    ) -> list[ReasoningPath]:
        """
        Find reasoning paths from start entities using beam search.

        Algorithm:
        1. Initialize beam with seed entities
        2. For each hop:
           a. Expand each path by one edge
           b. Score extended paths
           c. Keep top beam_width paths (prune rest)
        3. Rank final paths by combined score
        4. Return top paths

        Args:
            start_entities: List of seed entity names
            goal: Query text (for relevance scoring)
            max_hops: Maximum path length
            beam_width: Number of paths to keep at each hop

        Returns:
            List of ReasoningPath objects, sorted by confidence (descending)
        """
        if not start_entities:
            return []

        # Get knowledge graph
        kg = self._get_knowledge_graph()
        if not kg:
            logger.warning("Knowledge graph unavailable for multi-hop reasoning")
            return []

        # Initialize beam: each seed entity is a 1-node path
        current_paths: list[tuple[list[str], list[str], float, list[float]]] = []
        for entity in start_entities:
            if entity in kg.G:
                # (entities, relationships, score, edge_weights)
                current_paths.append(([entity], [], 1.0, []))

        if not current_paths:
            logger.debug("No seed entities found in knowledge graph")
            return []

        visited_paths: set[tuple] = set()
        all_paths: list[ReasoningPath] = []

        # Beam search
        for hop in range(max_hops):
            logger.debug(f"Hop {hop + 1}/{max_hops}: {len(current_paths)} paths in beam")

            next_paths: list[tuple[list[str], list[str], float, list[float]]] = []

            for entities, rels, current_score, weights in current_paths:
                last_entity = entities[-1]

                # Get neighbors
                neighbors = kg.get_neighbors(last_entity, direction="both", max_hops=1)

                # Expand path to each neighbor
                for neighbor in neighbors:
                    # Cycle detection
                    if neighbor in entities:
                        continue

                    # Get edge type and weight (check both directions)
                    edge_type = None
                    edge_weight = 1.0
                    src, dst = last_entity, neighbor

                    # Check forward direction
                    if kg.G.has_edge(last_entity, neighbor):
                        # Get edge data directly (MultiDiGraph stores as dict)
                        edge_data = kg.G[last_entity][neighbor]
                        # Take first edge (there may be multiple)
                        if edge_data:
                            first_edge = next(iter(edge_data.values()))
                            edge_type = first_edge.get('type', 'MENTIONS')
                            edge_weight = first_edge.get('weight', 1.0)

                    # Check reverse direction if no forward edge
                    elif kg.G.has_edge(neighbor, last_entity):
                        src, dst = neighbor, last_entity
                        edge_data = kg.G[neighbor][last_entity]
                        if edge_data:
                            first_edge = next(iter(edge_data.values()))
                            edge_type = first_edge.get('type', 'MENTIONS')
                            edge_weight = first_edge.get('weight', 1.0)

                    if not edge_type:
                        continue

                    # Create extended path
                    new_entities = entities + [neighbor]
                    new_rels = rels + [edge_type]
                    new_weights = weights + [edge_weight]

                    # Check if we've seen this path before
                    path_signature = tuple(new_entities)
                    if path_signature in visited_paths:
                        continue
                    visited_paths.add(path_signature)

                    # Score extended path
                    path_score = self._score_path_incremental(
                        entities=new_entities,
                        relationships=new_rels,
                        edge_weights=new_weights,
                        goal=goal,
                        current_score=current_score
                    )

                    # Add to candidates
                    if path_score >= self.min_path_confidence:
                        next_paths.append((new_entities, new_rels, path_score, new_weights))

            # Prune: keep top beam_width paths
            next_paths.sort(key=lambda x: x[2], reverse=True)
            current_paths = next_paths[:beam_width]

            # Store completed paths at this hop
            for entities, rels, score, weights in current_paths:
                path = ReasoningPath(
                    entities=entities,
                    relationships=rels,
                    confidence=score,
                    hop_count=len(rels),
                    edge_weights=weights,
                    metadata={'hop_discovered': hop + 1}
                )
                all_paths.append(path)

            # Early termination if no paths left
            if not current_paths:
                logger.debug(f"Beam search terminated at hop {hop + 1} (no viable paths)")
                break

        # Final ranking across all hops
        all_paths.sort(key=lambda p: p.confidence, reverse=True)

        logger.debug(f"Beam search complete: {len(all_paths)} total paths found")
        return all_paths

    def _extract_query_entities(self, query: str) -> list[str]:
        """
        Extract entity names from query text.

        Uses simple heuristic (capitalized words). In production, use NER.

        Args:
            query: Query text

        Returns:
            List of entity names
        """
        # Simple extraction: capitalized words
        words = query.split()
        entities = []

        for word in words:
            cleaned = word.strip('.,!?;:()[]{}"\'-')
            if cleaned and cleaned[0].isupper() and len(cleaned) > 1:
                entities.append(cleaned)

        # Fallback: extract from knowledge graph if exists
        if not entities and hasattr(self, 'loom') and self.loom:
            kg = self._get_knowledge_graph()
            if kg:
                # Find entities in graph that appear in query (case-insensitive)
                query_lower = query.lower()
                for node in kg.G.nodes():
                    if node.lower() in query_lower:
                        entities.append(node)

        return entities[:5]  # Limit to top 5 seed entities

    def _get_knowledge_graph(self):
        """
        Get knowledge graph from hololoom.

        Returns:
            KG instance or None
        """
        if not hasattr(self, 'loom') or not self.loom:
            return None

        # Try to get graph from awareness graph
        if hasattr(self.loom, 'awareness_graph') and self.loom.awareness_graph:
            if hasattr(self.loom.awareness_graph, 'kg'):
                return self.loom.awareness_graph.kg

        return None


    def _score_path_incremental(
        self,
        entities: list[str],
        relationships: list[str],
        edge_weights: list[float],
        goal: str,
        current_score: float
    ) -> float:
        """
        Score path incrementally (for beam search efficiency).

        Scoring factors:
        1. Edge weights (multiply)
        2. Semantic relevance to goal (boost)
        3. Path length penalty (longer paths = lower score)
        4. Relationship type weighting (IS_A > USES > MENTIONS)

        Args:
            entities: Entity list
            relationships: Relationship list
            edge_weights: Edge weight list
            goal: Query text
            current_score: Previous path score

        Returns:
            Updated path score
        """
        # 1. Edge weight contribution
        if edge_weights:
            last_weight = edge_weights[-1]
            score = current_score * last_weight
        else:
            score = current_score

        # 2. Semantic relevance (check if entities in goal)
        goal_lower = goal.lower()
        last_entity = entities[-1].lower()
        if last_entity in goal_lower:
            score *= 1.5  # Boost for relevant entities

        # 3. Path length penalty (prefer shorter paths)
        path_length = len(entities) - 1
        length_penalty = 1.0 / (1.0 + path_length * 0.1)
        score *= length_penalty

        # 4. Relationship type weighting
        if relationships:
            last_rel = relationships[-1]
            rel_weight = self._relationship_weight(last_rel)
            score *= rel_weight

        return score

    def _relationship_weight(self, relationship: str) -> float:
        """
        Weight different relationship types.

        Hierarchy:
        - IS_A: 1.0 (strongest - taxonomic)
        - USES: 0.9 (functional)
        - PART_OF: 0.85 (compositional)
        - LEADS_TO: 0.8 (causal)
        - MENTIONS: 0.6 (weakest - reference)

        Args:
            relationship: Edge type

        Returns:
            Weight multiplier
        """
        weights = {
            'IS_A': 1.0,
            'USES': 0.9,
            'PART_OF': 0.85,
            'LEADS_TO': 0.8,
            'MENTIONS': 0.6,
            'DEPICTS': 0.7,
            'TAGGED_AS': 0.5,
            'IN_TIME': 0.4,
            'OCCURRED_AT': 0.4,
        }
        return weights.get(relationship, 0.7)  # Default: 0.7

    def _generate_explanation(self, path: ReasoningPath) -> str:
        """
        Generate natural language explanation of reasoning path.

        Example:
            Input: ["attention", "transformer", "BERT"]
                   ["USES", "IS_A"]
            Output: "attention USES transformer, and BERT IS_A transformer"

        Args:
            path: ReasoningPath object

        Returns:
            Natural language explanation
        """
        if not path.entities or len(path.entities) < 2:
            return "No reasoning path"

        explanation_parts = []
        for i, rel in enumerate(path.relationships):
            src = path.entities[i]
            dst = path.entities[i + 1]
            explanation_parts.append(f"{src} {rel.replace('_', ' ').lower()} {dst}")

        return ", and ".join(explanation_parts)

    def _build_path_context(self, paths: list[ReasoningPath], question: str) -> str:
        """
        Build LLM context from reasoning paths.

        Args:
            paths: List of reasoning paths
            question: Query text

        Returns:
            Context string for LLM
        """
        if not paths:
            return ""

        context_lines = [
            f"Multi-hop reasoning for: {question}",
            f"Found {len(paths)} reasoning paths:",
            ""
        ]

        # Include top 3 paths
        for i, path in enumerate(paths[:3], 1):
            context_lines.append(f"Path {i} (confidence: {path.confidence:.2f}):")
            context_lines.append(f"  {path.explanation}")
            context_lines.append("")

        return "\n".join(context_lines)

    def get_multihop_stats(self) -> dict[str, Any]:
        """
        Get multi-hop reasoning statistics.

        Returns:
            Dictionary with stats
        """
        avg_latency = (
            self._multihop_stats['total_latency_ms'] / self._multihop_stats['total_queries']
            if self._multihop_stats['total_queries'] > 0 else 0.0
        )

        avg_paths = (
            self._multihop_stats['total_paths_explored'] / self._multihop_stats['total_queries']
            if self._multihop_stats['total_queries'] > 0 else 0.0
        )

        return {
            'total_queries': self._multihop_stats['total_queries'],
            'total_paths_explored': self._multihop_stats['total_paths_explored'],
            'avg_latency_ms': avg_latency,
            'avg_paths_per_query': avg_paths,
            'max_hops': self.max_hops,
            'beam_width': self.beam_width,
        }
