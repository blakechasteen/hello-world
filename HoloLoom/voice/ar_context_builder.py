"""
AR Context Builder
==================
Knowledge graph-based context retrieval for AR responses.

Queries HoloLoom's Yarn Graph (KG) to provide rich context for AR interactions,
including entity relationships, multi-hop reasoning, and spectral features.

Features:
- Entity extraction from AR queries and detected objects
- Multi-hop relationship traversal for context enrichment
- Spectral graph features for response quality
- Subgraph extraction for relevant context
- Temporal graph queries (bi-temporal support)

Integration:
- Uses KG (Yarn Graph) from HoloLoom.memory.graph
- Integrates with ARContext from HoloLoom.voice.ar_context
- Provides context for voice responses and AR overlays

Author: HoloLoom Team
Date: November 17, 2025
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re

import networkx as nx
import numpy as np

from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.voice.ar_context import ARContext, ARObject, ARObjectType

logger = logging.getLogger("HoloLoom.voice.ar_context_builder")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Entity:
    """
    An extracted entity from AR query or environment.

    Represents a concept, object, or topic that can be grounded in the
    knowledge graph.
    """
    text: str                    # Original text
    entity_type: str            # Type (e.g., "object", "action", "property")
    normalized: str             # Normalized form (lowercase, singular)
    confidence: float = 1.0     # Extraction confidence
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.normalized)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.normalized == other.normalized


@dataclass
class RelationshipPath:
    """
    A path through the knowledge graph connecting entities.

    Represents a reasoning chain like:
    beehive → IS_A → apiary_equipment → USES → inspection_tools
    """
    entities: List[str]          # Entity nodes in path
    relationships: List[str]     # Edge types connecting entities
    weights: List[float]         # Edge weights
    total_weight: float          # Total path weight (sum or product)
    hops: int                    # Number of hops

    def __str__(self) -> str:
        """Human-readable path representation."""
        parts = []
        for i, entity in enumerate(self.entities):
            parts.append(entity)
            if i < len(self.relationships):
                parts.append(f"--[{self.relationships[i]}]-->")
        return " ".join(parts)


@dataclass
class ARContextEnrichment:
    """
    Result of context enrichment from knowledge graph.

    Contains all information extracted from KG for AR response generation.
    """
    query: str                              # Original query
    extracted_entities: List[Entity]        # Entities found in query
    grounded_entities: List[str]            # Entities found in KG
    relationships: List[KGEdge]             # Direct relationships
    reasoning_paths: List[RelationshipPath] # Multi-hop paths
    subgraph: nx.MultiDiGraph               # Relevant subgraph
    spectral_features: Dict[str, Any]       # Spectral analysis
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Generate human-readable summary of enrichment."""
        parts = [
            f"Query: {self.query}",
            f"Extracted entities: {len(self.extracted_entities)} ({', '.join(e.text for e in self.extracted_entities[:5])}...)",
            f"Grounded in KG: {len(self.grounded_entities)} entities",
            f"Direct relationships: {len(self.relationships)}",
            f"Reasoning paths: {len(self.reasoning_paths)} multi-hop connections",
            f"Subgraph size: {self.subgraph.number_of_nodes()} nodes, {self.subgraph.number_of_edges()} edges",
        ]

        if self.reasoning_paths:
            parts.append(f"Example path: {self.reasoning_paths[0]}")

        return "\n".join(parts)


# ============================================================================
# Entity Extractor
# ============================================================================

class EntityExtractor:
    """
    Extracts entities from AR queries and AR environment.

    Uses pattern matching and heuristics to identify:
    - Object references ("this hive", "that tool")
    - Property queries ("health", "temperature")
    - Action requests ("inspect", "move", "show")
    """

    # Common AR query patterns
    OBJECT_PATTERNS = [
        r"(this|that|the)\s+(\w+)",        # "this hive", "that frame"
        r"(\w+)\s+(here|there|nearby)",    # "hive here", "tool nearby"
        r"(\w+)\s+\d+",                    # "hive 001", "frame 3"
    ]

    ACTION_PATTERNS = [
        r"(show|display|highlight|inspect|check|analyze|move|create|delete)\s+(\w+)",
        r"what\s+is\s+the\s+(\w+)",
        r"how\s+(\w+)\s+is",
    ]

    PROPERTY_PATTERNS = [
        r"(health|temperature|humidity|population|status|condition)",
        r"how\s+(healthy|strong|weak)",
    ]

    def __init__(self):
        """Initialize entity extractor."""
        self.object_regex = [re.compile(p, re.IGNORECASE) for p in self.OBJECT_PATTERNS]
        self.action_regex = [re.compile(p, re.IGNORECASE) for p in self.ACTION_PATTERNS]
        self.property_regex = [re.compile(p, re.IGNORECASE) for p in self.PROPERTY_PATTERNS]

    def extract(
        self,
        query: str,
        ar_context: Optional[ARContext] = None
    ) -> List[Entity]:
        """
        Extract entities from query and AR context.

        Args:
            query: User's query text
            ar_context: Current AR context (for spatial references)

        Returns:
            List of extracted entities
        """
        entities = []

        # Extract object references
        for pattern in self.object_regex:
            for match in pattern.finditer(query):
                object_name = match.group(2) if match.lastindex >= 2 else match.group(1)
                entities.append(Entity(
                    text=match.group(0),
                    entity_type="object",
                    normalized=object_name.lower(),
                    confidence=0.9
                ))

        # Extract action references
        for pattern in self.action_regex:
            for match in pattern.finditer(query):
                action_name = match.group(1)
                entities.append(Entity(
                    text=match.group(0),
                    entity_type="action",
                    normalized=action_name.lower(),
                    confidence=0.85
                ))

        # Extract property references
        for pattern in self.property_regex:
            for match in pattern.finditer(query):
                property_name = match.group(1) if match.lastindex >= 1 else match.group(0)
                entities.append(Entity(
                    text=match.group(0),
                    entity_type="property",
                    normalized=property_name.lower(),
                    confidence=0.8
                ))

        # Add AR context entities (objects in view)
        if ar_context:
            # Gaze target
            if ar_context.gaze_target:
                gaze_obj = ar_context.find_object_by_id(ar_context.gaze_target)
                if gaze_obj:
                    entities.append(Entity(
                        text=gaze_obj.name,
                        entity_type="ar_object",
                        normalized=gaze_obj.type.value,
                        confidence=1.0,
                        metadata={"object_id": gaze_obj.id, "source": "gaze"}
                    ))

            # Selected object
            if ar_context.selected_object:
                entities.append(Entity(
                    text=ar_context.selected_object.name,
                    entity_type="ar_object",
                    normalized=ar_context.selected_object.type.value,
                    confidence=1.0,
                    metadata={"object_id": ar_context.selected_object.id, "source": "selected"}
                ))

            # Nearby objects (spatial references)
            for ref, obj in ar_context.nearby_objects.items():
                entities.append(Entity(
                    text=f"{ref} ({obj.name})",
                    entity_type="ar_object",
                    normalized=obj.type.value,
                    confidence=0.9,
                    metadata={"object_id": obj.id, "source": "spatial_ref", "reference": ref}
                ))

        # Deduplicate entities (keep highest confidence)
        unique_entities = {}
        for entity in entities:
            key = (entity.normalized, entity.entity_type)
            if key not in unique_entities or entity.confidence > unique_entities[key].confidence:
                unique_entities[key] = entity

        return list(unique_entities.values())


# ============================================================================
# AR Context Builder
# ============================================================================

class ARContextBuilder:
    """
    Main context builder for AR interactions.

    Queries Yarn Graph (KG) to enrich AR queries with:
    - Entity grounding (map query entities to KG entities)
    - Direct relationships (1-hop connections)
    - Multi-hop reasoning paths (2-3 hop connections)
    - Spectral features (graph structure analysis)
    - Relevant subgraphs (context extraction)

    Usage:
        builder = ARContextBuilder(kg)

        enrichment = await builder.build_context(
            query="What's the health of this hive?",
            ar_context=ar_context
        )

        print(enrichment.get_summary())
        # Use enrichment for response generation
    """

    def __init__(
        self,
        kg: KG,
        max_hops: int = 3,
        max_paths: int = 10,
        enable_spectral_features: bool = True
    ):
        """
        Initialize AR context builder.

        Args:
            kg: Knowledge graph (Yarn Graph)
            max_hops: Maximum hops for multi-hop reasoning
            max_paths: Maximum number of reasoning paths to find
            enable_spectral_features: Whether to compute spectral features
        """
        self.kg = kg
        self.max_hops = max_hops
        self.max_paths = max_paths
        self.enable_spectral_features = enable_spectral_features

        self.entity_extractor = EntityExtractor()

        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    async def build_context(
        self,
        query: str,
        ar_context: Optional[ARContext] = None,
        temporal_query_time: Optional[datetime] = None
    ) -> ARContextEnrichment:
        """
        Build enriched context for AR query.

        Args:
            query: User's query text
            ar_context: Current AR context
            temporal_query_time: Point-in-time query (for bi-temporal KG)

        Returns:
            ARContextEnrichment with KG-derived context
        """
        logger.info(f"Building context for query: {query[:50]}...")

        # Extract entities from query and AR context
        extracted_entities = self.entity_extractor.extract(query, ar_context)
        logger.debug(f"Extracted {len(extracted_entities)} entities")

        # Ground entities in KG
        grounded_entities = self._ground_entities(extracted_entities, temporal_query_time)
        logger.debug(f"Grounded {len(grounded_entities)} entities in KG")

        # Find direct relationships
        relationships = self._find_direct_relationships(grounded_entities, temporal_query_time)
        logger.debug(f"Found {len(relationships)} direct relationships")

        # Find multi-hop reasoning paths
        reasoning_paths = self._find_reasoning_paths(
            grounded_entities,
            max_hops=self.max_hops,
            max_paths=self.max_paths,
            temporal_query_time=temporal_query_time
        )
        logger.debug(f"Found {len(reasoning_paths)} reasoning paths")

        # Extract relevant subgraph
        subgraph = self._extract_subgraph(
            grounded_entities,
            relationships,
            reasoning_paths,
            temporal_query_time
        )
        logger.debug(f"Extracted subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

        # Compute spectral features
        spectral_features = {}
        if self.enable_spectral_features and subgraph.number_of_nodes() > 0:
            spectral_features = self._compute_spectral_features(subgraph)
            logger.debug(f"Computed spectral features: {list(spectral_features.keys())}")

        # Build enrichment
        enrichment = ARContextEnrichment(
            query=query,
            extracted_entities=extracted_entities,
            grounded_entities=grounded_entities,
            relationships=relationships,
            reasoning_paths=reasoning_paths,
            subgraph=subgraph,
            spectral_features=spectral_features,
            metadata={
                "ar_context_summary": ar_context.to_dict() if ar_context else {},
                "temporal_query_time": temporal_query_time.isoformat() if temporal_query_time else None,
            }
        )

        logger.info(f"Context built: {len(grounded_entities)} entities, {len(relationships)} relationships, {len(reasoning_paths)} paths")

        return enrichment

    def _ground_entities(
        self,
        entities: List[Entity],
        temporal_query_time: Optional[datetime] = None
    ) -> List[str]:
        """
        Ground extracted entities in knowledge graph.

        Maps entity text to actual KG nodes using:
        - Exact matching (normalized)
        - Fuzzy matching (substring, partial)
        - Type-based filtering (entity_type)

        Args:
            entities: Extracted entities
            temporal_query_time: Point-in-time query

        Returns:
            List of KG node IDs that match entities
        """
        grounded = []

        for entity in entities:
            # Try exact match (normalized)
            if entity.normalized in self.kg.G.nodes:
                grounded.append(entity.normalized)
                continue

            # Try fuzzy match (find nodes containing entity text)
            for node in self.kg.G.nodes:
                if entity.normalized in node.lower():
                    grounded.append(node)
                    break

        return list(set(grounded))  # Deduplicate

    def _find_direct_relationships(
        self,
        entities: List[str],
        temporal_query_time: Optional[datetime] = None
    ) -> List[KGEdge]:
        """
        Find direct relationships between grounded entities.

        Retrieves 1-hop connections in the KG.

        Args:
            entities: Grounded entity IDs
            temporal_query_time: Point-in-time query

        Returns:
            List of KGEdges connecting entities
        """
        relationships = []

        for src in entities:
            if src not in self.kg.G:
                continue

            # Get all outgoing edges
            for dst in self.kg.G.successors(src):
                # Get edge data (may be multiple edges between same nodes)
                edge_data = self.kg.G.get_edge_data(src, dst)

                if edge_data:
                    # NetworkX MultiDiGraph returns dict of edge keys
                    for edge_key, data in edge_data.items():
                        edge = KGEdge(
                            src=src,
                            dst=dst,
                            type=data.get("type", "UNKNOWN"),
                            weight=data.get("weight", 1.0),
                            metadata=data.get("metadata", {}),
                            event_time=data.get("event_time"),
                            ingestion_time=data.get("ingestion_time"),
                            valid_from=data.get("valid_from"),
                            valid_to=data.get("valid_to"),
                        )

                        # Check temporal validity
                        if temporal_query_time:
                            if not edge.is_valid_at(temporal_query_time):
                                continue

                        relationships.append(edge)

        return relationships

    def _find_reasoning_paths(
        self,
        entities: List[str],
        max_hops: int = 3,
        max_paths: int = 10,
        temporal_query_time: Optional[datetime] = None
    ) -> List[RelationshipPath]:
        """
        Find multi-hop reasoning paths connecting entities.

        Uses NetworkX shortest path algorithms to find connections.

        Args:
            entities: Grounded entity IDs
            max_hops: Maximum path length
            max_paths: Maximum number of paths to return
            temporal_query_time: Point-in-time query

        Returns:
            List of RelationshipPaths (multi-hop connections)
        """
        paths = []

        # Try all entity pairs
        for i, src in enumerate(entities):
            for dst in entities[i + 1:]:
                if src not in self.kg.G or dst not in self.kg.G:
                    continue

                try:
                    # Find shortest path
                    nx_path = nx.shortest_path(self.kg.G, src, dst)

                    # Skip if too long
                    if len(nx_path) > max_hops + 1:
                        continue

                    # Build RelationshipPath
                    path_entities = nx_path
                    path_relationships = []
                    path_weights = []

                    for j in range(len(nx_path) - 1):
                        edge_src = nx_path[j]
                        edge_dst = nx_path[j + 1]

                        # Get edge data
                        edge_data = self.kg.G.get_edge_data(edge_src, edge_dst)
                        if edge_data:
                            # Take first edge (in MultiDiGraph, multiple edges possible)
                            first_edge_data = next(iter(edge_data.values()))

                            edge_type = first_edge_data.get("type", "UNKNOWN")
                            edge_weight = first_edge_data.get("weight", 1.0)

                            # Check temporal validity
                            if temporal_query_time:
                                edge = KGEdge(
                                    src=edge_src,
                                    dst=edge_dst,
                                    type=edge_type,
                                    weight=edge_weight,
                                    valid_from=first_edge_data.get("valid_from"),
                                    valid_to=first_edge_data.get("valid_to"),
                                )
                                if not edge.is_valid_at(temporal_query_time):
                                    continue

                            path_relationships.append(edge_type)
                            path_weights.append(edge_weight)

                    if path_relationships:  # Valid path found
                        paths.append(RelationshipPath(
                            entities=path_entities,
                            relationships=path_relationships,
                            weights=path_weights,
                            total_weight=sum(path_weights),
                            hops=len(path_relationships)
                        ))

                except nx.NetworkXNoPath:
                    # No path found
                    pass

                # Limit number of paths
                if len(paths) >= max_paths:
                    break

            if len(paths) >= max_paths:
                break

        # Sort by weight (prefer higher weight paths)
        paths.sort(key=lambda p: p.total_weight, reverse=True)

        return paths[:max_paths]

    def _extract_subgraph(
        self,
        entities: List[str],
        relationships: List[KGEdge],
        reasoning_paths: List[RelationshipPath],
        temporal_query_time: Optional[datetime] = None
    ) -> nx.MultiDiGraph:
        """
        Extract relevant subgraph from KG.

        Includes:
        - Grounded entities
        - Direct neighbors (1-hop)
        - Entities in reasoning paths

        Args:
            entities: Grounded entity IDs
            relationships: Direct relationships
            reasoning_paths: Multi-hop paths
            temporal_query_time: Point-in-time query

        Returns:
            Subgraph as NetworkX MultiDiGraph
        """
        # Collect all relevant nodes
        relevant_nodes = set(entities)

        # Add nodes from relationships
        for edge in relationships:
            relevant_nodes.add(edge.src)
            relevant_nodes.add(edge.dst)

        # Add nodes from reasoning paths
        for path in reasoning_paths:
            relevant_nodes.update(path.entities)

        # Extract subgraph
        subgraph = self.kg.G.subgraph(relevant_nodes).copy()

        return subgraph

    def _compute_spectral_features(self, subgraph: nx.MultiDiGraph) -> Dict[str, Any]:
        """
        Compute spectral graph features for response enrichment.

        Computes:
        - Laplacian eigenvalues (graph structure)
        - Centrality metrics (important nodes)
        - Clustering coefficients (community structure)

        Args:
            subgraph: Subgraph to analyze

        Returns:
            Dictionary of spectral features
        """
        features = {}

        if subgraph.number_of_nodes() == 0:
            return features

        try:
            # Convert to undirected for spectral analysis
            undirected = subgraph.to_undirected()

            # Laplacian eigenvalues (graph structure)
            laplacian = nx.laplacian_matrix(undirected).todense()
            eigenvalues = np.linalg.eigvalsh(laplacian)

            features["laplacian_eigenvalues"] = eigenvalues.tolist()
            features["spectral_gap"] = float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0

            # Centrality metrics
            if undirected.number_of_nodes() > 0:
                degree_centrality = nx.degree_centrality(undirected)
                features["avg_degree_centrality"] = float(np.mean(list(degree_centrality.values())))
                features["max_degree_centrality"] = float(max(degree_centrality.values()))
                features["top_central_nodes"] = sorted(
                    degree_centrality.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

            # Clustering coefficient
            if undirected.number_of_nodes() > 0:
                clustering = nx.clustering(undirected)
                features["avg_clustering_coefficient"] = float(np.mean(list(clustering.values())))

        except Exception as e:
            logger.warning(f"Error computing spectral features: {e}")

        return features


# ============================================================================
# Convenience Functions
# ============================================================================

def create_ar_context_builder(
    kg: KG,
    max_hops: int = 3,
    max_paths: int = 10,
    enable_spectral_features: bool = True
) -> ARContextBuilder:
    """
    Create AR context builder with optional configuration.

    Args:
        kg: Knowledge graph (Yarn Graph)
        max_hops: Maximum hops for multi-hop reasoning
        max_paths: Maximum number of reasoning paths
        enable_spectral_features: Whether to compute spectral features

    Returns:
        Configured ARContextBuilder instance
    """
    return ARContextBuilder(
        kg=kg,
        max_hops=max_hops,
        max_paths=max_paths,
        enable_spectral_features=enable_spectral_features
    )
