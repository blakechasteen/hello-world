"""
HoloLoom Shuttle - Weaving Orchestrator Integration

Integrates the MCTS-powered Shuttle (Warp↔Yarn intersection) into the
WeavingOrchestrator as Step 3 (replacing simple Yarn Graph thread selection).

This module provides:
- ShuttleStage: Drop-in replacement for Step 3 in weaving cycle
- Real Warp adapter using HoloLoom's Qdrant backend
- Real Yarn adapter using HoloLoom's Neo4j/NetworkX backend
- Configuration wiring between ShuttleConfig and HoloLoom.config

Created: 2025-01-21
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging
import time

# HoloLoom types
from hololoom.protocols.types import Query, MemoryShard
from hololoom.config import Config, ExecutionMode
from hololoom.memory.graph import KG
from hololoom.chrono.trigger import TemporalWindow

# Shuttle components
from .config import ShuttleConfig, ShuttleMode
from .orchestrator_v2 import Shuttle, WeaveResult, WarpInterface, YarnInterface
from .entity_extraction import Anchor
from .trajectories import TRAJECTORY_BY_NAME
from .mcts import NeighborMap
from .exceptions import ShuttleError, WarpSearchError, YarnTraversalError

logger = logging.getLogger(__name__)


# ============================================================================
# Warp Adapter - Qdrant Integration
# ============================================================================

class HoloLoomWarpAdapter(WarpInterface):
    """
    Warp adapter using HoloLoom's Qdrant backend for semantic search.

    Implements the WarpInterface protocol for Shuttle integration.
    """

    def __init__(self, retriever):
        """
        Args:
            retriever: HoloLoom retriever with Qdrant backend
        """
        self.retriever = retriever
        self.logger = logging.getLogger(__name__)

    def search(
        self,
        query: str,
        top_k: int = 10,
        timeout_ms: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search using Qdrant.

        Args:
            query: Natural language query
            top_k: Number of results
            timeout_ms: Optional timeout (not enforced yet)

        Returns:
            List of dicts with keys: 'id', 'text', 'score', 'metadata', 'entities'

        Raises:
            WarpSearchError: If search fails
        """
        try:
            self.logger.debug(f"[WARP] Searching for: '{query}' (top_k={top_k})")

            # Use HoloLoom retriever
            # Note: retriever.retrieve() returns List[MemoryShard]
            memories = self.retriever.retrieve(query, k=top_k)

            # Convert MemoryShard to Warp result format
            results = []
            for i, mem in enumerate(memories):
                result = {
                    'id': getattr(mem, 'id', f'shard_{i}'),
                    'text': mem.text,
                    'score': getattr(mem, 'score', 0.8),  # Default score if not available
                    'metadata': getattr(mem, 'metadata', {}),
                    # Add entities field for entity extraction
                    'entities': getattr(mem, 'entities', [])
                }
                results.append(result)

            self.logger.info(f"[WARP] Found {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"[WARP] Search failed: {e}")
            raise WarpSearchError(f"Qdrant search failed: {e}") from e


# ============================================================================
# Yarn Adapter - Neo4j/NetworkX Integration
# ============================================================================

class HoloLoomYarnAdapter(YarnInterface):
    """
    Yarn adapter using HoloLoom's KG (NetworkX MultiDiGraph) backend.

    Implements the YarnInterface protocol for Shuttle integration.
    """

    def __init__(self, kg: KG):
        """
        Args:
            kg: HoloLoom Knowledge Graph (NetworkX-based)
        """
        self.kg = kg
        self.logger = logging.getLogger(__name__)

    def build_neighbor_map(
        self,
        anchors: List[str],
        allowed_edge_types: List[str],
        max_depth: int,
        max_nodes: int,
        timeout_ms: Optional[int] = None
    ) -> tuple[NeighborMap, List[str]]:
        """
        Build neighbor map for MCTS from anchor nodes.

        Args:
            anchors: List of anchor node IDs
            allowed_edge_types: Edge types to traverse
            max_depth: Maximum traversal depth
            max_nodes: Maximum nodes to include
            timeout_ms: Optional timeout (not enforced yet)

        Returns:
            (neighbor_map, all_node_ids)

        Raises:
            YarnTraversalError: If graph traversal fails
        """
        try:
            self.logger.debug(
                f"[YARN] Building neighbor map from {len(anchors)} anchors "
                f"(max_depth={max_depth}, max_nodes={max_nodes})"
            )

            neighbor_map: NeighborMap = {}
            all_node_ids = set(anchors)
            frontier = list(anchors)
            visited = set()

            # BFS traversal with depth and node limits
            for depth in range(max_depth):
                if not frontier or len(all_node_ids) >= max_nodes:
                    break

                next_frontier = []

                for node_id in frontier:
                    if node_id in visited:
                        continue
                    visited.add(node_id)

                    # Get neighbors from KG
                    neighbors = self._get_neighbors(node_id, allowed_edge_types)

                    # Filter to unvisited neighbors
                    valid_neighbors = [n for n in neighbors if n not in visited]

                    # Add to neighbor map
                    neighbor_map[node_id] = valid_neighbors

                    # Add to frontier for next depth
                    next_frontier.extend(valid_neighbors)
                    all_node_ids.update(valid_neighbors)

                    # Check node limit
                    if len(all_node_ids) >= max_nodes:
                        break

                frontier = next_frontier

            all_nodes_list = list(all_node_ids)

            self.logger.info(
                f"[YARN] Built neighbor map: {len(neighbor_map)} nodes, "
                f"{depth + 1} depth, {len(all_nodes_list)} total nodes"
            )

            return neighbor_map, all_nodes_list

        except Exception as e:
            self.logger.error(f"[YARN] Graph traversal failed: {e}")
            raise YarnTraversalError(f"KG traversal failed: {e}") from e

    def _get_neighbors(self, node_id: str, allowed_edge_types: List[str]) -> List[str]:
        """
        Get neighbors of a node filtered by edge type.

        Args:
            node_id: Source node ID
            allowed_edge_types: Edge types to follow

        Returns:
            List of neighbor node IDs
        """
        neighbors = []

        # Check if node exists in graph
        if not self.kg.G.has_node(node_id):
            return neighbors

        # Get all outgoing edges
        for neighbor, edges in self.kg.G[node_id].items():
            # edges is a dict of {edge_key: edge_data}
            for edge_data in edges.values():
                edge_type = edge_data.get('type', 'RELATED_TO')

                # Filter by allowed edge types
                if edge_type in allowed_edge_types:
                    neighbors.append(neighbor)
                    break  # Don't add same neighbor multiple times

        return neighbors

    def get_node_description(
        self,
        node_ids: List[str],
        timeout_ms: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Get text descriptions for nodes.

        Args:
            node_ids: List of node IDs
            timeout_ms: Optional timeout (not enforced yet)

        Returns:
            Dict mapping node_id → description text

        Raises:
            YarnTraversalError: If retrieval fails
        """
        try:
            descriptions = {}

            for node_id in node_ids:
                if self.kg.G.has_node(node_id):
                    # Get node data
                    node_data = self.kg.G.nodes[node_id]

                    # Try to get description from various fields
                    desc = (
                        node_data.get('description') or
                        node_data.get('text') or
                        node_data.get('content') or
                        node_id  # Fallback to node ID
                    )

                    descriptions[node_id] = desc
                else:
                    descriptions[node_id] = node_id  # Fallback

            return descriptions

        except Exception as e:
            self.logger.error(f"[YARN] Failed to get node descriptions: {e}")
            raise YarnTraversalError(f"Node description retrieval failed: {e}") from e


# ============================================================================
# Shuttle Stage - Drop-in replacement for Step 3
# ============================================================================

class ShuttleStage:
    """
    Shuttle-powered thread selection for WeavingOrchestrator Step 3.

    Drop-in replacement for simple Yarn Graph thread selection.
    Uses MCTS to intelligently combine Warp (semantic) and Yarn (structural) search.
    """

    def __init__(
        self,
        config: Config,
        kg: KG,
        retriever,
        shuttle_config: Optional[ShuttleConfig] = None
    ):
        """
        Args:
            config: HoloLoom config
            kg: Knowledge graph (for Yarn)
            retriever: Vector retriever (for Warp)
            shuttle_config: Optional Shuttle config (auto-derived from config if None)
        """
        self.config = config
        self.kg = kg
        self.retriever = retriever
        self.logger = logging.getLogger(__name__)

        # Derive shuttle config from hololoom config if not provided
        if shuttle_config is None:
            shuttle_config = self._derive_shuttle_config(config)

        self.shuttle_config = shuttle_config

        # Create Warp and Yarn adapters
        self.warp = HoloLoomWarpAdapter(retriever)
        self.yarn = HoloLoomYarnAdapter(kg)

        # Create Shuttle
        self.shuttle = Shuttle(
            warp=self.warp,
            yarn=self.yarn,
            config=shuttle_config
        )

        self.logger.info(
            f"[SHUTTLE] Initialized (mode={shuttle_config.mode.value}, "
            f"mcts_sims={shuttle_config.mcts_simulations})"
        )

    def _derive_shuttle_config(self, config: Config) -> ShuttleConfig:
        """
        Derive ShuttleConfig from hololoom Config.

        Maps execution modes:
        - BARE → MINIMAL (lightweight)
        - FAST → LITE (balanced)
        - FUSED → FULL (feature-rich)
        """
        mode_mapping = {
            ExecutionMode.BARE: ShuttleMode.MINIMAL,
            ExecutionMode.FAST: ShuttleMode.LITE,
            ExecutionMode.FUSED: ShuttleMode.FULL,
        }

        shuttle_mode = mode_mapping.get(config.mode, ShuttleMode.AUTO)

        # Derive other parameters
        shuttle_config = ShuttleConfig(
            mode=shuttle_mode,
            mcts_simulations=32 if config.mode == ExecutionMode.FUSED else 16,
            mcts_timeout_ms=5000 if config.mode == ExecutionMode.FUSED else 2000,
            warp_top_k=config.retrieval_k,
            max_graph_depth=2 if config.mode == ExecutionMode.FUSED else 1,
            max_graph_nodes=40 if config.mode == ExecutionMode.FUSED else 20,
            enable_graceful_degradation=True,
            enable_entity_extraction=True,
            entity_extraction_method="spacy",  # spaCy with fallback to regex → payload
        )

        return shuttle_config

    async def select_threads(
        self,
        temporal_window: TemporalWindow,
        query: Query,
        trajectory_name: Optional[str] = None
    ) -> List[MemoryShard]:
        """
        Select threads using Shuttle (Warp↔Yarn intersection with MCTS).

        This is the drop-in replacement for yarn_graph.select_threads().

        Args:
            temporal_window: Temporal filter (currently not used by Shuttle)
            query: User query
            trajectory_name: Optional trajectory override

        Returns:
            List of MemoryShard objects (threads)
        """
        try:
            self.logger.info(f"[SHUTTLE] Selecting threads for: '{query.text}'")
            start_time = time.time()

            # Run Shuttle intersection
            weave_result = self.shuttle.intersect(
                query=query.text,
                trajectory_name=trajectory_name
            )

            # Convert WeaveResult to MemoryShard format
            threads = self._convert_to_shards(weave_result, query)

            duration_ms = (time.time() - start_time) * 1000

            self.logger.info(
                f"[SHUTTLE] Selected {len(threads)} threads in {duration_ms:.1f}ms "
                f"(trajectory={weave_result.metadata.get('trajectory_used', 'unknown')})"
            )

            return threads

        except (ShuttleError, AttributeError, Exception) as e:
            self.logger.error(f"[SHUTTLE] Thread selection failed: {e}")

            # Graceful degradation: Fall back to simple retrieval
            if self.shuttle_config.enable_graceful_degradation:
                self.logger.warning("[SHUTTLE] Falling back to simple retrieval")
                return self._fallback_retrieval(query)
            else:
                raise

    def _convert_to_shards(
        self,
        weave_result: WeaveResult,
        query: Query
    ) -> List[MemoryShard]:
        """
        Convert WeaveResult to List[MemoryShard] format expected by orchestrator.

        Args:
            weave_result: Shuttle's weave result
            query: Original query

        Returns:
            List of MemoryShard objects
        """
        shards = []

        # Add Warp results (semantic search)
        for i, item in enumerate(weave_result.fuzzy_evidence):
            shard = MemoryShard(
                id=f"warp_{item.get('id', i)}",
                text=item.get('text', ''),
                metadata={
                    'source': 'warp',
                    'score': item.get('score', 0.0),
                    'original_metadata': item.get('metadata', {}),
                }
            )
            shards.append(shard)

        # Add Yarn results (graph traversal)
        for node_id, description in weave_result.yarn_context.items():
            shard = MemoryShard(
                id=f"yarn_{node_id}",
                text=description,
                metadata={
                    'source': 'yarn',
                    'node_id': node_id,
                    'graph_depth': weave_result.metadata.get('mcts_depth', 0),
                }
            )
            shards.append(shard)

        return shards

    def _fallback_retrieval(self, query: Query) -> List[MemoryShard]:
        """
        Fallback to simple retrieval if Shuttle fails.

        Args:
            query: User query

        Returns:
            List of MemoryShard from simple retrieval
        """
        try:
            return self.retriever.retrieve(query.text, k=self.shuttle_config.warp_top_k)
        except Exception as e:
            self.logger.error(f"[SHUTTLE] Fallback retrieval failed: {e}")
            return []


# ============================================================================
# Integration Helper
# ============================================================================

def create_shuttle_stage(
    config: Config,
    kg: KG,
    retriever,
    shuttle_config: Optional[ShuttleConfig] = None
) -> ShuttleStage:
    """
    Factory function to create ShuttleStage for WeavingOrchestrator.

    Args:
        config: HoloLoom config
        kg: Knowledge graph
        retriever: Vector retriever
        shuttle_config: Optional shuttle config (auto-derived if None)

    Returns:
        ShuttleStage instance ready for integration

    Example:
        >>> from hololoom.config import Config
        >>> from hololoom.memory.graph import KG
        >>> from hololoom.memory.base import create_retriever
        >>> from hololoom.shuttle.weaving_integration import create_shuttle_stage
        >>>
        >>> config = Config.fast()
        >>> kg = KG()
        >>> retriever = create_retriever(config)
        >>> shuttle_stage = create_shuttle_stage(config, kg, retriever)
        >>>
        >>> # Use in WeavingOrchestrator:
        >>> # threads = await shuttle_stage.select_threads(temporal_window, query)
    """
    return ShuttleStage(
        config=config,
        kg=kg,
        retriever=retriever,
        shuttle_config=shuttle_config
    )
