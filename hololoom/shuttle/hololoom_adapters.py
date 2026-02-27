"""
HoloLoom Shuttle - HoloLoom-Specific Adapters

Adapters for connecting Shuttle to HoloLoom's Warp (Qdrant) and Yarn (Neo4j/KG) backends.
"""

from typing import List, Dict, Any, Optional, Tuple
from .mcts import NeighborMap


# ============================================================================
# Warp Adapter (Qdrant Vector Search)
# ============================================================================

class HoloLoomWarp:
    """
    Warp adapter for HoloLoom's Qdrant vector search backend.

    Wraps the Qdrant client to provide semantic/fuzzy search over embedded content.
    """

    def __init__(self, qdrant_client=None, collection_name: str = "hololoom_memories"):
        """
        Args:
            qdrant_client: Qdrant client instance (optional, falls back to mock)
            collection_name: Qdrant collection name
        """
        self.client = qdrant_client
        self.collection_name = collection_name

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Semantic search for anchor points.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of dicts with keys: 'id', 'text', 'score', 'metadata'
        """
        if self.client is None:
            # Mock implementation for testing
            return self._mock_search(query, top_k)

        try:
            # Real Qdrant search
            from hololoom.embedding.spectral import MatryoshkaEmbeddings

            # Get query embedding
            embedder = MatryoshkaEmbeddings()
            query_vector = embedder.get_embedding(query)

            # Search Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
            )

            # Convert to standard format
            results = []
            for hit in search_results:
                results.append({
                    'id': hit.id,
                    'text': hit.payload.get('text', ''),
                    'score': hit.score,
                    'metadata': hit.payload.get('metadata', {}),
                })

            return results

        except Exception as e:
            print(f"Warp search error: {e}")
            return self._mock_search(query, top_k)

    def _mock_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Mock search results for testing without Qdrant."""
        # Return mock anchors based on query keywords
        query_lower = query.lower()

        mock_results = []

        if "block" in query_lower or "delay" in query_lower:
            mock_results = [
                {
                    'id': 'task_integration_testing',
                    'text': 'Integration testing blocked by missing test data',
                    'score': 0.92,
                    'metadata': {'type': 'Task'},
                },
                {
                    'id': 'task_deployment',
                    'text': 'Deployment delayed due to integration testing',
                    'score': 0.85,
                    'metadata': {'type': 'Task'},
                },
                {
                    'id': 'person_alice',
                    'text': 'Alice is working on test data generation',
                    'score': 0.78,
                    'metadata': {'type': 'Person'},
                },
            ]
        else:
            mock_results = [
                {
                    'id': 'concept_' + query_lower.split()[0] if query_lower else 'unknown',
                    'text': f'Concept related to: {query}',
                    'score': 0.85,
                    'metadata': {'type': 'Concept'},
                },
            ]

        return mock_results[:top_k]


# ============================================================================
# Yarn Adapter (Neo4j Knowledge Graph)
# ============================================================================

class HoloLoomYarn:
    """
    Yarn adapter for HoloLoom's Neo4j knowledge graph backend.

    Provides structured traversal of entity relationships.
    """

    def __init__(self, neo4j_driver=None, kg_client=None):
        """
        Args:
            neo4j_driver: Neo4j driver instance (optional)
            kg_client: HoloLoom KG client (fallback to in-memory graph)
        """
        self.driver = neo4j_driver
        self.kg = kg_client

    def build_neighbor_map(
        self,
        anchors: List[str],
        allowed_edge_types: List[str],
        max_depth: int,
        max_nodes: int,
    ) -> Tuple[NeighborMap, List[str]]:
        """
        Build neighbor map for MCTS from anchor nodes.

        This is the most important method - MCTS uses this to explore the graph.

        Args:
            anchors: List of anchor node IDs
            allowed_edge_types: Edge types to traverse
            max_depth: Maximum traversal depth
            max_nodes: Maximum nodes to include

        Returns:
            (neighbor_map, all_node_ids)
            - neighbor_map: Dict[node_id, List[neighbor_ids]]
            - all_node_ids: List of all nodes in the map
        """
        if self.driver is not None:
            return self._neo4j_build_neighbor_map(anchors, allowed_edge_types, max_depth, max_nodes)
        elif self.kg is not None:
            return self._kg_build_neighbor_map(anchors, allowed_edge_types, max_depth, max_nodes)
        else:
            return self._mock_build_neighbor_map(anchors, allowed_edge_types, max_depth, max_nodes)

    def _neo4j_build_neighbor_map(
        self,
        anchors: List[str],
        allowed_edge_types: List[str],
        max_depth: int,
        max_nodes: int,
    ) -> Tuple[NeighborMap, List[str]]:
        """Build neighbor map using Neo4j driver."""
        neighbor_map: NeighborMap = {}
        visited_nodes = set()

        with self.driver.session() as session:
            # Start BFS from anchors
            frontier = list(anchors)
            depth = 0

            while frontier and depth < max_depth and len(visited_nodes) < max_nodes:
                next_frontier = []

                for node_id in frontier:
                    if node_id in visited_nodes:
                        continue

                    visited_nodes.add(node_id)

                    # Query neighbors
                    edge_type_filter = f":{' | :'.join(allowed_edge_types)}"
                    query = f"""
                    MATCH (n {{id: $node_id}})-[r{edge_type_filter}]-(neighbor)
                    RETURN neighbor.id AS neighbor_id
                    LIMIT {max_nodes - len(visited_nodes)}
                    """

                    result = session.run(query, node_id=node_id)

                    neighbors = []
                    for record in result:
                        neighbor_id = record['neighbor_id']
                        if neighbor_id not in visited_nodes:
                            neighbors.append(neighbor_id)
                            next_frontier.append(neighbor_id)

                    neighbor_map[node_id] = neighbors

                frontier = next_frontier
                depth += 1

        return neighbor_map, list(visited_nodes)

    def _kg_build_neighbor_map(
        self,
        anchors: List[str],
        allowed_edge_types: List[str],
        max_depth: int,
        max_nodes: int,
    ) -> Tuple[NeighborMap, List[str]]:
        """Build neighbor map using HoloLoom KG (NetworkX)."""
        from hololoom.memory.graph import KG

        neighbor_map: NeighborMap = {}
        visited_nodes = set()

        # BFS from anchors
        frontier = list(anchors)
        depth = 0

        while frontier and depth < max_depth and len(visited_nodes) < max_nodes:
            next_frontier = []

            for node_id in frontier:
                if node_id in visited_nodes:
                    continue

                visited_nodes.add(node_id)

                # Get neighbors from NetworkX graph
                neighbors = []

                if node_id in self.kg.graph:
                    for neighbor_id in self.kg.graph.neighbors(node_id):
                        # Check edge types
                        edges = self.kg.graph.get_edge_data(node_id, neighbor_id)

                        for edge_data in edges.values():
                            if edge_data.get('type') in allowed_edge_types:
                                if neighbor_id not in visited_nodes:
                                    neighbors.append(neighbor_id)
                                    next_frontier.append(neighbor_id)
                                break

                neighbor_map[node_id] = neighbors

            frontier = next_frontier
            depth += 1

        return neighbor_map, list(visited_nodes)

    def _mock_build_neighbor_map(
        self,
        anchors: List[str],
        allowed_edge_types: List[str],
        max_depth: int,
        max_nodes: int,
    ) -> Tuple[NeighborMap, List[str]]:
        """Mock neighbor map for testing."""
        # Create a simple mock graph based on common patterns
        neighbor_map: NeighborMap = {}

        if "task_integration_testing" in anchors:
            neighbor_map = {
                "task_integration_testing": ["task_test_data_gen", "person_alice"],
                "task_test_data_gen": ["person_alice", "artifact_test_db"],
                "task_deployment": ["task_integration_testing"],
                "person_alice": ["task_test_data_gen"],
                "artifact_test_db": [],
            }
        else:
            # Generic single-anchor mock
            anchor = anchors[0] if anchors else "unknown"
            neighbor_map = {
                anchor: [f"{anchor}_neighbor_{i}" for i in range(min(3, max_nodes))],
            }

        all_nodes = list(neighbor_map.keys())
        return neighbor_map, all_nodes

    def describe_nodes(self, node_ids: List[str]) -> str:
        """
        Get human-readable description of nodes.

        Args:
            node_ids: List of node IDs to describe

        Returns:
            Formatted string describing the nodes and their relationships
        """
        if self.driver is not None:
            return self._neo4j_describe_nodes(node_ids)
        elif self.kg is not None:
            return self._kg_describe_nodes(node_ids)
        else:
            return self._mock_describe_nodes(node_ids)

    def _neo4j_describe_nodes(self, node_ids: List[str]) -> str:
        """Describe nodes using Neo4j driver."""
        with self.driver.session() as session:
            query = """
            MATCH (n)
            WHERE n.id IN $node_ids
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n.id AS node_id, n.type AS node_type, n.name AS node_name,
                   collect({type: type(r), target: m.id}) AS relationships
            """

            result = session.run(query, node_ids=node_ids)

            descriptions = []
            for record in result:
                node_desc = f"- {record['node_name']} ({record['node_type']})"

                if record['relationships']:
                    rel_strs = []
                    for rel in record['relationships'][:5]:  # Limit to 5 relationships
                        rel_strs.append(f"{rel['type']} → {rel['target']}")

                    node_desc += "\n  " + "\n  ".join(rel_strs)

                descriptions.append(node_desc)

            return "\n".join(descriptions)

    def _kg_describe_nodes(self, node_ids: List[str]) -> str:
        """Describe nodes using HoloLoom KG."""
        descriptions = []

        for node_id in node_ids:
            if node_id not in self.kg.graph:
                continue

            node_data = self.kg.graph.nodes[node_id]
            node_desc = f"- {node_id}: {node_data.get('name', 'Unknown')}"

            # Get relationships
            neighbors = list(self.kg.graph.neighbors(node_id))[:5]
            if neighbors:
                rel_strs = []
                for neighbor in neighbors:
                    edges = self.kg.graph.get_edge_data(node_id, neighbor)
                    for edge_data in edges.values():
                        rel_strs.append(f"{edge_data.get('type', 'RELATED_TO')} → {neighbor}")

                node_desc += "\n  " + "\n  ".join(rel_strs)

            descriptions.append(node_desc)

        return "\n".join(descriptions) if descriptions else "No nodes found"

    def _mock_describe_nodes(self, node_ids: List[str]) -> str:
        """Mock node descriptions for testing."""
        descriptions = []

        node_info = {
            "task_integration_testing": ("Integration Testing", "BLOCKED_BY test data"),
            "task_test_data_gen": ("Test Data Generation", "ASSIGNED_TO Alice"),
            "task_deployment": ("Deployment", "DEPENDS_ON integration testing"),
            "person_alice": ("Alice", "OWNS test data generation"),
            "artifact_test_db": ("Test Database", "PART_OF test infrastructure"),
        }

        for node_id in node_ids:
            if node_id in node_info:
                name, relationship = node_info[node_id]
                descriptions.append(f"- {name} ({node_id})\n  {relationship}")
            else:
                descriptions.append(f"- {node_id}")

        return "\n".join(descriptions)


# ============================================================================
# Factory Functions
# ============================================================================

def create_hololoom_warp(qdrant_client=None) -> HoloLoomWarp:
    """
    Create Warp adapter for HoloLoom.

    Args:
        qdrant_client: Optional Qdrant client (falls back to mock)

    Returns:
        HoloLoomWarp instance
    """
    return HoloLoomWarp(qdrant_client=qdrant_client)


def create_hololoom_yarn(neo4j_driver=None, kg_client=None) -> HoloLoomYarn:
    """
    Create Yarn adapter for HoloLoom.

    Args:
        neo4j_driver: Optional Neo4j driver
        kg_client: Optional HoloLoom KG client

    Returns:
        HoloLoomYarn instance
    """
    return HoloLoomYarn(neo4j_driver=neo4j_driver, kg_client=kg_client)


def create_hololoom_shuttle(
    qdrant_client=None,
    neo4j_driver=None,
    kg_client=None,
    num_mcts_simulations: int = 50,
    enable_learning: bool = True,
):
    """
    Convenience function to create a fully configured Shuttle with HoloLoom backends.

    Args:
        qdrant_client: Optional Qdrant client
        neo4j_driver: Optional Neo4j driver
        kg_client: Optional HoloLoom KG client
        num_mcts_simulations: MCTS iterations per query
        enable_learning: Whether to learn from query outcomes

    Returns:
        Configured Shuttle instance with HoloLoom adapters
    """
    from .orchestrator import Shuttle

    warp = create_hololoom_warp(qdrant_client)
    yarn = create_hololoom_yarn(neo4j_driver, kg_client)

    return Shuttle(
        warp=warp,
        yarn=yarn,
        num_mcts_simulations=num_mcts_simulations,
        enable_learning=enable_learning,
    )
