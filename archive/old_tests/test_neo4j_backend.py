"""
Test Neo4j Backend
==================

Tests the Neo4j graph backend implementation for HoloLoom.
"""

import asyncio
from HoloLoom.memory.neo4j_graph import Neo4jKG, Neo4jConfig
from HoloLoom.config import Config, KGBackend


async def test_neo4j_connection():
    """Test basic Neo4j connection and health check."""
    print("=" * 60)
    print("Test 1: Neo4j Connection")
    print("=" * 60)

    # Create config with Neo4j backend
    config = Config.fast()
    config.kg_backend = KGBackend.NEO4J
    config.neo4j_uri = "bolt://localhost:7687"
    config.neo4j_username = "neo4j"
    config.neo4j_password = "hololoom123"

    # Initialize Neo4j graph
    neo4j_config = Neo4jConfig(
        uri=config.neo4j_uri,
        username=config.neo4j_username,
        password=config.neo4j_password
    )
    graph = Neo4jKG(neo4j_config)

    try:
        # Test connection
        print("\n1. Testing connection...")
        connected = await graph.connect()
        print(f"   Connected: {connected}")
        assert connected, "Failed to connect to Neo4j"

        # Test health check
        print("\n2. Testing health check...")
        health = await graph.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Backend: {health.get('backend', 'unknown')}")
        assert health['status'] == 'healthy', f"Health check failed: {health}"

        print("\n[OK] Neo4j connection test passed")
        return True

    finally:
        await graph.close()


async def test_neo4j_crud():
    """Test Create, Read, Update, Delete operations."""
    print("\n" + "=" * 60)
    print("Test 2: Neo4j CRUD Operations")
    print("=" * 60)

    config = Config.fast()
    config.kg_backend = KGBackend.NEO4J
    config.neo4j_uri = "bolt://localhost:7687"
    config.neo4j_username = "neo4j"
    config.neo4j_password = "hololoom123"

    graph = Neo4jGraph(
        uri=config.neo4j_uri,
        username=config.neo4j_username,
        password=config.neo4j_password
    )

    try:
        await graph.connect()

        # Clear test data
        print("\n1. Clearing test data...")
        await graph.execute_cypher(
            "MATCH (n:TestEntity) DETACH DELETE n"
        )

        # Create entities
        print("\n2. Creating entities...")
        await graph.add_entity("HoloLoom", "System", {
            "description": "Neural decision-making system",
            "test": True
        })
        await graph.add_entity("Thompson Sampling", "Algorithm", {
            "description": "Exploration/exploitation strategy",
            "test": True
        })
        await graph.add_entity("Matryoshka", "Embedding", {
            "description": "Multi-scale embeddings",
            "test": True
        })
        print("   Created 3 test entities")

        # Create relationships
        print("\n3. Creating relationships...")
        await graph.add_relationship(
            "HoloLoom",
            "Thompson Sampling",
            "USES",
            {"importance": 0.9}
        )
        await graph.add_relationship(
            "HoloLoom",
            "Matryoshka",
            "USES",
            {"importance": 0.95}
        )
        print("   Created 2 relationships")

        # Query entities
        print("\n4. Querying entities...")
        result = await graph.execute_cypher(
            "MATCH (n) WHERE n.test = true RETURN n.name as name, labels(n) as labels"
        )
        print(f"   Found {len(result)} entities:")
        for record in result:
            print(f"     - {record['name']} ({record['labels']})")

        # Query relationships
        print("\n5. Querying relationships...")
        result = await graph.execute_cypher(
            """
            MATCH (a)-[r:USES]->(b)
            WHERE a.test = true
            RETURN a.name as source, b.name as target, r.importance as importance
            """
        )
        print(f"   Found {len(result)} relationships:")
        for record in result:
            print(f"     - {record['source']} -> {record['target']} (importance: {record['importance']})")

        # Test subgraph extraction
        print("\n6. Testing subgraph extraction...")
        subgraph = await graph.get_subgraph("HoloLoom", max_depth=2)
        print(f"   Subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")

        # Test entity search
        print("\n7. Testing entity search...")
        entities = await graph.search_entities("Thompson")
        print(f"   Found {len(entities)} entities matching 'Thompson':")
        for entity in entities:
            print(f"     - {entity}")

        print("\n[OK] Neo4j CRUD test passed")
        return True

    finally:
        # Cleanup
        print("\n8. Cleaning up test data...")
        await graph.execute_cypher(
            "MATCH (n) WHERE n.test = true DETACH DELETE n"
        )
        await graph.close()


async def test_neo4j_spectral():
    """Test spectral features extraction from Neo4j graph."""
    print("\n" + "=" * 60)
    print("Test 3: Neo4j Spectral Features")
    print("=" * 60)

    config = Config.fast()
    config.kg_backend = KGBackend.NEO4J
    config.neo4j_uri = "bolt://localhost:7687"
    config.neo4j_username = "neo4j"
    config.neo4j_password = "hololoom123"

    graph = Neo4jGraph(
        uri=config.neo4j_uri,
        username=config.neo4j_username,
        password=config.neo4j_password
    )

    try:
        await graph.connect()

        # Create a small test graph
        print("\n1. Creating test graph...")
        await graph.execute_cypher(
            "MATCH (n:SpectralTest) DETACH DELETE n"
        )

        nodes = ["A", "B", "C", "D", "E"]
        for node in nodes:
            await graph.execute_cypher(
                "CREATE (n:SpectralTest {name: $name})",
                {"name": node}
            )

        # Create connections
        edges = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("D", "E")]
        for source, target in edges:
            await graph.execute_cypher(
                """
                MATCH (a:SpectralTest {name: $source})
                MATCH (b:SpectralTest {name: $target})
                CREATE (a)-[:CONNECTS]->(b)
                """,
                {"source": source, "target": target}
            )

        print(f"   Created graph with {len(nodes)} nodes and {len(edges)} edges")

        # Extract spectral features
        print("\n2. Extracting spectral features...")
        features = await graph.get_spectral_features(entities=nodes)
        print(f"   Spectral features shape: {features.shape}")
        print(f"   Features: {features}")

        # Test centrality measures
        print("\n3. Computing centrality...")
        centrality = await graph.get_centrality("A")
        print(f"   Node 'A' centrality metrics:")
        for metric, value in centrality.items():
            print(f"     - {metric}: {value:.4f}")

        print("\n[OK] Neo4j spectral features test passed")
        return True

    finally:
        # Cleanup
        await graph.execute_cypher(
            "MATCH (n:SpectralTest) DETACH DELETE n"
        )
        await graph.close()


async def main():
    """Run all Neo4j tests."""
    print("\n")
    print("="*60)
    print("Neo4j Backend Test Suite")
    print("="*60)

    tests = [
        test_neo4j_connection,
        test_neo4j_crud,
        test_neo4j_spectral
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
        except Exception as e:
            print(f"\n[FAILED] Test failed: {test.__name__}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == '__main__':
    success = asyncio.run(main())
    exit(0 if success else 1)
