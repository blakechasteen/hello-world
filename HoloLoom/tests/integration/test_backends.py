#!/usr/bin/env python3
"""
Quick Backend Connectivity Test
================================
Tests Neo4j and Qdrant connections.

Usage:
    python test_backends.py
"""

import sys
from pathlib import Path

def test_neo4j():
    """Test Neo4j connection"""
    print("\n" + "="*70)
    print("Testing Neo4j Connection")
    print("="*70)

    try:
        from neo4j import GraphDatabase
        print("✓ neo4j package installed")
    except ImportError:
        print("✗ neo4j package not found")
        print("  Install: pip install neo4j")
        return False

    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "hololoom123")
        )

        with driver.session() as session:
            result = session.run("RETURN 'Connected to Neo4j!' as message")
            message = result.single()["message"]
            print(f"✓ {message}")

            # Check version
            version_result = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions")
            for record in version_result:
                print(f"  Version: {record['versions'][0]}")

        driver.close()
        return True

    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Start Neo4j: docker-compose up -d neo4j")
        print("  2. Check status: docker ps | grep neo4j")
        print("  3. View logs: docker logs hololoom-neo4j")
        print("  4. Verify password: neo4j/hololoom123")
        return False

def test_qdrant():
    """Test Qdrant connection"""
    print("\n" + "="*70)
    print("Testing Qdrant Connection")
    print("="*70)

    try:
        from qdrant_client import QdrantClient
        print("✓ qdrant-client package installed")
    except ImportError:
        print("✗ qdrant-client package not found")
        print("  Install: pip install qdrant-client")
        return False

    try:
        client = QdrantClient(url="http://localhost:6333")

        # Check health
        health = client.http.health()
        print(f"✓ Connected to Qdrant!")
        print(f"  Status: {health}")

        # List collections
        collections = client.get_collections()
        print(f"  Collections: {len(collections.collections)}")

        return True

    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Start Qdrant: docker-compose up -d qdrant")
        print("  2. Check status: docker ps | grep qdrant")
        print("  3. View logs: docker logs hololoom-qdrant")
        print("  4. Check health: curl http://localhost:6333/health")
        return False

def test_embeddings():
    """Test sentence-transformers"""
    print("\n" + "="*70)
    print("Testing Embedding Model")
    print("="*70)

    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers package installed")
    except ImportError:
        print("✗ sentence-transformers package not found")
        print("  Install: pip install sentence-transformers")
        return False

    try:
        print("  Loading model (first time may take a while)...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Test embedding
        text = "This is a test sentence"
        embedding = model.encode(text)

        print(f"✓ Model loaded successfully")
        print(f"  Embedding dimensions: {len(embedding)}")
        print(f"  Sample vector: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]")

        return True

    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_hybrid_store():
    """Test hybrid Neo4j + Qdrant store"""
    print("\n" + "="*70)
    print("Testing Hybrid Store")
    print("="*70)

    try:
        # Add HoloLoom to path
        hololoom_path = Path(__file__).parent
        if str(hololoom_path) not in sys.path:
            sys.path.insert(0, str(hololoom_path))

        from memory.stores.hybrid_neo4j_qdrant import HybridNeo4jQdrant
        print("✓ HybridNeo4jQdrant imported")

        # Initialize store
        store = HybridNeo4jQdrant(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="hololoom123",
            qdrant_url="http://localhost:6333"
        )
        print("✓ Hybrid store initialized")

        # Test storage
        from datetime import datetime
        memory_id = store.add(
            text="Test memory for backend verification",
            user_id="test",
            timestamp=datetime.now(),
            context={"type": "test", "source": "test_backends.py"},
            metadata={}
        )
        print(f"✓ Memory stored: {memory_id}")

        # Test retrieval
        from memory.stores.hybrid_neo4j_qdrant import MemoryQuery
        query = MemoryQuery(
            text="test memory",
            user_id="test",
            limit=5
        )

        results = store.search(query, strategy="fused")
        print(f"✓ Search completed: {len(results.memories)} results")

        if results.memories:
            print(f"  First result: {results.memories[0].text[:50]}...")

        return True

    except Exception as e:
        print(f"✗ Hybrid store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*70)
    print("HoloLoom Backend Connectivity Test")
    print("="*70)

    results = {
        "Neo4j": test_neo4j(),
        "Qdrant": test_qdrant(),
        "Embeddings": test_embeddings(),
    }

    # Only test hybrid if both backends work
    if results["Neo4j"] and results["Qdrant"] and results["Embeddings"]:
        results["Hybrid Store"] = test_hybrid_store()
    else:
        print("\n" + "="*70)
        print("Skipping Hybrid Store Test (prerequisites not met)")
        print("="*70)

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 All tests passed! Backends are ready.")
        print("\nNext steps:")
        print("  1. Run: python demo_hololoom_integration.py")
        print("  2. Open Neo4j Browser: http://localhost:7474")
        print("  3. Open Qdrant Dashboard: http://localhost:6333/dashboard")
    else:
        print("\n⚠️  Some tests failed. Check troubleshooting steps above.")
        print("\nQuick fixes:")
        print("  1. Start backends: docker-compose up -d")
        print("  2. Install packages: pip install neo4j qdrant-client sentence-transformers")
        print("  3. Wait 30 seconds for services to start")

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
