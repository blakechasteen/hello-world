"""Quick test to verify Neo4j and Qdrant are accessible."""
import sys
import os

# Fix Windows encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

def test_neo4j():
    """Test Neo4j connection."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "hololoom123")
        )
        with driver.session() as session:
            result = session.run("RETURN 'Neo4j Connected!' AS message")
            message = result.single()["message"]
            print(f"✓ Neo4j: {message}")
        driver.close()
        return True
    except Exception as e:
        print(f"✗ Neo4j Error: {e}")
        return False

def test_qdrant():
    """Test Qdrant connection."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        print(f"✓ Qdrant: Connected! Collections: {len(collections.collections)}")
        return True
    except Exception as e:
        print(f"✗ Qdrant Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing HoloLoom backend services...\n")

    neo4j_ok = test_neo4j()
    qdrant_ok = test_qdrant()

    print("\n" + "="*50)
    if neo4j_ok and qdrant_ok:
        print("✓ All services operational!")
        sys.exit(0)
    else:
        print("✗ Some services failed")
        sys.exit(1)