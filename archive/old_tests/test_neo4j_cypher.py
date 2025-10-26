"""
Test Neo4j with some Cypher queries
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.memory.neo4j_graph import Neo4jKG, Neo4jConfig
from HoloLoom.memory.graph import KGEdge

print("=== Neo4j Cypher Demo ===\n")

# Connect to Neo4j
print("Connecting to Neo4j...")
config = Neo4jConfig()
kg = Neo4jKG(config)
print(f"✓ Connected to {config.uri}\n")

# Clear existing data
print("Clearing existing data...")
kg.clear()

# Build a knowledge graph about AI/ML
print("\nBuilding AI/ML knowledge graph...")
edges = [
    # Neural networks
    KGEdge("neural_network", "machine_learning", "IS_A", 1.0),
    KGEdge("deep_learning", "neural_network", "IS_A", 1.0),

    # Transformers
    KGEdge("transformer", "deep_learning", "IS_A", 1.0),
    KGEdge("attention", "transformer", "PART_OF", 1.0),
    KGEdge("multi-head_attention", "attention", "IS_A", 1.0),
    KGEdge("self-attention", "attention", "IS_A", 1.0),

    # Specific models
    KGEdge("BERT", "transformer", "IS_A", 1.0),
    KGEdge("GPT", "transformer", "IS_A", 1.0),
    KGEdge("Claude", "transformer", "IS_A", 1.0),
    KGEdge("BERT", "bidirectional", "USES", 0.9),
    KGEdge("GPT", "autoregressive", "USES", 0.9),

    # Training techniques
    KGEdge("transformer", "backpropagation", "USES", 1.0),
    KGEdge("deep_learning", "gradient_descent", "USES", 1.0),
    KGEdge("backpropagation", "gradient_descent", "USES", 1.0),

    # Applications
    KGEdge("BERT", "NLP", "APPLIES_TO", 1.0),
    KGEdge("GPT", "text_generation", "APPLIES_TO", 1.0),
    KGEdge("Claude", "conversation", "APPLIES_TO", 1.0),

    # HoloLoom specific
    KGEdge("HoloLoom", "neural_network", "USES", 1.0),
    KGEdge("HoloLoom", "knowledge_graph", "USES", 1.0),
    KGEdge("HoloLoom", "Thompson_sampling", "USES", 0.8),
]

kg.add_edges(edges)
print(f"✓ Added {len(edges)} edges")

# Graph stats
stats = kg.stats()
print(f"\nGraph statistics:")
print(f"  Nodes: {stats['num_nodes']}")
print(f"  Edges: {stats['num_edges']}")
print(f"  Avg degree: {stats['avg_degree']:.2f}")

# ============================================================================
# Cypher Query 1: Find all types of attention
# ============================================================================
print("\n" + "="*70)
print("Query 1: What are the types of attention?")
print("="*70)

results = kg.run_cypher("""
    MATCH (type)-[r:RELATES]->(attention:Entity {name: 'attention'})
    WHERE r.type = 'IS_A'
    RETURN type.name AS attention_type, r.weight AS confidence
    ORDER BY confidence DESC
""")

for record in results:
    print(f"  • {record['attention_type']} (confidence: {record['confidence']})")

# ============================================================================
# Cypher Query 2: What is BERT?
# ============================================================================
print("\n" + "="*70)
print("Query 2: What is BERT?")
print("="*70)

results = kg.run_cypher("""
    MATCH path = (bert:Entity {name: 'BERT'})-[:RELATES*1..3]-(related)
    WHERE ALL(r IN relationships(path) WHERE r.type IN ['IS_A', 'USES', 'APPLIES_TO'])
    RETURN DISTINCT
        related.name AS entity,
        [r IN relationships(path) | r.type][0] AS relationship,
        length(path) AS distance
    ORDER BY distance, entity
    LIMIT 10
""")

print("  BERT is related to:")
for record in results:
    print(f"  • {record['entity']} ({record['relationship']}, distance: {record['distance']})")

# ============================================================================
# Cypher Query 3: Path from HoloLoom to machine_learning
# ============================================================================
print("\n" + "="*70)
print("Query 3: How is HoloLoom connected to machine_learning?")
print("="*70)

results = kg.run_cypher("""
    MATCH path = shortestPath(
        (hololoom:Entity {name: 'HoloLoom'})-[:RELATES*1..5]-(ml:Entity {name: 'machine_learning'})
    )
    RETURN [node IN nodes(path) | node.name] AS path,
           [r IN relationships(path) | r.type] AS edge_types,
           length(path) AS hops
    LIMIT 5
""")

for record in results:
    path = record['path']
    edge_types = record['edge_types']
    print(f"  Path ({record['hops']} hops):")
    for i, node in enumerate(path):
        if i < len(edge_types):
            print(f"    {node} --[{edge_types[i]}]--> ", end="")
        else:
            print(node)

# ============================================================================
# Cypher Query 4: What uses transformers?
# ============================================================================
print("\n" + "="*70)
print("Query 4: What models are transformers?")
print("="*70)

results = kg.run_cypher("""
    MATCH (model)-[r:RELATES]->(transformer:Entity {name: 'transformer'})
    WHERE r.type = 'IS_A'
    RETURN model.name AS model, r.weight AS confidence
    ORDER BY confidence DESC
""")

print("  Transformer-based models:")
for record in results:
    print(f"  • {record['model']} (confidence: {record['confidence']})")

# ============================================================================
# Cypher Query 5: Neighborhood expansion
# ============================================================================
print("\n" + "="*70)
print("Query 5: What's in the neighborhood of 'deep_learning'?")
print("="*70)

results = kg.run_cypher("""
    MATCH (dl:Entity {name: 'deep_learning'})-[r:RELATES]-(neighbor)
    RETURN neighbor.name AS entity,
           r.type AS relationship,
           CASE
             WHEN startNode(r).name = 'deep_learning' THEN 'outgoing'
             ELSE 'incoming'
           END AS direction
    ORDER BY direction, entity
""")

print("  Neighbors:")
for record in results:
    arrow = "-->" if record['direction'] == 'outgoing' else "<--"
    print(f"  • deep_learning {arrow}[{record['relationship']}] {record['entity']}")

# ============================================================================
# Cypher Query 6: Find central nodes (high degree)
# ============================================================================
print("\n" + "="*70)
print("Query 6: Most connected entities")
print("="*70)

results = kg.run_cypher("""
    MATCH (e:Entity)-[r:RELATES]-()
    WITH e, count(r) AS degree
    RETURN e.name AS entity, degree
    ORDER BY degree DESC
    LIMIT 5
""")

print("  Most connected:")
for record in results:
    print(f"  • {record['entity']}: {record['degree']} connections")

# ============================================================================
# Cypher Query 7: Pattern matching - What USES what?
# ============================================================================
print("\n" + "="*70)
print("Query 7: What USES what?")
print("="*70)

results = kg.run_cypher("""
    MATCH (a)-[r:RELATES]->(b)
    WHERE r.type = 'USES'
    RETURN a.name AS user, b.name AS uses, r.weight AS confidence
    ORDER BY confidence DESC
""")

print("  Usage relationships:")
for record in results:
    print(f"  • {record['user']} uses {record['uses']} (confidence: {record['confidence']})")

# Close connection
kg.close()
print("\n✓ Demo complete!")