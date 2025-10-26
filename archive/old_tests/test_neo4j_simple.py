"""
Simple Neo4j Cypher test - standalone
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from neo4j import GraphDatabase

print("=== Neo4j Cypher Demo ===\n")

# Connect to Neo4j
uri = "bolt://localhost:7687"
username = "neo4j"
password = "hololoom123"

print(f"Connecting to {uri}...")
driver = GraphDatabase.driver(uri, auth=(username, password))

try:
    driver.verify_connectivity()
    print("✓ Connected!\n")

    with driver.session() as session:
        # Clear database
        print("Clearing database...")
        session.run("MATCH (n) DETACH DELETE n")

        # Create AI/ML knowledge graph
        print("\nBuilding AI/ML knowledge graph...")

        cypher = """
        // Neural networks
        MERGE (ml:Entity {name: 'machine_learning'})
        MERGE (nn:Entity {name: 'neural_network'})
        MERGE (dl:Entity {name: 'deep_learning'})
        MERGE (nn)-[:RELATES {type: 'IS_A', weight: 1.0}]->(ml)
        MERGE (dl)-[:RELATES {type: 'IS_A', weight: 1.0}]->(nn)

        // Transformers
        MERGE (transformer:Entity {name: 'transformer'})
        MERGE (attention:Entity {name: 'attention'})
        MERGE (mha:Entity {name: 'multi-head_attention'})
        MERGE (sa:Entity {name: 'self-attention'})
        MERGE (transformer)-[:RELATES {type: 'IS_A', weight: 1.0}]->(dl)
        MERGE (attention)-[:RELATES {type: 'PART_OF', weight: 1.0}]->(transformer)
        MERGE (mha)-[:RELATES {type: 'IS_A', weight: 1.0}]->(attention)
        MERGE (sa)-[:RELATES {type: 'IS_A', weight: 1.0}]->(attention)

        // Specific models
        MERGE (bert:Entity {name: 'BERT'})
        MERGE (gpt:Entity {name: 'GPT'})
        MERGE (claude:Entity {name: 'Claude'})
        MERGE (bi:Entity {name: 'bidirectional'})
        MERGE (auto:Entity {name: 'autoregressive'})
        MERGE (bert)-[:RELATES {type: 'IS_A', weight: 1.0}]->(transformer)
        MERGE (gpt)-[:RELATES {type: 'IS_A', weight: 1.0}]->(transformer)
        MERGE (claude)-[:RELATES {type: 'IS_A', weight: 1.0}]->(transformer)
        MERGE (bert)-[:RELATES {type: 'USES', weight: 0.9}]->(bi)
        MERGE (gpt)-[:RELATES {type: 'USES', weight: 0.9}]->(auto)

        // Training
        MERGE (bp:Entity {name: 'backpropagation'})
        MERGE (gd:Entity {name: 'gradient_descent'})
        MERGE (transformer)-[:RELATES {type: 'USES', weight: 1.0}]->(bp)
        MERGE (dl)-[:RELATES {type: 'USES', weight: 1.0}]->(gd)
        MERGE (bp)-[:RELATES {type: 'USES', weight: 1.0}]->(gd)

        // Applications
        MERGE (nlp:Entity {name: 'NLP'})
        MERGE (textgen:Entity {name: 'text_generation'})
        MERGE (convo:Entity {name: 'conversation'})
        MERGE (bert)-[:RELATES {type: 'APPLIES_TO', weight: 1.0}]->(nlp)
        MERGE (gpt)-[:RELATES {type: 'APPLIES_TO', weight: 1.0}]->(textgen)
        MERGE (claude)-[:RELATES {type: 'APPLIES_TO', weight: 1.0}]->(convo)

        // HoloLoom
        MERGE (hololoom:Entity {name: 'HoloLoom'})
        MERGE (kg:Entity {name: 'knowledge_graph'})
        MERGE (ts:Entity {name: 'Thompson_sampling'})
        MERGE (hololoom)-[:RELATES {type: 'USES', weight: 1.0}]->(nn)
        MERGE (hololoom)-[:RELATES {type: 'USES', weight: 1.0}]->(kg)
        MERGE (hololoom)-[:RELATES {type: 'USES', weight: 0.8}]->(ts)
        """

        session.run(cypher)
        print("✓ Graph created!")

        # Get stats
        result = session.run("""
            MATCH (n)
            OPTIONAL MATCH ()-[r]->()
            RETURN count(DISTINCT n) AS nodes, count(r) AS edges
        """)
        record = result.single()
        print(f"\nGraph statistics:")
        print(f"  Nodes: {record['nodes']}")
        print(f"  Edges: {record['edges']}")

        # Query 1: Types of attention
        print("\n" + "="*70)
        print("Query 1: What are the types of attention?")
        print("="*70)

        result = session.run("""
            MATCH (type)-[r:RELATES]->(attention:Entity {name: 'attention'})
            WHERE r.type = 'IS_A'
            RETURN type.name AS attention_type, r.weight AS confidence
            ORDER BY confidence DESC
        """)

        for record in result:
            print(f"  • {record['attention_type']} (confidence: {record['confidence']})")

        # Query 2: What is BERT?
        print("\n" + "="*70)
        print("Query 2: What is BERT's neighborhood?")
        print("="*70)

        result = session.run("""
            MATCH (bert:Entity {name: 'BERT'})-[r:RELATES]-(related)
            RETURN DISTINCT
                related.name AS entity,
                r.type AS relationship,
                CASE
                  WHEN startNode(r).name = 'BERT' THEN 'outgoing'
                  ELSE 'incoming'
                END AS direction
            ORDER BY direction, entity
        """)

        print("  BERT connections:")
        for record in result:
            arrow = "-->" if record['direction'] == 'outgoing' else "<--"
            print(f"  • BERT {arrow}[{record['relationship']}] {record['entity']}")

        # Query 3: Shortest path
        print("\n" + "="*70)
        print("Query 3: Shortest path from HoloLoom to machine_learning")
        print("="*70)

        result = session.run("""
            MATCH path = shortestPath(
                (hololoom:Entity {name: 'HoloLoom'})-[:RELATES*1..5]-(ml:Entity {name: 'machine_learning'})
            )
            RETURN [node IN nodes(path) | node.name] AS path,
                   [r IN relationships(path) | r.type] AS edge_types,
                   length(path) AS hops
        """)

        record = result.single()
        if record:
            path = record['path']
            edge_types = record['edge_types']
            print(f"  Path ({record['hops']} hops):")
            for i, node in enumerate(path):
                print(f"    {node}", end="")
                if i < len(edge_types):
                    print(f" --[{edge_types[i]}]-->", end=" ")
            print()

        # Query 4: Transformer-based models
        print("\n" + "="*70)
        print("Query 4: What models are transformers?")
        print("="*70)

        result = session.run("""
            MATCH (model)-[r:RELATES]->(transformer:Entity {name: 'transformer'})
            WHERE r.type = 'IS_A'
            RETURN model.name AS model, r.weight AS confidence
            ORDER BY confidence DESC
        """)

        print("  Transformer-based models:")
        for record in result:
            print(f"  • {record['model']} (confidence: {record['confidence']})")

        # Query 5: Most connected entities
        print("\n" + "="*70)
        print("Query 5: Most connected entities")
        print("="*70)

        result = session.run("""
            MATCH (e:Entity)-[r:RELATES]-()
            WITH e, count(r) AS degree
            RETURN e.name AS entity, degree
            ORDER BY degree DESC
            LIMIT 5
        """)

        print("  Most connected:")
        for record in result:
            print(f"  • {record['entity']}: {record['degree']} connections")

        # Query 6: What USES what?
        print("\n" + "="*70)
        print("Query 6: What USES what?")
        print("="*70)

        result = session.run("""
            MATCH (a)-[r:RELATES]->(b)
            WHERE r.type = 'USES'
            RETURN a.name AS user, b.name AS uses, r.weight AS confidence
            ORDER BY confidence DESC
        """)

        print("  Usage relationships:")
        for record in result:
            print(f"  • {record['user']} uses {record['uses']} (confidence: {record['confidence']})")

        # Query 7: Aggregation - count relationships by type
        print("\n" + "="*70)
        print("Query 7: Relationship type distribution")
        print("="*70)

        result = session.run("""
            MATCH ()-[r:RELATES]->()
            RETURN r.type AS rel_type, count(*) AS count
            ORDER BY count DESC
        """)

        print("  Relationship types:")
        for record in result:
            print(f"  • {record['rel_type']}: {record['count']}")

finally:
    driver.close()
    print("\n✓ Demo complete!")