"""
Query the beekeeping knowledge graph
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
username = "neo4j"
password = "hololoom123"

driver = GraphDatabase.driver(uri, auth=(username, password))

try:
    with driver.session() as session:
        # Stats
        print("=" * 70)
        print("BEEKEEPING KNOWLEDGE GRAPH - STATS")
        print("=" * 70)

        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] AS nodeType, count(*) AS count
            ORDER BY count DESC
        """)

        print("\nNode counts by type:")
        for record in result:
            print(f"  {record['nodeType']}: {record['count']}")

        # All hives
        print("\n" + "=" * 70)
        print("ALL HIVES")
        print("=" * 70)

        result = session.run("""
            MATCH (h:Hive)
            OPTIONAL MATCH (h)-[:SOURCED_FROM]->(b:Beekeeper)
            RETURN h.commonName AS hive,
                   h.configuration AS config,
                   h.genetics AS genetics,
                   h.colonyStatus AS status,
                   b.name AS source
            ORDER BY h.commonName
        """)

        for record in result:
            print(f"\n  {record['hive']}")
            print(f"    Config: {record['config']}")
            if record['genetics']:
                print(f"    Genetics: {record['genetics']}")
            if record['status']:
                print(f"    Status: {record['status']}")
            if record['source']:
                print(f"    Source: {record['source']}")

        # Population strengths
        print("\n" + "=" * 70)
        print("POPULATION STRENGTHS")
        print("=" * 70)

        result = session.run("""
            MATCH (h:Hive)<-[:STOCK_OF]-(p:PopulationStock)
            WHERE p.populationStrength IS NOT NULL
            RETURN h.commonName AS hive,
                   p.populationStrength AS strength,
                   p.framesOfBees AS frames
            ORDER BY
                CASE p.populationStrength
                    WHEN 'very_strong' THEN 1
                    WHEN 'good' THEN 2
                    WHEN 'moderate' THEN 3
                    WHEN 'very_weak' THEN 4
                END
        """)

        for record in result:
            frames = f" ({record['frames']} frames)" if record['frames'] else ""
            print(f"  {record['hive']}: {record['strength']}{frames}")

        # Critical tasks
        print("\n" + "=" * 70)
        print("CRITICAL TASKS")
        print("=" * 70)

        result = session.run("""
            MATCH (t:Task)
            WHERE t.priority = 'critical'
            OPTIONAL MATCH (t)-[:TARGETS]->(h:Hive)
            RETURN t.action AS action,
                   t.reason AS reason,
                   h.commonName AS hive
            ORDER BY t.taskId
        """)

        for record in result:
            print(f"\n  • {record['action']}")
            if record['hive']:
                print(f"    Target: {record['hive']}")
            if record['reason']:
                print(f"    Reason: {record['reason']}")

        # Risk assessments
        print("\n" + "=" * 70)
        print("WINTER MORTALITY RISK ASSESSMENT")
        print("=" * 70)

        result = session.run("""
            MATCH (h:Hive)<-[:ASSESSES]-(r:RiskAssessment)
            WHERE r.riskType = 'winter_mortality'
            RETURN h.commonName AS hive,
                   r.riskLevel AS risk,
                   r.reasoning AS reason
            ORDER BY
                CASE r.riskLevel
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 4
                END
        """)

        for record in result:
            print(f"\n  {record['hive']}: {record['risk'].upper()}")
            print(f"    {record['reason']}")

        # Spring breeding priorities
        print("\n" + "=" * 70)
        print("SPRING 2025 SPLIT PRIORITIES")
        print("=" * 70)

        result = session.run("""
            MATCH (sp:SplitPriority)-[:TARGETS]->(h:Hive)
            RETURN sp.rank AS rank,
                   h.commonName AS hive,
                   sp.reason AS reason,
                   sp.conditional AS conditional
            ORDER BY sp.rank
        """)

        for record in result:
            print(f"\n  {record['rank']}. {record['hive']}")
            print(f"     Reason: {record['reason']}")
            if record['conditional']:
                print(f"     Conditional: {record['conditional']}")

        # Treatment summary
        print("\n" + "=" * 70)
        print("TREATMENT ROUND 2 SUMMARY (Oct 12, 2024)")
        print("=" * 70)

        result = session.run("""
            MATCH (h:Hive)<-[:APPLIED_TO]-(t:Treatment)
            WHERE t.treatmentDate = '2024-10-12'
            RETURN h.commonName AS hive,
                   t.dosage AS dosage,
                   t.adjustmentReason AS reason
            ORDER BY t.dosage DESC
        """)

        for record in result:
            print(f"\n  {record['hive']}: {record['dosage']} units")
            if record['reason']:
                print(f"    Adjustment: {record['reason']}")

        # Genetics lineages
        print("\n" + "=" * 70)
        print("GENETIC LINEAGES")
        print("=" * 70)

        result = session.run("""
            MATCH (h:Hive)-[:SOURCED_FROM]->(b:Beekeeper)
            RETURN b.name AS beekeeper,
                   collect(h.commonName) AS hives,
                   b.reputation AS reputation
            ORDER BY b.name
        """)

        for record in result:
            print(f"\n  {record['beekeeper']}'s line", end="")
            if record['reputation']:
                print(f" ({record['reputation']})")
            else:
                print()
            for hive in record['hives']:
                print(f"    • {hive}")

        # Split lineage
        print("\n" + "=" * 70)
        print("SPLIT RELATIONSHIPS")
        print("=" * 70)

        result = session.run("""
            MATCH (child:Hive)-[r:SPLIT_FROM]->(parent:Hive)
            RETURN child.commonName AS child,
                   parent.commonName AS parent,
                   r.splitType AS splitType
        """)

        for record in result:
            print(f"  {record['child']} ← split from ← {record['parent']}")
            print(f"    Method: {record['splitType']}")

        # Research questions
        print("\n" + "=" * 70)
        print("RESEARCH QUESTIONS")
        print("=" * 70)

        result = session.run("""
            MATCH (rq:ResearchQuestion)
            WHERE rq.priority = 'critical'
            RETURN rq.question AS question, rq.topic AS topic
        """)

        for record in result:
            if record['topic']:
                print(f"\n  Topic: {record['topic']}")
            print(f"  Q: {record['question']}")

finally:
    driver.close()

print("\n" + "=" * 70)
print("✓ Query complete!")
print("=" * 70)