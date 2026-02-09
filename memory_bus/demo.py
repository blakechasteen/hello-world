"""
Memory Bus MVP demo — 2026-02-09

Exercises: resolve_entity → query → format_for_context → audit writes.

Usage:
    python -m memory_bus "What treatment should I apply to Aurora next?"
    python -m memory_bus.demo "What treatment should I apply to Aurora next?"

Requires: docker compose -f docker-compose.memory.yml up -d
"""

from __future__ import annotations

import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def seed_data(bus) -> None:
    """Seed sample entities and episodes for the demo."""
    from memory_bus.tools import MemoryBus

    logger.info("Seeding demo data...")

    # Create entities
    await bus.memory_store(
        entities=[
            {
                "id": "ent_hive_aurora",
                "type": "object",
                "name": "Hive Aurora",
                "importance": 0.8,
                "loom_origin": "demo",
                "properties": {"species": "apis_mellifera", "location": "north_yard", "frames": 10},
            },
            {
                "id": "ent_hive_nova",
                "type": "object",
                "name": "Hive Nova",
                "importance": 0.6,
                "loom_origin": "demo",
                "properties": {"species": "apis_mellifera", "location": "south_yard", "frames": 8},
            },
            {
                "id": "ent_oxalic_acid",
                "type": "concept",
                "name": "Oxalic Acid Treatment",
                "importance": 0.7,
                "loom_origin": "demo",
            },
        ],
        episodes=[
            {
                "id": "ep_inspect_jan",
                "type": "observation",
                "summary": "January inspection: Aurora has 8 frames brood, healthy queen, some varroa seen on drone brood",
                "content": "Full inspection of Aurora. 8 frames brood, 3 frames honey. Queen spotted on frame 5. Varroa mites visible on 3 drone cells. Recommend treatment before spring buildup.",
                "significance": 0.8,
                "loom_origin": "demo",
                "entity_ids": ["ent_hive_aurora"],
            },
            {
                "id": "ep_treatment_feb",
                "type": "action",
                "summary": "Applied oxalic acid vaporization to Aurora for varroa mite control",
                "content": "OA vaporization applied. 2g dose. Ambient temp 12C. Applied at 10am with sealed entrance for 10 min. Follow-up inspection in 7 days.",
                "significance": 0.7,
                "loom_origin": "demo",
                "entity_ids": ["ent_hive_aurora", "ent_oxalic_acid"],
            },
            {
                "id": "ep_inspect_feb",
                "type": "observation",
                "summary": "Follow-up inspection after treatment: varroa drop count high, treatment effective",
                "content": "7-day follow-up. Varroa drop board shows 150+ mites in 3 days. Treatment effective. Brood pattern still good. Consider second treatment in 7 days.",
                "significance": 0.7,
                "loom_origin": "demo",
                "entity_ids": ["ent_hive_aurora"],
            },
        ],
        aliases=[
            {"alias": "aurora", "entity_id": "ent_hive_aurora", "loom_id": "demo", "confidence": 0.95},
            {"alias": "hive aurora", "entity_id": "ent_hive_aurora", "loom_id": "demo", "confidence": 1.0},
            {"alias": "nova", "entity_id": "ent_hive_nova", "loom_id": "demo", "confidence": 0.95},
            {"alias": "oxalic acid", "entity_id": "ent_oxalic_acid", "loom_id": "demo", "confidence": 0.9},
        ],
        loom_id="demo",
        loop_id="demo_seed",
    )
    logger.info("Demo data seeded.")


async def run_demo(query_text: str) -> None:
    """Run the full demo pipeline."""
    from memory_bus.tools import MemoryBus

    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    pg_dsn = os.environ.get("PG_DSN", "postgresql://hololoom:hololoom@localhost:5432/hololoom")

    print(f"\n{'='*60}")
    print("  Memory Bus MVP Demo")
    print(f"{'='*60}\n")

    async with MemoryBus(neo4j_uri=neo4j_uri, pg_dsn=pg_dsn) as bus:
        # Step 0: Seed data
        await seed_data(bus)

        # Step 1: Resolve entity
        print(f"Query: {query_text!r}\n")
        print("--- Step 1: Resolve Entity ---")
        resolve = await bus.memory_resolve_entity(
            name="Aurora", loom_id="demo", loop_id="demo_loop"
        )
        if resolve.best_match:
            print(f"  Best match: {resolve.best_match.name} "
                  f"(id={resolve.best_match.entity_id}, "
                  f"confidence={resolve.best_match.confidence:.2f}, "
                  f"source={resolve.best_match.source})")
        else:
            print("  No match found!")
        print(f"  All candidates: {len(resolve.candidates)}")
        for c in resolve.candidates:
            print(f"    - {c.name} ({c.entity_id}) conf={c.confidence:.2f} src={c.source}")

        # Step 2: Query memory
        print("\n--- Step 2: Query Memory ---")
        entity_id = resolve.best_match.entity_id if resolve.best_match else None
        result = await bus.memory_query(
            text=query_text,
            entity_ids=[entity_id] if entity_id else [],
            detail_level="compact",
            loom_id="demo",
            loop_id="demo_loop",
        )
        print(f"  Resolution path: {result.resolution_path.value}")
        print(f"  Items found: {len(result.items)}")
        print(f"  Tokens used: {result.total_tokens_used}")
        print(f"  Detail level: {result.detail_level.value}")
        print(f"  Pressure tier: {result.pressure_tier.value}")

        # Step 3: Show formatted context
        print("\n--- Step 3: Formatted Context ---")
        if result.formatted_context:
            print(result.formatted_context)
        else:
            print("  (no context)")

        # Step 4: Check pressure
        print("\n--- Step 4: Pressure Status ---")
        pressure = bus.memory_pressure()
        print(f"  Tier: {pressure.tier.value}")
        print(f"  Context util: {pressure.signals.context_utilization_pct:.1f}%")
        print(f"  Policy: {json.dumps(pressure.policy, indent=4)}")

        # Step 5: Explain
        print("\n--- Step 5: Explain (provenance) ---")
        if result.items:
            item = result.items[0]
            explain = await bus.memory_explain(
                item_id=item.id, loom_id="demo", loop_id="demo_loop"
            )
            print(f"  Item: {explain.item_id}")
            print(f"  Confidence: {explain.confidence:.2f}")
            print(f"  Status: {explain.status}")
            print(f"  Derived from: {explain.derived_from}")
            print(f"  Audit trail entries: {len(explain.audit_trail)}")

        # Step 6: Check status
        print("\n--- Step 6: Memory Status ---")
        status = await bus.memory_status()
        print(f"  {json.dumps(status, indent=4)}")

        # Step 7: Show audit trail
        print("\n--- Step 7: Audit Trail (last 5) ---")
        rows = await bus._pg.get_audit_rows(loom_id="demo", limit=5)
        for row in rows:
            print(f"  [{row.action.value}] path={row.resolution_path} "
                  f"tokens={row.tokens_used} tier={row.pressure_tier.value}")

        # Step 8: Resolution path distribution
        print("\n--- Step 8: Resolution Path Distribution ---")
        dist = await bus._pg.get_resolution_path_distribution(loom_id="demo")
        for path, count in dist.items():
            print(f"  {path}: {count}")
        if dist.get("semantic", 0) > 0:
            total = sum(dist.values())
            pct = dist["semantic"] / total * 100
            if pct > 20:
                print(f"  WARNING: Semantic fallback at {pct:.1f}% (threshold: 20%)")

    print(f"\n{'='*60}")
    print("  Demo complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import asyncio
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What treatment should I apply to Aurora next?"
    asyncio.run(run_demo(query))
