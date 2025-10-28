#!/usr/bin/env python3
"""
Elegant Keep Demo - Showcasing Advanced Design Patterns

Demonstrates the elegant design patterns in Keep:
1. Fluent builders for readable object construction
2. Functional transformations for data processing
3. Composable analytics for insights
4. Narrative journaling for storytelling
5. HoloLoom integration for AI reasoning

Run with: python apps/demo_keep_elegant.py
"""

import asyncio
from datetime import datetime, timedelta

# Import with elegant patterns
from apps.keep import (
    # Core
    Apiary,
    BeeKeeper,
    # Fluent builders
    hive,
    colony,
    inspection,
    alert,
    # Functional transforms
    filter_healthy,
    filter_concerning,
    get_top_healthy_colonies,
    pipe,
    # Analytics
    ApiaryAnalytics,
    quick_health_check,
    productivity_summary,
    # Journal
    create_journal,
    EntryType,
    Sentiment,
    # HoloLoom
    export_to_memory,
)


def section(title: str):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


async def main():
    """Run elegant Keep demo."""

    section("Keep Elegant Patterns Demo")

    # =========================================================================
    # 1. Fluent Builders - Readable Object Construction
    # =========================================================================

    section("1. Fluent Builders")

    # Create apiary
    apiary = Apiary(name="Meadowbrook Apiary", location="Willow County")
    print(f"Created: {apiary}")

    # Build hives with fluent API
    print("\nBuilding hives with fluent builders...")

    hive1 = (hive("Alpha")
        .langstroth()
        .at("East meadow, near oak grove")
        .installed_on(datetime(2024, 3, 15))
        .notes("Excellent morning sun, sheltered from wind")
        .metadata(paint_color="white", frames=10)
        .build())
    apiary.add_hive(hive1)
    print(f"  + {hive1.name} (fluent builder)")

    hive2 = (hive("Beta")
        .top_bar()
        .at("West garden")
        .installed_on(datetime(2024, 4, 1))
        .build())
    apiary.add_hive(hive2)
    print(f"  + {hive2.name} (fluent builder)")

    hive3 = (hive("Gamma")
        .langstroth()
        .at("South field")
        .installed_on(datetime(2024, 4, 15))
        .build())
    apiary.add_hive(hive3)
    print(f"  + {hive3.name} (fluent builder)")

    # Build colonies with fluent API
    print("\nBuilding colonies with fluent builders...")

    colony1 = (colony()
        .in_hive(hive1.hive_id)
        .italian()
        .from_package()
        .excellent_health()
        .queen_laying()
        .population(60000)
        .queen_age(8)
        .established_on(datetime(2024, 3, 20))
        .notes("Thriving colony, prolific queen")
        .build())
    apiary.add_colony(colony1)
    print(f"  + Colony in {hive1.name} - Italian, 60k bees")

    colony2 = (colony()
        .in_hive(hive2.hive_id)
        .carniolan()
        .from_swarm()
        .healthy()
        .queen_laying()
        .population(35000)
        .established_on(datetime(2024, 4, 5))
        .build())
    apiary.add_colony(colony2)
    print(f"  + Colony in {hive2.name} - Carniolan, 35k bees")

    colony3 = (colony()
        .in_hive(hive3.hive_id)
        .italian()
        .from_nuc()
        .fair_health()
        .queen_not_laying()
        .population(20000)
        .established_on(datetime(2024, 4, 20))
        .notes("New nuc, queen needs time to establish")
        .build())
    apiary.add_colony(colony3)
    print(f"  + Colony in {hive3.name} - Italian nuc, 20k bees")

    # Build inspections with fluent API
    print("\nRecording inspections with fluent builders...")

    insp1 = (inspection()
        .for_hive(hive1.hive_id)
        .colony(colony1.colony_id)
        .routine()
        .on(datetime.now() - timedelta(days=2))
        .weather("Sunny, 74°F, light breeze")
        .temperature(74.0)
        .queen_seen()
        .eggs_present()
        .larvae_present()
        .capped_brood()
        .brood_frames(8)
        .honey_frames(6)
        .pollen_frames(3)
        .population(60000)
        .no_pests()
        .action("Checked all frames")
        .action("Added second honey super")
        .recommend("Monitor honey flow, may need third super soon")
        .inspector("Jane Doe")
        .duration(30)
        .build())
    apiary.record_inspection(insp1)
    print(f"  + Excellent inspection for {hive1.name}")

    insp2 = (inspection()
        .for_hive(hive3.hive_id)
        .colony(colony3.colony_id)
        .health_check()
        .on(datetime.now() - timedelta(days=1))
        .weather("Partly cloudy, 68°F")
        .queen_seen()
        .eggs_present(False)
        .larvae_present(False)
        .brood_frames(0)
        .honey_frames(4)
        .mites(False)
        .beetles(True)
        .finding("swarm_cells", 2)
        .action("Installed beetle traps")
        .action("Verified queen present")
        .recommend("Monitor queen performance, may need to requeen")
        .inspector("Jane Doe")
        .build())
    apiary.record_inspection(insp2)
    print(f"  + Concerning inspection for {hive3.name}")

    # =========================================================================
    # 2. Functional Transformations
    # =========================================================================

    section("2. Functional Transformations")

    colonies_list = list(apiary.colonies.values())

    # Simple filters
    print("\nApplying functional filters...")
    healthy = filter_healthy(colonies_list)
    print(f"  Healthy colonies: {len(healthy)}")

    concerning = filter_concerning(colonies_list)
    print(f"  Concerning colonies: {len(concerning)}")

    # Composable transforms
    print("\nComposable transforms (top 2 healthy by population)...")
    top_colonies = get_top_healthy_colonies(2)(colonies_list)
    for c in top_colonies:
        h = apiary.hives[c.hive_id]
        print(f"  - {h.name}: {c.population_estimate:,} bees ({c.health_status.value})")

    # =========================================================================
    # 3. Composable Analytics
    # =========================================================================

    section("3. Composable Analytics")

    analytics = ApiaryAnalytics(apiary)

    # Quick health check
    print("\nQuick health check...")
    health_check = quick_health_check(apiary)
    print(f"  Health Grade: {health_check['health_grade']}")
    print(f"  Health Score: {health_check['health_score']}/100")
    print(f"  Risk Level: {health_check['risk_level']}")
    print(f"  Top Actions:")
    for action in health_check['critical_actions'][:3]:
        print(f"    - {action}")

    # Health scoring
    print("\nDetailed health analysis...")
    health_score = analytics.compute_health_score()
    print(f"  Overall Score: {health_score['score']}/100 (Grade: {health_score['grade']})")
    print(f"  Distribution:")
    for status, count in health_score['distribution'].items():
        print(f"    {status}: {count} colonies")

    if health_score['strengths']:
        print(f"  Strengths:")
        for strength in health_score['strengths']:
            print(f"    + {strength}")

    if health_score['weaknesses']:
        print(f"  Weaknesses:")
        for weakness in health_score['weaknesses']:
            print(f"    ! {weakness}")

    # Risk assessment
    print("\nRisk assessment...")
    risk = analytics.assess_risk()
    print(f"  Overall Risk: {risk.overall_risk.upper()}")
    print(f"  Risk Factors ({len(risk.risk_factors)}):")
    for factor in risk.risk_factors[:3]:
        print(f"    - {factor}")

    # =========================================================================
    # 4. Narrative Journaling
    # =========================================================================

    section("4. Narrative Journaling")

    journal = create_journal(apiary)
    print(f"Created journal: {journal}")

    # Record narrative entries
    print("\nRecording narrative entries...")

    journal.observe(
        "Spring has arrived and the bees are incredibly active. "
        "Alpha hive is absolutely booming with brood on 8 frames!",
        hive_ids=[hive1.hive_id],
        tags=["spring", "brood", "active"],
    )
    print("  + Observation recorded")

    journal.celebrate(
        "First honey harvest from Alpha! Pulled 6 frames of beautifully capped "
        "honey. The bees have been working the wildflowers in the east meadow. "
        "Expecting around 40 lbs once extracted.",
        hive_ids=[hive1.hive_id],
        tags=["harvest", "milestone", "honey"],
        expected_yield_lbs=40.0,
    )
    print("  + Celebration recorded")

    journal.concern(
        "Gamma hive queen isn't laying yet. Saw her in the inspection but no "
        "eggs or larvae. Also spotted a few small hive beetles. Need to monitor "
        "closely and may need to requeen if no improvement in a week.",
        sentiment=Sentiment.WORRIED,
        hive_ids=[hive3.hive_id],
        tags=["queen_issue", "beetles", "monitoring"],
    )
    print("  + Concern recorded")

    journal.decide(
        "Decided to install beetle traps in Gamma and give the queen another "
        "week before considering requeening. Will check for eggs in 5 days.",
        hive_ids=[hive3.hive_id],
        tags=["decision", "treatment", "monitoring"],
    )
    print("  + Decision recorded")

    # Synthesize narrative
    print("\nSynthesizing narrative from journal...")
    since = datetime.now() - timedelta(days=30)
    narrative = journal.synthesize_narrative(since=since)
    print(f"  Summary: {narrative.summary}")
    print(f"  Key Themes: {', '.join(narrative.key_themes)}")

    if narrative.highlights:
        print(f"  Highlights:")
        for highlight in narrative.highlights[:2]:
            print(f"    - {highlight[:80]}...")

    if narrative.concerns:
        print(f"  Concerns:")
        for concern in narrative.concerns[:2]:
            print(f"    - {concern[:80]}...")

    # Extract insights
    print("\nExtracting insights from journal...")
    insights = await journal.extract_insights()
    if insights['insights']:
        print(f"  Insights:")
        for insight in insights['insights']:
            print(f"    + {insight}")

    # Timeline
    print("\nRecent timeline (merged journal + inspections)...")
    timeline = journal.get_timeline(days=7)
    for event in timeline[:5]:
        if event['type'] == 'journal':
            print(f"  [{event['timestamp'].strftime('%Y-%m-%d')}] Journal: {event['content'][:60]}...")
        else:
            print(f"  [{event['timestamp'].strftime('%Y-%m-%d')}] {event['summary']}")

    # =========================================================================
    # 5. HoloLoom Integration
    # =========================================================================

    section("5. HoloLoom Integration")

    print("\nExporting to HoloLoom memory shards...")
    shards = export_to_memory(apiary)
    print(f"  Generated {len(shards)} memory shards:")
    print(f"    - Apiary overview")
    print(f"    - {len(apiary.hives)} hive shards")
    print(f"    - {len(apiary.colonies)} colony shards")
    print(f"    - {min(20, len(apiary.inspections))} recent inspection shards")
    print(f"    - {min(10, len(apiary.get_active_alerts()))} alert shards")

    # Sample shard content
    if shards:
        print(f"\n  Sample shard (Apiary Overview):")
        print(f"    ID: {shards[0].id}")
        print(f"    Content: {shards[0].text[:150]}...")

    # =========================================================================
    # Summary
    # =========================================================================

    section("Demo Complete - Elegant Patterns Showcased")

    print("\nElegant patterns demonstrated:")
    print("  + Fluent Builders - Readable, chainable object construction")
    print("  + Functional Transforms - Composable data processing")
    print("  + Composable Analytics - Rich insights through composition")
    print("  + Narrative Journaling - Temporal storytelling and pattern recognition")
    print("  + HoloLoom Integration - Seamless AI reasoning integration")

    print("\nKey benefits:")
    print("  - Clean, expressive APIs")
    print("  - Testable, reusable components")
    print("  - Protocol-based extensibility")
    print("  - Separation of concerns")
    print("  - Graceful degradation")

    print("\nNext steps:")
    print("  - Explore apps/keep/protocols.py for extension points")
    print("  - See apps/keep/transforms.py for more functional utilities")
    print("  - Check apps/keep/analytics.py for advanced insights")
    print("  - Review apps/keep/journal.py for narrative patterns")

    # Generate comprehensive report
    print("\n" + "-" * 70)
    print("Generating comprehensive analytics report...")
    report = analytics.generate_report()
    print(f"Report timestamp: {report['timestamp']}")
    print(f"Apiary: {report['apiary']['name']}")
    print(f"Health score: {report['health']['score']['score']}/100")
    print(f"Risk level: {report['risk']['assessment']['overall_risk']}")


if __name__ == "__main__":
    asyncio.run(main())
