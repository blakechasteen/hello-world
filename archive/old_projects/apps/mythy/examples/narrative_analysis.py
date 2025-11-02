"""
Mythy Narrative Analysis Example
=================================
Demonstrates how to use Mythy to analyze narrative structure.

This example:
1. Analyzes Campbell's Hero's Journey stages
2. Detects characters and archetypes
3. Performs 5-level matryoshka depth analysis
4. Applies cross-domain narrative adaptation
"""

import asyncio
from mythy.api import NarrativeAnalyzerAPI, create_api


# Sample narrative texts
NARRATIVES = {
    "odyssey_return": """
    In the shadow of Mount Olympus, Odysseus stood before his home at last.
    Twenty years of trials, monsters, and gods had forged him anew.
    Athena appeared: 'The treasure you bring is not gold, but wisdom.'
    He bowed his head. 'The trials taught me that the greatest enemy is pride.'
    'And the greatest victory,' she replied, 'is returning home with humility.'
    """,

    "business_pivot": """
    The startup founders faced their moment of truth.
    Six months of building the wrong product had taught them everything.
    Their advisor smiled: 'You didn't fail. You learned what customers actually need.'
    They pivoted, applying hard-won insights to a simpler solution.
    Three months later, users loved it. The journey was the real product.
    """,

    "personal_transformation": """
    In therapy, she confronted the shadow she'd been running from.
    Years of fear dissolved when she finally faced the truth.
    Her therapist said gently: 'The monster was protecting you all along.'
    By accepting her pain, she found unexpected strength.
    Returning to her life, everything looked different yet familiar.
    """,
}


async def main():
    """Run narrative analysis example."""
    print("=" * 70)
    print("Mythy - Narrative Analysis Example")
    print("=" * 70)
    print()

    # 1. Initialize API
    print("1. Initializing Mythy API...")
    api = create_api(enable_cache=True, cache_capacity=1000)

    async with api:
        print("   ✓ API initialized")
        print(f"   ✓ Cache enabled: {api._enable_cache}")
        print(f"   ✓ Cache capacity: {api._cache_capacity}")
        print()

        # 2. Analyze Hero's Journey
        print("2. Analyzing Campbell's Hero's Journey...")
        print()

        result = await api.analyze_narrative(NARRATIVES["odyssey_return"])

        print(f"   Text: {NARRATIVES['odyssey_return'][:60]}...")
        print()
        print(f"   Primary Stage:    {result.primary_stage}")
        print(f"   Confidence:       {result.stage_confidence:.2%}")
        print(f"   Function:         {result.narrative_function}")
        print(f"   Overall confidence: {result.bayesian_confidence:.2%}")
        print()

        # Top 5 stages
        print("   Top 5 Campbell Stages:")
        sorted_stages = sorted(
            result.all_stage_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        for i, (stage, score) in enumerate(sorted_stages, 1):
            print(f"      {i}. {stage:<25} {score:>6.2%}")

        print()

        # Detected characters
        print(f"   Characters detected: {result.character_count}")
        for char in result.detected_characters[:3]:
            print(f"      • {char['name']:<15} ({char['mythology']}, {char['confidence']:.2%})")

        print()

        # Archetypes
        print(f"   Primary Archetypes:")
        for archetype in result.primary_archetypes[:3]:
            score = result.archetype_scores.get(archetype, 0.0)
            print(f"      • {archetype:<15} {score:>6.2%}")

        print()

        # Themes
        print(f"   Themes:")
        for theme in result.themes[:3]:
            score = result.theme_scores.get(theme, 0.0)
            print(f"      • {theme:<20} {score:>6.2%}")

        print()
        print(f"   Analysis time: {result.duration_ms:.1f}ms")
        print()

        # 3. Matryoshka depth analysis
        print("3. Performing 5-level depth analysis...")
        print()

        depth_result = await api.analyze_depth(NARRATIVES["odyssey_return"])

        print(f"   Max depth:    {depth_result.max_depth_achieved}")
        print(f"   Gates opened: {len(depth_result.gates_unlocked)}/{depth_result.total_gates}")
        print()

        print("   Interpretations by level:")
        print()

        if depth_result.surface_literal:
            print(f"   1. SURFACE (literal):")
            print(f"      {depth_result.surface_literal[:80]}...")
            print()

        if depth_result.symbolic_metaphor:
            print(f"   2. SYMBOLIC (metaphor):")
            print(f"      {depth_result.symbolic_metaphor[:80]}...")
            print()

        if depth_result.archetypal_pattern:
            print(f"   3. ARCHETYPAL (pattern):")
            print(f"      {depth_result.archetypal_pattern[:80]}...")
            print()

        if depth_result.mythic_resonance:
            print(f"   4. MYTHIC (resonance):")
            print(f"      {depth_result.mythic_resonance[:80]}...")
            print()

        if depth_result.cosmic_truth:
            print(f"   5. COSMIC (truth):")
            print(f"      {depth_result.cosmic_truth[:80]}...")
            print()

        print(f"   Analysis time: {depth_result.duration_ms:.1f}ms")
        print()

        # 4. Cross-domain analysis
        print("4. Cross-domain narrative adaptation...")
        print()

        # Analyze business narrative
        business_result = await api.analyze_cross_domain(
            NARRATIVES["business_pivot"],
            domain="business"
        )

        print(f"   Domain: {business_result.domain}")
        print(f"   Fit score: {business_result.domain_fit_score:.2%}")
        print()

        print("   Domain mappings (narrative → business):")
        if business_result.domain_mappings:
            for narrative_term, business_term in list(business_result.domain_mappings.items())[:5]:
                print(f"      {narrative_term:<20} → {business_term}")

        print()

        if business_result.domain_insights:
            print("   Domain insights:")
            for insight in business_result.domain_insights[:3]:
                print(f"      • {insight}")

        print()
        print(f"   Analysis time: {business_result.duration_ms:.1f}ms")
        print()

        # 5. Personal transformation analysis
        print("5. Analyzing personal transformation narrative...")
        print()

        personal_result = await api.analyze_cross_domain(
            NARRATIVES["personal_transformation"],
            domain="personal"
        )

        print(f"   Domain: {personal_result.domain}")
        base = personal_result.base_analysis
        if base and "narrative_arc" in base:
            stage = base["narrative_arc"].get("primary_stage", "unknown")
            print(f"   Stage (in personal context): {stage}")

        print()

        # 6. Cache performance
        print("6. Testing cache performance...")
        print()

        # Re-analyze same text (should hit cache)
        import time
        start = time.time()
        cached_result = await api.analyze_narrative(NARRATIVES["odyssey_return"], use_cache=True)
        cached_time = (time.time() - start) * 1000

        print(f"   First analysis:  {result.duration_ms:.1f}ms")
        print(f"   Cached analysis: {cached_time:.1f}ms")
        print(f"   Speedup:         {result.duration_ms / cached_time:.1f}x")
        print()

        # 7. System status
        print("7. System status...")
        status = await api.get_status()

        print()
        print(f"   Status:           {status.status}")
        print(f"   Uptime:           {status.uptime_seconds:.1f}s")
        print(f"   Total analyses:   {status.total_analyses}")
        print(f"   Avg time:         {status.avg_analysis_time_ms:.1f}ms")
        print(f"   Cache hit rate:   {status.cache_hit_rate:.1%}")
        print(f"   Cache size:       {status.cache_size}/{status.cache_capacity}")
        print()

        print("   Components:")
        print(f"      • Intelligence:   {'✓' if status.intelligence_ready else '✗'}")
        print(f"      • Depth analyzer: {'✓' if status.depth_analyzer_ready else '✗'}")
        print(f"      • Cross-domain:   {'✓' if status.cross_domain_ready else '✗'}")
        print(f"      • Cache:          {'✓' if status.cache_ready else '✗'}")
        print()

        # 8. Analytics
        print("8. Analytics summary...")
        analytics = await api.get_analytics()

        print()
        print(f"   Performance:")
        perf = analytics.get("performance", {})
        print(f"      • Total analyses: {perf.get('total_analyses', 0)}")
        print(f"      • Avg time:       {perf.get('avg_analysis_time_ms', 0):.1f}ms")

        print()
        print(f"   Cache metrics:")
        cache = analytics.get("cache_metrics", {})
        print(f"      • Hits:      {cache.get('hits', 0)}")
        print(f"      • Misses:    {cache.get('misses', 0)}")
        print(f"      • Hit rate:  {cache.get('hit_rate', 0):.1%}")
        print()

    print("=" * 70)
    print("Example complete!")
    print()
    print("Key insights:")
    print("  • Campbell's Hero's Journey maps to all domains")
    print("  • Depth analysis reveals layered meanings")
    print("  • Cross-domain adaptation enables broad application")
    print("  • Caching provides 10-100x speedup")
    print()
    print("Next steps:")
    print("  • Analyze your own narratives")
    print("  • Try different domains (science, history, product)")
    print("  • Build narrative-aware applications")
    print("  • Integrate with writing tools")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
