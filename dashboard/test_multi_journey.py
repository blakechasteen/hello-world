#!/usr/bin/env python3
"""
Test Multi-Journey Analysis with Universal Pattern Detection
"""

import requests
import json

# Business startup story that crosses multiple domains
test_text = """
In the ordinary world of my comfortable corporate job, I lived a routine life.
Then came the call to adventure - a business opportunity that could change everything.
At first, I refused out of fear and doubt about leaving my stable salary.
But after meeting an experienced mentor and advisor, I found the courage to begin.
I crossed the threshold by launching my MVP and committing to the startup journey.

Through early traction, I faced many tests, made new allies, and encountered fierce competitors.
As I prepared to scale, gathering resources and building my team for growth.
Then came the cash crunch - the darkest crisis when we almost ran out of runway.
Through perseverance, we achieved product-market fit and claimed our reward of sustainable growth.

Now scaling operations and bringing our solution back to the market at large scale.
We've undergone a complete transformation from idea to market leader.
Finally, we're ready to exit or build a lasting legacy and impact the industry forever.
"""

print("=" * 70)
print("MULTI-JOURNEY ANALYSIS TEST")
print("=" * 70)
print()

print("Test Text: Business startup story")
print(f"Length: {len(test_text.split())} words")
print()

# Test the multi-journey endpoint
try:
    print("Testing: POST /api/journey/analyze-multi")
    print()

    response = requests.post(
        "http://localhost:8001/api/journey/analyze-multi",
        json={
            "text": test_text,
            "journeys": ["hero", "business", "learning"],
            "include_resonance": True
        },
        timeout=30
    )

    if response.status_code == 200:
        data = response.json()

        print("[OK] Multi-Journey Analysis Complete!")
        print()

        # Summary
        print("ANALYSIS SUMMARY")
        print("-" * 70)
        print(f"Journeys Analyzed: {', '.join(data['journeys_analyzed'])}")
        print(f"Universal Patterns Found: {len(data['universal_patterns'])}")
        print(f"Recommended Journeys: {', '.join(data['recommended_journeys'])}")
        print()

        # Show each journey's results
        print("INDIVIDUAL JOURNEY RESULTS")
        print("-" * 70)
        for journey_id, result in data['journey_results'].items():
            journey_name = {
                'hero': "Hero's Journey",
                'business': 'Business Journey',
                'learning': 'Learning Journey'
            }.get(journey_id, journey_id)

            print(f"\n{journey_name}:")
            print(f"  Current Stage: {result['current_stage']}")
            print(f"  Dominant Stage: {result['dominant_stage']}")
            print(f"  Overall Progress: {result['overall_progress'] * 100:.1f}%")
            print(f"  Narrative Arc: {result['narrative_arc']}")
            print(f"  Key Transitions: {len(result['key_transitions'])}")

        # Universal Patterns (the magic!)
        print()
        print()
        print("UNIVERSAL PATTERNS DETECTED")
        print("=" * 70)
        for i, pattern in enumerate(data['universal_patterns'][:5], 1):
            print(f"\n{i}. {pattern['pattern_name']} (Energy: {pattern['energy']})")
            print(f"   Resonance Score: {pattern['resonance_score']:.2f}")
            print(f"   Avg Intensity: {pattern['avg_intensity'] * 100:.0f}%")
            print(f"   Journeys Matched:")

            for journey_id, journey_data in pattern['journeys_matched'].items():
                stage = journey_data['stage']
                intensity = journey_data['intensity']
                print(f"     - {journey_id:10s}: {stage:30s} [{intensity * 100:3.0f}%]")

        # Cross-Journey Insights
        print()
        print()
        print("CROSS-JOURNEY INSIGHTS")
        print("=" * 70)
        insights = data['cross_journey_insights']
        print(f"Highest Resonance: {insights['highest_resonance_pattern']}")
        print(f"Universal Current Stage: {insights.get('universal_current_stage', 'N/A')}")
        print(f"Average Progress Across All Journeys: {insights['avg_overall_progress'] * 100:.1f}%")

        print()
        print("=" * 70)
        print("VISUALIZATION READY!")
        print("This data powers the overlay radar chart in the UI:")
        print("  - Multiple colored polygons (one per journey)")
        print("  - Golden resonance zones where patterns align")
        print("  - Interactive journey toggles")
        print("  - Real-time cross-domain insights")

    else:
        print(f"[ERROR] API returned {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("[ERROR] Cannot connect to API")
    print("Start the server: python dashboard/enhanced_query_api.py")
    print()
    print("Expected Output:")
    print("  - Hero's Journey: 85% match (classic monomyth)")
    print("  - Business Journey: 92% match (startup narrative)")
    print("  - Learning Journey: 68% match (growth elements)")
    print()
    print("Universal Patterns:")
    print("  1. The Commitment - appears in all 3 journeys")
    print("  2. The Crisis - startup cash crunch = hero's ordeal")
    print("  3. The Breakthrough - product-market fit = reward")

except Exception as e:
    print(f"[ERROR] {e}")

print()
