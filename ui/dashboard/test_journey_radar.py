#!/usr/bin/env python3
"""
Test the Hero's Journey Radar Chart Integration
"""

import requests
import json

# Test text with hero's journey elements
test_text = """
In the ordinary world of his comfortable home, young hero lived a mundane daily routine.
Then came the call to adventure - a mysterious invitation that would change everything.
At first, he refused the call, filled with fear and doubt about leaving his familiar world.
But after meeting his wise mentor, who gave him guidance and training, he found courage.
He crossed the threshold, committing to begin his journey into the unknown.
Along the way, he faced many tests and made allies, while also encountering dangerous enemies.
As he approached the inmost cave, gathering his strength and preparing for what lay ahead.
The ordeal came - a darkest crisis and confrontation with death itself.
But through the battle, he achieved victory and claimed his reward.
Now on the road back, pursued by consequences of his actions, he raced toward home.
The final test came - a resurrection, a climactic purification and transformation.
Finally, he returned with the elixir, bringing wisdom and gifts to share with his people.
His journey was complete, and he had found new life through change.
"""

print("HERO'S JOURNEY RADAR CHART TEST")
print("=" * 70)
print()

# Test the API endpoint
print("Testing API endpoint: POST /api/journey/analyze")
print()

try:
    response = requests.post(
        "http://localhost:8001/api/journey/analyze",
        json={
            "text": test_text,
            "domain": "mythology"
        }
    )

    if response.status_code == 200:
        data = response.json()

        print("[OK] API Response Success!")
        print()
        print(f"Current Stage: {data['current_stage']}")
        print(f"Dominant Stage: {data['dominant_stage']}")
        print(f"Overall Progress: {data['overall_progress'] * 100:.1f}%")
        print(f"Narrative Arc: {data['narrative_arc']}")
        print(f"Key Transitions: {len(data['key_transitions'])}")
        print()

        print("📊 Stage Metrics:")
        print("-" * 70)

        for stage, metrics in data['stage_metrics'].items():
            intensity = metrics['intensity']
            completion = metrics['completion']
            relevance = metrics['relevance']

            bar = "█" * int(intensity * 20)
            print(f"{stage:25s} | {bar:20s} {intensity*100:3.0f}% | C:{completion*100:3.0f}% | R:{relevance*100:3.0f}%")

            if metrics['keywords_found']:
                print(f"{'':25s}   Keywords: {', '.join(metrics['keywords_found'])}")

        print()
        print("=" * 70)
        print("[*] The radar chart will visualize these metrics in real-time!")
        print("   - Orange polygon shows intensity")
        print("   - Purple dashed line shows completion")
        print("   - Interactive labels and tooltips")
        print("   - Smooth animations and transitions")

    else:
        print(f"[ERROR] API Error: {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("[ERROR] Cannot connect to API server")
    print("   Make sure the server is running: python dashboard/enhanced_query_api.py")
    print()
    print("[*] Manual Test - Analyzing Journey Metrics:")
    print()

    # Fallback: show what metrics would look like
    print("Sample metrics that would be returned:")
    print("-" * 70)
    stages = [
        ("Ordinary World", 0.8, 1.0, 0.7),
        ("Call to Adventure", 0.9, 1.0, 0.8),
        ("Refusal of Call", 0.7, 1.0, 0.6),
        ("Meeting Mentor", 0.85, 1.0, 0.75),
        ("Crossing Threshold", 0.8, 1.0, 0.7),
        ("Tests, Allies, Enemies", 0.75, 0.9, 0.65),
        ("Approach Inmost Cave", 0.7, 0.8, 0.6),
        ("Ordeal", 0.9, 0.7, 0.8),
        ("Reward", 0.7, 0.6, 0.6),
        ("Road Back", 0.75, 0.5, 0.65),
        ("Resurrection", 0.8, 0.4, 0.7),
        ("Return with Elixir", 0.85, 0.3, 0.75),
    ]

    for stage, intensity, completion, relevance in stages:
        bar = "█" * int(intensity * 20)
        print(f"{stage:25s} | {bar:20s} {intensity*100:3.0f}% | C:{completion*100:3.0f}% | R:{relevance*100:3.0f}%")

print()
print("Next Steps:")
print("1. Start the frontend: cd dashboard && npm run dev")
print("2. Open http://localhost:5173")
print("3. Enter some narrative text and enable streaming")
print("4. Watch the radar chart come alive with real-time metrics!")
print()
