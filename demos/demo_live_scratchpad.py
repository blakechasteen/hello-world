#!/usr/bin/env python3
"""
Live Scratchpad Demo
====================

Demonstrates the live transcription scratchpad system with simulated
audio transcriptions.

This demo shows:
1. Domain detection from natural language
2. Template auto-population
3. Field extraction from transcriptions
4. Finalization and memory storage

Author: Claude Code
Date: November 2025
"""

import asyncio
from datetime import datetime

from hololoom.spinningWheel.live_scratchpad import LiveScratchpad


# ============================================================================
# Demo Transcriptions
# ============================================================================

DEMO_TRANSCRIPTIONS = {
    'recipe': """
    Recipe for chocolate chip cookies. Serves 24 cookies.
    Prep time 15 minutes, cook time 12 minutes.
    Ingredients: 1 cup butter softened, 3/4 cup sugar, 3/4 cup brown sugar packed,
    2 eggs, 2 teaspoons vanilla extract, 2 and 1/4 cups flour,
    1 teaspoon baking soda, 1 teaspoon salt, 2 cups chocolate chips.
    First preheat oven to 375 degrees. Then cream butter and sugars until fluffy.
    Next beat in eggs and vanilla. Finally stir in flour mixture and chocolate chips.
    """,

    'bee_inspection': """
    Inspected Hive 3 today. Temperature was -5 degrees Celsius.
    Weather was cloudy and windy. Colony looks weak, brood pattern is spotty.
    I saw the queen but she doesn't seem very active.
    Honey stores are low. I noticed some varroa mites on a few bees.
    The colony will need feeding soon and we should treat for mites.
    """,

    'budget': """
    Spent $45.99 at the grocery store today. Bought food for the week
    including vegetables, chicken, and some snacks. Paid with my credit card.
    """,

    'time_tracking': """
    Worked on the HoloLoom project for 3 hours today.
    Task was implementing the live scratchpad feature.
    Started at 9:00 and finished at 12:00. This is billable time for the client.
    """,

    'expense': """
    Expense from Amazon for office supplies. Receipt number AMZ-12345.
    Amount was $127.50 for a new keyboard and mouse.
    This is reimbursable under the equipment budget.
    """,

    'sop': """
    SOP for winter beehive inspection. Purpose is to assess colony health
    during cold weather without causing harm. First step, check the weather
    forecast and only inspect when temperature is above 50 degrees.
    Next, prepare all equipment including smoker and hive tool.
    Then gently open the hive and observe activity. Finally, close up quickly
    to preserve heat. This should be done monthly during winter.
    Safety warning: wear protective gear at all times.
    """
}


# ============================================================================
# Demo Functions
# ============================================================================

async def demo_domain_detection():
    """Demo 1: Domain detection from transcriptions"""
    print("=" * 70)
    print("DEMO 1: Domain Detection")
    print("=" * 70)

    from hololoom.spinningWheel.domain_router import DomainRouter

    router = DomainRouter()

    for domain, transcription in DEMO_TRANSCRIPTIONS.items():
        print(f"\n--- Transcription ({domain.upper()}) ---")
        print(transcription[:100] + "...")

        detected = router.detect_domain(transcription, top_k=3)

        print("\nDetected domains:")
        for score in detected:
            print(f"  {score.domain:20} {score.confidence:.2%}")

        print()


async def demo_template_population():
    """Demo 2: Template auto-population"""
    print("\n" + "=" * 70)
    print("DEMO 2: Template Auto-Population")
    print("=" * 70)

    scratchpad = LiveScratchpad()

    # Demo: Recipe
    print("\n--- RECIPE DEMO ---")
    await scratchpad.start_recording('recipe')
    await scratchpad.update_from_transcription(DEMO_TRANSCRIPTIONS['recipe'])

    state = scratchpad.get_current_state()
    print(f"Domain: {state['domain']}")
    print(f"Valid: {state['is_valid']}")
    print("\nExtracted Data:")
    for key, value in state['data'].items():
        if value:
            print(f"  {key}: {value}")

    scratchpad.clear()

    # Demo: Bee Inspection
    print("\n--- BEE INSPECTION DEMO ---")
    await scratchpad.start_recording('bee_inspection')
    await scratchpad.update_from_transcription(DEMO_TRANSCRIPTIONS['bee_inspection'])

    state = scratchpad.get_current_state()
    print(f"Domain: {state['domain']}")
    print(f"Valid: {state['is_valid']}")
    print("\nExtracted Data:")
    for key, value in state['data'].items():
        if value:
            print(f"  {key}: {value}")

    scratchpad.clear()

    # Demo: Budget
    print("\n--- BUDGET DEMO ---")
    await scratchpad.start_recording('budget')
    await scratchpad.update_from_transcription(DEMO_TRANSCRIPTIONS['budget'])

    state = scratchpad.get_current_state()
    print(f"Domain: {state['domain']}")
    print(f"Valid: {state['is_valid']}")
    print("\nExtracted Data:")
    for key, value in state['data'].items():
        if value:
            print(f"  {key}: {value}")


async def demo_manual_editing():
    """Demo 3: Manual field editing"""
    print("\n" + "=" * 70)
    print("DEMO 3: Manual Field Editing")
    print("=" * 70)

    scratchpad = LiveScratchpad()

    # Start with recipe
    await scratchpad.start_recording('recipe')
    await scratchpad.update_from_transcription(DEMO_TRANSCRIPTIONS['recipe'])

    print("\nBefore manual edit:")
    state = scratchpad.get_current_state()
    print(f"  servings: {state['data'].get('servings')}")

    # Manually update a field
    scratchpad.update_field('servings', 48)
    scratchpad.update_field('notes', 'Double batch for party')

    print("\nAfter manual edit:")
    state = scratchpad.get_current_state()
    print(f"  servings: {state['data'].get('servings')}")
    print(f"  notes: {state['data'].get('notes')}")


async def demo_finalization():
    """Demo 4: Finalization and memory shard creation"""
    print("\n" + "=" * 70)
    print("DEMO 4: Finalization")
    print("=" * 70)

    scratchpad = LiveScratchpad()

    # Create a complete entry
    await scratchpad.start_recording('bee_inspection')
    await scratchpad.update_from_transcription(DEMO_TRANSCRIPTIONS['bee_inspection'])

    # Ensure required fields are filled
    scratchpad.update_field('hive_id', 'Hive 3')

    # Finalize
    print("\nFinalizing bee inspection entry...")
    try:
        shard = await scratchpad.finalize()

        print(f"[OK] Created MemoryShard: {shard.id}")
        print(f"  Episode: {shard.episode}")
        print(f"  Entities: {shard.entities}")
        print(f"  Motifs: {shard.motifs}")
        print(f"  Metadata keys: {list(shard.metadata.keys())}")
        print(f"  Structured data: {shard.metadata['structured_data']}")

    except ValueError as e:
        print(f"[FAIL] Validation failed: {e}")


async def demo_all_domains():
    """Demo 5: All domains"""
    print("\n" + "=" * 70)
    print("DEMO 5: All Domains")
    print("=" * 70)

    scratchpad = LiveScratchpad()

    for domain, transcription in DEMO_TRANSCRIPTIONS.items():
        print(f"\n--- {domain.upper().replace('_', ' ')} ---")

        await scratchpad.start_recording(domain)
        await scratchpad.update_from_transcription(transcription)

        state = scratchpad.get_current_state()

        # Count populated fields
        populated = sum(1 for v in state['data'].values() if v)
        total = len(state['data'])

        print(f"Fields populated: {populated}/{total}")
        print(f"Valid: {state['is_valid']}")

        if state['validation_errors']:
            print(f"Errors: {state['validation_errors']}")

        scratchpad.clear()


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("LIVE SCRATCHPAD DEMO")
    print("=" * 70)
    print("\nDemonstrating domain-specific template auto-population")
    print("from natural language transcriptions.\n")

    await demo_domain_detection()
    await demo_template_population()
    await demo_manual_editing()
    await demo_finalization()
    await demo_all_domains()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nTo use the live scratchpad web UI:")
    print("1. Start the API server:")
    print("   python HoloLoom/server/scratchpad_api.py")
    print("\n2. Open the web UI:")
    print("   HoloLoom/web_dashboard/live_scratchpad.html")
    print("\n3. Select a domain, record audio, and watch it auto-populate!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
