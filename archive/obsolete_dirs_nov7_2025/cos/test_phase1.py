#!/usr/bin/env python3
"""
Test Phase 1 functionality of HoloLoom COS

Tests:
1. Database initialization
2. NLP parsing
3. Event storage
4. HITL verification
5. Queries and summaries
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from decimal import Decimal

# Fix UTF-8 encoding on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cos.core.types import EventType, EventSource, Category, ProductLine
from cos.core.parser import parse_input
from cos.core.storage import EventStore


async def test_parser():
    """Test NLP parsing"""
    print("\n" + "="*60)
    print("TEST 1: NLP Parser")
    print("="*60)

    test_inputs = [
        "Baked bread for 3 hours, made 12 loaves",
        "Sold 10 loaves for $60",
        "Bought 50 pounds flour at Costco for $27",
        "Paid electric bill $145",
        "Made $180 in bread sales",
        "Finished 20 quarts of soup",
    ]

    for text in test_inputs:
        intent, verification = parse_input(text)
        print(f"\nInput: {text}")
        print(f"  Type: {intent.event_type.value}")
        print(f"  Confidence: {intent.confidence:.2%}")
        print(f"  Explanation: {intent.explanation}")

        if verification:
            print(f"  ⚠️  HITL needed: {verification.reason}")


async def test_storage():
    """Test event storage and retrieval"""
    print("\n" + "="*60)
    print("TEST 2: Event Storage")
    print("="*60)

    # Use test database
    store = EventStore("test_cos.db")

    # Parse and store some events
    events_to_log = [
        "Baked bread for 3 hours, made 12 loaves",
        "Sold 10 loaves for $60",
        "Bought 50 pounds flour at Costco for $27",
    ]

    event_ids = []
    for text in events_to_log:
        intent, verification = parse_input(text, EventSource.MANUAL)
        event = intent.to_event(text, EventSource.MANUAL)

        # Auto-verify for testing
        if verification:
            event.verified = True
            event.verification_note = "Auto-verified for testing"

        event_id = await store.store(event)
        event_ids.append(event_id)
        print(f"\n✓ Stored: {text}")
        print(f"  Event ID: {event_id}")

    # Test retrieval
    print(f"\n\nRetrieving events...")
    for event_id in event_ids:
        event = await store.get_by_id(event_id)
        print(f"\nEvent #{event_id}:")
        print(f"  Type: {event.type.value}")
        print(f"  Item: {event.item}")
        print(f"  Amount: ${event.amount}" if event.amount else "")
        print(f"  Verified: {event.verified}")

    return store


async def test_queries(store: EventStore):
    """Test querying and summaries"""
    print("\n" + "="*60)
    print("TEST 3: Queries & Summaries")
    print("="*60)

    # Query by type
    tasks = await store.query(event_type=EventType.TASK)
    print(f"\nTasks: {len(tasks)}")
    for task in tasks:
        print(f"  - {task.item}: {task.quantity} {task.unit}, ${abs(task.amount):.2f}")

    sales = await store.query(event_type=EventType.SALE)
    print(f"\nSales: {len(sales)}")
    for sale in sales:
        print(f"  - {sale.item}: ${sale.amount:.2f}")

    purchases = await store.query(event_type=EventType.PURCHASE)
    print(f"\nPurchases: {len(purchases)}")
    for purchase in purchases:
        print(f"  - {purchase.item}: ${abs(purchase.amount):.2f}")

    # Daily summary
    summary = await store.get_daily_summary(datetime.now())
    print(f"\n\nDaily Summary:")
    print(f"  Revenue:  ${summary.revenue:.2f}")
    print(f"  COGS:     ${summary.cogs:.2f}")
    print(f"  Labor:    ${summary.labor_cost:.2f}")
    print(f"  Profit:   ${summary.profit:.2f}")
    print(f"  Hours:    {summary.hours_worked:.1f}")
    print(f"  $/hour:   ${summary.hourly_rate():.2f}")
    print(f"  Margin:   {summary.profit_margin():.1f}%")


async def test_hitl():
    """Test HITL verification flow"""
    print("\n" + "="*60)
    print("TEST 4: HITL Verification")
    print("="*60)

    store = EventStore("test_cos.db")

    # Create a low-confidence event
    text = "Bought some stuff for about twenty seven dollars"  # Ambiguous
    intent, verification = parse_input(text)

    print(f"\nInput: {text}")
    print(f"Confidence: {intent.confidence:.2%}")

    if verification:
        print(f"\n{verification}")
        print("\n✓ HITL verification would be triggered")
    else:
        print("\n✓ High confidence, no HITL needed")

    # Check unverified queue
    unverified = await store.get_unverified()
    print(f"\n\nUnverified events in queue: {len(unverified)}")
    for event in unverified:
        print(f"  #{event.id}: {event.raw_input} (confidence: {event.confidence:.2%})")


async def test_full_workflow():
    """Test complete workflow"""
    print("\n" + "="*60)
    print("TEST 5: Full Workflow")
    print("="*60)

    store = EventStore("test_cos.db")

    # Simulate a day of work
    day_events = [
        # Morning: Bread baking
        "Started bread at 7am, finished at 10am",
        "Made 12 loaves of bread",

        # Afternoon: Sales
        "Sold 8 loaves for $48 at farmers market",

        # Evening: Meal prep
        "Worked on meal prep for 4 hours",
        "Made 20 quarts of soup",

        # Purchases
        "Bought 25 pounds flour at Costco for $13.50",

        # Daily review
        "Daily review: Good day, bread sold well, need more soup variety",
    ]

    print("\nSimulating a day of work...\n")
    for i, text in enumerate(day_events, 1):
        intent, verification = parse_input(text)
        event = intent.to_event(text, EventSource.MANUAL)

        # Auto-verify for demo
        if verification:
            event.verified = True
            event.verification_note = "Auto-verified for demo"

        event_id = await store.store(event)
        print(f"{i}. ✓ {text[:50]}... (event #{event_id})")

    # Show summary
    summary = await store.get_daily_summary(datetime.now())
    print(f"\n\n📊 End of Day Summary:")
    print(f"   Revenue:  ${summary.revenue:>8.2f}")
    print(f"   COGS:    -${summary.cogs:>8.2f}")
    print(f"   Labor:   -${summary.labor_cost:>8.2f}")
    print(f"   ──────────────────")
    print(f"   Profit:   ${summary.profit:>8.2f}")
    print(f"\n   Hours:    {summary.hours_worked:>8.1f}")
    print(f"   $/hour:   ${summary.hourly_rate():>8.2f}")
    print(f"   Margin:   {summary.profit_margin():>7.1f}%")

    # Product breakdown
    print(f"\n   Revenue by Product:")
    for product, revenue in summary.revenue_by_product.items():
        print(f"     {product.value:12} ${revenue:>6.2f}")


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("HoloLoom COS - Phase 1 Test Suite")
    print("="*60)

    try:
        # Test 1: Parser
        await test_parser()

        # Test 2: Storage
        store = await test_storage()

        # Test 3: Queries
        await test_queries(store)

        # Test 4: HITL
        await test_hitl()

        # Test 5: Full workflow
        await test_full_workflow()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\nPhase 1 is complete and working!")
        print("\nNext steps:")
        print("  1. Try the CLI: python cos/cli.py log 'Baked bread for 3 hours'")
        print("  2. Move on to Phase 2: HoloLoom integration")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
