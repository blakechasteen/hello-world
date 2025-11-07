#!/usr/bin/env python3
"""
HoloLoom COS - Command Line Interface

Usage:
    cos log "Baked bread for 3 hours, made 12 loaves"
    cos ask "What's my profit today?"
    cos verify
    cos summary today
    cos summary week
"""

import sys
import asyncio
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cos.core.types import Event, EventType, EventSource, VerificationRequest
from cos.core.parser import parse_input
from cos.core.storage import get_store


class COSCLI:
    """Command-line interface for COS"""

    def __init__(self):
        self.store = get_store()

    async def log(self, text: str, source: EventSource = EventSource.MANUAL):
        """
        Log an event from natural language input.

        Handles HITL verification automatically.
        """
        print(f"\n📝 Processing: {text}")

        # Parse input
        intent, verification = parse_input(text, source)

        if verification:
            # HITL needed
            print(f"\n{verification}")
            print(f"\nParsed as: {intent.explanation}")

            # Ask for confirmation
            response = input("\n✓ Is this correct? [y/n/edit]: ").strip().lower()

            if response == 'n':
                print("❌ Event discarded")
                return None

            elif response == 'edit':
                # TODO: Interactive editing
                print("📝 Interactive editing not yet implemented")
                print("   For now, please re-enter with corrections")
                return None

            # If 'y' or just enter, continue with verification
            event = intent.to_event(text, source)
            event.verified = True
            event.verification_note = "Manually verified via CLI"

        else:
            # High confidence, auto-process
            event = intent.to_event(text, source)
            print(f"✓ {intent.explanation}")

        # Store event
        event_id = await self.store.store(event)
        event.id = event_id

        print(f"✓ Logged as event #{event_id}")

        # Show impact
        await self._show_impact(event)

        return event

    async def _show_impact(self, event: Event):
        """Show the impact of this event on business metrics"""
        if event.type == EventType.TASK:
            print(f"   Labor cost: ${abs(event.amount):.2f}")
            if event.parsed_data.get('output'):
                print(f"   Output: {event.parsed_data['output']} units")

        elif event.type == EventType.SALE:
            print(f"   Revenue: ${event.amount:.2f}")
            today_summary = await self.store.get_daily_summary(datetime.now())
            print(f"   Today's revenue: ${today_summary.revenue:.2f}")

        elif event.type == EventType.PURCHASE:
            print(f"   Expense: ${abs(event.amount):.2f}")
            if event.quantity and event.unit:
                unit_cost = abs(event.amount) / event.quantity
                print(f"   Unit cost: ${unit_cost:.2f}/{event.unit}")

    async def verify_queue(self):
        """Show and verify pending events (HITL queue)"""
        unverified = await self.store.get_unverified()

        if not unverified:
            print("✓ No events pending verification")
            return

        print(f"\n📋 {len(unverified)} events need verification:\n")

        for event in unverified:
            print("─" * 60)
            print(f"Event #{event.id} - {event.timestamp.strftime('%Y-%m-%d %H:%M')}")
            print(f"Type: {event.type.value}")
            print(f"Input: {event.raw_input}")
            print(f"Parsed: {event}")
            if event.confidence:
                print(f"Confidence: {event.confidence:.2%}")

            response = input("\n✓ Verify this event? [y/n/edit/skip]: ").strip().lower()

            if response == 'y':
                await self.store.verify(event.id, verified=True, note="Verified via CLI")
                print("✓ Verified")

            elif response == 'n':
                await self.store.verify(event.id, verified=False, note="Rejected via CLI")
                print("❌ Rejected")

            elif response == 'edit':
                print("📝 What should it be?")
                corrected = input("   New input: ").strip()
                if corrected:
                    # Re-parse
                    intent, _ = parse_input(corrected, event.source)
                    # Update event
                    await self.store.update(event.id, {
                        'raw_input': corrected,
                        'parsed_data': intent.parsed_data,
                        'amount': float(intent.amount) if intent.amount else None,
                        'quantity': float(intent.quantity) if intent.quantity else None,
                        'item': intent.item,
                        'verified': True,
                        'verification_note': 'Corrected via CLI'
                    })
                    print("✓ Updated and verified")

            else:  # skip
                print("⏭ Skipped")

        print("\n✓ Verification complete")

    async def summary_today(self):
        """Show today's summary"""
        today = datetime.now()
        summary = await self.store.get_daily_summary(today)

        print(f"\n📊 Daily Summary - {today.strftime('%A, %B %d, %Y')}")
        print("─" * 60)
        print(f"Revenue:      ${summary.revenue:>10.2f}")
        print(f"COGS:        -${summary.cogs:>10.2f}")
        print(f"Labor:       -${summary.labor_cost:>10.2f}")
        print(f"             ──────────────")
        print(f"Profit:       ${summary.profit:>10.2f}")
        print(f"\nHours:        {summary.hours_worked:>10.1f}")
        print(f"Hourly Rate:  ${summary.hourly_rate():>10.2f}/hr")
        print(f"Margin:       {summary.profit_margin():>10.1f}%")

        if summary.revenue_by_product:
            print(f"\nRevenue by Product:")
            for product, revenue in summary.revenue_by_product.items():
                print(f"  {product.value:15} ${revenue:>8.2f}")

    async def summary_week(self):
        """Show this week's summary"""
        # Get current ISO week
        today = datetime.now()
        week = today.strftime('%Y-W%W')

        summary = await self.store.get_weekly_summary(week)

        print(f"\n📊 Weekly Summary - Week {week}")
        print("─" * 60)
        print(f"Revenue:      ${summary.revenue:>10.2f}")
        print(f"COGS:        -${summary.cogs:>10.2f}")
        print(f"Labor:       -${summary.labor_cost:>10.2f}")
        print(f"Overhead:    -${summary.overhead:>10.2f}")
        print(f"             ──────────────")
        print(f"Profit:       ${summary.profit:>10.2f}")
        print(f"\nHours:        {summary.hours_worked:>10.1f}")
        print(f"Products:     {summary.active_product_lines:>10}")

        if summary.revenue_target:
            variance = summary.variance()
            variance_pct = summary.variance_pct()
            status = "✓" if variance >= 0 else "⚠"
            print(f"\nTarget:       ${summary.revenue_target:>10.2f}")
            print(f"Variance:     {status} ${variance:>9.2f} ({variance_pct:+.1f}%)")

    async def inventory(self):
        """Show current inventory"""
        inventory = await self.store.get_inventory()

        if not inventory:
            print("📦 No inventory tracked yet")
            return

        print("\n📦 Current Inventory")
        print("─" * 60)
        print(f"{'Item':<20} {'Quantity':>15} {'Avg Cost':>15}")
        print("─" * 60)

        for item, data in sorted(inventory.items()):
            qty_str = f"{data['quantity']:.1f} {data['unit']}"
            cost_str = f"${data['avg_cost']:.2f}/{data['unit']}"
            print(f"{item:<20} {qty_str:>15} {cost_str:>15}")

    async def ask(self, question: str):
        """
        Ask a question about the business.

        This will eventually use HoloLoom reasoning.
        For now, handle common queries.
        """
        question_lower = question.lower()

        # Simple pattern matching for common questions
        if 'profit' in question_lower and 'today' in question_lower:
            await self.summary_today()

        elif 'profit' in question_lower and 'week' in question_lower:
            await self.summary_week()

        elif 'inventory' in question_lower:
            await self.inventory()

        elif 'most profitable' in question_lower:
            await self._most_profitable()

        elif 'hours' in question_lower:
            await self._time_breakdown()

        else:
            print(f"\n❓ Question: {question}")
            print("🔮 HoloLoom reasoning not yet implemented")
            print("   Try: 'profit today', 'profit week', 'inventory', 'most profitable'")

    async def _most_profitable(self):
        """Show most profitable product line"""
        from cos.core.types import ProductLine

        print("\n💰 Product Line Performance")
        print("─" * 80)
        print(f"{'Product':<15} {'Revenue':>12} {'Costs':>12} {'Profit':>12} {'Margin':>10} {'$/hr':>10}")
        print("─" * 80)

        performances = []
        for product in ProductLine:
            perf = await self.store.get_product_performance(product)
            if perf.revenue > 0:
                performances.append(perf)

        # Sort by net profit
        performances.sort(key=lambda p: p.net_profit(), reverse=True)

        for perf in performances:
            total_cost = perf.material_cost + perf.labor_cost
            print(f"{perf.product_line.value:<15} "
                  f"${perf.revenue:>11.2f} "
                  f"${total_cost:>11.2f} "
                  f"${perf.net_profit():>11.2f} "
                  f"{perf.net_margin():>9.1f}% "
                  f"${perf.profit_per_hour():>9.2f}")

    async def _time_breakdown(self):
        """Show time breakdown by task"""
        # Get this week's tasks
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())

        tasks = await self.store.query(
            event_type=EventType.TASK,
            start_date=week_start,
            limit=1000
        )

        if not tasks:
            print("⏱ No time tracked this week")
            return

        # Aggregate by product line
        from collections import defaultdict
        time_by_product = defaultdict(float)
        total_hours = 0

        for task in tasks:
            if task.quantity:
                hours = float(task.quantity)
                total_hours += hours
                if task.product_line:
                    time_by_product[task.product_line.value] += hours

        print(f"\n⏱ Time Breakdown - This Week")
        print("─" * 60)
        print(f"{'Product':<20} {'Hours':>15} {'% of Time':>15}")
        print("─" * 60)

        for product, hours in sorted(time_by_product.items(), key=lambda x: x[1], reverse=True):
            pct = (hours / total_hours * 100) if total_hours > 0 else 0
            print(f"{product:<20} {hours:>14.1f}h {pct:>14.1f}%")

        print("─" * 60)
        print(f"{'TOTAL':<20} {total_hours:>14.1f}h")


async def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print("HoloLoom COS - Company Operating System")
        print("\nUsage:")
        print("  cos log <text>        Log an event")
        print("  cos verify            Verify pending events (HITL)")
        print("  cos summary today     Today's summary")
        print("  cos summary week      This week's summary")
        print("  cos inventory         Show inventory")
        print("  cos ask <question>    Ask a question")
        print("\nExamples:")
        print("  cos log 'Baked bread for 3 hours, made 12 loaves'")
        print("  cos log 'Sold 10 loaves for $60'")
        print("  cos log 'Bought flour at Costco for $27, 50 pounds'")
        print("  cos ask 'What is my most profitable product?'")
        sys.exit(1)

    command = sys.argv[1]
    cli = COSCLI()

    if command == 'log':
        if len(sys.argv) < 3:
            print("Error: Please provide text to log")
            print("Usage: cos log 'Baked bread for 3 hours'")
            sys.exit(1)

        text = ' '.join(sys.argv[2:])
        await cli.log(text)

    elif command == 'verify':
        await cli.verify_queue()

    elif command == 'summary':
        if len(sys.argv) < 3:
            print("Usage: cos summary [today|week]")
            sys.exit(1)

        period = sys.argv[2]
        if period == 'today':
            await cli.summary_today()
        elif period == 'week':
            await cli.summary_week()
        else:
            print(f"Unknown period: {period}")
            print("Usage: cos summary [today|week]")

    elif command == 'inventory':
        await cli.inventory()

    elif command == 'ask':
        if len(sys.argv) < 3:
            print("Error: Please provide a question")
            print("Usage: cos ask 'What is my profit today?'")
            sys.exit(1)

        question = ' '.join(sys.argv[2:])
        await cli.ask(question)

    else:
        print(f"Unknown command: {command}")
        print("Try: cos (no args) for usage")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
