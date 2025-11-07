"""
HoloLoom COS - Daily Review Workflow

Automated prompts for morning planning and evening review.

Philosophy: Reflection leads to improvement
- Morning: Set intentions
- Evening: Learn from reality
- Compare: Plan vs actual
- Improve: Apply lessons tomorrow
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cos.core.storage import EventStore
from cos.core.types import EventType, ProductLine
from cos.intelligence.accounting import AccountingGenerator
from cos.interface.voice_chat import VoiceChat


class DailyReviewWorkflow:
    """
    Guided daily review workflow.

    Morning (7am):
    - What are today's goals?
    - How many hours available?
    - What products to focus on?

    Evening (7pm):
    - What got done?
    - What didn't?
    - Why?
    - What learned?
    - Tomorrow's priorities?
    """

    def __init__(
        self,
        store: EventStore,
        voice: Optional[VoiceChat] = None
    ):
        self.store = store
        self.voice = voice
        self.accounting = AccountingGenerator(store)

    async def morning_planning(self, voice_mode: bool = False) -> Dict:
        """
        Morning planning session.

        Returns:
            {
                'goals': List[str],
                'hours_available': float,
                'priority_products': List[str],
                'revenue_target': float
            }
        """
        print("\n" + "="*60)
        print("☀️  MORNING PLANNING")
        print("="*60)
        print(f"\n{datetime.now().strftime('%A, %B %d, %Y')}\n")

        # Get yesterday's summary
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_summary = await self.store.get_daily_summary(yesterday)

        if yesterday_summary.revenue > 0:
            print(f"Yesterday: ${yesterday_summary.revenue:.2f} revenue, "
                  f"{yesterday_summary.hours_worked:.1f} hours, "
                  f"${yesterday_summary.hourly_rate():.2f}/hr\n")

        # Get this week's progress
        week = datetime.now().strftime('%Y-W%W')
        week_summary = await self.store.get_weekly_summary(week)
        if week_summary.revenue > 0:
            print(f"This week so far: ${week_summary.revenue:.2f} revenue, "
                  f"{week_summary.hours_worked:.1f} hours\n")

        # Planning prompts
        if voice_mode and self.voice:
            goals = await self._voice_planning()
        else:
            goals = await self._text_planning()

        # Store as plan event
        from cos.core.types import Event, EventSource
        plan_text = f"Daily plan: {', '.join(goals['goals'])}"
        plan_event = Event(
            type=EventType.PLAN,
            raw_input=plan_text,
            source=EventSource.MANUAL,
            parsed_data=goals
        )
        await self.store.store(plan_event)

        print("\n✓ Plan saved!")
        print("="*60)

        return goals

    async def _text_planning(self) -> Dict:
        """Text-based planning prompts"""
        goals = []

        print("What are your top 3 goals for today?")
        for i in range(3):
            goal = input(f"  {i+1}. ").strip()
            if goal:
                goals.append(goal)

        hours = input("\nHow many hours do you have available today? ").strip()
        hours = float(hours) if hours else 8.0

        priority = input("\nWhat product(s) will you focus on? ").strip()
        priority_products = [p.strip() for p in priority.split(',') if p.strip()]

        target = input("\nRevenue target for today? $").strip()
        revenue_target = float(target) if target else 50.0

        return {
            'goals': goals,
            'hours_available': hours,
            'priority_products': priority_products,
            'revenue_target': revenue_target
        }

    async def _voice_planning(self) -> Dict:
        """Voice-based planning"""
        # TODO: Implement voice-based planning
        # Use Whisper to capture goals, hours, etc.
        return await self._text_planning()

    async def evening_review(self, voice_mode: bool = False) -> Dict:
        """
        Evening review session.

        Returns:
            {
                'completed': List[str],
                'not_completed': List[str],
                'learnings': List[str],
                'tomorrow_priorities': List[str],
                'energy_level': int  # 1-10
            }
        """
        print("\n" + "="*60)
        print("🌙 EVENING REVIEW")
        print("="*60)
        print(f"\n{datetime.now().strftime('%A, %B %d, %Y')}\n")

        # Get today's actual performance
        today = datetime.now()
        today_summary = await self.store.get_daily_summary(today)

        print("TODAY'S RESULTS:")
        print(f"  Revenue:  ${today_summary.revenue:.2f}")
        print(f"  Hours:    {today_summary.hours_worked:.1f}")
        print(f"  $/hour:   ${today_summary.hourly_rate():.2f}")
        print(f"  Profit:   ${today_summary.profit:.2f}")
        print(f"  Margin:   {today_summary.profit_margin():.1f}%\n")

        # Get today's plan (if any)
        today_start = today.replace(hour=0, minute=0, second=0)
        today_end = today.replace(hour=23, minute=59, second=59)
        plans = await self.store.query(
            event_type=EventType.PLAN,
            start_date=today_start,
            end_date=today_end
        )

        if plans:
            print("PLANNED GOALS:")
            plan_data = plans[0].parsed_data
            for i, goal in enumerate(plan_data.get('goals', []), 1):
                print(f"  {i}. {goal}")
            print()

        # Review prompts
        if voice_mode and self.voice:
            review = await self._voice_review()
        else:
            review = await self._text_review()

        # Store as review event
        from cos.core.types import Event, EventSource
        review_text = f"Daily review: {len(review['completed'])} completed, {len(review['learnings'])} learnings"
        review_event = Event(
            type=EventType.REVIEW,
            raw_input=review_text,
            source=EventSource.MANUAL,
            parsed_data=review
        )
        await self.store.store(review_event)

        # Generate insights
        await self._generate_insights(today_summary, plans[0] if plans else None, review)

        print("\n✓ Review saved!")
        print("="*60)

        return review

    async def _text_review(self) -> Dict:
        """Text-based review prompts"""
        print("REFLECTION:")

        completed = []
        print("\nWhat got done today?")
        for i in range(5):
            item = input(f"  {i+1}. ").strip()
            if not item:
                break
            completed.append(item)

        not_completed = []
        print("\nWhat didn't get done?")
        for i in range(3):
            item = input(f"  {i+1}. ").strip()
            if not item:
                break
            not_completed.append(item)

        learnings = []
        print("\nWhat did you learn today?")
        for i in range(3):
            learning = input(f"  {i+1}. ").strip()
            if not learning:
                break
            learnings.append(learning)

        tomorrow = []
        print("\nTop 3 priorities for tomorrow?")
        for i in range(3):
            priority = input(f"  {i+1}. ").strip()
            if not priority:
                break
            tomorrow.append(priority)

        energy = input("\nEnergy level (1-10)? ").strip()
        energy_level = int(energy) if energy else 7

        return {
            'completed': completed,
            'not_completed': not_completed,
            'learnings': learnings,
            'tomorrow_priorities': tomorrow,
            'energy_level': energy_level
        }

    async def _voice_review(self) -> Dict:
        """Voice-based review"""
        # TODO: Implement voice-based review
        return await self._text_review()

    async def _generate_insights(self, summary, plan, review):
        """Generate insights from plan vs actual"""
        print("\n" + "-"*60)
        print("INSIGHTS:")

        # Energy check
        energy = review['energy_level']
        if energy < 5:
            print(f"\n⚠️  Low energy ({energy}/10) - consider lightening workload tomorrow")
        elif energy > 8:
            print(f"\n✓ High energy ({energy}/10) - good sign of sustainable pace!")

        # Hours check
        if summary.hours_worked > 10:
            print(f"\n⚠️  Long day ({summary.hours_worked:.1f} hours) - watch for burnout")
        elif summary.hours_worked < 4:
            print(f"\nℹ️  Light day ({summary.hours_worked:.1f} hours) - opportunity to scale?")

        # Profitability check
        if summary.hourly_rate() < 15:
            print(f"\n⚠️  Low hourly rate (${summary.hourly_rate():.2f}/hr) - review pricing or efficiency")
        elif summary.hourly_rate() > 25:
            print(f"\n✓ Strong hourly rate (${summary.hourly_rate():.2f}/hr) - great work!")

        # Plan vs actual (if plan exists)
        if plan:
            plan_data = plan.parsed_data
            if 'hours_available' in plan_data:
                planned_hours = plan_data['hours_available']
                actual_hours = summary.hours_worked
                if actual_hours > planned_hours * 1.2:
                    print(f"\nℹ️  Worked {actual_hours:.1f}h vs planned {planned_hours:.1f}h - over by {actual_hours - planned_hours:.1f}h")

            if 'revenue_target' in plan_data:
                target = plan_data['revenue_target']
                actual = summary.revenue
                if actual >= target:
                    pct = (actual / target - 1) * 100
                    print(f"\n✓ Revenue target exceeded! ${actual:.2f} vs ${target:.2f} (+{pct:.0f}%)")
                else:
                    pct = (1 - actual / target) * 100
                    print(f"\nℹ️  Revenue below target: ${actual:.2f} vs ${target:.2f} (-{pct:.0f}%)")


async def demo():
    """Demo daily review workflow"""
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    store = EventStore("test_cos.db")
    workflow = DailyReviewWorkflow(store)

    print("\n" + "="*60)
    print("HoloLoom COS - Daily Review Demo")
    print("="*60)

    # Morning planning
    print("\n\nDEMO: Morning Planning")
    print("(In production, this would run at 7am automatically)")
    # goals = await workflow.morning_planning()

    # Evening review
    print("\n\nDEMO: Evening Review")
    print("(In production, this would run at 7pm automatically)")
    review = await workflow.evening_review()

    print("\n\n" + "="*60)
    print("✅ Daily Review Demo Complete")
    print("="*60)
    print("\nNext steps:")
    print("  - Set up cron job or Windows Task Scheduler")
    print("  - Morning: 7:00 AM")
    print("  - Evening: 7:00 PM")


if __name__ == '__main__':
    asyncio.run(demo())
