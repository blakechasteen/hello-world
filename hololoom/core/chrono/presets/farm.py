"""
Farm Calendar Preset — demonstrates the agent calendar with agricultural tasks.

Based on the Elle farm & kitchen cooperative calendar. Shows how a domain-specific
calendar maps onto the generic AgentCalendar using tags and readable recurrence.

Usage:
    from hololoom.core.chrono.calendar import AgentCalendar
    from hololoom.core.chrono.presets.farm import load_farm_calendar

    cal = AgentCalendar(agent_id="farm-agent")
    load_farm_calendar(cal)

    # Now cal has ~30 farm tasks with proper recurrence patterns.
    # The agent treats them like any other calendar tasks.

Created: 2026-04-02
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from ..calendar import AgentCalendar, CalendarTask, Priority


def load_farm_calendar(
    cal: Optional[AgentCalendar] = None,
    agent_id: str = "farm-agent",
) -> AgentCalendar:
    """
    Populate a calendar with farm & kitchen cooperative tasks.

    If no calendar is provided, creates a new one.
    Returns the populated calendar.
    """
    if cal is None:
        cal = AgentCalendar(agent_id=agent_id)

    # ── Seasonal production windows (as recurring check-in tasks) ────

    cal.add(CalendarTask(
        id="spring-planning",
        name="Spring season planning review",
        recurrence="yearly:mar:1",
        tags=["season", "planning"],
        priority=Priority.HIGH,
        description=(
            "Review seed inventory, soil amendments, and planting schedule. "
            "Focus: seedlings, greens, herbs. Products: microgreens, salad mix, "
            "herb bundles, seedling starts."
        ),
    ))

    cal.add(CalendarTask(
        id="summer-planning",
        name="Summer season planning review",
        recurrence="yearly:jun:1",
        tags=["season", "planning"],
        priority=Priority.HIGH,
        description=(
            "Peak production review. Focus: full production, preservation. "
            "Products: tomatoes, peppers, squash, ferments, dried herbs, sauces."
        ),
    ))

    cal.add(CalendarTask(
        id="fall-planning",
        name="Fall season planning review",
        recurrence="yearly:sep:1",
        tags=["season", "planning"],
        priority=Priority.HIGH,
        description=(
            "Transition review. Focus: harvest, storage crops, preservation. "
            "Products: root vegetables, winter squash, canned goods, stocks."
        ),
    ))

    cal.add(CalendarTask(
        id="winter-planning",
        name="Winter season planning review",
        recurrence="yearly:dec:1",
        tags=["season", "planning"],
        priority=Priority.MEDIUM,
        description=(
            "Off-season review. Focus: planning, maintenance, indoor growing. "
            "Products: microgreens, sprouts, preserved goods."
        ),
    ))

    # ── Planting tasks ───────────────────────────────────────────────

    cal.add(CalendarTask(
        id="start-seedlings",
        name="Start indoor seedlings",
        recurrence="yearly:feb:15",
        tags=["planting", "garden"],
        priority=Priority.HIGH,
        description="Tomatoes, peppers, eggplant, herbs under grow lights.",
        metadata={"related_sops": ["SOP-SEED-001"]},
    ))

    cal.add(CalendarTask(
        id="transplant-seedlings",
        name="Transplant seedlings to beds",
        recurrence="yearly:apr:15",
        tags=["planting", "garden"],
        priority=Priority.HIGH,
        description="After last frost. Harden off 1 week before.",
        metadata={"related_sops": ["SOP-TRANS-001"]},
    ))

    cal.add(CalendarTask(
        id="succession-planting",
        name="Succession planting round",
        recurrence="every:3:weeks",
        tags=["planting", "garden"],
        priority=Priority.MEDIUM,
        description="Lettuce, radish, beans for continuous harvest.",
        metadata={"active_months": [4, 5, 6, 7, 8]},
    ))

    cal.add(CalendarTask(
        id="fall-cover-crop",
        name="Plant fall cover crops",
        recurrence="yearly:sep:15",
        tags=["planting", "soil"],
        priority=Priority.MEDIUM,
        description="Winter rye, crimson clover in emptied beds.",
    ))

    # ── Harvest tasks ────────────────────────────────────────────────

    cal.add(CalendarTask(
        id="daily-harvest",
        name="Morning harvest check",
        recurrence="every:day:7am",
        tags=["harvest", "garden"],
        priority=Priority.MEDIUM,
        description="Check beds for ripe produce. Log quantities.",
        metadata={"active_months": [5, 6, 7, 8, 9, 10]},
    ))

    cal.add(CalendarTask(
        id="microgreen-harvest",
        name="Harvest microgreens",
        recurrence="every:3:days",
        tags=["harvest", "indoor"],
        priority=Priority.HIGH,
        description="Cut at soil level, wash, pack for market.",
        metadata={"related_products": ["microgreens"]},
    ))

    # ── Market tasks ─────────────────────────────────────────────────

    cal.add(CalendarTask(
        id="saturday-market-prep",
        name="Saturday market prep",
        recurrence="every:friday:2pm",
        tags=["market", "sales"],
        priority=Priority.HIGH,
        description="Wash, pack, label, load. Check inventory against pre-orders.",
    ))

    cal.add(CalendarTask(
        id="saturday-market",
        name="Saturday farmers market",
        recurrence="every:saturday:6am",
        tags=["market", "sales"],
        priority=Priority.CRITICAL,
        description="Setup by 7am, market 8am-1pm. Bring signage, scale, bags.",
        metadata={"active_months": [4, 5, 6, 7, 8, 9, 10, 11]},
    ))

    cal.add(CalendarTask(
        id="csa-box-assembly",
        name="CSA box assembly",
        recurrence="every:wednesday:8am",
        tags=["market", "csa"],
        priority=Priority.HIGH,
        description="Assemble weekly CSA boxes. Include newsletter.",
        metadata={"active_months": [5, 6, 7, 8, 9, 10]},
    ))

    # ── Maintenance tasks ────────────────────────────────────────────

    cal.add(CalendarTask(
        id="soil-ph-check",
        name="Check soil pH in raised beds",
        recurrence="every:monday:9am",
        tags=["maintenance", "soil"],
        priority=Priority.MEDIUM,
        description="Test pH in each zone. Target 6.2-6.8. Log results.",
    ))

    cal.add(CalendarTask(
        id="irrigation-check",
        name="Irrigation system check",
        recurrence="every:2:weeks:monday",
        tags=["maintenance", "equipment"],
        priority=Priority.MEDIUM,
        description="Check drip lines, timers, pressure. Fix leaks.",
    ))

    cal.add(CalendarTask(
        id="tool-maintenance",
        name="Tool cleaning and sharpening",
        recurrence="every:month:first:saturday",
        tags=["maintenance", "equipment"],
        priority=Priority.LOW,
        description="Clean, oil, sharpen. Inventory check.",
    ))

    cal.add(CalendarTask(
        id="greenhouse-maintenance",
        name="Greenhouse maintenance",
        recurrence="every:month:15",
        tags=["maintenance", "infrastructure"],
        priority=Priority.MEDIUM,
        description="Check ventilation, heaters, plastic, benches.",
    ))

    # ── Production / kitchen tasks ───────────────────────────────────

    cal.add(CalendarTask(
        id="fermentation-check",
        name="Check active ferments",
        recurrence="every:day:10am",
        tags=["production", "kitchen"],
        priority=Priority.HIGH,
        description="Check kraut, kimchi, hot sauce ferments. Log pH and taste.",
        metadata={"related_sops": ["SOP-FERM-001"]},
    ))

    cal.add(CalendarTask(
        id="sauce-production",
        name="Hot sauce production day",
        recurrence="every:2:weeks:thursday",
        tags=["production", "kitchen"],
        priority=Priority.MEDIUM,
        description="Batch production. Label, date, inventory.",
        metadata={"related_products": ["hot_sauce", "bbq_sauce"]},
    ))

    # ── Financial / regulatory ───────────────────────────────────────

    cal.add(CalendarTask(
        id="quarterly-taxes",
        name="Quarterly tax filing deadline",
        recurrence="yearly:apr:15",
        tags=["financial", "deadline"],
        priority=Priority.CRITICAL,
        description="Federal estimated taxes due. State may differ.",
        reminder_before=timedelta(days=14),
    ))

    cal.add(CalendarTask(
        id="q2-taxes",
        name="Q2 estimated tax payment",
        recurrence="yearly:jun:15",
        tags=["financial", "deadline"],
        priority=Priority.CRITICAL,
        reminder_before=timedelta(days=14),
    ))

    cal.add(CalendarTask(
        id="q3-taxes",
        name="Q3 estimated tax payment",
        recurrence="yearly:sep:15",
        tags=["financial", "deadline"],
        priority=Priority.CRITICAL,
        reminder_before=timedelta(days=14),
    ))

    cal.add(CalendarTask(
        id="q4-taxes",
        name="Q4 estimated tax payment",
        recurrence="yearly:jan:15",
        tags=["financial", "deadline"],
        priority=Priority.CRITICAL,
        reminder_before=timedelta(days=14),
    ))

    cal.add(CalendarTask(
        id="health-inspection",
        name="Annual health department inspection prep",
        recurrence="yearly:mar:1",
        tags=["regulatory", "compliance"],
        priority=Priority.CRITICAL,
        description="Review logs, clean deep, organize records.",
        reminder_before=timedelta(days=30),
    ))

    cal.add(CalendarTask(
        id="insurance-renewal",
        name="Farm insurance renewal",
        recurrence="yearly:jan:1",
        tags=["financial", "insurance"],
        priority=Priority.HIGH,
        description="Review coverage, compare quotes, renew.",
        reminder_before=timedelta(days=30),
    ))

    cal.add(CalendarTask(
        id="monthly-bookkeeping",
        name="Monthly bookkeeping reconciliation",
        recurrence="every:month:last",
        tags=["financial", "admin"],
        priority=Priority.MEDIUM,
        description="Reconcile accounts, categorize expenses, update P&L.",
    ))

    # ── General / planning ───────────────────────────────────────────

    cal.add(CalendarTask(
        id="weekly-team-meeting",
        name="Weekly team planning meeting",
        recurrence="every:monday:8am",
        tags=["planning", "team"],
        priority=Priority.MEDIUM,
        description="Review week ahead, assign tasks, address issues.",
    ))

    cal.add(CalendarTask(
        id="photo-documentation",
        name="Farm photo documentation",
        recurrence="every:sunday:4pm",
        tags=["documentation", "marketing"],
        priority=Priority.LOW,
        description="Walk property, photograph progress for social media and records.",
    ))

    return cal
