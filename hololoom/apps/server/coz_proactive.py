"""
Coz Proactive — Scheduled insight generation for Elle.

Generates actionable daily briefs by comparing current Coz data
to historical patterns in the memory bus. On Sundays, proposes
a weekly summary note to the vault.

Endpoints:
  POST /coz/proactive/brief   — Generate daily brief with insights
  GET  /coz/proactive/history  — Past briefs from memory bus

Created: 2026-03-13
"""

import json
import logging
import os
from datetime import date, datetime

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/coz/proactive", tags=["coz-proactive"])

OLLAMA_URL = os.environ.get(
    "PROMPTLY_OLLAMA_URL",
    os.environ.get("ELLE_OLLAMA_URL", "http://127.0.0.1:11434"),
)
OLLAMA_MODEL = os.environ.get("ELLE_MODEL", "qwen3.5:9b")


# ============================================================================
# Insight generation
# ============================================================================

class Insight(BaseModel):
    insight: str
    action: str
    urgency: str  # low | medium | high
    domain: str   # inventory | financial | seasonal | tasks


class ProactiveBrief(BaseModel):
    insights: list[Insight]
    status: str  # ok | coz_unavailable | llm_unavailable
    generated_at: str
    vault_proposal: str | None = None


async def generate_proactive_brief() -> ProactiveBrief:
    """Generate actionable daily brief with historical comparison."""
    # 1. Get today's data from Coz
    try:
        from hololoom.apps.server.coz_api import _get_sync_manager
        sm = _get_sync_manager()
        brief = sm.get_daily_brief()
        inventory = sm.get_inventory_status()
    except Exception as e:
        logger.warning("Coz data unavailable: %s", e)
        return ProactiveBrief(
            insights=[], status="coz_unavailable",
            generated_at=datetime.now().isoformat(),
        )

    # 2. Get historical context from memory bus
    history_context = ""
    try:
        from hololoom.apps.server.memory_bus import get_store
        store = get_store()
        past_briefs = store.search_semantic(
            query="daily brief proactive insights",
            agent="coz", limit=3, decay_half_life=14.0,
        )
        if past_briefs:
            history_context = "\n".join(
                f"- {b['content'][:300]}" for b in past_briefs[:3]
            )
    except Exception:
        pass  # Memory bus unavailable — generate without history

    # 3. Generate insights via LLM
    insights = await _generate_insights(brief, inventory, history_context)

    # 4. Store this brief as observation for future comparison
    try:
        from hololoom.apps.server.memory_bus import get_store
        store = get_store()
        summary = "; ".join(f"{i.insight} ({i.urgency})" for i in insights[:5])
        store.write(
            agent="coz",
            content=f"Daily brief {date.today().isoformat()}: {summary}",
            tags=["proactive", "daily-brief", date.today().strftime("%B").lower()],
            relevance_score=0.8,
        )
    except Exception:
        pass

    # 5. Weekly vault proposal (Sundays only)
    vault_path = None
    if datetime.now().weekday() == 6:
        vault_path = _maybe_propose_weekly_note(insights)

    return ProactiveBrief(
        insights=insights,
        status="ok",
        generated_at=datetime.now().isoformat(),
        vault_proposal=vault_path,
    )


async def _generate_insights(
    brief: dict, inventory: dict, history_context: str,
) -> list[Insight]:
    """Single LLM pass to produce 3-5 actionable insights."""
    # Format data for the prompt
    alerts = inventory.get("alerts", [])
    alerts_text = "\n".join(
        f"- {a.get('item', a) if isinstance(a, dict) else a}" for a in alerts[:5]
    ) if alerts else "None"

    tasks = brief.get("priority_tasks", brief.get("due_today", []))
    tasks_text = "\n".join(
        f"- {t.get('title', t) if isinstance(t, dict) else t}" for t in tasks[:5]
    ) if tasks else "None"

    focus = brief.get("focus_area", brief.get("seasonal_focus", ""))
    month = brief.get("current_month", date.today().strftime("%B"))

    prompt = f"""You are a farm/kitchen operations analyst. Based on today's data, produce 3-5 actionable insights.

## Today's Data ({date.today().isoformat()}, {month})

Focus: {focus}

Inventory Alerts:
{alerts_text}

Priority Tasks:
{tasks_text}

## Recent History
{history_context or "No historical data available yet."}

## Instructions
- Compare today to historical patterns if available
- Each insight must have: insight (1 sentence), action (specific next step), urgency (low/medium/high), domain (inventory/financial/seasonal/tasks)
- If nothing is actionable, return a single insight: "All systems nominal" with urgency "low"
- Return ONLY valid JSON array of objects. No markdown, no explanation.

Example format:
[{{"insight": "Sourdough flour below reorder threshold", "action": "Order 25kg bread flour from supplier", "urgency": "high", "domain": "inventory"}}]"""

    try:
        async with httpx.AsyncClient(
            base_url=OLLAMA_URL, timeout=60.0,
        ) as client:
            resp = await client.post("/api/chat", json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0.3},
            })
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")

            # Strip thinking tags (qwen quirk)
            import re
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            # Parse JSON
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return [Insight(**item) for item in parsed[:5]]
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON response for proactive brief")
    except Exception as e:
        logger.warning("LLM unavailable for proactive brief: %s", e)

    # Fallback: data-only summary
    if alerts:
        return [Insight(
            insight=f"{len(alerts)} inventory alert(s) need attention",
            action="Review inventory alerts in /coz/inventory",
            urgency="medium",
            domain="inventory",
        )]
    return [Insight(
        insight="All systems nominal",
        action="No action needed",
        urgency="low",
        domain="tasks",
    )]


def _maybe_propose_weekly_note(insights: list[Insight]) -> str | None:
    """On Sundays, propose a weekly summary note to the vault."""
    try:
        from hololoom.apps.server.vault_bridge import get_tracker, propose_note

        tracker = get_tracker()
        if not tracker.should_propose("coz"):
            return None

        today = date.today().isoformat()
        lines = [f"Weekly operational summary for the week ending {today}.\n"]
        for ins in insights:
            urgency_prefix = "(!)" if ins.urgency == "high" else ""
            lines.append(f"- {urgency_prefix}**{ins.domain}**: {ins.insight}")
            lines.append(f"  - Action: {ins.action}")

        content = "\n".join(lines)
        result = propose_note(
            title=f"Weekly Coz Summary — {today}",
            content=content,
            agent="coz",
        )
        if result.get("status") == "proposed":
            return result.get("path")
    except Exception as e:
        logger.debug("Weekly vault proposal failed: %s", e)
    return None


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/brief", response_model=ProactiveBrief)
async def proactive_brief():
    """Generate and return proactive daily brief with insights."""
    return await generate_proactive_brief()


@router.get("/history")
async def proactive_history(days: int = 7):
    """Get past proactive briefs from memory bus."""
    try:
        from hololoom.apps.server.memory_bus import get_store
        store = get_store()
        results = store.search_semantic(
            query="daily brief proactive",
            agent="coz", limit=days, decay_half_life=30.0,
        )
        return {"briefs": results, "total": len(results)}
    except Exception as e:
        return {"briefs": [], "total": 0, "error": str(e)}


@router.get("/formatted")
async def proactive_formatted():
    """Return a pre-formatted text brief suitable for direct posting to Matrix."""
    brief = await generate_proactive_brief()

    if brief.status != "ok" or not brief.insights:
        return {"text": "All systems nominal. Nothing actionable today."}

    lines = [f"**Daily Brief — {date.today().isoformat()}**\n"]
    for ins in brief.insights:
        prefix = {"high": "(!)", "medium": "(?)", "low": ""}.get(ins.urgency, "")
        lines.append(f"{prefix} **{ins.domain}**: {ins.insight}")
        lines.append(f"   → {ins.action}")

    if brief.vault_proposal:
        lines.append(f"\n_Weekly summary proposed to vault: {brief.vault_proposal}_")

    return {"text": "\n".join(lines)}
