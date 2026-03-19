"""
Coz Query API — Structured access to Company Operating System data.

Three layers:
  Layer 1: Direct lookups (GET, no LLM, <10ms)
  Layer 2: Analyzed queries (POST, single LLM pass, ~200ms)
  Layer 3: Context injection (passive, into Elle's system prompt)
"""
import logging
import os
import sys
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/coz", tags=["coz-query"])

# --- SyncManager singleton ---

_sync_manager = None
_last_refresh = 0.0
CACHE_TTL_SECONDS = 600  # 10 minutes


def _get_sync_manager():
    """Lazy-load SyncManager with caching."""
    global _sync_manager, _last_refresh

    now = time.time()

    if _sync_manager is None:
        try:
            # Add coz module to path if needed
            coz_root = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "apps", "elle", "coz"
            )
            coz_root = os.path.normpath(coz_root)
            if coz_root not in sys.path:
                sys.path.insert(0, os.path.dirname(coz_root))

            from elle.coz.sync_manager import SyncManager

            # Default data directory for Coz files
            data_dir = os.environ.get(
                "COZ_DATA_DIR",
                os.path.join(coz_root, "data"),
            )
            _sync_manager = SyncManager(coz_dir=data_dir)
            _sync_manager.parse_all()
            _last_refresh = now
            logger.info("SyncManager initialized from %s", data_dir)
        except Exception as e:
            logger.error("Failed to initialize SyncManager: %s", e)
            raise HTTPException(
                status_code=503,
                detail=f"Coz data unavailable: {e}",
            )

    # Refresh cache if stale
    if now - _last_refresh > CACHE_TTL_SECONDS:
        try:
            _sync_manager.parse_all()
            _last_refresh = now
            logger.debug("SyncManager cache refreshed")
        except Exception as e:
            logger.warning("SyncManager refresh failed (using stale cache): %s", e)

    return _sync_manager


# --- Response Models ---


class CozStatusResponse(BaseModel):
    tasks: dict
    alerts: list
    seasonal_focus: dict
    timestamp: str


class CozFinancialsResponse(BaseModel):
    products: list[dict]
    reinvestment: dict
    summary: dict


class CozInventoryResponse(BaseModel):
    summary: dict
    reorder_list: list
    alerts: list


class CozScheduleResponse(BaseModel):
    context: dict


class CozAnalyzeRequest(BaseModel):
    domain: str  # "financial", "inventory", "tasks", "seasonal", "general"
    question: str


class CozAnalyzeResponse(BaseModel):
    data: dict
    analysis: str
    domain: str
    latency_ms: float


class CozBriefResponse(BaseModel):
    brief: dict


# --- Layer 1: Direct Lookups ---


@router.get("/status", response_model=CozStatusResponse)
async def coz_status():
    """Operational status: today's tasks, alerts, seasonal focus."""
    sm = _get_sync_manager()
    return CozStatusResponse(
        tasks=sm.get_daily_tasks(),
        alerts=sm.get_inventory_status().get("alerts", []),
        seasonal_focus=sm.get_seasonal_context(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


@router.get("/financials", response_model=CozFinancialsResponse)
async def coz_financials():
    """Product financials: prices, margins, COGS, reinvestment."""
    sm = _get_sync_manager()
    fin = sm.get_financial_summary()
    return CozFinancialsResponse(
        products=fin.get("products", []),
        reinvestment=fin.get("reinvestment", {}),
        summary=fin.get("summary", {}),
    )


@router.get("/inventory", response_model=CozInventoryResponse)
async def coz_inventory():
    """Inventory status: quantities, reorder alerts, thresholds."""
    sm = _get_sync_manager()
    inv = sm.get_inventory_status()
    return CozInventoryResponse(
        summary=inv.get("summary", {}),
        reorder_list=inv.get("reorder_list", []),
        alerts=inv.get("alerts", []),
    )


@router.get("/schedule", response_model=CozScheduleResponse)
async def coz_schedule():
    """Seasonal schedule: current month focus, recommendations."""
    sm = _get_sync_manager()
    return CozScheduleResponse(context=sm.get_seasonal_context())


@router.get("/brief", response_model=CozBriefResponse)
async def coz_brief():
    """Comprehensive daily brief combining all domains."""
    sm = _get_sync_manager()
    return CozBriefResponse(brief=sm.get_daily_brief())


# --- Layer 2: Analyzed Queries ---


@router.post("/analyze", response_model=CozAnalyzeResponse)
async def coz_analyze(request: CozAnalyzeRequest):
    """Analyze Coz data with a single LLM pass for interpretation."""
    start = time.time()
    sm = _get_sync_manager()

    # Gather domain-specific data
    domain_data = _get_domain_data(sm, request.domain)

    # Single LLM pass for analysis
    analysis = await _analyze_with_llm(domain_data, request.question, request.domain)

    return CozAnalyzeResponse(
        data=domain_data,
        analysis=analysis,
        domain=request.domain,
        latency_ms=round((time.time() - start) * 1000, 1),
    )


def _get_domain_data(sm, domain: str) -> dict:
    """Extract relevant data for a domain query."""
    domain_map = {
        "financial": sm.get_financial_summary,
        "financials": sm.get_financial_summary,
        "inventory": sm.get_inventory_status,
        "tasks": sm.get_daily_tasks,
        "seasonal": sm.get_seasonal_context,
        "schedule": sm.get_seasonal_context,
    }

    getter = domain_map.get(domain)
    if getter:
        return getter()

    # "general" or unknown domain — return brief
    return sm.get_daily_brief()


async def _analyze_with_llm(data: dict, question: str, domain: str) -> str:
    """Single-pass LLM analysis of domain data. Follows Elle's pattern."""
    try:
        import json

        import httpx

        ollama_url = os.environ.get("PROMPTLY_OLLAMA_URL", "http://127.0.0.1:11434")
        model = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")

        # Compact data representation to fit context
        data_str = json.dumps(data, indent=2, default=str)
        if len(data_str) > 3000:
            data_str = data_str[:3000] + "\n... (truncated)"

        prompt = (
            f"You are a business analyst for a small kitchen/farm operation.\n"
            f"Domain: {domain}\n"
            f"Data:\n{data_str}\n\n"
            f"Question: {question}\n\n"
            f"Give a concise, actionable answer. Use specific numbers from the data. "
            f"Keep it under 200 words."
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 300},
                },
            )
            resp.raise_for_status()
            result = resp.json()
            response_text = result.get("response", "")

            # Strip thinking tags (qwen3.5 quirk)
            import re
            response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

            return response_text

    except Exception as e:
        logger.warning("LLM analysis failed (returning data-only): %s", e)
        return f"(LLM unavailable: {e}. Review the raw data above.)"


# --- Layer 3: Context Injection ---


def get_coz_context_block() -> str | None:
    """
    Generate a context block for injection into Elle's system prompt.
    Returns None if Coz data is unavailable.
    """
    try:
        sm = _get_sync_manager()
        alerts = sm.get_inventory_status().get("alerts", [])
        tasks = sm.get_daily_tasks()

        lines = ["## Today's Coz Context"]

        # Critical alerts first
        if alerts:
            lines.append(f"\n**Inventory Alerts** ({len(alerts)} items):")
            for alert in alerts[:5]:  # Cap at 5 to keep prompt manageable
                if isinstance(alert, dict):
                    lines.append(f"  - {alert.get('item', 'unknown')}: {alert.get('message', alert)}")
                else:
                    lines.append(f"  - {alert}")

        # Today's focus
        focus = tasks.get("focus_area", "")
        if focus:
            lines.append(f"\n**Today's Focus**: {focus}")

        # Priority tasks
        priority = tasks.get("priority_tasks", [])
        if priority:
            lines.append(f"\n**Priority Tasks** ({len(priority)}):")
            for task in priority[:3]:
                if isinstance(task, dict):
                    lines.append(f"  - {task.get('title', task.get('name', str(task)))}")
                else:
                    lines.append(f"  - {task}")

        return "\n".join(lines)

    except Exception as e:
        logger.debug("Coz context injection unavailable: %s", e)
        return None


# --- Status endpoint ---


@router.get("/health")
async def coz_health():
    """Coz subsystem health check."""
    try:
        _get_sync_manager()
        return {
            "status": "healthy",
            "cache_age_seconds": round(time.time() - _last_refresh, 1),
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
        }
    except Exception as e:
        return {
            "status": "unavailable",
            "error": str(e),
        }
