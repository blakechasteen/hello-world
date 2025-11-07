"""
HoloLoom COS - FastAPI Backend

REST API for web interfaces (dashboard, POS, voice input).

Endpoints:
- POST /voice/input - Voice transcription + event logging
- POST /log/text - Text event logging
- GET /summary/today - Today's summary
- GET /summary/week - This week's summary
- GET /accounting/pl - Income statement
- GET /accounting/bs - Balance sheet
- GET /accounting/cf - Cash flow
- POST /pos/sale - Record POS sale
- GET /review/morning - Morning planning prompts
- POST /review/evening - Evening review
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cos.core.storage import EventStore
from cos.core.parser import parse_input
from cos.core.types import EventSource, EventType
from cos.intelligence.accounting import AccountingGenerator
from cos.interface.voice_input import VoiceInputHandler
from cos.interface.daily_review import DailyReviewWorkflow

# Create app
app = FastAPI(
    title="HoloLoom COS API",
    description="Voice-first business management API",
    version="1.0.0"
)

# CORS for web interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
store = EventStore("cos_production.db")
voice_handler = VoiceInputHandler(store)
accounting = AccountingGenerator(store)
review_workflow = DailyReviewWorkflow(store)


# Pydantic models
class TextLogRequest(BaseModel):
    text: str
    source: str = "manual"


class POSSaleRequest(BaseModel):
    items: List[Dict[str, Any]]
    total: float
    payment_method: str
    timestamp: Optional[str] = None


class EveningReviewRequest(BaseModel):
    completed: List[str]
    not_completed: List[str]
    learnings: List[str]
    tomorrow_priorities: List[str]
    energy_level: int


# Health check
@app.get("/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


# Voice input
@app.post("/voice/input")
async def voice_input(audio: UploadFile = File(...)):
    """
    Process voice input.

    Upload audio file → transcribe → parse → log
    """
    # Save audio temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        content = await audio.read()
        f.write(content)
        temp_path = f.name

    try:
        result = await voice_handler.process_voice_input(temp_path)
        return result
    finally:
        Path(temp_path).unlink(missing_ok=True)


# Text logging
@app.post("/log/text")
async def log_text(request: TextLogRequest):
    """
    Log event from text input.

    Parses natural language → creates event
    """
    intent, verification = parse_input(
        request.text,
        EventSource.VOICE if request.source == "voice" else EventSource.MANUAL
    )

    event = intent.to_event(request.text)

    if verification:
        # Return verification request for HITL
        return {
            "needs_verification": True,
            "event": event.to_dict(),
            "reason": verification.reason,
            "confidence": float(intent.confidence)
        }

    # Auto-verify if high confidence
    event.verified = True
    event_id = await store.store(event)

    return {
        "needs_verification": False,
        "event_id": event_id,
        "confidence": float(intent.confidence)
    }


# Summaries
@app.get("/summary/today")
async def summary_today():
    """Get today's business summary"""
    today = datetime.now()
    summary = await store.get_daily_summary(today)

    return {
        "date": today.strftime('%Y-%m-%d'),
        "revenue": float(summary.revenue),
        "cogs": float(summary.cogs),
        "labor_cost": float(summary.labor_cost),
        "profit": float(summary.profit),
        "hours_worked": float(summary.hours_worked),
        "hourly_rate": float(summary.hourly_rate()),
        "profit_margin": float(summary.profit_margin()),
        "revenue_by_product": {
            k.value: float(v) for k, v in summary.revenue_by_product.items()
        }
    }


@app.get("/summary/week")
async def summary_week():
    """Get this week's business summary"""
    week = datetime.now().strftime('%Y-W%W')
    summary = await store.get_weekly_summary(week)

    return {
        "week": week,
        "revenue": float(summary.revenue),
        "cogs": float(summary.cogs),
        "labor_cost": float(summary.labor_cost),
        "overhead": float(summary.overhead),
        "profit": float(summary.profit),
        "hours_worked": float(summary.hours_worked),
        "active_product_lines": summary.active_product_lines
    }


# Accounting
@app.get("/accounting/pl")
async def income_statement(days: int = 7):
    """
    Generate Income Statement (P&L).

    Args:
        days: Number of days to include (default: 7)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    pl = await accounting.generate_income_statement(start_date, end_date)

    return {
        "period_start": pl.period_start.isoformat(),
        "period_end": pl.period_end.isoformat(),
        "revenue_by_product": {k: float(v) for k, v in pl.revenue_by_product.items()},
        "total_revenue": float(pl.total_revenue),
        "total_cogs": float(pl.total_cogs),
        "gross_profit": float(pl.gross_profit),
        "gross_margin": float(pl.gross_margin),
        "labor": float(pl.labor),
        "overhead": float(pl.total_operating_expenses - pl.labor),
        "net_profit": float(pl.net_profit),
        "net_margin": float(pl.net_margin)
    }


@app.get("/accounting/bs")
async def balance_sheet():
    """Generate Balance Sheet as of today"""
    bs = await accounting.generate_balance_sheet(datetime.now())

    return {
        "as_of_date": bs.as_of_date.isoformat(),
        "assets": {
            "cash": float(bs.cash),
            "inventory_rm": float(bs.inventory_raw_materials),
            "inventory_fg": float(bs.inventory_finished_goods),
            "equipment": float(bs.equipment),
            "total": float(bs.total_assets)
        },
        "liabilities": {
            "accounts_payable": float(bs.accounts_payable),
            "loans": float(bs.loans),
            "total": float(bs.total_liabilities)
        },
        "equity": {
            "owner_investment": float(bs.owner_investment),
            "retained_earnings": float(bs.retained_earnings),
            "total": float(bs.total_equity)
        },
        "is_balanced": bs.is_balanced
    }


@app.get("/accounting/cf")
async def cash_flow(days: int = 7):
    """Generate Cash Flow Statement"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    cf = await accounting.generate_cash_flow(start_date, end_date)

    return {
        "period_start": cf.period_start.isoformat(),
        "period_end": cf.period_end.isoformat(),
        "operating_cash_flow": float(cf.operating_cash_flow),
        "investing_cash_flow": float(cf.investing_cash_flow),
        "financing_cash_flow": float(cf.financing_cash_flow),
        "net_change_in_cash": float(cf.net_change_in_cash),
        "opening_cash": float(cf.opening_cash),
        "closing_cash": float(cf.closing_cash)
    }


# POS
@app.post("/pos/sale")
async def pos_sale(sale: POSSaleRequest):
    """
    Record POS sale.

    Creates sale events for each item.
    """
    event_ids = []

    for item in sale.items:
        # Create sale event
        text = f"Sold {item['quantity']} {item['name']} for ${item['total']:.2f}"

        intent, _ = parse_input(text, EventSource.MANUAL)
        event = intent.to_event(text)
        event.verified = True
        event.verification_note = f"POS sale ({sale.payment_method})"

        event_id = await store.store(event)
        event_ids.append(event_id)

    return {
        "success": True,
        "event_ids": event_ids,
        "total": sale.total,
        "items_count": len(sale.items)
    }


# Daily Review
@app.get("/review/morning")
async def morning_planning():
    """Get morning planning prompts + context"""
    # Get yesterday's summary
    yesterday = datetime.now() - timedelta(days=1)
    yesterday_summary = await store.get_daily_summary(yesterday)

    # Get this week's summary
    week = datetime.now().strftime('%Y-W%W')
    week_summary = await store.get_weekly_summary(week)

    return {
        "date": datetime.now().strftime('%Y-%m-%d'),
        "yesterday": {
            "revenue": float(yesterday_summary.revenue),
            "hours": float(yesterday_summary.hours_worked),
            "hourly_rate": float(yesterday_summary.hourly_rate())
        },
        "week_so_far": {
            "revenue": float(week_summary.revenue),
            "hours": float(week_summary.hours_worked)
        },
        "prompts": [
            "What are your top 3 goals for today?",
            "How many hours do you have available?",
            "What product(s) will you focus on?",
            "Revenue target for today?"
        ]
    }


@app.post("/review/evening")
async def evening_review(review: EveningReviewRequest):
    """
    Submit evening review.

    Stores review event and generates insights.
    """
    # Get today's summary
    today = datetime.now()
    today_summary = await store.get_daily_summary(today)

    # Create review event
    from cos.core.types import Event
    review_data = {
        'completed': review.completed,
        'not_completed': review.not_completed,
        'learnings': review.learnings,
        'tomorrow_priorities': review.tomorrow_priorities,
        'energy_level': review.energy_level
    }

    review_event = Event(
        type=EventType.REVIEW,
        raw_input=f"Evening review: {len(review.completed)} tasks completed",
        source=EventSource.MANUAL,
        parsed_data=review_data
    )

    event_id = await store.store(review_event)

    # Generate insights
    insights = []

    if review.energy_level < 5:
        insights.append({
            "type": "warning",
            "message": f"Low energy ({review.energy_level}/10) - consider lightening workload"
        })

    if today_summary.hours_worked > 10:
        insights.append({
            "type": "warning",
            "message": f"Long day ({today_summary.hours_worked:.1f} hours) - watch for burnout"
        })

    if today_summary.hourly_rate() > 25:
        insights.append({
            "type": "success",
            "message": f"Strong hourly rate (${today_summary.hourly_rate():.2f}/hr)!"
        })

    return {
        "event_id": event_id,
        "insights": insights,
        "today_summary": {
            "revenue": float(today_summary.revenue),
            "hours": float(today_summary.hours_worked),
            "hourly_rate": float(today_summary.hourly_rate()),
            "profit": float(today_summary.profit)
        }
    }


# Serve static files
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve dashboard"""
    dashboard_path = Path(__file__).parent / "dashboard.html"
    if dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text())
    return {"message": "COS API is running. Visit /docs for API documentation."}


@app.get("/pos", response_class=HTMLResponse)
async def pos_interface():
    """Serve POS interface"""
    pos_path = Path(__file__).parent / "pos.html"
    if pos_path.exists():
        return HTMLResponse(content=pos_path.read_text())
    return {"error": "POS interface not found"}


if __name__ == '__main__':
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Starting HoloLoom COS API Server")
    print("="*60)
    print("\nEndpoints:")
    print("  Dashboard:  http://localhost:8000/")
    print("  POS:        http://localhost:8000/pos")
    print("  API Docs:   http://localhost:8000/docs")
    print("\n" + "="*60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
