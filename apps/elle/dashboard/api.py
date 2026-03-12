"""
FastAPI Backend for Elle Core Dashboard

RESTful API exposing all Elle Core data:
- Budgets and variance analysis
- Task tracking and profit analysis
- SOPs and production data
- Decision engine recommendations
- Real-time analytics

Created: 2025-11-15
Author: Blake Chasteen (Agent C)
"""

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

# Elle Core imports
from elle.budget import BudgetBuilder, Budget, BudgetPeriod, BudgetCategory
from elle.tracker import TaskTracker, TaskResult, TaskStatus
from elle.sop_schema import SOP
from elle.mirrorcore import DecisionEngine, Recommendation

# Initialize Elle Core components
budget_builder = BudgetBuilder()
tracker = TaskTracker()
decision_engine = DecisionEngine(tracker=tracker)

# Create FastAPI app
app = FastAPI(
    title="Elle Core Dashboard API",
    description="Real-time operational intelligence for farm operations",
    version="1.0.0"
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================================================
# BUDGET ENDPOINTS
# ============================================================================

@app.get("/api/budgets")
async def list_budgets() -> List[Dict[str, Any]]:
    """List all budgets"""
    budgets = budget_builder.list_budgets()
    return [
        {
            "budget_id": b.budget_id,
            "name": b.name,
            "period": b.period.value,
            "start_date": b.start_date.isoformat(),
            "end_date": b.end_date.isoformat(),
            "total_planned": b.total_planned,
            "total_actual": b.total_actual,
            "variance": b.variance,
            "variance_pct": b.variance_pct
        }
        for b in budgets
    ]


@app.get("/api/budgets/{budget_id}")
async def get_budget(budget_id: str) -> Dict[str, Any]:
    """Get detailed budget information"""
    budget = budget_builder.get_budget(budget_id)
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")

    return {
        "budget_id": budget.budget_id,
        "name": budget.name,
        "period": budget.period.value,
        "start_date": budget.start_date.isoformat(),
        "end_date": budget.end_date.isoformat(),
        "lines": [line.to_dict() for line in budget.lines],
        "total_planned": budget.total_planned,
        "total_actual": budget.total_actual,
        "variance": budget.variance,
        "variance_pct": budget.variance_pct,
        "revenue_planned": budget.revenue_planned,
        "revenue_actual": budget.revenue_actual,
        "costs_planned": budget.costs_planned,
        "costs_actual": budget.costs_actual,
        "reinvestment_planned": budget.reinvestment_planned,
        "reinvestment_actual": budget.reinvestment_actual
    }


@app.get("/api/budgets/{budget_id}/variance")
async def get_variance_report(budget_id: str) -> Dict[str, Any]:
    """Get variance analysis for budget"""
    budget = budget_builder.get_budget(budget_id)
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")

    variance_report = budget_builder.variance_report(budget_id)
    return variance_report


@app.get("/api/budgets/{budget_id}/forecast")
async def get_cash_flow_forecast(budget_id: str, weeks: int = 12) -> Dict[str, Any]:
    """Get cash flow forecast for budget"""
    budget = budget_builder.get_budget(budget_id)
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")

    forecast = budget_builder.forecast_cash_flow(budget_id, weeks=weeks)
    return forecast


# ============================================================================
# SOP ENDPOINTS
# ============================================================================

@app.get("/api/sops")
async def list_sops() -> List[Dict[str, Any]]:
    """List all SOPs"""
    sops = decision_engine.sops
    return [
        {
            "sop_id": sop_id,
            "title": sop.title,
            "category": sop.category,
            "version": sop.version,
            "batch_size": sop.batch_size,
            "estimated_time": sop.estimated_time_minutes,
            "material_cost": sop.total_material_cost,
            "sale_price": sop.sale_price
        }
        for sop_id, sop in sops.items()
    ]


@app.get("/api/sops/{sop_id}")
async def get_sop(sop_id: str) -> Dict[str, Any]:
    """Get detailed SOP information"""
    sop = decision_engine.sops.get(sop_id)
    if not sop:
        raise HTTPException(status_code=404, detail="SOP not found")

    return {
        "sop_id": sop.sop_id,
        "title": sop.title,
        "category": sop.category,
        "version": sop.version,
        "description": sop.description,
        "batch_size": sop.batch_size,
        "estimated_time_minutes": sop.estimated_time_minutes,
        "ingredients": [ing.to_dict() for ing in sop.ingredients],
        "steps": [step.to_dict() for step in sop.steps],
        "total_material_cost": sop.total_material_cost,
        "sale_price": sop.sale_price,
        "profit_per_batch": sop.profit_per_batch,
        "profit_margin": sop.profit_margin
    }


# ============================================================================
# TASK ENDPOINTS
# ============================================================================

@app.get("/api/tasks")
async def list_tasks(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent task history"""
    tasks = tracker.get_recent_tasks(limit=limit)
    return [task.to_dict() for task in tasks]


@app.get("/api/tasks/active")
async def get_active_tasks() -> List[Dict[str, Any]]:
    """Get currently active tasks"""
    active = tracker.get_active_task()
    if active:
        return [active.to_dict()]
    return []


@app.post("/api/tasks/start")
async def start_task(
    task_name: str,
    sop_id: Optional[str] = None,
    category: str = "general"
) -> Dict[str, Any]:
    """Start a new task"""
    result = await tracker.start_task(task_name, sop_id=sop_id, category=category)
    return result.to_dict()


@app.post("/api/tasks/end")
async def end_task(
    units_produced: int = 0,
    quality_score: Optional[float] = None,
    notes: str = ""
) -> Dict[str, Any]:
    """End the active task"""
    result = await tracker.end_task(
        units_produced=units_produced,
        quality_score=quality_score,
        notes=notes
    )
    if not result:
        raise HTTPException(status_code=404, detail="No active task")
    return result.to_dict()


@app.post("/api/tasks/{task_id}/pause")
async def pause_task(task_id: str) -> Dict[str, str]:
    """Pause a task"""
    await tracker.pause_task()
    return {"status": "paused", "task_id": task_id}


@app.post("/api/tasks/{task_id}/resume")
async def resume_task(task_id: str) -> Dict[str, str]:
    """Resume a paused task"""
    await tracker.resume_task()
    return {"status": "resumed", "task_id": task_id}


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/api/analytics")
async def get_analytics() -> Dict[str, Any]:
    """Get summary analytics"""
    # Recent tasks (last 30 days)
    recent_tasks = tracker.get_recent_tasks(limit=1000)
    thirty_days_ago = datetime.now() - timedelta(days=30)
    recent_tasks = [
        t for t in recent_tasks
        if t.created_at >= thirty_days_ago
    ]

    # Calculate metrics
    total_revenue = sum(t.revenue for t in recent_tasks)
    total_costs = sum(t.total_cost for t in recent_tasks)
    total_profit = total_revenue - total_costs
    total_hours = sum(t.active_time_hours for t in recent_tasks)

    # Product breakdown
    product_stats = {}
    for task in recent_tasks:
        product = task.task_name
        if product not in product_stats:
            product_stats[product] = {
                "revenue": 0.0,
                "costs": 0.0,
                "profit": 0.0,
                "hours": 0.0,
                "units": 0
            }
        product_stats[product]["revenue"] += task.revenue
        product_stats[product]["costs"] += task.total_cost
        product_stats[product]["profit"] += task.profit
        product_stats[product]["hours"] += task.active_time_hours
        product_stats[product]["units"] += task.units_produced

    # Calculate ROI per product
    for product, stats in product_stats.items():
        if stats["hours"] > 0:
            stats["hourly_roi"] = stats["profit"] / stats["hours"]
        else:
            stats["hourly_roi"] = 0.0
        if stats["costs"] > 0:
            stats["profit_margin"] = (stats["profit"] / stats["revenue"]) * 100
        else:
            stats["profit_margin"] = 0.0

    return {
        "period": "Last 30 days",
        "total_revenue": total_revenue,
        "total_costs": total_costs,
        "total_profit": total_profit,
        "total_hours": total_hours,
        "avg_hourly_roi": total_profit / total_hours if total_hours > 0 else 0.0,
        "profit_margin": (total_profit / total_revenue * 100) if total_revenue > 0 else 0.0,
        "products": product_stats,
        "task_count": len(recent_tasks)
    }


@app.get("/api/recommendations")
async def get_recommendations() -> List[Dict[str, Any]]:
    """Get daily recommendations from decision engine"""
    recommendations = decision_engine.get_daily_recommendations()
    return [
        {
            "action": rec.action,
            "reasoning": rec.reasoning,
            "priority": rec.priority,
            "estimated_profit": rec.estimated_profit,
            "time_required": rec.time_required,
            "materials_needed": rec.materials_needed,
            "dependencies": rec.dependencies,
            "confidence": rec.confidence
        }
        for rec in recommendations
    ]


@app.get("/api/recommendations/weekly")
async def get_weekly_plan() -> Dict[str, Any]:
    """Get optimized weekly plan"""
    plan = decision_engine.create_weekly_plan(available_hours=40.0)
    return {
        "tasks": plan.tasks,
        "total_hours": plan.total_hours,
        "total_cost": plan.total_cost,
        "expected_revenue": plan.expected_revenue,
        "expected_profit": plan.expected_profit,
        "utilization": plan.utilization
    }


# ============================================================================
# STATIC PAGES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Main dashboard page"""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return index_file.read_text()
    return "<h1>Elle Core Dashboard</h1><p>Static files not found.</p>"


@app.get("/budget", response_class=HTMLResponse)
async def budget_page():
    """Budget dashboard page"""
    budget_file = static_dir / "budget.html"
    if budget_file.exists():
        return budget_file.read_text()
    raise HTTPException(status_code=404, detail="Page not found")


@app.get("/products", response_class=HTMLResponse)
async def products_page():
    """Product ROI dashboard page"""
    products_file = static_dir / "products.html"
    if products_file.exists():
        return products_file.read_text()
    raise HTTPException(status_code=404, detail="Page not found")


@app.get("/planner", response_class=HTMLResponse)
async def planner_page():
    """Weekly planner page"""
    planner_file = static_dir / "planner.html"
    if planner_file.exists():
        return planner_file.read_text()
    raise HTTPException(status_code=404, detail="Page not found")


@app.get("/tracker", response_class=HTMLResponse)
async def tracker_page():
    """Task tracker page"""
    tracker_file = static_dir / "tracker.html"
    if tracker_file.exists():
        return tracker_file.read_text()
    raise HTTPException(status_code=404, detail="Page not found")


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
