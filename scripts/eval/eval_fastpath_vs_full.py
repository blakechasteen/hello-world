import json
from pathlib import Path

from common import load_shards, default_config, run_battery, save_metrics, run_sync

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


def judge(trace, item):
    if getattr(trace, "metadata", None) and trace.metadata.get("status") == "error":
        return False
    return bool(trace.response and trace.response.strip())


def configure_orchestrator(shuttle):
    if getattr(shuttle, "tool_executor", None):
        shuttle.tool_executor.tools = ["answer"]


async def run_both(shards, queries):
    full_cfg = default_config()
    fast_cfg = default_config()
    fast_cfg.enable_refinement = False
    orch_kwargs = {"enable_semantic_cache": False}
    full_rr = await run_battery(
        full_cfg,
        shards,
        queries,
        judge,
        orchestrator_kwargs=orch_kwargs,
        configure_orchestrator=configure_orchestrator,
    )
    fast_rr = await run_battery(
        fast_cfg,
        shards,
        queries,
        judge,
        orchestrator_kwargs=orch_kwargs,
        configure_orchestrator=configure_orchestrator,
    )
    return full_rr, fast_rr


if __name__ == "__main__":
    shards = load_shards(DATA_DIR / "mixed_shards.jsonl")
    queries = [
        json.loads(l)
        for l in (DATA_DIR / "mixed_queries.jsonl").read_text().splitlines()
        if l.strip()
    ]
    full_rr, fast_rr = run_sync(run_both(shards, queries))
    save_metrics(REPORTS_DIR, "full_weave", full_rr)
    save_metrics(REPORTS_DIR, "fast_path", fast_rr)
