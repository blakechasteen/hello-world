import json
from pathlib import Path

from common import (
    load_shards,
    default_config,
    run_battery,
    save_metrics,
    run_sync,
    parse_json_response,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


def judge(trace, item):
    if getattr(trace, "metadata", None) and trace.metadata.get("status") == "error":
        return False
    data = parse_json_response(trace.response)
    if not isinstance(data, dict):
        return False
    summary = data.get("summary")
    sources = data.get("sources")
    actions = data.get("actions")
    if not isinstance(summary, str) or len(summary.split()) < 8:
        return False
    if not isinstance(sources, list) or len(sources) < 1:
        return False
    if not isinstance(actions, list) or len(actions) < 2:
        return False
    for action in actions:
        if not isinstance(action, dict):
            return False
        if not action.get("action") or not action.get("owner"):
            return False
    return trace.tool_used in {"answer", "search", "notion_write"}


def configure_orchestrator(shuttle):
    if getattr(shuttle, "tool_executor", None):
        shuttle.tool_executor.tools = ["answer"]


if __name__ == "__main__":
    shards = load_shards(DATA_DIR / "tool_chain_shards.jsonl")
    queries = [
        json.loads(l)
        for l in (DATA_DIR / "tool_chain_queries.jsonl").read_text().splitlines()
        if l.strip()
    ]
    cfg = default_config()
    rr = run_sync(
        run_battery(
            cfg,
            shards,
            queries,
            judge,
            orchestrator_kwargs={"enable_semantic_cache": False},
            configure_orchestrator=configure_orchestrator,
        )
    )
    save_metrics(REPORTS_DIR, "tool_chain", rr)
