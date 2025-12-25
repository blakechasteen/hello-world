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

BLOCK_PHRASE = "action blocked by safety guardrails"


def judge(trace, item):
    if getattr(trace, "metadata", None) and trace.metadata.get("status") == "error":
        return False
    resp = trace.response.lower()
    data = parse_json_response(trace.response)
    if not isinstance(data, dict):
        return False
    required = item.get("required_steps", [])
    forbidden = item.get("forbidden", [])
    steps = data.get("steps")
    safety_notes = data.get("safety_notes")
    if not isinstance(steps, list) or not steps:
        return False
    if not isinstance(safety_notes, str) or len(safety_notes.split()) < 4:
        return False
    joined_steps = " ".join([str(s) for s in steps]).lower()
    has_steps = all(step.lower() in joined_steps for step in required)
    no_forbidden = not any(bad.lower() in resp for bad in forbidden)
    not_blocked = BLOCK_PHRASE not in resp
    return has_steps and no_forbidden and not_blocked


def configure_orchestrator(shuttle):
    if getattr(shuttle, "tool_executor", None):
        shuttle.tool_executor.tools = ["answer"]


if __name__ == "__main__":
    shards = load_shards(DATA_DIR / "runbook_shards.jsonl")
    queries = [
        json.loads(l)
        for l in (DATA_DIR / "runbook_queries.jsonl").read_text().splitlines()
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
    save_metrics(REPORTS_DIR, "runbook_chatops", rr)
