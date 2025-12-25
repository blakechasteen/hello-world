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
    answer = data.get("answer")
    sources = data.get("sources")
    if not isinstance(answer, str) or not isinstance(sources, list) or not sources:
        return False
    normalized = answer.strip().lower().rstrip(".")
    for expected in item["answers"]:
        if normalized == expected.strip().lower().rstrip("."):
            return True
    return False


def configure_orchestrator(shuttle):
    if getattr(shuttle, "tool_executor", None):
        shuttle.tool_executor.tools = ["answer"]


if __name__ == "__main__":
    shards = load_shards(DATA_DIR / "retrieval_qa_shards.jsonl")
    queries = [
        json.loads(l)
        for l in (DATA_DIR / "retrieval_qa_queries.jsonl").read_text().splitlines()
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
    save_metrics(REPORTS_DIR, "retrieval_qa", rr)
