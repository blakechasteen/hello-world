import json
import asyncio
import time
import statistics as stats
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
import sys
import os
import re

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HOLOLOOM_DISABLE_SKILLS", "1")
os.environ.setdefault("HOLOLOOM_REQUIRE_LLM", "1")
os.environ.setdefault("HOLOLOOM_OLLAMA_MODEL", "llama3.2:3b")

from HoloLoom.config import Config, MemoryBackend, Environment
from HoloLoom.protocols.types import Query, MemoryShard
from HoloLoom.weaving_orchestrator import WeavingOrchestrator


def load_shards(path: Path) -> List[MemoryShard]:
    allowed_keys = set(MemoryShard.__dataclass_fields__.keys())
    shards: List[MemoryShard] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        filtered = {k: raw[k] for k in allowed_keys if k in raw}
        shards.append(MemoryShard(**filtered))
    return shards


@dataclass
class RunResult:
    latencies_ms: List[float]
    successes: List[bool]
    confidences: List[float]
    traces: List[Any]
    extras: Dict[str, Any]


def default_config() -> Config:
    cfg = Config.fast()
    cfg.memory_backend = MemoryBackend.HYBRID
    cfg.enable_refinement = True
    cfg.refinement_threshold = 0.75
    cfg.environment = Environment.DEVELOPMENT
    cfg.enable_safety_guardrails = False
    return cfg


def _aggregate_stage_timings(traces: List[Any]) -> Dict[str, Dict[str, float]]:
    stage_times: Dict[str, List[float]] = {}
    for t in traces:
        trace = getattr(t, "trace", None)
        if trace and getattr(trace, "stage_durations", None):
            for k, v in trace.stage_durations.items():
                stage_times.setdefault(k, []).append(v)
    agg: Dict[str, Dict[str, float]] = {}
    for k, vals in stage_times.items():
        agg[k] = {
            "p50_ms": stats.median(vals),
            "p95_ms": stats.quantiles(vals, n=20)[18] if len(vals) >= 20 else max(vals),
            "avg_ms": sum(vals) / len(vals),
        }
    return agg


def run_sync(coro):
    return asyncio.run(coro)


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


async def run_battery(
    cfg: Config,
    shards: List[MemoryShard],
    queries: List[Dict[str, Any]],
    judge: Callable[[Any, Dict[str, Any]], bool],
    orchestrator_kwargs: Optional[Dict[str, Any]] = None,
    configure_orchestrator: Optional[Callable[[WeavingOrchestrator], None]] = None,
) -> RunResult:
    lat, ok, conf, traces = [], [], [], []
    orchestrator_kwargs = orchestrator_kwargs or {}
    async with WeavingOrchestrator(cfg=cfg, shards=shards, **orchestrator_kwargs) as shuttle:
        if configure_orchestrator:
            configure_orchestrator(shuttle)
        if os.getenv("HOLOLOOM_REQUIRE_LLM", "").lower() in {"1", "true", "yes"}:
            llm = getattr(getattr(shuttle, "tool_executor", None), "llm", None)
            available = bool(llm and hasattr(llm, "is_available") and llm.is_available())
            if not available:
                raise RuntimeError(
                    "LLM required but unavailable. Start Ollama and pull the model "
                    f"'{os.getenv('HOLOLOOM_OLLAMA_MODEL', 'llama3.2:3b')}'."
                )
        for q in queries:
            t0 = time.perf_counter()
            result = await shuttle.weave(Query(text=q["text"], metadata=q.get("metadata")))
            lat.append((time.perf_counter() - t0) * 1000)
            ok.append(judge(result, q))
            conf.append(result.confidence)
            traces.append(result)
    stage_breakdowns = _aggregate_stage_timings(traces)
    extras = {
        "p50_ms": stats.median(lat) if lat else 0,
        "p95_ms": stats.quantiles(lat, n=20)[18] if len(lat) >= 20 else max(lat, default=0),
        "success_rate": sum(ok) / len(ok) if ok else 0,
        "avg_confidence": sum(conf) / len(conf) if conf else 0,
        "stage_timings": stage_breakdowns,
    }
    return RunResult(lat, ok, conf, traces, extras)


def save_metrics(out_dir: Path, name: str, rr: RunResult):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(rr.extras, indent=2))

    # Emit aggregated stage timings as CSV for quick spreadsheet review
    stage_csv = out_dir / f"{name}_stage_timings.csv"
    if rr.extras.get("stage_timings"):
        lines = ["stage,p50_ms,p95_ms,avg_ms"]
        for stage, stats_map in rr.extras["stage_timings"].items():
            lines.append(
                f"{stage},{stats_map.get('p50_ms', 0):.2f},{stats_map.get('p95_ms', 0):.2f},{stats_map.get('avg_ms', 0):.2f}"
            )
        stage_csv.write_text("\n".join(lines))

    # Emit per-query stage timings as CSV (one row per query)
    per_query_csv = out_dir / f"{name}_per_query_stage_timings.csv"
    per_query_lines = ["query_index,stage,duration_ms"]
    for idx, trace in enumerate(rr.traces):
        t = getattr(trace, "trace", None)
        if t and getattr(t, "stage_durations", None):
            for stage, duration in t.stage_durations.items():
                per_query_lines.append(f"{idx},{stage},{duration:.2f}")
    per_query_csv.write_text("\n".join(per_query_lines))

    for i, trace in enumerate(rr.traces[:5]):
        (out_dir / f"{name}_trace_{i}.json").write_text(
            json.dumps(trace.to_dict(), indent=2, default=_json_default)
        )
