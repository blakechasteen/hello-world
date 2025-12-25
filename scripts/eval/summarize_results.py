import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"


def load_metrics():
    metrics = {}
    if not REPORTS_DIR.exists():
        return metrics
    for path in REPORTS_DIR.glob("*_metrics.json"):
        name = path.stem.replace("_metrics", "")
        metrics[name] = json.loads(path.read_text())
    return metrics


def compute_overall(metrics):
    if not metrics:
        return {}

    def avg(key):
        vals = [
            m.get(key)
            for m in metrics.values()
            if isinstance(m.get(key), (int, float))
        ]
        return sum(vals) / len(vals) if vals else 0

    return {
        "avg_success_rate": avg("success_rate"),
        "avg_p50_ms": avg("p50_ms"),
        "avg_p95_ms": avg("p95_ms"),
        "avg_confidence": avg("avg_confidence"),
        "task_count": len(metrics),
    }


def write_summary(metrics):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"tasks": metrics, "overall": compute_overall(metrics)}
    (REPORTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["# Eval Summary", ""]
    if not metrics:
        lines.append("No metrics found.")
    else:
        lines.append("| task | success_rate | p50_ms | p95_ms | avg_confidence |")
        lines.append("| --- | --- | --- | --- | --- |")
        for name, m in sorted(metrics.items()):
            lines.append(
                f"| {name} | {m.get('success_rate', 0):.2f} | {m.get('p50_ms', 0):.2f} | "
                f"{m.get('p95_ms', 0):.2f} | {m.get('avg_confidence', 0):.2f} |"
            )
        lines.append("")
        overall = summary.get("overall", {})
        if overall:
            lines.append("Overall:")
            lines.append(f"- avg_success_rate: {overall.get('avg_success_rate', 0):.2f}")
            lines.append(f"- avg_p50_ms: {overall.get('avg_p50_ms', 0):.2f}")
            lines.append(f"- avg_p95_ms: {overall.get('avg_p95_ms', 0):.2f}")
            lines.append(f"- avg_confidence: {overall.get('avg_confidence', 0):.2f}")
    (REPORTS_DIR / "summary.md").write_text("\n".join(lines))

    stage_csv = REPORTS_DIR / "summary_stage_timings.csv"
    stage_lines = ["task,stage,p50_ms,p95_ms,avg_ms"]
    for name, m in sorted(metrics.items()):
        stage_map = m.get("stage_timings", {}) or {}
        for stage, stats_map in stage_map.items():
            stage_lines.append(
                f"{name},{stage},{stats_map.get('p50_ms', 0):.2f},{stats_map.get('p95_ms', 0):.2f},{stats_map.get('avg_ms', 0):.2f}"
            )
    stage_csv.write_text("\n".join(stage_lines))


if __name__ == "__main__":
    write_summary(load_metrics())
