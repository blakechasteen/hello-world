"""
Reproducibility Infrastructure

Provides:
- Deterministic seeding for all randomized components
- Run metadata capture (versions, platform, timestamps)
- Run comparison across multiple evaluation runs
- Fingerprinting for detecting dataset/config drift
"""

import hashlib
import json
import os
import platform
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


@dataclass
class RunMetadata:
    """Complete metadata for a single evaluation run."""
    run_id: str
    timestamp: str
    mode: str
    seed: int
    python_version: str
    platform_info: str
    framework_version: str
    corpus_fingerprint: str
    query_fingerprint: str
    dependency_versions: Dict[str, str]
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunComparison:
    """Comparison between two evaluation runs."""
    run_a_id: str
    run_b_id: str
    score_delta: float
    benchmark_deltas: Dict[str, float]
    config_diffs: List[str]
    improved: List[str]
    regressed: List[str]
    unchanged: List[str]
    summary: str


def generate_run_id(seed: int, mode: str) -> str:
    """Generate a deterministic run ID."""
    raw = f"{seed}-{mode}-{datetime.now().strftime('%Y%m%d')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def fingerprint_data(data: Any) -> str:
    """Generate SHA-256 fingerprint of data for drift detection."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def set_global_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    # Set numpy seed if available
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def capture_run_metadata(
    mode: str,
    seed: int,
    corpus: Any,
    queries: Any,
    framework_version: str = "2.0.0",
) -> RunMetadata:
    """Capture complete metadata for the current run."""
    dep_versions = {}

    for mod_name in ["numpy", "sentence_transformers", "torch", "networkx"]:
        try:
            mod = __import__(mod_name.replace("-", "_"))
            dep_versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            dep_versions[mod_name] = "not_installed"

    return RunMetadata(
        run_id=generate_run_id(seed, mode),
        timestamp=datetime.now().isoformat(),
        mode=mode,
        seed=seed,
        python_version=sys.version.split()[0],
        platform_info=f"{platform.system()} {platform.release()} {platform.machine()}",
        framework_version=framework_version,
        corpus_fingerprint=fingerprint_data(corpus),
        query_fingerprint=fingerprint_data(queries),
        dependency_versions=dep_versions,
    )


def compare_runs(
    run_a: Dict[str, Any],
    run_b: Dict[str, Any],
    threshold: float = 0.05,
) -> RunComparison:
    """
    Compare two evaluation runs and identify improvements/regressions.

    Args:
        run_a: First run report dict (baseline)
        run_b: Second run report dict (new)
        threshold: Minimum score delta to count as change
    """
    score_a = run_a.get("overall_score", 0)
    score_b = run_b.get("overall_score", 0)
    score_delta = score_b - score_a

    benchmarks_a = {b["name"]: b for b in run_a.get("benchmarks", [])}
    benchmarks_b = {b["name"]: b for b in run_b.get("benchmarks", [])}

    all_names = set(benchmarks_a.keys()) | set(benchmarks_b.keys())
    benchmark_deltas = {}
    improved = []
    regressed = []
    unchanged = []

    for name in sorted(all_names):
        sa = benchmarks_a.get(name, {}).get("score", 0)
        sb = benchmarks_b.get(name, {}).get("score", 0)
        delta = sb - sa
        benchmark_deltas[name] = delta

        if delta > threshold:
            improved.append(name)
        elif delta < -threshold:
            regressed.append(name)
        else:
            unchanged.append(name)

    # Config diffs
    config_diffs = []
    meta_a = run_a.get("metadata", run_a.get("system_info", {}))
    meta_b = run_b.get("metadata", run_b.get("system_info", {}))
    if meta_a.get("dependencies") != meta_b.get("dependencies"):
        config_diffs.append("Dependencies changed")
    if run_a.get("mode") != run_b.get("mode"):
        config_diffs.append(f"Mode: {run_a.get('mode')} -> {run_b.get('mode')}")

    # Summary
    if regressed:
        summary = f"REGRESSION: {len(regressed)} benchmarks regressed ({', '.join(regressed)})"
    elif improved:
        summary = f"IMPROVED: {len(improved)} benchmarks improved ({', '.join(improved)})"
    else:
        summary = "STABLE: No significant changes detected"

    return RunComparison(
        run_a_id=run_a.get("metadata", {}).get("run_id", "unknown"),
        run_b_id=run_b.get("metadata", {}).get("run_id", "unknown"),
        score_delta=score_delta,
        benchmark_deltas=benchmark_deltas,
        config_diffs=config_diffs,
        improved=improved,
        regressed=regressed,
        unchanged=unchanged,
        summary=summary,
    )


def load_previous_run(results_dir: str, mode: str) -> Optional[Dict[str, Any]]:
    """Load the most recent previous run for comparison."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return None

    latest_path = results_path / f"eval_report_{mode}.json"
    if not latest_path.exists():
        return None

    try:
        with open(latest_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_run_with_metadata(
    report_dict: Dict[str, Any],
    metadata: RunMetadata,
    results_dir: str,
) -> str:
    """Save run with full metadata for future comparison."""
    report_dict["metadata"] = metadata.to_dict()

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Save with run ID for history
    run_path = results_path / f"eval_run_{metadata.run_id}.json"
    with open(run_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    return str(run_path)
