"""
Heartbeat writer — updates heartbeat.md from rolling metrics.

This is the agent's self-report mechanism. After each evaluation window,
the heartbeat file gets updated with current vitals, observations, and
drift detection.

The delta between what heartbeat.md says and what the metrics actually show
is itself a measurement of groundedness.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from hololoom.agents.eval.runner import RollingMetrics

logger = logging.getLogger(__name__)

_IDENTITIES_ROOT = Path(__file__).resolve().parent.parent / "identities"


def update_heartbeat(
    agent_name: str,
    metrics: RollingMetrics,
    differentiation: float = 0.0,
    observations: list[str] | None = None,
    root: Path | None = None,
) -> None:
    """
    Write current metrics to heartbeat.md.

    This is the agent-owned file. Updated autonomously after measurement.
    """
    root = root or _IDENTITIES_ROOT
    heartbeat_path = root / agent_name / "heartbeat.md"

    if not heartbeat_path.parent.exists():
        logger.warning("No identity dir for %s", agent_name)
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    g = metrics.groundedness()
    c = metrics.coherence()
    n = len(metrics.results)

    # Detect trends (compare first half vs second half of window)
    half = n // 2
    if half > 5:
        early = metrics.results[:half]
        late = metrics.results[half:]
        from hololoom.agents.eval.axes import groundedness_score, coherence_score
        g_early = groundedness_score(early)
        g_late = groundedness_score(late)
        c_early = coherence_score(early)
        c_late = coherence_score(late)
        g_trend = _trend(g_late - g_early)
        c_trend = _trend(c_late - c_early)
    else:
        g_trend = "—"
        c_trend = "—"

    d_trend = "—"  # Differentiation needs cross-agent data

    obs_block = "\n".join(f"- {o}" for o in (observations or []))
    if not obs_block:
        obs_block = "No new observations."

    # Build drift log entries for significant changes
    drift_entries = []
    if g_trend == "▼":
        drift_entries.append(f"- {now}: groundedness declining ({g:.3f})")
    if c_trend == "▼":
        drift_entries.append(f"- {now}: coherence declining ({c:.3f})")
    drift_block = "\n".join(drift_entries) if drift_entries else "(stable)"

    content = f"""# heartbeat.md — {agent_name.capitalize()}
## Last updated: {now}

## Vitals (rolling, last {n} interactions)
| Axis | Value | Trend |
|------|-------|-------|
| coherence | {c:.3f} | {c_trend} |
| differentiation | {differentiation:.3f} | {d_trend} |
| groundedness | {g:.3f} | {g_trend} |

## Observations
{obs_block}

## Drift Log
{drift_block}
"""

    heartbeat_path.write_text(content, encoding="utf-8")
    logger.info("%s heartbeat updated: g=%.3f c=%.3f d=%.3f", agent_name, g, c, differentiation)


def _trend(delta: float) -> str:
    if delta > 0.02:
        return "▲"
    elif delta < -0.02:
        return "▼"
    return "—"
