"""
Demons — Condition-action pattern monitors for response quality.

Classical AI "demons" (from Pandemonium architecture) that watch
for patterns in generated responses and raise flags when triggered.

Each demon monitors a specific quality dimension:
  - GroundingDemon:   Response claims not supported by retrieved context
  - BrevityDemon:     Response too short relative to context richness
  - RepetitionDemon:  Response restates the query without adding info
  - HedgingDemon:     Response over-hedges instead of committing to answers

Demons post flags to the blackboard that production rules can consume.
For example, a "grounding_alert" flag can trigger a FLAG shell re-run.

Implements the KnowledgeSource protocol at POST_GENERATE phase.

No external dependencies. No LLM calls.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

from .knowledge_source import Phase

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DemonConfig:
    """Configuration for the Demon system."""
    grounding_threshold: float = 0.3   # Min fraction of response claims in context
    brevity_min_ratio: float = 0.3     # Min response_words / context_words
    repetition_threshold: float = 0.5  # Max Jaccard between query and response
    hedging_threshold: int = 3         # Max hedge phrases before alert
    enable_grounding: bool = True
    enable_brevity: bool = True
    enable_repetition: bool = True
    enable_hedging: bool = True


@dataclass
class DemonAlert:
    """A demon firing event."""
    demon_name: str
    severity: float         # 0.0-1.0 (higher = worse)
    detail: str
    metric: float           # The measured value that triggered


# =============================================================================
# Individual Demon Implementations
# =============================================================================

_HEDGE_PHRASES = frozenset({
    "it depends", "it is important to note", "there are many factors",
    "it could be argued", "some might say", "in some cases",
    "it is worth noting", "generally speaking", "it is possible that",
    "one could argue", "there is no simple answer", "it varies",
    "depending on the context", "this is a complex topic",
})

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "and",
    "but", "or", "nor", "so", "yet", "both", "either", "all", "any",
    "than", "too", "very", "just", "also", "about", "up", "out", "if",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "we", "our", "you", "your", "he", "she", "his", "her", "which",
    "who", "what", "where", "when", "why", "how", "there", "then",
    "not", "no",
})


def _content_words(text: str) -> set[str]:
    """Extract meaningful content words from text."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {w for w in tokens if w not in _STOP_WORDS and len(w) > 2}


def check_grounding(response: str, context: str, threshold: float) -> DemonAlert | None:
    """Check if response claims are grounded in retrieved context.

    Computes what fraction of response content words appear in the context.
    Low grounding suggests hallucination or confabulation.
    """
    resp_words = _content_words(response)
    ctx_words = _content_words(context)

    if not resp_words:
        return None

    grounded = len(resp_words & ctx_words)
    ratio = grounded / len(resp_words)

    if ratio < threshold:
        return DemonAlert(
            demon_name="grounding",
            severity=min(1.0, 1.0 - ratio),
            detail=f"{grounded}/{len(resp_words)} response words found in context ({ratio:.0%})",
            metric=ratio,
        )
    return None


def check_brevity(response: str, context: str, min_ratio: float) -> DemonAlert | None:
    """Check if response is too short relative to context richness.

    A rich context (many words) paired with a brief response suggests
    the LLM failed to engage with the material.
    """
    resp_words = len(re.findall(r"\b\w+\b", response))
    ctx_words = len(re.findall(r"\b\w+\b", context))

    if ctx_words < 20:  # Context too small to judge
        return None

    ratio = resp_words / ctx_words

    if ratio < min_ratio:
        return DemonAlert(
            demon_name="brevity",
            severity=min(1.0, 1.0 - ratio / min_ratio),
            detail=f"Response {resp_words} words vs context {ctx_words} words ({ratio:.1%})",
            metric=ratio,
        )
    return None


def check_repetition(response: str, query: str, threshold: float) -> DemonAlert | None:
    """Check if response just restates the query without adding info.

    High Jaccard overlap between query and response suggests
    the LLM parroted the question back.
    """
    resp_words = _content_words(response[:200])  # Check first 200 chars
    query_words = _content_words(query)

    if not resp_words or not query_words:
        return None

    intersection = len(resp_words & query_words)
    union = len(resp_words | query_words)

    if union == 0:
        return None

    jaccard = intersection / union

    if jaccard > threshold:
        return DemonAlert(
            demon_name="repetition",
            severity=min(1.0, (jaccard - threshold) / (1.0 - threshold)),
            detail=f"Query-response Jaccard = {jaccard:.2f} (threshold: {threshold})",
            metric=jaccard,
        )
    return None


def check_hedging(response: str, max_hedges: int) -> DemonAlert | None:
    """Check if response over-hedges instead of giving direct answers.

    Counts hedge phrases. Some hedging is normal; excessive hedging
    suggests low confidence or evasion.
    """
    response_lower = response.lower()
    hedge_count = sum(1 for phrase in _HEDGE_PHRASES if phrase in response_lower)

    if hedge_count > max_hedges:
        return DemonAlert(
            demon_name="hedging",
            severity=min(1.0, (hedge_count - max_hedges) / 5.0),
            detail=f"{hedge_count} hedge phrases detected (max: {max_hedges})",
            metric=float(hedge_count),
        )
    return None


# =============================================================================
# Demon System (KnowledgeSource)
# =============================================================================

class DemonSystem:
    """Pattern monitoring demons for response quality.

    Implements KnowledgeSource protocol.

    POST_GENERATE: After LLM generation, run all enabled demons on
    the (response, context, query) triple. Post alerts to
    blackboard.flags["demon_alerts"] for production rules / FLAG shell.

    The orchestrator can use demon alerts to decide:
      - Retry with different parameters
      - Trigger FLAG shell for manual review
      - Decrease confidence score
      - Log quality issues for training

    Usage:
        demons = DemonSystem()
        registry.register(demons)
        # Demons automatically monitor every response
    """

    def __init__(self, config: DemonConfig | None = None):
        self._config = config or DemonConfig()
        self._alert_history: list[DemonAlert] = []

    @property
    def name(self) -> str:
        return "demons"

    @property
    def phases(self) -> tuple:
        return (Phase.POST_GENERATE,)

    def activation_level(self, blackboard: Any) -> float:
        """Always active — monitoring is lightweight."""
        return 0.9

    def contribute(self, blackboard: Any, graph: Any, context: dict[str, Any]) -> None:
        """Run all enabled demons on the generated response."""
        if blackboard is None:
            return

        response = context.get("response", "")
        if not response:
            return

        # Get the formatted context that was sent to the LLM
        context_block = ""
        ctx_obj = context.get("context_block")
        if ctx_obj is not None:
            # ContextBlock has a .text or string representation
            context_block = getattr(ctx_obj, "text", str(ctx_obj))

        # If no context_block in this phase, try blackboard
        if not context_block:
            fmt = blackboard.flags.get("formatted_context", "")
            if isinstance(fmt, str):
                context_block = fmt

        query = blackboard.query

        # Run demons
        alerts: list[DemonAlert] = []

        if self._config.enable_grounding and context_block:
            alert = check_grounding(response, context_block, self._config.grounding_threshold)
            if alert:
                alerts.append(alert)

        if self._config.enable_brevity and context_block:
            alert = check_brevity(response, context_block, self._config.brevity_min_ratio)
            if alert:
                alerts.append(alert)

        if self._config.enable_repetition:
            alert = check_repetition(response, query, self._config.repetition_threshold)
            if alert:
                alerts.append(alert)

        if self._config.enable_hedging:
            alert = check_hedging(response, self._config.hedging_threshold)
            if alert:
                alerts.append(alert)

        if alerts:
            self._alert_history.extend(alerts)

            blackboard.post_flag("demon_alerts", [
                {
                    "demon": a.demon_name,
                    "severity": round(a.severity, 3),
                    "detail": a.detail,
                    "metric": round(a.metric, 3),
                }
                for a in alerts
            ])
            blackboard.post_flag("demons_fired", True)

            for a in alerts:
                logger.info(
                    "Demon '%s' fired (severity=%.2f): %s",
                    a.demon_name, a.severity, a.detail,
                )

    # =========================================================================
    # Public API
    # =========================================================================

    @property
    def alert_count(self) -> int:
        return len(self._alert_history)

    @property
    def alerts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for a in self._alert_history:
            counts[a.demon_name] = counts.get(a.demon_name, 0) + 1
        return counts

    def report(self) -> dict[str, Any]:
        """Diagnostic info about demon activity."""
        return {
            "total_alerts": len(self._alert_history),
            "alerts_by_type": self.alerts_by_type,
            "recent_alerts": [
                {
                    "demon": a.demon_name,
                    "severity": a.severity,
                    "detail": a.detail,
                }
                for a in self._alert_history[-10:]
            ],
        }
