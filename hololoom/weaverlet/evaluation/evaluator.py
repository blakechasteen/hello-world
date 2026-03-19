"""
Weaverlet Evaluator
====================

Composes two evaluation sources:

1. FabricInspector (signal-based): runs registered signals in parallel,
   produces a weighted confidence score. Reuses the full Tapestry signal
   pipeline without modification.

2. Model self-evaluation: asks the inference model to score and critique
   its own work. Parses a numeric score and textual critique from the
   response.

The final score is a weighted blend:
    score = signal_weight * inspector_score + self_eval_weight * model_score

Graceful degradation: if either source is unavailable, the available one
gets full weight. If neither is available, returns (0.5, "no evaluator").
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from hololoom.tapestry.inspector import FabricInspector
from hololoom.tapestry.protocol import Thread, ThreadStatus
from hololoom.weaverlet.schemas import WorkingState
from hololoom.weaverlet.shuttle.http_client import DaemonInferClient

logger = logging.getLogger(__name__)

SELF_EVAL_PROMPT = """You are evaluating progress on a task.

Goal: {goal}

Current artifact:
{artifact_summary}

Score history: {score_history}
Iteration: {iteration}

Rate the current progress on a scale of 0.0 to 1.0, and provide a brief critique
identifying what's working and what needs improvement.

Respond in this exact format:
SCORE: <number between 0.0 and 1.0>
CRITIQUE: <your assessment>
"""


class WeaverletEvaluator:

    def __init__(
        self,
        inspector: FabricInspector | None = None,
        infer_client: DaemonInferClient | None = None,
        signal_weight: float = 0.6,
        self_eval_weight: float = 0.4,
    ):
        self.inspector = inspector
        self.infer_client = infer_client
        self.signal_weight = signal_weight
        self.self_eval_weight = self_eval_weight

    async def evaluate(
        self,
        state: WorkingState,
        context: dict[str, Any] | None = None,
    ) -> tuple[float, str]:
        """
        Returns (composite_score, critique_text).

        Blends signal-based and model-based evaluation with graceful fallback.
        """
        context = context or {}
        signal_score: float | None = None
        model_score: float | None = None
        critique = ""

        # Signal-based evaluation via FabricInspector
        if self.inspector:
            try:
                signal_score, signal_critique = await self._run_signal_eval(
                    state, context
                )
                critique = signal_critique
            except Exception as e:
                logger.warning(f"Signal evaluation failed: {e}")

        # Model self-evaluation via daemon /infer
        if self.infer_client:
            try:
                model_score, model_critique = await self._run_model_eval(state)
                if critique:
                    critique = f"{critique}\n\n{model_critique}"
                else:
                    critique = model_critique
            except Exception as e:
                logger.warning(f"Model self-evaluation failed: {e}")

        # Blend scores
        score = self._blend(signal_score, model_score)

        if not critique:
            critique = "No evaluation available."

        return score, critique

    async def _run_signal_eval(
        self, state: WorkingState, context: dict[str, Any]
    ) -> tuple[float, str]:
        """Run FabricInspector on a synthetic thread."""
        thread = Thread(
            id=f"wvlt_iter_{state.iteration}",
            description=state.goal,
            status=ThreadStatus.WEAVING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        result = await self.inspector.inspect(thread, context)
        critique_parts = []
        if result.blockers:
            critique_parts.append(f"Blockers: {', '.join(result.blockers)}")
        if result.recommendations:
            critique_parts.append(
                "Recommendations:\n" +
                "\n".join(f"  - {r}" for r in result.recommendations)
            )
        return result.confidence, "\n".join(critique_parts) or "Signals passed."

    async def _run_model_eval(
        self, state: WorkingState
    ) -> tuple[float, str]:
        """Ask the model to self-evaluate via daemon /infer."""
        artifact_summary = ""
        if state.artifact:
            content = state.artifact.content
            if isinstance(content, str):
                artifact_summary = content[:2000]
            elif content is not None:
                artifact_summary = str(content)[:2000]
            else:
                artifact_summary = f"[{state.artifact.type.value}: {state.artifact.name}]"

        prompt = SELF_EVAL_PROMPT.format(
            goal=state.goal,
            artifact_summary=artifact_summary or "(no artifact yet)",
            score_history=state.score_history[-5:],
            iteration=state.iteration,
        )

        resp = await self.infer_client.infer(prompt, max_tokens=4096, temperature=0.3)

        # Charge tokens to ledger if available
        if state.ledger:
            state.ledger.charge_tokens(resp.tokens_used)

        return self._parse_eval_response(resp.content)

    def _parse_eval_response(self, text: str) -> tuple[float, str]:
        """Parse SCORE: and CRITIQUE: from model response."""
        score = 0.5
        critique = text

        score_match = re.search(r"SCORE:\s*([\d.]+)", text)
        if score_match:
            try:
                score = max(0.0, min(1.0, float(score_match.group(1))))
            except ValueError:
                pass

        critique_match = re.search(r"CRITIQUE:\s*(.+)", text, re.DOTALL)
        if critique_match:
            critique = critique_match.group(1).strip()

        return score, critique

    def _blend(
        self,
        signal_score: float | None,
        model_score: float | None,
    ) -> float:
        """Weighted blend with graceful degradation."""
        if signal_score is not None and model_score is not None:
            return (
                self.signal_weight * signal_score
                + self.self_eval_weight * model_score
            )
        if signal_score is not None:
            return signal_score
        if model_score is not None:
            return model_score
        return 0.5
