"""
ShadowDetectorMiddleware — runs DetectorRegistry per checkpoint, logs comparison.
"""

from __future__ import annotations

import logging
from typing import Optional, List

from ..context import GovernanceContext
from ..protocol import NextFn
from hololoom.alignment.detector_registry import DetectorRegistry
from hololoom.alignment.graduation_policy import GraduationVerdict

logger = logging.getLogger(__name__)


class ShadowDetectorMiddleware:
    """Post-reasoning middleware that evaluates all registered detectors.

    For each checkpoint in the registry:
    1. Runs primary detector (result used for governance)
    2. Runs shadow detector if present (result logged, not enforced)
    3. Records comparison in AgreementTracker
    4. Checks GraduationPolicy — auto-promotes when criteria met
    """

    name: str = "shadow_detector"

    def __init__(
        self,
        registry: DetectorRegistry,
        checkpoints: Optional[List[str]] = None,
        audit_trail=None,
    ):
        self.registry = registry
        self.checkpoints = checkpoints or registry.checkpoints
        self.audit_trail = audit_trail

    async def __call__(self, ctx: GovernanceContext, next: NextFn) -> GovernanceContext:
        # Run downstream first (this is post-reasoning)
        ctx = await next(ctx)

        # Get the response text for detection
        response_text = ""
        if ctx.response and isinstance(ctx.response, dict):
            response_text = ctx.response.get("response", "")
        elif ctx.response and isinstance(ctx.response, str):
            response_text = ctx.response

        if not response_text:
            return ctx

        detection_results = {}
        for checkpoint in self.checkpoints:
            try:
                primary_result, shadow_result = self.registry.evaluate(
                    checkpoint, response_text, {"query": ctx.query}
                )
                detection_results[checkpoint] = {
                    "primary_score": primary_result.score,
                    "primary_features": primary_result.activated_features,
                }
                if shadow_result is not None:
                    detection_results[checkpoint]["shadow_score"] = shadow_result.score
                    detection_results[checkpoint]["shadow_features"] = shadow_result.activated_features

                    # Check graduation
                    verdict = self.registry.check_graduation(checkpoint)
                    if verdict == GraduationVerdict.PROMOTE:
                        self.registry.promote_shadow(checkpoint)
                        logger.info(f"Auto-promoted shadow for checkpoint '{checkpoint}'")
                        detection_results[checkpoint]["promoted"] = True

            except Exception as e:
                logger.warning(f"Shadow detector error for '{checkpoint}': {e}")

        ctx.annotations["detector_results"] = detection_results

        # Log to audit trail if available
        if self.audit_trail and detection_results:
            try:
                self.audit_trail.log_decision(
                    "shadow_detection",
                    metadata=detection_results,
                )
            except Exception:
                pass

        return ctx
