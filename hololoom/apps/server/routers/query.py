"""
Query Router
=============
Main /query endpoint for agentic reasoning.

Extracted from agentic_api.py (March 2026 Refactor).
"""

import logging
from datetime import datetime
from time import time
from typing import Optional, List, Dict

from fastapi import APIRouter, HTTPException

from hololoom.agentic import create_agentic_orchestrator, ReasoningMode, AgenticResult
from hololoom.protocols.types import Query
from hololoom.alignment.audit_trail import DecisionType, OutcomeType
from hololoom.alignment.safety_guardrails import ActionRequest
from hololoom.alignment.deception_detection import BehavioralProbe, ProbeType
from hololoom.agentic.safety_adapter import create_safety_adapter

from hololoom.apps.server.schemas import (
    QueryRequest, VerificationResponse, ReasoningStepResponse, AgenticResponse,
)
from hololoom.apps.server.server_state import state

# Lazy imports for optional governance
try:
    from hololoom.agents.policy_governance import PolicyDecision
    GOVERNANCE_AVAILABLE = True
except ImportError:
    GOVERNANCE_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


# ============================================================================
# Helpers
# ============================================================================

async def get_orchestrator():
    """Get or create global orchestrator instance."""
    if state.orchestrator is None:
        logger.info("Initializing agentic orchestrator...")
        state.orchestrator = await create_agentic_orchestrator(
            state.config,
            state.shards,
            enable_verification=True,
            enable_goal_tracking=True,
            audit_trail=state.audit_trail,
            monitor=state.monitor,
            safety_adapter=create_safety_adapter(state.safety_guardrails) if state.safety_guardrails else None,
            deception_detector=state.deception_detector,
        )
        logger.info("Orchestrator ready!")

    return state.orchestrator


def _format_verification(verification) -> Optional[VerificationResponse]:
    """Format verification result for API response."""
    if verification is None:
        return None

    return VerificationResponse(
        verified=verification.verified,
        confidence=verification.confidence,
        contradictions=verification.contradictions,
        supporting_evidence=verification.supporting_evidence,
        suggested_refinements=verification.suggested_refinements
    )


def _format_steps(steps: List[Dict]) -> List[ReasoningStepResponse]:
    """Format reasoning steps for API response."""
    return [
        ReasoningStepResponse(
            type=step.get("type", "unknown"),
            query=step.get("query"),
            confidence=step.get("confidence"),
            finding=step.get("finding"),
            completed=step.get("completed"),
            tool=step.get("tool_used") or step.get("tool")
        )
        for step in steps
    ]


# ============================================================================
# Bus Signal Helpers
# ============================================================================

async def _bus_emit_query(text: str, mode: str, correlation_id: str) -> None:
    """Emit QUERY signal to bus. Non-blocking, non-fatal."""
    try:
        from hololoom.apps.server.bus_setup import get_bus
        bus = get_bus()
        if bus is None:
            return
        from hololoom.core.bus import Signal, SignalKind, SignalPriority
        await bus.emit(Signal(
            kind=SignalKind.QUERY,
            priority=SignalPriority.NORMAL,
            source="agentic_api",
            payload={"text": text[:500], "mode": mode, "speed_weight": 0.5},
            correlation_id=correlation_id,
        ))
    except Exception:
        pass


async def _bus_emit_result(
    correlation_id: str, success: bool, confidence: float, duration_ms: float,
    mode: str, steps: int, error: str = "",
) -> None:
    """Emit EXECUTION_COMPLETE/FAILED to bus. Strategy selector learns from this."""
    try:
        from hololoom.apps.server.bus_setup import get_bus
        bus = get_bus()
        if bus is None:
            return
        from hololoom.core.bus import Signal, SignalKind, SignalPriority
        kind = SignalKind.EXECUTION_COMPLETE if success else SignalKind.EXECUTION_FAILED
        await bus.emit(Signal(
            kind=kind,
            priority=SignalPriority.NORMAL,
            source="agentic_api",
            payload={
                "success": success,
                "confidence": confidence,
                "duration_ms": duration_ms,
                "mode": mode,
                "steps_executed": steps,
                "error": error,
            },
            correlation_id=correlation_id,
        ))
    except Exception:
        pass


# ============================================================================
# Endpoint
# ============================================================================

@router.post("/query", response_model=AgenticResponse)
async def query_endpoint(request: QueryRequest):
    """Main query endpoint for agentic reasoning."""
    start_time = datetime.now()
    start_time_ms = time() * 1000

    try:
        text_value = (request.text or "").strip()
        if not text_value:
            raise HTTPException(status_code=400, detail="Query text must not be empty.")

        mode_map = {
            "direct": ReasoningMode.DIRECT,
            "verify": ReasoningMode.VERIFY,
            "research": ReasoningMode.RESEARCH,
            "plan_execute": ReasoningMode.PLAN_EXECUTE,
        }
        mode_key = (request.mode or "verify").strip().lower()
        if mode_key not in mode_map:
            raise HTTPException(status_code=400, detail=f"Unsupported reasoning mode: {request.mode}")

        orchestrator = await get_orchestrator()
        mode = mode_map[mode_key]
        query = Query(text=text_value)

        if request.context:
            query.metadata = {
                "code_context": request.context.dict(),
                "language": request.context.languageId,
                "file": request.context.fileName,
                "selection": request.context.selection,
                "workspace": request.context.workspace,
            }

        # RBAC / GOVERNANCE CHECK
        if state.governance_engine and GOVERNANCE_AVAILABLE:
            from hololoom.agents.policy_governance import (
                CommunicationRequest as GovRequest, Priority as GovPriority
            )
            gov_request = GovRequest(
                from_agent=getattr(request, 'agent_id', 'anonymous'),
                to_agent="hololoom_server",
                message_type="query",
                topic=mode_key,
                content=text_value,
                priority=GovPriority.MEDIUM,
                metadata={"mode": mode_key, "max_steps": request.max_steps},
            )
            gov_decision, gov_reason = state.governance_engine.evaluate_communication_request(
                gov_request
            )

            if gov_decision == PolicyDecision.DENY:
                logger.warning(f"RBAC DENY: {gov_reason}")
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "governance_denied",
                        "reason": gov_reason,
                        "message": f"Request denied by governance policy: {gov_reason}",
                    }
                )
            elif gov_decision == PolicyDecision.ESCALATE:
                logger.info(f"RBAC ESCALATE: {gov_reason} (proceeding with audit)")

        # SAFETY GATING
        if state.safety_guardrails:
            from hololoom.alignment.safety_guardrails import ActionCategory

            action_request = ActionRequest(
                action="code_analysis" if request.context else "text_query",
                category=ActionCategory.QUERY if not request.context else ActionCategory.CODE_EXECUTION,
                context={
                    "query": text_value,
                    "mode": request.mode,
                    "max_steps": request.max_steps,
                    "has_code_context": request.context is not None,
                    "source": "vscode_extension",
                    "timestamp": start_time.isoformat()
                }
            )

            gate_result = state.safety_guardrails.evaluate(action_request)

            logger.info(f"Safety Gate: {gate_result.risk_level.value} risk "
                       f"(score={gate_result.safety_score:.2f}, allowed={gate_result.allowed})")

            if not gate_result.allowed:
                error_msg = (
                    f"Query blocked by safety guardrails: {gate_result.reason}. "
                    f"Risk level: {gate_result.risk_level.value} "
                    f"(safety score: {gate_result.safety_score:.2f})"
                )
                logger.warning(error_msg)
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "safety_guardrail_blocked",
                        "reason": gate_result.reason,
                        "risk_level": gate_result.risk_level.value,
                        "safety_score": gate_result.safety_score,
                        "message": error_msg
                    }
                )

            if not query.metadata:
                query.metadata = {}
            query.metadata["safety_evaluation"] = {
                "risk_level": gate_result.risk_level.value,
                "safety_score": gate_result.safety_score,
                "allowed": gate_result.allowed
            }

        # Notify bus
        correlation_id = f"query-{start_time.isoformat()}"
        await _bus_emit_query(text_value, request.mode, correlation_id)

        # Run agentic reasoning
        logger.info(f"Query: {request.text[:100]}... (mode={request.mode})")
        result: AgenticResult = await orchestrator.reason(
            query,
            mode=mode,
            max_steps=request.max_steps
        )

        response_text = result.spacetime.metadata.get("response", "")
        if not response_text:
            response_text = f"Processed query with {result.reasoning_mode.value} mode."

        end_time_ms = time() * 1000
        latency_ms = end_time_ms - start_time_ms

        if state.stats:
            state.stats.record_query(request.mode, latency_ms, success=True)

        await _bus_emit_result(
            correlation_id=correlation_id,
            success=True,
            confidence=result.spacetime.confidence,
            duration_ms=latency_ms,
            mode=request.mode,
            steps=len(result.steps_taken),
        )

        # AUDIT TRAIL LOGGING
        if state.audit_trail:
            try:
                state.audit_trail.log_decision(
                    decision_type=DecisionType.TOOL_SELECTION,
                    outcome=OutcomeType.APPROVED,
                    reason=f"Agentic reasoning completed successfully in {request.mode} mode",
                    query_text=text_value,
                    action_description=f"agentic_reasoning_{request.mode}",
                    risk_level=query.metadata.get("safety_evaluation", {}).get("risk_level") if query.metadata else None,
                    confidence=result.spacetime.confidence,
                    metadata={
                        "code_context": request.context.dict() if request.context else None,
                        "max_steps": request.max_steps,
                        "reasoning_mode": result.reasoning_mode.value,
                        "steps_taken": len(result.steps_taken),
                        "total_queries": result.total_queries,
                        "timestamp": start_time.isoformat(),
                        "query_id": result.spacetime.query_id,
                        "latency_ms": latency_ms,
                        "verification": result.verification is not None,
                        "safety_score": query.metadata.get("safety_evaluation", {}).get("safety_score") if query.metadata else None
                    }
                )
                logger.debug(f"Logged to audit trail: query_id={result.spacetime.query_id}")
            except Exception as e:
                logger.error(f"Failed to log to audit trail: {e}")

        # DECEPTION MONITORING
        if state.deception_detector:
            try:
                probe = BehavioralProbe(
                    probe_type=ProbeType.HONESTY,
                    scenario=f"Response to query: {text_value[:100]}",
                    expected_behavior=text_value,
                )
                _passed, deception_score = state.deception_detector.run_probe(
                    probe, actual_behavior=response_text
                )
                if state.audit_trail:
                    state.audit_trail.log_decision(
                        decision_type=DecisionType.DECEPTION_CHECK,
                        outcome=OutcomeType.APPROVED,
                        reason=f"Deception monitoring (Jaccard score={deception_score:.2f})",
                        query_text=text_value,
                        action_description="deception_monitoring",
                        confidence=1.0 - deception_score,
                        metadata={"deception_score": deception_score},
                    )
            except Exception as e:
                logger.debug(f"Deception monitoring skipped: {e}")

        response_obj = AgenticResponse(
            response=response_text,
            confidence=result.spacetime.confidence,
            reasoning_mode=result.reasoning_mode.value,
            steps_taken=_format_steps(result.steps_taken),
            total_queries=result.total_queries,
            total_duration_ms=result.total_duration_ms,
            verification=_format_verification(result.verification),
            timestamp=start_time.isoformat(),
            query_id=result.spacetime.query_id
        )

        if query.metadata and "safety_evaluation" in query.metadata:
            logger.info(f"Safety: {query.metadata['safety_evaluation']}")

        return response_obj

    except HTTPException as e:
        if state.stats:
            state.stats.record_query(request.mode, 0, success=False)
            state.stats.record_error(f"HTTP_{e.status_code}")
        raise

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)

        if state.stats:
            latency_ms = (time() * 1000) - start_time_ms
            state.stats.record_query(request.mode, latency_ms, success=False)
            state.stats.record_error(type(e).__name__)

        await _bus_emit_result(
            correlation_id=correlation_id,
            success=False, confidence=0.0,
            duration_ms=(time() * 1000) - start_time_ms,
            mode=request.mode, steps=0, error=str(e),
        )

        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}. Please try again."
        )


# ============================================================================
# v2 Endpoint — Governance Pipeline
# ============================================================================

@router.post("/v2/query", response_model=AgenticResponse)
async def query_v2(request: QueryRequest):
    """Governance pipeline endpoint — composable middleware chain.

    Functionally equivalent to /query but routes through the
    GovernancePipeline (RBAC -> Safety -> Reasoning -> Audit -> Deception)
    instead of inline procedural checks.
    """
    from hololoom.apps.server.governance import GovernanceContext

    if state.governance_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Governance pipeline not initialized",
        )

    text_value = (request.text or "").strip()
    if not text_value:
        raise HTTPException(status_code=400, detail="Query text must not be empty.")

    mode_key = (request.mode or "verify").strip().lower()
    valid_modes = {"direct", "verify", "research", "plan_execute"}
    if mode_key not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Unsupported reasoning mode: {request.mode}")

    start_time = datetime.now()

    ctx = GovernanceContext(
        query=text_value,
        mode=mode_key,
        request_metadata={
            "source": "api",
            "max_steps": request.max_steps,
            "has_code_context": request.context is not None,
            "code_context": request.context.dict() if request.context else None,
            "agent_id": getattr(request, "agent_id", "anonymous"),
            "timestamp": start_time.isoformat(),
        },
    )

    correlation_id = f"query-v2-{ctx.correlation_id}"

    # Bus notification (non-blocking)
    await _bus_emit_query(text_value, mode_key, correlation_id)

    try:
        ctx = await state.governance_pipeline.execute(ctx)
    except HTTPException:
        # RBAC / safety gates raise HTTPException directly — let them through
        if state.stats:
            state.stats.record_query(request.mode, 0, success=False)
        raise
    except Exception as e:
        logger.error("Governance pipeline failed: %s", e, exc_info=True)
        if state.stats:
            state.stats.record_query(request.mode, ctx.elapsed_ms, success=False)
            state.stats.record_error(type(e).__name__)
        await _bus_emit_result(
            correlation_id=correlation_id,
            success=False, confidence=0.0,
            duration_ms=ctx.elapsed_ms,
            mode=mode_key, steps=0, error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}. Please try again.",
        )

    # Build response from context
    result = ctx.reasoning_result
    if result is None:
        raise HTTPException(status_code=500, detail="Pipeline completed but no reasoning result produced.")

    if state.stats:
        state.stats.record_query(request.mode, ctx.elapsed_ms, success=True)

    await _bus_emit_result(
        correlation_id=correlation_id,
        success=True,
        confidence=result.spacetime.confidence,
        duration_ms=ctx.elapsed_ms,
        mode=mode_key,
        steps=len(result.steps_taken),
    )

    return AgenticResponse(
        response=ctx.response or "",
        confidence=result.spacetime.confidence,
        reasoning_mode=result.reasoning_mode.value,
        steps_taken=_format_steps(result.steps_taken),
        total_queries=result.total_queries,
        total_duration_ms=result.total_duration_ms,
        verification=_format_verification(result.verification),
        timestamp=start_time.isoformat(),
        query_id=result.spacetime.query_id,
    )
