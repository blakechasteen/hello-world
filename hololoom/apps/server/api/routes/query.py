"""
Query Route
===========

Main query endpoint for agentic reasoning.

Endpoints:
    POST /query - Execute agentic reasoning query
"""

import logging
from datetime import datetime
from time import time

from fastapi import APIRouter, HTTPException

from hololoom.agentic import ReasoningMode, AgenticResult
from hololoom.alignment.safety_guardrails import ActionRequest, ActionCategory
from hololoom.alignment.audit_trail import DecisionType, OutcomeType
from hololoom.protocols.types import Query

from ..models import QueryRequest, AgenticResponse
from ..state import state
from ..services import get_orchestrator, format_verification, format_steps

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Query"])


@router.post("/query", response_model=AgenticResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint for agentic reasoning.

    Matches VS Code extension's HoloLoomBridge.query() expectations.

    Args:
        request: QueryRequest with text, context, mode, max_steps

    Returns:
        AgenticResponse with reasoning results

    Example:
        POST /query
        {
          "text": "Explain this TypeScript code",
          "context": {
            "languageId": "typescript",
            "fileName": "example.ts",
            "selection": "function foo() { ... }"
          },
          "mode": "verify",
          "max_steps": 5
        }
    """
    start_time = datetime.now()
    start_time_ms = time() * 1000

    try:
        # Validate request
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

        # Get orchestrator
        orchestrator = await get_orchestrator()
        mode = mode_map[mode_key]

        # Create query
        query = Query(text=text_value)

        # Add code context to metadata
        if request.context:
            query.metadata = {
                "code_context": request.context.dict(),
                "language": request.context.languageId,
                "file": request.context.fileName,
                "selection": request.context.selection,
                "workspace": request.context.workspace,
            }

        # Safety gating
        if state.safety_guardrails:
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

        # Run agentic reasoning
        logger.info(f"Query: {request.text[:100]}... (mode={request.mode})")
        result: AgenticResult = await orchestrator.reason(
            query,
            mode=mode,
            max_steps=request.max_steps
        )

        # Extract response
        response_text = result.spacetime.metadata.get("response", "")
        if not response_text:
            response_text = f"Processed query with {result.reasoning_mode.value} mode."

        # Calculate latency
        end_time_ms = time() * 1000
        latency_ms = end_time_ms - start_time_ms

        # Track stats
        if state.stats:
            state.stats.record_query(request.mode, latency_ms, success=True)

        # Audit trail logging
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
            except Exception as e:
                logger.error(f"Failed to log to audit trail: {e}")

        # Format response
        response_obj = AgenticResponse(
            response=response_text,
            confidence=result.spacetime.confidence,
            reasoning_mode=result.reasoning_mode.value,
            steps_taken=format_steps(result.steps_taken),
            total_queries=result.total_queries,
            total_duration_ms=result.total_duration_ms,
            verification=format_verification(result.verification),
            timestamp=start_time.isoformat(),
            query_id=result.spacetime.query_id
        )

        return response_obj

    except HTTPException:
        if state.stats:
            state.stats.record_query(request.mode, 0, success=False)
        raise

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        if state.stats:
            latency_ms = (time() * 1000) - start_time_ms
            state.stats.record_query(request.mode, latency_ms, success=False)
            state.stats.record_error(type(e).__name__)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}. Please try again."
        )
