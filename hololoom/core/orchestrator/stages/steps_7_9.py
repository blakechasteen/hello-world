#!/usr/bin/env python3
"""
Weaving Pipeline Steps 7-9: Convergence, Execution, and Output
==============================================================

Part of the Elegance Pass architectural refactoring (December 2025).

Pure function implementations of the final 3 weaving steps:
- Step 7: Convergence Engine (collapse to discrete tool selection)
- Step 8: Tool Execution (with safety gating)
- Step 9: Spacetime Fabric (weave final output)

Each function follows the signature: step_n(ctx, deps...) -> ctx
- Takes WeavingContext as first parameter
- Returns WeavingContext (mutated or new)
- Has no `self` references
- Receives all dependencies as explicit parameters

Author: Claude Code (Elegance Pass - Phase 1)
Date: 2025-12-09
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

if TYPE_CHECKING:
    from hololoom.alignment.audit_trail import AuditTrail
    from hololoom.alignment.safety_guardrails import SafetyGuardrails
    from hololoom.config import Config
    from hololoom.orchestrator.context import WeavingContext
    from hololoom.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


# ============================================================================
# Step 7: Convergence Engine (Decision Collapse)
# ============================================================================

async def execute_step7_convergence(
    ctx: WeavingContext,
    cfg: Config,
    policy: Any,
    tool_executor: ToolExecutor,
    gradient_router: Any | None = None,
    emit_stage_event: Callable[[int, str, float | None], None] | None = None,
    log: logging.Logger | None = None,
) -> WeavingContext:
    """
    Step 7: Convergence Engine collapses to discrete tool selection.

    Gets neural predictions from policy, optionally blends with gradient flow,
    and collapses probability distributions to a discrete tool choice.

    Args:
        ctx: Weaving context with features and retrieval_context
        cfg: Configuration (bandit strategy, epsilon)
        policy: Policy engine for neural predictions
        tool_executor: Tool executor (for available tools list)
        gradient_router: Optional gradient flow router
        emit_stage_event: Optional callback for stage events
        log: Optional logger

    Returns:
        WeavingContext with action_plan, collapse_result, neural_probs set.

    Example:
        >>> ctx = await execute_step7_convergence(
        ...     ctx, cfg, policy, tool_executor
        ... )
        >>> print(f"Selected: {ctx.collapse_result.tool}")
        >>> print(f"Confidence: {ctx.collapse_result.confidence:.2f}")
    """
    log = log or logger
    step_start = ctx.start_timer()

    # Emit stage start event
    if emit_stage_event:
        emit_stage_event(7, "Convergence Engine", None)

    # Get neural predictions with timeout (prevent hung decisions)
    try:
        ctx.action_plan = await asyncio.wait_for(
            policy.decide(features=ctx.features, context=ctx.retrieval_context),
            timeout=0.2  # 200ms timeout
        )
    except asyncio.TimeoutError:
        log.error("Policy decision timed out after 200ms, using safe default")
        # Create safe default action plan
        from hololoom.protocols.types import ActionPlan
        ctx.action_plan = ActionPlan(
            tool="answer",
            confidence=0.5,
            tool_probs={"answer": 1.0},
            metadata={"timeout": True, "fallback": True}
        )

    # Get tool probabilities
    tools = tool_executor.tools if hasattr(tool_executor, 'tools') else ['answer', 'research', 'remember', 'clarify']
    ctx.neural_probs = np.array([
        ctx.action_plan.tool_probs.get(tool, 0.0)
        for tool in tools
    ])

    # Optional: Phase 1 Gradient Flow (physics-based tool selection)
    if gradient_router:
        try:
            ctx.gradient_decision = await gradient_router.select_tool(ctx.current_query_text)
            log.info(f"  [7a] Gradient flow suggests: {ctx.gradient_decision.target} (loss={ctx.gradient_decision.loss:.3f})")

            # Blend gradient flow with neural (70% neural, 30% gradient flow)
            for i, tool in enumerate(tools):
                if tool == ctx.gradient_decision.target:
                    ctx.neural_probs[i] = 0.7 * ctx.neural_probs[i] + 0.3
                else:
                    ctx.neural_probs[i] = 0.7 * ctx.neural_probs[i]
            # Renormalize
            ctx.neural_probs = ctx.neural_probs / ctx.neural_probs.sum()
            log.debug("  [7b] Blended neural + gradient flow probabilities")

        except Exception as e:
            log.warning(f"Gradient flow routing failed: {e}")

    # Create Convergence Engine and collapse
    from hololoom.convergence.engine import CollapseStrategy, ConvergenceEngine

    # Map bandit strategy to collapse strategy
    strategy_map = {
        'epsilon_greedy': CollapseStrategy.EPSILON_GREEDY,
        'bayesian_blend': CollapseStrategy.BAYESIAN_BLEND,
        'pure_thompson': CollapseStrategy.PURE_THOMPSON,
    }
    default_strategy = strategy_map.get(
        getattr(cfg, 'bandit_strategy', 'epsilon_greedy'),
        CollapseStrategy.EPSILON_GREEDY
    )

    ctx.convergence_engine = ConvergenceEngine(
        tools=tools,
        default_strategy=default_strategy,
        epsilon=getattr(cfg, 'epsilon', 0.1)
    )

    ctx.collapse_result = ctx.convergence_engine.collapse(ctx.neural_probs)

    # Record timing
    duration = ctx.record_timing_since('convergence', step_start)
    log.info(f"  [7] Convergence collapsed to tool: {ctx.collapse_result.tool} (confidence={ctx.collapse_result.confidence:.2f})")

    # Emit stage complete event
    if emit_stage_event:
        emit_stage_event(7, "Convergence Engine", duration)

    return ctx


# ============================================================================
# Step 8: Tool Execution (with Safety Gating)
# ============================================================================

async def execute_step8_tool_execution(
    ctx: WeavingContext,
    tool_executor: ToolExecutor,
    guardrails: SafetyGuardrails | None = None,
    audit_trail: AuditTrail | None = None,
    minimax_gate: Any | None = None,
    emit_stage_event: Callable[[int, str, float | None], None] | None = None,
    log: logging.Logger | None = None,
) -> WeavingContext:
    """
    Step 8: Tool Execution with Safety Gating + Minimax Verification.

    Gates the action through:
    1. Minimax verification (game-theoretic worst-case check) — optional
    2. Safety guardrails (alignment check) — optional
    Then executes the selected tool.

    Args:
        ctx: Weaving context with collapse_result
        tool_executor: Tool executor for running tools
        guardrails: Optional safety guardrails
        audit_trail: Optional audit trail for logging
        minimax_gate: Optional MinimaxGate for game-theoretic verification
        emit_stage_event: Optional callback for stage events
        log: Optional logger

    Returns:
        WeavingContext with tool_result, safety_decision set.
        If blocked by safety, safety_blocked=True.

    Example:
        >>> ctx = await execute_step8_tool_execution(
        ...     ctx, tool_executor, guardrails
        ... )
        >>> if ctx.safety_blocked:
        ...     print("Action blocked by safety guardrails")
        >>> else:
        ...     print(f"Result: {ctx.tool_result}")
    """
    log = log or logger
    step_start = ctx.start_timer()

    # 8pre. Minimax Verification Gate (Wire 3 — game-theoretic check)
    if minimax_gate and hasattr(ctx, 'collapse_result') and ctx.collapse_result:
        try:
            tools = tool_executor.tools if hasattr(tool_executor, 'tools') else ['answer', 'research', 'remember', 'clarify']
            n_tools = len(tools)

            # Build payoff matrix from available context
            # Rows = tools, Cols = environment states (normal, adversarial, degraded)
            tool_payoffs = np.ones((n_tools, 3)) * 0.5  # baseline
            for i, tool in enumerate(tools):
                tool_prob = ctx.neural_probs[i] if hasattr(ctx, 'neural_probs') and i < len(ctx.neural_probs) else 0.25
                confidence = ctx.collapse_result.confidence if ctx.collapse_result else 0.5
                # Normal state: payoff proportional to neural probability
                tool_payoffs[i, 0] = tool_prob * confidence
                # Adversarial state: safe tools do better
                tool_payoffs[i, 1] = 0.8 if tool in ('answer', 'clarify') else 0.2 * tool_prob
                # Degraded state: simple tools preferred
                tool_payoffs[i, 2] = 0.6 if tool in ('answer', 'remember') else 0.3 * tool_prob

            selected_idx = tools.index(ctx.collapse_result.tool) if ctx.collapse_result.tool in tools else 0
            gate_result = minimax_gate.verify(
                selected_tool=selected_idx,
                tool_payoffs=tool_payoffs,
            )
            ctx.minimax_result = gate_result

            if not gate_result.verified:
                log.info(
                    f"  [8pre] Minimax divergence: TS chose {ctx.collapse_result.tool}, "
                    f"minimax prefers {tools[gate_result.minimax_tool]} "
                    f"(regret={gate_result.regret:.3f})"
                )

        except Exception as e:
            log.debug(f"Minimax gate failed (non-blocking): {e}")

    # Emit stage start event
    if emit_stage_event:
        emit_stage_event(8, "Tool Execution", None)

    # 8a. Action Gating - Check safety before tool execution
    if guardrails:
        try:
            from hololoom.alignment.audit_trail import DecisionType, OutcomeType
            from hololoom.alignment.safety_guardrails import ActionRequest

            ctx.action_request = ActionRequest(
                action_id=str(uuid4()),
                category=_categorize_tool_to_category(ctx.collapse_result.tool),
                description=f"Execute '{ctx.collapse_result.tool}' for query: {ctx.current_query_text[:100]}",
                context={
                    "tool": ctx.collapse_result.tool,
                    "query": ctx.current_query_text,
                    "confidence": ctx.collapse_result.confidence,
                    "complexity": ctx.complexity.value if ctx.complexity else "unknown"
                }
            )

            ctx.safety_decision = guardrails.evaluate(ctx.action_request)

            # Log to audit trail
            if audit_trail:
                audit_trail.log_decision(
                    decision_id=ctx.action_request.action_id,
                    decision_type=DecisionType.TOOL_SELECTION,
                    action=ctx.collapse_result.tool,
                    outcome=OutcomeType.APPROVED if ctx.safety_decision.allowed else OutcomeType.REJECTED,
                    metadata={
                        "query": ctx.current_query_text[:200],
                        "confidence": ctx.collapse_result.confidence,
                        "risk_level": ctx.safety_decision.risk_level.value if hasattr(ctx.safety_decision.risk_level, 'value') else str(ctx.safety_decision.risk_level),
                        "safety_score": ctx.safety_decision.safety_score
                    }
                )

            if not ctx.safety_decision.allowed:
                log.warning(f"  [8] Tool execution BLOCKED by safety gating: {ctx.safety_decision.reason}")
                ctx.safety_blocked = True

                duration = ctx.record_timing_since('tool_execution', step_start)
                if emit_stage_event:
                    emit_stage_event(8, "Tool Execution (BLOCKED)", duration)

                return ctx

        except (ConnectionError, TimeoutError, OSError) as e:
            log.warning(f"Safety gating failed (service unavailable): {e}")
            ctx.add_warning(f"Safety gating error: {e}")
            # Continue with tool execution (graceful degradation)

    # 8b. Execute the selected tool
    ctx.tool_result = await tool_executor.execute(
        ctx.collapse_result.tool,
        ctx.current_query,
        ctx.retrieval_context
    )

    # Record timing
    duration = ctx.record_timing_since('tool_execution', step_start)
    log.info(f"  [8] Tool executed: {ctx.collapse_result.tool}")

    # Emit stage complete event
    if emit_stage_event:
        emit_stage_event(8, "Tool Execution", duration)

    return ctx


def _categorize_tool_to_category(tool: str) -> ActionCategory:
    """Categorize tool for safety classification, returning ActionCategory enum.

    Args:
        tool: Tool name (e.g., 'answer', 'research', 'execute')

    Returns:
        ActionCategory enum value for safety evaluation
    """
    from hololoom.alignment.safety_guardrails import ActionCategory

    safe_tools = {'answer', 'clarify', 'remember'}
    moderate_tools = {'research', 'search', 'retrieve'}
    risky_tools = {'execute', 'modify', 'delete', 'create'}

    if tool in safe_tools:
        return ActionCategory.QUERY
    elif tool in moderate_tools:
        return ActionCategory.RETRIEVAL
    elif tool in risky_tools:
        return ActionCategory.MODIFICATION
    else:
        return ActionCategory.QUERY  # Default to safe category


# ============================================================================
# Step 9: Spacetime Fabric (Weaving Results)
# ============================================================================

async def execute_step9_spacetime_fabric(
    ctx: WeavingContext,
    cfg: Config,
    semantic_cache: Any | None = None,
    dashboard_constructor: Any | None = None,
    awareness_context: dict[str, Any] | None = None,
    emit_stage_event: Callable[[int, str, float | None], None] | None = None,
    log: logging.Logger | None = None,
) -> WeavingContext:
    """
    Step 9: Spacetime Fabric - Weave final output with provenance.

    Detensions Warp Space, creates WeavingTrace with full provenance,
    and assembles the final Spacetime artifact.

    Args:
        ctx: Weaving context with all previous step results
        cfg: Configuration
        semantic_cache: Optional semantic cache for statistics
        dashboard_constructor: Optional dashboard constructor
        awareness_context: Optional awareness/consciousness context
        emit_stage_event: Optional callback for stage events
        log: Optional logger

    Returns:
        WeavingContext with spacetime and trace set.

    Example:
        >>> ctx = await execute_step9_spacetime_fabric(ctx, cfg)
        >>> print(f"Response: {ctx.spacetime.response}")
        >>> print(f"Confidence: {ctx.spacetime.confidence:.2f}")
    """
    log = log or logger
    step_start = ctx.start_timer()

    # Emit stage start event
    if emit_stage_event:
        emit_stage_event(9, "Spacetime Fabric", None)

    # Detension Warp Space
    if ctx.warp_space:
        warp_updates = ctx.warp_space.collapse()
        ctx.add_warp_operation("detension", len(warp_updates))

    # Calculate total duration
    end_time = datetime.now()
    duration_ms = ctx.total_duration_ms

    # Import fabric types
    from hololoom.fabric.spacetime import Artifact, Spacetime, WeavingTrace

    # Create WeavingTrace with full provenance
    ctx.trace = WeavingTrace(
        start_time=ctx.start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        stage_durations=ctx.stage_timings.copy(),
        motifs_detected=[
            m.pattern if hasattr(m, 'pattern') else str(m)
            for m in (ctx.features.motifs if ctx.features else [])
        ],
        embedding_scales_used=ctx.pattern_spec.scales if ctx.pattern_spec else [],
        spectral_features=ctx.features.metrics.get('spectral') if ctx.features else None,
        threads_activated=ctx.thread_ids,
        context_shards_count=len(ctx.shards),
        retrieval_mode=ctx.pattern_spec.retrieval_mode if ctx.pattern_spec else 'simple',
        policy_adapter=ctx.action_plan.adapter if ctx.action_plan else 'default',
        tool_selected=ctx.collapse_result.tool if ctx.collapse_result else 'unknown',
        tool_confidence=ctx.collapse_result.confidence if ctx.collapse_result else 0.0,
        bandit_statistics=ctx.collapse_result.bandit_stats if ctx.collapse_result else {},
        warp_operations=ctx.warp_operations,
        tensor_field_stats={"threads_tensioned": len(ctx.thread_ids)},
        errors=ctx.errors,
        warnings=ctx.warnings
    )

    # Build metadata
    metadata = {
        'pattern_card': ctx.pattern_spec.name if ctx.pattern_spec else 'unknown',
        'execution_mode': ctx.pattern_spec.card.value if ctx.pattern_spec else 'unknown',
        'loom_command': 'auto',
        'chrono_timeout': ctx.pattern_spec.pipeline_timeout if ctx.pattern_spec else 30.0
    }

    # Add semantic cache statistics if available
    if semantic_cache:
        try:
            cache_stats = semantic_cache.get_stats()
            metadata['semantic_cache'] = {
                'enabled': True,
                'hit_rate': cache_stats.get('cache_hit_rate', 0.0),
                'hot_hits': cache_stats.get('hot_hits', 0),
                'warm_hits': cache_stats.get('warm_hits', 0),
                'cold_misses': cache_stats.get('cold_misses', 0),
                'estimated_speedup': cache_stats.get('estimated_speedup', 1.0)
            }
        except Exception:
            metadata['semantic_cache'] = {'enabled': False}
    else:
        metadata['semantic_cache'] = {'enabled': False}

    # Inject awareness context if available
    if awareness_context or ctx.awareness_context:
        ac = awareness_context or ctx.awareness_context
        metadata['awareness'] = {
            'activation_level': ac.get('activation_level', 0.0),
            'coherence': ac.get('coherence', 0.0),
            'active_nodes': ac.get('active_nodes', 0),
            'shift_detected': ac.get('shift_detected', False),
            'semantic_position': ac.get('semantic_position'),
            'perception_time_ms': ac.get('perception_time_ms', 0.0),
        }
        log.info(
            f"[AWARENESS] Epistemic context added to Spacetime: "
            f"activation={ac.get('activation_level', 0.0):.3f}, "
            f"coherence={ac.get('coherence', 0.0):.3f}"
        )

    # Handle safety blocked case
    if ctx.safety_blocked:
        metadata['safety_blocked'] = True
        if ctx.safety_decision:
            metadata['risk_level'] = ctx.safety_decision.risk_level.value if hasattr(ctx.safety_decision.risk_level, 'value') else str(ctx.safety_decision.risk_level)
            metadata['safety_reason'] = ctx.safety_decision.reason
            metadata['safety_score'] = ctx.safety_decision.safety_score

        ctx.spacetime = Spacetime(
            query_text=ctx.current_query_text,
            response=f"[Action blocked by safety guardrails: {ctx.safety_decision.reason if ctx.safety_decision else 'unknown'}]",
            confidence=0.0,
            tool_used=ctx.collapse_result.tool if ctx.collapse_result else 'unknown',
            trace=ctx.trace,
            artifacts=[],
            metadata=metadata,
            context_summary=f"{len(ctx.shards)} shards",
            sources_used=[s.id for s in ctx.shards[:3]] if ctx.shards else []
        )
    else:
        # Parse artifacts from tool result
        raw_artifacts = ctx.tool_result.get('artifacts', []) if ctx.tool_result else []
        artifacts = [Artifact.from_dict(a) for a in raw_artifacts]

        ctx.spacetime = Spacetime(
            query_text=ctx.current_query_text,
            response=ctx.tool_result.get('result', 'No response') if ctx.tool_result else 'No response',
            tool_used=ctx.collapse_result.tool if ctx.collapse_result else 'unknown',
            confidence=ctx.collapse_result.confidence if ctx.collapse_result else 0.0,
            artifacts=artifacts,
            trace=ctx.trace,
            metadata=metadata,
            context_summary=f"{len(ctx.shards)} shards",
            sources_used=[s.id for s in ctx.shards[:3]] if ctx.shards else []
        )

    # Attach complexity and provenance if available
    if ctx.complexity:
        ctx.spacetime.complexity = ctx.complexity
    if ctx.provenance:
        ctx.spacetime.provenance = ctx.provenance
        # Add stage timings as protocol calls
        for stage, timing_ms in ctx.stage_timings.items():
            ctx.spacetime.provenance.add_protocol_call(
                protocol='weaving_orchestrator',
                method=stage,
                duration_ms=timing_ms,
                result_summary=f"Stage completed in {timing_ms:.1f}ms"
            )
        # Add final shuttle event
        ctx.spacetime.provenance.add_shuttle_event("weaving_complete", {
            'tool': ctx.collapse_result.tool if ctx.collapse_result else 'unknown',
            'confidence': ctx.collapse_result.confidence if ctx.collapse_result else 0.0,
            'duration_ms': duration_ms
        })

    # Generate dashboard if enabled
    if dashboard_constructor:
        try:
            dashboard = dashboard_constructor.construct(ctx.spacetime)
            ctx.spacetime.metadata['dashboard'] = dashboard
            log.info(f"[DASHBOARD] Generated {len(dashboard.panels)} panels ({dashboard.layout.value} layout)")
        except Exception as e:
            log.warning(f"[DASHBOARD] Failed to generate dashboard: {e}")

    # Record timing
    duration = ctx.record_timing_since('spacetime_assembly', step_start)
    log.info("  [9] Spacetime fabric woven!")

    # Emit stage complete event
    if emit_stage_event:
        emit_stage_event(9, "Spacetime Fabric", duration)

    # Mark pipeline complete
    if emit_stage_event:
        emit_stage_event(0, "Complete", None)

    return ctx


__all__ = [
    'execute_step7_convergence',
    'execute_step8_tool_execution',
    'execute_step9_spacetime_fabric',
]
