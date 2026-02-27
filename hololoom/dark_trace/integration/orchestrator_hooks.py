"""
Dark Trace Orchestrator Hooks

Deep integration with HoloLoom's WeavingOrchestrator through middleware
and hooks that inject interpretability analysis at key points in the
weaving cycle.

Author: HoloLoom Team
Created: December 2025
"""

from typing import Any, Dict, List, Optional, Callable, Awaitable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio

import numpy as np


class IntegrationMode(Enum):
    """Modes of Dark Trace integration with orchestrator."""
    DISABLED = "disabled"           # No integration
    PASSIVE = "passive"             # Monitor only, no intervention
    ADVISORY = "advisory"           # Provide recommendations
    ACTIVE = "active"               # Can modify responses
    STRICT = "strict"               # Block unsafe responses


class HookPoint(Enum):
    """Points in weaving cycle where hooks can be injected."""
    PRE_WEAVE = "pre_weave"                 # Before weaving starts
    POST_RETRIEVAL = "post_retrieval"       # After memory retrieval
    PRE_DECISION = "pre_decision"           # Before policy decision
    POST_DECISION = "post_decision"         # After policy decision
    PRE_EXECUTION = "pre_execution"         # Before tool execution
    POST_EXECUTION = "post_execution"       # After tool execution
    POST_WEAVE = "post_weave"               # After weaving completes


@dataclass
class HookContext:
    """Context passed to hooks during weaving."""
    query_text: str
    query_embedding: Optional[np.ndarray] = None
    retrieved_memories: Optional[List[Any]] = None
    activations: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    decision_probabilities: Optional[np.ndarray] = None
    selected_tool: Optional[str] = None
    tool_result: Optional[Any] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookResult:
    """Result returned by a hook."""
    allow: bool = True                      # Whether to continue
    modified_context: Optional[HookContext] = None  # Modified context
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Hook type definitions
PreWeaveHook = Callable[[HookContext], Awaitable[HookResult]]
PostWeaveHook = Callable[[HookContext, Any], Awaitable[HookResult]]
DecisionHook = Callable[[HookContext, np.ndarray], Awaitable[HookResult]]


@dataclass
class MiddlewareConfig:
    """Configuration for Dark Trace middleware."""
    mode: IntegrationMode = IntegrationMode.PASSIVE

    # Safety thresholds
    block_threshold: float = 0.8            # Block if concern severity >= this
    warn_threshold: float = 0.5             # Warn if concern severity >= this

    # Analysis options
    analyze_retrievals: bool = True         # Analyze retrieved memories
    analyze_decisions: bool = True          # Analyze decision patterns
    analyze_outputs: bool = True            # Analyze final outputs

    # Feature monitoring
    monitored_features: List[str] = field(default_factory=list)
    blocked_features: List[str] = field(default_factory=list)

    # Callbacks
    on_warning: Optional[Callable[[str, Dict], Awaitable[None]]] = None
    on_block: Optional[Callable[[str, Dict], Awaitable[None]]] = None
    on_analysis_complete: Optional[Callable[[Dict], Awaitable[None]]] = None

    # Performance
    enable_caching: bool = True
    max_analysis_time_ms: float = 50.0      # Max time for analysis


class DarkTraceMiddleware:
    """
    Middleware that wraps the weaving cycle with interpretability analysis.

    Provides automatic analysis at each stage of weaving and can block,
    modify, or warn based on detected patterns.
    """

    def __init__(
        self,
        config: Optional[MiddlewareConfig] = None,
        safety_gate: Optional[Any] = None,  # DarkTraceSafetyGate
        engine: Optional[Any] = None,       # DarkTraceEngine
    ):
        self.config = config or MiddlewareConfig()
        self.safety_gate = safety_gate
        self.engine = engine

        self._hooks: Dict[HookPoint, List[Callable]] = {
            point: [] for point in HookPoint
        }
        self._analysis_cache: Dict[str, Any] = {}
        self._statistics = {
            "total_weaves": 0,
            "blocked_weaves": 0,
            "warnings_issued": 0,
            "avg_analysis_time_ms": 0.0,
        }

    def register_hook(
        self,
        point: HookPoint,
        hook: Callable
    ) -> None:
        """Register a hook at a specific point in the weaving cycle."""
        self._hooks[point].append(hook)

    def unregister_hook(
        self,
        point: HookPoint,
        hook: Callable
    ) -> bool:
        """Unregister a hook. Returns True if found and removed."""
        try:
            self._hooks[point].remove(hook)
            return True
        except ValueError:
            return False

    async def _run_hooks(
        self,
        point: HookPoint,
        context: HookContext,
        extra_args: tuple = ()
    ) -> HookResult:
        """Run all hooks at a specific point."""
        combined_result = HookResult()

        for hook in self._hooks[point]:
            try:
                if extra_args:
                    result = await hook(context, *extra_args)
                else:
                    result = await hook(context)

                # Combine results
                if not result.allow:
                    combined_result.allow = False
                combined_result.warnings.extend(result.warnings)
                combined_result.recommendations.extend(result.recommendations)
                combined_result.metadata.update(result.metadata)

                # Use modified context if provided
                if result.modified_context:
                    context = result.modified_context
                    combined_result.modified_context = context

            except Exception as e:
                combined_result.warnings.append(f"Hook error: {str(e)}")

        return combined_result

    async def pre_weave(self, context: HookContext) -> HookResult:
        """
        Called before weaving starts.

        Can block the weave before any processing occurs.
        """
        if self.config.mode == IntegrationMode.DISABLED:
            return HookResult()

        self._statistics["total_weaves"] += 1

        # Run registered pre-weave hooks
        result = await self._run_hooks(HookPoint.PRE_WEAVE, context)

        return result

    async def post_retrieval(
        self,
        context: HookContext,
        memories: List[Any]
    ) -> HookResult:
        """
        Called after memory retrieval.

        Can analyze and filter retrieved memories.
        """
        if self.config.mode == IntegrationMode.DISABLED:
            return HookResult()

        if not self.config.analyze_retrievals:
            return HookResult()

        context.retrieved_memories = memories

        # Run registered post-retrieval hooks
        result = await self._run_hooks(
            HookPoint.POST_RETRIEVAL,
            context,
            (memories,)
        )

        return result

    async def pre_decision(
        self,
        context: HookContext,
        activations: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> HookResult:
        """
        Called before policy decision.

        This is a key integration point for safety analysis.
        """
        if self.config.mode == IntegrationMode.DISABLED:
            return HookResult()

        if not self.config.analyze_decisions:
            return HookResult()

        context.activations = activations
        context.feature_names = feature_names

        result = HookResult()

        # Use safety gate if available
        if self.safety_gate and self.config.mode in [
            IntegrationMode.ACTIVE,
            IntegrationMode.STRICT
        ]:
            gate_decision = await self.safety_gate.evaluate(
                activations=activations,
                confidence=context.confidence,
                feature_names=feature_names,
                context=context.metadata,
            )

            if gate_decision.should_block:
                result.allow = False
                result.metadata["blocked_reason"] = gate_decision.blocked_reason
                result.metadata["concerns"] = [
                    c.to_dict() for c in gate_decision.concerns
                ]

                self._statistics["blocked_weaves"] += 1

                if self.config.on_block:
                    await self.config.on_block(
                        gate_decision.blocked_reason or "Unknown",
                        {"decision": gate_decision.to_dict()}
                    )

            elif gate_decision.has_concerns:
                for concern in gate_decision.concerns:
                    if concern.severity >= self.config.warn_threshold:
                        result.warnings.append(concern.description)
                        self._statistics["warnings_issued"] += 1

                        if self.config.on_warning:
                            await self.config.on_warning(
                                concern.description,
                                {"concern": concern.to_dict()}
                            )

        # Run registered pre-decision hooks
        hook_result = await self._run_hooks(
            HookPoint.PRE_DECISION,
            context,
            (activations,)
        )

        # Combine results
        if not hook_result.allow:
            result.allow = False
        result.warnings.extend(hook_result.warnings)
        result.recommendations.extend(hook_result.recommendations)
        result.metadata.update(hook_result.metadata)

        return result

    async def post_decision(
        self,
        context: HookContext,
        probabilities: np.ndarray,
        selected_tool: str
    ) -> HookResult:
        """
        Called after policy decision.

        Can analyze and potentially override tool selection.
        """
        if self.config.mode == IntegrationMode.DISABLED:
            return HookResult()

        context.decision_probabilities = probabilities
        context.selected_tool = selected_tool

        # Run registered post-decision hooks
        result = await self._run_hooks(
            HookPoint.POST_DECISION,
            context,
            (probabilities, selected_tool)
        )

        return result

    async def post_weave(
        self,
        context: HookContext,
        spacetime: Any
    ) -> HookResult:
        """
        Called after weaving completes.

        Final analysis and logging.
        """
        if self.config.mode == IntegrationMode.DISABLED:
            return HookResult()

        # Run analysis if engine available
        analysis_result = None
        if self.engine and self.config.analyze_outputs:
            if context.activations is not None:
                analysis_result = self.engine.analyze(context.activations)

        # Run registered post-weave hooks
        result = await self._run_hooks(
            HookPoint.POST_WEAVE,
            context,
            (spacetime,)
        )

        # Add analysis to result
        if analysis_result:
            result.metadata["analysis"] = {
                "top_features": analysis_result.top_features[:10],
                "explanation": analysis_result.explanation,
            }

        # Call completion callback
        if self.config.on_analysis_complete:
            await self.config.on_analysis_complete({
                "context": {
                    "query": context.query_text,
                    "tool": context.selected_tool,
                    "confidence": context.confidence,
                },
                "result": result.metadata,
            })

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get middleware statistics."""
        return {
            **self._statistics,
            "mode": self.config.mode.value,
            "hooks_registered": {
                point.value: len(hooks)
                for point, hooks in self._hooks.items()
            },
        }


class DarkTraceIntegrator:
    """
    Main integration class that connects Dark Trace to the orchestrator.

    Provides a high-level API for enabling interpretability in the
    weaving pipeline.
    """

    def __init__(
        self,
        orchestrator: Any,  # WeavingOrchestrator
        engine: Optional[Any] = None,  # DarkTraceEngine
        safety_gate: Optional[Any] = None,  # DarkTraceSafetyGate
        mode: IntegrationMode = IntegrationMode.PASSIVE,
    ):
        self.orchestrator = orchestrator
        self.engine = engine
        self.safety_gate = safety_gate
        self.mode = mode

        self.middleware = DarkTraceMiddleware(
            config=MiddlewareConfig(mode=mode),
            safety_gate=safety_gate,
            engine=engine,
        )

        self._original_weave = None
        self._is_integrated = False
        self._weave_history: List[Dict[str, Any]] = []

    def integrate(self) -> None:
        """
        Integrate Dark Trace into the orchestrator.

        Wraps the orchestrator's weave method with middleware.
        """
        if self._is_integrated:
            return

        # Store original weave method
        self._original_weave = self.orchestrator.weave

        # Create wrapped version
        async def traced_weave(query, *args, **kwargs):
            return await self._traced_weave(query, *args, **kwargs)

        # Replace weave method
        self.orchestrator.weave = traced_weave
        self._is_integrated = True

    def remove(self) -> None:
        """Remove Dark Trace integration from orchestrator."""
        if not self._is_integrated:
            return

        if self._original_weave:
            self.orchestrator.weave = self._original_weave

        self._is_integrated = False

    async def _traced_weave(self, query, *args, **kwargs):
        """Internal traced weave implementation."""
        import time
        start_time = time.perf_counter()

        # Create hook context
        query_text = query.text if hasattr(query, 'text') else str(query)
        context = HookContext(query_text=query_text)

        # Pre-weave check
        pre_result = await self.middleware.pre_weave(context)
        if not pre_result.allow:
            raise IntegrationBlockedError(
                f"Weave blocked: {pre_result.metadata.get('blocked_reason', 'Unknown')}"
            )

        # Call original weave
        spacetime = await self._original_weave(query, *args, **kwargs)

        # Extract activations from spacetime if available
        if hasattr(spacetime, 'metadata'):
            activations = spacetime.metadata.get('activations')
            feature_names = spacetime.metadata.get('feature_names')

            if activations is not None:
                context.activations = activations
                context.feature_names = feature_names
                context.confidence = getattr(spacetime, 'confidence', 0.0)

                # Pre-decision analysis (retroactive)
                decision_result = await self.middleware.pre_decision(
                    context,
                    activations,
                    feature_names
                )

                if not decision_result.allow and self.mode == IntegrationMode.STRICT:
                    raise IntegrationBlockedError(
                        f"Response blocked: {decision_result.metadata.get('blocked_reason', 'Unknown')}"
                    )

                # Add warnings to spacetime metadata
                if decision_result.warnings:
                    spacetime.metadata['dark_trace_warnings'] = decision_result.warnings

        # Post-weave analysis
        post_result = await self.middleware.post_weave(context, spacetime)

        # Add analysis to spacetime
        if 'analysis' in post_result.metadata:
            spacetime.metadata['dark_trace_analysis'] = post_result.metadata['analysis']

        # Record in history
        duration_ms = (time.perf_counter() - start_time) * 1000
        self._weave_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query_text,
            "duration_ms": duration_ms,
            "warnings": post_result.warnings,
            "blocked": not pre_result.allow,
        })

        # Trim history
        if len(self._weave_history) > 1000:
            self._weave_history = self._weave_history[-1000:]

        return spacetime

    def set_mode(self, mode: IntegrationMode) -> None:
        """Change integration mode."""
        self.mode = mode
        self.middleware.config.mode = mode

    def register_hook(
        self,
        point: HookPoint,
        hook: Callable
    ) -> None:
        """Register a hook at a specific point."""
        self.middleware.register_hook(point, hook)

    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            "is_integrated": self._is_integrated,
            "mode": self.mode.value,
            "weave_count": len(self._weave_history),
            "middleware": self.middleware.get_statistics(),
        }

    def get_recent_weaves(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent weave history."""
        return self._weave_history[-limit:]


class IntegrationBlockedError(Exception):
    """Raised when integration blocks a weave operation."""
    pass


async def create_integrator(
    orchestrator: Any,
    engine: Optional[Any] = None,
    safety_gate: Optional[Any] = None,
    mode: IntegrationMode = IntegrationMode.PASSIVE,
    auto_integrate: bool = True,
) -> DarkTraceIntegrator:
    """
    Create and configure a Dark Trace integrator.

    Args:
        orchestrator: WeavingOrchestrator to integrate with
        engine: Optional DarkTraceEngine for analysis
        safety_gate: Optional DarkTraceSafetyGate for safety checks
        mode: Integration mode
        auto_integrate: Whether to immediately integrate

    Returns:
        Configured DarkTraceIntegrator
    """
    integrator = DarkTraceIntegrator(
        orchestrator=orchestrator,
        engine=engine,
        safety_gate=safety_gate,
        mode=mode,
    )

    if auto_integrate:
        integrator.integrate()

    return integrator


def integrate_dark_trace(
    orchestrator: Any,
    mode: IntegrationMode = IntegrationMode.PASSIVE,
    **kwargs
) -> DarkTraceIntegrator:
    """
    Convenience function to quickly integrate Dark Trace.

    Args:
        orchestrator: WeavingOrchestrator to integrate with
        mode: Integration mode
        **kwargs: Additional arguments for DarkTraceIntegrator

    Returns:
        Integrated DarkTraceIntegrator
    """
    integrator = DarkTraceIntegrator(
        orchestrator=orchestrator,
        mode=mode,
        **kwargs
    )
    integrator.integrate()
    return integrator
