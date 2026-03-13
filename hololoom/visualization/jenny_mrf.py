"""
Jenny MRF Integration - Metaprompt Refinement Framework Enhancement
====================================================================
Enhances Jenny panels with MRF's 7-component framework and Thompson Sampling.

Date: December 2025
Status: Phase 2 Implementation

This module provides:
1. MRF-enhanced WHY panel generation (ELEGANCE strategy)
2. Thompson Sampling learner for panel type selection
3. MRF-enhanced query analysis
4. Integration with JennyCompiler

Philosophy:
> "Every explanation deserves clarity, simplicity, and beauty."

Usage:
    from hololoom.visualization.jenny_mrf import (
        JennyMRFCompiler,
        PanelTypeLearner,
        generate_why_panel_mrf,
    )

    # MRF-enhanced compiler
    compiler = JennyMRFCompiler()
    specs = await compiler.compile(spacetime)

    # Panel type learning
    learner = PanelTypeLearner()
    panel_type = learner.select(query_type, candidates)
    learner.update(query_type, panel_type, success=True, confidence=0.9)

References:
- prompting/unified_mrf.py (MRF framework)
- jenny_compiler.py (Base compiler)
- jenny_spec.py (JennySpec dataclass)
"""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

from hololoom.bandits.beta_arm import BetaArm

from .jenny_spec import (
    JennySpec,
    PanelTypeJenny,
    PanelSizeJenny,
    LifecycleStage,
    BindingMode,
    get_default_actions,
)
from .jenny_compiler import (
    JennyCompiler,
    QueryAnalysis,
    analyze_query,
    generate_text_panel,
    generate_confidence_panel,
    generate_sources_panel,
    generate_graph_panel,
    generate_timeline_panel,
    generate_reasoning_panel,
    generate_metric_panel,
)
from hololoom.protocols.jenny import CompilationStrategy

# Try to import MRF
try:
    from hololoom.prompting.unified_mrf import (
        UnifiedMRF,
        RefinementStrategyType,
        ModelProvider,
    )
    MRF_AVAILABLE = True
except ImportError:
    MRF_AVAILABLE = False
    UnifiedMRF = None
    RefinementStrategyType = None
    ModelProvider = None

# Try to import Spacetime
try:
    from hololoom.fabric.spacetime import Spacetime, WeavingTrace
except ImportError:
    Spacetime = Any  # type: ignore
    WeavingTrace = Any  # type: ignore

logger = logging.getLogger(__name__)


# ============================================================================
# Thompson Sampling Panel Type Learner
# ============================================================================

@dataclass
class PanelTypePrior:
    """Beta distribution prior for panel type selection.

    Delegates Beta math to BetaArm (canonical implementation).
    """
    _arm: BetaArm = field(default_factory=BetaArm)

    @property
    def alpha(self) -> float:
        return self._arm.alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._arm.alpha = value

    @property
    def beta(self) -> float:
        return self._arm.beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._arm.beta = value

    def sample(self) -> float:
        """Sample from Beta(alpha, beta)."""
        return self._arm.sample()

    def expected_value(self) -> float:
        """Expected value E[X] = alpha / (alpha + beta)."""
        return self._arm.expected_value()

    def update_success(self, confidence: float = 1.0):
        """Update prior on success."""
        self._arm.update(success=True, weight=confidence)

    def update_failure(self, confidence: float = 1.0):
        """Update prior on failure."""
        self._arm.update(success=False, weight=(1.0 - confidence))


class PanelTypeLearner:
    """
    Thompson Sampling learner for optimal panel type selection.

    Learns which panel types work best for different query types based on
    user feedback (pin rate, dismiss rate, dwell time, etc.).

    Usage:
        learner = PanelTypeLearner()

        # Select panel type
        panel_type = learner.select("factual", [TEXT, CONFIDENCE, GRAPH])

        # Update based on outcome
        learner.update("factual", TEXT, success=True, confidence=0.85)

        # Get statistics
        stats = learner.get_statistics()
    """

    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize learner.

        Args:
            persist_path: Optional path to persist learning state
        """
        # Priors indexed by (query_type, panel_type)
        self.priors: Dict[Tuple[str, str], PanelTypePrior] = defaultdict(PanelTypePrior)
        self.selection_history: List[Dict[str, Any]] = []
        self.persist_path = persist_path

        # Load persisted state if available
        if persist_path:
            self._load_state()

        logger.info("PanelTypeLearner initialized")

    def select(
        self,
        query_type: str,
        candidates: List[PanelTypeJenny],
        exploration_bonus: float = 0.0
    ) -> PanelTypeJenny:
        """
        Select optimal panel type using Thompson Sampling.

        Args:
            query_type: Type of query (factual, procedural, analytical, etc.)
            candidates: List of candidate panel types
            exploration_bonus: Extra exploration boost (0.0-1.0)

        Returns:
            Selected panel type
        """
        if not candidates:
            return PanelTypeJenny.TEXT

        # Sample from each prior
        samples = {}
        for panel_type in candidates:
            key = (query_type, panel_type.value)
            prior = self.priors[key]
            sample = prior.sample()

            # Add exploration bonus if requested
            if exploration_bonus > 0:
                sample += random.uniform(0, exploration_bonus)

            samples[panel_type] = sample

        # Select highest sample
        selected = max(samples, key=samples.get)

        # Log selection
        self.selection_history.append({
            "timestamp": datetime.now().isoformat(),
            "query_type": query_type,
            "candidates": [p.value for p in candidates],
            "samples": {p.value: s for p, s in samples.items()},
            "selected": selected.value,
        })

        logger.debug(f"Selected {selected.value} for {query_type} (sample={samples[selected]:.3f})")

        return selected

    def update(
        self,
        query_type: str,
        panel_type: PanelTypeJenny,
        success: bool,
        confidence: float = 1.0
    ):
        """
        Update prior based on outcome.

        Args:
            query_type: Type of query
            panel_type: Panel type that was shown
            success: Whether interaction was successful (pin, positive feedback)
            confidence: Confidence weight (0.0-1.0)
        """
        key = (query_type, panel_type.value)
        prior = self.priors[key]

        if success:
            prior.update_success(confidence)
        else:
            prior.update_failure(confidence)

        logger.debug(
            f"Updated {panel_type.value} for {query_type}: "
            f"α={prior.alpha:.2f}, β={prior.beta:.2f}, E[X]={prior.expected_value():.3f}"
        )

        # Persist state if path configured
        if self.persist_path:
            self._save_state()

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        stats = {
            "total_selections": len(self.selection_history),
            "priors": {},
        }

        for (query_type, panel_type), prior in self.priors.items():
            key = f"{query_type}:{panel_type}"
            stats["priors"][key] = {
                "alpha": prior.alpha,
                "beta": prior.beta,
                "expected_value": prior.expected_value(),
                "total_updates": (prior.alpha - 1) + (prior.beta - 1),
            }

        return stats

    def get_best_panel_type(self, query_type: str) -> Optional[PanelTypeJenny]:
        """Get the current best panel type for a query type (without sampling)."""
        best_value = 0.0
        best_type = None

        for (qt, pt), prior in self.priors.items():
            if qt == query_type:
                ev = prior.expected_value()
                if ev > best_value:
                    best_value = ev
                    best_type = PanelTypeJenny(pt)

        return best_type

    def _save_state(self):
        """Persist learning state to disk."""
        if not self.persist_path:
            return

        state = {
            "priors": {
                f"{qt}:{pt}": {"alpha": p.alpha, "beta": p.beta}
                for (qt, pt), p in self.priors.items()
            },
            "selection_count": len(self.selection_history),
            "saved_at": datetime.now().isoformat(),
        }

        path = Path(self.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load persisted learning state."""
        if not self.persist_path:
            return

        path = Path(self.persist_path)
        if not path.exists():
            return

        try:
            with open(path, 'r') as f:
                state = json.load(f)

            for key, prior_data in state.get("priors", {}).items():
                qt, pt = key.split(":", 1)
                self.priors[(qt, pt)] = PanelTypePrior(
                    _arm=BetaArm(alpha=prior_data["alpha"], beta=prior_data["beta"])
                )

            logger.info(f"Loaded {len(self.priors)} priors from {self.persist_path}")
        except Exception as e:
            logger.warning(f"Failed to load state from {self.persist_path}: {e}")


# ============================================================================
# MRF-Enhanced WHY Panel Generation
# ============================================================================

async def generate_why_panel_mrf(
    spacetime: Spacetime,
    specs_generated: List[JennySpec],
    analysis: QueryAnalysis,
    mrf: Optional["UnifiedMRF"] = None,
    priority: int = 99,
) -> JennySpec:
    """
    Generate MRF-enhanced 'Why this UI?' meta-panel.

    Uses MRF ELEGANCE strategy (Clarity → Simplicity → Beauty) to generate
    better explanations than rule-based generation.

    Args:
        spacetime: The Spacetime output
        specs_generated: List of panels already generated
        analysis: Query analysis results
        mrf: Optional UnifiedMRF instance (creates one if not provided)
        priority: Panel priority (default: 99 = lowest)

    Returns:
        JennySpec for the WHY panel with enhanced explanation
    """
    # Build the base explanation context
    panel_types = [s.panel_type.value for s in specs_generated]
    decision_factors = _extract_decision_factors(spacetime, analysis)

    # If MRF available, enhance the explanation
    explanation = ""
    mrf_enhanced = False

    if MRF_AVAILABLE and mrf is not None:
        try:
            # Use ELEGANCE strategy for clear, simple, beautiful explanations
            enhanced = await mrf.refine_prompt(
                original_prompt=f"""
Explain why these UI panels were chosen for the user's query.

Query: "{spacetime.query_text}"
Query Type: {analysis.query_type}
Complexity: {analysis.complexity}
Confidence: {spacetime.confidence:.0%}

Panels Generated:
{chr(10).join(f'- {pt.upper()}' for pt in panel_types)}

Decision Factors:
{chr(10).join(f'- {k}: {v}' for k, v in decision_factors.items())}

Provide a concise (2-3 sentences), user-friendly explanation of why these specific panels were chosen.
Focus on what value each panel provides to the user.
""",
                strategy=RefinementStrategyType.ELEGANCE,
                context={
                    "audience": "end_user",
                    "tone": "helpful",
                    "query_type": analysis.query_type,
                },
                epistemic_confidence=spacetime.confidence,
            )

            explanation = enhanced.get("enhanced_prompt", "")
            mrf_enhanced = True
            logger.debug("Generated MRF-enhanced WHY panel explanation")

        except Exception as e:
            logger.warning(f"MRF enhancement failed, falling back to rule-based: {e}")

    # Fallback to rule-based explanation
    if not explanation:
        explanation = _generate_rule_based_explanation(analysis, panel_types, decision_factors)

    return JennySpec(
        spacetime_id=f"st_{id(spacetime)}",
        panel_type=PanelTypeJenny.WHY,
        title="Why This UI?",
        content={
            "explanation": explanation,
            "query_type": analysis.query_type,
            "complexity": analysis.complexity,
            "panels_generated": len(specs_generated),
            "panel_types": panel_types,
            "decision_factors": decision_factors,
            "mrf_enhanced": mrf_enhanced,
        },
        size=PanelSizeJenny.SMALL,
        priority=priority,
        lifecycle=LifecycleStage.SYSTEM,  # SYSTEM stage - not logged
        actions=get_default_actions(PanelTypeJenny.WHY),
    )


def _extract_decision_factors(spacetime: Spacetime, analysis: QueryAnalysis) -> Dict[str, Any]:
    """Extract factors that influenced panel selection."""
    factors = {
        "confidence_level": analysis.confidence_level,
        "has_sources": analysis.has_sources,
        "has_graph_context": analysis.has_graph_context,
        "has_reasoning": analysis.has_reasoning,
        "has_metrics": analysis.has_metrics,
    }

    if spacetime.trace:
        factors["threads_activated"] = len(getattr(spacetime.trace, 'threads_activated', []))
        factors["stage_count"] = len(getattr(spacetime.trace, 'stage_durations', {}))
        factors["duration_ms"] = getattr(spacetime.trace, 'duration_ms', 0)

    return factors


def _generate_rule_based_explanation(
    analysis: QueryAnalysis,
    panel_types: List[str],
    decision_factors: Dict[str, Any]
) -> str:
    """Generate rule-based explanation (fallback when MRF unavailable)."""
    parts = []

    # Opening based on query type
    type_explanations = {
        "factual": "For your factual question",
        "procedural": "For your how-to question",
        "analytical": "For your analytical query",
        "creative": "For your creative request",
        "debug": "For your debugging question",
    }
    parts.append(type_explanations.get(analysis.query_type, "For your question"))

    # Complexity modifier
    if analysis.complexity == "complex":
        parts.append("which involves multiple concepts,")
    elif analysis.complexity == "moderate":
        parts.append("with moderate complexity,")

    # Panel rationale
    parts.append(f"we generated {len(panel_types)} panels.")

    # Specific panel explanations
    panel_explanations = []
    if "text" in panel_types:
        panel_explanations.append("The TEXT panel shows the main response")
    if "confidence" in panel_types:
        panel_explanations.append("CONFIDENCE shows how certain the system is")
    if "graph" in panel_types:
        panel_explanations.append("GRAPH visualizes connected concepts")
    if "timeline" in panel_types:
        panel_explanations.append("TIMELINE shows the processing steps")
    if "sources" in panel_types:
        panel_explanations.append("SOURCES lists where information came from")

    if panel_explanations:
        parts.append(" ".join(panel_explanations[:2]) + ".")

    return " ".join(parts)


# ============================================================================
# MRF-Enhanced Jenny Compiler
# ============================================================================

class JennyMRFCompiler(JennyCompiler):
    """
    MRF-enhanced Jenny compiler with Thompson Sampling learning.

    Extends base JennyCompiler with:
    1. MRF-enhanced WHY panel generation
    2. Thompson Sampling panel type selection
    3. Learning from user interactions

    Usage:
        compiler = JennyMRFCompiler()
        specs = await compiler.compile(spacetime)

        # After user interaction
        compiler.update_learning("factual", PanelTypeJenny.TEXT, success=True)
    """

    VERSION = "2.0.0-mrf"

    def __init__(
        self,
        default_strategy: CompilationStrategy = CompilationStrategy.AUTO,
        include_why_panel: bool = True,
        enable_learning: bool = True,
        learning_persist_path: Optional[str] = None,
        mrf_instance: Optional["UnifiedMRF"] = None,
    ):
        """
        Initialize MRF-enhanced compiler.

        Args:
            default_strategy: Default compilation strategy
            include_why_panel: Whether to include "Why this UI?" panel
            enable_learning: Enable Thompson Sampling learning
            learning_persist_path: Path to persist learning state
            mrf_instance: Optional pre-configured MRF instance
        """
        super().__init__(default_strategy, include_why_panel)

        self.enable_learning = enable_learning
        self.learner = PanelTypeLearner(persist_path=learning_persist_path) if enable_learning else None

        # Initialize MRF if available
        self.mrf = mrf_instance
        if MRF_AVAILABLE and self.mrf is None:
            try:
                self.mrf = UnifiedMRF()
            except Exception as e:
                logger.warning(f"Failed to initialize MRF: {e}")
                self.mrf = None

        logger.info(
            f"JennyMRFCompiler v{self.VERSION} initialized "
            f"(learning={enable_learning}, mrf={'enabled' if self.mrf else 'disabled'})"
        )

    async def compile(
        self,
        spacetime: Spacetime,
        strategy: CompilationStrategy = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[JennySpec]:
        """
        Compile Spacetime into UI specifications with MRF enhancement.

        Args:
            spacetime: Woven output from hololoom weaving cycle
            strategy: Compilation strategy (None = use default)
            context: Optional user/session context

        Returns:
            List of JennySpec panels to render
        """
        strategy = strategy or self.default_strategy

        # Analyze query
        analysis = analyze_query(spacetime)

        # Determine effective strategy
        if strategy == CompilationStrategy.AUTO:
            strategy = self._auto_select_strategy(analysis)

        # Generate panels based on strategy
        specs = []

        # Always include text panel
        specs.append(generate_text_panel(spacetime, priority=0))

        # Use Thompson Sampling for panel selection if learning enabled
        if self.enable_learning and self.learner:
            # Get candidates based on strategy
            candidates = self._get_panel_candidates(strategy, analysis, spacetime)

            # Let learner influence selection
            for priority, candidate_set in enumerate(candidates, start=1):
                if candidate_set:
                    selected = self.learner.select(analysis.query_type, candidate_set)
                    panel = self._generate_panel(selected, spacetime, priority)
                    if panel:
                        specs.append(panel)
        else:
            # Fall back to base compiler logic
            specs = await super().compile(spacetime, strategy, context)
            # Remove the WHY panel from base compiler (we'll add MRF version)
            specs = [s for s in specs if s.panel_type != PanelTypeJenny.WHY]

        # Add MRF-enhanced WHY panel if enabled
        if self.include_why_panel:
            why_panel = await generate_why_panel_mrf(
                spacetime, specs, analysis, self.mrf, priority=99
            )
            specs.append(why_panel)

        # Sort by priority
        specs.sort(key=lambda s: s.priority)

        # Store query_type in metadata for learning callback (Phase C)
        for spec in specs:
            if spec.metadata is None:
                # JennySpec.metadata is a dict, but might not exist
                object.__setattr__(spec, 'metadata', {})
            spec.metadata['query_type'] = analysis.query_type
            spec.metadata['complexity'] = analysis.complexity

        logger.debug(
            f"MRF Compiled {len(specs)} panels for query "
            f"(strategy={strategy.value}, type={analysis.query_type})"
        )

        return specs

    def _get_panel_candidates(
        self,
        strategy: CompilationStrategy,
        analysis: QueryAnalysis,
        spacetime: Spacetime
    ) -> List[List[PanelTypeJenny]]:
        """Get candidate panel types based on strategy."""
        candidates = []

        # Priority 1: Always confidence
        candidates.append([PanelTypeJenny.CONFIDENCE])

        # Priority 2-3: Based on content
        if strategy in [CompilationStrategy.BALANCED, CompilationStrategy.COMPREHENSIVE]:
            # Sources vs Graph choice
            content_candidates = []
            if analysis.has_sources:
                content_candidates.append(PanelTypeJenny.SOURCES)
            if analysis.has_graph_context:
                content_candidates.append(PanelTypeJenny.GRAPH)
            if content_candidates:
                candidates.append(content_candidates)

        # Priority 4-5: Comprehensive extras
        if strategy == CompilationStrategy.COMPREHENSIVE:
            if analysis.has_metrics:
                candidates.append([PanelTypeJenny.TIMELINE, PanelTypeJenny.METRIC])
            if analysis.has_reasoning:
                candidates.append([PanelTypeJenny.REASONING])

        return candidates

    def _generate_panel(
        self,
        panel_type: PanelTypeJenny,
        spacetime: Spacetime,
        priority: int
    ) -> Optional[JennySpec]:
        """Generate a panel of the specified type."""
        generators = {
            PanelTypeJenny.CONFIDENCE: lambda: generate_confidence_panel(spacetime, priority),
            PanelTypeJenny.SOURCES: lambda: generate_sources_panel(spacetime, priority),
            PanelTypeJenny.GRAPH: lambda: generate_graph_panel(spacetime, priority),
            PanelTypeJenny.TIMELINE: lambda: generate_timeline_panel(spacetime, priority),
            PanelTypeJenny.REASONING: lambda: generate_reasoning_panel(spacetime, priority),
            PanelTypeJenny.METRIC: lambda: generate_metric_panel(
                spacetime, "Duration",
                f"{getattr(spacetime.trace, 'duration_ms', 0):.1f}ms",
                priority
            ),
        }

        generator = generators.get(panel_type)
        if generator:
            return generator()
        return None

    def update_learning(
        self,
        query_type: str,
        panel_type: PanelTypeJenny,
        success: bool,
        confidence: float = 1.0
    ):
        """
        Update learning based on user interaction.

        Call this when user interacts with a panel (pin, dismiss, etc.)

        Args:
            query_type: Type of query (factual, procedural, etc.)
            panel_type: Panel type that was interacted with
            success: True if positive interaction (pin), False if negative (quick dismiss)
            confidence: Confidence weight (0.0-1.0)
        """
        if self.learner:
            self.learner.update(query_type, panel_type, success, confidence)

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        if self.learner:
            return self.learner.get_statistics()
        return {"learning_enabled": False}


# ============================================================================
# Factory Functions
# ============================================================================

def create_mrf_compiler(
    enable_learning: bool = True,
    learning_persist_path: Optional[str] = None,
) -> JennyMRFCompiler:
    """
    Create an MRF-enhanced Jenny compiler.

    Args:
        enable_learning: Enable Thompson Sampling learning
        learning_persist_path: Path to persist learning state

    Returns:
        Configured JennyMRFCompiler instance
    """
    return JennyMRFCompiler(
        enable_learning=enable_learning,
        learning_persist_path=learning_persist_path,
    )


def create_panel_learner(persist_path: Optional[str] = None) -> PanelTypeLearner:
    """
    Create a standalone panel type learner.

    Args:
        persist_path: Path to persist learning state

    Returns:
        Configured PanelTypeLearner instance
    """
    return PanelTypeLearner(persist_path=persist_path)


# ============================================================================
# Action → Learning Signal Mapping (Phase C - Close the Learning Loop)
# ============================================================================

# Maps action handler names to (success: bool, confidence: float) tuples
# These values represent the learning signal strength for each action type
ACTION_LEARNING_MAP: Dict[str, Tuple[bool, float]] = {
    "pin_panel": (True, 0.9),      # Strong positive signal - user valued the panel
    "dismiss_panel": (False, 0.3), # Weak negative signal - could be cleanup, not dislike
    "show_why_panel": (True, 0.7), # Moderate positive - user engaged with meta-panel
    "copy_content": (True, 0.8),   # Strong positive - user found content useful
    "export_panel": (True, 0.9),   # Strong positive - user wants to save
    "expand_panel": (True, 0.6),   # Moderate positive - user wants more detail
    "collapse_panel": (True, 0.4), # Weak positive - still interacting but wants less
}


async def create_learning_callback(
    compiler: "JennyMRFCompiler",
) -> "Callable":
    """
    Create a learning callback for JennyActionHandler.

    The callback translates user actions (PIN, DISMISS, WHY, etc.) into
    Thompson Sampling learning updates. This closes the learning loop:

    User Action → ActionResult → Learning Callback → Thompson Sampling Update

    Args:
        compiler: JennyMRFCompiler instance with learning enabled

    Returns:
        Async callback function (ActionResult, JennySpec) -> None

    Usage:
        compiler = create_mrf_compiler(enable_learning=True)
        callback = await create_learning_callback(compiler)
        action_handler = create_action_handler(
            lifecycle_manager=lifecycle,
            ledger=ledger,
            learning_callback=callback,
        )
    """
    # Import here to avoid circular dependency
    from .jenny_actions import ActionResult

    async def callback(result: ActionResult, spec: JennySpec) -> None:
        """
        Process action result and update Thompson Sampling priors.

        Maps user actions to learning signals:
        - PIN → strong positive (α += 0.9)
        - DISMISS → weak negative (β += 0.7)
        - WHY → moderate positive (α += 0.7)
        - COPY → strong positive (α += 0.8)
        - EXPORT → strong positive (α += 0.9)
        """
        handler = result.handler

        if handler not in ACTION_LEARNING_MAP:
            logger.debug(f"Unknown action handler '{handler}' - skipping learning update")
            return

        success, confidence = ACTION_LEARNING_MAP[handler]

        # Extract query_type from spec metadata (set during compilation)
        query_type = spec.metadata.get('query_type', 'unknown')

        # Update Thompson Sampling priors
        compiler.update_learning(
            query_type=query_type,
            panel_type=spec.panel_type,
            success=success,
            confidence=confidence
        )

        logger.info(
            f"Learning update: {handler} on {spec.panel_type.value} "
            f"(query_type={query_type}, success={success}, conf={confidence:.1f})"
        )

    return callback


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Classes
    'JennyMRFCompiler',
    'PanelTypeLearner',
    'PanelTypePrior',

    # Functions
    'generate_why_panel_mrf',
    'create_mrf_compiler',
    'create_panel_learner',
    'create_learning_callback',

    # Constants
    'MRF_AVAILABLE',
    'ACTION_LEARNING_MAP',
]
