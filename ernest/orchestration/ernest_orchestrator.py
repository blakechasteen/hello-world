"""
Ernest Orchestrator - Phase 3
==============================
Full HoloLoom 9-step weaving cycle integration for creative writing.

This orchestrator wraps HoloLoom's WeavingOrchestrator and adds:
- Hemingway refinement in the synthesis step
- Ernest learning engine integration
- Creative writing metrics tracking
- Mode-aware query processing

Architecture:
- Uses complete 9-step weaving cycle (Loom → Chrono → Yarn → Resonance →
  Warp → Convergence → Tool → Spacetime → Reflection)
- Integrates Ernest learning for continuous improvement
- Tracks Hemingway scores in Spacetime provenance
- Adapts mode selection based on writing context

Usage:
```python
from ernest.orchestration import ErnestOrchestrator
from HoloLoom.config import Config

config = Config.fused()  # Full power for creative writing
async with ErnestOrchestrator(config=config, enable_learning=True) as ernest:
    # Query with automatic refinement
    spacetime = await ernest.weave_creative(
        "Analyze my opening paragraph for Chapter 5",
        context="narrative"
    )

    print(f"Response: {spacetime.response}")
    print(f"Hemingway Score: {spacetime.metadata['hemingway_score']:.0f}/100")

    # Get learning statistics
    stats = ernest.get_learning_statistics()
    print(f"Refinements performed: {stats['refinements_performed']}")
    print(f"Avg improvement: {stats['avg_improvement']:.1f} points")
```
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

from HoloLoom.config import Config
from HoloLoom.protocols.types import Query, MemoryShard
from HoloLoom.fabric.spacetime import Spacetime
from HoloLoom.weaving_orchestrator import WeavingOrchestrator

from ernest.learning import ErnestLearningEngine
from ernest.refinement.patterns import HemingwayMode, HemingwayPatterns


@dataclass
class CreativeContext:
    """
    Context for creative writing queries.

    Helps Ernest understand what type of writing you're working on
    so it can select the appropriate mode and refinement strategy.
    """
    context_type: str  # dialogue, action, description, narrative, etc.
    genre: Optional[str] = None  # literary, thriller, romance, etc.
    target_audience: Optional[str] = None  # adult, YA, middle-grade, etc.
    chapter: Optional[int] = None
    scene_type: Optional[str] = None  # opening, climax, resolution, etc.

    def get_hemingway_mode_hint(self) -> HemingwayMode:
        """
        Get recommended Hemingway mode based on context.

        This is a hint - Ernest's learning engine may override based
        on what it's learned works best.
        """
        if self.context_type in ["dialogue", "confrontation"]:
            return HemingwayMode.SPARSE  # Iceberg maximum
        elif self.context_type in ["action", "chase", "fight"]:
            return HemingwayMode.DIRECT  # Journalistic precision
        elif self.context_type in ["description", "setting", "character"]:
            return HemingwayMode.GRACE  # Economical elegance
        elif self.context_type in ["existential", "introspection", "loss"]:
            return HemingwayMode.LOST_GEN  # Disaffected truth
        else:
            return HemingwayMode.GRACE  # Default


class ErnestOrchestrator:
    """
    Hemingway-powered creative writing orchestrator.

    Integrates HoloLoom's full 9-step weaving cycle with Ernest's
    learning and refinement capabilities.

    **The 9-Step Weaving Cycle for Creative Writing**:

    1. **Loom Command**: Select pattern card (BARE/FAST/FUSED)
       - Creative writing typically uses FUSED (full power)

    2. **Chrono Trigger**: Fire temporal window
       - Load recent chapters, writing patterns, style preferences

    3. **Yarn Graph**: Select memory threads
       - Retrieve relevant creative writing examples
       - Load character notes, plot threads, world-building

    4. **Resonance Shed**: Lift feature threads
       - Motifs: Story patterns, narrative structure
       - Embeddings: Semantic similarity to your writing
       - Spectral: Writing style topology

    5. **Warp Space**: Tension into continuous manifold
       - Blend creative examples into coherent context

    6. **Convergence Engine**: Collapse to discrete decision
       - Select appropriate response strategy (analyze/refine/suggest)

    7. **Tool Execution**: Generate response
       - Apply Hemingway refinement if requested
       - Track creative writing metrics

    8. **Spacetime Fabric**: Weave output with provenance
       - Include Hemingway score, mode used, improvements made
       - Complete trace for learning

    9. **Reflection Buffer**: Learn from outcome
       - Update Ernest's mode preferences
       - Learn which refinement strategies work
       - Adapt Thompson Sampling priors
    """

    def __init__(
        self,
        config: Config,
        shards: Optional[List[MemoryShard]] = None,
        enable_learning: bool = True,
        learning_persistence_path: Optional[str] = None
    ):
        """
        Initialize Ernest orchestrator.

        Args:
            config: HoloLoom configuration (recommend Config.fused() for creative writing)
            shards: Memory shards (creative writing examples, chapters, etc.)
            enable_learning: Enable Ernest learning engine
            learning_persistence_path: Path to save/load Ernest's learning state
        """
        self.config = config
        self.shards = shards or []
        self.logger = logging.getLogger(f"{__name__}.ErnestOrchestrator")

        # Core orchestrator (will be initialized in __aenter__)
        self.orchestrator: Optional[WeavingOrchestrator] = None

        # Ernest learning engine
        self.learning_engine: Optional[ErnestLearningEngine] = None
        self.enable_learning = enable_learning
        self.learning_persistence_path = learning_persistence_path or "./ernest_learning_state.json"

    async def __aenter__(self):
        """Async context manager entry"""
        # Initialize HoloLoom orchestrator
        self.orchestrator = WeavingOrchestrator(
            cfg=self.config,
            shards=self.shards
        )
        await self.orchestrator.__aenter__()
        self.logger.info("HoloLoom orchestrator initialized")

        # Initialize Ernest learning engine
        if self.enable_learning:
            self.learning_engine = ErnestLearningEngine(
                enable_background_learning=True,
                learning_update_interval=60.0,
                persistence_path=self.learning_persistence_path
            )
            await self.learning_engine.__aenter__()
            self.logger.info("Ernest learning engine initialized")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Cleanup Ernest learning engine
        if self.learning_engine:
            await self.learning_engine.__aexit__(exc_type, exc_val, exc_tb)

        # Cleanup orchestrator
        if self.orchestrator:
            await self.orchestrator.__aexit__(exc_type, exc_val, exc_tb)

        return False

    async def weave_creative(
        self,
        query_text: str,
        context: Optional[CreativeContext] = None,
        enable_refinement: bool = True,
        force_mode: Optional[HemingwayMode] = None
    ) -> Spacetime:
        """
        Weave creative writing query through full 9-step cycle.

        This is the main entry point for creative writing queries.

        Args:
            query_text: Your query (e.g., "Analyze my opening paragraph")
            context: Creative context (dialogue/action/description/etc.)
            enable_refinement: Apply Hemingway refinement if applicable
            force_mode: Force specific Hemingway mode (overrides learning)

        Returns:
            Spacetime with response + Hemingway metrics in metadata

        Example:
        ```python
        spacetime = await ernest.weave_creative(
            "How can I make this dialogue sharper?",
            context=CreativeContext(context_type="dialogue")
        )
        ```
        """
        # Create query
        query = Query(text=query_text)

        # Step 1-6: Run through HoloLoom weaving cycle
        spacetime = await self.orchestrator.weave(query)

        # Determine if we should apply Hemingway refinement
        should_refine = (
            enable_refinement and
            self._should_apply_refinement(query_text, spacetime)
        )

        if should_refine and self.learning_engine:
            # Get context type
            context_type = context.context_type if context else "general"

            # Get mode (from context hint or Ernest's learned preferences)
            if force_mode:
                mode = force_mode
            elif context:
                # Get hint from context, but let learning engine override
                hint_mode = context.get_hemingway_mode_hint()
                mode = self.learning_engine.mode_prefs.get_best_mode(context_type)
                self.logger.info(f"Context hint: {hint_mode.value}, learned preference: {mode.value}")
            else:
                mode = self.learning_engine.mode_prefs.get_best_mode(context_type)

            # Apply Hemingway refinement to the response
            refinement_result = await self.learning_engine.refine_with_learning(
                text=spacetime.response,
                context=context_type,
                mode=mode
            )

            # Update spacetime with refined response
            spacetime.response = refinement_result["refined_text"]

            # Add Hemingway metrics to metadata
            spacetime.metadata.update({
                "ernest_applied": True,
                "hemingway_mode": mode.value,
                "hemingway_score_before": refinement_result["before_score"],
                "hemingway_score_after": refinement_result["after_score"],
                "hemingway_improvement": refinement_result["improvement"],
                "hemingway_metrics": {
                    "avg_words_per_sentence": refinement_result["after_metrics"].avg_words_per_sentence,
                    "active_voice_pct": refinement_result["after_metrics"].active_voice_pct,
                    "strong_verb_pct": refinement_result["after_metrics"].strong_verb_pct,
                    "iceberg_ratio": refinement_result["after_metrics"].iceberg_ratio,
                    "flesch_kincaid_grade": refinement_result["after_metrics"].flesch_kincaid_grade
                },
                "refinement_changes": len(refinement_result["changes"]),
                "pass1_improvement": refinement_result["pass1_improvement"],
                "pass2_improvement": refinement_result["pass2_improvement"],
                "pass3_improvement": refinement_result["pass3_improvement"]
            })

            self.logger.info(
                f"Ernest refinement applied: {refinement_result['before_score']:.0f} → "
                f"{refinement_result['after_score']:.0f} (+{refinement_result['improvement']:.0f}) "
                f"using {mode.value} mode"
            )

        else:
            # No refinement applied
            spacetime.metadata.update({
                "ernest_applied": False
            })

        return spacetime

    async def analyze_prose(self, text: str) -> Dict[str, Any]:
        """
        Analyze prose with Ernest (no refinement, just metrics).

        Useful for getting Hemingway score and recommendations without
        modifying the text.

        Args:
            text: Text to analyze

        Returns:
            Dict with Hemingway metrics and recommendations
        """
        ernest_patterns = HemingwayPatterns()
        metrics = ernest_patterns.analyze_prose(text)

        # Auto-recommend mode
        if metrics.iceberg_ratio < 50:
            recommended_mode = HemingwayMode.SPARSE
            reason = "Low showing vs. telling ratio - try SPARSE mode (90% subtext)"
        elif metrics.avg_words_per_sentence > 25:
            recommended_mode = HemingwayMode.DIRECT
            reason = "Long sentences - try DIRECT mode (journalistic precision)"
        elif metrics.active_voice_pct < 70:
            recommended_mode = HemingwayMode.GRACE
            reason = "Low active voice - try GRACE mode (economical elegance)"
        else:
            recommended_mode = HemingwayMode.GRACE
            reason = "Overall solid prose - GRACE mode for polish"

        return {
            "hemingway_score": metrics.hemingway_score,
            "metrics": {
                "avg_words_per_sentence": metrics.avg_words_per_sentence,
                "max_sentence_length": metrics.max_sentence_length,
                "active_voice_pct": metrics.active_voice_pct,
                "strong_verb_pct": metrics.strong_verb_pct,
                "flesch_kincaid_grade": metrics.flesch_kincaid_grade,
                "iceberg_ratio": metrics.iceberg_ratio,
                "filler_word_count": metrics.filler_word_count,
                "weak_verb_count": metrics.weak_verb_count,
                "emotion_word_count": metrics.emotion_word_count,
                "telling_verb_count": metrics.telling_verb_count
            },
            "recommended_mode": recommended_mode.value,
            "recommendation_reason": reason,
            "needs_improvement": self._identify_improvements(metrics)
        }

    async def refine_prose(
        self,
        text: str,
        context: str = "general",
        mode: Optional[HemingwayMode] = None
    ) -> Dict[str, Any]:
        """
        Refine prose with Ernest (3-pass refinement with learning).

        Direct refinement without the full weaving cycle. Useful when you
        just want to improve existing text.

        Args:
            text: Text to refine
            context: Writing context (dialogue/action/description/etc.)
            mode: Specific mode to use (None = auto-select)

        Returns:
            Dict with refinement results
        """
        if not self.learning_engine:
            raise RuntimeError("Learning engine not enabled. Set enable_learning=True")

        return await self.learning_engine.refine_with_learning(text, context, mode)

    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get Ernest's learning statistics.

        Shows how much Ernest has learned from your writing and which
        modes/strategies work best.
        """
        if not self.learning_engine:
            return {"learning_enabled": False}

        stats = self.learning_engine.get_learning_statistics()
        stats["learning_enabled"] = True
        return stats

    def _should_apply_refinement(self, query_text: str, spacetime: Spacetime) -> bool:
        """
        Determine if Hemingway refinement should be applied.

        Refinement is applied when:
        - Query mentions "refine", "improve", "polish", "edit"
        - Response is creative writing (not just analysis)
        - Response length is suitable (50-5000 words)
        """
        # Check query for refinement keywords
        query_lower = query_text.lower()
        refinement_keywords = [
            "refine", "improve", "polish", "edit", "rewrite",
            "sharpen", "tighten", "clean", "fix", "enhance"
        ]
        if any(kw in query_lower for kw in refinement_keywords):
            return True

        # Check response characteristics
        response = spacetime.response
        word_count = len(response.split())

        # Too short or too long to refine
        if word_count < 50 or word_count > 5000:
            return False

        # Check if response looks like creative writing
        # (vs. just analysis or conversation)
        analysis_keywords = [
            "The text", "This passage", "The writing", "You asked",
            "Here's what", "I think", "I would", "You should",
            "Consider", "Try", "Use"
        ]
        starts_with_analysis = any(
            response.strip().startswith(kw) for kw in analysis_keywords
        )

        # Don't refine if it's clearly analysis
        if starts_with_analysis:
            return False

        # Otherwise, apply refinement
        return True

    def _identify_improvements(self, metrics) -> List[str]:
        """Identify areas needing improvement based on metrics"""
        improvements = []

        if metrics.avg_words_per_sentence > 20:
            improvements.append(
                f"Sentence length: {metrics.avg_words_per_sentence:.1f} words "
                f"(target: 16) - Break into shorter sentences"
            )

        if metrics.active_voice_pct < 85:
            improvements.append(
                f"Active voice: {metrics.active_voice_pct:.0f}% "
                f"(target: 85%) - Convert passive to active"
            )

        if metrics.strong_verb_pct < 70:
            improvements.append(
                f"Strong verbs: {metrics.strong_verb_pct:.0f}% "
                f"(target: 70%) - Replace weak verbs (is/was/were)"
            )

        if metrics.iceberg_ratio < 70:
            improvements.append(
                f"Iceberg ratio: {metrics.iceberg_ratio:.0f}% showing "
                f"(target: 70%) - Show more, tell less"
            )

        if metrics.flesch_kincaid_grade > 8:
            improvements.append(
                f"Readability: Grade {metrics.flesch_kincaid_grade:.1f} "
                f"(target: 7) - Simplify language"
            )

        if metrics.filler_word_count > 0:
            improvements.append(
                f"Filler words: {metrics.filler_word_count} detected "
                f"- Remove 'very', 'really', 'just', etc."
            )

        if not improvements:
            improvements.append("All metrics within Hemingway targets - excellent prose!")

        return improvements


# Quick helper functions
async def quick_ernest_query(query: str, chapter_folder: Optional[str] = None) -> str:
    """
    Quick Ernest query with automatic setup.

    One-liner for getting creative writing feedback.

    Args:
        query: Your creative writing query
        chapter_folder: Path to folder with your chapters (optional)

    Returns:
        Ernest's response
    """
    # Load chapters if provided
    shards = []
    if chapter_folder:
        folder_path = Path(chapter_folder)
        if folder_path.exists():
            for chapter_file in folder_path.glob("chapter*"):
                content = chapter_file.read_text(encoding='utf-8', errors='ignore')
                from HoloLoom.protocols.types import MemoryShard
                shards.append(MemoryShard(
                    content=content,
                    source=str(chapter_file),
                    timestamp=datetime.now()
                ))

    # Create Ernest orchestrator
    config = Config.fused()  # Full power for creative writing
    async with ErnestOrchestrator(config=config, shards=shards) as ernest:
        spacetime = await ernest.weave_creative(query)
        return spacetime.response


async def quick_ernest_refine(text: str, context: str = "general") -> str:
    """
    Quick Ernest refinement.

    One-liner for refining prose with Hemingway principles.
    """
    config = Config.fused()
    async with ErnestOrchestrator(config=config, enable_learning=True) as ernest:
        result = await ernest.refine_prose(text, context)
        return result["refined_text"]
