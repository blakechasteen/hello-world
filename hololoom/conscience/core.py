"""
Conscience Core
===============
The main Conscience class - unified safety through composable concerns.

Three verbs, like Memory has three verbs:
- consider: Evaluate an action before execution
- witness: Record an action and its outcome
- learn: Improve from feedback

Philosophy: "Composable concerns, unified judgment."

Author: HoloLoom Team
Date: 2025-12-03
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .judgment import Concern, Judgment, Voice, Wisdom, quiet_judgment
from .lenses import CompositeLens, standard
from .protocol import ConscienceLens, LearnerProtocol, WitnessProtocol

logger = logging.getLogger("hololoom.conscience")


# =============================================================================
# Witness Implementation
# =============================================================================

class MemoryWitness:
    """
    In-memory witness that records actions and judgments.

    For production, extend with persistent storage (SQLite, PostgreSQL, etc.)
    """

    def __init__(self, max_records: int = 10000):
        """
        Initialize memory witness.

        Args:
            max_records: Maximum records to keep in memory
        """
        self._records: list[dict[str, Any]] = []
        self._max_records = max_records
        self._record_counter = 0

    async def record(
        self,
        action: str,
        context: dict[str, Any],
        judgment: Judgment,
        result: dict[str, Any] | None = None
    ) -> str:
        """
        Record an action and its judgment.

        Args:
            action: The action taken
            context: Context for the action
            judgment: The conscience's judgment
            result: The outcome of the action (if executed)

        Returns:
            Record ID for this observation
        """
        self._record_counter += 1
        record_id = f"witness_{self._record_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        record = {
            "id": record_id,
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "context": context,
            "judgment": {
                "voice": judgment.voice.name,
                "voice_level": judgment.voice.value,
                "concern_count": len(judgment.concerns),
                "summary": judgment.summary,
                "guidance": judgment.guidance,
            },
            "concerns": [
                {
                    "lens": c.lens,
                    "category": c.category,
                    "description": c.description,
                    "confidence": c.confidence,
                }
                for c in judgment.concerns
            ],
            "result": result,
        }

        self._records.append(record)

        # Prune old records if needed
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]

        logger.debug(f"Witnessed: {record_id} (voice={judgment.voice.name})")
        return record_id

    async def query(
        self,
        filters: dict[str, Any],
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Query past records.

        Args:
            filters: Query filters
                - voice: Filter by voice level (name or value)
                - action: Filter by action substring
                - category: Filter by concern category
                - start_time: ISO timestamp
                - end_time: ISO timestamp
            limit: Maximum records to return

        Returns:
            List of matching records (newest first)
        """
        results = []

        for record in reversed(self._records):
            # Check filters
            if "voice" in filters:
                target_voice = filters["voice"]
                if isinstance(target_voice, str):
                    if record["judgment"]["voice"] != target_voice:
                        continue
                else:
                    if record["judgment"]["voice_level"] != target_voice:
                        continue

            if "action" in filters:
                if filters["action"] not in record["action"]:
                    continue

            if "category" in filters:
                has_category = any(
                    filters["category"] in c["category"]
                    for c in record["concerns"]
                )
                if not has_category:
                    continue

            if "start_time" in filters:
                if record["timestamp"] < filters["start_time"]:
                    continue

            if "end_time" in filters:
                if record["timestamp"] > filters["end_time"]:
                    continue

            results.append(record)

            if len(results) >= limit:
                break

        return results

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about witnessed actions."""
        if not self._records:
            return {
                "total_records": 0,
                "voice_distribution": {},
                "concern_categories": {},
            }

        voice_counts = {}
        category_counts = {}

        for record in self._records:
            voice = record["judgment"]["voice"]
            voice_counts[voice] = voice_counts.get(voice, 0) + 1

            for concern in record["concerns"]:
                cat = concern["category"]
                category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_records": len(self._records),
            "voice_distribution": voice_counts,
            "concern_categories": category_counts,
        }


# =============================================================================
# Learner Implementation
# =============================================================================

class ThompsonLearner:
    """
    Thompson Sampling-based learner for conscience parameters.

    Learns optimal thresholds and responses based on outcomes.
    """

    def __init__(self, persist_path: Path | None = None):
        """
        Initialize Thompson learner.

        Args:
            persist_path: Optional path for persisting learned wisdom
        """
        self._wisdom: dict[str, Wisdom] = {}
        self._persist_path = persist_path

        # Load persisted wisdom if available
        if persist_path and persist_path.exists():
            self._load()

    async def update(
        self,
        pattern_id: str,
        success: bool,
        confidence: float = 1.0
    ) -> Wisdom:
        """
        Update a pattern based on outcome.

        Args:
            pattern_id: The pattern to update
            success: Whether the outcome was successful
            confidence: Confidence in this observation (0.0-1.0)

        Returns:
            Updated wisdom for this pattern
        """
        if pattern_id not in self._wisdom:
            self._wisdom[pattern_id] = Wisdom(
                pattern_id=pattern_id,
                description=f"Pattern: {pattern_id}",
                contexts=[],
            )

        wisdom = self._wisdom[pattern_id]

        if success:
            wisdom.update_success(confidence)
        else:
            wisdom.update_failure(confidence)

        logger.debug(
            f"Updated wisdom '{pattern_id}': "
            f"success_rate={wisdom.expected_success_rate:.2f}"
        )

        return wisdom

    async def get_wisdom(self, pattern_id: str) -> Wisdom | None:
        """Get learned wisdom for a pattern."""
        return self._wisdom.get(pattern_id)

    async def suggest_threshold(
        self,
        context: dict[str, Any]
    ) -> float:
        """
        Suggest a threshold based on learned patterns.

        Uses Thompson Sampling across relevant wisdom to suggest
        a threshold that balances safety and utility.

        Args:
            context: Context for the suggestion

        Returns:
            Suggested threshold (0.0-1.0)
        """
        if not self._wisdom:
            return 0.5  # Default

        # Find relevant wisdom based on context
        relevant = []
        context_str = str(context)

        for wisdom in self._wisdom.values():
            # Check if any wisdom contexts match
            if any(ctx in context_str for ctx in wisdom.contexts):
                relevant.append(wisdom)

        if not relevant:
            return 0.5

        # Thompson Sampling: Weight by expected reward
        total_weight = 0.0
        weighted_sum = 0.0

        for wisdom in relevant:
            # Sample from posterior
            import random
            sample = random.betavariate(wisdom.alpha, wisdom.beta)

            weight = wisdom.confidence
            weighted_sum += sample * weight
            total_weight += weight

        if total_weight == 0:
            return 0.5

        return weighted_sum / total_weight

    def _load(self) -> None:
        """Load wisdom from disk."""
        if not self._persist_path:
            return

        try:
            with open(self._persist_path) as f:
                data = json.load(f)

            for pattern_id, wisdom_data in data.items():
                self._wisdom[pattern_id] = Wisdom(
                    pattern_id=wisdom_data["pattern_id"],
                    description=wisdom_data["description"],
                    alpha=wisdom_data["alpha"],
                    beta=wisdom_data["beta"],
                    contexts=wisdom_data["contexts"],
                    confidence=wisdom_data["confidence"],
                )

            logger.info(f"Loaded {len(self._wisdom)} wisdom patterns")

        except Exception as e:
            logger.warning(f"Failed to load wisdom: {e}")

    def save(self) -> None:
        """Save wisdom to disk."""
        if not self._persist_path:
            return

        try:
            data = {}
            for pattern_id, wisdom in self._wisdom.items():
                data[pattern_id] = {
                    "pattern_id": wisdom.pattern_id,
                    "description": wisdom.description,
                    "alpha": wisdom.alpha,
                    "beta": wisdom.beta,
                    "contexts": wisdom.contexts,
                    "confidence": wisdom.confidence,
                }

            # Ensure parent directory exists
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self._persist_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self._wisdom)} wisdom patterns")

        except Exception as e:
            logger.warning(f"Failed to save wisdom: {e}")


# =============================================================================
# Main Conscience Class
# =============================================================================

class Conscience:
    """
    The Conscience - unified safety through composable concerns.

    Three verbs, like Memory has three verbs:
    - consider: Evaluate an action before execution
    - witness: Record an action and its outcome
    - learn: Improve from feedback

    Usage:
        async with Conscience() as conscience:
            judgment = await conscience.consider("execute_code", context)

            if judgment.voice <= Voice.VOICE:
                result = execute_code()
                await conscience.witness("execute_code", context, judgment, result)
            else:
                # Block or escalate
                pass

    Customization:
        # Custom lens combination
        conscience = Conscience(
            lens=HarmLens() | DeceptionLens(threshold=0.4),
            witness=CustomWitness(),
            learner=CustomLearner(),
        )
    """

    def __init__(
        self,
        lens: CompositeLens | ConscienceLens | None = None,
        witness: WitnessProtocol | None = None,
        learner: LearnerProtocol | None = None,
        persist_path: Path | None = None,
        auto_learn: bool = True,
    ):
        """
        Initialize conscience.

        Args:
            lens: Lens or composite lens for examination (default: standard())
            witness: Witness for recording (default: MemoryWitness)
            learner: Learner for improvement (default: ThompsonLearner)
            persist_path: Path for persisting learned wisdom
            auto_learn: Whether to auto-learn from witness records
        """
        # Use standard lens if none provided
        if lens is None:
            self._lens = standard()
        elif isinstance(lens, CompositeLens):
            self._lens = lens
        else:
            # Wrap single lens in composite
            self._lens = CompositeLens([lens])

        # Initialize witness
        self._witness = witness or MemoryWitness()

        # Initialize learner
        self._learner = learner or ThompsonLearner(persist_path)

        self._auto_learn = auto_learn
        self._initialized = False

    async def __aenter__(self) -> "Conscience":
        """Async context manager entry."""
        self._initialized = True
        logger.info(f"Conscience initialized with lens: {self._lens.name}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        # Save learned wisdom
        if isinstance(self._learner, ThompsonLearner):
            self._learner.save()

        logger.info("Conscience shutdown complete")

    # =========================================================================
    # Core Verbs
    # =========================================================================

    async def consider(
        self,
        action: str,
        context: dict[str, Any] | None = None
    ) -> Judgment:
        """
        Consider an action before execution.

        The core safety gate. Every action should pass through consider()
        before execution.

        Args:
            action: The action being considered
            context: Context for the action (user, session, etc.)

        Returns:
            Judgment with voice level and concerns

        Example:
            judgment = await conscience.consider("execute_code", {"code": "..."})
            if judgment.voice <= Voice.VOICE:
                # Safe to proceed
                result = execute()
            else:
                # Block or escalate
                logger.warning(judgment.guidance)
        """
        context = context or {}

        # Use composite lens judge() if available
        if hasattr(self._lens, "judge"):
            judgment = await self._lens.judge(action, context)
        else:
            # Fall back to examine only
            concerns = await self._lens.examine(action, context)
            judgment = self._concerns_to_judgment(concerns)

        logger.debug(
            f"Considered '{action[:50]}': voice={judgment.voice.name}, "
            f"concerns={len(judgment.concerns)}"
        )

        return judgment

    async def witness(
        self,
        action: str,
        context: dict[str, Any],
        judgment: Judgment,
        result: dict[str, Any] | None = None
    ) -> str:
        """
        Witness an action and its outcome.

        Records the action, judgment, and outcome for audit trail
        and learning purposes.

        Args:
            action: The action that was taken
            context: Context for the action
            judgment: The prior judgment
            result: The outcome of the action

        Returns:
            Record ID for this observation
        """
        record_id = await self._witness.record(action, context, judgment, result)

        # Auto-learn from outcome if enabled
        if self._auto_learn and result is not None:
            success = result.get("success", True)
            confidence = result.get("confidence", 0.8)

            # Learn for each concern that was raised
            for concern in judgment.concerns:
                pattern_id = f"{concern.lens}:{concern.category}"
                await self._learner.update(pattern_id, success, confidence)

            # Also learn at the action level
            pattern_id = f"action:{action[:50]}"
            await self._learner.update(pattern_id, success, confidence)

        return record_id

    async def learn(
        self,
        feedback: dict[str, Any]
    ) -> None:
        """
        Learn from feedback.

        Explicit learning signal for improving conscience accuracy.

        Args:
            feedback: Feedback on past judgments/actions
                - record_id: ID of witnessed record
                - correct: Was the judgment correct?
                - adjustment: How to adjust (optional)
        """
        record_id = feedback.get("record_id")
        correct = feedback.get("correct", True)
        confidence = feedback.get("confidence", 1.0)

        # Find the record
        records = await self._witness.query({"id": record_id}, limit=1)
        if not records:
            logger.warning(f"Record not found: {record_id}")
            return

        record = records[0]

        # Update learner for each concern
        for concern in record.get("concerns", []):
            pattern_id = f"{concern['lens']}:{concern['category']}"
            await self._learner.update(pattern_id, correct, confidence)

        # Update action-level pattern
        action = record.get("action", "")
        pattern_id = f"action:{action[:50]}"
        await self._learner.update(pattern_id, correct, confidence)

        logger.info(
            f"Learned from feedback on {record_id}: correct={correct}"
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _concerns_to_judgment(self, concerns: list[Concern]) -> Judgment:
        """Convert a list of concerns to a judgment."""
        if not concerns:
            return quiet_judgment()

        # Determine voice from max confidence
        max_confidence = max(c.confidence for c in concerns)

        if max_confidence >= 0.8:
            voice = Voice.SHOUT
        elif max_confidence >= 0.6:
            voice = Voice.VOICE
        elif max_confidence >= 0.3:
            voice = Voice.WHISPER
        else:
            voice = Voice.QUIET

        # Generate summary
        by_category: dict[str, int] = {}
        for concern in concerns:
            by_category[concern.category] = by_category.get(concern.category, 0) + 1

        parts = [f"{count} {cat}" for cat, count in sorted(by_category.items())]
        summary = ", ".join(parts)

        # Generate guidance
        if voice == Voice.QUIET:
            guidance = "Proceed normally"
        elif voice == Voice.WHISPER:
            guidance = "Proceed with monitoring"
        elif voice == Voice.VOICE:
            mitigations = [c.suggested_mitigation for c in concerns if c.suggested_mitigation]
            if mitigations:
                guidance = f"Request human review. Consider: {mitigations[0]}"
            else:
                guidance = "Request human review before proceeding"
        else:
            guidance = "Action blocked for safety. Do not proceed."

        return Judgment(
            voice=voice,
            concerns=concerns,
            summary=summary,
            guidance=guidance,
        )

    async def suggest_threshold(self, context: dict[str, Any]) -> float:
        """Suggest a threshold based on learned patterns."""
        return await self._learner.suggest_threshold(context)

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about conscience activity."""
        witness_stats = {}
        if isinstance(self._witness, MemoryWitness):
            witness_stats = self._witness.get_statistics()

        return {
            "lens": self._lens.name,
            "lens_count": len(self._lens) if hasattr(self._lens, "__len__") else 1,
            "witness": witness_stats,
            "auto_learn": self._auto_learn,
        }

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def is_safe(
        self,
        action: str,
        context: dict[str, Any] | None = None,
        max_voice: Voice = Voice.VOICE
    ) -> bool:
        """
        Quick check if an action is safe.

        Args:
            action: The action to check
            context: Context for the action
            max_voice: Maximum acceptable voice level

        Returns:
            True if action is safe (voice <= max_voice)
        """
        judgment = await self.consider(action, context)
        return judgment.voice <= max_voice

    async def gate(
        self,
        action: str,
        context: dict[str, Any] | None = None,
        execute_fn=None,
    ) -> dict[str, Any]:
        """
        Gate an action through conscience with automatic execution.

        Args:
            action: Description of the action
            context: Context for the action
            execute_fn: Async function to execute if safe

        Returns:
            {
                "allowed": bool,
                "judgment": Judgment,
                "result": Any (if executed),
            }
        """
        judgment = await self.consider(action, context)

        result = {
            "allowed": judgment.voice <= Voice.VOICE,
            "judgment": judgment,
            "result": None,
        }

        if result["allowed"] and execute_fn:
            try:
                execution_result = await execute_fn()
                result["result"] = {"success": True, "value": execution_result}

                # Witness the successful execution
                await self.witness(
                    action, context or {},
                    judgment, {"success": True}
                )

            except Exception as e:
                result["result"] = {"success": False, "error": str(e)}

                # Witness the failed execution
                await self.witness(
                    action, context or {},
                    judgment, {"success": False, "error": str(e)}
                )

        return result


# =============================================================================
# Factory Functions
# =============================================================================

def create_conscience(
    preset: str = "standard",
    persist_path: Path | None = None,
    auto_learn: bool = True,
) -> Conscience:
    """
    Create a conscience with a preset configuration.

    Args:
        preset: One of "standard", "paranoid", "research"
        persist_path: Path for persisting learned wisdom
        auto_learn: Whether to auto-learn from outcomes

    Returns:
        Configured Conscience instance
    """
    from .lenses import paranoid, research, standard

    presets = {
        "standard": standard,
        "paranoid": paranoid,
        "research": research,
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Use: {list(presets.keys())}")

    lens = presets[preset]()

    return Conscience(
        lens=lens,
        persist_path=persist_path,
        auto_learn=auto_learn,
    )


async def consider(
    action: str,
    context: dict[str, Any] | None = None,
    preset: str = "standard"
) -> Judgment:
    """
    Quick consideration without creating a full Conscience.

    For one-off safety checks. For repeated use, create a Conscience instance.

    Args:
        action: The action to consider
        context: Context for the action
        preset: Lens preset to use

    Returns:
        Judgment for the action
    """
    conscience = create_conscience(preset=preset, auto_learn=False)
    return await conscience.consider(action, context)
