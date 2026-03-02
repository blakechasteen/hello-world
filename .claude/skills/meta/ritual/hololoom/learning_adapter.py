"""
HoloLoom Learning Integration for Coding Ritual
===============================================

Uses Thompson Sampling to learn optimal ritual patterns:
- Which complexity level works best for which task types
- Which phases are most valuable for different scenarios
- Optimal refinement strategies based on past outcomes

This enables:
- Adaptive complexity recommendations
- Phase prioritization based on task
- Continuous improvement from session outcomes
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import random
import math

# Try to import HoloLoom learning systems
HOLOLOOM_LEARNING_AVAILABLE = False
try:
    hololoom_path = Path(__file__).parent.parent.parent.parent.parent.parent
    if str(hololoom_path) not in sys.path:
        sys.path.insert(0, str(hololoom_path))

    from hololoom.convergence.engine import ThompsonBandit
    HOLOLOOM_LEARNING_AVAILABLE = True
except ImportError:
    pass


@dataclass
class BetaPrior:
    """Beta distribution prior for Thompson Sampling."""

    alpha: float = 1.0  # Successes + 1
    beta: float = 1.0   # Failures + 1

    def update(self, success: bool, weight: float = 1.0) -> None:
        """Update prior with observation."""
        if success:
            self.alpha += weight
        else:
            self.beta += weight

    def sample(self) -> float:
        """Sample from Beta distribution."""
        # Use random.betavariate for simplicity
        return random.betavariate(self.alpha, self.beta)

    def expected_value(self) -> float:
        """Expected value (mean) of distribution."""
        return self.alpha / (self.alpha + self.beta)

    def confidence(self) -> float:
        """Confidence based on number of observations."""
        total = self.alpha + self.beta - 2  # Subtract initial priors
        return min(1.0, total / 20)  # Full confidence after 20 obs

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "BetaPrior":
        """Create from dictionary."""
        return cls(alpha=data.get("alpha", 1.0), beta=data.get("beta", 1.0))


@dataclass
class RitualLearningState:
    """State for ritual learning system."""

    # Complexity priors per task type
    complexity_priors: Dict[str, Dict[str, BetaPrior]] = field(default_factory=dict)

    # Phase value priors per complexity
    phase_priors: Dict[str, Dict[str, BetaPrior]] = field(default_factory=dict)

    # Build wave success priors
    wave_priors: Dict[int, BetaPrior] = field(default_factory=dict)

    # Check importance priors
    check_priors: Dict[str, BetaPrior] = field(default_factory=dict)

    # Total sessions tracked
    total_sessions: int = 0

    # Last update timestamp
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "complexity_priors": {
                task_type: {
                    complexity: prior.to_dict()
                    for complexity, prior in complexities.items()
                }
                for task_type, complexities in self.complexity_priors.items()
            },
            "phase_priors": {
                complexity: {
                    phase: prior.to_dict()
                    for phase, prior in phases.items()
                }
                for complexity, phases in self.phase_priors.items()
            },
            "wave_priors": {
                wave: prior.to_dict()
                for wave, prior in self.wave_priors.items()
            },
            "check_priors": {
                check_id: prior.to_dict()
                for check_id, prior in self.check_priors.items()
            },
            "total_sessions": self.total_sessions,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RitualLearningState":
        """Create state from dictionary."""
        state = cls()

        # Load complexity priors
        for task_type, complexities in data.get("complexity_priors", {}).items():
            state.complexity_priors[task_type] = {
                complexity: BetaPrior.from_dict(prior_data)
                for complexity, prior_data in complexities.items()
            }

        # Load phase priors
        for complexity, phases in data.get("phase_priors", {}).items():
            state.phase_priors[complexity] = {
                phase: BetaPrior.from_dict(prior_data)
                for phase, prior_data in phases.items()
            }

        # Load wave priors
        for wave_str, prior_data in data.get("wave_priors", {}).items():
            state.wave_priors[int(wave_str)] = BetaPrior.from_dict(prior_data)

        # Load check priors
        for check_id, prior_data in data.get("check_priors", {}).items():
            state.check_priors[check_id] = BetaPrior.from_dict(prior_data)

        state.total_sessions = data.get("total_sessions", 0)
        state.last_updated = data.get("last_updated")

        return state


class RitualLearningAdapter:
    """
    Adapter for learning optimal ritual patterns using Thompson Sampling.

    Learns:
    - Best complexity for task types
    - Most valuable phases per complexity
    - Build wave effectiveness
    - Check importance
    """

    COMPLEXITIES = ["lite", "fast", "full", "research"]
    PHASES = ["open", "design", "build", "check", "close"]
    TASK_TYPES = ["feature", "bugfix", "refactor", "docs", "research", "other"]

    def __init__(
        self,
        state_path: Optional[Path] = None,
        auto_save: bool = True,
    ):
        """
        Initialize learning adapter.

        Args:
            state_path: Path to persist learning state
            auto_save: Auto-save after updates
        """
        self.state_path = state_path or Path(".ritual/learning_state.json")
        self.auto_save = auto_save
        self.state = RitualLearningState()

        # Try to load existing state
        self._load_state()

        # Initialize default priors if empty
        self._initialize_priors()

    def _initialize_priors(self) -> None:
        """Initialize default priors."""
        # Complexity priors per task type
        for task_type in self.TASK_TYPES:
            if task_type not in self.state.complexity_priors:
                self.state.complexity_priors[task_type] = {
                    c: BetaPrior() for c in self.COMPLEXITIES
                }

        # Phase priors per complexity
        for complexity in self.COMPLEXITIES:
            if complexity not in self.state.phase_priors:
                self.state.phase_priors[complexity] = {
                    p: BetaPrior() for p in self.PHASES
                }

        # Wave priors
        for wave in [1, 2, 3]:
            if wave not in self.state.wave_priors:
                self.state.wave_priors[wave] = BetaPrior()

    def _load_state(self) -> None:
        """Load state from disk."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                self.state = RitualLearningState.from_dict(data)
            except Exception:
                pass

    def _save_state(self) -> None:
        """Save state to disk."""
        if self.auto_save:
            try:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.state_path, "w") as f:
                    json.dump(self.state.to_dict(), f, indent=2)
            except Exception:
                pass

    def classify_task(self, task: str) -> str:
        """
        Classify a task description into a task type.

        Args:
            task: Task description

        Returns:
            Task type (feature, bugfix, refactor, docs, research, other)
        """
        task_lower = task.lower()

        # Simple keyword matching
        if any(w in task_lower for w in ["fix", "bug", "error", "issue"]):
            return "bugfix"
        elif any(w in task_lower for w in ["refactor", "cleanup", "reorganize"]):
            return "refactor"
        elif any(w in task_lower for w in ["doc", "readme", "comment"]):
            return "docs"
        elif any(w in task_lower for w in ["research", "explore", "investigate"]):
            return "research"
        elif any(w in task_lower for w in ["add", "implement", "create", "build"]):
            return "feature"
        else:
            return "other"

    def recommend_complexity(
        self,
        task: str,
        use_thompson: bool = True,
    ) -> Tuple[str, float]:
        """
        Recommend complexity level for a task.

        Args:
            task: Task description
            use_thompson: Use Thompson Sampling (vs. expected value)

        Returns:
            (complexity, confidence) tuple
        """
        task_type = self.classify_task(task)
        priors = self.state.complexity_priors.get(task_type, {})

        if not priors:
            return ("fast", 0.5)  # Default

        if use_thompson:
            # Thompson Sampling: Sample from each prior
            samples = {c: priors[c].sample() for c in self.COMPLEXITIES}
            best = max(samples, key=samples.get)
            confidence = priors[best].confidence()
        else:
            # Expected value: Use mean
            expected = {c: priors[c].expected_value() for c in self.COMPLEXITIES}
            best = max(expected, key=expected.get)
            confidence = priors[best].confidence()

        return (best, confidence)

    def recommend_phases(
        self,
        complexity: str,
        limit: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Recommend phases to focus on for a complexity level.

        Args:
            complexity: Complexity level
            limit: Maximum phases to return

        Returns:
            List of (phase, expected_value) tuples, sorted by value
        """
        priors = self.state.phase_priors.get(complexity, {})

        if not priors:
            return [(p, 0.5) for p in self.PHASES[:limit]]

        # Get expected values
        expected = {p: priors[p].expected_value() for p in self.PHASES}

        # Sort by value and return top
        sorted_phases = sorted(expected.items(), key=lambda x: x[1], reverse=True)
        return sorted_phases[:limit]

    def update_from_session(
        self,
        task: str,
        complexity: str,
        success: bool,
        phases_completed: List[str],
        builds: Optional[List[Dict[str, Any]]] = None,
        checks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Update learning state from a completed session.

        Args:
            task: Task description
            complexity: Complexity used
            success: Whether session was successful
            phases_completed: Phases that were completed
            builds: Build wave results
            checks: Safety check results

        Returns:
            Update summary
        """
        task_type = self.classify_task(task)
        updates = {
            "task_type": task_type,
            "complexity": complexity,
            "success": success,
            "priors_updated": [],
        }

        # Update complexity prior for task type
        if task_type in self.state.complexity_priors:
            if complexity in self.state.complexity_priors[task_type]:
                self.state.complexity_priors[task_type][complexity].update(success)
                updates["priors_updated"].append(f"complexity:{task_type}:{complexity}")

        # Update phase priors
        if complexity in self.state.phase_priors:
            for phase in phases_completed:
                if phase in self.state.phase_priors[complexity]:
                    # Completed phases are treated as valuable if session succeeded
                    self.state.phase_priors[complexity][phase].update(success)
                    updates["priors_updated"].append(f"phase:{complexity}:{phase}")

        # Update wave priors
        if builds:
            for build in builds:
                wave = build.get("wave")
                wave_success = build.get("status") == "pass"
                if wave and wave in self.state.wave_priors:
                    self.state.wave_priors[wave].update(wave_success)
                    updates["priors_updated"].append(f"wave:{wave}")

        # Update check priors
        if checks:
            for check in checks:
                check_id = check.get("check_id")
                check_passed = check.get("passed", False)
                if check_id:
                    if check_id not in self.state.check_priors:
                        self.state.check_priors[check_id] = BetaPrior()
                    self.state.check_priors[check_id].update(check_passed)
                    updates["priors_updated"].append(f"check:{check_id}")

        # Update metadata
        self.state.total_sessions += 1
        self.state.last_updated = datetime.now().isoformat()

        # Save state
        self._save_state()

        return updates

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        stats = {
            "total_sessions": self.state.total_sessions,
            "last_updated": self.state.last_updated,
            "complexity_stats": {},
            "phase_stats": {},
            "wave_stats": {},
        }

        # Complexity stats per task type
        for task_type, priors in self.state.complexity_priors.items():
            stats["complexity_stats"][task_type] = {
                complexity: {
                    "expected_value": prior.expected_value(),
                    "confidence": prior.confidence(),
                    "alpha": prior.alpha,
                    "beta": prior.beta,
                }
                for complexity, prior in priors.items()
            }

        # Phase stats per complexity
        for complexity, priors in self.state.phase_priors.items():
            stats["phase_stats"][complexity] = {
                phase: {
                    "expected_value": prior.expected_value(),
                    "confidence": prior.confidence(),
                }
                for phase, prior in priors.items()
            }

        # Wave stats
        for wave, prior in self.state.wave_priors.items():
            stats["wave_stats"][wave] = {
                "expected_value": prior.expected_value(),
                "confidence": prior.confidence(),
            }

        return stats

    def get_recommendations(
        self,
        task: str,
    ) -> Dict[str, Any]:
        """
        Get comprehensive recommendations for a task.

        Args:
            task: Task description

        Returns:
            Recommendations including complexity, phases, warnings
        """
        task_type = self.classify_task(task)
        complexity, complexity_confidence = self.recommend_complexity(task)
        phases = self.recommend_phases(complexity)

        recommendations = {
            "task_type": task_type,
            "recommended_complexity": complexity,
            "complexity_confidence": complexity_confidence,
            "recommended_phases": [
                {"phase": p, "expected_value": v}
                for p, v in phases
            ],
            "warnings": [],
            "insights": [],
        }

        # Add warnings based on confidence
        if complexity_confidence < 0.3:
            recommendations["warnings"].append(
                "Low confidence in complexity recommendation - limited data"
            )

        # Add insights
        stats = self.get_statistics()
        if stats["total_sessions"] < 5:
            recommendations["insights"].append(
                f"Only {stats['total_sessions']} sessions tracked - recommendations will improve"
            )

        return recommendations


# Convenience functions
def recommend_ritual_complexity(task: str) -> Tuple[str, float]:
    """Get complexity recommendation for a task."""
    adapter = RitualLearningAdapter()
    return adapter.recommend_complexity(task)


def update_ritual_learning(
    task: str,
    complexity: str,
    success: bool,
    phases_completed: List[str],
) -> Dict[str, Any]:
    """Update learning from a completed session."""
    adapter = RitualLearningAdapter()
    return adapter.update_from_session(
        task=task,
        complexity=complexity,
        success=success,
        phases_completed=phases_completed,
    )


def get_ritual_recommendations(task: str) -> Dict[str, Any]:
    """Get comprehensive recommendations for a task."""
    adapter = RitualLearningAdapter()
    return adapter.get_recommendations(task)