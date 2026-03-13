"""
Thompson Sampling UX Learning System for Collaboration.

Learns optimal UX settings from user behavior using Bayesian Beta priors.
Adapts cursor visibility, voice defaults, presence indicators, and more.

Phase 3: Collaborative Intelligence - Adaptive UX.

Created: December 2025
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple
import logging
import random
import math
import json

from hololoom.bandits.beta_arm import BetaArm

logger = logging.getLogger(__name__)


class UXFeature(Enum):
    """UX features that can be optimized."""
    CURSOR_VISIBILITY = "cursor_visibility"      # Show/hide other cursors
    VOICE_DEFAULT = "voice_default"              # Mute/unmute on join
    PRESENCE_STYLE = "presence_style"            # Compact/detailed/minimal
    ANNOTATION_TYPE = "annotation_type"          # Default annotation type
    TYPING_INDICATOR = "typing_indicator"        # Show/hide typing indicators
    NOTIFICATION_LEVEL = "notification_level"    # Notification frequency
    COLLABORATION_MODE = "collaboration_mode"    # Real-time/async/hybrid
    WHITEBOARD_TOOLS = "whiteboard_tools"        # Default tool set


class PresenceStyle(Enum):
    """Presence indicator styles."""
    MINIMAL = "minimal"      # Just dot/name
    COMPACT = "compact"      # Name + status
    DETAILED = "detailed"    # Full activity info


class CollaborationMode(Enum):
    """Collaboration modes."""
    REALTIME = "realtime"    # Synchronous collaboration
    ASYNC = "async"          # Asynchronous (notifications)
    HYBRID = "hybrid"        # Context-dependent


class NotificationLevel(Enum):
    """Notification frequency levels."""
    ALL = "all"              # Every event
    IMPORTANT = "important"  # Mentions, replies, joins
    MINIMAL = "minimal"      # Just mentions
    NONE = "none"            # Disabled


@dataclass
class BetaPrior:
    """
    Beta distribution prior for Thompson Sampling.

    Delegates Beta math to BetaArm (canonical implementation).

    Beta(alpha, beta) represents belief about probability of success.
    - alpha: pseudo-count of successes (positive outcomes)
    - beta: pseudo-count of failures (negative outcomes)

    Higher alpha means higher expected success rate.
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

    @property
    def expected_value(self) -> float:
        """Expected success probability: E[X] = alpha / (alpha + beta)"""
        return self._arm.expected_value()

    @property
    def variance(self) -> float:
        """Variance of the Beta distribution."""
        return self._arm.variance()

    @property
    def confidence(self) -> float:
        """Confidence in the estimate (inverse of variance, normalized)."""
        return self._arm.confidence(saturation=50.0)

    def sample(self) -> float:
        """
        Sample from the Beta distribution (Thompson Sampling).

        Returns a random value representing believed success probability.
        """
        return self._arm.sample()

    def update(self, success: bool, weight: float = 1.0):
        """
        Update prior with observation.

        Args:
            success: Whether the action was successful
            weight: Weight of update (default 1.0)
        """
        self._arm.update(success, weight)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "expected_value": self.expected_value,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BetaPrior':
        return cls(_arm=BetaArm(alpha=data.get("alpha", 1.0), beta=data.get("beta", 1.0)))


@dataclass
class FeatureOption:
    """An option for a UX feature with its prior."""
    option_id: str
    option_name: str
    prior: BetaPrior = field(default_factory=BetaPrior)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "option_id": self.option_id,
            "option_name": self.option_name,
            "prior": self.prior.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureOption':
        return cls(
            option_id=data["option_id"],
            option_name=data["option_name"],
            prior=BetaPrior.from_dict(data.get("prior", {}))
        )


@dataclass
class LearningContext:
    """Context for a learning decision."""
    session_id: str
    user_id: str
    node_id: Optional[str] = None
    participant_count: int = 1
    session_duration_minutes: float = 0.0
    time_of_day: str = "day"  # morning, day, evening, night
    activity_level: str = "normal"  # low, normal, high
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_session(cls, session_id: str, user_id: str, **kwargs) -> 'LearningContext':
        """Create context from session parameters."""
        # Determine time of day
        hour = datetime.now().hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "day"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        return cls(
            session_id=session_id,
            user_id=user_id,
            time_of_day=time_of_day,
            **kwargs
        )


@dataclass
class LearningDecision:
    """A decision made by the learning system."""
    feature: UXFeature
    selected_option: str
    confidence: float
    exploration: bool  # True if this was an exploration vs exploitation
    context: Optional[LearningContext] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature.value,
            "selected_option": self.selected_option,
            "confidence": self.confidence,
            "exploration": self.exploration,
            "timestamp": self.timestamp.isoformat()
        }


class CollaborationUXLearner:
    """
    Thompson Sampling learner for collaboration UX optimization.

    Learns optimal settings for:
    - Cursor visibility (show/hide other users' cursors)
    - Voice defaults (mute/unmute on join)
    - Presence style (minimal/compact/detailed)
    - Annotation type suggestions
    - Notification levels
    - Collaboration mode

    Uses Beta priors that update based on user actions and feedback.
    """

    def __init__(
        self,
        exploration_rate: float = 0.1,
        decay_rate: float = 0.995,
        min_confidence_for_exploit: float = 0.3
    ):
        """
        Initialize the UX learner.

        Args:
            exploration_rate: Base probability of exploration (0.0-1.0)
            decay_rate: Decay rate for old observations (0.99 = 1% decay per update)
            min_confidence_for_exploit: Minimum confidence to prefer exploitation
        """
        self.exploration_rate = exploration_rate
        self.decay_rate = decay_rate
        self.min_confidence_for_exploit = min_confidence_for_exploit

        # Feature options with priors
        self.features: Dict[UXFeature, List[FeatureOption]] = self._init_features()

        # User-specific overrides
        self.user_priors: Dict[str, Dict[UXFeature, List[FeatureOption]]] = {}

        # Context-specific priors (e.g., by node type, time of day)
        self.context_priors: Dict[str, Dict[UXFeature, List[FeatureOption]]] = {}

        # Decision history for analysis
        self.decision_history: List[LearningDecision] = []

        # Feedback tracking
        self.pending_decisions: Dict[str, LearningDecision] = {}

    def _init_features(self) -> Dict[UXFeature, List[FeatureOption]]:
        """Initialize feature options with uniform priors."""
        return {
            UXFeature.CURSOR_VISIBILITY: [
                FeatureOption("show", "Show Cursors"),
                FeatureOption("hide", "Hide Cursors"),
                FeatureOption("focus_only", "Show When Focused"),
            ],
            UXFeature.VOICE_DEFAULT: [
                FeatureOption("muted", "Start Muted"),
                FeatureOption("unmuted", "Start Unmuted"),
                FeatureOption("push_to_talk", "Push to Talk"),
            ],
            UXFeature.PRESENCE_STYLE: [
                FeatureOption("minimal", "Minimal", BetaPrior(1.5, 1.0)),
                FeatureOption("compact", "Compact", BetaPrior(2.0, 1.0)),  # Slight preference
                FeatureOption("detailed", "Detailed"),
            ],
            UXFeature.ANNOTATION_TYPE: [
                FeatureOption("note", "Note", BetaPrior(2.0, 1.0)),
                FeatureOption("question", "Question"),
                FeatureOption("insight", "Insight"),
                FeatureOption("connection", "Connection"),
            ],
            UXFeature.TYPING_INDICATOR: [
                FeatureOption("show", "Show Typing", BetaPrior(2.0, 1.0)),
                FeatureOption("hide", "Hide Typing"),
            ],
            UXFeature.NOTIFICATION_LEVEL: [
                FeatureOption("all", "All Notifications"),
                FeatureOption("important", "Important Only", BetaPrior(2.0, 1.0)),
                FeatureOption("minimal", "Minimal"),
                FeatureOption("none", "None"),
            ],
            UXFeature.COLLABORATION_MODE: [
                FeatureOption("realtime", "Real-time", BetaPrior(2.0, 1.0)),
                FeatureOption("async", "Asynchronous"),
                FeatureOption("hybrid", "Hybrid"),
            ],
            UXFeature.WHITEBOARD_TOOLS: [
                FeatureOption("basic", "Basic Tools"),
                FeatureOption("standard", "Standard Set", BetaPrior(2.0, 1.0)),
                FeatureOption("advanced", "Advanced Tools"),
            ],
        }

    def select(
        self,
        feature: UXFeature,
        context: Optional[LearningContext] = None,
        force_exploration: bool = False
    ) -> Tuple[str, LearningDecision]:
        """
        Select an option for a feature using Thompson Sampling.

        Args:
            feature: The UX feature to select
            context: Optional context for personalization
            force_exploration: Force exploration (random selection)

        Returns:
            Tuple of (selected_option_id, decision)
        """
        options = self._get_options(feature, context)
        if not options:
            raise ValueError(f"No options for feature {feature}")

        # Decide exploration vs exploitation
        exploring = force_exploration or random.random() < self.exploration_rate

        if exploring:
            # Random exploration
            selected = random.choice(options)
            confidence = 0.0
        else:
            # Thompson Sampling: sample from each prior, pick highest
            samples = [(opt, opt.prior.sample()) for opt in options]
            selected, sample_value = max(samples, key=lambda x: x[1])

            # Confidence based on prior confidence
            confidence = selected.prior.confidence

            # If confidence too low, might still explore
            if confidence < self.min_confidence_for_exploit:
                if random.random() < (self.min_confidence_for_exploit - confidence):
                    selected = random.choice(options)
                    exploring = True
                    confidence = 0.0

        decision = LearningDecision(
            feature=feature,
            selected_option=selected.option_id,
            confidence=confidence,
            exploration=exploring,
            context=context
        )

        # Track for feedback
        decision_id = f"{feature.value}_{datetime.now().timestamp()}"
        self.pending_decisions[decision_id] = decision
        self.decision_history.append(decision)

        # Keep history bounded
        if len(self.decision_history) > 10000:
            self.decision_history = self.decision_history[-5000:]

        logger.debug(f"Selected {selected.option_id} for {feature.value} "
                     f"(explore={exploring}, conf={confidence:.2f})")

        return selected.option_id, decision

    def feedback(
        self,
        feature: UXFeature,
        option_id: str,
        success: bool,
        context: Optional[LearningContext] = None,
        weight: float = 1.0
    ):
        """
        Provide feedback on a selection.

        Args:
            feature: The feature that was selected
            option_id: The option that was used
            success: Whether the user had a positive experience
            context: Context for contextual learning
            weight: Weight of feedback (default 1.0)
        """
        # Update global priors
        options = self.features.get(feature, [])
        for opt in options:
            if opt.option_id == option_id:
                opt.prior.update(success, weight)
                break

        # Update user-specific priors if context provided
        if context and context.user_id:
            self._update_user_prior(context.user_id, feature, option_id, success, weight)

        # Update context-specific priors
        if context:
            context_key = self._get_context_key(context)
            if context_key:
                self._update_context_prior(context_key, feature, option_id, success, weight)

        # Apply decay to all options (forgetting old data)
        self._apply_decay(feature)

        logger.debug(f"Feedback for {feature.value}/{option_id}: "
                     f"success={success}, weight={weight}")

    def get_recommendation(
        self,
        feature: UXFeature,
        context: Optional[LearningContext] = None
    ) -> Dict[str, Any]:
        """
        Get a recommendation without committing to selection.

        Returns expected values and confidences for all options.
        """
        options = self._get_options(feature, context)

        recommendations = []
        for opt in options:
            recommendations.append({
                "option_id": opt.option_id,
                "option_name": opt.option_name,
                "expected_value": opt.prior.expected_value,
                "confidence": opt.prior.confidence,
                "alpha": opt.prior.alpha,
                "beta": opt.prior.beta
            })

        # Sort by expected value
        recommendations.sort(key=lambda x: x["expected_value"], reverse=True)

        return {
            "feature": feature.value,
            "recommendations": recommendations,
            "top_recommendation": recommendations[0]["option_id"] if recommendations else None
        }

    def get_all_recommendations(
        self,
        context: Optional[LearningContext] = None
    ) -> Dict[str, Any]:
        """Get recommendations for all features."""
        return {
            feature.value: self.get_recommendation(feature, context)
            for feature in UXFeature
        }

    def get_session_defaults(
        self,
        context: LearningContext
    ) -> Dict[str, str]:
        """
        Get recommended defaults for a new session.

        Returns dict mapping feature names to recommended option IDs.
        """
        defaults = {}
        for feature in UXFeature:
            option_id, _ = self.select(feature, context)
            defaults[feature.value] = option_id

        return defaults

    def infer_feedback_from_action(
        self,
        action: str,
        context: Optional[LearningContext] = None
    ) -> List[Tuple[UXFeature, str, bool]]:
        """
        Infer feedback from user actions.

        Maps user actions to feature feedback:
        - "changed_cursor_visibility" -> negative feedback for current setting
        - "dismissed_notification" -> negative for notification level
        - "used_voice" -> positive for voice default
        - etc.

        Returns list of (feature, option_id, success) tuples.
        """
        inferences = []

        action_mappings = {
            # Cursor actions
            "hid_cursors": (UXFeature.CURSOR_VISIBILITY, "hide", True),
            "showed_cursors": (UXFeature.CURSOR_VISIBILITY, "show", True),
            "toggled_cursor": None,  # Negative feedback for current setting

            # Voice actions
            "unmuted_self": (UXFeature.VOICE_DEFAULT, "unmuted", True),
            "muted_self": (UXFeature.VOICE_DEFAULT, "muted", True),
            "used_voice": (UXFeature.VOICE_DEFAULT, "unmuted", True),
            "enabled_push_to_talk": (UXFeature.VOICE_DEFAULT, "push_to_talk", True),

            # Annotation actions
            "created_note": (UXFeature.ANNOTATION_TYPE, "note", True),
            "created_question": (UXFeature.ANNOTATION_TYPE, "question", True),
            "created_insight": (UXFeature.ANNOTATION_TYPE, "insight", True),
            "created_connection": (UXFeature.ANNOTATION_TYPE, "connection", True),

            # Notification actions
            "dismissed_notification": (UXFeature.NOTIFICATION_LEVEL, "minimal", True),
            "clicked_notification": (UXFeature.NOTIFICATION_LEVEL, "all", True),
            "disabled_notifications": (UXFeature.NOTIFICATION_LEVEL, "none", True),

            # Presence actions
            "expanded_presence": (UXFeature.PRESENCE_STYLE, "detailed", True),
            "collapsed_presence": (UXFeature.PRESENCE_STYLE, "minimal", True),

            # Mode actions
            "enabled_realtime": (UXFeature.COLLABORATION_MODE, "realtime", True),
            "enabled_async": (UXFeature.COLLABORATION_MODE, "async", True),
        }

        mapping = action_mappings.get(action)
        if mapping:
            feature, option_id, success = mapping
            inferences.append((feature, option_id, success))
            self.feedback(feature, option_id, success, context)

        return inferences

    def _get_options(
        self,
        feature: UXFeature,
        context: Optional[LearningContext] = None
    ) -> List[FeatureOption]:
        """Get options for a feature, considering context."""
        # Start with global options
        global_options = self.features.get(feature, [])

        if not context:
            return global_options

        # Blend with user-specific priors if available
        user_options = self._get_user_options(context.user_id, feature)
        if user_options:
            # Blend priors: 70% user-specific, 30% global
            return self._blend_options(user_options, global_options, user_weight=0.7)

        # Blend with context-specific priors
        context_key = self._get_context_key(context)
        if context_key:
            context_options = self._get_context_options(context_key, feature)
            if context_options:
                return self._blend_options(context_options, global_options, user_weight=0.5)

        return global_options

    def _get_user_options(self, user_id: str, feature: UXFeature) -> Optional[List[FeatureOption]]:
        """Get user-specific options if available."""
        user_priors = self.user_priors.get(user_id, {})
        return user_priors.get(feature)

    def _get_context_options(self, context_key: str, feature: UXFeature) -> Optional[List[FeatureOption]]:
        """Get context-specific options if available."""
        context_priors = self.context_priors.get(context_key, {})
        return context_priors.get(feature)

    def _get_context_key(self, context: LearningContext) -> Optional[str]:
        """Generate a context key for context-specific learning."""
        # Use time of day and activity level as context
        if context.time_of_day and context.activity_level:
            return f"{context.time_of_day}_{context.activity_level}"
        return None

    def _blend_options(
        self,
        primary: List[FeatureOption],
        secondary: List[FeatureOption],
        user_weight: float = 0.7
    ) -> List[FeatureOption]:
        """Blend two option lists by their priors."""
        blended = []

        for p_opt in primary:
            # Find matching secondary option
            s_opt = next((s for s in secondary if s.option_id == p_opt.option_id), None)

            if s_opt:
                # Blend the priors
                blended_alpha = p_opt.prior.alpha * user_weight + s_opt.prior.alpha * (1 - user_weight)
                blended_beta = p_opt.prior.beta * user_weight + s_opt.prior.beta * (1 - user_weight)

                blended.append(FeatureOption(
                    option_id=p_opt.option_id,
                    option_name=p_opt.option_name,
                    prior=BetaPrior(alpha=blended_alpha, beta=blended_beta)
                ))
            else:
                blended.append(p_opt)

        return blended

    def _update_user_prior(
        self,
        user_id: str,
        feature: UXFeature,
        option_id: str,
        success: bool,
        weight: float
    ):
        """Update user-specific prior."""
        if user_id not in self.user_priors:
            # Initialize from global
            self.user_priors[user_id] = {}

        if feature not in self.user_priors[user_id]:
            # Copy from global
            self.user_priors[user_id][feature] = [
                FeatureOption(
                    option_id=opt.option_id,
                    option_name=opt.option_name,
                    prior=BetaPrior(opt.prior.alpha, opt.prior.beta)
                )
                for opt in self.features.get(feature, [])
            ]

        # Update the specific option
        for opt in self.user_priors[user_id][feature]:
            if opt.option_id == option_id:
                opt.prior.update(success, weight)
                break

    def _update_context_prior(
        self,
        context_key: str,
        feature: UXFeature,
        option_id: str,
        success: bool,
        weight: float
    ):
        """Update context-specific prior."""
        if context_key not in self.context_priors:
            self.context_priors[context_key] = {}

        if feature not in self.context_priors[context_key]:
            # Copy from global
            self.context_priors[context_key][feature] = [
                FeatureOption(
                    option_id=opt.option_id,
                    option_name=opt.option_name,
                    prior=BetaPrior(opt.prior.alpha, opt.prior.beta)
                )
                for opt in self.features.get(feature, [])
            ]

        # Update the specific option
        for opt in self.context_priors[context_key][feature]:
            if opt.option_id == option_id:
                opt.prior.update(success, weight)
                break

    def _apply_decay(self, feature: UXFeature):
        """Apply decay to priors (forgetting old data)."""
        options = self.features.get(feature, [])
        for opt in options:
            # Decay towards prior (1, 1)
            opt.prior.alpha = 1.0 + (opt.prior.alpha - 1.0) * self.decay_rate
            opt.prior.beta = 1.0 + (opt.prior.beta - 1.0) * self.decay_rate

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        stats = {
            "total_decisions": len(self.decision_history),
            "exploration_decisions": sum(1 for d in self.decision_history if d.exploration),
            "features": {}
        }

        for feature, options in self.features.items():
            stats["features"][feature.value] = {
                "options": [opt.to_dict() for opt in options],
                "top_option": max(options, key=lambda o: o.prior.expected_value).option_id
            }

        stats["user_count"] = len(self.user_priors)
        stats["context_count"] = len(self.context_priors)

        return stats

    def save_state(self, path: str):
        """Save learning state to file."""
        state = {
            "exploration_rate": self.exploration_rate,
            "decay_rate": self.decay_rate,
            "min_confidence_for_exploit": self.min_confidence_for_exploit,
            "features": {
                f.value: [opt.to_dict() for opt in opts]
                for f, opts in self.features.items()
            },
            "user_priors": {
                user_id: {
                    f.value: [opt.to_dict() for opt in opts]
                    for f, opts in priors.items()
                }
                for user_id, priors in self.user_priors.items()
            },
            "context_priors": {
                ctx: {
                    f.value: [opt.to_dict() for opt in opts]
                    for f, opts in priors.items()
                }
                for ctx, priors in self.context_priors.items()
            }
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved UX learning state to {path}")

    def load_state(self, path: str):
        """Load learning state from file."""
        try:
            with open(path, 'r') as f:
                state = json.load(f)

            self.exploration_rate = state.get("exploration_rate", 0.1)
            self.decay_rate = state.get("decay_rate", 0.995)
            self.min_confidence_for_exploit = state.get("min_confidence_for_exploit", 0.3)

            # Load feature priors
            for feature_str, opts_data in state.get("features", {}).items():
                try:
                    feature = UXFeature(feature_str)
                    self.features[feature] = [
                        FeatureOption.from_dict(opt_data) for opt_data in opts_data
                    ]
                except ValueError:
                    pass  # Skip unknown features

            logger.info(f"Loaded UX learning state from {path}")

        except FileNotFoundError:
            logger.warning(f"No learning state found at {path}")
        except Exception as e:
            logger.error(f"Error loading learning state: {e}")


# Factory function
def create_ux_learner(
    exploration_rate: float = 0.1,
    state_path: Optional[str] = None
) -> CollaborationUXLearner:
    """
    Create a UX learner with optional state loading.

    Args:
        exploration_rate: Base exploration rate (0.0-1.0)
        state_path: Optional path to load previous state

    Returns:
        Configured CollaborationUXLearner
    """
    learner = CollaborationUXLearner(exploration_rate=exploration_rate)

    if state_path:
        learner.load_state(state_path)

    return learner
