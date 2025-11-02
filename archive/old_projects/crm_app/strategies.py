"""
Intelligence Strategies

Composable, testable strategies for lead scoring, recommendations, and predictions.
Each strategy is independent and follows the Strategy pattern for easy swapping.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from crm_app.models import (
    Contact, Company, Deal, Activity,
    LeadScore, ActionRecommendation,
    DealStage, ActivityOutcome
)


# ============================================================================
# Feature Extraction (shared utilities)
# ============================================================================

class FeatureExtractor:
    """Extract scoring features from contacts and activities"""

    @staticmethod
    def engagement_frequency(activities: List[Activity]) -> float:
        """Calculate engagement frequency (0-1)"""
        if not activities:
            return 0.0

        days_active = (datetime.utcnow() - min(a.timestamp for a in activities)).days
        return min(1.0, len(activities) / max(1, days_active / 7))

    @staticmethod
    def recency_score(last_contact: Optional[datetime], decay_days: int = 90) -> float:
        """Calculate recency score with exponential decay"""
        if not last_contact:
            return 0.0

        days_since = (datetime.utcnow() - last_contact).days
        return max(0.0, 1.0 - (days_since / decay_days))

    @staticmethod
    def sentiment_score(activities: List[Activity]) -> float:
        """Calculate average sentiment from activities"""
        sentiment_values = []
        for activity in activities:
            if activity.outcome == ActivityOutcome.POSITIVE:
                sentiment_values.append(1.0)
            elif activity.outcome == ActivityOutcome.NEUTRAL:
                sentiment_values.append(0.5)
            elif activity.outcome == ActivityOutcome.NEGATIVE:
                sentiment_values.append(0.0)

        return np.mean(sentiment_values) if sentiment_values else 0.5

    @staticmethod
    def response_rate(activities: List[Activity]) -> float:
        """Calculate response rate from outreach activities"""
        outreach = [a for a in activities if a.type.value in ("email", "call")]
        if not outreach:
            return 0.5  # Neutral default

        positive = [a for a in outreach if a.outcome == ActivityOutcome.POSITIVE]
        return len(positive) / len(outreach)


# ============================================================================
# Scoring Strategies
# ============================================================================

class WeightedFeatureScoringStrategy:
    """
    Weighted feature-based lead scoring

    Uses multiple features with configurable weights for interpretable scoring.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "engagement_frequency": 0.25,
            "recency": 0.20,
            "activity_sentiment": 0.20,
            "deal_value": 0.15,
            "response_rate": 0.10,
            "company_fit": 0.10
        }
        self.extractor = FeatureExtractor()

    def score(
        self,
        contact: Contact,
        activities: List[Activity],
        deals: Optional[List[Deal]] = None,
        company: Optional[Company] = None
    ) -> LeadScore:
        """Score lead using weighted feature combination"""

        # Extract features
        features = {
            "engagement_frequency": self.extractor.engagement_frequency(activities),
            "recency": self.extractor.recency_score(contact.last_contact),
            "activity_sentiment": self.extractor.sentiment_score(activities),
            "deal_value": self._deal_value_score(deals),
            "response_rate": self.extractor.response_rate(activities),
            "company_fit": self._company_fit_score(company)
        }

        # Calculate weighted score
        score = sum(features[k] * self.weights[k] for k in self.weights.keys())

        # Determine engagement level
        engagement_level = self._classify_engagement(score)

        # Build reasoning
        reasoning = self._build_reasoning(features)

        # Confidence based on data availability
        confidence = min(1.0, len(activities) / 10.0)

        return LeadScore(
            contact_id=contact.id,
            score=score,
            confidence=confidence,
            engagement_level=engagement_level,
            factors=features,
            reasoning=reasoning
        )

    def _deal_value_score(self, deals: Optional[List[Deal]]) -> float:
        """Score based on deal value"""
        if not deals:
            return 0.0

        open_deals = [d for d in deals if not d.is_closed]
        total_value = sum(d.value for d in open_deals)
        return min(1.0, total_value / 50000.0)  # Normalize to 50k

    def _company_fit_score(self, company: Optional[Company]) -> float:
        """Score based on company characteristics"""
        if not company:
            return 0.5  # Neutral default

        size_scores = {
            "1-10": 0.3,
            "11-50": 0.5,
            "51-200": 0.7,
            "201-1000": 0.85,
            "1000+": 1.0
        }
        return size_scores.get(company.size.value, 0.5)

    def _classify_engagement(self, score: float) -> str:
        """Classify engagement level from score"""
        if score >= 0.75:
            return "hot"
        elif score >= 0.50:
            return "warm"
        elif score >= 0.25:
            return "cold"
        else:
            return "dead"

    def _build_reasoning(self, features: Dict[str, float]) -> str:
        """Build human-readable reasoning"""
        # Sort by contribution
        contributions = [
            (k, features[k] * self.weights[k])
            for k in self.weights.keys()
        ]
        contributions.sort(key=lambda x: x[1], reverse=True)

        # Top 3 factors
        parts = []
        for factor, contribution in contributions[:3]:
            factor_name = factor.replace('_', ' ').title()
            factor_value = features[factor]
            parts.append(f"{factor_name}: {factor_value:.2f}")

        return ". ".join(parts)


# ============================================================================
# Recommendation Strategies
# ============================================================================

class EngagementBasedRecommendationStrategy:
    """
    Engagement-based action recommendations

    Recommends actions based on engagement level and recent activity patterns.
    """

    def recommend(
        self,
        contact: Contact,
        activities: List[Activity],
        lead_score: Optional[LeadScore] = None
    ) -> ActionRecommendation:
        """Recommend next action based on engagement"""

        # Get or calculate engagement level
        if lead_score:
            engagement_level = lead_score.engagement_level
        else:
            # Quick estimation
            if contact.lead_score and contact.lead_score >= 0.75:
                engagement_level = "hot"
            elif contact.lead_score and contact.lead_score >= 0.50:
                engagement_level = "warm"
            else:
                engagement_level = "cold"

        # Get most recent activity
        recent_activities = sorted(activities, key=lambda a: a.timestamp, reverse=True)
        last_activity = recent_activities[0] if recent_activities else None

        # Decision logic
        if engagement_level == "hot":
            return self._recommend_for_hot_lead(contact, last_activity)
        elif engagement_level == "warm":
            return self._recommend_for_warm_lead(contact, last_activity)
        elif engagement_level == "cold":
            return self._recommend_for_cold_lead(contact, last_activity)
        else:  # dead
            return self._recommend_for_dead_lead(contact)

    def _recommend_for_hot_lead(
        self,
        contact: Contact,
        last_activity: Optional[Activity]
    ) -> ActionRecommendation:
        """Recommendations for hot leads"""
        if last_activity and (datetime.utcnow() - last_activity.timestamp).days < 3:
            return ActionRecommendation(
                contact_id=contact.id,
                action="send_proposal",
                priority=0.9,
                reasoning="Hot lead with recent engagement. Time to send proposal.",
                expected_outcome="High probability of conversion",
                suggested_timing=datetime.utcnow() + timedelta(hours=24)
            )
        else:
            return ActionRecommendation(
                contact_id=contact.id,
                action="schedule_call",
                priority=0.85,
                reasoning="Hot lead but no recent activity. Schedule call to re-engage.",
                expected_outcome="Re-establish momentum",
                suggested_timing=datetime.utcnow() + timedelta(hours=48)
            )

    def _recommend_for_warm_lead(
        self,
        contact: Contact,
        last_activity: Optional[Activity]
    ) -> ActionRecommendation:
        """Recommendations for warm leads"""
        if last_activity:
            days_since = (datetime.utcnow() - last_activity.timestamp).days
            if days_since < 7:
                return ActionRecommendation(
                    contact_id=contact.id,
                    action="wait",
                    priority=0.3,
                    reasoning="Recent activity. Give them space.",
                    expected_outcome="Maintain relationship",
                    suggested_timing=datetime.utcnow() + timedelta(days=7)
                )
            else:
                return ActionRecommendation(
                    contact_id=contact.id,
                    action="send_email",
                    priority=0.6,
                    reasoning="Warm lead, time for check-in email.",
                    expected_outcome="Keep conversation alive",
                    suggested_timing=datetime.utcnow() + timedelta(hours=24)
                )
        else:
            return ActionRecommendation(
                contact_id=contact.id,
                action="send_email",
                priority=0.5,
                reasoning="Warm lead with no activity. Start conversation.",
                expected_outcome="Initial engagement",
                suggested_timing=datetime.utcnow()
            )

    def _recommend_for_cold_lead(
        self,
        contact: Contact,
        last_activity: Optional[Activity]
    ) -> ActionRecommendation:
        """Recommendations for cold leads"""
        return ActionRecommendation(
            contact_id=contact.id,
            action="send_email",
            priority=0.3,
            reasoning="Cold lead. Low-effort email to gauge interest.",
            expected_outcome="Low probability response",
            suggested_timing=datetime.utcnow() + timedelta(days=7)
        )

    def _recommend_for_dead_lead(self, contact: Contact) -> ActionRecommendation:
        """Recommendations for dead leads"""
        return ActionRecommendation(
            contact_id=contact.id,
            action="wait",
            priority=0.1,
            reasoning="Dead lead. Consider archiving or long-term nurture.",
            expected_outcome="Minimal activity expected",
            suggested_timing=datetime.utcnow() + timedelta(days=30)
        )


# ============================================================================
# Prediction Strategies
# ============================================================================

class ActivityBasedPredictionStrategy:
    """
    Activity-based deal prediction

    Predicts deal success based on activity patterns and engagement signals.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "activity_count": 0.20,
            "positive_ratio": 0.25,
            "stage_progress": 0.25,
            "pipeline_age": 0.15,
            "contact_quality": 0.15
        }

    def predict(
        self,
        deal: Deal,
        activities: List[Activity],
        contact: Optional[Contact] = None
    ) -> Dict[str, Any]:
        """Predict deal success probability"""

        # Extract features
        features = {
            "activity_count": min(1.0, len(activities) / 20.0),
            "positive_ratio": self._positive_ratio(activities),
            "stage_progress": self._stage_progress(deal.stage),
            "pipeline_age": self._pipeline_age_score(deal.created_at),
            "contact_quality": contact.lead_score if contact and contact.lead_score else 0.5
        }

        # Calculate probability
        probability = sum(features[k] * self.weights[k] for k in self.weights.keys())

        # Confidence based on data
        confidence = min(1.0, len(activities) / 10.0)

        # Build reasoning
        reasoning = self._build_reasoning(features, activities)

        return {
            "deal_id": deal.id,
            "probability": probability,
            "confidence": confidence,
            "features": features,
            "reasoning": reasoning
        }

    def _positive_ratio(self, activities: List[Activity]) -> float:
        """Calculate ratio of positive activities"""
        if not activities:
            return 0.5

        positive = [a for a in activities if a.outcome == ActivityOutcome.POSITIVE]
        return len(positive) / len(activities)

    def _stage_progress(self, stage: DealStage) -> float:
        """Score based on pipeline stage"""
        stage_values = {
            DealStage.LEAD: 0.1,
            DealStage.QUALIFIED: 0.3,
            DealStage.PROPOSAL: 0.5,
            DealStage.NEGOTIATION: 0.7
        }
        return stage_values.get(stage, 0.5)

    def _pipeline_age_score(self, created_at: datetime) -> float:
        """Score based on time in pipeline (older = less likely)"""
        days_open = (datetime.utcnow() - created_at).days
        return 1.0 - min(1.0, days_open / 180.0)

    def _build_reasoning(self, features: Dict[str, float], activities: List[Activity]) -> str:
        """Build human-readable reasoning"""
        parts = []

        if features["positive_ratio"] > 0.7:
            parts.append("Strong positive engagement")
        if features["activity_count"] > 0.6:
            parts.append("High activity volume")
        if features["pipeline_age"] < 0.5:
            parts.append("Deal aging in pipeline")

        return ". ".join(parts) if parts else "Limited data available"


# ============================================================================
# Strategy Factory
# ============================================================================

class StrategyFactory:
    """Factory for creating intelligence strategies with optional configuration"""

    @staticmethod
    def create_scoring_strategy(strategy_type: str = "weighted") -> WeightedFeatureScoringStrategy:
        """Create a scoring strategy"""
        if strategy_type == "weighted":
            return WeightedFeatureScoringStrategy()
        else:
            raise ValueError(f"Unknown scoring strategy: {strategy_type}")

    @staticmethod
    def create_recommendation_strategy(
        strategy_type: str = "engagement"
    ) -> EngagementBasedRecommendationStrategy:
        """Create a recommendation strategy"""
        if strategy_type == "engagement":
            return EngagementBasedRecommendationStrategy()
        else:
            raise ValueError(f"Unknown recommendation strategy: {strategy_type}")

    @staticmethod
    def create_prediction_strategy(
        strategy_type: str = "activity"
    ) -> ActivityBasedPredictionStrategy:
        """Create a prediction strategy"""
        if strategy_type == "activity":
            return ActivityBasedPredictionStrategy()
        else:
            raise ValueError(f"Unknown prediction strategy: {strategy_type}")
