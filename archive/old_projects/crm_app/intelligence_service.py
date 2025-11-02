"""
Intelligence Service

Clean, composable intelligence layer using strategy pattern.
Replaces the monolithic intelligence.py with elegant, testable design.
"""

from typing import List, Dict, Any, Optional, Tuple

from crm_app.models import Contact, Company, Deal, LeadScore, ActionRecommendation
from crm_app.protocols import IntelligenceService, CRMService
from crm_app.strategies import (
    WeightedFeatureScoringStrategy,
    EngagementBasedRecommendationStrategy,
    ActivityBasedPredictionStrategy,
    StrategyFactory
)


class CRMIntelligenceService:
    """
    Intelligence service with composable strategies

    Provides lead scoring, action recommendations, and deal predictions
    using configurable strategy objects.
    """

    def __init__(
        self,
        crm_service: CRMService,
        scoring_strategy: Optional[WeightedFeatureScoringStrategy] = None,
        recommendation_strategy: Optional[EngagementBasedRecommendationStrategy] = None,
        prediction_strategy: Optional[ActivityBasedPredictionStrategy] = None
    ):
        """
        Initialize intelligence service with CRM data source and strategies

        Args:
            crm_service: CRM data access service
            scoring_strategy: Strategy for lead scoring (default: weighted features)
            recommendation_strategy: Strategy for recommendations (default: engagement-based)
            prediction_strategy: Strategy for predictions (default: activity-based)
        """
        self.crm = crm_service

        # Use provided strategies or create defaults
        factory = StrategyFactory()
        self.scoring = scoring_strategy or factory.create_scoring_strategy()
        self.recommendations = recommendation_strategy or factory.create_recommendation_strategy()
        self.predictions = prediction_strategy or factory.create_prediction_strategy()

    # ========================================================================
    # Core Intelligence Operations
    # ========================================================================

    def score_lead(self, contact_id: str) -> LeadScore:
        """
        Score a lead by contact ID

        Returns:
            LeadScore with score, confidence, and reasoning
        """
        contact = self.crm.contacts.get(contact_id)
        if not contact:
            raise ValueError(f"Contact {contact_id} not found")

        # Get related data
        activities = self.crm.activities.get_for_contact(contact_id)
        deals = self.crm.deals.list({"contact_id": contact_id})
        company = self.crm.companies.get(contact.company_id) if contact.company_id else None

        # Score using strategy
        lead_score = self.scoring.score(contact, activities, deals, company)

        # Update contact with score (side effect - consider if this is desired)
        contact.lead_score = lead_score.score
        contact.engagement_level = lead_score.engagement_level

        return lead_score

    def recommend_action(self, contact_id: str) -> ActionRecommendation:
        """
        Recommend next action for a contact

        Returns:
            ActionRecommendation with action, priority, and reasoning
        """
        contact = self.crm.contacts.get(contact_id)
        if not contact:
            raise ValueError(f"Contact {contact_id} not found")

        activities = self.crm.activities.get_for_contact(contact_id)

        # Get or calculate lead score
        lead_score = None
        if contact.lead_score is not None:
            # Use cached score
            lead_score = LeadScore(
                contact_id=contact.id,
                score=contact.lead_score,
                confidence=0.8,
                engagement_level=contact.engagement_level or "warm",
                factors={},
                reasoning=""
            )

        return self.recommendations.recommend(contact, activities, lead_score)

    def predict_deal_success(self, deal_id: str) -> Dict[str, Any]:
        """
        Predict deal success probability

        Returns:
            Dict with probability, confidence, and reasoning
        """
        deal = self.crm.deals.get(deal_id)
        if not deal:
            raise ValueError(f"Deal {deal_id} not found")

        activities = self.crm.activities.list({"deal_id": deal_id})
        contact = self.crm.contacts.get(deal.contact_id)

        return self.predictions.predict(deal, activities, contact)

    # ========================================================================
    # Batch Operations
    # ========================================================================

    def score_all_leads(self, min_score: float = 0.0) -> List[Tuple[Contact, LeadScore]]:
        """
        Score all contacts and return sorted by score

        Args:
            min_score: Minimum score threshold (default: 0.0 for all)

        Returns:
            List of (Contact, LeadScore) tuples sorted by score descending
        """
        results = []

        for contact in self.crm.contacts.list():
            # Skip archived
            if "archived" in contact.tags:
                continue

            try:
                lead_score = self.score_lead(contact.id)

                if lead_score.score >= min_score:
                    results.append((contact, lead_score))
            except Exception as e:
                # Log error but continue (production-ready)
                print(f"Warning: Failed to score contact {contact.id}: {e}")
                continue

        # Sort by score descending
        results.sort(key=lambda x: x[1].score, reverse=True)
        return results

    def get_top_leads(self, limit: int = 20) -> List[Tuple[Contact, LeadScore]]:
        """Get highest-scoring leads"""
        return self.score_all_leads()[:limit]

    def get_daily_actions(self, limit: int = 20) -> List[Tuple[Contact, ActionRecommendation]]:
        """
        Get daily action recommendations

        Returns top priority actions sorted by priority.
        """
        recommendations = []

        for contact in self.crm.contacts.list():
            # Skip archived
            if "archived" in contact.tags:
                continue

            try:
                recommendation = self.recommend_action(contact.id)
                recommendations.append((contact, recommendation))
            except Exception as e:
                print(f"Warning: Failed to recommend action for contact {contact.id}: {e}")
                continue

        # Sort by priority descending
        recommendations.sort(key=lambda x: x[1].priority, reverse=True)
        return recommendations[:limit]

    def get_at_risk_deals(self, threshold: float = 0.4, limit: int = 10) -> List[Tuple[Deal, Dict[str, Any]]]:
        """
        Get deals at risk of being lost

        Args:
            threshold: Probability threshold below which deals are considered at risk
            limit: Maximum number of deals to return

        Returns:
            List of (Deal, prediction) tuples for at-risk deals
        """
        at_risk = []

        open_deals = self.crm.deals.list({"open_only": True})
        for deal in open_deals:
            try:
                prediction = self.predict_deal_success(deal.id)

                if prediction["probability"] < threshold:
                    at_risk.append((deal, prediction))
            except Exception as e:
                print(f"Warning: Failed to predict deal {deal.id}: {e}")
                continue

        # Sort by probability ascending (most at risk first)
        at_risk.sort(key=lambda x: x[1]["probability"])
        return at_risk[:limit]

    # ========================================================================
    # Comprehensive Insights
    # ========================================================================

    def get_insights(self) -> Dict[str, Any]:
        """
        Get comprehensive CRM insights dashboard

        Returns:
            Dict with top leads, daily actions, at-risk deals, and pipeline summary
        """
        # Top leads
        top_leads = self.get_top_leads(limit=10)

        # Daily recommendations
        daily_actions = self.get_daily_actions(limit=10)

        # At-risk deals
        at_risk_deals = self.get_at_risk_deals(limit=10)

        # Pipeline summary
        pipeline = self.crm.deals.get_pipeline_summary()

        return {
            "top_leads": [
                {"contact": c.to_dict(), "score": s.to_dict()}
                for c, s in top_leads
            ],
            "daily_actions": [
                {"contact": c.to_dict(), "recommendation": r.to_dict()}
                for c, r in daily_actions
            ],
            "at_risk_deals": [
                {"deal": d.to_dict(), "prediction": p}
                for d, p in at_risk_deals
            ],
            "pipeline": pipeline
        }

    # ========================================================================
    # Strategy Management (allows runtime strategy swapping)
    # ========================================================================

    def set_scoring_strategy(self, strategy: WeightedFeatureScoringStrategy) -> None:
        """Change the scoring strategy at runtime"""
        self.scoring = strategy

    def set_recommendation_strategy(self, strategy: EngagementBasedRecommendationStrategy) -> None:
        """Change the recommendation strategy at runtime"""
        self.recommendations = strategy

    def set_prediction_strategy(self, strategy: ActivityBasedPredictionStrategy) -> None:
        """Change the prediction strategy at runtime"""
        self.predictions = strategy
