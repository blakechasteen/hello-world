"""
CRM Intelligence Module

Provides intelligent lead scoring, action recommendations, and predictions
using HoloLoom's policy engine and decision-making capabilities.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

from crm_app.models import (
    Contact, Company, Deal, Activity, LeadScore, ActionRecommendation,
    DealStage, ActivityOutcome
)
from crm_app.storage import CRMStorage
from HoloLoom.documentation.types import Query, Features
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config


class LeadScorer:
    """Intelligent lead scoring using feature extraction and policy engine"""

    def __init__(self, storage: CRMStorage):
        self.storage = storage

    def extract_features(self, contact: Contact) -> Dict[str, float]:
        """
        Extract scoring features from contact and activity history

        Features:
        - engagement_frequency: How often they interact (0-1)
        - recency: Time since last contact (0-1)
        - activity_sentiment: Avg sentiment of interactions (0-1)
        - deal_value: Total value of associated deals (0-1)
        - response_rate: How often they respond (0-1)
        - company_fit: Company size/industry alignment (0-1)
        """
        features = {}

        # Engagement frequency
        activities = self.storage.get_contact_activities(contact.id, limit=100)
        if activities:
            days_active = (datetime.utcnow() - min(a.timestamp for a in activities)).days
            features["engagement_frequency"] = min(1.0, len(activities) / max(1, days_active / 7))
        else:
            features["engagement_frequency"] = 0.0

        # Recency
        if contact.last_contact:
            days_since = (datetime.utcnow() - contact.last_contact).days
            features["recency"] = max(0.0, 1.0 - (days_since / 90.0))  # Decay over 90 days
        else:
            features["recency"] = 0.0

        # Activity sentiment
        sentiment_scores = []
        for activity in activities:
            if activity.outcome == ActivityOutcome.POSITIVE:
                sentiment_scores.append(1.0)
            elif activity.outcome == ActivityOutcome.NEUTRAL:
                sentiment_scores.append(0.5)
            elif activity.outcome == ActivityOutcome.NEGATIVE:
                sentiment_scores.append(0.0)

        if sentiment_scores:
            features["activity_sentiment"] = np.mean(sentiment_scores)
        else:
            features["activity_sentiment"] = 0.5  # Neutral default

        # Deal value
        contact_deals = self.storage.list_deals({"contact_id": contact.id, "open_only": True})
        total_value = sum(d.value for d in contact_deals)
        features["deal_value"] = min(1.0, total_value / 50000.0)  # Normalize to 50k

        # Response rate (simplified - count emails/calls with positive outcome)
        outreach_activities = [a for a in activities if a.type.value in ("email", "call")]
        positive_responses = [a for a in outreach_activities if a.outcome == ActivityOutcome.POSITIVE]
        if outreach_activities:
            features["response_rate"] = len(positive_responses) / len(outreach_activities)
        else:
            features["response_rate"] = 0.5  # Neutral default

        # Company fit
        if contact.company_id and contact.company_id in self.storage.companies:
            company = self.storage.companies[contact.company_id]

            # Size score (larger = better)
            size_scores = {
                "1-10": 0.3,
                "11-50": 0.5,
                "51-200": 0.7,
                "201-1000": 0.85,
                "1000+": 1.0
            }
            features["company_fit"] = size_scores.get(company.size.value, 0.5)
        else:
            features["company_fit"] = 0.5  # Neutral default

        return features

    def score_lead(self, contact: Contact) -> LeadScore:
        """
        Calculate lead score with weighted feature combination

        Returns LeadScore with explanation
        """
        features = self.extract_features(contact)

        # Weighted scoring
        weights = {
            "engagement_frequency": 0.25,
            "recency": 0.20,
            "activity_sentiment": 0.20,
            "deal_value": 0.15,
            "response_rate": 0.10,
            "company_fit": 0.10
        }

        score = sum(features[k] * weights[k] for k in weights.keys())

        # Determine engagement level
        if score >= 0.75:
            engagement_level = "hot"
        elif score >= 0.50:
            engagement_level = "warm"
        elif score >= 0.25:
            engagement_level = "cold"
        else:
            engagement_level = "dead"

        # Build reasoning
        top_factors = sorted(features.items(), key=lambda x: x[1] * weights[x[0]], reverse=True)[:3]
        reasoning_parts = []
        for factor, value in top_factors:
            contribution = value * weights[factor]
            reasoning_parts.append(f"{factor.replace('_', ' ')}: {value:.2f} (contributes {contribution:.2f})")

        reasoning = ". ".join(reasoning_parts)

        # Confidence based on data availability
        activities = self.storage.get_contact_activities(contact.id)
        confidence = min(1.0, len(activities) / 10.0)  # More data = higher confidence

        return LeadScore(
            contact_id=contact.id,
            score=score,
            confidence=confidence,
            engagement_level=engagement_level,
            factors=features,
            reasoning=reasoning
        )

    def score_all_leads(self, min_score: float = 0.0) -> List[Tuple[Contact, LeadScore]]:
        """Score all contacts and return sorted by score"""
        results = []

        for contact in self.storage.contacts.values():
            if "archived" in contact.tags:
                continue

            lead_score = self.score_lead(contact)

            if lead_score.score >= min_score:
                # Update contact with score
                contact.lead_score = lead_score.score
                contact.engagement_level = lead_score.engagement_level
                results.append((contact, lead_score))

        # Sort by score descending
        results.sort(key=lambda x: x[1].score, reverse=True)
        return results


class ActionRecommender:
    """Recommends next best actions for contacts/deals"""

    def __init__(self, storage: CRMStorage):
        self.storage = storage

    def recommend_action(self, contact: Contact, lead_score: Optional[LeadScore] = None) -> ActionRecommendation:
        """
        Recommend next action for a contact

        Actions:
        - send_email: Reach out via email
        - schedule_call: Set up phone call
        - send_proposal: Send formal proposal
        - schedule_meeting: In-person or video meeting
        - wait: No action needed yet
        """
        if not lead_score:
            scorer = LeadScorer(self.storage)
            lead_score = scorer.score_lead(contact)

        # Get recent activities
        activities = self.storage.get_contact_activities(contact.id, limit=10)
        last_activity = activities[0] if activities else None

        # Decision logic based on engagement level and recent activity
        if lead_score.engagement_level == "hot":
            # Hot leads - be aggressive
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

        elif lead_score.engagement_level == "warm":
            # Warm leads - nurture
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

        elif lead_score.engagement_level == "cold":
            # Cold leads - low-effort nurture
            return ActionRecommendation(
                contact_id=contact.id,
                action="send_email",
                priority=0.3,
                reasoning="Cold lead. Low-effort email to gauge interest.",
                expected_outcome="Low probability response",
                suggested_timing=datetime.utcnow() + timedelta(days=7)
            )

        else:  # dead
            # Dead leads - archive or revive
            return ActionRecommendation(
                contact_id=contact.id,
                action="wait",
                priority=0.1,
                reasoning="Dead lead. Consider archiving or long-term nurture.",
                expected_outcome="Minimal activity expected",
                suggested_timing=datetime.utcnow() + timedelta(days=30)
            )

    def get_daily_recommendations(self, limit: int = 20) -> List[Tuple[Contact, ActionRecommendation]]:
        """Get top daily action recommendations"""
        recommendations = []

        for contact in self.storage.contacts.values():
            if "archived" in contact.tags:
                continue

            recommendation = self.recommend_action(contact)
            recommendations.append((contact, recommendation))

        # Sort by priority descending
        recommendations.sort(key=lambda x: x[1].priority, reverse=True)
        return recommendations[:limit]


class DealPredictor:
    """Predict deal success probability"""

    def __init__(self, storage: CRMStorage):
        self.storage = storage

    def predict_success(self, deal: Deal) -> Dict[str, Any]:
        """
        Predict probability of deal closing successfully

        Returns:
            Dict with prediction, confidence, and reasoning
        """
        # Get deal context
        contact = self.storage.get_contact(deal.contact_id)
        activities = self.storage.list_activities({"deal_id": deal.id})

        # Extract predictive features
        features = {}

        # Activity volume
        features["activity_count"] = min(1.0, len(activities) / 20.0)

        # Positive sentiment ratio
        positive_activities = [a for a in activities if a.outcome == ActivityOutcome.POSITIVE]
        if activities:
            features["positive_ratio"] = len(positive_activities) / len(activities)
        else:
            features["positive_ratio"] = 0.5

        # Stage progress
        stage_values = {
            DealStage.LEAD: 0.1,
            DealStage.QUALIFIED: 0.3,
            DealStage.PROPOSAL: 0.5,
            DealStage.NEGOTIATION: 0.7
        }
        features["stage_progress"] = stage_values.get(deal.stage, 0.5)

        # Time in pipeline
        days_open = (datetime.utcnow() - deal.created_at).days
        features["pipeline_age"] = 1.0 - min(1.0, days_open / 180.0)  # Older = less likely

        # Contact engagement
        if contact and contact.lead_score:
            features["contact_quality"] = contact.lead_score
        else:
            features["contact_quality"] = 0.5

        # Weighted prediction
        weights = {
            "activity_count": 0.20,
            "positive_ratio": 0.25,
            "stage_progress": 0.25,
            "pipeline_age": 0.15,
            "contact_quality": 0.15
        }

        probability = sum(features[k] * weights[k] for k in weights.keys())

        # Confidence based on data
        confidence = min(1.0, len(activities) / 10.0)

        # Reasoning
        reasoning_parts = []
        if features["positive_ratio"] > 0.7:
            reasoning_parts.append("Strong positive engagement")
        if features["activity_count"] > 0.6:
            reasoning_parts.append("High activity volume")
        if features["pipeline_age"] < 0.5:
            reasoning_parts.append("Deal aging in pipeline")

        reasoning = ". ".join(reasoning_parts) if reasoning_parts else "Limited data available"

        return {
            "deal_id": deal.id,
            "probability": probability,
            "confidence": confidence,
            "features": features,
            "reasoning": reasoning
        }


class CRMIntelligence:
    """Unified intelligence interface"""

    def __init__(self, storage: CRMStorage):
        self.storage = storage
        self.lead_scorer = LeadScorer(storage)
        self.action_recommender = ActionRecommender(storage)
        self.deal_predictor = DealPredictor(storage)

    def get_insights(self) -> Dict[str, Any]:
        """Get comprehensive CRM insights"""
        # Top leads
        top_leads = self.lead_scorer.score_all_leads(min_score=0.5)[:10]

        # Daily recommendations
        recommendations = self.action_recommender.get_daily_recommendations(limit=10)

        # At-risk deals (low probability)
        open_deals = self.storage.list_deals({"open_only": True})
        at_risk_deals = []
        for deal in open_deals:
            prediction = self.deal_predictor.predict_success(deal)
            if prediction["probability"] < 0.4:
                at_risk_deals.append((deal, prediction))

        at_risk_deals.sort(key=lambda x: x[1]["probability"])

        # Pipeline summary
        pipeline = self.storage.get_pipeline_summary()

        return {
            "top_leads": [
                {"contact": c.to_dict(), "score": s.to_dict()}
                for c, s in top_leads
            ],
            "daily_actions": [
                {"contact": c.to_dict(), "recommendation": r.to_dict()}
                for c, r in recommendations
            ],
            "at_risk_deals": [
                {"deal": d.to_dict(), "prediction": p}
                for d, p in at_risk_deals[:10]
            ],
            "pipeline": pipeline
        }