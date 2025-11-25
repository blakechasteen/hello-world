#!/usr/bin/env python3
"""
HoloLoom Integration - AI-powered matching and learning for community food sharing

Integrates HoloLoom's memory and learning systems to enhance:
- Semantic understanding of food items and organizational needs
- Learning from past donation patterns
- Predictive matching based on historical success
- Smart route optimization
- Personalized recommendations
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from decimal import Decimal

# HoloLoom imports
try:
    from HoloLoom import HoloLoom
    from HoloLoom.config import Config
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False
    print("Warning: HoloLoom not available. AI features will be disabled.")

from sous.models.organization import Organization
from sous.models.food_item import FoodItem, DonationListing
from sous.services.donation_coordinator import DonationCoordinator, MatchScore


class HoloLoomMatchingEngine:
    """
    AI-powered matching engine using HoloLoom for semantic understanding
    and pattern learning.
    """

    def __init__(self, coordinator: DonationCoordinator):
        self.coordinator = coordinator
        self.loom: Optional[HoloLoom] = None
        self.enabled = HOLOLOOM_AVAILABLE

        if self.enabled:
            # Initialize HoloLoom with FAST config for balanced performance
            self.config = Config.fast()

    async def initialize(self):
        """Initialize HoloLoom memory system"""
        if not self.enabled:
            return

        self.loom = HoloLoom()
        await self.loom.__aenter__()

        # Load historical knowledge if available
        await self._load_historical_patterns()

    async def close(self):
        """Clean up HoloLoom resources"""
        if self.loom:
            await self.loom.__aexit__(None, None, None)

    # ========================================================================
    # Semantic Matching
    # ========================================================================

    async def semantic_food_matching(
        self,
        food_items: List[FoodItem],
        recipient_needs: str
    ) -> List[Tuple[FoodItem, float]]:
        """
        Match food items to recipient needs using semantic understanding.

        Args:
            food_items: Available food items
            recipient_needs: Description of recipient's needs
                            (e.g., "nutritious meals for families with children")

        Returns:
            List of (FoodItem, relevance_score) tuples
        """
        if not self.enabled or not self.loom:
            # Fallback: return all items with equal score
            return [(item, 0.5) for item in food_items]

        matches = []

        for item in food_items:
            # Create semantic query combining item description and recipient needs
            query = f"Match food donation: {item.name} ({item.description}) to recipient need: {recipient_needs}"

            # Use HoloLoom to understand semantic relevance
            memories = await self.loom.recall(query)

            # Calculate relevance based on memory activation
            relevance = 0.5  # Default
            if memories:
                # Use awareness metrics to gauge relevance
                metrics = self.loom.get_metrics()
                relevance = metrics.get('activation', {}).get('mean_activation', 0.5)

            matches.append((item, relevance))

        # Sort by relevance
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches

    async def learn_from_donation(
        self,
        listing: DonationListing,
        recipient_org_id: str,
        success: bool,
        feedback: Optional[str] = None
    ):
        """
        Learn from donation outcome to improve future matching.

        Args:
            listing: The donation listing
            recipient_org_id: Who received the donation
            success: Whether the donation was successful
            feedback: Optional textual feedback
        """
        if not self.enabled or not self.loom:
            return

        # Build experience description
        donor_org = self.coordinator.get_organization(listing.donor_org_id)
        recipient_org = self.coordinator.get_organization(recipient_org_id)

        if not donor_org or not recipient_org:
            return

        # Describe the donation
        food_summary = ", ".join([item.name for item in listing.items[:3]])
        if len(listing.items) > 3:
            food_summary += f" and {len(listing.items) - 3} more items"

        outcome = "successful" if success else "unsuccessful"

        experience = (
            f"{outcome.capitalize()} donation from {donor_org.name} ({donor_org.org_type.value}) "
            f"to {recipient_org.name} ({recipient_org.org_type.value}). "
            f"Items: {food_summary}. "
        )

        if feedback:
            experience += f"Feedback: {feedback}"

        # Store in HoloLoom memory
        await self.loom.experience(experience)

    async def predict_match_success(
        self,
        donor_org: Organization,
        recipient_org: Organization,
        food_items: List[FoodItem]
    ) -> float:
        """
        Predict likelihood of successful match based on historical patterns.

        Returns:
            Predicted success probability (0.0-1.0)
        """
        if not self.enabled or not self.loom:
            return 0.5  # Default neutral prediction

        # Build query about this type of match
        food_summary = ", ".join([item.name for item in food_items[:3]])

        query = (
            f"Historical success of donations from {donor_org.org_type.value} "
            f"to {recipient_org.org_type.value} with items like {food_summary}"
        )

        # Query HoloLoom for relevant past experiences
        memories = await self.loom.recall(query)

        if not memories:
            return 0.5  # No history, neutral prediction

        # Analyze sentiment/success from memories
        success_count = 0
        total_count = 0

        for memory in memories:
            total_count += 1
            # Simple heuristic: check for positive indicators
            text = memory.text.lower()
            if "successful" in text or "positive" in text or "good" in text:
                success_count += 1

        if total_count == 0:
            return 0.5

        return success_count / total_count

    # ========================================================================
    # Enhanced Matching
    # ========================================================================

    async def enhanced_match_score(
        self,
        listing: DonationListing,
        donor: Organization,
        recipient: Organization
    ) -> MatchScore:
        """
        Calculate enhanced match score using AI predictions.

        Combines standard scoring with HoloLoom predictions.
        """
        # Get base score from coordinator
        base_score = self.coordinator._calculate_match_score(listing, donor, recipient)

        if not self.enabled or not self.loom:
            return base_score

        # Add AI prediction factor
        predicted_success = await self.predict_match_success(
            donor, recipient, listing.items
        )

        # Blend base score (70%) with AI prediction (30%)
        enhanced_total = (base_score.total_score * 0.7) + (predicted_success * 0.3)

        # Create enhanced score
        enhanced_score = MatchScore(
            recipient_org_id=recipient.org_id,
            total_score=enhanced_total,
            subscores={
                **base_score.subscores,
                "ai_prediction": predicted_success
            },
            reasoning=base_score.reasoning + f" | AI prediction: {predicted_success:.2f}"
        )

        return enhanced_score

    async def find_enhanced_matches(
        self,
        listing: DonationListing,
        max_matches: int = 10
    ) -> List[MatchScore]:
        """
        Find best matches using enhanced AI scoring.
        """
        donor = self.coordinator.get_organization(listing.donor_org_id)
        if not donor:
            return []

        recipients = self.coordinator.get_recipients()
        if not recipients:
            return []

        # Score all recipients with AI enhancement
        scores = []
        for recipient in recipients:
            score = await self.enhanced_match_score(listing, donor, recipient)
            if score.total_score > 0:
                scores.append(score)

        # Sort by score
        scores.sort(reverse=True, key=lambda s: s.total_score)

        return scores[:max_matches]

    # ========================================================================
    # Route Optimization
    # ========================================================================

    async def optimize_pickup_route(
        self,
        starting_location: str,
        pickup_locations: List[Tuple[str, Organization]],
        max_distance_miles: float = 50.0
    ) -> List[str]:
        """
        Optimize pickup route using AI-learned patterns.

        Args:
            starting_location: Starting point (org_id)
            pickup_locations: List of (org_id, Organization) to visit
            max_distance_miles: Maximum total route distance

        Returns:
            Optimized list of org_ids to visit in order
        """
        if not pickup_locations:
            return []

        if not self.enabled or not self.loom:
            # Fallback: simple nearest-neighbor
            return self._simple_nearest_neighbor(starting_location, pickup_locations)

        # Query HoloLoom for optimal route patterns
        query = f"Optimal food pickup route for {len(pickup_locations)} locations"

        memories = await self.loom.recall(query)

        # Use historical route patterns if available
        if memories and len(memories) > 0:
            # Extract route recommendations from memory
            # For now, fall back to nearest-neighbor
            # In production, could learn complex route patterns
            pass

        return self._simple_nearest_neighbor(starting_location, pickup_locations)

    def _simple_nearest_neighbor(
        self,
        starting_location: str,
        pickup_locations: List[Tuple[str, Organization]]
    ) -> List[str]:
        """
        Simple nearest-neighbor route optimization.
        """
        start_org = self.coordinator.get_organization(starting_location)
        if not start_org:
            return [org_id for org_id, _ in pickup_locations]

        route = []
        remaining = list(pickup_locations)
        current = start_org

        while remaining:
            # Find nearest location
            nearest = min(
                remaining,
                key=lambda x: current.get_distance_to(x[1].location) or float('inf')
            )

            route.append(nearest[0])
            remaining.remove(nearest)
            current = nearest[1]

        return route

    # ========================================================================
    # Smart Recommendations
    # ========================================================================

    async def recommend_partnerships(
        self,
        org_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        AI-powered partnership recommendations.

        Uses HoloLoom to understand successful partnership patterns.
        """
        org = self.coordinator.get_organization(org_id)
        if not org:
            return []

        if not self.enabled or not self.loom:
            # Fallback to coordinator's suggestions
            from sous.services.community_network import CommunityNetwork
            network = CommunityNetwork()
            # If network available, use its suggestions
            return []

        # Query for successful partnership patterns
        query = f"Successful partnerships for {org.org_type.value} organizations"

        memories = await self.loom.recall(query)

        recommendations = []

        # Analyze memories for partnership insights
        for other_id, other_org in self.coordinator.organizations.items():
            if other_id == org_id:
                continue

            # Calculate AI recommendation score
            score = 0.0
            reasons = []

            # Check for similar patterns in memory
            similarity_query = (
                f"Partnership between {org.org_type.value} and {other_org.org_type.value}"
            )

            similar_memories = await self.loom.recall(similarity_query)

            if similar_memories:
                # Positive memories increase score
                for memory in similar_memories[:3]:
                    if "successful" in memory.text.lower():
                        score += 0.3
                        reasons.append("Similar successful partnerships exist")
                        break

            # Add geographic factor
            distance = org.get_distance_to(other_org.location)
            if distance and distance < 10:
                score += 0.2
                reasons.append(f"Close proximity ({distance:.1f} miles)")

            # Add complementary capabilities
            if org.can_donate and other_org.can_receive:
                score += 0.2
                reasons.append("Complementary capabilities")

            if score > 0:
                recommendations.append({
                    "org_id": other_id,
                    "org_name": other_org.name,
                    "score": score,
                    "reasons": reasons
                })

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:limit]

    # ========================================================================
    # Historical Learning
    # ========================================================================

    async def _load_historical_patterns(self):
        """Load historical patterns into HoloLoom memory"""
        if not self.enabled or not self.loom:
            return

        # Load common successful patterns
        patterns = [
            "Restaurants typically have surplus prepared food at closing time",
            "Grocery stores donate produce and bakery items approaching expiration",
            "Food banks can accept large volumes with proper refrigeration",
            "Churches with meal programs need prepared food donations",
            "Short distances (<5 miles) increase donation success rates",
            "Verified partnerships have higher success rates",
            "Urgent alerts require immediate response within 2 hours",
            "Frozen items require freezer storage at recipient location",
            "Dairy products have strict refrigeration requirements",
        ]

        for pattern in patterns:
            await self.loom.experience(pattern)

    async def get_insights(self, org_id: str) -> Dict:
        """
        Generate AI insights for an organization.

        Returns insights about donation patterns, opportunities, etc.
        """
        org = self.coordinator.get_organization(org_id)
        if not org:
            return {}

        if not self.enabled or not self.loom:
            return {"insights": [], "recommendations": []}

        # Query for insights
        query = f"Insights and opportunities for {org.name} ({org.org_type.value})"

        memories = await self.loom.recall(query)

        insights = []
        recommendations = []

        # Analyze historical patterns
        if org.can_donate:
            insights.append({
                "type": "donation_pattern",
                "message": f"{org.org_type.value} typically has surplus food available"
            })

        if org.can_receive:
            insights.append({
                "type": "capacity",
                "message": "Can accept donations matching storage capabilities"
            })

        # Generate recommendations based on memory
        if memories:
            recommendations.append({
                "priority": "high",
                "action": "Consider establishing regular donation schedules",
                "reasoning": "Historical patterns show consistent surplus"
            })

        return {
            "org_id": org_id,
            "insights": insights,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }


# ========================================================================
# Helper Functions
# ========================================================================

async def create_hololoom_engine(
    coordinator: DonationCoordinator
) -> HoloLoomMatchingEngine:
    """
    Create and initialize HoloLoom matching engine.

    Usage:
        engine = await create_hololoom_engine(coordinator)
        try:
            matches = await engine.find_enhanced_matches(listing)
            # ... use matches ...
        finally:
            await engine.close()
    """
    engine = HoloLoomMatchingEngine(coordinator)
    await engine.initialize()
    return engine


async def demo_hololoom_matching():
    """Demonstration of HoloLoom-enhanced matching"""
    if not HOLOLOOM_AVAILABLE:
        print("HoloLoom not available for demo")
        return

    from sous.services.donation_coordinator import DonationCoordinator
    from sous.models.organization import create_food_bank, create_restaurant

    # Create coordinator
    coordinator = DonationCoordinator()

    # Add organizations
    food_bank = create_food_bank(
        name="Community Food Bank",
        address="123 Main St",
        city="Springfield",
        state="IL",
        zip_code="62701",
        contact_name="John Doe",
        contact_email="john@foodbank.org",
        contact_phone="555-0100"
    )

    restaurant = create_restaurant(
        name="Bella Cucina",
        cuisine_type="Italian",
        address="789 Oak St",
        city="Springfield",
        state="IL",
        zip_code="62701",
        contact_name="Maria Rossi",
        contact_email="maria@bella.com",
        contact_phone="555-0200"
    )

    coordinator.register_organization(food_bank)
    coordinator.register_organization(restaurant)

    # Create engine
    engine = await create_hololoom_engine(coordinator)

    try:
        # Demonstrate learning
        from sous.models.food_item import create_prepared_meal, DonationListing

        meal = create_prepared_meal(
            name="Lasagna",
            servings=10,
            description="Homemade lasagna",
            source_org_id=restaurant.org_id
        )

        listing = DonationListing(
            listing_id="demo_listing_1",
            donor_org_id=restaurant.org_id,
            items=[meal],
            available_from=datetime.now(),
            available_until=datetime.now()
        )

        # Learn from successful donation
        await engine.learn_from_donation(
            listing,
            food_bank.org_id,
            success=True,
            feedback="Great quality food, perfect timing"
        )

        print("✓ Learned from donation")

        # Get AI insights
        insights = await engine.get_insights(restaurant.org_id)
        print(f"✓ Generated insights: {len(insights.get('insights', []))} insights")

        # Find enhanced matches
        matches = await engine.find_enhanced_matches(listing, max_matches=5)
        print(f"✓ Found {len(matches)} AI-enhanced matches")

    finally:
        await engine.close()


if __name__ == "__main__":
    asyncio.run(demo_hololoom_matching())
