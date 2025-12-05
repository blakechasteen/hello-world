"""
Social Physics

Models social relationships as physical springs with tension, attraction, and decay.
Integrates with HoloLoom's Spring Dynamics system.
"""

from typing import Dict, List, Any
from ..config import NeuroHoodConfig


class SocialPhysics:
    """
    Physics-based social relationship engine.

    Relationships are modeled as springs:
    - Strength: Spring stiffness (how connected)
    - Tension: Current stress on relationship
    - Decay: Natural weakening over time

    Based on Hooke's Law:
        F = -k × Δa - c × v

    Where:
    - k = stiffness (connection strength)
    - Δa = activation difference (tension)
    - c = damping (prevents oscillation)
    - v = velocity (rate of change)
    """

    def __init__(self, config: NeuroHoodConfig):
        """
        Initialize social physics engine.

        Args:
            config: NeuroHood configuration
        """
        self.config = config

        # Physics parameters from config
        self.stiffness = config.relationship_stiffness
        self.damping = config.relationship_damping
        self.decay = config.relationship_decay

    def update_relationships(
        self,
        relationships: Dict[str, Any],
        time_delta: float = 1.0
    ) -> Dict[str, Any]:
        """
        Update all relationships using spring dynamics.

        Args:
            relationships: Dict of relationships
            time_delta: Time step

        Returns:
            Updated relationships
        """
        # Placeholder - integrate with HoloLoom's SpringDynamics
        # For now, just apply decay

        for key, relationship in relationships.items():
            # Natural decay
            relationship.strength *= self.decay ** time_delta

            # Tension gradually returns to zero
            relationship.tension *= (self.damping ** time_delta)

        return relationships

    def apply_interaction(
        self,
        relationship: Any,
        interaction_type: str,
        intensity: float = 0.1
    ) -> Any:
        """
        Apply an interaction to a relationship.

        Args:
            relationship: Relationship object
            interaction_type: "positive" or "negative"
            intensity: How strong the interaction

        Returns:
            Updated relationship
        """
        if interaction_type == "positive":
            # Strengthen relationship
            relationship.strength = min(1.0, relationship.strength + intensity)
            relationship.tension -= intensity  # Reduce tension

        elif interaction_type == "negative":
            # Weaken relationship
            relationship.strength = max(0.0, relationship.strength - intensity)
            relationship.tension += intensity  # Increase tension

        # Clamp values
        relationship.tension = max(-1.0, min(1.0, relationship.tension))

        return relationship

    def compute_social_force(
        self,
        resident_a_activation: float,
        resident_b_activation: float,
        relationship_strength: float
    ) -> float:
        """
        Compute social force between two residents.

        Based on spring physics: F = -k × (a_i - a_j)

        Args:
            resident_a_activation: Activation level of resident A
            resident_b_activation: Activation level of resident B
            relationship_strength: Spring stiffness

        Returns:
            Social force (positive = attraction, negative = repulsion)
        """
        delta_activation = resident_a_activation - resident_b_activation
        force = -relationship_strength * delta_activation

        return force
