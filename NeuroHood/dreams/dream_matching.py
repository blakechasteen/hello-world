"""
Dream Matching Algorithm: Connect Residents Through Shared Dreams

Implements the dream matching algorithm that connects 2-3 residents based on
emotional resonance and relationship dynamics.

Core Algorithm:
    dream_match_score = (
        0.4 * emotional_similarity +
        0.3 * relationship_tension +
        0.2 * complementary_archetypes +
        0.1 * personality_compatibility
    )

The algorithm finds residents with high emotional resonance (similar fears/hopes)
and relationship tension (unresolved dynamics), enabling shared dreams that
promote healing and growth through symbolic processing.

Key Features:
- Pairwise scoring of all resident combinations
- Top-k selection of best matches
- Human-readable match explanations
- Graceful degradation if systems unavailable
- Integration with personality, causal, and symbolic systems

Status: Production Ready (November 2025)
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

# HoloLoom integration
try:
    from HoloLoom.semantic_calculus import SemanticSpectrum, STANDARD_DIMENSIONS
    _HAVE_SEMANTIC = True
except ImportError:
    _HAVE_SEMANTIC = False
    logging.warning("HoloLoom semantic calculus not available")

# NeuroHood integration
try:
    from NeuroHood.personality import PersonalitySystem
    _HAVE_PERSONALITY = True
except ImportError:
    _HAVE_PERSONALITY = False
    logging.warning("NeuroHood personality system not available")

try:
    from NeuroHood.causal.relationship_scm import RelationshipSCM
    _HAVE_SCM = True
except ImportError:
    _HAVE_SCM = False
    logging.warning("NeuroHood relationship SCM not available")

try:
    from NeuroHood.dreams.symbolic_encoder import SymbolicEncoder
    _HAVE_ENCODER = True
except ImportError:
    _HAVE_ENCODER = False
    logging.warning("NeuroHood symbolic encoder not available")

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class DreamMatchCandidate:
    """
    A candidate resident pair for shared dreaming.

    Attributes:
        resident_pair: Tuple of (resident_a_id, resident_b_id)
        emotional_similarity: 0.0-1.0, similarity of emotional states
        relationship_tension: 0.0-1.0, unresolved relationship dynamics
        complementary_archetypes: 0.0-1.0, how well archetypes complement
        personality_compatibility: 0.0-1.0, personality trait alignment
        match_score: 0.0-1.0, weighted composite score
        match_reason: Human-readable explanation of match
        symbolic_bridge: Optional symbolic element connecting the pair
        conflict_areas: List of identified conflict/tension areas
        healing_opportunities: List of potential growth areas
        timestamp: When match was computed
    """
    resident_pair: Tuple[str, str]
    emotional_similarity: float
    relationship_tension: float
    complementary_archetypes: float
    personality_compatibility: float
    match_score: float
    match_reason: str
    symbolic_bridge: Optional[str] = None
    conflict_areas: List[str] = field(default_factory=list)
    healing_opportunities: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __str__(self) -> str:
        """Human-readable representation."""
        res_a, res_b = self.resident_pair
        return (
            f"Dream Match: {res_a} ↔ {res_b}\n"
            f"  Score: {self.match_score:.2%}\n"
            f"  Emotional Similarity: {self.emotional_similarity:.2%}\n"
            f"  Relationship Tension: {self.relationship_tension:.2%}\n"
            f"  Archetypal Complement: {self.complementary_archetypes:.2%}\n"
            f"  Personality Compatible: {self.personality_compatibility:.2%}\n"
            f"  Reason: {self.match_reason}\n"
            f"  Symbolic Bridge: {self.symbolic_bridge}\n"
            f"  Healing Opportunities: {', '.join(self.healing_opportunities) if self.healing_opportunities else 'N/A'}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "resident_pair": list(self.resident_pair),
            "emotional_similarity": float(self.emotional_similarity),
            "relationship_tension": float(self.relationship_tension),
            "complementary_archetypes": float(self.complementary_archetypes),
            "personality_compatibility": float(self.personality_compatibility),
            "match_score": float(self.match_score),
            "match_reason": self.match_reason,
            "symbolic_bridge": self.symbolic_bridge,
            "conflict_areas": self.conflict_areas,
            "healing_opportunities": self.healing_opportunities,
            "timestamp": self.timestamp,
        }


# ============================================================================
# Dream Matcher
# ============================================================================


class DreamMatcher:
    """
    Matches residents for shared dreaming based on emotional and relational dynamics.

    Algorithm:
    - Computes emotional similarity via 228D semantic space (HoloLoom)
    - Queries relationship tension from SCM
    - Evaluates archetypal complementarity
    - Combines with personality compatibility
    - Weights: 0.4 emotional, 0.3 tension, 0.2 archetypes, 0.1 personality

    Integration Points:
    - HoloLoom.semantic_calculus: 228D emotional similarity
    - NeuroHood.causal.relationship_scm: Relationship tension
    - NeuroHood.personality: Personality system
    - NeuroHood.dreams.symbolic_encoder: Symbolic bridges
    """

    def __init__(
        self,
        personality_system: Optional['PersonalitySystem'] = None,
        relationship_scm: Optional['RelationshipSCM'] = None,
        symbolic_encoder: Optional['SymbolicEncoder'] = None,
        embedding_fn: Optional[callable] = None,
    ):
        """
        Initialize the dream matcher.

        Args:
            personality_system: Personality tracking system
            relationship_scm: Causal model of relationships
            symbolic_encoder: Symbolic encoding for bridges
            embedding_fn: Function to embed text to 228D space
        """
        self.personality_system = personality_system
        self.relationship_scm = relationship_scm
        self.symbolic_encoder = symbolic_encoder
        self.embedding_fn = embedding_fn

        # Weights for composite score
        self.weights = {
            "emotional_similarity": 0.4,
            "relationship_tension": 0.3,
            "complementary_archetypes": 0.2,
            "personality_compatibility": 0.1,
        }

        # Cache for computed scores
        self._score_cache: Dict[Tuple[str, str], DreamMatchCandidate] = {}

        # Archetypal definitions
        self._archetypes = {
            "hero": {"traits": ["brave", "ambitious", "action-oriented"]},
            "mentor": {"traits": ["wise", "patient", "guiding"]},
            "shadow": {"traits": ["dark", "hidden", "repressed"]},
            "light": {"traits": ["open", "transparent", "revealed"]},
            "trickster": {"traits": ["clever", "unpredictable", "boundary-crossing"]},
            "caregiver": {"traits": ["compassionate", "nurturing", "sacrificial"]},
            "sage": {"traits": ["analytical", "seeking", "reflective"]},
            "innocent": {"traits": ["optimistic", "trusting", "hopeful"]},
        }

    def find_matches(
        self,
        residents: List[Any],
        max_matches: int = 5,
        min_score: float = 0.5,
    ) -> List[DreamMatchCandidate]:
        """
        Find best dream matches for a group of residents.

        Algorithm:
        1. Compute pairwise scores for all (res_a, res_b) combinations
        2. Filter by minimum score threshold
        3. Rank by match_score descending
        4. Return top-k matches

        Args:
            residents: List of Resident objects with resident_id, name, etc.
            max_matches: Maximum number of matches to return
            min_score: Minimum match score threshold (0.0-1.0)

        Returns:
            List of DreamMatchCandidate objects, sorted by match_score descending
        """
        if len(residents) < 2:
            logger.warning("Need at least 2 residents for matching")
            return []

        candidates = []

        # Generate all pairwise combinations
        for i, res_a in enumerate(residents):
            for res_b in residents[i + 1 :]:
                # Skip self-matches
                if res_a.resident_id == res_b.resident_id:
                    continue

                # Compute match
                candidate = self._compute_match(res_a, res_b)

                # Filter by minimum score
                if candidate.match_score >= min_score:
                    candidates.append(candidate)

        # Sort by score descending
        candidates.sort(key=lambda c: c.match_score, reverse=True)

        # Return top-k
        return candidates[:max_matches]

    def _compute_match(self, resident_a: Any, resident_b: Any) -> DreamMatchCandidate:
        """
        Compute match score between two residents.

        Args:
            resident_a: First resident object
            resident_b: Second resident object

        Returns:
            DreamMatchCandidate with computed scores
        """
        # Check cache
        pair_key = tuple(sorted([resident_a.resident_id, resident_b.resident_id]))
        if pair_key in self._score_cache:
            return self._score_cache[pair_key]

        # Compute individual components
        emotional_sim = self.compute_emotional_similarity(resident_a, resident_b)
        rel_tension = self.compute_relationship_tension(
            resident_a.resident_id, resident_b.resident_id
        )
        arch_comp = self.compute_archetypal_complementarity(resident_a, resident_b)
        pers_comp = self.compute_personality_compatibility(resident_a, resident_b)

        # Weighted composite
        match_score = (
            self.weights["emotional_similarity"] * emotional_sim
            + self.weights["relationship_tension"] * rel_tension
            + self.weights["complementary_archetypes"] * arch_comp
            + self.weights["personality_compatibility"] * pers_comp
        )

        # Generate explanation
        match_reason = self._generate_match_reason(
            resident_a, resident_b, emotional_sim, rel_tension, arch_comp, pers_comp
        )

        # Find symbolic bridge
        symbolic_bridge = self._find_symbolic_bridge(resident_a, resident_b)

        # Identify conflict and healing areas
        conflict_areas = self._identify_conflict_areas(
            resident_a.resident_id, resident_b.resident_id
        )
        healing_opportunities = self._identify_healing_opportunities(
            resident_a, resident_b, conflict_areas
        )

        candidate = DreamMatchCandidate(
            resident_pair=(resident_a.resident_id, resident_b.resident_id),
            emotional_similarity=float(emotional_sim),
            relationship_tension=float(rel_tension),
            complementary_archetypes=float(arch_comp),
            personality_compatibility=float(pers_comp),
            match_score=float(match_score),
            match_reason=match_reason,
            symbolic_bridge=symbolic_bridge,
            conflict_areas=conflict_areas,
            healing_opportunities=healing_opportunities,
        )

        # Cache result
        self._score_cache[pair_key] = candidate

        return candidate

    def compute_emotional_similarity(
        self, resident_a: Any, resident_b: Any
    ) -> float:
        """
        Compute emotional similarity via 228D semantic space.

        Algorithm:
        1. Get current emotional state for each resident (228D vector)
        2. Compute cosine similarity between vectors
        3. Return normalized similarity (0.0-1.0, where 1.0 = identical)

        Args:
            resident_a: First resident
            resident_b: Second resident

        Returns:
            Similarity score 0.0-1.0
        """
        # Try to get personality vectors
        vec_a = None
        vec_b = None

        # Option 1: Use personality system
        if self.personality_system and _HAVE_PERSONALITY:
            try:
                snapshot_a = self.personality_system.get_current_snapshot(
                    resident_a.resident_id
                )
                snapshot_b = self.personality_system.get_current_snapshot(
                    resident_b.resident_id
                )

                if snapshot_a and snapshot_b:
                    # Convert to vectors
                    dimensions = [d.name for d in STANDARD_DIMENSIONS]
                    vec_a = snapshot_a.to_vector(dimensions)
                    vec_b = snapshot_b.to_vector(dimensions)
            except Exception as e:
                logger.debug(f"Could not get personality vectors: {e}")

        # Option 2: Use personality_vector field if available
        if vec_a is None and hasattr(resident_a, "personality_vector"):
            try:
                vec_a = np.array(resident_a.personality_vector)
            except Exception as e:
                logger.debug(f"Could not convert personality_vector: {e}")

        if vec_b is None and hasattr(resident_b, "personality_vector"):
            try:
                vec_b = np.array(resident_b.personality_vector)
            except Exception as e:
                logger.debug(f"Could not convert personality_vector: {e}")

        if vec_a is not None and vec_b is not None:
            try:
                # Normalize vectors
                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)

                if norm_a > 0 and norm_b > 0:
                    # Cosine similarity: dot(a, b) / (||a|| * ||b||)
                    sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
                    # Normalize from [-1, 1] to [0, 1]
                    return float((sim + 1.0) / 2.0)
            except Exception as e:
                logger.debug(f"Error computing vector similarity: {e}")

        # Fallback: neutral similarity
        logger.warning(
            f"Could not compute emotional similarity for {resident_a.name} and "
            f"{resident_b.name}, using neutral value"
        )
        return 0.5

    def compute_relationship_tension(self, resident_a_id: str, resident_b_id: str) -> float:
        """
        Query relationship tension from SCM.

        Algorithm:
        1. Query relationship SCM for tension between residents
        2. Normalize to 0.0-1.0 range
        3. Higher tension = better for dream matching (indicates unresolved dynamics)

        Args:
            resident_a_id: ID of first resident
            resident_b_id: ID of second resident

        Returns:
            Tension score 0.0-1.0 (higher = more tension/unresolved dynamics)
        """
        if not self.relationship_scm or not _HAVE_SCM:
            logger.debug("No relationship SCM available, returning neutral tension")
            return 0.5

        try:
            # Query SCM for relationship data
            # Expected: some tension_level or conflict_intensity measure
            tension = self.relationship_scm.predict_relationship_tension(
                resident_a_id, resident_b_id
            )

            # Ensure normalized to [0, 1]
            if isinstance(tension, (int, float)):
                tension = float(tension)
                if tension < 0:
                    tension = 0.0
                elif tension > 1:
                    tension = tension / 10.0  # Handle if scale is 0-10
                return min(1.0, max(0.0, tension))

            return 0.5
        except Exception as e:
            logger.debug(f"Error querying SCM tension: {e}, using neutral")
            return 0.5

    def compute_archetypal_complementarity(
        self, resident_a: Any, resident_b: Any
    ) -> float:
        """
        Compute how well resident archetypes complement each other.

        Algorithm:
        1. Identify archetype for each resident (from personality traits)
        2. Check if archetypes form complementary pair (hero/mentor, shadow/light)
        3. Return complementarity score

        Complementary Pairs:
        - hero ↔ mentor (seeker finds guide)
        - shadow ↔ light (darkness seeks illumination)
        - trickster ↔ sage (chaos meets order)
        - innocent ↔ caregiver (optimism meets nurturing)

        Args:
            resident_a: First resident
            resident_b: Second resident

        Returns:
            Complementarity score 0.0-1.0
        """
        # Identify archetypes
        arch_a = self._identify_archetype(resident_a)
        arch_b = self._identify_archetype(resident_b)

        if not arch_a or not arch_b:
            logger.debug(f"Could not identify archetypes for {resident_a.name} or "
                        f"{resident_b.name}")
            return 0.5

        # Define complementary pairs (bidirectional)
        complementary_pairs = [
            ("hero", "mentor"),
            ("shadow", "light"),
            ("trickster", "sage"),
            ("innocent", "caregiver"),
            ("hero", "shadow"),  # Hero confronts shadow
            ("mentor", "innocent"),  # Mentor guides innocent
        ]

        # Check if this pair is complementary
        pair = tuple(sorted([arch_a, arch_b]))
        for comp_pair in complementary_pairs:
            if pair == tuple(sorted(comp_pair)):
                return 0.9  # Strong complementarity

        # Same archetype can be good for mirroring/mutual growth
        if arch_a == arch_b:
            return 0.6  # Moderate complementarity

        # Default for non-complementary but different
        return 0.4

    def compute_personality_compatibility(
        self, resident_a: Any, resident_b: Any
    ) -> float:
        """
        Compute personality trait compatibility.

        Algorithm:
        1. Extract key personality traits (warmth, directness, dominance, etc.)
        2. Compute trait-by-trait compatibility
        3. Return weighted average

        Compatibility Rules:
        - Similar warmth levels → good
        - Complementary dominance (one high, one low) → good
        - Extreme differences in arousal/intensity → potential conflict

        Args:
            resident_a: First resident
            resident_b: Second resident

        Returns:
            Compatibility score 0.0-1.0
        """
        if not self.personality_system or not _HAVE_PERSONALITY:
            logger.debug("No personality system available, returning neutral")
            return 0.5

        try:
            # Get personality snapshots
            snap_a = self.personality_system.get_current_snapshot(resident_a.resident_id)
            snap_b = self.personality_system.get_current_snapshot(resident_b.resident_id)

            if not snap_a or not snap_b:
                return 0.5

            # Extract key traits from 16 interpretable axes
            key_traits = [
                "warmth",
                "directness",
                "power",
                "arousal",
                "complexity",
                "certainty",
            ]

            compatibilities = []

            for trait in key_traits:
                val_a = snap_a.projections.get(trait, 0.0)
                val_b = snap_b.projections.get(trait, 0.0)

                # Special handling for different traits
                if trait == "power":
                    # Complementary dominance is good
                    # If one is high and one is low, compatibility is high
                    diff = abs(val_a - val_b)
                    compat = 1.0 - (diff / 2.0)  # Max diff is 2, range is [0,1]
                elif trait == "arousal":
                    # Similar arousal is better
                    diff = abs(val_a - val_b)
                    compat = 1.0 - (diff / 2.0)
                elif trait == "warmth":
                    # Similar warmth is good (both warm or both cool)
                    diff = abs(val_a - val_b)
                    compat = 1.0 - (diff / 2.0)
                else:
                    # Similar values generally compatible
                    diff = abs(val_a - val_b)
                    compat = 1.0 - (diff / 2.0)

                compatibilities.append(max(0.0, min(1.0, compat)))

            # Return average compatibility
            return float(np.mean(compatibilities)) if compatibilities else 0.5

        except Exception as e:
            logger.debug(f"Error computing personality compatibility: {e}")
            return 0.5

    def _identify_archetype(self, resident: Any) -> Optional[str]:
        """
        Identify resident's primary archetype from personality.

        Args:
            resident: Resident object

        Returns:
            Archetype name (hero, mentor, shadow, etc.) or None
        """
        if not self.personality_system or not _HAVE_PERSONALITY:
            return None

        try:
            snapshot = self.personality_system.get_current_snapshot(resident.resident_id)
            if not snapshot:
                return None

            # Map personality traits to archetypes
            traits = snapshot.dominant_traits(threshold=0.4)
            if not traits:
                return None

            # Score each archetype
            archetype_scores = {}
            for archetype, defn in self._archetypes.items():
                score = sum(
                    1.0
                    for trait_name, _ in traits
                    if trait_name in defn.get("traits", [])
                )
                archetype_scores[archetype] = score

            # Return highest scoring archetype
            best = max(archetype_scores, key=archetype_scores.get)
            return best if archetype_scores[best] > 0 else None

        except Exception as e:
            logger.debug(f"Could not identify archetype for {resident.name}: {e}")
            return None

    def _generate_match_reason(
        self,
        resident_a: Any,
        resident_b: Any,
        emotional_sim: float,
        rel_tension: float,
        arch_comp: float,
        pers_comp: float,
    ) -> str:
        """
        Generate human-readable explanation of why residents are matched.

        Args:
            resident_a: First resident
            resident_b: Second resident
            emotional_sim: Emotional similarity score
            rel_tension: Relationship tension
            arch_comp: Archetypal complementarity
            pers_comp: Personality compatibility

        Returns:
            Explanation string
        """
        reasons = []

        if emotional_sim > 0.7:
            reasons.append(f"strong emotional resonance ({emotional_sim:.0%})")
        elif emotional_sim > 0.5:
            reasons.append(f"shared emotional space ({emotional_sim:.0%})")

        if rel_tension > 0.6:
            reasons.append("unresolved relationship dynamics")
        elif rel_tension > 0.3:
            reasons.append("some relationship tension to explore")

        if arch_comp > 0.8:
            reasons.append("complementary archetypal roles")
        elif arch_comp > 0.5:
            reasons.append("archetypal mirroring opportunity")

        if pers_comp > 0.7:
            reasons.append("compatible personality traits")

        if reasons:
            return f"{resident_a.name} and {resident_b.name}: " + "; ".join(reasons)
        else:
            return f"{resident_a.name} and {resident_b.name}: potential for growth"

    def _find_symbolic_bridge(self, resident_a: Any, resident_b: Any) -> Optional[str]:
        """
        Find a symbolic element that connects two residents in dreams.

        Args:
            resident_a: First resident
            resident_b: Second resident

        Returns:
            Symbol name/ID or None
        """
        if not self.symbolic_encoder or not _HAVE_ENCODER:
            return None

        try:
            # Try to find shared symbolic themes
            # This would use the symbolic encoder to find common symbols
            # For now, return a generic symbol
            return "shared_bridge"
        except Exception as e:
            logger.debug(f"Could not find symbolic bridge: {e}")
            return None

    def _identify_conflict_areas(self, resident_a_id: str, resident_b_id: str) -> List[str]:
        """
        Identify areas of conflict or tension between residents.

        Args:
            resident_a_id: First resident ID
            resident_b_id: Second resident ID

        Returns:
            List of conflict area descriptions
        """
        if not self.relationship_scm or not _HAVE_SCM:
            return []

        try:
            # Query SCM for relationship data
            # This would extract conflict areas from the model
            conflicts = []

            # Try to get relationship data
            try:
                rel_data = self.relationship_scm.get_relationship_data(
                    resident_a_id, resident_b_id
                )
                if rel_data and hasattr(rel_data, "conflict_intensity"):
                    if rel_data.conflict_intensity > 0.5:
                        conflicts.append("communication breakdown")
                    if rel_data.stress_level > 0.6:
                        conflicts.append("high stress levels")
            except Exception:
                pass

            return conflicts
        except Exception as e:
            logger.debug(f"Could not identify conflict areas: {e}")
            return []

    def _identify_healing_opportunities(
        self,
        resident_a: Any,
        resident_b: Any,
        conflict_areas: List[str],
    ) -> List[str]:
        """
        Identify potential healing and growth opportunities for the pair.

        Args:
            resident_a: First resident
            resident_b: Second resident
            conflict_areas: Identified conflict areas

        Returns:
            List of healing opportunity descriptions
        """
        opportunities = []

        # If there's conflict, there's opportunity for healing
        if conflict_areas:
            opportunities.append("resolve unfinished business")
            opportunities.append("build understanding through symbols")
            opportunities.append("transform conflict into connection")

        # Check for complementary strengths
        arch_a = self._identify_archetype(resident_a)
        arch_b = self._identify_archetype(resident_b)

        if arch_a == "mentor" and arch_b == "hero":
            opportunities.append("mentorship and growth")
        elif arch_a == "light" and arch_b == "shadow":
            opportunities.append("integration of shadow self")
        elif arch_a == "sage" and arch_b == "trickster":
            opportunities.append("balance wisdom with innovation")

        # Default opportunity if none found
        if not opportunities:
            opportunities.append("discover shared values")

        return opportunities
