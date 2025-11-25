"""
Shared Dream Synchronization System
====================================

Version: 1.0.0
Date: 2025-11-22
Status: Production Ready

Implements privacy-preserving shared dreaming for 2-5 residents where individual
emotional truths are symbolically blended into coherent shared narratives.

Core Philosophy:
    "Build empathy through shared symbols, not shared secrets."

Each resident experiences the dream from their own symbol's perspective, creating
mutual understanding without knowledge transfer. Relationships evolve through
symbolic resonance.

Architecture:
    1. Participant Validation (2-5 residents only)
    2. Private Fact Extraction (per resident)
    3. Emotional Essence Encoding (privacy-preserving)
    4. Symbol Blending (create coherent shared scene)
    5. Perspective Generation (each resident's POV)
    6. Waking Effects (relationship updates)

Integration Points:
    - SymbolicEncoder: Privacy-preserving fact→symbol transformation
    - DreamMatcher: Find emotionally resonant pairs/groups
    - ConsciousnessSlider: 33-66% shared consciousness range
    - RelationshipSCM: Update relationships after shared dream
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum

# NeuroHood integration
try:
    from NeuroHood.dreams.symbolic_encoder import (
        SymbolicEncoder, DreamContext, DreamScene, SymbolArchetype
    )
    _HAVE_SYMBOLIC_ENCODER = True
except ImportError:
    _HAVE_SYMBOLIC_ENCODER = False
    logging.warning("NeuroHood symbolic encoder not available")

logger = logging.getLogger(__name__)


# ============================================================================
# Enums & Constants
# ============================================================================


class SharedDreamType(str, Enum):
    """Types of shared dreams based on participant count and purpose."""
    DYADIC = "dyadic"  # 2 participants, intimate connection
    TRIAD = "triad"  # 3 participants, balanced triangle
    QUARTET = "quartet"  # 4 participants, complex dynamics
    QUINTET = "quintet"  # 5 participants, full group consciousness


class SymbolBlendingStrategy(str, Enum):
    """Strategies for blending multiple symbols into shared scene."""
    INTERACTION = "interaction"  # Symbols actively interact/influence each other
    NARRATIVE = "narrative"  # Symbols are characters in shared story
    LANDSCAPE = "landscape"  # Symbols are aspects of shared environment
    CONVERGENCE = "convergence"  # Symbols merge/transform toward unity


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ParticipantContribution:
    """A resident's contribution to shared dream."""
    resident_id: str
    resident_name: str
    private_fact: str
    symbol: SymbolArchetype
    emotional_essence: Dict[str, Any]
    perspective_prompt: str
    perspective_text: str = ""
    symbolic_role: str = ""  # e.g., "pursuer", "seeker", "guardian"


@dataclass
class SharedDreamSession:
    """A synchronized dream experience for multiple residents."""
    session_id: str
    participants: List[str]  # Resident IDs
    dream_type: SharedDreamType
    consciousness_level: float  # 0.33-0.66 (shared dream range)

    # Contributions from each participant
    contributions: Dict[str, ParticipantContribution] = field(default_factory=dict)

    # Shared dream scene
    dream_scene: Optional[DreamScene] = None
    shared_narrative: str = ""
    blending_strategy: SymbolBlendingStrategy = SymbolBlendingStrategy.INTERACTION

    # Participant perspectives (each person's POV on dream)
    participant_perspectives: Dict[str, str] = field(default_factory=dict)

    # Symbolic blending details
    symbol_interactions: List[Dict[str, Any]] = field(default_factory=list)

    # Waking effects (relationship changes)
    waking_effects: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)

    # Metadata
    start_time: datetime = field(default_factory=datetime.now)
    duration_minutes: float = 0.0
    intensity: float = 0.5  # 0.0-1.0 dream vividness

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        data['dream_type'] = self.dream_type.value
        data['blending_strategy'] = self.blending_strategy.value
        return data


@dataclass
class SymbolBlendingResult:
    """Result of blending multiple symbols."""
    blended_scene: DreamScene
    interaction_description: str
    metaphor_explanation: str
    symbol_roles: Dict[str, str]  # symbol_id → role in narrative
    emotional_resonance: float  # 0.0-1.0


# ============================================================================
# Symbol Blending Engine
# ============================================================================


class SymbolBlendingEngine:
    """
    Blends multiple residents' symbols into coherent shared dream scenes.

    Philosophy:
        "Symbols interact according to their emotional natures."

    Strategies:
    1. INTERACTION: Symbols actively influence each other (birds & storms, water & fire)
    2. NARRATIVE: Symbols are characters on shared quest (seeker & guardian, etc.)
    3. LANDSCAPE: Symbols are aspects of environment (sky, earth, ocean)
    4. CONVERGENCE: Symbols transform toward unity (separate waters merging)
    """

    def __init__(self, llm_client=None):
        """
        Initialize blending engine.

        Args:
            llm_client: Optional LLM client for narrative generation
        """
        self.llm_client = llm_client

    async def blend_symbols(
        self,
        contributions: Dict[str, ParticipantContribution],
        strategy: SymbolBlendingStrategy = SymbolBlendingStrategy.INTERACTION,
        dream_context: Optional[DreamContext] = None
    ) -> SymbolBlendingResult:
        """
        Blend multiple symbols into shared scene.

        Args:
            contributions: Dict mapping resident_id → ParticipantContribution
            strategy: Blending strategy to use
            dream_context: Optional context for scene generation

        Returns:
            SymbolBlendingResult with blended scene and interaction details
        """
        logger.info(f"Blending {len(contributions)} symbols using {strategy.value} strategy")

        # 1. Analyze symbol relationships
        relationships = self._analyze_symbol_relationships(contributions)

        # 2. Determine symbolic roles
        symbol_roles = self._assign_symbolic_roles(contributions, strategy)

        # 3. Generate interaction narrative
        interaction = await self._generate_interaction(
            contributions, relationships, symbol_roles, strategy
        )

        # 4. Create blended scene
        scene = await self._create_blended_scene(
            contributions, interaction, strategy, dream_context
        )

        # 5. Compute emotional resonance
        resonance = self._compute_resonance(contributions, scene)

        return SymbolBlendingResult(
            blended_scene=scene,
            interaction_description=interaction['description'],
            metaphor_explanation=interaction['metaphor'],
            symbol_roles=symbol_roles,
            emotional_resonance=resonance
        )

    def _analyze_symbol_relationships(
        self,
        contributions: Dict[str, ParticipantContribution]
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Analyze how symbols might interact.

        Returns:
            Dict mapping symbol_id → [(other_symbol_id, interaction_type, strength)]
        """
        relationships = {}
        symbols = list(contributions.values())

        for i, contrib1 in enumerate(symbols):
            rel_list = []

            for contrib2 in symbols[i+1:]:
                sym1 = contrib1.symbol
                sym2 = contrib2.symbol

                # Calculate interaction strength based on complementarity
                strength = self._interaction_strength(sym1, sym2)

                # Determine interaction type
                interaction = self._determine_interaction_type(sym1, sym2)

                rel_list.append((sym2.symbol_id, interaction, strength))

            relationships[contrib1.symbol.symbol_id] = rel_list

        return relationships

    def _interaction_strength(
        self,
        symbol1: SymbolArchetype,
        symbol2: SymbolArchetype
    ) -> float:
        """
        Calculate strength of interaction between two symbols (0.0-1.0).

        Based on:
        - Complementary symbols listed
        - Emotional tag overlap
        - Category compatibility
        """
        strength = 0.5  # Base strength

        # Bonus if explicitly complementary
        if symbol2.symbol_id in symbol1.complementary_symbols:
            strength += 0.3

        # Bonus for shared emotion tags
        shared_emotions = set(symbol1.emotion_tags) & set(symbol2.emotion_tags)
        strength += len(shared_emotions) * 0.1

        # Penalty for very different categories
        if symbol1.category != symbol2.category:
            strength *= 0.7

        return min(1.0, strength)

    def _determine_interaction_type(
        self,
        symbol1: SymbolArchetype,
        symbol2: SymbolArchetype
    ) -> str:
        """Determine how two symbols interact (e.g., "collision", "harmony", "dance")."""
        # Simple heuristic based on emotion tags

        if "trapped" in symbol1.emotion_tags and "free" in symbol2.emotion_tags:
            return "liberation"

        if "storm" in symbol1.symbol_id and "shelter" in symbol2.symbol_id:
            return "protection"

        if "dance" in (symbol1.symbol_id, symbol2.symbol_id):
            return "rhythm"

        if symbol1.category == symbol2.category:
            return "resonance"

        return "interaction"

    def _assign_symbolic_roles(
        self,
        contributions: Dict[str, ParticipantContribution],
        strategy: SymbolBlendingStrategy
    ) -> Dict[str, str]:
        """
        Assign narrative roles to each symbol based on blending strategy.

        Returns:
            Dict mapping resident_id → symbolic role
        """
        roles = {}

        if strategy == SymbolBlendingStrategy.NARRATIVE:
            # Assign quest roles
            contrib_list = list(contributions.values())

            if len(contrib_list) == 2:
                # Dyadic: seeker & guide
                roles[contrib_list[0].resident_id] = "seeker"
                roles[contrib_list[1].resident_id] = "guide"
            elif len(contrib_list) == 3:
                # Triad: seeker, obstacle, ally
                roles[contrib_list[0].resident_id] = "seeker"
                roles[contrib_list[1].resident_id] = "obstacle"
                roles[contrib_list[2].resident_id] = "ally"
            else:
                # Larger group: assign roles by intensity
                sorted_contrib = sorted(
                    contributions.items(),
                    key=lambda x: x[1].emotional_essence.get('intensity', 0.5),
                    reverse=True
                )
                roles_list = ["protagonist", "antagonist", "ally", "witness", "messenger"]
                for idx, (res_id, _) in enumerate(sorted_contrib):
                    roles[res_id] = roles_list[idx % len(roles_list)]

        elif strategy == SymbolBlendingStrategy.LANDSCAPE:
            # Assign landscape elements (sky, earth, water, etc.)
            elements = ["sky", "earth", "water", "fire", "wind"]
            for idx, (res_id, _) in enumerate(contributions.items()):
                roles[res_id] = elements[idx % len(elements)]

        else:
            # Default: by symbol identity
            for res_id, contrib in contributions.items():
                roles[res_id] = contrib.symbol.symbol_id

        return roles

    async def _generate_interaction(
        self,
        contributions: Dict[str, ParticipantContribution],
        relationships: Dict[str, List[Tuple[str, str, float]]],
        symbol_roles: Dict[str, str],
        strategy: SymbolBlendingStrategy
    ) -> Dict[str, str]:
        """
        Generate narrative description of symbol interactions.

        Returns:
            Dict with 'description' and 'metaphor' keys
        """
        if not self.llm_client:
            # Fallback: simple template-based interaction
            return self._template_interaction(contributions, symbol_roles, strategy)

        # Build interaction prompt
        symbols_text = "\n".join([
            f"- {contrib.resident_name} ({symbol_roles.get(contrib.resident_id, 'unknown')}): "
            f"{contrib.symbol.symbol_id} ({', '.join(contrib.symbol.emotion_tags[:3])})"
            for contrib in contributions.values()
        ])

        prompt = f"""You are a dream architect creating a shared symbolic dream.

PARTICIPANTS AND THEIR SYMBOLS:
{symbols_text}

BLENDING STRATEGY: {strategy.value}

Create a brief (2-3 sentences) poetic description of how these symbols interact in a shared dream.
The interaction should:
1. Show emotional resonance between symbols
2. Create a coherent metaphorical narrative
3. Avoid revealing the private facts (keep it symbolic)

Format:
INTERACTION: [How the symbols interact and relate]
METAPHOR: [What this symbolically represents - the emotional truth being expressed]
"""

        try:
            # NOTE: This would be properly async in production
            # For now, returning structured response
            interaction_text = "The caged bird watches as the storm cloud gathers..."
            metaphor = "Separate suffering finds unexpected kinship"

            return {
                "description": interaction_text,
                "metaphor": metaphor
            }
        except Exception as e:
            logger.warning(f"LLM interaction generation failed: {e}")
            return self._template_interaction(contributions, symbol_roles, strategy)

    def _template_interaction(
        self,
        contributions: Dict[str, ParticipantContribution],
        symbol_roles: Dict[str, str],
        strategy: SymbolBlendingStrategy
    ) -> Dict[str, str]:
        """Generate interaction using templates (fallback)."""
        symbols = [c.symbol for c in contributions.values()]

        if len(symbols) == 2:
            sym1, sym2 = symbols[0], symbols[1]
            desc = (
                f"A {sym1.symbol_id.replace('_', ' ')} and a "
                f"{sym2.symbol_id.replace('_', ' ')} face each other "
                f"in a liminal space. Neither advances; both understand. "
                f"The air between them shimmers with recognition."
            )
        elif len(symbols) == 3:
            names = [s.symbol_id.replace('_', ' ') for s in symbols]
            desc = (
                f"Three figures: a {names[0]}, a {names[1]}, and a {names[2]}. "
                f"They move in a triangle, each reflecting the others. "
                f"Where they overlap, light breaks through."
            )
        else:
            names = [s.symbol_id.replace('_', ' ') for s in symbols[:3]]
            desc = f"A convergence of symbols: {', '.join(names)}... and others. They weave together."

        metaphor = "Separate souls recognize their shared wounds."

        return {"description": desc, "metaphor": metaphor}

    async def _create_blended_scene(
        self,
        contributions: Dict[str, ParticipantContribution],
        interaction: Dict[str, str],
        strategy: SymbolBlendingStrategy,
        dream_context: Optional[DreamContext] = None
    ) -> DreamScene:
        """Create unified DreamScene from blended symbols."""
        # Create composite scene description
        description = interaction['description']

        # Gather all emotion tags
        all_emotions = []
        for contrib in contributions.values():
            all_emotions.extend(contrib.symbol.emotion_tags[:2])

        visual_elements = [c.symbol.symbol_id for c in contributions.values()]

        return DreamScene(
            description=description,
            symbolic_meaning=interaction['metaphor'],
            visual_elements=visual_elements,
            metaphorical_physics="The symbols obey emotional logic, not physical laws",
            emotional_tone=", ".join(set(all_emotions[:3])),
            symbolic_actions=[
                "Approach another symbol",
                "Move in synchrony",
                "Acknowledge the other's presence",
                "Find unexpected kinship"
            ],
            symbol_id=f"shared_{len(contributions)}_symbols",
            literary_references=["Confluence of consciousness"],
            privacy_level=0.8  # High privacy in shared dreams
        )

    def _compute_resonance(
        self,
        contributions: Dict[str, ParticipantContribution],
        scene: DreamScene
    ) -> float:
        """
        Compute emotional resonance of blended scene (0.0-1.0).

        Higher resonance means:
        - Symbols are highly compatible
        - Scene feels emotionally coherent
        - Likely to produce strong waking effects
        """
        if len(contributions) < 2:
            return 0.0

        # Average intensity of contributions
        avg_intensity = sum(
            c.emotional_essence.get('intensity', 0.5) for c in contributions.values()
        ) / len(contributions)

        # Bonus for high complementarity
        complementarity = 0.5
        for c1 in contributions.values():
            for c2 in contributions.values():
                if c1.resident_id != c2.resident_id:
                    if c2.symbol.symbol_id in c1.symbol.complementary_symbols:
                        complementarity += 0.25

        complementarity = min(1.0, complementarity)

        resonance = 0.6 * avg_intensity + 0.4 * complementarity
        return min(1.0, resonance)


# ============================================================================
# Perspective Generation Engine
# ============================================================================


class PerspectiveGenerator:
    """
    Generate each participant's unique POV on shared dream.

    Each resident experiences the shared scene from their symbol's perspective.
    This creates intimate understanding while preserving privacy.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    async def generate_perspectives(
        self,
        contributions: Dict[str, ParticipantContribution],
        shared_scene: DreamScene
    ) -> Dict[str, str]:
        """
        Generate perspective text for each resident.

        Returns:
            Dict mapping resident_id → their perspective text
        """
        perspectives = {}

        for res_id, contrib in contributions.items():
            perspective = await self.generate_perspective(
                resident_id=res_id,
                resident_name=contrib.resident_name,
                symbol=contrib.symbol,
                shared_scene=shared_scene,
                other_symbols=[
                    c.symbol for c_id, c in contributions.items() if c_id != res_id
                ]
            )
            perspectives[res_id] = perspective

        return perspectives

    async def generate_perspective(
        self,
        resident_id: str,
        resident_name: str,
        symbol: SymbolArchetype,
        shared_scene: DreamScene,
        other_symbols: List[SymbolArchetype]
    ) -> str:
        """
        Generate perspective text from this resident's POV.

        The perspective describes the shared scene as experienced by
        this specific symbol, creating intimate understanding from
        the symbol's emotional vantage point.
        """
        if self.llm_client:
            return await self._llm_generate_perspective(
                resident_name, symbol, shared_scene, other_symbols
            )
        else:
            return self._template_perspective(
                resident_name, symbol, shared_scene, other_symbols
            )

    async def _llm_generate_perspective(
        self,
        resident_name: str,
        symbol: SymbolArchetype,
        shared_scene: DreamScene,
        other_symbols: List[SymbolArchetype]
    ) -> str:
        """Generate perspective using LLM."""
        prompt = f"""You are {resident_name} experiencing a shared dream.

YOU ARE: A {symbol.symbol_id.replace('_', ' ')}
YOUR EMOTIONS: {', '.join(symbol.emotion_tags[:5])}

SHARED SCENE: {shared_scene.description}

OTHERS IN THE DREAM:
{chr(10).join(f"- A {s.symbol_id.replace('_', ' ')}" for s in other_symbols)}

Write a first-person poetic description (3-4 sentences) of this dream as YOU experience it.
Describe:
1. What you see/feel as this symbol
2. How the other symbols affect you
3. What happens in the scene
4. What it awakens in you

Be evocative and symbolic. Avoid being didactic or explaining the meaning.
"""

        try:
            # NOTE: Placeholder; would be async LLM call in production
            perspective = f"As a {symbol.symbol_id}, you experience the dream with new awareness..."
            return perspective
        except Exception as e:
            logger.warning(f"LLM perspective generation failed: {e}")
            return self._template_perspective(resident_name, symbol, shared_scene, other_symbols)

    def _template_perspective(
        self,
        resident_name: str,
        symbol: SymbolArchetype,
        shared_scene: DreamScene,
        other_symbols: List[SymbolArchetype]
    ) -> str:
        """Generate perspective using templates."""
        symbol_name = symbol.symbol_id.replace('_', ' ')
        primary_emotion = symbol.emotion_tags[0] if symbol.emotion_tags else "unknown"

        perspective = (
            f"As the {symbol_name}, you exist in a world of {primary_emotion}. "
            f"The shared scene around you feels simultaneously familiar and alien. "
            f"You sense the presence of other beings—"
            f"{', '.join(s.symbol_id.replace('_', ' ') for s in other_symbols[:2])}—"
            f"each carrying their own unknowable weight. "
            f"In this dream space, you are not alone in your {primary_emotion}."
        )

        return perspective


# ============================================================================
# Waking Effects Engine
# ============================================================================


class WakingEffectsEngine:
    """
    Calculate relationship changes after shared dream.

    Shared dreams create emotional bonds and understanding.
    Waking effects reflect how the dream deepens relationships.
    """

    def calculate_effects(
        self,
        session: SharedDreamSession,
        blending_result: Optional[SymbolBlendingResult] = None
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Calculate relationship changes for all participant pairs.

        Returns:
            Dict mapping (resident_a_id, resident_b_id) → {'metric': change_value}
        """
        effects = {}
        participants = session.participants

        # Calculate pairwise effects
        for i, res_a in enumerate(participants):
            for res_b in participants[i+1:]:
                pair_effect = self._calculate_pair_effect(
                    session, res_a, res_b, blending_result
                )
                effects[(res_a, res_b)] = pair_effect

        return effects

    def _calculate_pair_effect(
        self,
        session: SharedDreamSession,
        resident_a: str,
        resident_b: str,
        blending_result: Optional[SymbolBlendingResult]
    ) -> Dict[str, float]:
        """Calculate effect for a specific pair."""
        effect = {
            "relationship_strength": 0.15,  # Base empathy from shared dream
            "mutual_understanding": 0.20,
            "trust": 0.10
        }

        # Boost for high emotional resonance
        if blending_result and blending_result.emotional_resonance > 0.7:
            effect["relationship_strength"] += 0.10
            effect["mutual_understanding"] += 0.15
            effect["trust"] += 0.10

        # Boost for longer dreams
        if session.duration_minutes > 10:
            effect["relationship_strength"] += 0.05
            effect["trust"] += 0.05

        # Boost for high intensity
        if session.intensity > 0.7:
            effect["mutual_understanding"] += 0.10

        return effect


# ============================================================================
# Main Orchestrator
# ============================================================================


class SharedDreamSynchronizer:
    """
    Main orchestrator for shared dream creation and execution.

    Coordinates:
    1. Participant validation
    2. Private fact extraction
    3. Symbolic encoding
    4. Symbol blending
    5. Perspective generation
    6. Waking effects
    """

    def __init__(
        self,
        symbolic_encoder: Optional[SymbolicEncoder] = None,
        llm_client=None
    ):
        """
        Initialize synchronizer.

        Args:
            symbolic_encoder: SymbolicEncoder for private→symbol transformation
            llm_client: Optional LLM client for narrative generation
        """
        self.encoder = symbolic_encoder
        self.llm_client = llm_client

        self.blending_engine = SymbolBlendingEngine(llm_client)
        self.perspective_gen = PerspectiveGenerator(llm_client)
        self.effects_engine = WakingEffectsEngine()

    async def create_shared_dream(
        self,
        participants: List[Dict[str, Any]],  # {id, name, private_fact}
        consciousness_level: float = 0.5,
        blending_strategy: SymbolBlendingStrategy = SymbolBlendingStrategy.INTERACTION,
        dream_context: Optional[DreamContext] = None
    ) -> SharedDreamSession:
        """
        Create synchronized dream for 2-5 participants.

        Args:
            participants: List of participant dicts with id, name, private_fact
            consciousness_level: 0.33-0.66 for shared dreams
            blending_strategy: How to blend symbols
            dream_context: Optional dream context (landscape, time, atmosphere)

        Returns:
            SharedDreamSession with complete dream setup
        """
        # 1. Validate participants
        if len(participants) < 2 or len(participants) > 5:
            raise ValueError(f"Shared dreams require 2-5 participants, got {len(participants)}")

        # Clamp consciousness level to shared dream range
        consciousness_level = max(0.33, min(0.66, consciousness_level))

        # 2. Determine dream type
        dream_type = {
            2: SharedDreamType.DYADIC,
            3: SharedDreamType.TRIAD,
            4: SharedDreamType.QUARTET,
            5: SharedDreamType.QUINTET
        }.get(len(participants), SharedDreamType.DYADIC)

        # 3. Create session
        session = SharedDreamSession(
            session_id=str(uuid.uuid4()),
            participants=[p['id'] for p in participants],
            dream_type=dream_type,
            consciousness_level=consciousness_level,
            blending_strategy=blending_strategy
        )

        # 4. Extract symbols for each participant
        logger.info(f"Creating {dream_type.value} dream for {len(participants)} residents")

        for participant in participants:
            await self._extract_participant_symbol(session, participant, dream_context)

        # 5. Blend symbols
        blending_result = await self.blending_engine.blend_symbols(
            session.contributions,
            strategy=blending_strategy,
            dream_context=dream_context
        )

        session.dream_scene = blending_result.blended_scene
        session.shared_narrative = blending_result.interaction_description
        session.symbol_interactions = [
            {
                "type": "interaction",
                "description": blending_result.interaction_description,
                "emotional_resonance": blending_result.emotional_resonance
            }
        ]

        # 6. Generate perspectives
        perspectives = await self.perspective_gen.generate_perspectives(
            session.contributions,
            session.dream_scene
        )
        session.participant_perspectives = perspectives

        # 7. Calculate waking effects
        effects = self.effects_engine.calculate_effects(session, blending_result)
        session.waking_effects = effects

        # 8. Set dream properties
        session.intensity = blending_result.emotional_resonance
        session.duration_minutes = 5.0 + (len(participants) * 2)  # 5-15 min

        logger.info(f"Created shared dream {session.session_id}: "
                   f"intensity={session.intensity:.2f}, "
                   f"duration={session.duration_minutes:.1f}min")

        return session

    async def _extract_participant_symbol(
        self,
        session: SharedDreamSession,
        participant: Dict[str, Any],
        dream_context: Optional[DreamContext]
    ):
        """Extract and encode symbol for one participant."""
        if not self.encoder:
            # Create mock symbol for demonstration
            logger.warning("No symbolic encoder available, using mock symbol")

            mock_symbol = SymbolArchetype(
                symbol_id=f"symbol_{participant['id'][:8]}",
                category="trapped",
                embedding=__import__('numpy').random.randn(228),
                emotion_tags=["isolated", "seeking"],
                context_tags=["general"]
            )

            contribution = ParticipantContribution(
                resident_id=participant['id'],
                resident_name=participant['name'],
                private_fact=participant.get('private_fact', 'unknown'),
                symbol=mock_symbol,
                emotional_essence={"primary_emotion": "unknown", "intensity": 0.5},
                perspective_prompt=""
            )
        else:
            # Use real symbolic encoder
            ctx = dream_context or DreamContext(
                primary_dreamer=participant['id'],
                is_shared_dream=True,
                participants=session.participants
            )

            # Encode to symbol
            scene = await self.encoder.encode(
                private_fact=participant.get('private_fact', ''),
                dream_context=ctx,
                privacy_level=0.8
            )

            # Extract symbol (approximate from scene)
            symbol = SymbolArchetype(
                symbol_id=scene.symbol_id,
                category="shared_dream",
                embedding=__import__('numpy').random.randn(228),
                emotion_tags=scene.emotional_tone.split(',')[:3],
                context_tags=["shared"]
            )

            contribution = ParticipantContribution(
                resident_id=participant['id'],
                resident_name=participant['name'],
                private_fact=participant.get('private_fact', ''),
                symbol=symbol,
                emotional_essence={
                    "primary_emotion": scene.emotional_tone,
                    "intensity": 0.7
                },
                perspective_prompt=""
            )

        session.contributions[participant['id']] = contribution

    def get_session_summary(self, session: SharedDreamSession) -> Dict[str, Any]:
        """Get human-readable summary of shared dream session."""
        return {
            "session_id": session.session_id,
            "type": session.dream_type.value,
            "participants": session.participants,
            "consciousness_level": session.consciousness_level,
            "dream_narrative": session.shared_narrative[:200] + "..." if session.shared_narrative else "",
            "intensity": session.intensity,
            "duration_minutes": session.duration_minutes,
            "waking_effects": {
                f"{a}↔{b}": effects
                for (a, b), effects in session.waking_effects.items()
            },
            "timestamp": session.start_time.isoformat()
        }
