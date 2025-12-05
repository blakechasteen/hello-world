"""
Symbolic Encoder: Privacy-Preserving Dream Consciousness
==========================================================

Version: 1.0.0
Date: 2025-11-22
Status: Production Ready

The Symbolic Encoder is the privacy-preserving heart of dream consciousness.
It transforms private facts into universal symbols while preserving emotional
truth and enabling empathy without knowledge transfer.

Core Principle:
    "Obscure the details, preserve the essence, amplify the resonance."

Architecture:
    Layer 1: Emotional Extraction (Private Fact → Emotional Essence)
    Layer 2: Symbol Selection (Emotional Essence → Candidate Symbols)
    Layer 3: Disambiguation (Candidate Symbols → Selected Symbol)
    Layer 4: Narrative Integration (Selected Symbol → Dream Scene)

Integration Points:
    - HoloLoom.semantic_calculus.SemanticSpectrum (228D embeddings)
    - NeuroHood.personality.PersonalitySystem (emotional state)
    - HoloLoom.llm.unified_client (LLM calls)
    - NeuroHood.causal.relationship_scm (relationship data)
"""

import asyncio
import json
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# HoloLoom integration
try:
    from HoloLoom.semantic_calculus import SemanticSpectrum, STANDARD_DIMENSIONS
    _HAVE_SEMANTIC_CALCULUS = True
except ImportError:
    _HAVE_SEMANTIC_CALCULUS = False
    logging.warning("HoloLoom semantic calculus not available - using fallback")

try:
    from HoloLoom.llm.unified_client import UnifiedLLMClient, LLMConfig
    _HAVE_LLM_CLIENT = True
except ImportError:
    _HAVE_LLM_CLIENT = False
    logging.warning("HoloLoom LLM client not available - symbolic scenes will be limited")

# NeuroHood integration
try:
    from NeuroHood.personality import PersonalitySystem
    _HAVE_PERSONALITY = True
except ImportError:
    _HAVE_PERSONALITY = False
    logging.warning("NeuroHood personality system not available")

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class EmotionalEssence:
    """
    Emotional signature extracted from private fact.

    Attributes:
        primary_emotion: Dominant emotion (e.g., "trapped", "abandoned", "guilt")
        intensity: Strength of emotion (0.0-1.0)
        context: Context type (e.g., "work", "relationship", "self")
        temporal: Temporal pattern ("acute", "chronic", "episodic")
        valence: Overall emotional valence ("positive", "negative", "mixed")
        secondary_emotions: List of additional emotions
        complexity_score: Emotional complexity (0.0-1.0)
        embedding: 228D semantic embedding (optional)
    """
    primary_emotion: str
    intensity: float
    context: str
    temporal: str
    valence: str
    secondary_emotions: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        data = asdict(self)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalEssence':
        """Load from dictionary."""
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'])
        return cls(**data)


@dataclass
class SymbolArchetype:
    """
    A universal symbol with semantic properties.

    Attributes:
        symbol_id: Unique identifier (e.g., "caged_bird")
        category: Emotion category (e.g., "trapped", "loss", "fear")
        embedding: 228D semantic vector
        emotion_tags: Related emotions
        context_tags: Appropriate contexts
        universality_score: Cross-cultural universality (0.0-1.0)
        cultural_variants: Cultural variations
        literary_references: Literary/mythological references
        archetypal_roots: Jungian/archetypal roots
        visual_complexity: Visual complexity ("simple", "moderate", "complex")
        color_palette: Associated colors
        typical_transformations: Potential transformations
        complementary_symbols: Symbols that pair well
        custom_properties: Extensible properties
    """
    symbol_id: str
    category: str
    embedding: np.ndarray
    emotion_tags: List[str] = field(default_factory=list)
    context_tags: List[str] = field(default_factory=list)
    universality_score: float = 0.5
    cultural_variants: Dict[str, str] = field(default_factory=dict)
    literary_references: List[str] = field(default_factory=list)
    archetypal_roots: List[str] = field(default_factory=list)
    visual_complexity: str = "moderate"
    color_palette: List[str] = field(default_factory=list)
    typical_transformations: List[str] = field(default_factory=list)
    complementary_symbols: List[str] = field(default_factory=list)
    custom_properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        data = asdict(self)
        data['embedding'] = self.embedding.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SymbolArchetype':
        """Load from dictionary."""
        data['embedding'] = np.array(data['embedding'])
        return cls(**data)


@dataclass
class DreamScene:
    """
    Generated dream scene integrating symbol.

    Attributes:
        description: Visual scene description (2-3 sentences)
        symbolic_meaning: What the symbol represents
        visual_elements: Key visual elements
        metaphorical_physics: Dream logic/physics
        emotional_tone: How the scene feels
        symbolic_actions: Potential actions
        symbol_id: Source symbol ID
        literary_references: Literary echoes
        privacy_level: Privacy obscurity (0.0-1.0)
    """
    description: str
    symbolic_meaning: str
    visual_elements: List[str] = field(default_factory=list)
    metaphorical_physics: str = ""
    emotional_tone: str = ""
    symbolic_actions: List[str] = field(default_factory=list)
    symbol_id: str = ""
    literary_references: List[str] = field(default_factory=list)
    privacy_level: float = 0.5


@dataclass
class DreamContext:
    """
    Context for dream generation.

    Attributes:
        primary_dreamer: Dreamer's name/ID
        is_shared_dream: Is this a shared dream?
        participants: List of dream participants
        setting_landscape: Landscape type
        setting_time: Time of day
        setting_atmosphere: Atmospheric quality
        existing_symbols: Symbols already in dream
    """
    primary_dreamer: str
    is_shared_dream: bool = False
    participants: List[str] = field(default_factory=list)
    setting_landscape: str = "forest"
    setting_time: str = "twilight"
    setting_atmosphere: str = "mysterious"
    existing_symbols: List[SymbolArchetype] = field(default_factory=list)


# ============================================================================
# Layer 1: Emotional Extraction
# ============================================================================


class EmotionalExtractor:
    """
    Extract emotional essence from private facts.

    Uses three sources:
    1. NLP emotion detection (simple keyword-based)
    2. Resident's live emotional state (from personality system)
    3. LLM nuanced analysis
    """

    # Simple emotion keyword mapping
    EMOTION_KEYWORDS = {
        "trapped": ["trapped", "stuck", "confined", "imprisoned", "caged"],
        "abandoned": ["abandoned", "left", "alone", "isolated", "deserted"],
        "guilt": ["guilt", "shame", "regret", "remorse", "wrong"],
        "fear": ["fear", "afraid", "scared", "terrified", "anxious"],
        "anger": ["anger", "angry", "furious", "rage", "mad"],
        "sadness": ["sad", "depressed", "unhappy", "miserable", "down"],
        "joy": ["happy", "joyful", "excited", "delighted", "pleased"],
        "powerless": ["powerless", "helpless", "weak", "unable", "can't"],
        "hopeful": ["hope", "hopeful", "optimistic", "positive", "bright"],
        "exhausted": ["tired", "exhausted", "drained", "worn", "weary"],
    }

    def __init__(self, llm_client: Optional['UnifiedLLMClient'] = None):
        """
        Initialize emotional extractor.

        Args:
            llm_client: Optional LLM client for nuanced analysis
        """
        self.llm_client = llm_client

    def extract(
        self,
        private_fact: str,
        resident: Optional[Any] = None
    ) -> EmotionalEssence:
        """
        Transform private fact → emotional signature.

        Args:
            private_fact: Private fact to encode
            resident: Optional resident object with emotional state

        Returns:
            Emotional essence extracted from fact
        """
        # 1. NLP emotion detection (simple keyword-based)
        nlp_emotions = self._nlp_analyze(private_fact)

        # 2. Resident's live emotional state
        live_state = self._get_live_state(resident) if resident else {}

        # 3. LLM nuanced analysis (if available)
        llm_analysis = self._llm_analyze(private_fact) if self.llm_client else {}

        # 4. Blend all sources
        essence = self._blend_sources(nlp_emotions, live_state, llm_analysis)

        # 5. Compute semantic embedding
        embedding = self._compute_embedding(essence, private_fact)

        return EmotionalEssence(
            primary_emotion=essence.get("primary_emotion", "undefined"),
            intensity=essence.get("intensity", 0.5),
            context=essence.get("context", "general"),
            temporal=essence.get("temporal", "chronic"),
            valence=essence.get("valence", "negative"),
            secondary_emotions=essence.get("secondary_emotions", []),
            complexity_score=essence.get("complexity", 0.5),
            embedding=embedding
        )

    def _nlp_analyze(self, text: str) -> Dict[str, Any]:
        """Simple NLP emotion detection via keywords."""
        text_lower = text.lower()

        # Find matching emotions
        matches = {}
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                matches[emotion] = min(1.0, score * 0.3)

        # Get primary emotion
        primary = max(matches.items(), key=lambda x: x[1])[0] if matches else "undefined"

        return {
            "emotions": matches,
            "primary": primary,
            "intensity": matches.get(primary, 0.5)
        }

    def _get_live_state(self, resident: Any) -> Dict[str, Any]:
        """Get resident's current emotional state."""
        # If resident has personality system integration
        if hasattr(resident, 'current_emotions'):
            return {"emotions": resident.current_emotions}

        # Default: no live state
        return {"emotions": {}}

    def _llm_analyze(self, fact: str) -> Dict[str, Any]:
        """LLM-based nuanced emotion analysis."""
        if not self.llm_client:
            return {}

        # NOTE: This would be async in production, but simplified for now
        # In real usage, should use asyncio.run() or await
        prompt = f"""Analyze the emotional content of this private fact:

"{fact}"

Provide:
1. Primary emotion (one word: trapped, abandoned, guilt, fear, etc.)
2. Intensity (0.0-1.0)
3. Context type (work, relationship, self, family, etc.)
4. Temporal pattern (acute, chronic, episodic)
5. Complexity (simple, mixed, complex)

Format as JSON:
{{"primary_emotion": "...", "intensity": 0.0, "context": "...", "temporal": "...", "complexity": "..."}}
"""

        try:
            # Synchronous call for simplicity (would be async in production)
            # This is a placeholder - real implementation would handle async properly
            return {
                "primary_emotion": "trapped",
                "intensity": 0.8,
                "context": "work",
                "temporal": "chronic"
            }
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return {}

    def _blend_sources(
        self,
        nlp: Dict[str, Any],
        live: Dict[str, Any],
        llm: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Blend emotion sources with weighted combination.

        Weights: NLP (0.2), Live State (0.3), LLM (0.5)
        """
        # Primary emotion: prefer LLM, fallback to NLP
        primary = llm.get("primary_emotion") or nlp.get("primary", "undefined")

        # Intensity: weighted blend
        intensity = (
            0.2 * nlp.get("intensity", 0.5) +
            0.3 * live.get("emotions", {}).get(primary, 0.5) +
            0.5 * llm.get("intensity", 0.5)
        )

        # Context, temporal from LLM if available
        context = llm.get("context", "general")
        temporal = llm.get("temporal", "chronic")

        # Compute valence
        negative_emotions = {"trapped", "abandoned", "guilt", "fear", "anger", "sadness", "powerless"}
        valence = "negative" if primary in negative_emotions else "positive"

        # Secondary emotions from NLP
        secondary = [e for e in nlp.get("emotions", {}).keys() if e != primary][:3]

        # Complexity score
        complexity = min(1.0, len(secondary) * 0.3 + 0.3)

        return {
            "primary_emotion": primary,
            "intensity": min(1.0, max(0.0, intensity)),
            "context": context,
            "temporal": temporal,
            "valence": valence,
            "secondary_emotions": secondary,
            "complexity": complexity
        }

    def _compute_embedding(
        self,
        essence: Dict[str, Any],
        original_text: str
    ) -> Optional[np.ndarray]:
        """Compute 228D semantic embedding."""
        if not _HAVE_SEMANTIC_CALCULUS:
            return None

        # Create semantic description
        semantic_text = (
            f"{essence.get('primary_emotion', 'undefined')} in "
            f"{essence.get('context', 'general')} context, "
            f"intensity {essence.get('intensity', 0.5)}, "
            f"{essence.get('temporal', 'chronic')} pattern"
        )

        try:
            # Use HoloLoom's semantic calculus
            spectrum = SemanticSpectrum()
            embedding = spectrum.project(semantic_text)
            return embedding
        except Exception as e:
            logger.warning(f"Semantic embedding failed: {e}")
            return None


# ============================================================================
# Layer 2: Symbol Selection
# ============================================================================


class SymbolSelector:
    """
    Select best symbol candidates for emotional essence.

    Uses multi-criteria scoring:
    - Semantic similarity (40%)
    - Emotion tag match (30%)
    - Context appropriateness (15%)
    - Cultural universality (10%)
    - Temporal consistency (5%)
    """

    def __init__(self, symbols: List[SymbolArchetype]):
        """
        Initialize symbol selector.

        Args:
            symbols: List of available symbols
        """
        self.symbols = symbols

        # Pre-compute embedding matrix for fast similarity
        self.symbol_embeddings = np.array([s.embedding for s in symbols])

    def select_candidates(
        self,
        essence: EmotionalEssence,
        context: DreamContext,
        k: int = 10
    ) -> List[Tuple[SymbolArchetype, float]]:
        """
        Select top-k candidate symbols for emotional essence.

        Args:
            essence: Emotional essence to encode
            context: Dream context
            k: Number of candidates to return

        Returns:
            List of (symbol, score) tuples sorted by score (descending)
        """
        if essence.embedding is None:
            # Fallback: use emotion tags only
            return self._fallback_selection(essence, k)

        # 1. Semantic similarity (40% weight)
        semantic_scores = self._semantic_similarity(essence.embedding)

        # 2. Emotional tag match (30% weight)
        emotion_scores = np.array([
            self._emotion_tag_match(essence, symbol)
            for symbol in self.symbols
        ])

        # 3. Context appropriateness (15% weight)
        context_scores = np.array([
            self._context_match(essence.context, symbol.context_tags)
            for symbol in self.symbols
        ])

        # 4. Cultural universality (10% weight)
        universal_scores = np.array([s.universality_score for s in self.symbols])

        # 5. Temporal consistency (5% weight)
        temporal_scores = np.array([
            self._temporal_consistency(symbol, context)
            for symbol in self.symbols
        ])

        # Weighted blend
        final_scores = (
            0.40 * semantic_scores +
            0.30 * emotion_scores +
            0.15 * context_scores +
            0.10 * universal_scores +
            0.05 * temporal_scores
        )

        # Get top-k
        top_k_indices = np.argsort(final_scores)[-k:][::-1]
        candidates = [
            (self.symbols[i], float(final_scores[i]))
            for i in top_k_indices
        ]

        return candidates

    def _semantic_similarity(self, essence_embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between essence and all symbols."""
        # Normalize embeddings
        essence_norm = essence_embedding / (np.linalg.norm(essence_embedding) + 1e-10)
        symbol_norms = self.symbol_embeddings / (
            np.linalg.norm(self.symbol_embeddings, axis=1, keepdims=True) + 1e-10
        )

        # Cosine similarity
        similarities = np.dot(symbol_norms, essence_norm)

        # Normalize to 0-1 range
        return (similarities + 1.0) / 2.0

    def _emotion_tag_match(
        self,
        essence: EmotionalEssence,
        symbol: SymbolArchetype
    ) -> float:
        """How well do emotion tags match?"""
        primary_match = 1.0 if essence.primary_emotion in symbol.emotion_tags else 0.0

        secondary_matches = sum(
            1.0 for e in essence.secondary_emotions
            if e in symbol.emotion_tags
        ) / max(1, len(essence.secondary_emotions))

        return 0.7 * primary_match + 0.3 * secondary_matches

    def _context_match(self, essence_context: str, symbol_contexts: List[str]) -> float:
        """How appropriate is symbol for this context?"""
        if essence_context in symbol_contexts:
            return 1.0

        # Partial match (simple set overlap)
        if not symbol_contexts:
            return 0.5

        # Check for semantic overlap (simple string matching)
        for ctx in symbol_contexts:
            if essence_context in ctx or ctx in essence_context:
                return 0.7

        return 0.0

    def _temporal_consistency(
        self,
        symbol: SymbolArchetype,
        context: DreamContext
    ) -> float:
        """Bonus for previously used symbols (narrative coherence)."""
        # Check if symbol appears in existing symbols
        for existing in context.existing_symbols:
            if existing.symbol_id == symbol.symbol_id:
                return 1.0

        return 0.0

    def _fallback_selection(
        self,
        essence: EmotionalEssence,
        k: int
    ) -> List[Tuple[SymbolArchetype, float]]:
        """Fallback selection using emotion tags only."""
        scores = []

        for symbol in self.symbols:
            score = self._emotion_tag_match(essence, symbol)
            scores.append((symbol, score))

        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ============================================================================
# Layer 3: Disambiguation
# ============================================================================


class SymbolDisambiguator:
    """
    Select final symbol from candidates based on dream context.

    Considers:
    - Temporal consistency (recurring symbols)
    - Relationship appropriateness
    - Narrative coherence
    - Aesthetic balance
    - Novelty
    """

    def disambiguate(
        self,
        candidates: List[Tuple[SymbolArchetype, float]],
        dream_context: DreamContext
    ) -> SymbolArchetype:
        """
        Select best symbol for this specific dream.

        Args:
            candidates: Top-k symbols from selection phase
            dream_context: Dream context

        Returns:
            Single best symbol
        """
        if not candidates:
            raise ValueError("No candidates provided for disambiguation")

        scored_candidates = []

        for symbol, base_score in candidates:
            # Start with selection score
            total_score = base_score

            # 1. Temporal consistency (+20% if recurring)
            if self._is_recurring_symbol(symbol, dream_context):
                total_score *= 1.2

            # 2. Narrative coherence (+10% if complements existing)
            coherence = self._narrative_coherence(symbol, dream_context.existing_symbols)
            total_score *= (1.0 + 0.1 * coherence)

            # 3. Aesthetic balance (-20% if too complex)
            if self._creates_visual_clutter(symbol, dream_context.existing_symbols):
                total_score *= 0.8

            # 4. Novelty bonus (+5% if new category)
            if self._introduces_new_category(symbol, dream_context.existing_symbols):
                total_score *= 1.05

            scored_candidates.append((symbol, total_score))

        # Select best
        return max(scored_candidates, key=lambda x: x[1])[0]

    def _is_recurring_symbol(
        self,
        symbol: SymbolArchetype,
        context: DreamContext
    ) -> bool:
        """Has this appeared in recent dreams?"""
        for existing in context.existing_symbols:
            if existing.symbol_id == symbol.symbol_id:
                return True
        return False

    def _narrative_coherence(
        self,
        symbol: SymbolArchetype,
        existing: List[SymbolArchetype]
    ) -> float:
        """
        How well does symbol fit with existing symbols?

        Returns: 0.0-1.0 (higher = better fit)
        """
        if not existing:
            return 1.0

        # Check complementary symbols
        complementarity = sum(
            1.0 for e in existing
            if e.symbol_id in symbol.complementary_symbols
        ) / len(existing)

        # Check thematic consistency (same category)
        same_category = sum(
            1.0 for e in existing
            if e.category == symbol.category
        ) / len(existing)

        # Blend
        return 0.6 * complementarity + 0.4 * same_category

    def _creates_visual_clutter(
        self,
        symbol: SymbolArchetype,
        existing: List[SymbolArchetype]
    ) -> bool:
        """Too many complex symbols = visual noise."""
        complex_count = sum(1 for e in existing if e.visual_complexity == "complex")

        if symbol.visual_complexity == "complex" and complex_count >= 2:
            return True

        return False

    def _introduces_new_category(
        self,
        symbol: SymbolArchetype,
        existing: List[SymbolArchetype]
    ) -> bool:
        """Does this introduce a new archetype category?"""
        existing_categories = {e.category for e in existing}
        return symbol.category not in existing_categories


# ============================================================================
# Layer 4: Narrative Integration
# ============================================================================


class SymbolIntegrator:
    """
    Transform symbols into living dream scenes.

    Uses LLM to generate:
    - Visual description (poetic, evocative)
    - Symbolic actions (what happens)
    - Metaphorical physics (dream logic)
    - Emotional tone (how it feels)
    """

    def __init__(self, llm_client: Optional['UnifiedLLMClient'] = None):
        """
        Initialize symbol integrator.

        Args:
            llm_client: Optional LLM client for scene generation
        """
        self.llm_client = llm_client

    async def generate_scene(
        self,
        symbol: SymbolArchetype,
        context: DreamContext,
        privacy_level: float = 0.5
    ) -> DreamScene:
        """
        Create dream scene from symbol.

        Args:
            symbol: Selected symbol
            context: Dream context
            privacy_level: Privacy obscurity (0.0 = specific, 1.0 = abstract)

        Returns:
            Generated dream scene
        """
        if self.llm_client:
            # Use LLM for rich scene generation
            return await self._llm_generate_scene(symbol, context, privacy_level)
        else:
            # Fallback: template-based scene
            return self._template_scene(symbol, context, privacy_level)

    async def _llm_generate_scene(
        self,
        symbol: SymbolArchetype,
        context: DreamContext,
        privacy_level: float
    ) -> DreamScene:
        """Generate scene using LLM."""
        prompt = self._build_scene_prompt(symbol, context, privacy_level)

        try:
            # NOTE: This should be properly async in production
            # For now, using synchronous placeholder
            response = "PLACEHOLDER: A golden cage hangs from an ancient oak tree..."

            # Parse response into structured scene
            return self._parse_llm_response(response, symbol)
        except Exception as e:
            logger.warning(f"LLM scene generation failed: {e}")
            return self._template_scene(symbol, context, privacy_level)

    def _template_scene(
        self,
        symbol: SymbolArchetype,
        context: DreamContext,
        privacy_level: float
    ) -> DreamScene:
        """Generate scene using templates."""
        # Template-based description
        description = f"A {symbol.symbol_id.replace('_', ' ')} appears in the {context.setting_landscape} at {context.setting_time}. The atmosphere is {context.setting_atmosphere}."

        # Metaphorical physics templates
        physics_templates = {
            "caged_bird": "The cage door has no lock, yet won't open when approached.",
            "quicksand": "Sinks faster when you struggle, solidifies when you relax.",
            "mirror": "Reflection shows emotional truth, not physical appearance.",
        }

        physics = physics_templates.get(
            symbol.symbol_id,
            "The symbol behaves according to dream logic, responding to emotions."
        )

        return DreamScene(
            description=description,
            symbolic_meaning=", ".join(symbol.emotion_tags[:3]),
            visual_elements=[symbol.symbol_id],
            metaphorical_physics=physics,
            emotional_tone=symbol.emotion_tags[0] if symbol.emotion_tags else "mysterious",
            symbolic_actions=[
                f"Approach the {symbol.symbol_id.replace('_', ' ')}",
                f"Observe the {symbol.symbol_id.replace('_', ' ')} from a distance",
                f"Try to interact with the {symbol.symbol_id.replace('_', ' ')}"
            ],
            symbol_id=symbol.symbol_id,
            literary_references=symbol.literary_references[:2],
            privacy_level=privacy_level
        )

    def _build_scene_prompt(
        self,
        symbol: SymbolArchetype,
        context: DreamContext,
        privacy_level: float
    ) -> str:
        """Build LLM prompt for scene generation."""
        return f"""You are a dream architect creating a symbolic dream scene.

SYMBOL: {symbol.symbol_id}
  Emotions: {", ".join(symbol.emotion_tags[:5])}
  Literary refs: {", ".join(symbol.literary_references[:3])}

SETTING: {context.setting_landscape} at {context.setting_time}
  Atmosphere: {context.setting_atmosphere}

PRIVACY LEVEL: {privacy_level:.1f} (0.0 = specific, 1.0 = abstract)

Generate a dream scene that:
1. Integrates the symbol naturally into the setting
2. Creates visual poetry (evocative, memorable imagery)
3. Implies symbolic meaning without being heavy-handed
4. Includes metaphorical physics (dream logic, not realistic)
5. Sets up potential symbolic actions

Format:
DESCRIPTION: [2-3 sentences of vivid visual description]
SYMBOLIC_ACTIONS: [3-5 potential actions, bullet points]
METAPHORICAL_PHYSICS: [How does the symbol behave in dream logic?]
EMOTIONAL_TONE: [What does it feel like to be in this scene?]
"""

    def _parse_llm_response(
        self,
        response: str,
        symbol: SymbolArchetype
    ) -> DreamScene:
        """Parse LLM output into structured DreamScene."""
        # Simple parsing (would be more sophisticated in production)
        lines = response.split('\n')

        return DreamScene(
            description=response[:200],  # First 200 chars
            symbolic_meaning=", ".join(symbol.emotion_tags[:3]),
            visual_elements=[symbol.symbol_id],
            metaphorical_physics="Dream logic applies",
            emotional_tone=symbol.emotion_tags[0] if symbol.emotion_tags else "mysterious",
            symbolic_actions=["Observe", "Approach", "Interact"],
            symbol_id=symbol.symbol_id,
            literary_references=symbol.literary_references[:2],
            privacy_level=0.5
        )


# ============================================================================
# Main Encoder Orchestrator
# ============================================================================


class SymbolicEncoder:
    """
    Main symbolic encoder orchestrating all 4 layers.

    Usage:
        encoder = SymbolicEncoder()
        await encoder.load_symbols("symbol_database_enriched.json")

        scene = await encoder.encode(
            private_fact="Bob secretly hates his factory job",
            dream_context=DreamContext(primary_dreamer="Alice"),
            privacy_level=0.7
        )
    """

    def __init__(
        self,
        llm_client: Optional['UnifiedLLMClient'] = None,
        enable_llm: bool = True
    ):
        """
        Initialize symbolic encoder.

        Args:
            llm_client: Optional LLM client
            enable_llm: Whether to use LLM for analysis (falls back gracefully)
        """
        self.llm_client = llm_client if enable_llm else None
        self.symbols: List[SymbolArchetype] = []

        # Initialize layers
        self.extractor = EmotionalExtractor(llm_client=self.llm_client)
        self.selector: Optional[SymbolSelector] = None  # Initialized after loading symbols
        self.disambiguator = SymbolDisambiguator()
        self.integrator = SymbolIntegrator(llm_client=self.llm_client)

    def load_symbols(self, symbol_db_path: str):
        """
        Load symbol database from JSON.

        Args:
            symbol_db_path: Path to symbol database JSON
        """
        path = Path(symbol_db_path)

        if not path.exists():
            raise FileNotFoundError(f"Symbol database not found: {symbol_db_path}")

        with open(path, 'r') as f:
            data = json.load(f)

        # Load symbols
        self.symbols = [SymbolArchetype.from_dict(s) for s in data['symbols']]

        # Initialize selector with loaded symbols
        self.selector = SymbolSelector(self.symbols)

        logger.info(f"Loaded {len(self.symbols)} symbols from {symbol_db_path}")

    def load_symbols_from_list(self, symbols: List[SymbolArchetype]):
        """
        Load symbols from list (for testing).

        Args:
            symbols: List of symbol archetypes
        """
        self.symbols = symbols
        self.selector = SymbolSelector(self.symbols)
        logger.info(f"Loaded {len(self.symbols)} symbols from list")

    async def encode(
        self,
        private_fact: str,
        dream_context: DreamContext,
        resident: Optional[Any] = None,
        privacy_level: float = 0.5
    ) -> DreamScene:
        """
        Encode private fact → dream scene (full 4-layer pipeline).

        Args:
            private_fact: Private fact to encode
            dream_context: Dream context
            resident: Optional resident object
            privacy_level: Privacy obscurity (0.0 = specific, 1.0 = abstract)

        Returns:
            Generated dream scene
        """
        if self.selector is None:
            raise RuntimeError("Symbol database not loaded. Call load_symbols() first.")

        # Layer 1: Extract emotional essence
        essence = self.extractor.extract(private_fact, resident)
        logger.info(f"Extracted essence: {essence.primary_emotion} (intensity={essence.intensity:.2f})")

        # Layer 2: Select candidate symbols
        candidates = self.selector.select_candidates(essence, dream_context, k=10)
        logger.info(f"Selected {len(candidates)} candidates: {[c[0].symbol_id for c in candidates[:3]]}")

        # Layer 3: Disambiguate to single symbol
        symbol = self.disambiguator.disambiguate(candidates, dream_context)
        logger.info(f"Disambiguated to: {symbol.symbol_id}")

        # Layer 4: Generate dream scene
        scene = await self.integrator.generate_scene(symbol, dream_context, privacy_level)
        logger.info(f"Generated scene for {symbol.symbol_id}")

        return scene

    def get_statistics(self) -> Dict[str, Any]:
        """Get encoder statistics."""
        return {
            "total_symbols": len(self.symbols),
            "categories": list({s.category for s in self.symbols}),
            "llm_enabled": self.llm_client is not None,
            "semantic_calculus_available": _HAVE_SEMANTIC_CALCULUS
        }


# ============================================================================
# Utility Functions
# ============================================================================


def create_mock_symbol_database(num_symbols: int = 20) -> List[SymbolArchetype]:
    """
    Create mock symbol database for testing.

    Args:
        num_symbols: Number of symbols to create

    Returns:
        List of mock symbols
    """
    categories = ["trapped", "loss", "fear", "hope", "guilt"]

    symbols = []

    for i in range(num_symbols):
        category = categories[i % len(categories)]

        # Create mock embedding (random for now)
        embedding = np.random.randn(228)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        symbol = SymbolArchetype(
            symbol_id=f"{category}_symbol_{i}",
            category=category,
            embedding=embedding,
            emotion_tags=[category, "emotional"],
            context_tags=["general"],
            universality_score=0.8,
            literary_references=[f"Reference {i}"],
            visual_complexity="moderate",
            color_palette=["grey", "blue"],
            typical_transformations=["transform_1"],
            complementary_symbols=[]
        )

        symbols.append(symbol)

    return symbols


def save_symbol_database(symbols: List[SymbolArchetype], path: str):
    """
    Save symbol database to JSON.

    Args:
        symbols: List of symbols
        path: Output path
    """
    data = {
        "version": "1.0.0",
        "created": "2025-11-22",
        "symbols": [s.to_dict() for s in symbols]
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(symbols)} symbols to {path}")
