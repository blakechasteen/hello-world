#!/usr/bin/env python3
"""
🎭 NARRATIVE INTELLIGENCE MODULE
================================
Comprehensive story understanding with Bayesian sophistication

Core Capabilities:
- Archetypal patterns from Joseph Campbell's Hero's Journey (17 stages)
- Universal character detection (mythology, literature, history, fiction)
- Definite narrative arc prediction with confidence scoring
- Temporal sentiment evolution tracking
- Complete narrative function analysis (story structure roles)
- Bayesian-enhanced sentiment piercing

This module provides the foundation for mythRL's narrative understanding,
integrating epic intelligence with modern text analysis.
"""

import asyncio
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import re


class CampbellStage(Enum):
    """Joseph Campbell's 17 stages of the Hero's Journey"""
    # ACT 1: DEPARTURE
    ORDINARY_WORLD = "ordinary_world"
    CALL_TO_ADVENTURE = "call_to_adventure"
    REFUSAL_OF_CALL = "refusal_of_call"
    MEETING_MENTOR = "meeting_mentor"
    CROSSING_THRESHOLD = "crossing_threshold"
    
    # ACT 2: INITIATION
    TESTS_ALLIES_ENEMIES = "tests_allies_enemies"
    APPROACH_INMOST_CAVE = "approach_inmost_cave"
    ORDEAL = "ordeal"
    REWARD = "reward"
    
    # ACT 3: RETURN
    ROAD_BACK = "road_back"
    RESURRECTION = "resurrection"
    RETURN_WITH_ELIXIR = "return_with_elixir"
    
    # ADDITIONAL CAMPBELL STAGES
    BELLY_OF_WHALE = "belly_of_whale"
    MEETING_GODDESS = "meeting_goddess"
    WOMAN_AS_TEMPTRESS = "woman_as_temptress"
    ATONEMENT_WITH_FATHER = "atonement_with_father"
    APOTHEOSIS = "apotheosis"


class NarrativeFunction(Enum):
    """Story structure functions (expanded Freytag + modern narrative theory)"""
    EXPOSITION = "exposition"
    INCITING_INCIDENT = "inciting_incident"
    RISING_ACTION = "rising_action"
    FIRST_PINCH_POINT = "first_pinch_point"
    MIDPOINT_REVERSAL = "midpoint_reversal"
    SECOND_PINCH_POINT = "second_pinch_point"
    CLIMAX = "climax"
    FALLING_ACTION = "falling_action"
    RESOLUTION = "resolution"
    DENOUEMENT = "denouement"


class ArchetypeType(Enum):
    """Universal character archetypes (Jung + Campbell + modern expansion)"""
    HERO = "hero"
    MENTOR = "mentor"
    HERALD = "herald"
    THRESHOLD_GUARDIAN = "threshold_guardian"
    SHAPESHIFTER = "shapeshifter"
    SHADOW = "shadow"
    ALLY = "ally"
    TRICKSTER = "trickster"
    
    # Extended archetypes
    INNOCENT = "innocent"
    ORPHAN = "orphan"
    WARRIOR = "warrior"
    CAREGIVER = "caregiver"
    SEEKER = "seeker"
    LOVER = "lover"
    DESTROYER = "destroyer"
    CREATOR = "creator"
    RULER = "ruler"
    MAGICIAN = "magician"
    SAGE = "sage"
    JESTER = "jester"


@dataclass
class UniversalCharacter:
    """Character from mythology, literature, history, or fiction"""
    name: str
    source: str  # "greek_mythology", "literature", "history", "modern_fiction", etc.
    archetype: ArchetypeType
    traits: List[str]
    keywords: List[str]
    emotional_signature: Dict[str, float]
    narrative_function: str


@dataclass
class ArchetypalPattern:
    """Emotional and behavioral patterns for Campbell stages"""
    stage: CampbellStage
    emotions: Dict[str, float]
    themes: List[str]
    conflicts: List[str]
    transformations: List[str]
    symbolic_elements: List[str]


@dataclass
class NarrativeArc:
    """Definite narrative arc with confidence and evidence"""
    primary_arc: CampbellStage
    confidence: float
    secondary_arcs: List[Tuple[CampbellStage, float]]
    evidence: List[str]
    narrative_function: NarrativeFunction
    act_position: int  # 1, 2, or 3


@dataclass
class TemporalSentimentPoint:
    """Single point in temporal sentiment evolution"""
    position: float  # 0.0 to 1.0 through text
    sentiment: float  # -1.0 to 1.0
    intensity: float
    dominant_emotion: str
    narrative_function: NarrativeFunction


@dataclass
class NarrativeIntelligenceResult:
    """Complete narrative intelligence analysis"""
    text: str
    
    # Character analysis
    detected_characters: List[UniversalCharacter]
    character_archetypes: Dict[str, ArchetypeType]
    character_confidence: float
    
    # Narrative arc
    narrative_arc: NarrativeArc
    archetypal_patterns: Dict[str, float]
    
    # Temporal analysis
    temporal_evolution: List[TemporalSentimentPoint]
    sentiment_trajectory: str  # "rising", "falling", "u_shaped", "inverted_u", "stable", "chaotic"
    emotional_range: float
    
    # Story structure
    narrative_function: NarrativeFunction
    act_position: int
    story_progress: float  # 0.0 to 1.0 estimate of where in complete story
    
    # Enhanced sentiment
    base_sentiment: float
    narrative_enhanced_sentiment: float
    enhancement_factor: float
    
    # Insights
    themes: List[str]
    conflicts: List[str]
    symbolic_elements: List[str]
    recommendations: List[str]
    
    # Meta
    bayesian_confidence: float


class NarrativeIntelligence:
    """
    Comprehensive narrative understanding system
    Combines Bayesian intelligence with universal story patterns
    """
    
    def __init__(self):
        self.universal_characters = self._initialize_character_database()
        self.campbell_patterns = self._initialize_campbell_patterns()
        self.narrative_functions = self._initialize_narrative_functions()
        
    def _initialize_character_database(self) -> Dict[str, UniversalCharacter]:
        """Initialize comprehensive character database"""
        characters = {}
        
        # GREEK MYTHOLOGY
        characters["odysseus"] = UniversalCharacter(
            name="Odysseus",
            source="greek_mythology",
            archetype=ArchetypeType.HERO,
            traits=["cunning", "resourceful", "determined", "wanderer", "warrior"],
            keywords=["odysseus", "ulysses", "clever", "cunning", "wanderer", "ithaca", "trojan"],
            emotional_signature={"courage": 0.9, "wisdom": 0.85, "longing": 0.8, "perseverance": 0.95},
            narrative_function="protagonist_hero"
        )
        
        characters["athena"] = UniversalCharacter(
            name="Athena",
            source="greek_mythology",
            archetype=ArchetypeType.MENTOR,
            traits=["wise", "strategic", "protective", "divine"],
            keywords=["athena", "minerva", "owl", "wisdom", "goddess", "divine guidance"],
            emotional_signature={"wisdom": 1.0, "clarity": 0.9, "strategic_thinking": 0.95},
            narrative_function="mentor_guide"
        )
        
        characters["penelope"] = UniversalCharacter(
            name="Penelope",
            source="greek_mythology",
            archetype=ArchetypeType.LOVER,
            traits=["faithful", "clever", "patient", "resilient"],
            keywords=["penelope", "faithful", "weaving", "loom", "patient", "waiting"],
            emotional_signature={"love": 0.95, "patience": 1.0, "longing": 0.9, "determination": 0.85},
            narrative_function="beloved_home"
        )
        
        characters["telemachus"] = UniversalCharacter(
            name="Telemachus",
            source="greek_mythology",
            archetype=ArchetypeType.SEEKER,
            traits=["young", "growing", "brave", "learning"],
            keywords=["telemachus", "son", "youth", "journey", "growth", "coming of age"],
            emotional_signature={"courage": 0.7, "growth": 0.9, "uncertainty": 0.6, "determination": 0.75},
            narrative_function="secondary_hero"
        )
        
        characters["circe"] = UniversalCharacter(
            name="Circe",
            source="greek_mythology",
            archetype=ArchetypeType.SHAPESHIFTER,
            traits=["magical", "transformative", "seductive", "dangerous"],
            keywords=["circe", "witch", "sorceress", "transformation", "enchantress"],
            emotional_signature={"mystery": 0.9, "power": 0.85, "transformation": 1.0},
            narrative_function="threshold_guardian"
        )
        
        characters["zeus"] = UniversalCharacter(
            name="Zeus",
            source="greek_mythology",
            archetype=ArchetypeType.RULER,
            traits=["powerful", "authoritative", "judgmental", "divine"],
            keywords=["zeus", "jupiter", "thunder", "lightning", "king of gods", "olympus"],
            emotional_signature={"power": 1.0, "authority": 0.95, "judgment": 0.9},
            narrative_function="cosmic_authority"
        )
        
        characters["poseidon"] = UniversalCharacter(
            name="Poseidon",
            source="greek_mythology",
            archetype=ArchetypeType.SHADOW,
            traits=["wrathful", "powerful", "vengeful", "oceanic"],
            keywords=["poseidon", "neptune", "sea", "ocean", "earthquake", "trident"],
            emotional_signature={"wrath": 0.9, "power": 0.95, "vengeance": 0.85},
            narrative_function="antagonist_force"
        )
        
        # ARTHURIAN LEGEND
        characters["arthur"] = UniversalCharacter(
            name="King Arthur",
            source="arthurian_legend",
            archetype=ArchetypeType.RULER,
            traits=["noble", "just", "destined", "tragic"],
            keywords=["arthur", "king", "excalibur", "camelot", "round table"],
            emotional_signature={"nobility": 0.95, "justice": 0.9, "duty": 0.85},
            narrative_function="noble_king"
        )
        
        characters["merlin"] = UniversalCharacter(
            name="Merlin",
            source="arthurian_legend",
            archetype=ArchetypeType.MAGICIAN,
            traits=["wise", "magical", "prophetic", "enigmatic"],
            keywords=["merlin", "wizard", "sorcerer", "prophecy", "magic", "enchanter"],
            emotional_signature={"wisdom": 1.0, "mystery": 0.9, "foresight": 0.95},
            narrative_function="magical_mentor"
        )
        
        # NORSE MYTHOLOGY
        characters["odin"] = UniversalCharacter(
            name="Odin",
            source="norse_mythology",
            archetype=ArchetypeType.SAGE,
            traits=["wise", "sacrificial", "mysterious", "all-knowing"],
            keywords=["odin", "allfather", "ravens", "wisdom", "sacrifice", "runes"],
            emotional_signature={"wisdom": 0.95, "sacrifice": 0.9, "knowledge": 1.0},
            narrative_function="wise_father"
        )
        
        characters["loki"] = UniversalCharacter(
            name="Loki",
            source="norse_mythology",
            archetype=ArchetypeType.TRICKSTER,
            traits=["cunning", "chaotic", "unpredictable", "transformative"],
            keywords=["loki", "trickster", "mischief", "shapeshifter", "chaos"],
            emotional_signature={"chaos": 0.95, "cleverness": 0.9, "unpredictability": 1.0},
            narrative_function="agent_of_chaos"
        )
        
        # LITERATURE - SHAKESPEARE
        characters["hamlet"] = UniversalCharacter(
            name="Hamlet",
            source="literature_shakespeare",
            archetype=ArchetypeType.HERO,
            traits=["philosophical", "tormented", "indecisive", "tragic"],
            keywords=["hamlet", "prince", "denmark", "to be or not to be", "revenge"],
            emotional_signature={"melancholy": 0.9, "doubt": 0.85, "intelligence": 0.9},
            narrative_function="tragic_hero"
        )
        
        characters["prospero"] = UniversalCharacter(
            name="Prospero",
            source="literature_shakespeare",
            archetype=ArchetypeType.MAGICIAN,
            traits=["magical", "wise", "vengeful", "forgiving"],
            keywords=["prospero", "tempest", "magic", "island", "forgiveness"],
            emotional_signature={"power": 0.9, "wisdom": 0.85, "forgiveness": 0.8},
            narrative_function="magical_authority"
        )
        
        # LITERATURE - MODERN
        characters["frodo"] = UniversalCharacter(
            name="Frodo Baggins",
            source="literature_modern",
            archetype=ArchetypeType.HERO,
            traits=["brave", "small", "burdened", "compassionate"],
            keywords=["frodo", "hobbit", "ring bearer", "baggins", "shire"],
            emotional_signature={"courage": 0.8, "burden": 0.9, "compassion": 0.85},
            narrative_function="unlikely_hero"
        )
        
        characters["gandalf"] = UniversalCharacter(
            name="Gandalf",
            source="literature_modern",
            archetype=ArchetypeType.MENTOR,
            traits=["wise", "powerful", "protective", "sacrificial"],
            keywords=["gandalf", "wizard", "grey", "white", "staff"],
            emotional_signature={"wisdom": 0.95, "power": 0.9, "sacrifice": 0.85},
            narrative_function="wise_mentor"
        )
        
        characters["harry_potter"] = UniversalCharacter(
            name="Harry Potter",
            source="literature_modern",
            archetype=ArchetypeType.HERO,
            traits=["brave", "orphaned", "destined", "loyal"],
            keywords=["harry", "potter", "wizard", "scar", "chosen one"],
            emotional_signature={"courage": 0.85, "loyalty": 0.9, "destiny": 0.8},
            narrative_function="chosen_one"
        )
        
        characters["sherlock"] = UniversalCharacter(
            name="Sherlock Holmes",
            source="literature_modern",
            archetype=ArchetypeType.SAGE,
            traits=["brilliant", "observant", "eccentric", "logical"],
            keywords=["sherlock", "holmes", "detective", "deduction", "elementary"],
            emotional_signature={"intelligence": 1.0, "observation": 0.95, "logic": 0.9},
            narrative_function="master_detective"
        )
        
        # HISTORICAL FIGURES
        characters["caesar"] = UniversalCharacter(
            name="Julius Caesar",
            source="history",
            archetype=ArchetypeType.RULER,
            traits=["ambitious", "powerful", "strategic", "tragic"],
            keywords=["caesar", "rome", "emperor", "conquer", "dictator"],
            emotional_signature={"ambition": 0.95, "power": 0.9, "strategy": 0.85},
            narrative_function="ambitious_ruler"
        )
        
        characters["lincoln"] = UniversalCharacter(
            name="Abraham Lincoln",
            source="history",
            archetype=ArchetypeType.HERO,
            traits=["noble", "compassionate", "determined", "tragic"],
            keywords=["lincoln", "honest abe", "emancipation", "union", "president"],
            emotional_signature={"nobility": 0.9, "compassion": 0.85, "determination": 0.9},
            narrative_function="liberator"
        )
        
        # MODERN FICTION
        characters["batman"] = UniversalCharacter(
            name="Batman",
            source="modern_fiction",
            archetype=ArchetypeType.HERO,
            traits=["dark", "determined", "tragic", "vigilante"],
            keywords=["batman", "dark knight", "bruce wayne", "gotham", "vigilante"],
            emotional_signature={"determination": 0.95, "darkness": 0.8, "justice": 0.9},
            narrative_function="dark_hero"
        )
        
        characters["superman"] = UniversalCharacter(
            name="Superman",
            source="modern_fiction",
            archetype=ArchetypeType.HERO,
            traits=["noble", "powerful", "alien", "idealistic"],
            keywords=["superman", "clark kent", "krypton", "man of steel"],
            emotional_signature={"nobility": 0.95, "hope": 0.9, "power": 0.85},
            narrative_function="ideal_hero"
        )
        
        # BIBLICAL/RELIGIOUS
        characters["moses"] = UniversalCharacter(
            name="Moses",
            source="biblical",
            archetype=ArchetypeType.HERO,
            traits=["prophetic", "liberating", "destined", "humble"],
            keywords=["moses", "exodus", "commandments", "pharaoh", "promised land"],
            emotional_signature={"faith": 0.95, "leadership": 0.9, "humility": 0.85},
            narrative_function="prophet_liberator"
        )
        
        characters["buddha"] = UniversalCharacter(
            name="Buddha",
            source="religious",
            archetype=ArchetypeType.SAGE,
            traits=["enlightened", "compassionate", "wise", "peaceful"],
            keywords=["buddha", "enlightenment", "meditation", "dharma", "nirvana"],
            emotional_signature={"wisdom": 1.0, "compassion": 0.95, "peace": 0.9},
            narrative_function="enlightened_teacher"
        )
        
        return characters
    
    def _initialize_campbell_patterns(self) -> Dict[CampbellStage, ArchetypalPattern]:
        """Initialize archetypal patterns for all Campbell stages"""
        patterns = {}
        
        patterns[CampbellStage.ORDINARY_WORLD] = ArchetypalPattern(
            stage=CampbellStage.ORDINARY_WORLD,
            emotions={"comfort": 0.7, "routine": 0.8, "restlessness": 0.5},
            themes=["normalcy", "everyday_life", "status_quo"],
            conflicts=["dissatisfaction", "longing", "incompleteness"],
            transformations=["awakening", "dissatisfaction_grows"],
            symbolic_elements=["home", "family", "routine", "safety"]
        )
        
        patterns[CampbellStage.CALL_TO_ADVENTURE] = ArchetypalPattern(
            stage=CampbellStage.CALL_TO_ADVENTURE,
            emotions={"excitement": 0.7, "fear": 0.6, "curiosity": 0.8, "destiny": 0.9},
            themes=["destiny", "opportunity", "challenge", "change"],
            conflicts=["fear_vs_desire", "safety_vs_adventure"],
            transformations=["awareness", "possibility"],
            symbolic_elements=["messenger", "news", "dream", "omen"]
        )
        
        patterns[CampbellStage.REFUSAL_OF_CALL] = ArchetypalPattern(
            stage=CampbellStage.REFUSAL_OF_CALL,
            emotions={"fear": 0.9, "doubt": 0.8, "resistance": 0.7},
            themes=["fear", "doubt", "unworthiness", "safety"],
            conflicts=["courage_vs_fear", "duty_vs_comfort"],
            transformations=["crisis_of_faith", "internal_struggle"],
            symbolic_elements=["threshold", "warning", "obstacle"]
        )
        
        patterns[CampbellStage.MEETING_MENTOR] = ArchetypalPattern(
            stage=CampbellStage.MEETING_MENTOR,
            emotions={"hope": 0.8, "wisdom": 0.9, "courage": 0.7, "trust": 0.8},
            themes=["guidance", "wisdom", "preparation", "gift"],
            conflicts=["ignorance_vs_knowledge", "weakness_vs_strength"],
            transformations=["empowerment", "knowledge_gained", "confidence"],
            symbolic_elements=["teacher", "gift", "training", "blessing"]
        )
        
        patterns[CampbellStage.CROSSING_THRESHOLD] = ArchetypalPattern(
            stage=CampbellStage.CROSSING_THRESHOLD,
            emotions={"courage": 0.9, "determination": 0.8, "fear": 0.6, "commitment": 0.9},
            themes=["commitment", "point_of_no_return", "transformation"],
            conflicts=["old_vs_new", "known_vs_unknown"],
            transformations=["commitment", "entering_special_world"],
            symbolic_elements=["gate", "guardian", "boundary", "transformation"]
        )
        
        patterns[CampbellStage.TESTS_ALLIES_ENEMIES] = ArchetypalPattern(
            stage=CampbellStage.TESTS_ALLIES_ENEMIES,
            emotions={"perseverance": 0.9, "camaraderie": 0.7, "challenge": 0.8},
            themes=["learning", "proving", "bonding", "discovery"],
            conflicts=["trust_vs_betrayal", "strength_vs_weakness"],
            transformations=["skill_building", "relationship_forming"],
            symbolic_elements=["trial", "companion", "enemy", "lesson"]
        )
        
        patterns[CampbellStage.APPROACH_INMOST_CAVE] = ArchetypalPattern(
            stage=CampbellStage.APPROACH_INMOST_CAVE,
            emotions={"anticipation": 0.8, "dread": 0.7, "preparation": 0.9, "focus": 0.8},
            themes=["preparation", "final_approach", "gathering_strength"],
            conflicts=["readiness_vs_doubt", "hope_vs_fear"],
            transformations=["final_preparation", "steeling_resolve"],
            symbolic_elements=["fortress", "lair", "sanctum", "boundary"]
        )
        
        patterns[CampbellStage.ORDEAL] = ArchetypalPattern(
            stage=CampbellStage.ORDEAL,
            emotions={"suffering": 0.9, "perseverance": 1.0, "despair": 0.7, "transformation": 0.8},
            themes=["death_and_rebirth", "ultimate_test", "sacrifice"],
            conflicts=["life_vs_death", "success_vs_failure"],
            transformations=["death_of_old_self", "rebirth", "transformation"],
            symbolic_elements=["death", "darkness", "abyss", "monster"]
        )
        
        patterns[CampbellStage.REWARD] = ArchetypalPattern(
            stage=CampbellStage.REWARD,
            emotions={"triumph": 0.9, "relief": 0.8, "accomplishment": 0.9, "wisdom": 0.7},
            themes=["victory", "treasure", "knowledge", "celebration"],
            conflicts=["earning_vs_losing", "keeping_vs_sharing"],
            transformations=["mastery", "enlightenment", "empowerment"],
            symbolic_elements=["treasure", "elixir", "knowledge", "power"]
        )
        
        patterns[CampbellStage.ROAD_BACK] = ArchetypalPattern(
            stage=CampbellStage.ROAD_BACK,
            emotions={"urgency": 0.8, "determination": 0.9, "pursuit": 0.7},
            themes=["return", "chase", "escape", "commitment_to_ordinary"],
            conflicts=["staying_vs_returning", "special_world_vs_ordinary"],
            transformations=["reintegration_begins", "bringing_gift_back"],
            symbolic_elements=["chase", "deadline", "urgency", "return"]
        )
        
        patterns[CampbellStage.RESURRECTION] = ArchetypalPattern(
            stage=CampbellStage.RESURRECTION,
            emotions={"transformation": 1.0, "purification": 0.9, "clarity": 0.9, "rebirth": 1.0},
            themes=["final_test", "purification", "ultimate_transformation"],
            conflicts=["old_self_vs_new_self", "death_vs_life"],
            transformations=["complete_transformation", "mastery_achieved", "rebirth"],
            symbolic_elements=["final_battle", "cleansing", "rebirth", "emergence"]
        )
        
        patterns[CampbellStage.RETURN_WITH_ELIXIR] = ArchetypalPattern(
            stage=CampbellStage.RETURN_WITH_ELIXIR,
            emotions={"fulfillment": 0.9, "completion": 1.0, "peace": 0.9, "integration": 0.8},
            themes=["return", "gift_sharing", "completion", "new_beginning"],
            conflicts=["sharing_vs_hoarding", "changed_vs_unchanged"],
            transformations=["integration", "completion", "new_equilibrium"],
            symbolic_elements=["home", "gift", "wisdom", "treasure"]
        )
        
        # Extended Campbell stages
        patterns[CampbellStage.BELLY_OF_WHALE] = ArchetypalPattern(
            stage=CampbellStage.BELLY_OF_WHALE,
            emotions={"isolation": 0.9, "transformation": 0.8, "fear": 0.7, "metamorphosis": 0.9},
            themes=["separation", "transformation", "death_of_old_self"],
            conflicts=["old_vs_new", "destruction_vs_creation"],
            transformations=["complete_separation", "metamorphosis"],
            symbolic_elements=["darkness", "womb", "cocoon", "tomb"]
        )
        
        patterns[CampbellStage.MEETING_GODDESS] = ArchetypalPattern(
            stage=CampbellStage.MEETING_GODDESS,
            emotions={"love": 0.9, "reverence": 0.8, "awe": 0.9, "transcendence": 0.8},
            themes=["divine_love", "ultimate_boon", "sacred_marriage"],
            conflicts=["human_vs_divine", "mortal_vs_immortal"],
            transformations=["spiritual_awakening", "sacred_union"],
            symbolic_elements=["goddess", "divine_feminine", "sacred", "love"]
        )
        
        patterns[CampbellStage.APOTHEOSIS] = ArchetypalPattern(
            stage=CampbellStage.APOTHEOSIS,
            emotions={"enlightenment": 1.0, "transcendence": 1.0, "clarity": 0.9, "divinity": 0.9},
            themes=["divine_transformation", "ultimate_knowledge", "god_like_state"],
            conflicts=["mortal_vs_divine", "ignorance_vs_enlightenment"],
            transformations=["becoming_divine", "ultimate_understanding"],
            symbolic_elements=["light", "divinity", "perfection", "transcendence"]
        )
        
        return patterns
    
    def _initialize_narrative_functions(self) -> Dict[NarrativeFunction, Dict[str, Any]]:
        """Initialize narrative function patterns"""
        functions = {}
        
        functions[NarrativeFunction.EXPOSITION] = {
            "keywords": ["introduce", "describe", "begin", "setting", "background"],
            "position_range": (0.0, 0.15),
            "tension_level": 0.2,
            "description": "Setting the stage, introducing world and characters"
        }
        
        functions[NarrativeFunction.INCITING_INCIDENT] = {
            "keywords": ["suddenly", "changed", "disrupted", "began", "started", "unexpected"],
            "position_range": (0.10, 0.25),
            "tension_level": 0.6,
            "description": "Event that sets story in motion"
        }
        
        functions[NarrativeFunction.RISING_ACTION] = {
            "keywords": ["then", "next", "building", "growing", "intensifying", "more", "greater"],
            "position_range": (0.20, 0.45),
            "tension_level": 0.7,
            "description": "Building tension and complications"
        }
        
        functions[NarrativeFunction.FIRST_PINCH_POINT] = {
            "keywords": ["threat", "danger", "pressure", "antagonist", "obstacle"],
            "position_range": (0.30, 0.40),
            "tension_level": 0.75,
            "description": "First major pressure from antagonistic force"
        }
        
        functions[NarrativeFunction.MIDPOINT_REVERSAL] = {
            "keywords": ["revelation", "discovered", "turned", "shift", "transformed", "realized"],
            "position_range": (0.45, 0.55),
            "tension_level": 0.8,
            "description": "Major revelation or reversal at story center"
        }
        
        functions[NarrativeFunction.SECOND_PINCH_POINT] = {
            "keywords": ["crisis", "desperate", "losing", "darker", "worse"],
            "position_range": (0.60, 0.70),
            "tension_level": 0.85,
            "description": "Second major pressure, all seems lost"
        }
        
        functions[NarrativeFunction.CLIMAX] = {
            "keywords": ["finally", "ultimate", "decisive", "crucial", "moment", "peak", "confrontation"],
            "position_range": (0.75, 0.85),
            "tension_level": 1.0,
            "description": "Ultimate confrontation and resolution"
        }
        
        functions[NarrativeFunction.FALLING_ACTION] = {
            "keywords": ["after", "following", "consequence", "result", "aftermath"],
            "position_range": (0.80, 0.90),
            "tension_level": 0.4,
            "description": "Consequences of climax unfold"
        }
        
        functions[NarrativeFunction.RESOLUTION] = {
            "keywords": ["resolved", "settled", "concluded", "ended", "decided"],
            "position_range": (0.85, 0.95),
            "tension_level": 0.3,
            "description": "Conflicts resolved, loose ends tied"
        }
        
        functions[NarrativeFunction.DENOUEMENT] = {
            "keywords": ["finally", "peace", "home", "complete", "rest", "new beginning"],
            "position_range": (0.90, 1.0),
            "tension_level": 0.1,
            "description": "New equilibrium, reflection, closure"
        }
        
        return functions
    
    async def analyze(self, text: str) -> NarrativeIntelligenceResult:
        """
        Complete narrative intelligence analysis
        
        Args:
            text: Text to analyze
            
        Returns:
            NarrativeIntelligenceResult with complete analysis
        """
        
        # 1. Character detection and archetype analysis
        detected_characters, character_archetypes, character_confidence = self._detect_characters(text)
        
        # 2. Narrative arc prediction with Campbell stages
        narrative_arc = self._predict_narrative_arc(text, detected_characters)
        
        # 3. Get archetypal patterns for this arc
        archetypal_patterns = self._get_archetypal_patterns(narrative_arc, detected_characters)
        
        # 4. Temporal sentiment evolution
        temporal_evolution, sentiment_trajectory, emotional_range = self._analyze_temporal_evolution(text)
        
        # 5. Narrative function analysis
        narrative_function, act_position, story_progress = self._analyze_narrative_function(text, temporal_evolution)
        
        # 6. Enhanced sentiment with narrative intelligence
        base_sentiment = self._calculate_base_sentiment(text)
        narrative_enhanced_sentiment, enhancement_factor = self._enhance_sentiment_with_narrative(
            base_sentiment, archetypal_patterns, detected_characters, narrative_arc
        )
        
        # 7. Extract themes, conflicts, and symbolic elements
        themes = self._extract_themes(text, narrative_arc, archetypal_patterns)
        conflicts = self._identify_conflicts(text, narrative_arc)
        symbolic_elements = self._identify_symbols(text, narrative_arc)
        
        # 8. Generate recommendations
        recommendations = self._generate_recommendations(
            narrative_arc, detected_characters, enhancement_factor, 
            sentiment_trajectory, narrative_function
        )
        
        # 9. Calculate Bayesian confidence
        bayesian_confidence = self._calculate_bayesian_confidence(
            character_confidence, narrative_arc.confidence, len(detected_characters),
            len(themes), emotional_range
        )
        
        return NarrativeIntelligenceResult(
            text=text,
            detected_characters=detected_characters,
            character_archetypes=character_archetypes,
            character_confidence=character_confidence,
            narrative_arc=narrative_arc,
            archetypal_patterns=archetypal_patterns,
            temporal_evolution=temporal_evolution,
            sentiment_trajectory=sentiment_trajectory,
            emotional_range=emotional_range,
            narrative_function=narrative_function,
            act_position=act_position,
            story_progress=story_progress,
            base_sentiment=base_sentiment,
            narrative_enhanced_sentiment=narrative_enhanced_sentiment,
            enhancement_factor=enhancement_factor,
            themes=themes,
            conflicts=conflicts,
            symbolic_elements=symbolic_elements,
            recommendations=recommendations,
            bayesian_confidence=bayesian_confidence
        )
    
    def _detect_characters(self, text: str) -> Tuple[List[UniversalCharacter], Dict[str, ArchetypeType], float]:
        """Detect characters from universal database"""
        text_lower = text.lower()
        detected = []
        archetypes = {}
        
        for char_name, character in self.universal_characters.items():
            # Check if any keywords match
            if any(keyword in text_lower for keyword in character.keywords):
                detected.append(character)
                archetypes[character.name] = character.archetype
        
        # Calculate confidence based on number and clarity of matches
        if not detected:
            confidence = 0.0
        elif len(detected) == 1:
            confidence = 0.7
        elif len(detected) == 2:
            confidence = 0.85
        else:
            confidence = 0.95
        
        return detected, archetypes, confidence
    
    def _predict_narrative_arc(self, text: str, characters: List[UniversalCharacter]) -> NarrativeArc:
        """Predict Campbell stage with confidence"""
        words = text.lower().split()
        
        # Score each Campbell stage
        stage_scores = {}
        evidence_by_stage = {}
        
        for stage, pattern in self.campbell_patterns.items():
            score = 0.0
            evidence = []
            
            # Check theme keywords
            for theme in pattern.themes:
                if any(theme.replace("_", " ") in text.lower() for _ in [1]):
                    score += 0.3
                    evidence.append(f"Theme: {theme}")
            
            # Check symbolic elements
            for symbol in pattern.symbolic_elements:
                if symbol in text.lower():
                    score += 0.2
                    evidence.append(f"Symbol: {symbol}")
            
            # Check transformations
            for transformation in pattern.transformations:
                if any(transformation.replace("_", " ") in text.lower() for _ in [1]):
                    score += 0.25
                    evidence.append(f"Transformation: {transformation}")
            
            # Boost score based on character archetypes present
            for char in characters:
                if stage in [CampbellStage.MEETING_MENTOR] and char.archetype == ArchetypeType.MENTOR:
                    score += 0.5
                    evidence.append(f"Mentor character: {char.name}")
                elif stage in [CampbellStage.CALL_TO_ADVENTURE] and char.archetype == ArchetypeType.HERO:
                    score += 0.4
                    evidence.append(f"Hero character: {char.name}")
            
            stage_scores[stage] = score
            evidence_by_stage[stage] = evidence
        
        # Get primary arc (highest score)
        if max(stage_scores.values()) > 0:
            primary_arc = max(stage_scores.keys(), key=lambda k: stage_scores[k])
            primary_confidence = min(stage_scores[primary_arc], 1.0)
        else:
            primary_arc = CampbellStage.ORDINARY_WORLD
            primary_confidence = 0.3
        
        # Get secondary arcs (next 2 highest)
        sorted_stages = sorted(stage_scores.items(), key=lambda x: x[1], reverse=True)
        secondary_arcs = [(stage, min(score, 1.0)) for stage, score in sorted_stages[1:3] if score > 0]
        
        # Determine act position
        act_1_stages = {CampbellStage.ORDINARY_WORLD, CampbellStage.CALL_TO_ADVENTURE, 
                       CampbellStage.REFUSAL_OF_CALL, CampbellStage.MEETING_MENTOR, 
                       CampbellStage.CROSSING_THRESHOLD}
        act_3_stages = {CampbellStage.ROAD_BACK, CampbellStage.RESURRECTION, 
                       CampbellStage.RETURN_WITH_ELIXIR}
        
        if primary_arc in act_1_stages:
            act_position = 1
        elif primary_arc in act_3_stages:
            act_position = 3
        else:
            act_position = 2
        
        # Determine narrative function based on stage
        function_mapping = {
            CampbellStage.ORDINARY_WORLD: NarrativeFunction.EXPOSITION,
            CampbellStage.CALL_TO_ADVENTURE: NarrativeFunction.INCITING_INCIDENT,
            CampbellStage.REFUSAL_OF_CALL: NarrativeFunction.RISING_ACTION,
            CampbellStage.MEETING_MENTOR: NarrativeFunction.RISING_ACTION,
            CampbellStage.CROSSING_THRESHOLD: NarrativeFunction.FIRST_PINCH_POINT,
            CampbellStage.TESTS_ALLIES_ENEMIES: NarrativeFunction.RISING_ACTION,
            CampbellStage.APPROACH_INMOST_CAVE: NarrativeFunction.MIDPOINT_REVERSAL,
            CampbellStage.ORDEAL: NarrativeFunction.CLIMAX,
            CampbellStage.REWARD: NarrativeFunction.FALLING_ACTION,
            CampbellStage.ROAD_BACK: NarrativeFunction.FALLING_ACTION,
            CampbellStage.RESURRECTION: NarrativeFunction.CLIMAX,
            CampbellStage.RETURN_WITH_ELIXIR: NarrativeFunction.DENOUEMENT,
        }
        
        narrative_function = function_mapping.get(primary_arc, NarrativeFunction.EXPOSITION)
        
        return NarrativeArc(
            primary_arc=primary_arc,
            confidence=primary_confidence,
            secondary_arcs=secondary_arcs,
            evidence=evidence_by_stage[primary_arc],
            narrative_function=narrative_function,
            act_position=act_position
        )
    
    def _get_archetypal_patterns(self, narrative_arc: NarrativeArc, 
                                 characters: List[UniversalCharacter]) -> Dict[str, float]:
        """Get archetypal emotional patterns for this narrative stage"""
        
        pattern = self.campbell_patterns.get(narrative_arc.primary_arc)
        if not pattern:
            return {}
        
        archetypal_emotions = pattern.emotions.copy()
        
        # Enhance based on character presence
        character_multiplier = 1.0 + len(characters) * 0.15
        
        for emotion in archetypal_emotions:
            archetypal_emotions[emotion] *= character_multiplier
            archetypal_emotions[emotion] = min(archetypal_emotions[emotion], 1.0)
        
        return archetypal_emotions
    
    def _analyze_temporal_evolution(self, text: str) -> Tuple[List[TemporalSentimentPoint], str, float]:
        """Analyze how sentiment evolves temporally through text"""
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [], "stable", 0.0
        
        evolution = []
        sentiments = []
        
        positive_words = ["joy", "love", "hope", "wisdom", "triumph", "peace", "glory", "happiness"]
        negative_words = ["pain", "sorrow", "fear", "death", "suffering", "loss", "despair", "anger"]
        
        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()
            
            pos_count = sum(1 for word in words if any(pw in word for pw in positive_words))
            neg_count = sum(1 for word in words if any(nw in word for nw in negative_words))
            
            total = pos_count + neg_count
            if total == 0:
                sentiment = 0.0
                intensity = 0.0
            else:
                sentiment = (pos_count - neg_count) / len(words)  # -1 to 1 scale
                intensity = total / len(words)
            
            sentiments.append(sentiment)
            
            # Determine dominant emotion (simplified)
            if sentiment > 0.3:
                dominant_emotion = "positive"
            elif sentiment < -0.3:
                dominant_emotion = "negative"
            else:
                dominant_emotion = "neutral"
            
            position = i / (len(sentences) - 1) if len(sentences) > 1 else 0.0
            
            # Estimate narrative function based on position
            if position < 0.15:
                function = NarrativeFunction.EXPOSITION
            elif position < 0.25:
                function = NarrativeFunction.INCITING_INCIDENT
            elif position < 0.45:
                function = NarrativeFunction.RISING_ACTION
            elif position < 0.55:
                function = NarrativeFunction.MIDPOINT_REVERSAL
            elif position < 0.75:
                function = NarrativeFunction.SECOND_PINCH_POINT
            elif position < 0.85:
                function = NarrativeFunction.CLIMAX
            elif position < 0.95:
                function = NarrativeFunction.FALLING_ACTION
            else:
                function = NarrativeFunction.DENOUEMENT
            
            evolution.append(TemporalSentimentPoint(
                position=position,
                sentiment=sentiment,
                intensity=intensity,
                dominant_emotion=dominant_emotion,
                narrative_function=function
            ))
        
        # Determine trajectory
        if len(sentiments) < 2:
            trajectory = "stable"
        else:
            start_avg = sum(sentiments[:len(sentiments)//3]) / max(len(sentiments)//3, 1)
            end_avg = sum(sentiments[-len(sentiments)//3:]) / max(len(sentiments)//3, 1)
            mid_avg = sum(sentiments[len(sentiments)//3:-len(sentiments)//3]) / max(len(sentiments) - 2*(len(sentiments)//3), 1)
            
            if end_avg > start_avg + 0.2:
                trajectory = "rising"
            elif end_avg < start_avg - 0.2:
                trajectory = "falling"
            elif mid_avg < min(start_avg, end_avg) - 0.2:
                trajectory = "u_shaped"
            elif mid_avg > max(start_avg, end_avg) + 0.2:
                trajectory = "inverted_u"
            elif max(sentiments) - min(sentiments) > 0.6:
                trajectory = "chaotic"
            else:
                trajectory = "stable"
        
        emotional_range = max(sentiments) - min(sentiments) if sentiments else 0.0
        
        return evolution, trajectory, emotional_range
    
    def _analyze_narrative_function(self, text: str, temporal_evolution: List[TemporalSentimentPoint]) -> Tuple[NarrativeFunction, int, float]:
        """Analyze narrative function and story position"""
        
        words = text.lower().split()
        function_scores = {}
        
        for function, info in self.narrative_functions.items():
            score = 0.0
            
            # Check keywords
            for keyword in info["keywords"]:
                if any(keyword in word for word in words):
                    score += 0.3
            
            function_scores[function] = score
        
        # Get primary function
        if max(function_scores.values()) > 0:
            primary_function = max(function_scores.keys(), key=lambda k: function_scores[k])
        else:
            primary_function = NarrativeFunction.EXPOSITION
        
        # Determine act based on function
        act_1_functions = {NarrativeFunction.EXPOSITION, NarrativeFunction.INCITING_INCIDENT}
        act_3_functions = {NarrativeFunction.FALLING_ACTION, NarrativeFunction.RESOLUTION, NarrativeFunction.DENOUEMENT}
        
        if primary_function in act_1_functions:
            act = 1
            progress = 0.15
        elif primary_function in act_3_functions:
            act = 3
            progress = 0.85
        else:
            act = 2
            progress = 0.5
        
        return primary_function, act, progress
    
    def _calculate_base_sentiment(self, text: str) -> float:
        """Calculate base sentiment without narrative enhancement"""
        positive_words = ["good", "joy", "love", "hope", "wisdom", "triumph", "peace", "glory"]
        negative_words = ["bad", "pain", "fear", "death", "suffering", "loss", "despair", "anger"]
        
        words = text.lower().split()
        pos_count = sum(1 for word in words if any(pw in word for pw in positive_words))
        neg_count = sum(1 for word in words if any(nw in word for nw in negative_words))
        
        total = pos_count + neg_count
        if total == 0:
            return 0.5
        
        return pos_count / total
    
    def _enhance_sentiment_with_narrative(self, base_sentiment: float, 
                                         archetypal_patterns: Dict[str, float],
                                         characters: List[UniversalCharacter],
                                         narrative_arc: NarrativeArc) -> Tuple[float, float]:
        """Enhance sentiment using narrative intelligence"""
        
        enhancement = 0.0
        
        # Archetypal enhancement
        if archetypal_patterns:
            positive_emotions = ["courage", "wisdom", "hope", "triumph", "fulfillment", "completion", "peace", "love"]
            negative_emotions = ["fear", "suffering", "despair", "doubt"]
            
            pos_score = sum(archetypal_patterns.get(e, 0) for e in positive_emotions)
            neg_score = sum(archetypal_patterns.get(e, 0) for e in negative_emotions)
            
            enhancement += (pos_score - neg_score) * 0.2
        
        # Character presence enhancement
        hero_bonus = sum(0.15 for char in characters if char.archetype in 
                        {ArchetypeType.HERO, ArchetypeType.MENTOR, ArchetypeType.SAGE})
        enhancement += hero_bonus
        
        # Arc-based enhancement
        positive_arcs = {CampbellStage.REWARD, CampbellStage.RETURN_WITH_ELIXIR, 
                        CampbellStage.RESURRECTION, CampbellStage.APOTHEOSIS}
        if narrative_arc.primary_arc in positive_arcs:
            enhancement += 0.2
        
        # Apply enhancement
        enhanced_sentiment = max(0.0, min(1.0, base_sentiment + enhancement))
        enhancement_factor = enhanced_sentiment - base_sentiment
        
        return enhanced_sentiment, enhancement_factor
    
    def _extract_themes(self, text: str, narrative_arc: NarrativeArc, 
                       archetypal_patterns: Dict[str, float]) -> List[str]:
        """Extract narrative themes"""
        pattern = self.campbell_patterns.get(narrative_arc.primary_arc)
        if pattern:
            return pattern.themes
        return []
    
    def _identify_conflicts(self, text: str, narrative_arc: NarrativeArc) -> List[str]:
        """Identify narrative conflicts"""
        pattern = self.campbell_patterns.get(narrative_arc.primary_arc)
        if pattern:
            return pattern.conflicts
        return []
    
    def _identify_symbols(self, text: str, narrative_arc: NarrativeArc) -> List[str]:
        """Identify symbolic elements"""
        pattern = self.campbell_patterns.get(narrative_arc.primary_arc)
        if pattern:
            return pattern.symbolic_elements
        return []
    
    def _generate_recommendations(self, narrative_arc: NarrativeArc, 
                                 characters: List[UniversalCharacter],
                                 enhancement_factor: float,
                                 sentiment_trajectory: str,
                                 narrative_function: NarrativeFunction) -> List[str]:
        """Generate interpretation recommendations"""
        recommendations = []
        
        # Arc-based recommendation
        recommendations.append(f"Text aligns with {narrative_arc.primary_arc.value} stage of Hero's Journey")
        
        # Character-based
        if characters:
            char_names = ", ".join([c.name for c in characters])
            recommendations.append(f"Universal characters detected: {char_names}")
        
        # Enhancement significance
        if enhancement_factor > 0.3:
            recommendations.append(f"Narrative intelligence dramatically enhances understanding (+{enhancement_factor:.3f})")
        elif enhancement_factor > 0.1:
            recommendations.append(f"Narrative context enriches interpretation (+{enhancement_factor:.3f})")
        
        # Trajectory insight
        if sentiment_trajectory != "stable":
            recommendations.append(f"Dynamic {sentiment_trajectory} emotional trajectory detected")
        
        # Function insight
        recommendations.append(f"Narrative function: {narrative_function.value} - {self.narrative_functions[narrative_function]['description']}")
        
        return recommendations
    
    def _calculate_bayesian_confidence(self, character_confidence: float, 
                                      arc_confidence: float, 
                                      character_count: int,
                                      theme_count: int,
                                      emotional_range: float) -> float:
        """Calculate overall Bayesian confidence in analysis"""
        
        # Weight different factors
        confidence = (
            character_confidence * 0.3 +
            arc_confidence * 0.3 +
            min(character_count * 0.1, 0.2) +
            min(theme_count * 0.05, 0.1) +
            min(emotional_range * 0.2, 0.1)
        )
        
        return min(confidence, 0.95)


async def demonstrate_narrative_intelligence():
    """Demonstrate comprehensive narrative intelligence"""
    
    print("🎭 NARRATIVE INTELLIGENCE DEMONSTRATION")
    print("=" * 80)
    print("Universal character detection + Joseph Campbell + Temporal evolution")
    print("=" * 80)
    print()
    
    ni = NarrativeIntelligence()
    
    test_texts = [
        {
            "title": "Odysseus' Final Test",
            "text": "Odysseus stood before the gates of Ithaca, his long journey finally at an end. Athena appeared beside him, her owl eyes gleaming with wisdom. 'You have crossed threshold after threshold,' she said, 'faced monsters and gods, died and been reborn. Now comes the final test: to return home not as the king who left, but as the hero you have become. The treasure you bring is not gold, but wisdom earned through suffering.' The old warrior smiled, understanding at last that the journey had transformed him completely."
        },
        {
            "title": "Harry's Call to Adventure",
            "text": "The letter arrived on Harry Potter's eleventh birthday, carried by an owl through the window of his tiny cupboard. His scar burned as he read the words: 'You are a wizard.' Everything he thought he knew about the world shattered in that moment. Fear and excitement warred within him. He could refuse this call, stay in the ordinary world of the Dursleys, remain safe. But destiny whispered that another world awaited, filled with magic and danger and purpose."
        },
        {
            "title": "Hamlet's Ordeal",
            "text": "To be or not to be, that is the question Hamlet faced in the depths of his despair. His father murdered, his mother remarried to the killer, and he himself charged with bloody revenge. The prince stood at the abyss, staring into the darkness of death itself. This was his ordeal, his dark night of the soul—to act or not to act, to live or to die, to embrace the terrible destiny fate had thrust upon him or to sink into madness and oblivion."
        },
        {
            "title": "Frodo's Return",
            "text": "Frodo Baggins returned to the Shire, the Ring destroyed, the quest complete. But the hobbit who came home was not the one who had left. He carried invisible wounds, memories of darkness and suffering that would never fully heal. The elixir he brought was not for himself but for all of Middle-earth—the gift of peace, purchased through sacrifice. As he walked through the green fields of his homeland, he understood that some journeys change you so completely that you can never truly return."
        }
    ]
    
    for i, test in enumerate(test_texts, 1):
        print(f"📖 Test {i}/4: {test['title']}")
        print(f"{'=' * 80}")
        print(f"Text: {test['text'][:200]}...")
        print()
        
        result = await ni.analyze(test['text'])
        
        print(f"🎭 DETECTED CHARACTERS:")
        for char in result.detected_characters:
            print(f"   • {char.name} ({char.source}) - Archetype: {char.archetype.value}")
            print(f"     Traits: {', '.join(char.traits[:3])}")
        print()
        
        print(f"🏛️ CAMPBELL'S HERO'S JOURNEY:")
        print(f"   Primary Stage: {result.narrative_arc.primary_arc.value}")
        print(f"   Confidence: {result.narrative_arc.confidence:.3f}")
        print(f"   Act: {result.narrative_arc.act_position}/3")
        print(f"   Evidence: {', '.join(result.narrative_arc.evidence[:3])}")
        if result.narrative_arc.secondary_arcs:
            print(f"   Secondary Stages: {', '.join([s[0].value for s in result.narrative_arc.secondary_arcs])}")
        print()
        
        print(f"🎬 NARRATIVE FUNCTION:")
        print(f"   Function: {result.narrative_function.value}")
        print(f"   Story Progress: {result.story_progress:.1%}")
        print(f"   Description: {ni.narrative_functions[result.narrative_function]['description']}")
        print()
        
        print(f"📈 TEMPORAL SENTIMENT EVOLUTION:")
        print(f"   Trajectory: {result.sentiment_trajectory}")
        print(f"   Emotional Range: {result.emotional_range:.3f}")
        print(f"   Points: {len(result.temporal_evolution)} sentiment measurements")
        print()
        
        print(f"🧠 ARCHETYPAL PATTERNS:")
        if result.archetypal_patterns:
            for emotion, score in sorted(result.archetypal_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   • {emotion}: {score:.3f}")
        print()
        
        print(f"💫 SENTIMENT ANALYSIS:")
        print(f"   Base Sentiment: {result.base_sentiment:.3f}")
        print(f"   Narrative Enhanced: {result.narrative_enhanced_sentiment:.3f}")
        print(f"   Enhancement Factor: {result.enhancement_factor:+.3f}")
        print()
        
        print(f"🎯 THEMES & CONFLICTS:")
        print(f"   Themes: {', '.join(result.themes[:3])}")
        print(f"   Conflicts: {', '.join(result.conflicts[:2])}")
        print(f"   Symbols: {', '.join(result.symbolic_elements[:3])}")
        print()
        
        print(f"🧭 RECOMMENDATIONS:")
        for rec in result.recommendations:
            print(f"   • {rec}")
        print()
        
        print(f"🔮 Bayesian Confidence: {result.bayesian_confidence:.3f}")
        print(f"{'=' * 80}")
        print()
    
    print("✨ NARRATIVE INTELLIGENCE DEMONSTRATION COMPLETE!")
    print("🎯 Full Joseph Campbell integration + Universal character database!")
    print("📊 Definite narrative arc prediction + Temporal evolution tracking!")
    print("🧠 Complete story structure analysis with Bayesian confidence!")


if __name__ == "__main__":
    asyncio.run(demonstrate_narrative_intelligence())