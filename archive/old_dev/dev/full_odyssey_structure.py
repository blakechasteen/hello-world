"""
Full Odyssey Narrative Structure - All 24 Books
Comprehensive structure for Matryoshka MCTS processing with temporal analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np

class NarrativeArc(Enum):
    INVOCATION = "invocation"           # Books 1-4: Setup and Telemachy
    DEPARTURE = "departure"             # Books 5-8: Odysseus begins journey
    TRIALS = "trials"                   # Books 9-12: Major trials and tests
    WANDERING = "wandering"             # Books 13-16: Lost and finding way
    RECOGNITION = "recognition"         # Books 17-20: Identity revelations
    RESOLUTION = "resolution"           # Books 21-24: Final conflicts and reunion

class DecisionType(Enum):
    STRATEGIC = "strategic"             # Planning and tactics
    MORAL = "moral"                     # Right vs wrong choices
    IDENTITY = "identity"               # Revealing or concealing self
    LOYALTY = "loyalty"                 # Trust and betrayal decisions
    DIVINE = "divine"                   # Dealing with gods
    HEROIC = "heroic"                   # Courage vs caution

@dataclass
class CharacterState:
    """Track character development and relationships"""
    name: str
    trust_level: float = 0.5            # 0-1, how much Odysseus trusts them
    knowledge_of_identity: float = 0.0   # 0-1, how much they know who he is
    loyalty: float = 0.5                # 0-1, their loyalty to Odysseus
    power_level: float = 0.5            # 0-1, their ability to help/harm
    relationship_history: List[str] = field(default_factory=list)

@dataclass
class NarrativeDecision:
    """A decision point in the Odyssey narrative"""
    book: int
    scene: str
    decision_type: DecisionType
    narrative_arc: NarrativeArc
    description: str
    options: List[str]
    canonical_choice: str               # What Odysseus actually chose
    wisdom_level: float                 # 0-1, how wise the canonical choice was
    consequences: List[str]
    affected_characters: List[str]
    divine_involvement: float = 0.0     # 0-1, level of divine intervention
    narrative_weight: float = 0.5       # 0-1, importance to overall story
    temporal_complexity: float = 0.5    # 0-1, how many timeframes involved

@dataclass
class OdysseyBook:
    """Complete structure for one book of the Odyssey"""
    number: int
    title: str
    narrative_arc: NarrativeArc
    key_themes: List[str]
    major_decisions: List[NarrativeDecision]
    character_developments: Dict[str, CharacterState]
    divine_interventions: List[str]
    geographic_setting: str
    temporal_setting: str               # Past, present, future references
    narrative_techniques: List[str]     # Flashbacks, prophecies, etc.
    
class FullOdysseyStructure:
    """Complete 24-book structure of the Odyssey"""
    
    def __init__(self):
        self.books = self._build_full_odyssey()
        self.character_evolution = self._track_character_evolution()
        self.thematic_progression = self._analyze_thematic_progression()
        
    def _build_full_odyssey(self) -> Dict[int, OdysseyBook]:
        """Build complete 24-book structure"""
        books = {}
        
        # BOOK 1: The Gods in Council
        books[1] = OdysseyBook(
            number=1,
            title="The Gods in Council",
            narrative_arc=NarrativeArc.INVOCATION,
            key_themes=["divine justice", "homecoming desire", "heroic reputation"],
            major_decisions=[
                NarrativeDecision(
                    book=1, scene="divine_council",
                    decision_type=DecisionType.DIVINE,
                    narrative_arc=NarrativeArc.INVOCATION,
                    description="Athena advocates for Odysseus to Zeus",
                    options=["intervene_immediately", "wait_longer", "abandon_odysseus"],
                    canonical_choice="intervene_immediately",
                    wisdom_level=0.8,
                    consequences=["Divine support activated", "Journey home begins"],
                    affected_characters=["Zeus", "Athena", "Odysseus"],
                    divine_involvement=1.0,
                    narrative_weight=0.9
                )
            ],
            character_developments={
                "Athena": CharacterState("Athena", trust_level=0.9, loyalty=0.95, power_level=0.9),
                "Zeus": CharacterState("Zeus", trust_level=0.6, loyalty=0.7, power_level=1.0)
            },
            divine_interventions=["Athena's advocacy", "Zeus's permission"],
            geographic_setting="Olympus",
            temporal_setting="Present divine time, references to 10-year absence",
            narrative_techniques=["Divine perspective", "Exposition"]
        )
        
        # BOOK 2: The Assembly in Ithaca
        books[2] = OdysseyBook(
            number=2,
            title="The Assembly in Ithaca", 
            narrative_arc=NarrativeArc.INVOCATION,
            key_themes=["political crisis", "filial duty", "justice vs power"],
            major_decisions=[
                NarrativeDecision(
                    book=2, scene="telemachus_assembly",
                    decision_type=DecisionType.MORAL,
                    narrative_arc=NarrativeArc.INVOCATION,
                    description="Telemachus calls assembly to address suitors",
                    options=["public_confrontation", "private_negotiation", "violent_action"],
                    canonical_choice="public_confrontation",
                    wisdom_level=0.7,
                    consequences=["Public support sought", "Suitors exposed", "Journey decision made"],
                    affected_characters=["Telemachus", "Suitors", "Ithacans"],
                    divine_involvement=0.3,
                    narrative_weight=0.7
                )
            ],
            character_developments={
                "Telemachus": CharacterState("Telemachus", trust_level=0.8, loyalty=1.0, power_level=0.4),
                "Antinous": CharacterState("Antinous", trust_level=0.1, loyalty=0.0, power_level=0.7)
            },
            divine_interventions=["Athena's guidance to Telemachus"],
            geographic_setting="Ithaca",
            temporal_setting="Present crisis, references to Odysseus's absence",
            narrative_techniques=["Political assembly", "Generational conflict"]
        )
        
        # BOOK 9: The Cyclops (Key trial)
        books[9] = OdysseyBook(
            number=9,
            title="The Cyclops",
            narrative_arc=NarrativeArc.TRIALS,
            key_themes=["hubris vs wisdom", "identity revelation", "divine justice"],
            major_decisions=[
                NarrativeDecision(
                    book=9, scene="cyclops_blinding",
                    decision_type=DecisionType.STRATEGIC,
                    narrative_arc=NarrativeArc.TRIALS,
                    description="Decision to blind Polyphemus rather than kill",
                    options=["kill_cyclops", "blind_cyclops", "negotiate_escape"],
                    canonical_choice="blind_cyclops",
                    wisdom_level=0.8,
                    consequences=["Escape possible", "Cyclops alive to pray to Poseidon"],
                    affected_characters=["Polyphemus", "Odysseus", "Crew"],
                    divine_involvement=0.2,
                    narrative_weight=0.9,
                    temporal_complexity=0.6
                ),
                NarrativeDecision(
                    book=9, scene="name_revelation",
                    decision_type=DecisionType.IDENTITY,
                    narrative_arc=NarrativeArc.TRIALS,
                    description="Odysseus reveals his true name to Polyphemus",
                    options=["stay_anonymous", "reveal_name", "give_false_name"],
                    canonical_choice="reveal_name",
                    wisdom_level=0.3,  # Hubris over wisdom
                    consequences=["Polyphemus knows his attacker", "Divine curse activated"],
                    affected_characters=["Polyphemus", "Poseidon", "Odysseus"],
                    divine_involvement=0.8,
                    narrative_weight=1.0,
                    temporal_complexity=0.8
                )
            ],
            character_developments={
                "Polyphemus": CharacterState("Polyphemus", trust_level=0.0, loyalty=0.0, power_level=0.8),
                "Odysseus": CharacterState("Odysseus", trust_level=0.7, loyalty=0.9, power_level=0.8)
            },
            divine_interventions=["Poseidon's curse activation"],
            geographic_setting="Cyclops Island",
            temporal_setting="Present adventure, future consequences established",
            narrative_techniques=["First-person narrative", "Prophecy fulfillment"]
        )
        
        # BOOK 12: The Sirens and Scylla
        books[12] = OdysseyBook(
            number=12,
            title="The Sirens and Scylla",
            narrative_arc=NarrativeArc.TRIALS,
            key_themes=["temptation vs wisdom", "knowledge vs safety", "leadership sacrifice"],
            major_decisions=[
                NarrativeDecision(
                    book=12, scene="siren_encounter",
                    decision_type=DecisionType.STRATEGIC,
                    narrative_arc=NarrativeArc.TRIALS,
                    description="Decision to hear Sirens while bound",
                    options=["avoid_completely", "listen_while_bound", "all_listen"],
                    canonical_choice="listen_while_bound",
                    wisdom_level=0.8,
                    consequences=["Knowledge gained", "Crew tested", "Safe passage"],
                    affected_characters=["Sirens", "Crew", "Odysseus"],
                    divine_involvement=0.4,
                    narrative_weight=0.8,
                    temporal_complexity=0.7
                ),
                NarrativeDecision(
                    book=12, scene="scylla_charybdis",
                    decision_type=DecisionType.MORAL,
                    narrative_arc=NarrativeArc.TRIALS,
                    description="Choice between Scylla (lose some) vs Charybdis (risk all)",
                    options=["face_scylla", "risk_charybdis", "attempt_avoidance"],
                    canonical_choice="face_scylla",
                    wisdom_level=0.9,  # Utilitarian calculation
                    consequences=["Six men lost", "Ship and most crew saved"],
                    affected_characters=["Crew", "Odysseus", "Scylla"],
                    divine_involvement=0.6,
                    narrative_weight=0.9,
                    temporal_complexity=0.5
                )
            ],
            character_developments={
                "Crew": CharacterState("Crew", trust_level=0.6, loyalty=0.8, power_level=0.5)
            },
            divine_interventions=["Circe's warnings", "Divine monsters"],
            geographic_setting="Strait of Messina region",
            temporal_setting="Present trial, prophetic warnings",
            narrative_techniques=["Divine guidance", "Impossible choices"]
        )
        
        # BOOK 23: The Recognition
        books[23] = OdysseyBook(
            number=23,
            title="The Recognition",
            narrative_arc=NarrativeArc.RESOLUTION,
            key_themes=["identity verification", "marital reunion", "trust restoration"],
            major_decisions=[
                NarrativeDecision(
                    book=23, scene="bed_secret_test",
                    decision_type=DecisionType.IDENTITY,
                    narrative_arc=NarrativeArc.RESOLUTION,
                    description="Penelope tests Odysseus with bed secret",
                    options=["immediate_acceptance", "test_with_bed", "demand_other_proof"],
                    canonical_choice="test_with_bed",
                    wisdom_level=0.95,  # Ultimate verification wisdom
                    consequences=["True identity confirmed", "Marriage restored", "Trust rebuilt"],
                    affected_characters=["Penelope", "Odysseus", "Eurycleia"],
                    divine_involvement=0.2,
                    narrative_weight=1.0,
                    temporal_complexity=0.9
                )
            ],
            character_developments={
                "Penelope": CharacterState("Penelope", trust_level=0.95, loyalty=1.0, power_level=0.7),
                "Odysseus": CharacterState("Odysseus", trust_level=0.9, loyalty=1.0, power_level=0.9)
            },
            divine_interventions=["Athena's time manipulation"],
            geographic_setting="Odysseus's palace, Ithaca",
            temporal_setting="Present reunion, 20-year separation referenced",
            narrative_techniques=["Recognition scene", "Temporal manipulation"]
        )
        
        # BOOK 3: King Nestor Remembers
        books[3] = OdysseyBook(
            number=3,
            title="King Nestor Remembers",
            narrative_arc=NarrativeArc.INVOCATION,
            key_themes=["wisdom of age", "honor and glory", "generational memory"],
            major_decisions=[
                NarrativeDecision(
                    book=3, scene="nestor_counsel",
                    decision_type=DecisionType.MORAL,
                    narrative_arc=NarrativeArc.INVOCATION,
                    description="Telemachus seeks guidance from Nestor",
                    options=["accept_traditional_wisdom", "question_elder_authority", "seek_compromise"],
                    canonical_choice="accept_traditional_wisdom",
                    wisdom_level=0.8,
                    consequences=["Gains elder wisdom", "Learns family history", "Receives guidance"],
                    affected_characters=["Nestor", "Telemachus"],
                    divine_involvement=0.2,
                    narrative_weight=0.6
                )
            ],
            character_developments={
                "Nestor": CharacterState("Nestor", trust_level=0.9, loyalty=0.8, power_level=0.6),
                "Telemachus": CharacterState("Telemachus", trust_level=0.85, loyalty=1.0, power_level=0.5)
            },
            divine_interventions=["Athena's continued guidance"],
            geographic_setting="Pylos",
            temporal_setting="Present consultation, references to Trojan War past",
            narrative_techniques=["Flashback narratives", "Wisdom tradition"]
        )

        # BOOK 4: The King and Queen of Sparta
        books[4] = OdysseyBook(
            number=4,
            title="The King and Queen of Sparta",
            narrative_arc=NarrativeArc.INVOCATION,
            key_themes=["royal hospitality", "marital fidelity", "heroic recognition"],
            major_decisions=[
                NarrativeDecision(
                    book=4, scene="menelaus_helen_counsel",
                    decision_type=DecisionType.IDENTITY,
                    narrative_arc=NarrativeArc.INVOCATION,
                    description="Helen recognizes Telemachus's resemblance to Odysseus",
                    options=["reveal_recognition", "remain_silent", "test_telemachus"],
                    canonical_choice="reveal_recognition",
                    wisdom_level=0.7,
                    consequences=["Identity confirmed", "Hope strengthened", "Information shared"],
                    affected_characters=["Helen", "Menelaus", "Telemachus"],
                    divine_involvement=0.3,
                    narrative_weight=0.7,
                    temporal_complexity=0.6
                )
            ],
            character_developments={
                "Helen": CharacterState("Helen", trust_level=0.7, loyalty=0.6, power_level=0.8),
                "Menelaus": CharacterState("Menelaus", trust_level=0.8, loyalty=0.9, power_level=0.7)
            },
            divine_interventions=["Divine recognition"],
            geographic_setting="Sparta",
            temporal_setting="Present visit, references to Troy and aftermath",
            narrative_techniques=["Recognition scene", "Parallel narratives"]
        )

        # BOOK 5: Odysseus - Nymph and Shipwreck
        books[5] = OdysseyBook(
            number=5,
            title="Odysseus - Nymph and Shipwreck",
            narrative_arc=NarrativeArc.DEPARTURE,
            key_themes=["divine captivity", "longing for home", "divine mercy"],
            major_decisions=[
                NarrativeDecision(
                    book=5, scene="calypso_release",
                    decision_type=DecisionType.DIVINE,
                    narrative_arc=NarrativeArc.DEPARTURE,
                    description="Odysseus chooses mortality and home over immortality with Calypso",
                    options=["accept_immortality", "choose_mortality_home", "negotiate_compromise"],
                    canonical_choice="choose_mortality_home",
                    wisdom_level=0.9,  # Ultimate wisdom choice
                    consequences=["Freedom granted", "Journey resumed", "Divine favor earned"],
                    affected_characters=["Calypso", "Odysseus", "Hermes"],
                    divine_involvement=0.9,
                    narrative_weight=1.0,
                    temporal_complexity=0.8
                )
            ],
            character_developments={
                "Calypso": CharacterState("Calypso", trust_level=0.5, loyalty=0.4, power_level=0.9),
                "Odysseus": CharacterState("Odysseus", trust_level=0.8, loyalty=1.0, power_level=0.7)
            },
            divine_interventions=["Zeus's command", "Hermes's message"],
            geographic_setting="Ogygia (Calypso's island)",
            temporal_setting="Present captivity, eternal vs mortal time",
            narrative_techniques=["Divine intervention", "Existential choice"]
        )

        # BOOK 6: The Princess and the Stranger
        books[6] = OdysseyBook(
            number=6,
            title="The Princess and the Stranger",
            narrative_arc=NarrativeArc.DEPARTURE,
            key_themes=["hospitality customs", "divine guidance", "youth and wisdom"],
            major_decisions=[
                NarrativeDecision(
                    book=6, scene="nausicaa_encounter",
                    decision_type=DecisionType.STRATEGIC,
                    narrative_arc=NarrativeArc.DEPARTURE,
                    description="Odysseus approaches Nausicaa for help",
                    options=["direct_supplication", "modest_approach", "reveal_identity"],
                    canonical_choice="modest_approach",
                    wisdom_level=0.8,
                    consequences=["Gains assistance", "Maintains dignity", "Secures guidance"],
                    affected_characters=["Nausicaa", "Odysseus"],
                    divine_involvement=0.4,
                    narrative_weight=0.6
                )
            ],
            character_developments={
                "Nausicaa": CharacterState("Nausicaa", trust_level=0.7, loyalty=0.6, power_level=0.4)
            },
            divine_interventions=["Athena's dream to Nausicaa"],
            geographic_setting="Phaeacia",
            temporal_setting="Present encounter, hints of future",
            narrative_techniques=["Divine machination", "Youth meets experience"]
        )

        # BOOK 7: The Luxurious Palace
        books[7] = OdysseyBook(
            number=7,
            title="The Luxurious Palace",
            narrative_arc=NarrativeArc.DEPARTURE,
            key_themes=["royal hospitality", "divine protection", "storytelling power"],
            major_decisions=[
                NarrativeDecision(
                    book=7, scene="alcinous_court",
                    decision_type=DecisionType.IDENTITY,
                    narrative_arc=NarrativeArc.DEPARTURE,
                    description="Odysseus reveals himself to King Alcinous",
                    options=["remain_anonymous", "partial_revelation", "full_identity"],
                    canonical_choice="partial_revelation",
                    wisdom_level=0.7,
                    consequences=["Trust built gradually", "Safety maintained", "Story setup"],
                    affected_characters=["Alcinous", "Arete", "Odysseus"],
                    divine_involvement=0.3,
                    narrative_weight=0.7
                )
            ],
            character_developments={
                "Alcinous": CharacterState("Alcinous", trust_level=0.8, loyalty=0.7, power_level=0.8),
                "Arete": CharacterState("Arete", trust_level=0.7, loyalty=0.6, power_level=0.7)
            },
            divine_interventions=["Athena's continued protection"],
            geographic_setting="Alcinous's palace, Phaeacia",
            temporal_setting="Present hospitality, anticipation of story",
            narrative_techniques=["Court intrigue", "Gradual revelation"]
        )

        # BOOK 8: A Day for Songs and Contests
        books[8] = OdysseyBook(
            number=8,
            title="A Day for Songs and Contests",
            narrative_arc=NarrativeArc.DEPARTURE,
            key_themes=["heroic fame", "emotional revelation", "artistic truth"],
            major_decisions=[
                NarrativeDecision(
                    book=8, scene="demodocus_songs",
                    decision_type=DecisionType.MORAL,
                    narrative_arc=NarrativeArc.DEPARTURE,
                    description="Odysseus weeps at songs of Troy, revealing his identity",
                    options=["control_emotions", "reveal_through_tears", "deflect_attention"],
                    canonical_choice="reveal_through_tears",
                    wisdom_level=0.6,
                    consequences=["Emotional truth revealed", "Identity suspected", "Story demanded"],
                    affected_characters=["Demodocus", "Alcinous", "Odysseus"],
                    divine_involvement=0.2,
                    narrative_weight=0.8,
                    temporal_complexity=0.7
                )
            ],
            character_developments={
                "Demodocus": CharacterState("Demodocus", trust_level=0.8, loyalty=0.7, power_level=0.6)
            },
            divine_interventions=["Muse's inspiration"],
            geographic_setting="Phaeacian court",
            temporal_setting="Present performance, past events sung",
            narrative_techniques=["Meta-narrative", "Emotional revelation"]
        )

        # BOOK 10: The Bewitching Queen of Aeaea
        books[10] = OdysseyBook(
            number=10,
            title="The Bewitching Queen of Aeaea",
            narrative_arc=NarrativeArc.TRIALS,
            key_themes=["magical transformation", "leadership testing", "divine knowledge"],
            major_decisions=[
                NarrativeDecision(
                    book=10, scene="circe_encounter",
                    decision_type=DecisionType.STRATEGIC,
                    narrative_arc=NarrativeArc.TRIALS,
                    description="Odysseus confronts Circe to save his transformed crew",
                    options=["direct_attack", "magical_protection", "diplomatic_approach"],
                    canonical_choice="magical_protection",
                    wisdom_level=0.8,
                    consequences=["Crew restored", "Circe becomes ally", "Knowledge gained"],
                    affected_characters=["Circe", "Odysseus", "Crew"],
                    divine_involvement=0.7,
                    narrative_weight=0.9,
                    temporal_complexity=0.6
                )
            ],
            character_developments={
                "Circe": CharacterState("Circe", trust_level=0.6, loyalty=0.7, power_level=0.9)
            },
            divine_interventions=["Hermes's magical aid"],
            geographic_setting="Aeaea island",
            temporal_setting="Present trial, magical time",
            narrative_techniques=["Magical realism", "Transformation theme"]
        )

        # BOOK 11: The Kingdom of the Dead
        books[11] = OdysseyBook(
            number=11,
            title="The Kingdom of the Dead",
            narrative_arc=NarrativeArc.TRIALS,
            key_themes=["death and prophecy", "family legacy", "heroic knowledge"],
            major_decisions=[
                NarrativeDecision(
                    book=11, scene="underworld_consultation",
                    decision_type=DecisionType.DIVINE,
                    narrative_arc=NarrativeArc.TRIALS,
                    description="Odysseus seeks prophecy from Tiresias",
                    options=["accept_all_prophecies", "question_fate", "seek_alternatives"],
                    canonical_choice="accept_all_prophecies",
                    wisdom_level=0.8,
                    consequences=["Future knowledge gained", "Family status learned", "Path clarified"],
                    affected_characters=["Tiresias", "Anticlea", "Agamemnon"],
                    divine_involvement=1.0,
                    narrative_weight=1.0,
                    temporal_complexity=1.0
                )
            ],
            character_developments={
                "Tiresias": CharacterState("Tiresias", trust_level=0.9, loyalty=0.8, power_level=1.0)
            },
            divine_interventions=["Underworld access", "Prophetic revelation"],
            geographic_setting="The Underworld",
            temporal_setting="Eternal time, past-present-future convergence",
            narrative_techniques=["Nekyia", "Prophetic vision", "Temporal collapse"]
        )

        # BOOK 13: Ithaca at Last
        books[13] = OdysseyBook(
            number=13,
            title="Ithaca at Last",
            narrative_arc=NarrativeArc.WANDERING,
            key_themes=["homecoming reality", "divine deception", "identity concealment"],
            major_decisions=[
                NarrativeDecision(
                    book=13, scene="athena_disguise",
                    decision_type=DecisionType.STRATEGIC,
                    narrative_arc=NarrativeArc.WANDERING,
                    description="Athena disguises Odysseus for strategic return",
                    options=["immediate_revelation", "accept_disguise", "partial_concealment"],
                    canonical_choice="accept_disguise",
                    wisdom_level=0.9,
                    consequences=["Strategic advantage gained", "Safety ensured", "Plan developed"],
                    affected_characters=["Athena", "Odysseus"],
                    divine_involvement=0.8,
                    narrative_weight=0.9,
                    temporal_complexity=0.7
                )
            ],
            character_developments={
                "Athena": CharacterState("Athena", trust_level=1.0, loyalty=1.0, power_level=1.0)
            },
            divine_interventions=["Athena's disguise", "Divine strategy"],
            geographic_setting="Ithaca shore",
            temporal_setting="Present return, future planning",
            narrative_techniques=["Divine transformation", "Strategic concealment"]
        )

        # BOOK 14: The Loyal Swineherd
        books[14] = OdysseyBook(
            number=14,
            title="The Loyal Swineherd",
            narrative_arc=NarrativeArc.WANDERING,
            key_themes=["loyalty testing", "class and service", "storytelling identity"],
            major_decisions=[
                NarrativeDecision(
                    book=14, scene="eumaeus_test",
                    decision_type=DecisionType.LOYALTY,
                    narrative_arc=NarrativeArc.WANDERING,
                    description="Odysseus tests Eumaeus's loyalty while disguised",
                    options=["reveal_immediately", "test_thoroughly", "maintain_disguise"],
                    canonical_choice="test_thoroughly",
                    wisdom_level=0.8,
                    consequences=["Loyalty confirmed", "Trust built", "Alliance secured"],
                    affected_characters=["Eumaeus", "Odysseus"],
                    divine_involvement=0.2,
                    narrative_weight=0.7
                )
            ],
            character_developments={
                "Eumaeus": CharacterState("Eumaeus", trust_level=1.0, loyalty=1.0, power_level=0.4)
            },
            divine_interventions=["Divine disguise maintained"],
            geographic_setting="Eumaeus's hut",
            temporal_setting="Present testing, past loyalty referenced",
            narrative_techniques=["Loyalty test", "Class exploration"]
        )

        # BOOK 15: The Prince Sets Sail for Home
        books[15] = OdysseyBook(
            number=15,
            title="The Prince Sets Sail for Home",
            narrative_arc=NarrativeArc.WANDERING,
            key_themes=["generational reunion", "divine timing", "parallel journeys"],
            major_decisions=[
                NarrativeDecision(
                    book=15, scene="telemachus_return",
                    decision_type=DecisionType.STRATEGIC,
                    narrative_arc=NarrativeArc.WANDERING,
                    description="Telemachus decides to return to Ithaca",
                    options=["direct_return", "cautious_approach", "seek_more_allies"],
                    canonical_choice="cautious_approach",
                    wisdom_level=0.7,
                    consequences=["Safe return ensured", "Timing coordinated", "Danger avoided"],
                    affected_characters=["Telemachus", "Athena"],
                    divine_involvement=0.6,
                    narrative_weight=0.8
                )
            ],
            character_developments={
                "Telemachus": CharacterState("Telemachus", trust_level=0.9, loyalty=1.0, power_level=0.6)
            },
            divine_interventions=["Athena's guidance"],
            geographic_setting="Sparta to Ithaca",
            temporal_setting="Present journey, convergent timing",
            narrative_techniques=["Parallel narratives", "Divine timing"]
        )

        # BOOK 16: Father and Son
        books[16] = OdysseyBook(
            number=16,
            title="Father and Son",
            narrative_arc=NarrativeArc.RECOGNITION,
            key_themes=["family reunion", "identity revelation", "generational inheritance"],
            major_decisions=[
                NarrativeDecision(
                    book=16, scene="father_son_reunion",
                    decision_type=DecisionType.IDENTITY,
                    narrative_arc=NarrativeArc.RECOGNITION,
                    description="Odysseus reveals himself to Telemachus",
                    options=["gradual_revelation", "immediate_proof", "test_first"],
                    canonical_choice="immediate_proof",
                    wisdom_level=0.8,
                    consequences=["Family bond restored", "Alliance formed", "Plan developed"],
                    affected_characters=["Telemachus", "Odysseus"],
                    divine_involvement=0.4,
                    narrative_weight=1.0,
                    temporal_complexity=0.8
                )
            ],
            character_developments={
                "Telemachus": CharacterState("Telemachus", trust_level=1.0, loyalty=1.0, power_level=0.7)
            },
            divine_interventions=["Athena's revelation"],
            geographic_setting="Eumaeus's hut",
            temporal_setting="Present reunion, future planning",
            narrative_techniques=["Recognition scene", "Family restoration"]
        )

        # BOOK 17: Stranger at the Gates
        books[17] = OdysseyBook(
            number=17,
            title="Stranger at the Gates",
            narrative_arc=NarrativeArc.RECOGNITION,
            key_themes=["disguised testing", "hospitality violation", "animal loyalty"],
            major_decisions=[
                NarrativeDecision(
                    book=17, scene="beggar_entrance",
                    decision_type=DecisionType.STRATEGIC,
                    narrative_arc=NarrativeArc.RECOGNITION,
                    description="Odysseus enters his palace disguised as beggar",
                    options=["aggressive_approach", "humble_disguise", "reveal_partially"],
                    canonical_choice="humble_disguise",
                    wisdom_level=0.9,
                    consequences=["Information gathered", "Enemies revealed", "Strategy refined"],
                    affected_characters=["Suitors", "Odysseus", "Argos"],
                    divine_involvement=0.3,
                    narrative_weight=0.8
                )
            ],
            character_developments={
                "Argos": CharacterState("Argos", trust_level=1.0, loyalty=1.0, power_level=0.1)
            },
            divine_interventions=["Divine disguise maintained"],
            geographic_setting="Odysseus's palace",
            temporal_setting="Present infiltration, emotional recognition",
            narrative_techniques=["Dramatic irony", "Animal recognition"]
        )

        # BOOK 18: The Beggar-King of Ithaca
        books[18] = OdysseyBook(
            number=18,
            title="The Beggar-King of Ithaca",
            narrative_arc=NarrativeArc.RECOGNITION,
            key_themes=["strength revelation", "social hierarchy", "divine justice preview"],
            major_decisions=[
                NarrativeDecision(
                    book=18, scene="beggar_fight",
                    decision_type=DecisionType.HEROIC,
                    narrative_arc=NarrativeArc.RECOGNITION,
                    description="Odysseus fights Irus the beggar",
                    options=["reveal_full_strength", "controlled_victory", "avoid_conflict"],
                    canonical_choice="controlled_victory",
                    wisdom_level=0.8,
                    consequences=["Strength hinted", "Respect gained", "Cover maintained"],
                    affected_characters=["Irus", "Suitors", "Odysseus"],
                    divine_involvement=0.3,
                    narrative_weight=0.6
                )
            ],
            character_developments={
                "Irus": CharacterState("Irus", trust_level=0.1, loyalty=0.0, power_level=0.2)
            },
            divine_interventions=["Divine strength control"],
            geographic_setting="Palace courtyard",
            temporal_setting="Present demonstration, future foreshadowing",
            narrative_techniques=["Strength revelation", "Social commentary"]
        )

        # BOOK 19: Eurycleia Recognizes Odysseus
        books[19] = OdysseyBook(
            number=19,
            title="Eurycleia Recognizes Odysseus",
            narrative_arc=NarrativeArc.RECOGNITION,
            key_themes=["servant loyalty", "scar recognition", "identity verification"],
            major_decisions=[
                NarrativeDecision(
                    book=19, scene="scar_recognition",
                    decision_type=DecisionType.IDENTITY,
                    narrative_arc=NarrativeArc.RECOGNITION,
                    description="Eurycleia recognizes Odysseus by his scar",
                    options=["silence_nurse", "partial_revelation", "trust_completely"],
                    canonical_choice="silence_nurse",
                    wisdom_level=0.9,
                    consequences=["Secret maintained", "Ally secured", "Timing preserved"],
                    affected_characters=["Eurycleia", "Odysseus"],
                    divine_involvement=0.2,
                    narrative_weight=0.9,
                    temporal_complexity=0.7
                )
            ],
            character_developments={
                "Eurycleia": CharacterState("Eurycleia", trust_level=1.0, loyalty=1.0, power_level=0.3)
            },
            divine_interventions=["Divine timing"],
            geographic_setting="Palace chambers",
            temporal_setting="Present recognition, childhood past",
            narrative_techniques=["Physical recognition", "Servant loyalty"]
        )

        # BOOK 20: Portents Gather
        books[20] = OdysseyBook(
            number=20,
            title="Portents Gather",
            narrative_arc=NarrativeArc.RECOGNITION,
            key_themes=["divine omens", "final preparation", "justice approaching"],
            major_decisions=[
                NarrativeDecision(
                    book=20, scene="omens_interpretation",
                    decision_type=DecisionType.DIVINE,
                    narrative_arc=NarrativeArc.RECOGNITION,
                    description="Odysseus interprets divine omens of coming justice",
                    options=["act_immediately", "wait_for_sign", "prepare_carefully"],
                    canonical_choice="prepare_carefully",
                    wisdom_level=0.9,
                    consequences=["Perfect timing achieved", "Divine favor confirmed", "Victory ensured"],
                    affected_characters=["Odysseus", "Athena"],
                    divine_involvement=0.8,
                    narrative_weight=0.8,
                    temporal_complexity=0.8
                )
            ],
            character_developments={},
            divine_interventions=["Divine omens", "Zeus's signs"],
            geographic_setting="Palace",
            temporal_setting="Present preparation, imminent future",
            narrative_techniques=["Omen interpretation", "Divine foreshadowing"]
        )

        # BOOK 21: Odysseus Strings His Bow
        books[21] = OdysseyBook(
            number=21,
            title="Odysseus Strings His Bow",
            narrative_arc=NarrativeArc.RESOLUTION,
            key_themes=["heroic proof", "divine right", "contest resolution"],
            major_decisions=[
                NarrativeDecision(
                    book=21, scene="bow_contest",
                    decision_type=DecisionType.HEROIC,
                    narrative_arc=NarrativeArc.RESOLUTION,
                    description="Odysseus reveals his identity by stringing his bow",
                    options=["dramatic_revelation", "gradual_demonstration", "immediate_action"],
                    canonical_choice="dramatic_revelation",
                    wisdom_level=0.9,
                    consequences=["Identity proven", "Right established", "Justice begins"],
                    affected_characters=["Penelope", "Suitors", "Odysseus"],
                    divine_involvement=0.6,
                    narrative_weight=1.0,
                    temporal_complexity=0.6
                )
            ],
            character_developments={
                "Penelope": CharacterState("Penelope", trust_level=0.8, loyalty=1.0, power_level=0.6)
            },
            divine_interventions=["Divine strength", "Bow's recognition"],
            geographic_setting="Great hall",
            temporal_setting="Present proof, destiny fulfillment",
            narrative_techniques=["Object recognition", "Heroic demonstration"]
        )

        # BOOK 22: Slaughter in the Hall
        books[22] = OdysseyBook(
            number=22,
            title="Slaughter in the Hall",
            narrative_arc=NarrativeArc.RESOLUTION,
            key_themes=["divine justice", "heroic vengeance", "moral judgment"],
            major_decisions=[
                NarrativeDecision(
                    book=22, scene="suitor_slaughter",
                    decision_type=DecisionType.MORAL,
                    narrative_arc=NarrativeArc.RESOLUTION,
                    description="Odysseus decides extent of vengeance against suitors",
                    options=["total_elimination", "spare_some_suitors", "merciful_justice"],
                    canonical_choice="total_elimination",
                    wisdom_level=0.7,  # Justice but harsh
                    consequences=["Complete justice", "Order restored", "Peace through strength"],
                    affected_characters=["Suitors", "Odysseus", "Telemachus"],
                    divine_involvement=0.8,
                    narrative_weight=1.0,
                    temporal_complexity=0.5
                )
            ],
            character_developments={},
            divine_interventions=["Divine justice", "Athena's battle aid"],
            geographic_setting="Great hall",
            temporal_setting="Present justice, past crimes punished",
            narrative_techniques=["Divine justice", "Heroic aristeia"]
        )

        # BOOK 24: Peace
        books[24] = OdysseyBook(
            number=24,
            title="Peace",
            narrative_arc=NarrativeArc.RESOLUTION,
            key_themes=["divine reconciliation", "social healing", "cycle completion"],
            major_decisions=[
                NarrativeDecision(
                    book=24, scene="divine_peace",
                    decision_type=DecisionType.DIVINE,
                    narrative_arc=NarrativeArc.RESOLUTION,
                    description="Athena establishes peace between families",
                    options=["continued_vengeance", "divine_intervention", "negotiated_peace"],
                    canonical_choice="divine_intervention",
                    wisdom_level=1.0,  # Ultimate divine wisdom
                    consequences=["Peace restored", "Cycle completed", "Social order renewed"],
                    affected_characters=["Athena", "Odysseus", "Families of suitors"],
                    divine_involvement=1.0,
                    narrative_weight=1.0,
                    temporal_complexity=0.9
                )
            ],
            character_developments={
                "Athena": CharacterState("Athena", trust_level=1.0, loyalty=1.0, power_level=1.0)
            },
            divine_interventions=["Divine peace decree", "Eternal resolution"],
            geographic_setting="Ithaca",
            temporal_setting="Present resolution, eternal future peace",
            narrative_techniques=["Divine resolution", "Eternal peace"]
        )
        
        return books
        
    def _track_character_evolution(self) -> Dict[str, List[CharacterState]]:
        """Track how characters evolve across the narrative"""
        character_timeline = {}
        
        for book_num, book in self.books.items():
            for char_name, char_state in book.character_developments.items():
                if char_name not in character_timeline:
                    character_timeline[char_name] = []
                
                # Add book reference to state
                char_state.relationship_history.append(f"Book {book_num}: {book.title}")
                character_timeline[char_name].append(char_state)
                
        return character_timeline
        
    def _analyze_thematic_progression(self) -> Dict[str, List[float]]:
        """Analyze how themes develop across books"""
        theme_evolution = {}
        
        for book_num, book in self.books.items():
            for theme in book.key_themes:
                if theme not in theme_evolution:
                    theme_evolution[theme] = []
                    
                # Calculate theme intensity in this book
                theme_intensity = len([d for d in book.major_decisions 
                                    if theme in str(d.description).lower()]) / max(len(book.major_decisions), 1)
                theme_evolution[theme].append(theme_intensity)
                
        return theme_evolution
        
    def get_decision_sequence(self) -> List[NarrativeDecision]:
        """Get all decisions in chronological order"""
        all_decisions = []
        for book_num in sorted(self.books.keys()):
            all_decisions.extend(self.books[book_num].major_decisions)
        return all_decisions
        
    def get_books_by_arc(self, arc: NarrativeArc) -> List[OdysseyBook]:
        """Get all books in a specific narrative arc"""
        return [book for book in self.books.values() if book.narrative_arc == arc]
        
    def analyze_decision_complexity_evolution(self) -> List[float]:
        """Track how decision complexity evolves across the epic"""
        complexity_scores = []
        
        for book_num in sorted(self.books.keys()):
            book = self.books[book_num]
            if book.major_decisions:
                avg_complexity = np.mean([
                    (d.temporal_complexity + d.narrative_weight + d.divine_involvement) / 3
                    for d in book.major_decisions
                ])
                complexity_scores.append(avg_complexity)
            else:
                complexity_scores.append(0.0)
                
        return complexity_scores

def analyze_narrative_structure():
    """Analyze the complete Odyssey structure"""
    odyssey = FullOdysseyStructure()
    
    print("📚 FULL ODYSSEY NARRATIVE STRUCTURE")
    print("=" * 60)
    
    # Analyze by narrative arc
    for arc in NarrativeArc:
        books = odyssey.get_books_by_arc(arc)
        print(f"\n{arc.value.upper()} ARC: {len(books)} books")
        
        if books:
            total_decisions = sum(len(book.major_decisions) for book in books)
            avg_divine = np.mean([d.divine_involvement for book in books for d in book.major_decisions])
            avg_wisdom = np.mean([d.wisdom_level for book in books for d in book.major_decisions])
            
            print(f"  Total Decisions: {total_decisions}")
            print(f"  Average Divine Involvement: {avg_divine:.3f}")
            print(f"  Average Wisdom Level: {avg_wisdom:.3f}")
    
    # Decision complexity evolution
    complexity_evolution = odyssey.analyze_decision_complexity_evolution()
    print(f"\n📈 DECISION COMPLEXITY EVOLUTION")
    print(f"Starting Complexity: {complexity_evolution[0]:.3f}")
    print(f"Peak Complexity: {max(complexity_evolution):.3f}")
    print(f"Final Complexity: {complexity_evolution[-1]:.3f}")
    
    # Character development
    print(f"\n👥 CHARACTER EVOLUTION")
    char_timeline = odyssey.character_evolution
    for char_name, states in char_timeline.items():
        if len(states) > 1:
            initial_trust = states[0].trust_level
            final_trust = states[-1].trust_level
            trust_change = final_trust - initial_trust
            print(f"{char_name}: Trust {initial_trust:.2f} → {final_trust:.2f} (Δ{trust_change:+.2f})")
    
    return odyssey

if __name__ == "__main__":
    odyssey_structure = analyze_narrative_structure()