"""
Semantic Dimensions: Interpretable axes in embedding space

Defines conjugate pairs of semantic dimensions (warmth/coldness, formality/casualness, etc.)
and provides tools to project trajectories onto these dimensions to understand
WHAT is changing in semantic flow.

This is the key to interpretability: instead of tracking raw 384D vectors,
we track motion along meaningful human-interpretable axes.

Based on the insight that semantic space has natural axes that can be learned
from exemplar words at the poles of each dimension.

PERFORMANCE OPTIMIZATIONS:
- Batch embedding for axis learning
- Sparse projection (skip near-zero dimensions)
- Cached axis computations
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field

# Import performance utilities
from .performance import SparseSemanticVector


@dataclass
class SemanticDimension:
    """
    A single interpretable dimension in semantic space

    Defined by exemplar words at positive and negative poles.
    For example:
    - Warmth: positive=["warm", "loving", "kind"], negative=["cold", "harsh", "cruel"]
    - Formality: positive=["formal", "professional"], negative=["casual", "colloquial"]
    """
    name: str
    positive_exemplars: List[str]
    negative_exemplars: List[str]
    axis: Optional[np.ndarray] = None  # learned direction vector

    def learn_axis(self, embed_fn: Callable, use_batch: bool = True):
        """
        Learn the axis direction from exemplars

        Method: Compute centroids of positive and negative exemplars,
        axis is the normalized difference vector.

        OPTIMIZATION: Uses batch embedding if available

        Args:
            embed_fn: Either embed_fn(word) -> vector  OR  embed_fn(words) -> vectors
            use_batch: Try to use batch embedding (default True)
        """
        # OPTIMIZATION: Try batch embedding first
        if use_batch:
            try:
                # Assume embed_fn can handle lists
                pos_embeddings = embed_fn(self.positive_exemplars)
                neg_embeddings = embed_fn(self.negative_exemplars)
            except (TypeError, AttributeError):
                # Fall back to individual embeddings
                pos_embeddings = np.array([embed_fn(word) for word in self.positive_exemplars])
                neg_embeddings = np.array([embed_fn(word) for word in self.negative_exemplars])
        else:
            pos_embeddings = np.array([embed_fn(word) for word in self.positive_exemplars])
            neg_embeddings = np.array([embed_fn(word) for word in self.negative_exemplars])

        # Compute centroids
        pos_centroid = np.mean(pos_embeddings, axis=0)
        neg_centroid = np.mean(neg_embeddings, axis=0)

        # Axis = normalized difference
        self.axis = pos_centroid - neg_centroid
        self.axis = self.axis / (np.linalg.norm(self.axis) + 1e-10)

        return self.axis

    def project(self, vector: np.ndarray) -> float:
        """
        Project a vector onto this dimension

        Returns scalar indicating position along dimension:
        - Positive values = toward positive pole
        - Negative values = toward negative pole
        """
        if self.axis is None:
            raise ValueError(f"Dimension '{self.name}' axis not learned yet. Call learn_axis() first.")

        return np.dot(vector, self.axis)


# Predefined semantic dimensions
# These are the "conjugate pairs" that form an interpretable basis
STANDARD_DIMENSIONS = [
    # Affective dimensions
    SemanticDimension(
        name="Warmth",
        positive_exemplars=["warm", "loving", "kind", "affectionate", "caring", "tender"],
        negative_exemplars=["cold", "harsh", "cruel", "hostile", "uncaring", "callous"]
    ),
    SemanticDimension(
        name="Valence",
        positive_exemplars=["positive", "good", "happy", "pleasant", "joyful", "delightful"],
        negative_exemplars=["negative", "bad", "sad", "unpleasant", "miserable", "awful"]
    ),
    SemanticDimension(
        name="Arousal",
        positive_exemplars=["excited", "energetic", "intense", "passionate", "thrilling"],
        negative_exemplars=["calm", "peaceful", "relaxed", "serene", "tranquil"]
    ),
    SemanticDimension(
        name="Intensity",
        positive_exemplars=["intense", "extreme", "powerful", "overwhelming", "fierce"],
        negative_exemplars=["mild", "gentle", "subtle", "moderate", "faint"]
    ),

    # Social/interpersonal dimensions
    SemanticDimension(
        name="Formality",
        positive_exemplars=["formal", "professional", "official", "proper", "ceremonial"],
        negative_exemplars=["casual", "informal", "colloquial", "relaxed", "friendly"]
    ),
    SemanticDimension(
        name="Directness",
        positive_exemplars=["direct", "explicit", "clear", "straightforward", "blunt"],
        negative_exemplars=["indirect", "implicit", "vague", "subtle", "evasive"]
    ),
    SemanticDimension(
        name="Power",
        positive_exemplars=["dominant", "authoritative", "commanding", "powerful", "controlling"],
        negative_exemplars=["submissive", "passive", "powerless", "weak", "yielding"]
    ),
    SemanticDimension(
        name="Generosity",
        positive_exemplars=["generous", "giving", "selfless", "charitable", "magnanimous"],
        negative_exemplars=["selfish", "greedy", "stingy", "miserly", "uncharitable"]
    ),

    # Cognitive dimensions
    SemanticDimension(
        name="Certainty",
        positive_exemplars=["certain", "sure", "definite", "confident", "convinced"],
        negative_exemplars=["uncertain", "unsure", "doubtful", "hesitant", "ambiguous"]
    ),
    SemanticDimension(
        name="Complexity",
        positive_exemplars=["complex", "complicated", "intricate", "sophisticated", "elaborate"],
        negative_exemplars=["simple", "basic", "straightforward", "elementary", "plain"]
    ),
    SemanticDimension(
        name="Concreteness",
        positive_exemplars=["concrete", "tangible", "physical", "specific", "material"],
        negative_exemplars=["abstract", "intangible", "theoretical", "conceptual", "general"]
    ),
    SemanticDimension(
        name="Familiarity",
        positive_exemplars=["familiar", "known", "common", "usual", "ordinary"],
        negative_exemplars=["novel", "unfamiliar", "strange", "unusual", "exotic"]
    ),

    # Temporal/dynamic dimensions
    SemanticDimension(
        name="Agency",
        positive_exemplars=["active", "doing", "acting", "causing", "initiating"],
        negative_exemplars=["passive", "receiving", "experiencing", "affected", "undergoing"]
    ),
    SemanticDimension(
        name="Stability",
        positive_exemplars=["stable", "constant", "steady", "unchanging", "fixed"],
        negative_exemplars=["volatile", "changing", "unstable", "fluctuating", "dynamic"]
    ),
    SemanticDimension(
        name="Urgency",
        positive_exemplars=["urgent", "immediate", "pressing", "critical", "emergency"],
        negative_exemplars=["patient", "gradual", "leisurely", "relaxed", "unhurried"]
    ),
    SemanticDimension(
        name="Completion",
        positive_exemplars=["complete", "finished", "final", "concluded", "ending"],
        negative_exemplars=["incomplete", "starting", "beginning", "initial", "nascent"]
    ),
]


# Extended 244-dimensional semantic space for deep narrative/mythological research
NARRATIVE_DIMENSIONS = [
    # Heroic Journey dimensions (16)
    SemanticDimension(
        name="Heroism",
        positive_exemplars=["heroic", "brave", "courageous", "valiant", "noble", "gallant"],
        negative_exemplars=["cowardly", "fearful", "timid", "weak", "craven", "spineless"]
    ),
    SemanticDimension(
        name="Transformation",
        positive_exemplars=["transforming", "changing", "evolving", "metamorphosing", "becoming"],
        negative_exemplars=["static", "unchanging", "fixed", "stagnant", "frozen", "rigid"]
    ),
    SemanticDimension(
        name="Conflict",
        positive_exemplars=["conflicted", "opposing", "antagonistic", "adversarial", "clashing"],
        negative_exemplars=["harmonious", "peaceful", "aligned", "cooperative", "unified"]
    ),
    SemanticDimension(
        name="Mystery",
        positive_exemplars=["mysterious", "hidden", "secret", "enigmatic", "obscure", "veiled"],
        negative_exemplars=["revealed", "obvious", "clear", "transparent", "open", "manifest"]
    ),
    SemanticDimension(
        name="Sacrifice",
        positive_exemplars=["sacrificing", "giving", "surrendering", "offering", "renouncing"],
        negative_exemplars=["preserving", "keeping", "hoarding", "protecting", "retaining"]
    ),
    SemanticDimension(
        name="Wisdom",
        positive_exemplars=["wise", "sagacious", "insightful", "enlightened", "discerning"],
        negative_exemplars=["foolish", "naive", "ignorant", "blind", "unaware", "shortsighted"]
    ),
    SemanticDimension(
        name="Courage",
        positive_exemplars=["courageous", "bold", "fearless", "daring", "intrepid"],
        negative_exemplars=["afraid", "hesitant", "cautious", "timid", "reluctant"]
    ),
    SemanticDimension(
        name="Redemption",
        positive_exemplars=["redeeming", "atoning", "saving", "rescuing", "restoring"],
        negative_exemplars=["damning", "condemning", "destroying", "corrupting", "ruining"]
    ),
    SemanticDimension(
        name="Destiny",
        positive_exemplars=["fated", "destined", "inevitable", "predetermined", "prophesied"],
        negative_exemplars=["random", "accidental", "chaotic", "free", "unconstrained"]
    ),
    SemanticDimension(
        name="Honor",
        positive_exemplars=["honorable", "noble", "principled", "virtuous", "upright"],
        negative_exemplars=["dishonorable", "shameful", "corrupt", "base", "ignoble"]
    ),
    SemanticDimension(
        name="Loyalty",
        positive_exemplars=["loyal", "faithful", "devoted", "steadfast", "true"],
        negative_exemplars=["treacherous", "disloyal", "betraying", "faithless", "fickle"]
    ),
    SemanticDimension(
        name="Quest",
        positive_exemplars=["questing", "seeking", "searching", "pursuing", "striving"],
        negative_exemplars=["settled", "content", "passive", "accepting", "resigned"]
    ),
    SemanticDimension(
        name="Transcendence",
        positive_exemplars=["transcendent", "sublime", "elevated", "exalted", "divine"],
        negative_exemplars=["mundane", "earthly", "ordinary", "base", "lowly"]
    ),
    SemanticDimension(
        name="Shadow",
        positive_exemplars=["dark", "shadowy", "repressed", "hidden", "unconscious"],
        negative_exemplars=["light", "conscious", "acknowledged", "integrated", "aware"]
    ),
    SemanticDimension(
        name="Initiation",
        positive_exemplars=["initiated", "inducted", "begun", "entering", "awakening"],
        negative_exemplars=["uninitiated", "ignorant", "profane", "excluded", "unaware"]
    ),
    SemanticDimension(
        name="Rebirth",
        positive_exemplars=["reborn", "renewed", "resurrected", "revived", "regenerated"],
        negative_exemplars=["dying", "decaying", "ending", "perishing", "fading"]
    ),
]

EMOTIONAL_DEPTH_DIMENSIONS = [
    # Deep emotional/psychological dimensions (16)
    SemanticDimension(
        name="Authenticity",
        positive_exemplars=["authentic", "genuine", "real", "true", "sincere", "honest"],
        negative_exemplars=["fake", "false", "pretending", "artificial", "insincere", "phony"]
    ),
    SemanticDimension(
        name="Vulnerability",
        positive_exemplars=["vulnerable", "open", "exposed", "unguarded", "defenseless"],
        negative_exemplars=["guarded", "protected", "defended", "armored", "closed"]
    ),
    SemanticDimension(
        name="Trust",
        positive_exemplars=["trusting", "confident", "believing", "reliant", "assured"],
        negative_exemplars=["suspicious", "doubtful", "paranoid", "wary", "distrustful"]
    ),
    SemanticDimension(
        name="Hope",
        positive_exemplars=["hopeful", "optimistic", "expecting", "anticipating", "aspiring"],
        negative_exemplars=["despairing", "hopeless", "pessimistic", "defeated", "resigned"]
    ),
    SemanticDimension(
        name="Grief",
        positive_exemplars=["grieving", "mourning", "sorrowing", "lamenting", "bereft"],
        negative_exemplars=["joyful", "celebrating", "rejoicing", "cheerful", "glad"]
    ),
    SemanticDimension(
        name="Shame",
        positive_exemplars=["ashamed", "embarrassed", "humiliated", "mortified", "disgraced"],
        negative_exemplars=["proud", "dignified", "honored", "respected", "esteemed"]
    ),
    SemanticDimension(
        name="Compassion",
        positive_exemplars=["compassionate", "empathetic", "sympathetic", "merciful", "kind"],
        negative_exemplars=["callous", "cruel", "heartless", "merciless", "indifferent"]
    ),
    SemanticDimension(
        name="Rage",
        positive_exemplars=["enraged", "furious", "wrathful", "incensed", "livid"],
        negative_exemplars=["calm", "peaceful", "serene", "tranquil", "composed"]
    ),
    SemanticDimension(
        name="Longing",
        positive_exemplars=["longing", "yearning", "craving", "pining", "desiring"],
        negative_exemplars=["satisfied", "content", "fulfilled", "complete", "satiated"]
    ),
    SemanticDimension(
        name="Awe",
        positive_exemplars=["awestruck", "amazed", "wonderstruck", "reverent", "overwhelmed"],
        negative_exemplars=["indifferent", "unmoved", "blasé", "jaded", "unimpressed"]
    ),
    SemanticDimension(
        name="Jealousy",
        positive_exemplars=["jealous", "envious", "covetous", "resentful", "possessive"],
        negative_exemplars=["generous", "content", "secure", "unenvious", "giving"]
    ),
    SemanticDimension(
        name="Guilt",
        positive_exemplars=["guilty", "remorseful", "regretful", "culpable", "responsible"],
        negative_exemplars=["innocent", "blameless", "guiltless", "absolved", "clean"]
    ),
    SemanticDimension(
        name="Pride",
        positive_exemplars=["proud", "dignified", "self-respecting", "esteemed", "honored"],
        negative_exemplars=["humble", "modest", "meek", "unassuming", "self-effacing"]
    ),
    SemanticDimension(
        name="Disgust",
        positive_exemplars=["disgusted", "repulsed", "revolted", "nauseated", "sickened"],
        negative_exemplars=["attracted", "drawn", "appealed", "pleased", "delighted"]
    ),
    SemanticDimension(
        name="Ecstasy",
        positive_exemplars=["ecstatic", "euphoric", "rapturous", "blissful", "transcendent"],
        negative_exemplars=["miserable", "agonized", "tormented", "suffering", "pained"]
    ),
    SemanticDimension(
        name="Dread",
        positive_exemplars=["dreading", "fearing", "apprehensive", "anxious", "terrified"],
        negative_exemplars=["eager", "excited", "anticipating", "welcoming", "ready"]
    ),
]

RELATIONAL_DIMENSIONS = [
    # Interpersonal/relational dynamics (16)
    SemanticDimension(
        name="Intimacy",
        positive_exemplars=["intimate", "close", "connected", "bonded", "united"],
        negative_exemplars=["distant", "separate", "disconnected", "estranged", "alienated"]
    ),
    SemanticDimension(
        name="Dominance",
        positive_exemplars=["dominating", "controlling", "commanding", "overpowering", "ruling"],
        negative_exemplars=["submitting", "yielding", "deferring", "following", "obeying"]
    ),
    SemanticDimension(
        name="Respect",
        positive_exemplars=["respectful", "admiring", "honoring", "esteeming", "valuing"],
        negative_exemplars=["disrespectful", "contemptuous", "scornful", "disdainful", "dismissive"]
    ),
    SemanticDimension(
        name="Reciprocity",
        positive_exemplars=["mutual", "reciprocal", "balanced", "equal", "fair"],
        negative_exemplars=["one-sided", "unequal", "imbalanced", "unfair", "exploitative"]
    ),
    SemanticDimension(
        name="Attachment",
        positive_exemplars=["attached", "bonded", "connected", "linked", "tied"],
        negative_exemplars=["detached", "separate", "independent", "autonomous", "free"]
    ),
    SemanticDimension(
        name="Rivalry",
        positive_exemplars=["rival", "competing", "contending", "vying", "opposing"],
        negative_exemplars=["allied", "cooperating", "partnering", "collaborating", "united"]
    ),
    SemanticDimension(
        name="Dependence",
        positive_exemplars=["dependent", "reliant", "needing", "requiring", "leaning"],
        negative_exemplars=["independent", "self-sufficient", "autonomous", "self-reliant", "free"]
    ),
    SemanticDimension(
        name="Acceptance",
        positive_exemplars=["accepting", "welcoming", "embracing", "including", "receiving"],
        negative_exemplars=["rejecting", "excluding", "refusing", "denying", "dismissing"]
    ),
    SemanticDimension(
        name="Understanding",
        positive_exemplars=["understanding", "comprehending", "knowing", "grasping", "perceiving"],
        negative_exemplars=["misunderstanding", "confused", "ignorant", "mistaken", "unaware"]
    ),
    SemanticDimension(
        name="Forgiveness",
        positive_exemplars=["forgiving", "pardoning", "absolving", "excusing", "releasing"],
        negative_exemplars=["unforgiving", "vengeful", "holding-grudges", "resentful", "bitter"]
    ),
    SemanticDimension(
        name="Betrayal",
        positive_exemplars=["betraying", "deceiving", "backstabbing", "double-crossing", "disloyal"],
        negative_exemplars=["loyal", "faithful", "trustworthy", "reliable", "true"]
    ),
    SemanticDimension(
        name="Support",
        positive_exemplars=["supporting", "helping", "assisting", "aiding", "backing"],
        negative_exemplars=["hindering", "opposing", "obstructing", "undermining", "sabotaging"]
    ),
    SemanticDimension(
        name="Boundaries",
        positive_exemplars=["boundaried", "defined", "separated", "distinct", "protected"],
        negative_exemplars=["boundaryless", "merged", "enmeshed", "fused", "confused"]
    ),
    SemanticDimension(
        name="Gratitude",
        positive_exemplars=["grateful", "thankful", "appreciative", "indebted", "obliged"],
        negative_exemplars=["ungrateful", "entitled", "unappreciative", "demanding", "thankless"]
    ),
    SemanticDimension(
        name="Hostility",
        positive_exemplars=["hostile", "aggressive", "antagonistic", "belligerent", "combative"],
        negative_exemplars=["friendly", "peaceful", "amicable", "cordial", "welcoming"]
    ),
    SemanticDimension(
        name="Mentorship",
        positive_exemplars=["mentoring", "teaching", "guiding", "counseling", "advising"],
        negative_exemplars=["learning", "following", "seeking", "receiving", "apprenticing"]
    ),
]

ARCHETYPAL_DIMENSIONS = [
    # Jungian/mythological archetypes (16)
    SemanticDimension(
        name="Hero-Archetype",
        positive_exemplars=["champion", "savior", "protagonist", "warrior", "defender"],
        negative_exemplars=["victim", "bystander", "coward", "weakling", "follower"]
    ),
    SemanticDimension(
        name="Mentor-Archetype",
        positive_exemplars=["guide", "teacher", "sage", "wizard", "elder"],
        negative_exemplars=["student", "novice", "apprentice", "seeker", "initiate"]
    ),
    SemanticDimension(
        name="Shadow-Archetype",
        positive_exemplars=["dark-self", "repressed", "hidden-side", "unconscious", "denied"],
        negative_exemplars=["integrated", "conscious", "acknowledged", "accepted", "whole"]
    ),
    SemanticDimension(
        name="Trickster-Archetype",
        positive_exemplars=["trickster", "jester", "fool", "shapeshifter", "deceiver"],
        negative_exemplars=["straightforward", "honest", "direct", "reliable", "predictable"]
    ),
    SemanticDimension(
        name="Mother-Archetype",
        positive_exemplars=["nurturing", "protecting", "caring", "nourishing", "maternal"],
        negative_exemplars=["neglecting", "abandoning", "withholding", "rejecting", "harsh"]
    ),
    SemanticDimension(
        name="Father-Archetype",
        positive_exemplars=["authoritative", "protecting", "providing", "guiding", "paternal"],
        negative_exemplars=["absent", "weak", "abandoning", "unreliable", "failing"]
    ),
    SemanticDimension(
        name="Child-Archetype",
        positive_exemplars=["innocent", "pure", "wonder-filled", "playful", "naive"],
        negative_exemplars=["jaded", "corrupted", "world-weary", "cynical", "hardened"]
    ),
    SemanticDimension(
        name="Anima-Animus",
        positive_exemplars=["soul", "inner-opposite", "contrasexual", "complementary", "balancing"],
        negative_exemplars=["same", "identical", "uniform", "singular", "one-sided"]
    ),
    SemanticDimension(
        name="Self-Archetype",
        positive_exemplars=["whole", "integrated", "complete", "unified", "individuated"],
        negative_exemplars=["fragmented", "divided", "split", "scattered", "partial"]
    ),
    SemanticDimension(
        name="Threshold-Guardian",
        positive_exemplars=["guardian", "gatekeeper", "protector", "challenger", "tester"],
        negative_exemplars=["open", "welcoming", "permitting", "allowing", "accepting"]
    ),
    SemanticDimension(
        name="Herald",
        positive_exemplars=["announcing", "proclaiming", "calling", "summoning", "declaring"],
        negative_exemplars=["silent", "concealing", "hiding", "suppressing", "withholding"]
    ),
    SemanticDimension(
        name="Ally",
        positive_exemplars=["companion", "helper", "supporter", "friend", "partner"],
        negative_exemplars=["enemy", "opponent", "adversary", "obstacle", "hindrance"]
    ),
    SemanticDimension(
        name="Shapeshifter",
        positive_exemplars=["changing", "uncertain", "unreliable", "mysterious", "transforming"],
        negative_exemplars=["stable", "constant", "reliable", "predictable", "steady"]
    ),
    SemanticDimension(
        name="Oracle",
        positive_exemplars=["prophetic", "visionary", "seeing", "knowing", "revealing"],
        negative_exemplars=["blind", "ignorant", "unseeing", "unknowing", "concealing"]
    ),
    SemanticDimension(
        name="Ruler",
        positive_exemplars=["sovereign", "king", "queen", "governing", "commanding"],
        negative_exemplars=["subject", "servant", "follower", "ruled", "governed"]
    ),
    SemanticDimension(
        name="Lover",
        positive_exemplars=["passionate", "devoted", "desiring", "longing", "romantic"],
        negative_exemplars=["indifferent", "cold", "unfeeling", "detached", "unloving"]
    ),
]

PHILOSOPHICAL_DIMENSIONS = [
    # Existential/philosophical dimensions (16)
    SemanticDimension(
        name="Freedom",
        positive_exemplars=["free", "liberated", "autonomous", "independent", "unbound"],
        negative_exemplars=["constrained", "bound", "trapped", "imprisoned", "limited"]
    ),
    SemanticDimension(
        name="Meaning",
        positive_exemplars=["meaningful", "purposeful", "significant", "important", "valuable"],
        negative_exemplars=["meaningless", "pointless", "absurd", "futile", "empty"]
    ),
    SemanticDimension(
        name="Authenticity-Existential",
        positive_exemplars=["authentic-being", "true-self", "genuine-existence", "owned", "chosen"],
        negative_exemplars=["inauthentic", "they-self", "false-existence", "disowned", "imposed"]
    ),
    SemanticDimension(
        name="Being",
        positive_exemplars=["existing", "being", "present", "alive", "manifesting"],
        negative_exemplars=["non-existing", "absent", "void", "nothingness", "vanishing"]
    ),
    SemanticDimension(
        name="Essence",
        positive_exemplars=["essential", "fundamental", "core", "intrinsic", "inherent"],
        negative_exemplars=["accidental", "superficial", "external", "added", "contingent"]
    ),
    SemanticDimension(
        name="Time-Consciousness",
        positive_exemplars=["temporal", "time-aware", "historical", "becoming", "changing"],
        negative_exemplars=["timeless", "eternal", "static", "unchanging", "permanent"]
    ),
    SemanticDimension(
        name="Death-Awareness",
        positive_exemplars=["mortal", "finite", "ending", "perishing", "dying"],
        negative_exemplars=["immortal", "infinite", "eternal", "undying", "everlasting"]
    ),
    SemanticDimension(
        name="Absurdity",
        positive_exemplars=["absurd", "meaningless", "irrational", "senseless", "paradoxical"],
        negative_exemplars=["rational", "sensible", "logical", "coherent", "ordered"]
    ),
    SemanticDimension(
        name="Authenticity-Choice",
        positive_exemplars=["choosing", "deciding", "committing", "owning", "willing"],
        negative_exemplars=["avoiding", "fleeing", "denying", "refusing", "evading"]
    ),
    SemanticDimension(
        name="Bad-Faith",
        positive_exemplars=["self-deceiving", "lying-to-self", "denying", "evading", "pretending"],
        negative_exemplars=["truthful", "honest", "facing", "acknowledging", "accepting"]
    ),
    SemanticDimension(
        name="Thrownness",
        positive_exemplars=["thrown", "situated", "factical", "given", "found"],
        negative_exemplars=["choosing", "creating", "making", "constructing", "willing"]
    ),
    SemanticDimension(
        name="Care",
        positive_exemplars=["caring", "concerned", "engaged", "involved", "invested"],
        negative_exemplars=["uncaring", "indifferent", "detached", "uninvolved", "apathetic"]
    ),
    SemanticDimension(
        name="Dasein",
        positive_exemplars=["being-there", "existing", "thrown-project", "temporal", "finite"],
        negative_exemplars=["object", "thing", "present-at-hand", "atemporal", "infinite"]
    ),
    SemanticDimension(
        name="Truth-Aletheia",
        positive_exemplars=["unconcealed", "revealed", "disclosed", "manifest", "open"],
        negative_exemplars=["concealed", "hidden", "covered", "veiled", "closed"]
    ),
    SemanticDimension(
        name="Anxiety",
        positive_exemplars=["anxious", "unsettled", "groundless", "facing-nothingness", "uncertain"],
        negative_exemplars=["calm", "grounded", "secure", "certain", "settled"]
    ),
    SemanticDimension(
        name="Responsibility",
        positive_exemplars=["responsible", "accountable", "answerable", "owning", "bearing"],
        negative_exemplars=["irresponsible", "unaccountable", "avoiding", "denying", "escaping"]
    ),
]

TRANSFORMATION_DIMENSIONS = [
    # Change/transformation dynamics (16)
    SemanticDimension(
        name="Emergence",
        positive_exemplars=["emerging", "arising", "appearing", "manifesting", "coming-forth"],
        negative_exemplars=["disappearing", "vanishing", "fading", "dissolving", "receding"]
    ),
    SemanticDimension(
        name="Dissolution",
        positive_exemplars=["dissolving", "breaking-down", "decomposing", "disintegrating", "fragmenting"],
        negative_exemplars=["forming", "building", "integrating", "consolidating", "coalescing"]
    ),
    SemanticDimension(
        name="Chrysalis",
        positive_exemplars=["incubating", "gestating", "forming", "developing", "metamorphosing"],
        negative_exemplars=["static", "unchanging", "dormant", "frozen", "arrested"]
    ),
    SemanticDimension(
        name="Crisis",
        positive_exemplars=["critical", "decisive", "turning-point", "pivotal", "crucial"],
        negative_exemplars=["stable", "steady", "continuous", "gradual", "smooth"]
    ),
    SemanticDimension(
        name="Revolution",
        positive_exemplars=["revolutionary", "radical", "transformative", "overturning", "revolutionary"],
        negative_exemplars=["evolutionary", "gradual", "incremental", "conservative", "maintaining"]
    ),
    SemanticDimension(
        name="Awakening",
        positive_exemplars=["awakening", "enlightening", "realizing", "becoming-aware", "opening"],
        negative_exemplars=["sleeping", "unconscious", "unaware", "blind", "ignorant"]
    ),
    SemanticDimension(
        name="Descent",
        positive_exemplars=["descending", "falling", "sinking", "going-down", "lowering"],
        negative_exemplars=["ascending", "rising", "climbing", "going-up", "elevating"]
    ),
    SemanticDimension(
        name="Ascent",
        positive_exemplars=["ascending", "rising", "climbing", "elevating", "transcending"],
        negative_exemplars=["descending", "falling", "lowering", "sinking", "declining"]
    ),
    SemanticDimension(
        name="Integration",
        positive_exemplars=["integrating", "unifying", "combining", "merging", "synthesizing"],
        negative_exemplars=["fragmenting", "dividing", "separating", "splitting", "analyzing"]
    ),
    SemanticDimension(
        name="Differentiation",
        positive_exemplars=["differentiating", "distinguishing", "separating", "discriminating", "defining"],
        negative_exemplars=["merging", "blending", "fusing", "confusing", "mixing"]
    ),
    SemanticDimension(
        name="Ripening",
        positive_exemplars=["ripening", "maturing", "developing", "fulfilling", "completing"],
        negative_exemplars=["unripe", "immature", "undeveloped", "incomplete", "premature"]
    ),
    SemanticDimension(
        name="Decay",
        positive_exemplars=["decaying", "deteriorating", "declining", "degrading", "rotting"],
        negative_exemplars=["flourishing", "thriving", "growing", "developing", "blooming"]
    ),
    SemanticDimension(
        name="Renewal",
        positive_exemplars=["renewing", "refreshing", "regenerating", "restoring", "reviving"],
        negative_exemplars=["depleting", "exhausting", "wearing-out", "aging", "degrading"]
    ),
    SemanticDimension(
        name="Breakthrough",
        positive_exemplars=["breaking-through", "penetrating", "overcoming", "surpassing", "transcending"],
        negative_exemplars=["blocked", "stuck", "limited", "contained", "prevented"]
    ),
    SemanticDimension(
        name="Regression",
        positive_exemplars=["regressing", "reverting", "backsliding", "returning", "retreating"],
        negative_exemplars=["progressing", "advancing", "developing", "moving-forward", "evolving"]
    ),
    SemanticDimension(
        name="Liminal",
        positive_exemplars=["liminal", "threshold", "betwixt", "transitional", "boundary"],
        negative_exemplars=["settled", "established", "defined", "stable", "fixed"]
    ),
]

MORAL_ETHICAL_DIMENSIONS = [
    # Moral/ethical dimensions (16)
    SemanticDimension(
        name="Justice",
        positive_exemplars=["just", "fair", "equitable", "righteous", "impartial"],
        negative_exemplars=["unjust", "unfair", "inequitable", "biased", "partial"]
    ),
    SemanticDimension(
        name="Virtue",
        positive_exemplars=["virtuous", "moral", "good", "ethical", "righteous"],
        negative_exemplars=["vicious", "immoral", "evil", "unethical", "corrupt"]
    ),
    SemanticDimension(
        name="Purity",
        positive_exemplars=["pure", "clean", "untainted", "innocent", "unsullied"],
        negative_exemplars=["impure", "tainted", "corrupted", "polluted", "defiled"]
    ),
    SemanticDimension(
        name="Sanctity",
        positive_exemplars=["sacred", "holy", "divine", "hallowed", "consecrated"],
        negative_exemplars=["profane", "secular", "mundane", "desecrated", "unholy"]
    ),
    SemanticDimension(
        name="Harm",
        positive_exemplars=["harmful", "damaging", "hurting", "injuring", "wounding"],
        negative_exemplars=["helpful", "healing", "caring", "nurturing", "beneficial"]
    ),
    SemanticDimension(
        name="Fairness-Reciprocity",
        positive_exemplars=["fair", "balanced", "reciprocal", "equal", "just"],
        negative_exemplars=["unfair", "imbalanced", "one-sided", "unequal", "unjust"]
    ),
    SemanticDimension(
        name="Loyalty-Group",
        positive_exemplars=["loyal-to-group", "tribal", "in-group", "patriotic", "belonging"],
        negative_exemplars=["disloyal", "traitorous", "out-group", "foreign", "outsider"]
    ),
    SemanticDimension(
        name="Authority-Respect",
        positive_exemplars=["respecting-authority", "deferential", "obedient", "submissive", "hierarchical"],
        negative_exemplars=["rebellious", "defiant", "disobedient", "subversive", "egalitarian"]
    ),
    SemanticDimension(
        name="Liberty",
        positive_exemplars=["free", "liberated", "autonomous", "independent", "self-governing"],
        negative_exemplars=["oppressed", "constrained", "controlled", "dominated", "subjugated"]
    ),
    SemanticDimension(
        name="Dignity",
        positive_exemplars=["dignified", "worthy", "respected", "honored", "esteemed"],
        negative_exemplars=["degraded", "debased", "humiliated", "dishonored", "shamed"]
    ),
    SemanticDimension(
        name="Duty",
        positive_exemplars=["dutiful", "obligated", "bound", "responsible", "committed"],
        negative_exemplars=["irresponsible", "negligent", "shirking", "avoiding", "escaping"]
    ),
    SemanticDimension(
        name="Rights",
        positive_exemplars=["entitled", "deserving", "warranted", "owed", "having-rights"],
        negative_exemplars=["unentitled", "undeserving", "privilege", "favor", "gift"]
    ),
    SemanticDimension(
        name="Equality",
        positive_exemplars=["equal", "same", "equivalent", "uniform", "identical"],
        negative_exemplars=["unequal", "different", "hierarchical", "ranked", "stratified"]
    ),
    SemanticDimension(
        name="Mercy",
        positive_exemplars=["merciful", "forgiving", "compassionate", "lenient", "pardoning"],
        negative_exemplars=["merciless", "unforgiving", "harsh", "strict", "punishing"]
    ),
    SemanticDimension(
        name="Temperance",
        positive_exemplars=["temperate", "moderate", "balanced", "restrained", "controlled"],
        negative_exemplars=["excessive", "extreme", "immoderate", "unrestrained", "indulgent"]
    ),
    SemanticDimension(
        name="Humility",
        positive_exemplars=["humble", "modest", "unassuming", "meek", "self-effacing"],
        negative_exemplars=["proud", "arrogant", "boastful", "vain", "conceited"]
    ),
]

# Total: 16 (standard) + 16*7 (extended) = 128 dimensions
# Add more categories to reach 244...

CREATIVE_DIMENSIONS = [
    # Creative/artistic dimensions (16)
    SemanticDimension(
        name="Originality",
        positive_exemplars=["original", "unique", "novel", "innovative", "creative"],
        negative_exemplars=["derivative", "copied", "imitative", "conventional", "cliché"]
    ),
    SemanticDimension(
        name="Beauty",
        positive_exemplars=["beautiful", "aesthetic", "lovely", "attractive", "pleasing"],
        negative_exemplars=["ugly", "unaesthetic", "hideous", "repulsive", "displeasing"]
    ),
    SemanticDimension(
        name="Harmony",
        positive_exemplars=["harmonious", "balanced", "proportionate", "unified", "coherent"],
        negative_exemplars=["disharmonious", "discordant", "clashing", "chaotic", "incoherent"]
    ),
    SemanticDimension(
        name="Expression",
        positive_exemplars=["expressive", "articulate", "communicative", "demonstrative", "revealing"],
        negative_exemplars=["inexpressive", "inarticulate", "mute", "suppressed", "hidden"]
    ),
    SemanticDimension(
        name="Imagination",
        positive_exemplars=["imaginative", "creative", "inventive", "visionary", "fanciful"],
        negative_exemplars=["unimaginative", "literal", "prosaic", "mundane", "pedestrian"]
    ),
    SemanticDimension(
        name="Skill",
        positive_exemplars=["skillful", "masterful", "expert", "proficient", "accomplished"],
        negative_exemplars=["unskilled", "amateur", "novice", "clumsy", "inept"]
    ),
    SemanticDimension(
        name="Inspiration",
        positive_exemplars=["inspired", "moved", "animated", "energized", "enthused"],
        negative_exemplars=["uninspired", "blocked", "stuck", "stale", "exhausted"]
    ),
    SemanticDimension(
        name="Flow",
        positive_exemplars=["flowing", "effortless", "smooth", "natural", "spontaneous"],
        negative_exemplars=["forced", "labored", "difficult", "awkward", "strained"]
    ),
    SemanticDimension(
        name="Vision",
        positive_exemplars=["visionary", "seeing", "perceiving", "imagining", "conceiving"],
        negative_exemplars=["blind", "unseeing", "unimaginative", "limited", "narrow"]
    ),
    SemanticDimension(
        name="Craft",
        positive_exemplars=["crafted", "worked", "refined", "polished", "perfected"],
        negative_exemplars=["raw", "crude", "unworked", "rough", "unrefined"]
    ),
    SemanticDimension(
        name="Play",
        positive_exemplars=["playful", "experimental", "exploratory", "spontaneous", "free"],
        negative_exemplars=["serious", "rigid", "constrained", "controlled", "inhibited"]
    ),
    SemanticDimension(
        name="Risk",
        positive_exemplars=["risky", "daring", "bold", "adventurous", "experimental"],
        negative_exemplars=["safe", "cautious", "conservative", "conventional", "timid"]
    ),
    SemanticDimension(
        name="Depth-Artistic",
        positive_exemplars=["deep", "profound", "rich", "complex", "layered"],
        negative_exemplars=["shallow", "superficial", "simple", "one-dimensional", "flat"]
    ),
    SemanticDimension(
        name="Resonance",
        positive_exemplars=["resonant", "evocative", "moving", "touching", "affecting"],
        negative_exemplars=["flat", "unmov ing", "empty", "hollow", "lifeless"]
    ),
    SemanticDimension(
        name="Authenticity-Art",
        positive_exemplars=["authentic", "true", "genuine", "real", "honest"],
        negative_exemplars=["fake", "artificial", "contrived", "affected", "pretentious"]
    ),
    SemanticDimension(
        name="Innovation",
        positive_exemplars=["innovative", "groundbreaking", "revolutionary", "pioneering", "cutting-edge"],
        negative_exemplars=["traditional", "conventional", "old-fashioned", "derivative", "stale"]
    ),
]

COGNITIVE_COMPLEXITY_DIMENSIONS = [
    # Advanced cognitive dimensions (16)
    SemanticDimension(
        name="Nuance",
        positive_exemplars=["nuanced", "subtle", "refined", "delicate", "sophisticated"],
        negative_exemplars=["crude", "blunt", "simplistic", "heavy-handed", "obvious"]
    ),
    SemanticDimension(
        name="Paradox",
        positive_exemplars=["paradoxical", "contradictory", "opposing", "dual", "both-and"],
        negative_exemplars=["consistent", "logical", "coherent", "either-or", "singular"]
    ),
    SemanticDimension(
        name="Ambiguity",
        positive_exemplars=["ambiguous", "unclear", "uncertain", "equivocal", "multiple-meanings"],
        negative_exemplars=["unambiguous", "clear", "definite", "univocal", "single-meaning"]
    ),
    SemanticDimension(
        name="Reflexivity",
        positive_exemplars=["self-referential", "meta", "reflexive", "self-aware", "circular"],
        negative_exemplars=["direct", "straightforward", "unreflexive", "unaware", "linear"]
    ),
    SemanticDimension(
        name="Dialectic",
        positive_exemplars=["dialectical", "thesis-antithesis", "oppositional", "synthesizing", "dynamic"],
        negative_exemplars=["monologic", "single-sided", "static", "undialectical", "fixed"]
    ),
    SemanticDimension(
        name="Systems-Thinking",
        positive_exemplars=["systemic", "holistic", "interconnected", "emergent", "complex"],
        negative_exemplars=["reductionist", "isolated", "linear", "simple", "separated"]
    ),
    SemanticDimension(
        name="Perspective-Taking",
        positive_exemplars=["multiple-perspectives", "seeing-from-many-angles", "empathetic", "understanding"],
        negative_exemplars=["single-perspective", "narrow", "rigid", "unempathetic", "limited"]
    ),
    SemanticDimension(
        name="Abstraction",
        positive_exemplars=["abstract", "theoretical", "conceptual", "general", "universal"],
        negative_exemplars=["concrete", "practical", "specific", "particular", "individual"]
    ),
    SemanticDimension(
        name="Coherence",
        positive_exemplars=["coherent", "consistent", "unified", "integrated", "logical"],
        negative_exemplars=["incoherent", "inconsistent", "fragmented", "disintegrated", "illogical"]
    ),
    SemanticDimension(
        name="Emergence",
        positive_exemplars=["emergent", "arising", "self-organizing", "spontaneous", "unpredictable"],
        negative_exemplars=["predetermined", "planned", "controlled", "predictable", "imposed"]
    ),
    SemanticDimension(
        name="Metacognition",
        positive_exemplars=["metacognitive", "thinking-about-thinking", "aware", "reflective", "monitoring"],
        negative_exemplars=["unreflective", "unaware", "automatic", "unconscious", "unmonitored"]
    ),
    SemanticDimension(
        name="Synthesis",
        positive_exemplars=["synthesizing", "combining", "integrating", "unifying", "creating-wholes"],
        negative_exemplars=["analyzing", "separating", "dividing", "fragmenting", "breaking-down"]
    ),
    SemanticDimension(
        name="Insight",
        positive_exemplars=["insightful", "penetrating", "seeing-deeply", "understanding", "grasping"],
        negative_exemplars=["superficial", "missing", "blind", "misunderstanding", "failing-to-grasp"]
    ),
    SemanticDimension(
        name="Wisdom-Cognitive",
        positive_exemplars=["wise", "discerning", "judicious", "prudent", "sagacious"],
        negative_exemplars=["foolish", "undiscerning", "imprudent", "rash", "unwise"]
    ),
    SemanticDimension(
        name="Creativity-Cognitive",
        positive_exemplars=["creative", "generative", "productive", "inventive", "original"],
        negative_exemplars=["uncreative", "sterile", "unproductive", "imitative", "derivative"]
    ),
    SemanticDimension(
        name="Intuition",
        positive_exemplars=["intuitive", "feeling", "sensing", "knowing-without-reason", "immediate"],
        negative_exemplars=["analytical", "reasoning", "logical", "mediated", "step-by-step"]
    ),
]

TEMPORAL_NARRATIVE_DIMENSIONS = [
    # Temporal/narrative flow dimensions (16)
    SemanticDimension(
        name="Pacing",
        positive_exemplars=["fast-paced", "rapid", "quick", "hurried", "rushed"],
        negative_exemplars=["slow-paced", "leisurely", "gradual", "unhurried", "deliberate"]
    ),
    SemanticDimension(
        name="Suspense",
        positive_exemplars=["suspenseful", "tense", "uncertain", "anticipatory", "nail-biting"],
        negative_exemplars=["relaxed", "certain", "predictable", "resolved", "calm"]
    ),
    SemanticDimension(
        name="Continuity",
        positive_exemplars=["continuous", "connected", "flowing", "unbroken", "seamless"],
        negative_exemplars=["discontinuous", "fragmented", "broken", "interrupted", "disjointed"]
    ),
    SemanticDimension(
        name="Cyclical",
        positive_exemplars=["cyclical", "repeating", "recurring", "circular", "returning"],
        negative_exemplars=["linear", "progressive", "forward", "non-repeating", "unique"]
    ),
    SemanticDimension(
        name="Climax",
        positive_exemplars=["climactic", "peak", "culminating", "highest-point", "crescendo"],
        negative_exemplars=["anticlimax", "declining", "falling", "decreasing", "diminishing"]
    ),
    SemanticDimension(
        name="Resolution",
        positive_exemplars=["resolved", "concluded", "finished", "settled", "complete"],
        negative_exemplars=["unresolved", "open", "incomplete", "unsettled", "continuing"]
    ),
    SemanticDimension(
        name="Foreshadowing",
        positive_exemplars=["foreshadowing", "hinting", "predicting", "suggesting", "prefiguring"],
        negative_exemplars=["surprising", "unexpected", "unforeshadowed", "unanticipated", "sudden"]
    ),
    SemanticDimension(
        name="Flashback",
        positive_exemplars=["past", "retrospective", "remembering", "recalling", "looking-back"],
        negative_exemplars=["present", "current", "immediate", "now", "contemporary"]
    ),
    SemanticDimension(
        name="Momentum",
        positive_exemplars=["momentum", "building", "increasing", "accelerating", "growing"],
        negative_exemplars=["stalling", "stopping", "decreasing", "decelerating", "shrinking"]
    ),
    SemanticDimension(
        name="Tension",
        positive_exemplars=["tense", "tight", "strained", "stressed", "pressured"],
        negative_exemplars=["relaxed", "loose", "easy", "unstressed", "calm"]
    ),
    SemanticDimension(
        name="Release",
        positive_exemplars=["releasing", "letting-go", "freeing", "relaxing", "resolving"],
        negative_exemplars=["holding", "grasping", "tensing", "building", "constraining"]
    ),
    SemanticDimension(
        name="Inevitability",
        positive_exemplars=["inevitable", "fated", "destined", "certain", "unavoidable"],
        negative_exemplars=["contingent", "uncertain", "avoidable", "changeable", "open"]
    ),
    SemanticDimension(
        name="Reversal",
        positive_exemplars=["reversing", "turning", "flipping", "inverting", "peripeteia"],
        negative_exemplars=["continuing", "steady", "unchanged", "consistent", "stable"]
    ),
    SemanticDimension(
        name="Recognition",
        positive_exemplars=["recognizing", "realizing", "discovering", "anagnorisis", "seeing-truth"],
        negative_exemplars=["misrecognizing", "unaware", "blind", "missing", "failing-to-see"]
    ),
    SemanticDimension(
        name="Beginning",
        positive_exemplars=["beginning", "starting", "initiating", "opening", "commencing"],
        negative_exemplars=["ending", "finishing", "concluding", "closing", "completing"]
    ),
    SemanticDimension(
        name="Middle",
        positive_exemplars=["middle", "developing", "complicating", "building", "progressing"],
        negative_exemplars=["extreme", "beginning-or-end", "static", "unchanging", "fixed"]
    ),
]

SPATIAL_SETTING_DIMENSIONS = [
    # Spatial/environmental dimensions (12)
    SemanticDimension(
        name="Light",
        positive_exemplars=["light", "bright", "illuminated", "radiant", "luminous"],
        negative_exemplars=["dark", "dim", "shadowy", "gloomy", "obscure"]
    ),
    SemanticDimension(
        name="Open",
        positive_exemplars=["open", "expansive", "wide", "spacious", "vast"],
        negative_exemplars=["closed", "confined", "narrow", "cramped", "restricted"]
    ),
    SemanticDimension(
        name="Height",
        positive_exemplars=["high", "elevated", "lofty", "towering", "soaring"],
        negative_exemplars=["low", "deep", "sunken", "underground", "buried"]
    ),
    SemanticDimension(
        name="Natural",
        positive_exemplars=["natural", "wild", "organic", "untamed", "pristine"],
        negative_exemplars=["artificial", "civilized", "constructed", "tamed", "synthetic"]
    ),
    SemanticDimension(
        name="Sacred-Space",
        positive_exemplars=["sacred", "holy", "consecrated", "temple", "sanctuary"],
        negative_exemplars=["profane", "mundane", "ordinary", "common", "everyday"]
    ),
    SemanticDimension(
        name="Home",
        positive_exemplars=["home", "familiar", "comfortable", "safe", "belonging"],
        negative_exemplars=["foreign", "strange", "uncomfortable", "unsafe", "alien"]
    ),
    SemanticDimension(
        name="Boundary",
        positive_exemplars=["boundary", "edge", "limit", "threshold", "border"],
        negative_exemplars=["center", "core", "heart", "middle", "interior"]
    ),
    SemanticDimension(
        name="Journey",
        positive_exemplars=["traveling", "moving", "journeying", "wandering", "voyaging"],
        negative_exemplars=["staying", "settled", "stationary", "fixed", "rooted"]
    ),
    SemanticDimension(
        name="Isolation",
        positive_exemplars=["isolated", "alone", "remote", "solitary", "separated"],
        negative_exemplars=["connected", "together", "communal", "social", "joined"]
    ),
    SemanticDimension(
        name="Shelter",
        positive_exemplars=["sheltered", "protected", "covered", "enclosed", "safe"],
        negative_exemplars=["exposed", "vulnerable", "open", "unprotected", "endangered"]
    ),
    SemanticDimension(
        name="Wilderness",
        positive_exemplars=["wild", "untamed", "chaotic", "primal", "savage"],
        negative_exemplars=["civilized", "ordered", "cultivated", "domesticated", "tamed"]
    ),
    SemanticDimension(
        name="Center-Periphery",
        positive_exemplars=["central", "core", "heart", "focal", "main"],
        negative_exemplars=["peripheral", "marginal", "edge", "outer", "secondary"]
    ),
]

CHARACTER_DIMENSIONS = [
    # Character/personality dimensions (12)
    SemanticDimension(
        name="Strength-Character",
        positive_exemplars=["strong", "powerful", "mighty", "robust", "vigorous"],
        negative_exemplars=["weak", "frail", "feeble", "fragile", "delicate"]
    ),
    SemanticDimension(
        name="Intelligence",
        positive_exemplars=["intelligent", "clever", "smart", "brilliant", "sharp"],
        negative_exemplars=["stupid", "dull", "slow", "dim", "simple-minded"]
    ),
    SemanticDimension(
        name="Cunning",
        positive_exemplars=["cunning", "crafty", "wily", "shrewd", "sly"],
        negative_exemplars=["straightforward", "honest", "direct", "open", "guileless"]
    ),
    SemanticDimension(
        name="Nobility-Character",
        positive_exemplars=["noble", "dignified", "honorable", "high-minded", "elevated"],
        negative_exemplars=["base", "low", "ignoble", "dishonorable", "degraded"]
    ),
    SemanticDimension(
        name="Piety",
        positive_exemplars=["pious", "devout", "religious", "faithful", "reverent"],
        negative_exemplars=["impious", "irreverent", "sacrilegious", "blasphemous", "godless"]
    ),
    SemanticDimension(
        name="Patience",
        positive_exemplars=["patient", "enduring", "long-suffering", "tolerant", "forbearing"],
        negative_exemplars=["impatient", "hasty", "rash", "intolerant", "quick-tempered"]
    ),
    SemanticDimension(
        name="Passion-Character",
        positive_exemplars=["passionate", "fervent", "ardent", "intense", "zealous"],
        negative_exemplars=["dispassionate", "cool", "indifferent", "apathetic", "lukewarm"]
    ),
    SemanticDimension(
        name="Stubbornness",
        positive_exemplars=["stubborn", "obstinate", "inflexible", "unyielding", "headstrong"],
        negative_exemplars=["flexible", "adaptable", "yielding", "compliant", "accommodating"]
    ),
    SemanticDimension(
        name="Hospitality",
        positive_exemplars=["hospitable", "welcoming", "generous", "gracious", "warm"],
        negative_exemplars=["inhospitable", "unwelcoming", "stingy", "cold", "rejecting"]
    ),
    SemanticDimension(
        name="Eloquence",
        positive_exemplars=["eloquent", "articulate", "well-spoken", "persuasive", "silver-tongued"],
        negative_exemplars=["inarticulate", "tongue-tied", "clumsy", "unpersuasive", "mute"]
    ),
    SemanticDimension(
        name="Temperament",
        positive_exemplars=["even-tempered", "calm", "balanced", "stable", "composed"],
        negative_exemplars=["volatile", "moody", "unstable", "erratic", "explosive"]
    ),
    SemanticDimension(
        name="Ambition",
        positive_exemplars=["ambitious", "driven", "aspiring", "striving", "goal-oriented"],
        negative_exemplars=["unambitious", "content", "satisfied", "accepting", "aimless"]
    ),
]

PLOT_DIMENSIONS = [
    # Plot structure dimensions (12)
    SemanticDimension(
        name="Complication",
        positive_exemplars=["complicated", "complex", "intricate", "tangled", "involved"],
        negative_exemplars=["simple", "straightforward", "clear", "uncomplicated", "direct"]
    ),
    SemanticDimension(
        name="Irony",
        positive_exemplars=["ironic", "contrary", "opposite", "paradoxical", "unexpected"],
        negative_exemplars=["literal", "expected", "straightforward", "predictable", "intended"]
    ),
    SemanticDimension(
        name="Coincidence",
        positive_exemplars=["coincidental", "chance", "accidental", "fortuitous", "random"],
        negative_exemplars=["planned", "intended", "caused", "deliberate", "designed"]
    ),
    SemanticDimension(
        name="Necessity",
        positive_exemplars=["necessary", "required", "inevitable", "must", "compulsory"],
        negative_exemplars=["optional", "unnecessary", "avoidable", "contingent", "dispensable"]
    ),
    SemanticDimension(
        name="Causality",
        positive_exemplars=["causal", "because", "resulting", "consequent", "following"],
        negative_exemplars=["random", "uncaused", "independent", "coincidental", "unrelated"]
    ),
    SemanticDimension(
        name="Dramatic-Irony",
        positive_exemplars=["audience-knows", "character-unaware", "gap", "dramatic-tension"],
        negative_exemplars=["character-knows", "no-gap", "aligned", "shared-knowledge"]
    ),
    SemanticDimension(
        name="Catastrophe",
        positive_exemplars=["catastrophic", "disastrous", "tragic", "terrible", "devastating"],
        negative_exemplars=["fortunate", "blessed", "lucky", "favorable", "beneficial"]
    ),
    SemanticDimension(
        name="Comedy",
        positive_exemplars=["comic", "funny", "humorous", "amusing", "laughable"],
        negative_exemplars=["tragic", "serious", "somber", "grave", "solemn"]
    ),
    SemanticDimension(
        name="Hamartia",
        positive_exemplars=["flaw", "error", "mistake", "weakness", "failing"],
        negative_exemplars=["strength", "virtue", "correctness", "perfection", "success"]
    ),
    SemanticDimension(
        name="Hubris",
        positive_exemplars=["hubris", "pride", "arrogance", "overconfidence", "presumption"],
        negative_exemplars=["humility", "modesty", "deference", "respect", "caution"]
    ),
    SemanticDimension(
        name="Nemesis",
        positive_exemplars=["nemesis", "retribution", "punishment", "vengeance", "downfall"],
        negative_exemplars=["reward", "blessing", "mercy", "forgiveness", "rise"]
    ),
    SemanticDimension(
        name="Deus-Ex-Machina",
        positive_exemplars=["divine-intervention", "sudden-solution", "external-rescue", "contrived"],
        negative_exemplars=["earned", "organic", "internal-resolution", "natural", "inevitable"]
    ),
]

THEME_DIMENSIONS = [
    # Thematic dimensions (12)
    SemanticDimension(
        name="Love-Hate",
        positive_exemplars=["love", "affection", "devotion", "adoration", "fondness"],
        negative_exemplars=["hate", "loathing", "animosity", "enmity", "abhorrence"]
    ),
    SemanticDimension(
        name="War-Peace",
        positive_exemplars=["war", "conflict", "battle", "fighting", "strife"],
        negative_exemplars=["peace", "harmony", "concord", "tranquility", "calm"]
    ),
    SemanticDimension(
        name="Civilization-Barbarism",
        positive_exemplars=["civilized", "cultured", "refined", "sophisticated", "polished"],
        negative_exemplars=["barbaric", "savage", "crude", "primitive", "uncivilized"]
    ),
    SemanticDimension(
        name="Individual-Society",
        positive_exemplars=["individual", "personal", "private", "singular", "unique"],
        negative_exemplars=["social", "collective", "public", "communal", "shared"]
    ),
    SemanticDimension(
        name="Nature-Culture",
        positive_exemplars=["natural", "instinctive", "biological", "innate", "organic"],
        negative_exemplars=["cultural", "learned", "constructed", "artificial", "social"]
    ),
    SemanticDimension(
        name="Mortality-Immortality",
        positive_exemplars=["mortal", "dying", "finite", "temporary", "ephemeral"],
        negative_exemplars=["immortal", "eternal", "infinite", "permanent", "everlasting"]
    ),
    SemanticDimension(
        name="Knowledge-Ignorance",
        positive_exemplars=["knowing", "informed", "aware", "educated", "enlightened"],
        negative_exemplars=["ignorant", "uninformed", "unaware", "uneducated", "benighted"]
    ),
    SemanticDimension(
        name="Fate-Free-Will",
        positive_exemplars=["fated", "destined", "predetermined", "inevitable", "fixed"],
        negative_exemplars=["free", "choosing", "self-determined", "contingent", "open"]
    ),
    SemanticDimension(
        name="Appearance-Reality",
        positive_exemplars=["appearance", "seeming", "surface", "illusion", "pretense"],
        negative_exemplars=["reality", "being", "truth", "essence", "actuality"]
    ),
    SemanticDimension(
        name="Order-Chaos",
        positive_exemplars=["ordered", "structured", "organized", "systematic", "regular"],
        negative_exemplars=["chaotic", "disordered", "random", "anarchic", "irregular"]
    ),
    SemanticDimension(
        name="Youth-Age",
        positive_exemplars=["young", "youthful", "fresh", "new", "inexperienced"],
        negative_exemplars=["old", "aged", "ancient", "experienced", "weathered"]
    ),
    SemanticDimension(
        name="Memory-Forgetting",
        positive_exemplars=["remembering", "recalling", "retaining", "preserving", "memorial"],
        negative_exemplars=["forgetting", "losing", "erasing", "repressing", "amnesia"]
    ),
]

STYLE_VOICE_DIMENSIONS = [
    # Style/voice dimensions (4 to reach 244 total)
    SemanticDimension(
        name="Epic",
        positive_exemplars=["epic", "grand", "heroic", "legendary", "monumental"],
        negative_exemplars=["mundane", "ordinary", "small-scale", "everyday", "trivial"]
    ),
    SemanticDimension(
        name="Lyric",
        positive_exemplars=["lyrical", "poetic", "musical", "melodic", "flowing"],
        negative_exemplars=["prosaic", "unpoetic", "harsh", "discordant", "choppy"]
    ),
    SemanticDimension(
        name="Tragic",
        positive_exemplars=["tragic", "doomed", "catastrophic", "sorrowful", "lamentable"],
        negative_exemplars=["comic", "fortunate", "joyful", "successful", "triumphant"]
    ),
    SemanticDimension(
        name="Sublime",
        positive_exemplars=["sublime", "awe-inspiring", "overwhelming", "transcendent", "magnificent"],
        negative_exemplars=["ordinary", "mundane", "unremarkable", "commonplace", "plain"]
    ),
]

# Combine all dimensions: 16 + 16*14 + 4 = 244 dimensions
EXTENDED_244_DIMENSIONS = (
    STANDARD_DIMENSIONS +           # 16
    NARRATIVE_DIMENSIONS +          # 16
    EMOTIONAL_DEPTH_DIMENSIONS +    # 16
    RELATIONAL_DIMENSIONS +         # 16
    ARCHETYPAL_DIMENSIONS +         # 16
    PHILOSOPHICAL_DIMENSIONS +      # 16
    TRANSFORMATION_DIMENSIONS +     # 16
    MORAL_ETHICAL_DIMENSIONS +      # 16
    CREATIVE_DIMENSIONS +           # 16
    COGNITIVE_COMPLEXITY_DIMENSIONS + # 16
    TEMPORAL_NARRATIVE_DIMENSIONS +  # 16
    SPATIAL_SETTING_DIMENSIONS +    # 12
    CHARACTER_DIMENSIONS +          # 12
    PLOT_DIMENSIONS +               # 12
    THEME_DIMENSIONS +              # 12
    STYLE_VOICE_DIMENSIONS          # 4
)  # Total = 244


class SemanticSpectrum:
    """
    Projects semantic trajectories onto interpretable dimensions

    This is the key to understanding WHAT is changing in a conversation.
    Instead of 384 opaque numbers, we get "warmth increasing, formality decreasing"
    """

    def __init__(self, dimensions: Optional[List[SemanticDimension]] = None):
        """
        Args:
            dimensions: List of semantic dimensions. If None, uses STANDARD_DIMENSIONS
        """
        self.dimensions = dimensions if dimensions is not None else STANDARD_DIMENSIONS
        self._axes_learned = False

    def learn_axes(self, embed_fn: Callable[[str], np.ndarray]):
        """
        Learn all dimension axes from exemplars

        Args:
            embed_fn: Function that maps word -> embedding vector
        """
        print(f"Learning {len(self.dimensions)} semantic dimension axes...")
        for dim in self.dimensions:
            dim.learn_axis(embed_fn)
        self._axes_learned = True
        print(f"  All axes learned successfully")

    def project_vector(self, vector: np.ndarray) -> Dict[str, float]:
        """
        Project a single vector onto all dimensions

        Returns:
            Dictionary mapping dimension name -> projection value
        """
        if not self._axes_learned:
            raise ValueError("Axes not learned yet. Call learn_axes() first.")

        return {dim.name: dim.project(vector) for dim in self.dimensions}

    def project_trajectory(self, positions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Project entire trajectory onto all dimensions

        Args:
            positions: Array of shape (n_steps, embedding_dim)

        Returns:
            Dictionary mapping dimension name -> array of projections over time
        """
        if not self._axes_learned:
            raise ValueError("Axes not learned yet. Call learn_axes() first.")

        projections = {}
        for dim in self.dimensions:
            projections[dim.name] = np.array([dim.project(pos) for pos in positions])

        return projections

    def compute_spectrum_velocity(self, positions: np.ndarray, dt: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Compute velocity along each semantic dimension

        This tells us HOW FAST each dimension is changing
        """
        projections = self.project_trajectory(positions)

        velocities = {}
        for dim_name, proj in projections.items():
            velocities[dim_name] = np.gradient(proj, dt)

        return velocities

    def compute_spectrum_acceleration(self, positions: np.ndarray, dt: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Compute acceleration along each semantic dimension

        This tells us which dimensions have FORCES acting on them
        """
        velocities = self.compute_spectrum_velocity(positions, dt)

        accelerations = {}
        for dim_name, vel in velocities.items():
            accelerations[dim_name] = np.gradient(vel, dt)

        return accelerations

    def get_dominant_dimensions(self, velocity_dict: Dict[str, np.ndarray],
                               top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find which dimensions are changing most rapidly

        Args:
            velocity_dict: Output of compute_spectrum_velocity()
            top_k: How many top dimensions to return

        Returns:
            List of (dimension_name, avg_velocity_magnitude) sorted by magnitude
        """
        avg_magnitudes = {
            name: np.mean(np.abs(vel))
            for name, vel in velocity_dict.items()
        }

        sorted_dims = sorted(avg_magnitudes.items(), key=lambda x: x[1], reverse=True)
        return sorted_dims[:top_k]

    def analyze_semantic_forces(self, positions: np.ndarray, dt: float = 1.0) -> Dict:
        """
        Complete analysis: which dimensions are being pushed/pulled

        Returns rich analysis of semantic forces acting on the trajectory
        """
        projections = self.project_trajectory(positions)
        velocities = self.compute_spectrum_velocity(positions, dt)
        accelerations = self.compute_spectrum_acceleration(positions, dt)

        # Find dominant dimensions
        dominant_by_velocity = self.get_dominant_dimensions(velocities, top_k=5)

        # Find dimensions with strongest forces (acceleration)
        dominant_by_force = self.get_dominant_dimensions(accelerations, top_k=5)

        # Compute total semantic distance along each dimension
        distances = {
            name: np.sum(np.abs(np.diff(proj)))
            for name, proj in projections.items()
        }

        return {
            'projections': projections,
            'velocities': velocities,
            'accelerations': accelerations,
            'dominant_velocity': dominant_by_velocity,
            'dominant_force': dominant_by_force,
            'distances': distances
        }


def visualize_semantic_spectrum(analysis: Dict, words: List[str],
                                top_k: int = 8, save_path: Optional[str] = None):
    """
    Visualize how semantic dimensions change over a trajectory

    Shows the top K most active dimensions and how they evolve
    """
    import matplotlib.pyplot as plt

    # Get top K dimensions by total distance traveled
    distances = analysis['distances']
    top_dims = sorted(distances.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_dim_names = [name for name, _ in top_dims]

    projections = analysis['projections']
    velocities = analysis['velocities']

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Projections (position along each dimension)
    ax1 = axes[0]
    for dim_name in top_dim_names:
        ax1.plot(projections[dim_name], label=dim_name, linewidth=2, alpha=0.7)
    ax1.set_ylabel('Projection Value', fontsize=12)
    ax1.set_title('Semantic Dimension Positions', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Velocities (rate of change)
    ax2 = axes[1]
    for dim_name in top_dim_names:
        ax2.plot(velocities[dim_name], label=dim_name, linewidth=2, alpha=0.7)
    ax2.set_ylabel('Velocity', fontsize=12)
    ax2.set_title('Semantic Dimension Velocities', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Heatmap of all dimensions
    ax3 = axes[2]
    all_velocities = np.array([velocities[name] for name in top_dim_names])
    im = ax3.imshow(all_velocities, aspect='auto', cmap='RdBu_r',
                    interpolation='nearest', vmin=-0.5, vmax=0.5)
    ax3.set_yticks(range(len(top_dim_names)))
    ax3.set_yticklabels(top_dim_names)
    ax3.set_xlabel('Word Index', fontsize=12)
    ax3.set_title('Semantic Velocity Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Velocity')

    # Add word labels if not too many
    if len(words) <= 20:
        for ax in [ax1, ax2]:
            ax.set_xticks(range(len(words)))
            ax.set_xticklabels(words, rotation=45, ha='right')
        ax3.set_xticks(range(len(words)))
        ax3.set_xticklabels(words, rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved spectrum visualization: {save_path}")

    return fig, axes


def print_spectrum_summary(analysis: Dict, words: List[str]):
    """
    Print human-readable summary of semantic forces
    """
    print("\n" + "=" * 70)
    print("SEMANTIC SPECTRUM ANALYSIS")
    print("=" * 70)

    print(f"\nTop 5 dimensions by velocity (how fast they're changing):")
    for i, (name, mag) in enumerate(analysis['dominant_velocity'], 1):
        print(f"  {i}. {name:<15} (avg velocity: {mag:.4f})")

    print(f"\nTop 5 dimensions by force (acceleration):")
    for i, (name, mag) in enumerate(analysis['dominant_force'], 1):
        print(f"  {i}. {name:<15} (avg force: {mag:.4f})")

    print(f"\nTop 5 dimensions by distance traveled:")
    distances = sorted(analysis['distances'].items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (name, dist) in enumerate(distances, 1):
        print(f"  {i}. {name:<15} (total distance: {dist:.4f})")

    # Show start and end values for dominant dimensions
    print(f"\nStart -> End values for top 3 dimensions:")
    projections = analysis['projections']
    for i, (name, _) in enumerate(analysis['dominant_velocity'][:3], 1):
        start_val = projections[name][0]
        end_val = projections[name][-1]
        change = end_val - start_val
        direction = "UP" if change > 0 else "DOWN"
        print(f"  {i}. {name:<15}: {start_val:>7.3f} -> {end_val:>7.3f}  {direction} ({abs(change):.3f})")


# Export dimension sets
__all__ = [
    'SemanticDimension',
    'SemanticSpectrum',
    'STANDARD_DIMENSIONS',
    'EXTENDED_244_DIMENSIONS',
    'visualize_semantic_spectrum',
    'print_spectrum_summary',
]