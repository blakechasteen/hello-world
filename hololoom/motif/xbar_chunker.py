"""
Universal Grammar Phrase Chunker - X-bar Theory Implementation
==============================================================
Detects hierarchical phrase structure using Chomsky's X-bar theory.

Philosophy:
-----------
Chomsky's insight: ALL phrases in ALL human languages have the same structure:

    XP (Maximal Projection)
     ├─ Specifier
     └─ X' (Intermediate Projection)
         ├─ X (Head) ← Determines category!
         └─ Complement

This is UNIVERSAL across languages. Only parameters vary (head direction, etc.).

Example: "the big red ball"
    NP
     ├─ Spec: Det "the"
     └─ N'
         ├─ Adjunct: AP "big"
         └─ N'
             ├─ Adjunct: AP "red"
             └─ N "ball" ← HEAD

The HEAD determines everything:
- Head = N → It's an NP
- Head = V → It's a VP
- Head = P → It's a PP
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Import spaCy with graceful fallback
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Install with: pip install spacy")


# ============================================================================
# Universal Grammar Categories
# ============================================================================

class Category(Enum):
    """
    Syntactic categories (heads) from Universal Grammar.

    These are universal across all human languages.
    """
    N = "N"    # Noun
    V = "V"    # Verb
    A = "A"    # Adjective
    P = "P"    # Preposition
    D = "D"    # Determiner
    C = "C"    # Complementizer (that, if, whether)
    T = "T"    # Tense/Inflection


# ============================================================================
# X-bar Node Structure
# ============================================================================

@dataclass
class XBarNode:
    """
    Node in X-bar structure.

    Represents one level of projection:
    - X (level=0): Lexical head
    - X' (level=1): Intermediate projection
    - XP (level=2): Maximal projection

    Structure:
        XP
         ├─ Specifier (optional)
         └─ X'
             ├─ X (Head) ← Required!
             └─ Complement (optional)
             └─ Adjuncts (zero or more)

    Attributes:
        category: Syntactic category of head
        level: Projection level (0=X, 1=X', 2=XP)
        head: Head word (string)
        head_token: Original spaCy token (if available)
        specifier: Specifier node (optional)
        complement: Complement node (optional)
        adjuncts: List of adjunct nodes
        features: Syntactic features (tense, agreement, etc.)
    """
    category: Category
    level: int  # 0=X, 1=X', 2=XP
    head: str
    head_token: Optional[object] = None  # spaCy token
    specifier: Optional['XBarNode'] = None
    complement: Optional['XBarNode'] = None
    adjuncts: List['XBarNode'] = field(default_factory=list)
    features: Dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Syntactic label (e.g., NP, N', N)."""
        if self.level == 0:
            return self.category.value  # N, V, A, P, D, C, T
        elif self.level == 1:
            return f"{self.category.value}'"  # N', V', etc.
        else:  # level == 2
            return f"{self.category.value}P"  # NP, VP, AP, PP, DP, CP, TP

    @property
    def span(self) -> Tuple[int, int]:
        """Character span (start, end) of this phrase."""
        if self.head_token:
            # Get span from all children
            start = self.head_token.idx
            end = self.head_token.idx + len(self.head_token.text)

            # Expand to include specifier
            if self.specifier and hasattr(self.specifier, 'span'):
                spec_start, spec_end = self.specifier.span
                start = min(start, spec_start)
                end = max(end, spec_end)

            # Expand to include complement
            if self.complement and hasattr(self.complement, 'span'):
                comp_start, comp_end = self.complement.span
                start = min(start, comp_start)
                end = max(end, comp_end)

            # Expand to include adjuncts
            for adj in self.adjuncts:
                if hasattr(adj, 'span'):
                    adj_start, adj_end = adj.span
                    start = min(start, adj_start)
                    end = max(end, adj_end)

            return (start, end)
        else:
            return (0, len(self.head))

    def __repr__(self) -> str:
        return f"{self.label}(head={self.head!r})"


# ============================================================================
# Universal Grammar Chunker
# ============================================================================

class UniversalGrammarChunker:
    """
    Phrase chunker using Universal Grammar (X-bar theory).

    Detects hierarchical phrase structure based on universal principles:
    1. Every phrase has a head
    2. Phrases project three levels: X → X' → XP
    3. Specifier-Head-Complement relationships
    4. Adjuncts attach at X' level

    This works across ALL human languages (with parameter variation).

    Usage:
        chunker = UniversalGrammarChunker()
        phrases = chunker.chunk("the big red ball")

        for phrase in phrases:
            print(phrase.label, "→", phrase.head)
            # Output: NP → ball
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize Universal Grammar chunker.

        Args:
            spacy_model: spaCy model name (default: en_core_web_sm)
        """
        self.spacy_model = spacy_model
        self.nlp = None

        # Category mapping (spaCy POS → UG categories)
        self.pos_to_category = {
            "NOUN": Category.N,
            "PROPN": Category.N,  # Proper nouns are also N
            "VERB": Category.V,
            "AUX": Category.V,    # Auxiliaries are also V
            "ADJ": Category.A,
            "ADP": Category.P,    # Adpositions (prepositions)
            "DET": Category.D,
            "SCONJ": Category.C,  # Subordinating conjunctions
        }

        # Lazy load spaCy model
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Loaded spaCy model: {spacy_model}")
            except OSError:
                logger.warning(
                    f"spaCy model '{spacy_model}' not found. "
                    f"Run: python -m spacy download {spacy_model}"
                )
                self.nlp = None
        else:
            logger.warning("spaCy not available - chunker will not work")

    def chunk(self, text: str) -> List[XBarNode]:
        """
        Chunk text into X-bar phrase structures.

        Args:
            text: Input text

        Returns:
            List of maximal projections (XPs) found in text
        """
        if not self.nlp:
            logger.error("Cannot chunk: spaCy not loaded")
            return []

        # Parse with spaCy
        doc = self.nlp(text)

        # Build X-bar structures for each sentence
        phrases = []

        for sent in doc.sents:
            # Find main verb (root)
            root = sent.root

            # Build clause structure (CP or TP)
            clause = self._build_clause(root, doc)
            if clause:
                phrases.append(clause)

        logger.info(f"Chunked text into {len(phrases)} phrase(s)")
        return phrases

    # ========================================================================
    # Clause Building (CP, TP)
    # ========================================================================

    def _build_clause(self, verb, doc) -> Optional[XBarNode]:
        """
        Build clause structure (CP → TP → VP).

        Clause hierarchy:
            CP (Complementizer Phrase)
             └─ C'
                 ├─ C: "that" / "if" / null
                 └─ TP (Tense Phrase)
                     ├─ Spec: Subject (NP)
                     └─ T'
                         ├─ T: tense/agreement
                         └─ VP (Verb Phrase)

        Args:
            verb: Main verb token
            doc: spaCy doc

        Returns:
            CP or TP node (maximal projection of clause)
        """
        # Build VP (core predicate)
        vp = self._build_vp(verb, doc)
        if not vp:
            return None

        # Build TP (add subject)
        subject = self._find_subject(verb)
        if subject:
            subject_np = self._build_np(subject, doc)

            # TP = [Spec: Subject] T' [T, VP]
            t_bar = XBarNode(
                category=Category.T,
                level=1,  # T'
                head=verb.text,
                head_token=verb,
                complement=vp
            )

            tp = XBarNode(
                category=Category.T,
                level=2,  # TP
                head=verb.text,
                head_token=verb,
                specifier=subject_np,
                complement=t_bar,
                features={"tense": verb.tag_}
            )
        else:
            # No subject (imperative, etc.) - just return VP as TP
            tp = vp

        # Check for complementizer
        comp = self._find_complementizer(verb)
        if comp:
            # CP = [Spec: null] C' [C, TP]
            c_bar = XBarNode(
                category=Category.C,
                level=1,  # C'
                head=comp.text,
                head_token=comp,
                complement=tp
            )

            cp = XBarNode(
                category=Category.C,
                level=2,  # CP
                head=comp.text,
                head_token=comp,
                complement=c_bar
            )
            return cp

        return tp

    # ========================================================================
    # Phrase Building (VP, NP, PP, AP)
    # ========================================================================

    def _build_vp(self, verb, doc) -> Optional[XBarNode]:
        """
        Build VP (Verb Phrase) using X-bar theory.

        VP structure:
            VP
             ├─ Spec: (adverbs, etc.)
             └─ V'
                 ├─ V: verb head
                 └─ Comp: object (NP/CP/PP)

        Args:
            verb: Verb token
            doc: spaCy doc

        Returns:
            VP node (maximal projection)
        """
        category = self.pos_to_category.get(verb.pos_, Category.V)

        # Find complement (object)
        complement = self._find_complement(verb, doc)

        # Find specifier (adverbs modifying verb)
        specifier = self._find_vp_specifier(verb, doc)

        # Build V' (intermediate projection)
        v_bar = XBarNode(
            category=category,
            level=1,  # V'
            head=verb.text,
            head_token=verb,
            complement=complement
        )

        # Build VP (maximal projection)
        vp = XBarNode(
            category=category,
            level=2,  # VP
            head=verb.text,
            head_token=verb,
            specifier=specifier,
            complement=v_bar
        )

        return vp

    def _build_np(self, noun, doc) -> Optional[XBarNode]:
        """
        Build NP (Noun Phrase) using X-bar theory.

        NP structure:
            NP
             ├─ Spec: Det "the"
             └─ N'
                 ├─ Adjunct: AP "big"
                 └─ N'
                     ├─ Adjunct: AP "red"
                     └─ N: "ball"

        Args:
            noun: Noun token
            doc: spaCy doc

        Returns:
            NP node (maximal projection)
        """
        category = self.pos_to_category.get(noun.pos_, Category.N)

        # Find determiner (specifier)
        det = self._find_determiner(noun)
        det_node = None
        if det:
            det_node = XBarNode(
                category=Category.D,
                level=0,  # Just D
                head=det.text,
                head_token=det
            )

        # Find adjectives (adjuncts to N')
        adjectives = self._find_adjectives(noun)
        adjuncts = []
        for adj in adjectives:
            # Build AP for each adjective
            ap = XBarNode(
                category=Category.A,
                level=2,  # AP
                head=adj.text,
                head_token=adj,
                complement=XBarNode(
                    category=Category.A,
                    level=0,  # A
                    head=adj.text,
                    head_token=adj
                )
            )
            adjuncts.append(ap)

        # Build N (lexical head)
        n_head = XBarNode(
            category=category,
            level=0,  # N
            head=noun.text,
            head_token=noun
        )

        # Build N' with adjuncts
        # Adjuncts attach right-to-left (innermost first)
        current = n_head
        for adj in reversed(adjuncts):
            n_bar = XBarNode(
                category=category,
                level=1,  # N'
                head=noun.text,
                head_token=noun,
                complement=current,
                adjuncts=[adj]
            )
            current = n_bar

        # Final N' (or just N if no adjuncts)
        if not adjuncts:
            n_bar = XBarNode(
                category=category,
                level=1,  # N'
                head=noun.text,
                head_token=noun,
                complement=n_head
            )
        else:
            n_bar = current

        # Build NP (maximal projection)
        np = XBarNode(
            category=category,
            level=2,  # NP
            head=noun.text,
            head_token=noun,
            specifier=det_node,
            complement=n_bar
        )

        return np

    def _build_pp(self, prep, doc) -> Optional[XBarNode]:
        """
        Build PP (Prepositional Phrase).

        PP structure:
            PP
             └─ P'
                 ├─ P: "in"
                 └─ Comp: NP "the morning"

        Args:
            prep: Preposition token
            doc: spaCy doc

        Returns:
            PP node (maximal projection)
        """
        # Find NP complement
        np_head = None
        for child in prep.children:
            if child.dep_ == "pobj":  # Prepositional object
                np_head = child
                break

        complement = self._build_np(np_head, doc) if np_head else None

        # Build P' (intermediate)
        p_bar = XBarNode(
            category=Category.P,
            level=1,  # P'
            head=prep.text,
            head_token=prep,
            complement=complement
        )

        # Build PP (maximal)
        pp = XBarNode(
            category=Category.P,
            level=2,  # PP
            head=prep.text,
            head_token=prep,
            complement=p_bar
        )

        return pp

    # ========================================================================
    # Helper Methods (Finding Constituents)
    # ========================================================================

    def _find_subject(self, verb) -> Optional:
        """Find subject of verb."""
        for child in verb.children:
            if child.dep_ in ("nsubj", "nsubjpass", "csubj"):
                return child
        return None

    def _find_complement(self, verb, doc) -> Optional[XBarNode]:
        """Find complement of verb (object, clausal complement, etc.)."""
        for child in verb.children:
            if child.dep_ == "dobj":  # Direct object
                return self._build_np(child, doc)
            elif child.dep_ == "ccomp":  # Clausal complement
                return self._build_clause(child, doc)
            elif child.dep_ == "xcomp":  # Open clausal complement
                return self._build_vp(child, doc)
            elif child.dep_ == "prep":  # Prepositional phrase
                return self._build_pp(child, doc)
        return None

    def _find_vp_specifier(self, verb, doc) -> Optional[XBarNode]:
        """Find specifier of VP (adverbs)."""
        for child in verb.children:
            if child.dep_ == "advmod":
                # Build simple adverb node
                return XBarNode(
                    category=Category.A,  # Simplified: adverbs as adjectives
                    level=0,
                    head=child.text,
                    head_token=child
                )
        return None

    def _find_determiner(self, noun) -> Optional:
        """Find determiner of noun."""
        for child in noun.children:
            if child.dep_ == "det":
                return child
        return None

    def _find_adjectives(self, noun) -> List:
        """Find adjectives modifying noun."""
        adjectives = []
        for child in noun.children:
            if child.dep_ == "amod":  # Adjectival modifier
                adjectives.append(child)
        # Sort by position (left-to-right)
        adjectives.sort(key=lambda t: t.i)
        return adjectives

    def _find_complementizer(self, verb) -> Optional:
        """Find complementizer (that, if, whether)."""
        for child in verb.children:
            if child.dep_ == "mark":  # Marker (complementizer)
                return child
        return None


# ============================================================================
# Visualization
# ============================================================================

def visualize_xbar(node: XBarNode, indent: int = 0) -> str:
    """
    Visualize X-bar structure as tree.

    Example output:
        NP
         ├─ Spec: D "the"
         └─ N'
             ├─ Adjunct: AP "big"
             └─ N "ball"

    Args:
        node: Root node
        indent: Indentation level

    Returns:
        String representation of tree
    """
    prefix = "  " * indent
    result = f"{prefix}{node.label}"

    if node.level == 0:
        # Lexical head
        result += f' "{node.head}"\n'
    else:
        result += "\n"

        # Specifier
        if node.specifier:
            result += f"{prefix} |- Spec: "
            result += visualize_xbar(node.specifier, indent + 1).lstrip()

        # Adjuncts
        for adj in node.adjuncts:
            result += f"{prefix} |- Adjunct: "
            result += visualize_xbar(adj, indent + 1).lstrip()

        # Complement
        if node.complement:
            connector = "L-" if not (node.specifier or node.adjuncts) else "L-"
            result += f"{prefix} {connector} "
            result += visualize_xbar(node.complement, indent + 1).lstrip()

    return result


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("UNIVERSAL GRAMMAR CHUNKER DEMO")
    print("=" * 80)
    print()

    # Create chunker
    chunker = UniversalGrammarChunker()

    if not chunker.nlp:
        print("ERROR: spaCy not available")
        print("Install with: pip install spacy")
        print("Then download model: python -m spacy download en_core_web_sm")
        exit(1)

    # Test phrases
    test_cases = [
        "the big red ball",
        "John hit the ball",
        "Mary gave John the book",
        "The cat sat on the mat",
        "I think that he left",
    ]

    for test_text in test_cases:
        print(f"Input: \"{test_text}\"")
        print("-" * 80)

        phrases = chunker.chunk(test_text)

        for phrase in phrases:
            print(visualize_xbar(phrase))

        print()

    print("✓ Universal Grammar chunker operational!")