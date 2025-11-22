"""
HoloLoom Shuttle - Entity Extraction Strategies

Provides multiple strategies for extracting anchors from Warp search results:
1. Payload method (zero dependencies, fastest)
2. Regex method (lightweight, pattern-based)
3. spaCy method (highest quality, requires spaCy)

Created: 2025-01-21
"""

from typing import List, Dict, Any, Protocol, Optional
import re
import logging

from .exceptions import EntityExtractionError

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class Anchor:
    """
    An anchor point from Warp search results into the Yarn graph.

    Anchors are named entities extracted from semantic search results
    that serve as entry points for graph traversal.
    """

    def __init__(
        self,
        name: str,
        type: str,
        node_id: Optional[str] = None,
        confidence: float = 1.0,
        source: str = "unknown"
    ):
        self.name = name
        self.type = type
        self.node_id = node_id
        self.confidence = confidence
        self.source = source  # "payload", "regex", "spacy"

    def __repr__(self) -> str:
        return f"Anchor(name='{self.name}', type='{self.type}', id={self.node_id})"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "name": self.name,
            "type": self.type,
            "node_id": self.node_id,
            "confidence": self.confidence,
            "source": self.source,
        }


# ============================================================================
# Extraction Protocol
# ============================================================================

class EntityExtractor(Protocol):
    """
    Protocol for entity extraction strategies.

    All extractors must implement extract() method that takes
    Warp search results and returns a list of Anchors.
    """

    def extract(
        self,
        warp_results: List[Dict[str, Any]],
        max_anchors: int = 10
    ) -> List[Anchor]:
        """
        Extract anchors from Warp search results.

        Args:
            warp_results: List of results from Warp.search()
            max_anchors: Maximum number of anchors to return

        Returns:
            List of Anchor objects

        Raises:
            EntityExtractionError: If extraction fails
        """
        ...


# ============================================================================
# Payload Extractor (Zero Dependencies)
# ============================================================================

class PayloadExtractor:
    """
    Extract entities from Warp result payload.

    Assumes Warp results have an "entities" field in the payload.
    This is the fastest method but requires entities to be
    pre-extracted during ingestion.

    Zero dependencies, always works.
    """

    def extract(
        self,
        warp_results: List[Dict[str, Any]],
        max_anchors: int = 10
    ) -> List[Anchor]:
        """
        Extract anchors from payload "entities" field.

        Expected payload structure:
        {
            "entities": [
                {"name": "HoloLoom", "type": "Project", "node_id": "proj_123"},
                ...
            ]
        }

        Args:
            warp_results: Results from Warp.search()
            max_anchors: Max anchors to return

        Returns:
            List of Anchor objects
        """
        anchors = []

        for result in warp_results:
            # Check if result has entities field
            entities = result.get("entities", [])

            if not entities:
                # Fallback: treat the result itself as an entity
                if "node_id" in result and "name" in result:
                    anchors.append(Anchor(
                        name=result["name"],
                        type=result.get("type", "Unknown"),
                        node_id=result.get("node_id"),
                        confidence=result.get("score", 1.0),
                        source="payload_fallback"
                    ))
                continue

            # Extract from entities field
            for entity in entities:
                if len(anchors) >= max_anchors:
                    break

                anchors.append(Anchor(
                    name=entity.get("name", ""),
                    type=entity.get("type", "Unknown"),
                    node_id=entity.get("node_id"),
                    confidence=result.get("score", 1.0),
                    source="payload"
                ))

            if len(anchors) >= max_anchors:
                break

        logger.debug(f"Payload extractor found {len(anchors)} anchors")
        return anchors[:max_anchors]


# ============================================================================
# Regex Extractor (Lightweight)
# ============================================================================

class RegexExtractor:
    """
    Extract entities using regex patterns.

    Lightweight extractor that uses regex patterns to identify
    common entity types (projects, tasks, people, etc.).

    Requires no external dependencies.
    """

    # Entity patterns
    PATTERNS = {
        "Project": [
            r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+[Pp]roject\b',
            r'\b[Pp]roject\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b',
        ],
        "Task": [
            r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+[Tt]ask\b',
            r'\b[Tt]ask:\s+([A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+)*)\b',
            r'\bImplement\s+([A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+)*)\b',
            r'\bFix\s+([A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+)*)\b',
        ],
        "Person": [
            r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b',  # John Doe
        ],
        "Issue": [
            r'\b[Ii]ssue[:\s#]+([A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+)*)\b',
            r'\bBug:\s+([A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+)*)\b',
        ],
    }

    def extract(
        self,
        warp_results: List[Dict[str, Any]],
        max_anchors: int = 10
    ) -> List[Anchor]:
        """
        Extract anchors using regex patterns.

        Args:
            warp_results: Results from Warp.search()
            max_anchors: Max anchors to return

        Returns:
            List of Anchor objects
        """
        anchors = []
        seen_names = set()

        for result in warp_results:
            content = result.get("content", "") + " " + result.get("name", "")

            for entity_type, patterns in self.PATTERNS.items():
                for pattern in patterns:
                    matches = re.findall(pattern, content)

                    for match in matches:
                        if len(anchors) >= max_anchors:
                            break

                        # Extract name (handle tuple matches)
                        if isinstance(match, tuple):
                            name = " ".join(match).strip()
                        else:
                            name = match.strip()

                        # Skip duplicates
                        if name in seen_names or len(name) < 3:
                            continue

                        seen_names.add(name)
                        anchors.append(Anchor(
                            name=name,
                            type=entity_type,
                            node_id=None,  # Will be resolved in Yarn
                            confidence=0.7,  # Regex is less confident
                            source="regex"
                        ))

        logger.debug(f"Regex extractor found {len(anchors)} anchors")
        return anchors[:max_anchors]


# ============================================================================
# spaCy Extractor (Highest Quality)
# ============================================================================

class SpacyExtractor:
    """
    Extract entities using spaCy NER (Named Entity Recognition).

    Highest quality entity extraction but requires spaCy to be installed.
    Falls back gracefully if spaCy is not available.

    Install: pip install spacy && python -m spacy download en_core_web_sm
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize spaCy extractor.

        Args:
            model_name: spaCy model to use

        Raises:
            EntityExtractionError: If spaCy is not installed
        """
        try:
            import spacy
            self.nlp = spacy.load(model_name)
            self.available = True
            logger.info(f"spaCy extractor initialized with {model_name}")
        except (ImportError, OSError) as e:
            self.available = False
            logger.warning(
                f"spaCy not available: {e}. "
                "Install with: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            )

    def extract(
        self,
        warp_results: List[Dict[str, Any]],
        max_anchors: int = 10
    ) -> List[Anchor]:
        """
        Extract anchors using spaCy NER.

        Args:
            warp_results: Results from Warp.search()
            max_anchors: Max anchors to return

        Returns:
            List of Anchor objects

        Raises:
            EntityExtractionError: If spaCy is not available
        """
        if not self.available:
            raise EntityExtractionError(
                "spaCy not installed",
                extraction_method="spacy",
                fallback_available=True
            )

        anchors = []
        seen_names = set()

        # Entity type mapping
        type_mapping = {
            "ORG": "Organization",
            "PRODUCT": "Project",
            "PERSON": "Person",
            "WORK_OF_ART": "Project",
            "EVENT": "Event",
        }

        for result in warp_results:
            content = result.get("content", "")

            # Process with spaCy
            doc = self.nlp(content)

            for ent in doc.ents:
                if len(anchors) >= max_anchors:
                    break

                # Map spaCy entity type to our types
                entity_type = type_mapping.get(ent.label_, ent.label_)

                # Skip duplicates
                if ent.text in seen_names:
                    continue

                seen_names.add(ent.text)
                anchors.append(Anchor(
                    name=ent.text,
                    type=entity_type,
                    node_id=None,
                    confidence=0.9,  # spaCy is highly confident
                    source="spacy"
                ))

        logger.debug(f"spaCy extractor found {len(anchors)} anchors")
        return anchors[:max_anchors]


# ============================================================================
# Factory & Fallback Chain
# ============================================================================

class EntityExtractionFactory:
    """
    Factory for creating entity extractors with automatic fallback.

    Tries methods in order: spaCy → Regex → Payload
    """

    @staticmethod
    def create(
        method: str = "auto",
        enable_fallback: bool = True
    ) -> EntityExtractor:
        """
        Create entity extractor with fallback support.

        Args:
            method: Extraction method ("spacy", "regex", "payload", "auto")
            enable_fallback: Enable automatic fallback

        Returns:
            EntityExtractor instance
        """
        if method == "auto":
            # Try spaCy first, fall back to regex, then payload
            return ChainedExtractor(enable_fallback=enable_fallback)

        elif method == "spacy":
            extractor = SpacyExtractor()
            if not extractor.available and enable_fallback:
                logger.warning("spaCy unavailable, falling back to regex")
                return RegexExtractor()
            return extractor

        elif method == "regex":
            return RegexExtractor()

        elif method == "payload":
            return PayloadExtractor()

        else:
            raise ValueError(
                f"Unknown extraction method: {method}. "
                f"Valid: spacy, regex, payload, auto"
            )


class ChainedExtractor:
    """
    Tries multiple extractors in sequence with fallback.

    Chain: spaCy → Regex → Payload
    """

    def __init__(self, enable_fallback: bool = True):
        self.enable_fallback = enable_fallback
        self.extractors = [
            ("spacy", SpacyExtractor()),
            ("regex", RegexExtractor()),
            ("payload", PayloadExtractor()),
        ]

    def extract(
        self,
        warp_results: List[Dict[str, Any]],
        max_anchors: int = 10
    ) -> List[Anchor]:
        """
        Try extractors in sequence until one succeeds.

        Args:
            warp_results: Results from Warp.search()
            max_anchors: Max anchors to return

        Returns:
            List of Anchor objects

        Raises:
            EntityExtractionError: If all extractors fail
        """
        for name, extractor in self.extractors:
            try:
                anchors = extractor.extract(warp_results, max_anchors)
                if anchors:
                    logger.debug(f"Extractor '{name}' succeeded")
                    return anchors
            except Exception as e:
                logger.debug(f"Extractor '{name}' failed: {e}")
                if not self.enable_fallback:
                    raise EntityExtractionError(
                        f"Extractor '{name}' failed: {e}",
                        extraction_method=name,
                        fallback_available=False
                    )

        # All extractors failed
        raise EntityExtractionError(
            "All entity extraction methods failed",
            extraction_method="chained",
            fallback_available=False
        )