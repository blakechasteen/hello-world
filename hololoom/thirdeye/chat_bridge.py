"""
ChatBridge - Connect to multithreaded chat and extract concepts.

Listens to the chat WebSocket, extracts semantic concepts from messages,
and emits them for visualization in the 3rdEye panel.

Created: 2025-11-30
Author: HoloLoom Team
"""

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Dict, Any, List, Optional
from enum import Enum, auto

from hololoom.thirdeye.concept import (
    Concept,
    ConceptType,
    SemanticPosition,
    create_entity_concept,
    create_comparison_concept,
    create_process_concept,
)
from hololoom.thirdeye.transition import (
    Transition,
    calculate_transition,
    create_emphasis_transition,
)

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Role of the message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """Parsed chat message from WebSocket."""
    id: str
    role: MessageRole
    content: str
    thread_id: Optional[str] = None
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptExtraction:
    """Result of extracting concepts from a message."""
    concepts: List[Concept]
    transition: Optional[Transition] = None
    primary_concept: Optional[Concept] = None


class ConceptExtractor:
    """
    Extract semantic concepts from chat messages.

    Uses pattern matching and (optionally) embeddings to identify
    key concepts in conversation messages.
    """

    # Patterns that suggest certain concept types
    COMPARISON_PATTERNS = [
        r'\b(vs\.?|versus|compared?\s+to|differ(?:ence|ent)?|between)\b',
        r'\b(better|worse|faster|slower|pros?\s+and\s+cons?)\b',
        r'\b(trade-?offs?|advantages?|disadvantages?)\b',
    ]

    PROCESS_PATTERNS = [
        r'\b(steps?|process|algorithm|workflow|procedure|how\s+to)\b',
        r'\b(first|then|next|finally|after|before)\b',
        r'\b(implementation|implement|create|build|make)\b',
    ]

    ENTITY_PATTERNS = [
        r'(?:what\s+is|define|explain)\s+([A-Z][a-zA-Z\s]+)',
        r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',  # Capitalized phrases
    ]

    def __init__(self, embedder: Optional[Callable[[str], List[float]]] = None):
        """
        Initialize the concept extractor.

        Args:
            embedder: Optional function to compute embeddings for concepts
        """
        self._embedder = embedder
        self._concept_cache: Dict[str, Concept] = {}

    def extract(self, message: ChatMessage) -> ConceptExtraction:
        """
        Extract concepts from a chat message.

        Args:
            message: The chat message to analyze

        Returns:
            ConceptExtraction with identified concepts
        """
        content = message.content
        concepts: List[Concept] = []

        # Check for comparison patterns
        if self._matches_patterns(content, self.COMPARISON_PATTERNS):
            comparison = self._extract_comparison(content, message.id)
            if comparison:
                concepts.append(comparison)

        # Check for process patterns
        if self._matches_patterns(content, self.PROCESS_PATTERNS):
            process = self._extract_process(content, message.id)
            if process:
                concepts.append(process)

        # Extract named entities
        entities = self._extract_entities(content, message.id)
        concepts.extend(entities)

        # If no specific concepts found, create a general topic concept
        if not concepts:
            topic = self._extract_topic(content, message.id)
            if topic:
                concepts.append(topic)

        # Determine primary concept (highest importance)
        primary = max(concepts, key=lambda c: c.importance) if concepts else None

        # Calculate transition from previous concept
        transition = None
        if primary and self._last_concept_embedding is not None:
            transition = calculate_transition(
                self._last_concept_embedding,
                primary.position.embedding,
                self._last_concept_type,
                primary.concept_type.name.lower(),
            )
            transition.from_concept_id = self._last_concept_id
            transition.to_concept_id = primary.id

        # Update tracking
        if primary:
            self._last_concept_id = primary.id
            self._last_concept_type = primary.concept_type.name.lower()
            self._last_concept_embedding = primary.position.embedding

        return ConceptExtraction(
            concepts=concepts,
            transition=transition,
            primary_concept=primary,
        )

    # Tracking for transitions
    _last_concept_id: Optional[str] = None
    _last_concept_type: Optional[str] = None
    _last_concept_embedding: Optional[List[float]] = None

    def _matches_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the patterns."""
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in patterns)

    def _extract_comparison(self, content: str, message_id: str) -> Optional[Concept]:
        """Extract a comparison concept."""
        # Look for "A vs B" or "A compared to B" patterns
        vs_match = re.search(r'(\w+)\s+(?:vs\.?|versus|compared?\s+to)\s+(\w+)', content, re.I)
        if vs_match:
            a, b = vs_match.groups()
            return create_comparison_concept(
                id=f"comparison_{message_id}",
                label=f"{a} vs {b}",
                concepts_compared=[a.lower(), b.lower()],
                importance=0.7,
            )
        return None

    def _extract_process(self, content: str, message_id: str) -> Optional[Concept]:
        """Extract a process/algorithm concept."""
        # Look for step indicators
        steps = re.findall(r'(?:step\s+\d+|first|then|next|finally)[:\s]+([^.]+)', content, re.I)
        if steps:
            return create_process_concept(
                id=f"process_{message_id}",
                label="Process",
                steps=[s.strip() for s in steps],
                importance=0.6,
            )
        return None

    def _extract_entities(self, content: str, message_id: str) -> List[Concept]:
        """Extract named entity concepts."""
        entities: List[Concept] = []

        # Look for capitalized terms (likely proper nouns/technical terms)
        matches = re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b', content)

        # Filter common words and duplicates
        common = {'The', 'This', 'That', 'What', 'How', 'Why', 'When', 'Where', 'I', 'We', 'They'}
        seen = set()
        for match in matches:
            if match not in common and match not in seen:
                seen.add(match)

                # Compute embedding if embedder available
                embedding = None
                if self._embedder:
                    try:
                        embedding = self._embedder(match)
                    except Exception:
                        pass

                entity = create_entity_concept(
                    id=f"entity_{match.lower().replace(' ', '_')}_{message_id[:8]}",
                    label=match,
                    importance=0.5,
                    embedding=embedding,
                )
                entities.append(entity)

        return entities[:5]  # Limit to 5 entities per message

    def _extract_topic(self, content: str, message_id: str) -> Optional[Concept]:
        """Extract a general topic concept when no specific concepts found."""
        # Use first significant phrase as topic
        words = content.split()
        if len(words) < 3:
            return None

        # Take first few words as topic label
        label = ' '.join(words[:5]) + ('...' if len(words) > 5 else '')

        embedding = None
        if self._embedder:
            try:
                embedding = self._embedder(content[:200])
            except Exception:
                pass

        return Concept(
            id=f"topic_{message_id}",
            label=label,
            concept_type=ConceptType.ENTITY,
            position=SemanticPosition.from_embedding(embedding) if embedding else SemanticPosition(),
            importance=0.4,
            source_message_id=message_id,
        )


class ChatBridge:
    """
    Connect to the multithreaded chat WebSocket and stream concepts.

    Listens for chat messages, extracts concepts, and yields them
    for visualization in the 3rdEye panel.
    """

    def __init__(
        self,
        chat_ws_url: str = "ws://localhost:8002",
        embedder: Optional[Callable[[str], List[float]]] = None,
    ):
        """
        Initialize the chat bridge.

        Args:
            chat_ws_url: WebSocket URL of the chat server
            embedder: Optional function to compute embeddings
        """
        self.chat_ws_url = chat_ws_url
        self.extractor = ConceptExtractor(embedder)
        self._running = False
        self._websocket = None

    async def listen(self) -> AsyncIterator[ConceptExtraction]:
        """
        Listen to chat and yield concept extractions.

        Yields:
            ConceptExtraction for each relevant message
        """
        self._running = True

        try:
            # Try to import websockets
            import websockets
        except ImportError:
            logger.warning("websockets not installed, using mock mode")
            async for extraction in self._mock_listen():
                yield extraction
            return

        try:
            async with websockets.connect(self.chat_ws_url) as ws:
                self._websocket = ws
                logger.info(f"Connected to chat at {self.chat_ws_url}")

                async for raw_message in ws:
                    if not self._running:
                        break

                    try:
                        data = json.loads(raw_message)
                        message = self._parse_message(data)

                        if message and message.role in [MessageRole.USER, MessageRole.ASSISTANT]:
                            extraction = self.extractor.extract(message)
                            if extraction.concepts:
                                yield extraction

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from chat: {raw_message[:100]}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            # Fall back to mock mode
            async for extraction in self._mock_listen():
                yield extraction

    async def _mock_listen(self) -> AsyncIterator[ConceptExtraction]:
        """Mock listener for testing without WebSocket."""
        mock_messages = [
            ChatMessage(
                id="1",
                role=MessageRole.USER,
                content="What is Thompson Sampling?",
            ),
            ChatMessage(
                id="2",
                role=MessageRole.ASSISTANT,
                content="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            ),
            ChatMessage(
                id="3",
                role=MessageRole.USER,
                content="How does it compare to UCB?",
            ),
        ]

        for msg in mock_messages:
            if not self._running:
                break
            await asyncio.sleep(2.0)  # Simulate message timing
            extraction = self.extractor.extract(msg)
            if extraction.concepts:
                yield extraction

    def _parse_message(self, data: Dict[str, Any]) -> Optional[ChatMessage]:
        """Parse raw WebSocket data into a ChatMessage."""
        try:
            msg_type = data.get("type", "")

            # Handle different message formats
            if msg_type == "message" or "content" in data:
                return ChatMessage(
                    id=data.get("id", str(uuid.uuid4())),
                    role=MessageRole(data.get("role", "user")),
                    content=data.get("content", ""),
                    thread_id=data.get("thread_id"),
                    timestamp=data.get("timestamp", 0.0),
                    metadata=data.get("metadata", {}),
                )

            # Handle streaming tokens (accumulate until complete)
            if msg_type == "token":
                # Would need state to accumulate - skip for now
                return None

            return None

        except Exception as e:
            logger.warning(f"Failed to parse message: {e}")
            return None

    async def stop(self) -> None:
        """Stop listening to chat."""
        self._running = False
        if self._websocket:
            await self._websocket.close()


async def create_chat_bridge(
    chat_ws_url: str = "ws://localhost:8002",
    embedder: Optional[Callable[[str], List[float]]] = None,
) -> ChatBridge:
    """
    Create a chat bridge instance.

    Args:
        chat_ws_url: WebSocket URL of the chat server
        embedder: Optional embedding function

    Returns:
        Configured ChatBridge instance
    """
    return ChatBridge(chat_ws_url=chat_ws_url, embedder=embedder)
