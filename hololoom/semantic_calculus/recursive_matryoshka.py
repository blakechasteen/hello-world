#!/usr/bin/env python3
"""
🪆🔄 RECURSIVE MATRYOSHKA STREAMING
===================================
True compositional semantics: Each level embeds the previous level's understanding.

The key innovation: RECURSIVE COMPOSITION
- Word-level produces 16D semantic understanding
- Phrase-level USES word-level 16D + new 192D embedding → 64D
- Sentence-level USES phrase-level 64D + word-level 16D + new 384D → 128D
- Each level has access to ALL previous levels' interpretations

This mirrors human comprehension:
- We don't re-analyze words when reading sentences
- We COMPOSE: word meanings → phrase meanings → sentence meanings
- Lower-level understanding feeds upward

Mathematical Framework:
    S_word = f_word(E_96d)                                    # 16D
    S_phrase = f_phrase(E_192d, S_word)                       # 64D
    S_sentence = f_sentence(E_384d, S_phrase, S_word)         # 128D
    S_paragraph = f_paragraph(E_384d, S_sentence, S_phrase, S_word)  # 244D

Where:
    E_*d = Embedding at dimension *
    S_* = Semantic understanding at scale *
    f_* = Projection function with recursive inputs

Fusion Strategies:
1. CONCATENATE: Simply concat lower-level semantics
2. CROSS_ATTENTION: Higher level attends to lower levels
3. RESIDUAL: Add upsampled lower-level as residual
4. GATED: Learn gates to weight lower-level contributions
"""

import sys
import os
# Add repository root to path for imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import asyncio
import numpy as np
from typing import Dict, List, Optional, AsyncIterator, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time

from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.semantic_calculus.dimensions import SemanticSpectrum, STANDARD_DIMENSIONS


class FusionStrategy(Enum):
    """How to combine lower-level semantics with current level."""
    CONCATENATE = "concatenate"      # Simple concatenation
    WEIGHTED_SUM = "weighted_sum"    # Weighted average
    RESIDUAL = "residual"            # Add as residual connection
    GATED = "gated"                  # Learned gating


@dataclass
class RecursiveState:
    """State at a recursive Matryoshka level."""
    embedding: np.ndarray           # Raw embedding at this scale's dimension
    semantic: np.ndarray            # Projected semantic understanding
    inherited_semantics: Dict[str, np.ndarray]  # From lower levels
    timestamp: float
    token_count: int


class RecursiveMatryoshkaScale(Enum):
    """Scales with recursive composition."""
    WORD = ("word", 1, 5, 96, 16)
    PHRASE = ("phrase", 5, 15, 192, 64)
    SENTENCE = ("sentence", 15, 50, 384, 128)
    PARAGRAPH = ("paragraph", 50, 200, 384, 244)

    def __init__(self, name, min_tokens, max_tokens, embedding_dim, semantic_dim):
        self.scale_name = name
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.embedding_dim = embedding_dim
        self.semantic_dim = semantic_dim


class RecursiveFusion:
    """Handles fusion of lower-level semantics into higher levels."""

    def __init__(self, strategy: FusionStrategy = FusionStrategy.WEIGHTED_SUM):
        self.strategy = strategy
        # Learnable weights (in practice, these could be trained)
        self.weights = {
            'word_to_phrase': 0.3,
            'phrase_to_sentence': 0.4,
            'word_to_sentence': 0.2,
            'all_to_paragraph': 0.5
        }

    def fuse_for_phrase(
        self,
        phrase_embedding: np.ndarray,  # 192D
        word_semantic: Optional[np.ndarray]  # 16D
    ) -> np.ndarray:
        """Fuse word-level semantics into phrase embedding."""
        if word_semantic is None:
            return phrase_embedding

        if self.strategy == FusionStrategy.CONCATENATE:
            # Concatenate: [192D embedding | 16D word semantic] → 208D
            return np.concatenate([phrase_embedding, word_semantic])

        elif self.strategy == FusionStrategy.WEIGHTED_SUM:
            # Upsample word_semantic to 192D and add weighted
            upsampled = self._upsample(word_semantic, 192)
            weight = self.weights['word_to_phrase']
            return phrase_embedding * (1 - weight) + upsampled * weight

        elif self.strategy == FusionStrategy.RESIDUAL:
            # Residual connection: phrase + α * word (upsampled)
            upsampled = self._upsample(word_semantic, 192)
            return phrase_embedding + 0.2 * upsampled

        elif self.strategy == FusionStrategy.GATED:
            # Simple gating: learn to weight
            upsampled = self._upsample(word_semantic, 192)
            gate = self._compute_gate(phrase_embedding, upsampled)
            return gate * phrase_embedding + (1 - gate) * upsampled

        return phrase_embedding

    def fuse_for_sentence(
        self,
        sentence_embedding: np.ndarray,  # 384D
        phrase_semantic: Optional[np.ndarray],  # 64D
        word_semantic: Optional[np.ndarray]  # 16D
    ) -> np.ndarray:
        """Fuse phrase-level and word-level semantics into sentence."""
        if phrase_semantic is None and word_semantic is None:
            return sentence_embedding

        if self.strategy == FusionStrategy.CONCATENATE:
            # Concatenate all: [384D | 64D phrase | 16D word] → 464D
            parts = [sentence_embedding]
            if phrase_semantic is not None:
                parts.append(phrase_semantic)
            if word_semantic is not None:
                parts.append(word_semantic)
            return np.concatenate(parts)

        elif self.strategy == FusionStrategy.WEIGHTED_SUM:
            result = sentence_embedding
            if phrase_semantic is not None:
                phrase_up = self._upsample(phrase_semantic, 384)
                weight = self.weights['phrase_to_sentence']
                result = result * (1 - weight) + phrase_up * weight
            if word_semantic is not None:
                word_up = self._upsample(word_semantic, 384)
                weight = self.weights['word_to_sentence']
                result = result * (1 - weight) + word_up * weight
            return result

        elif self.strategy == FusionStrategy.RESIDUAL:
            result = sentence_embedding
            if phrase_semantic is not None:
                result = result + 0.3 * self._upsample(phrase_semantic, 384)
            if word_semantic is not None:
                result = result + 0.1 * self._upsample(word_semantic, 384)
            return result

        return sentence_embedding

    def fuse_for_paragraph(
        self,
        paragraph_embedding: np.ndarray,  # 384D
        sentence_semantic: Optional[np.ndarray],  # 128D
        phrase_semantic: Optional[np.ndarray],  # 64D
        word_semantic: Optional[np.ndarray]  # 16D
    ) -> np.ndarray:
        """Fuse all lower levels into paragraph."""
        if self.strategy == FusionStrategy.CONCATENATE:
            # Stack all semantics: [384D | 128D | 64D | 16D] → 592D
            parts = [paragraph_embedding]
            if sentence_semantic is not None:
                parts.append(sentence_semantic)
            if phrase_semantic is not None:
                parts.append(phrase_semantic)
            if word_semantic is not None:
                parts.append(word_semantic)
            return np.concatenate(parts)

        elif self.strategy == FusionStrategy.WEIGHTED_SUM:
            result = paragraph_embedding
            weight = self.weights['all_to_paragraph']

            combined_lower = np.zeros(384)
            count = 0

            if sentence_semantic is not None:
                combined_lower += self._upsample(sentence_semantic, 384)
                count += 1
            if phrase_semantic is not None:
                combined_lower += self._upsample(phrase_semantic, 384)
                count += 1
            if word_semantic is not None:
                combined_lower += self._upsample(word_semantic, 384)
                count += 1

            if count > 0:
                combined_lower /= count
                result = result * (1 - weight) + combined_lower * weight

            return result

        elif self.strategy == FusionStrategy.RESIDUAL:
            result = paragraph_embedding
            if sentence_semantic is not None:
                result = result + 0.3 * self._upsample(sentence_semantic, 384)
            if phrase_semantic is not None:
                result = result + 0.2 * self._upsample(phrase_semantic, 384)
            if word_semantic is not None:
                result = result + 0.1 * self._upsample(word_semantic, 384)
            return result

        return paragraph_embedding

    def _upsample(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        """Upsample vector to target dimension using interpolation."""
        if len(vector) >= target_dim:
            return vector[:target_dim]

        # Linear interpolation
        indices = np.linspace(0, len(vector) - 1, target_dim)
        upsampled = np.interp(indices, np.arange(len(vector)), vector)
        return upsampled

    def _compute_gate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Simple gating function based on vector similarity."""
        # Gate is high when vectors are similar (trust both)
        # Gate is low when vectors are dissimilar (favor current)
        cosine_sim = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-10)
        return float(np.clip((cosine_sim + 1) / 2, 0, 1))  # Normalize to [0,1]


class RecursiveMatryoshkaCalculus:
    """
    Recursive Matryoshka streaming with compositional semantics.

    Each level inherits understanding from all previous levels.
    """

    def __init__(
        self,
        matryoshka_embedder: MatryoshkaEmbeddings,
        fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_SUM,
        snapshot_interval: float = 1.0
    ):
        self.embedder = matryoshka_embedder
        self.fusion = RecursiveFusion(strategy=fusion_strategy)
        self.snapshot_interval = snapshot_interval

        # Create semantic spectrums (same as before)
        self.spectrum_word = SemanticSpectrum(dimensions=STANDARD_DIMENSIONS[:16])

        from HoloLoom.semantic_calculus.dimensions import (
            NARRATIVE_DIMENSIONS,
            EMOTIONAL_DEPTH_DIMENSIONS, RELATIONAL_DIMENSIONS,
            ARCHETYPAL_DIMENSIONS, PHILOSOPHICAL_DIMENSIONS,
            EXTENDED_244_DIMENSIONS
        )
        phrase_dims = (STANDARD_DIMENSIONS + NARRATIVE_DIMENSIONS +
                      EMOTIONAL_DEPTH_DIMENSIONS + RELATIONAL_DIMENSIONS)
        self.spectrum_phrase = SemanticSpectrum(dimensions=phrase_dims[:64])

        sentence_dims = phrase_dims + ARCHETYPAL_DIMENSIONS + PHILOSOPHICAL_DIMENSIONS
        self.spectrum_sentence = SemanticSpectrum(dimensions=sentence_dims[:128])

        self.spectrum_paragraph = SemanticSpectrum(dimensions=EXTENDED_244_DIMENSIONS)

        print("🔧 Learning recursive Matryoshka axes...")

        def word_embed(w):
            return self.embedder.encode_base([w])[0][:96]
        self.spectrum_word.learn_axes(word_embed)

        def phrase_embed(p):
            return self.embedder.encode_base([p])[0][:192]
        self.spectrum_phrase.learn_axes(phrase_embed)

        def sentence_embed(s):
            return self.embedder.encode_base([s])[0][:384]
        self.spectrum_sentence.learn_axes(sentence_embed)
        self.spectrum_paragraph.learn_axes(sentence_embed)

        print("✅ Recursive axes ready!\n")

        # Recursive state tracking
        self.current_states: Dict[RecursiveMatryoshkaScale, RecursiveState] = {}

        # Windows for token buffering
        self.windows = {
            scale: deque(maxlen=scale.max_tokens)
            for scale in RecursiveMatryoshkaScale
        }

        self.snapshots = []
        self.event_callbacks = []
        self.total_words = 0

    def on_snapshot(self, callback: Callable):
        self.event_callbacks.append(callback)

    async def stream_analyze(
        self,
        text_stream: AsyncIterator[str]
    ) -> AsyncIterator[Dict]:
        """Stream analyze with recursive composition."""
        last_snapshot = time.time()

        async for token in text_stream:
            # Add to all windows
            for window in self.windows.values():
                window.append(token)

            self.total_words += 1

            # Analyze recursively (bottom-up)
            await self._analyze_recursive()

            # Snapshot
            now = time.time()
            if now - last_snapshot >= self.snapshot_interval:
                snapshot = self._create_snapshot()

                for cb in self.event_callbacks:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(snapshot)
                    else:
                        cb(snapshot)

                yield snapshot
                last_snapshot = now

        # Final
        final = self._create_snapshot()
        for cb in self.event_callbacks:
            if asyncio.iscoroutinefunction(cb):
                await cb(final)
            else:
                cb(final)
        yield final

    async def _analyze_recursive(self):
        """Analyze all scales with recursive composition (bottom-up)."""

        # 1. WORD LEVEL (base case, no inheritance)
        if len(self.windows[RecursiveMatryoshkaScale.WORD]) > 0:
            text = " ".join(list(self.windows[RecursiveMatryoshkaScale.WORD]))
            embedding = self.embedder.encode_base([text])[0][:96]
            semantic = self._project_to_semantic(embedding, self.spectrum_word)

            self.current_states[RecursiveMatryoshkaScale.WORD] = RecursiveState(
                embedding=embedding,
                semantic=semantic,
                inherited_semantics={},
                timestamp=time.time(),
                token_count=len(self.windows[RecursiveMatryoshkaScale.WORD])
            )

        # 2. PHRASE LEVEL (inherits from word)
        if len(self.windows[RecursiveMatryoshkaScale.PHRASE]) >= 5:
            text = " ".join(list(self.windows[RecursiveMatryoshkaScale.PHRASE]))
            embedding = self.embedder.encode_base([text])[0][:192]

            # Get word-level semantic
            word_semantic = None
            if RecursiveMatryoshkaScale.WORD in self.current_states:
                word_semantic = self.current_states[RecursiveMatryoshkaScale.WORD].semantic

            # Fuse word understanding into phrase embedding
            fused_embedding = self.fusion.fuse_for_phrase(embedding, word_semantic)

            # Project fused embedding to semantic space
            semantic = self._project_to_semantic(fused_embedding, self.spectrum_phrase)

            self.current_states[RecursiveMatryoshkaScale.PHRASE] = RecursiveState(
                embedding=embedding,
                semantic=semantic,
                inherited_semantics={'word': word_semantic} if word_semantic is not None else {},
                timestamp=time.time(),
                token_count=len(self.windows[RecursiveMatryoshkaScale.PHRASE])
            )

        # 3. SENTENCE LEVEL (inherits from phrase and word)
        if len(self.windows[RecursiveMatryoshkaScale.SENTENCE]) >= 15:
            text = " ".join(list(self.windows[RecursiveMatryoshkaScale.SENTENCE]))
            embedding = self.embedder.encode_base([text])[0][:384]

            # Get lower-level semantics
            phrase_semantic = None
            word_semantic = None
            if RecursiveMatryoshkaScale.PHRASE in self.current_states:
                phrase_semantic = self.current_states[RecursiveMatryoshkaScale.PHRASE].semantic
            if RecursiveMatryoshkaScale.WORD in self.current_states:
                word_semantic = self.current_states[RecursiveMatryoshkaScale.WORD].semantic

            # Fuse
            fused_embedding = self.fusion.fuse_for_sentence(
                embedding, phrase_semantic, word_semantic
            )

            semantic = self._project_to_semantic(fused_embedding, self.spectrum_sentence)

            inherited = {}
            if phrase_semantic is not None:
                inherited['phrase'] = phrase_semantic
            if word_semantic is not None:
                inherited['word'] = word_semantic

            self.current_states[RecursiveMatryoshkaScale.SENTENCE] = RecursiveState(
                embedding=embedding,
                semantic=semantic,
                inherited_semantics=inherited,
                timestamp=time.time(),
                token_count=len(self.windows[RecursiveMatryoshkaScale.SENTENCE])
            )

        # 4. PARAGRAPH LEVEL (inherits from all)
        if len(self.windows[RecursiveMatryoshkaScale.PARAGRAPH]) >= 50:
            text = " ".join(list(self.windows[RecursiveMatryoshkaScale.PARAGRAPH]))
            embedding = self.embedder.encode_base([text])[0][:384]

            # Get all lower-level semantics
            sentence_semantic = None
            phrase_semantic = None
            word_semantic = None
            if RecursiveMatryoshkaScale.SENTENCE in self.current_states:
                sentence_semantic = self.current_states[RecursiveMatryoshkaScale.SENTENCE].semantic
            if RecursiveMatryoshkaScale.PHRASE in self.current_states:
                phrase_semantic = self.current_states[RecursiveMatryoshkaScale.PHRASE].semantic
            if RecursiveMatryoshkaScale.WORD in self.current_states:
                word_semantic = self.current_states[RecursiveMatryoshkaScale.WORD].semantic

            # Fuse all
            fused_embedding = self.fusion.fuse_for_paragraph(
                embedding, sentence_semantic, phrase_semantic, word_semantic
            )

            semantic = self._project_to_semantic(fused_embedding, self.spectrum_paragraph)

            inherited = {}
            if sentence_semantic is not None:
                inherited['sentence'] = sentence_semantic
            if phrase_semantic is not None:
                inherited['phrase'] = phrase_semantic
            if word_semantic is not None:
                inherited['word'] = word_semantic

            self.current_states[RecursiveMatryoshkaScale.PARAGRAPH] = RecursiveState(
                embedding=embedding,
                semantic=semantic,
                inherited_semantics=inherited,
                timestamp=time.time(),
                token_count=len(self.windows[RecursiveMatryoshkaScale.PARAGRAPH])
            )

    def _project_to_semantic(
        self,
        embedding: np.ndarray,
        spectrum: SemanticSpectrum
    ) -> np.ndarray:
        """Project embedding to semantic space."""
        # Handle different embedding sizes due to fusion
        # Truncate or pad to match spectrum's expected dimension
        expected_dim = spectrum.dimensions[0].axis.shape[0] if spectrum.dimensions[0].axis is not None else len(embedding)

        if len(embedding) > expected_dim:
            embedding = embedding[:expected_dim]
        elif len(embedding) < expected_dim:
            embedding = np.pad(embedding, (0, expected_dim - len(embedding)))

        semantic_proj = spectrum.project_vector(embedding)
        return np.array([semantic_proj[dim.name] for dim in spectrum.dimensions])

    def _create_snapshot(self) -> Dict:
        """Create snapshot with recursive information."""
        snapshot = {
            'timestamp': time.time(),
            'word_count': self.total_words,
            'states': {},
            'inheritance_depth': 0
        }

        for scale, state in self.current_states.items():
            snapshot['states'][scale.scale_name] = {
                'semantic_dim': len(state.semantic),
                'inherited_from': list(state.inherited_semantics.keys()),
                'dominant_dims': self._get_dominant_dims(state.semantic)[:3]
            }
            # Track max inheritance depth
            depth = len(state.inherited_semantics)
            snapshot['inheritance_depth'] = max(snapshot['inheritance_depth'], depth)

        return snapshot

    def _get_dominant_dims(self, semantic: np.ndarray) -> List[str]:
        """Get dimension names with highest magnitudes."""
        # This is a placeholder - would need to track which spectrum
        top_indices = np.argsort(np.abs(semantic))[-5:][::-1]
        return [f"dim_{i}" for i in top_indices]


async def demonstrate_recursive():
    """Demonstrate recursive composition."""
    print("🪆🔄 RECURSIVE MATRYOSHKA STREAMING")
    print("=" * 80)
    print("   Each level embeds previous levels' understanding")
    print("=" * 80)
    print()

    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

    calculator = RecursiveMatryoshkaCalculus(
        matryoshka_embedder=embedder,
        fusion_strategy=FusionStrategy.WEIGHTED_SUM,
        snapshot_interval=1.5
    )

    text = """
    The hero began his journey in darkness. Fear gripped him. But courage emerged.
    He crossed the threshold into the unknown. Tests came immediately. Some he passed.
    Others he failed. But each failure taught him. Allies appeared when needed. The
    supreme ordeal awaited. In the deepest darkness he found his true power. The reward
    was hard-won. Now he returns transformed. The elixir of wisdom he brings to others.
    """

    def on_snapshot(snapshot):
        print(f"\n📊 Word {snapshot['word_count']}:")
        print(f"   Inheritance depth: {snapshot['inheritance_depth']}")

        for scale_name, info in snapshot['states'].items():
            inherited = ", ".join(info['inherited_from']) if info['inherited_from'] else "none"
            print(f"   {scale_name:12} ({info['semantic_dim']:3}D) ← inherits from: {inherited}")
            if info['dominant_dims']:
                print(f"                  Active: {', '.join(info['dominant_dims'][:2])}")

    calculator.on_snapshot(on_snapshot)

    async def word_stream():
        for word in text.split():
            yield word
            await asyncio.sleep(0.1)

    print("🚀 Starting recursive analysis...\n")

    async for snapshot in calculator.stream_analyze(word_stream()):
        pass

    print("\n" + "=" * 80)
    print("✅ RECURSIVE COMPOSITION DEMONSTRATED")
    print("=" * 80)
    print("""
    🪆 Key insight: TRUE COMPOSITIONAL SEMANTICS

    Unlike traditional approaches that analyze each scale independently:
    ✅ Word semantics feed into phrase understanding
    ✅ Phrase semantics feed into sentence understanding
    ✅ All lower levels inform paragraph understanding

    This mirrors human comprehension:
    - We don't re-analyze words when reading paragraphs
    - We COMPOSE: word meaning → phrase meaning → sentence meaning
    - Each level BUILDS on previous levels

    Result: More coherent, context-aware semantic understanding!
    """)
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(demonstrate_recursive())