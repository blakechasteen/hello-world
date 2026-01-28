"""
MemoryLLM - An LLM that remembers. Simple as that.

Usage:
    llm = MemoryLLM(lambda p: your_model(p))

    llm("What is Thompson Sampling?")
    llm("Tell me more")  # Remembers previous context!

    # Or memory-only
    mem = MemoryLLM()
    mem.remember("Important fact about X")
    mem.recall("X")  # Returns relevant memories

Author: HoloLoom
Date: December 2025
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable


@dataclass
class Memory:
    """A stored memory."""
    text: str
    timestamp: float = field(default_factory=time.time)
    embedding: Any = None
    score: float = 0.0


class MemoryLLM:
    """
    An LLM that remembers. Just call it.

    Examples:
        # With any LLM
        llm = MemoryLLM(lambda p: ollama.generate(model='llama3.2', prompt=p)['response'])
        print(llm("What is Python?"))
        print(llm("Give me an example"))  # Uses memory!

        # Memory-only mode
        mem = MemoryLLM()
        mem.remember("Python is a programming language")
        mem.remember("Python uses indentation")
        results = mem.recall("programming")
    """

    def __init__(
        self,
        llm: Optional[Callable[[str], str]] = None,
        system: str = "You are a helpful assistant with memory of our conversation.",
        max_context: int = 5,
    ):
        """
        Args:
            llm: Function(prompt) -> response. None for memory-only.
            system: System prompt.
            max_context: Max memories to include.
        """
        self.llm = llm
        self.system = system
        self.max_context = max_context

        self._memories: List[Memory] = []
        self._history: List[tuple] = []  # (role, content)
        self._embedder = None
        self._init_embedder()

    def _init_embedder(self):
        """Try to load embedder for semantic search."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            pass  # Fall back to keyword matching

    def __call__(self, message: str) -> str:
        """Chat with memory. Just call the object."""
        # Get relevant memories
        memories = self.recall(message)

        # Build prompt
        prompt = self._build_prompt(message, memories)

        # Generate
        if self.llm:
            try:
                response = self.llm(prompt)
            except Exception as e:
                response = f"[Error: {e}]"
        else:
            response = f"[Memory-only mode. Found {len(memories)} relevant memories.]"

        # Remember this exchange
        self.remember(f"User asked: {message}")
        if self.llm:
            self.remember(f"I answered: {response[:200]}...")

        # Track history
        self._history.append(("user", message))
        self._history.append(("assistant", response))

        return response

    def remember(self, text: str, metadata: Optional[Dict] = None) -> None:
        """Store a memory."""
        embedding = None
        if self._embedder:
            try:
                embedding = self._embedder.encode(text)
            except Exception:
                pass

        self._memories.append(Memory(
            text=text,
            embedding=embedding,
        ))

    def recall(self, query: str, k: Optional[int] = None) -> List[Memory]:
        """Find relevant memories."""
        if not self._memories:
            return []

        k = k or self.max_context

        # Score all memories
        scored = []
        if self._embedder:
            import numpy as np
            q_emb = self._embedder.encode(query)
            for mem in self._memories:
                if mem.embedding is not None:
                    sim = float(np.dot(q_emb, mem.embedding) / (
                        np.linalg.norm(q_emb) * np.linalg.norm(mem.embedding) + 1e-8
                    ))
                    scored.append((mem, sim))
                else:
                    scored.append((mem, 0.0))
        else:
            # Keyword matching
            q_words = set(query.lower().split())
            for mem in self._memories:
                m_words = set(mem.text.lower().split())
                overlap = len(q_words & m_words) / max(len(q_words), 1)
                scored.append((mem, overlap))

        # Sort and filter
        scored.sort(key=lambda x: x[1], reverse=True)
        result = []
        for mem, score in scored[:k]:
            if score > 0.1:
                mem.score = score
                result.append(mem)

        return result

    def _build_prompt(self, message: str, memories: List[Memory]) -> str:
        """Build prompt with context."""
        parts = [self.system]

        if memories:
            ctx = "\n".join(f"- {m.text}" for m in memories)
            parts.append(f"\nRelevant context:\n{ctx}")

        # Recent history (last 3 exchanges)
        if self._history:
            recent = self._history[-6:]
            hist = "\n".join(f"{r}: {c[:100]}" for r, c in recent)
            parts.append(f"\nRecent:\n{hist}")

        parts.append(f"\nUser: {message}\nAssistant:")
        return "\n".join(parts)

    def clear(self):
        """Clear all memory and history."""
        self._memories.clear()
        self._history.clear()

    @property
    def memories(self) -> List[Memory]:
        """Access all memories."""
        return self._memories

    def __repr__(self):
        return f"MemoryLLM({len(self._memories)} memories, llm={'yes' if self.llm else 'no'})"


# =============================================================================
# Quick helpers
# =============================================================================

def with_ollama(model: str = "llama3.2:3b") -> Callable[[str], str]:
    """Ollama wrapper."""
    import ollama
    return lambda p: ollama.generate(model=model, prompt=p)['response']


def with_openai(model: str = "gpt-4") -> Callable[[str], str]:
    """OpenAI wrapper."""
    from openai import OpenAI
    client = OpenAI()
    return lambda p: client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": p}]
    ).choices[0].message.content


def with_anthropic(model: str = "claude-3-5-sonnet-20241022") -> Callable[[str], str]:
    """Anthropic wrapper."""
    import anthropic
    client = anthropic.Anthropic()
    return lambda p: client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": p}]
    ).content[0].text
