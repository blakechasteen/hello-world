"""
Agent Kit — composable pieces for building chat agents.

    # Protocols (swap implementations freely)
    from hololoom.apps.server.kit import Memory, LLMBackend, Soul

    # Pieces
    from hololoom.apps.server.kit import (
        conversation, load_soul, ollama, emit,
        ChatRequest, ChatResponse,
    )

    # Pipeline
    from hololoom.apps.server.kit import (
        Draft, Pass, think, named, converged, loop, compose,
    )

    # Agent
    from hololoom.apps.server.kit import Agent

    # Deployment
    from hololoom.apps.server.kit import deploy, discover

    # Introspection
    from hololoom.apps.server.kit import tabula_rasa, consult, NullMemory

    # Testing (import from .testing directly for test files)
    from hololoom.apps.server.kit.testing import mock, MockBackend
"""

from .deploy import deploy, discover
from .draft import Agent, Draft, Pass, Soul, compose, converged, loop, named, think
from .emit import emit
from .introspect import NullMemory, consult, tabula_rasa
from .llm import LLMBackend, OllamaBackend, extract_content, extract_tokens, ollama
from .memory import ConversationMemory, Memory, conversation
from .models import ChatRequest, ChatResponse
from .soul import load_soul

__all__ = [
    # Protocols
    "Memory",
    "LLMBackend",
    "Soul",
    # Pieces
    "ConversationMemory",
    "conversation",
    "load_soul",
    "OllamaBackend",
    "ollama",
    "extract_content",
    "extract_tokens",
    "emit",
    "ChatRequest",
    "ChatResponse",
    # Pipeline
    "Draft",
    "Pass",
    "think",
    "named",
    "converged",
    "loop",
    "compose",
    # Agent
    "Agent",
    # Deployment
    "deploy",
    "discover",
    # Introspection
    "NullMemory",
    "tabula_rasa",
    "consult",
]
