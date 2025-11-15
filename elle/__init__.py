"""Elle: AR guide for unfolding work.

A quiet, observant presence that helps you see what you're looking at
and decide what to do next.

Architecture:
    adapters → engine → core → tools → infra

Layers:
    - Interface Adapters: AR / Matrix / CLI
    - Orchestrator: ElleEngine routes requests
    - Core: Policy + prompting + LLM calls
    - Domain & Services: Models, memory, tools
    - Infrastructure: Config, logging, persistence

Key principles:
    - LLM is policy, not glue
    - Event in → Decision → Command out
    - Stateless per-request, stateful via memory
    - Everything is replaceable

See ELLE_ARCHITECTURE.md for complete design.
"""

__version__ = "0.1.0"

from .domain import (
    SceneSnapshot,
    UserIntent,
    UserState,
    ElleAction,
    ElleRequest,
)
from .engine import ElleEngine, create_request
from .core import EllePolicy, PromptBuilder, create_llm_client
from .memory import MemoryStore, MemorySnapshot, InMemoryMemoryStore
from .tools import ToolRegistry, create_default_registry
from .infra import ElleConfig

__all__ = [
    # Domain
    'SceneSnapshot',
    'UserIntent',
    'UserState',
    'ElleAction',
    'ElleRequest',
    
    # Engine
    'ElleEngine',
    'create_request',
    
    # Core
    'EllePolicy',
    'PromptBuilder',
    'create_llm_client',
    
    # Memory
    'MemoryStore',
    'MemorySnapshot',
    'InMemoryMemoryStore',
    
    # Tools
    'ToolRegistry',
    'create_default_registry',
    
    # Config
    'ElleConfig',
]
