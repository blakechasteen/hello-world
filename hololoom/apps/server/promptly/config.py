"""
Promptly configuration constants.

All tunables live here so sibling modules can import them without
pulling in heavy dependencies.
"""
from __future__ import annotations

import os
from pathlib import Path

# ============================================================================
# Ollama / LLM
# ============================================================================

_raw_ollama_host = os.environ.get("PROMPTLY_OLLAMA_URL", "").strip()
if not _raw_ollama_host:
    _raw_ollama_host = "http://100.66.145.78:11434"
OLLAMA_URL = _raw_ollama_host if _raw_ollama_host.startswith("http") else f"http://{_raw_ollama_host}"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:30b-155k")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

# llama-server (llama.cpp HTTP server)
_raw_llamacpp_url = os.environ.get("LLAMACPP_URL", "").strip()
if not _raw_llamacpp_url:
    _raw_llamacpp_url = "http://127.0.0.1:8080"
LLAMACPP_URL = _raw_llamacpp_url if _raw_llamacpp_url.startswith("http") else f"http://{_raw_llamacpp_url}"
MAX_HISTORY_TURNS = 20  # Per-room rolling window

# ============================================================================
# Multi-pass refinement
# ============================================================================

REFINEMENT_ENABLED = os.environ.get("PROMPTLY_REFINEMENT", "true").lower() == "true"
REFINEMENT_TIMEOUT = int(os.environ.get("PROMPTLY_REFINEMENT_TIMEOUT", "60"))
REFINEMENT_TTL = 300  # Seconds to keep refinement results before cleanup
MAX_PASSES = int(os.environ.get("PROMPTLY_MAX_PASSES", "4"))  # Including pass 1
DEFAULT_MAX_TOKENS = int(os.environ.get("PROMPTLY_MAX_TOKENS", "8192"))

# ============================================================================
# Jenny visualization
# ============================================================================

JENNY_ENABLED = os.environ.get("PROMPTLY_JENNY", "true").lower() == "true"
JENNY_CONVERSATION_ENABLED = os.environ.get(
    "PROMPTLY_JENNY_CONVERSATION", "true"
).lower() == "true"
JENNY_SPATIAL_ENABLED = os.environ.get(
    "PROMPTLY_JENNY_SPATIAL", ""
).lower() == "true"

# ============================================================================
# Paths
# ============================================================================

# Root of the promptly_chat.py file's package (server/)
_SERVER_DIR = Path(__file__).resolve().parent.parent
# Root of the mythRL repo
_REPO_ROOT = _SERVER_DIR.parent.parent.parent

MEMORY_DB_PATH = os.environ.get(
    "PROMPTLY_MEMORY_DB",
    str(_REPO_ROOT / "data" / "promptly_memory.db"),
)

# Conversation graph GC timeout (seconds)
GRAPH_IDLE_TIMEOUT = 30 * 60  # 30 minutes
