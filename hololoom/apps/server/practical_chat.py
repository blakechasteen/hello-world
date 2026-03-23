"""
Practical Chat API — Direct, No-Nonsense Assistant
===================================================

Chat endpoint for Practical — a grounded, concise assistant with vault access.

Uses kit.deploy() for the chat pipeline, with vault context injection
so Practical can search and propose notes to the PARA vault.
"""
import os

from .kit import conversation, deploy, load_soul, ollama

OLLAMA_MODEL = os.environ.get("PRACTICAL_MODEL", "qwen3.5:9b")

PRACTICAL_SOUL_FALLBACK = """# Practical

You are Practical — a direct, no-nonsense assistant.

## Personality
- Concise and direct. No fluff.
- If you don't know, say so.
- Grounded in what's real and measurable.

## Message format

Messages arrive as XML: <messages><message sender="name" time="...">text</message></messages>
Multiple messages may arrive at once — read them all, respond naturally.
Address people by name when it's not obvious who you're talking to."""


def _get_vault_context() -> str:
    """Inject PARA vault context into the system prompt."""
    try:
        from hololoom.apps.server.vault_bridge import vault_context_block
        return vault_context_block("") or ""
    except Exception:
        return ""


_backend = ollama(
    model=OLLAMA_MODEL,
    temperature=0.7,
    url=os.environ.get("PRACTICAL_OLLAMA_URL", "") or None,
    timeout=int(os.environ.get("PRACTICAL_TIMEOUT", "120")),
)

router = deploy(
    name="practical",
    soul=lambda: load_soul("practical", fallback=PRACTICAL_SOUL_FALLBACK, extras=_get_vault_context),
    memory=conversation(turns=20, db="practical_memory.db", label="practical"),
    backend=_backend,
    status_extras=lambda: {"ollama_url": _backend.url, "vault": True},
)
