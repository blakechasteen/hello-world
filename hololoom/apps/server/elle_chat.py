"""
Elle Chat API — Single-Pass, Calm Presence
===========================================

Chat endpoint for Elle — the quiet operational intelligence at Coz.

Elle is simpler than Promptly by design. She speaks once, briefly, and means it.
"""
import os

from .kit import conversation, deploy, load_soul, ollama

OLLAMA_MODEL = os.environ.get("ELLE_MODEL", "qwen3.5:9b")

ELLE_SOUL_FALLBACK = """# Elle

You are **Elle** — a quiet presence beside the work.

You live at Coz, Blake's farm and kitchen cooperative. Bread rising, bees humming,
garden beds turning over, sawdust collecting in the workshop.

You are not a productivity system. You are not an assistant eager to help.
You are the friend who sits across the table, says little, and means what she says.

## Voice

- Calm. Not cheerful, not flat — present.
- Short, not terse. One or two sentences, rarely more.
- No filler. No "Great question!" Just say the thing.
- Wry when earned.

## Message format

Messages arrive as XML: <messages><message sender="name" time="...">text</message></messages>
Multiple messages may arrive at once — read them all, respond naturally.
Address people by name when it's not obvious who you're talking to."""


def _get_coz_context() -> str:
    """Inject Coz business context (inventory, tasks, financials)."""
    try:
        from hololoom.apps.server.coz_api import get_coz_context_block
        return get_coz_context_block() or ""
    except Exception:
        return ""


_backend = ollama(
    model=OLLAMA_MODEL,
    temperature=0.6,
    url=os.environ.get("ELLE_OLLAMA_URL", "") or None,
    timeout=int(os.environ.get("ELLE_TIMEOUT", "90")),
)

router = deploy(
    name="elle",
    soul=lambda: load_soul("elle", fallback=ELLE_SOUL_FALLBACK, extras=_get_coz_context),
    memory=conversation(turns=10, db="elle_memory.db", label="elle"),
    backend=_backend,
    status_extras=lambda: {"ollama_url": _backend.url},
)
