"""
Agent introspection tools.

    tabula_rasa  — same soul, no memory (fresh perspective)
    consult      — wrap an agent as a pipeline Pass (uses tabula_rasa internally)

These tools treat agents as composable primitives. An agent can consult
another agent as part of its pipeline without side effects on the
consulted agent's memory.

Naming: tabula rasa = blank slate. Same identity (soul + backend),
no experience (memory). Nature without nurture.
"""
from __future__ import annotations

import logging
from dataclasses import replace

from .draft import Agent, Draft, Pass
from .models import ChatRequest

logger = logging.getLogger(__name__)


# ============================================================================
# NullMemory — remembers nothing
# ============================================================================

class NullMemory:
    """Memory that remembers nothing. For tabula rasa agents."""

    max_turns = 0

    def get_messages(self, room_id: str):
        return []

    def add_turn(self, room_id: str, user_msg: str, assistant_msg: str):
        pass

    def room_count(self):
        return 0

    def turn_count(self, room_id: str):
        return 0

    def total_turns(self):
        return 0


# ============================================================================
# tabula_rasa — same soul, no memory
# ============================================================================

def tabula_rasa(agent: Agent) -> Agent:
    """
    Same soul and backend, no memory. Fresh perspective.

    The returned agent thinks like the original but remembers nothing —
    no conversation history loaded, no turns stored.

    Useful for:
    - Consultation without side effects (via consult())
    - One-shot queries to an agent's personality
    - Testing an agent's reasoning without context
    """
    return replace(agent, memory=NullMemory(), name=f"{agent.name}:tabula_rasa")


# ============================================================================
# consult — ask another agent, as a Pass
# ============================================================================

def consult(agent: Agent, as_role: str | None = None) -> Pass:
    """
    Wrap an agent as a pipeline Pass.

    Uses tabula_rasa internally — the consulted agent thinks like itself
    but doesn't remember the consultation. No side effects on the
    consulted agent's memory.

    The consultation response is:
    - Stored in draft.metadata["consultations"][role]
    - Appended to draft.messages so the next think() sees it

    Usage in a pipeline:
        pipeline = [think(fast), consult(elle), think(quality)]
    """
    fresh = tabula_rasa(agent)
    role = as_role or agent.name

    async def _consult(draft: Draft) -> Draft:
        resp = await fresh(ChatRequest(text=draft.text, room_id=draft.room_id))

        # Store for programmatic access
        consultations = draft.metadata.setdefault("consultations", {})
        consultations[role] = resp.response

        # Inject into messages so the next think() sees it
        draft.messages.append({
            "role": "user",
            "content": f"[Consultation from {role}]:\n{resp.response}",
        })

        return draft

    _consult.__pass_name__ = f"consult({role})"
    return _consult
