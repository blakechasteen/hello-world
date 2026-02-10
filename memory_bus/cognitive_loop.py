"""
Cognitive loop — 2026-02-10

One full reasoning cycle with Memory Bus in the middle:

    user question
        → prime (memory_query with aggressive_prime)
        → Claude API call (system prompt + primed context + tools)
        → model reasons, optionally calls memory tools
        → capture response
        → memory_verify (entity grounding check)
        → audit log
        → print results

Usage:
    # With live databases + ANTHROPIC_API_KEY
    python -m memory_bus.cognitive_loop "What treatment should I apply to Aurora next?"

    # Dry-run (no Claude API, prints what would be sent)
    python -m memory_bus.cognitive_loop --dry-run "Tell me about Aurora's varroa history"

Requires:
    docker compose -f docker-compose.memory.yml up -d
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-...
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (Claude tool_use format)
# ---------------------------------------------------------------------------

MEMORY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "memory_query",
        "description": (
            "Query the Memory Bus for entities, episodes, and context. "
            "Structured-first routing: exact ID → name resolution → graph → semantic stub. "
            "Returns formatted context with token budget awareness."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Free-text query (used for graph/semantic fallback)",
                },
                "entity_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Direct entity IDs to fetch",
                },
                "entity_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Entity names to resolve and fetch",
                },
                "episode_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["observation", "action", "decision", "plan", "inference", "other"],
                    },
                    "description": "Filter by episode type",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max items to return (default 20)",
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["title", "summary", "compact", "full"],
                    "description": "Detail level for formatted output (default compact)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "memory_store",
        "description": (
            "Store new entities, episodes, aliases, or plans in memory. "
            "Write-through by default (immediate persistence)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Entity objects to store",
                },
                "episodes": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Episode objects to store",
                },
                "aliases": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Entity alias mappings to store",
                },
            },
            "required": [],
        },
    },
    {
        "name": "memory_resolve_entity",
        "description": (
            "Resolve a name to ranked entity candidates. "
            "Checks aliases first, then Neo4j name search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Entity name to resolve",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "memory_explain",
        "description": (
            "Return provenance chain for a memory item. "
            "Shows where data came from and audit trail."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "string",
                    "description": "ID of the memory item to explain",
                },
            },
            "required": ["item_id"],
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a beekeeping assistant with access to a durable Memory Bus.

Your memory is external, not in your weights. When the user asks about hives,
treatments, inspections, or plans, you MUST query the Memory Bus first.

Rules:
1. Before answering any factual question about hives, call memory_query with
   entity_names for any hive or entity mentioned.
2. If memory returns results, ground your answer in those results. Cite specific
   episodes (dates, observations) when possible.
3. If memory returns no results or a semantic stub warning, say so honestly.
   Do NOT make up hive data from your training data.
4. If the user mentions a new observation or action, offer to store it using
   memory_store.
5. You can call memory_resolve_entity to disambiguate names (e.g., "Aurora"
   could be "Hive Aurora" or a queen name).

Your goal is zero confabulation about specific hive data. General beekeeping
knowledge from your training is fine, but specific hive facts must come from memory.
"""


# ---------------------------------------------------------------------------
# Loop result
# ---------------------------------------------------------------------------

@dataclass
class LoopResult:
    """Full result of one cognitive loop iteration."""
    loop_id: str
    question: str
    # Priming phase
    primed_entity_names: list[str]
    primed_items_count: int
    primed_context: str
    priming_ms: float
    # Model response
    model_response: str
    tool_calls_made: list[dict[str, Any]]
    model_ms: float
    # Verification
    verification: dict[str, Any]
    verify_ms: float
    # Totals
    total_ms: float
    # Audit summary
    audit_actions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Entity name extraction (simple heuristic for priming)
# ---------------------------------------------------------------------------

def extract_likely_entity_names(question: str) -> list[str]:
    """Pull capitalized phrases from the question for priming."""
    import re
    pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
    stopwords = {
        "What", "When", "Where", "How", "Why", "Who", "Which",
        "Tell", "Show", "Give", "Can", "Should", "Would", "Could",
        "The", "This", "That", "These", "Those", "There",
    }
    matches = pattern.findall(question)
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        m = m.strip()
        if m in stopwords or m.lower() in seen:
            continue
        seen.add(m.lower())
        result.append(m)
    return result


# ---------------------------------------------------------------------------
# Tool executor (bridges Claude tool_use → MemoryBus methods)
# ---------------------------------------------------------------------------

async def execute_tool_call(
    bus,
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    loom_id: str,
    loop_id: str,
) -> Any:
    """Execute a single tool call against the MemoryBus."""
    if tool_name == "memory_query":
        result = await bus.memory_query(
            text=tool_input.get("text", ""),
            entity_ids=tool_input.get("entity_ids"),
            entity_names=tool_input.get("entity_names"),
            episode_types=tool_input.get("episode_types"),
            max_results=tool_input.get("max_results", 20),
            detail_level=tool_input.get("detail_level", "compact"),
            loom_id=loom_id,
            loop_id=loop_id,
        )
        return {
            "items_count": len(result.items),
            "resolution_path": result.resolution_path.value,
            "formatted_context": result.formatted_context,
            "warnings": result.warnings,
            "semantic_stub_reached": result.semantic_stub_reached,
        }

    elif tool_name == "memory_store":
        result = await bus.memory_store(
            entities=tool_input.get("entities"),
            episodes=tool_input.get("episodes"),
            aliases=tool_input.get("aliases"),
            loom_id=loom_id,
            loop_id=loop_id,
        )
        return result

    elif tool_name == "memory_resolve_entity":
        result = await bus.memory_resolve_entity(
            name=tool_input["name"],
            loom_id=loom_id,
            loop_id=loop_id,
        )
        return {
            "query_name": result.query_name,
            "candidates": [
                {
                    "entity_id": c.entity_id,
                    "name": c.name,
                    "confidence": c.confidence,
                    "source": c.source,
                }
                for c in result.candidates
            ],
            "best_match": (
                {
                    "entity_id": result.best_match.entity_id,
                    "name": result.best_match.name,
                    "confidence": result.best_match.confidence,
                }
                if result.best_match
                else None
            ),
        }

    elif tool_name == "memory_explain":
        result = await bus.memory_explain(
            item_id=tool_input["item_id"],
            loom_id=loom_id,
            loop_id=loop_id,
        )
        return {
            "item_id": result.item_id,
            "confidence": result.confidence,
            "status": result.status,
            "derived_from": result.derived_from,
        }

    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Main cognitive loop
# ---------------------------------------------------------------------------

async def run_cognitive_loop(
    question: str,
    *,
    bus=None,
    loom_id: str = "demo",
    dry_run: bool = False,
    max_tool_rounds: int = 5,
) -> LoopResult:
    """
    Execute one full cognitive loop:

    1. Extract entity names from question
    2. Prime: query Memory Bus with aggressive_prime=True
    3. Build messages: system + primed context + user question
    4. Call Claude API (with tool definitions)
    5. Handle tool_use responses (loop until text response or max rounds)
    6. Verify: run memory_verify on final response
    7. Return LoopResult with full provenance
    """
    loop_id = f"loop_{uuid.uuid4().hex[:12]}"
    t_start = time.monotonic()

    # ---- Phase 1: Extract entity names from question -----------------------
    entity_names = extract_likely_entity_names(question)
    logger.info("Extracted entity names: %s", entity_names)

    # ---- Phase 2: Prime ----------------------------------------------------
    t_prime = time.monotonic()
    primed_context = ""
    primed_items_count = 0
    primed_entity_ids: list[str] = []

    if bus and entity_names:
        try:
            prime_result = await bus.memory_query(
                text=question,
                entity_names=entity_names,
                detail_level="compact",
                aggressive_prime=True,
                loom_id=loom_id,
                loop_id=loop_id,
            )
            primed_context = prime_result.formatted_context
            primed_items_count = len(prime_result.items)
            primed_entity_ids = [
                item.id for item in prime_result.items if item.kind == "entity"
            ]
            if prime_result.warnings:
                logger.warning("Priming warnings: %s", prime_result.warnings)
        except Exception as exc:
            logger.error("Priming failed: %s", exc)

    priming_ms = (time.monotonic() - t_prime) * 1000.0

    # ---- Phase 3: Build messages -------------------------------------------
    user_content = question
    if primed_context:
        user_content = (
            f"<memory_context>\n{primed_context}\n</memory_context>\n\n"
            f"Question: {question}"
        )

    messages = [{"role": "user", "content": user_content}]

    # ---- Phase 4: Call Claude API ------------------------------------------
    t_model = time.monotonic()
    model_response = ""
    tool_calls_made: list[dict[str, Any]] = []

    if dry_run:
        # Dry-run: print what would be sent, produce synthetic response
        print("\n--- DRY RUN: Claude API request ---")
        print(f"System: {SYSTEM_PROMPT[:200]}...")
        print(f"User: {user_content[:300]}...")
        print(f"Tools: {[t['name'] for t in MEMORY_TOOLS]}")
        model_response = (
            f"[DRY RUN] I would answer '{question}' using the primed context "
            f"({primed_items_count} items). Entity names detected: {entity_names}."
        )
    else:
        try:
            import anthropic

            client = anthropic.Anthropic()

            for round_idx in range(max_tool_rounds):
                response = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    tools=MEMORY_TOOLS,
                    messages=messages,
                )

                # Process response content blocks
                text_parts: list[str] = []
                has_tool_use = False

                for block in response.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        has_tool_use = True
                        tool_name = block.name
                        tool_input = block.input

                        logger.info(
                            "Tool call [round %d]: %s(%s)",
                            round_idx + 1,
                            tool_name,
                            json.dumps(tool_input, default=str)[:200],
                        )

                        # Execute the tool
                        tool_result = await execute_tool_call(
                            bus,
                            tool_name,
                            tool_input,
                            loom_id=loom_id,
                            loop_id=loop_id,
                        )

                        tool_calls_made.append({
                            "round": round_idx + 1,
                            "tool": tool_name,
                            "input": tool_input,
                            "result_summary": _summarize_tool_result(tool_result),
                        })

                        # Add assistant message + tool result to conversation
                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": json.dumps(tool_result, default=str),
                                }
                            ],
                        })

                if text_parts and not has_tool_use:
                    model_response = "\n".join(text_parts)
                    break
                elif text_parts and has_tool_use:
                    # Model returned text + tool_use; continue for tool results
                    pass
                elif not has_tool_use:
                    model_response = response.content[0].text if response.content else ""
                    break
            else:
                # Exhausted rounds — take whatever text we have
                model_response = "[Max tool rounds reached] " + " ".join(text_parts)

        except ImportError:
            logger.error("anthropic package not installed: pip install anthropic")
            model_response = "[ERROR] anthropic package not installed"
        except Exception as exc:
            logger.error("Claude API call failed: %s", exc)
            model_response = f"[ERROR] API call failed: {exc}"

    model_ms = (time.monotonic() - t_model) * 1000.0

    # ---- Phase 5: Verify ---------------------------------------------------
    t_verify = time.monotonic()
    verification: dict[str, Any] = {}

    if bus and model_response and not model_response.startswith("[ERROR]"):
        try:
            verification = await bus.memory_verify(
                response_text=model_response,
                primed_entity_ids=primed_entity_ids,
                primed_entity_names=entity_names,
                loom_id=loom_id,
                loop_id=loop_id,
            )
        except Exception as exc:
            logger.error("Verification failed: %s", exc)
            verification = {"error": str(exc)}

    verify_ms = (time.monotonic() - t_verify) * 1000.0
    total_ms = (time.monotonic() - t_start) * 1000.0

    return LoopResult(
        loop_id=loop_id,
        question=question,
        primed_entity_names=entity_names,
        primed_items_count=primed_items_count,
        primed_context=primed_context,
        priming_ms=priming_ms,
        model_response=model_response,
        tool_calls_made=tool_calls_made,
        model_ms=model_ms,
        verification=verification,
        verify_ms=verify_ms,
        total_ms=total_ms,
    )


def _summarize_tool_result(result: Any) -> str:
    """Short summary of a tool result for logging."""
    if isinstance(result, dict):
        if "items_count" in result:
            return f"{result['items_count']} items via {result.get('resolution_path', '?')}"
        if "best_match" in result:
            bm = result["best_match"]
            return f"resolved → {bm['name'] if bm else 'no match'}"
        if "error" in result:
            return f"error: {result['error']}"
    return str(result)[:100]


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_loop_result(result: LoopResult) -> None:
    """Print a structured summary of the cognitive loop result."""
    print(f"\n{'='*64}")
    print(f"  COGNITIVE LOOP RESULT — {result.loop_id}")
    print(f"{'='*64}")

    print(f"\n  Question: {result.question}")

    print(f"\n--- Phase 1: Priming ---")
    print(f"  Entity names detected: {result.primed_entity_names}")
    print(f"  Items primed: {result.primed_items_count}")
    print(f"  Priming latency: {result.priming_ms:.1f}ms")
    if result.primed_context:
        ctx_preview = result.primed_context[:300]
        if len(result.primed_context) > 300:
            ctx_preview += "..."
        print(f"  Context preview:\n    {ctx_preview}")

    print(f"\n--- Phase 2: Model Response ---")
    print(f"  Tool calls: {len(result.tool_calls_made)}")
    for tc in result.tool_calls_made:
        print(f"    [{tc['round']}] {tc['tool']} → {tc['result_summary']}")
    print(f"  Model latency: {result.model_ms:.1f}ms")
    print(f"  Response:")
    # Indent response
    for line in result.model_response.split("\n"):
        print(f"    {line}")

    print(f"\n--- Phase 3: Verification ---")
    if result.verification:
        v = result.verification
        print(f"  Total mentions: {v.get('total_mentions', 0)}")
        print(f"  Grounded: {v.get('grounded_count', 0)}")
        print(f"  Ungrounded: {v.get('ungrounded_count', 0)}")
        print(f"  Confabulation rate: {v.get('confabulation_rate', 0):.2%}")
        print(f"  False positive count: {v.get('false_positive_count', 0)}")
        if v.get("ungrounded"):
            print(f"  Ungrounded entities:")
            for u in v["ungrounded"][:5]:
                print(f"    - {u['text']}")
        if v.get("grounded"):
            print(f"  Grounded entities:")
            for g in v["grounded"][:5]:
                print(f"    - {g['text']} (via {g['grounded_via']})")
    else:
        print("  (skipped)")
    print(f"  Verify latency: {result.verify_ms:.1f}ms")

    print(f"\n--- Timing ---")
    print(f"  Prime: {result.priming_ms:>8.1f}ms")
    print(f"  Model: {result.model_ms:>8.1f}ms")
    print(f"  Verify: {result.verify_ms:>7.1f}ms")
    print(f"  Total: {result.total_ms:>8.1f}ms")

    print(f"\n{'='*64}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """CLI entry point for cognitive loop."""
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    if dry_run:
        args.remove("--dry-run")

    question = " ".join(args) if args else "What treatment should I apply to Aurora next?"

    if dry_run:
        # Dry-run: no database connections needed
        result = await run_cognitive_loop(question, dry_run=True)
        print_loop_result(result)
        return

    # Live mode: connect to databases
    from memory_bus.tools import MemoryBus
    from memory_bus.demo import seed_data

    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    pg_dsn = os.environ.get("PG_DSN", "postgresql://hololoom:hololoom@localhost:5432/hololoom")

    async with MemoryBus(neo4j_uri=neo4j_uri, pg_dsn=pg_dsn) as bus:
        # Seed data if this is a fresh database
        await seed_data(bus)

        # Run the cognitive loop
        result = await run_cognitive_loop(
            question,
            bus=bus,
            loom_id="demo",
        )
        print_loop_result(result)

        # Print audit trail for this loop
        print("--- Audit Trail (this loop) ---")
        rows = await bus._pg.get_audit_rows(loop_id=result.loop_id, limit=20)
        for row in rows:
            print(
                f"  [{row.action.value:>8}] "
                f"path={row.resolution_path.value if row.resolution_path else '-':>12} "
                f"tokens={row.tokens_used:>5} "
                f"tier={row.pressure_tier.value}"
            )

        # Print confabulation stats
        print("\n--- Confabulation Stats (all loops) ---")
        stats = await bus.confabulation_stats(loom_id="demo")
        print(f"  Total verifications: {stats['total_verifications']}")
        print(f"  Avg confabulation rate: {stats['avg_confabulation_rate']:.2%}")
        print(f"  Avg grounding rate: {stats['avg_grounding_rate']:.2%}")
        print()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
