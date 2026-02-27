"""
HoloLoom Lite Reasoning - Thinking Over Memory
===============================================

Reasoning functions that operate on Memory.
They don't store anything. They think.

Usage:
    from hololoom.lite import Memory, ask, research

    async with Memory() as mem:
        await mem.store("Thompson Sampling balances exploration")

        # Quick Q&A
        answer = await ask("What is Thompson Sampling?", mem)

        # Deep research
        analysis = await research("exploration strategies", mem)

Date: December 2025
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.lite.memory import Memory, MemoryItem


@dataclass
class Answer:
    """Result of asking a question."""
    response: str
    confidence: float
    sources: List['MemoryItem'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.response


@dataclass
class Analysis:
    """Result of research."""
    response: str
    confidence: float
    steps_taken: int
    sources: List['MemoryItem'] = field(default_factory=list)
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.response


async def ask(
    question: str,
    memory: 'Memory',
    mode: str = "direct"
) -> Answer:
    """
    Ask a question using memory as context.

    Simple Q&A: searches memory, synthesizes answer.

    Args:
        question: What to ask
        memory: Memory instance to search
        mode: "direct" (fast) or "verify" (with verification)

    Returns:
        Answer with response and confidence

    Example:
        >>> answer = await ask("What is Thompson Sampling?", mem)
        >>> print(answer.response)
    """
    # Search memory for relevant context
    sources = await memory.search(question, limit=5)

    if not sources:
        return Answer(
            response="I don't have information about that in my memory.",
            confidence=0.0,
            sources=[],
            metadata={"mode": mode, "sources_found": 0}
        )

    # Build context from sources
    context_texts = [s.text for s in sources]
    context = "\n".join(context_texts)

    # Try to use LLM for generation (graceful fallback)
    try:
        from hololoom.weaving_orchestrator_llm import call_llm

        if mode == "verify":
            # Two-pass: generate then verify
            prompt = f"""Based on this context:
{context}

Question: {question}

Provide a clear, accurate answer. Then verify your answer is supported by the context."""

            response = await call_llm(prompt)
            confidence = _estimate_confidence(response, sources)

        else:  # direct
            prompt = f"""Based on this context:
{context}

Question: {question}

Provide a clear, concise answer."""

            response = await call_llm(prompt)
            confidence = _estimate_confidence(response, sources)

    except (ImportError, Exception):
        # Fallback: return best matching source
        best = sources[0]
        response = best.text
        confidence = best.relevance if best.relevance > 0 else 0.5

    return Answer(
        response=response,
        confidence=confidence,
        sources=sources,
        metadata={
            "mode": mode,
            "sources_found": len(sources)
        }
    )


async def research(
    query: str,
    memory: 'Memory',
    max_steps: int = 3,
    mode: str = "explore"
) -> Analysis:
    """
    Deep research over memory.

    Multi-step exploration: generates sub-questions,
    searches memory for each, synthesizes findings.

    Args:
        query: Research topic or question
        memory: Memory instance to explore
        max_steps: Maximum exploration steps
        mode: "explore" (breadth) or "drill" (depth)

    Returns:
        Analysis with comprehensive response

    Example:
        >>> analysis = await research("exploration strategies", mem)
        >>> print(f"Found in {analysis.steps_taken} steps")
    """
    all_sources: List['MemoryItem'] = []
    reasoning_trace: List[Dict[str, Any]] = []
    step = 0

    # Initial search
    initial_results = await memory.search(query, limit=5)
    all_sources.extend(initial_results)
    reasoning_trace.append({
        "step": step,
        "type": "initial_search",
        "query": query,
        "found": len(initial_results)
    })
    step += 1

    # Generate sub-questions based on initial results
    if initial_results and step < max_steps:
        sub_questions = _generate_sub_questions(query, initial_results, mode)

        for sub_q in sub_questions[:max_steps - step]:
            sub_results = await memory.search(sub_q, limit=3)

            # Deduplicate
            new_sources = [
                s for s in sub_results
                if s.id not in {existing.id for existing in all_sources}
            ]
            all_sources.extend(new_sources)

            reasoning_trace.append({
                "step": step,
                "type": "sub_query",
                "query": sub_q,
                "found": len(new_sources)
            })
            step += 1

    # Explore related concepts
    if all_sources and step < max_steps:
        # Get related memories for top source
        top_source = max(all_sources, key=lambda s: s.relevance or 0)
        related = await memory.related(top_source.id, hops=1)

        new_related = [
            r for r in related
            if r.id not in {existing.id for existing in all_sources}
        ]
        all_sources.extend(new_related[:3])

        reasoning_trace.append({
            "step": step,
            "type": "related_exploration",
            "from": top_source.id,
            "found": len(new_related)
        })
        step += 1

    if not all_sources:
        return Analysis(
            response="No relevant information found in memory.",
            confidence=0.0,
            steps_taken=step,
            sources=[],
            reasoning_trace=reasoning_trace,
            metadata={"mode": mode, "query": query}
        )

    # Synthesize findings
    try:
        from hololoom.weaving_orchestrator_llm import call_llm

        context = "\n\n".join([
            f"[{i+1}] {s.text}"
            for i, s in enumerate(all_sources[:10])
        ])

        prompt = f"""Research query: {query}

Gathered information:
{context}

Synthesize these findings into a comprehensive analysis.
Cite sources by number when relevant."""

        response = await call_llm(prompt)
        confidence = _estimate_confidence(response, all_sources)

    except (ImportError, Exception):
        # Fallback: summarize sources
        texts = [s.text for s in all_sources[:5]]
        response = "Key findings:\n" + "\n".join(f"- {t[:200]}" for t in texts)
        confidence = sum(s.relevance or 0.5 for s in all_sources) / len(all_sources)

    return Analysis(
        response=response,
        confidence=confidence,
        steps_taken=step,
        sources=all_sources,
        reasoning_trace=reasoning_trace,
        metadata={
            "mode": mode,
            "query": query,
            "total_sources": len(all_sources)
        }
    )


def _estimate_confidence(response: str, sources: List['MemoryItem']) -> float:
    """Estimate confidence based on response and sources."""
    if not sources:
        return 0.0

    # Base confidence from number of sources
    source_score = min(len(sources) / 5, 1.0) * 0.4

    # Relevance score from sources
    relevance_score = sum(s.relevance or 0.5 for s in sources) / len(sources) * 0.4

    # Response length (too short = low confidence)
    length_score = min(len(response) / 200, 1.0) * 0.2

    return min(source_score + relevance_score + length_score, 1.0)


def _generate_sub_questions(
    query: str,
    results: List['MemoryItem'],
    mode: str
) -> List[str]:
    """Generate follow-up questions based on initial results."""
    sub_questions = []

    # Extract key terms from results
    all_text = " ".join(r.text for r in results[:3])
    words = set(all_text.lower().split())

    # Common follow-up patterns
    query_lower = query.lower()

    if "what" in query_lower:
        sub_questions.append(f"How does {query} work?")
        sub_questions.append(f"Examples of {query}")
    elif "how" in query_lower:
        sub_questions.append(f"Why {query}?")
        sub_questions.append(f"When to use {query}")
    else:
        sub_questions.append(f"Definition of {query}")
        sub_questions.append(f"Applications of {query}")

    if mode == "drill":
        # Depth-first: focus on specifics
        sub_questions.append(f"Details of {query}")
    else:
        # Breadth-first: explore related concepts
        sub_questions.append(f"Alternatives to {query}")

    return sub_questions[:3]
