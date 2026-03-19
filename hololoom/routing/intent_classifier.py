"""
Intent Classifier — Keyword-based query intent classification.

Classifies queries into primary intent categories for model routing
and downstream processing.

Extracted from apps/server/model_router.py for reuse.
"""

from enum import Enum


class Intent(str, Enum):
    CHAT = "chat"           # Casual conversation, greetings
    CODE = "code"           # Code generation, debugging, review
    REASONING = "reasoning" # Multi-step logic, math, analysis
    FACTUAL = "factual"     # Lookup, recall, Q&A
    CREATIVE = "creative"   # Writing, brainstorming, ideation
    PLANNING = "planning"   # Architecture, design, project planning


# Keyword → intent mapping. Cheap, fast, good enough for routing.
_INTENT_KEYWORDS: dict[Intent, list[str]] = {
    Intent.CODE: [
        "code", "function", "class", "def ", "import ", "bug", "error",
        "debug", "refactor", "implement", "compile", "syntax", "```",
        "typescript", "python", "javascript", "rust", "sql", "api",
        "endpoint", "docker", "git", "test", "lint", "type",
    ],
    Intent.REASONING: [
        "why", "how does", "explain", "prove", "analyze", "compare",
        "difference between", "tradeoff", "trade-off", "cause", "because",
        "therefore", "logic", "math", "calculate", "derive", "evaluate",
    ],
    Intent.PLANNING: [
        "plan", "architect", "design", "roadmap", "strategy", "structure",
        "organize", "breakdown", "steps to", "approach", "migrate", "refactor",
        "restructure", "sequence", "priority", "timeline",
    ],
    Intent.CREATIVE: [
        "write", "story", "poem", "brainstorm", "imagine", "creative",
        "generate", "draft", "compose", "describe", "narrative",
        "name for", "ideas for", "suggest",
    ],
    Intent.FACTUAL: [
        "what is", "who is", "when did", "where is", "define", "list",
        "how many", "which", "lookup", "find", "search",
    ],
    Intent.CHAT: [
        "hi", "hello", "hey", "thanks", "ok", "yes", "no", "lol",
        "haha", "cool", "nice", "sup", "yo", "good morning",
        "good night", "how are you",
    ],
}


def classify_intent(text: str) -> tuple[Intent, float]:
    """
    Classify query into primary intent + confidence.

    Returns (intent, confidence) where confidence is 0.0-1.0.
    """
    lower = text.lower().strip()
    scores: dict[Intent, float] = dict.fromkeys(Intent, 0.0)

    for intent, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                scores[intent] += 1.0

    # Length-based adjustments
    word_count = len(lower.split())
    if word_count <= 5:
        scores[Intent.CHAT] += 1.5
    if word_count > 50:
        scores[Intent.REASONING] += 0.5
        scores[Intent.PLANNING] += 0.5

    # Code block detection (strong signal)
    if "```" in text or text.count("\n") > 5:
        scores[Intent.CODE] += 3.0

    total = sum(scores.values())
    if total == 0:
        return Intent.CHAT, 0.3  # Default to chat with low confidence

    best_intent = max(scores, key=scores.get)  # type: ignore
    confidence = scores[best_intent] / total
    return best_intent, min(1.0, confidence)
