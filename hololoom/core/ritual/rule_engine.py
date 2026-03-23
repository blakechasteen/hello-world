"""
PatternCard — JIT rule activation for ritual execution.
=========================================================

Jacquard cards encode per-fabric instructions. PatternCards are loaded
based on what fabric is being woven. They activate domain-specific
behavioral rules at dispatch time.

Rules are YAML. The engine is code. Rules shape retrieval (via recall/
exclude keywords) and guide body behavior (via ctx.active_rules).

Rules are in-body context, NOT dispatch gates. BindingObligation gates
dispatch (structural). PatternCard guides execution (behavioral).
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rule:
    """A Jacquard card — behavioral instruction for a specific fabric."""
    id: str
    domain: str                        # "coding" | "beekeeping" | "universal"
    text: str                          # the rule (human-readable)
    keywords: list[str] = field(default_factory=list)
    recall: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    always_on: bool = False
    weight: float = 1.0
    confidence_below: float | None = None


@dataclass
class ActiveRule:
    """A rule that matched the current context."""
    rule: Rule
    activation_score: float
    triggered_by: str = ""             # keyword that triggered activation


@dataclass
class ActiveRuleSet:
    """The rules active for a specific dispatch."""
    rules: list[ActiveRule] = field(default_factory=list)
    token_cost: int = 0

    @property
    def recall_keywords(self) -> list[str]:
        """All recall keywords from active rules — shapes ThreadPrimer retrieval."""
        kws = []
        for ar in self.rules:
            kws.extend(ar.rule.recall)
        return kws

    @property
    def exclude_keywords(self) -> list[str]:
        """All exclude keywords from active rules — ThreadPrimer skips these."""
        kws = []
        for ar in self.rules:
            kws.extend(ar.rule.exclude)
        return kws


class RuleEngine:
    """
    JIT rule activation based on dispatch context.

    Loads rules programmatically or from YAML. At dispatch time,
    matches context against keywords/domain/confidence and returns
    the active rule set.
    """

    def __init__(self) -> None:
        self._rules: list[Rule] = []

    def register(self, rule: Rule) -> None:
        """Register a rule."""
        self._rules.append(rule)

    def register_many(self, rules: list[Rule]) -> None:
        """Register multiple rules."""
        self._rules.extend(rules)

    def activate(self, ctx: Any) -> ActiveRuleSet:
        """
        Determine which rules are active for this context.

        1. Always-on rules always included
        2. Domain match: rule.domain == ctx.domain or "universal"
        3. Keyword match: any rule.keywords in ctx.task_intent
        4. Confidence gate: rule.confidence_below and ctx.confidence < threshold
        5. Exclude: any rule.exclude keyword in task_intent → skip
        6. Sort by activation_score descending
        """
        active: list[ActiveRule] = []
        intent_lower = ctx.task_intent.lower() if hasattr(ctx, "task_intent") else ""
        domain = ctx.domain if hasattr(ctx, "domain") else "universal"
        confidence = ctx.confidence if hasattr(ctx, "confidence") else 1.0

        for rule in self._rules:
            # Domain filter
            if rule.domain != "universal" and rule.domain != domain:
                continue

            # Exclude check
            excluded = any(
                kw.lower() in intent_lower for kw in rule.exclude
            )
            if excluded:
                continue

            # Always-on
            if rule.always_on:
                active.append(ActiveRule(
                    rule=rule,
                    activation_score=rule.weight,
                    triggered_by="always_on",
                ))
                continue

            # Keyword match
            score = 0.0
            triggered_by = ""
            for kw in rule.keywords:
                if kw.lower() in intent_lower:
                    score += rule.weight
                    triggered_by = kw
                    break  # one match is enough

            # Confidence gate
            if rule.confidence_below is not None and confidence < rule.confidence_below:
                score += rule.weight * 0.5
                if not triggered_by:
                    triggered_by = f"confidence<{rule.confidence_below}"

            if score > 0:
                active.append(ActiveRule(
                    rule=rule,
                    activation_score=score,
                    triggered_by=triggered_by,
                ))

        # Sort by score descending
        active.sort(key=lambda r: r.activation_score, reverse=True)

        # Estimate token cost (~10 tokens per rule text)
        token_cost = sum(len(ar.rule.text.split()) * 2 for ar in active)

        return ActiveRuleSet(rules=active, token_cost=token_cost)

    @property
    def rule_count(self) -> int:
        return len(self._rules)


def load_rules_from_yaml(path: str) -> list[Rule]:
    """
    Load rules from a YAML file. Returns empty list on failure.

    Expected format:
    - id: rule_name
      domain: coding
      text: "The rule text"
      keywords: [fix, refactor]
      weight: 1.0
    """
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            return []
        return [
            Rule(
                id=item.get("id", ""),
                domain=item.get("domain", "universal"),
                text=item.get("text", ""),
                keywords=item.get("keywords", []),
                recall=item.get("recall", []),
                exclude=item.get("exclude", []),
                always_on=item.get("always_on", False),
                weight=item.get("weight", 1.0),
                confidence_below=item.get("confidence_below"),
            )
            for item in data
            if isinstance(item, dict) and "id" in item
        ]
    except Exception as e:
        logger.warning("Failed to load rules from %s: %s", path, e)
        return []
