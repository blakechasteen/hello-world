"""Tests for PatternCard — JIT rule activation."""

import pytest
import os

from hololoom.core.ritual.rule_engine import (
    Rule, ActiveRule, ActiveRuleSet, RuleEngine, load_rules_from_yaml,
)
from hololoom.core.ritual.types import RitualContext, TokenBudget


def _make_ctx(**overrides) -> RitualContext:
    defaults = dict(
        agent_id="agent-1", agent_role="coder", domain="coding",
        task_id="t-1", task_type="code_modify", task_intent="fix the login bug",
        task_complexity="standard", yarn=None, warp=None,
        confidence=0.8, confidence_history=[0.8],
        ritual_outputs={}, ritual_log=[],
        token_budget=TokenBudget(total=5000, used=0),
        time_budget=60.0, escalation_path=[], session_id="sess-1",
    )
    defaults.update(overrides)
    return RitualContext(**defaults)


CODING_RULE = Rule(
    id="coding_test",
    domain="coding",
    text="Run tests after changes.",
    keywords=["fix", "refactor", "implement"],
    weight=1.0,
)

UNIVERSAL_RULE = Rule(
    id="always_degrade",
    domain="universal",
    text="Every dependency is optional.",
    always_on=True,
    weight=2.0,
)

BEEKEEPING_RULE = Rule(
    id="bee_inspect",
    domain="beekeeping",
    text="Check for queen presence.",
    keywords=["inspect", "hive"],
    weight=1.0,
)

CONFIDENCE_RULE = Rule(
    id="low_conf",
    domain="universal",
    text="Be cautious when confidence is low.",
    confidence_below=0.5,
    weight=1.0,
)

EXCLUDE_RULE = Rule(
    id="no_chat",
    domain="coding",
    text="Don't apply coding rules to chat.",
    keywords=["fix", "code"],
    exclude=["chat", "question"],
    weight=1.0,
)


class TestRuleEngine:
    def test_empty_engine(self):
        engine = RuleEngine()
        ctx = _make_ctx()
        result = engine.activate(ctx)
        assert result.rules == []
        assert result.token_cost == 0

    def test_always_on_activates(self):
        engine = RuleEngine()
        engine.register(UNIVERSAL_RULE)
        ctx = _make_ctx()
        result = engine.activate(ctx)
        assert len(result.rules) == 1
        assert result.rules[0].rule.id == "always_degrade"
        assert result.rules[0].triggered_by == "always_on"

    def test_keyword_match(self):
        engine = RuleEngine()
        engine.register(CODING_RULE)
        ctx = _make_ctx(task_intent="fix the login bug")
        result = engine.activate(ctx)
        assert len(result.rules) == 1
        assert result.rules[0].triggered_by == "fix"

    def test_keyword_no_match(self):
        engine = RuleEngine()
        engine.register(CODING_RULE)
        ctx = _make_ctx(task_intent="explain how auth works")
        result = engine.activate(ctx)
        assert result.rules == []

    def test_domain_filter(self):
        engine = RuleEngine()
        engine.register(BEEKEEPING_RULE)
        ctx = _make_ctx(domain="coding", task_intent="inspect the hive")
        result = engine.activate(ctx)
        assert result.rules == []  # wrong domain

    def test_domain_match(self):
        engine = RuleEngine()
        engine.register(BEEKEEPING_RULE)
        ctx = _make_ctx(domain="beekeeping", task_intent="inspect the hive")
        result = engine.activate(ctx)
        assert len(result.rules) == 1

    def test_universal_domain_always_matches(self):
        engine = RuleEngine()
        engine.register(UNIVERSAL_RULE)
        ctx = _make_ctx(domain="beekeeping")
        result = engine.activate(ctx)
        assert len(result.rules) == 1

    def test_confidence_gate_activates_below_threshold(self):
        engine = RuleEngine()
        engine.register(CONFIDENCE_RULE)
        ctx = _make_ctx(confidence=0.3)
        result = engine.activate(ctx)
        assert len(result.rules) == 1

    def test_confidence_gate_inactive_above_threshold(self):
        engine = RuleEngine()
        engine.register(CONFIDENCE_RULE)
        ctx = _make_ctx(confidence=0.8)
        result = engine.activate(ctx)
        assert result.rules == []

    def test_exclude_keyword_blocks(self):
        engine = RuleEngine()
        engine.register(EXCLUDE_RULE)
        ctx = _make_ctx(task_intent="fix the chat handler")
        result = engine.activate(ctx)
        assert result.rules == []  # "chat" in exclude

    def test_exclude_not_triggered(self):
        engine = RuleEngine()
        engine.register(EXCLUDE_RULE)
        ctx = _make_ctx(task_intent="fix the auth module")
        result = engine.activate(ctx)
        assert len(result.rules) == 1

    def test_sorted_by_score(self):
        engine = RuleEngine()
        engine.register(Rule(id="low", domain="coding", text="low", keywords=["fix"], weight=0.5))
        engine.register(Rule(id="high", domain="coding", text="high", keywords=["fix"], weight=5.0))
        ctx = _make_ctx(task_intent="fix it")
        result = engine.activate(ctx)
        assert result.rules[0].rule.id == "high"
        assert result.rules[1].rule.id == "low"

    def test_register_many(self):
        engine = RuleEngine()
        engine.register_many([CODING_RULE, UNIVERSAL_RULE])
        assert engine.rule_count == 2

    def test_token_cost_estimated(self):
        engine = RuleEngine()
        engine.register(CODING_RULE)
        ctx = _make_ctx(task_intent="fix the bug")
        result = engine.activate(ctx)
        assert result.token_cost > 0


class TestActiveRuleSet:
    def test_recall_keywords(self):
        rule = Rule(id="r1", domain="coding", text="t", recall=["spring_dynamics", "test"])
        ruleset = ActiveRuleSet(rules=[ActiveRule(rule=rule, activation_score=1.0)])
        assert "spring_dynamics" in ruleset.recall_keywords
        assert "test" in ruleset.recall_keywords

    def test_exclude_keywords(self):
        rule = Rule(id="r1", domain="coding", text="t", exclude=["weather"])
        ruleset = ActiveRuleSet(rules=[ActiveRule(rule=rule, activation_score=1.0)])
        assert "weather" in ruleset.exclude_keywords

    def test_empty_ruleset(self):
        ruleset = ActiveRuleSet()
        assert ruleset.recall_keywords == []
        assert ruleset.exclude_keywords == []


class TestLoadRulesFromYaml:
    def test_load_coding_rules(self):
        yaml_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "core", "ritual", "rules", "coding.yaml",
        )
        if not os.path.exists(yaml_path):
            pytest.skip("YAML rules not found")
        rules = load_rules_from_yaml(yaml_path)
        assert len(rules) >= 3
        assert any(r.id == "coding_test_after_change" for r in rules)

    def test_load_nonexistent_returns_empty(self):
        rules = load_rules_from_yaml("/nonexistent/path.yaml")
        assert rules == []
