"""Regression tests: the learning loop must actually close.

Before this change, ``WeavingOrchestrator.apply_learning_signals()`` logged the
signals but never changed any state — so "reflection" was telemetry, not
learning. These tests don't require torch: they exercise the bandit-update
math on the live ``TSBandit`` against ``LearningSignal`` inputs that match
what the reflection buffer actually emits.
"""

from __future__ import annotations

import logging
import types

import pytest

# The policy package eagerly imports torch via NeuralCore, so these tests are
# skipped in torch-free environments. The learning-loop logic itself is pure
# numpy — torch only shows up through the policy package's __init__.
pytest.importorskip("torch")

from hololoom.policy.thompson_sampling import TSBandit
from hololoom.reflection.buffer import LearningSignal


def _stub_orchestrator(bandit: TSBandit | None = None):
    """Construct a minimal object with the shape ``_apply_bandit_update`` expects.

    We can't instantiate WeavingOrchestrator in a torch-free environment, so
    we build a types.SimpleNamespace with the attributes the method reads.
    """
    from hololoom.weaving_orchestrator import WeavingOrchestrator

    stub = types.SimpleNamespace(
        _persistent_bandit=bandit,
        cfg=types.SimpleNamespace(
            bandit_strategy=TSBandit(n_arms=4).strategy,
            epsilon=0.1,
        ),
        logger=logging.getLogger("test-stub"),
        _BANDIT_TOOLS=WeavingOrchestrator._BANDIT_TOOLS,
    )
    # Bind the unbound method to the stub so `self` resolves correctly.
    stub._apply_bandit_update = types.MethodType(
        WeavingOrchestrator._apply_bandit_update, stub
    )
    stub._apply_pattern_preference = types.MethodType(
        WeavingOrchestrator._apply_pattern_preference, stub
    )
    return stub


class TestBanditUpdateClosesTheLoop:
    def test_positive_reward_increments_success(self):
        bandit = TSBandit(n_arms=4)
        before = float(bandit.success[1])  # "search"
        stub = _stub_orchestrator(bandit)

        applied = stub._apply_bandit_update(
            LearningSignal(
                signal_type="bandit_update",
                tool="search",
                reward=0.9,  # centered +0.8 → success
            )
        )

        assert applied is True
        assert float(bandit.success[1]) == pytest.approx(before + 0.8, abs=1e-6)

    def test_negative_reward_increments_failure(self):
        bandit = TSBandit(n_arms=4)
        before = float(bandit.fail[0])  # "answer"
        stub = _stub_orchestrator(bandit)

        applied = stub._apply_bandit_update(
            LearningSignal(
                signal_type="bandit_update",
                tool="answer",
                reward=0.1,  # centered -0.8 → failure
            )
        )

        assert applied is True
        assert float(bandit.fail[0]) > before

    def test_unknown_tool_is_ignored(self):
        bandit = TSBandit(n_arms=4)
        before_success = bandit.success.copy()
        before_fail = bandit.fail.copy()
        stub = _stub_orchestrator(bandit)

        applied = stub._apply_bandit_update(
            LearningSignal(
                signal_type="bandit_update",
                tool="not_a_real_tool",
                reward=0.9,
            )
        )

        assert applied is False
        assert (bandit.success == before_success).all()
        assert (bandit.fail == before_fail).all()

    def test_bandit_lazy_initializes_on_first_signal(self):
        stub = _stub_orchestrator(bandit=None)  # no persistent bandit yet

        applied = stub._apply_bandit_update(
            LearningSignal(
                signal_type="bandit_update",
                tool="calc",
                reward=0.75,
            )
        )

        assert applied is True
        assert stub._persistent_bandit is not None
        assert stub._persistent_bandit.n_arms == 4

    def test_missing_reward_is_ignored(self):
        stub = _stub_orchestrator(TSBandit(n_arms=4))

        applied = stub._apply_bandit_update(
            LearningSignal(
                signal_type="bandit_update",
                tool="search",
                reward=None,
            )
        )

        assert applied is False


class TestPatternPreference:
    def test_switches_default_pattern(self):
        from hololoom.loom.command import PatternCard

        stub = _stub_orchestrator(TSBandit(n_arms=4))
        stub.default_pattern = None

        applied = stub._apply_pattern_preference(
            LearningSignal(signal_type="pattern_preference", pattern="fast")
        )

        assert applied is True
        assert stub.default_pattern == PatternCard.FAST

    def test_idempotent_when_already_set(self):
        from hololoom.loom.command import PatternCard

        stub = _stub_orchestrator(TSBandit(n_arms=4))
        stub.default_pattern = PatternCard.FAST

        applied = stub._apply_pattern_preference(
            LearningSignal(signal_type="pattern_preference", pattern="fast")
        )

        assert applied is False  # no-op, no false positive

    def test_unknown_pattern_does_not_switch(self):
        stub = _stub_orchestrator(TSBandit(n_arms=4))
        stub.default_pattern = "sentinel"

        applied = stub._apply_pattern_preference(
            LearningSignal(signal_type="pattern_preference", pattern="not-a-pattern")
        )

        assert applied is False
        assert stub.default_pattern == "sentinel"
