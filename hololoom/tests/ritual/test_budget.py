"""Tests for admission control."""

from hololoom.core.ritual.budget import check_admission
from hololoom.core.ritual.types import RitualBudget, TokenBudget


class TestCheckAdmission:
    def test_sufficient_budget(self):
        ritual_budget = RitualBudget(max_tokens=200, max_seconds=10.0, priority_class="standard")
        agent_budget = TokenBudget(total=1000, used=500)
        assert check_admission(ritual_budget, agent_budget) is True

    def test_insufficient_budget(self):
        ritual_budget = RitualBudget(max_tokens=600, max_seconds=10.0, priority_class="standard")
        agent_budget = TokenBudget(total=1000, used=500)
        assert check_admission(ritual_budget, agent_budget) is False

    def test_exact_budget(self):
        ritual_budget = RitualBudget(max_tokens=500, max_seconds=10.0, priority_class="standard")
        agent_budget = TokenBudget(total=1000, used=500)
        assert check_admission(ritual_budget, agent_budget) is True

    def test_critical_always_admitted(self):
        ritual_budget = RitualBudget(max_tokens=9999, max_seconds=10.0, priority_class="critical")
        agent_budget = TokenBudget(total=100, used=99)
        assert check_admission(ritual_budget, agent_budget) is True

    def test_best_effort_budget_checked(self):
        ritual_budget = RitualBudget(max_tokens=200, max_seconds=10.0, priority_class="best_effort")
        agent_budget = TokenBudget(total=100, used=0)
        assert check_admission(ritual_budget, agent_budget) is False

    def test_zero_remaining(self):
        ritual_budget = RitualBudget(max_tokens=1, max_seconds=1.0, priority_class="standard")
        agent_budget = TokenBudget(total=100, used=100)
        assert check_admission(ritual_budget, agent_budget) is False
