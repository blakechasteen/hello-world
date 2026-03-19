"""
Unit tests for hololoom/core/reflection/ modules.

Covers:
- rewards.py: RewardConfig, RewardExtractor, compute_shaped_reward, estimate_potential
- feedback_store.py: FeedbackRecord, LearningSignals, FeedbackStore
- ppo_trainer.py: PPOConfig, PPOTrainer
- semantic_learning.py: SemanticLearningConfig, SemanticExperience,
    SemanticTrajectoryAnalyzer, SemanticCurriculumDesigner
- __init__.py: Re-exports

Target: bring hololoom/core/reflection/ from 0% to 60%+ coverage.
"""

import json
import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from hololoom.fabric.spacetime import Spacetime, WeavingTrace


# =============================================================================
# Helpers
# =============================================================================

def make_trace(
    duration_ms: float = 500.0,
    errors: list = None,
    warnings: list = None,
    threads_activated: list = None,
) -> WeavingTrace:
    now = datetime.now()
    return WeavingTrace(
        start_time=now,
        end_time=now,
        duration_ms=duration_ms,
        errors=errors or [],
        warnings=warnings or [],
        threads_activated=threads_activated or [],
    )


def make_spacetime(
    query: str = "test query",
    tool: str = "answer",
    confidence: float = 0.8,
    duration_ms: float = 500.0,
    quality_score: float = None,
    errors: list = None,
    warnings: list = None,
    threads_activated: list = None,
    pattern_card: str = "fast",
) -> Spacetime:
    trace = make_trace(
        duration_ms=duration_ms,
        errors=errors,
        warnings=warnings,
        threads_activated=threads_activated,
    )
    return Spacetime(
        query_text=query,
        response="test response",
        tool_used=tool,
        confidence=confidence,
        trace=trace,
        quality_score=quality_score,
        metadata={"pattern_card": pattern_card},
    )


# =============================================================================
# rewards.py — RewardConfig
# =============================================================================

class TestRewardConfig:
    def test_defaults(self):
        from hololoom.core.reflection.rewards import RewardConfig
        cfg = RewardConfig()
        assert cfg.base_weight == 0.6
        assert cfg.quality_weight == 0.3
        assert cfg.efficiency_weight == 0.1
        assert cfg.error_penalty == -0.5
        assert cfg.warning_penalty == -0.1
        assert cfg.timeout_penalty == -0.3
        assert cfg.min_reward == -1.0
        assert cfg.max_reward == 1.0
        assert cfg.time_budget_ms == 1000.0

    def test_custom_values(self):
        from hololoom.core.reflection.rewards import RewardConfig
        cfg = RewardConfig(base_weight=0.8, time_budget_ms=2000.0)
        assert cfg.base_weight == 0.8
        assert cfg.time_budget_ms == 2000.0


# =============================================================================
# rewards.py — RewardExtractor
# =============================================================================

class TestRewardExtractor:
    def test_default_construction(self):
        from hololoom.core.reflection.rewards import RewardExtractor
        ext = RewardExtractor()
        assert ext.config is not None

    def test_custom_config(self):
        from hololoom.core.reflection.rewards import RewardExtractor, RewardConfig
        cfg = RewardConfig(base_weight=0.9)
        ext = RewardExtractor(config=cfg)
        assert ext.config.base_weight == 0.9

    def test_compute_reward_returns_float(self):
        from hololoom.core.reflection.rewards import RewardExtractor
        ext = RewardExtractor()
        st = make_spacetime(confidence=0.8)
        reward = ext.compute_reward(st)
        assert isinstance(reward, float)

    def test_reward_in_valid_range(self):
        from hololoom.core.reflection.rewards import RewardExtractor
        ext = RewardExtractor()
        st = make_spacetime(confidence=0.9)
        reward = ext.compute_reward(st)
        assert -1.0 <= reward <= 1.0

    def test_high_confidence_higher_reward(self):
        from hololoom.core.reflection.rewards import RewardExtractor
        ext = RewardExtractor()
        high = ext.compute_reward(make_spacetime(confidence=0.95))
        low = ext.compute_reward(make_spacetime(confidence=0.1))
        assert high > low

    def test_errors_reduce_reward(self):
        from hololoom.core.reflection.rewards import RewardExtractor
        ext = RewardExtractor()
        clean = ext.compute_reward(make_spacetime(confidence=0.8))
        errored = ext.compute_reward(
            make_spacetime(confidence=0.8, errors=[{"msg": "boom"}])
        )
        assert errored < clean

    def test_warnings_reduce_reward(self):
        from hololoom.core.reflection.rewards import RewardExtractor
        ext = RewardExtractor()
        clean = ext.compute_reward(make_spacetime(confidence=0.8))
        warned = ext.compute_reward(
            make_spacetime(confidence=0.8, warnings=["warn1"])
        )
        assert warned < clean

    def test_fast_execution_gives_efficiency_bonus(self):
        from hololoom.core.reflection.rewards import RewardExtractor
        ext = RewardExtractor()
        fast = ext.compute_reward(make_spacetime(duration_ms=100.0))
        slow = ext.compute_reward(make_spacetime(duration_ms=900.0))
        assert fast >= slow

    def test_timeout_penalty_for_very_slow(self):
        from hololoom.core.reflection.rewards import RewardExtractor
        ext = RewardExtractor()
        normal = ext.compute_reward(make_spacetime(duration_ms=500.0))
        timeout = ext.compute_reward(make_spacetime(duration_ms=2500.0))
        assert timeout < normal

    def test_user_feedback_overrides_computed(self):
        from hololoom.core.reflection.rewards import RewardExtractor
        ext = RewardExtractor()
        st = make_spacetime(confidence=0.1)  # Low confidence
        feedback = {"rating": 5}  # Excellent rating
        reward = ext.compute_reward(st, user_feedback=feedback)
        assert reward == pytest.approx(1.0)

    def test_user_feedback_rating_zero(self):
        from hololoom.core.reflection.rewards import RewardExtractor
        ext = RewardExtractor()
        reward = ext.compute_reward(make_spacetime(), user_feedback={"rating": 0})
        assert reward == pytest.approx(-1.0)

    def test_user_feedback_rating_midpoint(self):
        from hololoom.core.reflection.rewards import RewardExtractor
        ext = RewardExtractor()
        reward = ext.compute_reward(make_spacetime(), user_feedback={"rating": 2.5})
        assert reward == pytest.approx(0.0)

    def test_quality_score_used_when_available(self):
        from hololoom.core.reflection.rewards import RewardExtractor
        ext = RewardExtractor()
        with_quality = ext.compute_reward(
            make_spacetime(confidence=0.5, quality_score=0.95)
        )
        without_quality = ext.compute_reward(
            make_spacetime(confidence=0.5, quality_score=None)
        )
        # With high quality_score, reward should be higher
        assert with_quality > without_quality

    def test_normalize_rating_5_point_scale(self):
        from hololoom.core.reflection.rewards import RewardExtractor
        ext = RewardExtractor()
        assert ext._normalize_rating(0) == pytest.approx(-1.0)
        assert ext._normalize_rating(5) == pytest.approx(1.0)
        assert ext._normalize_rating(2.5) == pytest.approx(0.0)

    def test_compute_components_keys(self):
        from hololoom.core.reflection.rewards import RewardExtractor
        ext = RewardExtractor()
        st = make_spacetime()
        components = ext._compute_components(st)
        expected_keys = {
            "base", "quality", "efficiency",
            "error_penalty", "warning_penalty", "timeout_penalty",
        }
        assert set(components.keys()) == expected_keys

    def test_extract_experience_structure(self):
        from hololoom.core.reflection.rewards import RewardExtractor
        ext = RewardExtractor()
        st = make_spacetime(tool="search")
        exp = ext.extract_experience(st)
        assert exp["action"] == "search"
        assert exp["done"] is True
        assert isinstance(exp["reward"], float)
        assert "observation" in exp
        assert "info" in exp

    def test_extract_experience_info_keys(self):
        from hololoom.core.reflection.rewards import RewardExtractor
        ext = RewardExtractor()
        exp = ext.extract_experience(make_spacetime())
        info = exp["info"]
        assert "confidence" in info
        assert "duration_ms" in info
        assert "num_errors" in info
        assert "num_warnings" in info


# =============================================================================
# rewards.py — compute_shaped_reward
# =============================================================================

class TestComputeShapedReward:
    def test_shaping_adds_potential_difference(self):
        from hololoom.core.reflection.rewards import compute_shaped_reward
        # F(s,a,s') = gamma * phi(s') - phi(s)
        # shaped = base + F
        result = compute_shaped_reward(
            base_reward=0.5, potential_old=0.3, potential_new=0.6, gamma=0.99
        )
        expected = 0.5 + (0.99 * 0.6 - 0.3)
        assert result == pytest.approx(expected)

    def test_zero_potentials(self):
        from hololoom.core.reflection.rewards import compute_shaped_reward
        result = compute_shaped_reward(0.5, 0.0, 0.0, 0.99)
        assert result == pytest.approx(0.5)

    def test_negative_shaping(self):
        from hololoom.core.reflection.rewards import compute_shaped_reward
        result = compute_shaped_reward(0.5, 0.8, 0.2, 0.99)
        # Shaping is negative (moved to worse state)
        assert result < 0.5


# =============================================================================
# rewards.py — estimate_potential
# =============================================================================

class TestEstimatePotential:
    def test_returns_float(self):
        from hololoom.core.reflection.rewards import estimate_potential
        st = make_spacetime()
        p = estimate_potential(st)
        assert isinstance(p, float)

    def test_high_confidence_high_potential(self):
        from hololoom.core.reflection.rewards import estimate_potential
        high = estimate_potential(make_spacetime(confidence=0.95))
        low = estimate_potential(make_spacetime(confidence=0.1))
        assert high > low

    def test_errors_reduce_potential(self):
        from hololoom.core.reflection.rewards import estimate_potential
        clean = estimate_potential(make_spacetime(confidence=0.8))
        errored = estimate_potential(
            make_spacetime(confidence=0.8, errors=[{"msg": "e1"}])
        )
        assert errored <= clean

    def test_threads_increase_potential(self):
        from hololoom.core.reflection.rewards import estimate_potential
        no_threads = estimate_potential(
            make_spacetime(threads_activated=[])
        )
        many_threads = estimate_potential(
            make_spacetime(threads_activated=[f"t{i}" for i in range(8)])
        )
        assert many_threads > no_threads


# =============================================================================
# feedback_store.py — FeedbackRecord
# =============================================================================

class TestFeedbackRecord:
    def test_construction(self):
        from hololoom.core.reflection.feedback_store import FeedbackRecord
        rec = FeedbackRecord(
            feedback_id="abc123",
            query_text="test",
            response_text="response",
            tool_used="answer",
            confidence=0.9,
            user_rating=1.0,
            feedback_type="helpful",
        )
        assert rec.feedback_id == "abc123"
        assert rec.user_id == "@unknown:matrix.org"
        assert rec.metadata == {}

    def test_to_dict(self):
        from hololoom.core.reflection.feedback_store import FeedbackRecord
        rec = FeedbackRecord(
            feedback_id="abc",
            query_text="q",
            response_text="r",
            tool_used="answer",
            confidence=0.9,
            user_rating=1.0,
            feedback_type="helpful",
        )
        d = rec.to_dict()
        assert d["feedback_id"] == "abc"
        assert d["tool_used"] == "answer"
        assert "timestamp" in d

    def test_to_dict_serializable(self):
        from hololoom.core.reflection.feedback_store import FeedbackRecord
        rec = FeedbackRecord(
            feedback_id="x",
            query_text="q",
            response_text="r",
            tool_used="t",
            confidence=0.5,
            user_rating=0.5,
            feedback_type="rating",
            metadata={"key": "val"},
        )
        d = rec.to_dict()
        json.dumps(d)  # Should not raise


# =============================================================================
# feedback_store.py — LearningSignals
# =============================================================================

class TestLearningSignals:
    def test_construction(self):
        from hololoom.core.reflection.feedback_store import LearningSignals
        now = datetime.now()
        sig = LearningSignals(
            tool_name="answer",
            total_samples=10,
            successful_samples=8,
            failed_samples=2,
            avg_rating=0.85,
            avg_confidence=0.9,
            success_rate=0.8,
            alpha=9.0,
            beta=3.0,
            expected_reward=0.75,
            window_start=now,
            window_end=now,
        )
        assert sig.tool_name == "answer"
        assert sig.alpha == 9.0

    def test_to_dict(self):
        from hololoom.core.reflection.feedback_store import LearningSignals
        now = datetime.now()
        sig = LearningSignals(
            tool_name="answer",
            total_samples=10,
            successful_samples=8,
            failed_samples=2,
            avg_rating=0.85,
            avg_confidence=0.9,
            success_rate=0.8,
            alpha=9.0,
            beta=3.0,
            expected_reward=0.75,
            window_start=now,
            window_end=now,
        )
        d = sig.to_dict()
        assert d["tool_name"] == "answer"
        assert d["total_samples"] == 10
        json.dumps(d)  # Serializable


# =============================================================================
# feedback_store.py — FeedbackStore
# =============================================================================

class TestFeedbackStore:
    def test_init_creates_parent_dir(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        db_path = tmp_path / "subdir" / "feedback.db"
        store = FeedbackStore(db_path=str(db_path))
        assert db_path.parent.exists()

    @pytest.mark.asyncio
    async def test_initialize_and_close(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        store = FeedbackStore(db_path=str(tmp_path / "test.db"))
        await store.initialize()
        assert store.db is not None
        await store.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        async with FeedbackStore(db_path=str(tmp_path / "ctx.db")) as store:
            assert store.db is not None

    @pytest.mark.asyncio
    async def test_store_feedback(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        async with FeedbackStore(db_path=str(tmp_path / "fb.db")) as store:
            fid = await store.store_feedback(
                query="What is RL?",
                response="RL is...",
                tool_used="answer",
                confidence=0.9,
                user_rating=1.0,
                feedback_type="helpful",
            )
            assert isinstance(fid, str)
            assert len(fid) == 16

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        async with FeedbackStore(db_path=str(tmp_path / "fb.db")) as store:
            await store.store_feedback(
                query="Q1", response="R1", tool_used="answer",
                confidence=0.9, user_rating=0.8, feedback_type="helpful",
            )
            await store.store_feedback(
                query="Q2", response="R2", tool_used="search",
                confidence=0.7, user_rating=0.5, feedback_type="not_helpful",
            )
            records = await store.get_feedback_history(limit=10)
            assert len(records) == 2

    @pytest.mark.asyncio
    async def test_get_feedback_history_filter_by_tool(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        async with FeedbackStore(db_path=str(tmp_path / "fb.db")) as store:
            await store.store_feedback(
                query="Q1", response="R1", tool_used="answer",
                confidence=0.9, user_rating=0.8, feedback_type="helpful",
            )
            await store.store_feedback(
                query="Q2", response="R2", tool_used="search",
                confidence=0.7, user_rating=0.5, feedback_type="helpful",
            )
            answer_recs = await store.get_feedback_history(tool="answer")
            assert len(answer_recs) == 1
            assert answer_recs[0].tool_used == "answer"

    @pytest.mark.asyncio
    async def test_get_feedback_history_filter_by_user(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        async with FeedbackStore(db_path=str(tmp_path / "fb.db")) as store:
            await store.store_feedback(
                query="Q1", response="R1", tool_used="answer",
                confidence=0.9, user_rating=0.8, feedback_type="helpful",
                user_id="@alice:local",
            )
            await store.store_feedback(
                query="Q2", response="R2", tool_used="answer",
                confidence=0.9, user_rating=0.9, feedback_type="helpful",
                user_id="@bob:local",
            )
            alice = await store.get_feedback_history(user_id="@alice:local")
            assert len(alice) == 1

    @pytest.mark.asyncio
    async def test_get_feedback_history_with_offset(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        async with FeedbackStore(db_path=str(tmp_path / "fb.db")) as store:
            for i in range(5):
                await store.store_feedback(
                    query=f"Q{i}", response=f"R{i}", tool_used="answer",
                    confidence=0.9, user_rating=0.8, feedback_type="helpful",
                )
            recs = await store.get_feedback_history(limit=2, offset=2)
            assert len(recs) == 2

    @pytest.mark.asyncio
    async def test_get_learning_signals(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        async with FeedbackStore(db_path=str(tmp_path / "fb.db")) as store:
            for i in range(12):
                await store.store_feedback(
                    query=f"Q{i}", response=f"R{i}", tool_used="answer",
                    confidence=0.85, user_rating=0.8 + (i % 3) * 0.05,
                    feedback_type="helpful",
                )
            signals = await store.get_learning_signals(tool="answer", min_samples=10)
            assert "answer" in signals
            sig = signals["answer"]
            assert sig.total_samples == 12
            assert sig.alpha >= 1
            assert sig.beta >= 1

    @pytest.mark.asyncio
    async def test_get_learning_signals_min_samples_filter(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        async with FeedbackStore(db_path=str(tmp_path / "fb.db")) as store:
            for i in range(3):
                await store.store_feedback(
                    query=f"Q{i}", response=f"R{i}", tool_used="answer",
                    confidence=0.9, user_rating=0.9, feedback_type="helpful",
                )
            signals = await store.get_learning_signals(min_samples=10)
            assert "answer" not in signals

    @pytest.mark.asyncio
    async def test_get_learning_signals_with_time_window(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        async with FeedbackStore(db_path=str(tmp_path / "fb.db")) as store:
            for i in range(5):
                await store.store_feedback(
                    query=f"Q{i}", response=f"R{i}", tool_used="answer",
                    confidence=0.9, user_rating=0.9, feedback_type="helpful",
                )
            signals = await store.get_learning_signals(
                min_samples=1, time_window=timedelta(hours=1)
            )
            assert "answer" in signals

    @pytest.mark.asyncio
    async def test_update_thompson_priors(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        async with FeedbackStore(db_path=str(tmp_path / "fb.db")) as store:
            fid = await store.store_feedback(
                query="Q", response="R", tool_used="answer",
                confidence=0.9, user_rating=0.9, feedback_type="helpful",
            )
            alpha, beta = await store.update_thompson_priors("answer", fid)
            assert alpha >= 1.0
            assert beta >= 1.0

    @pytest.mark.asyncio
    async def test_update_thompson_priors_missing_feedback(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        async with FeedbackStore(db_path=str(tmp_path / "fb.db")) as store:
            with pytest.raises(ValueError, match="not found"):
                await store.update_thompson_priors("answer", "nonexistent")

    @pytest.mark.asyncio
    async def test_update_thompson_priors_low_rating(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        async with FeedbackStore(db_path=str(tmp_path / "fb.db")) as store:
            fid = await store.store_feedback(
                query="Q", response="R", tool_used="answer",
                confidence=0.5, user_rating=0.3, feedback_type="not_helpful",
            )
            alpha, beta = await store.update_thompson_priors("answer", fid)
            # Low rating should increment beta more
            assert beta > alpha

    @pytest.mark.asyncio
    async def test_get_statistics(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        async with FeedbackStore(db_path=str(tmp_path / "fb.db")) as store:
            await store.store_feedback(
                query="Q1", response="R1", tool_used="answer",
                confidence=0.9, user_rating=0.9, feedback_type="helpful",
            )
            await store.store_feedback(
                query="Q2", response="R2", tool_used="search",
                confidence=0.7, user_rating=0.5, feedback_type="not_helpful",
            )
            stats = await store.get_statistics()
            assert stats["total_feedback_count"] == 2
            assert "answer" in stats["tool_statistics"]
            assert "search" in stats["tool_statistics"]
            assert "feedback_types" in stats

    @pytest.mark.asyncio
    async def test_store_with_metadata(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        async with FeedbackStore(db_path=str(tmp_path / "fb.db")) as store:
            fid = await store.store_feedback(
                query="Q", response="R", tool_used="answer",
                confidence=0.9, user_rating=0.9, feedback_type="helpful",
                metadata={"session": "abc"},
            )
            records = await store.get_feedback_history(limit=1)
            assert records[0].metadata == {"session": "abc"}

    @pytest.mark.asyncio
    async def test_store_with_explicit_feedback(self, tmp_path):
        from hololoom.core.reflection.feedback_store import FeedbackStore
        async with FeedbackStore(db_path=str(tmp_path / "fb.db")) as store:
            await store.store_feedback(
                query="Q", response="R", tool_used="answer",
                confidence=0.9, user_rating=1.0, feedback_type="explicit",
                explicit_feedback="Great answer!",
            )
            records = await store.get_feedback_history(limit=1)
            assert records[0].explicit_feedback == "Great answer!"


# =============================================================================
# ppo_trainer.py — PPOConfig
# =============================================================================

class TestPPOConfig:
    def test_defaults(self):
        from hololoom.core.reflection.ppo_trainer import PPOConfig
        cfg = PPOConfig()
        assert cfg.learning_rate == 3e-4
        assert cfg.clip_epsilon == 0.2
        assert cfg.value_loss_coef == 0.5
        assert cfg.entropy_coef == 0.01
        assert cfg.max_grad_norm == 0.5
        assert cfg.gamma == 0.99
        assert cfg.gae_lambda == 0.95
        assert cfg.n_epochs == 4
        assert cfg.batch_size == 64
        assert cfg.target_kl == 0.01

    def test_custom_values(self):
        from hololoom.core.reflection.ppo_trainer import PPOConfig
        cfg = PPOConfig(learning_rate=1e-3, clip_epsilon=0.3)
        assert cfg.learning_rate == 1e-3
        assert cfg.clip_epsilon == 0.3


# =============================================================================
# ppo_trainer.py — PPOTrainer
# =============================================================================

class TestPPOTrainer:
    @staticmethod
    def _make_dummy_policy():
        """Create a minimal nn.Module for testing."""
        import torch
        import torch.nn as nn

        class DummyPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(128, 5)
                self.value_head = nn.Linear(128, 1)

            def forward(self, x):
                logits = self.fc(x)
                value = self.value_head(x).squeeze(-1)
                return {"logits": logits, "value": value}

        return DummyPolicy()

    def test_construction(self):
        from hololoom.core.reflection.ppo_trainer import PPOTrainer
        policy = self._make_dummy_policy()
        trainer = PPOTrainer(policy=policy)
        assert trainer.update_count == 0
        assert trainer.total_samples == 0
        assert trainer.metrics_history == []

    def test_construction_with_config(self):
        from hololoom.core.reflection.ppo_trainer import PPOTrainer, PPOConfig
        cfg = PPOConfig(learning_rate=1e-3)
        policy = self._make_dummy_policy()
        trainer = PPOTrainer(policy=policy, config=cfg)
        assert trainer.config.learning_rate == 1e-3

    def test_get_metrics_summary_empty(self):
        from hololoom.core.reflection.ppo_trainer import PPOTrainer
        policy = self._make_dummy_policy()
        trainer = PPOTrainer(policy=policy)
        summary = trainer.get_metrics_summary()
        assert summary == {}

    def test_get_metrics_summary_with_history(self):
        from hololoom.core.reflection.ppo_trainer import PPOTrainer
        policy = self._make_dummy_policy()
        trainer = PPOTrainer(policy=policy)
        trainer.metrics_history = [
            {"policy_loss": 0.1, "value_loss": 0.2, "entropy": 0.5, "kl_divergence": 0.01},
            {"policy_loss": 0.08, "value_loss": 0.15, "entropy": 0.45, "kl_divergence": 0.008},
        ]
        summary = trainer.get_metrics_summary()
        assert "avg_policy_loss" in summary
        assert "avg_value_loss" in summary
        assert "avg_entropy" in summary

    def test_compute_advantages(self):
        import torch
        from hololoom.core.reflection.ppo_trainer import PPOTrainer, PPOConfig
        policy = self._make_dummy_policy()
        trainer = PPOTrainer(policy=policy)

        rewards = torch.tensor([0.5, 0.8, 0.3, 0.9])
        values = torch.tensor([0.4, 0.7, 0.5, 0.6])
        dones = torch.tensor([0.0, 0.0, 0.0, 1.0])

        advantages, returns = trainer._compute_advantages(rewards, values, dones)
        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape
        # Advantages should be normalized (mean ~0, std ~1)
        assert abs(advantages.mean().item()) < 0.1

    def test_compute_advantages_all_done(self):
        import torch
        from hololoom.core.reflection.ppo_trainer import PPOTrainer
        policy = self._make_dummy_policy()
        trainer = PPOTrainer(policy=policy)

        rewards = torch.tensor([0.5, 0.8])
        values = torch.tensor([0.4, 0.7])
        dones = torch.tensor([1.0, 1.0])

        advantages, returns = trainer._compute_advantages(rewards, values, dones)
        assert advantages.shape == (2,)

    def test_batch_to_tensors(self):
        import torch
        from hololoom.core.reflection.ppo_trainer import PPOTrainer
        policy = self._make_dummy_policy()
        trainer = PPOTrainer(policy=policy)

        batch = {
            "observations": [{"motifs": 1}, {"motifs": 2}],
            "actions": ["answer", "search"],
            "rewards": [0.5, 0.8],
            "dones": [True, True],
            "infos": [{}, {}],
        }
        tensors = trainer._batch_to_tensors(batch)
        assert "observations" in tensors
        assert "actions" in tensors
        assert "rewards" in tensors
        assert tensors["rewards"].shape == (2,)

    def test_ppo_update(self):
        import torch
        from hololoom.core.reflection.ppo_trainer import PPOTrainer, PPOConfig
        policy = self._make_dummy_policy()
        cfg = PPOConfig(n_epochs=2)
        trainer = PPOTrainer(policy=policy, config=cfg)

        B = 8
        observations = torch.randn(B, 128)
        actions = torch.randint(0, 5, (B,))
        log_probs_old = torch.zeros(B)
        advantages = torch.randn(B)
        returns = torch.randn(B)

        metrics = trainer._ppo_update(
            observations, actions, log_probs_old, advantages, returns
        )
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "kl_divergence" in metrics

    @pytest.mark.asyncio
    async def test_train_on_buffer_not_enough_samples(self):
        from hololoom.core.reflection.ppo_trainer import PPOTrainer
        from hololoom.core.reflection.buffer import ReflectionBuffer
        policy = self._make_dummy_policy()
        trainer = PPOTrainer(policy=policy)

        buf = ReflectionBuffer(capacity=50)
        for _ in range(5):
            await buf.store(make_spacetime(), reward=0.8)

        metrics = await trainer.train_on_buffer(buf, min_samples=32)
        assert metrics == {}

    @pytest.mark.asyncio
    async def test_train_on_buffer_with_enough_samples(self):
        from hololoom.core.reflection.ppo_trainer import PPOTrainer, PPOConfig
        from hololoom.core.reflection.buffer import ReflectionBuffer
        policy = self._make_dummy_policy()
        cfg = PPOConfig(n_epochs=1)
        trainer = PPOTrainer(policy=policy, config=cfg)

        buf = ReflectionBuffer(capacity=100)
        for _ in range(40):
            await buf.store(make_spacetime(), reward=0.8)

        metrics = await trainer.train_on_buffer(buf, min_samples=32)
        assert "policy_loss" in metrics
        assert trainer.update_count == 1
        assert trainer.total_samples == 40

    def test_save_and_load_checkpoint(self, tmp_path):
        import torch
        from hololoom.core.reflection.ppo_trainer import PPOTrainer
        policy = self._make_dummy_policy()
        trainer = PPOTrainer(policy=policy)
        trainer.update_count = 5
        trainer.total_samples = 100
        trainer.metrics_history = [{"policy_loss": 0.1, "value_loss": 0.2, "entropy": 0.3, "kl_divergence": 0.01}]

        path = str(tmp_path / "checkpoint.pt")
        trainer.save_checkpoint(path)
        assert Path(path).exists()

        # Load into new trainer
        policy2 = self._make_dummy_policy()
        trainer2 = PPOTrainer(policy=policy2)
        trainer2.load_checkpoint(path)
        assert trainer2.update_count == 5
        assert trainer2.total_samples == 100
        assert len(trainer2.metrics_history) == 1


# =============================================================================
# semantic_learning.py — SemanticLearningConfig
# =============================================================================

class TestSemanticLearningConfig:
    def test_defaults(self):
        from hololoom.core.reflection.semantic_learning import SemanticLearningConfig
        cfg = SemanticLearningConfig()
        assert cfg.enable_dimension_prediction is True
        assert cfg.enable_tool_effect_learning is True
        assert cfg.enable_goal_achievement is True
        assert cfg.enable_contrastive_pairs is True
        assert cfg.contrastive_margin == 0.5
        assert cfg.forecast_horizon == 3
        assert cfg.curriculum_stages == 5


# =============================================================================
# semantic_learning.py — SemanticExperience
# =============================================================================

class TestSemanticExperience:
    def test_construction_minimal(self):
        from hololoom.core.reflection.semantic_learning import SemanticExperience
        exp = SemanticExperience(
            observation={"key": 1},
            action="answer",
            reward=0.5,
            next_observation={"key": 2},
            done=False,
            semantic_state={"dim1": 0.5},
            semantic_velocity={"dim1": 0.1},
            semantic_categories={"cat1": 0.3},
            next_semantic_state={"dim1": 0.6},
            next_semantic_velocity={"dim1": 0.05},
            next_semantic_categories={"cat1": 0.4},
        )
        assert exp.action == "answer"
        assert exp.reward == 0.5
        assert exp.semantic_goal is None
        assert exp.tool_semantic_delta == {}
        assert exp.semantic_history == []

    def test_construction_full(self):
        from hololoom.core.reflection.semantic_learning import SemanticExperience
        exp = SemanticExperience(
            observation={},
            action="search",
            reward=0.9,
            next_observation={},
            done=True,
            semantic_state={"dim1": 0.3},
            semantic_velocity={"dim1": 0.2},
            semantic_categories={},
            next_semantic_state={"dim1": 0.8},
            next_semantic_velocity={"dim1": 0.0},
            next_semantic_categories={},
            semantic_goal={"dim1": 1.0},
            goal_alignment_before=0.3,
            goal_alignment_after=0.8,
            tool_semantic_delta={"dim1": 0.5},
            success=True,
            query_text="test",
        )
        assert exp.success is True
        assert exp.goal_alignment_after == 0.8


# =============================================================================
# semantic_learning.py — SemanticTrajectoryAnalyzer
# =============================================================================

class TestSemanticTrajectoryAnalyzer:
    @staticmethod
    def _make_experience(
        action="answer",
        success=True,
        semantic_state=None,
        next_semantic_state=None,
        semantic_velocity=None,
        semantic_goal=None,
        goal_alignment_before=0.5,
        goal_alignment_after=0.7,
    ):
        from hololoom.core.reflection.semantic_learning import SemanticExperience
        state = semantic_state or {"Complexity": 0.3, "Warmth": 0.6}
        next_state = next_semantic_state or {"Complexity": 0.5, "Warmth": 0.7}
        velocity = semantic_velocity or {"Complexity": 0.2, "Warmth": 0.1}
        return SemanticExperience(
            observation={},
            action=action,
            reward=0.8 if success else 0.2,
            next_observation={},
            done=True,
            semantic_state=state,
            semantic_velocity=velocity,
            semantic_categories={},
            next_semantic_state=next_state,
            next_semantic_velocity={},
            next_semantic_categories={},
            semantic_goal=semantic_goal,
            goal_alignment_before=goal_alignment_before,
            goal_alignment_after=goal_alignment_after,
            success=success,
        )

    def test_construction(self):
        from hololoom.core.reflection.semantic_learning import (
            SemanticTrajectoryAnalyzer, SemanticLearningConfig,
        )
        cfg = SemanticLearningConfig()
        analyzer = SemanticTrajectoryAnalyzer(config=cfg)
        assert analyzer.tool_semantic_effects is not None

    def test_analyze_experience_returns_signals(self):
        from hololoom.core.reflection.semantic_learning import (
            SemanticTrajectoryAnalyzer, SemanticLearningConfig,
        )
        analyzer = SemanticTrajectoryAnalyzer(config=SemanticLearningConfig())
        exp = self._make_experience()
        signals = analyzer.analyze_experience(exp)
        assert "tool_effect" in signals
        assert "dimension_importance" in signals

    def test_analyze_tool_effect(self):
        from hololoom.core.reflection.semantic_learning import (
            SemanticTrajectoryAnalyzer, SemanticLearningConfig,
        )
        analyzer = SemanticTrajectoryAnalyzer(config=SemanticLearningConfig())
        exp = self._make_experience(
            semantic_state={"dim1": 0.3},
            next_semantic_state={"dim1": 0.8},
        )
        signals = analyzer.analyze_experience(exp)
        assert "dim1" in signals["tool_effect"]
        assert signals["tool_effect"]["dim1"] == pytest.approx(0.5)

    def test_tool_effects_accumulated(self):
        from hololoom.core.reflection.semantic_learning import (
            SemanticTrajectoryAnalyzer, SemanticLearningConfig,
        )
        analyzer = SemanticTrajectoryAnalyzer(config=SemanticLearningConfig())
        for _ in range(3):
            exp = self._make_experience(action="search")
            analyzer.analyze_experience(exp)
        assert "search" in analyzer.tool_semantic_effects
        assert len(analyzer.tool_semantic_effects["search"]["Complexity"]) == 3

    def test_success_states_tracked(self):
        from hololoom.core.reflection.semantic_learning import (
            SemanticTrajectoryAnalyzer, SemanticLearningConfig,
        )
        analyzer = SemanticTrajectoryAnalyzer(config=SemanticLearningConfig())
        analyzer.analyze_experience(self._make_experience(success=True))
        analyzer.analyze_experience(self._make_experience(success=False))
        assert len(analyzer.successful_semantic_states) == 1
        assert len(analyzer.failed_semantic_states) == 1

    def test_goal_progress_when_goal_set(self):
        from hololoom.core.reflection.semantic_learning import (
            SemanticTrajectoryAnalyzer, SemanticLearningConfig,
        )
        analyzer = SemanticTrajectoryAnalyzer(config=SemanticLearningConfig())
        exp = self._make_experience(
            semantic_goal={"Complexity": 1.0},
            semantic_state={"Complexity": 0.3, "Warmth": 0.6},
            next_semantic_state={"Complexity": 0.5, "Warmth": 0.7},
            goal_alignment_before=0.3,
            goal_alignment_after=0.5,
        )
        signals = analyzer.analyze_experience(exp)
        assert "goal_progress" in signals
        assert signals["goal_progress"]["achieved"] is True
        assert signals["goal_progress"]["alignment_delta"] == pytest.approx(0.2)

    def test_no_goal_progress_without_goal(self):
        from hololoom.core.reflection.semantic_learning import (
            SemanticTrajectoryAnalyzer, SemanticLearningConfig,
        )
        analyzer = SemanticTrajectoryAnalyzer(config=SemanticLearningConfig())
        exp = self._make_experience(semantic_goal=None)
        signals = analyzer.analyze_experience(exp)
        assert "goal_progress" not in signals

    def test_dimension_importance_nonzero_for_active_dims(self):
        from hololoom.core.reflection.semantic_learning import (
            SemanticTrajectoryAnalyzer, SemanticLearningConfig,
        )
        analyzer = SemanticTrajectoryAnalyzer(config=SemanticLearningConfig())
        exp = self._make_experience(
            semantic_velocity={"dim1": 0.8, "dim2": 0.0},
        )
        signals = analyzer.analyze_experience(exp)
        importance = signals["dimension_importance"]
        assert importance["dim1"] > importance["dim2"]

    def test_detect_anomalies_extreme_velocity(self):
        from hololoom.core.reflection.semantic_learning import (
            SemanticTrajectoryAnalyzer, SemanticLearningConfig,
        )
        analyzer = SemanticTrajectoryAnalyzer(config=SemanticLearningConfig())
        exp = self._make_experience(
            semantic_velocity={"dim1": 2.0},  # Extreme
        )
        signals = analyzer.analyze_experience(exp)
        assert "anomalies" in signals
        assert any("Extreme velocity" in a for a in signals["anomalies"])

    def test_detect_contradictory_dimensions(self):
        from hololoom.core.reflection.semantic_learning import (
            SemanticTrajectoryAnalyzer, SemanticLearningConfig,
        )
        analyzer = SemanticTrajectoryAnalyzer(config=SemanticLearningConfig())
        exp = self._make_experience(
            semantic_state={"Complexity": 0.8, "Simplicity": 0.9},
            semantic_velocity={"Complexity": 0.0, "Simplicity": 0.0},
        )
        signals = analyzer.analyze_experience(exp)
        assert "anomalies" in signals
        assert any("Contradictory" in a for a in signals["anomalies"])

    def test_get_tool_effect_model_empty(self):
        from hololoom.core.reflection.semantic_learning import (
            SemanticTrajectoryAnalyzer, SemanticLearningConfig,
        )
        analyzer = SemanticTrajectoryAnalyzer(config=SemanticLearningConfig())
        model = analyzer.get_tool_effect_model("nonexistent")
        assert model == {}

    def test_get_tool_effect_model_with_data(self):
        from hololoom.core.reflection.semantic_learning import (
            SemanticTrajectoryAnalyzer, SemanticLearningConfig,
        )
        analyzer = SemanticTrajectoryAnalyzer(config=SemanticLearningConfig())
        for _ in range(5):
            analyzer.analyze_experience(self._make_experience(action="answer"))
        model = analyzer.get_tool_effect_model("answer")
        assert len(model) > 0
        for dim, (mean, std) in model.items():
            assert isinstance(mean, float)
            assert isinstance(std, float)

    def test_suggest_contrastive_pairs_empty(self):
        from hololoom.core.reflection.semantic_learning import (
            SemanticTrajectoryAnalyzer, SemanticLearningConfig,
        )
        analyzer = SemanticTrajectoryAnalyzer(config=SemanticLearningConfig())
        pairs = analyzer.suggest_contrastive_pairs()
        assert pairs == []

    def test_suggest_contrastive_pairs_with_data(self):
        from hololoom.core.reflection.semantic_learning import (
            SemanticTrajectoryAnalyzer, SemanticLearningConfig,
        )
        analyzer = SemanticTrajectoryAnalyzer(config=SemanticLearningConfig())
        for _ in range(5):
            analyzer.analyze_experience(self._make_experience(success=True))
            analyzer.analyze_experience(self._make_experience(success=False))
        pairs = analyzer.suggest_contrastive_pairs(min_pairs=3)
        assert len(pairs) == 3
        for success_state, failure_state in pairs:
            assert isinstance(success_state, dict)
            assert isinstance(failure_state, dict)


# =============================================================================
# semantic_learning.py — SemanticCurriculumDesigner
# =============================================================================

class TestSemanticCurriculumDesigner:
    def test_construction(self):
        from hololoom.core.reflection.semantic_learning import SemanticCurriculumDesigner
        designer = SemanticCurriculumDesigner(n_stages=5)
        assert designer.n_stages == 5
        assert designer.current_stage == 0

    def test_get_stage_goals_stage_0(self):
        from hololoom.core.reflection.semantic_learning import SemanticCurriculumDesigner
        designer = SemanticCurriculumDesigner(n_stages=5)
        goals = {"dim1": 0.8, "dim2": 0.6, "dim3": 0.9, "dim4": 0.7, "dim5": 0.5}
        stage_goals = designer.get_stage_goals(goals, stage=0)
        # Stage 0 should have fewest goals
        assert len(stage_goals) <= len(goals)

    def test_stage_goals_increase_with_stage(self):
        from hololoom.core.reflection.semantic_learning import SemanticCurriculumDesigner
        designer = SemanticCurriculumDesigner(n_stages=5)
        goals = {"d1": 0.9, "d2": 0.8, "d3": 0.7, "d4": 0.6, "d5": 0.5}
        early = designer.get_stage_goals(goals, stage=0)
        late = designer.get_stage_goals(goals, stage=4)
        assert len(late) >= len(early)

    def test_should_advance_stage(self):
        from hololoom.core.reflection.semantic_learning import SemanticCurriculumDesigner
        designer = SemanticCurriculumDesigner(n_stages=5)
        assert designer.current_stage == 0
        advanced = designer.should_advance_stage(success_rate=0.8, threshold=0.7)
        assert advanced is True
        assert designer.current_stage == 1

    def test_should_not_advance_below_threshold(self):
        from hololoom.core.reflection.semantic_learning import SemanticCurriculumDesigner
        designer = SemanticCurriculumDesigner(n_stages=5)
        advanced = designer.should_advance_stage(success_rate=0.5, threshold=0.7)
        assert advanced is False
        assert designer.current_stage == 0

    def test_should_not_advance_past_last_stage(self):
        from hololoom.core.reflection.semantic_learning import SemanticCurriculumDesigner
        designer = SemanticCurriculumDesigner(n_stages=3)
        designer.current_stage = 2  # Last stage
        advanced = designer.should_advance_stage(success_rate=0.9, threshold=0.7)
        assert advanced is False
        assert designer.current_stage == 2

    def test_get_stage_goals_uses_current_stage(self):
        from hololoom.core.reflection.semantic_learning import SemanticCurriculumDesigner
        designer = SemanticCurriculumDesigner(n_stages=5)
        designer.current_stage = 2
        goals = {"d1": 0.9, "d2": 0.8, "d3": 0.7}
        stage_goals = designer.get_stage_goals(goals)  # No explicit stage
        assert len(stage_goals) > 0


# =============================================================================
# __init__.py — Re-exports
# =============================================================================

class TestReflectionInit:
    def test_imports_reflection_buffer(self):
        from hololoom.core.reflection import ReflectionBuffer
        assert ReflectionBuffer is not None

    def test_imports_reflection_metrics(self):
        from hololoom.core.reflection import ReflectionMetrics
        assert ReflectionMetrics is not None

    def test_imports_learning_signal(self):
        from hololoom.core.reflection import LearningSignal
        assert LearningSignal is not None
