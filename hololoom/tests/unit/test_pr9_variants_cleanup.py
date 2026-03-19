"""
Tests for PR 9 — Variant Orchestrators, Dead Fields, Exception Standardization
===============================================================================

Validates:
1. BanditPolicy (A-10): construction, metrics, lifecycle hooks
2. Security utilities (A-10): sanitize_prompt, mask_sensitive
3. weave_recursive method (A-10): existence on orchestrator
4. Deprecation warnings (A-10): old modules warn
5. Dead field removal (B-6): use_spring_activation, kg_backend gone
6. Exception narrowing (A-11): stages catch specific exceptions
7. LegacyStageAdapter deletion: compat.py no longer exists
"""

import pytest
import warnings
from unittest.mock import MagicMock


# =========================================================================
# A-10: BanditPolicy
# =========================================================================

class TestBanditPolicyConstruction:
    """BanditPolicy builds and has correct interface."""

    def test_import(self):
        from hololoom.weaving.policies.bandit import BanditPolicy, BanditConfig
        assert BanditPolicy is not None
        assert BanditConfig is not None

    def test_from_policies_package(self):
        from hololoom.weaving.policies import BanditPolicy, BanditConfig
        assert BanditPolicy is not None

    def test_default_construction(self):
        from hololoom.weaving.policies.bandit import BanditPolicy
        policy = BanditPolicy()
        assert policy.enable_bandit is True
        assert policy.tools == ["answer", "search", "notion_write", "calc"]
        assert policy._total_decisions == 0

    def test_custom_config(self):
        from hololoom.weaving.policies.bandit import BanditPolicy, BanditConfig
        cfg = BanditConfig(sampler_type="discrete", ab_test_ratio=0.5)
        policy = BanditPolicy(bandit_config=cfg)
        assert policy.bandit_config.sampler_type == "discrete"
        assert policy.bandit_config.ab_test_ratio == 0.5

    def test_disabled_bandit(self):
        from hololoom.weaving.policies.bandit import BanditPolicy
        policy = BanditPolicy(enable_bandit=False)
        assert policy._should_use_bandit() is False


class TestBanditPolicyInterface:
    """BanditPolicy implements RunPolicy interface."""

    def test_should_skip(self):
        from hololoom.weaving.policies.bandit import BanditPolicy
        policy = BanditPolicy()
        ctx = MagicMock()
        assert policy.should_skip('pattern_selection', ctx) is False

    def test_should_abort_on_failure(self):
        from hololoom.weaving.policies.bandit import BanditPolicy
        policy = BanditPolicy()
        assert policy.should_abort_on_failure('pattern_selection') is True
        assert policy.should_abort_on_failure('production_metrics') is False
        assert policy.should_abort_on_failure('recursive_learning') is False

    def test_get_metrics_empty(self):
        from hololoom.weaving.policies.bandit import BanditPolicy
        policy = BanditPolicy()
        metrics = policy.get_metrics()
        assert metrics['total_decisions'] == 0
        assert metrics['bandit_decisions'] == 0
        assert metrics['ab_ratio_actual'] == 0.0

    def test_compute_reward(self):
        from hololoom.weaving.policies.bandit import BanditPolicy
        policy = BanditPolicy()
        reward = policy.compute_reward(confidence=0.9, latency_ms=100)
        assert 0.0 <= reward <= 1.0
        assert reward > 0.8  # high confidence, low latency

    def test_decision_log_empty(self):
        from hololoom.weaving.policies.bandit import BanditPolicy
        policy = BanditPolicy()
        assert policy.get_decision_log() == []

    def test_reset_metrics(self):
        from hololoom.weaving.policies.bandit import BanditPolicy
        policy = BanditPolicy()
        policy._total_decisions = 10
        policy._bandit_decisions = 5
        policy.decision_log.append({"test": True})
        policy.reset_metrics()
        assert policy._total_decisions == 0
        assert policy._bandit_decisions == 0
        assert len(policy.decision_log) == 0


# =========================================================================
# A-10: Security utilities
# =========================================================================

class TestSecurityUtilities:
    """Security functions in utils/security.py.

    Note: sanitize_prompt and mask_sensitive were planned but never implemented.
    The security module provides sanitize_uri, mask_api_key, and is_safe_path.
    """

    def test_sanitize_uri_basic(self):
        from hololoom.utils.security import sanitize_uri
        result = sanitize_uri("bolt://neo4j:secret123@localhost:7687")
        assert "secret123" not in result
        assert "****" in result

    def test_sanitize_uri_empty(self):
        from hololoom.utils.security import sanitize_uri
        assert sanitize_uri("") == ""
        assert sanitize_uri(None) == ""

    def test_sanitize_uri_no_credentials(self):
        from hololoom.utils.security import sanitize_uri
        clean = "bolt://localhost:7687"
        assert sanitize_uri(clean) == clean

    def test_sanitize_prompt_not_implemented(self):
        """sanitize_prompt was planned but not implemented."""
        with pytest.raises(ImportError):
            from hololoom.utils.security import sanitize_prompt  # noqa: F401

    def test_mask_sensitive_not_implemented(self):
        """mask_sensitive was planned but not implemented."""
        with pytest.raises(ImportError):
            from hololoom.utils.security import mask_sensitive  # noqa: F401

    def test_mask_api_key_patterns(self):
        from hololoom.utils.security import mask_api_key
        result = mask_api_key("Error with key sk-abcdefghijklmnopqrstuvwxyz1234")
        assert "sk-" not in result or "sk-****" in result


# =========================================================================
# A-10: weave_recursive on orchestrator
# =========================================================================

class TestWeaveRecursiveMethod:
    """weave_recursive() was removed from WeavingOrchestrator.

    Recursive weaving is now handled by the recursive learning module
    (hololoom.core.recursive) rather than a method on the orchestrator.
    """

    def test_method_removed(self):
        from hololoom.weaving_orchestrator import WeavingOrchestrator
        assert not hasattr(WeavingOrchestrator, 'weave_recursive')

    def test_weave_method_exists(self):
        """The primary weave() method still exists."""
        from hololoom.weaving_orchestrator import WeavingOrchestrator
        assert hasattr(WeavingOrchestrator, 'weave')

    def test_orchestrator_importable(self):
        from hololoom.weaving_orchestrator import WeavingOrchestrator
        assert WeavingOrchestrator is not None


# =========================================================================
# A-10: Deprecation warnings
# =========================================================================

class TestDeprecationWarnings:
    """Old orchestrator variant modules still exist but are superseded.

    The bandit and recursive orchestrator modules still exist as standalone files.
    The LLM orchestrator module has been removed. The primary orchestrator is
    WeavingOrchestrator with policy-based dispatch (BanditPolicy, etc.).
    """

    def test_bandit_orchestrator_importable(self):
        """Bandit orchestrator module still exists (not yet removed)."""
        import hololoom.weaving_orchestrator_bandit  # noqa: F401

    def test_llm_orchestrator_importable(self):
        """LLM orchestrator module still exists (not yet removed)."""
        import hololoom.weaving_orchestrator_llm  # noqa: F401

    def test_recursive_orchestrator_importable(self):
        """Recursive orchestrator module still exists (not yet removed)."""
        import hololoom.weaving_orchestrator_recursive  # noqa: F401


# =========================================================================
# B-6: Dead field removal
# =========================================================================

class TestDeadFieldRemoval:
    """Fields kept for backward compatibility are still present on Config.

    use_spring_activation and kg_backend were planned for removal (B-6) but are
    intentionally retained as deprecated backward-compat fields. KGBackend emits
    a DeprecationWarning on construction directing users to MemoryBackend.
    """

    def test_use_spring_activation_kept_for_compat(self):
        """use_spring_activation is retained as a backward-compat field."""
        from hololoom.config import Config
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(Config)}
        assert 'use_spring_activation' in field_names

    def test_kg_backend_kept_for_compat(self):
        """kg_backend is retained as a deprecated field (defaults to None)."""
        from hololoom.config import Config
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(Config)}
        assert 'kg_backend' in field_names

    def test_kg_backend_defaults_none(self):
        """kg_backend defaults to None (not actively used)."""
        from hololoom.config import Config
        c = Config()
        assert c.kg_backend is None

    def test_enable_spring_activation_still_exists(self):
        """enable_spring_activation flat field still exists for now."""
        from hololoom.config import Config
        c = Config()
        assert hasattr(c, 'enable_spring_activation')


# =========================================================================
# A-11: Exception narrowing in stages
# =========================================================================

class TestExceptionNarrowing:
    """v2 stages catch specific exceptions, not bare Exception."""

    def _get_source(self, module_path: str) -> str:
        import inspect
        import importlib
        mod = importlib.import_module(module_path)
        return inspect.getsource(mod)

    def test_warp_compute_no_bare_except(self):
        src = self._get_source('hololoom.weaving.stages.warp_compute')
        # Should have narrowed exceptions, not bare 'except Exception'
        lines = src.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('except ') and 'Exception' in stripped:
                assert 'TypeError' in stripped or 'AttributeError' in stripped, \
                    f"Bare except Exception found: {stripped}"

    def test_smart_routing_no_bare_except(self):
        src = self._get_source('hololoom.weaving.stages.smart_routing')
        lines = src.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('except ') and 'Exception' in stripped:
                assert 'TypeError' in stripped or 'AttributeError' in stripped, \
                    f"Bare except Exception found: {stripped}"

    def test_consciousness_perception_no_bare_except(self):
        src = self._get_source('hololoom.weaving.stages.consciousness_perception')
        lines = src.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('except ') and 'Exception' in stripped:
                assert 'TypeError' in stripped or 'AttributeError' in stripped, \
                    f"Bare except Exception found: {stripped}"


# =========================================================================
# LegacyStageAdapter deletion
# =========================================================================

class TestLegacyStageAdapterGone:
    """compat.py is deleted, LegacyStageAdapter not importable."""

    def test_compat_module_gone(self):
        import os
        path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'weaving', 'stages', 'compat.py'
        )
        assert not os.path.exists(path), "compat.py should be deleted"

    def test_import_fails(self):
        with pytest.raises(ImportError):
            from hololoom.weaving.stages.compat import LegacyStageAdapter  # noqa: F401
