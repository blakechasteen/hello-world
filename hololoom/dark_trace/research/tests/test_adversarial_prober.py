"""
Tests for Dark Trace Adversarial Probing Suite
==============================================

Tests cover:
- ProbeConfig presets
- GradientProber operations
- AdversarialProber vulnerability detection
- GeneticAdversarialSearch evolution
- AdversarialCatalog persistence

Created: 2025-12-28
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os


# =============================================================================
# Test Configuration
# =============================================================================


class TestProbeConfig:
    """Tests for ProbeConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from hololoom.dark_trace.research.adversarial_prober import ProbeConfig

        config = ProbeConfig()

        assert config.epsilon == 0.1
        assert config.gradient_steps == 50
        assert config.population_size == 50
        assert config.generations == 30
        assert config.vulnerability_threshold == 0.3
        assert config.critical_threshold == 0.7
        assert config.enable_safety_checks is True

    def test_quick_preset(self):
        """Test quick preset for fast probing."""
        from hololoom.dark_trace.research.adversarial_prober import ProbeConfig

        config = ProbeConfig.quick()

        assert config.gradient_steps == 20
        assert config.population_size == 20
        assert config.generations == 10

    def test_balanced_preset(self):
        """Test balanced preset."""
        from hololoom.dark_trace.research.adversarial_prober import ProbeConfig

        config = ProbeConfig.balanced()

        assert config.gradient_steps == 50
        assert config.population_size == 50

    def test_thorough_preset(self):
        """Test thorough preset for comprehensive probing."""
        from hololoom.dark_trace.research.adversarial_prober import ProbeConfig

        config = ProbeConfig.thorough()

        assert config.gradient_steps == 100
        assert config.population_size == 100
        assert config.generations == 50


# =============================================================================
# Test Vulnerability Level
# =============================================================================


class TestVulnerabilityLevel:
    """Tests for vulnerability level classification."""

    def test_level_from_score(self):
        """Test vulnerability level determination from score."""
        from hololoom.dark_trace.research.adversarial_prober import VulnerabilityLevel

        # Test all levels
        assert VulnerabilityLevel.ROBUST.value == "robust"
        assert VulnerabilityLevel.LOW.value == "low"
        assert VulnerabilityLevel.MEDIUM.value == "medium"
        assert VulnerabilityLevel.HIGH.value == "high"
        assert VulnerabilityLevel.CRITICAL.value == "critical"


# =============================================================================
# Test Gradient Prober
# =============================================================================


class TestGradientProber:
    """Tests for GradientProber."""

    def test_initialization(self):
        """Test GradientProber initialization."""
        from hololoom.dark_trace.research.adversarial_prober import (
            GradientProber, ProbeConfig
        )

        config = ProbeConfig()
        prober = GradientProber(config)

        assert prober.config == config
        assert prober.config.epsilon == 0.1

    def test_fgsm_probe(self):
        """Test FGSM probing."""
        from hololoom.dark_trace.research.adversarial_prober import (
            GradientProber, ProbeConfig
        )

        config = ProbeConfig(gradient_steps=5)
        prober = GradientProber(config)

        # Mock input and functions
        input_data = np.random.randn(1, 384).astype(np.float32)

        def activation_fn(x):
            return float(np.mean(x))

        def gradient_fn(x):
            return np.ones_like(x)  # Simple gradient

        result = prober.probe(
            input_data=input_data,
            feature_idx=42,
            activation_fn=activation_fn,
            gradient_fn=gradient_fn,
            target_type="minimize"  # Suppress = minimize activation
        )

        assert result.feature_idx == 42
        assert result.perturbation is not None
        assert result.perturbation.shape == input_data.shape
        assert np.max(np.abs(result.perturbation)) <= config.epsilon + 1e-6

    def test_pgd_probe(self):
        """Test PGD probing."""
        from hololoom.dark_trace.research.adversarial_prober import (
            GradientProber, ProbeConfig
        )

        config = ProbeConfig(gradient_steps=10)
        prober = GradientProber(config)

        input_data = np.random.randn(1, 384).astype(np.float32)

        def activation_fn(x):
            return float(np.sum(x ** 2))  # Quadratic activation

        def gradient_fn(x):
            return 2 * x  # Gradient of quadratic

        result = prober.probe(
            input_data=input_data,
            feature_idx=100,
            activation_fn=activation_fn,
            gradient_fn=gradient_fn,
            target_type="maximize"  # Amplify = maximize activation
        )

        assert result.feature_idx == 100
        assert result.activation_change != 0.0  # Some change expected


# =============================================================================
# Test Adversarial Prober
# =============================================================================


class TestAdversarialProber:
    """Tests for AdversarialProber."""

    def test_initialization(self):
        """Test AdversarialProber initialization."""
        from hololoom.dark_trace.research.adversarial_prober import (
            AdversarialProber, ProbeConfig
        )

        config = ProbeConfig()
        mock_adapter = Mock()

        prober = AdversarialProber(config, mock_adapter)

        assert prober.config == config
        assert prober.model_adapter == mock_adapter
        assert prober.safety_guardrails is None

    def test_with_safety_guardrails(self):
        """Test prober with safety guardrails."""
        from hololoom.dark_trace.research.adversarial_prober import (
            AdversarialProber, ProbeConfig
        )

        config = ProbeConfig(enable_safety_checks=True)
        mock_adapter = Mock()
        mock_guardrails = Mock()

        prober = AdversarialProber(config, mock_adapter, mock_guardrails)

        assert prober.safety_guardrails == mock_guardrails

    def test_probe_feature(self):
        """Test probing a single feature."""
        from hololoom.dark_trace.research.adversarial_prober import (
            AdversarialProber, ProbeConfig, VulnerabilityLevel
        )

        config = ProbeConfig.quick()
        mock_adapter = Mock()
        mock_adapter.get_activations = Mock(return_value=np.random.randn(1, 384))

        prober = AdversarialProber(config, mock_adapter)

        test_inputs = np.random.randn(5, 384).astype(np.float32)

        def activation_fn(x):
            return float(np.mean(np.abs(x)))

        def gradient_fn(x):
            return np.sign(x)

        report = prober.probe_feature(
            feature_idx=42,
            test_inputs=test_inputs,
            activation_fn=activation_fn,
            gradient_fn=gradient_fn
        )

        assert report.feature_idx == 42
        assert 0.0 <= report.vulnerability_score <= 1.0
        assert isinstance(report.vulnerability_level, VulnerabilityLevel)

    def test_vulnerability_summary(self):
        """Test vulnerability summary generation."""
        from hololoom.dark_trace.research.adversarial_prober import (
            AdversarialProber, ProbeConfig
        )

        config = ProbeConfig()
        mock_adapter = Mock()

        prober = AdversarialProber(config, mock_adapter)

        # Initially empty
        summary = prober.get_vulnerability_summary()

        assert "total_probed" in summary
        assert "by_level" in summary
        assert summary["total_probed"] == 0


# =============================================================================
# Test Genetic Prober
# =============================================================================


class TestGeneticConfig:
    """Tests for GeneticConfig."""

    def test_default_config(self):
        """Test default genetic config."""
        from hololoom.dark_trace.research.genetic_prober import GeneticConfig

        config = GeneticConfig()

        assert config.population_size == 50
        assert config.generations == 30
        assert config.mutation_rate == 0.1
        assert config.crossover_rate == 0.8
        assert config.elite_count == 5

    def test_fast_preset(self):
        """Test fast genetic config."""
        from hololoom.dark_trace.research.genetic_prober import GeneticConfig

        config = GeneticConfig.fast()

        assert config.population_size == 20
        assert config.generations == 15

    def test_thorough_preset(self):
        """Test thorough genetic config."""
        from hololoom.dark_trace.research.genetic_prober import GeneticConfig

        config = GeneticConfig.thorough()

        assert config.population_size == 100
        assert config.generations == 100


class TestGeneticOperators:
    """Tests for genetic operators."""

    def test_tournament_selection(self):
        """Test tournament selection."""
        from hololoom.dark_trace.research.genetic_prober import (
            GeneticConfig, GeneticOperators, Individual
        )

        config = GeneticConfig(tournament_size=3)
        operators = GeneticOperators(config)

        # Create population with varying fitness
        population = [
            Individual(genes=np.zeros(10), fitness=float(i))
            for i in range(10)
        ]

        parents = operators.select_parents(population, 5)

        assert len(parents) == 5
        # Higher fitness individuals should be more likely selected
        avg_fitness = sum(p.fitness for p in parents) / len(parents)
        assert avg_fitness > 2.0  # Should be above random selection average

    def test_crossover(self):
        """Test crossover operations."""
        from hololoom.dark_trace.research.genetic_prober import (
            GeneticConfig, GeneticOperators, Individual, CrossoverMethod
        )

        config = GeneticConfig(crossover_rate=1.0)
        operators = GeneticOperators(config)

        parent1 = Individual(genes=np.ones(10))
        parent2 = Individual(genes=np.zeros(10))

        child1, child2 = operators.crossover(parent1, parent2)

        # Children should be different from parents (most likely)
        assert child1.genes is not parent1.genes
        assert child2.genes is not parent2.genes

    def test_mutation(self):
        """Test mutation operations."""
        from hololoom.dark_trace.research.genetic_prober import (
            GeneticConfig, GeneticOperators, Individual
        )

        config = GeneticConfig(mutation_rate=1.0, epsilon=0.1)  # Always mutate
        operators = GeneticOperators(config)

        individual = Individual(genes=np.zeros(100))

        mutated = operators.mutate(individual)

        # With 100% mutation rate, genes should change
        assert not np.allclose(mutated.genes, individual.genes)
        # Mutation should be bounded
        assert np.max(np.abs(mutated.genes)) <= config.epsilon + 1e-6


class TestGeneticAdversarialSearch:
    """Tests for GeneticAdversarialSearch."""

    def test_initialization(self):
        """Test search initialization."""
        from hololoom.dark_trace.research.genetic_prober import (
            GeneticAdversarialSearch, GeneticConfig
        )

        config = GeneticConfig()
        search = GeneticAdversarialSearch(config)

        assert search.config == config

    def test_evolution(self):
        """Test basic evolution."""
        from hololoom.dark_trace.research.genetic_prober import (
            GeneticAdversarialSearch, GeneticConfig
        )

        config = GeneticConfig.fast()
        search = GeneticAdversarialSearch(config)

        def fitness_fn(perturbation):
            # Simple fitness: maximize L2 norm (within bounds)
            return np.linalg.norm(perturbation)

        result = search.evolve(
            input_shape=(10,),
            fitness_fn=fitness_fn
        )

        assert result.best_individual is not None
        assert result.generations_run > 0
        assert len(result.fitness_history) > 0
        assert result.best_individual.fitness > 0

    def test_early_stopping(self):
        """Test early stopping when fitness converges."""
        from hololoom.dark_trace.research.genetic_prober import (
            GeneticAdversarialSearch, GeneticConfig
        )

        config = GeneticConfig(
            population_size=10,
            generations=100,
            early_stopping_patience=3
        )
        search = GeneticAdversarialSearch(config)

        # Constant fitness should trigger early stopping
        def fitness_fn(perturbation):
            return 1.0  # Always same fitness

        result = search.evolve(
            input_shape=(10,),
            fitness_fn=fitness_fn
        )

        # Should stop early due to no improvement
        assert result.early_stopped or result.generations_run < 100


# =============================================================================
# Test Adversarial Catalog
# =============================================================================


class TestAdversarialCatalog:
    """Tests for AdversarialCatalog."""

    def test_initialization_in_memory(self):
        """Test in-memory catalog."""
        from hololoom.dark_trace.research.adversarial_catalog import AdversarialCatalog

        catalog = AdversarialCatalog()

        assert catalog.storage_path is None
        assert len(catalog.get_all()) == 0

    def test_add_vulnerability(self):
        """Test adding vulnerability."""
        from hololoom.dark_trace.research.adversarial_catalog import AdversarialCatalog

        catalog = AdversarialCatalog()

        vuln = catalog.add_vulnerability(
            feature_idx=42,
            model_version="v1.0",
            vulnerability_score=0.75,
            vulnerability_level="HIGH",
            attack_method="FGSM",
            epsilon=0.1
        )

        assert vuln.feature_idx == 42
        assert vuln.vulnerability_score == 0.75
        assert vuln.vulnerability_level == "HIGH"

        # Should be retrievable
        all_vulns = catalog.get_all()
        assert len(all_vulns) == 1
        assert all_vulns[0].feature_idx == 42

    def test_query_by_level(self):
        """Test querying by vulnerability level."""
        from hololoom.dark_trace.research.adversarial_catalog import AdversarialCatalog

        catalog = AdversarialCatalog()

        # Add multiple
        catalog.add_vulnerability(42, "v1.0", 0.75, "HIGH", "FGSM", 0.1)
        catalog.add_vulnerability(43, "v1.0", 0.35, "MEDIUM", "PGD", 0.1)
        catalog.add_vulnerability(44, "v1.0", 0.85, "CRITICAL", "GENETIC", 0.1)

        high_vulns = catalog.get_by_level("HIGH")
        assert len(high_vulns) == 1
        assert high_vulns[0].feature_idx == 42

        critical_vulns = catalog.get_by_level("CRITICAL")
        assert len(critical_vulns) == 1

    def test_update_status(self):
        """Test updating vulnerability status."""
        from hololoom.dark_trace.research.adversarial_catalog import (
            AdversarialCatalog, VulnerabilityStatus
        )

        catalog = AdversarialCatalog()

        vuln = catalog.add_vulnerability(42, "v1.0", 0.75, "HIGH", "FGSM", 0.1)

        # Update status
        result = catalog.update_status(vuln.id, VulnerabilityStatus.MITIGATED)
        assert result is True

        # Verify update
        updated = catalog.get_vulnerability(vuln.id)
        assert updated.status == VulnerabilityStatus.MITIGATED

    def test_persistence(self):
        """Test catalog persistence."""
        from hololoom.dark_trace.research.adversarial_catalog import AdversarialCatalog

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate catalog
            catalog = AdversarialCatalog(tmpdir)
            catalog.add_vulnerability(42, "v1.0", 0.75, "HIGH", "FGSM", 0.1)

            # Create new catalog from same path
            catalog2 = AdversarialCatalog(tmpdir)

            # Should have loaded the vulnerability
            assert len(catalog2.get_all()) == 1
            assert catalog2.get_all()[0].feature_idx == 42

    def test_version_comparison(self):
        """Test comparing versions for regression."""
        from hololoom.dark_trace.research.adversarial_catalog import AdversarialCatalog

        catalog = AdversarialCatalog()

        # v1.0 vulnerabilities
        catalog.add_vulnerability(42, "v1.0", 0.75, "HIGH", "FGSM", 0.1)
        catalog.add_vulnerability(43, "v1.0", 0.35, "MEDIUM", "PGD", 0.1)

        # v1.1 vulnerabilities (42 fixed, 44 new)
        catalog.add_vulnerability(43, "v1.1", 0.30, "MEDIUM", "PGD", 0.1)
        catalog.add_vulnerability(44, "v1.1", 0.60, "MEDIUM", "GENETIC", 0.1)

        report = catalog.compare_versions("v1.0", "v1.1")

        assert report.summary["new_count"] == 1  # feature 44
        assert report.summary["fixed_count"] == 1  # feature 42

    def test_catalog_summary(self):
        """Test catalog summary generation."""
        from hololoom.dark_trace.research.adversarial_catalog import AdversarialCatalog

        catalog = AdversarialCatalog()

        # Add various vulnerabilities
        catalog.add_vulnerability(42, "v1.0", 0.75, "HIGH", "FGSM", 0.1)
        catalog.add_vulnerability(43, "v1.0", 0.35, "MEDIUM", "PGD", 0.1)
        catalog.add_vulnerability(44, "v1.0", 0.85, "CRITICAL", "GENETIC", 0.1)

        summary = catalog.get_summary()

        assert summary.total_vulnerabilities == 3
        assert summary.by_level["HIGH"] == 1
        assert summary.by_level["CRITICAL"] == 1
        assert summary.by_method["FGSM"] == 1
        assert 44 in summary.most_vulnerable_features  # Highest score


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_prober(self):
        """Test prober factory function."""
        from hololoom.dark_trace.research.adversarial_prober import create_prober

        mock_adapter = Mock()

        prober = create_prober(mock_adapter, preset="quick")

        assert prober is not None
        assert prober.config.gradient_steps == 20

    def test_create_genetic_search(self):
        """Test genetic search factory function."""
        from hololoom.dark_trace.research.genetic_prober import create_genetic_search

        search = create_genetic_search(preset="fast")

        assert search is not None
        assert search.config.population_size == 20

    def test_create_catalog(self):
        """Test catalog factory function."""
        from hololoom.dark_trace.research.adversarial_catalog import create_catalog

        catalog = create_catalog()

        assert catalog is not None
        assert catalog.storage_path is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for adversarial probing suite."""

    def test_full_probing_workflow(self):
        """Test complete probing workflow."""
        from hololoom.dark_trace.research.adversarial_prober import (
            AdversarialProber, ProbeConfig
        )
        from hololoom.dark_trace.research.adversarial_catalog import AdversarialCatalog

        # Setup
        config = ProbeConfig.quick()
        mock_adapter = Mock()
        mock_adapter.get_activations = Mock(return_value=np.random.randn(1, 384))

        prober = AdversarialProber(config, mock_adapter)
        catalog = AdversarialCatalog()

        # Probe a feature
        test_inputs = np.random.randn(3, 384).astype(np.float32)

        def activation_fn(x):
            return float(np.mean(np.abs(x)))

        def gradient_fn(x):
            return np.sign(x)

        report = prober.probe_feature(
            feature_idx=42,
            test_inputs=test_inputs,
            activation_fn=activation_fn,
            gradient_fn=gradient_fn
        )

        # Catalog the result
        vuln = catalog.add_vulnerability(
            feature_idx=report.feature_idx,
            model_version="test_v1.0",
            vulnerability_score=report.vulnerability_score,
            vulnerability_level=report.vulnerability_level.value,
            attack_method=report.methods_tried[0] if report.methods_tried else "HYBRID",
            epsilon=config.epsilon,
            activation_before=report.activation_before,
            activation_after=report.activation_after
        )

        # Verify
        assert vuln.feature_idx == 42
        assert catalog.get_all()[0].id == vuln.id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
