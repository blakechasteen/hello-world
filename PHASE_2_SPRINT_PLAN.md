# Phase 2: Testing Sprint Plan

**Goal**: Increase test coverage from 60% → 85% lines, 23% → 90% modules
**Duration**: 4 weeks (Nov 18 - Dec 13, 2025)
**Test Target**: +592 new tests (2,408 → 3,000)
**Quality Score Target**: 74/100 → 89/100 (+15 points)

---

## 📊 Current State (Baseline)

**Coverage Metrics**:
- **Line Coverage**: 60%
- **Module Coverage**: 23% (208/897 files have tests)
- **Test Functions**: 2,408 total
- **Quality Score**: 74/100

**Test Distribution**:
- Unit Tests: 1,148
- Integration Tests: 832
- E2E Tests: 231

**Modules Without Tests** (115 files):
- Dreamweaving: ~20 files
- Writing: ~15 files
- Physics: ~12 files
- Visualization: ~10 files (partial)
- Server/API: ~15 files (partial)
- Infrastructure: ~25 files
- Math pipeline: ~18 files

---

## 🎯 Sprint Breakdown (4 Weeks)

### Week 1 (Nov 18-22): High-Priority Modules (150 tests)

**Target**: Dreamweaving + Writing core functionality

#### Day 1-2: Dreamweaving Tests (50 tests)
**Files to Test**:
- `HoloLoom/dreamweaving/core.py` (DreamWeaver orchestrator) - 15 tests
- `HoloLoom/dreamweaving/__init__.py` (Data structures) - 10 tests
- `HoloLoom/dreamweaving/world_fabric.py` (WorldFabric) - 8 tests
- `HoloLoom/dreamweaving/narrative_memory.py` (NarrativeMemory) - 8 tests
- `HoloLoom/dreamweaving/consistency_engine.py` (ConsistencyEngine) - 9 tests

**Test Focus**:
```python
# core.py tests
def test_dreamweaver_initialization()
def test_generate_narrative_procedural_mode()
def test_generate_narrative_creative_mode()
def test_consistency_check_strict()
def test_world_state_update()
# ... 10 more

# __init__.py tests
def test_world_entity_creation()
def test_narrative_event_creation()
def test_world_state_serialization()
# ... 7 more

# world_fabric.py tests
def test_add_entity()
def test_remove_entity()
def test_update_entity_attributes()
# ... 5 more
```

#### Day 3-4: Writing Tests (30 tests)
**Files to Test**:
- `HoloLoom/writing/core/writer.py` (Writer class) - 10 tests
- `HoloLoom/writing/core/composer.py` (Composer refinement) - 8 tests
- `HoloLoom/writing/modes/narrative.py` (NarrativeWriter) - 6 tests
- `HoloLoom/writing/modes/technical.py` (TechnicalWriter) - 6 tests

**Test Focus**:
```python
# writer.py tests
def test_writer_initialization()
def test_write_auto_mode_detection()
def test_write_narrative_mode()
def test_write_technical_mode()
def test_write_with_refinement()
# ... 5 more

# composer.py tests
def test_compose_elegance_strategy()
def test_compose_verify_strategy()
def test_quality_scoring()
# ... 5 more
```

#### Day 5: Physics Core Tests (40 tests)
**Files to Test**:
- `HoloLoom/physics/gradient_flow.py` - 8 tests
- `HoloLoom/physics/thermodynamics.py` - 8 tests
- `HoloLoom/physics/wave_mechanics.py` - 8 tests
- `HoloLoom/physics/statistical_mechanics.py` - 8 tests
- `HoloLoom/physics/fluid_dynamics.py` - 8 tests

**Test Focus**:
```python
# gradient_flow.py tests
def test_gradient_flow_engine_initialization()
def test_route_decision_lowest_loss()
def test_gradient_descent_convergence()
# ... 5 more

# thermodynamics.py tests
def test_temperature_scheduler_exponential()
def test_boltzmann_action_selection()
def test_free_energy_minimization()
# ... 5 more
```

#### Day 6-7: Review & Integration (30 tests)
- Integration tests for Dreamweaving + Writing interaction (15 tests)
- Integration tests for Physics phases integration (15 tests)

**Week 1 Deliverables**: 150 tests, +5% module coverage, +3% line coverage

---

### Week 2 (Nov 25-29): Visualization & API (148 tests)

**Target**: Visualization components + Server/API endpoints

#### Day 1-2: Visualization Tests (55 tests)
**Files to Test**:
- `HoloLoom/visualization/confidence_trajectory.py` - 10 tests
- `HoloLoom/visualization/cache_gauge.py` - 8 tests
- `HoloLoom/visualization/knowledge_graph.py` - 10 tests
- `HoloLoom/visualization/stage_waterfall.py` - 7 tests
- `HoloLoom/visualization/small_multiples.py` - 10 tests
- `HoloLoom/visualization/density_table.py` - 10 tests

**Test Focus**:
```python
# confidence_trajectory.py tests
def test_render_simple_trajectory()
def test_anomaly_detection_sudden_drop()
def test_anomaly_detection_prolonged_low()
def test_cache_markers_rendering()
# ... 6 more

# knowledge_graph.py tests
def test_render_from_kg()
def test_force_directed_layout()
def test_node_sizing_by_degree()
def test_path_highlighting()
# ... 6 more
```

#### Day 3-4: Server/API Tests (70 tests)
**Files to Test**:
- `HoloLoom/server/agentic_api.py` - 20 tests
- `HoloLoom/server/agentic_api_integrated.py` - 15 tests
- `HoloLoom/web_dashboard/agent_orchestration.py` - 15 tests
- `HoloLoom/web_dashboard/adversarial_orchestration.py` - 10 tests
- `HoloLoom/web_dashboard/workflow_executor.py` - 10 tests

**Test Focus**:
```python
# agentic_api.py tests
def test_health_endpoint()
def test_query_endpoint_simple()
def test_query_endpoint_with_context()
def test_query_endpoint_reasoning_modes()
def test_stats_endpoint()
def test_audit_trail_endpoint()
# ... 14 more

# agent_orchestration.py tests
def test_orchestrate_single_agent()
def test_orchestrate_parallel_agents()
def test_agent_error_handling()
# ... 12 more
```

#### Day 5-7: RAG & Agentic Tests (23 tests)
**Files to Test**:
- `HoloLoom/rag/simple_rag.py` (already has 24/25) - 5 more tests
- `HoloLoom/agentic/core.py` (needs improvement) - 10 tests
- `HoloLoom/agentic/web_research.py` - 8 tests

**Week 2 Deliverables**: 148 tests, +10% module coverage, +5% line coverage

---

### Week 3 (Dec 2-6): Infrastructure & Math Pipeline (147 tests)

**Target**: Infrastructure utilities + Math pipeline

#### Day 1-2: Infrastructure Tests (60 tests)
**Files to Test**:
- `HoloLoom/tools/handler_factory.py` - 10 tests
- `HoloLoom/tools/validation.py` - 8 tests
- `HoloLoom/convergence/engine.py` - 12 tests
- `HoloLoom/loom/command.py` - 10 tests
- `HoloLoom/chrono/trigger.py` - 10 tests
- `HoloLoom/resonance/shed.py` - 10 tests

**Test Focus**:
```python
# convergence/engine.py tests
def test_convergence_argmax_strategy()
def test_convergence_epsilon_greedy()
def test_convergence_bayesian_blend()
def test_convergence_pure_thompson()
# ... 8 more

# loom/command.py tests
def test_pattern_card_selection()
def test_loom_command_bare_mode()
def test_loom_command_fast_mode()
def test_loom_command_fused_mode()
# ... 6 more
```

#### Day 3-4: Math Pipeline Tests (55 tests)
**Files to Test**:
- `HoloLoom/semantic_calculus/integrator.py` - 15 tests
- `HoloLoom/semantic_calculus/dimension_selector.py` - 10 tests
- `HoloLoom/semantic_calculus/axes.py` - 15 tests
- `HoloLoom/embedding/spectral.py` - 15 tests

**Test Focus**:
```python
# integrator.py tests
def test_semantic_spectrum_projection()
def test_228d_space_projection()
def test_16_interpretable_axes_extraction()
# ... 12 more

# axes.py tests
def test_warmth_axis_projection()
def test_formality_axis_projection()
def test_all_16_axes_available()
# ... 12 more
```

#### Day 5-7: Recursive Learning Tests (32 tests)
**Files to Test**:
- `HoloLoom/recursive/scratchpad.py` - 8 tests
- `HoloLoom/recursive/advanced_refinement.py` - 12 tests
- `HoloLoom/recursive/hot_pattern_feedback.py` - 12 tests

**Week 3 Deliverables**: 147 tests, +15% module coverage, +7% line coverage

---

### Week 4 (Dec 9-13): Integration & Performance (147 tests)

**Target**: Cross-module integration + performance benchmarks

#### Day 1-3: Cross-Module Integration Tests (90 tests)
**Focus Areas**:
- Dreamweaving + RAG integration (15 tests)
- Writing + Recursive Refinement (15 tests)
- Physics + Warp Space (15 tests)
- Visualization + Query Pipeline (15 tests)
- API + Alignment Framework (15 tests)
- Complete 9-step weaving cycle (15 tests)

**Test Focus**:
```python
# Dreamweaving + RAG integration
def test_narrative_generation_with_rag_context()
def test_world_building_from_knowledge_graph()
def test_consistency_check_with_multimodal_rag()
# ... 12 more

# Physics + Warp Space integration
def test_gradient_flow_warp_space_routing()
def test_fluid_dynamics_context_packing()
def test_thermodynamic_action_selection()
# ... 12 more
```

#### Day 4-5: Performance Tests (40 tests)
**Files to Create**:
- `HoloLoom/tests/performance/test_latency_benchmarks.py` - 15 tests
- `HoloLoom/tests/performance/test_throughput_benchmarks.py` - 10 tests
- `HoloLoom/tests/performance/test_memory_usage.py` - 15 tests

**Test Focus**:
```python
# Latency benchmarks
def test_bare_mode_latency_under_50ms()
def test_fast_mode_latency_under_150ms()
def test_fused_mode_latency_under_300ms()
# ... 12 more

# Memory usage
def test_memory_usage_within_bounds()
def test_no_memory_leaks_after_1000_queries()
def test_cache_memory_management()
# ... 12 more
```

#### Day 6-7: E2E Scenario Tests (17 tests)
**Files to Test**:
- `HoloLoom/tests/e2e/test_full_research_workflow.py` (new) - 5 tests
- `HoloLoom/tests/e2e/test_multimodal_pipeline.py` (new) - 6 tests
- `HoloLoom/tests/e2e/test_production_scenarios.py` (new) - 6 tests

**Test Focus**:
```python
# Full research workflow
def test_research_query_with_verification()
def test_multiquery_research_pipeline()
def test_plan_execute_complex_task()
# ... 2 more

# Production scenarios
def test_high_load_concurrent_queries()
def test_graceful_degradation_failures()
def test_end_to_end_with_alignment_checks()
# ... 3 more
```

**Week 4 Deliverables**: 147 tests, +20% module coverage, +10% line coverage

---

## 📈 Cumulative Progress Tracker

| Week | Tests Added | Cumulative Tests | Module Coverage | Line Coverage | Quality Score |
|------|-------------|------------------|-----------------|---------------|---------------|
| **Baseline** | 0 | 2,408 | 23% | 60% | 74/100 |
| **Week 1** | +150 | 2,558 | 28% (+5%) | 63% (+3%) | 76/100 (+2) |
| **Week 2** | +148 | 2,706 | 38% (+10%) | 68% (+5%) | 80/100 (+4) |
| **Week 3** | +147 | 2,853 | 53% (+15%) | 75% (+7%) | 84/100 (+4) |
| **Week 4** | +147 | 3,000 | 73% (+20%) | 85% (+10%) | 89/100 (+5) |

**Final Target**: 3,000 tests, 73% module coverage, 85% line coverage, 89/100 quality score

---

## 🎯 Success Metrics

**Per Week**:
- Minimum 145 new tests written
- All new tests pass
- No regression in existing tests
- Code review completed
- Tests committed to branch

**Overall Phase 2 Success Criteria**:
- ✅ Test count: 2,408 → 3,000 (+592)
- ✅ Line coverage: 60% → 85% (+25%)
- ✅ Module coverage: 23% → 73%+ (+50%)
- ✅ Quality score: 74 → 89 (+15 points)
- ✅ Zero critical bugs introduced
- ✅ All CI/CD checks passing

---

## 🚀 Execution Strategy

### Daily Workflow
1. **Morning** (9-10am): Review yesterday's tests, run coverage
2. **Focus Time** (10am-12pm): Write tests for target module
3. **Lunch Break** (12-1pm)
4. **Afternoon** (1-3pm): Continue test writing
5. **Review** (3-4pm): Run full test suite, fix failures
6. **EOD** (4-5pm): Commit tests, update progress tracker

### Test Writing Best Practices
- **Follow AAA pattern**: Arrange, Act, Assert
- **Use pytest fixtures**: Reduce duplication
- **Test edge cases**: Not just happy paths
- **Mock external dependencies**: Fast, isolated tests
- **Document test intent**: Clear docstrings
- **Use parametrize**: Test multiple inputs efficiently

### Quality Checks Before Commit
```bash
# Run tests
pytest HoloLoom/tests -v --cov=HoloLoom

# Check coverage increased
make coverage

# Lint new test files
ruff check HoloLoom/tests/

# Type check
mypy HoloLoom/tests/

# Commit
git add HoloLoom/tests/
git commit -m "test: Add [module] tests (+N tests, +X% coverage)"
```

---

## 📝 Test Template

Use this template for new test files:

```python
"""
Tests for HoloLoom.[module].[submodule]

Author: Claude Code
Date: YYYY-MM-DD
Coverage Target: XX% → YY%
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from HoloLoom.[module] import [ClassToTest]


class Test[ClassName]:
    """Test suite for [ClassName]."""

    @pytest.fixture
    def instance(self):
        """Create test instance."""
        return [ClassName](param1=value1)

    def test_initialization(self, instance):
        """Test [ClassName] initialization."""
        assert instance.param1 == value1
        assert isinstance(instance.internal_state, dict)

    def test_method_happy_path(self, instance):
        """Test method with valid inputs."""
        result = instance.method(input_data)
        assert result.success is True
        assert result.value > 0

    def test_method_edge_case(self, instance):
        """Test method with edge case inputs."""
        result = instance.method(edge_case_data)
        assert result.handled_gracefully is True

    def test_method_error_handling(self, instance):
        """Test method error handling."""
        with pytest.raises(ValueError, match="Invalid input"):
            instance.method(invalid_data)

    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_method_parametrized(self, instance, input, expected):
        """Test method with multiple inputs."""
        result = instance.method(input)
        assert result == expected
```

---

## 🔄 Progress Reporting

**Weekly Status Update Format**:
```markdown
### Week N Status (MMM DD-DD, 2025)

**Tests Added**: XXX
**Cumulative**: XXXX/3000
**Module Coverage**: XX% (+Y%)
**Line Coverage**: XX% (+Y%)
**Quality Score**: XX/100 (+Y)

**Completed**:
- ✅ Module A tests (NN tests)
- ✅ Module B tests (NN tests)
- ✅ Integration tests (NN tests)

**Blockers**: None | [Issue description]
**Next Week Focus**: [Module C, Module D]
```

---

## 🎉 Phase 2 Completion Celebration

When Phase 2 is complete, we will have:
- **3,000 test functions** (from 2,408)
- **85% line coverage** (from 60%)
- **73%+ module coverage** (from 23%)
- **89/100 quality score** (from 74)
- **Comprehensive test coverage across all major modules**

This represents a **+15 point improvement** in quality score and sets us up perfectly for Phase 3 (Architecture Refactoring).

---

**Next Steps After Phase 2**:
- Phase 3: Architecture Perfection (3 weeks)
- Phase 4: Developer Experience (2 weeks)
- Phase 5: CI/CD Perfection (1 week)
- **Final Goal**: 96/100 quality score (Excellent tier)

---

**Created**: Nov 15, 2025
**Author**: Claude Code (HoloLoom Core Team)
**Status**: Ready for execution 🚀
