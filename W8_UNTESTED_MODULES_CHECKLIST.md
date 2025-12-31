# W8: Untested Modules - Quick Reference Checklist

**Generated**: December 31, 2025
**Purpose**: Fast lookup of which modules need tests
**Format**: Checkbox lists for tracking

---

## CRITICAL - IMMEDIATE ATTENTION (No Tests, Central Systems)

### Orchestrator & Weaving (THE CORE)
- [ ] HoloLoom/orchestrator/ (45+ files)
  - [ ] orchestrator/core/complexity_detection.py
  - [ ] orchestrator/core/metrics_collection.py
  - [ ] orchestrator/core/background_tasks.py
  - [ ] orchestrator/core/stat_mech_integration.py
  - [ ] orchestrator/stages/steps_0_3.py (Meta-prompt, Pattern, Chrono, Thread)
  - [ ] orchestrator/stages/steps_4_6.py (Feature, Warp, Memory - PARALLEL)
  - [ ] orchestrator/stages/steps_7_9.py (Convergence, Tool, Spacetime)
  - [ ] orchestrator/stages/executors/meta_prompt_executor.py
  - [ ] orchestrator/stages/executors/pattern_executor.py
  - [ ] orchestrator/stages/executors/chrono_executor.py
  - [ ] orchestrator/stages/executors/thread_executor.py
  - [ ] orchestrator/stages/executors/parallel_executor.py
  - [ ] orchestrator/stages/executors/convergence_executor.py
  - [ ] orchestrator/stages/executors/tool_executor.py
  - [ ] orchestrator/stages/executors/spacetime_executor.py
  - [ ] orchestrator/initialization/*.py (5 files)
  - [ ] orchestrator/retrieval/multipass_retrieval.py
  - [ ] orchestrator/physics/physics_integration.py
  - [ ] orchestrator/learning/recursive_learning.py
  - [ ] orchestrator/jenny/panel_detection.py
  - [ ] orchestrator/context.py
  - [ ] orchestrator/pipeline.py
  - [ ] orchestrator/protocol_factory.py

- [ ] HoloLoom/weaving/ (16 files)
  - [ ] weaving/strategies/base.py
  - [ ] weaving/strategies/lite_strategy.py
  - [ ] weaving/strategies/fast_strategy.py
  - [ ] weaving/strategies/full_strategy.py
  - [ ] weaving/strategies/research_strategy.py
  - [ ] weaving/stages/pattern_selection.py
  - [ ] weaving/stages/temporal_control.py
  - [ ] weaving/stages/feature_extraction.py
  - [ ] weaving/stages/memory_retrieval.py
  - [ ] weaving/stages/decision_collapse.py
  - [ ] weaving/protocols.py
  - [ ] weaving/eggroll_weave.py

### Policy & Neural Core
- [ ] HoloLoom/policy/ (6 files, NO tests)
  - [ ] policy/unified.py (Main neural core + Thompson Sampling)
  - [ ] policy/thompson_sampling.py
  - [ ] policy/bayesian_policy.py
  - [ ] policy/semantic_nudging.py
  - [ ] policy/gp_policy.py
  - [ ] policy/gp_policy.py

- [ ] HoloLoom/neural/ (4 files, NO tests)
  - [ ] neural/meta_learning.py
  - [ ] neural/twin_networks.py
  - [ ] neural/value_functions.py

### Embedding System (37x speedup unvalidated!)
- [ ] HoloLoom/embedding/ (8 files, NO tests)
  - [ ] embedding/zero_copy.py ⚠️ CRITICAL - 37x speedup claim
  - [ ] embedding/spectral.py (Multi-scale features)
  - [ ] embedding/spectral_multiscale.py
  - [ ] embedding/matryoshka_gate.py
  - [ ] embedding/linguistic_matryoshka_gate.py
  - [ ] embedding/matryoshka_interpreter.py
  - [ ] embedding/riemannian_matryoshka.py

### Input Adapters (36+ complete black box)
- [ ] HoloLoom/spinningWheel/ (36+ adapters, NO tests)
  - [ ] spinningWheel/pdf_spinner.py
  - [ ] spinningWheel/youtube_spinner.py
  - [ ] spinningWheel/email_spinner.py
  - [ ] spinningWheel/git_spinner.py
  - [ ] spinningWheel/codebase_spinner.py
  - [ ] spinningWheel/image_spinner.py
  - [ ] spinningWheel/whisper_spinner.py (Audio)
  - [ ] spinningWheel/voice_correction.py
  - [ ] spinningWheel/spreadsheet_spinner.py
  - [ ] spinningWheel/url_spinner.py
  - [ ] spinningWheel/matrix_spinner.py
  - [ ] spinningWheel/file_upload_spinner.py
  - [ ] spinningWheel/multimodal_spinner.py
  - [ ] spinningWheel/schema_aware_receipt_spinner.py
  - [ ] spinningWheel/receipt_spinner.py
  - [ ] spinningWheel/handwritten_spinner.py
  - [ ] spinningWheel/deepseek_ocr_spinner.py
  - [ ] spinningWheel/live_scratchpad.py
  - [ ] spinningWheel/voice_scratchpad.py
  - [ ] spinningWheel/chat_history.py
  - [ ] spinningWheel/website.py
  - [ ] spinningWheel/recursive_crawler.py
  - [ ] spinningWheel/browser_history.py
  - [ ] spinningWheel/domain_router.py
  - [ ] spinningWheel/auto.py
  - [ ] spinningWheel/batch_utils.py
  - [ ] spinningWheel/protocol.py
  - [ ] spinningWheel/ocr_protocol.py
  - [ ] spinningWheel/schema_registry.py
  - [ ] spinningWheel/utils.py
  - [ ] spinningWheel/importance.py
  - [ ] spinningWheel/workspace.py
  - [ ] spinningWheel/mcp_server.py
  - [ ] spinningWheel/dream_spinner.py

### Memory Awareness & Dynamics
- [ ] HoloLoom/memory/awareness/ (Multiple files)
  - [ ] memory/awareness_graph.py ⚠️ Core activation system (800+ lines)
  - [ ] memory/activation_field.py
  - [ ] memory/awareness_types.py

- [ ] HoloLoom/memory/spring_dynamics.py ⚠️ Physics-based (699 lines)
  - [ ] Physics ODE integration tests

- [ ] HoloLoom/memory/multi_wave_engine.py ⚠️ Consolidation (623 lines)

- [ ] HoloLoom/memory/yarn/ - Knowledge graph

- [ ] HoloLoom/memory/stores/ (Neo4j, Qdrant)
  - [ ] memory/stores/qdrant_store.py
  - [ ] memory/stores/neo4j_graph.py

- [ ] HoloLoom/memory/visual_compression.py (674 lines)

### Prompting & Refinement
- [ ] HoloLoom/prompting/ (Missing MRF tests!)
  - [ ] prompting/unified_mrf.py (Main 7-component framework)
  - [ ] prompting/model_adapters.py (Claude, Gemini, GPT, Ollama)
  - [ ] prompting/quality_assessment.py
  - [ ] prompting/analytics/dashboard.py
  - [ ] prompting/analytics/learning.py
  - [ ] prompting/analytics/ab_testing.py
  - [ ] prompting/testing/* (All test infrastructure)

---

## HIGH PRIORITY (No Tests, Feature-Complete Systems)

### Input & Multimodal
- [ ] HoloLoom/input/ (9 files)
  - [ ] input/audio_processor.py
  - [ ] input/image_processor.py
  - [ ] input/text_processor.py
  - [ ] input/structured_processor.py
  - [ ] input/router.py
  - [ ] input/fusion.py
  - [ ] input/protocol.py
  - [ ] input/simple_embedder.py

### Visualization & UI (37+ files)
- [ ] HoloLoom/visualization/ (37 files, NO tests)
  - [ ] visualization/jenny_runtime.py
  - [ ] visualization/jenny_renderer.py
  - [ ] visualization/jenny_spec.py
  - [ ] visualization/jenny_accessibility.py
  - [ ] visualization/jenny_analytics.py
  - [ ] visualization/jenny_llm_client.py
  - [ ] visualization/jenny_mrf.py
  - [ ] visualization/stage_waterfall.py ⚠️ Tufte visualization
  - [ ] visualization/confidence_trajectory.py
  - [ ] visualization/cache_gauge.py
  - [ ] visualization/knowledge_graph.py
  - [ ] visualization/semantic_space.py
  - [ ] visualization/small_multiples.py
  - [ ] visualization/density_table.py
  - [ ] visualization/html_renderer.py
  - [ ] visualization/dashboard.py
  - [ ] visualization/dashboard_constructor.py
  - [ ] visualization/rag_dashboard.py

### Spatial (AR/VR) - 20+ files
- [ ] HoloLoom/spatial/ (NO tests)
  - [ ] spatial/webxr_graph.py
  - [ ] spatial/spatial_anchors.py
  - [ ] spatial/hand_tracking.py
  - [ ] spatial/gaze_tracking.py
  - [ ] spatial/avatar_system.py
  - [ ] spatial/environment_mapping.py
  - [ ] spatial/spatial_audio.py
  - [ ] spatial/spatial_ui.py
  - [ ] spatial/mobile_spatial_ui.py
  - [ ] spatial/whiteboard_3d.py
  - [ ] spatial/knowledge_overlay.py
  - [ ] spatial/physics_objects.py
  - [ ] spatial/haptic_feedback.py
  - [ ] spatial/session_recording.py
  - [ ] spatial/presence.py
  - [ ] spatial/collaborative_session.py
  - [ ] spatial/voice_commands.py

### Causal Reasoning - 8 files
- [ ] HoloLoom/causal/ (NO tests)
  - [ ] causal/dag.py (Directed Acyclic Graph)
  - [ ] causal/discovery.py (Structure discovery)
  - [ ] causal/intervention.py (Do-calculus)
  - [ ] causal/counterfactual.py
  - [ ] causal/neural_scm.py (Structural Causal Model)
  - [ ] causal/query.py
  - [ ] causal/temporal.py

### Writing System - 21 files
- [ ] HoloLoom/writing/ (NO tests)
  - [ ] writing/core/writer.py
  - [ ] writing/core/composer.py
  - [ ] writing/core/protocol.py
  - [ ] writing/modes/creative.py
  - [ ] writing/modes/technical.py
  - [ ] writing/modes/narrative.py
  - [ ] writing/modes/analysis.py
  - [ ] writing/refinement/basic.py
  - [ ] writing/refinement/elegance.py
  - [ ] writing/refinement/verify.py
  - [ ] writing/export/html.py
  - [ ] writing/export/markdown.py
  - [ ] writing/templates/base.py
  - [ ] writing/templates/email.py
  - [ ] writing/templates/report.py

---

## MEDIUM PRIORITY (No Tests, Advanced/Specialized Systems)

### Advanced Analysis
- [ ] HoloLoom/clustering/ (4 files)
  - [ ] clustering/core.py
  - [ ] clustering/thompson.py
  - [ ] clustering/labeler.py

- [ ] HoloLoom/physics/ (Helmholtz Free Energy)
  - [ ] physics/physics_integration.py (if separate file)

- [ ] HoloLoom/cve/ (5 files - Cognitive Visual Extractors)
  - [ ] cve/cognitive_extractors.py
  - [ ] cve/cognitive_protocol.py
  - [ ] cve/cve_server.py
  - [ ] cve/tufte_renderer.py

### Infrastructure
- [ ] HoloLoom/llm/ (4 files)
  - [ ] llm/unified_client.py
  - [ ] llm/cost_tracker.py

- [ ] HoloLoom/infrastructure/ (15+ files)
  - [ ] infrastructure/kubernetes/* (K8s manifests)
  - [ ] infrastructure/grafana/* (Metrics)
  - [ ] infrastructure/sql/* (SQL integration)
  - [ ] infrastructure/mcp/* (MCP server)

- [ ] HoloLoom/telemetry/ (20+ files across 5 subdirs)
  - [ ] telemetry/analytics/*
  - [ ] telemetry/exporters/*
  - [ ] telemetry/metrics/*
  - [ ] telemetry/monitoring/*
  - [ ] telemetry/tracing/*

### Miscellaneous
- [ ] HoloLoom/synthesis/ (NO tests)
- [ ] HoloLoom/motif/ (NO tests)
- [ ] HoloLoom/resonance/ (Feature fusion)
- [ ] HoloLoom/fabric/ (4 files - Spacetime output)
  - [ ] fabric/spacetime.py
  - [ ] fabric/fabric.py
  - [ ] fabric/materializer.py

- [ ] HoloLoom/collaboration/ (Multi-user workspaces)
- [ ] HoloLoom/chrono/ (Temporal control)

---

## PARTIAL COVERAGE (Tests Exist But Gaps)

### Memory Systems (Coverage: 50%)
- [x] test_adaptive_expansion.py (Phase 1)
- [x] test_streaming_expansion.py (Phase 2)
- [x] test_interleaved_generation.py (Phase 3)
- [x] test_phase4_concurrent.py (Phase 4)
- [x] test_advanced_features.py
- [ ] ❌ awareness_graph.py (CRITICAL)
- [ ] ❌ spring_dynamics.py (HIGH)
- [ ] ❌ multi_wave_engine.py (HIGH)
- [ ] ❌ visual_compression.py (MEDIUM)
- [ ] ❌ graph.py operations (MEDIUM)
- [ ] ❌ memory/stores/* (HIGH)

### Agentic Reasoning (Coverage: 70%)
- [x] test_conscience_adapter.py
- [x] test_conscience_calibrator.py
- [x] test_conscience_e2e.py
- [x] test_orchestrator_conscience.py
- [x] test_per_step_gating.py
- [x] test_context_handoff.py
- [x] test_agentic_safety.py
- [ ] ❌ Core agentic modes (DIRECT/VERIFY/RESEARCH/PLAN_EXECUTE)
- [ ] ❌ Multi-query orchestration

### Search (Coverage: 70%)
- [x] test_matryoshka_search.py
- [x] test_cache.py
- [x] test_citation.py
- [ ] ❌ search/providers/* (Completely untested)

### Routing (Coverage: 30%)
- [x] test_query_classifier.py
- [ ] ❌ routing/context_aware/*
- [ ] ❌ routing/ml/*
- [ ] ❌ routing/learning/* (Pattern mining/validation)

### Voice (Coverage: 60%)
- [x] test_voice_agent.py
- [x] test_language.py
- [x] test_personality.py
- [x] test_tts_cache.py
- [ ] ❌ voice/languages/* (Language-specific)
- [ ] ❌ voice/personalities/* (Personality variants)
- [ ] ❌ voice/ux/* (User experience)

### Dark Trace (Coverage: 40%)
- [x] tests/test_core.py (SAE core)
- [ ] ❌ sae/tests/* (Sparse autoencoder variants)
- [ ] ❌ multilayer/tests/* (Circuit decomposition)
- [ ] ❌ visualization/tests/* (Interpretability viz)
- [ ] ❌ models/fingerprinting (Multi-model support)

---

## LEGACY/LOWER PRIORITY

- [ ] HoloLoom/promptly/ (Prompt CLI - legacy, integrated into alignment)
- [ ] HoloLoom/tools/ (Developer tools)
- [ ] HoloLoom/examples/ (Example code)
- [ ] HoloLoom/utils/ (Utility functions)
- [ ] HoloLoom/ml/ (ML utilities)
- [ ] HoloLoom/tuning/ (Hyperparameter tuning)
- [ ] HoloLoom/tui/ (Terminal UI)
- [ ] HoloLoom/workflows/ (Workflow definitions)
- [ ] HoloLoom/datapig/ (Data quality assurance)

---

## SUMMARY STATISTICS

**Total Modules Without Tests**: 83+
**Total Modules With Incomplete Tests**: 15+
**Total Python Files Untested**: 300+ estimated

**By Criticality**:
- CRITICAL (0% coverage): 35+ modules
- HIGH (1-50% coverage): 25+ modules
- MEDIUM (51-90% coverage): 15+ modules
- LOW/LEGACY (untested): 8+ modules

**Estimated Test Writing Effort**: 150-200+ hours for critical coverage

---

## QUICK COMMANDS

```bash
# Find all Python files without test files
find HoloLoom -name "*.py" -type f | while read f; do
  base=$(echo $f | sed 's/\.py$//')
  if [ ! -f "${base}_test.py" ] && [ ! -f "$(dirname $f)/tests/test_$(basename $f)" ]; then
    echo "$f"
  fi
done

# Count untested modules
find HoloLoom -type d -maxdepth 1 | while read dir; do
  if [ ! -d "$dir/tests" ]; then
    echo "NO TESTS: $dir"
  fi
done | wc -l

# Find empty test files
find HoloLoom -path "*/tests/test_*.py" -type f | while read f; do
  lines=$(wc -l < "$f")
  if [ $lines -lt 20 ]; then
    echo "MINIMAL TESTS ($lines lines): $f"
  fi
done
```

---

**Last Updated**: December 31, 2025
**Status**: Research phase complete, ready for implementation
