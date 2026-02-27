# HoloLoom: Frontier Expansion Plan (Sci-Fi Roadmap)

**Version**: 0.1.0-alpha (Frontier Edition)
**Date**: December 7, 2025
**Scope**: Advanced Architectures, Distributed Evolution, and Semantic Steering (Dark Trace)
**Target Hardware**: OLCF Frontier (AMD MI250X)

---

## 🌌 Executive Summary

This roadmap outlines the progression of HoloLoom from a local prototyping environment to a **massive-scale distributed intelligence forge** running on supercomputing infrastructure. We introduce "Sci-Fi" architectures (Recursive, Liquid, Neuromorphic) and "Dark Trace" interpretability tools to ensure these complex models remain aligned and consistent.

**Core Objectives:**
1.  **Exotic Architectures**: Validate TRM (Tiny Recursive Models) and Liquid Neural Networks at scale.
2.  **Distributed Evolution**: Scale `EggrollIntegration` from local threads to thousands of GPU nodes.
3.  **Mind Reading & Steering**: Operationalize `DarkTrace` for automated consistency enforcement.
4.  **Budget-Aware Reasoning**: Implement "Thinking Budgets" via dynamic recursion depth.

---

## 📅 Timeline Overview

```
Phase 1: Architecture Maturity (Week 1-2)
  - Verify TRM/Liquid/Spiking correctness.
  - Optimize memory usage for "Tiny" models.
  - Benchmarking on single-node GPU.

Phase 2: Dark Trace "First Light" (Week 3-4)
  - Automate SAE training during Evolution.
  - Develop "Auto-Steerer" for consistency.
  - Demonstrate "Mind Control" on TRM.

Phase 3: Frontier Scaling (Week 5-8)
  - Port `EggrollIntegration` to Ray/TorchDistributed.
  - Implement Sharded Large Models (Teacher).
  - 10,000+ Agent Population experiments.

Phase 4: The Holo-Mind (Week 9+)
  - Unified Memory across population.
  - Collective Steering (Hive Mind alignment).
```

---

## 🛠️ Phase 1: Architecture Maturity

**Goal**: Ensure our exotic models are robust, efficient, and ready for evolution.

### Deliverables

#### 1. Validated TRM (Tiny Recursive Model)
*   **Action**: Create unit tests verifying gradient flow through 50+ recursion steps.
*   **Optimization**: Implement "Gradient Checkpointing" for infinite depth without OOM.
*   **File**: `hololoom/eggroll/architectures.py` (Enhancement)

#### 2. Liquid Neural Network (LNN) Stability
*   **Action**: Fix numerical instability in ODE solvers for long sequences.
*   **Feature**: Adaptive time-stepping for processing irregular data streams.

#### 3. Neuromorphic Spiking Encoders
*   **Action**: Implement surrogate gradients (e.g., SuperSpike) for backprop.
*   **Goal**: Demonstrate < 10% energy usage compared to dense Transformers.

---

## 🌑 Phase 2: Dark Trace "First Light"

**Goal**: Turn "Mind Reading" from a demo into a core safety protocol.

### Deliverables

#### 1. Automated MindProbe
*   **Concept**: A probe that attaches to *every* agent born in the Eggroll hatchery.
*   **Mechanism**:
    *   buffer activations -> mini-batch SAE train -> extract top features -> log to Yarn.
*   **File**: `hololoom/dark_trace/auto_probe.py`

#### 2. The Semantic Steering Loop (Auto-Alignment)
*   **Concept**: If an agent deviates (low consistency), the system automatically:
    1.  Identifies the "Confusion" feature via SAE.
    2.  Injects a corrective Steering Vector (negative confusion).
    3.  Re-evaluates.
*   **File**: `hololoom/dark_trace/steering_policy.py`

#### 3. Thinking Budget Protocol
*   **Concept**: Formalize `recur_depth` as a resource.
*   **API**: `agent.set_budget(tokens: int)`
*   **Logic**: Complex queries get high budget; simple greetings get low budget.

---

## 🚀 Phase 3: Frontier Scaling

**Goal**: Unleash the swarm on OLCF Frontier.

### Deliverables

#### 1. Ray / MPI Backend for Eggroll
*   **Problem**: `asyncio` is concurrent, not parallel. Python GIL limits us.
*   **Solution**:
    *   **Master Node**: Runs `EggrollIntegration` (Evolution Strategy).
    *   **Worker Nodes**: 100s of Ray Actors running `LoomNode` + `MirrorCoreAgent`.
*   **File**: `hololoom/eggroll/distributed_backend.py`

#### 2. Sharded Large Model (The Teacher)
*   **Concept**: A 70B+ model (Llama-3) sharded across 8 GPUs acting as the "Reference".
*   **Usage**: TRM agents try to mimic the Teacher's output (Distillation) + maximize Reward.

#### 3. Mass Evolution Experiment
*   **Config**: Population=10,000, Generations=500.
*   **Metric**: "Reasoning per Parameter" (Target: TRM beats dense Transformer).

---

## 🧠 Phase 4: The Holo-Mind

**Goal**: Collective intelligence.

*   **Unified Memory**: All agents share a vector database (already in user's prompt history).
*   **Collective Steering**: Identify a "Goal Vector" for the entire swarm and steer all agents towards it simultaneously.

---

## 📋 Immediate Action Items (Next 48 Hours)

1.  [ ] **Benchmarking**: Run `test_architectures_demo.py` with `recur_depth=100` to check memory usage.
2.  [ ] **Probe Integration**: Modify `LoomNode` to accept a `MindProbe` in its constructor.
3.  [ ] **Steering UI**: (Optional) Simple TUI to adjust steering vectors live.

---

**Signed**: *Antigravity* (Lead Architect)
