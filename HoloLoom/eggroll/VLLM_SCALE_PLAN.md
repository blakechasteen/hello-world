# EGGROLL Hyper-Scale Architecture: Justice Protocol (Phase II)

## 1. Executive Summary
This document outlines the expansion plan to transition **EGGROLL** from a simulated evolutionary loop to a **high-throughput, Multi-LoRA parallel evolution engine** using **vLLM** and **Ray**.

**The Goal**: Achieving "Real-Time Evolution" where the system can fork, mutate, and evaluate 50-1000+ variations of itself in seconds, eventually scaling across multiple GPUs or nodes.

**The Key Enablers**:
1.  **vLLM** (Multi-LoRA Inference)
2.  **Ray** (Distributed Worker Orchestration)
3.  **Offline Consolidation** ("Sleep" Learning)

---

## 2. The "Tri-Engine" Architecture

To maximize performance, we introduce a third layer for orchestration.

### Engine A: The Interface (Ollama / Light-LLM)
*   **Role**: Latency-critical user interaction.
*   **Responsibility**:
    *   Maintains the "Persona".
    *   Streaming tokens to the UI.
    *   Quick RAG formatting.
*   **State**: Holds the currently active "Master Adapter".

### Engine B: The Forge (vLLM)
*   **Role**: Massive batch throughput.
*   **Responsibility**:
    *   Serving the frozen **Base Model** (e.g., Llama-3-70B).
    *   Dynamically swapping **hundreds** of LoRA adapters per forward pass.
    *   Handling the "Population Evaluation" phase.

### Engine C: The Hive (Ray Cluster)
*   **Role**: Orchestration & Distribution.
*   **Responsibility**:
    *   Distributing tasks across multiple vLLM instances (if using >1 GPU).
    *   Managing the lifecycle of "Mutant" workers.
    *   Aggregating results for the `EggrollIntegration` controller.

---

## 3. The Hyper-Loop (Detailed Cycle)

This is the concrete workflow for a single "Epoch" of thought.

### Phase 1: Mitosis (The Fork)
**Responsible**: `MirrorCore`
*   **Action**: Instead of modifying the *live* model, we generate N sets of `lora_B` delta matrices using fast Numpy operations.
*   **IO Strategy**:
    *   Use a **RamDisk** (`/dev/shm` on Linux) to store temporary Adapter Configs.
    *   Write 50 lightweight JSON/Bin files.
    *   Time cost: < 50ms for 50 rank-16 adapters.

### Phase 2: The Assessment (Async Batch)
**Responsible**: `VLLMInferenceEngine`
*   **Batch Construction**: Create a single "Mega-Batch" of prompts.
    *   *Self-Critique*: "Does this code compile?"
    *   *Creativity*: "Is this response novel?"
*   **Execution**:
    *   Ray dispatches the batch to the vLLM engine.
    *   vLLM uses **PagedAttention** to process all 50 mutants in parallel.
    *   Memory footprint: Only 1 Base Model copy + small overhead per adapter.
*   **Throughput**: 50 variants checked in the time it takes to generate ~1.5 standard responses.

### Phase 3: Natural Selection (Scoring)
**Responsible**: `Warp` (The Judge)
*   LLM-as-a-Judge is too slow. We use **Vectorized Reward Models**:
    1.  **Semantic Distance**: Embed outputs (GPU-accelerated `sentence-transformers`) and check cosine similarity to a "Gold Standard" or "Goal Vector".
    2.  **Constraint Checking**: Regex/AST parsing for code (100% fast).
    3.  **Heuristic Filters**: Length, repetition penalties.

### Phase 4: Evolution (Update)
**Responsible**: `EggrollIntegration`
*   Compute **Evolution Strategy (ES)** gradient update.
*   Apply update to the "Master Adapter".
*   **Real-Time Injection**:
    *   Save `master_v{N+1}` to disk.
    *   Signal Engine A (Ollama) to hot-swap the adapter.

---

## 4. Advanced Capabilities (Phase II Expansion)

### A. "Sleep Mode" (Memory Consolidation)
Training on adapters indefinitely creates "drift" and instability.
*   **The Process**:
    *   Every night (or after X epochs), the system enters "Sleep Mode".
    *   **Merge**: The best LoRA weights are merged permanently into a new "Checkpoint" of the base model.
    *   **Prune**: Failed branches (adapters) are deleted.
    *   **Distill**: High-scoring interaction traces are saved to a dataset for traditional Supervised Fine-Tuning (SFT).

### B. "The Guardian" (Automated Red Teaming)
Evolution can lead to "Reward Hacking" (e.g., the model generating gibberish that statistically looks like a good answer).
*   **The Protocol**:
    *   Before any "Mutant" answer is shown to the user, it passes through a lightweight **Safety Filter** (e.g., Llama-Guard or a Regex filter).
    *   Any mutant triggering the filter gets a Reward of -infinity, effectively killing that lineage.

### C. Method Acting (Persona Scaling)
*   **Concept**: We can evolve different *parts* of the network for different roles.
    *   *Pattern: "Coding"* -> Evolve MLP Layers 10-20.
    *   *Pattern: "Creative"* -> Evolve Attention Layers 20-30.
*   **Implementation**: `MirrorCore` defines different `target_modules` in the LoRA config based on the selected **Pattern Card**.

---

## 5. Infrastructure Requirements (The Rig)

To run this "Justice" architecture locally:

*   **OS**: Windows 11 with **WSL2** (Ubuntu 22.04) or pure Linux.
*   **GPU**:
    *   *Entry*: NVIDIA RTX 3090/4090 (24GB). Supports ~7B models with ~50 adapters.
    *   *Pro*: Dual RTX 3090/4090 (48GB NVLink). Supports 70B models (4-bit) or massive batches.
*   **Software Stack**:
    *   `vllm` (Inference)
    *   `ray[default]` (Orchestration)
    *   `torch`
    *   `peft` (Adapter management)

---

## 6. Implementation Stages (Revised)

### Stage 1: The vLLM Bridge (Done/Prototype)
*   [x] `VLLMInferenceEngine` class structure.
*   [x] Mock/Simulation fallback.

### Stage 2: The Adapter Factory
*   [ ] `MirrorCore.spawn_mutants(n, scale)`: Fast concurrent disk I/O.
*   [ ] `RamDisk` management for temp files.

### Stage 3: The Parallel Warp
*   [ ] Batch scoring engine (Vectorized).

### Stage 4: Ray Integration
*   [ ] Wrap `LoomNode` in `@ray.remote`.
*   [ ] Create a `RayLeaf` actor for managing vLLM handles.

### Stage 5: The Interface
*   [ ] "Live MRI": A UI view showing the active "Hot" neurons or active Adapter paths lighting up during generation.
