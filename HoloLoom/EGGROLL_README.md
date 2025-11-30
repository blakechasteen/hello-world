# EGGROLL-HoloLoom Integration

**A Gradient-Free Evolutionary Training Framework for Distributed, Symbolic, Federated AI Agents**

Reference: [arXiv:2511.16652](https://arxiv.org/abs/2511.16652)

## Overview

This directory contains the implementation of the EGGROLL integration for HoloLoom. This framework transforms the Loom into a distributed evolutionary trainer using low-rank perturbations (LoRA-based Evolution Strategies).

## Components

*   **Shuttle (`HoloLoom/shuttle/eggroll_shuttle.py`)**: Exploration Orchestrator. Uses Neural Thompson Sampling to propose perturbations.
*   **Warp (`HoloLoom/warp/eggroll_warp.py`)**: Evaluator. Embeds outputs and computes scores (coherence, novelty, etc.).
*   **Yarn (`HoloLoom/yarn/eggroll_yarn.py`)**: Lineage + Memory. Logs ancestry and perturbation traces.
*   **Weave (`HoloLoom/weaving/eggroll_weave.py`)**: Data + Reward Fabric. Generates tasks and computes composite rewards.
*   **Nodes (`HoloLoom/eggroll/nodes.py`)**: Federated Loom Nodes. Apply perturbations and run inference.
*   **MirrorCore (`HoloLoom/eggroll/mirror_core.py`)**: Agent wrapper that maintains weights and applies updates.

## Usage

To run the evolutionary training loop:

```bash
python HoloLoom/eggroll/integration.py
```

## Integration with Weaving Orchestrator

The EGGROLL framework runs as a training loop (`integration.py`). The `MirrorCoreAgent` represents the model being trained. In a production setting, the updated weights from `MirrorCoreAgent` would be hot-swapped into the `WeavingOrchestrator`'s inference model.

## Architecture

```
Weave → Shuttle → Federated Nodes → Warp → Yarn → Parameter Update → MirrorCore Agent
```

1.  **Weave** generates tasks.
2.  **Shuttle** proposes low-rank perturbations (A, B matrices).
3.  **Nodes** apply perturbations and evaluate tasks.
4.  **Warp** scores the outputs.
5.  **Yarn** logs the lineage.
6.  **MirrorCore** updates weights based on aggregated rewards.
