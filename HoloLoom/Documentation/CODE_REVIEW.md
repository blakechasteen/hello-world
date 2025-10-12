# HoloLoom Orchestrator – Initial Code Review

This document captures a first-pass review of the new `HoloLoom` orchestrator codebase. It summarizes the core pipeline, highlights strengths, and documents issues or risks that should be addressed before relying on the system in production.

## High-level architecture

* **Entry point:** `HoloLoom/Orchestrator.py` defines configuration (`Config`), memory shards (`MemoryShard`), the retrieval stack, spectral feature extraction, a neural policy core, and I/O dataclasses. The async `main()` demo exercises the full pipeline over demo shards.
* **Type surface:** The repo ships an independent `Modules/Types.py` file with light-weight dataclasses that mirror some of the types defined inline in the orchestrator. The orchestrator currently duplicates similar structures rather than reusing these shared types.

## Strengths

1. **Graceful degradation of optional dependencies.** spaCy, sentence-transformers, BM25, and SciPy all degrade with warnings, letting the system run in limited environments.
2. **Spectral retrieval features.** `SpectralFusion.features` fuses graph spectra with embedding-derived topic statistics, giving downstream policies richer signals than raw similarity scores alone.
3. **Policy modularity.** `UnifiedPolicy` cleanly separates the neural core from adapter selection and bandit exploration logic, making it easier to iterate on tool routing without rewriting retrieval or memory code.

## Issues & risks

1. **Disconnected bandit feedback.** `UnifiedPolicy.decide` samples an arm from `TSBandit`, but ultimately selects the tool using the neural core’s argmax. The arm that receives a reward is therefore unrelated to the chosen tool, so the Thompson sampler never learns from outcomes. Aligning the arm selection with the tool choice (or updating the bandit with `tool_idx`) is necessary for meaningful exploration.
2. **Background task lifecycle.** `MemoryManager._run_archiver()` spawns a fire-and-forget task on construction. There is no shutdown hook, and instantiating the orchestrator outside an active event loop will raise `RuntimeError`. Consider injecting an event loop handle, adding explicit start/stop hooks, or delaying task creation until `process()` is first awaited.
3. **Duplication with shared types.** The orchestrator redefines `Query`, `Context`, `Response`, and feature dataclasses instead of reusing the `Modules/Types.py` abstractions. Consolidating these would avoid divergence in future updates.
4. **Empty feature module.** `Modules/Features.py` is currently empty, so any downstream import will fail at runtime. Either populate the module or remove references until the implementation lands.
5. **Demo dependency footprint.** Running the demo requires heavy libraries (`numpy`, `torch`, optional NLP toolkits). Document minimal install instructions or provide a lightweight mock runner so contributors can validate changes without a full ML stack.

## Suggested next steps

1. Align the bandit’s chosen arm with the executed tool (or update rewards using the actual tool index).
2. Add lifecycle management for background tasks created in the memory manager.
3. Centralize dataclass definitions in `Modules/Types.py` and have the orchestrator import them.
4. Flesh out `Modules/Features.py` or remove it from the module surface.
5. Ship an installation section in the README (or a requirements file) so the demo can run without manual dependency discovery.
