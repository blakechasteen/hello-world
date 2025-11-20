#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS-STAR Pattern
===============

Data Science with Self-Taught Autonomous Reasoning.

Based on Google Research paper, adapted for HoloLoom.

Quick Start:
```python
from hololoom.patterns.dsstar import DSStarOrchestrator

# Create orchestrator
orchestrator = DSStarOrchestrator(max_iterations=3)

# Process query
result = await orchestrator.process(
    query="What is the average sales by region?",
    data_file="sales_data.csv"
)

# Check results
print(result.summary())
if result.success:
    print("Outputs:", result.final_outputs)
```

Components:
- DataAnalyzer: Extract anchors from data files
- Planner: Create execution plans
- Executor: Run code safely
- Verifier: Check goal satisfaction
- Orchestrator: Iterative loop with refinement

Architecture:
1. Analyze → Extract data anchors (schema, types, samples)
2. Plan → Create execution steps based on query
3. Execute → Run code in safe sandbox
4. Verify → Check if goal achieved
5. Iterate → Refine and retry if needed
"""

# Core components
from .analyzer import (
    DataAnalyzer,
    DataProfile,
    DataAnchor,
    analyze_data
)

from .planner import (
    Planner,
    ExecutionPlan,
    PlanStep,
    ActionType,
    create_plan
)

from .executor import (
    SafeExecutor,
    ExecutionResult,
    execute_code
)

from .verifier import (
    Verifier,
    VerificationResult,
    verify_execution
)

from .orchestrator import (
    DSStarOrchestrator,
    DSStarResult,
    DSStarIteration,
    process_query
)

# Scratchpad integration (internal dialogue)
from .scratchpad.recursive_scratchpad import (
    RecursiveScratchpad,
    Thought,
    DialogueTree
)

from .scratchpad.internal_dialogue import (
    InternalDialogue,
    DialogueMode
)


__all__ = [
    # Analyzer
    'DataAnalyzer',
    'DataProfile',
    'DataAnchor',
    'analyze_data',

    # Planner
    'Planner',
    'ExecutionPlan',
    'PlanStep',
    'ActionType',
    'create_plan',

    # Executor
    'SafeExecutor',
    'ExecutionResult',
    'execute_code',

    # Verifier
    'Verifier',
    'VerificationResult',
    'verify_execution',

    # Orchestrator
    'DSStarOrchestrator',
    'DSStarResult',
    'DSStarIteration',
    'process_query',

    # Scratchpad
    'RecursiveScratchpad',
    'Thought',
    'DialogueTree',
    'InternalDialogue',
    'DialogueMode',
]


# Version
__version__ = "1.0.0"
