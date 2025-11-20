"""
HoloLoom Chaining System - Declarative Prompt Chain Orchestration

**Purpose**: Provides declarative chain definition and execution for sequencing
HoloLoom department operations. Enables building complex workflows from simple
step definitions with automatic context passing, conditional branching, and
loop support.

**Status**: ✅ Complete (November 2025)
**Location**: `HoloLoom/chaining/`
**Total Code**: ~2,500 lines

**Key Features**:
- Declarative chain definition (Python dataclasses)
- Sequential step execution with context passing
- Conditional branching (if/else logic)
- Loop support (while loops with exit conditions)
- Error handling and rollback
- Pre-built patterns for common workflows
- Execution tracing and debugging
- Full async/await support

**Quick Start**:
```python
from HoloLoom.chaining import Chain, ChainStep, StepType, ChainPatterns
from HoloLoom.chaining import ChainOrchestrator
from HoloLoom.departments.rag_department import RAGDepartment

# Use a pre-built pattern
chain = ChainPatterns.verified_query()  # execute → verify

# Or define custom chain
chain = Chain(name="MyChain")
chain.add_step("execute", ChainStep(
    step_type=StepType.EXECUTE,
    params={"mode": "verify", "max_sources": 5}
))

# Execute
async with RAGDepartment() as rag_dept:
    orchestrator = ChainOrchestrator(rag_dept)
    result = await orchestrator.execute_chain(chain, "What is Thompson Sampling?")
    print(result.final_response)
    print(result.trace)  # Execution trace for debugging
```

**Architecture**:
- `chain.py` - Core chain definition (Chain, ChainStep, StepType)
- `orchestrator.py` - Chain execution engine
- `patterns.py` - Pre-built chain patterns
- `conditions.py` - Common condition helpers
- `types.py` - Result types and data structures

**Integration**:
Works with any department implementing the DepartmentProtocol:
- RAG Department (SimpleRAG queries, verification, refinement)
- Quality Assurance Department (code analysis, fixing)
- Context Department (contextual reasoning)
- Any custom department

Author: HoloLoom Architecture Team
Date: November 2025
"""

from .chain import Chain, ChainStep, StepType
from .orchestrator import ChainOrchestrator, ChainResult, ExecutionTrace
from .patterns import ChainPatterns
from .conditions import Conditions, CommonConditions
from .types import StepResult, LoopConfig, ConditionalBranch

__all__ = [
    "Chain",
    "ChainStep",
    "StepType",
    "ChainOrchestrator",
    "ChainResult",
    "ExecutionTrace",
    "ChainPatterns",
    "Conditions",
    "CommonConditions",
    "StepResult",
    "LoopConfig",
    "ConditionalBranch",
]
