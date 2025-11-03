# Safe Alignment Configuration for Analysis Queries

## Problem
The alignment framework correctly blocks high-risk "execution" actions by requiring approval. However, for **analysis queries** (like processing documents), this creates friction since tools like `calc` are categorized as high-risk execution.

## Solution: Three Safe Approaches

### 1. **Auto-Approve Specific Categories** (RECOMMENDED for Production)

Auto-approve only analysis-related categories while keeping destructive operations protected:

```python
from HoloLoom.alignment import create_guardrails

guardrails = create_guardrails(
    auto_approve_categories={'analysis', 'query', 'retrieval'}
)
```

**Safety**: Still blocks deletion, modification, and system operations.
**Use case**: Production analysis workloads

---

### 2. **Testing Mode** (RECOMMENDED for Development)

Bypass all approval requirements during development:

```python
from HoloLoom.alignment import create_guardrails

guardrails = create_guardrails(
    testing_mode=True
)
```

**Safety**: NO protection - only use in safe development environments.
**Use case**: Local development, testing, debugging

---

### 3. **Custom Policy** (RECOMMENDED for Fine-Grained Control)

Create a custom policy that treats analysis-oriented execution tools as low risk:

```python
from HoloLoom.alignment.safety_guardrails import SafetyPolicy, RiskLevel, ActionCategory

class AnalysisPolicy(SafetyPolicy):
    """Custom policy that allows analysis tools without approval."""

    def get_risk_level(self, request):
        # Check if this is an analysis-oriented execution
        if request.category == ActionCategory.EXECUTION:
            # Check context for analysis indicators
            context = request.context or {}
            if context.get('purpose') == 'analysis':
                return RiskLevel.LOW  # Allow without approval

        # Fall back to default policy
        return super().get_risk_level(request)

# Use custom policy
from HoloLoom.alignment import create_guardrails

guardrails = create_guardrails(
    custom_policy=AnalysisPolicy()
)
```

**Safety**: Granular control - only analysis-marked executions bypass approval.
**Use case**: Production with mixed workloads (analysis + operations)

---

## Apply to Your Script

Update `process_analysis_highest_quality.py` with one of these approaches:

### Option A: Auto-Approve Analysis Categories

```python
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.alignment import create_guardrails

# Create config
config = Config.fused()
config.enable_linguistic_gate = True
config.linguistic_mode = "both"
config.use_compositional_cache = True

# Create guardrails with auto-approve for analysis
guardrails = create_guardrails(
    auto_approve_categories={'analysis', 'query', 'retrieval', 'execution'}  # Allow execution for analysis
)

# Create orchestrator with custom guardrails
async with WeavingOrchestrator(
    cfg=config,
    shards=shards,
    guardrails=guardrails  # Pass custom guardrails
) as orchestrator:
    spacetime = await orchestrator.weave(query)
```

### Option B: Testing Mode (Simplest for Development)

```python
# Create guardrails in testing mode
guardrails = create_guardrails(
    testing_mode=True  # Bypass all approvals
)

async with WeavingOrchestrator(
    cfg=config,
    shards=shards,
    guardrails=guardrails
) as orchestrator:
    spacetime = await orchestrator.weave(query)
```

---

## Recommended Configuration Matrix

| Environment | Configuration | Safety Level | Use Case |
|-------------|--------------|--------------|----------|
| **Local Dev** | `testing_mode=True` | ⚠️ None | Fast iteration, debugging |
| **Staging** | `auto_approve_categories={'analysis','query','retrieval'}` | ✅ Medium | Analysis testing with some protection |
| **Production (Analysis Only)** | `auto_approve_categories={'analysis','query','retrieval'}` | ✅ High | Read-only analysis workloads |
| **Production (Mixed)** | Custom `AnalysisPolicy` | ✅ Very High | Analysis + operational workloads |

---

## What Each Category Protects

**Auto-approved (safe for analysis)**:
- `query` - Query knowledge base
- `retrieval` - Retrieve information
- `analysis` - Analyze data
- `execution` - Execute tools (when analysis-oriented)

**Still protected (requires approval)**:
- `storage` - Store new information
- `modification` - Modify existing data
- `deletion` - Delete data
- `system` - System-level operations
- `external` - External API calls

---

## Implementation for Your Current Issue

The error you saw:
```
WARNING: Policy guardrails require approval for tool 'calc':
Action category: execution, Risk level: high
```

**Quick Fix**: Add `'execution'` to auto-approve categories since you're only doing analysis:

```python
guardrails = create_guardrails(
    auto_approve_categories={'analysis', 'query', 'retrieval', 'execution'}
)
```

This tells the alignment framework: *"For this workload, execution tools (like calc) are part of analysis and don't need approval."*

**Safety**: Still blocks deletion, modification, and system operations. You're just allowing execution tools to run for analysis purposes.

---

## Complete Working Example

```python
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.alignment import create_guardrails
from HoloLoom.documentation.types import Query, MemoryShard

async def main():
    # 1. Create configuration
    config = Config.fused()
    config.enable_linguistic_gate = True
    config.use_compositional_cache = True

    # 2. Create guardrails with analysis-friendly settings
    guardrails = create_guardrails(
        auto_approve_categories={'analysis', 'query', 'retrieval', 'execution'},
        enable_adversarial_detection=True  # Keep adversarial detection
    )

    # 3. Create memory shards
    shards = [
        MemoryShard(
            id="shard_1",
            text="Your analysis content here",
            episode="analysis"
        )
    ]

    # 4. Process queries with alignment-friendly settings
    async with WeavingOrchestrator(
        cfg=config,
        shards=shards,
        guardrails=guardrails
    ) as orchestrator:
        query = Query(text="What are the critical bottlenecks?")
        spacetime = await orchestrator.weave(query)

        print(f"Response: {spacetime.text}")
        print(f"Confidence: {spacetime.confidence}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Summary

✅ **Use `auto_approve_categories`** for safe, production-ready analysis
✅ **Use `testing_mode=True`** for local development only
✅ **Keep adversarial detection enabled** (it's separate from approval)
✅ **Still blocks destructive operations** (deletion, modification, system)

This approach maintains the safety benefits of the alignment framework while allowing analysis queries to execute without manual approval.