# Tutorial: Build an Agentic Workflow

Create a multi-step reasoning workflow with verification, planning, and execution capabilities.

## Overview

In this tutorial, you'll build an agentic workflow that:
1. Analyzes a complex question
2. Creates an execution plan
3. Executes steps with verification
4. Synthesizes final results with confidence scores

**Time**: ~30 minutes
**Difficulty**: Advanced
**Prerequisites**: [RAG Pipeline Tutorial](rag-pipeline.md)

## What You'll Build

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Input   │───▶│ Analyzer │───▶│ Planner  │───▶│ Executor │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                      │
                     ┌────────────────────────────────┘
                     ▼
              ┌──────────┐    ┌──────────┐    ┌──────────┐
              │ Verifier │───▶│ Synth    │───▶│  Output  │
              └──────────┘    └──────────┘    └──────────┘
```

## Step 1: Create the Workflow

1. Open the Workflow Builder
2. Click **New Workflow** (or `Ctrl+N`)
3. Name it "Agentic Reasoning Pipeline"

## Step 2: Add Input Node

1. From **I/O** palette, drag **Input Node**
2. Configure:
   - **Label**: "Complex Question"
   - **Input Type**: `object`
   - **Schema**:

```json
{
  "question": "string",
  "context": "string (optional)",
  "max_steps": "number (default: 5)"
}
```

## Step 3: Add Query Analyzer

The analyzer determines the question type and complexity.

1. From **Processing** palette, drag **Synthesizer** node
2. Configure:
   - **Label**: "Query Analyzer"
   - **Mode**: `analyze`
   - **Output Schema**:

```javascript
// Query Analyzer Configuration
{
  "label": "Query Analyzer",
  "mode": "analyze",
  "analysis_dimensions": [
    "question_type",      // factual, analytical, procedural
    "complexity",         // simple, moderate, complex
    "required_knowledge", // domains needed
    "ambiguity_level"     // clear, ambiguous, very_ambiguous
  ],
  "output_format": {
    "type": "object",
    "properties": {
      "question_type": "string",
      "complexity": "string",
      "sub_questions": "array",
      "estimated_steps": "number"
    }
  }
}
```

3. Connect **Input** → **Query Analyzer**

## Step 4: Add Plan Generator

Creates an execution plan based on analysis.

1. From **Processing** palette, drag **Synthesizer** node
2. Configure:
   - **Label**: "Plan Generator"
   - **Mode**: `plan`

```javascript
// Plan Generator Configuration
{
  "label": "Plan Generator",
  "mode": "plan",
  "planning_strategy": "decompose",  // or "chain", "tree"
  "max_steps": "${input.max_steps || 5}",
  "step_template": {
    "action": "string",
    "expected_output": "string",
    "dependencies": "array",
    "verification_required": "boolean"
  }
}
```

3. Connect **Query Analyzer** → **Plan Generator**

## Step 5: Add Loop Iterator

Executes each step in the plan.

1. From **Control Flow** palette, drag **Loop Iterator**
2. Configure:
   - **Label**: "Step Executor"
   - **Iterator Variable**: `current_step`
   - **Collection**: `${plan.steps}`
   - **Max Iterations**: `10`

```
Loop Configuration:
┌─────────────────────────────────────┐
│ Label: Step Executor                │
│ Collection: [${plan.steps}]         │
│ Variable: current_step              │
│ Max Iterations: [10]                │
│ ☑ Continue on Error                 │
│ ☑ Collect Results                   │
└─────────────────────────────────────┘
```

3. Connect **Plan Generator** → **Loop Iterator**

## Step 6: Add Step Execution Logic (Inside Loop)

Inside the loop, add the execution chain:

### 6a. HoloLoom Query Node

1. Drag **HoloLoom Query** inside the loop body
2. Configure:
   - **Label**: "Execute Step"
   - **Query**: `${current_step.action}`
   - **Mode**: `fused`

```javascript
{
  "label": "Execute Step",
  "query_template": "${current_step.action}",
  "mode": "fused",
  "include_reasoning": true,
  "timeout_ms": 30000
}
```

### 6b. Conditional Verification

1. Drag **Conditional Branch** after HoloLoom Query
2. Configure:
   - **Condition**: `${current_step.verification_required}`

```
Branch Configuration:
┌─────────────────────────────────────┐
│ Condition: ${current_step.          │
│            verification_required}   │
│                                     │
│ True Branch ─▶ [Verification Node]  │
│ False Branch ─▶ [Next Step]         │
└─────────────────────────────────────┘
```

### 6c. Verification Node (True Branch)

1. Drag **Synthesizer** for verification
2. Configure:
   - **Label**: "Verify Result"
   - **Mode**: `verify`

```javascript
{
  "label": "Verify Result",
  "mode": "verify",
  "verification_checks": [
    "factual_accuracy",
    "logical_consistency",
    "completeness"
  ],
  "min_confidence": 0.7,
  "retry_on_fail": true,
  "max_retries": 2
}
```

## Step 7: Add Results Aggregator

After the loop, aggregate all step results.

1. From **Processing** palette, drag **Synthesizer**
2. Position after the **Loop Iterator**
3. Configure:
   - **Label**: "Results Aggregator"
   - **Mode**: `aggregate`

```javascript
{
  "label": "Results Aggregator",
  "mode": "aggregate",
  "aggregation_strategy": "weighted_merge",
  "weights": {
    "verified_results": 1.0,
    "unverified_results": 0.7
  },
  "deduplication": true,
  "conflict_resolution": "highest_confidence"
}
```

## Step 8: Add Final Synthesizer

Creates the final coherent response.

1. Drag another **Synthesizer** node
2. Configure:
   - **Label**: "Final Synthesis"
   - **Mode**: `synthesize`

```javascript
{
  "label": "Final Synthesis",
  "mode": "synthesize",
  "synthesis_style": "comprehensive",
  "include_confidence": true,
  "include_reasoning_trace": true,
  "format": "markdown",
  "sections": [
    "summary",
    "detailed_findings",
    "confidence_assessment",
    "limitations"
  ]
}
```

## Step 9: Add Output Node

1. From **I/O** palette, drag **Output Node**
2. Configure:
   - **Label**: "Agentic Result"
   - **Output Schema**:

```json
{
  "answer": "string",
  "confidence": "number",
  "reasoning_steps": "array",
  "verification_results": "array",
  "metadata": {
    "total_steps": "number",
    "verified_steps": "number",
    "execution_time_ms": "number"
  }
}
```

## Complete Workflow Structure

```
┌────────────┐
│  Complex   │
│  Question  │
└─────┬──────┘
      │
      ▼
┌────────────┐
│   Query    │
│  Analyzer  │
└─────┬──────┘
      │
      ▼
┌────────────┐
│    Plan    │
│  Generator │
└─────┬──────┘
      │
      ▼
┌─────────────────────────────────────────┐
│            Loop Iterator                 │
│  ┌────────────┐    ┌────────────┐       │
│  │  Execute   │───▶│  Verify?   │       │
│  │   Step     │    │ (if needed)│       │
│  └────────────┘    └────────────┘       │
└─────────────────────────────────────────┘
      │
      ▼
┌────────────┐
│  Results   │
│ Aggregator │
└─────┬──────┘
      │
      ▼
┌────────────┐
│   Final    │
│  Synthesis │
└─────┬──────┘
      │
      ▼
┌────────────┐
│  Agentic   │
│   Result   │
└────────────┘
```

## Step 10: Test the Pipeline

1. Click **Execute** (▶️) or press `Ctrl+Enter`
2. Enter test input:

```json
{
  "question": "Compare Thompson Sampling with UCB for multi-armed bandits, including theoretical guarantees and practical performance",
  "max_steps": 5
}
```

3. Watch execution progress through each node

**Expected Output**:
```json
{
  "answer": "## Comparison: Thompson Sampling vs UCB\n\n### Summary\nThompson Sampling and UCB are...",
  "confidence": 0.87,
  "reasoning_steps": [
    {
      "step": 1,
      "action": "Define Thompson Sampling",
      "result": "Thompson Sampling is a Bayesian...",
      "verified": true
    },
    {
      "step": 2,
      "action": "Define UCB algorithm",
      "result": "Upper Confidence Bound (UCB)...",
      "verified": true
    },
    {
      "step": 3,
      "action": "Compare theoretical properties",
      "result": "Both algorithms achieve...",
      "verified": true
    }
  ],
  "verification_results": [
    {"step": 1, "passed": true, "confidence": 0.92},
    {"step": 2, "passed": true, "confidence": 0.89},
    {"step": 3, "passed": true, "confidence": 0.85}
  ],
  "metadata": {
    "total_steps": 5,
    "verified_steps": 5,
    "execution_time_ms": 3450
  }
}
```

## Step 11: Add Error Handling (Optional)

Add robust error handling for production:

1. From **Control Flow** palette, drag **Conditional Branch**
2. Insert after **Loop Iterator**
3. Configure error detection:

```javascript
{
  "condition": "${loop_results.errors.length > 0}",
  "true_branch": "error_handler",
  "false_branch": "results_aggregator"
}
```

4. Add **Error Handler** node:

```javascript
{
  "label": "Error Handler",
  "strategy": "graceful_degradation",
  "fallback_response": true,
  "log_errors": true,
  "retry_failed_steps": false
}
```

## Step 12: Export as Python

Export the workflow for programmatic use:

```python
"""
Workflow: Agentic Reasoning Pipeline
Description: Multi-step reasoning with verification
Generated: 2025-12-15
"""

import asyncio
from typing import List, Dict, Any
from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.config import Config
from hololoom.protocols.types import Query


async def run_agentic_pipeline(
    question: str,
    context: str = None,
    max_steps: int = 5
) -> Dict[str, Any]:
    """Execute the agentic reasoning workflow."""

    config = Config.fused()

    async with WeavingOrchestrator(cfg=config) as orchestrator:
        # Step 1: Analyze the question
        analysis = await analyze_question(question, context)

        # Step 2: Generate execution plan
        plan = await generate_plan(analysis, max_steps)

        # Step 3: Execute steps with verification
        results = []
        for step in plan['steps']:
            step_result = await execute_step(orchestrator, step)

            if step.get('verification_required', True):
                step_result = await verify_result(step_result)

            results.append(step_result)

        # Step 4: Aggregate and synthesize
        aggregated = aggregate_results(results)
        final = await synthesize_response(aggregated, question)

        return {
            'answer': final['response'],
            'confidence': final['confidence'],
            'reasoning_steps': results,
            'metadata': {
                'total_steps': len(results),
                'verified_steps': sum(1 for r in results if r.get('verified')),
                'execution_time_ms': final['timing']['total_ms']
            }
        }


async def analyze_question(question: str, context: str = None) -> Dict:
    """Analyze question type and complexity."""
    # Implementation uses Synthesizer in analyze mode
    pass


async def generate_plan(analysis: Dict, max_steps: int) -> Dict:
    """Generate execution plan based on analysis."""
    # Implementation uses Synthesizer in plan mode
    pass


async def execute_step(orchestrator, step: Dict) -> Dict:
    """Execute a single reasoning step."""
    result = await orchestrator.weave(Query(text=step['action']))
    return {
        'step': step['id'],
        'action': step['action'],
        'result': result.response,
        'confidence': result.confidence
    }


async def verify_result(result: Dict) -> Dict:
    """Verify step result for accuracy."""
    # Implementation uses Synthesizer in verify mode
    result['verified'] = True  # After verification
    return result


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate all step results."""
    return {
        'results': results,
        'total_confidence': sum(r['confidence'] for r in results) / len(results)
    }


async def synthesize_response(aggregated: Dict, question: str) -> Dict:
    """Create final synthesized response."""
    # Implementation uses Synthesizer in synthesize mode
    pass


async def main():
    """Main entry point."""
    result = await run_agentic_pipeline(
        question="Compare Thompson Sampling with UCB",
        max_steps=5
    )

    print(f"Answer: {result['answer'][:200]}...")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Steps executed: {result['metadata']['total_steps']}")
    print(f"Steps verified: {result['metadata']['verified_steps']}")


if __name__ == '__main__':
    asyncio.run(main())
```

## Advanced: Parallel Step Execution

For independent steps, use parallel execution:

1. Modify **Plan Generator** to mark independent steps
2. Replace **Loop Iterator** with **Parallel Executor**
3. Configure parallel branches:

```javascript
{
  "label": "Parallel Step Executor",
  "branches": "${plan.parallel_groups}",
  "max_concurrent": 3,
  "timeout_per_branch_ms": 30000,
  "merge_strategy": "collect_all"
}
```

```
Parallel Execution Flow:
                    ┌─── Step 1 ───┐
Input ─▶ Planner ──┼─── Step 2 ───┼──▶ Merge ─▶ Synthesize
                    └─── Step 3 ───┘
```

## Advanced: Reasoning Modes

Configure different reasoning strategies:

### Direct Mode (Fast)
```javascript
{
  "reasoning_mode": "direct",
  "max_steps": 1,
  "verification": false
}
```

### Verify Mode (Accurate)
```javascript
{
  "reasoning_mode": "verify",
  "max_steps": 3,
  "verification": true,
  "verification_threshold": 0.8
}
```

### Research Mode (Comprehensive)
```javascript
{
  "reasoning_mode": "research",
  "max_steps": 10,
  "verification": true,
  "exploration_breadth": "wide"
}
```

### Plan-Execute Mode (Structured)
```javascript
{
  "reasoning_mode": "plan_execute",
  "max_steps": 5,
  "verification": true,
  "plan_refinement": true
}
```

## Debugging Tips

### Loop Not Iterating
- Check collection expression syntax: `${plan.steps}`
- Verify plan generator outputs an array
- Check max iterations limit

### Verification Failing
- Lower `min_confidence` threshold temporarily
- Check verification mode is set correctly
- Review verification checks configuration

### Slow Execution
- Reduce `max_steps` for testing
- Enable parallel execution for independent steps
- Use `fast` mode instead of `fused` for HoloLoom nodes
- Add timeouts to prevent runaway execution

### Inconsistent Results
- Enable `include_reasoning_trace` to debug
- Check aggregation strategy matches your needs
- Verify deduplication settings

## Summary

You've built a production-ready agentic workflow that:
- ✅ Analyzes complex questions
- ✅ Creates structured execution plans
- ✅ Executes steps with optional verification
- ✅ Handles errors gracefully
- ✅ Synthesizes comprehensive responses
- ✅ Provides confidence scores and metadata

## Next Steps

- [Integration Tutorial](integration.md) - Connect to HoloLoom backend
- [Custom Agents](../advanced/custom-agents.md) - Build domain-specific agents
- [Performance Optimization](../advanced/performance.md) - Scale to production

---

← [RAG Pipeline](rag-pipeline.md) | [Integration Tutorial](integration.md) →
