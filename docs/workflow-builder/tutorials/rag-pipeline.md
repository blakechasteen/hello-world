# Tutorial: Build a RAG Pipeline

Create a complete Retrieval-Augmented Generation (RAG) workflow from scratch.

## Overview

In this tutorial, you'll build a multi-hop RAG pipeline that:
1. Takes a user query
2. Retrieves relevant context from memory
3. Verifies retrieved information
4. Generates a synthesized response

**Time**: ~20 minutes
**Difficulty**: Intermediate
**Prerequisites**: [First Workflow](../getting-started/first-workflow.md)

## What You'll Build

```
┌──────────┐    ┌─────────────┐    ┌───────────┐    ┌──────────┐    ┌────────┐
│  Input   │───▶│ Multi-Query │───▶│  Memory   │───▶│ Verifier │───▶│ Output │
└──────────┘    └─────────────┘    └───────────┘    └──────────┘    └────────┘
                      │                  ▲
                      │                  │
                      └──────────────────┘
                       (parallel queries)
```

## Step 1: Create the Workflow

1. Open the Workflow Builder
2. Click **New Workflow** (or `Ctrl+N`)
3. Name it "RAG Pipeline Tutorial"

## Step 2: Add the Input Node

1. From the **I/O** palette, drag **Input Node** onto the canvas
2. Configure it:
   - **Label**: "User Query"
   - **Input Type**: `string`
   - **Required**: ✓

```
Input Configuration:
┌─────────────────────────────────────┐
│ Label: User Query                   │
│ Type:  [string ▼]                   │
│ ☑ Required                          │
│ Default: (empty)                    │
└─────────────────────────────────────┘
```

## Step 3: Add Query Expansion

The Multi-Query agent breaks complex questions into sub-questions for more comprehensive retrieval.

1. From **Query** palette, drag **Multi-Query** node
2. Position it to the right of Input
3. Connect **Input** → **Multi-Query**
4. Configure:
   - **Label**: "Query Expander"
   - **Max Queries**: `3`
   - **Strategy**: `diverse` (covers different aspects)

```javascript
// Multi-Query Configuration
{
  "label": "Query Expander",
  "max_queries": 3,
  "strategy": "diverse",
  "include_original": true
}
```

## Step 4: Add Memory Retrieval

1. From **Memory** palette, drag **Context Retriever** node
2. Position it to the right of Multi-Query
3. Connect **Multi-Query** → **Context Retriever**
4. Configure:
   - **Label**: "Memory Search"
   - **Top K**: `10`
   - **Strategy**: `hybrid` (BM25 + semantic)
   - **Include Graph**: ✓

```
Context Retriever Configuration:
┌─────────────────────────────────────┐
│ Label: Memory Search                │
│ Top K: [10      ]                   │
│ Strategy: [hybrid ▼]                │
│ ☑ Include Graph Context             │
│ ☑ Enable Spreading Activation       │
│ Max Hops: [2]                       │
└─────────────────────────────────────┘
```

## Step 5: Add Verification

Verification ensures retrieved context is accurate and relevant.

1. From **Processing** palette, drag **Synthesizer** node
2. Position it to the right of Context Retriever
3. Connect **Context Retriever** → **Synthesizer**
4. Configure:
   - **Label**: "Verify Sources"
   - **Mode**: `verify`
   - **Min Confidence**: `0.7`

```javascript
// Verification Configuration
{
  "label": "Verify Sources",
  "mode": "verify",
  "min_confidence": 0.7,
  "check_contradictions": true,
  "require_citations": true
}
```

## Step 6: Add Response Generation

1. From **Output** palette, drag **Response Generator** node
2. Position it to the right of Synthesizer
3. Connect **Synthesizer** → **Response Generator**
4. Configure:
   - **Label**: "Generate Answer"
   - **Format**: `markdown`
   - **Style**: `technical`
   - **Include Sources**: ✓

```
Response Generator Configuration:
┌─────────────────────────────────────┐
│ Label: Generate Answer              │
│ Format: [markdown ▼]                │
│ Style: [technical ▼]               │
│ Max Length: [500    ]               │
│ ☑ Include Sources                   │
│ ☑ Include Confidence                │
└─────────────────────────────────────┘
```

## Step 7: Add Output Node

1. From **I/O** palette, drag **Output Node**
2. Position it at the end
3. Connect **Response Generator** → **Output**
4. Configure:
   - **Label**: "Final Response"
   - **Output Type**: `object`

## Your Workflow Should Look Like This

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ User Query │───▶│   Query    │───▶│  Memory    │───▶│  Verify    │───▶│  Generate  │───▶│   Final    │
│  (Input)   │    │  Expander  │    │  Search    │    │  Sources   │    │   Answer   │    │  Response  │
└────────────┘    └────────────┘    └────────────┘    └────────────┘    └────────────┘    └────────────┘
```

## Step 8: Test the Pipeline

1. Click **Execute** (▶️) or press `Ctrl+Enter`
2. Enter a test query:
   ```
   What is Thompson Sampling and how does it relate to exploration?
   ```
3. Watch the execution progress through each node
4. Review the output

**Expected Output**:
```json
{
  "response": "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem...",
  "confidence": 0.89,
  "sources": [
    {"id": "mem-001", "relevance": 0.95, "text": "Thompson Sampling balances..."},
    {"id": "mem-002", "relevance": 0.87, "text": "Exploration strategies include..."}
  ],
  "queries_expanded": [
    "What is Thompson Sampling?",
    "How does exploration work in bandits?",
    "What are Bayesian methods in decision making?"
  ]
}
```

## Step 9: Add Conditional Logic (Optional Enhancement)

Add a confidence check to handle low-quality retrievals:

1. From **Control Flow** palette, drag **Conditional Branch**
2. Insert it between **Memory Search** and **Verify Sources**
3. Configure:
   - **Condition**: `${input.confidence} >= 0.5`
   - **True Branch**: Continue to verification
   - **False Branch**: Return "insufficient context" response

```
Updated Flow:
                                    ┌──────────────┐
                              yes ──▶│   Verify     │───▶ Generate
┌────────────┐    ┌──────────┐     │              │
│  Memory    │───▶│ Confident?│     └──────────────┘
│  Search    │    │  >= 0.5   │
└────────────┘    └──────────┘     ┌──────────────┐
                              no ──▶│   Fallback   │───▶ Output
                                    │   Response   │
                                    └──────────────┘
```

## Step 10: Save and Export

1. Click **Save** (`Ctrl+S`)
2. Export as Python:
   - Click **Export** → **Python**
   - Enable "Include docstrings" and "Include main()"
3. Save the file as `rag_pipeline.py`

## Generated Python Code

The exported Python code will look like this:

```python
"""
Workflow: RAG Pipeline Tutorial
Description: Multi-hop RAG with verification
Generated: 2025-12-15
"""

import asyncio
from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.config import Config
from hololoom.protocols.types import Query


async def run_rag_pipeline(query_text: str) -> dict:
    """Execute the RAG pipeline workflow."""

    config = Config.fused()

    async with WeavingOrchestrator(cfg=config) as orchestrator:
        # Step 1: Expand query into sub-questions
        expanded_queries = await expand_query(query_text, max_queries=3)

        # Step 2: Retrieve context for each query
        all_context = []
        for q in expanded_queries:
            context = await orchestrator.recall(q, k=10, strategy='hybrid')
            all_context.extend(context)

        # Step 3: Deduplicate and verify sources
        verified_context = await verify_sources(all_context, min_confidence=0.7)

        # Step 4: Generate response
        result = await orchestrator.weave(
            Query(text=query_text),
            context=verified_context
        )

        return {
            'response': result.response,
            'confidence': result.confidence,
            'sources': verified_context,
            'queries_expanded': expanded_queries
        }


async def main():
    """Main entry point."""
    result = await run_rag_pipeline("What is Thompson Sampling?")
    print(f"Response: {result['response']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Sources: {len(result['sources'])}")


if __name__ == '__main__':
    asyncio.run(main())
```

## Advanced: Multi-Hop Retrieval

For deeper context, add a second retrieval pass:

1. Add another **Context Retriever** node
2. Configure it for graph-based retrieval:
   - **Strategy**: `connected`
   - **Max Hops**: `3`
   - **Min Edge Weight**: `0.5`
3. Connect after the first retrieval
4. Use **Knowledge Fusion** to merge results

```
Multi-Hop Flow:
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Initial  │───▶│  Graph   │───▶│ Knowledge│───▶│  Verify  │
│ Retrieve │    │ Traverse │    │  Fusion  │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

## Debugging Tips

### Low Confidence Results

If confidence is consistently low:
1. Check memory has relevant content (`Memory Store` first)
2. Increase `Top K` in retrieval
3. Enable spreading activation for related concepts
4. Lower verification threshold temporarily to diagnose

### Slow Execution

If the pipeline is slow:
1. Reduce `Max Queries` in Multi-Query
2. Enable caching on retrieval nodes
3. Lower `Max Hops` in graph traversal
4. Use `FAST` mode instead of `FUSED`

### Missing Context

If relevant context isn't being retrieved:
1. Enable hybrid search (BM25 + semantic)
2. Increase spreading activation hops
3. Check query expansion is generating useful sub-questions
4. Verify memory store contains expected content

## Summary

You've built a production-ready RAG pipeline that:
- ✅ Expands queries for comprehensive coverage
- ✅ Retrieves hybrid (keyword + semantic) results
- ✅ Verifies source quality and relevance
- ✅ Generates formatted responses with citations
- ✅ Handles edge cases with conditional logic

## Next Steps

- [Agentic Workflow Tutorial](agentic-workflow.md) - Add multi-step reasoning
- [Integration Tutorial](integration.md) - Connect to HoloLoom backend
- [Performance Optimization](../advanced/performance.md) - Scale to production

---

← [Advanced: API Reference](../advanced/api-reference.md) | [Agentic Workflow](agentic-workflow.md) →
