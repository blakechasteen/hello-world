# Tutorial: Integration with HoloLoom

Connect your workflows to HoloLoom's memory, learning, and reasoning systems.

## Overview

In this tutorial, you'll learn how to:
1. Connect the Workflow Builder to HoloLoom backend
2. Configure memory persistence
3. Enable learning and adaptation
4. Monitor workflow performance

**Time**: ~25 minutes
**Difficulty**: Advanced
**Prerequisites**: [Agentic Workflow Tutorial](agentic-workflow.md)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Workflow Builder UI                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  Node   │──│  Node   │──│  Node   │──│  Node   │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└────────────────────────┬────────────────────────────────────┘
                         │ WebSocket
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Workflow Executor                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Node Runner  │  │ State Manager│  │ Event Stream │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    HoloLoom Core                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   Memory   │  │  Learning  │  │  Reasoning │            │
│  │   System   │  │   Engine   │  │   Engine   │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ Yarn Graph │  │  Thompson  │  │  Alignment │            │
│  │    (KG)    │  │  Sampling  │  │ Framework  │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## Step 1: Start the Backend Services

### 1a. Start Docker Services

HoloLoom requires Neo4j and Qdrant for persistent storage:

```bash
# From repository root
docker-compose up -d

# Verify services are running
docker-compose ps

# Expected output:
# NAME                STATUS
# mythrl-neo4j-1      Up (healthy)
# mythrl-qdrant-1     Up (healthy)
```

### 1b. Start the Workflow Executor

```bash
cd HoloLoom/web_dashboard
python workflow_executor.py

# Expected output:
# INFO:     Started server process [12345]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8001
```

### 1c. Verify Connection

```bash
# Health check
curl http://localhost:8001/health

# Expected response:
{
  "status": "healthy",
  "memory_backend": "hybrid",
  "services": {
    "neo4j": "connected",
    "qdrant": "connected"
  }
}
```

## Step 2: Configure HoloLoom Integration

### 2a. Backend Configuration

In the Workflow Builder, click **Settings** (⚙️) and configure:

```
Backend Settings:
┌─────────────────────────────────────┐
│ API Endpoint: [http://localhost:8001]│
│ WebSocket:    [ws://localhost:8001/ws]│
│                                      │
│ Memory Backend: [Hybrid ▼]           │
│   ☑ Neo4j (Knowledge Graph)         │
│   ☑ Qdrant (Vector Store)           │
│                                      │
│ Config Mode: [FUSED ▼]               │
│   ○ BARE  (minimal, <50ms)          │
│   ○ FAST  (balanced, <150ms)        │
│   ● FUSED (full features, <300ms)   │
└─────────────────────────────────────┘
```

### 2b. Programmatic Configuration

```python
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.web_dashboard.workflow_executor import WorkflowExecutor

# Create FUSED configuration
config = Config.fused()
config.memory_backend = MemoryBackend.HYBRID

# Initialize executor with HoloLoom integration
executor = WorkflowExecutor(
    config=config,
    enable_learning=True,
    enable_alignment=True
)

await executor.start()
```

## Step 3: Memory Integration

### 3a. Add Memory Store Node

Connect workflows to HoloLoom's persistent memory:

1. From **Memory** palette, drag **Memory Store** node
2. Configure:
   - **Label**: "Store to Memory"
   - **Collection**: `workflow_memories`
   - **Include Metadata**: ✓

```javascript
// Memory Store Configuration
{
  "label": "Store to Memory",
  "collection": "workflow_memories",
  "include_metadata": true,
  "metadata_fields": [
    "workflow_id",
    "node_id",
    "timestamp",
    "confidence"
  ]
}
```

### 3b. Add Context Retriever Node

Retrieve context from HoloLoom's memory system:

1. From **Memory** palette, drag **Context Retriever** node
2. Configure:
   - **Strategy**: `hybrid` (BM25 + semantic)
   - **Top K**: `10`
   - **Include Graph**: ✓

```javascript
// Context Retriever Configuration
{
  "label": "Retrieve Context",
  "strategy": "hybrid",
  "top_k": 10,
  "include_graph_context": true,
  "spreading_activation": {
    "enabled": true,
    "max_hops": 2,
    "decay": 0.7
  }
}
```

### 3c. Memory-Augmented Workflow

```
┌────────────┐    ┌────────────┐    ┌────────────┐
│   Input    │───▶│  Retrieve  │───▶│  Process   │
│            │    │  Context   │    │            │
└────────────┘    └────────────┘    └─────┬──────┘
                                          │
                       ┌──────────────────┘
                       ▼
                 ┌────────────┐    ┌────────────┐
                 │   Store    │◀───│   Output   │
                 │  to Memory │    │            │
                 └────────────┘    └────────────┘
```

## Step 4: Enable Learning

### 4a. Thompson Sampling Integration

Add adaptive exploration to your workflows:

1. From **Decision** palette, drag **Thompson Sampler** node
2. Configure:
   - **Strategy**: `bayesian_blend`
   - **Exploration Rate**: `0.1`

```javascript
// Thompson Sampler Configuration
{
  "label": "Adaptive Decision",
  "strategy": "bayesian_blend",
  "exploration_rate": 0.1,
  "prior_alpha": 1.0,
  "prior_beta": 1.0,
  "learning_rate": 0.05
}
```

### 4b. Feedback Loop

Enable the workflow to learn from outcomes:

```javascript
// Feedback Configuration
{
  "enable_feedback": true,
  "feedback_source": "confidence",
  "update_rule": "thompson",
  "success_threshold": 0.7
}
```

### 4c. Learning Workflow Example

```
┌────────────┐    ┌────────────┐    ┌────────────┐
│   Query    │───▶│  Thompson  │───▶│  Execute   │
│            │    │  Sampler   │    │   Action   │
└────────────┘    └────────────┘    └─────┬──────┘
                        ▲                  │
                        │                  ▼
                  ┌─────┴──────┐    ┌────────────┐
                  │   Update   │◀───│  Evaluate  │
                  │   Priors   │    │  Outcome   │
                  └────────────┘    └────────────┘
```

## Step 5: Alignment Framework Integration

### 5a. Add Safety Guardrails

1. From **Decision** palette, drag **Safety Guardrails** node
2. Position before any action nodes
3. Configure:

```javascript
// Safety Guardrails Configuration
{
  "label": "Safety Check",
  "risk_threshold": "medium",
  "enable_human_in_loop": true,
  "blocked_actions": [
    "delete_all",
    "external_api_write"
  ],
  "audit_logging": true
}
```

### 5b. Risk-Based Routing

```
                              ┌────────────┐
                        low ──▶│  Execute   │
                              └────────────┘
┌────────────┐    ┌────────────┐
│   Query    │───▶│   Safety   │
│            │    │   Check    │
└────────────┘    └─────┬──────┘
                        │
                        │      ┌────────────┐
                      high ───▶│   Human    │
                              │  Approval  │
                              └────────────┘
```

### 5c. Audit Trail Integration

```javascript
// Audit Trail Configuration
{
  "enable_audit": true,
  "log_level": "detailed",
  "include_inputs": true,
  "include_outputs": true,
  "retention_days": 90
}
```

## Step 6: Real-Time Monitoring

### 6a. Connect Performance Dashboard

1. Click **Dashboard** (📊) in toolbar
2. Enable real-time metrics:

```
Dashboard Configuration:
┌─────────────────────────────────────┐
│ ☑ Execution Latency                 │
│ ☑ Node Throughput                   │
│ ☑ Memory Usage                      │
│ ☑ Cache Hit Rate                    │
│ ☑ Learning Progress                 │
│                                      │
│ Refresh Rate: [1 second ▼]          │
│ History: [1 hour ▼]                 │
└─────────────────────────────────────┘
```

### 6b. WebSocket Event Streaming

```javascript
// Subscribe to execution events
const ws = new WebSocket('ws://localhost:8001/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['execution', 'learning', 'memory']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case 'node_start':
      updateNodeStatus(data.node_id, 'running');
      break;
    case 'node_complete':
      updateNodeStatus(data.node_id, 'complete');
      updateMetrics(data.metrics);
      break;
    case 'learning_update':
      updateLearningPanel(data.priors);
      break;
    case 'memory_store':
      updateMemoryPanel(data.memory_id);
      break;
  }
};
```

### 6c. Prometheus Metrics

Export metrics for external monitoring:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'workflow-executor'
    static_configs:
      - targets: ['localhost:8001']
    metrics_path: '/metrics'
```

**Available Metrics**:
```
workflow_executions_total{workflow_id, status}
workflow_execution_duration_ms{workflow_id, quantile}
node_executions_total{node_type, status}
memory_operations_total{operation, collection}
learning_updates_total{strategy}
cache_hit_rate
```

## Step 7: Complete Integration Example

### Full Workflow with All Integrations

```python
"""
Workflow: Complete HoloLoom Integration
Description: RAG + Learning + Safety + Monitoring
"""

import asyncio
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.alignment import SafetyGuardrails, AuditTrail
from HoloLoom.recursive import FullLearningEngine


async def run_integrated_workflow(query: str) -> dict:
    """Execute workflow with full HoloLoom integration."""

    # Configuration
    config = Config.fused()

    # Initialize components
    guardrails = SafetyGuardrails(enable_human_in_loop=True)
    audit_trail = AuditTrail()

    async with FullLearningEngine(
        cfg=config,
        enable_background_learning=True
    ) as engine:

        # Step 1: Safety check
        safety_result = await guardrails.evaluate(query)

        if safety_result.risk_level == 'high':
            await audit_trail.log('high_risk_blocked', query=query)
            return {'status': 'blocked', 'reason': safety_result.reason}

        # Step 2: Retrieve context from memory
        context = await engine.memory.recall(query, k=10)

        # Step 3: Execute with learning
        spacetime = await engine.weave(
            query,
            context=context,
            enable_refinement=True
        )

        # Step 4: Store result to memory
        await engine.memory.store(
            content=spacetime.response,
            metadata={
                'query': query,
                'confidence': spacetime.confidence,
                'timestamp': spacetime.timestamp
            }
        )

        # Step 5: Log to audit trail
        await audit_trail.log(
            'workflow_complete',
            query=query,
            confidence=spacetime.confidence,
            tool_used=spacetime.metadata.get('tool_used')
        )

        return {
            'response': spacetime.response,
            'confidence': spacetime.confidence,
            'sources': [s.id for s in context],
            'learning_stats': engine.get_learning_statistics()
        }


async def main():
    result = await run_integrated_workflow(
        "Compare Thompson Sampling with UCB for exploration"
    )

    print(f"Response: {result['response'][:200]}...")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Sources: {len(result['sources'])} memories")
    print(f"Learning: {result['learning_stats']['total_updates']} updates")


if __name__ == '__main__':
    asyncio.run(main())
```

## Step 8: Export Integrated Workflow

Export your workflow for deployment:

1. Click **Export** → **Python (with HoloLoom)**
2. Select options:
   - ☑ Include HoloLoom imports
   - ☑ Include memory integration
   - ☑ Include learning hooks
   - ☑ Include safety checks
   - ☑ Include monitoring

The exported code includes all HoloLoom integrations.

## Debugging Integration Issues

### Connection Problems

```bash
# Check backend health
curl http://localhost:8001/health

# Check Docker services
docker-compose ps
docker-compose logs neo4j
docker-compose logs qdrant
```

### Memory Not Persisting

1. Verify HYBRID backend is configured
2. Check Neo4j connection:
   ```bash
   curl http://localhost:7474
   ```
3. Check Qdrant connection:
   ```bash
   curl http://localhost:6333/health
   ```

### Learning Not Updating

1. Verify `enable_learning: true` in config
2. Check feedback threshold (default: 0.7)
3. Review learning logs:
   ```bash
   curl http://localhost:8001/api/learning/stats
   ```

### Safety Blocking Too Much

1. Adjust `risk_threshold` (low/medium/high)
2. Review blocked patterns in config
3. Check audit logs for patterns:
   ```bash
   curl http://localhost:8001/api/audit-trail?limit=10
   ```

## Summary

You've learned how to:
- ✅ Connect Workflow Builder to HoloLoom backend
- ✅ Configure persistent memory storage
- ✅ Enable Thompson Sampling learning
- ✅ Integrate safety guardrails
- ✅ Set up real-time monitoring
- ✅ Export fully integrated workflows

## Next Steps

- [Custom Agents](../advanced/custom-agents.md) - Create domain-specific agents
- [Performance Optimization](../advanced/performance.md) - Scale to production
- [API Reference](../advanced/api-reference.md) - Complete API documentation

---

← [Agentic Workflow](agentic-workflow.md) | [Advanced: Custom Agents](../advanced/custom-agents.md) →
