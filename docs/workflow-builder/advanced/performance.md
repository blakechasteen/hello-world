# Performance Optimization

Optimize workflow execution for large-scale deployments with 100+ nodes.

## Overview

The Workflow Builder is designed to handle complex workflows efficiently. This guide covers:
- Canvas rendering optimization
- Execution performance tuning
- Memory management
- Scaling strategies

## Canvas Performance

### Virtual Scrolling

For workflows with 100+ nodes, virtual scrolling renders only visible nodes:

```javascript
// Automatically enabled for large workflows
workflowBuilder.setOption('virtualScrolling', true);
workflowBuilder.setOption('virtualScrollingThreshold', 50); // Enable at 50+ nodes
```

**How it works**:
- Only nodes in viewport (+ buffer) are rendered
- Off-screen nodes removed from DOM
- Re-rendered when scrolled into view
- 60 FPS maintained regardless of total node count

### Connection Culling

Connections to off-screen nodes are culled:

```javascript
workflowBuilder.setOption('connectionCulling', true);
workflowBuilder.setOption('cullingPadding', 200); // 200px buffer
```

### Lazy Property Loading

Node configurations loaded on-demand:

```javascript
// Config loaded only when node selected
workflowBuilder.setOption('lazyPropertyLoading', true);

// Threshold for lazy loading (bytes)
workflowBuilder.setOption('lazyLoadThreshold', 10000);
```

## Rendering Optimizations

### Frame Rate Monitoring

```javascript
// Enable performance overlay
workflowBuilder.enablePerformanceMonitor();

// Monitor frame time
workflowBuilder.on('frame', (stats) => {
  if (stats.frameTime > 16) { // Below 60 FPS
    console.warn(`Slow frame: ${stats.frameTime.toFixed(1)}ms`);
  }
});
```

### Batch DOM Updates

Group DOM updates to minimize reflows:

```javascript
// Bad: Multiple individual updates
nodes.forEach(n => updateNodePosition(n));

// Good: Batch updates
workflowBuilder.batchUpdate(() => {
  nodes.forEach(n => updateNodePosition(n));
});
```

### Debounced Rendering

Prevent excessive re-renders during rapid changes:

```javascript
workflowBuilder.setOption('renderDebounce', 16); // 60 FPS cap
workflowBuilder.setOption('layoutDebounce', 100); // Layout recalc delay
```

## Execution Performance

### Parallel Execution

Independent nodes execute concurrently:

```yaml
# Workflow with parallel branches
nodes:
  - id: start
    type: input_node
  - id: branch_a
    type: hololoom_query
  - id: branch_b
    type: hololoom_query
  - id: branch_c
    type: hololoom_query
  - id: merge
    type: synthesizer

connections:
  - source: start
    target: branch_a
  - source: start
    target: branch_b
  - source: start
    target: branch_c
  - source: branch_a
    target: merge
  - source: branch_b
    target: merge
  - source: branch_c
    target: merge
```

**Execution timeline**:
```
start ─────┬─── branch_a ───┐
           ├─── branch_b ───┼─── merge
           └─── branch_c ───┘

Time:    0ms     150ms      200ms
         │       │          │
         └───────┴──────────┴─ Total: 200ms (not 450ms)
```

### Execution Caching

Cache node results for repeated inputs:

```javascript
// Enable execution caching
workflowBuilder.setOption('executionCache', true);
workflowBuilder.setOption('cacheMaxSize', 1000);
workflowBuilder.setOption('cacheTTL', 3600); // 1 hour
```

Backend configuration:
```python
# workflow_executor.py
executor = WorkflowExecutor(
    enable_cache=True,
    cache_backend='redis',  # or 'memory'
    cache_ttl=3600
)
```

### Streaming Execution

For long-running workflows, stream results:

```javascript
// Enable streaming
const stream = workflowBuilder.executeStreaming(workflow, input);

stream.on('node_start', (nodeId) => {
  console.log(`Starting: ${nodeId}`);
});

stream.on('node_complete', (nodeId, result) => {
  console.log(`Complete: ${nodeId}`, result);
});

stream.on('workflow_complete', (finalResult) => {
  console.log('Done!', finalResult);
});
```

## Memory Management

### Node Cleanup

Remove unused nodes and connections:

```javascript
// Find orphaned nodes (no connections)
const orphans = workflowBuilder.findOrphanedNodes();

// Remove unused nodes
workflowBuilder.removeNodes(orphans);

// Compact internal data structures
workflowBuilder.compact();
```

### Large Data Handling

For nodes processing large data:

```javascript
// Use streaming for large inputs
{
  type: 'data_processor',
  config: {
    streaming: true,
    chunkSize: 1000,  // Process 1000 items at a time
    maxMemoryMB: 512  // Memory limit
  }
}
```

### Garbage Collection Hints

```javascript
// Trigger cleanup after large operations
workflowBuilder.on('workflow_complete', () => {
  workflowBuilder.releaseTemporaryResources();
});
```

## Backend Optimization

### Worker Pool

Configure worker pool for parallel execution:

```python
# workflow_executor.py
from concurrent.futures import ProcessPoolExecutor

executor = WorkflowExecutor(
    worker_pool=ProcessPoolExecutor(max_workers=8),
    max_concurrent_nodes=16
)
```

### Connection Pooling

Reuse database and HTTP connections:

```python
# Database connection pool
import asyncpg

pool = await asyncpg.create_pool(
    database_url,
    min_size=5,
    max_size=20
)

executor = WorkflowExecutor(
    resources={'db_pool': pool}
)
```

### Request Batching

Batch multiple node requests:

```python
class BatchingExecutor:
    """Batch multiple node executions together."""

    def __init__(self, batch_size=10, batch_timeout=0.1):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending = []

    async def execute_batched(self, nodes):
        # Collect nodes for batching
        self.pending.extend(nodes)

        # Execute when batch full or timeout
        if len(self.pending) >= self.batch_size:
            return await self._flush_batch()
```

## Scaling Strategies

### Horizontal Scaling

Distribute workflow execution across multiple servers:

```yaml
# docker-compose.yml
services:
  workflow-executor-1:
    image: hololoom/workflow-executor
    environment:
      - WORKER_ID=1
      - REDIS_URL=redis://redis:6379

  workflow-executor-2:
    image: hololoom/workflow-executor
    environment:
      - WORKER_ID=2
      - REDIS_URL=redis://redis:6379

  load-balancer:
    image: nginx
    depends_on:
      - workflow-executor-1
      - workflow-executor-2
```

### Queue-Based Execution

Use message queues for async processing:

```python
# Producer (API server)
async def submit_workflow(workflow, input_data):
    job_id = str(uuid.uuid4())
    await redis.rpush('workflow_queue', json.dumps({
        'job_id': job_id,
        'workflow': workflow,
        'input': input_data
    }))
    return job_id

# Consumer (worker)
async def worker_loop():
    while True:
        job = await redis.blpop('workflow_queue')
        result = await execute_workflow(job)
        await redis.set(f"result:{job['job_id']}", result)
```

### Workflow Sharding

Split large workflows across servers:

```python
def shard_workflow(workflow, num_shards):
    """Split workflow into independent sub-workflows."""
    # Find cut points (nodes with no cross-shard dependencies)
    cut_points = find_cut_points(workflow, num_shards)

    # Create sub-workflows
    shards = []
    for i, (start, end) in enumerate(cut_points):
        shard = extract_subworkflow(workflow, start, end)
        shards.append(shard)

    return shards
```

## Monitoring and Profiling

### Performance Metrics

```javascript
// Get performance statistics
const stats = workflowBuilder.getPerformanceStats();

console.log({
  fps: stats.fps,
  nodeCount: stats.nodeCount,
  connectionCount: stats.connectionCount,
  renderTime: stats.avgRenderTime,
  memoryUsage: stats.memoryUsage
});
```

### Execution Profiling

```python
# Enable profiling
executor = WorkflowExecutor(profiling=True)

result = await executor.execute(workflow, input_data)

# Get timing breakdown
print(result.profile)
# {
#   'total_ms': 1250,
#   'nodes': {
#     'query-1': {'duration_ms': 145, 'cache_hit': False},
#     'process-1': {'duration_ms': 89, 'cache_hit': True},
#     ...
#   },
#   'parallel_efficiency': 0.85
# }
```

### Bottleneck Detection

```javascript
// Find performance bottlenecks
const bottlenecks = workflowBuilder.analyzeBottlenecks(executionResult);

bottlenecks.forEach(b => {
  console.log(`Bottleneck: ${b.nodeId}`);
  console.log(`  Duration: ${b.duration}ms (${b.percentage}% of total)`);
  console.log(`  Suggestion: ${b.suggestion}`);
});
```

## Configuration Reference

### Frontend Options

| Option | Default | Description |
|--------|---------|-------------|
| `virtualScrolling` | `true` | Enable virtual scrolling |
| `virtualScrollingThreshold` | `50` | Node count to enable |
| `connectionCulling` | `true` | Cull off-screen connections |
| `cullingPadding` | `200` | Buffer around viewport (px) |
| `lazyPropertyLoading` | `true` | Load configs on-demand |
| `renderDebounce` | `16` | Render debounce (ms) |
| `maxUndoHistory` | `100` | Undo stack size |

### Backend Options

| Option | Default | Description |
|--------|---------|-------------|
| `max_workers` | `4` | Worker pool size |
| `max_concurrent_nodes` | `10` | Parallel node limit |
| `execution_timeout` | `300` | Timeout (seconds) |
| `cache_enabled` | `true` | Enable result caching |
| `cache_ttl` | `3600` | Cache TTL (seconds) |
| `memory_limit_mb` | `1024` | Memory limit per worker |

## Best Practices

### Workflow Design

1. **Minimize sequential chains**: Use parallel branches where possible
2. **Cache expensive operations**: Enable caching for slow nodes
3. **Use appropriate complexity**: Don't over-engineer simple workflows
4. **Group related nodes**: Use composites to organize complexity

### Resource Management

1. **Set timeouts**: Always configure execution timeouts
2. **Limit concurrency**: Don't overwhelm backend services
3. **Monitor memory**: Watch for memory leaks in long-running workflows
4. **Clean up**: Remove completed workflow data periodically

### Testing at Scale

1. **Load test**: Test with realistic node counts
2. **Profile regularly**: Monitor performance over time
3. **Test edge cases**: Large data, many connections, deep nesting
4. **Simulate failures**: Test recovery and error handling

---

← [Custom Agents](custom-agents.md) | [API Reference](api-reference.md) →
