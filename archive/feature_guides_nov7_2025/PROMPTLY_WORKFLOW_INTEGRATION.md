# Promptly × Workflow Builder Integration Strategy

**Date**: November 3, 2025
**Status**: Architecture Design
**Integration Goal**: Combine prompt management with visual workflow building

---

## Executive Summary

**Question**: "Does the workflow builder work inside Promptly, or should we rearchitect?"

**Answer**: **Both systems complement each other perfectly!** Here's the integration strategy:

---

## Architecture Overview

### Current State

**Promptly** (Archived):
- Prompt version control (Git-like)
- Branch management
- Evaluation framework
- SQLite storage
- CLI interface

**Workflow Builder** (New):
- Visual workflow design
- Multi-agent orchestration
- Real-time execution
- WebSocket updates
- Auto-generation from NL

### Integration Strategy: **Hybrid Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    Promptly Integration Layer                │
│  (Prompt Management + Versioning + Workflow Orchestration)   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴──────────────────────┐
        │                                             │
┌───────▼────────┐                          ┌────────▼────────┐
│   Promptly     │                          │    Workflow     │
│   (Prompts)    │◄────── Synergy ─────────►│    Builder      │
│                │                          │   (Execution)   │
└────────────────┘                          └─────────────────┘
        │                                             │
        │                                             │
    Version                                      Execute
    Branch                                       Monitor
    Evaluate                                     Optimize
```

---

## Integration Scenarios

### Scenario 1: **Workflow-as-Prompt** (Recommended)

**Concept**: Treat workflows as versioned prompts

```python
# Store workflow in Promptly
promptly.save_workflow(
    name="research_pipeline",
    workflow=workflow_json,
    branch="main",
    version=1,
    metadata={
        "nodes": 6,
        "type": "research",
        "performance": {"avg_latency": 1.2}
    }
)

# Branch for experimentation
promptly.branch("research_pipeline", "experiment-parallel")

# Modify workflow in visual builder
workflow_v2 = workflow_builder.edit(workflow_json)

# Commit new version
promptly.commit(
    name="research_pipeline",
    workflow=workflow_v2,
    message="Add parallel execution for sub-queries"
)

# Compare versions
promptly.diff("research_pipeline", version1=1, version2=2)
# Shows: Added parallel executor, changed config
```

**Benefits**:
- Version control for workflows
- A/B testing (branch comparison)
- Rollback to previous workflows
- Audit trail for changes

### Scenario 2: **Prompt-Enhanced Workflows**

**Concept**: Workflows execute versioned prompts

```python
# Workflow node references Promptly prompt
workflow = {
    "nodes": [
        {
            "id": "node-1",
            "agentType": "hololoom",
            "config": {
                "prompt_source": "promptly://research_prompt@v3",
                "pattern": "fused"
            }
        }
    ]
}

# Workflow executor fetches prompt from Promptly
prompt = promptly.get("research_prompt", version=3)
result = await hololoom.query(prompt.content, pattern="fused")
```

**Benefits**:
- Decouple prompts from workflows
- Reuse prompts across workflows
- Update prompts without changing workflows
- Centralized prompt management

### Scenario 3: **Evaluation-Driven Workflow Optimization**

**Concept**: Use Promptly's eval framework to optimize workflows

```python
# Define test cases
test_cases = [
    {"query": "What is RL?", "expected_confidence": 0.8},
    {"query": "Explain MCTS", "expected_confidence": 0.85}
]

# Run evaluation
promptly.evaluate_workflow(
    workflow="research_pipeline",
    test_cases=test_cases,
    metrics=["latency", "confidence", "quality"]
)

# Results stored in Promptly DB
results = promptly.get_eval_results("research_pipeline")
# Shows: 90% confidence average, 1.2s latency

# Auto-optimize workflow based on eval
optimized = workflow_optimizer.optimize(
    workflow=workflow,
    eval_results=results,
    target="minimize_latency"
)
```

**Benefits**:
- Systematic workflow testing
- Performance tracking
- Data-driven optimization
- Regression detection

---

## Recommended Architecture: **Unified System**

### New Integrated System: "HoloLoom Studio"

```
HoloLoom Studio
├── Prompt Management (Promptly Core)
│   ├── Version control
│   ├── Branch management
│   ├── Evaluation framework
│   └── SQLite storage
│
├── Workflow Builder (New)
│   ├── Visual designer
│   ├── Auto-generation
│   ├── Real-time execution
│   └── WebSocket monitoring
│
└── Integration Layer (New)
    ├── Workflow versioning
    ├── Prompt → Workflow binding
    ├── Evaluation automation
    └── A/B testing
```

### Key Components

#### 1. **Unified Storage**

Extend Promptly's SQLite schema:

```sql
-- Add workflows table
CREATE TABLE workflows (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    workflow_json TEXT NOT NULL,  -- Workflow definition
    branch TEXT NOT NULL DEFAULT 'main',
    version INTEGER NOT NULL,
    parent_id INTEGER,
    commit_hash TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT,  -- nodes, connections, performance
    FOREIGN KEY (parent_id) REFERENCES workflows(id)
);

-- Link prompts to workflows
CREATE TABLE workflow_prompts (
    workflow_id INTEGER NOT NULL,
    prompt_id INTEGER NOT NULL,
    node_id TEXT NOT NULL,  -- Which node uses this prompt
    FOREIGN KEY (workflow_id) REFERENCES workflows(id),
    FOREIGN KEY (prompt_id) REFERENCES prompts(id)
);

-- Workflow evaluation results
CREATE TABLE workflow_evaluations (
    id INTEGER PRIMARY KEY,
    workflow_name TEXT NOT NULL,
    commit_hash TEXT NOT NULL,
    test_case TEXT NOT NULL,
    metrics TEXT,  -- JSON: {latency, confidence, quality}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (commit_hash) REFERENCES workflows(commit_hash)
);
```

#### 2. **Unified CLI**

Extend Promptly CLI:

```bash
# Create workflow
promptly workflow create research_pipeline --from-description "Research with safety"

# Edit visually
promptly workflow edit research_pipeline
# Opens browser with visual builder

# Execute workflow
promptly workflow run research_pipeline --input "What is Thompson Sampling?"

# Version management
promptly workflow commit research_pipeline -m "Add parallel execution"
promptly workflow branch research_pipeline experiment
promptly workflow diff research_pipeline main..experiment

# Evaluation
promptly workflow eval research_pipeline --test-cases test.json
promptly workflow optimize research_pipeline --target latency
```

#### 3. **Unified Web UI**

Enhanced workflow builder:

```
┌──────────────────────────────────────────────────────────┐
│  HoloLoom Studio                        [Version: main]  │
├──────────────────────────────────────────────────────────┤
│  [Workflows] [Prompts] [Evaluations] [Analytics]         │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐      ┌──────────────────────────┐     │
│  │  Templates  │      │    Canvas                 │     │
│  │             │      │                           │     │
│  │ • Research  │      │  [Node] → [Node] → [Node] │     │
│  │ • Safety    │      │                           │     │
│  │ • Memory    │      └──────────────────────────┘     │
│  │             │                                        │
│  │  History    │      ┌──────────────────────────┐     │
│  │ • v1.0 (✓)  │      │  Properties               │     │
│  │ • v1.1      │      │                           │     │
│  │ • v1.2 (*)  │      │  [Agent Config]           │     │
│  └─────────────┘      └──────────────────────────┘     │
│                                                          │
│  [🪄 Generate] [💾 Save] [🔀 Branch] [📊 Evaluate]     │
└──────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Core Integration (Week 1)

**Goal**: Basic workflow versioning in Promptly

```python
# File: HoloLoom/promptly/workflow_store.py

class WorkflowStore(PromptlyDB):
    """Extends Promptly with workflow storage."""

    def save_workflow(self, name: str, workflow: Dict, branch: str = "main"):
        """Save workflow with versioning."""
        commit_hash = self._hash_workflow(workflow)
        version = self._get_next_version(name, branch)

        self.conn.execute("""
            INSERT INTO workflows (name, workflow_json, branch, version, commit_hash, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, json.dumps(workflow), branch, version, commit_hash, json.dumps(workflow.get('metadata', {}))))

    def get_workflow(self, name: str, version: Optional[int] = None, branch: str = "main"):
        """Retrieve workflow by name/version."""
        if version:
            result = self.conn.execute("""
                SELECT * FROM workflows WHERE name=? AND version=? AND branch=?
            """, (name, version, branch))
        else:
            result = self.conn.execute("""
                SELECT * FROM workflows WHERE name=? AND branch=?
                ORDER BY version DESC LIMIT 1
            """, (name, branch))

        row = result.fetchone()
        return json.loads(row['workflow_json']) if row else None

    def diff_workflows(self, name: str, version1: int, version2: int):
        """Compare two workflow versions."""
        w1 = self.get_workflow(name, version1)
        w2 = self.get_workflow(name, version2)

        diff = {
            'nodes_added': [],
            'nodes_removed': [],
            'nodes_modified': [],
            'connections_changed': []
        }

        # Compute diff
        nodes1 = {n['id']: n for n in w1['nodes']}
        nodes2 = {n['id']: n for n in w2['nodes']}

        for node_id in nodes2:
            if node_id not in nodes1:
                diff['nodes_added'].append(nodes2[node_id])
            elif nodes1[node_id] != nodes2[node_id]:
                diff['nodes_modified'].append({
                    'id': node_id,
                    'before': nodes1[node_id],
                    'after': nodes2[node_id]
                })

        for node_id in nodes1:
            if node_id not in nodes2:
                diff['nodes_removed'].append(nodes1[node_id])

        return diff
```

### Phase 2: Visual Integration (Week 2)

**Goal**: Add version control UI to workflow builder

```javascript
// File: workflow_builder.js

class WorkflowVersionControl {
    constructor() {
        this.currentBranch = 'main';
        this.currentVersion = null;
    }

    async saveVersion(workflow, message) {
        const response = await fetch('/api/workflow/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                name: workflow.name,
                workflow: workflow,
                branch: this.currentBranch,
                message: message
            })
        });

        const result = await response.json();
        this.currentVersion = result.version;

        showToast(`Saved as v${result.version}`, 'success');
        this.updateVersionUI();
    }

    async createBranch(branchName) {
        await fetch('/api/workflow/branch', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                name: workflow.name,
                fromBranch: this.currentBranch,
                toBranch: branchName
            })
        });

        this.currentBranch = branchName;
        showToast(`Created branch: ${branchName}`, 'success');
    }

    async loadVersion(version) {
        const workflow = await fetch(`/api/workflow/get?name=${workflow.name}&version=${version}`)
            .then(r => r.json());

        loadWorkflow(workflow);
        this.currentVersion = version;
        showToast(`Loaded v${version}`, 'success');
    }

    async diff(version1, version2) {
        const diff = await fetch(`/api/workflow/diff?name=${workflow.name}&v1=${version1}&v2=${version2}`)
            .then(r => r.json());

        showDiffModal(diff);
    }
}
```

### Phase 3: Evaluation Framework (Week 3)

**Goal**: Automated workflow testing

```python
# File: HoloLoom/promptly/workflow_evaluator.py

class WorkflowEvaluator:
    """Evaluate workflow performance."""

    def __init__(self, workflow_store: WorkflowStore):
        self.store = workflow_store

    async def evaluate(
        self,
        workflow_name: str,
        test_cases: List[Dict],
        metrics: List[str] = ["latency", "confidence", "quality"]
    ) -> Dict[str, Any]:
        """
        Run test cases against workflow.

        Args:
            workflow_name: Name of workflow to evaluate
            test_cases: List of {input, expected_output} dicts
            metrics: Metrics to measure

        Returns:
            Evaluation results with scores
        """
        workflow = self.store.get_workflow(workflow_name)
        executor = WorkflowExecutor(workflow)

        results = []
        for test_case in test_cases:
            start = time.time()

            # Execute workflow
            output = await executor.execute(test_case['input'])

            # Measure metrics
            latency = time.time() - start
            confidence = output.get('confidence', 0.0)
            quality = self._score_quality(output, test_case.get('expected_output'))

            results.append({
                'test_case': test_case,
                'output': output,
                'metrics': {
                    'latency': latency,
                    'confidence': confidence,
                    'quality': quality
                }
            })

        # Aggregate results
        avg_metrics = {
            'latency': sum(r['metrics']['latency'] for r in results) / len(results),
            'confidence': sum(r['metrics']['confidence'] for r in results) / len(results),
            'quality': sum(r['metrics']['quality'] for r in results) / len(results)
        }

        # Store in database
        self._store_eval_results(workflow_name, results, avg_metrics)

        return {
            'workflow': workflow_name,
            'test_cases': len(test_cases),
            'results': results,
            'averages': avg_metrics,
            'timestamp': datetime.now().isoformat()
        }

    def _score_quality(self, output, expected):
        """Score output quality (0-1)."""
        # Compare output to expected (can use LLM for semantic similarity)
        if not expected:
            return 1.0  # No expected output to compare

        # Simple word overlap for now
        output_words = set(str(output).lower().split())
        expected_words = set(str(expected).lower().split())
        overlap = len(output_words & expected_words)
        union = len(output_words | expected_words)

        return overlap / union if union > 0 else 0.0
```

### Phase 4: Auto-Optimization (Week 4)

**Goal**: Use eval results to improve workflows

```python
# File: HoloLoom/promptly/workflow_optimizer.py

class WorkflowOptimizer:
    """Optimize workflows based on evaluation results."""

    def optimize(
        self,
        workflow: Dict,
        eval_results: Dict,
        target: str = "balance"  # "latency", "confidence", "quality", "balance"
    ) -> Dict:
        """
        Optimize workflow based on evaluation results.

        Strategies:
        - latency: Remove unnecessary nodes, increase parallelism
        - confidence: Add refinement, increase model quality
        - quality: Add synthesis, multiple perspectives
        - balance: Optimize for all three
        """
        optimizations = []

        if target in ["latency", "balance"]:
            # Detect sequential nodes that can be parallelized
            parallel_candidates = self._find_parallel_opportunities(workflow)
            if parallel_candidates:
                optimizations.append({
                    'type': 'add_parallel_executor',
                    'nodes': parallel_candidates,
                    'expected_speedup': '2-3x'
                })

            # Remove redundant nodes
            redundant = self._find_redundant_nodes(workflow, eval_results)
            if redundant:
                optimizations.append({
                    'type': 'remove_redundant',
                    'nodes': redundant,
                    'expected_speedup': '1.2x'
                })

        if target in ["confidence", "balance"]:
            # Add conditional refinement for low confidence
            if eval_results['averages']['confidence'] < 0.8:
                optimizations.append({
                    'type': 'add_conditional_refiner',
                    'threshold': 0.75,
                    'expected_improvement': '+10% confidence'
                })

        if target in ["quality", "balance"]:
            # Add synthesis for better quality
            if eval_results['averages']['quality'] < 0.8:
                optimizations.append({
                    'type': 'add_synthesizer',
                    'expected_improvement': '+15% quality'
                })

        # Apply optimizations
        optimized_workflow = self._apply_optimizations(workflow, optimizations)

        return {
            'original_workflow': workflow,
            'optimized_workflow': optimized_workflow,
            'optimizations': optimizations,
            'target': target
        }
```

---

## Benefits of Integration

### 1. **Version Control for Workflows**
- Git-like workflow versioning
- Branch for experiments
- Rollback to previous versions
- Audit trail

### 2. **Systematic Testing**
- Automated evaluation
- Performance tracking
- Regression detection
- A/B testing

### 3. **Data-Driven Optimization**
- Optimize based on metrics
- Automatic workflow improvements
- Learn from production data

### 4. **Reusable Components**
- Prompts as first-class objects
- Workflows reference prompts
- Update prompts without changing workflows

### 5. **Collaboration**
- Team members work on branches
- Review workflow changes
- Share optimized workflows

---

## Migration Path

### From Archived Promptly → Integrated System

**Step 1**: Extract Promptly core (Week 1)
```bash
# Move Promptly from archive to active
cp -r archive/old_projects/Promptly HoloLoom/promptly
```

**Step 2**: Add workflow tables (Week 1)
```python
# Extend database schema
python -m HoloLoom.promptly.migrate_add_workflows
```

**Step 3**: Integrate with workflow builder (Week 2)
```python
# Add API endpoints
# Update UI with version control
```

**Step 4**: Add evaluation (Week 3)
```python
# Implement WorkflowEvaluator
# Add CLI commands
```

**Step 5**: Add optimization (Week 4)
```python
# Implement WorkflowOptimizer
# Add auto-optimization UI
```

---

## Conclusion

### Recommendation: **Full Integration**

**Don't rearchitect - ENHANCE both systems through integration!**

**Why?**
1. ✅ Promptly's versioning + Workflow Builder's visual design = Perfect combo
2. ✅ Evaluation framework enables data-driven optimization
3. ✅ No architectural conflicts - clean integration points
4. ✅ Minimal code duplication (~200 lines integration layer)
5. ✅ Huge value: Version control + Visual design + Auto-optimization

**Result**: **"HoloLoom Studio"** - The ultimate workflow management system

---

## Next Steps

**Immediate** (Can start now):
1. ✅ Use workflow builder standalone (already works)
2. ✅ Save workflows as JSON files
3. ✅ Use git for version control (manual)

**Phase 1** (1 week):
1. Resurrect Promptly from archive
2. Add workflows table to SQLite
3. Basic save/load/version

**Phase 2** (2-3 weeks):
1. Integrate version control into UI
2. Add evaluation framework
3. Auto-optimization

**Phase 3** (4 weeks):
1. ML-based intent detection
2. Reinforcement learning optimization
3. Team collaboration features

---

**Status**: Architecture defined ✅
**Feasibility**: High - clean integration points
**Value**: Massive - version control + visual + optimization
**Recommendation**: **INTEGRATE, don't rearchitect**

