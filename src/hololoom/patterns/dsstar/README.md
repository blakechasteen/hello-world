# DS-STAR Pattern

**Data Science with Self-Taught Autonomous Reasoning**

Based on Google Research paper, fully integrated with HoloLoom's architecture.

## Overview

DS-STAR is an agentic data science system that iteratively plans, executes, and verifies data analysis tasks. It combines:

- **Data Analysis** (anchors extraction)
- **Planning** (execution step generation)
- **Safe Execution** (sandboxed code running)
- **Verification** (goal satisfaction checking)
- **Iterative Refinement** (self-improvement loop)

## Quick Start

```python
from hololoom.patterns.dsstar import DSStarOrchestrator

# Create orchestrator
orchestrator = DSStarOrchestrator(
    max_iterations=3,
    verification_threshold=0.7,
    enable_refinement=True
)

# Process data analysis query
result = await orchestrator.process(
    query="What is the average sales by region?",
    data_file="sales_data.csv"
)

# Check results
print(result.summary())
if result.success:
    print("Final outputs:", result.final_outputs.keys())
    print(f"Confidence: {result.final_verification.confidence:.1%}")
else:
    print("Issues:", result.final_verification.issues)
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  DS-STAR Loop                            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. ANALYZE                                              │
│     ├─ Extract data anchors (schema, types, samples)    │
│     ├─ Profile dataset (rows, columns, relationships)   │
│     └─ Build context for planning                       │
│                                                          │
│  2. PLAN                                                 │
│     ├─ Parse query intent                               │
│     ├─ Generate execution steps                         │
│     └─ Create code templates                            │
│                                                          │
│  3. EXECUTE                                              │
│     ├─ Safety checks (no dangerous operations)          │
│     ├─ Sandbox execution (isolated context)             │
│     └─ Capture outputs and errors                       │
│                                                          │
│  4. VERIFY                                               │
│     ├─ Check expected outputs present                   │
│     ├─ Validate output types                            │
│     ├─ Assess query intent satisfaction                 │
│     └─ Calculate confidence score                       │
│                                                          │
│  5. ITERATE (if confidence < threshold)                 │
│     ├─ Extract improvement suggestions                  │
│     ├─ Refine plan based on issues                      │
│     └─ Retry with refined approach                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Components

### 1. DataAnalyzer (`analyzer.py`)

Extracts "anchors" (key information) from data files.

```python
from hololoom.patterns.dsstar import DataAnalyzer

analyzer = DataAnalyzer()
profile = analyzer.analyze_file("data.csv")

print(f"Rows: {profile.row_count}")
print(f"Columns: {profile.column_count}")

for anchor in profile.anchors:
    print(f"{anchor.name} ({anchor.type}): {anchor.unique_count} unique values")
```

**Features**:
- Automatic format detection (CSV, Excel, JSON, Parquet)
- Type inference (numeric, categorical, text, datetime)
- Statistical profiling (min, max, mean, mode, etc.)
- Relationship detection (correlations)
- Sample extraction

**Supported Formats**:
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- JSON (`.json`)
- Parquet (`.parquet`)

### 2. Planner (`planner.py`)

Creates execution plans from natural language queries.

```python
from hololoom.patterns.dsstar import Planner, create_plan

planner = Planner()
plan = planner.create_plan(
    query="Show average temperature by month",
    data_profile=profile
)

print(planner.visualize_plan(plan))

for step in plan.steps:
    print(f"{step.step_id}. {step.action.value}: {step.description}")
    print(f"   Code: {step.code_template}")
```

**Plan Operations**:
- `LOAD_DATA` - Load data from file
- `FILTER` - Filter rows by condition
- `AGGREGATE` - Group and aggregate
- `JOIN` - Merge datasets
- `TRANSFORM` - Create new columns
- `VISUALIZE` - Create plots
- `ANALYZE` - Statistical analysis
- `MODEL` - Build ML models

### 3. SafeExecutor (`executor.py`)

Executes generated code in a sandboxed environment.

```python
from hololoom.patterns.dsstar import SafeExecutor

executor = SafeExecutor(
    max_execution_time=30.0,
    enable_safety_checks=True
)

result = executor.execute(
    code="df = pd.read_csv('data.csv'); result = df['price'].mean()",
    context={'data_file': 'data.csv'}
)

if result.success:
    print("Outputs:", result.outputs)
else:
    print("Error:", result.error)
```

**Safety Features**:
- Restricted imports (pandas, numpy, matplotlib only)
- No file system access (except data files)
- No system commands
- No dynamic code execution (eval/exec)
- Stdout/stderr capture
- Timeout protection

### 4. Verifier (`verifier.py`)

Checks if execution achieved the goal.

```python
from hololoom.patterns.dsstar import Verifier

verifier = Verifier(strict_mode=False)
verification = verifier.verify(plan, execution_result)

print(f"Verified: {verification.verified}")
print(f"Confidence: {verification.confidence:.1%}")

if verification.issues:
    print("Issues:", verification.issues)

if verification.suggestions:
    print("Suggestions:", verification.suggestions)
```

**Verification Checks**:
1. Execution succeeded (no errors)
2. Expected outputs present
3. Output types reasonable
4. Query intent satisfied
5. No suspicious patterns (warnings, long execution)

**Confidence Scoring**:
- 1.0 = Perfect execution, all outputs as expected
- 0.9 = Minor issues (warnings)
- 0.7 = Missing some outputs
- 0.0 = Execution failed

### 5. DSStarOrchestrator (`orchestrator.py`)

Main loop coordinator with iterative refinement.

```python
from hololoom.patterns.dsstar import DSStarOrchestrator

orchestrator = DSStarOrchestrator(
    max_iterations=3,
    verification_threshold=0.7,
    enable_refinement=True
)

result = await orchestrator.process(
    query="Calculate monthly sales totals",
    data_file="sales.csv"
)

# View iteration history
for iteration in result.iterations:
    print(f"Iteration {iteration.iteration + 1}:")
    print(f"  Confidence: {iteration.verification.confidence:.1%}")
    print(f"  Issues: {iteration.verification.issues}")

# Explain entire process
print(orchestrator.explain_process(result))
```

## Integration with HoloLoom

DS-STAR leverages HoloLoom's existing infrastructure:

| DS-STAR Component | HoloLoom Integration |
|-------------------|----------------------|
| **DataAnalyzer** | SpinningWheel adapters (47 input formats) |
| **Planner** | Query routing + policy engine |
| **Executor** | Tool execution framework |
| **Verifier** | Alignment framework + reflection buffer |
| **Iterative Loop** | Recursive learning system (5 phases) |
| **Scratchpad** | Internal dialogue (Hofstadter strange loops) |

### Using with HoloLoom Core

```python
from hololoom import HoloLoom
from hololoom.patterns.dsstar import DSStarOrchestrator

# Initialize HoloLoom
async with HoloLoom() as loom:
    # Create DS-STAR orchestrator
    orchestrator = DSStarOrchestrator(use_hololoom_integration=True)

    # Process query
    result = await orchestrator.process(
        query="Analyze customer churn by region",
        data_file="customers.csv"
    )

    # Store result in HoloLoom memory
    await loom.experience(result.summary())
```

## Examples

### Example 1: Basic Data Analysis

```python
from hololoom.patterns.dsstar import process_query

result = await process_query(
    query="What is the correlation between age and income?",
    data_file="survey_data.csv",
    max_iterations=3
)

if result.success:
    correlation = result.final_outputs.get('correlation', 'N/A')
    print(f"Correlation: {correlation}")
```

### Example 2: Visualization

```python
orchestrator = DSStarOrchestrator()

result = await orchestrator.process(
    query="Plot sales trends over time",
    data_file="sales.csv"
)

# Plot is saved to 'plot' output
if 'plot' in result.final_outputs:
    print("Visualization created successfully")
```

### Example 3: Aggregation

```python
result = await process_query(
    query="Group by department and sum salaries",
    data_file="employees.csv"
)

if result.success:
    grouped_data = result.final_outputs.get('result')
    print(grouped_data)
```

### Example 4: With Scratchpad (Internal Dialogue)

```python
from hololoom.patterns.dsstar import DSStarOrchestrator, RecursiveScratchpad

orchestrator = DSStarOrchestrator()

async with RecursiveScratchpad() as scratchpad:
    # DS-STAR process with internal reasoning
    result = await orchestrator.process(
        query="Find outliers in temperature data",
        data_file="weather.csv"
    )

    # Record reasoning in scratchpad
    thought = await scratchpad.think(result.summary())

    # Explore reasoning
    dialogue = await scratchpad.dialogue_loop(
        initial_thought=thought,
        max_depth=3
    )

    print(dialogue.tree_visualization())
```

## Iteration History

DS-STAR tracks complete iteration history for analysis:

```python
result = await orchestrator.process(query, data_file)

for i, iteration in enumerate(result.iterations):
    print(f"\n=== Iteration {i+1} ===")
    print(f"Plan: {len(iteration.plan.steps)} steps")
    print(f"Execution: {'✓' if iteration.execution_result.success else '✗'}")
    print(f"Confidence: {iteration.verification.confidence:.1%}")

    if iteration.verification.issues:
        print("Issues:")
        for issue in iteration.verification.issues:
            print(f"  - {issue}")
```

## Configuration

```python
orchestrator = DSStarOrchestrator(
    max_iterations=3,              # Max refinement iterations
    verification_threshold=0.7,    # Min confidence to accept
    enable_refinement=True,        # Allow plan refinement
    use_hololoom_integration=True  # Use HoloLoom routing
)

# Configure components individually
orchestrator.analyzer = DataAnalyzer(
    max_sample_size=5,             # Sample values per column
    max_unique_show=10             # Max unique values to show
)

orchestrator.executor = SafeExecutor(
    max_execution_time=30.0,       # Timeout seconds
    enable_safety_checks=True      # Safety validation
)

orchestrator.verifier = Verifier(
    strict_mode=False              # Strict verification
)
```

## Testing

```bash
# Run DS-STAR tests
pytest src/hololoom/patterns/dsstar/tests/ -v

# Test individual components
pytest src/hololoom/patterns/dsstar/tests/test_analyzer.py
pytest src/hololoom/patterns/dsstar/tests/test_planner.py
pytest src/hololoom/patterns/dsstar/tests/test_executor.py
pytest src/hololoom/patterns/dsstar/tests/test_verifier.py
pytest src/hololoom/patterns/dsstar/tests/test_orchestrator.py
```

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| **Data analysis** | 50-200ms | Depends on file size |
| **Plan creation** | 10-50ms | Fast for simple queries |
| **Code execution** | 100ms-30s | Depends on computation |
| **Verification** | 5-10ms | Lightweight checks |
| **Full iteration** | 200ms-30s | Typically <2s |

## Limitations

Current limitations (future improvements):

1. **Single file only** - Multi-file joins not yet supported
2. **Pandas-centric** - Other libraries (Polars, Spark) not integrated
3. **Simple refinement** - Could use more sophisticated plan modification
4. **No async execution** - Execution is synchronous (blocking)
5. **Limited model support** - Basic ML, no deep learning

## Comparison to DS-STAR Paper

| Feature | Google DS-STAR | HoloLoom DS-STAR |
|---------|----------------|------------------|
| **Data Analyzer** | Custom | Pandas-based + SpinningWheel |
| **Planner** | LLM-based | Rule-based + HoloLoom routing |
| **Coder** | LLM code gen | Template-based + refinement |
| **Executor** | Jupyter kernel | Sandboxed Python exec |
| **Verifier** | LLM-based | Rule-based + HoloLoom alignment |
| **Router** | RL-based | HoloLoom policy engine |
| **Iterative Loop** | Multi-turn | Recursive learning (5 phases) |
| **Safety** | Basic | HoloLoom alignment framework |

## Future Roadmap

**Phase 2** (Next 3 months):
- LLM-based plan generation (optional)
- Multi-file support (joins, merges)
- Advanced refinement strategies
- Async execution with timeout

**Phase 3** (6 months):
- Polars/Spark support
- Deep learning model integration
- Multi-agent collaboration
- Streaming data support

**Phase 4** (12 months):
- Real-time data processing
- AutoML integration
- Production deployment tools
- Enterprise features

## References

- **Google Research DS-STAR Paper**: "Data Science with Self-Taught Autonomous Reasoning"
- **HoloLoom Documentation**: See `HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md`
- **Alignment Framework**: See `src/hololoom/weave/agentic/alignment/README.md`
- **Recursive Learning**: See `RECURSIVE_LEARNING_COMPLETE.md`

## License

MIT License - Same as HoloLoom

## Credits

- **DS-STAR Concept**: Google Research
- **HoloLoom Integration**: Blake Chasteen + Claude Code
- **Created**: January 2025
