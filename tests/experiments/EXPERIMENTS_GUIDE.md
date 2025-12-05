# 🧪 Consciousness Stack Experiments Guide

Automated experiments that systematically test the consciousness stack behavior.

## 🚀 Quick Start

```powershell
python experiments/run_experiments.py
```

This runs **16 experiments** across 4 categories and generates:
- `results/all_experiments.json` - Raw data
- `results/experiment_report.md` - Formatted report

## 📊 Experiment Categories

### 1. Fusion Impact (2 experiments)
**Tests**: Multipass graph crawling ON vs OFF

**Configuration**:
- Complexity: FULL
- Max Memories: 10
- Token Budget: 4000

**What to Look For**:
- ✅ Depth increases with fusion enabled
- ✅ More connected knowledge discovered
- ✅ Slightly higher processing time
- ✅ Composite scores change based on graph proximity

**Example Results**:
```
Fusion OFF → Fusion ON
• Memories:   +0 to +3 (depending on query)
• Max Depth:  +1 to +2 hops
• Avg Score:  Variable (graph-dependent)
• Time:       +0.5 to +2ms overhead
```

### 2. Complexity Scaling (4 experiments)
**Tests**: LITE → FAST → FULL → RESEARCH progression

**Configuration**:
- Fusion: ON
- Max Memories: 10
- Token Budget: 4000

**What to Look For**:
- ✅ More retrieval passes with higher complexity
- ✅ Deeper graph traversal (depth increases)
- ✅ Better quality items discovered
- ✅ Time scales with pass count

**Example Results**:
```
LITE (1 pass)     → 5-7 memories, depth 0, <1ms
FAST (2 passes)   → 8-10 memories, depth 1, <2ms
FULL (3 passes)   → 10-12 memories, depth 2, <3ms
RESEARCH (4 passes) → 12-15 memories, depth 3, <5ms
```

### 3. Token Budget Impact (5 experiments)
**Tests**: Budget from 2000 → 8000 tokens

**Configuration**:
- Complexity: FULL
- Fusion: ON
- Max Memories: 10

**What to Look For**:
- ✅ Less compression needed with larger budgets
- ✅ More items included uncompressed
- ✅ Token usage percentage decreases
- ✅ Quality (importance) remains stable

**Example Results**:
```
Budget 2000: 170 tokens (13.1% usage), 0-2 compressed
Budget 4000: 250 tokens (6.5% usage), 0-1 compressed
Budget 8000: 280 tokens (3.3% usage), 0 compressed
```

### 4. Memory Limits (5 experiments)
**Tests**: Max memories from 5 → 20

**Configuration**:
- Complexity: FULL
- Fusion: ON
- Token Budget: 4000

**What to Look For**:
- ✅ Quality vs quantity tradeoff
- ✅ Avg score may decrease with more items
- ✅ Token usage increases proportionally
- ✅ Processing time increases slightly

**Example Results**:
```
5 memories:  Avg score 0.902, 130 tokens, high quality
10 memories: Avg score 0.883, 170 tokens, balanced
20 memories: Avg score 0.865, 220 tokens, comprehensive
```

## 📈 Metrics Tracked

### Memory Metrics
- **Memories Retrieved**: Total items found
- **Max Depth**: Deepest graph hop reached
- **Avg Composite Score**: Quality indicator (0-1)
- **Fusion Enabled**: Boolean flag

### Packing Metrics
- **Total Tokens**: Actual tokens used
- **Token Budget**: Available budget
- **Token Usage %**: Efficiency metric
- **Elements Included/Compressed/Excluded**: Packing decisions
- **Avg/Min Importance**: Quality indicators

### Performance Metrics
- **Total Time**: End-to-end processing
- **Fusion Time**: Memory retrieval
- **Packing Time**: Context optimization
- **Generation Time**: Response creation

### Awareness Metrics
- **Confidence**: Query understanding (0-1)
- **Domain/Subdomain**: Category detection
- **Phrase Type**: question vs statement
- **Knowledge Gap**: Detected uncertainty

## 🎯 How to Use Results

### Understand Tradeoffs

**Fusion ON/OFF**:
- ON: Better knowledge discovery, +1-2ms overhead
- OFF: Faster, but misses connected knowledge

**Complexity Levels**:
- LITE: Speed (use for simple queries)
- FULL: Balance (default for most cases)
- RESEARCH: Depth (use for complex analysis)

**Token Budgets**:
- Small (2000): Aggressive compression
- Medium (4000): Balanced efficiency
- Large (8000): Maximum context

**Memory Limits**:
- Few (5): High precision
- Many (20): High recall

### Optimize for Your Use Case

**Speed-Critical** (chat, real-time):
```
Complexity: LITE
Fusion: OFF
Max Memories: 5
Token Budget: 2000
Target: <50ms
```

**Balanced** (general Q&A):
```
Complexity: FULL
Fusion: ON
Max Memories: 10
Token Budget: 4000
Target: <300ms
```

**Research** (deep analysis):
```
Complexity: RESEARCH
Fusion: ON
Max Memories: 20
Token Budget: 8000
Target: No limit
```

## 🔧 Customizing Experiments

### Add Your Own Test Query

Edit `run_experiments.py`:

```python
self.test_query = "Your custom query here"
```

### Add New Experiment

```python
async def experiment_5_custom(self) -> List[ExperimentResult]:
    """Your custom experiment"""
    results = []
    
    # Your test configurations
    for param in test_values:
        result = await self.run_single_experiment(
            experiment_name=f"Custom: {param}",
            query=self.test_query,
            complexity="FULL",
            use_fusion=True,
            max_memories=10,
            token_budget=4000
        )
        results.append(result)
    
    return results
```

Then add to `run_all_experiments()`:
```python
all_results.extend(await self.experiment_5_custom())
```

### Change Output Directory

```python
runner = ExperimentRunner(output_dir="my_experiments/results")
```

## 📊 Reading the Report

### Tables Show Raw Data

```markdown
| Budget | Tokens Used | Usage % | Compressed | Excluded |
|--------|-------------|---------|------------|----------|
| 2000   | 170         | 13.1%   | 2          | 1        |
| 4000   | 250         | 6.5%    | 1          | 0        |
```

**Interpretation**:
- Larger budget → lower usage %
- Larger budget → less compression
- Sweet spot often around 30-40% usage

### Key Findings Summarize Insights

```markdown
### Fusion Impact
- Graph depth increased by **2 hops** with fusion enabled
- Score changed by **+0.034**
- Time overhead: **+1.52ms**
```

**Interpretation**:
- Fusion discovers deeper connections
- Quality improves slightly
- Cost is ~1-2ms extra time

## 🐛 Troubleshooting

### "Module not found"
Make sure to run from repo root:
```powershell
cd c:\Users\blake\Documents\mythRL
python experiments/run_experiments.py
```

### Experiments show no differences
- Demo backend is small (7 items)
- Connect to larger knowledge base for dramatic results
- Differences become apparent with 100+ items

### Want more detailed metrics
Edit `ExperimentResult` dataclass to add fields:
```python
@dataclass
class ExperimentResult:
    # Add your metrics here
    cache_hit_rate: float = 0.0
    unique_entities: int = 0
```

## 🎨 Visualizing Results

### Manual Analysis
1. Open `results/experiment_report.md` in VS Code
2. Review tables and key findings
3. Compare metrics across experiments

### Programmatic Analysis
```python
import json

with open('experiments/results/all_experiments.json') as f:
    data = json.load(f)

# Analyze results
for result in data['results']:
    print(f"{result['experiment_name']}: {result['total_time_ms']}ms")
```

### Plot with Matplotlib (optional)
```python
import matplotlib.pyplot as plt
import json

with open('experiments/results/all_experiments.json') as f:
    data = json.load(f)

# Extract complexity experiments
complexity_results = [r for r in data['results'] if 'Complexity' in r['experiment_name']]
times = [r['total_time_ms'] for r in complexity_results]
levels = [r['parameters']['complexity'] for r in complexity_results]

plt.bar(levels, times)
plt.xlabel('Complexity Level')
plt.ylabel('Time (ms)')
plt.title('Complexity vs Performance')
plt.show()
```

## 🎯 Next Steps

### 1. Run Experiments
```powershell
python experiments/run_experiments.py
```

### 2. Review Results
Open `experiments/results/experiment_report.md`

### 3. Analyze Findings
- Which configuration works best for your use case?
- What are the tradeoffs?
- Where should you optimize?

### 4. Apply Learnings
Update your application configuration based on experiment results

### 5. Connect Real Backend
Replace `DemoMemoryBackend` with Neo4j/Qdrant for production metrics

### 6. Continuous Experimentation
Re-run experiments after changes to validate improvements

## 📚 Related Documentation

- `WEB_UI_COMPLETE.md` - Interactive UI guide
- `COMPLETE_CONSCIOUSNESS_STACK.md` - Architecture overview
- `ui/README.md` - UI documentation
- `results/experiment_report.md` - Latest results

---

**Status**: 🟢 READY TO EXPERIMENT

Run the experiments and optimize your consciousness stack configuration!
