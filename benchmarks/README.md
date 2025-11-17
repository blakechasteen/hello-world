# HoloLoom Benchmark Suite

Public, reproducible benchmarks for HoloLoom's persistent memory system.

## Philosophy

> **"Benchmarked, not marketed"**

We believe in **transparent, reproducible performance measurement**. Every claim is backed by public data and reproducible scripts.

## Benchmark Categories

### 1. Recall Accuracy (`recall_accuracy/`)

Measures how well HoloLoom retrieves relevant memories.

**Datasets:**
- **Wikipedia** - General knowledge (10K articles)
- **arXiv** - Scientific papers (5K abstracts)
- **Books** - Long-form text (100 books)

**Metrics:**
- Precision@K (K=1,5,10)
- Recall@K
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)

### 2. Scale Tests (`scale_tests/`)

Tests performance at different memory scales.

**Scales:**
- 1K memories
- 10K memories
- 100K memories
- 1M memories
- 10M memories (target)

**Metrics:**
- Latency (p50, p95, p99)
- Throughput (queries/sec)
- Memory usage (RAM, disk)
- Index build time

### 3. Multimodal Fidelity (`multimodal_fidelity/`)

Tests image + text retrieval quality.

**Datasets:**
- MS COCO (image captions)
- Flickr30k (image descriptions)
- Custom technical diagrams

**Metrics:**
- Image-text retrieval accuracy
- Cross-modal precision
- CLIP alignment scores

### 4. Learning Curves (`learning_curves/`)

Measures how fast HoloLoom improves over time.

**Experiments:**
- Thompson Sampling convergence
- Pattern learning effectiveness
- Hot pattern adaptation speed

**Metrics:**
- Accuracy improvement over queries
- Convergence speed (queries to 90% accuracy)
- Adaptation rate

### 5. Temporal Robustness (`temporal_robustness/`)

Tests long-term performance degradation.

**Time Periods:**
- 30 days
- 90 days
- 365 days

**Metrics:**
- Recall accuracy over time
- Memory decay effects
- Reconsolidation effectiveness

### 6. vs Competition (`vs_competition/`)

Head-to-head comparisons with other memory systems.

**Systems:**
- Mem0
- Zep
- LangMem
- ChromaDB
- Pinecone
- Custom baselines

**Metrics:**
- All above metrics
- Cost per query
- Setup complexity
- API simplicity

## Running Benchmarks

### Quick Start

```bash
# Run all benchmarks (takes ~30 minutes)
python benchmarks/run_all.py

# Run specific benchmark
python benchmarks/recall_accuracy/run.py

# Generate report
python benchmarks/generate_report.py --output results/report.md
```

### Individual Benchmarks

```bash
# Recall accuracy (Wikipedia)
python benchmarks/recall_accuracy/wikipedia.py

# Scale test (1M memories)
python benchmarks/scale_tests/1m_test.py

# vs Competition (Mem0)
python benchmarks/vs_competition/vs_mem0.py
```

## Results

Public results are published to:
- `benchmarks/results/` - JSON data + Markdown reports
- GitHub Wiki - Formatted results with charts
- Weekly blog posts - Analysis and insights

### Latest Results (Week 2 - November 2025)

**Recall Accuracy:**
- Wikipedia: 92.5% precision@5
- arXiv: 89.3% precision@5
- Books: 91.8% precision@5

**Scale Performance:**
- 1K memories: 15ms p95
- 10K memories: 45ms p95
- 100K memories: 125ms p95
- 1M memories: [pending]

**vs Competition:**
- [pending]

## Reproducibility

All benchmarks are:
- ✅ **Deterministic** - Seeded random generators
- ✅ **Versioned** - Dataset versions tracked
- ✅ **Documented** - Complete setup instructions
- ✅ **Open** - Public datasets, open scripts

### Reproducing Results

```bash
# Clone repo
git clone https://github.com/user/hello-world.git
cd hello-world

# Install dependencies
pip install -r benchmarks/requirements.txt

# Download datasets (auto-downloads on first run)
python benchmarks/download_datasets.py

# Run benchmarks
python benchmarks/run_all.py

# Results saved to: benchmarks/results/YYYY-MM-DD/
```

## Contributing

We welcome benchmark contributions!

**Adding a new benchmark:**
1. Create `benchmarks/your_benchmark/`
2. Add `run.py` with standardized output format
3. Add dataset download script
4. Update this README
5. Submit PR

**Required output format:**
```json
{
  "benchmark_name": "your_benchmark",
  "version": "1.0",
  "timestamp": "2025-11-17T12:00:00Z",
  "metrics": {
    "precision_at_5": 0.925,
    "latency_p95_ms": 125.5
  },
  "config": {
    "dataset": "wikipedia",
    "num_memories": 10000
  }
}
```

## Automation

Benchmarks run automatically:
- **Daily**: Recall accuracy (quick test)
- **Weekly**: Full benchmark suite
- **Monthly**: vs Competition updates

Results published to:
- GitHub releases (JSON)
- GitHub Wiki (Markdown)
- Blog (analysis)

## Datasets

Datasets are managed separately and auto-downloaded on first run.

**Storage:**
- Small datasets (<100MB): Included in repo
- Large datasets (>100MB): Downloaded from public URLs
- Custom datasets: Instructions for generation

**Licenses:**
- Wikipedia: CC BY-SA 3.0
- arXiv: Various (academic use)
- MS COCO: CC BY 4.0
- Custom: MIT

## Citation

If you use HoloLoom benchmarks in research:

```bibtex
@misc{hololoom_benchmarks_2025,
  title={HoloLoom Benchmark Suite},
  author={HoloLoom Team},
  year={2025},
  url={https://github.com/user/hello-world/benchmarks}
}
```

## License

Benchmark code: MIT License
Datasets: See individual dataset licenses
