# Promptly Diff & Merge System

Git-like diff and merge capabilities for intelligent prompt version management.

## Overview

The Promptly Diff & Merge system provides advanced tools for tracking, comparing, and merging prompt changes across versions and branches. Built with the same rigor as modern version control systems, it enables teams to collaborate on prompt engineering with confidence.

## Key Features

### 🔍 **Advanced Diffing**
- **Myers Algorithm**: Industry-standard diff algorithm for efficient change detection
- **Multiple Granularities**: Character, word, line, and semantic-level diffs
- **Smart Statistics**: Additions, deletions, changes, and similarity scores
- **Multiple Outputs**: Terminal colors, HTML reports, unified format, side-by-side

### 🔀 **Intelligent Merging**
- **Three-Way Merge**: Proper merge using common ancestor
- **Auto-Resolution**: Automatic conflict resolution with multiple strategies
- **Conflict Detection**: Precise identification of conflicting changes
- **Interactive Resolution**: Step-by-step conflict resolution workflow

### 📊 **Comprehensive Comparison**
- **Version Comparison**: Compare any two versions with detailed analysis
- **Branch Comparison**: See all differences between branches at once
- **Evaluation Tracking**: Monitor quality improvements and regressions
- **Metadata Tracking**: Track changes in prompt metadata and tags

### 🎨 **Beautiful Visualization**
- **Terminal Colors**: ANSI-colored diffs for easy reading
- **HTML Reports**: Professional reports for documentation and review
- **Side-by-Side View**: Compare versions in parallel
- **Context Highlighting**: Semantic annotations for meaningful changes

## Installation

The diff and merge system is included with Promptly:

```bash
pip install promptly
```

No additional dependencies required for basic functionality.

## Quick Start

### CLI Usage

```bash
# Initialize repository
promptly init

# Add and modify prompts
promptly add greeter "Hello {name}!"
promptly add greeter "Hey {name}, welcome!"

# View differences
promptly diff greeter --from 1 --to 2

# Create and compare branches
promptly branch feature
promptly checkout feature
promptly add greeter "Hey there, {name}!"
promptly branch-diff main feature

# Merge branches
promptly checkout main
promptly merge feature --strategy auto
```

### Python API

```python
from promptly import Promptly
from promptly.diff import DiffEngine, ComparisonEngine, TerminalDiff
from promptly.merge import MergeTool, MergeStrategy

# Initialize
promptly = Promptly()
promptly.init()

# Add versions
promptly.add("my_prompt", "Version 1 content")
promptly.add("my_prompt", "Version 2 content with changes")

# Compare versions
engine = ComparisonEngine(promptly)
comparison = engine.compare_versions("my_prompt", 1, 2)

# Display diff
print(TerminalDiff.render_comparison(comparison))

# Merge branches
merge_tool = MergeTool(promptly)
results = merge_tool.merge_branches("feature", "main", MergeStrategy.AUTO)
```

## Architecture

```
Promptly/promptly/
├── diff/
│   ├── __init__.py       # Public API
│   ├── engine.py         # Myers algorithm, diff strategies
│   ├── compare.py        # Version/branch comparison
│   ├── visual.py         # Terminal & HTML rendering
│   └── README.md         # Module documentation
│
├── merge/
│   ├── __init__.py       # Public API
│   └── tool.py           # Three-way merge, conflict resolution
│
├── test_diff_merge.py    # Comprehensive test suite
├── DIFF_MERGE_GUIDE.md   # Complete usage guide
└── examples/
    └── diff_merge_demo.py # Interactive demos
```

## Core Concepts

### Diff Levels

**Character Level** - Most granular, tracks every character change
```python
result = engine.diff(old, new, DiffLevel.CHARACTER)
```

**Word Level** - Best for natural language prompts
```python
result = engine.diff(old, new, DiffLevel.WORD)
```

**Line Level** - Default, most efficient for most cases
```python
result = engine.diff(old, new, DiffLevel.LINE)
```

**Semantic Level** - Understands meaning, not just syntax
```python
result = engine.diff(old, new, DiffLevel.SEMANTIC)
```

### Merge Strategies

**AUTO** - Attempt automatic resolution (recommended)
```python
MergeStrategy.AUTO
```

**OURS** - Keep our changes in conflicts
```python
MergeStrategy.OURS
```

**THEIRS** - Accept their changes in conflicts
```python
MergeStrategy.THEIRS
```

**UNION** - Keep both (concatenate)
```python
MergeStrategy.UNION
```

**MANUAL** - Mark all conflicts for manual resolution
```python
MergeStrategy.MANUAL
```

### Output Formats

**Terminal** - Colored ANSI output for console
```bash
promptly diff my_prompt --format terminal
```

**HTML** - Beautiful reports for documentation
```bash
promptly diff my_prompt --format html --output report.html
```

**Side-by-Side** - Parallel column comparison
```bash
promptly diff my_prompt --format side-by-side
```

**Unified** - Traditional unified diff format
```bash
promptly diff my_prompt --format unified
```

## CLI Reference

### diff
Show differences between prompt versions
```bash
promptly diff <name> [--from V1] [--to V2] [--level LEVEL] [--format FORMAT]
```

### compare
Compare two different prompts
```bash
promptly compare <prompt1> <prompt2> [--level LEVEL] [--format FORMAT]
```

### branch-diff
Compare two branches
```bash
promptly branch-diff <branch1> [branch2] [--format FORMAT]
```

### merge
Merge source branch into target
```bash
promptly merge <source> [--into TARGET] [--strategy STRATEGY] [--dry-run]
```

### conflicts
Manage merge conflicts
```bash
promptly conflicts list
promptly conflicts resolve <name> [--strategy STRATEGY]
```

## Common Workflows

### Prompt Refinement

```bash
# Create experiment branch
promptly branch experiment

# Make changes
promptly add my_prompt "Improved version..."

# Compare with main
promptly branch-diff main experiment

# If good, merge back
promptly checkout main
promptly merge experiment
```

### Team Collaboration

```bash
# Alice's changes
promptly branch alice-feature
promptly add shared_prompt "Alice's version"

# Bob's changes
promptly branch bob-feature
promptly add shared_prompt "Bob's version"

# Compare approaches
promptly branch-diff alice-feature bob-feature

# Merge and resolve conflicts
promptly checkout main
promptly merge alice-feature
promptly merge bob-feature
```

### Quality Assurance

```python
# Run evaluations on both versions
old_results = promptly.eval_prompt("my_prompt", test_cases)

promptly.add("my_prompt", "New version...")
new_results = promptly.eval_prompt("my_prompt", test_cases)

# Compare quality
engine = ComparisonEngine(promptly)
comparison = engine.compare_evaluations("my_prompt", old_results, new_results)

if comparison.score_delta < 0:
    print("Warning: Quality regression detected!")
```

## Examples

See [`examples/diff_merge_demo.py`](examples/diff_merge_demo.py) for interactive demonstrations of:

1. Basic diffing at multiple levels
2. Version comparison with metadata
3. Branch comparison
4. Merging with conflict resolution
5. HTML report generation
6. Semantic diff analysis
7. Statistics and similarity metrics

Run the demo:
```bash
cd Promptly/promptly
python examples/diff_merge_demo.py
```

## Testing

Run the comprehensive test suite:

```bash
cd Promptly/promptly
python test_diff_merge.py
```

Tests cover:
- Diff engine (character, word, line, semantic)
- Comparison tools (versions, branches, evaluations)
- Visual rendering (terminal, HTML)
- Three-way merge
- Merge strategies
- Conflict resolution
- Full integration workflows

## Performance

| Operation | Complexity | Typical Time |
|-----------|-----------|--------------|
| Character diff | O(N*M) | ~10ms for 1KB |
| Word diff | O(N*M) | ~5ms for 1KB |
| Line diff | O(N*M) | ~1ms for 100 lines |
| Three-way merge | O(N) | ~5ms clean, ~50ms conflicts |
| HTML generation | O(N) | ~2ms per chunk |

Optimized for prompts up to 100KB. For larger texts, use line-level diff.

## Advanced Features

### Custom Semantic Analysis

```python
class CustomDiffEngine(DiffEngine):
    def _infer_semantic_context(self, chunk):
        # Add LLM-powered semantic understanding
        return analyze_with_llm(chunk.old_text, chunk.new_text)
```

### Batch Processing

```python
def batch_diff_all_prompts(promptly, branch1, branch2):
    engine = ComparisonEngine(promptly)
    comparison = engine.compare_branches(branch1, branch2)

    reports = {}
    for name in comparison.prompts_modified:
        reports[name] = HTMLDiff.render(comparison.diff_results[name])

    return reports
```

### Automated Quality Checks

```python
def check_for_regressions(promptly, prompt_name, old_v, new_v):
    engine = ComparisonEngine(promptly)
    comparison = engine.compare_versions(prompt_name, old_v, new_v)

    if comparison.diff_result.stats.similarity < 0.5:
        raise ValueError("Major changes detected, review required")

    # Run evaluations...
```

## Documentation

- **[DIFF_MERGE_GUIDE.md](DIFF_MERGE_GUIDE.md)** - Complete guide with examples
- **[diff/README.md](diff/README.md)** - Module-specific documentation
- **[examples/diff_merge_demo.py](examples/diff_merge_demo.py)** - Interactive demos

## Troubleshooting

**Q: Diffs are too slow for large prompts**
A: Use line-level diff instead of character-level, or use `diff_stats_only()` for just metrics.

**Q: Merge conflicts not resolving automatically**
A: Try different strategies (UNION, OURS, THEIRS) or use interactive resolution.

**Q: HTML files are too large**
A: Limit context in diff generation or create summary-only reports.

**Q: Semantic diff not providing context**
A: Semantic analysis uses heuristics; for better results, integrate with LLM.

## Future Enhancements

- [ ] LLM-powered semantic diff and merge
- [ ] Patch file generation (`.diff` files)
- [ ] Visual conflict resolution UI
- [ ] Diff caching for performance
- [ ] Binary content diffing
- [ ] Conflict prediction
- [ ] Automated regression testing

## Contributing

Contributions welcome! Please:

1. Add tests to `test_diff_merge.py`
2. Update documentation
3. Follow existing code style
4. Add CLI commands if applicable

## License

Same as Promptly main project.

## Credits

- Myers diff algorithm: Eugene W. Myers (1986)
- Inspired by Git's diff and merge capabilities
- Built with Python's difflib and modern diff/merge algorithms

---

**Ready to level up your prompt management?** Start with the [Quick Start](#quick-start) or run the [demo](examples/diff_merge_demo.py)!
