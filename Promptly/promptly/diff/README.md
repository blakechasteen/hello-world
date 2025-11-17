# Promptly Diff & Merge System

Advanced git-like diff and merge capabilities for prompt version management.

## Quick Start

```python
from promptly import Promptly
from promptly.diff import DiffEngine, ComparisonEngine, TerminalDiff
from promptly.merge import MergeTool, MergeStrategy

# Initialize Promptly
promptly = Promptly()

# Create diff engine
diff_engine = DiffEngine()
comparison_engine = ComparisonEngine(promptly)

# Compare two versions
result = diff_engine.diff(old_text, new_text)
print(TerminalDiff.render(result))

# Merge branches
merge_tool = MergeTool(promptly)
results = merge_tool.merge_branches("feature", "main", MergeStrategy.AUTO)
```

## Features

### 🔍 Diff Engine

- **Myers Algorithm**: Industry-standard diff algorithm
- **Multiple Levels**: Character, word, line, semantic
- **Statistics**: Additions, deletions, changes, similarity scores
- **Output Formats**: Unified, side-by-side, terminal colors, HTML

### 🔀 Merge Tool

- **Three-Way Merge**: Base + Ours + Theirs
- **Merge Strategies**: AUTO, OURS, THEIRS, UNION, MANUAL
- **Conflict Detection**: Automatic conflict identification
- **Interactive Resolution**: Step-by-step conflict resolution

### 🎨 Visual Rendering

- **Terminal Colors**: ANSI color codes for deletions/insertions
- **HTML Reports**: Beautiful diff reports for team reviews
- **Side-by-Side**: Compare versions in parallel columns

### 📊 Comparison Tools

- **Version Comparison**: Compare any two versions
- **Branch Comparison**: See all changes between branches
- **Evaluation Comparison**: Track quality improvements/regressions

## CLI Commands

```bash
# Diff two versions
promptly diff my_prompt --from 1 --to 3 --format terminal

# Compare prompts
promptly compare prompt1 prompt2 --level word

# Compare branches
promptly branch-diff main feature --format detailed

# Merge branches
promptly merge feature --strategy auto

# Manage conflicts
promptly conflicts list
promptly conflicts resolve my_prompt --strategy ours
```

## Architecture

```
promptly/diff/
├── __init__.py      # Public API
├── engine.py        # Core diff algorithms (Myers, character, word, line, semantic)
├── compare.py       # Comparison tools (versions, branches, evaluations)
└── visual.py        # Rendering (terminal colors, HTML)

promptly/merge/
├── __init__.py      # Public API
└── tool.py          # Merge algorithms (three-way, conflict resolution)
```

## Examples

### Simple Diff

```python
from promptly.diff import quick_diff

result = quick_diff("Old text", "New text", level="word")
print(f"Similarity: {result.stats.similarity:.1%}")
```

### Branch Merge

```python
merge_tool = MergeTool(promptly)
results = merge_tool.merge_branches("feature", "main")

for name, result in results.items():
    if result.success:
        print(f"✓ {name} merged")
    else:
        print(f"✗ {name} has conflicts")
```

### HTML Export

```python
from promptly.diff import HTMLDiff

comparison = engine.compare_versions("my_prompt", 1, 2)
html = HTMLDiff.render_comparison(comparison)

with open("diff_report.html", "w") as f:
    f.write(html)
```

## Documentation

See [DIFF_MERGE_GUIDE.md](../DIFF_MERGE_GUIDE.md) for comprehensive documentation, examples, and best practices.

## Testing

```bash
# Run tests
cd Promptly/promptly
python test_diff_merge.py
```

## Performance

- **Character diff**: O(N*M) where N, M are text lengths
- **Line diff**: O(N*M) where N, M are line counts
- **Three-way merge**: O(N) for clean merges, O(N*M) for conflicts
- **HTML generation**: O(N) where N is number of chunks

## Future Enhancements

- [ ] LLM-powered semantic diff
- [ ] Conflict recommendation system
- [ ] Diff visualization in web UI
- [ ] Patch generation and application
- [ ] Binary diff support
- [ ] Performance optimization for very large prompts

## Contributing

1. Add new features to appropriate module
2. Write tests in `test_diff_merge.py`
3. Update documentation
4. Add CLI commands if applicable

## License

Same as Promptly main project.
