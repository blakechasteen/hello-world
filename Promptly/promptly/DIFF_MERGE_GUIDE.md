# Promptly Diff & Merge Guide

Complete guide to using Promptly's advanced diff and merge capabilities for prompt management.

## Table of Contents

1. [Overview](#overview)
2. [Diff Engine](#diff-engine)
3. [Comparison Tools](#comparison-tools)
4. [Merge Tool](#merge-tool)
5. [Visual Diff](#visual-diff)
6. [CLI Commands](#cli-commands)
7. [Advanced Usage](#advanced-usage)
8. [Best Practices](#best-practices)

---

## Overview

Promptly's diff and merge system provides git-like capabilities for managing prompt versions:

- **Character, word, and line-level diffs** for precise change tracking
- **Semantic diff** with context understanding
- **Three-way merge** with automatic conflict resolution
- **Branch comparison** for reviewing changes across branches
- **Visual rendering** with terminal colors and HTML reports
- **Interactive conflict resolution** for manual merging

### Key Features

- 🔍 **Myers diff algorithm** - Industry-standard diffing
- 🎨 **Multiple output formats** - Terminal, HTML, side-by-side
- 🤖 **Semantic analysis** - Understand meaning changes, not just text
- 🔀 **Smart merging** - Automatic conflict resolution strategies
- 📊 **Statistics** - Detailed change metrics and similarity scores
- 🎭 **Conflict management** - Full conflict tracking and resolution

---

## Diff Engine

### Basic Diffing

```python
from promptly.diff import DiffEngine, DiffLevel

engine = DiffEngine()

# Line-level diff (default)
result = engine.diff(old_text, new_text, DiffLevel.LINE)

# Word-level diff (more granular)
result = engine.diff(old_text, new_text, DiffLevel.WORD)

# Character-level diff (most granular)
result = engine.diff(old_text, new_text, DiffLevel.CHARACTER)

# Semantic diff (with context understanding)
result = engine.diff(old_text, new_text, DiffLevel.SEMANTIC)
```

### Understanding Diff Results

```python
# Access diff chunks
for chunk in result.chunks:
    print(f"Type: {chunk.type}")
    print(f"Old: {chunk.old_text}")
    print(f"New: {chunk.new_text}")

# View statistics
print(result.stats)
# Output: DiffStats(+5 -2 ~1, 85.3% similar)

print(f"Additions: {result.stats.additions}")
print(f"Deletions: {result.stats.deletions}")
print(f"Changes: {result.stats.changes}")
print(f"Similarity: {result.stats.similarity:.1%}")

# Get unified diff format
print(result.unified_format)
```

### Quick Diff

```python
from promptly.diff import quick_diff

# Quick line diff
result = quick_diff(old_text, new_text)

# Quick word diff
result = quick_diff(old_text, new_text, level="word")
```

### Side-by-Side Diff

```python
side_by_side = engine.side_by_side(
    old_text,
    new_text,
    width=100,  # Total display width
    margin=2    # Space between columns
)
print(side_by_side)
```

---

## Comparison Tools

### Compare Prompt Versions

```python
from promptly import Promptly
from promptly.diff import ComparisonEngine

promptly = Promptly()
engine = ComparisonEngine(promptly)

# Compare two versions
comparison = engine.compare_versions(
    name="my_prompt",
    version1=1,
    version2=3,
    level=DiffLevel.LINE
)

# View summary
print(comparison.summary())

# Access diff result
print(comparison.diff_result.stats)

# Check metadata changes
if comparison.metadata_diff['changed']:
    print("Metadata changed:")
    for key, change in comparison.metadata_diff['changed'].items():
        print(f"  {key}: {change['old']} -> {change['new']}")
```

### Compare Branches

```python
# Compare two branches
comparison = engine.compare_branches("main", "feature")

# View summary
print(comparison.summary())

# Prompts only in main
print(comparison.prompts_only_in_old)

# Prompts only in feature
print(comparison.prompts_only_in_new)

# Modified prompts
for name in comparison.prompts_modified:
    diff = comparison.diff_results[name]
    print(f"{name}: {diff.stats}")
```

### Compare Evaluation Results

```python
from promptly.diff import EvaluationComparison

comparison = engine.compare_evaluations(
    prompt_name="my_prompt",
    old_results=old_eval_results,
    new_results=new_eval_results
)

print(comparison.summary())
# Output:
# Evaluation comparison for: my_prompt
# Score: 0.750 -> 0.820 (+0.070, improvement)
# Improved tests: 5
# Regressed tests: 1
# Unchanged tests: 4

# Identify which tests improved
for test_idx in comparison.improved_tests:
    print(f"Test {test_idx} improved")
```

---

## Merge Tool

### Three-Way Merge

```python
from promptly.merge import ThreeWayMerge, MergeStrategy

# Perform three-way merge
result = ThreeWayMerge.merge(
    base="Common ancestor content",
    ours="Our version",
    theirs="Their version",
    strategy=MergeStrategy.AUTO
)

# Check for success
if result.success:
    print("Merge successful!")
    print(result.merged_content)
else:
    print(f"Merge has {len(result.conflicts)} conflicts")
```

### Merge Strategies

```python
# AUTO - Attempt automatic resolution
result = ThreeWayMerge.merge(base, ours, theirs, MergeStrategy.AUTO)

# OURS - Always keep our changes
result = ThreeWayMerge.merge(base, ours, theirs, MergeStrategy.OURS)

# THEIRS - Always keep their changes
result = ThreeWayMerge.merge(base, ours, theirs, MergeStrategy.THEIRS)

# UNION - Keep both (concatenate)
result = ThreeWayMerge.merge(base, ours, theirs, MergeStrategy.UNION)

# MANUAL - Leave all conflicts unresolved
result = ThreeWayMerge.merge(base, ours, theirs, MergeStrategy.MANUAL)
```

### Merge Branches

```python
from promptly.merge import MergeTool

promptly = Promptly()
merge_tool = MergeTool(promptly)

# Merge feature branch into main
results = merge_tool.merge_branches(
    source_branch="feature",
    target_branch="main",
    strategy=MergeStrategy.AUTO
)

# Check results
for prompt_name, result in results.items():
    if result.success:
        print(f"{prompt_name}: merged successfully")
    else:
        print(f"{prompt_name}: has conflicts")
        for conflict in result.conflicts:
            print(f"  Conflict at lines {conflict.start_line}-{conflict.end_line}")
```

### Handle Merge Conflicts

```python
# View conflict details
for conflict in result.conflicts:
    print(f"Conflict: {conflict}")
    print(f"Base: {conflict.base_content}")
    print(f"Ours: {conflict.ours_content}")
    print(f"Theirs: {conflict.theirs_content}")
    print(conflict.to_marker_format())

# Manually resolve conflict
merge_tool.resolve_conflict(
    conflict=conflict,
    resolution="Manually merged content"
)
```

### Interactive Merge

```python
from promptly.merge import InteractiveMerge

# Create interactive merge session
interactive = InteractiveMerge(merge_result)

# Process conflicts one by one
while interactive.has_more_conflicts():
    conflict = interactive.next_conflict()

    print(f"\nConflict {interactive.current_conflict_index}:")
    print(conflict.to_marker_format())

    # Get user choice
    choice = input("Choose (ours/theirs/union/edit): ")

    if choice == "edit":
        resolution = input("Enter resolved content: ")
        interactive.resolve_current(resolution)
    else:
        interactive.resolve_current(choice)

    # Show progress
    resolved, total = interactive.get_progress()
    print(f"Progress: {resolved}/{total} conflicts resolved")
```

---

## Visual Diff

### Terminal Colors

```python
from promptly.diff import TerminalDiff

# Render diff with colors
colored_output = TerminalDiff.render(diff_result, show_stats=True)
print(colored_output)

# Inline highlighting
inline_output = TerminalDiff.render_inline(diff_result)
print(inline_output)

# Side-by-side with colors
side_by_side = TerminalDiff.render_side_by_side(
    old_text,
    new_text,
    width=120,
    colored=True
)
print(side_by_side)

# Render comparison
comparison_output = TerminalDiff.render_comparison(version_comparison)
print(comparison_output)

# Render branch comparison
branch_output = TerminalDiff.render_branch_comparison(branch_comparison)
print(branch_output)
```

### HTML Reports

```python
from promptly.diff import HTMLDiff

# Generate HTML diff
html = HTMLDiff.render(diff_result, title="My Diff Report")

# Save to file
with open("diff_report.html", "w") as f:
    f.write(html)

# Generate version comparison HTML
html = HTMLDiff.render_comparison(version_comparison)

# Generate side-by-side HTML
html = HTMLDiff.render_side_by_side(
    old_text,
    new_text,
    old_label="Version 1",
    new_label="Version 2"
)
```

---

## CLI Commands

### Diff Command

```bash
# Compare prompt versions
promptly diff my_prompt --from 1 --to 3

# Use different diff levels
promptly diff my_prompt --level word
promptly diff my_prompt --level char
promptly diff my_prompt --level semantic

# Different output formats
promptly diff my_prompt --format terminal
promptly diff my_prompt --format unified
promptly diff my_prompt --format side-by-side

# Export to HTML
promptly diff my_prompt --format html --output report.html

# Compare latest to previous
promptly diff my_prompt
```

### Compare Command

```bash
# Compare two different prompts
promptly compare prompt1 prompt2

# With specific diff level
promptly compare prompt1 prompt2 --level word

# HTML output
promptly compare prompt1 prompt2 --format html > comparison.html
```

### Branch Diff Command

```bash
# Compare branches
promptly branch-diff main feature

# Detailed comparison
promptly branch-diff main feature --format detailed

# Compare with current branch
promptly branch-diff main
```

### Merge Command

```bash
# Merge feature into current branch
promptly merge feature

# Merge with specific target
promptly merge feature --into main

# Choose merge strategy
promptly merge feature --strategy ours
promptly merge feature --strategy theirs
promptly merge feature --strategy union
promptly merge feature --strategy auto

# Dry run (show what would be merged)
promptly merge feature --dry-run
```

### Conflict Management

```bash
# List conflicts
promptly conflicts list

# Resolve conflicts
promptly conflicts resolve my_prompt --strategy ours
promptly conflicts resolve my_prompt --strategy edit
```

---

## Advanced Usage

### Custom Diff Strategies

```python
class CustomDiffEngine(DiffEngine):
    def _semantic_diff(self, old_text, new_text):
        # Custom semantic analysis
        result = super()._semantic_diff(old_text, new_text)

        # Add custom annotations
        for chunk in result.chunks:
            if chunk.type == DiffType.REPLACE:
                # Use LLM for semantic understanding
                chunk.context = analyze_with_llm(chunk)

        return result
```

### Batch Diff Processing

```python
def diff_all_prompts(promptly, branch1, branch2):
    """Generate diffs for all prompts between branches"""
    engine = ComparisonEngine(promptly)
    comparison = engine.compare_branches(branch1, branch2)

    reports = {}
    for name in comparison.prompts_modified:
        diff = comparison.diff_results[name]
        html = HTMLDiff.render(diff, title=f"Diff: {name}")
        reports[name] = html

    return reports
```

### Automated Merge Pipeline

```python
def auto_merge_pipeline(promptly, source, target):
    """Automated merge with conflict reporting"""
    merge_tool = MergeTool(promptly)

    # Try automatic merge
    results = merge_tool.merge_branches(source, target, MergeStrategy.AUTO)

    # Separate successful and conflicted
    successful = {k: v for k, v in results.items() if v.success}
    conflicted = {k: v for k, v in results.items() if not v.success}

    # Generate conflict report
    if conflicted:
        report = generate_conflict_report(conflicted)
        notify_team(report)

    return {
        'successful': len(successful),
        'conflicted': len(conflicted),
        'details': results
    }
```

### Statistical Analysis

```python
def analyze_prompt_evolution(promptly, prompt_name):
    """Analyze how a prompt evolved over time"""
    engine = ComparisonEngine(promptly)

    # Get all versions
    current = promptly.get(prompt_name)
    max_version = current['version']

    stats = []
    for v in range(1, max_version):
        comparison = engine.compare_versions(prompt_name, v, v+1)
        stats.append({
            'version': v+1,
            'additions': comparison.diff_result.stats.additions,
            'deletions': comparison.diff_result.stats.deletions,
            'similarity': comparison.diff_result.stats.similarity,
            'time_delta': comparison.time_delta
        })

    return stats
```

---

## Best Practices

### 1. Choose Appropriate Diff Level

- **Line level**: Default, best for most cases
- **Word level**: Better for prose and prompt text
- **Character level**: Use for precise character changes
- **Semantic level**: Use when meaning matters more than exact text

```python
# For code-like prompts
diff = engine.diff(old, new, DiffLevel.LINE)

# For natural language prompts
diff = engine.diff(old, new, DiffLevel.WORD)

# For understanding intent changes
diff = engine.diff(old, new, DiffLevel.SEMANTIC)
```

### 2. Use Appropriate Merge Strategy

```python
# Development workflow
MergeStrategy.AUTO      # Try automatic resolution first

# Production deployment
MergeStrategy.MANUAL    # Review all conflicts manually

# Rollback scenarios
MergeStrategy.OURS      # Keep current version

# Accepting upstream changes
MergeStrategy.THEIRS    # Accept incoming changes
```

### 3. Regular Branch Comparisons

```python
# Before merging, always compare
comparison = engine.compare_branches("feature", "main")

if len(comparison.prompts_modified) > 10:
    print("Warning: Large merge, review carefully")

if comparison.prompts_only_in_new:
    print(f"New prompts: {comparison.prompts_only_in_new}")
```

### 4. Document Conflict Resolutions

```python
# Add metadata when resolving conflicts
conflict.metadata = {
    'resolved_by': 'user@example.com',
    'resolved_at': datetime.now().isoformat(),
    'strategy': 'manual',
    'reason': 'Combined best of both versions'
}
```

### 5. Use HTML Reports for Reviews

```python
# Generate comprehensive review document
html_parts = []

# Summary
comparison = engine.compare_branches("feature", "main")
html_parts.append(TerminalDiff.render_branch_comparison(comparison))

# Detailed diffs
for name in comparison.prompts_modified:
    diff = comparison.diff_results[name]
    html_parts.append(HTMLDiff.render(diff, title=name))

# Save for team review
with open("merge_review.html", "w") as f:
    f.write("\n".join(html_parts))
```

### 6. Automate Regression Detection

```python
def detect_regressions(promptly, old_version, new_version):
    """Detect if new version is worse than old"""
    old_eval = promptly.eval_prompt(name, test_cases)
    new_eval = promptly.eval_prompt(name, test_cases)

    comparison = engine.compare_evaluations(name, old_eval, new_eval)

    if comparison.score_delta < -0.1:  # 10% regression
        raise RegressionError(f"Score dropped: {comparison.score_delta}")

    if len(comparison.regressed_tests) > 3:
        raise RegressionError(f"{len(comparison.regressed_tests)} tests regressed")
```

---

## Troubleshooting

### Large Diffs Taking Too Long

```python
# Use line-level instead of character-level
result = engine.diff(old, new, DiffLevel.LINE)  # Faster

# For very large texts, use quick_diff with stats only
from promptly.diff import diff_stats_only
stats = diff_stats_only(old, new)  # Just statistics
```

### Merge Conflicts Not Resolving

```python
# Check conflict details
for conflict in result.conflicts:
    print(conflict.to_marker_format())

# Try different strategy
result = ThreeWayMerge.merge(base, ours, theirs, MergeStrategy.UNION)

# Or resolve manually
for conflict in result.conflicts:
    conflict.merged_content = custom_merge_logic(conflict)
    conflict.resolved = True
```

### HTML Output Too Large

```python
# Limit context in HTML
# Modify CSS to hide context or truncate large diffs

# Or generate summary only
summary_html = f"""
<html>
<body>
    <h1>Diff Summary</h1>
    <p>Additions: {result.stats.additions}</p>
    <p>Deletions: {result.stats.deletions}</p>
    <p>Similarity: {result.stats.similarity:.1%}</p>
</body>
</html>
"""
```

---

## Examples

### Example 1: Prompt Refinement Workflow

```python
# Initial prompt
promptly.add("summarizer", "Summarize the text: {text}")

# Experiment with improvements
promptly.branch("experiment", "main")
promptly.checkout("experiment")
promptly.add("summarizer", "Provide a concise summary of the following text, highlighting key points: {text}")

# Compare
engine = ComparisonEngine(promptly)
diff = engine.compare_versions("summarizer", 1, 2, DiffLevel.SEMANTIC)

# Review changes
print(TerminalDiff.render(diff.diff_result))

# If good, merge
promptly.checkout("main")
merge_tool = MergeTool(promptly)
merge_tool.merge_branches("experiment", "main", MergeStrategy.AUTO)
```

### Example 2: Team Collaboration

```python
# Alice creates feature
promptly.branch("alice-feature", "main")
promptly.add("greeter", "Hello {name}!")

# Bob creates feature
promptly.branch("bob-feature", "main")
promptly.add("greeter", "Hi {name}, welcome!")

# Compare approaches
comparison = engine.compare_branches("alice-feature", "bob-feature")
print(comparison.diff_results["greeter"])

# Merge best of both
result = merge_tool.merge_prompts("greeter", 1, 2, 3, MergeStrategy.MANUAL)

# Interactive resolution
interactive = InteractiveMerge(result)
conflict = interactive.next_conflict()
interactive.resolve_current("Hey {name}, welcome!")
```

### Example 3: Quality Assurance

```python
# Run evaluations on both versions
v1_results = promptly.eval_prompt("my_prompt", test_cases)

promptly.add("my_prompt", "Improved version...")
v2_results = promptly.eval_prompt("my_prompt", test_cases)

# Compare performance
eval_comparison = engine.compare_evaluations(
    "my_prompt",
    v1_results,
    v2_results
)

# Make decision
if eval_comparison.score_delta > 0.1:
    print("✓ New version is significantly better!")
elif eval_comparison.score_delta < -0.05:
    print("✗ New version regressed, reverting...")
    # Revert to v1
else:
    print("~ Marginal change, needs more testing")
```

---

## API Reference

See individual module documentation:

- [diff/engine.py](diff/engine.py) - Core diff algorithms
- [diff/compare.py](diff/compare.py) - Comparison tools
- [diff/visual.py](diff/visual.py) - Visual rendering
- [merge/tool.py](merge/tool.py) - Merge functionality

---

## Contributing

To add new diff or merge features:

1. Implement in appropriate module (`diff/` or `merge/`)
2. Add tests to `test_diff_merge.py`
3. Update this guide with examples
4. Add CLI commands to `promptly.py`

---

## License

Same as Promptly main project.
