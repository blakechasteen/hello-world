# Progress Bars - Quick Reference Guide

**Date**: 2025-11-16
**Status**: ✅ Implementation Complete

---

## Quick Start

### Basic Counted Loop Progress
```python
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console

console = Console()

items = ["item1", "item2", "item3"]

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    console=console,
    transient=True
) as progress:
    task = progress.add_task("[cyan]Processing items[/cyan]", total=len(items))

    for item in items:
        progress.update(task, description=f"[cyan]Item: {item}[/cyan]")
        # Do work...
        progress.advance(task)
```

### Expected Output
```
⠙ Item: item3 ━━━━━━━━━━━━━━━━━━━━━━━━ 100% 3/3 0:00:01 < 0:00:01
```

---

## Files with Progress Bars

| File | Purpose | Progress Bars |
|------|---------|---------------|
| `demos/demo_dashboard.py` | Dashboard demo with async operations | 2 bars (queries + skills) |
| `demos/demo_rag_dashboard.py` | RAG performance analysis demo | 4 bars (generation + build + render + save) |

---

## Common Patterns

### Pattern: Loop with Counter
```python
with Progress(...) as progress:
    task = progress.add_task("Description", total=len(items))
    for i, item in enumerate(items, 1):
        progress.update(task, description=f"[cyan]{i}/{len(items)}: {item}[/cyan]")
        # Work...
        progress.advance(task)
```

### Pattern: Single Operation (Indeterminate)
```python
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
    transient=True
) as progress:
    task = progress.add_task("[cyan]Loading...[/cyan]")
    # Do work...
    progress.update(task, completed=100)
```

### Pattern: Timed Loop
```python
import time

with Progress(...) as progress:
    task = progress.add_task("Processing", total=len(items))
    for item in items:
        start = time.time()
        # Do work...
        elapsed = time.time() - start
        progress.update(task, description=f"[cyan]{item} ({elapsed:.2f}s)[/cyan]")
        progress.advance(task)
```

---

## Demo Execution

### Run demo_dashboard.py
```bash
cd /home/user/hello-world
PYTHONPATH=. python demos/demo_dashboard.py
```

**Output**: Progress bars for 6 queries + 3 skills

### Run demo_rag_dashboard.py
```bash
cd /home/user/hello-world
PYTHONPATH=. python demos/demo_rag_dashboard.py
```

**Output**: Progress bars for query generation, building, rendering, and saving

---

## Customization

### Change Progress Bar Colors
```python
# Cyan description (default)
description=f"[cyan]Processing {item}[/cyan]"

# Green description
description=f"[green]Processing {item}[/green]"

# Yellow description (warning)
description=f"[yellow]Processing {item}[/yellow]"
```

### Add More Columns
```python
from rich.progress import TimeElapsedColumn, TimeRemainingColumn

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),    # Shows elapsed time
    TimeRemainingColumn(),   # Shows estimated remaining
    console=console,
    transient=True
) as progress:
    # ...
```

### Remove Spinner (Bar Only)
```python
with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    console=console,
    transient=True
) as progress:
    # ...
```

### Keep Progress Bar After Completion
```python
# Remove transient=True to keep the bar visible
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    console=console
    # transient=False (default)
) as progress:
    # ...
```

---

## Column Reference

| Column | Description | Example |
|--------|-------------|---------|
| `SpinnerColumn()` | Animated spinner | ⠙ |
| `TextColumn()` | Dynamic text | "Processing item 5/10" |
| `BarColumn()` | Progress bar | ━━━━━━━━ |
| `TaskProgressColumn()` | X/Y counter | "5/10" |
| `PercentageColumn()` | Percentage | "50%" |
| `TimeElapsedColumn()` | Time spent | "0:00:05" |
| `TimeRemainingColumn()` | Time left | "< 0:00:03" |
| `DownloadColumn()` | Download size | "1.5MB / 10MB" |
| `BarColumn()` | Custom bar width | ━━━━━ (50%) |

---

## Tips & Tricks

1. **Use `transient=True` for clean output**
   - Progress disappears after task completion
   - Prevents cluttering terminal with completed bars

2. **Update description dynamically**
   - Show current item being processed
   - Users see real-time progress of what's happening

3. **Integrate with existing Console**
   - Use `console=console` parameter
   - Matches existing Rich styling in your app

4. **Time estimates**
   - `TimeRemainingColumn()` automatically calculates ETA
   - Helps users plan for long operations

5. **Combine with async/await**
   - Works great with `async for` loops
   - Useful for dashboard demos with async orchestrators

---

## Troubleshooting

### Progress Bar Not Showing
**Check**:
- Is terminal width > 40 characters?
- Is output redirected to file/pipe? (Progress bars only in interactive terminal)
- Are you using `transient=True`? (Bar disappears after completion)

### Flickering or Jumpy Updates
**Solution**:
- Only call `progress.update()` and `progress.advance()` when needed
- Avoid rapid updates (>100 per second)

### Progress Stuck at 0%
**Check**:
- Did you call `progress.advance(task)`?
- Is `total` parameter set correctly?

### Unicode Characters Not Showing
**Solution**:
- Use `console=Console(force_terminal=True)` for proper Unicode
- Windows console may need special handling

---

## Integration with HoloLoom Demos

### Adding Progress Bar to Your Demo

1. **Import Progress Components**
```python
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)
from rich.console import Console
console = Console()
```

2. **Wrap Your Loop**
```python
with Progress(..., console=console, transient=True) as progress:
    task = progress.add_task("Description", total=len(items))
    for item in items:
        progress.update(task, description=f"[cyan]{item}[/cyan]")
        # Do work...
        progress.advance(task)
```

3. **Run Demo**
```bash
PYTHONPATH=. python demos/demo_your_demo.py
```

---

## Performance

- **Overhead per update**: <1ms
- **Memory cost**: Negligible (<1KB per task)
- **Thread-safe**: Yes, Progress is thread-safe
- **Async-compatible**: Yes, works with async/await

---

## Further Reading

- [Rich Progress Documentation](https://rich.readthedocs.io/en/latest/progress.html)
- [Rich Console Markup](https://rich.readthedocs.io/en/latest/markup.html)
- [Example Implementations](../../demos/demo_dashboard.py)

