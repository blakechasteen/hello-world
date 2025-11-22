# Trough Quick Start Guide

**Version**: 1.0.0
**Time to Complete**: 5-10 minutes
**Status**: Production Ready

Get started detecting AI code quality issues in 5 minutes.

---

## Installation

### Prerequisites

- Python 3.8+
- No external dependencies (pure stdlib)

### Setup

```bash
# Clone the repository (if not already done)
git clone https://github.com/mythRL/trough.git
cd trough

# Optional: Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

---

## Your First Analysis (2 minutes)

### Step 1: Import and Create Detector

```python
import asyncio
from trough import AISlopDetector, Language

# Create detector (works immediately, no models to download)
detector = AISlopDetector()
```

### Step 2: Analyze Code

```python
# Example code with issues
code = '''
import os
password = "admin123"  # Hardcoded secret!

def process_data(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)  # Inefficient

    return result / 0  # Division by zero!
'''

# Analyze
async def main():
    issues = await detector.detect_all(code, Language.PYTHON, "app.py")

    # Print results
    for issue in issues:
        print(f"Line {issue.line_number}: {issue.category.value}")
        print(f"  Message: {issue.message}")
        print(f"  Fix: {issue.suggestion}\n")

asyncio.run(main())
```

### Step 3: Run It

```bash
python your_script.py
```

**Output**:
```
Line 3: hardcoded_values
  Message: Hardcoded password found
  Fix: password = os.getenv("PASSWORD")

Line 6: performance
  Message: Inefficient list operation
  Fix: Use list comprehension: [x * 2 for x in data]

Line 9: security
  Message: Potential division by zero
  Fix: Add validation: if denominator != 0
```

---

## Next Steps (3-5 minutes)

### Analyze Your Own Code

```python
from pathlib import Path

# Read your file
code = Path("your_file.py").read_text()

# Analyze it
issues = await detector.detect_all(code, Language.PYTHON, "your_file.py")

# Print summary
critical = [i for i in issues if i.severity.value == "CRITICAL"]
print(f"Found {len(critical)} critical issues")
```

### Filter by Category

```python
from trough import SlopCategory

# Only check security issues
security_issues = await detector.detect_category(
    code,
    Language.PYTHON,
    SlopCategory.SECURITY,
    "app.py"
)

for issue in security_issues:
    print(f"🔒 {issue.message}")
```

### Generate HTML Report

```python
from trough.report_generator import ReportGenerator

# Get issues
slop_issues = await detector.detect_all(code, Language.PYTHON, "app.py")

# Generate report
generator = ReportGenerator()
html = generator.generate_html_report(
    issues=slop_issues,
    errors=[],  # Logic errors if available
    file_path="app.py",
    title="Code Quality Report"
)

# Save report
with open("report.html", "w") as f:
    f.write(html)

print("✅ Report saved to report.html - open in browser")
```

---

## Common Tasks

### Task 1: Check Multiple Files

```python
from pathlib import Path

async def check_all_python_files(directory: str):
    detector = AISlopDetector()

    for file_path in Path(directory).glob("**/*.py"):
        code = file_path.read_text()
        issues = await detector.detect_all(code, Language.PYTHON, str(file_path))

        if issues:
            print(f"\n{file_path}")
            print(f"  Issues: {len(issues)}")

asyncio.run(check_all_python_files("src/"))
```

### Task 2: Only Show Critical Issues

```python
from trough import Severity

critical_only = [i for i in issues if i.severity == Severity.CRITICAL]

for issue in critical_only:
    print(f"🔴 Line {issue.line_number}: {issue.message}")
```

### Task 3: Export to JSON

```python
import json

results = {
    "file": "app.py",
    "total_issues": len(issues),
    "issues": [
        {
            "line": i.line_number,
            "category": i.category.value,
            "severity": i.severity.value,
            "message": i.message,
            "suggestion": i.suggestion
        }
        for i in issues
    ]
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Task 4: Integrate with Git Pre-Commit

```bash
# Create .git/hooks/pre-commit
#!/bin/bash
python -c "
import asyncio
from trough import AISlopDetector, Severity, Language
from pathlib import Path

async def check():
    detector = AISlopDetector()
    files = Path('.').glob('**/*.py')

    for f in files:
        code = f.read_text()
        issues = await detector.detect_all(code, Language.PYTHON, str(f))
        critical = [i for i in issues if i.severity == Severity.CRITICAL]

        if critical:
            print(f'{f}: {len(critical)} critical issues - commit blocked')
            return 1

    print('✅ All checks passed')
    return 0

exit(asyncio.run(check()))
"

# Make executable
chmod +x .git/hooks/pre-commit
```

---

## Understanding Results

### Severity Levels

| Level | Meaning | Action |
|-------|---------|--------|
| **CRITICAL** | Security/data loss risk | Fix immediately |
| **HIGH** | Logic error/resource leak | Fix before shipping |
| **MEDIUM** | Performance/code smell | Fix when possible |
| **LOW** | Style/minor improvement | Nice to have |
| **INFO** | Suggestion/best practice | Consider for future |

### Issue Categories

- **Security** - SQL injection, XSS, command injection
- **Hardcoded Values** - Secrets, API keys, passwords
- **Resource Leak** - Unclosed files/connections
- **Error Handling** - Missing try/except blocks
- **Type Mismatch** - Wrong type usage
- **Performance** - N+1 queries, inefficient code
- **Dead Code** - Unused imports/variables
- **Logic Errors** - Off-by-one, contradictions, etc.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'trough'"

```bash
# Make sure you're in the right directory
cd /path/to/trough

# Or install in development mode
pip install -e .
```

### Analysis is slow

```python
# Use timeout for very large files
import asyncio

issues = asyncio.wait_for(
    detector.detect_all(code, Language.PYTHON, "file.py"),
    timeout=30.0  # 30 second timeout
)
```

### "UnicodeDecodeError" when reading files

```python
# Specify encoding
code = Path("file.py").read_text(encoding="utf-8")
# Try different encoding if utf-8 fails
code = Path("file.py").read_text(encoding="iso-8859-1")
```

---

## Next Resources

- **[API Reference](./API_REFERENCE.md)** - Complete API documentation
- **[Examples](../examples/trough/)** - More code examples
- **[Integration Guide](../HOLOLOOM_INTEGRATION.md)** - Use with HoloLoom
- **[Troubleshooting](../TROUBLESHOOTING.md)** - Common issues and solutions

---

## Tips & Tricks

### Tip 1: Use Async for Batch Processing

```python
import asyncio

async def analyze_files(files: List[str]):
    detector = AISlopDetector()

    # Run all analyses in parallel
    tasks = [
        detector.detect_all(Path(f).read_text(), Language.PYTHON, f)
        for f in files
    ]

    results = await asyncio.gather(*tasks)
    return results

# Much faster than sequential analysis
issues_per_file = asyncio.run(analyze_files(["a.py", "b.py", "c.py"]))
```

### Tip 2: Cache Results

```python
# For repeated analysis of same file
results_cache = {}

async def analyze_cached(file_path: str):
    if file_path in results_cache:
        return results_cache[file_path]

    code = Path(file_path).read_text()
    results = await detector.detect_all(code, Language.PYTHON, file_path)
    results_cache[file_path] = results

    return results
```

### Tip 3: Create Custom Severity Filter

```python
def show_important_issues(issues, max_severity_level=2):
    """Show only important issues (0=critical, 4=info)"""

    severity_order = ["critical", "high", "medium", "low", "info"]

    for issue in issues:
        level = severity_order.index(issue.severity.value)
        if level <= max_severity_level:
            print(f"Line {issue.line_number}: {issue.message}")

# Show only CRITICAL and HIGH
show_important_issues(issues, max_severity_level=1)
```

---

## Success Metrics

After implementing Trough, you should see:

- ✅ **Reduced security issues** (hardcoded secrets, injections)
- ✅ **Fewer runtime errors** (null dereference, division by zero)
- ✅ **Better code quality** (resource leaks, error handling)
- ✅ **Faster code review** (automated issue detection)
- ✅ **Confidence in AI-generated code** (verified before merge)

---

## Support

- **Issues**: Open an issue on GitHub
- **Questions**: Check the [FAQ](../TROUBLESHOOTING.md#faq)
- **Examples**: See [examples/trough/](../examples/trough/)

---

## What's Next?

After mastering Trough, explore:

1. **[BossPig Quick Start](../bosspig/QUICK_START.md)** - Analyze business writing
2. **[HoloLoom Integration](../HOLOLOOM_INTEGRATION.md)** - Use as HoloLoom department
3. **[Workflow Examples](../examples/integration/)** - Multi-step analysis pipelines
4. **[CI/CD Integration](../CICD_INTEGRATION.md)** - Automate quality checks

---

**You're ready to start!** 🚀

Next: Analyze your first file - see [Your First Analysis](#your-first-analysis-2-minutes)
