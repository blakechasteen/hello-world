# BossPig Quick Start Guide

**Version**: 0.1.0
**Time to Complete**: 5-10 minutes
**Status**: Production Ready

Improve your business writing quality in 5 minutes.

---

## Installation

### Prerequisites

- Python 3.8+
- spaCy (optional, for advanced passive voice detection)

### Setup

```bash
# Clone or navigate to bosspig directory
cd bosspig

# Optional: Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Optional: Install spaCy for better NLP
pip install spacy
python -m spacy download en_core_web_sm
```

---

## Your First Analysis (2 minutes)

### Step 1: Import and Create Detector

```python
from bosspig import BossPigDetector

# Create detector (works immediately, no training needed)
detector = BossPigDetector()
```

### Step 2: Analyze Text

```python
# Example business text with issues
proposal = """
Subject: Q4 Planning Initiative

We need to leverage our core competencies to synergize efforts and drive ROI.
We should circle back on this ASAP to ensure alignment with stakeholders.
The team will work on enhancing performance metrics in the coming weeks.

Best regards,
Management
"""

# Analyze
results = detector.analyze(proposal)

# Print results
print(f"Quality Score: {results.metrics.quality_score:.0f}/100\n")

for finding in results.findings:
    print(f"Line {finding.line_number}: {finding.category.value}")
    print(f"  Issue: {finding.message}")
    print(f"  Better: {finding.suggestion}\n")
```

### Step 3: Run It

```bash
python your_script.py
```

**Output**:
```
Quality Score: 58/100

Line 1: corporate_jargon
  Issue: Corporate jargon detected: 'leverage synergies'
  Better: Replace with specific action: 'combine our strengths'

Line 1: corporate_jargon
  Issue: Corporate jargon detected: 'synergize'
  Better: Use a specific verb like 'coordinate' or 'align'

Line 2: vague_commitments
  Issue: Vague commitment: 'ASAP'
  Better: Replace with specific date: 'by Friday EOD' or 'December 15'

Line 3: missing_dates
  Issue: Action item without specific deadline
  Better: Specify deadline: 'The team will complete this by [DATE]'

Line 3: passive_voice
  Issue: Passive voice - unclear responsibility
  Better: Use active voice: 'We will enhance performance metrics by Q1'
```

---

## Next Steps (3-5 minutes)

### Analyze Your Own Document

```python
from pathlib import Path

# Read your file
text = Path("proposal.txt").read_text()

# Analyze
results = detector.analyze(text)

# Show summary
print(f"Quality Score: {results.metrics.quality_score:.0f}/100")
print(f"Critical Issues: {results.metrics.critical_count}")
print(f"Total Issues: {results.metrics.total_issues}")
```

### Check Specific Category

```python
from bosspig import FindingCategory

# Get jargon findings only
jargon_issues = results.findings_by_category(FindingCategory.CORPORATE_JARGON)

for issue in jargon_issues:
    print(f"Line {issue.line_number}: {issue.text}")
    print(f"  → {issue.suggestion}\n")
```

### Get Improvement Suggestions

```python
# Filter findings by severity
from bosspig import Severity

critical = results.findings_by_severity(Severity.CRITICAL)

print(f"Must fix ({len(critical)} issues):")
for issue in critical:
    print(f"  • {issue.message}")
```

---

## Common Tasks

### Task 1: Before Sending Email

```python
def is_email_ready(text: str) -> bool:
    """Check if email is ready to send"""

    detector = BossPigDetector()
    results = detector.analyze(text)

    # Email is ready if:
    # - Score >= 80
    # - No critical issues
    return (results.metrics.quality_score >= 80 and
            results.metrics.critical_count == 0)

email = "We will leverage synergies ASAP..."

if is_email_ready(email):
    print("✅ Ready to send")
else:
    print("⚠️ Review before sending")
```

### Task 2: Batch Check Multiple Documents

```python
from pathlib import Path

def check_all_documents(directory: str):
    detector = BossPigDetector()

    for doc_path in Path(directory).glob("*.txt"):
        results = detector.analyze(doc_path)

        status = "✅" if results.metrics.quality_score >= 80 else "⚠️"
        print(f"{status} {doc_path.name}: {results.metrics.quality_score:.0f}/100")

check_all_documents("documents/")
```

### Task 3: Export Findings for Team Review

```python
import json

# Analyze
results = detector.analyze(text)

# Export to JSON
findings_dict = [f.to_dict() for f in results.findings]

with open("findings.json", "w") as f:
    json.dump(findings_dict, f, indent=2)

print(f"Exported {len(findings_dict)} findings to findings.json")
```

### Task 4: Track Quality Improvements

```python
def track_improvements(original_text: str, revised_text: str):
    detector = BossPigDetector()

    results_original = detector.analyze(original_text)
    results_revised = detector.analyze(revised_text)

    score_improvement = results_revised.metrics.quality_score - results_original.metrics.quality_score
    issues_removed = results_original.metrics.total_issues - results_revised.metrics.total_issues

    print(f"Score improved: {score_improvement:+.0f} points")
    print(f"Issues removed: {issues_removed} found {results_revised.metrics.total_issues} remaining")

    if score_improvement > 10:
        print("✅ Great improvement!")
    elif score_improvement > 0:
        print("✅ Good progress")
    else:
        print("⚠️ Score decreased")

# Usage
track_improvements(original, revised)
```

---

## Understanding Results

### Quality Score Interpretation

| Score | Status | Action |
|-------|--------|--------|
| **90-100** | Excellent | Ready to publish |
| **80-89** | Good | Minor fixes recommended |
| **70-79** | Fair | Review and revise |
| **60-69** | Poor | Significant revision needed |
| **0-59** | Critical | Rewrite recommended |

### Finding Categories

- **CORPORATE_JARGON** - Meaningless buzzwords
  - Examples: "leverage", "synergize", "circle back", "low-hanging fruit"
  - Fix: Replace with specific, clear language

- **VAGUE_COMMITMENTS** - Unclear promises
  - Examples: "ASAP", "soon", "improve", "enhance"
  - Fix: Add specific deadlines or metrics

- **MISSING_DATES** - No timeline specified
  - Examples: Action items without deadlines
  - Fix: Add specific date or deadline

- **AI_HALLUCINATION** - Signs of AI-generated text
  - Examples: "Furthermore", "In conclusion", unusual patterns
  - Fix: Rewrite in your own voice

- **PASSIVE_VOICE** - Unclear responsibility
  - Examples: "It was decided", "Will be reviewed by"
  - Fix: Use active voice with clear subject

### Severity Levels

| Level | Meaning |
|-------|---------|
| **CRITICAL** | Major issues affecting clarity and professionalism |
| **WARNING** | Issues that should be fixed before sending |
| **INFO** | Minor suggestions for improvement |

---

## Quick Wins (Immediate Improvements)

### Win 1: Replace Jargon (5 point boost)

```python
# Bad
"We need to leverage our synergies"

# Good
"We need to combine our strengths"
```

### Win 2: Add Specific Dates (10 point boost)

```python
# Bad
"We'll do this ASAP"

# Good
"We'll complete this by December 15"
```

### Win 3: Use Active Voice (8 point boost)

```python
# Bad
"It was decided that we should improve performance"

# Good
"We will improve performance"
```

### Win 4: Remove Buzzwords (15 point boost)

```python
# Bad
"We will circle back and reach out to stakeholders"

# Good
"We will contact stakeholders next week"
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'bosspig'"

```bash
# Make sure you're in the right directory
cd /path/to/bosspig

# Or install in development mode
pip install -e .
```

### "spaCy not available" warning

```bash
# Install spaCy for better NLP
pip install spacy
python -m spacy download en_core_web_sm
```

You can still use BossPig without spaCy - it uses regex fallback (slightly less accurate).

### Analysis of very long documents is slow

```python
# Split into sections
from pathlib import Path

text = Path("long_document.txt").read_text()
sections = text.split("\n\n")

detector = BossPigDetector()
all_findings = []

for i, section in enumerate(sections):
    results = detector.analyze(section)
    for finding in results.findings:
        # Adjust line numbers for original document
        finding.line_number += i * 10
        all_findings.append(finding)
```

---

## Real-World Example

### Before: Poor Business Proposal

```
Subject: Q4 Initiative

We need to leverage our core competencies to synergize efforts
and improve ROI. We should circle back on this to ensure alignment
with stakeholders. The team will enhance performance metrics.

Thanks,
John
```

**Score: 52/100** ❌

### After: Improved Proposal

```
Subject: Q4 Growth Initiative

We are combining our three departments' strengths to accelerate
product development. We will meet on December 15 to finalize our
approach with key stakeholders. Our goal is to increase revenue by
15% by end of Q1.

Thanks,
John
```

**Score: 87/100** ✅

**Improvements Made**:
1. Removed jargon: "leverage synergies" → "combining strengths"
2. Added specific date: "ASAP" → "December 15"
3. Added measurable goal: "improve ROI" → "15% revenue increase by Q1"
4. Used active voice: "team will enhance" → "we will increase"

---

## Next Resources

- **[API Reference](./API_REFERENCE.md)** - Complete API documentation
- **[Examples](../examples/bosspig/)** - More code examples
- **[Integration Guide](../HOLOLOOM_INTEGRATION.md)** - Use with HoloLoom
- **[Troubleshooting](../TROUBLESHOOTING.md)** - Common issues and solutions

---

## Tips & Tricks

### Tip 1: Custom Jargon Dictionary

```python
from pathlib import Path

# Create custom_jargon.json with your industry-specific terms
detector = BossPigDetector(
    jargon_dict_path=Path("custom_jargon.json")
)
```

### Tip 2: Pre-Check Before Publishing

```python
def quality_gate(text: str, min_score: int = 80):
    detector = BossPigDetector()
    results = detector.analyze(text)

    if results.metrics.quality_score >= min_score:
        return True, f"✅ Ready ({results.metrics.quality_score:.0f}/100)"
    else:
        return False, f"⚠️ {results.metrics.quality_score:.0f}/100 - needs fixes"

approved, message = quality_gate(email_draft)
print(message)
```

### Tip 3: Automated Improvements

```python
# Get all suggestions and apply them
results = detector.analyze(text)

improved = text
for finding in sorted(results.findings,
                      key=lambda f: f.column_number or 0,
                      reverse=True):
    if finding.text in improved:
        improved = improved.replace(
            finding.text,
            finding.suggestion,
            1
        )

print("BEFORE:")
print(text)
print("\nAFTER:")
print(improved)
```

---

## Success Metrics

After using BossPig, you should see:

- ✅ **Clearer communication** (jargon removed)
- ✅ **Better accountability** (clear owners and deadlines)
- ✅ **Faster decision-making** (specific commitments)
- ✅ **Improved credibility** (professional writing)
- ✅ **Less misunderstanding** (active voice, clarity)

---

## Support

- **Issues**: Open an issue on GitHub
- **Questions**: Check the [FAQ](../TROUBLESHOOTING.md#faq)
- **Examples**: See [examples/bosspig/](../examples/bosspig/)

---

## What's Next?

After mastering BossPig, explore:

1. **[Trough Quick Start](../trough/QUICK_START.md)** - Analyze code quality
2. **[HoloLoom Integration](../HOLOLOOM_INTEGRATION.md)** - Use as HoloLoom department
3. **[Workflow Examples](../examples/integration/)** - Combined analysis pipelines
4. **[CI/CD Integration](../CICD_INTEGRATION.md)** - Automate document checks

---

**You're ready to start!** 🚀

Next: Analyze your first document - see [Your First Analysis](#your-first-analysis-2-minutes)
