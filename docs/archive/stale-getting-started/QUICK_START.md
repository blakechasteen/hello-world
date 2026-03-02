# BossPig Quick Start Guide

**Get started with BossPig in 5 minutes!**

## What is BossPig?

BossPig is a business writing quality analyzer that detects and fixes common issues in business documents:

- 🎯 **15 detection categories** - from corporate jargon to governance violations
- ⚡ **Fast** - analyzes 1000-word documents in <30ms
- 🔧 **Auto-fix** - automatically fixes many issues
- 📊 **Quality scoring** - get instant feedback on document quality

---

## Installation

### 1. Install BossPig

```bash
pip install bosspig
```

### 2. Verify Installation

```bash
python -c "from bosspig import BossPigDetector; print('BossPig installed successfully!')"
```

---

## Your First Analysis (30 seconds)

### 1. Create a test document

Save this as `test.txt`:

```
# Q4 Strategic Initiative

We need to synergize our core competencies to leverage the low-hanging fruit
and move the needle on our key performance indicators.

Action Items:
- Circle back on the deep dive
- Touch base with stakeholders
- We will try to finish this soon

As an AI language model, I cannot provide specific recommendations.
```

### 2. Analyze it

```python
from bosspig.detector import BossPigDetector

# Create detector
detector = BossPigDetector()

# Analyze document
results = detector.analyze("test.txt")

# Print summary
print(results.summary())
```

### 3. See results

```
BossPig Analysis Report
=======================
Overall Quality Score: D (45.2%)

📊 Findings Summary:
- Corporate Jargon: 4 issues
- Vague Commitments: 1 issue
- AI Hallucination Markers: 1 issue

🎯 Top Issues:
1. Line 3: Corporate jargon: 'synergize our core competencies'
   Suggestion: Replace with 'combine our strengths'

2. Line 11: Vague commitment: 'we will try to finish this soon'
   Suggestion: Replace with specific action, owner, and date

3. Line 13: AI hallucination marker: 'As an AI language model'
   Suggestion: Remove LLM boilerplate entirely
```

---

## Common Use Cases

### 1. Analyze Text String

```python
from bosspig.detector import BossPigDetector

detector = BossPigDetector()
text = "We will try to synergize our efforts soon."

results = detector.analyze(text)
print(f"Quality Score: {results.quality_metrics.overall_score:.1%}")
print(f"Findings: {len(results.findings)}")
```

### 2. Analyze File

```python
from pathlib import Path

detector = BossPigDetector()
results = detector.analyze(Path("document.md"))

# Print findings by category
for finding in results.findings:
    print(f"{finding.category.value}: {finding.text}")
    print(f"  → {finding.suggestion}")
```

### 3. Filter by Severity

```python
# Get only CRITICAL issues
critical_findings = [
    f for f in results.findings
    if f.severity.value == "critical"
]

print(f"Critical issues: {len(critical_findings)}")
```

### 4. Check Specific Categories

```python
# Brand Guidelines Check
from bosspig.detector import BossPigDetector

detector = BossPigDetector(
    brand_config_path=Path("config/brand_config.json")
)

results = detector.analyze("marketing_copy.txt")

# Get brand violations
brand_issues = [
    f for f in results.findings
    if "brand" in f.category.value
]
```

### 5. Governance Validation

```python
# Healthcare Document Validation
detector = BossPigDetector(
    document_type="healthcare",
    governance_config_path=Path("config/governance_config.json")
)

results = detector.analyze("hipaa_policy.md")

# Check compliance score
from bosspig.detector.governance import calculate_governance_score

metrics = calculate_governance_score(results.findings)
print(f"Governance Score: {metrics.governance_score:.1%}")
print(f"Compliance Violations: {metrics.compliance_violations}")
```

---

## Configuration Options

### Document Types

Choose the appropriate document type for governance validation:

```python
detector = BossPigDetector(document_type="healthcare")        # HIPAA compliance
detector = BossPigDetector(document_type="data_policies")     # SOC2, GDPR
detector = BossPigDetector(document_type="technical_documentation")  # Default
```

### Enable NLP (Passive Voice Detection)

```python
# Requires spaCy: pip install spacy
# python -m spacy download en_core_web_sm

detector = BossPigDetector(enable_nlp=True)
```

### Custom Configuration Files

```python
from pathlib import Path

detector = BossPigDetector(
    jargon_dict_path=Path("custom_jargon.json"),
    brand_config_path=Path("custom_brand.json"),
    governance_config_path=Path("custom_governance.json")
)
```

---

## Understanding Results

### Quality Metrics

BossPig calculates 5 quality scores (0.0 - 1.0):

- **Clarity Score**: Low jargon, clear language
- **Specificity Score**: Concrete, measurable statements
- **Actionability Score**: Clear ownership, active voice
- **Professionalism Score**: No AI boilerplate or placeholders
- **Completeness Score**: All required sections present

**Overall Score** = Average of all 5 scores

### Quality Grades

- **A (90-100%)**: Excellent - publication ready
- **B (80-89%)**: Good - minor improvements needed
- **C (70-79%)**: Fair - notable issues to address
- **D (60-69%)**: Poor - significant rewrite needed
- **F (<60%)**: Failing - major problems

### Finding Categories

**Content Issues** (5 categories):
1. Corporate Jargon - buzzwords, filler language
2. Vague Commitments - unclear ownership
3. Missing Dates - no specific timelines
4. AI Hallucination Markers - LLM boilerplate
5. Passive Voice - unclear responsibility

**Specificity** (3 categories):
6. Vague Quantifiers - "several", "many"
7. Unmeasurable Claims - "world-class" without metrics
8. Relative Statements - "faster" without baseline

**Brand Compliance** (4 categories):
9. Brand Capitalization - proper brand names
10. Prohibited Terms - banned words/phrases
11. Non-Preferred Terminology - deprecated terms
12. Tone Violations - inappropriate tone

**Governance** (5 categories):
13. Missing Required Sections
14. Missing Required Disclaimers
15. Missing Approval Metadata
16. Missing Version Control
17. Compliance Framework Violations

---

## Production Usage

### Logging

```python
from bosspig.logging_config import setup_logging
from pathlib import Path

# Setup JSON logging for production
logger = setup_logging(
    level="INFO",
    log_file=Path("bosspig.log"),
    format_type="json",
    enable_console=True
)

# Analyze with logging
detector = BossPigDetector()
results = detector.analyze("document.txt")

# Logs will include:
# - Analysis start/complete events
# - Performance metrics (duration per detector)
# - Findings count and quality score
```

### Error Handling

```python
from bosspig.exceptions import (
    BossPigFileError,
    BossPigValidationError,
    BossPigAnalysisError
)

try:
    detector = BossPigDetector()
    results = detector.analyze("document.txt")
except BossPigFileError as e:
    print(f"File error: {e.message}")
    print(f"Suggestion: {e.suggestion}")
except BossPigValidationError as e:
    print(f"Validation error: {e.message}")
except BossPigAnalysisError as e:
    print(f"Analysis error: {e.message}")
```

### Performance Monitoring

```python
from bosspig.performance.benchmarks import BossPigBenchmark
from pathlib import Path

# Run benchmarks
benchmark = BossPigBenchmark()
detector = BossPigDetector()

results = benchmark.run_scalability_test(
    detector,
    base_text="Your test text here",
    sizes=[100, 500, 1000, 2000, 5000]
)

# Save report
benchmark.save_report(Path("benchmark_report.md"))

# Expected performance:
# - 1000 words: <50ms
# - 2000 words: <100ms
```

---

## Next Steps

### Learn More

- **[User Manual](USER_MANUAL.md)** - Complete feature reference
- **[Configuration Guide](CONFIGURATION.md)** - Customize BossPig
- **[API Documentation](API_REFERENCE.md)** - Full API reference

### Advanced Topics

- **Auto-Fix** - Automatically fix common issues
- **CI/CD Integration** - GitHub Actions, pre-commit hooks
- **Custom Rules** - Add your own detection patterns
- **Batch Processing** - Analyze multiple documents

### Get Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/bosspig/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/bosspig/discussions)
- **Email**: support@bosspig.dev

---

## Example: Complete Workflow

```python
from bosspig.detector import BossPigDetector
from bosspig.logging_config import setup_logging
from pathlib import Path

# Setup
setup_logging(level="INFO", log_file=Path("analysis.log"))
detector = BossPigDetector(
    document_type="technical_documentation",
    enable_nlp=True
)

# Analyze
results = detector.analyze("technical_spec.md")

# Generate report
with open("report.txt", "w") as f:
    f.write(results.summary())

# Print key metrics
print(f"Quality Score: {results.quality_metrics.overall_score:.1%}")
print(f"Grade: {results.quality_metrics.grade}")
print(f"Findings: {len(results.findings)}")

# Show critical issues
critical = [f for f in results.findings if f.severity.value == "critical"]
if critical:
    print("\nCritical Issues:")
    for finding in critical:
        print(f"  Line {finding.line_number}: {finding.message}")
        print(f"    Fix: {finding.suggestion}")
```

---

**Ready to improve your business writing? Get started with BossPig today!**

*Version: 1.0.0 (Beta) | Last Updated: 2025-11-22*
