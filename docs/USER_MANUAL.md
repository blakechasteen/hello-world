# BossPig User Manual

**Complete Reference Guide**

Version: 1.0.0 (Beta)
Last Updated: 2025-11-22

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Detection Categories](#detection-categories)
5. [Quality Scoring](#quality-scoring)
6. [Configuration](#configuration)
7. [Advanced Usage](#advanced-usage)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

---

## Introduction

BossPig is a business writing quality analyzer that detects 18 categories of issues across three domains:

- **Content Quality** - jargon, vague commitments, missing dates, AI hallucinations, passive voice
- **Specificity** - vague quantifiers, unmeasurable claims, relative statements without baselines
- **Brand Compliance** - capitalization, prohibited terms, tone violations
- **Governance** - required sections, disclaimers, approvals, version control, compliance frameworks

### Who Should Use BossPig?

- **Business Writers** - Improve clarity and specificity
- **Content Teams** - Enforce brand guidelines
- **Compliance Officers** - Validate governance requirements
- **Engineering Teams** - Integrate into CI/CD pipelines

### Performance

- **Fast**: Analyzes 1000-word documents in <30ms
- **Scalable**: Linear O(n) complexity up to 10,000+ words
- **Production-Ready**: 137/137 tests passing, comprehensive error handling

---

## Installation

### Basic Installation

```bash
pip install bosspig
```

### With Optional Dependencies

```bash
# For NLP-based passive voice detection
pip install bosspig[nlp]
pip install spacy
python -m spacy download en_core_web_sm

# For development
pip install bosspig[dev]
```

### Verify Installation

```bash
python -c "from bosspig import BossPigDetector; print('BossPig ready!')"
```

---

## Core Concepts

### Findings

A **Finding** represents a detected issue:

```python
from bosspig.detector import BossPigDetector

detector = BossPigDetector()
results = detector.analyze("We need to synergize our core competencies.")

for finding in results.findings:
    print(f"{finding.category.value}: {finding.text}")
    print(f"Severity: {finding.severity.value}")
    print(f"Line: {finding.line_number}")
    print(f"Suggestion: {finding.suggestion}")
```

**Finding Attributes**:
- `category` - FindingCategory enum (e.g., CORPORATE_JARGON)
- `severity` - Severity enum (INFO, WARNING, CRITICAL)
- `line_number` - Line where issue was found
- `column_number` - Optional column position
- `text` - The problematic text
- `message` - Description of the issue
- `suggestion` - How to fix it
- `context` - Surrounding text for context

### Quality Metrics

BossPig calculates **5 quality scores** (0.0 - 1.0):

1. **Clarity Score** - Low jargon, clear language
2. **Specificity Score** - Concrete, measurable statements
3. **Actionability Score** - Clear ownership, active voice
4. **Professionalism Score** - No AI boilerplate
5. **Completeness Score** - All required sections present

**Overall Score** = Average of all 5 scores

```python
metrics = results.quality_metrics

print(f"Clarity: {metrics.clarity_score:.1%}")
print(f"Specificity: {metrics.specificity_score:.1%}")
print(f"Overall: {metrics.overall_score:.1%}")
print(f"Grade: {metrics.grade}")  # A, B, C, D, F
```

### Document Stats

Automatic document statistics:

```python
stats = results.document_stats

print(f"Words: {stats.word_count}")
print(f"Sentences: {stats.sentence_count}")
print(f"Paragraphs: {stats.paragraph_count}")
print(f"Sections: {stats.section_count}")
print(f"Reading Time: {stats.reading_time_minutes:.1f} min")
```

---

## Detection Categories

### 1. Corporate Jargon (5 issues)

**What it detects**: Buzzwords, filler language, corporate speak

**Examples**:
- "synergize our core competencies" → "combine our strengths"
- "leverage low-hanging fruit" → "pursue easy wins"
- "move the needle" → "make progress"

**Configuration**: Uses `jargon_dict.json` (customizable)

**Code**:
```python
detector = BossPigDetector(jargon_dict_path=Path("custom_jargon.json"))
```

---

### 2. Vague Commitments (8 patterns)

**What it detects**: Unclear ownership, no accountability

**Examples**:
- "We will try to..." → "@alice will complete by 2025-12-01"
- "Hopefully we can..." → "Jane will deliver by Friday"
- "Someone should..." → "@bob is responsible"

**Severity**: CRITICAL (blocks publication)

---

### 3. Missing Dates/Deadlines (13 patterns)

**What it detects**: Vague timelines, no specific dates

**Examples**:
- "soon" → "2025-12-15"
- "ASAP" → "by 5pm today"
- "TBD" → "decision by 2025-12-10"

**Severity**: WARNING

---

### 4. AI Hallucination Markers (9 patterns)

**What it detects**: LLM boilerplate, placeholders

**Examples**:
- "As an AI language model..." → DELETE
- "[INSERT X HERE]" → Fill in actual content
- "[TODO: complete]" → Complete section

**Severity**: CRITICAL

---

### 5. Passive Voice (NLP + Regex)

**What it detects**: Unclear responsibility

**Examples**:
- "The work was completed" → "@alice completed the work"
- "Mistakes were made" → "Bob made mistakes in the API design"

**Enable NLP Mode**:
```python
detector = BossPigDetector(enable_nlp=True)  # Requires spaCy
```

**Fallback**: Regex patterns if spaCy unavailable

---

### 6. Vague Quantifiers (Specificity)

**What it detects**: Unspecific amounts

**Examples**:
- "several issues" → "7 issues"
- "many users" → "142 users"
- "significant improvement" → "15% improvement"

---

### 7. Unmeasurable Claims (Specificity)

**What it detects**: Claims without metrics

**Examples**:
- "We will improve performance" → "We will reduce latency from 200ms to 50ms"
- "Expect to achieve success" → "Expect 95% uptime"

---

### 8. Relative Statements (Specificity)

**What it detects**: Comparisons without baselines

**Examples**:
- "faster than before" → "faster than 200ms (now 50ms)"
- "better results" → "10% higher conversion (from 2.5% to 2.75%)"

---

### 9. Brand Capitalization (Brand Guidelines)

**What it detects**: Incorrect brand name capitalization

**Examples** (from config):
- "iphone" → "iPhone"
- "Github" → "GitHub"

**Configuration**: `brand_config.json`

---

### 10. Prohibited Terms (Brand Guidelines)

**What it detects**: Banned words/phrases

**Examples** (configurable):
- "cheap" → "affordable"
- "fail" → "did not meet expectations"

---

### 11. Non-Preferred Terminology (Brand Guidelines)

**What it detects**: Deprecated terms with replacements

**Examples** (configurable):
- "master/slave" → "primary/replica"
- "whitelist" → "allowlist"

---

### 12. Tone Violations (Brand Guidelines)

**What it detects**: Language that doesn't match brand voice

**Examples** (configurable):
- Too casual: "gonna", "wanna"
- Too formal: "heretofore", "henceforth"

---

### 13-17. Governance Violations

**Required Sections**: Purpose, Scope, Responsibilities, Version History

**Required Disclaimers**: Confidentiality, HIPAA, Data Retention

**Approval Workflows**: Author, Reviewer, Approval Date, Status

**Version Control**: Version Number, Last Updated Date

**Compliance Frameworks**: HIPAA, SOC2, GDPR

**Document Types**:
- `technical_documentation` - Basic governance
- `healthcare` - HIPAA compliance
- `data_policies` - SOC2, GDPR
- `strategic_plans` - Confidentiality focus

**Configuration**: `governance_config.json`

---

## Quality Scoring

### Score Calculation

```python
# Clarity: 1.0 - (jargon_count / total_words)
clarity = 1.0 - (4 / 100)  # 4 jargon in 100 words = 0.96

# Specificity: Uses advanced SpecificityDetector
# Penalty: 0.10 per vague, 0.15 per unmeasurable, 0.12 per relative
specificity = 1.0 - (vague*0.10 + unmeasurable*0.15 + relative*0.12)

# Actionability: 1.0 - (passive_count / total_paragraphs)
actionability = 1.0 - (2 / 5)  # 2 passive in 5 paragraphs = 0.60

# Professionalism: 1.0 - (hallucinations / total_sections)
professionalism = 1.0 - (1 / 3)  # 1 hallucination in 3 sections = 0.67

# Completeness: Same as professionalism
completeness = professionalism

# Overall: Average of all 5
overall = (clarity + specificity + actionability + professionalism + completeness) / 5
```

### Grade Thresholds

| Grade | Score Range | Interpretation |
|-------|-------------|----------------|
| **A** | 90-100% | Excellent - publication ready |
| **B** | 80-89% | Good - minor improvements |
| **C** | 70-79% | Fair - notable issues |
| **D** | 60-69% | Poor - significant rewrite |
| **F** | <60% | Failing - major problems |

### Governance Scoring

```python
from bosspig.detector.governance import calculate_governance_score

metrics = calculate_governance_score(results.findings)

print(f"Governance Score: {metrics.governance_score:.1%}")
print(f"Missing Sections: {metrics.missing_sections}")
print(f"Missing Disclaimers: {metrics.missing_disclaimers}")
print(f"Compliance Violations: {metrics.compliance_violations}")
```

**Governance Penalties**:
- Missing Section: -0.15 per section
- Missing Disclaimer: -0.20 per disclaimer
- Missing Approval: -0.10 per approval field
- Missing Version Control: -0.08 per field
- Compliance Violation: -0.25 per violation (most severe)

---

## Configuration

### Jargon Dictionary

**File**: `jargon_dict.json`

```json
{
  "synergize": {
    "category": "corporate_buzzwords",
    "severity": "warning",
    "replacement": "combine",
    "explanation": "Vague business jargon"
  }
}
```

**Load Custom Dictionary**:
```python
detector = BossPigDetector(jargon_dict_path=Path("custom_jargon.json"))
```

---

### Brand Guidelines

**File**: `brand_config.json`

```json
{
  "brand_name": "Acme Corp",
  "document_type": "marketing",
  "version": "1.0.0",

  "brand_capitalization": {
    "iPhone": {
      "incorrect_forms": ["iphone", "Iphone", "IPhone"],
      "severity": "high"
    }
  },

  "prohibited_terms": {
    "cheap": {
      "reason": "Negative connotation",
      "replacement": "affordable",
      "severity": "high"
    }
  },

  "tone_guidelines": {
    "formality_level": "professional",
    "avoid_casual": ["gonna", "wanna"],
    "avoid_overly_formal": ["heretofore"]
  }
}
```

---

### Governance Configuration

**File**: `governance_config.json`

```json
{
  "document_type": "technical_documentation",
  "version": "1.0.0",

  "required_sections": {
    "sections": [
      {
        "name": "Purpose",
        "patterns": ["^\\s*#+\\s*Purpose", "^\\s*#+\\s*Overview"],
        "severity": "high",
        "required": true
      }
    ]
  },

  "compliance_frameworks": {
    "frameworks": [
      {
        "name": "HIPAA",
        "required_elements": [
          {
            "element": "PHI Handling",
            "patterns": ["Protected Health Information", "\\bPHI\\b"],
            "severity": "critical"
          }
        ]
      }
    ]
  }
}
```

---

## Advanced Usage

### Batch Processing

```python
from pathlib import Path

detector = BossPigDetector()
docs_dir = Path("documents")

results = {}
for doc_file in docs_dir.glob("*.md"):
    results[doc_file.name] = detector.analyze(doc_file)

# Summary report
for filename, result in results.items():
    print(f"{filename}: {result.quality_metrics.grade} ({len(result.findings)} issues)")
```

### Filter by Category

```python
# Get only jargon findings
jargon_only = [
    f for f in results.findings
    if f.category.value == "corporate_jargon"
]

# Get governance violations
governance_only = [
    f for f in results.findings
    if "missing" in f.category.value or "compliance" in f.category.value
]
```

### Generate HTML Report

```python
def generate_html_report(results):
    html = f"""
    <html>
    <head><title>BossPig Report</title></head>
    <body>
        <h1>Quality Report</h1>
        <p>Overall Score: {results.quality_metrics.overall_score:.1%} ({results.quality_metrics.grade})</p>

        <h2>Findings ({len(results.findings)})</h2>
        <ul>
    """

    for finding in results.findings:
        html += f"""
        <li>
            <strong>Line {finding.line_number}</strong>: {finding.category.value}<br>
            {finding.message}<br>
            <em>Suggestion: {finding.suggestion}</em>
        </li>
        """

    html += """
        </ul>
    </body>
    </html>
    """

    return html

# Save report
with open("report.html", "w") as f:
    f.write(generate_html_report(results))
```

### Custom Severity Filtering

```python
# Only critical and warnings
important_findings = [
    f for f in results.findings
    if f.severity.value in ["critical", "warning"]
]

# Sort by severity
from bosspig.detector.core import Severity

severity_order = {"critical": 0, "warning": 1, "info": 2}
sorted_findings = sorted(
    results.findings,
    key=lambda f: severity_order[f.severity.value]
)
```

---

## Production Deployment

### Logging Configuration

```python
from bosspig.logging_config import setup_logging
from pathlib import Path

# Production logging with JSON format
logger = setup_logging(
    level="INFO",
    log_file=Path("/var/log/bosspig/analysis.log"),
    format_type="json",
    enable_console=False  # Disable console in production
)

# Analyze with logging
detector = BossPigDetector()
results = detector.analyze("document.txt")

# Logs include:
# - Analysis start/complete
# - Per-detector performance (ms)
# - Quality score
# - Findings count
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
    # File not found, permission denied, encoding issues
    logger.error(f"File error: {e.message}")
    logger.info(f"Suggestion: {e.suggestion}")

except BossPigValidationError as e:
    # Empty text, invalid document type
    logger.warning(f"Validation error: {e.message}")

except BossPigAnalysisError as e:
    # Analysis failure (regex error, etc.)
    logger.error(f"Analysis error: {e.message}", exc_info=True)

except Exception as e:
    # Unexpected error
    logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
```

### Performance Monitoring

```python
from bosspig.performance.benchmarks import BossPigBenchmark

# Run periodic benchmarks
benchmark = BossPigBenchmark()
detector = BossPigDetector()

results = benchmark.run_scalability_test(
    detector,
    base_text=sample_text,
    sizes=[100, 500, 1000, 2000, 5000]
)

# Generate report
benchmark.save_report(Path("/var/log/bosspig/benchmark.md"))

# Alert if performance degrades
if results[2].duration_ms > 50:  # 1000-word target
    logger.warning(f"Performance degradation: 1000 words took {results[2].duration_ms:.1f}ms")
```

---

## Troubleshooting

### Issue: Empty Findings

**Symptom**: `results.findings` is empty

**Causes**:
1. Document is actually high quality
2. Wrong document type (governance config doesn't apply)
3. Custom config files not loaded

**Solution**:
```python
# Check quality metrics
print(results.quality_metrics.overall_score)  # If >0.9, document is excellent

# Verify config loading
detector = BossPigDetector(
    governance_config_path=Path("config/governance_config.json"),
    document_type="healthcare"  # Ensure correct type
)
```

---

### Issue: Performance Slow

**Symptom**: Analysis takes >100ms for 1000 words

**Causes**:
1. NLP mode enabled (adds overhead)
2. Very large document (>10,000 words)
3. Disk I/O bottleneck

**Solution**:
```python
# Disable NLP if not needed
detector = BossPigDetector(enable_nlp=False)

# Batch process large documents
text_chunks = split_into_chunks(large_text, chunk_size=5000)
for chunk in text_chunks:
    results = detector.analyze(chunk)
```

---

### Issue: Import Errors

**Symptom**: `ImportError: cannot import name 'BossPigDetector'`

**Causes**:
1. Package not installed
2. Python path issues

**Solution**:
```bash
# Reinstall package
pip uninstall bosspig
pip install bosspig

# Verify installation
python -c "from bosspig import BossPigDetector; print('OK')"
```

---

## API Reference

See [API_REFERENCE.md](API_REFERENCE.md) for complete API documentation.

### Quick Reference

**Main Classes**:
- `BossPigDetector` - Main detection engine
- `BossPigFindings` - Analysis results
- `Finding` - Individual issue
- `QualityMetrics` - Quality scores
- `DocumentStats` - Document statistics

**Enums**:
- `FindingCategory` - Category of finding
- `Severity` - Issue severity (INFO, WARNING, CRITICAL)

**Exceptions**:
- `BossPigError` - Base exception
- `BossPigFileError` - File operations
- `BossPigValidationError` - Input validation
- `BossPigAnalysisError` - Analysis errors

---

## Appendix

### Default Jargon Dictionary

See `bosspig/detector/jargon_dict.py` for complete list (100+ entries)

### Supported Document Types

- `technical_documentation` (default)
- `healthcare`
- `data_policies`
- `strategic_plans`
- Custom types via `governance_config.json`

### Performance Benchmarks

| Document Size | Expected Duration |
|--------------|-------------------|
| 100 words | <5ms |
| 500 words | <20ms |
| 1000 words | <30ms |
| 2000 words | <55ms |
| 5000 words | <160ms |
| 10000 words | <310ms |

---

**Questions? Issues? Feedback?**

- GitHub Issues: https://github.com/yourusername/bosspig/issues
- Documentation: https://bosspig.readthedocs.io
- Email: support@bosspig.dev

---

*Version: 1.0.0 (Beta)*
*Last Updated: 2025-11-22*
*© 2025 BossPig Project*
