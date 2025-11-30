# HoloLoom Integration: Trough & BossPig

**Version**: 1.0.0
**Created**: November 2025
**Status**: Production Ready

Complete guide to integrating Trough (code quality) and BossPig (business writing) as HoloLoom departments.

---

## Table of Contents

1. [Overview](#overview)
2. [Department Architecture](#department-architecture)
3. [Trough Department](#trough-department)
4. [BossPig Department](#bossiig-department)
5. [Multi-Department Workflows](#multi-department-workflows)
6. [Configuration](#configuration)
7. [Examples](#examples)

---

## Overview

Trough and BossPig integrate as specialized HoloLoom departments:

- **QA Department** - Trough provides code quality analysis (AI slop detection, logic error detection)
- **Writing Department** - BossPig analyzes business documentation (jargon, clarity, ownership)

Both departments:
- Follow HoloLoom's Department Protocol
- Return structured findings via Spacetime
- Support multi-step workflows via orchestration
- Include safety guardrails and audit trails
- Integrate with alignment framework

---

## Department Architecture

### HoloLoom Department Protocol

All departments implement this protocol:

```python
from HoloLoom.departments.protocol import DepartmentProtocol
from typing import Dict, Any

class Department(DepartmentProtocol):
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process request and return structured result.

        Args:
            request: {
                "action": str,              # "analyze", "fix", "report", etc.
                "input": str or Dict,       # Content to process
                "context": Dict[str, Any],  # Optional context
                "options": Dict[str, Any]   # Detection/fixing options
            }

        Returns: {
            "success": bool,
            "result": Any,                  # Analysis result
            "metadata": Dict[str, Any],     # Timing, confidence, etc.
            "audit_trail": List[Dict]       # Decision log
        }
        """
        ...

    async def health_check(self) -> bool:
        """Check if department is operational."""
        ...

    def get_capabilities(self) -> Dict[str, Any]:
        """List supported actions and options."""
        ...
```

---

## Trough Department

Trough integrates as the QA Department.

### Initialization

```python
from HoloLoom.departments import register_department
from HoloLoom.agentic.trough_department import TroughDepartment

# Register Trough as QA department
qa_dept = TroughDepartment()
register_department("quality_assurance", qa_dept)

# Now accessible via HoloLoom
from HoloLoom.departments import get_department
qa = get_department("quality_assurance")
```

### Request Format

```python
request = {
    "action": "analyze",              # analyze, analyze_file, fix, report, etc.
    "input": code_string,             # Code to analyze
    "context": {
        "language": "python",         # Programming language
        "file_path": "app.py",        # File path for context
        "severity_threshold": "HIGH"  # Only return HIGH+ issues
    },
    "options": {
        "include_logic_errors": True,
        "include_slop_issues": True,
        "max_issues": 100
    }
}
```

### Response Format

```python
response = {
    "success": True,
    "result": {
        "slop_issues": [SlopIssue, ...],
        "logic_errors": [LogicError, ...],
        "total_issues": 23,
        "quality_score": 72
    },
    "metadata": {
        "processing_time_ms": 145,
        "lines_analyzed": 1250,
        "confidence": 0.92
    },
    "audit_trail": [
        {"timestamp": "2025-11-22T10:30:00Z", "action": "started_analysis"},
        {"timestamp": "2025-11-22T10:30:00Z", "action": "completed_slop_detection"},
        {"timestamp": "2025-11-22T10:30:00Z", "action": "completed_logic_detection"},
        {"timestamp": "2025-11-22T10:30:00Z", "action": "returned_results"}
    ]
}
```

### Supported Actions

```python
# Analyze code for issues
response = await qa.process({
    "action": "analyze",
    "input": code,
    "context": {"language": "python", "file_path": "app.py"}
})

# Analyze specific category
response = await qa.process({
    "action": "analyze_category",
    "input": code,
    "context": {"language": "python", "category": "SECURITY"}
})

# Generate HTML report
response = await qa.process({
    "action": "report",
    "input": code,
    "context": {"language": "python", "file_path": "app.py"},
    "options": {"include_code": True}
})

# Auto-fix issues (if available)
response = await qa.process({
    "action": "fix",
    "input": code,
    "context": {"language": "python", "dry_run": True}
})

# Check health
is_healthy = await qa.health_check()

# Get capabilities
caps = qa.get_capabilities()
```

---

## BossPig Department

BossPig integrates as the Writing Department.

### Initialization

```python
from HoloLoom.departments import register_department
from HoloLoom.agentic.bosspig_department import BossPigDepartment

# Register BossPig as writing department
writing_dept = BossPigDepartment()
register_department("writing_quality", writing_dept)

# Accessible via HoloLoom
from HoloLoom.departments import get_department
writing = get_department("writing_quality")
```

### Request Format

```python
request = {
    "action": "analyze",              # analyze, suggest_fixes, score, etc.
    "input": document_text,           # Text or file path to analyze
    "context": {
        "document_type": "proposal",  # proposal, email, report, etc.
        "audience": "stakeholders",
        "style_guide": "ap"           # Associated Press, Chicago, etc.
    },
    "options": {
        "enable_nlp": True,           # Use spaCy for advanced analysis
        "custom_jargon": "jargon.json",
        "strict_mode": False
    }
}
```

### Response Format

```python
response = {
    "success": True,
    "result": {
        "findings": [Finding, ...],
        "metrics": QualityMetrics(
            total_issues=8,
            critical_count=2,
            warning_count=4,
            info_count=2,
            quality_score=72,
            score_breakdown={...}
        ),
        "summary": "Document has clarity issues and vague commitments"
    },
    "metadata": {
        "processing_time_ms": 87,
        "word_count": 1250,
        "estimated_reading_time": 5
    },
    "audit_trail": [
        {"action": "started_analysis", "timestamp": "..."},
        {"action": "detected_jargon", "findings": 3, "timestamp": "..."},
        {"action": "detected_vague_commitments", "findings": 2, "timestamp": "..."},
        {"action": "returned_results", "timestamp": "..."}
    ]
}
```

### Supported Actions

```python
# Analyze document
response = await writing.process({
    "action": "analyze",
    "input": "We will leverage our synergies ASAP...",
    "context": {"document_type": "proposal"}
})

# Get quality score
response = await writing.process({
    "action": "score",
    "input": "Document text"
})

# Get improvement suggestions
response = await writing.process({
    "action": "suggest_fixes",
    "input": "Document text"
})

# Analyze specific finding category
response = await writing.process({
    "action": "analyze_category",
    "input": "Document text",
    "context": {"category": "CORPORATE_JARGON"}
})

# Check health
is_healthy = await writing.health_check()

# Get capabilities
caps = writing.get_capabilities()
```

---

## Multi-Department Workflows

### Workflow 1: Code Review Pipeline

Analyze code for both AI slop and security issues:

```python
from HoloLoom.departments import get_department, DepartmentOrchestrator

async def code_review_pipeline(code: str, file_path: str) -> Dict:
    """Multi-step code review with QA and optional writing review"""

    orchestrator = DepartmentOrchestrator()
    qa = get_department("quality_assurance")
    writing = get_department("writing_quality")  # For docstrings/comments

    # Step 1: Run QA analysis
    qa_result = await qa.process({
        "action": "analyze",
        "input": code,
        "context": {"language": "python", "file_path": file_path}
    })

    # Step 2: Filter critical issues
    critical_issues = [
        i for i in qa_result["result"]["slop_issues"]
        if i.severity.value == "CRITICAL"
    ]

    # Step 3: Check docstrings/comments for clarity
    comments = extract_comments(code)
    if comments:
        writing_result = await writing.process({
            "action": "analyze",
            "input": comments,
            "context": {"document_type": "code_comments"}
        })
    else:
        writing_result = None

    # Step 4: Synthesize results
    return {
        "code_quality": qa_result,
        "documentation_quality": writing_result,
        "ready_for_merge": len(critical_issues) == 0,
        "review_summary": f"{len(critical_issues)} critical issues found"
    }

# Usage
result = await code_review_pipeline(code, "src/app.py")
if result["ready_for_merge"]:
    print("✅ Code approved for merge")
else:
    print(f"❌ {result['review_summary']}")
```

### Workflow 2: Documentation Quality Review

Review business documents for clarity and compliance:

```python
async def document_quality_review(file_path: str) -> Dict:
    """Review document for quality, clarity, and compliance"""

    writing = get_department("writing_quality")

    # Main analysis
    result = await writing.process({
        "action": "analyze",
        "input": file_path,
        "context": {"document_type": "proposal"}
    })

    findings = result["result"]["findings"]
    metrics = result["result"]["metrics"]

    # Categorize by type
    critical = [f for f in findings if f.severity.value == "CRITICAL"]
    by_category = {}
    for f in findings:
        cat = f.category.value
        by_category[cat] = by_category.get(cat, 0) + 1

    return {
        "quality_score": metrics.quality_score,
        "critical_issues": len(critical),
        "total_issues": metrics.total_issues,
        "issues_by_category": by_category,
        "ready_for_publication": metrics.quality_score >= 80,
        "recommended_actions": generate_recommendations(findings)
    }

# Usage
review = await document_quality_review("proposal.txt")
print(f"Quality Score: {review['quality_score']}/100")
if review['ready_for_publication']:
    print("✅ Document approved for publication")
else:
    print(f"⚠️ {review['critical_issues']} critical issues to fix")
```

### Workflow 3: Integrated Code & Writing Review

Combined review for code quality and inline documentation:

```python
async def integrated_code_review(code: str, file_path: str) -> Dict:
    """Complete code review including documentation quality"""

    qa = get_department("quality_assurance")
    writing = get_department("writing_quality")

    # 1. Analyze code structure
    code_analysis = await qa.process({
        "action": "analyze",
        "input": code,
        "context": {"language": "python", "file_path": file_path}
    })

    # 2. Extract and analyze docstrings
    docstrings = extract_docstrings(code)
    doc_analysis = None
    if docstrings:
        doc_analysis = await writing.process({
            "action": "analyze",
            "input": docstrings,
            "context": {"document_type": "code_docstrings"}
        })

    # 3. Extract and analyze comments
    comments = extract_comments(code)
    comment_analysis = None
    if comments:
        comment_analysis = await writing.process({
            "action": "analyze",
            "input": comments,
            "context": {"document_type": "code_comments"}
        })

    # 4. Synthesize report
    qa_critical = len([i for i in code_analysis["result"]["slop_issues"]
                       if i.severity.value == "CRITICAL"])
    doc_warnings = doc_analysis["result"]["metrics"].warning_count if doc_analysis else 0
    comment_warnings = comment_analysis["result"]["metrics"].warning_count if comment_analysis else 0

    return {
        "code_quality_score": code_analysis["result"]["quality_score"],
        "code_critical_issues": qa_critical,
        "documentation_quality_score": doc_analysis["result"]["metrics"].quality_score if doc_analysis else 100,
        "comment_quality_score": comment_analysis["result"]["metrics"].quality_score if comment_analysis else 100,
        "overall_approval": (qa_critical == 0 and
                            (not doc_analysis or doc_analysis["result"]["metrics"].quality_score >= 75) and
                            (not comment_analysis or comment_analysis["result"]["metrics"].quality_score >= 75)),
        "detailed_results": {
            "code": code_analysis,
            "documentation": doc_analysis,
            "comments": comment_analysis
        }
    }

# Usage
review = await integrated_code_review(code, "main.py")
if review["overall_approval"]:
    print("✅ Code approved - good quality and documentation")
else:
    print(f"⚠️ Code: {review['code_critical_issues']} critical issues")
    print(f"⚠️ Docs: {100 - review['documentation_quality_score']} issues")
```

---

## Configuration

### Department-Level Configuration

```python
from HoloLoom.config import Config

# Configure Trough department
config = Config.fused()
config.qa_department = {
    "enabled": True,
    "include_logic_errors": True,
    "include_slop_issues": True,
    "strictness": "medium",  # low, medium, high
    "timeout_seconds": 30
}

# Configure BossPig department
config.writing_department = {
    "enabled": True,
    "enable_nlp": True,
    "strictness": "medium",
    "timeout_seconds": 20,
    "custom_jargon_path": None
}
```

### Custom Jargon Dictionary for BossPig

Create `custom_jargon.json`:

```json
{
  "corporate_jargon": [
    {
      "term": "leverage",
      "replacement": "use",
      "category": "vague_action",
      "severity": "warning"
    },
    {
      "term": "synergies",
      "replacement": "combined strength",
      "category": "meaningless",
      "severity": "critical"
    }
  ],
  "vague_commitments": [
    {
      "term": "ASAP",
      "replacement": "by [specific date]",
      "category": "no_deadline",
      "severity": "critical"
    }
  ]
}
```

Then use:

```python
writing = get_department("writing_quality")
result = await writing.process({
    "action": "analyze",
    "input": text,
    "options": {"custom_jargon": "custom_jargon.json"}
})
```

---

## Examples

### Example 1: Pre-Commit Hook

Prevent commits with critical code issues:

```python
#!/usr/bin/env python3
"""Pre-commit hook - check code quality"""

import asyncio
import subprocess
from HoloLoom.departments import get_department

async def check_staged_code():
    """Check staged files for critical issues"""

    # Get staged files
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True, text=True
    )
    staged_files = result.stdout.strip().split('\n')

    qa = get_department("quality_assurance")
    critical_count = 0

    for file in staged_files:
        if not file.endswith('.py'):
            continue

        code = subprocess.run(
            ["git", "show", f":{file}"],
            capture_output=True, text=True
        ).stdout

        response = await qa.process({
            "action": "analyze",
            "input": code,
            "context": {"language": "python", "file_path": file}
        })

        critical = len([i for i in response["result"]["slop_issues"]
                       if i.severity.value == "CRITICAL"])
        critical_count += critical

    # Block if critical issues found
    if critical_count > 0:
        print(f"❌ Commit blocked: {critical_count} critical issues")
        return False

    print("✅ Code quality check passed")
    return True

# Run check
if asyncio.run(check_staged_code()):
    exit(0)
else:
    exit(1)
```

### Example 2: CI/CD Integration

```yaml
# .github/workflows/quality-check.yml
name: Code Quality Check

on: [pull_request, push]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Run Trough (Code Quality)
        run: |
          python -c "
          import asyncio
          from HoloLoom.departments import get_department

          async def check():
              qa = get_department('quality_assurance')
              with open('app.py') as f:
                  code = f.read()
              response = await qa.process({
                  'action': 'analyze',
                  'input': code,
                  'context': {'language': 'python', 'file_path': 'app.py'}
              })
              critical = len([i for i in response['result']['slop_issues']
                             if i.severity.value == 'CRITICAL'])
              exit(0 if critical == 0 else 1)

          asyncio.run(check())
          "

      - name: Run BossPig (Documentation Quality)
        run: |
          python -c "
          import asyncio
          from HoloLoom.departments import get_department

          async def check():
              writing = get_department('writing_quality')
              with open('README.md') as f:
                  text = f.read()
              response = await writing.process({
                  'action': 'analyze',
                  'input': text,
                  'context': {'document_type': 'readme'}
              })
              score = response['result']['metrics'].quality_score
              exit(0 if score >= 75 else 1)

          asyncio.run(check())
          "
```

---

## See Also

- [Trough API Reference](./trough/API_REFERENCE.md)
- [BossPig API Reference](./bosspig/API_REFERENCE.md)
- [HoloLoom Departments Guide](./HoloLoom/departments/README.md)
- [Examples Directory](./examples/integration/)
