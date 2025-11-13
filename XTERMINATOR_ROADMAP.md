# xTerminator: Automated Code Fixer - Roadmap
**Date**: November 9, 2025
**Based on**: Real Trough scan of HoloLoom (1,246 issues found)

---

## Key Insight: False Positives Are Real

**Observation**: The 11 "critical SQL injection" issues were false positives - SQL keywords in comments, not actual code.

**Lesson**: xTerminator needs **context-aware classification**, not just pattern matching.

---

## The Confidence Pyramid

```
         /\
        /  \  MANUAL (Security: 11 issues)
       /____\
      /      \ REVIEW (Error handling: 300 issues)
     /________\
    /          \ AUTO-FIX (Dead code, hardcoded: 500 issues)
   /____________\
  /              \ SKIP (False positives: 435 issues)
 /________________\
```

---

## xTerminator Architecture (6 Components)

### 1. CLASSIFIER
**Input**: SlopIssue from Trough  
**Output**: Classified with confidence + risk + strategy

```python
classify(issue):
    context = detect_context(issue)  # comment/string/code?
    confidence = score_confidence(issue, context)
    risk = assess_risk(issue.category, issue.severity)
    strategy = select_strategy(confidence, risk)
    return ClassifiedIssue(context, confidence, risk, strategy)
```

**Context Detection**:
- In comment? → Low confidence (likely false positive)
- In string literal? → Medium confidence (investigate)
- In executable code? → High confidence (likely real issue)

### 2. FIXER ENGINE (3 sub-engines)

**AST Fixer** (High confidence, automated):
- Dead code removal
- Hardcoded value extraction
- Import cleanup
- Simple refactoring

**Template Fixer** (Medium confidence, review):
- Error handling patterns
- Resource leak fixes (context managers)
- Type hints
- Docstrings

**Manual Queue** (Low confidence or high risk):
- Security issues (always manual)
- Complex refactoring
- Uncertain fixes

### 3. VALIDATOR
- ✅ Syntax check (AST parsing)
- ✅ Test execution (pytest/jest)
- ✅ Re-scan with Trough (no regressions)
- ✅ Human approval (medium/high risk)

### 4. GIT APPLICATOR
```bash
# One issue, one commit
git add file.py
git commit -m "fix(category): Brief description

Issue: category (severity: X, confidence: 0.XX)
Line: NNN
Strategy: AST/Template/Manual

🤖 xTerminator v1.0.0"
```

### 5. ROLLBACK MANAGER
```bash
# Easy undo
xterminator rollback --last
xterminator rollback --commit abc123
xterminator rollback --category hardcoded_values
```

### 6. LEARNING LOOP
- Track fix acceptance rate
- Calibrate confidence scores
- Learn false positive patterns
- Improve over time

---

## Fix Strategy by Category

| Category | Count | Strategy | Confidence | Auto? |
|----------|-------|----------|------------|-------|
| copy_paste | 450 | Extract function | 0.85 | ✅ Yes |
| error_handling | 287 | Template wrapper | 0.70 | ⚠️ Review |
| hardcoded_values | 220 | Move to env | 0.90 | ✅ Yes |
| timezone | 51 | Use aware datetime | 0.95 | ✅ Yes |
| dead_code | 38 | AST removal | 1.00 | ✅ Yes |
| security | 9 | **Manual only** | N/A | ❌ No |

---

## Implementation Phases

### Phase 1: Classification (Week 1-2)
```bash
xterminator classify report.md
# Output: Classified issues with confidence scores
```

**Deliverable**: Classification engine

### Phase 2: AST Auto-Fixer (Week 3-4)
```bash
xterminator fix --auto --category dead_code
# Automatically fixes high-confidence, low-risk issues
```

**Deliverable**: Auto-fix for 40% of issues

### Phase 3: Template Fixer (Week 5-6)
```bash
xterminator fix --template --review
# Suggests fixes, waits for approval
```

**Deliverable**: Suggested fixes for 24% of issues

### Phase 4: Git Integration (Week 7-8)
```bash
xterminator apply --batch --dry-run
xterminator apply --commit
xterminator rollback --last
```

**Deliverable**: Safe git workflow

### Phase 5: Validation (Week 9-10)
```bash
xterminator validate
# Runs tests, re-scans, checks for regressions
```

**Deliverable**: Production-ready pipeline

### Phase 6: Learning (Month 3+)
```bash
xterminator stats
# Shows fix success rate, calibration metrics
```

**Deliverable**: Self-improving system

---

## Example: Fixing Dead Code (Automated)

```python
# 1. DETECT (Trough)
Issue: unused import 'numpy' in config.py:15
Category: dead_code
Severity: LOW
Confidence: 1.0

# 2. CLASSIFY (xTerminator)
Context: import statement (executable code)
Risk: LOW
Strategy: AST removal
Decision: AUTO_FIX

# 3. FIX (AST Engine)
- import numpy as np  # Line 15

# 4. VALIDATE
✓ Syntax OK
✓ Tests pass
✓ No new issues

# 5. APPLY
git commit -m "fix(dead_code): Remove unused numpy import

Issue: dead_code (confidence: 1.0)
🤖 xTerminator"

# 6. VERIFY
Re-scan: Issue resolved ✓
```

---

## Example: Fixing Error Handling (Review Required)

```python
# 1. DETECT
Issue: file opened without error handling
Line: 145
Confidence: 0.8

# 2. CLASSIFY
Context: executable code
Risk: MEDIUM
Strategy: TEMPLATE
Decision: REVIEW

# 3. PROPOSE FIX
# Before:
f = open("data.txt")
data = f.read()

# After (proposed):
try:
    with open("data.txt") as f:
        data = f.read()
except IOError as e:
    logger.error(f"Failed to read data.txt: {e}")
    data = None

# 4. HUMAN REVIEW
User: Approve / Reject / Edit
→ Approved

# 5. APPLY
# ... commit ...
```

---

## Safety Rules

1. **Never auto-fix security issues** (always manual review)
2. **Always run tests** before committing
3. **Always allow rollback** (git-based)
4. **Start conservative** (expand automation gradually)
5. **Learn from feedback** (improve confidence over time)

---

## Success Metrics

### Month 1 Targets
- Fix acceptance: >80%
- False positive: <20%
- Regression: <5%
- Auto-fix: >30%
- Time saved: 10+ hrs/week

### Month 6 Targets
- Fix acceptance: >95%
- False positive: <10%
- Regression: <2%
- Auto-fix: >60%
- Time saved: 40+ hrs/week

---

## Commands Reference

```bash
# Scan codebase
trough scan ./HoloLoom --output report.md

# Classify issues
xterminator classify report.md

# Auto-fix safe issues
xterminator fix --auto --category dead_code,hardcoded_values

# Review and fix manually
xterminator fix --review --category error_handling

# Apply fixes with git
xterminator apply --dry-run
xterminator apply --commit

# Rollback if needed
xterminator rollback --last

# Check stats
xterminator stats
```

---

## Next Steps

1. ✅ **Trough working** - 1,246 issues found
2. ⏳ **Classification engine** - Week 1-2
3. ⏳ **AST auto-fixer** - Week 3-4
4. ⏳ **Template fixer** - Week 5-6
5. ⏳ **Git integration** - Week 7-8
6. ⏳ **Validation** - Week 9-10
7. ⏳ **Learning loop** - Month 3+

**Status**: Roadmap complete, ready to build
**Timeline**: 10 weeks to production

---

**Built with**: Observations from real Trough dogfooding
**Validated by**: 50 files, 1,246 issues, real false positives discovered
**Philosophy**: Safety first, automate gradually, learn continuously

🤖 xTerminator: Because bugs deserve swift justice ⚡
