# xTerminator Complete Workflow Guide

## System Architecture

```
Phase 1: Issue Detection (Trough)
              ↓
Phase 1: Classification Engine
  ├─ Context Detection
  ├─ Risk Assessment
  ├─ Strategy Selection
  └─ Confidence Scoring
              ↓
          Decision Point
              ↓
     ┌────────┴────────┐
     ↓                 ↓
LOW-RISK          HIGH-RISK
(AutoFix)         (Review)
     ↓                 ↓
Direct Commit    Feature Branch
to Main/Master   (for approval)
     ↓                 ↓
.xterminator/commits.json
(Complete Audit Trail)
     ↓
Rollback (if needed)
  ├─ Last N
  ├─ By Category
  └─ By File
```

## Phase 1: Classification Pipeline

```
Issue Input
    ↓
ContextDetector: Where is the issue?
  - Code vs Comment vs String
  - Production-critical path
  - Test coverage
    ↓
RiskAssessor: How risky is it?
  - Category-based risk
  - Context-based escalation
  - Complexity assessment
    ↓
StrategySelector: How to fix it?
  - AST transformation
  - Template-based
  - Manual required
    ↓
ConfidenceScorer: How confident?
  - Multi-factor analysis
  - 0.0-1.0 confidence
    ↓
FixProposal Ready
  - original_code
  - proposed_code
  - explanation
  - risk_level
  - confidence
```

## Phase 4: Low-Risk Path (AUTO-APPLY)

```
FixProposal Input
├─ Risk: LOW
├─ Confidence >= 0.85
└─ safe_to_autofix: True
    ↓
GitApplicator.apply_fix()
    ├─ Check uncommitted changes
    ├─ Write fixed code
    ├─ Git add file
    ├─ Create commit
    │  └─ fix(category): description
    │     File: path
    │     Line: number
    │     Strategy: method
    │     Confidence: 0.92
    │     Risk: low
    │     Fix ID: FIX_001
    └─ Save metadata
         └─ .xterminator/commits.json

Result: Commit to Main/Master
```

## Phase 4: High-Risk Path (FEATURE BRANCH)

```
FixProposal Input
├─ Risk: HIGH or CRITICAL
└─ requires_approval: True
    ↓
GitApplicator.apply_fix()
    ├─ Create feature branch
    │  └─ xterminator/{fix_id}/{category}
    ├─ Write fixed code
    ├─ Create commit
    ├─ Switch back to original branch
    └─ Save metadata
         └─ .xterminator/commits.json

Result: Commit to Feature Branch
        (awaiting manual review)
```

## Rollback Strategies

```
RollbackManager Options:

1. Rollback Last N
   rollback_last(n=2)
   └─ Revert 2 most recent fixes

2. Rollback by Category
   rollback_category("hardcoded_values")
   └─ Revert all hardcoded value fixes

3. Rollback by File
   rollback_file("config.py")
   └─ Revert all fixes to config.py

Safety Checks:
- Prevents rollback of pushed commits
- Requires force=True for pushed commits
- Updates metadata after rollback
```

## Risk-Based Automation Decision

```
Confidence Check:
├─ < 0.70: MANUAL REVIEW (always)
├─ 0.70-0.85: FEATURE BRANCH (needs review)
└─ >= 0.85: CHECK RISK

Risk Check (if conf >= 0.85):
├─ CRITICAL: MANUAL ONLY (never auto)
├─ HIGH: FEATURE BRANCH (review required)
├─ MEDIUM: FEATURE BRANCH (prefer review)
└─ LOW: AUTO-FIX (direct commit)

Test Coverage:
├─ No tests: Escalate risk level
└─ Tests present: Use assessed risk
```

## Metadata Structure

```
.xterminator/commits.json

{
  "commit-hash-abc123": {
    "commit_hash": "abc123...",
    "timestamp": 1731417600.0,
    "file_path": "config.py",
    "issue_category": "hardcoded_values",
    "risk_level": "low",
    "confidence": 0.92,
    "fix_strategy": "template",
    "fix_id": "FIX_HARDCODED_001"
  }
}

Enables:
- Complete audit trail
- Intelligent rollback
- Statistics
- Compliance reporting
```

## Typical Daily Workflow

```
Morning:
1. Run Trough → Detect issues
2. Filter by risk/confidence
3. Auto-apply LOW-risk fixes → main
4. Create branches for HIGH-risk → review queue

Afternoon:
5. Review feature branches
6. Request changes or approve
7. Merge approved branches → main
8. Reject and rollback if needed

Evening:
9. View statistics
10. Archive old branches
11. Backup metadata
```

## Production Deployment

```
Prerequisites:
- Git initialized and configured
- user.name and user.email set
- Main/master branch protected
- CI/CD runs on feature branches

Process:
1. Deploy xTerminator + Trough
2. Run issue detection
3. Apply LOW-risk fixes (auto)
4. Queue HIGH-risk for review
5. Manual approval process
6. Merge to main after tests pass
7. Monitor results
8. Rollback if issues found
```

## Performance Characteristics

```
Operation               Time      Notes
────────────────────────────────────────
Commit creation         <100ms    Per fix
Metadata save           <10ms     JSON I/O
Rollback commit         <50ms     Per revert
Dry-run test            <5ms      No git ops
Multiple commits        <1s       10 fixes
```

## Error Recovery

```
Error: "Not a git repository"
Solution: Initialize git first

Error: "Uncommitted changes"
Solution: Commit or stash

Error: "Commit already pushed"
Solution: Requires force=True

Error: "Invalid git config"
Solution: Set user.name/email
```

---

**Philosophy**: "Templeton commits carefully, rolls back fearlessly!"
