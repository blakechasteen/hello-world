# xTerminator Quick Start Guide

## 5-Minute Overview

xTerminator is an automated code fixing system with two phases:

1. **Phase 1**: Classify issues detected by Trough
2. **Phase 4**: Apply fixes as safe, reversible git commits

## Installation

```bash
cd mythRL
python -m pip install -r requirements.txt
```

## Basic Usage

### 1. Classify an Issue (Phase 1)

```python
from xterminator import ClassificationEngine

engine = ClassificationEngine()

# Assume you have an issue from Trough
classification = await engine.classify(
    issue=issue_from_trough,
    full_code=file_content,
    file_path="config.py"
)

proposal = classification.to_fix_proposal(issue, original_code)

# Check if safe to autofix
if proposal.safe_to_autofix:
    print(f"Safe to autofix: {proposal.confidence:.2f} confidence")
else:
    print(f"Requires review: {proposal.risk_level.value} risk")
```

### 2. Apply Fix with Git (Phase 4)

```python
from xterminator import GitApplicator

applicator = GitApplicator(repo_path=".")

result = await applicator.apply_fix(
    file_path="config.py",
    fixed_code=proposal.proposed_code,
    proposal=proposal
)

if result.success:
    print(f"✓ Committed: {result.commit_hash[:8]}")
else:
    print(f"✗ Failed: {result.message}")
```

### 3. Manage Rollback

```python
from xterminator import RollbackManager

rollback = RollbackManager()

# Show history
history = await rollback.get_rollback_history()

# Rollback last fix
result = await rollback.rollback_last(1)

# Rollback all hardcoded_values fixes
result = await rollback.rollback_category("hardcoded_values")
```

## Common Patterns

### Pattern 1: Safe Auto-fix

```python
async def auto_fix(issue, file_content):
    """Automatically fix issues that are safe."""
    classifier = ClassificationEngine()

    # Classify
    classification = await classifier.classify(
        issue=issue,
        full_code=file_content,
        file_path=issue.file_path
    )

    proposal = classification.to_fix_proposal(issue, file_content)

    # Only apply if safe
    if proposal.is_automated():
        applicator = GitApplicator()
        result = await applicator.apply_fix(
            file_path=issue.file_path,
            fixed_code=proposal.proposed_code,
            proposal=proposal
        )
        return result.success

    return False
```

### Pattern 2: Review Required

```python
async def review_fix(issue, file_content):
    """Apply high-risk fixes to feature branch."""
    classifier = ClassificationEngine()
    classification = await classifier.classify(
        issue=issue,
        full_code=file_content,
        file_path=issue.file_path
    )

    proposal = classification.to_fix_proposal(issue, file_content)

    # Requires review - creates feature branch
    if proposal.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
        applicator = GitApplicator()
        result = await applicator.apply_fix(
            file_path=issue.file_path,
            fixed_code=proposal.proposed_code,
            proposal=proposal,
            auto_branch=True  # Create feature branch
        )
        print(f"Created branch for review: {result.details}")
        return result.success

    return False
```

### Pattern 3: Dry-Run First

```python
async def test_fix(issue, file_content):
    """Test fix before applying."""
    applicator = GitApplicator()

    # Dry-run
    result = await applicator.apply_fix(
        file_path=issue.file_path,
        fixed_code=proposal.proposed_code,
        proposal=proposal,
        dry_run=True  # No actual commit
    )

    if result.success:
        print(f"Fix would work: {result.details}")

        # Actually apply
        result = await applicator.apply_fix(
            file_path=issue.file_path,
            fixed_code=proposal.proposed_code,
            proposal=proposal,
            dry_run=False  # Actually commit
        )
```

## Running Tests

```bash
# All tests
pytest xterminator/test_git_applicator.py -v

# Specific test class
pytest xterminator/test_git_applicator.py::TestGitApplicator -v

# Single test
pytest xterminator/test_git_applicator.py::TestGitApplicator::test_initialization -v
```

## Running Demo

```bash
PYTHONPATH=. python xterminator/demo_git_integration.py
```

## Key Types

### RiskLevel
```python
from xterminator import RiskLevel

RiskLevel.LOW        # Safe to autofix
RiskLevel.MEDIUM     # Needs review
RiskLevel.HIGH       # Careful review needed
RiskLevel.CRITICAL   # Manual only
```

### FixStrategy
```python
from xterminator import FixStrategy

FixStrategy.AST      # AST transformation
FixStrategy.TEMPLATE # Template-based
FixStrategy.MANUAL   # Requires human
FixStrategy.SKIP     # Don't fix
```

### FixProposal

```python
proposal.fix_id                # Unique identifier
proposal.issue_category        # Type of issue
proposal.risk_level           # LOW/MEDIUM/HIGH/CRITICAL
proposal.fix_strategy         # How to fix
proposal.confidence           # 0.0-1.0
proposal.original_code        # Before
proposal.proposed_code        # After
proposal.explanation          # Why fix it
proposal.safe_to_autofix      # Can apply automatically
proposal.requires_approval    # Needs review
```

## Understanding Results

### GitOperationResult

```python
result.success               # True if successful
result.message              # Human-readable message
result.commit_hash          # Git commit ID (if successful)
result.details              # Dict with additional info

# Example
if result.success:
    print(f"Applied to {result.details['file']}")
    print(f"Risk: {result.details['risk']}")
    print(f"Confidence: {result.details['confidence']:.2f}")
```

## Risk-Based Branching

**Automatic behavior**:

| Risk Level | Action |
|-----------|--------|
| LOW | Direct commit to main/master |
| MEDIUM | Direct commit to main/master |
| HIGH | Create feature branch |
| CRITICAL | Create feature branch |

```python
# Disable auto-branching
result = await applicator.apply_fix(
    file_path="file.py",
    fixed_code=code,
    proposal=proposal,
    auto_branch=False  # Force direct commit
)
```

## Metadata Tracking

Fixes are automatically tracked in `.xterminator/commits.json`:

```json
{
  "cd75e127abc...": {
    "commit_hash": "cd75e127abc...",
    "timestamp": 1731417600.0,
    "file_path": "config.py",
    "issue_category": "hardcoded_values",
    "risk_level": "low",
    "confidence": 0.92,
    "fix_strategy": "template",
    "fix_id": "FIX_HARDCODED_001"
  }
}
```

## Troubleshooting

### "Not a git repository"
```python
# Solution: Initialize git first
applicator = GitApplicator(repo_path="/path/to/git/repo")
```

### "Repository has uncommitted changes"
```python
# Commit or stash changes first
# Or use dry-run mode to test
result = await applicator.apply_fix(..., dry_run=True)
```

### "Commit already pushed"
```python
# Can't rollback pushed commits by default
# Option 1: Use force flag (not recommended)
result = await rollback.rollback_last(1, force=True)

# Option 2: Create new revert commit instead
result = await rollback.rollback_last(1, force=False)
```

## Performance Tips

1. **Batch Operations**: Apply multiple fixes in loop
2. **Dry-Run First**: Test before actual commit
3. **Leverage Metadata**: Use rollback history for decisions
4. **Clean Commits**: One fix per commit for easy cherry-pick

## Complete Example

```python
import asyncio
from xterminator import (
    ClassificationEngine,
    GitApplicator,
    RollbackManager,
    RiskLevel
)

async def main():
    # Setup
    classifier = ClassificationEngine()
    applicator = GitApplicator()
    rollback_mgr = RollbackManager()

    # Simulate an issue from Trough
    issue = {
        'file_path': 'config.py',
        'line_number': 5,
        'category': 'hardcoded_values',
        'severity': 'medium',
        'message': 'Hardcoded API key'
    }

    file_content = """
import os
API_KEY = 'secret_key_12345'
DEBUG = True
"""

    # Phase 1: Classify
    classification = await classifier.classify(
        issue=issue,
        full_code=file_content,
        file_path='config.py'
    )

    proposal = classification.to_fix_proposal(issue, file_content)

    print(f"Issue: {proposal.issue_category}")
    print(f"Risk: {proposal.risk_level.value}")
    print(f"Confidence: {proposal.confidence:.2f}")

    # Phase 4: Apply if safe
    if proposal.safe_to_autofix:
        fixed_code = """
import os
API_KEY = os.getenv('API_KEY', 'default')
DEBUG = os.getenv('DEBUG', 'False') == 'True'
"""

        result = await applicator.apply_fix(
            file_path='config.py',
            fixed_code=fixed_code,
            proposal=proposal
        )

        if result.success:
            print(f"✓ Applied: {result.commit_hash[:8]}")

            # Show history
            history = await rollback_mgr.get_rollback_history()
            print(f"History: {len(history)} commits")

            # Could rollback
            # result = await rollback_mgr.rollback_last(1)
    else:
        print(f"Requires review: {proposal.risk_level.value} risk")

# Run
asyncio.run(main())
```

## Resources

- **Phase 4 Documentation**: `XTERMINATOR_PHASE_4_GIT_INTEGRATION.md`
- **Implementation Summary**: `XTERMINATOR_IMPLEMENTATION_SUMMARY.md`
- **Test Examples**: `xterminator/test_git_applicator.py`
- **Working Demo**: `xterminator/demo_git_integration.py`

## Command Reference

```python
# Classification
await ClassificationEngine().classify(issue, code, path)

# Apply fix
await GitApplicator().apply_fix(path, code, proposal, dry_run=False, auto_branch=True)

# Rollback
await RollbackManager().rollback_last(n=1, force=False)
await RollbackManager().rollback_category(category)
await RollbackManager().rollback_file(path)
await RollbackManager().get_rollback_history()

# Branch management
await BranchManager().merge_branch(name, delete_after=True)
BranchManager().get_feature_branch_name(proposal)
```

## Success Indicators

You know xTerminator is working when:

- ✓ Fixes are applied with git commits
- ✓ Metadata is saved in .xterminator/commits.json
- ✓ High-risk fixes create feature branches
- ✓ Rollback works without errors
- ✓ Demo completes successfully

---

**Questions?** See full documentation in the repository.
