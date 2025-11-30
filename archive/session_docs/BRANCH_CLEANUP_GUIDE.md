# Branch Cleanup Guide
## Practical Scripts and Procedures for Branch Management

**Created:** November 19, 2025
**Purpose:** Provide scripts and procedures for safely merging, archiving, and cleaning up branches

---

## 📋 Prerequisites

Before running any cleanup operations:

```bash
# 1. Ensure you're on master with latest changes
git checkout master
git pull origin master

# 2. Create backup of current state
git tag backup/pre-cleanup-$(date +%Y%m%d)
git push origin --tags

# 3. Ensure all tests pass on master
pytest HoloLoom/tests/ -v
```

---

## 🚀 Safe Merge Procedures

### Procedure 1: Documentation-Only Merges (Zero Risk)

For branches that only add documentation (no code changes):

```bash
#!/bin/bash
# safe-doc-merge.sh - Merge documentation-only branches

BRANCHES=(
    "claude/12-factor-agents-01NfmppAMNn6JcqaNkZB3tC2"
    "claude/contract-first-prompting-01C12FNhza2QfLhPEzhm6yKh"
    "claude/plan-skills-integration-0111BHXCrujpp6og5Jm5smcN"
)

for BRANCH in "${BRANCHES[@]}"; do
    echo "Merging: $BRANCH"

    # Check if branch exists
    if ! git ls-remote --exit-code --heads origin "$BRANCH" > /dev/null 2>&1; then
        echo "⚠️  Branch $BRANCH not found, skipping"
        continue
    fi

    # Fetch latest
    git fetch origin "$BRANCH"

    # Create test branch
    git checkout -b "test/${BRANCH##*/}"
    git merge "origin/$BRANCH" --no-ff -m "Merge documentation from $BRANCH"

    # Verify no Python files changed
    PYTHON_CHANGES=$(git diff master --name-only | grep -c "\.py$" || true)

    if [ "$PYTHON_CHANGES" -eq 0 ]; then
        echo "✅ No Python files changed, safe to merge"
        git checkout master
        git merge "origin/$BRANCH" --no-ff -m "docs: Merge $BRANCH"
        git push origin master
        echo "✅ Merged: $BRANCH"
    else
        echo "⚠️  Python files changed, manual review needed"
        git checkout master
    fi

    # Cleanup test branch
    git branch -D "test/${BRANCH##*/}"
done
```

---

### Procedure 2: Feature Branch Merge (With Testing)

For branches with code changes that need validation:

```bash
#!/bin/bash
# safe-feature-merge.sh - Merge feature branches with testing

merge_feature_branch() {
    local BRANCH=$1
    local FEATURE_NAME=${BRANCH##*/}

    echo "========================================="
    echo "Testing feature branch: $BRANCH"
    echo "========================================="

    # Fetch latest
    git fetch origin "$BRANCH"

    # Create test branch
    git checkout -b "test/$FEATURE_NAME"

    # Merge
    if ! git merge "origin/$BRANCH" --no-ff -m "Test merge: $BRANCH"; then
        echo "❌ Merge conflicts detected. Manual resolution needed."
        git merge --abort
        git checkout master
        git branch -D "test/$FEATURE_NAME"
        return 1
    fi

    # Run tests
    echo "Running tests..."
    if ! pytest HoloLoom/tests/ -v --tb=short; then
        echo "❌ Tests failed. Branch needs fixes."
        git checkout master
        git branch -D "test/$FEATURE_NAME"
        return 1
    fi

    # Check for new files that might need documentation
    NEW_PY_FILES=$(git diff master --name-only --diff-filter=A | grep "\.py$" || true)
    if [ -n "$NEW_PY_FILES" ]; then
        echo "📝 New Python files added:"
        echo "$NEW_PY_FILES"
        read -p "Update CLAUDE.md? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Please update CLAUDE.md before merging"
            git checkout master
            git branch -D "test/$FEATURE_NAME"
            return 1
        fi
    fi

    # All checks passed, merge to master
    echo "✅ All checks passed"
    read -p "Proceed with merge to master? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git checkout master
        git merge "origin/$BRANCH" --no-ff -m "feat: Merge $BRANCH"
        git push origin master
        echo "✅ Successfully merged: $BRANCH"
    else
        echo "Merge cancelled"
        git checkout master
    fi

    # Cleanup test branch
    git branch -D "test/$FEATURE_NAME"
}

# Usage examples:
# merge_feature_branch "claude/filesystem-rag-ingestion-01LZ4UKcM5K4jbae4C2mSVKV"
# merge_feature_branch "claude/tenant-isolation-pii-018g5itHiyhrwcgFdSoFJWXa"
```

---

### Procedure 3: Large Changeset Merge (Extra Careful)

For branches with extensive changes (like memory enhancements):

```bash
#!/bin/bash
# careful-merge.sh - Extra careful merge for large changesets

careful_merge() {
    local BRANCH=$1
    local FEATURE_NAME=${BRANCH##*/}

    echo "========================================="
    echo "CAREFUL MERGE: $BRANCH"
    echo "========================================="

    # Fetch and checkout branch locally
    git fetch origin "$BRANCH"
    git checkout -b "$FEATURE_NAME" "origin/$BRANCH"

    # Show summary of changes
    echo "Files changed vs master:"
    git diff master --stat
    echo ""

    read -p "Continue with review? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        git checkout master
        git branch -D "$FEATURE_NAME"
        return 1
    fi

    # Show new files
    echo "New files added:"
    git diff master --name-only --diff-filter=A
    echo ""

    # Show modified files
    echo "Modified files:"
    git diff master --name-only --diff-filter=M
    echo ""

    # Check for potential conflicts with recent changes
    echo "Checking for potential conflicts..."
    git checkout master
    git merge --no-commit --no-ff "$FEATURE_NAME"

    if [ $? -ne 0 ]; then
        echo "❌ Merge conflicts detected:"
        git status --short
        echo ""
        echo "Conflicts need manual resolution"
        git merge --abort
        git branch -D "$FEATURE_NAME"
        return 1
    fi

    echo "No conflicts detected"
    git merge --abort  # Abort test merge

    # Create integration test branch
    git checkout -b "integration/$FEATURE_NAME"
    git merge "$FEATURE_NAME" --no-ff -m "Integration test: $BRANCH"

    # Run full test suite
    echo "Running unit tests..."
    pytest HoloLoom/tests/unit/ -v || {
        echo "❌ Unit tests failed"
        git checkout master
        git branch -D "integration/$FEATURE_NAME"
        git branch -D "$FEATURE_NAME"
        return 1
    }

    echo "Running integration tests..."
    pytest HoloLoom/tests/integration/ -v || {
        echo "❌ Integration tests failed"
        git checkout master
        git branch -D "integration/$FEATURE_NAME"
        git branch -D "$FEATURE_NAME"
        return 1
    }

    echo "Running e2e tests..."
    pytest HoloLoom/tests/e2e/ -v || {
        echo "⚠️  E2E tests failed (may be acceptable)"
    }

    # Performance check
    echo "Running quick performance check..."
    PYTHONPATH=. python -c "
from HoloLoom import HoloLoom
import asyncio
import time

async def test():
    async with HoloLoom() as loom:
        start = time.time()
        await loom.experience('test memory')
        memories = await loom.recall('test')
        duration = time.time() - start
        print(f'Basic operation took {duration*1000:.2f}ms')
        assert duration < 1.0, 'Performance regression detected'

asyncio.run(test())
"

    if [ $? -ne 0 ]; then
        echo "⚠️  Performance regression detected"
    fi

    # Final review
    echo ""
    echo "========================================="
    echo "Review Summary"
    echo "========================================="
    git diff master --shortstat
    echo ""

    read -p "Proceed with merge to master? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git checkout master
        git merge "$FEATURE_NAME" --no-ff -m "feat: Merge $BRANCH

See BRANCH_REVIEW_AND_STATUS.md for details"
        git push origin master
        echo "✅ Successfully merged: $BRANCH"
    else
        echo "Merge cancelled"
        git checkout master
    fi

    # Cleanup
    git branch -D "integration/$FEATURE_NAME"
    git branch -D "$FEATURE_NAME"
}

# Usage:
# careful_merge "claude/enhance-hololoom-memory-01YFMtm1vRKUmwaNAigKR95q"
```

---

## 🗑️ Branch Archival Procedures

### Procedure 4: Archive Merged Branches

For branches already merged or no longer needed:

```bash
#!/bin/bash
# archive-branches.sh - Archive old branches with tags

archive_branch() {
    local BRANCH=$1
    local TAG_NAME="archive/${BRANCH##*/}"

    echo "Archiving: $BRANCH"

    # Create tag for reference
    git tag "$TAG_NAME" "origin/$BRANCH"
    git push origin "$TAG_NAME"

    # Delete remote branch
    git push origin --delete "$BRANCH"

    echo "✅ Archived: $BRANCH (tag: $TAG_NAME)"
}

# Archive old review branches
REVIEW_BRANCHES=(
    "claude/review-updates-011CUVGwFWS9AwV7dCtR8W7q"
    "claude/review-updates-01G1dZsbn7iMATnPMUTbyCVP"
    "claude/review-unfinished-tasks-011CUVLxdWWm4VaPzv4mMf4F"
    "claude/review-hololoom-writing-011CUsZxiffPYPQ7fy8ciVQC"
    "claude/code-review-updates-011CUSAqCnMcYkQZ8X4r7UWz"
)

echo "Archiving review branches..."
for BRANCH in "${REVIEW_BRANCHES[@]}"; do
    archive_branch "$BRANCH"
done

# Archive old codex branches
CODEX_BRANCHES=$(git branch -r | grep "origin/codex/" | sed 's/origin\///')
echo "Archiving codex branches..."
for BRANCH in $CODEX_BRANCHES; do
    archive_branch "$BRANCH"
done
```

---

### Procedure 5: List Branches by Status

Diagnostic script to understand branch status:

```bash
#!/bin/bash
# branch-status.sh - Analyze branch status

echo "========================================="
echo "Branch Status Report"
echo "========================================="
echo ""

# Merged branches
echo "Branches already merged to master:"
git branch -r --merged origin/master | grep "claude/" | wc -l
echo ""

# Unmerged branches
echo "Unmerged branches:"
UNMERGED=$(git branch -r --no-merged origin/master | grep -E "claude/|codex/")
echo "$UNMERGED" | wc -l
echo ""

# Recent branches (last 7 days)
echo "Branches updated in last 7 days:"
git for-each-ref --sort=-committerdate refs/remotes/ \
    --format='%(committerdate:short) %(refname:short)' \
    | grep -E "claude/|codex/" \
    | head -10
echo ""

# Old branches (>30 days)
echo "Branches not updated in 30+ days:"
CUTOFF_DATE=$(date -d '30 days ago' +%Y-%m-%d)
git for-each-ref --sort=committerdate refs/remotes/ \
    --format='%(committerdate:short) %(refname:short)' \
    | grep -E "claude/|codex/" \
    | awk -v cutoff="$CUTOFF_DATE" '$1 < cutoff' \
    | wc -l
echo ""

# Branches by category
echo "Branches by category:"
echo "Documentation-only branches:"
for BRANCH in $(git branch -r | grep "claude/" | sed 's/origin\///'); do
    PYTHON_CHANGES=$(git diff master "origin/$BRANCH" --name-only 2>/dev/null | grep -c "\.py$" || true)
    if [ "$PYTHON_CHANGES" -eq 0 ]; then
        echo "  - $BRANCH"
    fi
done
echo ""

echo "Feature branches (with code changes):"
for BRANCH in $(git branch -r | grep "claude/" | sed 's/origin\///'); do
    PYTHON_CHANGES=$(git diff master "origin/$BRANCH" --name-only 2>/dev/null | grep -c "\.py$" || true)
    if [ "$PYTHON_CHANGES" -gt 0 ]; then
        echo "  - $BRANCH (${PYTHON_CHANGES} Python files changed)"
    fi
done
```

---

## 📊 Batch Operations

### Batch Merge Documentation Branches

```bash
#!/bin/bash
# batch-merge-docs.sh - Merge all documentation-only branches at once

# List of doc-only branches to merge
DOC_BRANCHES=(
    "claude/12-factor-agents-01NfmppAMNn6JcqaNkZB3tC2"
    "claude/contract-first-prompting-01C12FNhza2QfLhPEzhm6yKh"
    "claude/plan-skills-integration-0111BHXCrujpp6og5Jm5smcN"
)

# Create merge commit list
MERGE_LIST="merge-list-$(date +%Y%m%d).txt"
> "$MERGE_LIST"

echo "Verifying all branches are documentation-only..."
for BRANCH in "${DOC_BRANCHES[@]}"; do
    PYTHON_CHANGES=$(git diff master "origin/$BRANCH" --name-only 2>/dev/null | grep -c "\.py$" || true)
    if [ "$PYTHON_CHANGES" -gt 0 ]; then
        echo "❌ WARNING: $BRANCH has Python file changes!"
        exit 1
    fi
    echo "✅ $BRANCH is doc-only"
    echo "$BRANCH" >> "$MERGE_LIST"
done

echo ""
echo "All branches verified as documentation-only"
read -p "Proceed with batch merge? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Merge all branches
for BRANCH in "${DOC_BRANCHES[@]}"; do
    echo "Merging: $BRANCH"
    git merge "origin/$BRANCH" --no-ff -m "docs: Merge $BRANCH"
    if [ $? -ne 0 ]; then
        echo "❌ Merge failed for $BRANCH"
        echo "Resolve conflicts and run: git merge --continue"
        exit 1
    fi
done

# Push all at once
git push origin master
echo "✅ All documentation branches merged successfully"
```

---

## 🔍 Quality Checks

### Pre-Merge Checklist Script

```bash
#!/bin/bash
# pre-merge-check.sh - Run all quality checks before merging

check_branch() {
    local BRANCH=$1
    local FAILURES=0

    echo "========================================="
    echo "Pre-Merge Quality Checks: $BRANCH"
    echo "========================================="
    echo ""

    # Checkout branch
    git fetch origin "$BRANCH"
    git checkout -b "check/$BRANCH" "origin/$BRANCH"

    # 1. Test coverage
    echo "1. Running tests..."
    if ! pytest HoloLoom/tests/ -v --cov=HoloLoom --cov-report=term-missing; then
        echo "❌ Tests failed"
        ((FAILURES++))
    else
        echo "✅ Tests passed"
    fi
    echo ""

    # 2. Code style
    echo "2. Checking code style..."
    if command -v black &> /dev/null; then
        black --check HoloLoom/ || {
            echo "⚠️  Code style issues detected (black)"
            ((FAILURES++))
        }
    fi
    echo ""

    # 3. Import checks
    echo "3. Checking imports..."
    if ! python -c "import HoloLoom; print('✅ Import successful')"; then
        echo "❌ Import failed"
        ((FAILURES++))
    fi
    echo ""

    # 4. Documentation check
    echo "4. Checking for undocumented new files..."
    NEW_PY_FILES=$(git diff master --name-only --diff-filter=A | grep "\.py$" || true)
    if [ -n "$NEW_PY_FILES" ]; then
        echo "New Python files:"
        echo "$NEW_PY_FILES"
        if ! git diff master --name-only | grep -q "CLAUDE.md"; then
            echo "⚠️  New files added but CLAUDE.md not updated"
            ((FAILURES++))
        fi
    fi
    echo ""

    # 5. Security check (basic)
    echo "5. Basic security check..."
    if git diff master | grep -i -E "password|secret|api_key|token" | grep -v "test"; then
        echo "⚠️  Potential sensitive data detected"
        ((FAILURES++))
    fi
    echo ""

    # Cleanup
    git checkout master
    git branch -D "check/$BRANCH"

    # Summary
    echo "========================================="
    if [ $FAILURES -eq 0 ]; then
        echo "✅ All checks passed - SAFE TO MERGE"
        return 0
    else
        echo "❌ $FAILURES check(s) failed - REVIEW NEEDED"
        return 1
    fi
}

# Usage:
# check_branch "claude/filesystem-rag-ingestion-01LZ4UKcM5K4jbae4C2mSVKV"
```

---

## 📝 Merge Log Template

Keep a log of all merges:

```bash
# Create merge log
cat > MERGE_LOG_$(date +%Y%m).md << 'EOF'
# Branch Merge Log - [Month Year]

## Merged Branches

### [Date]

**Branch:** [branch-name]
**Type:** [feature/docs/fix/security]
**Merged By:** [name]
**Tests:** [pass/fail/skipped]
**Conflicts:** [yes/no - describe if yes]
**Notes:** [any special notes]

---

### Example Entry

**Date:** 2025-11-19
**Branch:** claude/12-factor-agents-01NfmppAMNn6JcqaNkZB3tC2
**Type:** Documentation
**Merged By:** Blake Chasteen
**Tests:** Passed
**Conflicts:** None
**Notes:** Zero-risk merge, documentation only. Added 12-Factor methodology for AI agents.

EOF
```

---

## 🎯 Recommended Workflow

**Week 1: Documentation Merges**
```bash
# Day 1: Verify and merge doc-only branches
./safe-doc-merge.sh

# Day 2: Update documentation
# - Update CLAUDE.md with merged features
# - Update ARCHITECTURE_VISUAL_MAP.md
```

**Week 2: Feature Merges**
```bash
# Day 1: Filesystem RAG
./safe-feature-merge.sh claude/filesystem-rag-ingestion-01LZ4UKcM5K4jbae4C2mSVKV

# Day 2: Security fixes
./safe-feature-merge.sh claude/tenant-isolation-pii-018g5itHiyhrwcgFdSoFJWXa

# Day 3: Test and validate
pytest HoloLoom/tests/ -v
```

**Week 3: Large Changesets**
```bash
# Full week: Memory enhancements
./careful-merge.sh claude/enhance-hololoom-memory-01YFMtm1vRKUmwaNAigKR95q
# Review conflicts, test thoroughly, update docs
```

**Week 4: Cleanup**
```bash
# Archive old branches
./archive-branches.sh

# Update branch status document
./branch-status.sh > BRANCH_STATUS_REPORT.txt
```

---

## ⚠️ Safety Guidelines

1. **Always create backups** before bulk operations
2. **Test in isolation** - use test branches first
3. **One feature at a time** - don't batch-merge unrelated features
4. **Update documentation** - CLAUDE.md must stay current
5. **Run full test suite** - unit + integration + e2e
6. **Check performance** - ensure no regressions
7. **Archive, don't delete** - create tags before removing branches
8. **Keep logs** - document all merge decisions

---

## 🆘 Recovery Procedures

If something goes wrong:

```bash
# Revert last merge
git revert -m 1 HEAD
git push origin master

# Restore from backup tag
git checkout master
git reset --hard backup/pre-cleanup-20251119
git push origin master --force  # ONLY if no one else has pulled

# Restore archived branch
git checkout -b restored-branch archive/branch-name
git push origin restored-branch
```

---

**Last Updated:** November 19, 2025
**Maintained By:** Repository maintainers
**Review Schedule:** After each major merge wave
