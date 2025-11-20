# Pre-Commit Quick Start

**TL;DR**: Install once, then code normally. Automatic quality checks on every commit.

## 30-Second Setup

```bash
# Install once
pip install pre-commit
pre-commit install

# Now code normally - pre-commit runs automatically on git commit!
```

## Daily Usage

```bash
# Just commit as usual - checks run automatically
git add .
git commit -m "feat: Add new feature"

# Pre-commit auto-fixes formatting, shows errors if any
# If it auto-fixed, add again and commit:
git add .
git commit -m "feat: Add new feature"
```

## What Gets Checked

| Check | Action | Example |
|-------|--------|---------|
| Trailing whitespace | Auto-remove | `  \n` → `\n` |
| Missing newline at EOF | Auto-add | File without `\n` at end → adds it |
| Python syntax | Error if bad | `if x = 1:` ❌ |
| Code formatting | Auto-fix | `func(a,b)` → `func(a, b)` |
| Import sorting | Auto-fix | Sorts imports alphabetically |
| Linting | Auto-fix minor | Unused imports, undefined names |
| Commit message | Error if bad | `fix stuff` ❌ → `fix(scope): description` ✅ |
| No push to main | Error | Blocks commits to main/master |

## Common Commands

```bash
# Check status
pre-commit run

# Fix all files
pre-commit run --all-files

# Run specific check
pre-commit run black --all-files
pre-commit run ruff --all-files

# Type checking (slow, optional)
pre-commit run mypy --hook-stage manual --all-files

# Skip checks (emergency only!)
git commit --no-verify
```

## Commit Message Format

```bash
# Correct ✅
git commit -m "feat(embeddings): Add multi-scale support"
git commit -m "fix(policy): Correct Thompson Sampling"
git commit -m "docs: Improve quickstart"

# Incorrect ❌
git commit -m "fix stuff"
git commit -m "update code"
git commit -m "wip"
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Hooks not running | `pre-commit install` |
| Slow commits | Run `pre-commit run --all-files` once to cache |
| Code rejected | Read error, usually auto-fixable |
| Commit message rejected | Use format: `type(scope): description` |
| Want to skip checks | `git commit --no-verify` (not recommended) |

## Files

- **`.pre-commit-config.yaml`** - Main configuration (what to check)
- **`pyproject.toml`** - Tool settings (how to check)
- **`.editorconfig`** - Editor consistency
- **`.markdownlintrc`** - Markdown rules
- **`CONTRIBUTING.md`** - Full developer guide
- **`CODE_QUALITY_GUIDE.md`** - Detailed documentation

## More Info

- **Quick questions?** → See **CONTRIBUTING.md** "Pre-Commit Hooks" section
- **Detailed setup?** → See **CODE_QUALITY_GUIDE.md**
- **Configuration?** → See **pyproject.toml** and **.pre-commit-config.yaml**

---

**Status**: ✅ Ready to use
**Time to setup**: 2 minutes
**Impact on workflow**: None (automatic)
