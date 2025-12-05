# Code Quality Guide for HoloLoom

**Philosophy**: "Reliable Systems: Safety First"

This guide explains the automated code quality system for HoloLoom and how to work with it effectively.

## Overview

HoloLoom uses a comprehensive code quality system with automated checks:

```
Local Development          CI/CD Pipeline
    ↓                          ↓
pre-commit (fast)         pre-commit (complete)
   ↓                           ↓
git commit                 GitHub Actions
   ↓                           ↓
editor (EditorConfig)      All hooks + tests
   ↓                           ↓
pyproject.toml (config)    Code merged to main
```

## Configuration Files

### 1. `.pre-commit-config.yaml` (257 lines)

The main configuration file for automated code quality checks on every commit.

**What it does**:
- Runs before each `git commit`
- Auto-fixes formatting issues (Black, isort, trailing-whitespace)
- Detects syntax errors and security issues
- Validates commit messages (conventional-commit format)
- Prevents accidental commits to main/master

**Key hooks**:

| Hook | Speed | Stage | Auto-Fix |
|------|-------|-------|----------|
| `trailing-whitespace` | <10ms | commit | ✅ Yes |
| `end-of-file-fixer` | <10ms | commit | ✅ Yes |
| `check-yaml` | <50ms | commit | ❌ No (show error) |
| `check-json` | <50ms | commit | ✅ Yes |
| `check-ast` | <50ms | commit | ❌ No (show error) |
| `black` | 500ms-1s | commit | ✅ Yes |
| `ruff` | 100-200ms | commit | ✅ Partial (--fix) |
| `isort` | 100-200ms | commit | ✅ Yes |
| `markdownlint` | 100-300ms | commit | ✅ Yes |
| `conventional-pre-commit` | <50ms | commit-msg | ❌ No (show error) |
| `no-commit-to-branches` | <10ms | commit | ❌ No (block) |
| `mypy` | 5-10s | manual | ❌ No (type errors) |
| `bandit` | 2-5s | manual | ❌ No (security issues) |

**Total commit time**: ~1-2 seconds on average
**Run manually**: `pre-commit run --all-files`

### 2. `pyproject.toml` (252 lines)

Modern Python project configuration with tool settings.

**Contains**:
- **Black**: Code formatting (line-length: 100)
- **ruff**: Fast linting (F, E, W, I, C, UP, B, A rules)
- **isort**: Import sorting (Black-compatible profile)
- **mypy**: Type checking (optional, strict settings)
- **pytest**: Test configuration and markers
- **coverage**: Code coverage settings

**Usage**:
- Tools read configuration from this file automatically
- No need for separate `.flake8`, `.isort.cfg`, `mypy.ini`
- Single source of truth for tool configuration

### 3. `.markdownlintrc` (65 lines)

Markdown linting configuration.

**Key rules**:
- **MD013**: Line length 100 characters
- **MD007**: List indent 2 spaces
- **MD029**: Ordered lists (1, 2, 3...)
- **MD033**: Allow specific HTML tags (br, details, kbd)

**Enforces**:
- Consistent markdown formatting
- Proper heading hierarchy
- Consistent list styles

### 4. `.editorconfig` (115 lines)

Universal editor configuration for IDE consistency.

**Supported editors**:
- VS Code (with EditorConfig extension)
- PyCharm / IntelliJ
- Vim / Neovim (with editorconfig plugin)
- Sublime Text
- Atom
- Emacs

**Settings**:
- Line endings: LF (Unix-style)
- Encoding: UTF-8
- Indentation: 4 spaces (Python), 2 spaces (YAML, JSON, markdown)
- Line length: 100 characters
- Trim trailing whitespace

**Install**:
- VS Code: `ext install EditorConfig.EditorConfig`
- PyCharm: Built-in, enable in Settings → Editor → Code Style

### 5. `CONTRIBUTING.md` (updated)

Developer guide with setup instructions.

**New section**: "Pre-Commit Hooks" with:
- Installation instructions
- Common workflows
- Troubleshooting
- CI/CD integration

## Setup Instructions

### Initial Setup (One-Time)

```bash
# 1. Clone repository
git clone https://github.com/blakechasteen/mythRL.git
cd mythRL

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or: .venv\Scripts\activate on Windows

# 3. Install dependencies + dev tools
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg

# 5. (Optional) Run on all existing files
pre-commit run --all-files
```

### After First Commit

Pre-commit runs automatically:
```bash
git add .
git commit -m "feat: Add new feature"

# Output:
# trailing-whitespace...........................................Passed
# end-of-file-fixer.............................................Passed
# black.........................................................Passed
# isort.........................................................Passed
# ruff.........................................................Passed
# markdownlint...................................................Passed
# conventional-pre-commit........................................Passed
```

## Common Workflows

### Normal Development

```bash
# 1. Make changes
# vim HoloLoom/my_feature.py

# 2. Commit (pre-commit runs automatically)
git add HoloLoom/my_feature.py
git commit -m "feat(embeddings): Add new feature"

# 3. Pre-commit auto-fixes issues, if needed:
#    - Removes trailing whitespace
#    - Reformats with Black
#    - Sorts imports with isort
#    - Shows any syntax/security errors

# 4. If auto-fixes were applied, add again and retry
git add .
git commit -m "feat(embeddings): Add new feature"
```

### Run Type Checking

Type checking is slow, so it's manual by default:

```bash
# Type check before pushing
pre-commit run mypy --hook-stage manual --all-files

# Or with specific files
mypy HoloLoom/my_feature.py --ignore-missing-imports
```

### Run Security Checks

Security checking is slow, so it's manual by default:

```bash
# Security check before pushing
pre-commit run bandit --hook-stage manual --all-files

# Or with specific rules
bandit HoloLoom/ -ll  # Only medium/high severity
```

### Fix All Files at Once

For bulk cleanup:

```bash
# Auto-fix formatting on entire codebase
pre-commit run --all-files

# Then commit
git add .
git commit -m "style: Auto-format code"
```

### Emergency Bypass (Use Sparingly)

```bash
# Skip pre-commit checks (not recommended!)
git commit --no-verify -m "hotfix: Critical bug"
```

## Commit Message Format

Pre-commit enforces conventional commits:

**Format**: `<type>(<scope>): <subject>`

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation change
- `test`: Test changes
- `perf`: Performance improvement
- `refactor`: Code refactoring
- `style`: Formatting/style change
- `chore`: Build/tooling changes

**Examples**:
```bash
# Good
git commit -m "feat(embeddings): Add multi-scale support"
git commit -m "fix(policy): Correct Thompson Sampling update"
git commit -m "docs(readme): Improve quickstart guide"

# Bad (will be rejected)
git commit -m "fix stuff"
git commit -m "update code"
git commit -m "wip"
```

## Tool Configurations Summary

### Black (Formatter)

**Line length**: 100 characters
**Indentation**: 4 spaces
**Quotes**: Double quotes (")
**Run**: `black HoloLoom/ --line-length 100`

### Ruff (Linter)

**Rules**: F, E, W, I, C, UP, B, A
**Line length**: 100 characters
**Auto-fix**: `ruff check --fix HoloLoom/`
**Check only**: `ruff check HoloLoom/`

### isort (Import Sorter)

**Profile**: Black-compatible
**Line length**: 100 characters
**Run**: `isort HoloLoom/ --profile black`

### mypy (Type Checker)

**Target**: Python 3.10+
**Settings**: No implicit optional, warn on return_any
**Run**: `mypy HoloLoom/ --ignore-missing-imports`

### Markdownlint

**Line length**: 100 characters
**List indent**: 2 spaces
**Run**: `markdownlint . --fix`

## Performance Characteristics

### Local Commit (Typical)

```
Code check:        ~50ms
Black formatting:  ~500ms
ruff linting:      ~150ms
isort sorting:     ~100ms
Markdownlint:      ~200ms
Commit message:    ~50ms
─────────────────────
Total:             ~1-2 seconds
```

### First Run (All Files)

```bash
pre-commit run --all-files
# Duration: 5-15 seconds (depending on codebase size)
```

### Type Checking (Manual)

```bash
pre-commit run mypy --hook-stage manual
# Duration: 5-10 seconds
```

### Security Analysis (Manual)

```bash
pre-commit run bandit --hook-stage manual
# Duration: 2-5 seconds
```

## Integration with CI/CD

Add to `.github/workflows/code-quality.yml`:

```yaml
name: Code Quality

on: [push, pull_request]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - uses: pre-commit/action@v3
        with:
          extra_dependencies: ["safety"]
```

This runs all hooks (including slow checks) on every PR.

## Troubleshooting

### Pre-commit hooks not running

```bash
# Verify installation
cat .git/hooks/pre-commit

# Reinstall
pre-commit install
pre-commit install --hook-type commit-msg
```

### Hooks running but not fixing

```bash
# Verify hooks are configured to auto-fix
grep "args:" .pre-commit-config.yaml

# Run manually with fixes
pre-commit run --all-files
```

### Slow commits

```bash
# Profile which hooks are slow
time pre-commit run

# Skip slow hooks locally (run in CI instead)
# They're already marked as "manual" stage
```

### Editor not respecting formatting

```bash
# Ensure EditorConfig extension is installed
# VS Code: ext install EditorConfig.EditorConfig
# PyCharm: Built-in (enable in Settings)

# Or format file before committing
black HoloLoom/file.py
isort HoloLoom/file.py
```

### Commit message rejected

```bash
# Error: "Please enter a commit message in conventional commit format"

# Solution: Use correct format
git commit -m "feat(scope): description"
# NOT: git commit -m "fix stuff"

# Or bypass (not recommended)
git commit --no-verify
```

## Best Practices

### Before Pushing a PR

```bash
# 1. Format code
pre-commit run --all-files

# 2. Run type checking
pre-commit run mypy --hook-stage manual --all-files

# 3. Run security checks
pre-commit run bandit --hook-stage manual --all-files

# 4. Run tests
pytest HoloLoom/tests/ -v

# 5. Push
git push origin feature-branch
```

### Code Review Checklist

- ✅ Pre-commit hooks passed (auto on commit)
- ✅ Type checking passed (`mypy`)
- ✅ Security checks passed (`bandit`)
- ✅ Tests added and passing
- ✅ Commit messages follow conventional-commit
- ✅ No commits to main/master branch

### Documentation Best Practices

- ✅ Markdown formatted with markdownlint
- ✅ Lines wrap at 100 characters
- ✅ Proper heading hierarchy (# → ## → ###)
- ✅ Code examples tested/verified
- ✅ Links to relevant files included

## Configuration Files at a Glance

```
HoloLoom/
├── .pre-commit-config.yaml      # Main pre-commit hooks (257 lines)
├── pyproject.toml               # Tool configs (252 lines)
├── .markdownlintrc              # Markdown rules (65 lines)
├── .editorconfig                # Editor settings (115 lines)
├── CONTRIBUTING.md              # Updated with pre-commit section
├── CODE_QUALITY_GUIDE.md        # This file
└── setup.py                      # Original setup (unchanged)
```

## References

- **Pre-commit Documentation**: https://pre-commit.com/
- **Black Documentation**: https://black.readthedocs.io/
- **Ruff Documentation**: https://docs.astral.sh/ruff/
- **isort Documentation**: https://pycqa.github.io/isort/
- **mypy Documentation**: https://mypy.readthedocs.io/
- **EditorConfig**: https://editorconfig.org/
- **Conventional Commits**: https://www.conventionalcommits.org/

## Support

For issues or questions:
- Check `.pre-commit-config.yaml` comments (detailed explanations)
- Run `pre-commit --help`
- Check CONTRIBUTING.md "Pre-Commit Hooks" section
- Open a GitHub issue with logs

---

**Last Updated**: 2025-11-15
**Python Version**: 3.8+ (3.10+ recommended)
**Status**: Production Ready
