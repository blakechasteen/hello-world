# Code Quality

Tools and workflows for keeping the HoloLoom codebase clean.

## Setup

```bash
pip install -e ".[dev]"
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg
```

## What Runs on Every Commit

| Hook | Action | Speed |
|------|--------|-------|
| `trailing-whitespace` | Auto-remove | <10ms |
| `end-of-file-fixer` | Auto-add newline | <10ms |
| `black` | Auto-format code | ~500ms |
| `isort` | Auto-sort imports | ~100ms |
| `ruff` | Lint + partial auto-fix | ~150ms |
| `markdownlint` | Auto-fix markdown | ~200ms |
| `conventional-pre-commit` | Validate commit msg | <50ms |
| `no-commit-to-branches` | Block main/master | <10ms |

Total commit overhead: ~1-2 seconds.

## Commit Message Format

```bash
# Required format: type(scope): description
git commit -m "feat(memory): add hybrid retrieval strategy"
git commit -m "fix(policy): correct Thompson Sampling update"
git commit -m "docs: update quickstart guide"

# Types: feat, fix, docs, test, perf, refactor, style, chore
```

## Common Commands

```bash
# Auto-fix all files
pre-commit run --all-files

# Type checking (slow, manual)
mypy hololoom/ --ignore-missing-imports

# Security scan (slow, manual)
bandit hololoom/ -ll

# Before pushing a PR
pre-commit run --all-files
pytest hololoom/tests/ -v
```

## Tool Configuration

All tools read from `pyproject.toml`:
- **Black**: line-length 100, Python 3.10+
- **Ruff**: rules F, E, W, I, C, UP, B, A
- **isort**: Black-compatible profile
- **mypy**: strict warnings, `ignore_missing_imports = true`
- **pytest**: `asyncio_mode = "auto"`, 3-tier markers (unit/integration/e2e)

## Editor Setup

Install the EditorConfig extension for your editor. The `.editorconfig` file ensures:
- LF line endings
- UTF-8 encoding
- 4 spaces (Python), 2 spaces (YAML/JSON/markdown)
- 100 char line length
- Trim trailing whitespace

## CI/CD

GitHub Actions workflow at `.github/workflows/code-quality.yml` runs all hooks (including slow mypy/bandit) on every PR.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Hooks not running | `pre-commit install` |
| Code auto-fixed, commit failed | `git add .` then re-commit |
| Commit message rejected | Use `type(scope): description` format |
| Slow first run | Normal — hooks cache after first execution |
| Need to bypass (emergency) | `git commit --no-verify` |
