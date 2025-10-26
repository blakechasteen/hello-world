# Promptly - Project Overview

## What is Promptly?

**Promptly manage your prompts** - A production-ready CLI tool for managing AI prompts with Git-like versioning, branching, evaluation, and chaining.

## Key Stats

- **1,300+ lines** of Python code
- **All tests passing** ✓
- **Complete documentation** (5 guides)
- **Working examples** included
- **Production-ready** right now

## Core Features

### ✅ Version Control
Every prompt change is tracked automatically with commit hashes, creating a complete audit trail.

### ✅ Branching
Create unlimited branches to experiment with prompt variants without affecting your main prompts.

### ✅ Evaluation
Define test cases in JSON/YAML and run automated evaluations to ensure prompt quality.

### ✅ Chaining
Build multi-step workflows where prompts feed into each other for complex tasks.

### ✅ Local Storage
All data stored locally in SQLite + YAML - no cloud dependencies.

### ✅ Simple CLI
Intuitive command-line interface with color-coded output and helpful error messages.

## Quick Start

```bash
pip install -e .
promptly init
promptly add hello "Hello, {name}!"
promptly get hello
```

## File Structure

```
promptly/
├── promptly.py              # Main CLI (1,300 lines)
├── test_promptly.py         # Test suite
├── demo.sh                  # Interactive demo
├── setup.py                 # Installation
├── requirements.txt         # Dependencies
├── START_HERE.md           # Main guide
├── README.md               # Full docs
├── QUICKSTART.md           # 5-min tutorial
├── TUTORIAL.md             # Real examples
├── CHANGELOG.md            # Version history
└── examples/
    ├── test_cases.json     # Sample tests
    ├── chain_input.yaml    # Sample inputs
    └── advanced_integration.py  # LLM examples
```

## Commands

```bash
# Repository
promptly init
promptly list
promptly log

# Prompts
promptly add NAME "content"
promptly get NAME
promptly get NAME --version 2

# Branching
promptly branch NAME
promptly checkout NAME

# Evaluation
promptly eval run NAME tests.json

# Chaining
promptly chain create NAME step1 step2
promptly chain run NAME input.yaml
```

## Use Cases

1. **Prompt Engineering** - Track evolution of prompts
2. **A/B Testing** - Test different approaches
3. **Content Pipelines** - Multi-stage workflows
4. **Team Collaboration** - Multiple developers
5. **Production Deployment** - Safe rollout
6. **Quality Assurance** - Automated testing

## Technical Architecture

- **Language**: Python 3.7+
- **CLI Framework**: Click
- **Database**: SQLite3
- **Config**: YAML
- **Hashing**: SHA-256
- **Storage**: Local filesystem

## Database Schema

- `prompts` - Prompt versions and content
- `branches` - Branch metadata
- `evaluations` - Test results
- `chains` - Workflow definitions
- `config` - System configuration

## Integration

Works with any LLM:

```python
from promptly import Promptly

promptly = Promptly()
prompt = promptly.get('summarizer')
formatted = prompt['content'].format(text="...")

# Use with your LLM
response = your_llm.complete(formatted)
```

## Testing

Run comprehensive test suite:

```bash
python3 test_promptly.py
```

All tests pass! ✓

## Demo

See all features in action:

```bash
./demo.sh
```

## Documentation

- **START_HERE.md** - Quick overview (5 min)
- **README.md** - Complete docs (20 min)
- **QUICKSTART.md** - Fast tutorial (5 min)
- **TUTORIAL.md** - Real examples (30 min)
- **CHANGELOG.md** - Version history

## What's Next?

### Immediate
1. `pip install -e .`
2. `promptly init`
3. `./demo.sh`

### This Week
- Add your prompts
- Create test cases
- Build a chain
- Integrate with your LLM

### Future Features
- Branch merging
- Remote repositories
- Built-in LLM integrations
- Web UI
- VSCode extension

## Why Promptly?

✨ **Complete** - Not just code, full ecosystem  
✨ **Tested** - Comprehensive test coverage  
✨ **Documented** - 5 detailed guides  
✨ **Professional** - Clean, production code  
✨ **Extensible** - Easy to add features  
✨ **Local** - No cloud dependencies  

## The Name

**Promptly** = Prompt + ly (adverb)

Perfect wordplay: "**Promptly manage your prompts**"

Implies speed, efficiency, and immediate action!

---

**Start now:** `promptly init` 🚀

*Version: 0.1.0*  
*License: MIT*
