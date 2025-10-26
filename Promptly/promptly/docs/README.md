# Promptly 🚀

**Promptly manage your prompts** - A powerful CLI tool with versioning, branching, evaluation, and chaining.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
promptly init
promptly add summarizer "Summarize: {text}"
promptly get summarizer
promptly list
```

## Features

- 📦 **Versioning** - Track all prompt changes
- 🌿 **Branching** - Experiment with variants  
- ✅ **Evaluation** - Test against test cases
- ⛓️ **Chaining** - Multi-step workflows
- 💾 **Local Storage** - SQLite + YAML
- 🎯 **Simple CLI** - Easy to use

## Commands

```bash
# Repository
promptly init                    # Initialize repo
promptly list                    # List prompts
promptly log                     # View history

# Prompts  
promptly add NAME "content"      # Add/update prompt
promptly get NAME                # Get prompt
promptly get NAME --version 2    # Get specific version

# Branching
promptly branch NAME             # Create branch
promptly checkout NAME           # Switch branch

# Evaluation
promptly eval run NAME tests.json

# Chaining
promptly chain create NAME step1 step2
promptly chain run NAME input.yaml
```

## Example: Content Pipeline

```bash
promptly add research "Research: {topic}"
promptly add outline "Outline: {output}"
promptly add draft "Draft: {output}"
promptly add edit "Edit: {output}"

promptly chain create pipeline research outline draft edit
promptly chain run pipeline input.yaml
```

## Example: A/B Testing

```bash
promptly branch variant-a
promptly checkout variant-a  
promptly add ad "Emotional approach: {product}"

promptly checkout main
promptly branch variant-b
promptly checkout variant-b
promptly add ad "Data-driven approach: {product}"

# Test both variants
promptly eval run ad tests.json
```

## Integration with LLMs

```python
from promptly import Promptly

promptly = Promptly()
prompt_data = promptly.get('summarizer')
formatted = prompt_data['content'].format(text="Your text...")

# Use with any LLM API
response = your_llm.complete(formatted)
```

## Documentation

- **START_HERE.md** - Quick overview
- **QUICKSTART.md** - 5-minute tutorial
- **TUTORIAL.md** - Real-world examples
- **examples/** - Code examples

## Run Demo

```bash
./demo.sh
```

## Run Tests

```bash
python3 test_promptly.py
```

## License

MIT

---

**Promptly manage your prompts. Built with ❤️**
