# Promptly Quick Start

Get up and running in 5 minutes!

## Installation

```bash
cd promptly
pip install -e .
```

Or directly:
```bash
pip install click PyYAML
python3 promptly.py --help
```

## 5-Minute Tutorial

### Step 1: Initialize (10 seconds)

```bash
promptly init
```

Creates `.promptly` directory for storage.

### Step 2: Add Prompts (30 seconds)

```bash
promptly add summarizer "Summarize: {text}"
promptly add translator "Translate to Spanish: {text}"
promptly add analyzer "Analyze sentiment: {text}"
```

### Step 3: View Prompts (1 minute)

```bash
# View a prompt
promptly get summarizer

# List all prompts
promptly list

# View history
promptly log --name summarizer
```

### Step 4: Version Control (1 minute)

```bash
# Update prompt (creates v2)
promptly add summarizer "Provide concise summary: {text}"

# View history
promptly log --name summarizer

# Get specific version
promptly get summarizer --version 1
```

### Step 5: Branching (1 minute)

```bash
# Create experimental branch
promptly branch experimental

# Switch to it
promptly checkout experimental

# Make changes
promptly add summarizer "EXPERIMENTAL: {text}"

# Switch back
promptly checkout main
```

### Step 6: Create Chain (1 minute)

```bash
# Add prompts
promptly add outline "Create outline: {topic}"
promptly add draft "Write from outline: {output}"
promptly add edit "Edit: {output}"

# Create chain
promptly chain create writing-flow outline draft edit
```

### Step 7: Evaluation (30 seconds)

Create `test.json`:
```json
[{"inputs": {"text": "AI is amazing!"}, "expected": "positive"}]
```

Run:
```bash
promptly eval run analyzer test.json
```

## Run the Demo

```bash
./demo.sh
```

## What's Next?

- Read **README.md** for full docs
- Check **TUTORIAL.md** for examples
- Integrate with your LLM

## Commands Cheat Sheet

```bash
promptly init                          # Initialize
promptly add NAME "content"            # Add prompt
promptly get NAME                      # Get prompt
promptly list                          # List all
promptly log                           # View history
promptly branch NAME                   # Create branch
promptly checkout NAME                 # Switch branch
promptly eval run NAME tests.json      # Run tests
promptly chain create NAME s1 s2       # Create chain
promptly chain run NAME input.yaml     # Run chain
```

## Tips

1. Use descriptive names: `summarize-technical` not `sum1`
2. Branch often - experiment freely
3. Version everything - it's automatic
4. Test with eval - catch issues early
5. Chain wisely - break tasks into steps

Ready? Start with: `promptly init` 🚀
