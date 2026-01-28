# Promptly

**Git for your prompts.** Version control, branch, evaluate, and optimize your AI prompts with a local-first CLI tool.

```bash
pip install promptly
```

---

## Why Promptly?

Your prompts are scattered across files, notebooks, and chat histories. You lose the good ones. You can't track what works. Promptly fixes this.

```bash
# Save a prompt
promptly add greeting "Hello, {{name}}! Welcome to {{project}}."

# Branch for experiments
promptly branch experiments
promptly checkout experiments
promptly add greeting "Hey {{name}}, ready to build {{project}}?"

# Compare versions
promptly log greeting

# Switch back
promptly checkout main
```

## Features

**Version Control** - Git-like branches, commits, and history for every prompt.

**Dual Storage** - Global (`~/.promptly/`) for prompts you use everywhere, local (`.promptly/`) for project-specific ones. Local overrides global.

**LLM Judge** - Evaluate prompts with 12 criteria across 6 judging methods using any LLM backend (Ollama, Claude, GPT).

**Chains** - Multi-step prompt workflows where output from one step feeds into the next.

**Skills** - Reusable prompts with attached files for complex tasks.

**Analytics** - Thompson Sampling recommendations, quality trends, anomaly detection, and HTML dashboards.

**MCP Server** - Use Promptly directly from Claude Desktop.

**VS Code Extension** - Manage prompts from the Command Palette.

## Quick Start

### Install

```bash
# Core (minimal dependencies)
pip install promptly

# All features
pip install promptly[all]

# Specific extras
pip install promptly[rich]        # Rich terminal output
pip install promptly[mcp]         # Claude Desktop MCP server
pip install promptly[anthropic]   # Claude API for evaluation
pip install promptly[ollama]      # Local LLM evaluation
```

### Basic Usage

```bash
# Add prompts
promptly add summarize "Summarize the following text in {{style}} style:\n\n{{text}}"
promptly add code-review "Review this {{language}} code for bugs and improvements:\n\n{{code}}"

# List all prompts
promptly list

# Get a prompt
promptly get summarize

# View history
promptly log summarize

# Delete a prompt
promptly delete old-prompt
```

### Branching

```bash
# Create and switch to a branch
promptly branch v2-experiments
promptly checkout v2-experiments

# Make changes on the branch
promptly add summarize "Provide a {{style}} summary of:\n\n{{text}}\n\nKeep it under {{words}} words."

# List branches
promptly branches

# Switch back to main
promptly checkout main
```

### Chains (Multi-Step Workflows)

```bash
# Create a chain
promptly chain create research-flow \
  --step "research:findings" \
  --step "summarize:summary" \
  --step "format:output"

# Run a chain
promptly chain run research-flow --input "topic=quantum computing"
```

### Skills (Prompts + Files)

```bash
# Add a skill with attached files
promptly skill add code-analyzer \
  --description "Analyze code quality" \
  --template "Analyze this code:\n\n{{code}}" \
  --file ./src/main.py

# Run a skill
promptly skill run code-analyzer
```

### Evaluation

```bash
# Evaluate a prompt with LLM Judge
promptly eval summarize --input "text=Hello world" --criteria quality,relevance

# Compare two prompts
promptly analytics compare summarize-v1 summarize-v2

# Get Thompson Sampling recommendation
promptly analytics recommend summarization

# View quality trends
promptly analytics trend summarize

# Generate analytics dashboard
promptly analytics dashboard
```

### Import/Export

```bash
# Export to YAML
promptly export summarize > summarize.yaml

# Import from YAML
promptly import prompts.yaml
```

## Python API

```python
from promptly import Promptly

# Initialize (auto-detects project root)
p = Promptly()
p.connect()

# Add a prompt
p.add("greeting", "Hello, {{name}}!")

# Get and render
prompt = p.get("greeting")
rendered = prompt.render(name="World")
print(rendered)  # "Hello, World!"

# Branch operations
p.create_branch("experiments")
p.checkout("experiments")
p.add("greeting", "Hey {{name}}, what's up?")

# List all prompts (merged local + global view)
prompts = p.list_prompts()
```

## MCP Server (Claude Desktop)

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "promptly": {
      "command": "promptly",
      "args": ["mcp", "serve"]
    }
  }
}
```

Then use natural language in Claude: *"Add a prompt called summarize that..."*

## VS Code Extension

Install the Promptly VS Code extension for Command Palette integration:

- `Promptly: Add Prompt`
- `Promptly: List Prompts`
- `Promptly: Execute Chain`
- `Promptly: Evaluate Prompt`

See [vscode/README.md](vscode/README.md) for setup.

## Storage Architecture

```
~/.promptly/           # Global (shared across projects)
├── prompts.db         # SQLite database
├── config.yaml        # User preferences
└── prompts/           # YAML prompt files

.promptly/             # Project-local (overrides global)
├── prompts.db         # Project-specific database
└── prompts/           # Project prompt files
```

Local prompts with the same name override global prompts. This lets you customize prompts per-project while maintaining a shared library.

## All Commands

| Command | Description |
|---------|-------------|
| `promptly add <name> <content>` | Add or update a prompt |
| `promptly get <name>` | Retrieve a prompt |
| `promptly list` | List all prompts |
| `promptly log <name>` | Show version history |
| `promptly delete <name>` | Delete a prompt |
| `promptly branch <name>` | Create a branch |
| `promptly checkout <branch>` | Switch branches |
| `promptly branches` | List all branches |
| `promptly eval <name>` | Evaluate with LLM Judge |
| `promptly chain create <name>` | Create a chain |
| `promptly chain run <name>` | Run a chain |
| `promptly skill add <name>` | Add a skill |
| `promptly skill run <name>` | Run a skill |
| `promptly export <name>` | Export to YAML |
| `promptly import <file>` | Import from YAML |
| `promptly info` | Show configuration |
| `promptly demo` | Run interactive demos |
| `promptly analytics ...` | Analytics commands |
| `promptly mrf ...` | MRF refinement commands |

## Requirements

- Python 3.9+
- Core: `click`, `pyyaml` (installed automatically)
- Optional: `rich`, `mcp`, `anthropic`, `ollama`

## License

MIT License. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.
