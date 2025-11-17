# Promptly CLI & TUI - Quick Start

Modern interactive command-line and terminal user interfaces for Promptly.

## Installation

```bash
# Install dependencies
pip install click pyyaml rich prompt_toolkit pygments textual

# Install shell completion
cd shell_completion && ./install.sh
```

## Quick Start

### 1. Interactive REPL (Best for daily use)

```bash
python -m promptly.cli.interactive
```

Features:
- ✅ Command history and auto-completion
- ✅ Syntax highlighting
- ✅ Context-aware prompts
- ✅ Multi-line editing

### 2. Terminal UI (Best for visualization)

```bash
python -m promptly.tui.app
```

Features:
- ✅ Tabbed interface
- ✅ Split-pane layout
- ✅ Tree view for branches
- ✅ Real-time updates

### 3. Enhanced CLI (Best for scripting)

```bash
python -m promptly.cli.enhanced status
python -m promptly.cli.enhanced list
python -m promptly.cli.enhanced show summarizer
```

Features:
- ✅ Rich tables and formatting
- ✅ Progress indicators
- ✅ Syntax highlighting
- ✅ Export/import tools

### 4. Setup Wizards (Best for beginners)

```bash
python -m promptly.cli.wizards project     # Create new project
python -m promptly.cli.wizards prompt      # Create prompt
python -m promptly.cli.wizards chain       # Create chain
python -m promptly.cli.wizards evaluation  # Setup tests
python -m promptly.cli.wizards template    # Use templates
```

Features:
- ✅ Step-by-step guidance
- ✅ Input validation
- ✅ Preview before saving
- ✅ Predefined templates

## Usage Examples

### Interactive REPL Session

```bash
$ python -m promptly.cli.interactive

promptly> init
✓ Initialized empty Promptly repository

promptly (main)> add summarizer "Summarize: {text}"
✓ Added prompt 'summarizer' (v1) on branch 'main'

promptly (main)> list
┏━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Name        ┃ Version ┃ Commit   ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ summarizer  │ v1      │ a1b2c3d4 │
└─────────────┴─────────┴──────────┘

promptly (main)> get summarizer
[Shows prompt with syntax highlighting]

promptly (main)> branch development
✓ Created branch 'development' from 'main'

promptly (main)> checkout development
✓ Switched to branch 'development'

promptly (development)> exit
✓ Goodbye!
```

### TUI Navigation

```
1. Launch: python -m promptly.tui.app
2. Press 1-5 to switch tabs
3. Click prompts to view details
4. Press q to quit
```

### Enhanced CLI

```bash
# Show status with rich formatting
python -m promptly.cli.enhanced status

# List prompts in a table
python -m promptly.cli.enhanced list

# Show prompt with syntax highlighting
python -m promptly.cli.enhanced show summarizer

# View branch tree
python -m promptly.cli.enhanced branches

# Compare versions
python -m promptly.cli.enhanced diff summarizer 1 2

# Export with progress bar
python -m promptly.cli.enhanced export backup.json
```

### Project Wizard

```bash
$ python -m promptly.cli.wizards project

Step 1/5: Choose project location
Project directory: /home/user/my-prompts

Step 2/5: Initialize Promptly repository
✓ Initialized empty Promptly repository

Step 3/5: Create initial prompts
Would you like to create some starter prompts? y
✓ Created prompt: summarizer
✓ Created prompt: translator

Step 4/5: Set up branches
Create development and production branches? y
✓ Created 'development' branch
✓ Created 'production' branch

Step 5/5: Create configuration file
✓ Created config file

╭─ Setup Complete ─╮
│ ✓ Project ready! │
╰──────────────────╯
```

## Keyboard Shortcuts

### Interactive REPL

| Key | Action |
|-----|--------|
| `Tab` | Auto-complete |
| `Ctrl+R` | Search history |
| `Ctrl+D` | Exit |
| `Up/Down` | Navigate history |

### TUI

| Key | Action |
|-----|--------|
| `q` | Quit |
| `1-5` | Switch tabs |
| `r` | Refresh |
| `Tab` | Next widget |
| `Ctrl+H` | Help |

## Shell Completion

After installation, use Tab to complete:

```bash
promptly <TAB>              # Show all commands
promptly get <TAB>          # Complete prompt names
promptly checkout <TAB>     # Complete branch names
promptly chain run <TAB>    # Complete chain names
```

## Features Overview

| Feature | REPL | TUI | Enhanced | Wizards |
|---------|------|-----|----------|---------|
| Command history | ✅ | - | - | - |
| Auto-completion | ✅ | - | - | - |
| Visual navigation | - | ✅ | - | - |
| Rich formatting | ✅ | ✅ | ✅ | ✅ |
| Syntax highlighting | ✅ | ✅ | ✅ | - |
| Progress bars | - | - | ✅ | - |
| Step-by-step guidance | - | - | - | ✅ |
| Mouse support | - | ✅ | - | - |
| Keyboard shortcuts | ✅ | ✅ | - | - |

## When to Use Each Interface

### Use Interactive REPL when:
- Working daily with prompts
- Exploring repository
- Quick operations
- Learning commands

### Use TUI when:
- Need visual overview
- Comparing prompts/branches
- Navigating large repositories
- Prefer mouse interaction

### Use Enhanced CLI when:
- Writing scripts
- Automation tasks
- CI/CD pipelines
- Export/import operations

### Use Wizards when:
- Setting up new projects
- Creating complex prompts
- Building chains
- First-time users

## Configuration

Create `~/.promptly/config.yaml`:

```yaml
interactive:
  history_size: 1000
  completion_enabled: true

tui:
  default_tab: prompts

enhanced:
  color_output: true
```

## Troubleshooting

### Missing dependencies

```bash
pip install rich prompt_toolkit pygments textual
```

### Completion not working

```bash
# Reload shell
source ~/.bashrc  # or ~/.zshrc

# Verify
complete -p | grep promptly
```

### Colors not showing

```bash
export TERM=xterm-256color
export FORCE_COLOR=1
```

## Documentation

- **Complete Guide**: `CLI_TUI_GUIDE.md`
- **API Reference**: See module docstrings
- **Examples**: `examples/` directory

## Support

- File issues on GitHub
- Check troubleshooting guide
- See complete documentation

---

**Choose your interface and start managing prompts beautifully! 🎨**
