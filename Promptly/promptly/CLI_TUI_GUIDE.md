# Promptly CLI & TUI User Guide

Complete guide to using Promptly's interactive command-line interfaces.

## Table of Contents

1. [Installation](#installation)
2. [Interactive REPL](#interactive-repl)
3. [Terminal UI (TUI)](#terminal-ui-tui)
4. [Enhanced CLI](#enhanced-cli)
5. [Setup Wizards](#setup-wizards)
6. [Shell Completion](#shell-completion)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Dependencies

Install required packages:

```bash
# Core dependencies (required)
pip install click pyyaml

# For enhanced features (recommended)
pip install rich prompt_toolkit pygments textual

# All at once
pip install click pyyaml rich prompt_toolkit pygments textual
```

### Verify Installation

```bash
python -m promptly.cli.interactive --help
python -m promptly.tui.app --help
```

---

## Interactive REPL

The Interactive REPL provides a command-line shell with history, auto-completion, and rich formatting.

### Starting the REPL

```bash
# From Promptly directory
python -m promptly.cli.interactive

# Or if installed as package
promptly-interactive
```

### Features

- **Command History**: Navigate with Up/Down arrows
- **Auto-Completion**: Press Tab to complete commands and prompt names
- **Reverse Search**: Press Ctrl+R to search history
- **Multi-line Input**: Supported for long prompts
- **Syntax Highlighting**: Code and prompts displayed with colors
- **Context-Aware Prompts**: Shows current branch in prompt

### Available Commands

#### Repository Management

```
init                    Initialize a new promptly repository
status                  Show repository status
config                  Show configuration
```

#### Prompt Management

```
add <name> <content>    Add or update a prompt
get <name>              Get a prompt
list                    List all prompts
show <name>             Show prompt with syntax highlighting
search <query>          Search prompts by content
log [name]              Show commit history
```

#### Branch Management

```
branch <name>           Create a new branch
checkout <name>         Switch to a branch
branches                List all branches
```

#### Evaluation & Chains

```
eval <name> <file>      Evaluate a prompt
chain create <name>     Create a chain
chain run <name>        Run a chain
```

#### Utilities

```
diff <name> <v1> <v2>   Compare versions
export <format>         Export repository
import <file>           Import prompts
clear                   Clear screen
history                 Show command history
help                    Show help
exit/quit               Exit REPL
```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Tab` | Auto-complete command or name |
| `Ctrl+R` | Reverse search history |
| `Ctrl+C` | Cancel current input |
| `Ctrl+D` | Exit REPL |
| `Up/Down` | Navigate command history |
| `Ctrl+A` | Move to line start |
| `Ctrl+E` | Move to line end |
| `Ctrl+K` | Delete to line end |
| `Ctrl+U` | Delete entire line |

### Example Session

```
$ python -m promptly.cli.interactive

╭─────────────────────────────────────────╮
│          Welcome                        │
│  Promptly Interactive Mode              │
│                                         │
│  Type help for available commands       │
│  Type exit or press Ctrl+D to quit     │
╰─────────────────────────────────────────╯

promptly> init
✓ Initialized empty Promptly repository

promptly (main)> add summarizer "Summarize the following text: {text}"
✓ Added prompt 'summarizer' (v1) on branch 'main' [a1b2c3d4e5f6]

promptly (main)> list
┏━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Name        ┃ Version ┃ Commit   ┃ Created            ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ summarizer  │ v1      │ a1b2c3d4 │ 2024-01-15 10:30:00│
└─────────────┴─────────┴──────────┴────────────────────┘

promptly (main)> get summarizer
Prompt: summarizer
Branch: main
Version: 1
Commit: a1b2c3d4e5f6
Created: 2024-01-15 10:30:00

╭─ Content ──────────────────────────────╮
│ Summarize the following text: {text}   │
╰────────────────────────────────────────╯

promptly (main)> exit
✓ Goodbye!
```

---

## Terminal UI (TUI)

Full-featured terminal user interface with split-pane layout, navigation, and real-time updates.

### Starting the TUI

```bash
# From Promptly directory
python -m promptly.tui.app

# Or if installed
promptly-tui
```

### Features

- **Tabbed Interface**: Switch between Prompts, Branches, Log, Eval, Chains, and Diff
- **Split Panes**: Browse prompts on left, view details on right
- **Tree View**: Visual branch hierarchy
- **Syntax Highlighting**: Code displayed with colors
- **Keyboard Navigation**: Full keyboard control
- **Real-time Updates**: Live view of repository state

### Interface Layout

```
┌─ Promptly TUI ─────────────────────────────────────────────┐
│ [Prompts] [Branches] [Log] [Eval] [Chains] [Diff]          │
├─────────────────────────┬──────────────────────────────────┤
│ Prompt List             │ Prompt Details                   │
│                         │                                  │
│ • summarizer (v1)       │ Prompt: summarizer               │
│ • translator (v2)       │ Branch: main                     │
│ • code_reviewer (v1)    │ Version: 1                       │
│                         │ Commit: a1b2c3d4                 │
│                         │                                  │
│                         │ Content:                         │
│                         │ Summarize the following...       │
│                         │                                  │
└─────────────────────────┴──────────────────────────────────┘
│ q: Quit | r: Refresh | 1-5: Switch tabs | Ctrl+H: Help    │
└────────────────────────────────────────────────────────────┘
```

### Tabs

#### 1. Prompts Tab

- Left panel: List of all prompts (clickable)
- Right panel: Selected prompt details with syntax highlighting
- Navigate with arrow keys or mouse
- Click prompt to view details

#### 2. Branches Tab

- Tree view of all branches
- Shows current branch with `*` marker
- Displays prompts under each branch
- Hierarchical structure

#### 3. Log Tab

- Commit history with metadata
- Shows commit hash, prompt name, version, date
- Scrollable list
- Newest commits first

#### 4. Eval Tab

- Evaluation center
- List of prompts to evaluate
- Recent evaluation results
- Scores and metrics

#### 5. Chains Tab

- Prompt chains overview
- Chain descriptions and steps
- Run buttons for execution
- Step visualization (A → B → C)

#### 6. Diff Tab

- Compare prompt versions
- Side-by-side view (coming soon)
- Highlighting changes

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `q` | Quit application |
| `r` | Refresh current view |
| `1` | Switch to Prompts tab |
| `2` | Switch to Branches tab |
| `3` | Switch to Log tab |
| `4` | Switch to Eval tab |
| `5` | Switch to Chains tab |
| `Ctrl+H` | Show help |
| `Tab` | Navigate between widgets |
| `Arrow keys` | Navigate lists/trees |
| `Enter` | Select item |
| `Esc` | Go back |

### Navigation Tips

1. **Use Tab**: Move focus between panels
2. **Arrow Keys**: Navigate lists and trees
3. **Number Keys**: Quick tab switching
4. **Mouse Support**: Click on items and buttons
5. **Scroll**: Use scroll wheel or PgUp/PgDn

---

## Enhanced CLI

Rich CLI with beautiful tables, progress bars, and interactive prompts.

### Usage

```bash
# Status with rich formatting
python -m promptly.cli.enhanced status

# List with table
python -m promptly.cli.enhanced list

# Show with syntax highlighting
python -m promptly.cli.enhanced show summarizer

# Branch tree
python -m promptly.cli.enhanced branches

# Log with table
python -m promptly.cli.enhanced log --limit 20

# Diff comparison
python -m promptly.cli.enhanced diff summarizer 1 2

# Export with progress
python -m promptly.cli.enhanced export backup.json
```

### Features

#### Rich Tables

```bash
$ python -m promptly.cli.enhanced list

╭─ Prompts on branch 'main' ──────────────────────────────╮
│ ┏━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━┓ │
│ ┃ Name        ┃ Version ┃ Commit   ┃ Created       ┃ │
│ ┡━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━┩ │
│ │ summarizer  │ v1      │ a1b2c3d4 │ 2024-01-15... │ │
│ │ translator  │ v2      │ b2c3d4e5 │ 2024-01-16... │ │
│ └─────────────┴─────────┴──────────┴───────────────┘ │
╰──────────────────────────────────────────────────────────╯
```

#### Syntax Highlighting

```bash
$ python -m promptly.cli.enhanced show summarizer

╭─ Prompt: summarizer ──────────────────────────────────────╮
│ Branch: main                                              │
│ Version: 1                                                │
│ Commit: a1b2c3d4e5f6                                      │
│ Created: 2024-01-15 10:30:00                              │
╰───────────────────────────────────────────────────────────╯

Content:

  1 │ Summarize the following text:
  2 │
  3 │ {text}
```

#### Tree View

```bash
$ python -m promptly.cli.enhanced branches

Branches
├── [*] main (a1b2c3d4)
│   └── prompts
│       ├── summarizer v1
│       └── translator v2
├── development (b2c3d4e5)
│   └── prompts
│       └── test_prompt v1
└── production (c3d4e5f6)
    └── prompts
        └── summarizer v1
```

#### Progress Indicators

```bash
$ python -m promptly.cli.enhanced export backup.json
⠋ Exporting to backup.json...
✓ Exported 15 prompts to backup.json
```

---

## Setup Wizards

Interactive step-by-step guides for common tasks.

### Available Wizards

```bash
python -m promptly.cli.wizards project     # Project setup
python -m promptly.cli.wizards prompt      # Create prompt
python -m promptly.cli.wizards chain       # Create chain
python -m promptly.cli.wizards evaluation  # Setup evaluation
python -m promptly.cli.wizards template    # Template-based creation
```

### Project Wizard

Creates a new Promptly project with initial setup.

```bash
$ python -m promptly.cli.wizards project

╭─────────────────────────────────────────────────────╮
│  Promptly Project Setup Wizard                      │
│  This wizard will guide you through setting up      │
│  a new Promptly project.                            │
╰─────────────────────────────────────────────────────╯

Step 1/5: Choose project location

Project directory [/home/user/projects]: /home/user/my-prompts

Step 2/5: Initialize Promptly repository

✓ Initialized empty Promptly repository

Step 3/5: Create initial prompts

Would you like to create some starter prompts? (Y/n): y
Create 'summarizer' prompt? (Y/n): y
✓ Created prompt: summarizer
Create 'translator' prompt? (Y/n): y
✓ Created prompt: translator

Step 4/5: Set up branches

Create development and production branches? (Y/n): y
✓ Created 'development' branch
✓ Created 'production' branch

Step 5/5: Create configuration file

Create a configuration file? (Y/n): y
✓ Created config file: /home/user/my-prompts/promptly_config.yaml

╭─ Setup Complete ────────────────────────────────────╮
│ ✓ Project setup complete!                           │
│                                                     │
│ Location: /home/user/my-prompts                     │
│                                                     │
│ Next steps:                                         │
│   • cd /home/user/my-prompts                        │
│   • promptly list  # View prompts                   │
│   • promptly add <name> <content>  # Add prompts    │
│   • promptly-tui  # Launch TUI interface            │
╰─────────────────────────────────────────────────────╯
```

### Prompt Wizard

Interactive prompt creation with variable detection and metadata.

```bash
$ python -m promptly.cli.wizards prompt

Step 1/4: Basic Information

Prompt name: code_optimizer
Description (optional): Optimizes code for performance

Step 2/4: Prompt Content

Enter prompt content. Use {variable} for placeholders.
Examples: {text}, {language}, {query}

Use multi-line editor? (y/N): n
Content: Optimize the following {language} code for performance:\n\n{code}\n\nFocus on: {optimization_goals}

Step 3/4: Metadata

Add metadata? (Y/n): y
Category (optional) [general]: code
Tags (comma-separated, optional): optimization, performance, refactoring

Step 4/4: Review and Save

╭─ Review ────────────────────────────────────────────╮
│ Name: code_optimizer                                │
│ Description: Optimizes code for performance         │
│                                                     │
│ Content:                                            │
│ Optimize the following {language} code for          │
│ performance:                                        │
│                                                     │
│ {code}                                              │
│                                                     │
│ Focus on: {optimization_goals}                      │
│                                                     │
│ Metadata: {'category': 'code', 'tags': [...]}      │
╰─────────────────────────────────────────────────────╯

Save this prompt? (Y/n): y
✓ Added prompt 'code_optimizer' (v1) on branch 'main'
```

### Chain Wizard

Create prompt chains by selecting existing prompts.

```bash
$ python -m promptly.cli.wizards chain

Step 1/3: Basic Information

Chain name: full_analysis
Description (optional): Complete analysis pipeline

Step 2/3: Select Prompts

Available prompts:
  1. summarizer (v1)
  2. translator (v2)
  3. code_optimizer (v1)
  4. sentiment_analyzer (v1)

Enter prompt numbers to add to chain (comma-separated): 1,4

Step 3/3: Review and Save

╭─ Review Chain ──────────────────────────────────────╮
│ Name: full_analysis                                 │
│ Description: Complete analysis pipeline             │
│                                                     │
│ Steps:                                              │
│ summarizer → sentiment_analyzer                     │
╰─────────────────────────────────────────────────────╯

Create this chain? (Y/n): y
✓ Created chain 'full_analysis' with 2 steps
```

### Evaluation Wizard

Set up test cases for prompt evaluation.

```bash
$ python -m promptly.cli.wizards evaluation

Step 1/3: Select Prompt

Available prompts:
  1. summarizer (v1)
  2. translator (v2)

Select prompt number: 1

Prompt content:
Summarize the following text: {text}

Detected variables: text

Step 2/3: Create Test Cases

Number of test cases to create [3]: 2

Test case 1:
  text: Machine learning is a subset of artificial intelligence...
  Expected output (optional): ML is part of AI focusing on learning from data

Test case 2:
  text: Python is a high-level programming language...
  Expected output (optional): Python is a versatile programming language

Step 3/3: Save Test Cases

Test file name [summarizer_tests.json]:
✓ Saved test cases to: summarizer_tests.json

To run evaluation:
  promptly eval run summarizer summarizer_tests.json
```

### Template Wizard

Create prompts from predefined templates.

```bash
$ python -m promptly.cli.wizards template

Available templates:
  1. Text Processing Template
     Variables: instruction, text, format
  2. Code Generation Template
     Variables: language, task, requirements
  3. Analysis Template
     Variables: subject, content, analysis_points
  4. Q&A Template
     Variables: context, question

Select template [1]: 2

Template: Code Generation Template

Content preview:
Generate {language} code for the following task:

{task}

Requirements:
{requirements}

Provide clean, well-commented code.

Prompt name: code_generator
Customize content? (y/N): n
✓ Added prompt 'code_generator' (v1) on branch 'main'
```

---

## Shell Completion

Auto-completion for Bash, Zsh, and Fish shells.

### Installation

```bash
cd Promptly/promptly/shell_completion

# Automatic installation for current shell
./install.sh

# Or specify shell
./install.sh bash
./install.sh zsh
./install.sh fish

# Install for all shells
./install.sh all
```

### Manual Installation

#### Bash

```bash
# Copy completion file
cp promptly.bash ~/.bash_completion.d/promptly

# Add to .bashrc
echo '[ -f ~/.bash_completion.d/promptly ] && source ~/.bash_completion.d/promptly' >> ~/.bashrc

# Reload
source ~/.bashrc
```

#### Zsh

```bash
# Copy completion file
mkdir -p ~/.zsh/completions
cp promptly.zsh ~/.zsh/completions/_promptly

# Add to .zshrc
echo 'fpath=(~/.zsh/completions $fpath)' >> ~/.zshrc
echo 'autoload -Uz compinit && compinit' >> ~/.zshrc

# Reload
source ~/.zshrc
```

#### Fish

```bash
# Copy completion file
cp promptly.fish ~/.config/fish/completions/

# Reload (automatic in new sessions)
```

### Usage Examples

```bash
# Complete commands
$ promptly <TAB>
add       branch    chain     checkout  diff      eval
export    get       help      import    init      list
log       show      status    tui       version   wizard

# Complete prompt names
$ promptly get <TAB>
summarizer    translator    code_optimizer

# Complete chain names
$ promptly chain run <TAB>
full_analysis    translation_chain

# Complete branches
$ promptly checkout <TAB>
main    development    production

# Complete file formats
$ promptly export backup --format <TAB>
json    yaml
```

---

## Examples

### Complete Workflow Example

```bash
# 1. Initialize project
$ python -m promptly.cli.wizards project
# Follow prompts to create project

# 2. Start interactive mode
$ cd my-prompts
$ python -m promptly.cli.interactive

# 3. Create prompts
promptly (main)> add summarizer "Summarize: {text}"
✓ Added prompt 'summarizer' (v1)

promptly (main)> add analyzer "Analyze sentiment: {text}"
✓ Added prompt 'analyzer' (v1)

# 4. Create chain
promptly (main)> exit
$ python -m promptly.cli.wizards chain
# Select: summarizer, analyzer

# 5. View in TUI
$ python -m promptly.tui.app
# Navigate visually

# 6. Export
$ python -m promptly.cli.enhanced export backup.json
✓ Exported 2 prompts
```

### Testing Workflow

```bash
# Create test cases
$ python -m promptly.cli.wizards evaluation
# Follow prompts

# Run evaluation
$ promptly eval run summarizer summarizer_tests.json

# View results in TUI
$ python -m promptly.tui.app
# Navigate to Eval tab
```

### Branch Workflow

```bash
# Create feature branch
$ python -m promptly.cli.interactive
promptly (main)> branch feature/new-prompts
✓ Created branch 'feature/new-prompts'

promptly (main)> checkout feature/new-prompts
✓ Switched to branch 'feature/new-prompts'

# Work on prompts
promptly (feature/new-prompts)> add test_prompt "Test: {input}"
✓ Added prompt 'test_prompt' (v1)

# View branch tree
promptly (feature/new-prompts)> branches
Branches
├── [*] feature/new-prompts (...)
│   └── test_prompt v1
└── main (...)
    ├── summarizer v1
    └── analyzer v1
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'rich'`

**Solution**:
```bash
pip install rich prompt_toolkit pygments textual
```

#### 2. Permission Denied

**Problem**: Cannot write to completion directories

**Solution**:
```bash
# Use user-level installation
./install.sh  # Will create ~/.bash_completion.d automatically
```

#### 3. TUI Not Displaying Correctly

**Problem**: Characters appear garbled

**Solution**:
```bash
# Ensure UTF-8 locale
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

#### 4. Completion Not Working

**Problem**: Tab completion doesn't work

**Solution**:
```bash
# Reload shell configuration
source ~/.bashrc  # or ~/.zshrc

# Verify completion is loaded
complete -p | grep promptly  # bash
which _promptly               # zsh
```

#### 5. Colors Not Showing

**Problem**: No colors in output

**Solution**:
```bash
# Enable colors
export TERM=xterm-256color

# Force colors
export FORCE_COLOR=1
```

### Getting Help

```bash
# Interactive help
$ python -m promptly.cli.interactive
promptly> help

# Enhanced CLI help
$ python -m promptly.cli.enhanced --help

# Wizard help
$ python -m promptly.cli.wizards --help

# TUI help
# Press Ctrl+H while in TUI
```

### Debug Mode

```bash
# Enable verbose output
export PROMPTLY_DEBUG=1

# Run commands
python -m promptly.cli.interactive
```

---

## Tips & Best Practices

### 1. Use Interactive Mode for Exploration

```bash
# Great for learning and quick operations
python -m promptly.cli.interactive
```

### 2. Use TUI for Visual Overview

```bash
# Perfect for understanding repository structure
python -m promptly.tui.app
```

### 3. Use Wizards for Complex Setup

```bash
# Simplifies multi-step processes
python -m promptly.cli.wizards project
```

### 4. Use Enhanced CLI for Scripts

```bash
# Best for automation and scripting
python -m promptly.cli.enhanced export backup.json
```

### 5. Enable Shell Completion

```bash
# Saves time and reduces errors
./shell_completion/install.sh
```

### 6. Customize with Aliases

```bash
# Add to .bashrc or .zshrc
alias pi="python -m promptly.cli.interactive"
alias pt="python -m promptly.tui.app"
alias pw="python -m promptly.cli.wizards"
alias pe="python -m promptly.cli.enhanced"
```

### 7. Use History

```bash
# In interactive mode
promptly> history  # View recent commands
# Ctrl+R to search history
```

---

## Advanced Usage

### Custom Themes (TUI)

Coming soon: Customizable color schemes and layouts.

### Plugins

Coming soon: Extend functionality with custom commands.

### Configuration

Create `~/.promptly/config.yaml`:

```yaml
interactive:
  history_size: 1000
  completion_enabled: true
  syntax_theme: monokai

tui:
  default_tab: prompts
  refresh_interval: 5

enhanced:
  table_style: rounded
  color_output: true
```

---

## Keyboard Reference Card

### Interactive REPL

| Key | Action |
|-----|--------|
| `Tab` | Complete |
| `Ctrl+R` | Search history |
| `Ctrl+D` | Exit |
| `Up/Down` | History |
| `Ctrl+C` | Cancel |

### TUI

| Key | Action |
|-----|--------|
| `q` | Quit |
| `1-5` | Switch tabs |
| `r` | Refresh |
| `Tab` | Next widget |
| `Ctrl+H` | Help |

---

## Screenshots

### Interactive REPL
```
╭─────────────────────────────────────────╮
│  Promptly Interactive Mode              │
╰─────────────────────────────────────────╯
promptly (main)> list
┏━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Name        ┃ Version ┃ Commit   ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ summarizer  │ v1      │ a1b2c3d4 │
└─────────────┴─────────┴──────────┘
```

### Terminal UI
```
┌─ Promptly TUI ─────────────────────────┐
│ [Prompts] [Branches] [Log] [Eval]      │
├────────────────┬───────────────────────┤
│ summarizer (v1)│ Prompt: summarizer    │
│ translator (v2)│ Content: ...          │
└────────────────┴───────────────────────┘
```

---

## Support

- Documentation: `CLI_TUI_GUIDE.md`
- Issues: GitHub Issues
- Examples: `examples/` directory

---

**Happy prompting with Promptly! 🚀**
