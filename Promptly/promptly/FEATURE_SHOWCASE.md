# Promptly Interactive CLI & TUI - Feature Showcase

Visual demonstration of all interactive features.

## 🎨 Interface Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PROMPTLY INTERFACES                          │
└─────────────────────────────────────────────────────────────────────┘

1. INTERACTIVE REPL                    2. TERMINAL UI (TUI)
   ┌─────────────────────────┐           ┌──────────────────────────┐
   │ promptly (main)> _      │           │ [Prompts] [Branches]     │
   │                         │           ├────────────┬─────────────┤
   │ • History (↑↓)          │           │ Prompts    │ Details     │
   │ • Completion (Tab)      │           │            │             │
   │ • Search (Ctrl+R)       │           │ • Click    │ • Markdown  │
   │ • Syntax highlighting   │           │ • Navigate │ • Syntax HL │
   └─────────────────────────┘           └────────────┴─────────────┘

3. ENHANCED CLI                        4. SETUP WIZARDS
   ┌─────────────────────────┐           ┌──────────────────────────┐
   │ $ promptly-enhanced     │           │ Step 1/5: Setup...       │
   │                         │           │                          │
   │ ┏━━━━━━━┳━━━━━━━━━┓     │           │ [•••••]                  │
   │ ┃ Name  ┃ Version ┃     │           │                          │
   │ ┡━━━━━━━╇━━━━━━━━━┩     │           │ → Interactive prompts    │
   │ │ ...   │ ...     │     │           │ → Validation             │
   └─────────────────────────┘           └──────────────────────────┘
```

## 🚀 Quick Start Examples

### Example 1: REPL Session

```bash
$ ./promptly-interactive

╭─────────────────────────────────────────╮
│          Welcome                        │
│  Promptly Interactive Mode              │
╰─────────────────────────────────────────╯

promptly> init
✓ Initialized empty Promptly repository

promptly (main)> add summarizer "Summarize the following text:\n\n{text}"
✓ Added prompt 'summarizer' (v1) on branch 'main' [a1b2c3d4e5f6]

promptly (main)> add translator "Translate to {language}:\n\n{text}"
✓ Added prompt 'translator' (v1) on branch 'main' [b2c3d4e5f6g7]

promptly (main)> list
┏━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Name        ┃ Version ┃ Commit   ┃ Created            ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ summarizer  │ v1      │ a1b2c3d4 │ 2024-01-15 10:30   │
│ translator  │ v1      │ b2c3d4e5 │ 2024-01-15 10:31   │
└─────────────┴─────────┴──────────┴────────────────────┘

promptly (main)> get summarizer

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

promptly (main)> branch development
✓ Created branch 'development' from 'main'

promptly (main)> checkout development
✓ Switched to branch 'development'

promptly (development)> branches
Branches
├── [*] development (a1b2c3d4)
│   └── prompts
│       ├── summarizer v1
│       └── translator v1
└── main (a1b2c3d4)
    └── prompts
        ├── summarizer v1
        └── translator v1

promptly (development)> help
[Shows comprehensive help]

promptly (development)> exit
✓ Goodbye!
```

### Example 2: TUI Navigation

```
Launch:
$ ./promptly-tui

┌─ Promptly TUI ────────────────────────────────────────────────────────┐
│ Prompt Management System                                              │
├───────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ [Prompts] [Branches] [Log] [Eval] [Chains] [Diff]              │  │
│  ├──────────────────────────┬──────────────────────────────────────┤  │
│  │ Prompt List              │ Prompt Details                       │  │
│  │                          │                                      │  │
│  │ ┌────────────────────┐   │ # summarizer                         │  │
│  │ │ summarizer (v1)    │◄──┼─                                     │  │
│  │ │ translator (v2)    │   │ **Branch:** main                     │  │
│  │ │ code_reviewer (v1) │   │ **Version:** 1                       │  │
│  │ └────────────────────┘   │ **Commit:** a1b2c3d4                 │  │
│  │                          │                                      │  │
│  │                          │ ## Content                           │  │
│  │                          │                                      │  │
│  │                          │ ```                                  │  │
│  │                          │ Summarize the following text:        │  │
│  │                          │                                      │  │
│  │                          │ {text}                               │  │
│  │                          │ ```                                  │  │
│  └──────────────────────────┴──────────────────────────────────────┘  │
│                                                                        │
├───────────────────────────────────────────────────────────────────────┤
│ q: Quit | r: Refresh | 1-5: Tabs | Ctrl+H: Help                      │
└───────────────────────────────────────────────────────────────────────┘

Keyboard Navigation:
• Press 1-5 to switch tabs
• Use arrow keys to navigate lists
• Click on prompts to view details
• Press r to refresh
• Press Ctrl+H for help
• Press q to quit
```

### Example 3: Enhanced CLI

```bash
$ ./promptly-enhanced status

╭─ Promptly Status ─────────────────────────────────────────╮
│ Repository Status                                         │
│                                                           │
│ Location: /home/user/projects/my-prompts/.promptly        │
│ Current Branch: development                               │
│                                                           │
│ Statistics:                                               │
│   • Prompts: 15                                           │
│   • Branches: 3                                           │
│   • Evaluations: 25                                       │
│   • Chains: 5                                             │
╰───────────────────────────────────────────────────────────╯

$ ./promptly-enhanced list

╭─ Prompts on branch 'development' ─────────────────────────╮
│ ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┓ │
│ ┃ Name           ┃ Version ┃ Commit   ┃ Created      ┃ │
│ ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━┩ │
│ │ summarizer     │ v1      │ a1b2c3d4 │ 2024-01-15.. │ │
│ │ translator     │ v2      │ b2c3d4e5 │ 2024-01-16.. │ │
│ │ code_reviewer  │ v1      │ c3d4e5f6 │ 2024-01-17.. │ │
│ └────────────────┴─────────┴──────────┴──────────────┘ │
╰───────────────────────────────────────────────────────────╯

$ ./promptly-enhanced show summarizer

╭─ Prompt: summarizer ──────────────────────────────────────╮
│ Branch: development                                       │
│ Version: 1                                                │
│ Commit: a1b2c3d4e5f6                                      │
│ Created: 2024-01-15 10:30:00                              │
╰───────────────────────────────────────────────────────────╯

Content:

  1 │ Summarize the following text:
  2 │
  3 │ {text}

$ ./promptly-enhanced branches

Branches
├── [*] development (a1b2c3d4)
│   └── prompts
│       ├── summarizer v1
│       ├── translator v2
│       └── code_reviewer v1
├── main (b2c3d4e5)
│   └── prompts
│       ├── summarizer v1
│       └── translator v1
└── production (c3d4e5f6)
    └── prompts
        └── summarizer v1

$ ./promptly-enhanced export backup.json
⠋ Exporting to backup.json...
✓ Exported 15 prompts to backup.json
```

### Example 4: Project Wizard

```bash
$ ./promptly-wizard project

╭─────────────────────────────────────────────────────╮
│  Promptly Project Setup Wizard                      │
│                                                     │
│  This wizard will guide you through setting up      │
│  a new Promptly project.                            │
╰─────────────────────────────────────────────────────╯

Step 1/5: Choose project location

Project directory [/home/user/projects]: /home/user/my-prompts
Directory does not exist. Create it? (Y/n): y
✓ Created directory

Step 2/5: Initialize Promptly repository

✓ Initialized empty Promptly repository

Step 3/5: Create initial prompts

Would you like to create some starter prompts? (Y/n): y

Create 'summarizer' prompt? (Y/n): y
✓ Created prompt: summarizer

Create 'translator' prompt? (Y/n): y
✓ Created prompt: translator

Create 'code_reviewer' prompt? (Y/n): n

Step 4/5: Set up branches

Create development and production branches? (Y/n): y
✓ Created 'development' branch
✓ Created 'production' branch

Step 5/5: Create configuration file

Create a configuration file? (Y/n): y
✓ Created config file: /home/user/my-prompts/promptly_config.yaml

╭─ Setup Complete ─────────────────────────────────────╮
│ ✓ Project setup complete!                            │
│                                                      │
│ Location: /home/user/my-prompts                      │
│                                                      │
│ Next steps:                                          │
│   • cd /home/user/my-prompts                         │
│   • promptly list  # View prompts                    │
│   • promptly add <name> <content>  # Add prompts     │
│   • promptly-tui  # Launch TUI interface             │
╰──────────────────────────────────────────────────────╯
```

### Example 5: Prompt Wizard

```bash
$ ./promptly-wizard prompt

╭─────────────────────────────────────────────────────╮
│  Prompt Creation Wizard                             │
│                                                     │
│  This wizard will guide you through creating        │
│  a new prompt.                                      │
╰─────────────────────────────────────────────────────╯

Step 1/4: Basic Information

Prompt name: code_optimizer
Description (optional): Optimizes code for performance and readability

Step 2/4: Prompt Content

Enter prompt content. Use {variable} for placeholders.
Examples: {text}, {language}, {query}

Use multi-line editor? (y/N): n
Content: Optimize the following {language} code:\n\n{code}\n\nFocus on: {goals}

Step 3/4: Metadata

Add metadata? (Y/n): y
Category (optional) [general]: code
Tags (comma-separated, optional): optimization, performance, refactoring

Step 4/4: Review and Save

╭─ Review ─────────────────────────────────────────────╮
│ Name: code_optimizer                                 │
│ Description: Optimizes code for performance and...   │
│                                                      │
│ Content:                                             │
│ Optimize the following {language} code:              │
│                                                      │
│ {code}                                               │
│                                                      │
│ Focus on: {goals}                                    │
│                                                      │
│ Metadata: {'category': 'code', 'tags': [...]}       │
╰──────────────────────────────────────────────────────╯

Save this prompt? (Y/n): y
✓ Added prompt 'code_optimizer' (v1) on branch 'main' [d4e5f6g7h8i9]
```

## 🎯 Feature Matrix

```
╔════════════════════╦═════════╦══════╦═══════════╦══════════╗
║ Feature            ║ REPL    ║ TUI  ║ Enhanced  ║ Wizards  ║
╠════════════════════╬═════════╬══════╬═══════════╬══════════╣
║ History            ║    ✓    ║  -   ║     -     ║    -     ║
║ Auto-complete      ║    ✓    ║  -   ║     -     ║    -     ║
║ Visual Navigation  ║    -    ║  ✓   ║     -     ║    -     ║
║ Mouse Support      ║    -    ║  ✓   ║     -     ║    -     ║
║ Rich Formatting    ║    ✓    ║  ✓   ║     ✓     ║    ✓     ║
║ Syntax Highlight   ║    ✓    ║  ✓   ║     ✓     ║    -     ║
║ Progress Bars      ║    -    ║  -   ║     ✓     ║    -     ║
║ Step-by-Step       ║    -    ║  -   ║     -     ║    ✓     ║
║ Interactive        ║    ✓    ║  ✓   ║     -     ║    ✓     ║
║ Keyboard Shortcuts ║    ✓    ║  ✓   ║     -     ║    -     ║
║ Scriptable         ║    -    ║  -   ║     ✓     ║    -     ║
║ Validation         ║    -    ║  -   ║     -     ║    ✓     ║
╚════════════════════╩═════════╩══════╩═══════════╩══════════╝
```

## 🔧 Shell Completion Demo

```bash
# After installation
$ promptly <TAB>
add       branch    chain     checkout  diff      eval
export    get       help      import    init      list
log       show      status    tui       version   wizard

$ promptly get <TAB>
summarizer    translator    code_optimizer    code_reviewer

$ promptly checkout <TAB>
main    development    production    feature/new-feature

$ promptly chain run <TAB>
analysis_pipeline    translation_chain    full_workflow

$ promptly export backup --format <TAB>
json    yaml

$ promptly eval run sum<TAB>
summarizer

$ promptly show summ<TAB>
summarizer
```

## 📊 Use Case Scenarios

### Scenario 1: Learning Promptly (New User)

```
1. Run wizard:    ./promptly-wizard project
2. Explore TUI:   ./promptly-tui
3. Try REPL:      ./promptly-interactive
4. Create prompt: Use wizard or REPL
```

**Best Interface**: Wizards → TUI → REPL

### Scenario 2: Daily Prompt Management

```
1. Launch REPL:   ./promptly-interactive
2. Quick edits:   add, get, list commands
3. Use history:   ↑ to recall commands
4. Auto-complete: Tab for names
```

**Best Interface**: REPL (with shell completion)

### Scenario 3: Repository Overview

```
1. Launch TUI:    ./promptly-tui
2. Browse tabs:   Press 1-5
3. View details:  Click on items
4. Check history: Log tab
```

**Best Interface**: TUI

### Scenario 4: Automation/Scripts

```bash
#!/bin/bash
# Backup script
./promptly-enhanced export backup-$(date +%Y%m%d).json
./promptly-enhanced log --limit 50 > recent-changes.txt
./promptly-enhanced list --branch production > prod-prompts.txt
```

**Best Interface**: Enhanced CLI

### Scenario 5: Team Onboarding

```
1. Setup project: ./promptly-wizard project
2. Create prompts: ./promptly-wizard prompt (multiple times)
3. Build chain:   ./promptly-wizard chain
4. Set up tests:  ./promptly-wizard evaluation
```

**Best Interface**: Wizards

## 🎨 Customization Examples

### Custom Aliases

```bash
# Add to ~/.bashrc or ~/.zshrc
alias pi="cd ~/my-prompts && ./promptly-interactive"
alias pt="cd ~/my-prompts && ./promptly-tui"
alias ps="./promptly-enhanced status"
alias pl="./promptly-enhanced list"
```

### Custom Functions

```bash
# Quick prompt creation
pnew() {
    cd ~/my-prompts
    ./promptly-interactive <<EOF
add $1 "$2"
get $1
exit
EOF
}

# Usage: pnew myname "Content here"
```

### Tab Completion Integration

```bash
# After installing completions
$ source ~/.bashrc

# Now enjoy completions everywhere
$ promptly <TAB>
$ cd my-prompts && promptly get <TAB>
```

## 📈 Performance Features

### Progress Indicators

```bash
$ ./promptly-enhanced export large-backup.json

╭─ Exporting Repository ────────────────────────────────╮
│                                                       │
│  ⠋ Processing prompts...                             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 75% (150/200)     │
│                                                       │
╰───────────────────────────────────────────────────────╯
```

### Spinners

```bash
$ ./promptly-enhanced import prompts.yaml

⠋ Importing prompts from prompts.yaml...
✓ Successfully imported 50 prompts
```

### Real-time Updates (TUI)

```
TUI automatically refreshes when:
• Files change on disk
• Manual refresh (press 'r')
• Tab switches
```

## 🎓 Learning Path

### Week 1: Getting Started
- Day 1-2: Use project wizard
- Day 3-4: Explore TUI
- Day 5: Try REPL with completion

### Week 2: Daily Usage
- Use REPL for daily tasks
- Create shortcuts/aliases
- Learn keyboard shortcuts

### Week 3: Advanced Features
- Create chains
- Set up evaluations
- Use enhanced CLI for scripts

### Week 4: Mastery
- Customize workflows
- Create automation scripts
- Contribute improvements

## 🏆 Best Practices

1. **Start with Wizards** - Get comfortable
2. **Use REPL Daily** - It's fastest for regular work
3. **TUI for Review** - Great for understanding structure
4. **Enhanced for Scripts** - Automate repetitive tasks
5. **Enable Completion** - Saves tons of time
6. **Learn Shortcuts** - Boost productivity
7. **Customize Aliases** - Make it yours

## 💡 Pro Tips

1. **REPL**: Use Ctrl+R to search history quickly
2. **TUI**: Learn number keys (1-5) for quick tab switching
3. **Enhanced**: Pipe output to files for reports
4. **Wizards**: Use templates for common patterns
5. **Completion**: Works in subshells and scripts too
6. **All**: Read the help - lots of hidden gems!

---

**Choose the right tool for the job and enjoy beautiful prompt management!** 🎨✨
