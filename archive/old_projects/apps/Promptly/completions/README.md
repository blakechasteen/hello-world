# Promptly Shell Completions

Auto-completion scripts for bash, zsh, and fish shells.

---

## Installation

### Bash

**Option 1: User-level (recommended)**
```bash
# Copy completion script
mkdir -p ~/.local/share/bash-completion/completions
cp completions/promptly.bash ~/.local/share/bash-completion/completions/promptly

# Or source directly in ~/.bashrc
echo 'source /path/to/Promptly/completions/promptly.bash' >> ~/.bashrc
source ~/.bashrc
```

**Option 2: System-wide (requires sudo)**
```bash
sudo cp completions/promptly.bash /etc/bash_completion.d/promptly
```

**Test it:**
```bash
promptly <TAB><TAB>
# Should show: search create update delete list execute ...
```

---

### Zsh

**Option 1: Using oh-my-zsh (recommended)**
```zsh
# Copy to custom completions
mkdir -p ~/.oh-my-zsh/custom/completions
cp completions/promptly.zsh ~/.oh-my-zsh/custom/completions/_promptly

# Reload completions
exec zsh
```

**Option 2: Manual installation**
```zsh
# Add to fpath
mkdir -p ~/.zsh/completions
cp completions/promptly.zsh ~/.zsh/completions/_promptly

# Add to ~/.zshrc
echo 'fpath=(~/.zsh/completions $fpath)' >> ~/.zshrc
echo 'autoload -Uz compinit && compinit' >> ~/.zshrc

# Reload
source ~/.zshrc
```

**Test it:**
```zsh
promptly <TAB>
# Should show command descriptions
```

---

### Fish

**Installation**
```fish
# Copy to fish completions directory
mkdir -p ~/.config/fish/completions
cp completions/promptly.fish ~/.config/fish/completions/

# Fish auto-loads completions, no need to reload
```

**Test it:**
```fish
promptly <TAB>
# Should show commands with descriptions
```

---

## Supported Commands

All shells support tab completion for:

### Main Commands
- `search` - Search for prompts
- `create` - Create a new prompt
- `update` - Update an existing prompt
- `delete` - Delete a prompt
- `list` - List all prompts
- `execute` - Execute a prompt with recursive loops
- `branch` - Create a branch from a prompt
- `merge` - Merge prompt branches
- `diff` - Show differences between prompts
- `analytics` - View prompt analytics
- `skills` - Manage UltraPrompt skills
- `help` - Show help message
- `version` - Show version

### Options (Context-Aware)

Each command has its own set of options that appear after typing the command.

**Example:**
```bash
promptly search --<TAB>
# Shows: --tags --limit --format

promptly execute --loop <TAB>
# Shows: REFINE CRITIQUE DECOMPOSE VERIFY EXPLORE HOFSTADTER
```

---

## Short Alias

All completions also work with the `p` alias:

```bash
p search<TAB>  # Same as promptly search<TAB>
p list<TAB>    # Same as promptly list<TAB>
```

---

## Troubleshooting

### Bash: Completions not working

**Check if bash-completion is installed:**
```bash
# Ubuntu/Debian
sudo apt install bash-completion

# macOS (using Homebrew)
brew install bash-completion@2
```

**Make sure completion is enabled in ~/.bashrc:**
```bash
if [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
fi
```

### Zsh: Completions not showing descriptions

**Make sure compinit is called:**
```zsh
# Add to ~/.zshrc
autoload -Uz compinit && compinit
```

**Rebuild completion cache:**
```zsh
rm -f ~/.zcompdump
compinit
```

### Fish: Completions not loading

**Check fish version (need 3.0+):**
```fish
fish --version
```

**Manually reload:**
```fish
# Delete cache
rm -rf ~/.cache/fish

# Restart fish
exec fish
```

---

## Examples

### Search with Tab Completion
```bash
$ promptly search --tags<TAB>
# Shows: --tags option

$ promptly search --tags sql<TAB>
# (Future: could show available tags)
```

### Execute with Loop Type
```bash
$ promptly execute --loop <TAB>
# Shows: REFINE CRITIQUE DECOMPOSE VERIFY EXPLORE HOFSTADTER

$ promptly execute --loop REFINE<TAB>
# Continues with other options
```

### List with Format
```bash
$ promptly list --format <TAB>
# Shows: table json yaml tree
```

---

## Custom Completions

Want to add dynamic completions (e.g., prompt names from database)?

### Bash Example
Edit `promptly.bash`:
```bash
# Add after line 18
case "${prev}" in
    execute|update|delete)
        # Get prompt names from database
        local prompts=$(promptly list --format json | jq -r '.[].name')
        opts="${prompts}"
        ;;
```

### Zsh Example
Edit `promptly.zsh`:
```zsh
# Add helper function
_promptly_prompts() {
    local prompts
    prompts=(${(f)"$(promptly list --format json | jq -r '.[].name')"})
    _describe 'prompts' prompts
}

# Use in arguments
execute)
    _arguments \
        '1:prompt:_promptly_prompts' \
        '--input[Input data]:input:'
    ;;
```

---

## Uninstall

### Bash
```bash
rm ~/.local/share/bash-completion/completions/promptly
# Or remove source line from ~/.bashrc
```

### Zsh
```bash
rm ~/.oh-my-zsh/custom/completions/_promptly
# Or rm ~/.zsh/completions/_promptly
```

### Fish
```bash
rm ~/.config/fish/completions/promptly.fish
```

---

## Contributing

Found a bug or want to add more completions?

1. Edit the appropriate script in `completions/`
2. Test thoroughly in each shell
3. Submit a PR!

---

**Tab completion makes Promptly even faster!** ⚡
