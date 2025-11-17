#!/usr/bin/env bash
# Installation script for Promptly shell completions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Promptly Shell Completion Installer${NC}\n"

# Detect shell
CURRENT_SHELL=$(basename "$SHELL")

install_bash_completion() {
    echo "Installing Bash completion..."

    # Determine completion directory
    if [ -d "$HOME/.bash_completion.d" ]; then
        COMPLETION_DIR="$HOME/.bash_completion.d"
    elif [ -d "/usr/local/etc/bash_completion.d" ]; then
        COMPLETION_DIR="/usr/local/etc/bash_completion.d"
    elif [ -d "/etc/bash_completion.d" ]; then
        COMPLETION_DIR="/etc/bash_completion.d"
    else
        # Create user-level completion directory
        COMPLETION_DIR="$HOME/.bash_completion.d"
        mkdir -p "$COMPLETION_DIR"
    fi

    # Copy completion file
    cp "$SCRIPT_DIR/promptly.bash" "$COMPLETION_DIR/promptly"

    # Add source to .bashrc if not already present
    if ! grep -q "bash_completion.d/promptly" "$HOME/.bashrc"; then
        echo "" >> "$HOME/.bashrc"
        echo "# Promptly completion" >> "$HOME/.bashrc"
        echo "[ -f $COMPLETION_DIR/promptly ] && source $COMPLETION_DIR/promptly" >> "$HOME/.bashrc"
    fi

    echo -e "${GREEN}✓${NC} Bash completion installed to: $COMPLETION_DIR/promptly"
    echo -e "${YELLOW}Note:${NC} Run 'source ~/.bashrc' or start a new terminal to activate"
}

install_zsh_completion() {
    echo "Installing Zsh completion..."

    # Determine completion directory
    if [ -d "$HOME/.oh-my-zsh/completions" ]; then
        COMPLETION_DIR="$HOME/.oh-my-zsh/completions"
    elif [ -d "/usr/local/share/zsh/site-functions" ]; then
        COMPLETION_DIR="/usr/local/share/zsh/site-functions"
    else
        # Create user-level completion directory
        COMPLETION_DIR="$HOME/.zsh/completions"
        mkdir -p "$COMPLETION_DIR"

        # Add to fpath in .zshrc if not already present
        if ! grep -q "fpath=.*$COMPLETION_DIR" "$HOME/.zshrc"; then
            echo "" >> "$HOME/.zshrc"
            echo "# Promptly completion" >> "$HOME/.zshrc"
            echo "fpath=($COMPLETION_DIR \$fpath)" >> "$HOME/.zshrc"
            echo "autoload -Uz compinit && compinit" >> "$HOME/.zshrc"
        fi
    fi

    # Copy completion file
    cp "$SCRIPT_DIR/promptly.zsh" "$COMPLETION_DIR/_promptly"

    echo -e "${GREEN}✓${NC} Zsh completion installed to: $COMPLETION_DIR/_promptly"
    echo -e "${YELLOW}Note:${NC} Run 'source ~/.zshrc' or start a new terminal to activate"
}

install_fish_completion() {
    echo "Installing Fish completion..."

    # Fish completion directory
    COMPLETION_DIR="$HOME/.config/fish/completions"
    mkdir -p "$COMPLETION_DIR"

    # Copy completion file
    cp "$SCRIPT_DIR/promptly.fish" "$COMPLETION_DIR/promptly.fish"

    echo -e "${GREEN}✓${NC} Fish completion installed to: $COMPLETION_DIR/promptly.fish"
    echo -e "${YELLOW}Note:${NC} Completion will be available in new fish sessions"
}

# Main installation logic
if [ "$1" = "bash" ] || [ "$CURRENT_SHELL" = "bash" ]; then
    install_bash_completion
elif [ "$1" = "zsh" ] || [ "$CURRENT_SHELL" = "zsh" ]; then
    install_zsh_completion
elif [ "$1" = "fish" ] || [ "$CURRENT_SHELL" = "fish" ]; then
    install_fish_completion
elif [ "$1" = "all" ]; then
    install_bash_completion
    install_zsh_completion
    install_fish_completion
else
    echo -e "${YELLOW}Usage:${NC} $0 [bash|zsh|fish|all]"
    echo ""
    echo "Detected shell: $CURRENT_SHELL"
    echo ""
    echo "Install for current shell? (y/n)"
    read -r response
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        case "$CURRENT_SHELL" in
            bash)
                install_bash_completion
                ;;
            zsh)
                install_zsh_completion
                ;;
            fish)
                install_fish_completion
                ;;
            *)
                echo -e "${RED}Error:${NC} Unsupported shell: $CURRENT_SHELL"
                exit 1
                ;;
        esac
    fi
fi

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Try these commands:"
echo "  promptly <TAB>          - Show available commands"
echo "  promptly get <TAB>      - Complete prompt names"
echo "  promptly checkout <TAB> - Complete branch names"
