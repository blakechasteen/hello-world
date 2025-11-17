#!/usr/bin/env bash
# Promptly CLI & TUI Demo Script
# Demonstrates all interactive features

set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Promptly CLI & TUI Demo                     ║${NC}"
echo -e "${CYAN}╔════════════════════════════════════════════════╗${NC}"
echo ""

# Function to show section
section() {
    echo ""
    echo -e "${GREEN}═══ $1 ═══${NC}"
    echo ""
}

# Function to run command with description
demo_cmd() {
    echo -e "${YELLOW}$ $1${NC}"
    sleep 1
    eval "$1"
    sleep 2
}

# Create demo directory
DEMO_DIR="/tmp/promptly_demo_$$"
mkdir -p "$DEMO_DIR"
cd "$DEMO_DIR"

section "1. Interactive REPL Demo"
echo "The Interactive REPL provides a command-line shell with:"
echo "  • Command history and auto-completion"
echo "  • Syntax highlighting"
echo "  • Context-aware prompts"
echo ""
echo "Commands to try in REPL:"
echo "  - init                    # Initialize repository"
echo "  - add <name> <content>    # Add a prompt"
echo "  - list                    # List all prompts"
echo "  - get <name>              # View prompt details"
echo "  - branches                # View all branches"
echo "  - help                    # Show help"
echo "  - exit                    # Exit REPL"
echo ""
echo "Launch with: python -m promptly.cli.interactive"
echo ""
read -p "Press Enter to continue..."

section "2. Enhanced CLI Demo"
echo "Demonstrating enhanced CLI with rich formatting..."
echo ""

# Initialize repository
demo_cmd "python -m promptly.cli.enhanced --help || python -c \"
import sys
sys.path.insert(0, '/home/user/hello-world/Promptly/promptly')
from promptly import Promptly
p = Promptly()
try:
    p.init()
    print('✓ Initialized Promptly repository')
except:
    print('Repository already exists')

# Add some demo prompts
p.add('summarizer', 'Summarize the following text in 2-3 sentences:\\n\\n{text}', {'category': 'text-processing'})
p.add('translator', 'Translate the following text to {target_language}:\\n\\n{text}', {'category': 'translation'})
p.add('code_reviewer', 'Review the following code and provide feedback:\\n\\n{code}', {'category': 'code'})
print('✓ Added 3 demo prompts')
\""

echo ""
read -p "Press Enter to see prompt list with rich formatting..."

demo_cmd "python -c \"
import sys
sys.path.insert(0, '/home/user/hello-world/Promptly/promptly')
from promptly import Promptly

p = Promptly()
prompts = p.list_prompts()

# Simple table display
print('\\nPrompts:')
print('-' * 60)
for prompt in prompts:
    print(f'{prompt[\"name\"]:20} v{prompt[\"version\"]:5} {prompt[\"commit_hash\"][:8]}')
print('-' * 60)
\""

echo ""
read -p "Press Enter to continue..."

section "3. Setup Wizards Demo"
echo "Setup wizards provide step-by-step guidance for:"
echo "  • Project setup"
echo "  • Prompt creation"
echo "  • Chain building"
echo "  • Evaluation setup"
echo "  • Template usage"
echo ""
echo "Example wizard commands:"
echo "  python -m promptly.cli.wizards project"
echo "  python -m promptly.cli.wizards prompt"
echo "  python -m promptly.cli.wizards chain"
echo "  python -m promptly.cli.wizards evaluation"
echo "  python -m promptly.cli.wizards template"
echo ""
read -p "Press Enter to continue..."

section "4. Terminal UI (TUI) Demo"
echo "The TUI provides a full-featured terminal interface with:"
echo "  • Tabbed interface (Prompts, Branches, Log, Eval, Chains)"
echo "  • Split-pane layout"
echo "  • Tree view for branches"
echo "  • Keyboard navigation"
echo "  • Mouse support"
echo ""
echo "Keyboard shortcuts:"
echo "  q        - Quit"
echo "  1-5      - Switch tabs"
echo "  r        - Refresh"
echo "  Tab      - Navigate widgets"
echo "  Ctrl+H   - Help"
echo ""
echo "Launch with: python -m promptly.tui.app"
echo ""
read -p "Press Enter to continue..."

section "5. Shell Completion Demo"
echo "Shell completion provides auto-complete for:"
echo "  • Commands"
echo "  • Prompt names"
echo "  • Branch names"
echo "  • Chain names"
echo "  • File formats"
echo ""
echo "Installation:"
echo "  cd /home/user/hello-world/Promptly/promptly/shell_completion"
echo "  ./install.sh"
echo ""
echo "Usage examples:"
echo "  promptly <TAB>              # Show all commands"
echo "  promptly get <TAB>          # Complete prompt names"
echo "  promptly checkout <TAB>     # Complete branch names"
echo ""
read -p "Press Enter to continue..."

section "6. Feature Comparison"
echo ""
echo "╔═══════════════╦══════╦═════╦══════════╦═════════╗"
echo "║ Feature       ║ REPL ║ TUI ║ Enhanced ║ Wizards ║"
echo "╠═══════════════╬══════╬═════╬══════════╬═════════╣"
echo "║ History       ║  ✓   ║  -  ║    -     ║    -    ║"
echo "║ Completion    ║  ✓   ║  -  ║    -     ║    -    ║"
echo "║ Visual Nav    ║  -   ║  ✓  ║    -     ║    -    ║"
echo "║ Rich Format   ║  ✓   ║  ✓  ║    ✓     ║    ✓    ║"
echo "║ Syntax Hl     ║  ✓   ║  ✓  ║    ✓     ║    -    ║"
echo "║ Progress Bars ║  -   ║  -  ║    ✓     ║    -    ║"
echo "║ Step-by-step  ║  -   ║  -  ║    -     ║    ✓    ║"
echo "║ Mouse Support ║  -   ║  ✓  ║    -     ║    -    ║"
echo "╚═══════════════╩══════╩═════╩══════════╩═════════╝"
echo ""

section "7. Use Cases"
echo ""
echo "Use Interactive REPL when:"
echo "  • Working daily with prompts"
echo "  • Exploring repository"
echo "  • Quick operations"
echo ""
echo "Use TUI when:"
echo "  • Need visual overview"
echo "  • Comparing prompts/branches"
echo "  • Prefer mouse interaction"
echo ""
echo "Use Enhanced CLI when:"
echo "  • Writing scripts"
echo "  • Automation tasks"
echo "  • CI/CD pipelines"
echo ""
echo "Use Wizards when:"
echo "  • Setting up new projects"
echo "  • Creating complex prompts"
echo "  • First-time users"
echo ""

section "Demo Complete!"
echo ""
echo "Try it yourself:"
echo ""
echo "  ${CYAN}# Interactive REPL${NC}"
echo "  python -m promptly.cli.interactive"
echo ""
echo "  ${CYAN}# Terminal UI${NC}"
echo "  python -m promptly.tui.app"
echo ""
echo "  ${CYAN}# Enhanced CLI${NC}"
echo "  python -m promptly.cli.enhanced status"
echo ""
echo "  ${CYAN}# Setup Wizard${NC}"
echo "  python -m promptly.cli.wizards project"
echo ""
echo "Documentation:"
echo "  • CLI_README.md - Quick start guide"
echo "  • CLI_TUI_GUIDE.md - Complete documentation"
echo ""
echo "Demo directory: $DEMO_DIR"
echo ""
echo -e "${GREEN}Happy prompting! 🚀${NC}"
