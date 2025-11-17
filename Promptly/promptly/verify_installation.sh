#!/usr/bin/env bash
# Verification script for Promptly CLI & TUI installation

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Promptly CLI & TUI Installation Verifier    ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════╝${NC}"
echo ""

ERRORS=0
WARNINGS=0

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1 - MISSING"
        ((ERRORS++))
        return 1
    fi
}

# Function to check executable
check_executable() {
    if [ -x "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 (executable)"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $1 (not executable)"
        ((WARNINGS++))
        return 1
    fi
}

# Function to check Python module
check_module() {
    if python3 -c "import $1" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Python module: $1"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} Python module: $1 (optional)"
        ((WARNINGS++))
        return 1
    fi
}

echo -e "${CYAN}Checking Python modules...${NC}"
echo ""

check_module "click"
check_module "yaml"
echo ""

echo -e "${CYAN}Checking optional dependencies...${NC}"
echo ""

check_module "rich"
check_module "prompt_toolkit"
check_module "pygments"
check_module "textual"
echo ""

echo -e "${CYAN}Checking CLI module files...${NC}"
echo ""

check_file "cli/__init__.py"
check_file "cli/interactive.py"
check_file "cli/enhanced.py"
check_file "cli/wizards.py"
echo ""

echo -e "${CYAN}Checking TUI module files...${NC}"
echo ""

check_file "tui/__init__.py"
check_file "tui/app.py"
echo ""

echo -e "${CYAN}Checking shell completion files...${NC}"
echo ""

check_file "shell_completion/promptly.bash"
check_file "shell_completion/promptly.zsh"
check_file "shell_completion/promptly.fish"
check_executable "shell_completion/install.sh"
echo ""

echo -e "${CYAN}Checking entry point scripts...${NC}"
echo ""

check_executable "promptly-interactive"
check_executable "promptly-tui"
check_executable "promptly-enhanced"
check_executable "promptly-wizard"
check_executable "cli_tui_demo.sh"
echo ""

echo -e "${CYAN}Checking documentation files...${NC}"
echo ""

check_file "CLI_README.md"
check_file "CLI_TUI_GUIDE.md"
check_file "CLI_TUI_IMPLEMENTATION.md"
check_file "FEATURE_SHOWCASE.md"
check_file "../INTERACTIVE_CLI_DELIVERABLE.md"
echo ""

echo -e "${CYAN}Testing imports...${NC}"
echo ""

if python3 -c "import sys; sys.path.insert(0, '.'); from cli.interactive import InteractiveREPL" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} CLI interactive import"
else
    echo -e "${RED}✗${NC} CLI interactive import - FAILED"
    ((ERRORS++))
fi

if python3 -c "import sys; sys.path.insert(0, '.'); from cli.enhanced import EnhancedCLI" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} CLI enhanced import"
else
    echo -e "${RED}✗${NC} CLI enhanced import - FAILED"
    ((ERRORS++))
fi

if python3 -c "import sys; sys.path.insert(0, '.'); from cli.wizards import ProjectWizard" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} CLI wizards import"
else
    echo -e "${RED}✗${NC} CLI wizards import - FAILED"
    ((ERRORS++))
fi

# TUI might fail if textual not installed
if python3 -c "import sys; sys.path.insert(0, '.'); from tui.app import PromptlyTUI" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} TUI import"
else
    echo -e "${YELLOW}⚠${NC} TUI import - Install textual for TUI support"
    ((WARNINGS++))
fi

echo ""
echo -e "${CYAN}Checking file permissions...${NC}"
echo ""

if [ -r "cli/interactive.py" ] && [ -r "cli/enhanced.py" ] && [ -r "cli/wizards.py" ]; then
    echo -e "${GREEN}✓${NC} All CLI files are readable"
else
    echo -e "${RED}✗${NC} Some files are not readable"
    ((ERRORS++))
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ Installation verified successfully!${NC}"
    echo ""
    echo "All components are installed and ready to use."
    echo ""
    echo "Try these commands:"
    echo "  ./promptly-interactive    # Interactive REPL"
    echo "  ./promptly-tui            # Terminal UI"
    echo "  ./promptly-enhanced       # Enhanced CLI"
    echo "  ./promptly-wizard         # Setup wizards"
    echo "  ./cli_tui_demo.sh         # Run demo"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Installation verified with warnings${NC}"
    echo ""
    echo "Core components are installed."
    echo "Warnings: $WARNINGS (optional features)"
    echo ""
    echo "To install optional dependencies:"
    echo "  pip install rich prompt_toolkit pygments textual"
elif [ $ERRORS -gt 0 ]; then
    echo -e "${RED}✗ Installation has errors${NC}"
    echo ""
    echo "Errors: $ERRORS"
    echo "Warnings: $WARNINGS"
    echo ""
    echo "Please check the errors above and fix them."
    exit 1
fi

echo ""
echo "Documentation:"
echo "  cat CLI_README.md              # Quick start"
echo "  cat CLI_TUI_GUIDE.md           # Complete guide"
echo "  cat FEATURE_SHOWCASE.md        # Examples"
echo ""
