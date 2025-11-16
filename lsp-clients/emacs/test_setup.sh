#!/bin/bash
################################################################################
# HoloLoom LSP Client Setup Validation for Emacs
################################################################################
#
# This script validates that Emacs is properly configured for HoloLoom LSP.
#
# Checks performed:
#  ✓ Emacs version >= 27.1
#  ✓ Python 3.8+ installed
#  ✓ HoloLoom Python package available
#  ✓ lsp-mode package installed
#  ✓ lsp-ui package installed (recommended)
#  ✓ LSP server can start successfully
#  ✓ Configuration file is in place
#
# Usage:
#  bash lsp-clients/emacs/test_setup.sh
#
# Status: Production Ready (November 2025)
################################################################################

set -e  # Exit on first error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

print_check() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASSED++))
}

print_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((WARNINGS++))
}

print_fail() {
    echo -e "${RED}❌ $1${NC}"
    ((FAILED++))
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# ============================================================================
# Validation Functions
# ============================================================================

check_emacs() {
    print_header "Checking Emacs Installation"

    if ! command -v emacs &> /dev/null; then
        print_fail "Emacs not found in PATH"
        echo "  Install from: https://www.gnu.org/software/emacs/download.html"
        return 1
    fi

    emacs_version=$(emacs --version 2>&1 | head -n1 | grep -oP '\d+\.\d+')
    print_info "Emacs version: $emacs_version"

    # Check if version >= 27.1
    required_version="27.1"
    if [ "$(printf '%s\n' "$required_version" "$emacs_version" | sort -V | head -n1)" = "$required_version" ]; then
        print_check "Emacs version $emacs_version (required: >= 27.1)"
        return 0
    else
        print_fail "Emacs version $emacs_version (required: >= 27.1)"
        return 1
    fi
}

check_python() {
    print_header "Checking Python Installation"

    if ! command -v python3 &> /dev/null; then
        print_fail "Python 3 not found in PATH"
        echo "  Install from: https://www.python.org/downloads/"
        return 1
    fi

    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_info "Python version: $python_version"

    # Check if version >= 3.8
    required_version="3.8"
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
        print_check "Python $python_version (required: >= 3.8)"
        return 0
    else
        print_fail "Python $python_version (required: >= 3.8)"
        return 1
    fi
}

check_hololoom() {
    print_header "Checking HoloLoom Package"

    if python3 -c "import HoloLoom" 2>/dev/null; then
        hololoom_path=$(python3 -c "import HoloLoom; print(HoloLoom.__file__)")
        print_check "HoloLoom found at: $hololoom_path"
        return 0
    else
        print_fail "HoloLoom package not installed"
        echo "  Install with: pip install HoloLoom"
        echo "  Or from source:"
        echo "    git clone https://github.com/hololoom/hololoom.git"
        echo "    cd hololoom"
        echo "    pip install -e ."
        return 1
    fi
}

check_lsp_mode() {
    print_header "Checking lsp-mode Package"

    # Check common installation locations
    found=0

    # Check if installed via package.el
    emacs_out=$(emacs --batch -l ~/.emacs.d/init.el \
        -eval "(require 'lsp-mode nil t)" \
        -eval "(print (if (featurep 'lsp-mode) 'loaded 'not-loaded))" \
        2>/dev/null || echo "")

    if echo "$emacs_out" | grep -q "loaded"; then
        print_check "lsp-mode package is installed and loadable"
        return 0
    fi

    # Check elpa-admin or straight.el locations
    if [ -d "$HOME/.emacs.d/elpa/lsp-mode"* ]; then
        print_check "lsp-mode found in elpa directory"
        return 0
    fi

    if [ -d "$HOME/.emacs.d/straight/repos/lsp-mode" ]; then
        print_check "lsp-mode found in straight.el directory"
        return 0
    fi

    print_warn "lsp-mode not found in standard installation locations"
    echo "  Install via package.el (M-x package-install RET lsp-mode RET)"
    echo "  Or add to ~/.emacs.d/init.el:"
    echo "    (use-package lsp-mode"
    echo "      :ensure t)"
    return 1
}

check_lsp_ui() {
    print_header "Checking lsp-ui Package (Recommended)"

    emacs_out=$(emacs --batch -l ~/.emacs.d/init.el \
        -eval "(require 'lsp-ui nil t)" \
        -eval "(print (if (featurep 'lsp-ui) 'loaded 'not-loaded))" \
        2>/dev/null || echo "")

    if echo "$emacs_out" | grep -q "loaded"; then
        print_check "lsp-ui package is installed"
        return 0
    fi

    if [ -d "$HOME/.emacs.d/elpa/lsp-ui"* ]; then
        print_check "lsp-ui found in elpa directory"
        return 0
    fi

    print_warn "lsp-ui not found (optional, for better UI)"
    echo "  Install via package.el (M-x package-install RET lsp-ui RET)"
    echo "  Or add to ~/.emacs.d/init.el:"
    echo "    (use-package lsp-ui"
    echo "      :ensure t)"
    return 1
}

check_hololoom_config() {
    print_header "Checking HoloLoom LSP Configuration"

    emacs_config="$HOME/.emacs.d/init.el"

    if [ ! -f "$emacs_config" ]; then
        print_warn "Emacs configuration file not found: $emacs_config"
        echo "  Create it with basic HoloLoom LSP setup"
        return 1
    fi

    if grep -q "hololoom" "$emacs_config" 2>/dev/null; then
        print_check "HoloLoom configuration found in $emacs_config"
        return 0
    else
        print_warn "HoloLoom configuration not found in $emacs_config"
        echo "  Copy from repository:"
        echo "    cat lsp-clients/emacs/hololoom.el >> ~/.emacs.d/init.el"
        echo ""
        echo "  Or add manually:"
        echo "    (require 'hololoom nil t)"
        return 1
    fi
}

check_lsp_server() {
    print_header "Testing LSP Server Startup"

    # Try to start server with 5 second timeout
    if timeout 5 python3 -m HoloLoom.lsp.server --help &> /dev/null; then
        print_check "HoloLoom LSP server can start successfully"
        return 0
    else
        print_fail "HoloLoom LSP server failed to start"
        echo "  Try manually:"
        echo "    python3 -m HoloLoom.lsp.server --log-level DEBUG"
        return 1
    fi
}

check_json_rpc() {
    print_header "Checking json-rpc Library"

    if python3 -c "import jsonrpc" 2>/dev/null; then
        print_check "jsonrpc library installed"
        return 0
    else
        print_warn "jsonrpc library not found (needed for LSP communication)"
        echo "  Install with: pip install python-jsonrpc"
        return 1
    fi
}

check_emacs_init() {
    print_header "Checking Emacs Initialization"

    emacs_config="$HOME/.emacs.d/init.el"

    if [ ! -f "$emacs_config" ]; then
        print_fail "No init.el configuration found"
        echo "  Create ~/.emacs.d/init.el and add:"
        echo "    (require 'lsp-mode)"
        echo "    (require 'hololoom nil t)"
        echo "    (hololoom-setup)"
        return 1
    fi

    if grep -q "lsp-mode" "$emacs_config" 2>/dev/null; then
        print_check "lsp-mode is loaded in init.el"
        return 0
    else
        print_warn "lsp-mode not explicitly loaded in init.el"
        echo "  Add to ~/.emacs.d/init.el:"
        echo "    (require 'lsp-mode)"
        return 1
    fi
}

check_emacs_version_details() {
    print_header "Checking Emacs Features"

    # Check for native JSON support (Emacs 27+)
    if emacs --batch -eval "(print (fboundp 'json-parse-string))" 2>/dev/null | grep -q "t"; then
        print_check "Native JSON support available (Emacs 27.1+)"
        return 0
    else
        print_warn "Native JSON support may not be available"
        echo "  Consider upgrading to Emacs 27.1 or later"
        return 1
    fi
}

# ============================================================================
# Test Suite Execution
# ============================================================================

run_all_checks() {
    print_header "HoloLoom LSP Setup Validation for Emacs"

    # Run all checks
    check_emacs
    check_python
    check_hololoom
    check_lsp_mode
    check_lsp_ui
    check_json_rpc
    check_emacs_version_details
    check_emacs_init
    check_hololoom_config
    check_lsp_server

    # Print summary
    print_header "Validation Summary"

    echo -e "Results:"
    echo -e "  ${GREEN}✅ Passed:${NC}  $PASSED"
    echo -e "  ${YELLOW}⚠️  Warnings:${NC}  $WARNINGS"
    echo -e "  ${RED}❌ Failed:${NC}  $FAILED"
    echo ""

    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}✅ All checks passed! HoloLoom LSP is ready to use.${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Ensure this is in ~/.emacs.d/init.el:"
        echo "     (require 'lsp-mode)"
        echo "     (require 'hololoom nil t)"
        echo ""
        echo "  2. Start Emacs:"
        echo "     emacs test.py"
        echo ""
        echo "  3. Test features:"
        echo "     - Type and wait for completion popup"
        echo "     - Use M-x lsp-describe-session for hover info"
        echo "     - M-. to go to definition"
        echo "     - M-x lsp-workspace-symbol for symbol search"
        echo ""
        return 0
    else
        if [ $WARNINGS -gt 0 ]; then
            echo -e "${YELLOW}⚠️  Setup has some warnings (see above)${NC}"
        fi
        if [ $FAILED -gt 0 ]; then
            echo -e "${RED}❌ Setup failed. Please fix errors above and retry.${NC}"
        fi
        echo ""
        echo "Troubleshooting:"
        echo "  1. Check Emacs LSP status: M-x lsp-describe-session"
        echo "  2. View LSP logs: M-x lsp-switch-to-io-log-buffer"
        echo "  3. Test server directly: python3 -m HoloLoom.lsp.server --log-level DEBUG"
        echo "  4. Verify installation: python3 -c 'import HoloLoom; print(HoloLoom.__file__)'"
        echo ""
        return 1
    fi
}

# ============================================================================
# Main Entry Point
# ============================================================================

main() {
    echo ""
    echo "Starting HoloLoom LSP Setup Validation..."
    echo ""

    run_all_checks
    exit_code=$?

    echo ""
    echo "Validation complete at $(date)"
    echo ""

    exit $exit_code
}

# Run main if script is executed (not sourced)
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi
