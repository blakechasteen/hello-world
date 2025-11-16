#!/bin/bash
################################################################################
# HoloLoom LSP Client Setup Validation for Neovim
################################################################################
#
# This script validates that Neovim is properly configured for HoloLoom LSP.
#
# Checks performed:
#  ✓ Neovim version >= 0.8.0
#  ✓ Python 3.8+ installed
#  ✓ HoloLoom Python package available
#  ✓ nvim-lspconfig plugin installed
#  ✓ LSP server can start successfully
#  ✓ Configuration file is in place
#
# Usage:
#  bash lsp-clients/neovim/test_setup.sh
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

check_neovim() {
    print_header "Checking Neovim Installation"

    if ! command -v nvim &> /dev/null; then
        print_fail "Neovim not found in PATH"
        echo "  Install from: https://neovim.io/download/"
        return 1
    fi

    nvim_version=$(nvim --version | head -n1 | awk '{print $2}')
    print_info "Neovim version: $nvim_version"

    # Parse version (e.g., "v0.9.0" -> 0.9.0)
    version_num=$(echo $nvim_version | sed 's/^v//')

    # Check if version >= 0.8.0
    required_version="0.8.0"
    if [ "$(printf '%s\n' "$required_version" "$version_num" | sort -V | head -n1)" = "$required_version" ]; then
        print_check "Neovim version $version_num (required: >= 0.8.0)"
        return 0
    else
        print_fail "Neovim version $version_num (required: >= 0.8.0)"
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

    if ! python3 -c "import HoloLoom" 2>&1 | grep -q "No module named"; then
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

check_lspconfig() {
    print_header "Checking nvim-lspconfig Plugin"

    lspconfig_path="$HOME/.config/nvim/pack/packer/start/nvim-lspconfig"
    lazy_path="$HOME/.local/share/nvim/lazy/nvim-lspconfig"

    if [ -d "$lspconfig_path" ]; then
        print_check "nvim-lspconfig found at: $lspconfig_path"
        return 0
    elif [ -d "$lazy_path" ]; then
        print_check "nvim-lspconfig found at: $lazy_path"
        return 0
    else
        print_warn "nvim-lspconfig not found in standard locations"
        echo "  Please ensure nvim-lspconfig is installed via your plugin manager:"
        echo ""
        echo "  Using lazy.nvim:"
        echo "    { 'neovim/nvim-lspconfig', lazy = false }"
        echo ""
        echo "  Using packer.nvim:"
        echo "    use 'neovim/nvim-lspconfig'"
        echo ""
        echo "  Manual installation:"
        echo "    git clone https://github.com/neovim/nvim-lspconfig.git \\"
        echo "      ~/.config/nvim/pack/packer/start/nvim-lspconfig"
        return 1
    fi
}

check_hololoom_config() {
    print_header "Checking HoloLoom LSP Configuration"

    config_path="$HOME/.config/nvim/lua/hololoom.lua"

    if [ -f "$config_path" ]; then
        print_check "hololoom.lua found at: $config_path"
        return 0
    else
        print_warn "hololoom.lua not found at: $config_path"
        echo "  Copy from repository:"
        echo "    mkdir -p ~/.config/nvim/lua/"
        echo "    cp lsp-clients/neovim/hololoom.lua ~/.config/nvim/lua/"
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

check_nvim_lua() {
    print_header "Checking Neovim Lua Support"

    if nvim -c "lua print('success')" -c "q!" 2>&1 | grep -q "success"; then
        print_check "Neovim Lua support working"
        return 0
    else
        print_warn "Neovim Lua support may not be working"
        echo "  Try checking in Neovim:"
        echo "    :lua print('test')"
        return 1
    fi
}

check_config_format() {
    print_header "Checking Neovim Configuration"

    if [ -f "$HOME/.config/nvim/init.lua" ]; then
        print_check "Using init.lua configuration"
        if grep -q "hololoom\|HoloLoom\|lsp" "$HOME/.config/nvim/init.lua" 2>/dev/null; then
            print_info "LSP configuration found in init.lua"
        else
            print_warn "HoloLoom LSP not yet configured in init.lua"
            echo "  Add to ~/.config/nvim/init.lua:"
            echo "    require('hololoom').setup({})"
        fi
        return 0
    elif [ -f "$HOME/.config/nvim/init.vim" ]; then
        print_info "Using init.vim configuration (legacy)"
        print_warn "We recommend migrating to init.lua for better LSP support"
        return 0
    else
        print_fail "No Neovim configuration found"
        echo "  Create ~/.config/nvim/init.lua with:"
        echo "    require('hololoom').setup({})"
        return 1
    fi
}

# ============================================================================
# Test Suite Execution
# ============================================================================

run_all_checks() {
    print_header "HoloLoom LSP Setup Validation for Neovim"

    # Run all checks
    check_neovim
    check_python
    check_hololoom
    check_lspconfig
    check_hololoom_config
    check_nvim_lua
    check_config_format
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
        echo "  1. Add to ~/.config/nvim/init.lua:"
        echo "     require('hololoom').setup({})"
        echo ""
        echo "  2. Start Neovim:"
        echo "     nvim test.py"
        echo ""
        echo "  3. Test features:"
        echo "     - Type '.' to trigger completion"
        echo "     - Press 'K' for hover information"
        echo "     - Press 'gd' for go-to-definition"
        echo "     - Press '<leader>ws' for workspace symbol search"
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
        echo "  1. Check Neovim logs: :LspInfo and :LspLog"
        echo "  2. Test server directly: python3 -m HoloLoom.lsp.server --log-level DEBUG"
        echo "  3. Verify installation: python3 -c 'import HoloLoom; print(HoloLoom.__file__)'"
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
