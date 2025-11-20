# HoloLoom LSP Client for Neovim

Complete Language Server Protocol (LSP) client configuration for Neovim, enabling access to HoloLoom's neural memory system directly from your editor.

**Status**: Production Ready (November 2025)
**Language**: Lua (Neovim configuration)
**Dependencies**: Neovim 0.7+, nvim-lspconfig

## Overview

The HoloLoom LSP client provides semantic code intelligence powered by HoloLoom's neural memory system:

- **Code Completion**: Autocomplete from HoloLoom memories and knowledge graph
- **Hover Information**: Entity definitions and context from the knowledge graph
- **Go-to-Definition**: Navigate via semantic relationships
- **Workspace Symbol Search**: Find entities across your project semantically
- **Diagnostic Reports**: Alignment framework safety checks and insights
- **Zero Configuration**: Works out of the box with sensible defaults

## Features

### 1. Code Completion
Triggered by typing `.` or manually with `<C-Space>`:
```python
hololoom. # Shows: memory, weave, recall, reflect
```

### 2. Hover Information
Press `K` to see entity documentation:
```
**HoloLoom Entity**
Entity definition from knowledge graph
Related entities and relationships
Usage examples from semantic memory
Confidence scores
```

### 3. Go-to-Definition
Press `gd` to jump to symbol definition via knowledge graph relationships.

### 4. Find References
Press `gr` to find all references to a symbol across your workspace.

### 5. Workspace Symbol Search
Press `<leader>ws` to search for entities semantically across the workspace:
```
:Workspace symbol search for "Thompson Sampling"
```

### 6. Document Symbols
Press `<leader>ds` to see all symbols in current document.

### 7. Diagnostic Reports
Automatic diagnostics from HoloLoom's alignment framework with:
- Risk assessment (LOW/MEDIUM/HIGH/CRITICAL)
- Safety guardrails feedback
- Goal transparency checks

## Prerequisites

### Required
- **Neovim** >= 0.7.0 (check with `:version`)
- **Python** >= 3.8 (for running HoloLoom LSP server)
- **nvim-lspconfig** plugin

### Optional (for enhanced features)
- **nvim-cmp**: Better autocompletion with window and docs
- **nvim-telescope**: Fuzzy finder for symbol search
- **luasnip**: Snippet support in completions

### Check Your Setup

```bash
# Check Neovim version
nvim --version

# Check Python availability
python3 --version

# Check if HoloLoom is installed
python3 -c "import HoloLoom; print(HoloLoom.__file__)"
```

## Installation

Choose the method that matches your Neovim setup:

### Method 1: Using lazy.nvim (RECOMMENDED)

1. **Copy the configuration file** to your Neovim config:
```bash
mkdir -p ~/.config/nvim/lua/
cp lsp-clients/neovim/hololoom.lua ~/.config/nvim/lua/hololoom.lua
```

2. **Add to your `lazy.nvim` spec** in `~/.config/nvim/init.lua`:
```lua
{
    'lazy/hololoom-lsp',
    ft = { 'python', 'typescript', 'javascript', 'tsx', 'jsx' },
    config = function()
        require('hololoom').setup({
            enabled = true,
            auto_start = true,
            log_level = "INFO",
        })
    end,
}
```

3. **Run `:Lazy sync`** to install and setup the plugin.

### Method 2: Using packer.nvim

1. **Copy the configuration file** to your Neovim config:
```bash
mkdir -p ~/.config/nvim/lua/
cp lsp-clients/neovim/hololoom.lua ~/.config/nvim/lua/hololoom.lua
```

2. **Add to your `packer.nvim` spec** in `~/.config/nvim/init.lua`:
```lua
use {
    'hololoom/lsp',
    config = function()
        require('hololoom').setup({
            enabled = true,
            log_level = "INFO",
        })
    end,
}
```

3. **Run `:PackerSync`** to install and setup the plugin.

### Method 3: Manual Setup (No Plugin Manager)

1. **Copy the configuration file**:
```bash
mkdir -p ~/.config/nvim/lua/
cp lsp-clients/neovim/hololoom.lua ~/.config/nvim/lua/hololoom.lua
```

2. **Manually install nvim-lspconfig** (if not already installed):
```bash
mkdir -p ~/.config/nvim/pack/packer/start
cd ~/.config/nvim/pack/packer/start
git clone https://github.com/neovim/nvim-lspconfig.git
```

3. **Add to your `~/.config/nvim/init.lua`**:
```lua
require('hololoom').setup({
    enabled = true,
    auto_start = true,
    log_level = "INFO",
})
```

4. **Restart Neovim**.

## Configuration

The HoloLoom LSP client is configured via the `setup()` function. All options are optional:

```lua
require('hololoom').setup({
    -- Enable or disable the LSP client
    enabled = true,

    -- Auto-start the server when opening supported filetypes
    auto_start = true,

    -- Path to Python executable (use full path if not in PATH)
    cmd_path = "python",

    -- Arguments to pass to Python when starting the server
    cmd_args = { "-m", "HoloLoom.lsp.server" },

    -- Logging level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    log_level = "INFO",

    -- File types to attach the LSP server to
    filetypes = { "python", "typescript", "javascript", "jsx", "tsx" },

    -- Patterns to detect project root directory
    root_patterns = { ".git", "pyproject.toml", "package.json", "setup.py" },

    -- Completion settings
    completion = {
        enabled = true,
        trigger_characters = { "." },
    },

    -- Hover settings
    hover = {
        enabled = true,
        border = "rounded",  -- or "solid", "double", "shadow", "none"
    },

    -- Definition settings
    definition = {
        enabled = true,
    },

    -- Symbol search settings
    symbol = {
        enabled = true,
    },
})
```

## Keybindings Reference

All keybindings are automatically configured by the LSP client. Here's the complete reference:

### Navigation
| Key | Action | Description |
|-----|--------|-------------|
| `gd` | Go to Definition | Jump to symbol definition |
| `gD` | Go to Declaration | Jump to symbol declaration |
| `gr` | Find References | Find all symbol references |
| `<leader>gi` | Implementation | Go to implementation |
| `<leader>td` | Type Definition | Go to type definition |

### Information
| Key | Action | Description |
|-----|--------|-------------|
| `K` | Hover | Show hover documentation |
| `<C-k>` (Insert) | Signature Help | Show function signature |

### Search & Symbols
| Key | Action | Description |
|-----|--------|-------------|
| `<leader>ws` | Workspace Symbol | Search for symbols in workspace |
| `<leader>ds` | Document Symbols | Show symbols in current document |

### Editing
| Key | Action | Description |
|-----|--------|-------------|
| `<leader>rn` | Rename | Rename symbol at cursor |
| `<leader>ca` | Code Action | Show code actions menu |
| `<leader>fm` | Format Document | Format entire document |

### Diagnostics
| Key | Action | Description |
|-----|--------|-------------|
| `<leader>dn` | Next Diagnostic | Jump to next diagnostic |
| `<leader>dp` | Previous Diagnostic | Jump to previous diagnostic |
| `<leader>dd` | Show Diagnostics | Show diagnostics for current line |
| `<leader>dl` | Diagnostics List | Open diagnostics in quickfix list |

### Server Control
Use the custom commands (type in command mode):

```vim
:HoloLoomReload    " Reload the LSP server
:HoloLoomStop      " Stop the LSP server
:HoloLoomRestart   " Restart the LSP server
```

## Usage Examples

### Example 1: Code Completion

Open a Python file and start typing:
```python
from HoloLoom import

# Type <C-Space> or wait for auto-trigger
# Shows: hololoom, weave, recall, reflect, etc.
```

### Example 2: Hover Information

Position cursor on a symbol and press `K`:
```python
weave_orchestrator = WeavingOrchestrator()
                     ^ Press K here

# Shows: WeavingOrchestrator class documentation
```

### Example 3: Go to Definition

Position cursor on a symbol and press `gd`:
```python
result = orchestrator.weave(query)
                     ^ Press gd here

# Jumps to weave() method definition in knowledge graph
```

### Example 4: Find References

Position cursor on a symbol and press `gr`:
```python
Config.fast()
^ Press gr here

# Shows all references to Config.fast() across your project
```

### Example 5: Workspace Symbol Search

Press `<leader>ws` and type a query:
```
:Workspace symbol search for: Thompson
# Shows: ThompsonSampling, thompson_strategy, etc.
```

### Example 6: Rename Symbol

Position cursor on a symbol and press `<leader>rn`:
```python
def old_function_name():
   ^ Press <leader>rn here

# Enter new name: new_function_name
# Updates all references automatically
```

## Troubleshooting

### Issue: "nvim-lspconfig not found"

**Solution**: Install nvim-lspconfig:

Using lazy.nvim:
```lua
{
    'neovim/nvim-lspconfig',
    lazy = true,
}
```

Using packer.nvim:
```lua
use 'neovim/nvim-lspconfig'
```

Manually:
```bash
git clone https://github.com/neovim/nvim-lspconfig ~/.config/nvim/pack/packer/start/nvim-lspconfig
```

### Issue: "Failed to load HoloLoom LSP client"

**Solution**: Verify the hololoom.lua file is in the correct location:
```bash
# Check if file exists
ls ~/.config/nvim/lua/hololoom.lua

# If not, copy it:
cp lsp-clients/neovim/hololoom.lua ~/.config/nvim/lua/hololoom.lua
```

### Issue: "Language Server crashed or failed to start"

**Solution**: Check server logs:
```vim
:LspInfo          " Show LSP status and logs
:LspLog           " View LSP log file
```

Or check Python directly:
```bash
# Test if HoloLoom server can start
python -m HoloLoom.lsp.server --log-level DEBUG

# If not found, install HoloLoom:
pip install HoloLoom
# or from source:
git clone https://github.com/hololoom/hololoom.git
cd hololoom
pip install -e .
```

### Issue: Completion not working

**Solution**: Enable debug logging to see what's happening:

In your Neovim config:
```lua
require('hololoom').setup({
    log_level = "DEBUG",  -- More verbose logging
})
```

Then check logs:
```vim
:LspLog
```

### Issue: Server won't start on remote system

**Solution**: Specify full Python path:

```lua
require('hololoom').setup({
    cmd_path = "/usr/local/bin/python3",  -- Full path to Python
})
```

Or check if HoloLoom is installed in the remote Python:
```bash
ssh user@remote
python3 -c "import HoloLoom; print(HoloLoom.__file__)"
```

### Issue: Diagnostics not showing

**Solution**: Enable diagnostics explicitly in your config:

```lua
-- Add this to your init.lua
vim.diagnostic.config({
    virtual_text = true,
    signs = true,
    underline = true,
    update_in_insert = false,
    severity_sort = true,
})
```

Then verify in Neovim:
```vim
:LspInfo    " Check if server is attached and showing diagnostics
```

### Issue: Keybindings not working

**Solution**: Check if they're conflicting with other plugins:

```vim
" Check mapping in Neovim
:nmap gd    " Shows what's mapped to 'gd'
:nmap K     " Shows what's mapped to 'K'
```

If conflicting, customize keybindings in your config:
```lua
-- After require('hololoom').setup()
local opts = { noremap = true, silent = true }
vim.keymap.set('n', '<leader>ld', vim.lsp.buf.definition, opts)
vim.keymap.set('n', '<leader>lh', vim.lsp.buf.hover, opts)
```

## Performance Optimization

### Reduce Server Overhead

For slower systems, optimize the server:

```lua
require('hololoom').setup({
    -- Start only for specific filetypes
    filetypes = { "python" },  -- Remove javascript, typescript if not needed

    -- Use INFO logging instead of DEBUG
    log_level = "INFO",

    -- Disable features you don't use
    completion = { enabled = false },
    hover = { enabled = false },
})
```

### Enable Completion Window (Optional)

For a better autocomplete experience, install nvim-cmp:

```lua
{
    'hrsh7th/nvim-cmp',
    dependencies = {
        'neovim/nvim-lspconfig',
        'hrsh7th/cmp-nvim-lsp',
    },
    config = function()
        local cmp = require('cmp')
        cmp.setup({
            mapping = cmp.mapping.preset.insert({
                ['<C-b>'] = cmp.mapping.scroll_docs(-4),
                ['<C-f>'] = cmp.mapping.scroll_docs(4),
                ['<C-Space>'] = cmp.mapping.complete(),
                ['<CR>'] = cmp.mapping.confirm({ select = true }),
            }),
            sources = cmp.config.sources({
                { name = 'nvim_lsp' },
                { name = 'buffer' },
            }),
        })
    end,
}
```

## Advanced Configuration

### Custom Root Pattern Detection

Detect project root by checking for specific files:

```lua
require('hololoom').setup({
    root_patterns = {
        ".git",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "Makefile",
        "build.gradle",
        ".hololoom.config",
    },
})
```

### Enable Autoformat on Save

Uncomment in hololoom.lua's `setup()` function or add:

```lua
vim.api.nvim_create_autocmd("BufWritePre", {
    group = vim.api.nvim_create_augroup("HoloLoomFormat", { clear = true }),
    pattern = "*.py,*.ts,*.js",
    callback = function()
        vim.lsp.buf.format()
    end,
})
```

### Customize Diagnostic Display

```lua
vim.diagnostic.config({
    -- Show inline virtual text
    virtual_text = {
        prefix = "● ",
        source = "if_many",
    },
    -- Customize signs (✓, ✗, ⚠, ℹ)
    signs = {
        Error = "✗",
        Warn = "⚠",
        Hint = "ℹ",
        Info = "ℹ",
    },
    -- Underline and color
    underline = true,
    -- Update in insert mode
    update_in_insert = false,
    -- Sort by severity
    severity_sort = true,
})
```

## Testing the Setup

After installation, verify everything works:

1. **Open a Python file**:
```bash
nvim test.py
```

2. **Trigger completion**:
```
# Type:
from HoloLoom<C-Space>

# Should show completions from HoloLoom
```

3. **Test hover**:
```
# Position cursor on HoloLoom and press K
# Should show documentation
```

4. **Check server status**:
```vim
:LspInfo
# Should show HoloLoom language server attached
```

## Debugging

### Enable Debug Logging

```lua
require('hololoom').setup({
    log_level = "DEBUG",  -- Very verbose
})

-- Then check logs:
-- :LspLog  (in Neovim)
```

### Test Server Directly

```bash
# Start server in debug mode
python -m HoloLoom.lsp.server --log-level DEBUG

# In another terminal, send test requests
# (requires lsp-test tool or similar)
```

### Check Python Environment

```bash
# Ensure HoloLoom is installed
python -c "import HoloLoom; print(HoloLoom.__version__)"

# Check installed dependencies
pip list | grep -i hololoom
```

## Support & Documentation

- **HoloLoom Documentation**: https://github.com/hololoom/hololoom
- **LSP Specification**: https://microsoft.github.io/language-server-protocol/
- **Neovim LSP Guide**: https://neovim.io/doc/user/lsp.html
- **nvim-lspconfig**: https://github.com/neovim/nvim-lspconfig

## FAQ

**Q: Can I use HoloLoom LSP with other LSPs (Pylance, Pyright, etc.)?**
A: Yes! Multiple LSP servers can be attached to the same buffer. Each provides its own features. Configure:
```lua
require('pyright').setup({...})  -- Pyright for type checking
require('hololoom').setup({...}) -- HoloLoom for semantic intelligence
```

**Q: Does HoloLoom LSP require Internet?**
A: No. The server runs locally and uses your local HoloLoom knowledge graph.

**Q: Will HoloLoom LSP slow down my editor?**
A: Minimal impact (<50ms per request). Server runs in background. Diagnostics are async.

**Q: Can I customize the keybindings?**
A: Yes! Override in your config after setup or modify hololoom.lua's `on_attach()` function.

**Q: What file types are supported?**
A: By default: Python, TypeScript, JavaScript, JSX, TSX. Configurable via `filetypes` option.

**Q: How do I update to the latest version?**
A: Update HoloLoom:
```bash
pip install --upgrade HoloLoom
```
Restart Neovim or run `:HoloLoomRestart` in the editor.

## License

Same as HoloLoom project. See LICENSE file in HoloLoom repository.

## Contributing

Contributions welcome! To report issues or suggest improvements:

1. Check HoloLoom LSP GitHub issues
2. File a new issue with reproduction steps
3. Submit pull requests with improvements

---

**Last Updated**: November 2025
**Version**: 0.1.0
**Status**: Production Ready
