# HoloLoom LSP Quick Start Guide

**Setup Time**: 5 minutes per editor
**Difficulty**: Beginner-friendly
**Updated**: November 2025

---

## Table of Contents

1. [Installation (All Editors)](#installation-all-editors)
2. [VS Code (5 minutes)](#vs-code-5-minutes)
3. [Neovim (5 minutes)](#neovim-5-minutes)
4. [Emacs (5 minutes)](#emacs-5-minutes)
5. [Vim (7 minutes)](#vim-7-minutes)
6. [Sublime Text (5 minutes)](#sublime-text-5-minutes)
7. [Quick Reference: Commands](#quick-reference-commands)
8. [Troubleshooting Flowchart](#troubleshooting-flowchart)

---

## Installation (All Editors)

### Step 1: Install HoloLoom

```bash
# Clone repository
git clone https://github.com/user/hololoom.git
cd hololoom

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install pygls asyncio-contextmanager
pip install -e .

# Verify installation
python -c "from HoloLoom.lsp.server import server; print('✓ Ready!')"
```

### Step 2: Start the Server

**Option A: Automatic (Recommended)**
- Editor will start server automatically when opening a file
- No manual action needed

**Option B: Manual (For debugging)**
```bash
# Terminal 1: Start server
PYTHONPATH=. python -m HoloLoom.lsp.server --log-level INFO

# Terminal 2: Open your editor
# The server will now be ready to serve LSP requests
```

---

## VS Code (5 minutes)

### 1. Install LSP Client Extension

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "LSP Client"
4. Install the **LSP Client** extension by Jo Bingen

### 2. Configure LSP Client

1. Open Settings (Ctrl+,)
2. Search for "lsp"
3. Click "Edit in settings.json"
4. Add this configuration:

```json
{
  "lsp": {
    "hololoom": {
      "command": "python",
      "args": ["-m", "HoloLoom.lsp.server"],
      "languages": ["python", "typescript", "javascript"],
      "initializationOptions": {},
      "trace.server": "off"
    }
  }
}
```

### 3. Verify Connection

1. Open a Python file
2. You should see "hololoom" in the status bar at the bottom
3. Try pressing Ctrl+Space for code completion
4. Try hovering over a variable

### ✓ Complete!

**Keybindings**:
| Action | Shortcut |
|--------|----------|
| Completion | Ctrl+Space |
| Hover | Hover mouse |
| Go to Definition | Ctrl+Click or F12 |
| Find References | Ctrl+Shift+H |

---

## Neovim (5 minutes)

### 1. Install nvim-lspconfig

If using a plugin manager like `packer.nvim`:

```lua
use 'neovim/nvim-lspconfig'
```

Or with `vim-plug`:

```vim
Plug 'neovim/nvim-lspconfig'
```

Then run `:PlugInstall`

### 2. Configure Neovim

Add to `~/.config/nvim/init.lua`:

```lua
local lspconfig = require('lspconfig')

lspconfig.hololoom.setup {
    cmd = {"python", "-m", "HoloLoom.lsp.server"},
    filetypes = {"python"},
    root_dir = lspconfig.util.root_pattern(".git", "setup.py"),
}

-- (Optional) Set keybindings
local opts = { noremap=true, silent=true }
vim.keymap.set('n', 'gd', vim.lsp.buf.definition, opts)
vim.keymap.set('n', 'K', vim.lsp.buf.hover, opts)
vim.keymap.set('n', 'gr', vim.lsp.buf.references, opts)
vim.keymap.set('n', '<space>ca', vim.lsp.buf.code_action, opts)
```

### 3. Verify Connection

1. Open a Python file
2. Run `:LspInfo` in Neovim
3. Should show "hololoom" as "running"
4. Try pressing `gd` to go to definition

### ✓ Complete!

**Default Keybindings** (LSP-standard in Neovim):
| Action | Shortcut |
|--------|----------|
| Completion | Ctrl+X Ctrl+O (insert mode) |
| Hover | K (normal mode) |
| Go to Definition | gd |
| Find References | gr |

**Or customize** in your init.lua with:
```lua
vim.keymap.set('n', '<leader>d', vim.lsp.buf.definition, opts)
vim.keymap.set('n', '<leader>h', vim.lsp.buf.hover, opts)
```

---

## Emacs (5 minutes)

### 1. Install lsp-mode

```elisp
M-x package-install RET lsp-mode
```

Or add to `~/.emacs.d/init.el`:
```elisp
(use-package lsp-mode
  :ensure t)
```

### 2. Configure Emacs

Add to `~/.emacs.d/init.el`:

```elisp
(use-package lsp-mode
  :hook (python-mode . lsp-deferred)
  :commands lsp
  :config
  (lsp-register-client
   (make-lsp-client
    :new-connection (lsp-stdio-connection
                     '("python" "-m" "HoloLoom.lsp.server"))
    :major-modes '(python-mode)
    :server-id 'hololoom-lsp))
  (setq lsp-ui-sideline-enable t
        lsp-ui-peek-enable t))
```

### 3. Verify Connection

1. Restart Emacs (or run `M-x eval-buffer`)
2. Open a Python file (should auto-enable lsp)
3. Look for "LSP" in the mode line
4. Press `M-x lsp-ui-doc-show` to see hover info

### ✓ Complete!

**Keybindings**:
| Action | Shortcut |
|--------|----------|
| Completion | M-x completion-at-point or C-M-i |
| Hover | M-x lsp-ui-doc-show |
| Go to Definition | M-x lsp-find-definition or M-. |
| Find References | M-x lsp-find-references |
| Code Actions | M-x lsp-execute-code-action |

**Shortcut Setup** (optional, add to init.el):
```elisp
(global-set-key (kbd "C-c d") 'lsp-find-definition)
(global-set-key (kbd "C-c h") 'lsp-ui-doc-show)
(global-set-key (kbd "C-c r") 'lsp-find-references)
```

---

## Vim (7 minutes)

Vim has multiple LSP options. Here are two:

### Option A: Using vim-lsp (Simpler)

1. Install [vim-lsp](https://github.com/prabirshrestha/vim-lsp):

```vim
Plug 'prabirshrestha/vim-lsp'
Plug 'prabirshrestha/asyncomplete.vim'
Plug 'prabirshrestha/asyncomplete-lsp.vim'
```

2. Configure `~/.vimrc`:

```vim
" Register HoloLoom LSP
if executable('python')
    au User lsp_setup call lsp#register_server({
        \ 'name': 'hololoom',
        \ 'cmd': {server_info -> ['python', '-m', 'HoloLoom.lsp.server']},
        \ 'allowlist': ['python'],
        \ })
endif

" Keybindings
nmap <leader>d <plug>(lsp-definition)
nmap <leader>h <plug>(lsp-hover)
nmap <leader>r <plug>(lsp-references)
```

3. Run `:PlugInstall`

### Option B: Using coc.nvim (More Features)

1. Install [coc.nvim](https://github.com/neoclide/coc.nvim)
2. Create `~/.vim/coc-settings.json`:

```json
{
  "languageserver": {
    "hololoom": {
      "command": "python",
      "args": ["-m", "HoloLoom.lsp.server"],
      "filetypes": ["python"],
      "rootPatterns": [".git", "setup.py"]
    }
  }
}
```

3. Restart Vim

### ✓ Complete!

**Keybindings** (add to `~/.vimrc`):
```vim
" vim-lsp
nmap <leader>d <plug>(lsp-definition)
nmap <leader>h <plug>(lsp-hover)
nmap <leader>r <plug>(lsp-references)

" coc.nvim
nmap <leader>d <Plug>(coc-definition)
nmap <leader>h :call CocAction('doHover')<CR>
nmap <leader>r <Plug>(coc-references)
```

---

## Sublime Text (5 minutes)

### 1. Install LSP Package

1. Open Sublime Text
2. Install Package Control if not already installed:
   - Ctrl+Shift+P → "Install Package Control"

3. Install LSP package:
   - Ctrl+Shift+P → "Package Control: Install Package"
   - Search for "LSP"
   - Install by **sublimelsp**

### 2. Configure Sublime

1. Ctrl+Shift+P → "LSP: Enable Language Server"
2. Or manually in `~/.config/sublime-text/Packages/User/LSP.json`:

```json
{
  "clients": {
    "hololoom": {
      "enabled": true,
      "command": ["python", "-m", "HoloLoom.lsp.server"],
      "languages": [
        {
          "language": "python",
          "scopes": ["source.python"]
        }
      ]
    }
  }
}
```

### 3. Verify Connection

1. Open a Python file
2. Look at bottom-right for "HoloLoom" indicator
3. Right-click to see LSP commands
4. Try "Goto Definition"

### ✓ Complete!

**Commands**:
| Action | Command (Ctrl+Shift+P) |
|--------|------------------------|
| Completion | (automatic on `.` or partial word) |
| Hover | LSP: Hover |
| Definition | LSP: Goto Definition |
| References | LSP: Goto References |

---

## Quick Reference: Commands

### All Editors

**Common Actions**:
| Action | Description |
|--------|-------------|
| **Completion** | Show code suggestions (auto-trigger or Ctrl+Space) |
| **Hover** | Show symbol information and docs |
| **Go to Definition** | Jump to symbol definition |
| **Find References** | Find all uses of symbol |
| **Rename** | Rename symbol everywhere |
| **Format** | Auto-format code |

### VS Code
```
Ctrl+Space       → Completion
Hover            → Hover Info
F12              → Go to Definition
Ctrl+Shift+H     → Find References
F2               → Rename
Shift+Alt+F      → Format
```

### Neovim
```
gd               → Go to Definition
K                → Hover Info
gr               → Find References
<space>ca        → Code Actions (custom keybinding)
```

### Emacs
```
M-x completion-at-point
                 → Completion
M-x lsp-ui-doc-show
                 → Hover Info
M-x lsp-find-definition
                 → Go to Definition
M-x lsp-find-references
                 → Find References
M-x lsp-format-buffer
                 → Format
```

### Vim
```
<leader>d        → Go to Definition
<leader>h        → Hover Info
<leader>r        → Find References
```

### Sublime Text
```
Ctrl+Shift+P → LSP: Goto Definition
Ctrl+Shift+P → LSP: Hover
Ctrl+Shift+P → LSP: Goto References
```

---

## Troubleshooting Flowchart

### Problem: Server won't start

```
Is Python 3.9+ installed?
├─ NO  → Update Python (python.org)
└─ YES ↓

Does "pygls" module exist?
├─ NO  → pip install pygls
└─ YES ↓

Can you run: python -m HoloLoom.lsp.server?
├─ NO  → Check PYTHONPATH=. and pwd
└─ YES ✓ Server is working!
```

### Problem: Editor won't connect

```
Is server running?
├─ NO  → Start the server (python -m HoloLoom.lsp.server)
└─ YES ↓

Is editor configuration correct?
├─ NO  → Check settings (see editor guide)
└─ YES ↓

Check editor logs:
├─ VS Code → View → Output → Select "LSP Client"
├─ Neovim  → :LspInfo
├─ Emacs   → *lsp-log* buffer
└─ Verify server path in logs
```

### Problem: No completions appearing

```
Is server actually connected?
├─ NO  → Fix connection first (see above)
└─ YES ↓

Try manual trigger:
├─ VS Code → Ctrl+Space
├─ Neovim  → Ctrl+X Ctrl+O
├─ Emacs   → M-x completion-at-point
└─ Works?
    ├─ YES ✓ Completion is working!
    └─ NO  ↓

Check server logs:
PYTHONPATH=. python -m HoloLoom.lsp.server --log-level DEBUG
└─ Look for error messages in output
```

### Problem: Slow responses

```
Check system resources:
├─ High CPU?  → Server is overloaded
├─ High RAM?  → Close unused files
└─ Disk busy? → Wait for background tasks

Check network:
├─ Latency? → Check "ping localhost"
└─ Loss?    → Check connection stability

Check logs:
PYTHONPATH=. python -m HoloLoom.lsp.server --log-level DEBUG
└─ Look for "Latency:" messages
```

### Problem: "ModuleNotFoundError: No module named 'pygls'"

```bash
# Install pygls
pip install pygls

# Verify
python -c "import pygls; print(pygls.__version__)"
```

### Problem: "PYTHONPATH not set"

```bash
# Solution 1: Explicit PYTHONPATH
PYTHONPATH=/path/to/hololoom python -m HoloLoom.lsp.server

# Solution 2: Install as package
pip install -e /path/to/hololoom
python -m HoloLoom.lsp.server  # Now works without PYTHONPATH
```

---

## Quick Diagnostics

### Test if server works (without editor)

```bash
# Start server on TCP
PYTHONPATH=. python -m HoloLoom.lsp.server --port 8080 &
SERVER_PID=$!

# In another terminal, test connection
python -c "
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect(('127.0.0.1', 8080))
    print('✓ Server is accepting connections')
except:
    print('✗ Server not listening')
finally:
    sock.close()
"

# Kill server
kill $SERVER_PID
```

### Check all dependencies

```bash
# Check Python version
python --version  # Should be 3.9+

# Check required packages
python -c "
import pygls
import asyncio
print('✓ All dependencies installed')
"

# Check HoloLoom installation
python -c "
from HoloLoom.lsp.server import server
print('✓ HoloLoom LSP server is installed')
"
```

---

## Getting Help

### Documentation

- **Server README**: `HoloLoom/lsp/README.md`
- **Full Specification**: `LSP_PROTOCOL_SPEC.md`
- **API Audit**: `LSP_API_AUDIT.md`
- **Master Summary**: `PHASE_4_LSP_SERVER_SUMMARY.md`
- **Architecture**: `LSP_ARCHITECTURE.md`

### Debugging Tips

1. **Enable debug logging**:
   ```bash
   PYTHONPATH=. python -m HoloLoom.lsp.server --log-level DEBUG 2>&1 | tee lsp.log
   ```

2. **Check editor logs**:
   - **VS Code**: View → Output → Select "LSP Client"
   - **Neovim**: `:messages` or `:LspInfo`
   - **Emacs**: `M-x switch-to-buffer *lsp-log*`

3. **Test specific features**:
   - Try completion first (easiest)
   - Then try hover (medium)
   - Then go to definition (hardest)

4. **Check processes**:
   ```bash
   ps aux | grep "HoloLoom.lsp"
   ps aux | grep python
   ```

---

## Next Steps

Once your editor is set up:

1. **Try the features**:
   - Open a Python file
   - Type a variable name and press Ctrl+Space
   - Hover over a function to see its docs
   - Click go to definition to navigate

2. **Learn more**:
   - Read `LSP_PROTOCOL_SPEC.md` for what's available
   - Check `LSP_ARCHITECTURE.md` for how it works

3. **Customize**:
   - Add your own keybindings
   - Adjust hover timeout settings
   - Configure which languages to use

4. **Report issues**:
   - File bugs on GitHub with:
     - Editor version
     - Python version
     - Error logs (with --log-level DEBUG)

---

## Success Checklist

- [ ] Python 3.9+ installed
- [ ] HoloLoom cloned and installed
- [ ] Editor configured with LSP server
- [ ] Server starts without errors
- [ ] Editor shows "LSP connected" or similar
- [ ] Completion works (Ctrl+Space)
- [ ] Hover shows symbol info
- [ ] Go to definition navigates to symbol

**If all boxes checked**: You're ready to use HoloLoom LSP! 🎉

---

**Last Updated**: November 2025
**Status**: Current for Phase 4.0
**Questions?** See full documentation in LSP_PROTOCOL_SPEC.md or PHASE_4_LSP_SERVER_SUMMARY.md
