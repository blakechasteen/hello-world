# HoloLoom Emacs LSP Client - Quick Start

## Fastest Setup (2 minutes)

### Step 1: Install HoloLoom and dependencies
```bash
pip install HoloLoom pygls
```

### Step 2: Add to your `init.el` (use-package method)
```elisp
(use-package lsp-mode
  :ensure t
  :init (setq lsp-keymap-prefix "C-c l"))

(use-package hololoom
  :load-path "/path/to/hololoom/lsp-clients/emacs"
  :after lsp-mode
  :hook (python-mode . lsp-deferred))
```

### Step 3: Reload Emacs
```
C-c C-x  (or restart Emacs)
```

### Step 4: Open a Python file and test
```
1. Open any .py file
2. Type "from HoloLoom." 
3. Press C-M-i (or M-x company-complete)
4. See completion suggestions!
```

## Key Keybindings

| Action | Keybinding |
|--------|-----------|
| Jump to definition | `C-c l d` |
| Find references | `C-c l r` |
| Show hover docs | `C-c l h` |
| Search symbols | `C-c l s` |
| Code actions | `C-c l a` |
| Format buffer | `C-c l g` |
| Restart server | `C-c l x` |

## Verify Installation

Open Emacs and run:
```
M-x lsp-describe-session
```

Should show:
- Server: `hololoom-lsp`
- Status: `Running`
- Language: `python` (or your mode)

## Troubleshooting

### LSP not starting?
1. Check server installation: `python -m HoloLoom.lsp.server --help`
2. Check major mode: `M-x describe-major-mode`
3. Enable debug: `(setq lsp-log-io t)` then check logs in `*lsp-log*`

### Completions not working?
1. Enable company-mode: `M-x company-mode`
2. Try manual complete: `C-M-i`
3. Check position in code (after `.` character usually)

### Server crashes?
```
M-x lsp-workspace-restart
```

## Full Documentation

See **README.md** for:
- Detailed installation methods
- Advanced configuration
- All features and examples
- Complete troubleshooting guide
- Performance tuning

## File Structure

```
lsp-clients/emacs/
├── hololoom.el              # LSP client configuration (285 lines)
├── init.el                  # Integration examples (218 lines)
├── README.md                # Full documentation (680 lines)
└── INSTALLATION_QUICK_START.md  # This file
```

## Support

- **Repository**: https://github.com/hololoom/hololoom
- **Issues**: Report on GitHub
- **Docs**: See README.md for comprehensive guide

---

**You're ready to code with HoloLoom in Emacs!**
