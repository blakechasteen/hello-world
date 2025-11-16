# HoloLoom LSP Client for Emacs

**Complete Language Server Protocol integration for HoloLoom's neural memory system in GNU Emacs**

Transform your editor into an AI-augmented coding environment with semantic completion, intelligent navigation, and knowledge graph-powered suggestions.

## Overview

HoloLoom LSP brings the power of a neural decision-making system to Emacs, providing:

- **Semantic Code Completion** - Suggestions from HoloLoom's multi-scale embeddings and knowledge graph
- **Rich Hover Information** - Entity context, relationships, and confidence scores
- **Graph-Powered Navigation** - Go-to-definition through knowledge graph relationships
- **Workspace Symbol Search** - Semantic entity search across your codebase
- **Smart Diagnostics** - Alignment framework safety reports integrated into Flycheck
- **Multi-Language Support** - Python, TypeScript, JavaScript, JSON, and more

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Methods](#installation-methods)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Keybindings](#keybindings)
6. [Usage Examples](#usage-examples)
7. [Features](#features)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tips](#performance-tips)
10. [Contributing](#contributing)

## Prerequisites

### Emacs Version
- **Minimum**: Emacs 27.1
- **Recommended**: Emacs 28.0 or later (better async support)

### Required Packages
- `lsp-mode` (≥ 9.0) - Language Server Protocol client
- `company` (≥ 0.9.13) - Completion framework
- `flycheck` (≥ 32) - Diagnostic framework
- `emacs-lsp-booster` (optional) - Speed up LSP responses by 2-4×

### Python Setup
1. **HoloLoom installed**:
   ```bash
   pip install HoloLoom
   ```

2. **LSP server dependencies**:
   ```bash
   pip install pygls  # Language Server Protocol implementation
   ```

3. **Verify installation**:
   ```bash
   python -m HoloLoom.lsp.server --help
   ```

## Installation Methods

### Method 1: use-package (Recommended)

If you use `use-package` (most common), add to your `init.el`:

```elisp
;; 1. Install and configure base LSP mode
(use-package lsp-mode
  :ensure t
  :init (setq lsp-keymap-prefix "C-c l"))

;; 2. Install LSP UI enhancements
(use-package lsp-ui
  :ensure t
  :commands lsp-ui-mode)

;; 3. Install company for completions
(use-package company
  :ensure t
  :commands company-mode)

;; 4. Load HoloLoom LSP client
(use-package hololoom
  :load-path "path/to/lsp-clients/emacs"
  :after lsp-mode
  :hook
  ((python-mode . lsp-deferred)
   (typescript-mode . lsp-deferred)
   (javascript-mode . lsp-deferred)))
```

**Where is `path/to/lsp-clients/emacs`?**
- Clone HoloLoom: `git clone https://github.com/hololoom/hololoom`
- Use: `:load-path "/path/to/hololoom/lsp-clients/emacs"`

### Method 2: straight.el (For Reproducible Configs)

If you use `straight.el` with `use-package`:

```elisp
(use-package lsp-mode
  :straight t
  :init (setq lsp-keymap-prefix "C-c l"))

(use-package lsp-ui
  :straight t
  :after lsp-mode)

(use-package company
  :straight t)

(use-package hololoom
  :straight (hololoom :local-repo "/path/to/lsp-clients/emacs"
                       :type built-in)
  :after lsp-mode)
```

### Method 3: Manual Configuration

For minimal setups, add to `init.el`:

```elisp
;; Load dependencies
(require 'lsp-mode)
(require 'company)
(require 'flycheck)

;; Load HoloLoom
(load-file "/path/to/lsp-clients/emacs/hololoom.el")

;; Enable for your languages
(add-hook 'python-mode-hook #'lsp-deferred)
(add-hook 'typescript-mode-hook #'lsp-deferred)
```

### Method 4: Direct File Copy

1. Copy `hololoom.el` to your Emacs `load-path`:
   ```bash
   cp lsp-clients/emacs/hololoom.el ~/.emacs.d/lisp/
   ```

2. Add to `init.el`:
   ```elisp
   (load-file "~/.emacs.d/lisp/hololoom.el")
   ```

## Quick Start

1. **Start Emacs** (or reload config: `C-c C-x`)

2. **Open a Python file**: `C-x C-f example.py`

3. **Enable LSP** (if not already auto-enabled):
   ```
   M-x lsp-deferred
   ```

4. **Test completion**: Type and press `M-x company-complete` (or `C-M-i`)

5. **Check LSP status**:
   ```
   M-x lsp-describe-session
   ```

## Configuration

### Basic Configuration

Minimal working setup in `init.el`:

```elisp
(use-package lsp-mode
  :ensure t
  :init (setq lsp-keymap-prefix "C-c l")
  :hook (python-mode . lsp-deferred))

(use-package hololoom
  :load-path "path/to/lsp-clients/emacs"
  :after lsp-mode)
```

### Advanced Configuration

Fine-tune HoloLoom LSP behavior:

```elisp
(use-package hololoom
  :load-path "path/to/lsp-clients/emacs"
  :after lsp-mode
  :custom
  ;; Auto-enable LSP for these modes
  (hololoom-auto-enable t)
  ;; Show diagnostics in Flycheck
  (hololoom-use-flycheck t)
  ;; Enable snippet expansion in completions
  (hololoom-completion-snippet t)
  ;; Delay before showing hover info (seconds)
  (hololoom-hover-delay 0.5)
  :config
  ;; Additional hook setup
  (add-hook 'json-mode-hook #'lsp-deferred))
```

### LSP Mode Configuration

Customize core LSP behavior:

```elisp
(with-eval-after-load 'lsp-mode
  (setq
   ;; Completion settings
   lsp-completion-provider :company
   lsp-completion-show-kind t

   ;; Performance settings
   lsp-enable-file-watchers t
   lsp-file-watch-threshold 2000
   lsp-idle-delay 0.500

   ;; UI settings
   lsp-enable-symbol-highlighting t
   lsp-enable-on-type-formatting nil

   ;; Logging (useful for debugging)
   lsp-log-io nil))
```

### Company Mode Configuration

Tune completion behavior:

```elisp
(use-package company
  :ensure t
  :config
  (setq
   ;; Show completions after this many characters
   company-minimum-prefix-length 1
   ;; Delay before showing completions (seconds)
   company-idle-delay 0.3
   ;; Number of completions to show
   company-tooltip-limit 10
   ;; Wrap around when at end of list
   company-selection-wrap-around t))
```

### LSP UI Configuration

Customize hover and UI features:

```elisp
(use-package lsp-ui
  :ensure t
  :config
  (setq
   ;; Hover documentation
   lsp-ui-doc-enable t
   lsp-ui-doc-position 'top
   lsp-ui-doc-alignment 'window
   lsp-ui-doc-use-childframe t

   ;; Code peek
   lsp-ui-peek-enable t

   ;; Sideline hints
   lsp-ui-sideline-enable t
   lsp-ui-sideline-show-code-actions t
   lsp-ui-sideline-delay 0.5))
```

## Keybindings

### Standard LSP Keybindings

All HoloLoom LSP keybindings use the `C-c l` prefix:

| Keybinding | Command | Description |
|-----------|---------|-------------|
| `C-c l d` | `lsp-find-definition` | Jump to definition |
| `C-c l r` | `lsp-find-references` | Find all references |
| `C-c l h` | `lsp-describe-thing-at-point` | Show hover documentation |
| `C-c l s` | `lsp-workspace-symbol` | Search symbols in workspace |
| `C-c l a` | `lsp-execute-code-action` | Execute code action at point |
| `C-c l c` | `lsp-ui-doc-focus-frame` | Focus documentation frame |
| `C-c l g` | `lsp-format-buffer` | Format entire buffer |
| `C-c l x` | `lsp-workspace-restart` | Restart LSP server |

### Completion Keybindings

Used within completion menu:

| Keybinding | Action |
|-----------|--------|
| `C-n` | Select next candidate |
| `C-p` | Select previous candidate |
| `<tab>` | Accept completion |
| `RET` | Accept completion |
| `C-g` | Cancel completion |

### Common Completion Triggers

- **In Emacs**: `M-x company-complete` or `C-M-i`
- **In Spacemacs**: `C-<space>` or just type (auto-complete enabled)
- **Trigger character**: `.` (automatically triggers)

### Custom Keybindings

Add custom keybindings to your config:

```elisp
(with-eval-after-load 'lsp-mode
  ;; Quick symbol search
  (define-key lsp-mode-map (kbd "C-c l ?") 'lsp-workspace-symbol)

  ;; Rename symbol
  (define-key lsp-mode-map (kbd "C-c l n") 'lsp-rename)

  ;; Code actions
  (define-key lsp-mode-map (kbd "C-c l .") 'lsp-execute-code-action)

  ;; Show diagnostic details
  (define-key lsp-mode-map (kbd "C-c l e") 'flycheck-list-errors))
```

## Usage Examples

### Example 1: Code Completion

1. Open a Python file containing HoloLoom code
2. Type: `from HoloLoom.`
3. Press `C-M-i` (or `M-x company-complete`)
4. Completion menu shows available modules and classes
5. Navigate with `C-n`/`C-p`, select with `<tab>`

**Result**: Semantic completions from HoloLoom module structure

### Example 2: Hover Documentation

1. Position cursor on a symbol (e.g., `WeavingOrchestrator`)
2. Press `C-c l h` (or hover mouse, if enabled)
3. Documentation frame appears showing:
   - Entity definition
   - Type information
   - Related entities
   - Usage context

**Result**: Rich hover information from knowledge graph

### Example 3: Go to Definition

1. Position cursor on `lsp-find-definition` reference
2. Press `C-c l d`
3. Jumps to definition location in source file

**Result**: Fast navigation through codebase

### Example 4: Find All References

1. Position cursor on a function/class name
2. Press `C-c l r`
3. Shows all locations where symbol is referenced
4. Use `lsp-ui-peek-jump-forward`/`lsp-ui-peek-jump-backward` to navigate

**Result**: See all usage patterns of a symbol

### Example 5: Workspace Symbol Search

1. Press `C-c l s` (or `M-x lsp-workspace-symbol`)
2. Type symbol name to search: `WeavingOrchestrator`
3. See all matching symbols in workspace
4. Select one to jump to it

**Result**: Fast symbol navigation across entire workspace

### Example 6: Code Actions

Some completions include code actions (e.g., imports, refactoring):

1. Position cursor on a symbol with available actions
2. Press `C-c l a` (or `M-x lsp-execute-code-action`)
3. Menu shows available actions
4. Select one to apply it

**Result**: Automated code fixes and transformations

## Features

### Semantic Completion

HoloLoom LSP provides context-aware completions powered by:

- **Knowledge Graph**: Entity relationships and structure
- **Multi-Scale Embeddings**: Semantic similarity at 96D, 192D, and 384D scales
- **Usage Patterns**: Learned from HoloLoom memory and feedback
- **Type Information**: Integrated with Python type hints

**Trigger**: `.` character or `C-M-i`

### Rich Hover Information

Hovering displays:

- **Entity Definition**: What the symbol is
- **Type Information**: Parameter and return types
- **Related Entities**: Connected nodes in knowledge graph
- **Confidence Score**: How confident HoloLoom is
- **Usage Examples**: Example usage patterns

**Trigger**: `C-c l h` or mouse hover

### Knowledge Graph Navigation

Go-to-definition uses Yarn Graph relationships:

- **Direct References**: Find where symbols are defined
- **Type Relationships**: Navigate through inheritance chains
- **Usage Relationships**: Find who uses this symbol
- **Semantic Relationships**: Find related concepts

**Trigger**: `C-c l d` (definition), `C-c l r` (references)

### Workspace Symbol Search

Search entire workspace for symbols:

- **Semantic Search**: Find by meaning, not just name
- **Fuzzy Matching**: Tolerates typos and partial names
- **Type Filtering**: Filter by class, function, variable, etc.
- **Ranking**: Results ranked by relevance

**Trigger**: `C-c l s`

### Diagnostic Reports

Integration with Flycheck shows:

- **Alignment Framework Alerts**: Safety and goal transparency issues
- **Type Errors**: Type checking and inference
- **Style Issues**: Code quality suggestions
- **Semantic Problems**: Logic and knowledge graph inconsistencies

**Display**: Underlines in buffer, list via `C-c l e`

## Troubleshooting

### Problem: "Language server 'hololoom-lsp' not found"

**Solution**:
1. Verify HoloLoom is installed:
   ```bash
   python -c "import HoloLoom; print(HoloLoom.__version__)"
   ```

2. Verify LSP server can start:
   ```bash
   python -m HoloLoom.lsp.server --log-level DEBUG
   ```

3. Check Python path in Emacs:
   ```elisp
   M-x eval-expression: (executable-find "python")
   ```

4. If using venv, add to Emacs config:
   ```elisp
   (setenv "PATH" "/path/to/venv/bin:..." 'replace)
   ```

### Problem: Completions not showing

**Solution**:
1. Verify company-mode is enabled:
   ```
   M-x company-mode
   ```

2. Check LSP is active:
   ```
   M-x lsp-describe-session
   ```

3. Try manual completion:
   ```
   M-x company-complete (or C-M-i)
   ```

4. Increase verbosity:
   ```elisp
   (setq company-idle-delay 0.1
         company-minimum-prefix-length 0)
   ```

### Problem: LSP server crashes or disconnects

**Solution**:
1. Check server logs:
   ```
   M-x lsp-switch-to-logs-buffer
   ```

2. Restart server:
   ```
   C-c l x (or M-x lsp-workspace-restart)
   ```

3. Check for Python errors:
   ```bash
   python -m HoloLoom.lsp.server 2>&1 | head -20
   ```

4. Enable LSP logging:
   ```elisp
   (setq lsp-log-io t)
   ```

### Problem: High CPU usage or slow response

**Solution**:
1. Reduce file watching:
   ```elisp
   (setq lsp-file-watch-threshold 5000)
   ```

2. Increase idle delay:
   ```elisp
   (setq lsp-idle-delay 1.0
         company-idle-delay 0.5)
   ```

3. Disable unnecessary features:
   ```elisp
   (setq lsp-enable-text-document-color nil
         lsp-enable-on-type-formatting nil)
   ```

4. Use emacs-lsp-booster:
   ```bash
   pip install emacs-lsp-booster
   ```

### Problem: Server doesn't detect mode

**Check**:
1. Is correct major mode active?
   ```
   M-x describe-major-mode
   ```

2. Is LSP enabled for that mode?
   ```elisp
   ;; Add to config:
   (add-hook 'your-mode-hook #'lsp-deferred)
   ```

3. Verify hook is set:
   ```
   C-h v your-mode-hook
   ```

### Problem: Emacs freezes when using LSP

**Solution**:
1. Disable LSP UI peek:
   ```elisp
   (setq lsp-ui-peek-enable nil)
   ```

2. Disable file watchers:
   ```elisp
   (setq lsp-enable-file-watchers nil)
   ```

3. Use async completion:
   ```elisp
   (setq company-async-timeout 5)
   ```

### Problem: Hover shows "Language client initialization failed"

**Solution**:
1. Check HoloLoom version:
   ```bash
   pip show HoloLoom | grep Version
   ```

2. Reinstall dependencies:
   ```bash
   pip install --upgrade HoloLoom pygls
   ```

3. Check Python:
   ```bash
   python --version  # Should be 3.8+
   ```

4. Check Emacs and LSP mode versions:
   ```
   M-x emacs-version
   M-x eval-expression: lsp-version
   ```

## Performance Tips

### Speed Up LSP Responses

1. **Use emacs-lsp-booster** (2-4× speedup):
   ```bash
   pip install emacs-lsp-booster
   ```

2. **Increase idle delay** (reduce overhead):
   ```elisp
   (setq lsp-idle-delay 1.0)
   ```

3. **Disable unused features**:
   ```elisp
   (setq lsp-enable-on-type-formatting nil
         lsp-enable-semantic-highlighting nil)
   ```

4. **Reduce file watch threshold**:
   ```elisp
   (setq lsp-file-watch-threshold 10000)
   ```

### Optimize for Large Projects

```elisp
(setq
 ;; Don't watch too many files
 lsp-file-watch-threshold 10000
 ;; Less frequent updates
 lsp-idle-delay 2.0
 ;; Simpler completion
 company-idle-delay 0.5
 company-tooltip-limit 5)
```

### Memory Usage

Monitor memory with:
```
M-x garbage-collect
M-x memory-usage
```

If high, consider:
```elisp
(setq gc-cons-threshold 100000000)  ; Increase GC threshold
```

## Contributing

Found a bug or have a suggestion? Help improve HoloLoom LSP:

1. **Report Issues**: GitHub issues with minimal reproduction
2. **Submit PRs**: Fork and submit improvements
3. **Share Configs**: Post working configurations in discussions
4. **Test Coverage**: Help test on different Emacs versions

## Resources

- **HoloLoom Repository**: https://github.com/hololoom/hololoom
- **LSP Mode Documentation**: https://emacs-lsp.github.io/lsp-mode/
- **Language Server Protocol Spec**: https://microsoft.github.io/language-server-protocol/
- **Company Mode**: https://company-mode.github.io/
- **Emacs LSP Tips**: https://www.reddit.com/r/emacs/wiki/lsp

## License

HoloLoom LSP is part of the HoloLoom project and follows the same license terms.

---

**Happy coding with HoloLoom in Emacs!**

For questions or support, refer to the [main HoloLoom documentation](https://github.com/hololoom/hololoom/wiki) or open an issue on GitHub.

Last Updated: 2025-11-16
