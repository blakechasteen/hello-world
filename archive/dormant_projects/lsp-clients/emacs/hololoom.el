;;; hololoom.el --- LSP client configuration for HoloLoom neural memory system

;; Copyright (C) 2025 HoloLoom Contributors
;; Author: HoloLoom Team
;; Version: 0.1.0
;; Package-Requires: ((emacs "27.1") (lsp-mode "9.0") (company "0.9.13") (flycheck "32"))
;; Keywords: lsp, completion, languages
;; URL: https://github.com/hololoom/hololoom

;;; Commentary:

;; This package provides LSP client integration for HoloLoom, the neural
;; decision-making system that combines multi-scale embeddings, knowledge
;; graph memory, and adaptive policy learning.
;;
;; HoloLoom LSP provides:
;;   - Semantic code completion from HoloLoom memories
;;   - Rich hover information with entity context
;;   - Go-to-definition via knowledge graph relationships
;;   - Workspace-wide symbol search
;;   - Diagnostic reports from alignment framework
;;
;; Installation:
;;   With use-package:
;;     (use-package hololoom
;;       :after lsp-mode
;;       :load-path "path/to/lsp-clients/emacs")
;;
;;   Or manually:
;;     (load-file "path/to/lsp-clients/emacs/hololoom.el")
;;     (add-hook 'python-mode-hook #'lsp-deferred)
;;
;; Configuration:
;;   ;; Auto-enable LSP for multiple languages
;;   (add-hook 'python-mode-hook #'lsp-deferred)
;;   (add-hook 'typescript-mode-hook #'lsp-deferred)
;;   (add-hook 'javascript-mode-hook #'lsp-deferred)
;;
;; Keybindings (C-c l prefix):
;;   C-c l d   lsp-find-definition
;;   C-c l r   lsp-find-references
;;   C-c l h   lsp-describe-thing-at-point
;;   C-c l s   lsp-workspace-symbol
;;   C-c l a   lsp-execute-code-action
;;   C-c l c   lsp-ui-doc-focus-frame
;;
;; Usage:
;;   ;; Basic completion (bound to company-mode)
;;   M-x company-complete or C-M-i
;;
;;   ;; Go to definition
;;   M-x lsp-find-definition or C-c l d
;;
;;   ;; Find all references
;;   M-x lsp-find-references or C-c l r
;;
;;   ;; Hover documentation
;;   M-x lsp-describe-thing-at-point or C-c l h
;;
;;   ;; Search workspace symbols
;;   M-x lsp-workspace-symbol or C-c l s

;;; Code:

(require 'lsp-mode)

;; ============================================================================
;; LSP SERVER REGISTRATION
;; ============================================================================

;;;###autoload
(lsp-register-client
 (make-lsp-client
  :new-connection (lsp-stdio-connection
                   (list "python" "-m" "HoloLoom.lsp.server")
                   #'ignore)
  :activation-fn (lsp-activate-on
                  "python"           ; Python files
                  "typescript"       ; TypeScript
                  "javascript"       ; JavaScript
                  "typescriptreact"  ; TSX/JSX
                  "javascriptreact"
                  "json")            ; Configuration files
  :server-id 'hololoom-lsp
  :priority 1
  :notification-handlers
  (make-hash-table :test 'equal)
  :action-handlers
  (make-hash-table :test 'equal)
  :completion-in-comments? nil
  :initialization-options '((
                             ("logLevel" . "info")
                             ("trace" . "off")))))

;; ============================================================================
;; FEATURE CONFIGURATION
;; ============================================================================

;; Configure LSP completion provider
(with-eval-after-load 'lsp-completion
  (setf (alist-get 'hololoom-lsp lsp-completion-provider-resolve-functions)
        '(lsp-completion-resolve))
  (setf (alist-get 'hololoom-lsp lsp-completion-provider-trigger-characters)
        '(".")))

;; Configure LSP UI features
(with-eval-after-load 'lsp-ui
  (setq lsp-ui-doc-enable t
        lsp-ui-doc-position 'top
        lsp-ui-doc-alignment 'window
        lsp-ui-doc-use-childframe t
        lsp-ui-peek-enable t
        lsp-ui-sideline-enable t
        lsp-ui-sideline-show-code-actions t))

;; Configure code action handling
(with-eval-after-load 'lsp-mode
  (setq lsp-enable-file-watchers t
        lsp-file-watch-threshold 2000
        lsp-enable-text-document-color nil))

;; ============================================================================
;; KEYBINDINGS
;; ============================================================================

(with-eval-after-load 'lsp-mode
  (define-key lsp-mode-map (kbd "C-c l d") 'lsp-find-definition)
  (define-key lsp-mode-map (kbd "C-c l r") 'lsp-find-references)
  (define-key lsp-mode-map (kbd "C-c l h") 'lsp-describe-thing-at-point)
  (define-key lsp-mode-map (kbd "C-c l s") 'lsp-workspace-symbol)
  (define-key lsp-mode-map (kbd "C-c l a") 'lsp-execute-code-action)
  (define-key lsp-mode-map (kbd "C-c l c") 'lsp-ui-doc-focus-frame)
  (define-key lsp-mode-map (kbd "C-c l g") 'lsp-format-buffer)
  (define-key lsp-mode-map (kbd "C-c l x") 'lsp-workspace-restart))

;; Company mode integration for completion
(with-eval-after-load 'company
  (define-key company-active-map (kbd "C-n") 'company-select-next)
  (define-key company-active-map (kbd "C-p") 'company-select-previous)
  (define-key company-active-map (kbd "<tab>") 'company-complete-selection))

;; ============================================================================
;; AUTO-ENABLE HOOKS
;; ============================================================================

;;;###autoload
(defun hololoom-enable ()
  "Enable HoloLoom LSP for the current buffer."
  (interactive)
  (lsp-deferred))

;;;###autoload
(defun hololoom-enable-for-mode (mode-hook)
  "Enable HoloLoom LSP for a specific major mode.

Args:
  MODE-HOOK: Major mode hook symbol (e.g., 'python-mode-hook')"
  (add-hook mode-hook #'lsp-deferred))

;; Auto-enable for common languages
(add-hook 'python-mode-hook #'lsp-deferred)
(add-hook 'typescript-mode-hook #'lsp-deferred)
(add-hook 'javascript-mode-hook #'lsp-deferred)
(add-hook 'json-mode-hook #'lsp-deferred)

;; ============================================================================
;; DIAGNOSTIC CONFIGURATION
;; ============================================================================

;; Configure flycheck integration for diagnostics
(with-eval-after-load 'lsp-diagnostics
  (setq lsp-diagnostics-provider :flycheck
        lsp-diagnostics-attributes '((unnecessary :foreground "#c19a6b")
                                     (deprecated :strike-through t)
                                     (deprecated :foreground "#cccccc"))))

;; ============================================================================
;; COMPANY MODE INTEGRATION
;; ============================================================================

(with-eval-after-load 'company
  (setq company-minimum-prefix-length 1
        company-selection-wrap-around t
        company-idle-delay 0.3
        company-tooltip-limit 10))

(with-eval-after-load 'lsp-completion
  (setq lsp-completion-show-kind t))

;; ============================================================================
;; HELPER FUNCTIONS
;; ============================================================================

(defun hololoom-search-symbol (query)
  "Search for a symbol matching QUERY in the workspace.

Args:
  QUERY: String to search for"
  (interactive "sSearch for symbol: ")
  (lsp-workspace-symbol query))

(defun hololoom-show-hover-info ()
  "Show hover information for the symbol at point."
  (interactive)
  (lsp-describe-thing-at-point))

(defun hololoom-go-to-definition ()
  "Go to the definition of the symbol at point."
  (interactive)
  (lsp-find-definition))

(defun hololoom-find-references ()
  "Find all references to the symbol at point."
  (interactive)
  (lsp-find-references))

(defun hololoom-execute-action ()
  "Execute a code action at point."
  (interactive)
  (lsp-execute-code-action))

;; ============================================================================
;; CUSTOM FACES
;; ============================================================================

(defgroup hololoom nil
  "HoloLoom LSP client configuration."
  :group 'lsp-mode
  :prefix "hololoom-"
  :link '(url-link "https://github.com/hololoom/hololoom"))

(defface hololoom-entity
  '((t (:inherit font-lock-function-name-face)))
  "Face for HoloLoom entities in completion candidates."
  :group 'hololoom)

(defface hololoom-memory
  '((t (:inherit font-lock-constant-face)))
  "Face for HoloLoom memory references."
  :group 'hololoom)

;; ============================================================================
;; CUSTOM VARIABLES
;; ============================================================================

(defcustom hololoom-auto-enable t
  "Whether to automatically enable LSP for supported modes."
  :type 'boolean
  :group 'hololoom)

(defcustom hololoom-use-flycheck t
  "Whether to use Flycheck for diagnostics."
  :type 'boolean
  :group 'hololoom)

(defcustom hololoom-completion-snippet t
  "Whether to enable snippet expansion in completions."
  :type 'boolean
  :group 'hololoom)

(defcustom hololoom-hover-delay 0.5
  "Delay in seconds before showing hover information."
  :type 'float
  :group 'hololoom)

;; ============================================================================
;; INITIALIZATION
;; ============================================================================

(defun hololoom-setup-hooks ()
  "Set up HoloLoom LSP hooks based on custom configuration."
  (when hololoom-auto-enable
    (add-hook 'python-mode-hook #'lsp-deferred)
    (add-hook 'typescript-mode-hook #'lsp-deferred)
    (add-hook 'javascript-mode-hook #'lsp-deferred)))

(hololoom-setup-hooks)

;; ============================================================================
;; PROVIDE PACKAGE
;; ============================================================================

(provide 'hololoom)

;;; hololoom.el ends here
