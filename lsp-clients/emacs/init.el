;;; init.el --- Example Emacs configuration integrating HoloLoom LSP

;; This file shows how to integrate HoloLoom LSP client into your Emacs configuration.
;; Choose the installation method that best fits your setup.

;;; Installation Methods

;; ============================================================================
;; METHOD 1: Using use-package (Recommended for most users)
;; ============================================================================

;; Add this to your init.el:

(use-package lsp-mode
  :ensure t
  :commands (lsp lsp-deferred)
  :init
  (setq lsp-keymap-prefix "C-c l")
  :config
  (lsp-enable-which-key-integration t))

(use-package lsp-ui
  :ensure t
  :commands lsp-ui-mode
  :config
  (setq lsp-ui-doc-position 'top
        lsp-ui-doc-alignment 'window
        lsp-ui-peek-enable t
        lsp-ui-sideline-enable t))

(use-package company
  :ensure t
  :commands company-mode
  :config
  (setq company-idle-delay 0.3
        company-minimum-prefix-length 1))

;; Load HoloLoom LSP client
(use-package hololoom
  :load-path "path/to/lsp-clients/emacs"  ; Adjust path to your setup
  :after lsp-mode
  :config
  ;; Enable HoloLoom LSP for supported modes
  (add-hook 'python-mode-hook #'lsp-deferred)
  (add-hook 'typescript-mode-hook #'lsp-deferred)
  (add-hook 'javascript-mode-hook #'lsp-deferred))

;; ============================================================================
;; METHOD 2: Using straight.el (For reproducible configurations)
;; ============================================================================

;; Add this to your init.el if using straight.el:

;; (straight-use-package 'lsp-mode)
;; (straight-use-package 'lsp-ui)
;; (straight-use-package 'company)

;; ;; Load HoloLoom from git repository
;; (use-package hololoom
;;   :straight (hololoom :local-repo "path/to/lsp-clients/emacs"
;;                        :type built-in)
;;   :after lsp-mode)

;; ============================================================================
;; METHOD 3: Manual configuration (For minimal setups)
;; ============================================================================

;; Add this to your init.el for manual loading:

;; (require 'lsp-mode)
;; (load-file "path/to/lsp-clients/emacs/hololoom.el")

;; ;; Enable LSP for Python
;; (add-hook 'python-mode-hook #'lsp-deferred)

;; ============================================================================
;; COMPLETE CONFIGURATION EXAMPLE
;; ============================================================================

;; Below is a complete, working example configuration that you can copy
;; and customize for your setup:

;; --------- LSP Mode Configuration ---------

(use-package lsp-mode
  :ensure t
  :hook
  ((python-mode . lsp-deferred)
   (typescript-mode . lsp-deferred)
   (javascript-mode . lsp-deferred))
  :init
  (setq lsp-keymap-prefix "C-c l"
        lsp-completion-provider :company
        lsp-enable-file-watchers t
        lsp-enable-text-document-color nil)
  :config
  (lsp-enable-which-key-integration t))

;; --------- LSP UI Enhancement ---------

(use-package lsp-ui
  :ensure t
  :after lsp-mode
  :commands lsp-ui-mode
  :config
  (setq lsp-ui-doc-enable t
        lsp-ui-doc-position 'top
        lsp-ui-doc-alignment 'window
        lsp-ui-doc-use-childframe t
        lsp-ui-peek-enable t
        lsp-ui-sideline-enable t
        lsp-ui-sideline-show-code-actions t
        lsp-ui-sideline-delay 0.5))

;; --------- Company Mode for Completion ---------

(use-package company
  :ensure t
  :after lsp-mode
  :config
  (setq company-idle-delay 0.3
        company-minimum-prefix-length 1
        company-selection-wrap-around t
        company-tooltip-limit 10)
  :bind
  (:map company-active-map
        ("C-n" . company-select-next)
        ("C-p" . company-select-previous)
        ("<tab>" . company-complete-selection)))

;; --------- Flycheck for Diagnostics ---------

(use-package flycheck
  :ensure t
  :config
  (setq flycheck-indication-mode 'left-fringe))

;; --------- HoloLoom LSP Client ---------

(use-package hololoom
  :load-path "path/to/lsp-clients/emacs"
  :after lsp-mode
  :custom
  (hololoom-auto-enable t)
  (hololoom-use-flycheck t)
  (hololoom-completion-snippet t)
  (hololoom-hover-delay 0.5)
  :config
  (hololoom-setup-hooks))

;; ============================================================================
;; OPTIONAL: Additional LSP Extensions
;; ============================================================================

;; LSP Ivy integration (for symbol search)
;; (use-package lsp-ivy
;;   :ensure t
;;   :commands lsp-ivy-workspace-symbol)

;; Which Key integration for key hints
(use-package which-key
  :ensure t
  :config
  (which-key-mode))

;; ============================================================================
;; CUSTOM KEYBINDINGS (Optional)
;; ============================================================================

;; Add custom keybindings for frequently used commands:

;; (with-eval-after-load 'lsp-mode
;;   ;; Quick symbol search
;;   (define-key lsp-mode-map (kbd "C-c l ?") 'lsp-workspace-symbol)
;;
;;   ;; Quick code actions
;;   (define-key lsp-mode-map (kbd "C-c l .") 'lsp-execute-code-action)
;;
;;   ;; Format buffer
;;   (define-key lsp-mode-map (kbd "C-c l =") 'lsp-format-buffer)
;;
;;   ;; Rename symbol
;;   (define-key lsp-mode-map (kbd "C-c l n") 'lsp-rename))

;; ============================================================================
;; PERFORMANCE TUNING (Optional)
;; ============================================================================

;; For large projects, you may want to adjust these settings:

;; (setq lsp-file-watch-threshold 5000  ; Increase if watching too many files
;;       lsp-idle-delay 0.500            ; Increase if CPU usage is high
;;       lsp-log-io t)                   ; Set to nil to reduce overhead

;; ============================================================================
;; TROUBLESHOOTING
;; ============================================================================

;; If HoloLoom LSP is not working:
;;
;; 1. Check that the LSP server is installed:
;;    $ pip install HoloLoom pygls
;;
;; 2. Verify the server can start:
;;    $ python -m HoloLoom.lsp.server --log-level DEBUG
;;
;; 3. Check Emacs messages buffer:
;;    M-x lsp-switch-to-logs-buffer
;;
;; 4. Enable LSP logging in your config:
;;    (setq lsp-log-io t)
;;
;; 5. Check that the correct major mode is active:
;;    M-x describe-major-mode
;;
;; See README.md for detailed troubleshooting steps.

;;; init.el ends here
