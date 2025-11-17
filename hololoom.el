;;; hololoom.el --- HoloLoom AI integration for Emacs  -*- lexical-binding: t; -*-

;; Copyright (C) 2025

;; Author: HoloLoom Contributors
;; Version: 0.1.0
;; Package-Requires: ((emacs "27.1"))
;; Keywords: org, ai, productivity
;; URL: https://github.com/blakechasteen/hello-world

;;; Commentary:

;; HoloLoom brings AI-powered decision making to your Emacs org-mode workflow.
;;
;; Features:
;; - Natural language queries over your org files
;; - AI suggestions for next tasks
;; - Deadline tracking and prioritization
;; - Temporal queries (when did I finish X?)
;; - Learning from your patterns
;;
;; Usage:
;;   M-x hololoom-query RET What should I work on? RET
;;   C-c h n   - Suggest next task
;;   C-c h d   - Show deadlines
;;   C-c h s   - Start background monitoring
;;
;; Setup:
;;   (require 'hololoom)
;;   (hololoom-mode 1)
;;   (global-set-key (kbd "C-c h") hololoom-command-map)

;;; Code:

(require 'org)
(require 'json)

;;; Customization

(defgroup hololoom nil
  "HoloLoom AI integration for org-mode."
  :group 'org
  :prefix "hololoom-")

(defcustom hololoom-python-command "python"
  "Python command to use for HoloLoom."
  :type 'string
  :group 'hololoom)

(defcustom hololoom-module-path "HoloLoom.query"
  "Python module path for HoloLoom queries."
  :type 'string
  :group 'hololoom)

(defcustom hololoom-org-directory org-directory
  "Directory containing org files for HoloLoom to monitor."
  :type 'directory
  :group 'hololoom)

(defcustom hololoom-knowledge-graph-path "~/hololoom-knowledge.jsonl"
  "Path to knowledge graph file."
  :type 'file
  :group 'hololoom)

(defcustom hololoom-enable-mode-line t
  "Show AI suggestions in mode line."
  :type 'boolean
  :group 'hololoom)

(defcustom hololoom-update-interval 300
  "Seconds between mode line updates."
  :type 'integer
  :group 'hololoom)

(defcustom hololoom-enable-learning t
  "Track query usage for learning."
  :type 'boolean
  :group 'hololoom)

;;; Variables

(defvar hololoom--monitor-process nil
  "Background monitoring process.")

(defvar hololoom--mode-line-suggestion ""
  "Current AI suggestion for mode line.")

(defvar hololoom--update-timer nil
  "Timer for mode line updates.")

(defvar hololoom--query-history '()
  "History of queries for learning.")

;;; Core Functions

(defun hololoom--call-python (query &optional format monitor-dir kg-path)
  "Call HoloLoom Python query interface.
QUERY is the natural language query string.
FORMAT is the output format (text, json, org).
MONITOR-DIR is the directory to monitor for org files.
KG-PATH is the path to the knowledge graph file."
  (let* ((format-arg (if format (format "--format %s" format) ""))
         (monitor-arg (if monitor-dir
                         (format "--monitor '%s'" (expand-file-name monitor-dir))
                       ""))
         (kg-arg (if kg-path
                    (format "--kg '%s'" (expand-file-name kg-path))
                  ""))
         (cmd (format "%s -m %s '%s' %s %s %s"
                     hololoom-python-command
                     hololoom-module-path
                     query
                     format-arg
                     monitor-arg
                     kg-arg)))
    (shell-command-to-string cmd)))

(defun hololoom--log-query (query result accepted)
  "Log query for learning loop.
QUERY is the query string.
RESULT is the response.
ACCEPTED is t if user accepted the suggestion."
  (when hololoom-enable-learning
    (let ((log-entry (list :query query
                          :result result
                          :accepted accepted
                          :timestamp (current-time))))
      (push log-entry hololoom--query-history)
      ;; Also write to file for persistent learning
      (hololoom--write-learning-log log-entry))))

(defun hololoom--write-learning-log (entry)
  "Write learning log ENTRY to file."
  (let ((log-file (expand-file-name "~/.hololoom-learning.jsonl")))
    (with-temp-buffer
      (insert (json-encode entry))
      (insert "\n")
      (append-to-file (point-min) (point-max) log-file))))

;;; Interactive Commands

;;;###autoload
(defun hololoom-query (query-string)
  "Query HoloLoom with natural language QUERY-STRING."
  (interactive "sHoloLoom Query: ")
  (let* ((result (hololoom--call-python query-string
                                       "text"
                                       hololoom-org-directory
                                       hololoom-knowledge-graph-path))
         (buffer (get-buffer-create "*HoloLoom Query*")))
    (with-current-buffer buffer
      (let ((inhibit-read-only t))
        (erase-buffer)
        (insert (propertize (format "Query: %s\n" query-string)
                           'face 'bold))
        (insert (propertize (make-string 80 ?─) 'face 'shadow))
        (insert "\n\n")
        (insert result)
        (insert "\n\n")
        (insert (propertize (make-string 80 ?─) 'face 'shadow))
        (insert "\n")
        (insert (propertize "Press 'q' to quit, 'r' to refresh, 'a' to accept suggestion\n"
                           'face 'shadow))
        (goto-char (point-min))
        (hololoom-results-mode)))
    (switch-to-buffer-other-window buffer)
    ;; Log query
    (hololoom--log-query query-string result nil)))

;;;###autoload
(defun hololoom-suggest-next-task ()
  "Ask HoloLoom what to work on next."
  (interactive)
  (let ((result (hololoom--call-python "What should I work on?"
                                      "text"
                                      hololoom-org-directory
                                      hololoom-knowledge-graph-path)))
    (message "HoloLoom: %s" (string-trim result))
    (hololoom--log-query "What should I work on?" result nil)))

;;;###autoload
(defun hololoom-show-deadlines (&optional timeframe)
  "Show upcoming deadlines.
TIMEFRAME can be 'today', 'this week', 'this month'."
  (interactive)
  (let* ((tf (or timeframe "this week"))
         (query (format "What's due %s?" tf)))
    (hololoom-query query)))

;;;###autoload
(defun hololoom-show-current-tasks ()
  "Show tasks currently in progress."
  (interactive)
  (hololoom-query "What am I working on?"))

;;;###autoload
(defun hololoom-search-notes (topic)
  "Search notes about TOPIC."
  (interactive "sSearch for: ")
  (hololoom-query (format "Show my notes about %s" topic)))

;;;###autoload
(defun hololoom-show-statistics ()
  "Show knowledge graph statistics."
  (interactive)
  (hololoom-query "Show me statistics"))

;;;###autoload
(defun hololoom-when-finished (task)
  "Ask when TASK was finished."
  (interactive "sTask name: ")
  (hololoom-query (format "When did I finish %s?" task)))

;;; Background Monitoring

;;;###autoload
(defun hololoom-start-monitoring ()
  "Start HoloLoom background monitoring of org files."
  (interactive)
  (if hololoom--monitor-process
      (message "HoloLoom monitoring already running")
    (condition-case err
        (progn
          (setq hololoom--monitor-process
                (start-process "hololoom-monitor"
                             "*hololoom-monitor*"
                             hololoom-python-command
                             "-c"
                             (format "import asyncio; from HoloLoom.spinningWheel.orgmode_live import OrgLiveMonitor; from HoloLoom.memory.graph import KG; kg = KG(); monitor = OrgLiveMonitor(kg, watch_dir='%s'); asyncio.run(monitor.start()); import time; time.sleep(999999)"
                                    (expand-file-name hololoom-org-directory))))
          (message "HoloLoom monitoring started for %s" hololoom-org-directory))
      (error (message "Failed to start HoloLoom monitoring: %s" err)))))

;;;###autoload
(defun hololoom-stop-monitoring ()
  "Stop HoloLoom background monitoring."
  (interactive)
  (when hololoom--monitor-process
    (delete-process hololoom--monitor-process)
    (setq hololoom--monitor-process nil)
    (message "HoloLoom monitoring stopped")))

;;;###autoload
(defun hololoom-restart-monitoring ()
  "Restart HoloLoom background monitoring."
  (interactive)
  (hololoom-stop-monitoring)
  (sit-for 1)
  (hololoom-start-monitoring))

;;; Mode Line Integration

(defun hololoom-update-mode-line ()
  "Update mode line with latest AI suggestion."
  (when hololoom-enable-mode-line
    (condition-case err
        (let ((suggestion (string-trim
                          (hololoom--call-python "What should I work on?"
                                                "text"
                                                hololoom-org-directory
                                                hololoom-knowledge-graph-path))))
          ;; Extract just the task name (first line usually)
          (when (string-match "^[^:]+: \\(.+\\)$" suggestion)
            (setq hololoom--mode-line-suggestion (match-string 1 suggestion)))
          (force-mode-line-update t))
      (error (message "HoloLoom mode line update failed: %s" err)))))

(defun hololoom--mode-line-format ()
  "Format mode line string for HoloLoom."
  (when (and hololoom-enable-mode-line
             (not (string-empty-p hololoom--mode-line-suggestion)))
    (concat " [AI: "
            (propertize hololoom--mode-line-suggestion
                       'face 'font-lock-keyword-face
                       'help-echo "HoloLoom AI suggestion - C-c h n for details")
            "]")))

;;; Results Buffer Mode

(defvar hololoom-results-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "q") 'quit-window)
    (define-key map (kbd "r") 'hololoom-refresh-results)
    (define-key map (kbd "a") 'hololoom-accept-suggestion)
    (define-key map (kbd "n") 'next-line)
    (define-key map (kbd "p") 'previous-line)
    map)
  "Keymap for HoloLoom results buffer.")

(define-derived-mode hololoom-results-mode special-mode "HoloLoom"
  "Major mode for HoloLoom query results.

\\{hololoom-results-mode-map}"
  (setq truncate-lines nil)
  (setq buffer-read-only t)
  (setq show-trailing-whitespace nil))

(defun hololoom-refresh-results ()
  "Refresh current query results."
  (interactive)
  (message "Refreshing HoloLoom results...")
  ;; TODO: Store last query and re-run it
  (message "Refresh not yet implemented"))

(defun hololoom-accept-suggestion ()
  "Accept the current AI suggestion."
  (interactive)
  (message "Suggestion accepted - learning from your choice")
  ;; Log acceptance
  (hololoom--log-query "last-query" "last-result" t)
  (quit-window))

;;; Org Integration

;;;###autoload
(defun hololoom-org-capture-with-context ()
  "Capture new task with AI context."
  (interactive)
  (let ((context (hololoom--call-python "What am I currently working on?"
                                       "text"
                                       hololoom-org-directory
                                       hololoom-knowledge-graph-path)))
    (org-capture nil "h")
    (insert (format "\n:PROPERTIES:\n:AI-CONTEXT: %s\n:END:\n"
                   (string-trim context)))))

;;; Minor Mode

(defvar hololoom-command-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "q") 'hololoom-query)
    (define-key map (kbd "n") 'hololoom-suggest-next-task)
    (define-key map (kbd "d") 'hololoom-show-deadlines)
    (define-key map (kbd "c") 'hololoom-show-current-tasks)
    (define-key map (kbd "s") 'hololoom-search-notes)
    (define-key map (kbd "t") 'hololoom-show-statistics)
    (define-key map (kbd "w") 'hololoom-when-finished)
    (define-key map (kbd "S") 'hololoom-start-monitoring)
    (define-key map (kbd "Q") 'hololoom-stop-monitoring)
    map)
  "Keymap for HoloLoom commands.")

;;;###autoload
(define-minor-mode hololoom-mode
  "Toggle HoloLoom AI integration.

When enabled, provides AI-powered assistance for org-mode:
- Query your knowledge graph with natural language
- Get AI suggestions for next tasks
- Track deadlines and priorities
- Learn from your patterns

\\{hololoom-command-map}"
  :global t
  :group 'hololoom
  :lighter " HL"
  (if hololoom-mode
      (progn
        ;; Enable mode
        (when hololoom-enable-mode-line
          ;; Add to mode line
          (add-to-list 'mode-line-misc-info '(:eval (hololoom--mode-line-format)))
          ;; Start update timer
          (setq hololoom--update-timer
                (run-with-timer 0 hololoom-update-interval
                              'hololoom-update-mode-line)))
        (message "HoloLoom mode enabled - C-c h for commands"))
    ;; Disable mode
    (when hololoom--update-timer
      (cancel-timer hololoom--update-timer)
      (setq hololoom--update-timer nil))
    (setq hololoom--mode-line-suggestion "")
    (force-mode-line-update t)
    (message "HoloLoom mode disabled")))

;;; Setup Helper

;;;###autoload
(defun hololoom-setup ()
  "Set up HoloLoom for first use.
Creates knowledge graph from org files."
  (interactive)
  (message "Setting up HoloLoom...")
  (let ((setup-script (format "
import asyncio
from HoloLoom.memory.graph import KG
from HoloLoom.spinningWheel.orgmode import OrgModeSpinner
from HoloLoom.spinningWheel.base import SpinnerConfig
from pathlib import Path

async def setup():
    kg = KG()
    spinner = OrgModeSpinner(SpinnerConfig())

    # Load all org files
    for org_file in Path('%s').rglob('*.org'):
        print(f'Loading {org_file}...')
        with open(org_file) as f:
            content = f.read()
        shards = await spinner.spin({
            'content': content,
            'source': str(org_file),
            'episode': f'org:{org_file.stem}'
        })
        print(f'  -> {len(shards)} headings')

    # Save knowledge graph
    kg.save('%s')
    print(f'Saved knowledge graph: {kg.stats()}')

asyncio.run(setup())
" hololoom-org-directory hololoom-knowledge-graph-path)))
    (with-temp-buffer
      (insert setup-script)
      (write-file "/tmp/hololoom-setup.py"))
    (shell-command (format "%s /tmp/hololoom-setup.py" hololoom-python-command))
    (message "HoloLoom setup complete! Knowledge graph saved to %s"
             hololoom-knowledge-graph-path)))

(provide 'hololoom)

;;; hololoom.el ends here
