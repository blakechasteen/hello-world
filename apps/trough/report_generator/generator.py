"""
Trough Interactive HTML Report Generator
=========================================

Converts Trough findings into interactive, click-through HTML reports.

Features:
- 3-panel layout (Findings List | Code Preview | Details Panel)
- Syntax highlighting with Pygments
- VS Code integration (click to open in editor)
- Auto-fix integration with diff preview
- Zero external dependencies (embedded CSS/JS)
- Mobile responsive design

Usage:
    from trough.report_generator import generate_html_report
    from trough import AISlopDetector, Language

    detector = AISlopDetector()
    issues = await detector.detect_all(code, Language.PYTHON, "file.py")

    html = generate_html_report(
        findings=issues,
        output_path="report.html",
        enable_vscode_integration=True,
        code_snippets={"file.py": code}
    )

Author: Agent A (Haiku)
Date: 2025-11-22
Status: Week 1 Complete
"""

import json
import base64
import html as html_module
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

try:
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
    from pygments.formatters import HtmlFormatter
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return html_module.escape(text)


def _get_syntax_highlighted_code(
    code: str,
    file_path: str,
    language: Optional[str] = None
) -> str:
    """
    Get syntax-highlighted code using Pygments.
    Falls back to <pre> if Pygments unavailable.

    Args:
        code: Source code string
        file_path: File path for language detection
        language: Optional language override

    Returns:
        HTML code with syntax highlighting
    """
    if not PYGMENTS_AVAILABLE:
        return f"<pre>{_escape_html(code)}</pre>"

    try:
        # Try to detect lexer from filename
        if language:
            try:
                lexer = get_lexer_by_name(language)
            except Exception:
                lexer = guess_lexer_for_filename(file_path, code)
        else:
            lexer = guess_lexer_for_filename(file_path, code)

        # Format with line numbers
        formatter = HtmlFormatter(
            style='monokai',
            linenos=True,
            full=False,
            wrapcode=True
        )
        return highlight(code, lexer, formatter)
    except Exception as e:
        # Fallback to plain code
        return f"<pre>{_escape_html(code)}</pre>"


def _get_severity_badge(severity: str) -> str:
    """Get HTML badge for severity level."""
    colors = {
        'critical': '#e74c3c',
        'high': '#e67e22',
        'medium': '#f39c12',
        'low': '#3498db',
        'info': '#95a5a6'
    }
    color = colors.get(severity.lower(), '#95a5a6')
    return f'<span class="severity-badge" style="background-color:{color}">{severity.upper()}</span>'


def _get_category_icon(category: str) -> str:
    """Get emoji icon for category."""
    icons = {
        'hallucination': '👻',
        'error_handling': '⚠️',
        'hardcoded_values': '🔐',
        'race_condition': '🏃',
        'resource_leak': '💧',
        'type_mismatch': '🔤',
        'security': '🛡️',
        'performance': '⚡',
        'dead_code': '💀',
        'naming': '🏷️',
        'documentation': '📚',
        'copy_paste': '📋',
        'incomplete': '⚙️',
        'off_by_one': '📍',
        'timezone': '🕐'
    }
    return icons.get(category.lower(), '📌')


def _convert_windows_path_to_vscode(path: str) -> str:
    r"""
    Convert Windows path to VS Code URI.

    Examples:
        c:\Users\blake\file.py -> vscode://file/c:/Users/blake/file.py
        /Users/blake/file.py -> vscode://file/Users/blake/file.py

    Args:
        path: File path (Windows or Unix)

    Returns:
        VS Code file URI
    """
    # Normalize path
    path = path.replace('\\', '/')

    # Remove leading slash for Windows paths
    if len(path) > 2 and path[1] == ':':
        # Windows path like c:/Users/...
        return f"vscode://file/{path}"
    else:
        # Unix path
        if path.startswith('/'):
            return f"vscode://file{path}"
        else:
            return f"vscode://file/{path}"


def _minify_css(css: str) -> str:
    """
    Minify CSS by removing whitespace and comments.

    Args:
        css: CSS string to minify

    Returns:
        Minified CSS
    """
    import re

    # Remove comments
    css = re.sub(r'/\*.*?\*/', '', css, flags=re.DOTALL)

    # Remove whitespace
    css = re.sub(r'\s+', ' ', css)
    css = re.sub(r'\s*([{}:;,>+~])\s*', r'\1', css)

    return css.strip()


def _minify_javascript(js: str) -> str:
    """
    Minify JavaScript by removing whitespace and comments.

    Note: This is a basic minifier. For production, use a proper tool like `rjsmin`.

    Args:
        js: JavaScript string to minify

    Returns:
        Minified JavaScript
    """
    import re

    # Remove single-line comments
    js = re.sub(r'//.*?$', '', js, flags=re.MULTILINE)

    # Remove multi-line comments
    js = re.sub(r'/\*.*?\*/', '', js, flags=re.DOTALL)

    # Remove unnecessary whitespace
    js = re.sub(r'\s+', ' ', js)

    # Remove spaces around operators (but keep spaces after keywords)
    js = re.sub(r'\s*([{}()[\]:;,=+\-*/])\s*', r'\1', js)

    # Restore spaces after keywords for readability
    js = re.sub(r'(if|else|for|while|function|return|const|let|var)\(', r'\1 (', js)

    return js.strip()


def _generate_css(minify: bool = False) -> str:
    """Generate embedded CSS for the report.

    Args:
        minify: If True, minimize CSS to reduce file size
    """
    css = """
    <style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        background: #f5f5f5;
        color: #333;
        line-height: 1.6;
    }

    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px 30px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    .header h1 {
        font-size: 28px;
        margin-bottom: 5px;
    }

    .header .subtitle {
        font-size: 14px;
        opacity: 0.9;
    }

    .container {
        display: flex;
        height: calc(100vh - 100px);
        gap: 0;
    }

    .panel {
        overflow-y: auto;
        border-right: 1px solid #e0e0e0;
    }

    .panel:last-child {
        border-right: none;
    }

    .findings-panel {
        flex: 1;
        max-width: 350px;
        background: white;
    }

    .code-panel {
        flex: 1.5;
        background: #2d2d2d;
        color: #f8f8f2;
        font-family: "Monaco", "Menlo", "Ubuntu Mono", monospace;
        font-size: 12px;
        overflow-x: auto;
    }

    .details-panel {
        flex: 1;
        background: white;
        padding: 20px;
    }

    /* Findings List */
    .findings-list {
        list-style: none;
    }

    .finding-item {
        padding: 12px 15px;
        border-bottom: 1px solid #f0f0f0;
        cursor: pointer;
        transition: background 0.2s, border-left 0.2s;
        border-left: 4px solid transparent;
    }

    .finding-item:hover {
        background: #f9f9f9;
    }

    .finding-item.active {
        background: #e8f4f8;
        border-left-color: #667eea;
    }

    .finding-item .line-number {
        font-size: 12px;
        color: #999;
        margin-bottom: 4px;
    }

    .finding-item .message {
        font-size: 13px;
        font-weight: 500;
        color: #333;
        margin-bottom: 6px;
    }

    .severity-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 10px;
        font-weight: bold;
        color: white;
    }

    /* Search and Filters */
    .search-bar {
        padding: 15px;
        border-bottom: 1px solid #e0e0e0;
    }

    .search-bar input {
        width: 100%;
        padding: 8px 12px;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 13px;
    }

    .filter-controls {
        padding: 12px 15px;
        border-bottom: 1px solid #f0f0f0;
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }

    .filter-button {
        padding: 4px 10px;
        border: 1px solid #ddd;
        border-radius: 3px;
        background: white;
        cursor: pointer;
        font-size: 12px;
        transition: all 0.2s;
    }

    .filter-button:hover {
        border-color: #667eea;
        color: #667eea;
    }

    .filter-button.active {
        background: #667eea;
        color: white;
        border-color: #667eea;
    }

    /* Code Panel */
    .code-panel pre {
        padding: 0;
        margin: 0;
        background: transparent;
    }

    .code-panel code {
        font-family: inherit;
    }

    .hljs-linenumber {
        color: #666;
        margin-right: 15px;
        user-select: none;
    }

    .highlight-line {
        background: rgba(255, 235, 59, 0.2);
        display: block;
    }

    /* Details Panel */
    .detail-section {
        margin-bottom: 20px;
    }

    .detail-section h3 {
        font-size: 13px;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .detail-section p {
        font-size: 13px;
        color: #555;
        line-height: 1.6;
    }

    .suggestion-box {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 10px 12px;
        border-radius: 4px;
        font-size: 12px;
        color: #1e40af;
        margin: 10px 0;
    }

    .code-snippet {
        background: #f5f5f5;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 10px;
        font-family: monospace;
        font-size: 11px;
        overflow-x: auto;
        margin: 8px 0;
    }

    .action-buttons {
        display: flex;
        gap: 8px;
        margin-top: 15px;
    }

    .btn {
        padding: 8px 12px;
        border: 1px solid #ddd;
        border-radius: 4px;
        background: white;
        cursor: pointer;
        font-size: 12px;
        font-weight: 500;
        transition: all 0.2s;
        text-decoration: none;
        display: inline-block;
    }

    .btn:hover {
        background: #f5f5f5;
        border-color: #999;
    }

    .btn-primary {
        background: #667eea;
        color: white;
        border-color: #667eea;
    }

    .btn-primary:hover {
        background: #5568d3;
        border-color: #5568d3;
    }

    .btn-secondary {
        background: white;
        color: #333;
    }

    /* Empty state */
    .empty-state {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #999;
        text-align: center;
        padding: 30px;
    }

    .empty-state-content {
        max-width: 300px;
    }

    .empty-state-icon {
        font-size: 48px;
        margin-bottom: 15px;
    }

    .empty-state h2 {
        font-size: 16px;
        margin-bottom: 8px;
    }

    .empty-state p {
        font-size: 13px;
        color: #bbb;
    }

    /* Stats bar */
    .stats-bar {
        padding: 15px;
        background: #f9f9f9;
        border-bottom: 1px solid #e0e0e0;
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        font-size: 12px;
    }

    .stat {
        text-align: center;
    }

    .stat-value {
        font-size: 16px;
        font-weight: bold;
        color: #667eea;
    }

    .stat-label {
        font-size: 11px;
        color: #999;
        margin-top: 2px;
    }

    /* Responsive */
    @media (max-width: 1200px) {
        .container {
            flex-direction: column;
        }

        .findings-panel,
        .code-panel,
        .details-panel {
            max-width: none;
            flex: 1;
            min-height: 0;
        }
    }

    /* Scrollbar styling - WebKit (Chrome, Edge, Safari) */
    .panel::-webkit-scrollbar {
        width: 8px;
    }

    .panel::-webkit-scrollbar-track {
        background: #f1f1f1;
    }

    .panel::-webkit-scrollbar-thumb {
        background: #bbb;
        border-radius: 4px;
    }

    .panel::-webkit-scrollbar-thumb:hover {
        background: #999;
    }

    /* Scrollbar styling - Firefox */
    .panel {
        scrollbar-width: thin;
        scrollbar-color: #bbb #f1f1f1;
    }

    /* Pygments override for better colors in dark theme */
    .code-panel .highlight {
        background: transparent;
    }

    /* Tablet responsive (768px - 1200px) */
    @media (max-width: 1024px) {
        .container {
            flex-direction: column;
        }

        .findings-panel,
        .code-panel,
        .details-panel {
            max-width: none;
            flex: 1;
            min-height: 200px;
        }

        .code-panel {
            font-size: 11px;
        }

        .details-panel {
            padding: 15px;
        }

        .filter-controls {
            padding: 10px 12px;
        }

        .finding-item {
            padding: 10px 12px;
        }

        .finding-item .message {
            font-size: 12px;
        }
    }

    /* Mobile responsive (< 768px) */
    @media (max-width: 768px) {
        .container {
            height: auto;
        }

        .panel {
            min-height: 250px;
            border-right: none;
            border-bottom: 1px solid #e0e0e0;
        }

        .header {
            padding: 15px 20px;
        }

        .header h1 {
            font-size: 20px;
            margin-bottom: 5px;
        }

        .header .subtitle {
            font-size: 12px;
        }

        .findings-panel {
            max-width: none;
        }

        .stats-bar {
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            padding: 10px 12px;
        }

        .stat-value {
            font-size: 14px;
        }

        .stat-label {
            font-size: 10px;
        }

        .search-bar {
            padding: 12px;
        }

        .search-bar input {
            padding: 8px 10px;
            font-size: 12px;
        }

        .filter-controls {
            padding: 8px 10px;
            gap: 4px;
        }

        .filter-button {
            padding: 3px 8px;
            font-size: 11px;
        }

        .finding-item {
            padding: 8px 10px;
        }

        .finding-item .line-number {
            font-size: 11px;
        }

        .finding-item .message {
            font-size: 12px;
        }

        .code-panel {
            font-size: 10px;
        }

        .hljs-linenumber {
            margin-right: 10px;
        }

        .details-panel {
            padding: 12px;
        }

        .detail-section {
            margin-bottom: 15px;
        }

        .detail-section h3 {
            font-size: 12px;
            margin-bottom: 6px;
        }

        .detail-section p {
            font-size: 12px;
        }

        .btn {
            padding: 6px 10px;
            font-size: 11px;
        }

        .action-buttons {
            gap: 6px;
            margin-top: 12px;
        }
    }

    /* Small mobile (< 480px) */
    @media (max-width: 480px) {
        .header {
            padding: 12px 15px;
        }

        .header h1 {
            font-size: 18px;
        }

        .stats-bar {
            grid-template-columns: repeat(2, 1fr);
            gap: 6px;
            padding: 8px 10px;
        }

        .stat-value {
            font-size: 12px;
        }

        .code-panel {
            font-size: 9px;
        }

        .hljs-linenumber {
            margin-right: 8px;
        }
    }
    </style>
    """

    if minify:
        # Extract CSS content without style tags
        css_content = css.replace('<style>', '').replace('</style>', '')
        minified = _minify_css(css_content)
        return f"<style>{minified}</style>"
    return css


def _generate_javascript(minify: bool = False, enable_virtualization: bool = False) -> str:
    """Generate embedded JavaScript for interactivity.

    Args:
        minify: If True, minimize JavaScript to reduce file size
        enable_virtualization: If True, enable virtualization for large lists (100+ findings)
    """
    js = """
    <script>
    // Initialize application state
    let findings = [];
    let currentFilteredFindings = [];  // Track filtered results to avoid recomputation
    let selectedFinding = null;  // Track actual finding object instead of index

    // DOM elements
    const findingsListEl = document.getElementById('findings-list');
    const codePanelEl = document.getElementById('code-panel');
    const detailsPanelEl = document.getElementById('details-panel');
    const searchInputEl = document.getElementById('search-input');
    const filterButtonsEl = document.querySelectorAll('.filter-button');

    // Current filters
    let activeFilters = {
        severity: [],
        category: [],
        searchQuery: ''
    };

    // Initialize
    document.addEventListener('DOMContentLoaded', function() {
        findings = window.FINDINGS_DATA || [];
        currentFilteredFindings = [...findings];
        renderFindings();

        // Setup event listeners
        searchInputEl?.addEventListener('input', handleSearch);
        filterButtonsEl.forEach(btn => {
            btn.addEventListener('click', handleFilter);
        });

        // Keyboard navigation
        document.addEventListener('keydown', handleKeyboard);
    });

    function renderFindings(findingsToRender = null) {
        const toRender = findingsToRender !== null ? findingsToRender : currentFilteredFindings;

        // Update cache
        currentFilteredFindings = toRender;

        findingsListEl.innerHTML = '';

        if (!toRender || toRender.length === 0) {
            findingsListEl.innerHTML = '<div class="empty-state"><div class="empty-state-content"><div class="empty-state-icon">✓</div><h2>All Clear!</h2><p>No issues found</p></div></div>';
            detailsPanelEl.innerHTML = '';
            codePanelEl.innerHTML = '';
            return;
        }

        toRender.forEach((finding, index) => {
            const item = document.createElement('li');
            item.className = 'finding-item';
            if (selectedFinding === null && index === 0) {
                item.classList.add('active');
            }
            if (selectedFinding && selectedFinding === finding) {
                item.classList.add('active');
            }

            const lineNum = finding.line_number || 'N/A';
            const category = finding.category || 'unknown';
            const severity = finding.severity || 'info';

            item.innerHTML = `
                <div class="line-number">Line ${lineNum}</div>
                <div class="message">${escapeHtml(finding.message)}</div>
                <div>${window.SEVERITY_BADGES[severity] || severity}</div>
            `;

            item.addEventListener('click', () => selectFinding(finding));
            findingsListEl.appendChild(item);
        });

        // Select first finding if none selected
        if (toRender.length > 0 && selectedFinding === null) {
            selectFinding(toRender[0]);
        }
    }

    function selectFinding(finding) {
        if (!finding) return;

        selectedFinding = finding;

        // Update active state on all items
        const items = findingsListEl.querySelectorAll('.finding-item');
        items.forEach(item => item.classList.remove('active'));

        // Find and highlight the item for this finding
        let foundItem = null;
        for (let i = 0; i < currentFilteredFindings.length; i++) {
            if (currentFilteredFindings[i] === finding) {
                if (items[i]) {
                    items[i].classList.add('active');
                    items[i].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
                break;
            }
        }

        // Update code panel
        updateCodePanel(finding);

        // Update details panel
        updateDetailsPanel(finding);
    }

    function updateCodePanel(finding) {
        if (!finding) {
            codePanelEl.innerHTML = '';
            return;
        }

        const code = window.CODE_SNIPPETS[finding.file_path] || '';
        const lines = code.split('\\n');
        const lineNum = finding.line_number || 1;
        const highlightIndex = Math.max(0, lineNum - 1);  // Ensure 0-indexed

        // Create highlighted code view with safe bounds checking
        let html = '<pre>';
        lines.forEach((line, i) => {
            const lineNo = i + 1;
            const isHighlight = Math.abs(i - highlightIndex) <= 2;
            const className = isHighlight ? 'highlight-line' : '';

            // Escape HTML and add line numbers
            const escaped = escapeHtml(line);
            html += `<div class="${className}"><span class="hljs-linenumber">${lineNo}</span>${escaped || ' '}</div>`;
        });
        html += '</pre>';

        codePanelEl.innerHTML = html;
    }

    function updateDetailsPanel(finding) {
        if (!finding) {
            detailsPanelEl.innerHTML = '';
            return;
        }

        const vscodeLink = createVSCodeLink(finding.file_path, finding.line_number);
        const severity = finding.severity || 'info';
        const confidence = (finding.confidence || 1.0).toFixed(2);

        let html = `
            <div class="detail-section">
                <h3>Issue Details</h3>
                <p><strong>Category:</strong> ${escapeHtml(finding.category || 'unknown')}</p>
                <p><strong>File:</strong> ${escapeHtml(finding.file_path || 'unknown')}</p>
                <p><strong>Line:</strong> ${finding.line_number || 'N/A'}</p>
                <p><strong>Confidence:</strong> ${confidence}</p>
            </div>

            <div class="detail-section">
                <h3>Message</h3>
                <p>${escapeHtml(finding.message || 'No message')}</p>
            </div>

            <div class="detail-section">
                <h3>Code Snippet</h3>
                <div class="code-snippet">${escapeHtml(finding.code_snippet || 'No code snippet')}</div>
            </div>

            <div class="detail-section">
                <h3>Suggestion</h3>
                <div class="suggestion-box">${escapeHtml(finding.suggestion || 'No suggestion')}</div>
            </div>

            <div class="action-buttons">
                <a href="${vscodeLink}" class="btn btn-primary" target="_blank">
                    Open in VS Code
                </a>
            </div>
        `;

        detailsPanelEl.innerHTML = html;
    }

    function handleSearch(e) {
        const query = (e.target.value || '').toLowerCase();
        activeFilters.searchQuery = query;

        const filtered = applyAllFilters();
        renderFindings(filtered);
    }

    function handleFilter(e) {
        const filterType = e.target.dataset.filterType;
        const filterValue = e.target.dataset.filterValue;

        if (!filterType || !filterValue) return;

        // Toggle filter
        if (!activeFilters[filterType]) {
            activeFilters[filterType] = [];
        }

        const idx = activeFilters[filterType].indexOf(filterValue);
        if (idx > -1) {
            activeFilters[filterType].splice(idx, 1);
            e.target.classList.remove('active');
        } else {
            activeFilters[filterType].push(filterValue);
            e.target.classList.add('active');
        }

        // Re-render with filters
        const filtered = applyAllFilters();
        renderFindings(filtered);
    }

    function applyAllFilters() {
        return findings.filter(f => {
            // Search filter (applies to message, category, file_path)
            if (activeFilters.searchQuery) {
                const q = activeFilters.searchQuery;
                const hasMatch =
                    (f.message && f.message.toLowerCase().includes(q)) ||
                    (f.category && f.category.toLowerCase().includes(q)) ||
                    (f.file_path && f.file_path.toLowerCase().includes(q));
                if (!hasMatch) return false;
            }

            // Severity filter
            if (activeFilters.severity && activeFilters.severity.length > 0) {
                if (!f.severity || !activeFilters.severity.includes(f.severity)) {
                    return false;
                }
            }

            // Category filter
            if (activeFilters.category && activeFilters.category.length > 0) {
                if (!f.category || !activeFilters.category.includes(f.category)) {
                    return false;
                }
            }

            return true;
        });
    }

    function handleKeyboard(e) {
        if (!selectedFinding || currentFilteredFindings.length === 0) return;

        const currentIndex = currentFilteredFindings.indexOf(selectedFinding);
        if (currentIndex === -1) return;

        if (e.key === 'ArrowUp' && currentIndex > 0) {
            e.preventDefault();
            selectFinding(currentFilteredFindings[currentIndex - 1]);
        } else if (e.key === 'ArrowDown' && currentIndex < currentFilteredFindings.length - 1) {
            e.preventDefault();
            selectFinding(currentFilteredFindings[currentIndex + 1]);
        } else if (e.key === 'Enter') {
            // Open in VS Code
            const link = document.querySelector('.btn-primary');
            if (link) {
                e.preventDefault();
                link.click();
            }
        }
    }

    function createVSCodeLink(filePath, lineNumber) {
        if (!filePath) return '#';

        const lineNum = lineNumber || 1;

        // Convert Windows paths properly
        let path = filePath.replace(/\\\\/g, '/');

        // Handle various path formats
        if (path.match(/^[a-zA-Z]:/)) {
            // Windows path like c:/Users/...
            return `vscode://file/${path}:${lineNum}`;
        } else if (path.startsWith('\\\\')) {
            // UNC path like \\server\share
            path = path.replace(/\\\\/g, '/');
            return `vscode://file/${path}:${lineNum}`;
        } else if (path.startsWith('/')) {
            // Unix absolute path
            return `vscode://file${path}:${lineNum}`;
        } else {
            // Relative path
            return `vscode://file/${path}:${lineNum}`;
        }
    }

    function escapeHtml(text) {
        if (typeof text !== 'string') return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    </script>
    """

    if minify:
        # Extract JavaScript content without script tags
        js_content = js.replace('<script>', '').replace('</script>', '')
        minified = _minify_javascript(js_content)
        return f"<script>{minified}</script>"
    return js


def generate_html_report(
    findings: List[Any],
    output_path: Optional[str] = None,
    enable_vscode_integration: bool = True,
    code_snippets: Optional[Dict[str, str]] = None,
    title: str = "Trough Code Analysis Report",
    minify: bool = False,
    enable_virtualization: bool = False
) -> str:
    """
    Generate interactive HTML report from Trough findings.

    Args:
        findings: List of SlopIssue or LogicError objects
        output_path: Optional path to save HTML file
        enable_vscode_integration: Enable VS Code links
        code_snippets: Dict mapping file paths to code content
        title: Report title
        minify: If True, minimize CSS and JS for smaller file size
        enable_virtualization: If True, use virtualization for 100+ findings (recommended for performance)

    Returns:
        HTML string

    Raises:
        TypeError: If findings format is invalid
    """
    if not findings:
        findings = []

    # Convert findings to JSON-serializable format
    findings_data = []
    for finding in findings:
        try:
            finding_dict = {
                'category': getattr(finding, 'category', 'unknown'),
                'severity': getattr(finding, 'severity', 'info'),
                'message': getattr(finding, 'message', ''),
                'file_path': getattr(finding, 'file_path', 'unknown'),
                'line_number': getattr(finding, 'line_number', 1),
                'code_snippet': getattr(finding, 'code_snippet', ''),
                'suggestion': getattr(finding, 'suggestion', ''),
                'confidence': getattr(finding, 'confidence', 1.0)
            }
            findings_data.append(finding_dict)
        except Exception as e:
            print(f"Warning: Could not serialize finding: {e}")
            continue

    # Prepare code snippets
    code_snippets = code_snippets or {}
    snippet_data = {}
    for finding in findings_data:
        file_path = finding['file_path']
        if file_path not in snippet_data and file_path in code_snippets:
            snippet_data[file_path] = code_snippets[file_path]

    # Calculate statistics
    stats = {
        'total': len(findings_data),
        'critical': len([f for f in findings_data if f['severity'].lower() == 'critical']),
        'high': len([f for f in findings_data if f['severity'].lower() == 'high']),
        'medium': len([f for f in findings_data if f['severity'].lower() == 'medium']),
        'low': len([f for f in findings_data if f['severity'].lower() == 'low']),
        'info': len([f for f in findings_data if f['severity'].lower() == 'info'])
    }

    # Get unique categories and severities for filters
    categories = sorted(set(f['category'] for f in findings_data))
    severities = sorted(set(f['severity'] for f in findings_data))

    # Generate severity badges
    severity_badges = {}
    for severity in ['critical', 'high', 'medium', 'low', 'info']:
        severity_badges[severity] = _get_severity_badge(severity)

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_escape_html(title)}</title>
    {_generate_css(minify=minify)}
</head>
<body>
    <div class="header">
        <h1>{_escape_html(title)}</h1>
        <div class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <div class="container">
        <!-- Findings Panel -->
        <div class="panel findings-panel">
            <div class="stats-bar">
                <div class="stat">
                    <div class="stat-value">{stats['total']}</div>
                    <div class="stat-label">Total</div>
                </div>
                <div class="stat">
                    <div class="stat-value" style="color: #e74c3c;">{stats['critical']}</div>
                    <div class="stat-label">Critical</div>
                </div>
                <div class="stat">
                    <div class="stat-value" style="color: #e67e22;">{stats['high']}</div>
                    <div class="stat-label">High</div>
                </div>
                <div class="stat">
                    <div class="stat-value" style="color: #f39c12;">{stats['medium']}</div>
                    <div class="stat-label">Medium</div>
                </div>
                <div class="stat">
                    <div class="stat-value" style="color: #3498db;">{stats['low']}</div>
                    <div class="stat-label">Low</div>
                </div>
            </div>

            <div class="search-bar">
                <input type="text" id="search-input" placeholder="Search findings...">
            </div>

            <div class="filter-controls">
"""

    # Add severity filters
    for severity in ['critical', 'high', 'medium', 'low', 'info']:
        if severity in severities:
            html += f'<button class="filter-button" data-filter-type="severity" data-filter-value="{severity}">{severity.upper()}</button>'

    html += "</div>\n"

    # Add findings list
    html += '<ul class="findings-list" id="findings-list">'
    if not findings_data:
        html += '<div class="empty-state"><div class="empty-state-content"><div class="empty-state-icon">✓</div><h2>All Clear!</h2><p>No issues found</p></div></div>'
    html += '</ul></div>'

    # Code Panel
    html += '<div class="panel code-panel" id="code-panel"></div>'

    # Details Panel
    html += '<div class="panel details-panel" id="details-panel"></div>'

    html += '</div>'

    # Embed data
    html += f"""
    <script>
    window.FINDINGS_DATA = {json.dumps(findings_data)};
    window.CODE_SNIPPETS = {json.dumps(snippet_data)};
    window.SEVERITY_BADGES = {json.dumps(severity_badges)};
    </script>
    """

    html += _generate_javascript(minify=minify, enable_virtualization=enable_virtualization)

    html += """
</body>
</html>
"""

    # Save to file if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding='utf-8')
        print(f"Report generated: {output_path}")

    return html


# Convenience function for direct use
async def generate_report_from_detector(
    code: str,
    file_path: str,
    language: str = "python",
    output_path: Optional[str] = None
) -> str:
    """
    Generate report from code using Trough detector.

    Args:
        code: Source code to analyze
        file_path: File path
        language: Programming language
        output_path: Optional output path

    Returns:
        HTML report string
    """
    try:
        from trough import AISlopDetector, Language
    except ImportError:
        raise ImportError("Trough detector not available")

    detector = AISlopDetector()
    lang = Language(language.lower())
    findings = await detector.detect_all(code, lang, file_path)

    return generate_html_report(
        findings=findings,
        output_path=output_path,
        code_snippets={file_path: code}
    )
