/**
 * HoloLoom Issues Tracker
 * Lightweight client-side issue tracking system with localStorage persistence
 *
 * Features:
 * - CRUD operations for issues
 * - Comments and discussion threads
 * - Labels, priority, component tracking
 * - Search and filtering
 * - Markdown support
 * - Export/import functionality
 */

// ============================================================================
// ISSUES MANAGER - Core CRUD operations
// ============================================================================

const IssuesManager = {
    STORAGE_KEY: 'hololoom_issues',
    DEFAULT_STATE: {
        issues: [],
        nextId: 1,
        nextCommentId: 1
    },

    /**
     * Initialize storage with default state if empty
     */
    init() {
        if (!this.getState()) {
            this.setState(this.DEFAULT_STATE);
        }
    },

    /**
     * Get entire state from localStorage
     */
    getState() {
        try {
            const data = localStorage.getItem(this.STORAGE_KEY);
            return data ? JSON.parse(data) : null;
        } catch (e) {
            console.error('Failed to read from localStorage:', e);
            return null;
        }
    },

    /**
     * Save entire state to localStorage
     */
    setState(state) {
        try {
            localStorage.setItem(this.STORAGE_KEY, JSON.stringify(state));
            this.updateStorageInfo();
            return true;
        } catch (e) {
            console.error('Failed to write to localStorage:', e);
            return false;
        }
    },

    /**
     * Create a new issue
     * @param {Object} issueData - { title, description, component, priority, labels }
     * @returns {number} Issue ID
     */
    createIssue(issueData) {
        const state = this.getState();
        const id = state.nextId++;

        const issue = {
            id: id,
            title: issueData.title,
            description: issueData.description,
            status: 'open',
            component: issueData.component,
            priority: issueData.priority,
            labels: issueData.labels || [],
            created: new Date().toISOString(),
            updated: new Date().toISOString(),
            comments: []
        };

        state.issues.push(issue);
        this.setState(state);
        return id;
    },

    /**
     * Get a single issue by ID
     */
    getIssue(id) {
        const state = this.getState();
        return state.issues.find(issue => issue.id === id);
    },

    /**
     * Get all issues
     */
    getAllIssues() {
        const state = this.getState();
        return state.issues || [];
    },

    /**
     * Update an issue
     */
    updateIssue(id, updates) {
        const state = this.getState();
        const issue = state.issues.find(i => i.id === id);
        if (issue) {
            Object.assign(issue, updates);
            issue.updated = new Date().toISOString();
            this.setState(state);
            return true;
        }
        return false;
    },

    /**
     * Delete an issue
     */
    deleteIssue(id) {
        const state = this.getState();
        state.issues = state.issues.filter(i => i.id !== id);
        this.setState(state);
        return true;
    },

    /**
     * Add a comment to an issue
     */
    addComment(issueId, commentData) {
        const state = this.getState();
        const issue = state.issues.find(i => i.id === issueId);

        if (!issue) return null;

        if (!issue.comments) issue.comments = [];

        const commentId = state.nextCommentId++;
        const comment = {
            id: commentId,
            author: commentData.author || 'Anonymous',
            text: commentData.text,
            timestamp: new Date().toISOString()
        };

        issue.comments.push(comment);
        issue.updated = new Date().toISOString();
        this.setState(state);
        return commentId;
    },

    /**
     * Update storage info display
     */
    updateStorageInfo() {
        const storageInfo = document.getElementById('storage-info');
        if (storageInfo) {
            try {
                const data = localStorage.getItem(this.STORAGE_KEY);
                const bytes = new Blob([data]).size;
                const kb = (bytes / 1024).toFixed(2);
                const lastUpdated = document.getElementById('last-updated');
                if (lastUpdated) {
                    lastUpdated.textContent = new Date().toLocaleTimeString();
                }
                storageInfo.textContent = `Storage: ${kb} KB`;
            } catch (e) {
                storageInfo.textContent = 'Storage: unavailable';
            }
        }
    },

    /**
     * Export issues to JSON file
     */
    exportToJSON() {
        const state = this.getState();
        const dataStr = JSON.stringify(state, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `hololoom-issues-${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        URL.revokeObjectURL(url);
    },

    /**
     * Import issues from JSON file
     */
    async importFromJSON(file) {
        if (!file) return false;

        try {
            const text = await file.text();
            const state = JSON.parse(text);

            // Validate structure
            if (!state.issues || !Array.isArray(state.issues)) {
                throw new Error('Invalid format: missing issues array');
            }

            if (typeof state.nextId !== 'number') {
                throw new Error('Invalid format: missing nextId');
            }

            // Merge with existing data (avoid ID conflicts)
            const current = this.getState();
            const maxId = Math.max(
                current.nextId,
                ...state.issues.map(i => i.id || 0)
            );

            state.nextId = Math.max(state.nextId, maxId + 1);
            state.issues = current.issues.concat(state.issues);

            // Recalculate next comment ID if needed
            let maxCommentId = current.nextCommentId || 1;
            state.issues.forEach(issue => {
                if (issue.comments) {
                    const max = Math.max(...issue.comments.map(c => c.id || 0));
                    maxCommentId = Math.max(maxCommentId, max + 1);
                }
            });
            state.nextCommentId = maxCommentId;

            this.setState(state);
            alert(`✓ Imported ${state.issues.length} issue(s). Page will reload.`);
            location.reload();
            return true;
        } catch (error) {
            alert(`✗ Import failed: ${error.message}`);
            return false;
        }
    }
};

// ============================================================================
// MARKDOWN SUPPORT - Simple Markdown to HTML converter
// ============================================================================

const IssuesMarkdown = {
    /**
     * Convert simple Markdown to HTML
     * Supports: **bold**, *italic*, `code`, [links](url), # headings, - lists
     */
    toHTML(markdown) {
        if (!markdown) return '';

        let html = markdown;

        // Escape HTML
        html = html
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Re-unescape for our markdown processing
        const lines = html.split('\n');
        const processed = [];

        let inCodeBlock = false;

        for (let line of lines) {
            // Code blocks (triple backticks)
            if (line.trim().startsWith('```')) {
                inCodeBlock = !inCodeBlock;
                if (!inCodeBlock) {
                    processed.push('</code></pre>');
                } else {
                    processed.push('<pre><code>');
                }
                continue;
            }

            if (inCodeBlock) {
                processed.push(line);
                continue;
            }

            // Headings
            if (line.match(/^#{1,6}\s/)) {
                const level = line.match(/^#+/)[0].length;
                const text = line.replace(/^#+\s/, '');
                processed.push(`<h${level + 1}>${this.applyInlineFormatting(text)}</h${level + 1}>`);
                continue;
            }

            // Lists
            if (line.match(/^[-*]\s/)) {
                const text = line.replace(/^[-*]\s/, '');
                processed.push(`<li>${this.applyInlineFormatting(text)}</li>`);
                continue;
            }

            // Paragraphs
            if (line.trim()) {
                processed.push(`<p>${this.applyInlineFormatting(line)}</p>`);
            } else if (processed.length > 0 && !processed[processed.length - 1].startsWith('<li')) {
                processed.push('');
            }
        }

        html = processed.join('\n');

        // Wrap lists in <ul>
        html = html.replace(/(<li>.*?<\/li>\n?)+/gs, match => `<ul>${match}</ul>`);

        return html;
    },

    /**
     * Apply inline formatting: **bold**, *italic*, `code`, [links](url)
     */
    applyInlineFormatting(text) {
        // Links [text](url)
        text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

        // Bold **text**
        text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Italic *text*
        text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Inline code `text`
        text = text.replace(/`([^`]+)`/g, '<code>$1</code>');

        return text;
    }
};

// ============================================================================
// UI MANAGER - Rendering and filtering
// ============================================================================

const IssuesUI = {
    ITEMS_PER_PAGE: 10,
    currentPage: 1,
    filteredIssues: [],

    /**
     * Get current filter values
     */
    getFilters() {
        return {
            status: document.getElementById('filter-status')?.value || '',
            priority: document.getElementById('filter-priority')?.value || '',
            component: document.getElementById('filter-component')?.value || '',
            label: document.getElementById('filter-label')?.value || '',
            search: document.getElementById('search-input')?.value.toLowerCase() || ''
        };
    },

    /**
     * Filter and sort issues
     */
    getFilteredAndSorted() {
        const allIssues = IssuesManager.getAllIssues();
        const filters = this.getFilters();

        let filtered = allIssues.filter(issue => {
            // Status filter
            if (filters.status && issue.status !== filters.status) return false;

            // Priority filter
            if (filters.priority && issue.priority !== filters.priority) return false;

            // Component filter
            if (filters.component && issue.component !== filters.component) return false;

            // Label filter
            if (filters.label && !issue.labels.includes(filters.label)) return false;

            // Search filter
            if (filters.search) {
                const searchText = filters.search;
                const titleMatch = issue.title.toLowerCase().includes(searchText);
                const descMatch = issue.description.toLowerCase().includes(searchText);
                if (!titleMatch && !descMatch) return false;
            }

            return true;
        });

        // Sort
        const sortBy = document.getElementById('sort-by')?.value || 'newest';
        filtered.sort((a, b) => {
            switch (sortBy) {
                case 'newest':
                    return new Date(b.created) - new Date(a.created);
                case 'oldest':
                    return new Date(a.created) - new Date(b.created);
                case 'title':
                    return a.title.localeCompare(b.title);
                case 'priority':
                    const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
                    return (priorityOrder[a.priority] || 3) - (priorityOrder[b.priority] || 3);
                case 'updated':
                    return new Date(b.updated) - new Date(a.updated);
                default:
                    return 0;
            }
        });

        this.filteredIssues = filtered;
        this.currentPage = 1;
        return filtered;
    },

    /**
     * Render the issues list
     */
    render() {
        const filtered = this.getFilteredAndSorted();
        const list = document.getElementById('issues-list');
        const empty = document.getElementById('empty-state');
        const pagination = document.getElementById('pagination');

        if (filtered.length === 0) {
            list.style.display = 'none';
            empty.style.display = 'block';
            pagination.style.display = 'none';
            return;
        }

        list.style.display = 'block';
        empty.style.display = 'none';

        // Pagination
        const totalPages = Math.ceil(filtered.length / this.ITEMS_PER_PAGE);
        const start = (this.currentPage - 1) * this.ITEMS_PER_PAGE;
        const end = start + this.ITEMS_PER_PAGE;
        const paginated = filtered.slice(start, end);

        // Render issues
        list.innerHTML = paginated.map(issue => this.renderIssueCard(issue)).join('');

        // Show/hide pagination
        if (totalPages > 1) {
            pagination.style.display = 'flex';
            document.getElementById('page-number').textContent = this.currentPage;
            document.getElementById('total-pages').textContent = totalPages;
            document.getElementById('prev-page').disabled = this.currentPage === 1;
            document.getElementById('next-page').disabled = this.currentPage === totalPages;
        } else {
            pagination.style.display = 'none';
        }
    },

    /**
     * Render a single issue card
     */
    renderIssueCard(issue) {
        const statusColor = issue.status === 'open' ? 'open' : 'closed';
        const statusEmoji = issue.status === 'open' ? '🟢' : '🔴';
        const priorityClass = `priority-${issue.priority}`;
        const labels = issue.labels
            .map(label => `<span class="label-badge label-${label}">${label}</span>`)
            .join('');

        const commentCount = issue.comments?.length || 0;

        return `
            <div class="issue-card">
                <div class="issue-card-main">
                    <div class="issue-card-header">
                        <a href="./view.html?id=${issue.id}" class="issue-link">
                            <span class="issue-number">#${issue.id}</span>
                            <span class="issue-title">${this.escapeHTML(issue.title)}</span>
                        </a>
                        <span class="status-badge status-${statusColor}">${statusEmoji} ${issue.status.toUpperCase()}</span>
                    </div>

                    <div class="issue-card-meta">
                        <span class="component-badge">${this.escapeHTML(issue.component)}</span>
                        <span class="priority-badge ${priorityClass}">
                            ${issue.priority.charAt(0).toUpperCase() + issue.priority.slice(1)}
                        </span>
                        ${labels ? `<div class="labels-inline">${labels}</div>` : ''}
                    </div>

                    <p class="issue-preview">${this.escapeHTML(issue.description.substring(0, 120))}...</p>
                </div>

                <div class="issue-card-footer">
                    <span class="text-muted">
                        Created ${this.formatDate(issue.created)}
                    </span>
                    ${commentCount > 0 ? `<span class="comment-icon">💬 ${commentCount}</span>` : ''}
                </div>
            </div>
        `;
    },

    /**
     * Update statistics
     */
    updateStats() {
        const issues = IssuesManager.getAllIssues();
        document.getElementById('stat-total').textContent = issues.length;
        document.getElementById('stat-open').textContent = issues.filter(i => i.status === 'open').length;
        document.getElementById('stat-closed').textContent = issues.filter(i => i.status === 'closed').length;
        document.getElementById('stat-high').textContent = issues.filter(i => i.priority === 'high' || i.priority === 'critical').length;
    },

    /**
     * Reset all filters
     */
    resetFilters() {
        document.getElementById('filter-status').value = '';
        document.getElementById('filter-priority').value = '';
        document.getElementById('filter-component').value = '';
        document.getElementById('filter-label').value = '';
        document.getElementById('search-input').value = '';
        document.getElementById('sort-by').value = 'newest';
        this.render();
    },

    /**
     * Pagination controls
     */
    previousPage() {
        if (this.currentPage > 1) {
            this.currentPage--;
            this.render();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    },

    nextPage() {
        const totalPages = Math.ceil(this.filteredIssues.length / this.ITEMS_PER_PAGE);
        if (this.currentPage < totalPages) {
            this.currentPage++;
            this.render();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    },

    /**
     * Format date for display
     */
    formatDate(isoDate) {
        const date = new Date(isoDate);
        const now = new Date();
        const days = Math.floor((now - date) / (1000 * 60 * 60 * 24));

        if (days === 0) return 'today';
        if (days === 1) return 'yesterday';
        if (days < 7) return `${days} days ago`;
        if (days < 30) return `${Math.floor(days / 7)} weeks ago`;
        if (days < 365) return `${Math.floor(days / 30)} months ago`;
        return date.toLocaleDateString();
    },

    /**
     * Escape HTML special characters
     */
    escapeHTML(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
};

// ============================================================================
// EXPORT - Make utilities globally available
// ============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { IssuesManager, IssuesUI, IssuesMarkdown };
}
