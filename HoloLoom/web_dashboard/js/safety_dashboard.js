/**
 * Safety & Alignment Dashboard
 * ==============================
 * Real-time monitoring of HoloLoom's alignment framework.
 *
 * Features:
 * - Guardrail status (active/inactive, actions gated/blocked)
 * - Audit trail browser with search/filtering
 * - Deception detection alerts
 * - Risk level distribution
 * - Safety score trends
 *
 * Philosophy: Framework → Elegance → Safety First
 */

class SafetyDashboard {
    constructor(apiBase = 'http://localhost:8000') {
        this.apiBase = apiBase;
        this.updateInterval = 3000; // 3 seconds (more frequent for safety)
        this.intervalId = null;
        this.auditTrailPage = 0;
        this.auditTrailLimit = 20;
        this.searchQuery = '';
    }

    /**
     * Initialize dashboard
     */
    async initialize() {
        console.log('Initializing Safety Dashboard...');

        // Initial data load
        await Promise.all([
            this.refreshStatus(),
            this.refreshAuditTrail()
        ]);

        // Start polling
        this.intervalId = setInterval(() => {
            this.refreshStatus();
        }, this.updateInterval);

        // Setup event listeners
        this.setupEventListeners();

        console.log('✓ Safety Dashboard initialized');
    }

    /**
     * Stop polling and cleanup
     */
    destroy() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }

    /**
     * Setup event listeners for interactive elements
     */
    setupEventListeners() {
        // Search button
        const searchBtn = document.getElementById('audit-search-btn');
        if (searchBtn) {
            searchBtn.addEventListener('click', () => this.handleSearch());
        }

        // Search input (Enter key)
        const searchInput = document.getElementById('audit-search-input');
        if (searchInput) {
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.handleSearch();
            });
        }

        // Refresh button
        const refreshBtn = document.getElementById('audit-refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshAuditTrail());
        }

        // Load more button
        const loadMoreBtn = document.getElementById('audit-load-more-btn');
        if (loadMoreBtn) {
            loadMoreBtn.addEventListener('click', () => this.loadMoreAuditEntries());
        }
    }

    /**
     * Refresh safety status
     */
    async refreshStatus() {
        try {
            const response = await fetch(`${this.apiBase}/safety/status`);

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const status = await response.json();
            this.updateStatus(status);
        } catch (error) {
            console.error('Failed to refresh safety status:', error);
            this.showError('status', error.message);
        }
    }

    /**
     * Refresh audit trail
     */
    async refreshAuditTrail() {
        try {
            const offset = this.auditTrailPage * this.auditTrailLimit;
            const url = `${this.apiBase}/safety/audit-trail?limit=${this.auditTrailLimit}&offset=${offset}`;

            const response = await fetch(url);

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const data = await response.json();
            this.updateAuditTrail(data.entries, data.total);
        } catch (error) {
            console.error('Failed to refresh audit trail:', error);
            this.showError('audit', error.message);
        }
    }

    /**
     * Update status display
     */
    updateStatus(status) {
        // Guardrails status
        this.updateStatusIndicator('guardrails-status', status.guardrails_active);

        // Deception detector status
        this.updateStatusIndicator('deception-detector-status', status.deception_detector_active);

        // Audit trail status
        this.updateStatusIndicator('audit-trail-status', status.audit_trail_active);

        // Metrics
        this.updateMetric('total-actions-gated', status.total_actions_gated || 0);
        this.updateMetric('blocked-actions', status.blocked_actions || 0);

        // Block rate
        const blockRate = status.total_actions_gated > 0
            ? ((status.blocked_actions / status.total_actions_gated) * 100).toFixed(1)
            : '0.0';
        this.updateMetric('block-rate', `${blockRate}%`);

        // Block rate badge
        const blockRateBadge = document.getElementById('block-rate-badge');
        if (blockRateBadge) {
            const rate = parseFloat(blockRate);
            let className = 'badge ';
            if (rate >= 20) className += 'danger';
            else if (rate >= 10) className += 'warning';
            else className += 'success';

            blockRateBadge.className = className;
            blockRateBadge.textContent = rate >= 10 ? 'High' : rate >= 5 ? 'Medium' : 'Low';
        }
    }

    /**
     * Update audit trail display
     */
    updateAuditTrail(entries, total) {
        const container = document.getElementById('audit-trail-container');
        if (!container) return;

        if (entries.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M9 13h6v-2H9v2zm0 4h6v-2H9v2zm0-8h6V7H9v2zm11-6H4c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H4V5h16v14z"/>
                    </svg>
                    <div>No audit trail entries yet</div>
                </div>
            `;
            return;
        }

        // Render audit trail table
        let html = '<table class="audit-trail-table"><thead><tr>';
        html += '<th>Timestamp</th>';
        html += '<th>Query</th>';
        html += '<th>Action</th>';
        html += '<th>Outcome</th>';
        html += '<th>Confidence</th>';
        html += '<th>Safety Score</th>';
        html += '</tr></thead><tbody>';

        entries.forEach(entry => {
            const timestamp = new Date(entry.timestamp * 1000).toLocaleString();
            const queryPreview = entry.query ? this.truncate(entry.query, 40) : '—';

            html += '<tr>';
            html += `<td>${timestamp}</td>`;
            html += `<td title="${this.escapeHtml(entry.query || '')}">${this.escapeHtml(queryPreview)}</td>`;
            html += `<td><code>${this.escapeHtml(entry.action)}</code></td>`;
            html += `<td>${this.renderOutcomeBadge(entry.outcome)}</td>`;
            html += `<td>${entry.confidence ? entry.confidence.toFixed(2) : '—'}</td>`;
            html += `<td>${this.renderSafetyScoreBadge(entry.safety_score)}</td>`;
            html += '</tr>';
        });

        html += '</tbody></table>';

        // Add pagination info
        html += `<div class="audit-pagination">Showing ${entries.length} of ${total} entries</div>`;

        container.innerHTML = html;

        // Show/hide load more button
        const loadMoreBtn = document.getElementById('audit-load-more-btn');
        if (loadMoreBtn) {
            loadMoreBtn.style.display = entries.length < total ? 'block' : 'none';
        }
    }

    /**
     * Handle search
     */
    async handleSearch() {
        const input = document.getElementById('audit-search-input');
        if (!input) return;

        this.searchQuery = input.value.trim();
        this.auditTrailPage = 0; // Reset to first page

        await this.refreshAuditTrail();
    }

    /**
     * Load more audit entries
     */
    async loadMoreAuditEntries() {
        this.auditTrailPage++;
        await this.refreshAuditTrail();
    }

    /**
     * Update status indicator
     */
    updateStatusIndicator(id, active) {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = active ? 'Active' : 'Inactive';
            el.className = `badge ${active ? 'success' : 'warning'}`;
        }
    }

    /**
     * Update metric value
     */
    updateMetric(id, value) {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = value;
        }
    }

    /**
     * Render outcome badge
     */
    renderOutcomeBadge(outcome) {
        let className = 'badge ';
        if (outcome === 'success') className += 'success';
        else if (outcome === 'failure') className += 'danger';
        else className += 'info';

        return `<span class="${className}">${this.escapeHtml(outcome)}</span>`;
    }

    /**
     * Render safety score badge
     */
    renderSafetyScoreBadge(score) {
        if (score === null || score === undefined) return '—';

        let className = 'badge ';
        if (score >= 0.8) className += 'success';
        else if (score >= 0.6) className += 'info';
        else if (score >= 0.4) className += 'warning';
        else className += 'danger';

        return `<span class="${className}">${score.toFixed(2)}</span>`;
    }

    /**
     * Truncate text
     */
    truncate(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    /**
     * Show error message
     */
    showError(section, message) {
        const container = document.getElementById(`safety-error-${section}`);
        if (container) {
            container.innerHTML = `
                <div class="error">
                    <div class="error-title">Error</div>
                    <div>${this.escapeHtml(message)}</div>
                </div>
            `;
            container.style.display = 'block';
        }
    }

    /**
     * Escape HTML
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Export for use in control panel
window.SafetyDashboard = SafetyDashboard;
