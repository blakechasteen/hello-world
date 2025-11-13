/**
 * Memory Graph Explorer
 * ======================
 * Interactive knowledge graph exploration for HoloLoom's memory systems.
 *
 * Features:
 * - Entity search with auto-complete
 * - Relationship visualization
 * - Memory statistics and health metrics
 * - Backend status (INMEMORY/HYBRID/HYPERSPACE)
 *
 * Philosophy: Framework → Elegance → Visual Clarity
 */

class MemoryExplorer {
    constructor(apiBase = 'http://localhost:8000') {
        this.apiBase = apiBase;
        this.updateInterval = 10000; // 10 seconds
        this.intervalId = null;
        this.searchResults = [];
        this.selectedEntity = null;
    }

    /**
     * Initialize explorer
     */
    async initialize() {
        console.log('Initializing Memory Explorer...');

        // Initial data load
        await this.refreshStats();

        // Start polling for stats
        this.intervalId = setInterval(() => this.refreshStats(), this.updateInterval);

        // Setup event listeners
        this.setupEventListeners();

        console.log('✓ Memory Explorer initialized');
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
     * Setup event listeners
     */
    setupEventListeners() {
        // Search button
        const searchBtn = document.getElementById('memory-search-btn');
        if (searchBtn) {
            searchBtn.addEventListener('click', () => this.handleSearch());
        }

        // Search input (Enter key)
        const searchInput = document.getElementById('memory-search-input');
        if (searchInput) {
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.handleSearch();
            });

            // Live search (with debounce)
            let searchTimeout;
            searchInput.addEventListener('input', (e) => {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(() => {
                    if (e.target.value.trim().length >= 2) {
                        this.handleSearch();
                    }
                }, 500);
            });
        }

        // Refresh button
        const refreshBtn = document.getElementById('memory-refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshStats());
        }
    }

    /**
     * Refresh memory statistics
     */
    async refreshStats() {
        try {
            const response = await fetch(`${this.apiBase}/memory/stats`);

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const stats = await response.json();
            this.updateStats(stats);
        } catch (error) {
            console.error('Failed to refresh memory stats:', error);
            this.showError('stats', error.message);
        }
    }

    /**
     * Handle search
     */
    async handleSearch() {
        const input = document.getElementById('memory-search-input');
        if (!input) return;

        const query = input.value.trim();
        if (query.length === 0) return;

        try {
            const response = await fetch(`${this.apiBase}/memory/search`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    query: query,
                    limit: 10,
                    threshold: 0.5
                })
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const data = await response.json();
            this.searchResults = data.results || [];
            this.updateSearchResults();
        } catch (error) {
            console.error('Search failed:', error);
            this.showError('search', error.message);
        }
    }

    /**
     * Update statistics display
     */
    updateStats(stats) {
        // Metrics
        this.updateMetric('total-entities', stats.total_entities || 0);
        this.updateMetric('total-relationships', stats.total_relationships || 0);
        this.updateMetric('total-memories', stats.total_memories || 0);

        // Backend status
        const backendEl = document.getElementById('memory-backend-status');
        if (backendEl) {
            backendEl.textContent = stats.backend || 'UNKNOWN';
            let className = 'badge ';
            if (stats.backend === 'HYBRID') className += 'success';
            else if (stats.backend === 'INMEMORY') className += 'info';
            else if (stats.backend === 'HYPERSPACE') className += 'warning';
            else className += 'secondary';

            backendEl.className = className;
        }

        // Health score (calculated)
        const healthScore = this.calculateHealthScore(stats);
        this.updateMetric('memory-health-score', healthScore.toFixed(1));

        const healthBadge = document.getElementById('memory-health-badge');
        if (healthBadge) {
            let className = 'badge ';
            if (healthScore >= 80) className += 'success';
            else if (healthScore >= 60) className += 'warning';
            else className += 'danger';

            healthBadge.className = className;
            healthBadge.textContent = healthScore >= 80 ? 'Healthy' : healthScore >= 60 ? 'Fair' : 'Poor';
        }
    }

    /**
     * Update search results display
     */
    updateSearchResults() {
        const container = document.getElementById('memory-search-results');
        if (!container) return;

        if (this.searchResults.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
                    </svg>
                    <div>No results found</div>
                    <div style="margin-top: 0.5rem; font-size: 0.875rem; color: var(--secondary);">
                        Try different search terms
                    </div>
                </div>
            `;
            return;
        }

        // Render search results as cards
        let html = '<div class="memory-results-grid">';

        this.searchResults.forEach(result => {
            html += `
                <div class="memory-result-card" onclick="memoryExplorer.selectEntity('${this.escapeHtml(result.entity)}')">
                    <div class="result-entity">${this.escapeHtml(result.entity)}</div>
                    <div class="result-content">${this.escapeHtml(this.truncate(result.content, 100))}</div>
                    <div class="result-footer">
                        <span class="badge info">Similarity: ${result.similarity.toFixed(2)}</span>
                        ${result.metadata?.source ? `<span class="badge secondary">${this.escapeHtml(result.metadata.source)}</span>` : ''}
                    </div>
                </div>
            `;
        });

        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * Select an entity to view details
     */
    selectEntity(entity) {
        this.selectedEntity = entity;
        console.log('Selected entity:', entity);

        // TODO: Fetch entity details and relationships
        // For now, just show selection
        alert(`Selected entity: ${entity}\n\nEntity details and relationship visualization coming in Phase 3!`);
    }

    /**
     * Calculate health score based on stats
     */
    calculateHealthScore(stats) {
        const entities = stats.total_entities || 0;
        const relationships = stats.total_relationships || 0;
        const memories = stats.total_memories || 0;

        // Simple heuristic:
        // - Good entity count (>100): +30 points
        // - Good relationship/entity ratio (>2): +40 points
        // - Good memory count (>100): +30 points

        let score = 0;

        if (entities >= 100) score += 30;
        else if (entities >= 50) score += 20;
        else if (entities >= 10) score += 10;

        const ratio = entities > 0 ? relationships / entities : 0;
        if (ratio >= 3) score += 40;
        else if (ratio >= 2) score += 30;
        else if (ratio >= 1) score += 20;

        if (memories >= 100) score += 30;
        else if (memories >= 50) score += 20;
        else if (memories >= 10) score += 10;

        return score;
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
        const container = document.getElementById(`memory-error-${section}`);
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
window.MemoryExplorer = MemoryExplorer;
