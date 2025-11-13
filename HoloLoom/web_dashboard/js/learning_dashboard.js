/**
 * Recursive Learning Dashboard
 * ============================
 * Real-time monitoring of HoloLoom's 5-phase recursive learning system.
 *
 * Features:
 * - Learning loop statistics (queries, refinements, hot patterns)
 * - Thompson Sampling arm performance (alpha/beta, expected rewards)
 * - Policy weight evolution over time
 * - Hot pattern visualization (table with sparklines)
 * - Refinement strategy effectiveness
 *
 * Philosophy: Framework → Elegance → Zero Dependencies
 */

class LearningDashboard {
    constructor(apiBase = 'http://localhost:8000') {
        this.apiBase = apiBase;
        this.updateInterval = 5000; // 5 seconds
        this.intervalId = null;
        this.statsHistory = [];
        this.maxHistoryLength = 50;
    }

    /**
     * Initialize dashboard and start polling
     */
    async initialize() {
        console.log('Initializing Learning Dashboard...');

        // Initial data load
        await this.refreshData();

        // Start polling
        this.intervalId = setInterval(() => this.refreshData(), this.updateInterval);

        console.log('✓ Learning Dashboard initialized');
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
     * Refresh all data from API
     */
    async refreshData() {
        try {
            const stats = await this.fetchLearningStats();

            if (stats) {
                // Store in history
                this.statsHistory.push({
                    timestamp: Date.now(),
                    ...stats
                });

                // Trim history
                if (this.statsHistory.length > this.maxHistoryLength) {
                    this.statsHistory.shift();
                }

                // Update UI
                this.updateOverview(stats);
                this.updateThompsonSampling(stats.thompson_sampling_stats);
                this.updatePolicyWeights(stats.policy_weights);
                this.updateHotPatterns();
            }
        } catch (error) {
            console.error('Failed to refresh learning data:', error);
            this.showError(error.message);
        }
    }

    /**
     * Fetch learning statistics from API
     */
    async fetchLearningStats() {
        const response = await fetch(`${this.apiBase}/learning/status`);

        if (response.status === 503) {
            // Learning engine not initialized
            this.showWarning('Learning engine not initialized');
            return null;
        }

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Fetch hot patterns from API
     */
    async fetchHotPatterns(limit = 20) {
        const response = await fetch(`${this.apiBase}/learning/patterns?limit=${limit}`);

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Update overview metrics
     */
    updateOverview(stats) {
        // Learning enabled status
        const statusEl = document.getElementById('learning-enabled-status');
        if (statusEl) {
            statusEl.textContent = stats.learning_enabled ? 'Active' : 'Inactive';
            statusEl.className = `badge ${stats.learning_enabled ? 'success' : 'warning'}`;
        }

        // Background learning status
        const bgLearningEl = document.getElementById('background-learning-status');
        if (bgLearningEl) {
            bgLearningEl.textContent = stats.background_learning_active ? 'Running' : 'Stopped';
            bgLearningEl.className = `badge ${stats.background_learning_active ? 'success' : 'warning'}`;
        }

        // Metrics
        this.updateMetric('queries-processed', stats.total_queries_processed || 0);
        this.updateMetric('total-refinements', stats.total_refinements || 0);
        this.updateMetric('hot-patterns-count', stats.hot_patterns_count || 0);

        // Refinement rate (%)
        const refinementRate = stats.total_queries_processed > 0
            ? ((stats.total_refinements / stats.total_queries_processed) * 100).toFixed(1)
            : '0.0';
        this.updateMetric('refinement-rate', `${refinementRate}%`);
    }

    /**
     * Update Thompson Sampling arm statistics
     */
    updateThompsonSampling(thompsonStats) {
        if (!thompsonStats) return;

        const container = document.getElementById('thompson-arms-container');
        if (!container) return;

        const totalArms = thompsonStats.total_arms || 0;
        const expectedRewards = thompsonStats.expected_rewards || [];

        if (totalArms === 0 || expectedRewards.length === 0) {
            container.innerHTML = '<div class="empty-state">No Thompson Sampling data available</div>';
            return;
        }

        // Render arm statistics
        let html = '<table><thead><tr>';
        html += '<th>Arm</th>';
        html += '<th>Expected Reward</th>';
        html += '<th>Confidence</th>';
        html += '<th>Trend</th>';
        html += '</tr></thead><tbody>';

        expectedRewards.forEach((reward, idx) => {
            const confidence = this.calculateConfidenceWidth(reward);
            const trend = this.getRewardTrend(idx);

            html += '<tr>';
            html += `<td>Arm ${idx + 1}</td>`;
            html += `<td>${reward.toFixed(3)}</td>`;
            html += `<td><div class="progress-bar"><div class="progress-fill" style="width: ${confidence}%"></div></div></td>`;
            html += `<td class="sparkline">${this.renderSparkline(trend)}</td>`;
            html += '</tr>';
        });

        html += '</tbody></table>';
        container.innerHTML = html;
    }

    /**
     * Update policy weight evolution
     */
    updatePolicyWeights(policyWeights) {
        if (!policyWeights) return;

        const container = document.getElementById('policy-weights-container');
        if (!container) return;

        // Render policy weights as horizontal bars
        let html = '<div class="policy-weights">';

        Object.entries(policyWeights).forEach(([mode, weight]) => {
            const percentage = (weight * 100).toFixed(1);
            html += `
                <div class="policy-weight-item">
                    <div class="policy-label">${mode.toUpperCase()}</div>
                    <div class="policy-bar">
                        <div class="policy-fill" style="width: ${percentage}%"></div>
                    </div>
                    <div class="policy-value">${percentage}%</div>
                </div>
            `;
        });

        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * Update hot patterns table
     */
    async updateHotPatterns() {
        const container = document.getElementById('hot-patterns-container');
        if (!container) return;

        try {
            const data = await this.fetchHotPatterns(10);
            const patterns = data.patterns || [];

            if (patterns.length === 0) {
                container.innerHTML = '<div class="empty-state">No hot patterns yet. System is still learning.</div>';
                return;
            }

            // Render patterns table
            let html = '<table><thead><tr>';
            html += '<th>Motif</th>';
            html += '<th>Tool</th>';
            html += '<th>Confidence</th>';
            html += '<th>Access Count</th>';
            html += '<th>Heat Score</th>';
            html += '</tr></thead><tbody>';

            patterns.forEach(pattern => {
                html += '<tr>';
                html += `<td><code>${this.escapeHtml(pattern.motif)}</code></td>`;
                html += `<td><span class="badge info">${this.escapeHtml(pattern.tool)}</span></td>`;
                html += `<td>${pattern.confidence.toFixed(2)}</td>`;
                html += `<td>${pattern.access_count}</td>`;
                html += `<td>${this.renderHeatBadge(pattern.heat_score)}</td>`;
                html += '</tr>';
            });

            html += '</tbody></table>';
            container.innerHTML = html;
        } catch (error) {
            console.error('Failed to update hot patterns:', error);
            container.innerHTML = '<div class="error">Failed to load hot patterns</div>';
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
     * Calculate confidence bar width
     */
    calculateConfidenceWidth(reward) {
        return Math.min(100, Math.max(0, reward * 100));
    }

    /**
     * Get reward trend for arm (from history)
     */
    getRewardTrend(armIdx) {
        // Extract last 10 expected rewards for this arm from history
        return this.statsHistory
            .slice(-10)
            .map(stats => {
                const rewards = stats.thompson_sampling_stats?.expected_rewards || [];
                return rewards[armIdx] || 0;
            });
    }

    /**
     * Render sparkline (simplified Tufte-style)
     */
    renderSparkline(data) {
        if (!data || data.length === 0) return '—';

        const min = Math.min(...data);
        const max = Math.max(...data);
        const range = max - min || 1;

        // Simple ASCII sparkline
        const chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        return data.map(val => {
            const normalized = (val - min) / range;
            const idx = Math.floor(normalized * (chars.length - 1));
            return chars[idx];
        }).join('');
    }

    /**
     * Render heat badge
     */
    renderHeatBadge(heatScore) {
        let className = 'badge ';
        if (heatScore >= 0.8) className += 'danger';
        else if (heatScore >= 0.6) className += 'warning';
        else if (heatScore >= 0.4) className += 'info';
        else className += 'success';

        return `<span class="${className}">${heatScore.toFixed(2)}</span>`;
    }

    /**
     * Show error message
     */
    showError(message) {
        const container = document.getElementById('learning-error-container');
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
     * Show warning message
     */
    showWarning(message) {
        const container = document.getElementById('learning-warning-container');
        if (container) {
            container.innerHTML = `
                <div class="warning-box">
                    <strong>⚠ ${this.escapeHtml(message)}</strong>
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
window.LearningDashboard = LearningDashboard;
