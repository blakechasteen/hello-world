/**
 * Orchestrator Pipeline Visualizer
 * =================================
 * Real-time visualization of HoloLoom's 9-step weaving cycle.
 *
 * Features:
 * - Animated 9-step pipeline progression
 * - Stage timing and bottleneck detection
 * - Confidence trajectory tracking
 * - Cache effectiveness monitoring
 * - Stage waterfall visualization
 *
 * Philosophy: Framework → Elegance → Real-Time Visibility
 */

class OrchestratorVisualizer {
    constructor(apiBase = 'http://localhost:8000') {
        this.apiBase = apiBase;
        this.updateInterval = 1000; // 1 second for real-time feel
        this.intervalId = null;
        this.currentQuery = null;
        this.stageHistory = []; // Last 20 queries
        this.maxHistory = 20;

        // Phase 3.2: Historical stage timing for sparklines
        this.stageTrends = {}; // stage_name → [duration1, duration2, ...]
        this.maxTrendPoints = 20; // Keep last 20 data points per stage

        // 9-step weaving cycle stages
        this.stages = [
            { id: 1, name: 'Loom Command', description: 'Pattern Card Selection', color: '#3498db' },
            { id: 2, name: 'Chrono Trigger', description: 'Temporal Window Creation', color: '#9b59b6' },
            { id: 3, name: 'Yarn Graph', description: 'Memory Thread Selection', color: '#e74c3c' },
            { id: 4, name: 'Resonance Shed', description: 'Feature Extraction → DotPlasma', color: '#f39c12' },
            { id: 5, name: 'Warp Space', description: 'Continuous Manifold Tensioning', color: '#1abc9c' },
            { id: 6, name: 'Convergence Engine', description: 'Decision Collapse', color: '#34495e' },
            { id: 7, name: 'Tool Execution', description: 'Action with Results', color: '#16a085' },
            { id: 8, name: 'Spacetime Fabric', description: 'Provenance & Trace', color: '#2c3e50' },
            { id: 9, name: 'Reflection Buffer', description: 'Learning from Outcome', color: '#27ae60' }
        ];
    }

    /**
     * Initialize visualizer
     */
    async initialize() {
        console.log('Initializing Orchestrator Visualizer...');

        // Initial load
        await this.refreshPipelineStatus();

        // Start polling
        this.intervalId = setInterval(() => this.refreshPipelineStatus(), this.updateInterval);

        // Setup event listeners
        this.setupEventListeners();

        console.log('✓ Orchestrator Visualizer initialized');
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
        const refreshBtn = document.getElementById('orchestrator-refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshPipelineStatus());
        }
    }

    /**
     * Refresh pipeline status from API
     */
    async refreshPipelineStatus() {
        try {
            const response = await fetch(`${this.apiBase}/monitor/orchestrator`);

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const data = await response.json();
            this.updateVisualization(data);
        } catch (error) {
            console.error('Failed to refresh orchestrator status:', error);
            this.showError(error.message);
        }
    }

    /**
     * Update visualization with new data (Phase 3.2: Enhanced with trend tracking)
     */
    updateVisualization(data) {
        // Update status
        const statusEl = document.getElementById('orchestrator-status');
        if (statusEl) {
            statusEl.textContent = data.status || 'Unknown';
            statusEl.className = `badge ${data.status === 'active' ? 'success' : 'warning'}`;
        }

        // Update current query info
        if (data.current_query) {
            this.currentQuery = data.current_query;
            this.updateCurrentQuery(data.current_query);
        }

        // Phase 3.2: Update stage trends for sparklines
        if (data.stage_durations && Object.keys(data.stage_durations).length > 0) {
            this.updateStageTrends(data.stage_durations);
        }

        // Update stage visualization
        if (data.current_stage !== undefined) {
            this.updateStageProgress(data.current_stage, data.stage_durations);
        }

        // Update history
        if (data.recent_traces) {
            this.updateStageHistory(data.recent_traces);
        }

        // Update metrics
        this.updateMetrics(data.metrics || {});
    }

    /**
     * Update current query display
     */
    updateCurrentQuery(query) {
        const container = document.getElementById('current-query-container');
        if (!container) return;

        if (!query || !query.text) {
            container.innerHTML = '<div class="empty-state">No active query</div>';
            return;
        }

        const html = `
            <div class="current-query-card">
                <div class="query-text">${this.escapeHtml(this.truncate(query.text, 100))}</div>
                <div class="query-meta">
                    <span class="badge info">${query.mode || 'DIRECT'}</span>
                    <span style="margin-left: 0.5rem; font-size: 0.875rem;">
                        ${query.elapsed_ms ? Math.round(query.elapsed_ms) + 'ms' : 'In progress...'}
                    </span>
                </div>
            </div>
        `;
        container.innerHTML = html;
    }

    /**
     * Update stage progress visualization (Phase 3.2: Enhanced with bottleneck detection)
     */
    updateStageProgress(currentStage, stageDurations = {}) {
        const container = document.getElementById('stage-progress-container');
        if (!container) return;

        // Calculate total duration and detect bottlenecks
        const totalDuration = Object.values(stageDurations).reduce((sum, d) => sum + d, 0);
        const bottleneckThreshold = 0.4; // 40% of total time

        let html = '<div class="stage-pipeline">';

        this.stages.forEach(stage => {
            const isActive = stage.id === currentStage;
            const isComplete = stage.id < currentStage;
            const duration = stageDurations[stage.name] || null;

            // Detect bottleneck (Phase 3.2)
            const isBottleneck = duration && totalDuration > 0 &&
                                (duration / totalDuration) > bottleneckThreshold;
            const percentage = duration && totalDuration > 0 ?
                              ((duration / totalDuration) * 100).toFixed(0) : null;

            let stageClass = 'stage-box';
            if (isActive) stageClass += ' stage-active';
            else if (isComplete) stageClass += ' stage-complete';
            if (isBottleneck) stageClass += ' stage-bottleneck';

            // Build tooltip with detailed metrics
            const tooltip = duration ?
                `${stage.name}: ${duration.toFixed(1)}ms${percentage ? ` (${percentage}% of total)` : ''}${isBottleneck ? ' ⚠️ BOTTLENECK' : ''}` :
                stage.description;

            html += `
                <div class="${stageClass}"
                     style="border-color: ${isBottleneck ? '#e74c3c' : stage.color};"
                     title="${this.escapeHtml(tooltip)}">
                    <div class="stage-number" style="background: ${isBottleneck ? '#e74c3c' : stage.color};">${stage.id}</div>
                    <div class="stage-info">
                        <div class="stage-name">
                            ${stage.name}
                            ${isBottleneck ? '<span class="bottleneck-icon">⚠️</span>' : ''}
                        </div>
                        <div class="stage-description">${stage.description}</div>
                        ${duration ? `
                            <div class="stage-duration">
                                ${duration.toFixed(1)}ms
                                ${percentage ? `<span class="stage-percentage">(${percentage}%)</span>` : ''}
                                ${this.stageTrends[stage.name] && this.stageTrends[stage.name].length >= 2 ?
                                    this.renderSparkline(this.stageTrends[stage.name], 40, 16, isBottleneck ? '#e74c3c' : stage.color)
                                    : ''}
                            </div>
                        ` : ''}
                    </div>
                    ${isActive ? '<div class="stage-pulse"></div>' : ''}
                </div>
            `;

            // Add arrow between stages (except after last)
            if (stage.id < this.stages.length) {
                html += '<div class="stage-arrow">→</div>';
            }
        });

        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * Update stage history (waterfall view) (Phase 3.2: Enhanced with bottleneck detection)
     */
    updateStageHistory(recentTraces) {
        const container = document.getElementById('stage-history-container');
        if (!container) return;

        if (!recentTraces || recentTraces.length === 0) {
            container.innerHTML = '<div class="empty-state">No recent traces</div>';
            return;
        }

        // Store in history
        this.stageHistory = recentTraces.slice(-this.maxHistory);

        // Render waterfall
        let html = '<div class="stage-waterfall">';

        const bottleneckThreshold = 0.4; // 40% of total time

        recentTraces.slice(-10).forEach((trace, idx) => {
            const totalDuration = trace.total_duration_ms || 0;
            const stages = trace.stages || {};

            // Detect bottleneck in this trace
            let bottleneckStage = null;
            let maxPercentage = 0;
            Object.entries(stages).forEach(([stageName, duration]) => {
                const percentage = (duration / totalDuration);
                if (percentage > bottleneckThreshold && percentage > maxPercentage) {
                    bottleneckStage = stageName;
                    maxPercentage = percentage;
                }
            });

            const hasBottleneck = bottleneckStage !== null;

            html += `
                <div class="waterfall-row ${hasBottleneck ? 'has-bottleneck' : ''}">
                    <div class="waterfall-label">
                        Query ${trace.query_id || idx + 1}
                        <span class="waterfall-total">${totalDuration.toFixed(0)}ms</span>
                        ${hasBottleneck ? '<span class="bottleneck-badge">⚠️</span>' : ''}
                    </div>
                    <div class="waterfall-bars">
            `;

            // Calculate cumulative positions for stacked bars
            let cumulativeTime = 0;
            this.stages.forEach(stage => {
                const duration = stages[stage.name] || 0;
                if (duration > 0) {
                    const percentage = (duration / totalDuration) * 100;
                    const leftOffset = (cumulativeTime / totalDuration) * 100;
                    const isBottleneck = stage.name === bottleneckStage;

                    // Build tooltip
                    const tooltip = `${stage.name}: ${duration.toFixed(1)}ms (${percentage.toFixed(0)}%)${isBottleneck ? ' ⚠️ BOTTLENECK' : ''}`;

                    html += `
                        <div class="waterfall-bar ${isBottleneck ? 'waterfall-bar-bottleneck' : ''}"
                             style="left: ${leftOffset}%; width: ${percentage}%; background: ${isBottleneck ? '#e74c3c' : stage.color};"
                             title="${this.escapeHtml(tooltip)}">
                             ${isBottleneck ? '<div class="bottleneck-stripe"></div>' : ''}
                        </div>
                    `;

                    cumulativeTime += duration;
                }
            });

            html += `
                    </div>
                </div>
            `;
        });

        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * Update metrics
     */
    updateMetrics(metrics) {
        this.updateMetric('avg-pipeline-latency',
            metrics.avg_latency_ms ? `${Math.round(metrics.avg_latency_ms)}ms` : '—');
        this.updateMetric('pipeline-throughput',
            metrics.queries_per_second ? `${metrics.queries_per_second.toFixed(2)}/s` : '—');
        this.updateMetric('bottleneck-stage',
            metrics.bottleneck_stage || 'None detected');

        // Update bottleneck badge
        const bottleneckBadge = document.getElementById('bottleneck-badge');
        if (bottleneckBadge) {
            if (metrics.bottleneck_stage) {
                bottleneckBadge.className = 'badge warning';
                bottleneckBadge.textContent = 'Detected';
            } else {
                bottleneckBadge.className = 'badge success';
                bottleneckBadge.textContent = 'None';
            }
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
     * Render inline sparkline (Phase 3.2)
     */
    renderSparkline(values, width = 40, height = 16, color = '#3498db') {
        if (!values || values.length < 2) return '';

        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min || 1; // Avoid division by zero

        // Generate SVG path
        const points = values.map((v, i) => {
            const x = (i / (values.length - 1)) * width;
            const y = height - ((v - min) / range) * height;
            return `${x.toFixed(1)},${y.toFixed(1)}`;
        }).join(' ');

        return `
            <svg width="${width}" height="${height}" class="sparkline" style="display: inline-block; vertical-align: middle; margin-left: 0.25rem;">
                <polyline
                    fill="none"
                    stroke="${color}"
                    stroke-width="1.5"
                    points="${points}"
                />
            </svg>
        `;
    }

    /**
     * Update stage trends (Phase 3.2)
     */
    updateStageTrends(stageDurations) {
        Object.entries(stageDurations).forEach(([stageName, duration]) => {
            if (!this.stageTrends[stageName]) {
                this.stageTrends[stageName] = [];
            }

            this.stageTrends[stageName].push(duration);

            // Keep only last N points
            if (this.stageTrends[stageName].length > this.maxTrendPoints) {
                this.stageTrends[stageName].shift();
            }
        });
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
    showError(message) {
        const container = document.getElementById('orchestrator-error');
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
window.OrchestratorVisualizer = OrchestratorVisualizer;
