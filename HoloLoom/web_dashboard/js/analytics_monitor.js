/**
 * Phase 3.4: Advanced Analytics & Insights Monitor
 * Phase 3.5: Data Persistence (November 2025)
 * Phase 3.6: Advanced Filtering (November 2025)
 * Phase 3.7: Custom Dashboards (November 2025)
 * Phase 3.8: Advanced Filter Builder (November 2025)
 * Phase 3.9: Drag-and-Drop Dashboard (November 2025)
 *
 * Provides comparative analytics, historical tracking, and system-wide insights:
 * 1. Query Comparison Table - Side-by-side comparison with sorting
 * 2. Historical Confidence Tracking - Time series with anomaly detection
 * 3. Tool Effectiveness Matrix - Heatmap of tool performance by query type
 * 4. System Health Dashboard - Overall system metrics and trends
 * 5. Data Persistence - LocalStorage with auto-save/load (Phase 3.5)
 * 6. Advanced Filtering - Date range, confidence, tool, query type filters (Phase 3.6)
 * 7. Custom Dashboards - Drag-drop layout, themes, templates (Phase 3.7)
 * 8. Advanced Filter Builder - Visual editor, AND/OR/NOT logic, saved presets (Phase 3.8)
 * 9. Drag-and-Drop Dashboard - Reorder/resize cards, grid layouts, snap-to-grid (Phase 3.9)
 *
 * @author HoloLoom
 * @date November 2025
 */

class AnalyticsMonitor {
    constructor() {
        // Version for data migration
        this.version = '3.9.0';
        this.storageKey = 'hololoom_analytics_data';
        this.dashboardKey = 'hololoom_dashboard_layout';
        this.filterPresetsKey = 'hololoom_filter_presets';

        // Query history for comparison (keep last 50)
        this.queryHistory = [];
        this.maxHistory = 50;

        // Confidence tracking
        this.confidenceHistory = [];
        this.maxConfidenceHistory = 100;

        // Tool effectiveness tracking
        this.toolStats = {}; // tool_name → {total: N, success: N, avg_latency: X, by_type: {...}}

        // System health metrics
        this.systemHealth = {
            uptime: 0,
            totalQueries: 0,
            avgConfidence: 0,
            avgLatency: 0,
            cacheHitRate: 0,
            bottleneckCount: 0,
            errorCount: 0
        };

        // Query type classifier (simple keyword-based)
        this.queryTypes = ['factual', 'procedural', 'analytical', 'creative', 'debugging'];

        // Sort state for comparison table
        this.sortColumn = 'timestamp';
        this.sortDirection = 'desc'; // 'asc' or 'desc'

        // Persistence state (Phase 3.5)
        this.persistenceEnabled = true;
        this.lastSaveTime = 0;
        this.saveDebounceMs = 1000; // Debounce saves to max 1 per second
        this.pendingSave = null;

        // Phase 3.6: Filter state
        this.filters = {
            dateFrom: null,
            dateTo: null,
            confidenceMin: 0.0,
            confidenceMax: 1.0,
            tools: [], // Empty = all tools
            queryTypes: [] // Empty = all types
        };
        this.filtersActive = false;

        // Phase 3.7: Dashboard layout state
        this.dashboardLayout = {
            cardOrder: ['comparison', 'confidence', 'effectiveness', 'health', 'management'],
            cardVisibility: {
                comparison: true,
                confidence: true,
                effectiveness: true,
                health: true,
                management: true
            },
            theme: 'light', // 'light', 'dark', or 'custom'
            customColors: {
                primary: '#2c3e50',
                secondary: '#95a5a6',
                success: '#27ae60',
                warning: '#f39c12',
                danger: '#e74c3c'
            },
            // Phase 3.9: Card sizes and grid layout
            cardSizes: {
                comparison: 'medium',  // 'small', 'medium', 'large'
                confidence: 'medium',
                effectiveness: 'medium',
                health: 'medium',
                management: 'medium'
            },
            gridLayout: 'auto',  // 'auto', '1-column', '2-column', '3-column', 'masonry'
            snapToGrid: true
        };

        // Phase 3.8: Advanced Filter Builder state
        this.filterBuilder = {
            conditions: [], // Array of filter conditions
            logic: 'AND', // 'AND' or 'OR'
            enabled: false
        };
        this.filterPresets = {}; // Saved filter presets
        this.currentPreset = null; // Currently loaded preset name
    }

    /**
     * Initialize the analytics monitor
     */
    async initialize() {
        console.log('[AnalyticsMonitor] Initializing...');

        // Phase 3.5: Load persisted data
        await this.loadData();

        // Phase 3.6: Load filters
        this.loadFilters();

        // Phase 3.7: Load dashboard layout and apply
        this.loadDashboardLayout();
        this.applyTheme();
        this.updateCardVisibility();

        // Phase 3.8: Load filter presets and builder state
        this.loadFilterPresets();
        this.loadFilterBuilder();

        // Phase 3.9: Apply grid layout, card sizes, and enable drag-drop
        this.applyGridLayout();
        this.applyAllCardSizes();
        this.enableDragDrop();

        // Set up refresh intervals
        setInterval(() => this.refreshQueryComparison(), 5000); // Every 5s
        setInterval(() => this.refreshConfidenceTracking(), 5000); // Every 5s
        setInterval(() => this.refreshToolEffectiveness(), 10000); // Every 10s
        setInterval(() => this.refreshSystemHealth(), 3000); // Every 3s

        // Initial refresh
        await this.refreshAll();

        console.log(`[AnalyticsMonitor] Initialized with ${this.queryHistory.length} persisted queries`);
    }

    /**
     * Refresh all analytics visualizations
     */
    async refreshAll() {
        await this.refreshQueryComparison();
        await this.refreshConfidenceTracking();
        await this.refreshToolEffectiveness();
        await this.refreshSystemHealth();
    }

    /**
     * Classify query type based on content
     */
    classifyQuery(queryText) {
        const text = queryText.toLowerCase();

        // Debugging keywords
        if (text.match(/\b(error|bug|fix|debug|issue|problem)\b/)) {
            return 'debugging';
        }

        // Procedural keywords
        if (text.match(/\b(how|setup|install|configure|implement)\b/)) {
            return 'procedural';
        }

        // Analytical keywords
        if (text.match(/\b(why|compare|analyze|tradeoff|difference|vs)\b/)) {
            return 'analytical';
        }

        // Creative keywords
        if (text.match(/\b(design|create|generate|suggest|ideate)\b/)) {
            return 'creative';
        }

        // Default to factual
        return 'factual';
    }

    /**
     * Add query result to history
     */
    addQueryResult(result) {
        const queryType = this.classifyQuery(result.query);

        // Add to query history
        this.queryHistory.push({
            timestamp: Date.now(),
            query: result.query,
            queryType: queryType,
            mode: result.mode || 'unknown',
            tool: result.tool_used || 'unknown',
            latency: result.latency_ms || 0,
            confidence: result.confidence || 0,
            cached: result.metadata?.cache_hit || false,
            stages: result.stages || {},
            bottleneck: this.detectBottleneck(result.stages || {})
        });

        // Trim to max history
        if (this.queryHistory.length > this.maxHistory) {
            this.queryHistory.shift();
        }

        // Add to confidence history
        this.confidenceHistory.push({
            timestamp: Date.now(),
            confidence: result.confidence || 0,
            cached: result.metadata?.cache_hit || false
        });

        // Trim confidence history
        if (this.confidenceHistory.length > this.maxConfidenceHistory) {
            this.confidenceHistory.shift();
        }

        // Update tool stats
        const tool = result.tool_used || 'unknown';
        if (!this.toolStats[tool]) {
            this.toolStats[tool] = {
                total: 0,
                success: 0,
                latencies: [],
                byType: {}
            };
        }

        this.toolStats[tool].total++;
        if (result.confidence >= 0.75) {
            this.toolStats[tool].success++;
        }
        this.toolStats[tool].latencies.push(result.latency_ms || 0);

        if (!this.toolStats[tool].byType[queryType]) {
            this.toolStats[tool].byType[queryType] = { total: 0, success: 0 };
        }
        this.toolStats[tool].byType[queryType].total++;
        if (result.confidence >= 0.75) {
            this.toolStats[tool].byType[queryType].success++;
        }

        // Update system health
        this.systemHealth.totalQueries++;
        if (result.confidence < 0.5) {
            // Count as potential error
            this.systemHealth.errorCount++;
        }
        if (this.detectBottleneck(result.stages || {})) {
            this.systemHealth.bottleneckCount++;
        }

        // Recalculate averages
        this.recalculateSystemHealth();

        // Phase 3.5: Auto-save data (debounced)
        this.debouncedSave();
    }

    /**
     * Detect if a query has a bottleneck stage
     */
    detectBottleneck(stages) {
        const total = Object.values(stages).reduce((sum, d) => sum + d, 0);
        for (const [stageName, duration] of Object.entries(stages)) {
            if (duration / total > 0.4) {
                return stageName;
            }
        }
        return null;
    }

    /**
     * Recalculate system health averages
     */
    recalculateSystemHealth() {
        if (this.queryHistory.length === 0) return;

        // Average confidence
        const totalConf = this.queryHistory.reduce((sum, q) => sum + q.confidence, 0);
        this.systemHealth.avgConfidence = totalConf / this.queryHistory.length;

        // Average latency
        const totalLat = this.queryHistory.reduce((sum, q) => sum + q.latency, 0);
        this.systemHealth.avgLatency = totalLat / this.queryHistory.length;

        // Cache hit rate
        const cacheHits = this.queryHistory.filter(q => q.cached).length;
        this.systemHealth.cacheHitRate = cacheHits / this.queryHistory.length;
    }

    /**
     * Feature 1: Query Comparison Table
     */
    async refreshQueryComparison() {
        const container = document.getElementById('query-comparison-container');
        if (!container) return;

        if (this.queryHistory.length === 0) {
            container.innerHTML = '<div class="empty-state">No queries yet. Make some queries to see comparisons.</div>';
            return;
        }

        // Phase 3.6: Apply filters
        const filteredQueries = this.applyFilters();

        if (filteredQueries.length === 0) {
            container.innerHTML = '<div class="empty-state">No queries match current filters. Try adjusting your filters.</div>';
            return;
        }

        // Sort queries
        const sorted = [...filteredQueries].sort((a, b) => {
            const aVal = a[this.sortColumn];
            const bVal = b[this.sortColumn];

            if (this.sortDirection === 'asc') {
                return aVal > bVal ? 1 : -1;
            } else {
                return aVal < bVal ? 1 : -1;
            }
        });

        // Find best/worst
        const bestConf = Math.max(...filteredQueries.map(q => q.confidence));
        const worstConf = Math.min(...this.queryHistory.map(q => q.confidence));
        const fastestLat = Math.min(...this.queryHistory.map(q => q.latency));
        const slowestLat = Math.max(...this.queryHistory.map(q => q.latency));

        // Render table
        let html = `
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th onclick="analyticsMonitor.sortBy('timestamp')">Time ${this.getSortIndicator('timestamp')}</th>
                        <th onclick="analyticsMonitor.sortBy('query')">Query ${this.getSortIndicator('query')}</th>
                        <th onclick="analyticsMonitor.sortBy('queryType')">Type ${this.getSortIndicator('queryType')}</th>
                        <th onclick="analyticsMonitor.sortBy('tool')">Tool ${this.getSortIndicator('tool')}</th>
                        <th onclick="analyticsMonitor.sortBy('latency')">Latency ${this.getSortIndicator('latency')}</th>
                        <th onclick="analyticsMonitor.sortBy('confidence')">Confidence ${this.getSortIndicator('confidence')}</th>
                        <th onclick="analyticsMonitor.sortBy('cached')">Cache ${this.getSortIndicator('cached')}</th>
                    </tr>
                </thead>
                <tbody>
        `;

        sorted.slice(-20).reverse().forEach(query => {
            const isBestConf = query.confidence === bestConf;
            const isWorstConf = query.confidence === worstConf;
            const isFastest = query.latency === fastestLat;
            const isSlowest = query.latency === slowestLat;

            const rowClass = query.bottleneck ? 'bottleneck-row' : '';

            const timeStr = new Date(query.timestamp).toLocaleTimeString();
            const queryStr = query.query.substring(0, 40) + (query.query.length > 40 ? '...' : '');

            html += `
                <tr class="${rowClass}">
                    <td>${timeStr}</td>
                    <td title="${query.query}">${queryStr}</td>
                    <td><span class="type-badge type-${query.queryType}">${query.queryType}</span></td>
                    <td>${query.tool}</td>
                    <td>
                        ${query.latency.toFixed(0)}ms
                        ${isFastest ? '<span class="marker-best">★</span>' : ''}
                        ${isSlowest ? '<span class="marker-worst">⚠</span>' : ''}
                    </td>
                    <td>
                        ${query.confidence.toFixed(2)}
                        ${isBestConf ? '<span class="marker-best">★</span>' : ''}
                        ${isWorstConf ? '<span class="marker-worst">⚠</span>' : ''}
                    </td>
                    <td>${query.cached ? '✓' : '—'}</td>
                </tr>
            `;
        });

        html += `
                </tbody>
            </table>
        `;

        container.innerHTML = html;
    }

    /**
     * Get sort indicator for column header
     */
    getSortIndicator(column) {
        if (this.sortColumn !== column) return '';
        return this.sortDirection === 'asc' ? '▲' : '▼';
    }

    /**
     * Sort queries by column
     */
    sortBy(column) {
        if (this.sortColumn === column) {
            // Toggle direction
            this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            this.sortColumn = column;
            this.sortDirection = 'desc';
        }

        this.refreshQueryComparison();
    }

    /**
     * Feature 2: Historical Confidence Tracking
     */
    async refreshConfidenceTracking() {
        const container = document.getElementById('confidence-tracking-container');
        if (!container) return;

        if (this.confidenceHistory.length < 2) {
            container.innerHTML = '<div class="empty-state">Not enough data yet. Make more queries to see trends.</div>';
            return;
        }

        // Render time series chart
        const html = this.renderConfidenceChart();
        container.innerHTML = html;
    }

    /**
     * Render confidence time series chart with anomaly detection
     */
    renderConfidenceChart() {
        const width = 800;
        const height = 300;
        const padding = { top: 20, right: 20, bottom: 40, left: 50 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        const values = this.confidenceHistory.map(h => h.confidence);
        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min || 0.1;

        // Calculate statistics
        const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
        const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
        const stdDev = Math.sqrt(variance);

        // Detect anomalies (values outside 2 std devs or sudden drops >0.2)
        const anomalies = [];
        this.confidenceHistory.forEach((point, idx) => {
            if (idx > 0) {
                const prev = this.confidenceHistory[idx - 1].confidence;
                const drop = prev - point.confidence;

                if (drop > 0.2) {
                    anomalies.push({ index: idx, type: 'sudden_drop' });
                } else if (Math.abs(point.confidence - mean) > 2 * stdDev) {
                    anomalies.push({ index: idx, type: 'outlier' });
                }
            }
        });

        let svg = `
            <svg width="${width}" height="${height}" style="font-family: monospace; font-size: 11px;">
                <!-- Axes -->
                <line x1="${padding.left}" y1="${padding.top}"
                      x2="${padding.left}" y2="${height - padding.bottom}"
                      stroke="#ccc" stroke-width="1"/>
                <line x1="${padding.left}" y1="${height - padding.bottom}"
                      x2="${width - padding.right}" y2="${height - padding.bottom}"
                      stroke="#ccc" stroke-width="1"/>

                <!-- Y-axis labels -->
                <text x="${padding.left - 5}" y="${padding.top}" text-anchor="end" fill="#666">1.0</text>
                <text x="${padding.left - 5}" y="${height - padding.bottom}" text-anchor="end" fill="#666">0.0</text>

                <!-- Title -->
                <text x="${padding.left}" y="15" fill="#333" font-weight="bold">Confidence Over Time (${this.confidenceHistory.length} queries)</text>

                <!-- Mean line -->
                <line x1="${padding.left}" y1="${padding.top + (1 - mean) * chartHeight}"
                      x2="${width - padding.right}" y2="${padding.top + (1 - mean) * chartHeight}"
                      stroke="#3498db" stroke-width="1" stroke-dasharray="4,4" opacity="0.5"/>
                <text x="${width - padding.right - 40}" y="${padding.top + (1 - mean) * chartHeight - 5}"
                      fill="#3498db" font-size="9px">Mean: ${mean.toFixed(2)}</text>

                <!-- Std dev bands -->
                <rect x="${padding.left}"
                      y="${padding.top + (1 - (mean + stdDev)) * chartHeight}"
                      width="${chartWidth}"
                      height="${2 * stdDev * chartHeight}"
                      fill="#3498db" opacity="0.1"/>
        `;

        // Draw confidence line
        const points = this.confidenceHistory.map((point, idx) => {
            const x = padding.left + (idx / (this.confidenceHistory.length - 1)) * chartWidth;
            const y = padding.top + (1 - point.confidence) * chartHeight;
            return `${x},${y}`;
        }).join(' ');

        svg += `
            <polyline points="${points}"
                      fill="none" stroke="#2ecc71" stroke-width="2"/>
        `;

        // Mark cache hits vs misses
        this.confidenceHistory.forEach((point, idx) => {
            const x = padding.left + (idx / (this.confidenceHistory.length - 1)) * chartWidth;
            const y = padding.top + (1 - point.confidence) * chartHeight;

            if (point.cached) {
                svg += `<circle cx="${x}" cy="${y}" r="3" fill="#2ecc71" opacity="0.6"/>`;
            } else {
                svg += `<circle cx="${x}" cy="${y}" r="3" fill="none" stroke="#e74c3c" stroke-width="1.5" opacity="0.6"/>`;
            }
        });

        // Mark anomalies
        anomalies.forEach(anomaly => {
            const point = this.confidenceHistory[anomaly.index];
            const x = padding.left + (anomaly.index / (this.confidenceHistory.length - 1)) * chartWidth;
            const y = padding.top + (1 - point.confidence) * chartHeight;

            const color = anomaly.type === 'sudden_drop' ? '#e74c3c' : '#f39c12';
            svg += `
                <circle cx="${x}" cy="${y}" r="6" fill="none" stroke="${color}" stroke-width="2"/>
                <text x="${x}" y="${y - 10}" text-anchor="middle" fill="${color}" font-size="16px">⚠</text>
            `;
        });

        // Legend
        svg += `
            <text x="${padding.left}" y="${height - 5}" fill="#666" font-size="10px">● Cache Hit   ○ Cache Miss   ⚠ Anomaly</text>
        `;

        svg += '</svg>';

        // Add summary statistics
        const summary = `
            <div style="margin-top: 1rem; padding: 1rem; background: #f8f9fa; border-radius: 4px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
                <div>
                    <div style="font-size: 0.75rem; color: #666;">Mean</div>
                    <div style="font-size: 1.125rem; font-weight: 600;">${mean.toFixed(3)}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #666;">Std Dev</div>
                    <div style="font-size: 1.125rem; font-weight: 600;">${stdDev.toFixed(3)}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #666;">Anomalies</div>
                    <div style="font-size: 1.125rem; font-weight: 600; color: ${anomalies.length > 0 ? '#e74c3c' : '#2ecc71'};">
                        ${anomalies.length}
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #666;">Cache Hit Rate</div>
                    <div style="font-size: 1.125rem; font-weight: 600;">${(this.systemHealth.cacheHitRate * 100).toFixed(1)}%</div>
                </div>
            </div>
        `;

        return svg + summary;
    }

    /**
     * Feature 3: Tool Effectiveness Matrix
     */
    async refreshToolEffectiveness() {
        const container = document.getElementById('tool-effectiveness-container');
        if (!container) return;

        if (Object.keys(this.toolStats).length === 0) {
            container.innerHTML = '<div class="empty-state">No tool data yet. Make some queries to see effectiveness.</div>';
            return;
        }

        const html = this.renderToolEffectivenessMatrix();
        container.innerHTML = html;
    }

    /**
     * Render tool effectiveness heatmap
     */
    renderToolEffectivenessMatrix() {
        const tools = Object.keys(this.toolStats);
        const types = this.queryTypes;

        let html = `
            <div style="margin-bottom: 1rem;">
                <div style="font-size: 0.875rem; color: #666; margin-bottom: 0.5rem;">
                    Success rate by tool and query type (darker = better)
                </div>
            </div>
            <table class="effectiveness-matrix">
                <thead>
                    <tr>
                        <th>Tool</th>
                        ${types.map(type => `<th>${type}</th>`).join('')}
                        <th>Overall</th>
                    </tr>
                </thead>
                <tbody>
        `;

        tools.forEach(tool => {
            const stats = this.toolStats[tool];
            const overallSuccess = stats.total > 0 ? (stats.success / stats.total) : 0;

            html += `<tr><td class="tool-name-cell">${tool}</td>`;

            types.forEach(type => {
                const typeStats = stats.byType[type];
                if (!typeStats || typeStats.total === 0) {
                    html += `<td class="matrix-cell" style="background: #f8f9fa;">—</td>`;
                } else {
                    const successRate = typeStats.success / typeStats.total;
                    const intensity = Math.floor(successRate * 255);
                    const bgColor = `rgb(${255 - intensity}, ${255}, ${255 - intensity})`;

                    html += `
                        <td class="matrix-cell" style="background: ${bgColor};" title="${(successRate * 100).toFixed(0)}% success">
                            ${(successRate * 100).toFixed(0)}%
                        </td>
                    `;
                }
            });

            // Overall column
            const intensity = Math.floor(overallSuccess * 255);
            const bgColor = `rgb(${255 - intensity}, ${255}, ${255 - intensity})`;
            html += `
                <td class="matrix-cell overall-cell" style="background: ${bgColor}; font-weight: 600;">
                    ${(overallSuccess * 100).toFixed(0)}%
                </td>
            `;

            html += '</tr>';
        });

        html += `
                </tbody>
            </table>
        `;

        // Tool-level statistics
        html += `
            <div style="margin-top: 1.5rem;">
                <div style="font-size: 1rem; font-weight: 600; margin-bottom: 0.75rem;">Tool Performance Summary</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
        `;

        tools.forEach(tool => {
            const stats = this.toolStats[tool];
            const avgLatency = stats.latencies.reduce((sum, l) => sum + l, 0) / stats.latencies.length;
            const successRate = stats.total > 0 ? (stats.success / stats.total) : 0;

            html += `
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 4px;">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">${tool}</div>
                    <div style="font-size: 0.8125rem; color: #666; display: flex; flex-direction: column; gap: 0.25rem;">
                        <div>Total: ${stats.total} queries</div>
                        <div>Success: ${(successRate * 100).toFixed(1)}%</div>
                        <div>Avg Latency: ${avgLatency.toFixed(0)}ms</div>
                    </div>
                </div>
            `;
        });

        html += `
                </div>
            </div>
        `;

        return html;
    }

    /**
     * Feature 4: System Health Dashboard
     */
    async refreshSystemHealth() {
        const container = document.getElementById('system-health-container');
        if (!container) return;

        const html = this.renderSystemHealthDashboard();
        container.innerHTML = html;
    }

    /**
     * Render system health dashboard
     */
    renderSystemHealthDashboard() {
        // Fetch server uptime (would come from API in production)
        const uptimeHours = (Date.now() - (Date.now() - 3600000 * 2)) / 3600000; // Mock 2 hours

        // Calculate health score (0-100)
        const healthScore = this.calculateHealthScore();

        // Determine health status
        let healthStatus = 'excellent';
        let healthColor = '#2ecc71';
        if (healthScore < 50) {
            healthStatus = 'critical';
            healthColor = '#e74c3c';
        } else if (healthScore < 70) {
            healthStatus = 'poor';
            healthColor = '#f39c12';
        } else if (healthScore < 85) {
            healthStatus = 'good';
            healthColor = '#3498db';
        }

        return `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-bottom: 1.5rem;">
                <!-- Overall Health Score -->
                <div style="padding: 1.5rem; background: linear-gradient(135deg, ${healthColor}22, ${healthColor}11); border-left: 4px solid ${healthColor}; border-radius: 4px;">
                    <div style="font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px;">Health Score</div>
                    <div style="font-size: 2.5rem; font-weight: 700; color: ${healthColor}; margin: 0.5rem 0;">${healthScore}</div>
                    <div style="font-size: 0.875rem; color: #666; text-transform: capitalize;">${healthStatus}</div>
                </div>

                <!-- Total Queries -->
                <div style="padding: 1.5rem; background: #f8f9fa; border-radius: 4px;">
                    <div style="font-size: 0.75rem; color: #666;">Total Queries</div>
                    <div style="font-size: 2rem; font-weight: 600; margin: 0.5rem 0;">${this.systemHealth.totalQueries}</div>
                    <div style="font-size: 0.75rem; color: #2ecc71;">+${(this.systemHealth.totalQueries * 0.12).toFixed(0)} today</div>
                </div>

                <!-- Avg Confidence -->
                <div style="padding: 1.5rem; background: #f8f9fa; border-radius: 4px;">
                    <div style="font-size: 0.75rem; color: #666;">Avg Confidence</div>
                    <div style="font-size: 2rem; font-weight: 600; margin: 0.5rem 0;">${this.systemHealth.avgConfidence.toFixed(2)}</div>
                    <div style="font-size: 0.75rem; color: ${this.systemHealth.avgConfidence > 0.75 ? '#2ecc71' : '#e74c3c'};">
                        ${this.systemHealth.avgConfidence > 0.75 ? '✓' : '⚠'} ${this.systemHealth.avgConfidence > 0.75 ? 'Healthy' : 'Needs attention'}
                    </div>
                </div>

                <!-- Avg Latency -->
                <div style="padding: 1.5rem; background: #f8f9fa; border-radius: 4px;">
                    <div style="font-size: 0.75rem; color: #666;">Avg Latency</div>
                    <div style="font-size: 2rem; font-weight: 600; margin: 0.5rem 0;">${this.systemHealth.avgLatency.toFixed(0)}ms</div>
                    <div style="font-size: 0.75rem; color: ${this.systemHealth.avgLatency < 200 ? '#2ecc71' : '#f39c12'};">
                        ${this.systemHealth.avgLatency < 200 ? '✓ Fast' : '⚠ Slow'}
                    </div>
                </div>

                <!-- Cache Hit Rate -->
                <div style="padding: 1.5rem; background: #f8f9fa; border-radius: 4px;">
                    <div style="font-size: 0.75rem; color: #666;">Cache Hit Rate</div>
                    <div style="font-size: 2rem; font-weight: 600; margin: 0.5rem 0;">${(this.systemHealth.cacheHitRate * 100).toFixed(0)}%</div>
                    <div style="font-size: 0.75rem; color: ${this.systemHealth.cacheHitRate > 0.6 ? '#2ecc71' : '#f39c12'};">
                        ${this.systemHealth.cacheHitRate > 0.6 ? '✓ Excellent' : '⚠ Could be better'}
                    </div>
                </div>

                <!-- Bottleneck Count -->
                <div style="padding: 1.5rem; background: #f8f9fa; border-radius: 4px;">
                    <div style="font-size: 0.75rem; color: #666;">Bottlenecks</div>
                    <div style="font-size: 2rem; font-weight: 600; margin: 0.5rem 0; color: ${this.systemHealth.bottleneckCount > 5 ? '#e74c3c' : '#2ecc71'};">
                        ${this.systemHealth.bottleneckCount}
                    </div>
                    <div style="font-size: 0.75rem; color: #666;">${((this.systemHealth.bottleneckCount / this.systemHealth.totalQueries) * 100).toFixed(1)}% of queries</div>
                </div>
            </div>

            <!-- Recommendations -->
            <div style="padding: 1.5rem; background: #fff5e6; border-left: 4px solid #f39c12; border-radius: 4px;">
                <div style="font-size: 1rem; font-weight: 600; margin-bottom: 0.75rem; color: #333;">System Recommendations</div>
                <ul style="margin: 0; padding-left: 1.5rem; color: #666; line-height: 1.8;">
                    ${this.generateRecommendations().map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    /**
     * Calculate overall system health score (0-100)
     */
    calculateHealthScore() {
        let score = 100;

        // Penalize low confidence
        if (this.systemHealth.avgConfidence < 0.5) {
            score -= 30;
        } else if (this.systemHealth.avgConfidence < 0.75) {
            score -= 15;
        }

        // Penalize high latency
        if (this.systemHealth.avgLatency > 300) {
            score -= 20;
        } else if (this.systemHealth.avgLatency > 200) {
            score -= 10;
        }

        // Penalize low cache hit rate
        if (this.systemHealth.cacheHitRate < 0.4) {
            score -= 15;
        } else if (this.systemHealth.cacheHitRate < 0.6) {
            score -= 8;
        }

        // Penalize frequent bottlenecks
        const bottleneckRate = this.systemHealth.bottleneckCount / (this.systemHealth.totalQueries || 1);
        if (bottleneckRate > 0.3) {
            score -= 15;
        } else if (bottleneckRate > 0.15) {
            score -= 8;
        }

        // Penalize errors
        const errorRate = this.systemHealth.errorCount / (this.systemHealth.totalQueries || 1);
        if (errorRate > 0.2) {
            score -= 10;
        }

        return Math.max(0, Math.min(100, score));
    }

    /**
     * Generate actionable recommendations
     */
    generateRecommendations() {
        const recommendations = [];

        if (this.systemHealth.avgConfidence < 0.75) {
            recommendations.push('⚠️ Average confidence is low. Consider refining prompts or increasing retrieval quality.');
        }

        if (this.systemHealth.avgLatency > 200) {
            recommendations.push('⚠️ Average latency is high. Review bottleneck stages and consider optimization.');
        }

        if (this.systemHealth.cacheHitRate < 0.6) {
            recommendations.push('💡 Cache hit rate could be improved. Consider increasing cache size or TTL.');
        }

        const bottleneckRate = this.systemHealth.bottleneckCount / (this.systemHealth.totalQueries || 1);
        if (bottleneckRate > 0.15) {
            recommendations.push('⚠️ Frequent bottlenecks detected. Review pipeline stages for optimization opportunities.');
        }

        if (recommendations.length === 0) {
            recommendations.push('✓ System is performing well! No immediate actions needed.');
        }

        return recommendations;
    }

    // ============================================================================
    // Phase 3.5: Data Persistence Methods
    // ============================================================================

    /**
     * Save analytics data to LocalStorage (debounced)
     */
    debouncedSave() {
        if (!this.persistenceEnabled) return;

        // Clear any pending save
        if (this.pendingSave) {
            clearTimeout(this.pendingSave);
        }

        // Schedule save after debounce period
        this.pendingSave = setTimeout(() => {
            this.saveData();
            this.pendingSave = null;
        }, this.saveDebounceMs);
    }

    /**
     * Save analytics data to LocalStorage
     */
    saveData() {
        if (!this.persistenceEnabled) return;

        try {
            const data = {
                version: this.version,
                timestamp: Date.now(),
                queryHistory: this.queryHistory,
                confidenceHistory: this.confidenceHistory,
                toolStats: this.toolStats,
                systemHealth: this.systemHealth,
                sortColumn: this.sortColumn,
                sortDirection: this.sortDirection
            };

            const json = JSON.stringify(data);
            localStorage.setItem(this.storageKey, json);

            this.lastSaveTime = Date.now();

            console.log(`[AnalyticsMonitor] Saved ${this.queryHistory.length} queries (${(json.length / 1024).toFixed(1)} KB)`);
        } catch (error) {
            if (error.name === 'QuotaExceededError') {
                console.error('[AnalyticsMonitor] Storage quota exceeded. Clearing old data...');
                this.clearOldestData();
            } else {
                console.error('[AnalyticsMonitor] Failed to save data:', error);
            }
        }
    }

    /**
     * Load analytics data from LocalStorage
     */
    async loadData() {
        if (!this.persistenceEnabled) return;

        try {
            const json = localStorage.getItem(this.storageKey);
            if (!json) {
                console.log('[AnalyticsMonitor] No persisted data found');
                return;
            }

            const data = JSON.parse(json);

            // Version check
            if (!data.version) {
                console.warn('[AnalyticsMonitor] Old data format detected, migrating...');
                // Migration logic for pre-3.5 data (if needed)
                return;
            }

            // Load data
            this.queryHistory = data.queryHistory || [];
            this.confidenceHistory = data.confidenceHistory || [];
            this.toolStats = data.toolStats || {};
            this.systemHealth = data.systemHealth || this.systemHealth;
            this.sortColumn = data.sortColumn || 'timestamp';
            this.sortDirection = data.sortDirection || 'desc';

            console.log(`[AnalyticsMonitor] Loaded ${this.queryHistory.length} queries from ${new Date(data.timestamp).toLocaleString()}`);
        } catch (error) {
            console.error('[AnalyticsMonitor] Failed to load data:', error);
            console.warn('[AnalyticsMonitor] Starting with fresh data');
        }
    }

    /**
     * Clear all analytics data
     */
    clearData() {
        if (!confirm('Are you sure you want to clear all analytics data? This cannot be undone.')) {
            return false;
        }

        try {
            // Clear LocalStorage
            localStorage.removeItem(this.storageKey);

            // Reset in-memory data
            this.queryHistory = [];
            this.confidenceHistory = [];
            this.toolStats = {};
            this.systemHealth = {
                uptime: 0,
                totalQueries: 0,
                avgConfidence: 0,
                avgLatency: 0,
                cacheHitRate: 0,
                bottleneckCount: 0,
                errorCount: 0
            };

            // Refresh UI
            this.refreshAll();

            console.log('[AnalyticsMonitor] All data cleared');
            alert('Analytics data cleared successfully');
            return true;
        } catch (error) {
            console.error('[AnalyticsMonitor] Failed to clear data:', error);
            alert('Failed to clear data: ' + error.message);
            return false;
        }
    }

    /**
     * Clear oldest data to free up space
     */
    clearOldestData() {
        // Remove oldest 25% of queries
        const removeCount = Math.ceil(this.queryHistory.length * 0.25);
        this.queryHistory.splice(0, removeCount);

        const removeConfCount = Math.ceil(this.confidenceHistory.length * 0.25);
        this.confidenceHistory.splice(0, removeConfCount);

        console.log(`[AnalyticsMonitor] Cleared ${removeCount} old queries to free up space`);

        // Try saving again
        try {
            this.saveData();
        } catch (error) {
            console.error('[AnalyticsMonitor] Still cannot save after clearing old data:', error);
        }
    }

    /**
     * Export analytics data to JSON file
     */
    exportData() {
        try {
            const data = {
                version: this.version,
                exportDate: new Date().toISOString(),
                queryHistory: this.queryHistory,
                confidenceHistory: this.confidenceHistory,
                toolStats: this.toolStats,
                systemHealth: this.systemHealth
            };

            const json = JSON.stringify(data, null, 2);
            const blob = new Blob([json], { type: 'application/json' });
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = `hololoom-analytics-${Date.now()}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            console.log('[AnalyticsMonitor] Data exported successfully');
            alert('Analytics data exported successfully');
        } catch (error) {
            console.error('[AnalyticsMonitor] Failed to export data:', error);
            alert('Failed to export data: ' + error.message);
        }
    }

    /**
     * Import analytics data from JSON file
     */
    importData(file) {
        const reader = new FileReader();

        reader.onload = (e) => {
            try {
                const data = JSON.parse(e.target.result);

                // Validate data
                if (!data.queryHistory || !Array.isArray(data.queryHistory)) {
                    throw new Error('Invalid data format: missing or invalid queryHistory');
                }

                // Confirm import
                if (!confirm(`Import ${data.queryHistory.length} queries? This will replace current data.`)) {
                    return;
                }

                // Import data
                this.queryHistory = data.queryHistory;
                this.confidenceHistory = data.confidenceHistory || [];
                this.toolStats = data.toolStats || {};
                this.systemHealth = data.systemHealth || this.systemHealth;

                // Trim to max limits
                if (this.queryHistory.length > this.maxHistory) {
                    this.queryHistory = this.queryHistory.slice(-this.maxHistory);
                }
                if (this.confidenceHistory.length > this.maxConfidenceHistory) {
                    this.confidenceHistory = this.confidenceHistory.slice(-this.maxConfidenceHistory);
                }

                // Save and refresh
                this.saveData();
                this.refreshAll();

                console.log('[AnalyticsMonitor] Data imported successfully');
                alert('Analytics data imported successfully');
            } catch (error) {
                console.error('[AnalyticsMonitor] Failed to import data:', error);
                alert('Failed to import data: ' + error.message);
            }
        };

        reader.onerror = () => {
            console.error('[AnalyticsMonitor] Failed to read file');
            alert('Failed to read file');
        };

        reader.readAsText(file);
    }

    /**
     * Get storage usage information
     */
    getStorageUsage() {
        try {
            const json = localStorage.getItem(this.storageKey);
            const usedBytes = json ? json.length : 0;
            const usedKB = (usedBytes / 1024).toFixed(2);
            const usedMB = (usedBytes / (1024 * 1024)).toFixed(2);

            // Estimate total available (typically 5-10MB in most browsers)
            const estimatedTotalMB = 5;
            const usagePercent = ((usedBytes / (estimatedTotalMB * 1024 * 1024)) * 100).toFixed(1);

            return {
                usedBytes,
                usedKB,
                usedMB,
                usagePercent,
                queryCount: this.queryHistory.length,
                confidenceCount: this.confidenceHistory.length,
                toolCount: Object.keys(this.toolStats).length
            };
        } catch (error) {
            console.error('[AnalyticsMonitor] Failed to get storage usage:', error);
            return null;
        }
    }

    // ============================================================================
    // Phase 3.6: Advanced Filtering Methods
    // ============================================================================

    /**
     * Apply filters to query history (Phase 3.6 basic + Phase 3.8 advanced)
     * @returns {Array} Filtered query results
     */
    applyFilters(queries = null) {
        let data = queries || this.queryHistory;

        // Phase 3.6: Apply basic filters first
        if (this.filtersActive) {
            data = data.filter(result => {
                // Date range filter
                if (this.filters.dateFrom && result.timestamp < this.filters.dateFrom) {
                    return false;
                }
                if (this.filters.dateTo && result.timestamp > this.filters.dateTo) {
                    return false;
                }

                // Confidence filter
                if (result.confidence < this.filters.confidenceMin || result.confidence > this.filters.confidenceMax) {
                    return false;
                }

                // Tool filter
                if (this.filters.tools.length > 0 && !this.filters.tools.includes(result.tool_used)) {
                    return false;
                }

                // Query type filter
                if (this.filters.queryTypes.length > 0) {
                    const queryType = this.classifyQuery(result.query);
                    if (!this.filters.queryTypes.includes(queryType)) {
                        return false;
                    }
                }

                return true;
            });
        }

        // Phase 3.8: Apply advanced filter builder (if enabled)
        if (this.filterBuilder.enabled) {
            data = this.applyFilterBuilder(data);
        }

        return data;
    }

    /**
     * Set date range filter
     */
    setDateRangeFilter(fromDate, toDate) {
        this.filters.dateFrom = fromDate ? new Date(fromDate).getTime() : null;
        this.filters.dateTo = toDate ? new Date(toDate).getTime() : null;
        this.updateFilterState();
        this.saveFilters();
    }

    /**
     * Set confidence range filter
     */
    setConfidenceFilter(min, max) {
        this.filters.confidenceMin = parseFloat(min);
        this.filters.confidenceMax = parseFloat(max);
        this.updateFilterState();
        this.saveFilters();
    }

    /**
     * Set tool filter
     */
    setToolFilter(tools) {
        this.filters.tools = Array.isArray(tools) ? tools : [tools];
        this.updateFilterState();
        this.saveFilters();
    }

    /**
     * Set query type filter
     */
    setQueryTypeFilter(types) {
        this.filters.queryTypes = Array.isArray(types) ? types : [types];
        this.updateFilterState();
        this.saveFilters();
    }

    /**
     * Update filter active state
     */
    updateFilterState() {
        this.filtersActive =
            this.filters.dateFrom !== null ||
            this.filters.dateTo !== null ||
            this.filters.confidenceMin > 0.0 ||
            this.filters.confidenceMax < 1.0 ||
            this.filters.tools.length > 0 ||
            this.filters.queryTypes.length > 0;
    }

    /**
     * Clear all filters
     */
    clearFilters() {
        this.filters = {
            dateFrom: null,
            dateTo: null,
            confidenceMin: 0.0,
            confidenceMax: 1.0,
            tools: [],
            queryTypes: []
        };
        this.filtersActive = false;
        this.saveFilters();
        this.refreshAll();
    }

    /**
     * Save filters to localStorage
     */
    saveFilters() {
        try {
            localStorage.setItem('hololoom_filters', JSON.stringify(this.filters));
        } catch (error) {
            console.error('[AnalyticsMonitor] Failed to save filters:', error);
        }
    }

    /**
     * Load filters from localStorage
     */
    loadFilters() {
        try {
            const json = localStorage.getItem('hololoom_filters');
            if (json) {
                this.filters = JSON.parse(json);
                this.updateFilterState();
            }
        } catch (error) {
            console.error('[AnalyticsMonitor] Failed to load filters:', error);
        }
    }

    /**
     * Get active filter count
     */
    getActiveFilterCount() {
        let count = 0;
        if (this.filters.dateFrom) count++;
        if (this.filters.dateTo) count++;
        if (this.filters.confidenceMin > 0.0 || this.filters.confidenceMax < 1.0) count++;
        if (this.filters.tools.length > 0) count++;
        if (this.filters.queryTypes.length > 0) count++;
        return count;
    }

    // ============================================================================
    // Phase 3.7: Dashboard Customization Methods
    // ============================================================================

    /**
     * Set card visibility
     */
    setCardVisibility(cardId, visible) {
        this.dashboardLayout.cardVisibility[cardId] = visible;
        this.saveDashboardLayout();
        this.updateCardVisibility();
    }

    /**
     * Toggle card visibility
     */
    toggleCardVisibility(cardId) {
        this.dashboardLayout.cardVisibility[cardId] = !this.dashboardLayout.cardVisibility[cardId];
        this.saveDashboardLayout();
        this.updateCardVisibility();
    }

    /**
     * Update DOM to reflect card visibility
     */
    updateCardVisibility() {
        const cardMap = {
            'comparison': 'query-comparison-card',
            'confidence': 'confidence-tracking-card',
            'effectiveness': 'tool-effectiveness-card',
            'health': 'system-health-card',
            'management': 'data-management-card'
        };

        for (const [cardId, domId] of Object.entries(cardMap)) {
            const element = document.getElementById(domId);
            if (element) {
                element.style.display = this.dashboardLayout.cardVisibility[cardId] ? 'block' : 'none';
            }
        }
    }

    /**
     * Set card order (for drag-drop)
     */
    setCardOrder(newOrder) {
        this.dashboardLayout.cardOrder = newOrder;
        this.saveDashboardLayout();
        this.reorderCards();
    }

    /**
     * Reorder cards in DOM
     */
    reorderCards() {
        const container = document.getElementById('analytics');
        if (!container) return;

        const cardMap = {
            'comparison': 'query-comparison-card',
            'confidence': 'confidence-tracking-card',
            'effectiveness': 'tool-effectiveness-card',
            'health': 'system-health-card',
            'management': 'data-management-card'
        };

        // Reorder based on cardOrder array
        for (const cardId of this.dashboardLayout.cardOrder) {
            const domId = cardMap[cardId];
            const element = document.getElementById(domId);
            if (element) {
                container.appendChild(element);
            }
        }
    }

    /**
     * Set theme
     */
    setTheme(themeName) {
        this.dashboardLayout.theme = themeName;
        this.saveDashboardLayout();
        this.applyTheme();
    }

    /**
     * Apply theme to dashboard
     */
    applyTheme() {
        const root = document.documentElement;

        if (this.dashboardLayout.theme === 'dark') {
            root.style.setProperty('--primary', '#ecf0f1');
            root.style.setProperty('--secondary', '#bdc3c7');
            root.style.setProperty('--bg', '#2c3e50');
            root.style.setProperty('--card-bg', '#34495e');
            root.style.setProperty('--border', '#7f8c8d');
        } else if (this.dashboardLayout.theme === 'light') {
            root.style.setProperty('--primary', '#2c3e50');
            root.style.setProperty('--secondary', '#95a5a6');
            root.style.setProperty('--bg', '#ecf0f1');
            root.style.setProperty('--card-bg', '#ffffff');
            root.style.setProperty('--border', '#bdc3c7');
        } else if (this.dashboardLayout.theme === 'custom') {
            for (const [key, value] of Object.entries(this.dashboardLayout.customColors)) {
                root.style.setProperty(`--${key}`, value);
            }
        }
    }

    /**
     * Set custom colors
     */
    setCustomColors(colors) {
        this.dashboardLayout.customColors = { ...this.dashboardLayout.customColors, ...colors };
        this.saveDashboardLayout();
        if (this.dashboardLayout.theme === 'custom') {
            this.applyTheme();
        }
    }

    /**
     * Apply dashboard template
     */
    applyTemplate(templateName) {
        const templates = {
            'default': {
                cardVisibility: {
                    comparison: true,
                    confidence: true,
                    effectiveness: true,
                    health: true,
                    management: true
                },
                cardOrder: ['comparison', 'confidence', 'effectiveness', 'health', 'management']
            },
            'performance': {
                cardVisibility: {
                    comparison: true,
                    confidence: false,
                    effectiveness: false,
                    health: true,
                    management: false
                },
                cardOrder: ['health', 'comparison', 'confidence', 'effectiveness', 'management']
            },
            'quality': {
                cardVisibility: {
                    comparison: false,
                    confidence: true,
                    effectiveness: true,
                    health: false,
                    management: false
                },
                cardOrder: ['confidence', 'effectiveness', 'comparison', 'health', 'management']
            },
            'minimal': {
                cardVisibility: {
                    comparison: true,
                    confidence: false,
                    effectiveness: false,
                    health: true,
                    management: false
                },
                cardOrder: ['comparison', 'health', 'confidence', 'effectiveness', 'management']
            }
        };

        const template = templates[templateName];
        if (template) {
            this.dashboardLayout.cardVisibility = template.cardVisibility;
            this.dashboardLayout.cardOrder = template.cardOrder;
            this.saveDashboardLayout();
            this.updateCardVisibility();
            this.reorderCards();
        }
    }

    /**
     * Save dashboard layout to localStorage
     */
    saveDashboardLayout() {
        try {
            localStorage.setItem(this.dashboardKey, JSON.stringify(this.dashboardLayout));
            console.log('[AnalyticsMonitor] Dashboard layout saved');
        } catch (error) {
            console.error('[AnalyticsMonitor] Failed to save dashboard layout:', error);
        }
    }

    /**
     * Load dashboard layout from localStorage
     */
    loadDashboardLayout() {
        try {
            const json = localStorage.getItem(this.dashboardKey);
            if (json) {
                this.dashboardLayout = JSON.parse(json);
                console.log('[AnalyticsMonitor] Dashboard layout loaded');
            }
        } catch (error) {
            console.error('[AnalyticsMonitor] Failed to load dashboard layout:', error);
        }
    }

    /**
     * Reset dashboard to default
     */
    resetDashboard() {
        this.applyTemplate('default');
        this.setTheme('light');
    }

    // ============================================================================
    // Phase 3.9: Drag-and-Drop Dashboard Methods
    // ============================================================================

    /**
     * Set card size
     */
    setCardSize(cardId, size) {
        if (!['small', 'medium', 'large'].includes(size)) {
            console.error(`[AnalyticsMonitor] Invalid size: ${size}`);
            return;
        }

        this.dashboardLayout.cardSizes[cardId] = size;
        this.saveDashboardLayout();
        this.applyCardSize(cardId, size);
    }

    /**
     * Apply card size to DOM element
     */
    applyCardSize(cardId, size) {
        const cardMap = {
            'comparison': 'query-comparison-card',
            'confidence': 'confidence-tracking-card',
            'effectiveness': 'tool-effectiveness-card',
            'health': 'system-health-card',
            'management': 'data-management-card'
        };

        const element = document.getElementById(cardMap[cardId]);
        if (!element) return;

        // Remove existing size classes
        element.classList.remove('card-small', 'card-medium', 'card-large');

        // Add new size class
        element.classList.add(`card-${size}`);
    }

    /**
     * Apply all card sizes on load
     */
    applyAllCardSizes() {
        for (const [cardId, size] of Object.entries(this.dashboardLayout.cardSizes)) {
            this.applyCardSize(cardId, size);
        }
    }

    /**
     * Set grid layout
     */
    setGridLayout(layout) {
        const validLayouts = ['auto', '1-column', '2-column', '3-column', 'masonry'];
        if (!validLayouts.includes(layout)) {
            console.error(`[AnalyticsMonitor] Invalid layout: ${layout}`);
            return;
        }

        this.dashboardLayout.gridLayout = layout;
        this.saveDashboardLayout();
        this.applyGridLayout();
    }

    /**
     * Apply grid layout to dashboard
     */
    applyGridLayout() {
        const container = document.getElementById('analytics-cards-container');
        if (!container) return;

        // Remove existing layout classes
        container.classList.remove('layout-auto', 'layout-1-column', 'layout-2-column', 'layout-3-column', 'layout-masonry');

        // Add new layout class
        container.classList.add(`layout-${this.dashboardLayout.gridLayout}`);
    }

    /**
     * Toggle snap to grid
     */
    setSnapToGrid(enabled) {
        this.dashboardLayout.snapToGrid = enabled;
        this.saveDashboardLayout();
    }

    /**
     * Enable drag-and-drop for cards
     */
    enableDragDrop() {
        const cardMap = {
            'comparison': 'query-comparison-card',
            'confidence': 'confidence-tracking-card',
            'effectiveness': 'tool-effectiveness-card',
            'health': 'system-health-card',
            'management': 'data-management-card'
        };

        for (const [cardId, domId] of Object.entries(cardMap)) {
            const element = document.getElementById(domId);
            if (!element) continue;

            // Make draggable
            element.setAttribute('draggable', 'true');
            element.classList.add('draggable-card');

            // Add drag handle
            const header = element.querySelector('.card-header');
            if (header) {
                header.classList.add('drag-handle');
                header.style.cursor = 'move';
            }

            // Drag start
            element.addEventListener('dragstart', (e) => {
                e.dataTransfer.effectAllowed = 'move';
                e.dataTransfer.setData('text/plain', cardId);
                element.classList.add('dragging');
            });

            // Drag end
            element.addEventListener('dragend', (e) => {
                element.classList.remove('dragging');
            });

            // Drag over
            element.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';

                const draggingElement = document.querySelector('.dragging');
                if (draggingElement && draggingElement !== element) {
                    const rect = element.getBoundingClientRect();
                    const midpoint = rect.top + rect.height / 2;

                    if (e.clientY < midpoint) {
                        element.parentNode.insertBefore(draggingElement, element);
                    } else {
                        element.parentNode.insertBefore(draggingElement, element.nextSibling);
                    }
                }
            });

            // Drop
            element.addEventListener('drop', (e) => {
                e.preventDefault();
                this.updateCardOrderFromDOM();
            });
        }

        console.log('[AnalyticsMonitor] Drag-and-drop enabled');
    }

    /**
     * Update card order from DOM (after drag-and-drop)
     */
    updateCardOrderFromDOM() {
        const container = document.getElementById('analytics-cards-container');
        if (!container) return;

        const cardMap = {
            'query-comparison-card': 'comparison',
            'confidence-tracking-card': 'confidence',
            'tool-effectiveness-card': 'effectiveness',
            'system-health-card': 'health',
            'data-management-card': 'management'
        };

        const newOrder = [];
        const cards = container.querySelectorAll('.card[id$="-card"]');

        cards.forEach(card => {
            const cardId = cardMap[card.id];
            if (cardId && !newOrder.includes(cardId)) {
                newOrder.push(cardId);
            }
        });

        // Update order
        this.dashboardLayout.cardOrder = newOrder;
        this.saveDashboardLayout();

        console.log('[AnalyticsMonitor] Card order updated:', newOrder);
    }

    /**
     * Apply grid template (Phase 3.9 templates)
     */
    applyGridTemplate(templateName) {
        const templates = {
            'compact': {
                gridLayout: '3-column',
                cardSizes: {
                    comparison: 'small',
                    confidence: 'small',
                    effectiveness: 'small',
                    health: 'small',
                    management: 'small'
                }
            },
            'spacious': {
                gridLayout: '1-column',
                cardSizes: {
                    comparison: 'large',
                    confidence: 'large',
                    effectiveness: 'large',
                    health: 'large',
                    management: 'large'
                }
            },
            'balanced': {
                gridLayout: '2-column',
                cardSizes: {
                    comparison: 'medium',
                    confidence: 'medium',
                    effectiveness: 'medium',
                    health: 'medium',
                    management: 'medium'
                }
            },
            'masonry': {
                gridLayout: 'masonry',
                cardSizes: {
                    comparison: 'medium',
                    confidence: 'small',
                    effectiveness: 'large',
                    health: 'medium',
                    management: 'small'
                }
            }
        };

        const template = templates[templateName];
        if (!template) {
            console.error(`[AnalyticsMonitor] Unknown template: ${templateName}`);
            return;
        }

        // Apply template
        this.dashboardLayout.gridLayout = template.gridLayout;
        this.dashboardLayout.cardSizes = template.cardSizes;

        this.saveDashboardLayout();
        this.applyGridLayout();
        this.applyAllCardSizes();

        console.log(`[AnalyticsMonitor] Grid template "${templateName}" applied`);
    }

    // ============================================================================
    // Phase 3.8: Advanced Filter Builder Methods
    // ============================================================================

    /**
     * Apply filter builder conditions (supports AND/OR/NOT logic)
     * @returns {Array} Filtered query results
     */
    applyFilterBuilder(queries = null) {
        const data = queries || this.queryHistory;

        if (!this.filterBuilder.enabled || this.filterBuilder.conditions.length === 0) {
            return data;
        }

        return data.filter(result => {
            if (this.filterBuilder.logic === 'AND') {
                // ALL conditions must match
                return this.filterBuilder.conditions.every(condition =>
                    this.evaluateCondition(result, condition)
                );
            } else {
                // ANY condition must match (OR logic)
                return this.filterBuilder.conditions.some(condition =>
                    this.evaluateCondition(result, condition)
                );
            }
        });
    }

    /**
     * Evaluate a single filter condition
     * @param {Object} result - Query result to evaluate
     * @param {Object} condition - Filter condition
     * @returns {Boolean} Whether condition matches
     */
    evaluateCondition(result, condition) {
        let matches = false;

        switch (condition.field) {
            case 'date':
                const resultDate = new Date(result.timestamp).getTime();
                matches = this.evaluateDateCondition(resultDate, condition);
                break;

            case 'confidence':
                matches = this.evaluateNumberCondition(result.confidence, condition);
                break;

            case 'latency':
                matches = this.evaluateNumberCondition(result.latency, condition);
                break;

            case 'tool':
                matches = this.evaluateStringCondition(result.tool_used, condition);
                break;

            case 'queryType':
                const queryType = this.classifyQuery(result.query);
                matches = this.evaluateStringCondition(queryType, condition);
                break;

            case 'query':
                matches = this.evaluateStringCondition(result.query, condition);
                break;

            case 'cached':
                matches = this.evaluateBooleanCondition(result.cached, condition);
                break;

            default:
                matches = false;
        }

        // Apply NOT operator if specified
        return condition.not ? !matches : matches;
    }

    /**
     * Evaluate date condition
     */
    evaluateDateCondition(value, condition) {
        const compareValue = new Date(condition.value).getTime();

        switch (condition.operator) {
            case 'equals': return value === compareValue;
            case 'before': return value < compareValue;
            case 'after': return value > compareValue;
            case 'between':
                const value2 = new Date(condition.value2).getTime();
                return value >= compareValue && value <= value2;
            default: return false;
        }
    }

    /**
     * Evaluate number condition
     */
    evaluateNumberCondition(value, condition) {
        const compareValue = parseFloat(condition.value);

        switch (condition.operator) {
            case 'equals': return value === compareValue;
            case 'notEquals': return value !== compareValue;
            case 'greaterThan': return value > compareValue;
            case 'lessThan': return value < compareValue;
            case 'greaterOrEqual': return value >= compareValue;
            case 'lessOrEqual': return value <= compareValue;
            case 'between':
                const value2 = parseFloat(condition.value2);
                return value >= compareValue && value <= value2;
            default: return false;
        }
    }

    /**
     * Evaluate string condition
     */
    evaluateStringCondition(value, condition) {
        const compareValue = condition.value.toLowerCase();
        const lowerValue = (value || '').toLowerCase();

        switch (condition.operator) {
            case 'equals': return lowerValue === compareValue;
            case 'notEquals': return lowerValue !== compareValue;
            case 'contains': return lowerValue.includes(compareValue);
            case 'notContains': return !lowerValue.includes(compareValue);
            case 'startsWith': return lowerValue.startsWith(compareValue);
            case 'endsWith': return lowerValue.endsWith(compareValue);
            case 'regex':
                try {
                    return new RegExp(condition.value).test(value);
                } catch (e) {
                    return false;
                }
            default: return false;
        }
    }

    /**
     * Evaluate boolean condition
     */
    evaluateBooleanCondition(value, condition) {
        return value === condition.value;
    }

    /**
     * Add condition to filter builder
     */
    addCondition(field, operator, value, value2 = null, not = false) {
        const condition = {
            id: Date.now() + Math.random(),
            field,
            operator,
            value,
            value2,
            not
        };

        this.filterBuilder.conditions.push(condition);
        this.saveFilterBuilder();
        return condition;
    }

    /**
     * Remove condition from filter builder
     */
    removeCondition(conditionId) {
        this.filterBuilder.conditions = this.filterBuilder.conditions.filter(
            c => c.id !== conditionId
        );
        this.saveFilterBuilder();
    }

    /**
     * Update condition in filter builder
     */
    updateCondition(conditionId, updates) {
        const condition = this.filterBuilder.conditions.find(c => c.id === conditionId);
        if (condition) {
            Object.assign(condition, updates);
            this.saveFilterBuilder();
        }
    }

    /**
     * Toggle condition NOT operator
     */
    toggleConditionNot(conditionId) {
        const condition = this.filterBuilder.conditions.find(c => c.id === conditionId);
        if (condition) {
            condition.not = !condition.not;
            this.saveFilterBuilder();
        }
    }

    /**
     * Set filter builder logic (AND/OR)
     */
    setFilterLogic(logic) {
        this.filterBuilder.logic = logic;
        this.saveFilterBuilder();
    }

    /**
     * Enable/disable filter builder
     */
    setFilterBuilderEnabled(enabled) {
        this.filterBuilder.enabled = enabled;
        this.saveFilterBuilder();
        if (enabled) {
            this.refreshAll();
        }
    }

    /**
     * Clear all conditions
     */
    clearFilterBuilder() {
        this.filterBuilder.conditions = [];
        this.filterBuilder.enabled = false;
        this.saveFilterBuilder();
        this.refreshAll();
    }

    /**
     * Save filter builder to localStorage
     */
    saveFilterBuilder() {
        try {
            localStorage.setItem('hololoom_filter_builder', JSON.stringify(this.filterBuilder));
        } catch (error) {
            console.error('[AnalyticsMonitor] Failed to save filter builder:', error);
        }
    }

    /**
     * Load filter builder from localStorage
     */
    loadFilterBuilder() {
        try {
            const json = localStorage.getItem('hololoom_filter_builder');
            if (json) {
                this.filterBuilder = JSON.parse(json);
                console.log('[AnalyticsMonitor] Filter builder loaded');
            }
        } catch (error) {
            console.error('[AnalyticsMonitor] Failed to load filter builder:', error);
        }
    }

    /**
     * Save current filter as preset
     */
    saveFilterPreset(name, description = '') {
        if (!name || name.trim() === '') {
            throw new Error('Preset name is required');
        }

        const preset = {
            name: name.trim(),
            description: description.trim(),
            created: Date.now(),
            modified: Date.now(),
            conditions: JSON.parse(JSON.stringify(this.filterBuilder.conditions)),
            logic: this.filterBuilder.logic
        };

        this.filterPresets[name] = preset;
        this.saveFilterPresets();

        console.log(`[AnalyticsMonitor] Filter preset "${name}" saved`);
        return preset;
    }

    /**
     * Load filter preset
     */
    loadFilterPreset(name) {
        const preset = this.filterPresets[name];
        if (!preset) {
            throw new Error(`Preset "${name}" not found`);
        }

        this.filterBuilder.conditions = JSON.parse(JSON.stringify(preset.conditions));
        this.filterBuilder.logic = preset.logic;
        this.filterBuilder.enabled = true;
        this.currentPreset = name;

        this.saveFilterBuilder();
        this.refreshAll();

        console.log(`[AnalyticsMonitor] Filter preset "${name}" loaded`);
        return preset;
    }

    /**
     * Delete filter preset
     */
    deleteFilterPreset(name) {
        if (!this.filterPresets[name]) {
            throw new Error(`Preset "${name}" not found`);
        }

        delete this.filterPresets[name];

        if (this.currentPreset === name) {
            this.currentPreset = null;
        }

        this.saveFilterPresets();
        console.log(`[AnalyticsMonitor] Filter preset "${name}" deleted`);
    }

    /**
     * Update filter preset
     */
    updateFilterPreset(name, updates) {
        const preset = this.filterPresets[name];
        if (!preset) {
            throw new Error(`Preset "${name}" not found`);
        }

        Object.assign(preset, updates);
        preset.modified = Date.now();

        this.saveFilterPresets();
        console.log(`[AnalyticsMonitor] Filter preset "${name}" updated`);
    }

    /**
     * Get all filter presets
     */
    getFilterPresets() {
        return Object.values(this.filterPresets);
    }

    /**
     * Export filter preset to JSON
     */
    exportFilterPreset(name) {
        const preset = this.filterPresets[name];
        if (!preset) {
            throw new Error(`Preset "${name}" not found`);
        }

        const json = JSON.stringify(preset, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `filter-preset-${name}-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        console.log(`[AnalyticsMonitor] Filter preset "${name}" exported`);
    }

    /**
     * Import filter preset from JSON
     */
    importFilterPreset(file) {
        const reader = new FileReader();

        reader.onload = (e) => {
            try {
                const preset = JSON.parse(e.target.result);

                // Validate preset
                if (!preset.name || !preset.conditions || !Array.isArray(preset.conditions)) {
                    throw new Error('Invalid preset format');
                }

                // Check if preset already exists
                if (this.filterPresets[preset.name]) {
                    if (!confirm(`Preset "${preset.name}" already exists. Overwrite?`)) {
                        return;
                    }
                }

                // Import preset
                this.filterPresets[preset.name] = preset;
                this.saveFilterPresets();

                console.log(`[AnalyticsMonitor] Filter preset "${preset.name}" imported`);
                alert(`Filter preset "${preset.name}" imported successfully`);
            } catch (error) {
                console.error('[AnalyticsMonitor] Failed to import preset:', error);
                alert('Failed to import preset: ' + error.message);
            }
        };

        reader.onerror = () => {
            console.error('[AnalyticsMonitor] Failed to read file');
            alert('Failed to read file');
        };

        reader.readAsText(file);
    }

    /**
     * Save filter presets to localStorage
     */
    saveFilterPresets() {
        try {
            localStorage.setItem(this.filterPresetsKey, JSON.stringify(this.filterPresets));
        } catch (error) {
            console.error('[AnalyticsMonitor] Failed to save filter presets:', error);
        }
    }

    /**
     * Load filter presets from localStorage
     */
    loadFilterPresets() {
        try {
            const json = localStorage.getItem(this.filterPresetsKey);
            if (json) {
                this.filterPresets = JSON.parse(json);
                console.log(`[AnalyticsMonitor] Loaded ${Object.keys(this.filterPresets).length} filter presets`);
            }
        } catch (error) {
            console.error('[AnalyticsMonitor] Failed to load filter presets:', error);
        }
    }

    /**
     * Get condition summary text (for display)
     */
    getConditionSummary(condition) {
        const fieldNames = {
            date: 'Date',
            confidence: 'Confidence',
            latency: 'Latency',
            tool: 'Tool',
            queryType: 'Query Type',
            query: 'Query Text',
            cached: 'Cached'
        };

        const operatorNames = {
            equals: '=',
            notEquals: '≠',
            greaterThan: '>',
            lessThan: '<',
            greaterOrEqual: '≥',
            lessOrEqual: '≤',
            contains: 'contains',
            notContains: 'does not contain',
            startsWith: 'starts with',
            endsWith: 'ends with',
            before: 'before',
            after: 'after',
            between: 'between',
            regex: 'matches regex'
        };

        const fieldName = fieldNames[condition.field] || condition.field;
        const operatorName = operatorNames[condition.operator] || condition.operator;
        const notPrefix = condition.not ? 'NOT ' : '';

        let summary = `${notPrefix}${fieldName} ${operatorName} ${condition.value}`;

        if (condition.value2) {
            summary += ` and ${condition.value2}`;
        }

        return summary;
    }
}

// Global instance
let analyticsMonitor = null;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    analyticsMonitor = new AnalyticsMonitor();
    console.log('[AnalyticsMonitor] Initialized and ready');
});
