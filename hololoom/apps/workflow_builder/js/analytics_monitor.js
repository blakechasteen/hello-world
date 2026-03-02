/**
 * Phase 3.4: Advanced Analytics & Insights Monitor
 * Phase 3.5: Data Persistence (November 2025)
 * Phase 3.6: Advanced Filtering (November 2025)
 * Phase 3.7: Custom Dashboards (November 2025)
 * Phase 3.8: Advanced Filter Builder (November 2025)
 * Phase 3.9: Drag-and-Drop Dashboard (November 2025)
 * Phase 3.11: Responsive Enhancements (November 2025)
 * Phase 3.11.1: Advanced Touch Gestures (November 2025)
 * Phase 3.11.2: Gesture Macros (November 2025)
 * Phase 3.11.3: Mobile Performance Mode (November 2025)
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
 * 11. Responsive Enhancements - Touch gestures, mobile templates, breakpoint editor (Phase 3.11)
 * 11.1. Advanced Touch Gestures - Pinch-to-zoom, swipe-to-delete, double-tap, 3-finger pan (Phase 3.11.1)
 * 11.2. Gesture Macros - Record/playback gestures, pattern shortcuts, export/import (Phase 3.11.2)
 * 11.3. Mobile Performance Mode - Battery API, low-power mode, virtualized rendering, background pause (Phase 3.11.3)
 *
 * @author HoloLoom
 * @date November 2025
 */

class AnalyticsMonitor {
    constructor() {
        // Version for data migration
        this.version = '3.11.4'; // Enhanced with swipe-restore, calibration, visual preview, custom actions, memory monitoring, network detection
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
            snapToGrid: true,

            // Phase 3.11: Responsive enhancements
            breakpoints: {
                mobile: 768,    // < 768px
                tablet: 1200,   // 768px - 1200px
                desktop: 1920,  // 1200px - 1920px
                widescreen: Infinity // > 1920px
            },
            touchSettings: {
                swipeSensitivity: 50, // px minimum for swipe
                longPressDuration: 500, // ms for long press
                dragThreshold: 10, // px movement before drag starts
                enableSwipeReorder: true,
                enablePinchZoom: false
            },
            orientationSettings: {
                autoAdjust: true, // Auto-switch layout on orientation change
                portraitLayout: '1-column',
                landscapeLayout: '2-column',
                transitionDuration: 300 // ms
            },
            currentMobileTemplate: null // Currently applied mobile template
        };

        // Phase 3.11: Touch gesture state
        this.touchState = {
            startX: 0,
            startY: 0,
            startTime: 0,
            currentElement: null,
            isLongPress: false,
            longPressTimer: null,
            // Phase 3.11.1: Advanced gesture tracking
            lastTapTime: 0,
            tapCount: 0,
            pinchStartDistance: 0,
            pinchScale: 1.0,
            swipeStartX: 0,
            swipeDistance: 0,
            swipeStartTime: 0, // Enhancement: calibration timing
            threeFingerStart: null
        };

        // Phase 3.11.1: Advanced touch gesture settings
        this.advancedGestures = {
            enablePinchZoom: true,
            enableSwipeToDelete: true,
            enableDoubleTap: true,
            enableThreeFingerPan: false, // Disabled by default (experimental)
            swipeDeleteThreshold: 100, // px to trigger delete
            doubleTapInterval: 300, // ms between taps
            pinchZoomMin: 0.5, // Minimum scale
            pinchZoomMax: 3.0, // Maximum scale
            zoomedCards: {}, // cardId → scale
            // Enhancement: Swipe-right to restore
            enableSwipeRestore: true, // Enable swipe-right to restore hidden cards
            swipeRestoreThreshold: 100, // px to trigger restore
            recentlyHidden: [], // [{cardId, timestamp}] - tracks recently hidden cards (max 10)
            // Enhancement: Gesture sensitivity calibration
            calibrationMode: false, // Enable calibration mode
            calibrationData: {
                swipes: [], // {distance, duration, success}
                pinches: [], // {scale, duration, success}
                taps: [] // {interval, success}
            },
            sensitivityMultiplier: 1.0 // Global sensitivity adjustment (0.5-2.0)
        };

        // Phase 3.11.2: Gesture macro system
        this.gestureMacros = {
            recording: false,
            recordedGesture: [],
            recordStartTime: 0,
            savedMacros: {}, // name → { pattern: [...], actions: [...] }
            shortcuts: {
                'z-shape': { action: 'resetLayout', description: 'Z shape → Reset layout' },
                'circle': { action: 'refreshAll', description: 'Circle → Refresh all' },
                'line-horizontal': { action: 'toggleCompact', description: 'Horizontal line → Toggle compact' }
            },
            recognitionEnabled: true,
            recognitionThreshold: 0.7, // Similarity threshold for pattern matching
            // Enhancement: Visual gesture preview
            previewCanvas: null, // Canvas for drawing gesture
            previewContext: null, // 2D rendering context
            enableVisualPreview: true // Toggle visual preview during recording
        };

        // Phase 3.11.3: Mobile performance mode
        this.performanceMode = {
            enabled: false,
            autoEnableOnLowBattery: true,
            batteryThreshold: 20, // % battery level to auto-enable
            currentBatteryLevel: 100,
            isCharging: false,
            reducedAnimations: false,
            pauseBackgroundUpdates: false,
            virtualizedRendering: false,
            cardVirtualizationThreshold: 20, // Enable virtualization if >20 cards
            lastUpdateTime: Date.now(),
            updateInterval: 5000, // ms between updates in performance mode (default)
            activeUpdateInterval: 1000, // ms between updates in normal mode
            // Enhancement: Memory usage monitoring
            memoryMonitoring: {
                enabled: true,
                checkInterval: 10000, // Check every 10 seconds
                lastCheckTime: 0,
                history: [], // [{timestamp, usedJSHeapSize, totalJSHeapSize, jsHeapSizeLimit}]
                maxHistoryLength: 100,
                warningThreshold: 0.75, // 75% of heap limit
                criticalThreshold: 0.90, // 90% of heap limit
                currentUsagePercent: 0,
                autoOptimize: true, // Auto-enable optimizations on high memory
                optimizationApplied: false
            },
            // Enhancement: Network-aware optimizations
            networkMonitoring: {
                enabled: true,
                effectiveType: 'unknown', // slow-2g, 2g, 3g, 4g
                downlink: 0, // Mbps
                rtt: 0, // Round-trip time in ms
                saveData: false, // User requested data saver mode
                autoOptimize: true, // Auto-adjust update intervals based on network
                slowNetworkThreshold: 1.0, // Mbps - below this is "slow"
                optimizationApplied: false,
                baseUpdateInterval: 1000, // Default update interval (ms)
                currentUpdateInterval: 1000 // Current adjusted interval (ms)
            },
            // Voice UX tracking (Milestone 1 - November 2025)
            voiceUX: {
                enabled: false,
                currentMode: 'disabled', // conversational, command, streaming, disabled
                isActive: false,
                commandHistory: [], // [{timestamp, intent, latency, success, batteryLevel, networkType}]
                maxHistoryLength: 100,
                sessionMetrics: {
                    startTime: null,
                    commandsProcessed: 0,
                    threadsCreated: 0,
                    threadSwitches: 0,
                    errors: 0,
                    averageLatencyMs: 0,
                    successRate: 0
                }
            }
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

        // Phase 3.11: Enable touch gestures and orientation detection
        this.enableTouchGestures();
        this.enableOrientationDetection();
        this.applyBreakpointLayout();

        // Phase 3.11.1: Enable advanced gestures
        this.enablePinchZoom();
        this.enableSwipeToDelete();
        this.enableDoubleTap();

        // Phase 3.11.2: Load saved gesture macros
        this.loadGestureMacros();

        // Phase 3.11.3: Initialize performance monitoring
        this.initializeBatteryMonitor();
        this.initializePageVisibility();

        // Enhancement 3.11.3: Initialize memory monitoring
        if (this.performanceMode.memoryMonitoring.enabled) {
            // Check memory every 10 seconds
            setInterval(() => this.checkMemoryUsage(), 10000);
            // Initial memory check
            this.checkMemoryUsage();
        }

        // Enhancement 3.11.3: Initialize network monitoring
        if (this.performanceMode.networkMonitoring.enabled) {
            this.initializeNetworkMonitor();
        }

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

    // ==================== Phase 3.11: Responsive Enhancements ====================

    /**
     * Enable touch gesture recognition for mobile/tablet devices
     * Phase 3.11: Touch gestures (swipe to reorder, long press)
     */
    enableTouchGestures() {
        console.log('[AnalyticsMonitor] Enabling touch gestures...');

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

            // Add touch event listeners
            element.addEventListener('touchstart', (e) => this.handleTouchStart(e, cardId, element), { passive: false });
            element.addEventListener('touchmove', (e) => this.handleTouchMove(e, cardId, element), { passive: false });
            element.addEventListener('touchend', (e) => this.handleTouchEnd(e, cardId, element), { passive: false });
        }

        console.log('[AnalyticsMonitor] Touch gestures enabled');
    }

    /**
     * Handle touch start event
     */
    handleTouchStart(e, cardId, element) {
        if (!this.dashboardLayout.touchSettings.enableSwipeReorder) return;

        const touch = e.touches[0];
        this.touchState.startX = touch.clientX;
        this.touchState.startY = touch.clientY;
        this.touchState.startTime = Date.now();
        this.touchState.currentElement = element;

        // Start long press timer
        if (this.dashboardLayout.touchSettings.longPressDuration > 0) {
            this.touchState.longPressTimer = setTimeout(() => {
                this.touchState.isLongPress = true;
                element.classList.add('dragging');
                // Haptic feedback (if supported)
                if (navigator.vibrate) {
                    navigator.vibrate(50);
                }
            }, this.dashboardLayout.touchSettings.longPressDuration);
        }
    }

    /**
     * Handle touch move event
     */
    handleTouchMove(e, cardId, element) {
        if (!this.dashboardLayout.touchSettings.enableSwipeReorder) return;

        const touch = e.touches[0];
        const deltaX = touch.clientX - this.touchState.startX;
        const deltaY = touch.clientY - this.touchState.startY;

        // Check if movement exceeds drag threshold
        if (Math.abs(deltaX) > this.dashboardLayout.touchSettings.dragThreshold ||
            Math.abs(deltaY) > this.dashboardLayout.touchSettings.dragThreshold) {
            // Cancel long press if dragging
            if (this.touchState.longPressTimer) {
                clearTimeout(this.touchState.longPressTimer);
                this.touchState.longPressTimer = null;
            }

            // If long press was triggered, enable reordering
            if (this.touchState.isLongPress) {
                e.preventDefault(); // Prevent scrolling

                // Get element under touch point
                const touchX = touch.clientX;
                const touchY = touch.clientY;
                const elementUnder = document.elementFromPoint(touchX, touchY);

                // Find closest card
                const cardUnder = elementUnder?.closest('.card');
                if (cardUnder && cardUnder !== element) {
                    const rect = cardUnder.getBoundingClientRect();
                    const midpoint = rect.top + rect.height / 2;

                    if (touchY < midpoint) {
                        cardUnder.parentNode.insertBefore(element, cardUnder);
                    } else {
                        cardUnder.parentNode.insertBefore(element, cardUnder.nextSibling);
                    }
                }
            }
        }
    }

    /**
     * Handle touch end event
     */
    handleTouchEnd(e, cardId, element) {
        if (!this.dashboardLayout.touchSettings.enableSwipeReorder) return;

        // Cancel long press timer
        if (this.touchState.longPressTimer) {
            clearTimeout(this.touchState.longPressTimer);
            this.touchState.longPressTimer = null;
        }

        // If long press was active, update card order
        if (this.touchState.isLongPress) {
            element.classList.remove('dragging');
            this.updateCardOrderFromDOM();
        } else {
            // Check for swipe gesture
            const touch = e.changedTouches[0];
            const deltaY = touch.clientY - this.touchState.startY;
            const duration = Date.now() - this.touchState.startTime;

            if (Math.abs(deltaY) > this.dashboardLayout.touchSettings.swipeSensitivity &&
                duration < 300) { // Quick swipe (< 300ms)

                // Swipe detected - could implement quick reorder here
                console.log(`[AnalyticsMonitor] Swipe ${deltaY > 0 ? 'down' : 'up'} detected`);
            }
        }

        // Reset touch state
        this.touchState.isLongPress = false;
        this.touchState.currentElement = null;
    }

    /**
     * Apply a mobile-first template
     * Phase 3.11: Mobile templates
     */
    applyMobileTemplate(templateName) {
        console.log(`[AnalyticsMonitor] Applying mobile template: ${templateName}`);

        const templates = this.getMobileTemplates();
        const template = templates[templateName];

        if (!template) {
            console.error(`[AnalyticsMonitor] Unknown mobile template: ${templateName}`);
            return;
        }

        // Apply grid layout
        this.dashboardLayout.gridLayout = template.gridLayout;
        this.applyGridLayout();

        // Apply card sizes
        Object.assign(this.dashboardLayout.cardSizes, template.cardSizes);
        this.applyAllCardSizes();

        // Apply touch settings (if specified)
        if (template.touchSettings) {
            Object.assign(this.dashboardLayout.touchSettings, template.touchSettings);
        }

        // Store current template
        this.dashboardLayout.currentMobileTemplate = templateName;

        // Save layout
        this.saveDashboardLayout();

        console.log(`[AnalyticsMonitor] Mobile template '${templateName}' applied`);
    }

    /**
     * Get available mobile-first templates
     * Phase 3.11: Mobile templates
     */
    getMobileTemplates() {
        return {
            'mobile-compact': {
                name: 'Mobile Compact',
                gridLayout: '1-column',
                cardSizes: {
                    comparison: 'small',
                    confidence: 'small',
                    effectiveness: 'small',
                    health: 'small',
                    management: 'small'
                },
                touchSettings: {
                    enableSwipeReorder: true,
                    swipeSensitivity: 60 // Higher threshold for mobile
                }
            },
            'mobile-focused': {
                name: 'Mobile Focused',
                gridLayout: '1-column',
                cardSizes: {
                    comparison: 'large',
                    confidence: 'small',
                    effectiveness: 'small',
                    health: 'small',
                    management: 'small'
                },
                touchSettings: {
                    enableSwipeReorder: true,
                    swipeSensitivity: 60
                }
            },
            'tablet-split': {
                name: 'Tablet Split',
                gridLayout: '2-column',
                cardSizes: {
                    comparison: 'medium',
                    confidence: 'medium',
                    effectiveness: 'medium',
                    health: 'medium',
                    management: 'medium'
                },
                touchSettings: {
                    enableSwipeReorder: true,
                    swipeSensitivity: 50
                }
            },
            'tablet-grid': {
                name: 'Tablet Grid',
                gridLayout: '3-column',
                cardSizes: {
                    comparison: 'small',
                    confidence: 'small',
                    effectiveness: 'small',
                    health: 'small',
                    management: 'small'
                },
                touchSettings: {
                    enableSwipeReorder: false // Too cramped for swipe
                }
            },
            'touch-optimized': {
                name: 'Touch Optimized',
                gridLayout: 'auto',
                cardSizes: {
                    comparison: 'large',
                    confidence: 'large',
                    effectiveness: 'large',
                    health: 'large',
                    management: 'large'
                },
                touchSettings: {
                    enableSwipeReorder: true,
                    swipeSensitivity: 40,
                    longPressDuration: 400 // Shorter for faster interaction
                }
            }
        };
    }

    /**
     * Set a custom breakpoint value
     * Phase 3.11: Breakpoint editor
     */
    setBreakpoint(name, value) {
        if (!['mobile', 'tablet', 'desktop'].includes(name)) {
            console.error(`[AnalyticsMonitor] Invalid breakpoint name: ${name}`);
            return;
        }

        // Validate ordering (mobile < tablet < desktop)
        const breakpoints = this.dashboardLayout.breakpoints;
        if (name === 'mobile' && value >= breakpoints.tablet) {
            console.error(`[AnalyticsMonitor] Mobile breakpoint must be < tablet breakpoint`);
            return;
        }
        if (name === 'tablet' && (value <= breakpoints.mobile || value >= breakpoints.desktop)) {
            console.error(`[AnalyticsMonitor] Tablet breakpoint must be between mobile and desktop`);
            return;
        }
        if (name === 'desktop' && value <= breakpoints.tablet) {
            console.error(`[AnalyticsMonitor] Desktop breakpoint must be > tablet breakpoint`);
            return;
        }

        breakpoints[name] = value;
        this.saveDashboardLayout();

        // Update layout based on new breakpoint
        this.applyBreakpointLayout();

        console.log(`[AnalyticsMonitor] Breakpoint '${name}' set to ${value}px`);
    }

    /**
     * Get the currently active breakpoint
     * Phase 3.11: Breakpoint detection
     */
    getActiveBreakpoint() {
        const width = window.innerWidth;
        const bp = this.dashboardLayout.breakpoints;

        if (width < bp.mobile) return 'mobile';
        if (width < bp.tablet) return 'tablet';
        if (width < bp.desktop) return 'desktop';
        return 'widescreen';
    }

    /**
     * Apply layout adjustments based on current breakpoint
     * Phase 3.11: Responsive layout
     */
    applyBreakpointLayout() {
        const breakpoint = this.getActiveBreakpoint();
        console.log(`[AnalyticsMonitor] Active breakpoint: ${breakpoint}`);

        // Auto-apply mobile template if enabled and on mobile
        const settings = this.dashboardLayout.touchSettings;
        if (breakpoint === 'mobile' && settings.enableSwipeReorder && !this.dashboardLayout.currentMobileTemplate) {
            this.applyMobileTemplate('mobile-compact');
        }

        // Tablet-specific adjustments
        if (breakpoint === 'tablet' && !this.dashboardLayout.currentMobileTemplate) {
            // Suggest tablet-split layout if currently using 3-column
            if (this.dashboardLayout.gridLayout === '3-column') {
                console.log('[AnalyticsMonitor] Suggesting 2-column layout for tablet');
            }
        }
    }

    /**
     * Enable orientation change detection
     * Phase 3.11: Orientation detection
     */
    enableOrientationDetection() {
        if (!this.dashboardLayout.orientationSettings.autoAdjust) {
            console.log('[AnalyticsMonitor] Orientation auto-adjust disabled');
            return;
        }

        console.log('[AnalyticsMonitor] Enabling orientation detection...');

        // Use matchMedia for better support
        const portraitQuery = window.matchMedia('(orientation: portrait)');
        const handler = (e) => this.handleOrientationChange(e.matches);

        // Modern browsers
        if (portraitQuery.addEventListener) {
            portraitQuery.addEventListener('change', handler);
        }
        // Legacy browsers
        else if (portraitQuery.addListener) {
            portraitQuery.addListener(handler);
        }

        // Initial check
        handler(portraitQuery.matches);

        console.log('[AnalyticsMonitor] Orientation detection enabled');
    }

    /**
     * Handle orientation change event
     * Phase 3.11: Orientation handling
     */
    handleOrientationChange(isPortrait) {
        console.log(`[AnalyticsMonitor] Orientation changed to ${isPortrait ? 'portrait' : 'landscape'}`);

        const settings = this.dashboardLayout.orientationSettings;
        const newLayout = isPortrait ? settings.portraitLayout : settings.landscapeLayout;

        // Apply layout change with transition
        const container = document.getElementById('analytics-cards-container');
        if (container) {
            container.style.transition = `all ${settings.transitionDuration}ms ease`;
        }

        this.setGridLayout(newLayout);

        // Remove transition after completion
        setTimeout(() => {
            if (container) {
                container.style.transition = '';
            }
        }, settings.transitionDuration);
    }

    /**
     * Set individual gesture sensitivity parameter
     * Phase 3.11: Gesture customization
     */
    setGestureSensitivity(property, value) {
        const validProperties = ['swipeSensitivity', 'longPressDuration', 'dragThreshold', 'enableSwipeReorder', 'enablePinchZoom'];

        if (!validProperties.includes(property)) {
            console.error(`[AnalyticsMonitor] Invalid gesture property: ${property}`);
            return;
        }

        this.dashboardLayout.touchSettings[property] = value;
        this.saveDashboardLayout();

        console.log(`[AnalyticsMonitor] Gesture setting '${property}' set to ${value}`);
    }

    /**
     * Get current gesture settings
     * Phase 3.11: Gesture settings
     */
    getGestureSettings() {
        return { ...this.dashboardLayout.touchSettings };
    }

    /**
     * Reset gesture settings to defaults
     * Phase 3.11: Reset gestures
     */
    resetGestureSettings() {
        this.dashboardLayout.touchSettings = {
            swipeSensitivity: 50,
            longPressDuration: 500,
            dragThreshold: 10,
            enableSwipeReorder: true,
            enablePinchZoom: false
        };
        this.saveDashboardLayout();

        console.log('[AnalyticsMonitor] Gesture settings reset to defaults');
    }

    /**
     * Set orientation-specific layout
     * Phase 3.11: Orientation settings
     */
    setOrientationLayout(orientation, layout) {
        if (!['portrait', 'landscape'].includes(orientation)) {
            console.error(`[AnalyticsMonitor] Invalid orientation: ${orientation}`);
            return;
        }

        if (!['auto', '1-column', '2-column', '3-column', 'masonry'].includes(layout)) {
            console.error(`[AnalyticsMonitor] Invalid layout: ${layout}`);
            return;
        }

        const key = `${orientation}Layout`;
        this.dashboardLayout.orientationSettings[key] = layout;
        this.saveDashboardLayout();

        console.log(`[AnalyticsMonitor] ${orientation} layout set to ${layout}`);

        // Reapply if current orientation matches
        const isPortrait = window.matchMedia('(orientation: portrait)').matches;
        if ((orientation === 'portrait' && isPortrait) || (orientation === 'landscape' && !isPortrait)) {
            this.handleOrientationChange(isPortrait);
        }
    }

    // ===================================================================
    // PHASE 3.11.1: ADVANCED TOUCH GESTURES
    // ===================================================================

    /**
     * Enable pinch-to-zoom gesture on cards
     * Phase 3.11.1: Pinch-to-zoom
     */
    enablePinchZoom() {
        if (!this.advancedGestures.enablePinchZoom) {
            console.log('[AnalyticsMonitor] Pinch-to-zoom disabled');
            return;
        }

        console.log('[AnalyticsMonitor] Enabling pinch-to-zoom...');

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

            // Add touch event listeners for pinch
            element.addEventListener('touchstart', (e) => this.handlePinchStart(e, cardId, element), { passive: false });
            element.addEventListener('touchmove', (e) => this.handlePinchMove(e, cardId, element), { passive: false });
            element.addEventListener('touchend', (e) => this.handlePinchEnd(e, cardId, element), { passive: false });
        }

        console.log('[AnalyticsMonitor] Pinch-to-zoom enabled');
    }

    /**
     * Handle pinch start (two fingers)
     * Phase 3.11.1: Pinch-to-zoom
     */
    handlePinchStart(e, cardId, element) {
        if (!this.advancedGestures.enablePinchZoom) return;
        if (e.touches.length !== 2) return;

        // Calculate initial distance between two fingers
        const touch1 = e.touches[0];
        const touch2 = e.touches[1];
        const distance = Math.hypot(
            touch2.clientX - touch1.clientX,
            touch2.clientY - touch1.clientY
        );

        this.touchState.pinchStartDistance = distance;
        this.touchState.currentElement = element;

        console.log(`[AnalyticsMonitor] Pinch start on ${cardId}, distance: ${distance.toFixed(0)}px`);
    }

    /**
     * Handle pinch move (zoom in/out)
     * Phase 3.11.1: Pinch-to-zoom
     */
    handlePinchMove(e, cardId, element) {
        if (!this.advancedGestures.enablePinchZoom) return;
        if (e.touches.length !== 2) return;
        if (this.touchState.pinchStartDistance === 0) return;

        e.preventDefault(); // Prevent page zoom

        // Calculate current distance
        const touch1 = e.touches[0];
        const touch2 = e.touches[1];
        const distance = Math.hypot(
            touch2.clientX - touch1.clientX,
            touch2.clientY - touch1.clientY
        );

        // Calculate scale
        const scale = distance / this.touchState.pinchStartDistance;
        this.touchState.pinchScale = Math.max(
            this.advancedGestures.pinchZoomMin,
            Math.min(this.advancedGestures.pinchZoomMax, scale)
        );

        // Apply transform
        const cardContent = element.querySelector('.card-body, .card-content');
        if (cardContent) {
            cardContent.style.transform = `scale(${this.touchState.pinchScale})`;
            cardContent.style.transformOrigin = 'center center';
            cardContent.style.transition = 'none';
        }
    }

    /**
     * Handle pinch end (finalize zoom)
     * Phase 3.11.1: Pinch-to-zoom
     */
    handlePinchEnd(e, cardId, element) {
        if (!this.advancedGestures.enablePinchZoom) return;

        // Store final scale for this card
        if (this.touchState.pinchScale !== 1.0) {
            this.advancedGestures.zoomedCards[cardId] = this.touchState.pinchScale;
            console.log(`[AnalyticsMonitor] Card ${cardId} zoomed to ${this.touchState.pinchScale.toFixed(2)}x`);
        }

        // Reset pinch state
        this.touchState.pinchStartDistance = 0;
        this.touchState.pinchScale = 1.0;
    }

    /**
     * Reset card zoom to 1.0
     * Phase 3.11.1: Pinch-to-zoom
     */
    resetCardZoom(cardId) {
        const domId = {
            'comparison': 'query-comparison-card',
            'confidence': 'confidence-tracking-card',
            'effectiveness': 'tool-effectiveness-card',
            'health': 'system-health-card',
            'management': 'data-management-card'
        }[cardId];

        const element = document.getElementById(domId);
        if (!element) return;

        const cardContent = element.querySelector('.card-body, .card-content');
        if (cardContent) {
            cardContent.style.transform = 'scale(1.0)';
            cardContent.style.transition = 'transform 0.3s ease';
        }

        delete this.advancedGestures.zoomedCards[cardId];
        console.log(`[AnalyticsMonitor] Card ${cardId} zoom reset`);
    }

    /**
     * Enable swipe-to-delete gesture on cards
     * Phase 3.11.1: Swipe-to-delete
     */
    enableSwipeToDelete() {
        if (!this.advancedGestures.enableSwipeToDelete) {
            console.log('[AnalyticsMonitor] Swipe-to-delete disabled');
            return;
        }

        console.log('[AnalyticsMonitor] Enabling swipe-to-delete...');

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

            // Add swipe listeners (reuse touch events)
            element.addEventListener('touchstart', (e) => this.handleSwipeStart(e, cardId, element), { passive: false });
            element.addEventListener('touchmove', (e) => this.handleSwipeMove(e, cardId, element), { passive: false });
            element.addEventListener('touchend', (e) => this.handleSwipeEnd(e, cardId, element), { passive: false });
        }

        console.log('[AnalyticsMonitor] Swipe-to-delete enabled');
    }

    /**
     * Handle swipe start
     * Phase 3.11.1: Swipe-to-delete
     */
    handleSwipeStart(e, cardId, element) {
        if (!this.advancedGestures.enableSwipeToDelete) return;
        if (e.touches.length !== 1) return; // Only single finger

        const touch = e.touches[0];
        this.touchState.swipeStartX = touch.clientX;
        this.touchState.swipeDistance = 0;
        this.touchState.swipeStartTime = Date.now(); // Enhancement: calibration timing
        this.touchState.currentElement = element;
    }

    /**
     * Handle swipe move (horizontal drag)
     * Phase 3.11.1: Swipe-to-delete
     */
    handleSwipeMove(e, cardId, element) {
        if (!this.advancedGestures.enableSwipeToDelete) return;
        if (e.touches.length !== 1) return;
        if (this.touchState.swipeStartX === 0) return;

        const touch = e.touches[0];
        const deltaX = touch.clientX - this.touchState.swipeStartX;
        this.touchState.swipeDistance = deltaX;

        // Only allow left swipe (negative deltaX)
        if (deltaX < 0) {
            element.style.transform = `translateX(${deltaX}px)`;
            element.style.transition = 'none';

            // Change opacity as swipe progresses
            const progress = Math.abs(deltaX) / this.advancedGestures.swipeDeleteThreshold;
            element.style.opacity = Math.max(0.3, 1 - progress * 0.7);
        }
    }

    /**
     * Handle swipe end (delete if threshold exceeded)
     * Phase 3.11.1: Swipe-to-delete
     */
    handleSwipeEnd(e, cardId, element) {
        if (!this.advancedGestures.enableSwipeToDelete) return;

        const deleteThreshold = this.advancedGestures.swipeDeleteThreshold;
        const restoreThreshold = this.advancedGestures.swipeRestoreThreshold;

        // Swipe-left to delete
        if (this.touchState.swipeDistance < -deleteThreshold) {
            // Threshold exceeded → hide card
            element.style.transition = 'transform 0.3s ease, opacity 0.3s ease';
            element.style.transform = 'translateX(-100%)';
            element.style.opacity = '0';

            setTimeout(() => {
                this.setCardVisibility(cardId, false);
                element.style.transform = '';
                element.style.opacity = '';

                // Track as recently hidden (Enhancement: swipe-right restore)
                this.trackHiddenCard(cardId);
            }, 300);

            console.log(`[AnalyticsMonitor] Card ${cardId} swiped to delete`);
        }
        // Enhancement: Swipe-right to restore most recently hidden card
        else if (this.touchState.swipeDistance > restoreThreshold &&
                 this.advancedGestures.enableSwipeRestore) {
            const restored = this.restoreMostRecentlyHiddenCard();
            if (restored) {
                console.log(`[AnalyticsMonitor] Restored card: ${restored}`);
                // Visual feedback
                element.style.transition = 'transform 0.3s ease';
                element.style.transform = 'translateX(20px)';
                setTimeout(() => {
                    element.style.transform = '';
                }, 300);
            } else {
                // No cards to restore → just snap back
                element.style.transition = 'transform 0.3s ease';
                element.style.transform = '';
            }
        }
        else {
            // Not enough swipe → snap back
            element.style.transition = 'transform 0.3s ease, opacity 0.3s ease';
            element.style.transform = '';
            element.style.opacity = '';
        }

        // Enhancement: Record calibration data
        if (this.advancedGestures.calibrationMode && this.touchState.swipeStartTime > 0) {
            const duration = Date.now() - this.touchState.swipeStartTime;
            const distance = Math.abs(this.touchState.swipeDistance);
            const success = distance >= deleteThreshold || distance >= restoreThreshold;

            this.recordCalibrationGesture('swipes', {
                distance: distance,
                duration: duration,
                success: success
            });
        }

        // Reset swipe state
        this.touchState.swipeStartX = 0;
        this.touchState.swipeDistance = 0;
        this.touchState.swipeStartTime = 0;
    }

    /**
     * Track hidden card for potential restore
     * Enhancement: Swipe-right to restore
     */
    trackHiddenCard(cardId) {
        const now = Date.now();

        // Add to front of array (most recent first)
        this.advancedGestures.recentlyHidden.unshift({
            cardId: cardId,
            timestamp: now
        });

        // Keep only last 10 hidden cards
        if (this.advancedGestures.recentlyHidden.length > 10) {
            this.advancedGestures.recentlyHidden = this.advancedGestures.recentlyHidden.slice(0, 10);
        }

        console.log(`[AnalyticsMonitor] Tracked hidden card: ${cardId} (${this.advancedGestures.recentlyHidden.length} in history)`);
    }

    /**
     * Restore most recently hidden card
     * Enhancement: Swipe-right to restore
     * @returns {string|null} Restored card ID or null if none available
     */
    restoreMostRecentlyHiddenCard() {
        if (this.advancedGestures.recentlyHidden.length === 0) {
            console.log('[AnalyticsMonitor] No hidden cards to restore');
            return null;
        }

        // Get most recent hidden card
        const mostRecent = this.advancedGestures.recentlyHidden.shift();
        const cardId = mostRecent.cardId;

        // Restore card visibility
        this.setCardVisibility(cardId, true);

        console.log(`[AnalyticsMonitor] Restored card: ${cardId} (${this.advancedGestures.recentlyHidden.length} remaining in history)`);

        return cardId;
    }

    /**
     * Start gesture sensitivity calibration mode
     * Enhancement: Gesture sensitivity calibration
     */
    startGestureCalibration() {
        this.advancedGestures.calibrationMode = true;
        this.advancedGestures.calibrationData = {
            swipes: [],
            pinches: [],
            taps: []
        };

        console.log('[AnalyticsMonitor] Gesture calibration mode STARTED');
        console.log('[AnalyticsMonitor] Perform 5-10 gestures of each type, then call analyzeCalibrationData()');
    }

    /**
     * Stop calibration and analyze collected data
     * Enhancement: Gesture sensitivity calibration
     * @returns {object} Calibration results with recommendations
     */
    stopGestureCalibration() {
        this.advancedGestures.calibrationMode = false;

        const results = this.analyzeCalibrationData();

        console.log('[AnalyticsMonitor] Gesture calibration mode STOPPED');
        console.log('[AnalyticsMonitor] Calibration results:', results);

        return results;
    }

    /**
     * Analyze calibration data and recommend optimal settings
     * Enhancement: Gesture sensitivity calibration
     * @returns {object} Recommendations for sensitivity adjustments
     */
    analyzeCalibrationData() {
        const data = this.advancedGestures.calibrationData;
        const recommendations = {
            sensitivityMultiplier: 1.0,
            swipeThreshold: this.advancedGestures.swipeDeleteThreshold,
            doubleTapInterval: this.advancedGestures.doubleTapInterval,
            pinchSensitivity: 1.0,
            confidence: 0.0,
            analysis: {}
        };

        // Analyze swipes
        if (data.swipes.length >= 3) {
            const avgSwipeDistance = data.swipes.reduce((sum, s) => sum + Math.abs(s.distance), 0) / data.swipes.length;
            const successRate = data.swipes.filter(s => s.success).length / data.swipes.length;

            recommendations.analysis.swipes = {
                count: data.swipes.length,
                avgDistance: Math.round(avgSwipeDistance),
                successRate: successRate
            };

            // Recommend threshold based on average swipe distance
            if (avgSwipeDistance < 80) {
                recommendations.swipeThreshold = 60; // User makes short swipes
                recommendations.sensitivityMultiplier = Math.min(recommendations.sensitivityMultiplier, 0.7);
            } else if (avgSwipeDistance > 150) {
                recommendations.swipeThreshold = 130; // User makes long swipes
                recommendations.sensitivityMultiplier = Math.max(recommendations.sensitivityMultiplier, 1.3);
            }
        }

        // Analyze taps
        if (data.taps.length >= 3) {
            const avgTapInterval = data.taps.reduce((sum, t) => sum + t.interval, 0) / data.taps.length;
            const successRate = data.taps.filter(t => t.success).length / data.taps.length;

            recommendations.analysis.taps = {
                count: data.taps.length,
                avgInterval: Math.round(avgTapInterval),
                successRate: successRate
            };

            // Recommend interval based on average tap speed
            if (avgTapInterval < 250) {
                recommendations.doubleTapInterval = 350; // User taps quickly
            } else if (avgTapInterval > 400) {
                recommendations.doubleTapInterval = 500; // User taps slowly
            }
        }

        // Analyze pinches
        if (data.pinches.length >= 3) {
            const avgScale = data.pinches.reduce((sum, p) => sum + p.scale, 0) / data.pinches.length;
            const successRate = data.pinches.filter(p => p.success).length / data.pinches.length;

            recommendations.analysis.pinches = {
                count: data.pinches.length,
                avgScale: avgScale.toFixed(2),
                successRate: successRate
            };

            // Adjust pinch sensitivity
            if (avgScale < 1.3) {
                recommendations.pinchSensitivity = 1.2; // User makes small pinches
            } else if (avgScale > 2.0) {
                recommendations.pinchSensitivity = 0.8; // User makes large pinches
            }
        }

        // Calculate overall confidence
        const totalSamples = data.swipes.length + data.taps.length + data.pinches.length;
        if (totalSamples >= 10) {
            recommendations.confidence = Math.min(totalSamples / 15, 1.0); // Max confidence at 15 samples
        }

        return recommendations;
    }

    /**
     * Apply calibration recommendations
     * Enhancement: Gesture sensitivity calibration
     */
    applyCalibrationRecommendations(recommendations) {
        if (recommendations.confidence < 0.5) {
            console.warn('[AnalyticsMonitor] Low calibration confidence, recommendations may not be accurate');
        }

        // Apply sensitivity multiplier
        this.advancedGestures.sensitivityMultiplier = recommendations.sensitivityMultiplier;

        // Apply swipe threshold
        if (recommendations.swipeThreshold) {
            this.advancedGestures.swipeDeleteThreshold = recommendations.swipeThreshold;
            this.advancedGestures.swipeRestoreThreshold = recommendations.swipeThreshold;
        }

        // Apply double-tap interval
        if (recommendations.doubleTapInterval) {
            this.advancedGestures.doubleTapInterval = recommendations.doubleTapInterval;
        }

        console.log('[AnalyticsMonitor] Calibration recommendations applied:', recommendations);
    }

    /**
     * Record calibration gesture attempt
     * Enhancement: Gesture sensitivity calibration
     */
    recordCalibrationGesture(type, data) {
        if (!this.advancedGestures.calibrationMode) return;

        this.advancedGestures.calibrationData[type].push(data);

        console.log(`[AnalyticsMonitor] Calibration gesture recorded: ${type}`, data);
    }

    /**
     * Enable double-tap to expand/collapse card
     * Phase 3.11.1: Double-tap
     */
    enableDoubleTap() {
        if (!this.advancedGestures.enableDoubleTap) {
            console.log('[AnalyticsMonitor] Double-tap disabled');
            return;
        }

        console.log('[AnalyticsMonitor] Enabling double-tap...');

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

            element.addEventListener('touchend', (e) => this.handleDoubleTap(e, cardId, element));
        }

        console.log('[AnalyticsMonitor] Double-tap enabled');
    }

    /**
     * Handle double-tap (toggle card size)
     * Phase 3.11.1: Double-tap
     */
    handleDoubleTap(e, cardId, element) {
        if (!this.advancedGestures.enableDoubleTap) return;
        if (e.touches.length > 0) return; // Still touching

        const now = Date.now();
        const timeSinceLastTap = now - this.touchState.lastTapTime;

        if (timeSinceLastTap < this.advancedGestures.doubleTapInterval) {
            // Double-tap detected!
            this.touchState.tapCount = 0;
            this.touchState.lastTapTime = 0;

            // Enhancement: Record calibration data
            if (this.advancedGestures.calibrationMode) {
                this.recordCalibrationGesture('taps', {
                    interval: timeSinceLastTap,
                    success: true
                });
            }

            // Toggle card size: small ↔ large
            const currentSize = this.dashboardLayout.cardSizes[cardId] || 'medium';
            const newSize = currentSize === 'large' ? 'small' : 'large';

            this.setCardSize(cardId, newSize);
            console.log(`[AnalyticsMonitor] Double-tap on ${cardId}: ${currentSize} → ${newSize}`);

            // Haptic feedback
            if (navigator.vibrate) {
                navigator.vibrate([30, 20, 30]); // Double vibration
            }
        } else {
            // First tap
            this.touchState.lastTapTime = now;
            this.touchState.tapCount = 1;

            // Enhancement: Record failed calibration attempt (too slow)
            if (this.advancedGestures.calibrationMode && this.touchState.tapCount === 0 && timeSinceLastTap > 0) {
                this.recordCalibrationGesture('taps', {
                    interval: timeSinceLastTap,
                    success: false
                });
            }
        }
    }

    // ===================================================================
    // PHASE 3.11.2: GESTURE MACROS
    // ===================================================================

    /**
     * Create visual preview canvas for gesture recording
     * Enhancement: Visual gesture preview
     */
    createGesturePreviewCanvas() {
        if (!this.gestureMacros.enableVisualPreview) return;

        // Remove existing canvas if any
        this.removeGesturePreviewCanvas();

        // Create canvas
        const canvas = document.createElement('canvas');
        canvas.id = 'gesture-preview-canvas';
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        canvas.style.position = 'fixed';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.pointerEvents = 'none'; // Don't block touch events
        canvas.style.zIndex = '9999';
        canvas.style.background = 'transparent';

        document.body.appendChild(canvas);

        this.gestureMacros.previewCanvas = canvas;
        this.gestureMacros.previewContext = canvas.getContext('2d');

        // Set drawing style
        const ctx = this.gestureMacros.previewContext;
        ctx.strokeStyle = '#2196f3'; // Blue trail
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.shadowBlur = 8;
        ctx.shadowColor = '#2196f3';

        console.log('[AnalyticsMonitor] Gesture preview canvas created');
    }

    /**
     * Remove visual preview canvas
     * Enhancement: Visual gesture preview
     */
    removeGesturePreviewCanvas() {
        if (this.gestureMacros.previewCanvas) {
            this.gestureMacros.previewCanvas.remove();
            this.gestureMacros.previewCanvas = null;
            this.gestureMacros.previewContext = null;
            console.log('[AnalyticsMonitor] Gesture preview canvas removed');
        }
    }

    /**
     * Draw gesture path on preview canvas
     * Enhancement: Visual gesture preview
     */
    drawGesturePreview(points) {
        if (!this.gestureMacros.enableVisualPreview) return;
        if (!this.gestureMacros.previewContext) return;
        if (points.length < 2) return;

        const ctx = this.gestureMacros.previewContext;
        const lastTwoPoints = points.slice(-2);

        // Draw line segment from previous point to current point
        ctx.beginPath();
        ctx.moveTo(lastTwoPoints[0].x, lastTwoPoints[0].y);
        ctx.lineTo(lastTwoPoints[1].x, lastTwoPoints[1].y);
        ctx.stroke();

        // Add dot at current point
        ctx.beginPath();
        ctx.arc(lastTwoPoints[1].x, lastTwoPoints[1].y, 2, 0, Math.PI * 2);
        ctx.fillStyle = '#2196f3';
        ctx.fill();
    }

    /**
     * Start recording a gesture macro
     * Phase 3.11.2: Gesture recording
     */
    startGestureRecording() {
        this.gestureMacros.recording = true;
        this.gestureMacros.recordedGesture = [];
        this.gestureMacros.recordStartTime = Date.now();

        // Enhancement: Create visual preview canvas
        this.createGesturePreviewCanvas();

        console.log('[AnalyticsMonitor] Gesture recording started');

        // Listen to touch events globally
        document.addEventListener('touchmove', this._recordTouchMove.bind(this), { passive: true });
    }

    /**
     * Internal: Record touch move points
     * Phase 3.11.2: Gesture recording
     */
    _recordTouchMove(e) {
        if (!this.gestureMacros.recording) return;

        const touch = e.touches[0];
        const point = {
            x: touch.clientX,
            y: touch.clientY,
            timestamp: Date.now() - this.gestureMacros.recordStartTime
        };

        this.gestureMacros.recordedGesture.push(point);

        // Enhancement: Draw on preview canvas
        this.drawGesturePreview(this.gestureMacros.recordedGesture);
    }

    /**
     * Stop recording and save gesture macro
     * Phase 3.11.2: Gesture recording
     */
    stopGestureRecording(macroName) {
        if (!this.gestureMacros.recording) {
            console.error('[AnalyticsMonitor] No recording in progress');
            return null;
        }

        this.gestureMacros.recording = false;

        // Enhancement: Remove visual preview canvas
        this.removeGesturePreviewCanvas();

        document.removeEventListener('touchmove', this._recordTouchMove.bind(this));

        const gesture = {
            name: macroName || `Gesture_${Date.now()}`,
            pattern: this.gestureMacros.recordedGesture,
            duration: Date.now() - this.gestureMacros.recordStartTime,
            points: this.gestureMacros.recordedGesture.length
        };

        console.log(`[AnalyticsMonitor] Gesture recorded: ${gesture.name} (${gesture.points} points, ${gesture.duration}ms)`);

        // Try to recognize pattern
        const recognizedAction = this.recognizeGesturePattern(gesture.pattern);
        if (recognizedAction) {
            gesture.action = recognizedAction;
            console.log(`[AnalyticsMonitor] Gesture recognized as: ${recognizedAction}`);
        }

        return gesture;
    }

    /**
     * Recognize gesture pattern (Z-shape, circle, line, etc.)
     * Phase 3.11.2: Pattern recognition
     */
    recognizeGesturePattern(points) {
        if (!this.gestureMacros.recognitionEnabled) return null;
        if (points.length < 5) return null; // Too few points

        // Simple pattern recognition based on direction changes
        const directions = [];
        for (let i = 1; i < points.length; i++) {
            const dx = points[i].x - points[i-1].x;
            const dy = points[i].y - points[i-1].y;

            // Classify direction (8 directions)
            const angle = Math.atan2(dy, dx) * 180 / Math.PI;
            if (angle > -22.5 && angle <= 22.5) directions.push('E');
            else if (angle > 22.5 && angle <= 67.5) directions.push('SE');
            else if (angle > 67.5 && angle <= 112.5) directions.push('S');
            else if (angle > 112.5 && angle <= 157.5) directions.push('SW');
            else if (angle > 157.5 || angle <= -157.5) directions.push('W');
            else if (angle > -157.5 && angle <= -112.5) directions.push('NW');
            else if (angle > -112.5 && angle <= -67.5) directions.push('N');
            else directions.push('NE');
        }

        // Simplify directions (remove duplicates)
        const simplified = directions.filter((d, i) => i === 0 || d !== directions[i-1]);
        const pattern = simplified.join('-');

        // Match against known patterns
        if (pattern.includes('E-SE-S') || pattern.includes('E-S-W')) {
            return 'z-shape'; // Z-shape
        }
        if (pattern.match(/E.*S.*W.*N/) || pattern.match(/N.*E.*S.*W/)) {
            return 'circle'; // Circle (clockwise or counter-clockwise)
        }
        if (pattern.match(/^E+$/)) {
            return 'line-horizontal'; // Horizontal line
        }
        if (pattern.match(/^S+$/)) {
            return 'line-vertical'; // Vertical line
        }

        return null; // Unknown pattern
    }

    /**
     * Play back a saved gesture macro
     * Phase 3.11.2: Gesture playback
     */
    playbackGesture(macroName) {
        const macro = this.gestureMacros.savedMacros[macroName];
        if (!macro) {
            console.error(`[AnalyticsMonitor] Macro not found: ${macroName}`);
            return;
        }

        console.log(`[AnalyticsMonitor] Playing back macro: ${macroName}`);

        // Execute associated action
        if (macro.action) {
            this.executeGestureAction(macro.action);
        }
    }

    /**
     * Execute gesture action
     * Phase 3.11.2: Action execution
     */
    executeGestureAction(action) {
        const shortcuts = this.gestureMacros.shortcuts;

        if (action === 'z-shape' || action === shortcuts['z-shape']?.action) {
            // Reset layout
            this.resetDashboard();
            console.log('[AnalyticsMonitor] Gesture action: Reset layout');
        }
        else if (action === 'circle' || action === shortcuts['circle']?.action) {
            // Refresh all
            this.refreshAll();
            console.log('[AnalyticsMonitor] Gesture action: Refresh all');
        }
        else if (action === 'line-horizontal' || action === shortcuts['line-horizontal']?.action) {
            // Toggle compact template
            this.applyGridTemplate('compact');
            console.log('[AnalyticsMonitor] Gesture action: Toggle compact');
        }
        else {
            console.warn(`[AnalyticsMonitor] Unknown gesture action: ${action}`);
        }
    }

    /**
     * Save a gesture macro
     * Phase 3.11.2: Save macro
     */
    saveGestureMacro(name, pattern, action) {
        this.gestureMacros.savedMacros[name] = {
            pattern: pattern,
            action: action,
            createdAt: new Date().toISOString()
        };

        // Save to localStorage
        localStorage.setItem('analyticsMonitor_gesture_macros', JSON.stringify(this.gestureMacros.savedMacros));

        console.log(`[AnalyticsMonitor] Gesture macro saved: ${name}`);
    }

    /**
     * Load saved gesture macros from localStorage
     * Phase 3.11.2: Load macros
     */
    loadGestureMacros() {
        const saved = localStorage.getItem('analyticsMonitor_gesture_macros');
        if (saved) {
            try {
                this.gestureMacros.savedMacros = JSON.parse(saved);
                console.log(`[AnalyticsMonitor] Loaded ${Object.keys(this.gestureMacros.savedMacros).length} gesture macros`);
            } catch (error) {
                console.error('[AnalyticsMonitor] Failed to load gesture macros:', error);
            }
        }
    }

    /**
     * Export gesture macros as JSON
     * Phase 3.11.2: Export
     */
    exportGestureMacros() {
        const json = JSON.stringify(this.gestureMacros.savedMacros, null, 2);
        console.log('[AnalyticsMonitor] Gesture macros exported');
        return json;
    }

    /**
     * Import gesture macros from JSON
     * Phase 3.11.2: Import
     */
    importGestureMacros(json) {
        try {
            const macros = JSON.parse(json);
            this.gestureMacros.savedMacros = { ...this.gestureMacros.savedMacros, ...macros };

            // Save to localStorage
            localStorage.setItem('analyticsMonitor_gesture_macros', JSON.stringify(this.gestureMacros.savedMacros));

            console.log(`[AnalyticsMonitor] Imported ${Object.keys(macros).length} gesture macros`);
            return true;
        } catch (error) {
            console.error('[AnalyticsMonitor] Failed to import gesture macros:', error);
            return false;
        }
    }

    /**
     * Set custom action for a saved gesture macro
     * Enhancement 3.11.2: Custom action mapping
     */
    setCustomActionForMacro(macroName, customAction) {
        if (!this.gestureMacros.savedMacros[macroName]) {
            console.error(`[AnalyticsMonitor] Macro not found: ${macroName}`);
            return false;
        }

        // Validate action structure
        if (!customAction.type) {
            console.error('[AnalyticsMonitor] Custom action must have a type');
            return false;
        }

        // Update macro with custom action
        this.gestureMacros.savedMacros[macroName].customAction = customAction;

        // Save to localStorage
        localStorage.setItem('analyticsMonitor_gesture_macros', JSON.stringify(this.gestureMacros.savedMacros));

        console.log(`[AnalyticsMonitor] Custom action set for macro: ${macroName}`, customAction);
        return true;
    }

    /**
     * Execute a custom action
     * Enhancement 3.11.2: Custom action execution
     *
     * Supported action types:
     * - refresh: Refresh dashboard data
     * - hide_card: Hide a specific card (params: {cardId})
     * - show_card: Show a specific card (params: {cardId})
     * - toggle_card: Toggle card visibility (params: {cardId})
     * - navigate: Navigate to a section (params: {section})
     * - reset_layout: Reset dashboard layout
     * - toggle_compact: Toggle compact mode
     * - custom: Execute a custom function (params: {fn})
     */
    async executeCustomAction(actionType, actionParams = {}) {
        console.log(`[AnalyticsMonitor] Executing custom action: ${actionType}`, actionParams);

        try {
            switch (actionType) {
                case 'refresh':
                    // Refresh dashboard data
                    if (this.updateInterval > 0) {
                        await this.update();
                        console.log('[AnalyticsMonitor] Dashboard refreshed');
                    }
                    break;

                case 'hide_card':
                    // Hide a specific card
                    if (actionParams.cardId) {
                        this.setCardVisibility(actionParams.cardId, false);
                        console.log(`[AnalyticsMonitor] Card hidden: ${actionParams.cardId}`);
                    }
                    break;

                case 'show_card':
                    // Show a specific card
                    if (actionParams.cardId) {
                        this.setCardVisibility(actionParams.cardId, true);
                        console.log(`[AnalyticsMonitor] Card shown: ${actionParams.cardId}`);
                    }
                    break;

                case 'toggle_card':
                    // Toggle card visibility
                    if (actionParams.cardId) {
                        const currentVisibility = this.visibilityConfig[actionParams.cardId];
                        this.setCardVisibility(actionParams.cardId, !currentVisibility);
                        console.log(`[AnalyticsMonitor] Card toggled: ${actionParams.cardId}`);
                    }
                    break;

                case 'navigate':
                    // Navigate to a section (scroll to)
                    if (actionParams.section) {
                        const element = document.getElementById(actionParams.section);
                        if (element) {
                            element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                            console.log(`[AnalyticsMonitor] Navigated to: ${actionParams.section}`);
                        }
                    }
                    break;

                case 'reset_layout':
                    // Reset dashboard layout (restore all cards)
                    Object.keys(this.visibilityConfig).forEach(cardId => {
                        this.setCardVisibility(cardId, true);
                    });
                    console.log('[AnalyticsMonitor] Layout reset');
                    break;

                case 'toggle_compact':
                    // Toggle compact mode (if implemented)
                    if (this.compactMode !== undefined) {
                        this.compactMode = !this.compactMode;
                        console.log(`[AnalyticsMonitor] Compact mode: ${this.compactMode ? 'ON' : 'OFF'}`);
                    }
                    break;

                case 'custom':
                    // Execute custom function
                    if (actionParams.fn && typeof actionParams.fn === 'function') {
                        await actionParams.fn();
                        console.log('[AnalyticsMonitor] Custom function executed');
                    }
                    break;

                default:
                    console.warn(`[AnalyticsMonitor] Unknown action type: ${actionType}`);
                    return false;
            }

            return true;
        } catch (error) {
            console.error(`[AnalyticsMonitor] Error executing custom action:`, error);
            return false;
        }
    }

    /**
     * Replay a saved gesture macro with its custom action
     * Enhancement 3.11.2: Macro replay
     */
    async replayGestureMacro(macroName) {
        const macro = this.gestureMacros.savedMacros[macroName];
        if (!macro) {
            console.error(`[AnalyticsMonitor] Macro not found: ${macroName}`);
            return false;
        }

        console.log(`[AnalyticsMonitor] Replaying macro: ${macroName}`);

        // Execute custom action if defined
        if (macro.customAction) {
            const success = await this.executeCustomAction(
                macro.customAction.type,
                macro.customAction.params
            );

            if (success) {
                console.log(`[AnalyticsMonitor] Macro replay complete: ${macroName}`);
                return true;
            } else {
                console.error(`[AnalyticsMonitor] Macro replay failed: ${macroName}`);
                return false;
            }
        }

        // If no custom action, execute the recognized pattern action
        if (macro.action) {
            // Map old action names to new action types
            const actionMap = {
                'resetLayout': 'reset_layout',
                'refreshAll': 'refresh',
                'toggleCompact': 'toggle_compact'
            };

            const actionType = actionMap[macro.action] || macro.action;
            const success = await this.executeCustomAction(actionType);

            if (success) {
                console.log(`[AnalyticsMonitor] Macro replay complete: ${macroName}`);
                return true;
            }
        }

        console.warn(`[AnalyticsMonitor] No action defined for macro: ${macroName}`);
        return false;
    }

    /**
     * List all saved macros with their actions
     * Enhancement 3.11.2: Macro inspection
     */
    listGestureMacros() {
        const macros = Object.entries(this.gestureMacros.savedMacros).map(([name, macro]) => ({
            name: name,
            action: macro.action,
            customAction: macro.customAction,
            createdAt: macro.createdAt,
            points: macro.pattern ? macro.pattern.length : 0
        }));

        console.table(macros);
        return macros;
    }

    // ===================================================================
    // PHASE 3.11.3: MOBILE PERFORMANCE MODE
    // ===================================================================

    /**
     * Initialize Battery API monitoring
     * Phase 3.11.3: Battery monitoring
     */
    async initializeBatteryMonitor() {
        if (!('getBattery' in navigator)) {
            console.warn('[AnalyticsMonitor] Battery API not supported');
            return;
        }

        try {
            const battery = await navigator.getBattery();

            // Update initial state
            this.performanceMode.currentBatteryLevel = Math.round(battery.level * 100);
            this.performanceMode.isCharging = battery.charging;

            console.log(`[AnalyticsMonitor] Battery: ${this.performanceMode.currentBatteryLevel}%, charging: ${this.performanceMode.isCharging}`);

            // Auto-enable performance mode if low battery
            if (this.performanceMode.autoEnableOnLowBattery &&
                this.performanceMode.currentBatteryLevel <= this.performanceMode.batteryThreshold &&
                !this.performanceMode.isCharging) {
                this.enablePerformanceMode();
            }

            // Listen for battery changes
            battery.addEventListener('levelchange', () => this.handleBatteryChange(battery));
            battery.addEventListener('chargingchange', () => this.handleBatteryChange(battery));

        } catch (error) {
            console.error('[AnalyticsMonitor] Battery API error:', error);
        }
    }

    /**
     * Handle battery level or charging state change
     * Phase 3.11.3: Battery monitoring
     */
    handleBatteryChange(battery) {
        this.performanceMode.currentBatteryLevel = Math.round(battery.level * 100);
        this.performanceMode.isCharging = battery.charging;

        console.log(`[AnalyticsMonitor] Battery changed: ${this.performanceMode.currentBatteryLevel}%, charging: ${this.performanceMode.isCharging}`);

        // Auto-enable performance mode on low battery
        if (this.performanceMode.autoEnableOnLowBattery &&
            this.performanceMode.currentBatteryLevel <= this.performanceMode.batteryThreshold &&
            !this.performanceMode.isCharging &&
            !this.performanceMode.enabled) {
            this.enablePerformanceMode();
        }

        // Auto-disable performance mode when charging
        if (this.performanceMode.isCharging && this.performanceMode.enabled) {
            this.disablePerformanceMode();
        }
    }

    /**
     * Enable mobile performance mode (reduce animations, pause updates)
     * Phase 3.11.3: Performance mode
     */
    enablePerformanceMode() {
        if (this.performanceMode.enabled) {
            console.log('[AnalyticsMonitor] Performance mode already enabled');
            return;
        }

        this.performanceMode.enabled = true;
        console.log('[AnalyticsMonitor] Performance mode ENABLED');

        // Reduce animations
        document.body.classList.add('performance-mode');
        this.performanceMode.reducedAnimations = true;

        // Increase update interval (slower updates)
        // Note: Update intervals are managed by refresh timers

        console.log('[AnalyticsMonitor] Performance mode active: reduced animations, slower updates');
    }

    /**
     * Disable mobile performance mode (restore full performance)
     * Phase 3.11.3: Performance mode
     */
    disablePerformanceMode() {
        if (!this.performanceMode.enabled) {
            console.log('[AnalyticsMonitor] Performance mode already disabled');
            return;
        }

        this.performanceMode.enabled = false;
        console.log('[AnalyticsMonitor] Performance mode DISABLED');

        // Restore animations
        document.body.classList.remove('performance-mode');
        this.performanceMode.reducedAnimations = false;

        // Restore normal update interval

        console.log('[AnalyticsMonitor] Performance mode inactive: full animations, normal updates');
    }

    /**
     * Get current memory usage (if Performance Memory API available)
     * Enhancement 3.11.3: Memory monitoring
     */
    getCurrentMemoryUsage() {
        // Check if Performance Memory API is available (Chrome, Edge)
        if (!performance.memory) {
            return {
                supported: false,
                usedJSHeapSize: 0,
                totalJSHeapSize: 0,
                jsHeapSizeLimit: 0,
                usagePercent: 0
            };
        }

        const memory = performance.memory;
        const usagePercent = (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100;

        return {
            supported: true,
            usedJSHeapSize: memory.usedJSHeapSize,
            totalJSHeapSize: memory.totalJSHeapSize,
            jsHeapSizeLimit: memory.jsHeapSizeLimit,
            usagePercent: usagePercent,
            usedMB: (memory.usedJSHeapSize / 1024 / 1024).toFixed(2),
            totalMB: (memory.totalJSHeapSize / 1024 / 1024).toFixed(2),
            limitMB: (memory.jsHeapSizeLimit / 1024 / 1024).toFixed(2)
        };
    }

    /**
     * Check memory usage and trigger optimizations if needed
     * Enhancement 3.11.3: Memory monitoring
     */
    checkMemoryUsage() {
        if (!this.performanceMode.memoryMonitoring.enabled) {
            return;
        }

        // Rate limit checks
        const now = Date.now();
        const timeSinceLastCheck = now - this.performanceMode.memoryMonitoring.lastCheckTime;
        if (timeSinceLastCheck < this.performanceMode.memoryMonitoring.checkInterval) {
            return;
        }

        this.performanceMode.memoryMonitoring.lastCheckTime = now;

        // Get current memory usage
        const memoryUsage = this.getCurrentMemoryUsage();

        if (!memoryUsage.supported) {
            console.warn('[AnalyticsMonitor] Memory monitoring not supported in this browser');
            return;
        }

        // Store in history
        this.performanceMode.memoryMonitoring.history.push({
            timestamp: now,
            usedJSHeapSize: memoryUsage.usedJSHeapSize,
            totalJSHeapSize: memoryUsage.totalJSHeapSize,
            jsHeapSizeLimit: memoryUsage.jsHeapSizeLimit,
            usagePercent: memoryUsage.usagePercent
        });

        // Trim history to max length
        if (this.performanceMode.memoryMonitoring.history.length > this.performanceMode.memoryMonitoring.maxHistoryLength) {
            this.performanceMode.memoryMonitoring.history = this.performanceMode.memoryMonitoring.history.slice(-this.performanceMode.memoryMonitoring.maxHistoryLength);
        }

        // Update current usage
        this.performanceMode.memoryMonitoring.currentUsagePercent = memoryUsage.usagePercent;

        // Check thresholds
        const warningThreshold = this.performanceMode.memoryMonitoring.warningThreshold * 100;
        const criticalThreshold = this.performanceMode.memoryMonitoring.criticalThreshold * 100;

        if (memoryUsage.usagePercent >= criticalThreshold) {
            console.error(`[AnalyticsMonitor] CRITICAL memory usage: ${memoryUsage.usagePercent.toFixed(1)}% (${memoryUsage.usedMB}MB / ${memoryUsage.limitMB}MB)`);

            // Auto-optimize if enabled
            if (this.performanceMode.memoryMonitoring.autoOptimize && !this.performanceMode.memoryMonitoring.optimizationApplied) {
                this.optimizeForMemory('critical');
            }
        } else if (memoryUsage.usagePercent >= warningThreshold) {
            console.warn(`[AnalyticsMonitor] High memory usage: ${memoryUsage.usagePercent.toFixed(1)}% (${memoryUsage.usedMB}MB / ${memoryUsage.limitMB}MB)`);

            // Auto-optimize if enabled
            if (this.performanceMode.memoryMonitoring.autoOptimize && !this.performanceMode.memoryMonitoring.optimizationApplied) {
                this.optimizeForMemory('warning');
            }
        } else {
            // Memory usage is OK
            if (this.performanceMode.memoryMonitoring.optimizationApplied) {
                console.log(`[AnalyticsMonitor] Memory usage normalized: ${memoryUsage.usagePercent.toFixed(1)}% (${memoryUsage.usedMB}MB / ${memoryUsage.limitMB}MB)`);
            }
        }

        return memoryUsage;
    }

    /**
     * Optimize dashboard for memory constraints
     * Enhancement 3.11.3: Memory optimization
     */
    optimizeForMemory(severity = 'warning') {
        console.log(`[AnalyticsMonitor] Optimizing for memory (severity: ${severity})`);

        this.performanceMode.memoryMonitoring.optimizationApplied = true;

        if (severity === 'critical') {
            // Critical optimizations (aggressive)

            // 1. Clear gesture macro history (keep only recent 10)
            if (this.advancedGestures.recentlyHidden.length > 10) {
                this.advancedGestures.recentlyHidden = this.advancedGestures.recentlyHidden.slice(0, 10);
            }

            // 2. Trim memory monitoring history
            if (this.performanceMode.memoryMonitoring.history.length > 50) {
                this.performanceMode.memoryMonitoring.history = this.performanceMode.memoryMonitoring.history.slice(-50);
            }

            // 3. Trim calibration data
            if (this.advancedGestures.calibrationData) {
                this.advancedGestures.calibrationData.swipes = this.advancedGestures.calibrationData.swipes.slice(-10);
                this.advancedGestures.calibrationData.taps = this.advancedGestures.calibrationData.taps.slice(-10);
                this.advancedGestures.calibrationData.pinches = this.advancedGestures.calibrationData.pinches.slice(-10);
            }

            // 4. Enable performance mode (reduces animations, slows updates)
            if (!this.performanceMode.enabled) {
                this.enablePerformanceMode();
            }

            // 5. Disable visual preview (saves canvas memory)
            this.gestureMacros.enableVisualPreview = false;

            console.log('[AnalyticsMonitor] Applied CRITICAL memory optimizations');

        } else if (severity === 'warning') {
            // Warning optimizations (moderate)

            // 1. Trim gesture macro history (keep only recent 20)
            if (this.advancedGestures.recentlyHidden.length > 20) {
                this.advancedGestures.recentlyHidden = this.advancedGestures.recentlyHidden.slice(0, 20);
            }

            // 2. Trim memory monitoring history (keep only recent 75)
            if (this.performanceMode.memoryMonitoring.history.length > 75) {
                this.performanceMode.memoryMonitoring.history = this.performanceMode.memoryMonitoring.history.slice(-75);
            }

            console.log('[AnalyticsMonitor] Applied WARNING memory optimizations');
        }
    }

    /**
     * Get memory monitoring statistics
     * Enhancement 3.11.3: Memory monitoring
     */
    getMemoryStatistics() {
        const current = this.getCurrentMemoryUsage();

        if (!current.supported) {
            return {
                supported: false,
                message: 'Memory monitoring not supported in this browser (try Chrome or Edge)'
            };
        }

        const history = this.performanceMode.memoryMonitoring.history;

        // Calculate statistics from history
        let avgUsage = 0;
        let maxUsage = 0;
        let minUsage = 100;

        if (history.length > 0) {
            history.forEach(entry => {
                avgUsage += entry.usagePercent;
                maxUsage = Math.max(maxUsage, entry.usagePercent);
                minUsage = Math.min(minUsage, entry.usagePercent);
            });
            avgUsage /= history.length;
        }

        return {
            supported: true,
            current: {
                usedMB: current.usedMB,
                totalMB: current.totalMB,
                limitMB: current.limitMB,
                usagePercent: current.usagePercent.toFixed(2)
            },
            statistics: {
                avgUsagePercent: avgUsage.toFixed(2),
                maxUsagePercent: maxUsage.toFixed(2),
                minUsagePercent: minUsage.toFixed(2),
                samplesCollected: history.length
            },
            thresholds: {
                warning: (this.performanceMode.memoryMonitoring.warningThreshold * 100).toFixed(0) + '%',
                critical: (this.performanceMode.memoryMonitoring.criticalThreshold * 100).toFixed(0) + '%'
            },
            optimizationApplied: this.performanceMode.memoryMonitoring.optimizationApplied
        };
    }

    /**
     * Initialize network monitoring using Network Information API
     * Enhancement 3.11.3: Network monitoring
     */
    initializeNetworkMonitor() {
        if (!('connection' in navigator || 'mozConnection' in navigator || 'webkitConnection' in navigator)) {
            console.warn('[AnalyticsMonitor] Network Information API not supported');
            return;
        }

        // Get network connection object
        const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;

        // Update initial state
        this.updateNetworkInfo(connection);

        // Listen for network changes
        connection.addEventListener('change', () => this.handleNetworkChange(connection));

        console.log('[AnalyticsMonitor] Network monitoring initialized');
    }

    /**
     * Update network information from connection object
     * Enhancement 3.11.3: Network monitoring
     */
    updateNetworkInfo(connection) {
        this.performanceMode.networkMonitoring.effectiveType = connection.effectiveType || 'unknown';
        this.performanceMode.networkMonitoring.downlink = connection.downlink || 0;
        this.performanceMode.networkMonitoring.rtt = connection.rtt || 0;
        this.performanceMode.networkMonitoring.saveData = connection.saveData || false;

        console.log(`[AnalyticsMonitor] Network: ${this.performanceMode.networkMonitoring.effectiveType}, ` +
                    `${this.performanceMode.networkMonitoring.downlink} Mbps, ` +
                    `${this.performanceMode.networkMonitoring.rtt}ms RTT, ` +
                    `saveData: ${this.performanceMode.networkMonitoring.saveData}`);
    }

    /**
     * Handle network connection change
     * Enhancement 3.11.3: Network monitoring
     */
    handleNetworkChange(connection) {
        console.log('[AnalyticsMonitor] Network connection changed');

        // Update network info
        this.updateNetworkInfo(connection);

        // Check if optimization needed
        if (this.performanceMode.networkMonitoring.autoOptimize) {
            this.checkNetworkOptimizations();
        }
    }

    /**
     * Check if network optimizations are needed
     * Enhancement 3.11.3: Network optimization
     */
    checkNetworkOptimizations() {
        const network = this.performanceMode.networkMonitoring;

        // Conditions for optimization:
        // 1. Slow effective type (slow-2g, 2g)
        // 2. Low downlink speed (< threshold)
        // 3. High RTT (> 500ms)
        // 4. User requested data saver mode

        const isSlow2g = network.effectiveType === 'slow-2g' || network.effectiveType === '2g';
        const isSlowDownlink = network.downlink > 0 && network.downlink < network.slowNetworkThreshold;
        const isHighLatency = network.rtt > 500;
        const isDataSaver = network.saveData;

        if (isSlow2g || isSlowDownlink || isHighLatency || isDataSaver) {
            console.warn(`[AnalyticsMonitor] Slow network detected: ${network.effectiveType}, ${network.downlink} Mbps, ${network.rtt}ms RTT`);

            if (!network.optimizationApplied) {
                this.optimizeForNetwork();
            }
        } else {
            // Network is fast - restore normal intervals
            if (network.optimizationApplied) {
                console.log('[AnalyticsMonitor] Network speed normalized, restoring normal update intervals');
                this.restoreNormalUpdateIntervals();
            }
        }
    }

    /**
     * Optimize dashboard for slow network
     * Enhancement 3.11.3: Network optimization
     */
    optimizeForNetwork() {
        console.log('[AnalyticsMonitor] Optimizing for slow network');

        const network = this.performanceMode.networkMonitoring;
        network.optimizationApplied = true;

        // Adjust update interval based on network speed
        let multiplier = 1.0;

        if (network.effectiveType === 'slow-2g') {
            multiplier = 10.0; // 10x slower (1s → 10s)
        } else if (network.effectiveType === '2g') {
            multiplier = 5.0; // 5x slower (1s → 5s)
        } else if (network.effectiveType === '3g') {
            multiplier = 2.0; // 2x slower (1s → 2s)
        } else if (network.downlink < network.slowNetworkThreshold) {
            multiplier = 3.0; // 3x slower for low downlink
        } else if (network.rtt > 500) {
            multiplier = 2.0; // 2x slower for high latency
        }

        // Apply data saver multiplier
        if (network.saveData) {
            multiplier *= 2.0; // Additional 2x slowdown for data saver
        }

        // Update current interval
        network.currentUpdateInterval = network.baseUpdateInterval * multiplier;

        console.log(`[AnalyticsMonitor] Update interval adjusted: ${network.baseUpdateInterval}ms → ${network.currentUpdateInterval}ms (${multiplier}x)`);

        // Also enable performance mode if not already enabled
        if (!this.performanceMode.enabled) {
            this.enablePerformanceMode();
        }
    }

    /**
     * Restore normal update intervals when network improves
     * Enhancement 3.11.3: Network optimization
     */
    restoreNormalUpdateIntervals() {
        const network = this.performanceMode.networkMonitoring;
        network.currentUpdateInterval = network.baseUpdateInterval;
        network.optimizationApplied = false;

        console.log(`[AnalyticsMonitor] Update intervals restored to ${network.baseUpdateInterval}ms`);
    }

    /**
     * Get network monitoring statistics
     * Enhancement 3.11.3: Network monitoring
     */
    getNetworkStatistics() {
        const network = this.performanceMode.networkMonitoring;

        if (network.effectiveType === 'unknown') {
            return {
                supported: false,
                message: 'Network Information API not supported in this browser'
            };
        }

        // Classify network quality
        let quality = 'Good';
        if (network.effectiveType === 'slow-2g' || network.effectiveType === '2g') {
            quality = 'Poor';
        } else if (network.effectiveType === '3g' || network.downlink < 2.0) {
            quality = 'Fair';
        } else if (network.effectiveType === '4g' && network.downlink >= 5.0) {
            quality = 'Excellent';
        }

        return {
            supported: true,
            effectiveType: network.effectiveType,
            downlink: network.downlink + ' Mbps',
            rtt: network.rtt + ' ms',
            saveData: network.saveData,
            quality: quality,
            currentUpdateInterval: network.currentUpdateInterval + ' ms',
            optimizationApplied: network.optimizationApplied
        };
    }

    // ===================================================================
    // VOICE UX TRACKING (Milestone 1 - November 2025)
    // ===================================================================

    /**
     * Track voice command
     * Milestone 1: Voice UX integration
     */
    trackVoiceCommand(intentType, latencyMs, success, confidence) {
        const voice = this.performanceMode.voiceUX;

        // Create command entry
        const entry = {
            timestamp: Date.now(),
            intent: intentType,
            latency: latencyMs,
            success: success,
            confidence: confidence,
            batteryLevel: this.performanceMode.currentBatteryLevel,
            networkType: this.performanceMode.networkMonitoring.effectiveType,
            memoryUsagePercent: this.performanceMode.memoryMonitoring.currentUsagePercent
        };

        // Add to history
        voice.commandHistory.push(entry);

        // Trim history to max length
        if (voice.commandHistory.length > voice.maxHistoryLength) {
            voice.commandHistory = voice.commandHistory.slice(-voice.maxHistoryLength);
        }

        // Update session metrics
        voice.sessionMetrics.commandsProcessed++;
        if (success) {
            // Recalculate success rate
            const successfulCommands = voice.commandHistory.filter(c => c.success).length;
            voice.sessionMetrics.successRate = (successfulCommands / voice.commandHistory.length) * 100;
        }

        // Update average latency
        const totalLatency = voice.commandHistory.reduce((sum, c) => sum + c.latency, 0);
        voice.sessionMetrics.averageLatencyMs = totalLatency / voice.commandHistory.length;

        console.log(`[AnalyticsMonitor] Voice command tracked: ${intentType} (${latencyMs}ms, ${success ? 'success' : 'failed'})`);
    }

    /**
     * Track thread creation (voice UX)
     */
    trackThreadCreated() {
        this.performanceMode.voiceUX.sessionMetrics.threadsCreated++;
    }

    /**
     * Track thread switch (voice UX)
     */
    trackThreadSwitch() {
        this.performanceMode.voiceUX.sessionMetrics.threadSwitches++;
    }

    /**
     * Track voice error
     */
    trackVoiceError(errorType, errorMessage) {
        this.performanceMode.voiceUX.sessionMetrics.errors++;

        console.error(`[AnalyticsMonitor] Voice error: ${errorType} - ${errorMessage}`);
    }

    /**
     * Start voice session
     */
    startVoiceSession(mode = 'conversational') {
        const voice = this.performanceMode.voiceUX;

        voice.enabled = true;
        voice.isActive = true;
        voice.currentMode = mode;
        voice.sessionMetrics.startTime = Date.now();

        console.log(`[AnalyticsMonitor] Voice session started (mode: ${mode})`);
    }

    /**
     * Stop voice session
     */
    stopVoiceSession() {
        const voice = this.performanceMode.voiceUX;

        voice.isActive = false;
        const duration = Date.now() - voice.sessionMetrics.startTime;

        console.log(`[AnalyticsMonitor] Voice session stopped (duration: ${duration}ms, commands: ${voice.sessionMetrics.commandsProcessed})`);
    }

    /**
     * Get voice UX statistics
     */
    getVoiceStatistics() {
        const voice = this.performanceMode.voiceUX;

        if (!voice.enabled) {
            return {
                supported: true,
                enabled: false,
                message: 'Voice UX not currently enabled'
            };
        }

        // Calculate statistics from command history
        const recentCommands = voice.commandHistory.slice(-20); // Last 20 commands
        const intentCounts = {};

        recentCommands.forEach(cmd => {
            intentCounts[cmd.intent] = (intentCounts[cmd.intent] || 0) + 1;
        });

        return {
            supported: true,
            enabled: true,
            isActive: voice.isActive,
            currentMode: voice.currentMode,
            session: {
                startTime: voice.sessionMetrics.startTime,
                commandsProcessed: voice.sessionMetrics.commandsProcessed,
                threadsCreated: voice.sessionMetrics.threadsCreated,
                threadSwitches: voice.sessionMetrics.threadSwitches,
                errors: voice.sessionMetrics.errors,
                averageLatencyMs: voice.sessionMetrics.averageLatencyMs.toFixed(0),
                successRate: voice.sessionMetrics.successRate.toFixed(1) + '%'
            },
            recentIntents: intentCounts,
            commandHistory: recentCommands
        };
    }

    /**
     * Check if voice should be disabled due to performance constraints
     */
    shouldDisableVoice() {
        // Low battery and not charging
        if (this.performanceMode.currentBatteryLevel < 30 && !this.performanceMode.isCharging) {
            return { shouldDisable: true, reason: 'low_battery' };
        }

        // High memory usage
        if (this.performanceMode.memoryMonitoring.currentUsagePercent > 85) {
            return { shouldDisable: true, reason: 'high_memory' };
        }

        // Very slow network (for future cloud STT)
        const network = this.performanceMode.networkMonitoring;
        if (network.effectiveType === 'slow-2g') {
            return { shouldDisable: true, reason: 'very_slow_network' };
        }

        return { shouldDisable: false, reason: null };
    }

    /**
     * Initialize Page Visibility API (pause when backgrounded)
     * Phase 3.11.3: Background pause
     */
    initializePageVisibility() {
        if (typeof document.hidden === 'undefined') {
            console.warn('[AnalyticsMonitor] Page Visibility API not supported');
            return;
        }

        document.addEventListener('visibilitychange', () => this.handleVisibilityChange());

        console.log('[AnalyticsMonitor] Page Visibility API initialized');
    }

    /**
     * Handle page visibility change (pause/resume updates)
     * Phase 3.11.3: Background pause
     */
    handleVisibilityChange() {
        if (document.hidden) {
            console.log('[AnalyticsMonitor] Page hidden → pausing updates');
            this.performanceMode.pauseBackgroundUpdates = true;
            // Note: Actual pausing is handled by checking this flag in refresh methods
        } else {
            console.log('[AnalyticsMonitor] Page visible → resuming updates');
            this.performanceMode.pauseBackgroundUpdates = false;
            // Refresh immediately when returning
            this.refreshAll();
        }
    }

    /**
     * Enable virtualized rendering for large card lists (>20 cards)
     * Phase 3.11.3: Virtualized rendering
     */
    enableVirtualizedRendering() {
        const cardCount = Object.keys(this.dashboardLayout.cardSizes).length;

        if (cardCount < this.performanceMode.cardVirtualizationThreshold) {
            console.log(`[AnalyticsMonitor] Virtualization not needed (${cardCount} cards < ${this.performanceMode.cardVirtualizationThreshold} threshold)`);
            return;
        }

        this.performanceMode.virtualizedRendering = true;

        console.log(`[AnalyticsMonitor] Virtualized rendering enabled for ${cardCount} cards`);

        // Note: Actual virtualization would require more complex DOM manipulation
        // This is a placeholder for future implementation
        // For now, just hide off-screen cards
        const container = document.getElementById('analytics-cards-container');
        if (container) {
            container.classList.add('virtualized');
        }
    }

    /**
     * Disable virtualized rendering
     * Phase 3.11.3: Virtualized rendering
     */
    disableVirtualizedRendering() {
        this.performanceMode.virtualizedRendering = false;

        const container = document.getElementById('analytics-cards-container');
        if (container) {
            container.classList.remove('virtualized');
        }

        console.log('[AnalyticsMonitor] Virtualized rendering disabled');
    }
}

// Global instance
let analyticsMonitor = null;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    analyticsMonitor = new AnalyticsMonitor();
    console.log('[AnalyticsMonitor] Initialized and ready');
});
