/**
 * Enhanced Safety Dashboard
 * =========================
 * WebSocket-enabled safety dashboard with real-time alignment metrics.
 *
 * Features:
 * - WebSocket connection for live safety feed
 * - Risk level distribution chart
 * - Deception alerts panel
 * - Resource bounds gauges
 * - Integration with AlignmentMonitor metrics (E2.1)
 *
 * December 2025 - E2.2 Implementation
 */

class EnhancedSafetyDashboard extends SafetyDashboard {
    constructor(apiBase = 'http://localhost:8000', wsUrl = 'ws://localhost:8000/ws/safety') {
        super(apiBase);
        this.wsUrl = wsUrl;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 3000;
        this.feedItems = [];
        this.maxFeedItems = 100;
        this.alerts = [];
        this.alignmentMetrics = {
            safetyDecisions: {},
            deceptionFlags: {},
            convergenceViolations: {},
            autonomyActions: {},
            resourceUtilization: {}
        };
    }

    /**
     * Initialize enhanced dashboard with WebSocket
     */
    async initialize() {
        console.log('Initializing Enhanced Safety Dashboard...');

        // Initialize base dashboard
        await super.initialize();

        // Connect WebSocket
        this.connectWebSocket();

        // Fetch initial alignment metrics
        await this.fetchAlignmentMetrics();

        // Setup enhanced event listeners
        this.setupEnhancedEventListeners();

        // Start alignment metrics polling (every 5 seconds)
        this.alignmentInterval = setInterval(() => {
            this.fetchAlignmentMetrics();
        }, 5000);

        console.log('[OK] Enhanced Safety Dashboard initialized');
    }

    /**
     * Cleanup resources
     */
    destroy() {
        super.destroy();
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        if (this.alignmentInterval) {
            clearInterval(this.alignmentInterval);
            this.alignmentInterval = null;
        }
    }

    /**
     * Connect to WebSocket for live events
     */
    connectWebSocket() {
        console.log(`Connecting to WebSocket: ${this.wsUrl}`);

        try {
            this.ws = new WebSocket(this.wsUrl);

            this.ws.onopen = () => {
                console.log('[OK] WebSocket connected');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);

                // Subscribe to safety events
                this.ws.send(JSON.stringify({
                    type: 'subscribe',
                    pattern: 'safety:*'
                }));
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (e) {
                    console.error('Failed to parse WebSocket message:', e);
                }
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                this.attemptReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.updateConnectionStatus(false);
        }
    }

    /**
     * Attempt to reconnect WebSocket
     */
    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Max reconnect attempts reached, falling back to polling');
            return;
        }

        this.reconnectAttempts++;
        console.log(`Reconnecting in ${this.reconnectDelay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        setTimeout(() => {
            this.connectWebSocket();
        }, this.reconnectDelay);
    }

    /**
     * Handle WebSocket messages
     */
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'safety_decision':
                this.handleSafetyDecision(data);
                break;
            case 'deception_flag':
                this.handleDeceptionFlag(data);
                break;
            case 'convergence_violation':
                this.handleConvergenceViolation(data);
                break;
            case 'autonomy_action':
                this.handleAutonomyAction(data);
                break;
            case 'resource_update':
                this.handleResourceUpdate(data);
                break;
            case 'alert':
                this.handleAlert(data);
                break;
            case 'metrics_update':
                this.updateMetricsFromWS(data);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }

    /**
     * Handle safety decision event
     */
    handleSafetyDecision(data) {
        const riskLevel = data.risk_level || 'UNKNOWN';
        const outcome = data.outcome || 'unknown';
        const action = data.action || '';

        // Add to feed
        this.addFeedItem({
            type: outcome === 'blocked' ? 'danger' : (riskLevel === 'HIGH' || riskLevel === 'CRITICAL' ? 'warning' : 'success'),
            message: `Safety decision: ${action} [${riskLevel}] - ${outcome}`,
            timestamp: new Date()
        });

        // Update risk distribution
        this.updateRiskDistribution(riskLevel);

        // Update metrics
        this.updateDecisionMetrics();
    }

    /**
     * Handle deception flag event
     */
    handleDeceptionFlag(data) {
        const flagType = data.flag_type || 'UNKNOWN';
        const description = data.description || 'Deception behavior detected';

        // Add critical alert
        this.addAlert({
            level: 'critical',
            title: `Deception Detected: ${flagType}`,
            description: description,
            timestamp: new Date()
        });

        // Add to feed
        this.addFeedItem({
            type: 'danger',
            message: `[DECEPTION] ${flagType}: ${description}`,
            timestamp: new Date()
        });
    }

    /**
     * Handle convergence violation event
     */
    handleConvergenceViolation(data) {
        const violationType = data.violation_type || 'UNKNOWN';
        const description = data.description || 'Convergence violation detected';

        // Add warning alert
        this.addAlert({
            level: 'warning',
            title: `Convergence Violation: ${violationType}`,
            description: description,
            timestamp: new Date()
        });

        // Add to feed
        this.addFeedItem({
            type: 'warning',
            message: `[CONVERGENCE] ${violationType}: ${description}`,
            timestamp: new Date()
        });
    }

    /**
     * Handle autonomy action event
     */
    handleAutonomyAction(data) {
        const stepType = data.step_type || 'unknown';
        const description = data.description || '';

        // Add to feed (info level)
        this.addFeedItem({
            type: 'info',
            message: `[AUTONOMY] ${stepType}: ${description}`,
            timestamp: new Date()
        });
    }

    /**
     * Handle resource update event
     */
    handleResourceUpdate(data) {
        const resourceType = data.resource_type || 'UNKNOWN';
        const ratio = data.ratio || 0;

        this.updateResourceGauge(resourceType, ratio);
    }

    /**
     * Handle alert event
     */
    handleAlert(data) {
        this.addAlert({
            level: data.level === 'CRITICAL' ? 'critical' : 'warning',
            title: data.title || 'Alert',
            description: data.message || '',
            timestamp: new Date(data.timestamp * 1000 || Date.now())
        });
    }

    /**
     * Update metrics from WebSocket data
     */
    updateMetricsFromWS(data) {
        if (data.alignment_metrics) {
            this.alignmentMetrics = data.alignment_metrics;
            this.renderAlignmentMetrics();
        }
    }

    /**
     * Fetch alignment metrics from API
     */
    async fetchAlignmentMetrics() {
        try {
            const response = await fetch(`${this.apiBase}/safety/alignment-metrics`);
            if (response.ok) {
                const data = await response.json();
                this.alignmentMetrics = data;
                this.renderAlignmentMetrics();
            }
        } catch (error) {
            console.error('Failed to fetch alignment metrics:', error);
        }
    }

    /**
     * Render alignment metrics
     */
    renderAlignmentMetrics() {
        // Update risk distribution
        const decisions = this.alignmentMetrics.safety_decisions || {};
        this.renderRiskDistribution(decisions);

        // Update resource gauges
        const resources = this.alignmentMetrics.resource_utilization || {};
        for (const [type, ratio] of Object.entries(resources)) {
            this.updateResourceGauge(type, ratio);
        }

        // Update decision counts
        let totalDecisions = 0;
        let blockedCount = 0;
        for (const [key, count] of Object.entries(decisions)) {
            totalDecisions += count;
            if (key.includes('blocked')) {
                blockedCount += count;
            }
        }

        this.updateMetric('total-decisions', totalDecisions);
        this.updateMetric('blocked-count', blockedCount);
        const blockRate = totalDecisions > 0 ? ((blockedCount / totalDecisions) * 100).toFixed(1) : '0.0';
        this.updateMetric('block-rate', `${blockRate}%`);

        // Update alert count
        this.updateMetric('alert-count', this.alerts.length);
    }

    /**
     * Render risk distribution bars
     */
    renderRiskDistribution(decisions) {
        const levels = ['SAFE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];
        const counts = {};
        let maxCount = 0;

        // Count decisions by risk level
        for (const level of levels) {
            counts[level] = 0;
            for (const [key, count] of Object.entries(decisions)) {
                if (key.startsWith(level)) {
                    counts[level] += count;
                }
            }
            if (counts[level] > maxCount) {
                maxCount = counts[level];
            }
        }

        // Update bars
        for (const level of levels) {
            const percentage = maxCount > 0 ? (counts[level] / maxCount) * 100 : 0;
            const bar = document.getElementById(`risk-${level.toLowerCase()}`);
            const value = document.getElementById(`risk-${level.toLowerCase()}-value`);

            if (bar) {
                bar.style.width = `${percentage}%`;
            }
            if (value) {
                value.textContent = counts[level];
            }
        }
    }

    /**
     * Update risk distribution for single event
     */
    updateRiskDistribution(riskLevel) {
        const key = riskLevel.toUpperCase();
        if (!this.alignmentMetrics.safety_decisions) {
            this.alignmentMetrics.safety_decisions = {};
        }
        this.alignmentMetrics.safety_decisions[key] = (this.alignmentMetrics.safety_decisions[key] || 0) + 1;
        this.renderAlignmentMetrics();
    }

    /**
     * Update resource gauge
     */
    updateResourceGauge(resourceType, ratio) {
        const typeMap = {
            'API_CALLS': 'api',
            'COMPUTE': 'compute',
            'MEMORY': 'memory',
            'STORAGE': 'storage',
            'NETWORK': 'network'
        };

        const gaugeId = typeMap[resourceType] || resourceType.toLowerCase();
        const gauge = document.getElementById(`gauge-${gaugeId}`);

        if (gauge) {
            const percentage = Math.min(ratio * 100, 100);
            const angle = (percentage / 100) * 360;
            gauge.style.setProperty('--gauge-angle', `${angle}deg`);

            // Update color based on usage
            let color = 'var(--accent-green)';
            if (percentage >= 80) color = 'var(--accent-red)';
            else if (percentage >= 60) color = 'var(--accent-orange)';
            else if (percentage >= 40) color = 'var(--accent-yellow)';

            gauge.style.background = `conic-gradient(${color} 0deg, ${color} ${angle}deg, var(--bg-tertiary) ${angle}deg, var(--bg-tertiary) 360deg)`;

            // Update value text
            const valueEl = gauge.querySelector('.gauge-value');
            if (valueEl) {
                valueEl.textContent = `${percentage.toFixed(0)}%`;
            }
        }
    }

    /**
     * Update decision metrics
     */
    updateDecisionMetrics() {
        const totalEl = document.getElementById('total-decisions');
        if (totalEl) {
            const current = parseInt(totalEl.textContent) || 0;
            totalEl.textContent = current + 1;
        }
    }

    /**
     * Add item to live feed
     */
    addFeedItem(item) {
        this.feedItems.unshift(item);

        // Trim to max items
        if (this.feedItems.length > this.maxFeedItems) {
            this.feedItems.pop();
        }

        this.renderFeed();
    }

    /**
     * Render live feed
     */
    renderFeed() {
        const container = document.getElementById('live-feed');
        if (!container) return;

        if (this.feedItems.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm0-2a8 8 0 1 0 0-16 8 8 0 0 0 0 16zm-1-5h2v2h-2v-2zm0-8h2v6h-2V7z"/>
                    </svg>
                    <div>Waiting for events...</div>
                </div>
            `;
            return;
        }

        container.innerHTML = this.feedItems.slice(0, 20).map((item, index) => `
            <div class="feed-item ${index === 0 ? 'new' : ''}">
                ${this.getFeedIcon(item.type)}
                <div class="feed-content">
                    <div class="feed-message">${this.escapeHtml(item.message)}</div>
                    <div class="feed-time">${this.formatTime(item.timestamp)}</div>
                </div>
            </div>
        `).join('');

        // Update feed count
        const countEl = document.getElementById('feed-count');
        if (countEl) {
            countEl.textContent = `${this.feedItems.length} events`;
        }
    }

    /**
     * Get feed icon SVG
     */
    getFeedIcon(type) {
        const icons = {
            success: '<svg class="feed-icon success" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm-.997-6l7.07-7.071-1.414-1.414-5.656 5.657-2.829-2.829-1.414 1.414L11.003 16z"/></svg>',
            warning: '<svg class="feed-icon warning" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm-1-7v2h2v-2h-2zm0-8v6h2V7h-2z"/></svg>',
            danger: '<svg class="feed-icon danger" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm-1-7v2h2v-2h-2zm0-8v6h2V7h-2z"/></svg>',
            info: '<svg class="feed-icon info" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm-1-11v6h2v-6h-2zm0-4v2h2V7h-2z"/></svg>'
        };
        return icons[type] || icons.info;
    }

    /**
     * Add alert to alerts panel
     */
    addAlert(alert) {
        this.alerts.unshift(alert);
        this.renderAlerts();

        // Update alert count
        this.updateMetric('alert-count', this.alerts.length);
    }

    /**
     * Render alerts panel
     */
    renderAlerts() {
        const container = document.getElementById('alerts-container');
        if (!container) return;

        if (this.alerts.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm-.997-6l7.07-7.071-1.414-1.414-5.656 5.657-2.829-2.829-1.414 1.414L11.003 16z"/>
                    </svg>
                    <div>No active alerts</div>
                </div>
            `;
            return;
        }

        container.innerHTML = this.alerts.map(alert => `
            <div class="alert-item ${alert.level}">
                <div class="alert-content">
                    <div class="alert-title">${this.escapeHtml(alert.title)}</div>
                    <div class="alert-description">${this.escapeHtml(alert.description)}</div>
                </div>
                <div class="alert-time">${this.formatTime(alert.timestamp)}</div>
            </div>
        `).join('');
    }

    /**
     * Update connection status indicator
     */
    updateConnectionStatus(connected) {
        const dot = document.getElementById('ws-status');
        const text = document.getElementById('ws-status-text');

        if (dot) {
            dot.classList.toggle('connected', connected);
        }
        if (text) {
            text.textContent = connected ? 'Connected' : 'Disconnected';
        }
    }

    /**
     * Format timestamp
     */
    formatTime(date) {
        if (!(date instanceof Date)) {
            date = new Date(date);
        }
        return date.toLocaleTimeString();
    }

    /**
     * Setup enhanced event listeners
     */
    setupEnhancedEventListeners() {
        // Clear alerts button
        const clearBtn = document.getElementById('clear-alerts-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.alerts = [];
                this.renderAlerts();
                this.updateMetric('alert-count', 0);
            });
        }
    }
}

// Export for use
window.EnhancedSafetyDashboard = EnhancedSafetyDashboard;
