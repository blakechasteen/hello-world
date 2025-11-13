/**
 * Policy & Bandit Monitor
 * =======================
 * Real-time visualization of Thompson Sampling exploration and policy evolution.
 *
 * Features:
 * - Thompson Sampling arm performance over time (line charts)
 * - Exploration vs Exploitation balance
 * - Policy weight evolution (area charts)
 * - Tool selection distribution (bar charts)
 * - Confidence trajectory tracking
 *
 * Philosophy: Framework → Elegance → Real-Time Visibility
 */

class PolicyMonitor {
    constructor(apiBase = 'http://localhost:8000') {
        this.apiBase = apiBase;
        this.updateInterval = 3000; // 3 seconds for policy monitoring
        this.intervalId = null;

        // History buffers (last 50 data points for time series)
        this.maxHistory = 50;
        this.armHistory = new Map(); // tool → [{timestamp, expected_reward, alpha, beta}]
        this.policyWeightHistory = []; // [{timestamp, BARE, FAST, FUSED}]
        this.explorationHistory = []; // [{timestamp, exploration_rate, exploitation_rate}]
        this.toolSelectionHistory = new Map(); // tool → count
    }

    /**
     * Initialize policy monitor
     */
    async initialize() {
        console.log('Initializing Policy Monitor...');

        // Initial load
        await this.refreshPolicyData();

        // Start polling
        this.intervalId = setInterval(() => this.refreshPolicyData(), this.updateInterval);

        // Setup event listeners
        this.setupEventListeners();

        console.log('✓ Policy Monitor initialized');
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
        const refreshBtn = document.getElementById('policy-refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshPolicyData());
        }

        const resetBtn = document.getElementById('policy-reset-history-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetHistory());
        }
    }

    /**
     * Refresh policy data from API
     */
    async refreshPolicyData() {
        try {
            const response = await fetch(`${this.apiBase}/learning/status`);

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const data = await response.json();

            // Update visualizations
            this.updateThompsonSamplingChart(data.thompson_sampling_stats);
            this.updatePolicyWeightChart(data.policy_weights);
            this.updateExplorationBalance(data);
            this.updateToolDistribution(data);

        } catch (error) {
            console.error('Failed to refresh policy data:', error);
            this.showError(error.message);
        }
    }

    /**
     * Update Thompson Sampling arm performance chart
     */
    updateThompsonSamplingChart(stats) {
        const container = document.getElementById('thompson-chart-container');
        if (!container || !stats || !stats.arms) return;

        const timestamp = Date.now();

        // Update history for each arm
        Object.entries(stats.arms).forEach(([tool, arm]) => {
            if (!this.armHistory.has(tool)) {
                this.armHistory.set(tool, []);
            }

            const history = this.armHistory.get(tool);
            history.push({
                timestamp,
                expected_reward: arm.expected_reward,
                alpha: arm.alpha,
                beta: arm.beta
            });

            // Keep only last 50 points
            if (history.length > this.maxHistory) {
                history.shift();
            }
        });

        // Render line chart (Tufte-style)
        this.renderThompsonChart(container);
    }

    /**
     * Render Thompson Sampling line chart (Phase 3.3: Enhanced with win rates and sparklines)
     */
    renderThompsonChart(container) {
        if (this.armHistory.size === 0) {
            container.innerHTML = '<div class="empty-state">No Thompson Sampling data yet</div>';
            return;
        }

        // Get all tools
        const tools = Array.from(this.armHistory.keys());

        // Get min/max for Y-axis scaling
        let minReward = 1.0;
        let maxReward = 0.0;

        this.armHistory.forEach(history => {
            history.forEach(point => {
                minReward = Math.min(minReward, point.expected_reward);
                maxReward = Math.max(maxReward, point.expected_reward);
            });
        });

        const range = maxReward - minReward || 0.1;
        const yMin = Math.max(0, minReward - range * 0.1);
        const yMax = Math.min(1, maxReward + range * 0.1);

        // Chart dimensions
        const width = 600;
        const height = 250;
        const padding = { top: 20, right: 150, bottom: 30, left: 50 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        // Color palette (distinct colors for tools)
        const colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];

        let svg = `
            <svg width="${width}" height="${height}" style="font-family: monospace; font-size: 11px;">
                <!-- Y-axis -->
                <line x1="${padding.left}" y1="${padding.top}"
                      x2="${padding.left}" y2="${height - padding.bottom}"
                      stroke="#ccc" stroke-width="1"/>

                <!-- X-axis -->
                <line x1="${padding.left}" y1="${height - padding.bottom}"
                      x2="${width - padding.right}" y2="${height - padding.bottom}"
                      stroke="#ccc" stroke-width="1"/>

                <!-- Y-axis labels -->
                <text x="${padding.left - 5}" y="${padding.top}" text-anchor="end" fill="#666">${yMax.toFixed(2)}</text>
                <text x="${padding.left - 5}" y="${height - padding.bottom}" text-anchor="end" fill="#666">${yMin.toFixed(2)}</text>

                <!-- Title -->
                <text x="${padding.left}" y="15" fill="#333" font-weight="bold">Thompson Sampling: Expected Reward Over Time</text>
        `;

        // Draw lines for each tool with enhanced legend
        tools.forEach((tool, toolIdx) => {
            const history = this.armHistory.get(tool);
            if (history.length < 2) return;

            const color = colors[toolIdx % colors.length];
            const points = history.map((point, idx) => {
                const x = padding.left + (idx / (this.maxHistory - 1)) * chartWidth;
                const y = height - padding.bottom - ((point.expected_reward - yMin) / (yMax - yMin)) * chartHeight;
                return `${x},${y}`;
            }).join(' ');

            // Calculate win rate (Phase 3.3)
            const latest = history[history.length - 1];
            const totalTrials = latest.alpha + latest.beta;
            const winRate = totalTrials > 0 ? (latest.alpha / totalTrials) : 0;

            // Render inline sparkline for this tool
            const rewardValues = history.map(p => p.expected_reward);
            const sparkline = this.renderInlineSparkline(rewardValues, 30, 12, color);

            svg += `
                <!-- Line for ${tool} -->
                <polyline points="${points}"
                          fill="none" stroke="${color}" stroke-width="2"/>

                <!-- Enhanced Legend with win rate and sparkline -->
                <line x1="${width - padding.right + 10}"
                      y1="${padding.top + toolIdx * 25}"
                      x2="${width - padding.right + 25}"
                      y2="${padding.top + toolIdx * 25}"
                      stroke="${color}" stroke-width="2"/>
                <text x="${width - padding.right + 30}"
                      y="${padding.top + toolIdx * 25 + 4}"
                      fill="#333" font-size="10px">${tool}</text>
                <text x="${width - padding.right + 30}"
                      y="${padding.top + toolIdx * 25 + 15}"
                      fill="#666" font-size="9px">Win: ${(winRate * 100).toFixed(1)}%</text>
            `;
        });

        svg += '</svg>';
        container.innerHTML = svg;
    }

    /**
     * Render inline sparkline for legend (Phase 3.3)
     */
    renderInlineSparkline(values, width, height, color) {
        if (!values || values.length < 2) return '';

        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min || 1;

        const points = values.map((v, i) => {
            const x = (i / (values.length - 1)) * width;
            const y = height - ((v - min) / range) * height;
            return `${x.toFixed(1)},${y.toFixed(1)}`;
        }).join(' ');

        return `
            <polyline points="${points}"
                      fill="none" stroke="${color}" stroke-width="1" opacity="0.5"/>
        `;
    }

    /**
     * Update policy weight evolution chart
     */
    updatePolicyWeightChart(weights) {
        const container = document.getElementById('policy-weight-chart-container');
        if (!container || !weights) return;

        const timestamp = Date.now();

        // Update history
        this.policyWeightHistory.push({
            timestamp,
            BARE: weights.BARE || 0,
            FAST: weights.FAST || 0,
            FUSED: weights.FUSED || 0
        });

        // Keep only last 50 points
        if (this.policyWeightHistory.length > this.maxHistory) {
            this.policyWeightHistory.shift();
        }

        // Render stacked area chart
        this.renderPolicyWeightChart(container);
    }

    /**
     * Render policy weight stacked area chart
     */
    renderPolicyWeightChart(container) {
        if (this.policyWeightHistory.length === 0) {
            container.innerHTML = '<div class="empty-state">No policy weight data yet</div>';
            return;
        }

        const width = 600;
        const height = 200;
        const padding = { top: 30, right: 100, bottom: 30, left: 50 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        const colors = {
            BARE: '#3498db',
            FAST: '#2ecc71',
            FUSED: '#9b59b6'
        };

        let svg = `
            <svg width="${width}" height="${height}" style="font-family: monospace; font-size: 11px;">
                <!-- Y-axis -->
                <line x1="${padding.left}" y1="${padding.top}"
                      x2="${padding.left}" y2="${height - padding.bottom}"
                      stroke="#ccc" stroke-width="1"/>

                <!-- X-axis -->
                <line x1="${padding.left}" y1="${height - padding.bottom}"
                      x2="${width - padding.right}" y2="${height - padding.bottom}"
                      stroke="#ccc" stroke-width="1"/>

                <!-- Y-axis labels -->
                <text x="${padding.left - 5}" y="${padding.top}" text-anchor="end" fill="#666">1.0</text>
                <text x="${padding.left - 5}" y="${height - padding.bottom}" text-anchor="end" fill="#666">0.0</text>

                <!-- Title -->
                <text x="${padding.left}" y="20" fill="#333" font-weight="bold">Policy Weight Evolution (Stacked)</text>
        `;

        // Compute stacked areas (BARE at bottom, FUSED at top)
        const history = this.policyWeightHistory;

        ['BARE', 'FAST', 'FUSED'].forEach((mode, modeIdx) => {
            const points = [];

            // Top edge of area
            history.forEach((point, idx) => {
                const x = padding.left + (idx / (this.maxHistory - 1)) * chartWidth;

                // Cumulative weight up to this mode
                let cumulativeWeight = 0;
                for (let i = 0; i <= modeIdx; i++) {
                    const m = ['BARE', 'FAST', 'FUSED'][i];
                    cumulativeWeight += point[m];
                }

                const y = height - padding.bottom - cumulativeWeight * chartHeight;
                points.push(`${x},${y}`);
            });

            // Bottom edge of area (cumulative weight of previous modes)
            for (let idx = history.length - 1; idx >= 0; idx--) {
                const point = history[idx];
                const x = padding.left + (idx / (this.maxHistory - 1)) * chartWidth;

                let cumulativeWeight = 0;
                for (let i = 0; i < modeIdx; i++) {
                    const m = ['BARE', 'FAST', 'FUSED'][i];
                    cumulativeWeight += point[m];
                }

                const y = height - padding.bottom - cumulativeWeight * chartHeight;
                points.push(`${x},${y}`);
            }

            svg += `
                <polygon points="${points.join(' ')}"
                         fill="${colors[mode]}"
                         opacity="0.6"
                         stroke="${colors[mode]}"
                         stroke-width="1"/>
            `;
        });

        // Enhanced Legend with sparklines and current values (Phase 3.3)
        ['BARE', 'FAST', 'FUSED'].forEach((mode, idx) => {
            // Get weight history for this mode
            const weightValues = history.map(p => p[mode]);
            const currentWeight = weightValues[weightValues.length - 1];

            svg += `
                <rect x="${width - padding.right + 10}"
                      y="${padding.top + idx * 30}"
                      width="15" height="15"
                      fill="${colors[mode]}" opacity="0.6"/>
                <text x="${width - padding.right + 30}"
                      y="${padding.top + idx * 30 + 12}"
                      fill="#333" font-size="10px">${mode}: ${(currentWeight * 100).toFixed(0)}%</text>
            `;

            // Add inline sparkline (Phase 3.3)
            if (weightValues.length >= 2) {
                const sparklineY = padding.top + idx * 30 + 20;
                const sparklineX = width - padding.right + 10;

                const min = Math.min(...weightValues);
                const max = Math.max(...weightValues);
                const range = max - min || 0.1;

                const sparklinePoints = weightValues.map((v, i) => {
                    const x = sparklineX + (i / (weightValues.length - 1)) * 60;
                    const y = sparklineY - ((v - min) / range) * 10;
                    return `${x},${y}`;
                }).join(' ');

                svg += `
                    <polyline points="${sparklinePoints}"
                              fill="none" stroke="${colors[mode]}" stroke-width="1" opacity="0.5"/>
                `;
            }
        });

        svg += '</svg>';
        container.innerHTML = svg;
    }

    /**
     * Update exploration vs exploitation balance
     */
    updateExplorationBalance(data) {
        const container = document.getElementById('exploration-balance-container');
        if (!container) return;

        // Extract exploration rate from Thompson Sampling
        const ts = data.thompson_sampling_stats;
        if (!ts || !ts.arms) {
            container.innerHTML = '<div class="empty-state">No exploration data</div>';
            return;
        }

        // Calculate exploration rate (variance in arm rewards)
        const rewards = Object.values(ts.arms).map(arm => arm.expected_reward);
        const avgReward = rewards.reduce((a, b) => a + b, 0) / rewards.length;
        const variance = rewards.reduce((sum, r) => sum + Math.pow(r - avgReward, 2), 0) / rewards.length;

        // High variance = more exploration, low variance = more exploitation
        const explorationRate = Math.min(variance * 10, 1.0); // Scale to 0-1
        const exploitationRate = 1.0 - explorationRate;

        const timestamp = Date.now();
        this.explorationHistory.push({ timestamp, explorationRate, exploitationRate });

        // Keep only last 50 points
        if (this.explorationHistory.length > this.maxHistory) {
            this.explorationHistory.shift();
        }

        // Render radial gauge and sparkline (Phase 3.3: Enhanced)
        const html = `
            <div class="exploration-balance" style="display: flex; gap: 2rem; align-items: center;">
                <div style="flex: 1;">
                    ${this.renderRadialGauge(explorationRate, exploitationRate)}
                </div>
                <div style="flex: 1;">
                    <div class="balance-sparkline">
                        <div style="font-size: 0.875rem; color: var(--secondary); margin-bottom: 0.5rem; font-weight: 600;">
                            Balance History (Last ${this.explorationHistory.length} updates)
                        </div>
                        ${this.renderExplorationSparkline()}
                        <div style="margin-top: 0.5rem; font-size: 0.75rem; color: var(--secondary);">
                            ${this.getBalanceRecommendation(explorationRate, exploitationRate)}
                        </div>
                    </div>
                </div>
            </div>
        `;
        container.innerHTML = html;
    }

    /**
     * Render radial gauge for exploration/exploitation balance (Phase 3.3)
     */
    renderRadialGauge(exploration, exploitation) {
        const size = 180;
        const cx = size / 2;
        const cy = size / 2;
        const radius = 60;
        const strokeWidth = 20;

        // Calculate arc paths
        const explorationAngle = exploration * 180; // 0-180 degrees (left half)
        const exploitationAngle = exploitation * 180; // 0-180 degrees (right half)

        // Convert to radians and calculate end points
        const exploreEnd = this.polarToCartesian(cx, cy, radius, 180 - explorationAngle);
        const exploitEnd = this.polarToCartesian(cx, cy, radius, exploitationAngle);

        return `
            <svg width="${size}" height="${size}" style="display: block; margin: 0 auto;">
                <!-- Background circle -->
                <circle cx="${cx}" cy="${cy}" r="${radius}"
                        fill="none" stroke="#f0f0f0" stroke-width="${strokeWidth}"/>

                <!-- Exploration arc (left, orange) -->
                <path d="M ${cx - radius} ${cy}
                         A ${radius} ${radius} 0 0 1 ${exploreEnd.x} ${exploreEnd.y}"
                      fill="none" stroke="#f39c12" stroke-width="${strokeWidth}"
                      stroke-linecap="round"/>

                <!-- Exploitation arc (right, green) -->
                <path d="M ${cx + radius} ${cy}
                         A ${radius} ${radius} 0 0 0 ${exploitEnd.x} ${exploitEnd.y}"
                      fill="none" stroke="#2ecc71" stroke-width="${strokeWidth}"
                      stroke-linecap="round"/>

                <!-- Center text -->
                <text x="${cx}" y="${cy - 10}" text-anchor="middle" fill="#333" font-size="12px" font-weight="bold">
                    Balance
                </text>
                <text x="${cx}" y="${cy + 10}" text-anchor="middle" fill="#666" font-size="24px" font-weight="bold">
                    ${(exploration * 50).toFixed(0)}:${(exploitation * 50).toFixed(0)}
                </text>

                <!-- Labels -->
                <text x="20" y="${cy}" text-anchor="start" fill="#f39c12" font-size="11px" font-weight="600">
                    Explore
                </text>
                <text x="20" y="${cy + 15}" text-anchor="start" fill="#f39c12" font-size="14px" font-weight="bold">
                    ${(exploration * 100).toFixed(0)}%
                </text>

                <text x="${size - 20}" y="${cy}" text-anchor="end" fill="#2ecc71" font-size="11px" font-weight="600">
                    Exploit
                </text>
                <text x="${size - 20}" y="${cy + 15}" text-anchor="end" fill="#2ecc71" font-size="14px" font-weight="bold">
                    ${(exploitation * 100).toFixed(0)}%
                </text>
            </svg>
        `;
    }

    /**
     * Convert polar coordinates to cartesian (helper for radial gauge)
     */
    polarToCartesian(cx, cy, radius, angleDegrees) {
        const angleRadians = (angleDegrees - 90) * Math.PI / 180.0;
        return {
            x: cx + (radius * Math.cos(angleRadians)),
            y: cy + (radius * Math.sin(angleRadians))
        };
    }

    /**
     * Get balance recommendation based on current rates (Phase 3.3)
     */
    getBalanceRecommendation(exploration, exploitation) {
        if (exploration > 0.7) {
            return '⚠️ High exploration - consider focusing on best performers';
        } else if (exploration < 0.2) {
            return '⚠️ Low exploration - may miss better alternatives';
        } else {
            return '✓ Balanced exploration/exploitation';
        }
    }

    /**
     * Render exploration rate sparkline
     */
    renderExplorationSparkline() {
        if (this.explorationHistory.length === 0) return '';

        const values = this.explorationHistory.map(h => h.explorationRate);
        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min || 0.01;

        const chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        const sparkline = values.map(val => {
            const normalized = (val - min) / range;
            const idx = Math.floor(normalized * (chars.length - 1));
            return chars[idx];
        }).join('');

        return `<div style="font-family: monospace; font-size: 1.5rem; color: #f39c12;">${sparkline}</div>`;
    }

    /**
     * Update tool selection distribution
     */
    updateToolDistribution(data) {
        const container = document.getElementById('tool-distribution-container');
        if (!container) return;

        const ts = data.thompson_sampling_stats;
        if (!ts || !ts.arms) {
            container.innerHTML = '<div class="empty-state">No tool selection data</div>';
            return;
        }

        // Get tool selection counts from Thompson Sampling arms
        const tools = Object.entries(ts.arms)
            .map(([tool, arm]) => ({
                tool,
                count: arm.alpha + arm.beta - 2, // Total selections (subtract prior)
                success_rate: arm.alpha / (arm.alpha + arm.beta)
            }))
            .sort((a, b) => b.count - a.count);

        if (tools.length === 0) {
            container.innerHTML = '<div class="empty-state">No tool usage yet</div>';
            return;
        }

        // Render horizontal bar chart
        const maxCount = Math.max(...tools.map(t => t.count), 1);

        let html = '<div class="tool-distribution">';

        tools.forEach(tool => {
            const percentage = (tool.count / maxCount) * 100;
            const successColor = tool.success_rate >= 0.7 ? '#2ecc71' :
                                tool.success_rate >= 0.5 ? '#f39c12' : '#e74c3c';

            html += `
                <div class="tool-bar">
                    <div class="tool-name">${tool.tool}</div>
                    <div class="tool-bar-container">
                        <div class="tool-bar-fill" style="width: ${percentage}%; background: ${successColor};"></div>
                    </div>
                    <div class="tool-stats">
                        <span>${tool.count} selections</span>
                        <span style="margin-left: 0.5rem; color: ${successColor};">
                            ${(tool.success_rate * 100).toFixed(0)}% success
                        </span>
                    </div>
                </div>
            `;
        });

        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * Reset history
     */
    resetHistory() {
        this.armHistory.clear();
        this.policyWeightHistory = [];
        this.explorationHistory = [];
        this.toolSelectionHistory.clear();

        // Clear visualizations
        const containers = [
            'thompson-chart-container',
            'policy-weight-chart-container',
            'exploration-balance-container',
            'tool-distribution-container'
        ];

        containers.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.innerHTML = '<div class="empty-state">History cleared - collecting new data...</div>';
            }
        });

        console.log('Policy monitor history reset');
    }

    /**
     * Show error message
     */
    showError(message) {
        const container = document.getElementById('policy-error');
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
window.PolicyMonitor = PolicyMonitor;
