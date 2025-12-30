/**
 * Dark Trace Dashboard
 * ====================
 *
 * D3.js-based real-time visualization dashboard for Dark Trace.
 * Provides layer heatmaps, correlation networks, and feature details.
 *
 * Usage:
 *   const dashboard = new DarkTraceDashboard({
 *     wsUrl: 'ws://localhost:8765',
 *     apiUrl: '/api/dark-trace',
 *   });
 *   dashboard.start();
 *
 * Created: 2025-12-28
 */

class DarkTraceDashboard {
    /**
     * Create a new Dark Trace Dashboard.
     * @param {Object} config - Configuration options
     * @param {string} config.wsUrl - WebSocket URL for streaming
     * @param {string} config.apiUrl - REST API URL
     * @param {number} config.updateInterval - Update interval in ms
     */
    constructor(config = {}) {
        this.config = {
            wsUrl: config.wsUrl || 'ws://localhost:8765',
            apiUrl: config.apiUrl || '/api/dark-trace',
            updateInterval: config.updateInterval || 1000,
            maxHistoryPoints: config.maxHistoryPoints || 60,
            heatmapRows: config.heatmapRows || 10,
            heatmapCols: config.heatmapCols || 50,
            networkNodeLimit: config.networkNodeLimit || 50,
        };

        // State
        this.connected = false;
        this.ws = null;
        this.features = [];
        this.activations = new Map();
        this.correlations = [];
        this.history = {
            activeFeatures: [],
            meanActivation: [],
            maxActivation: [],
            sparsity: [],
        };

        // D3 elements
        this.heatmapSvg = null;
        this.networkSvg = null;
        this.tooltip = null;

        // Renderers
        this.heatmapRenderer = null;
        this.networkRenderer = null;

        // Bind methods
        this._onMessage = this._onMessage.bind(this);
        this._onOpen = this._onOpen.bind(this);
        this._onClose = this._onClose.bind(this);
        this._onError = this._onError.bind(this);
    }

    /**
     * Start the dashboard.
     */
    start() {
        this._initializeUI();
        this._connectWebSocket();
        this._startUpdateLoop();
    }

    /**
     * Stop the dashboard.
     */
    stop() {
        if (this.ws) {
            this.ws.close();
        }
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }

    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------

    _initializeUI() {
        // Get containers
        const heatmapContainer = document.getElementById('heatmapContainer');
        const networkContainer = document.getElementById('networkContainer');
        this.tooltip = document.getElementById('tooltip');

        // Initialize heatmap
        if (heatmapContainer) {
            this.heatmapRenderer = new LayerHeatmapRenderer(heatmapContainer, {
                rows: this.config.heatmapRows,
                cols: this.config.heatmapCols,
                onCellHover: this._onHeatmapHover.bind(this),
            });
        }

        // Initialize network
        if (networkContainer) {
            this.networkRenderer = new CorrelationNetworkRenderer(networkContainer, {
                nodeLimit: this.config.networkNodeLimit,
                onNodeClick: this._onNetworkNodeClick.bind(this),
            });
        }

        // Initialize sparklines
        this._initSparklines();

        // Set up event listeners
        this._setupEventListeners();
    }

    _initSparklines() {
        const sparklineIds = [
            'sparklineActive',
            'sparklineMean',
            'sparklineMax',
            'sparklineSparsity',
        ];

        sparklineIds.forEach(id => {
            const svg = d3.select(`#${id}`);
            if (svg.node()) {
                const width = svg.node().clientWidth || 100;
                const height = svg.node().clientHeight || 30;

                svg.attr('viewBox', `0 0 ${width} ${height}`)
                   .attr('preserveAspectRatio', 'none');
            }
        });
    }

    _setupEventListeners() {
        // Layer filter buttons
        document.querySelectorAll('[data-layer]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('[data-layer]').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this._filterLayer(e.target.dataset.layer);
            });
        });

        // Sort buttons
        document.querySelectorAll('[data-sort]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('[data-sort]').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this._sortFeatures(e.target.dataset.sort);
            });
        });

        // Network controls
        const resetZoom = document.getElementById('resetZoom');
        if (resetZoom) {
            resetZoom.addEventListener('click', () => {
                if (this.networkRenderer) {
                    this.networkRenderer.resetZoom();
                }
            });
        }

        const toggleLabels = document.getElementById('toggleLabels');
        if (toggleLabels) {
            toggleLabels.addEventListener('click', () => {
                if (this.networkRenderer) {
                    this.networkRenderer.toggleLabels();
                }
            });
        }
    }

    // -------------------------------------------------------------------------
    // WebSocket
    // -------------------------------------------------------------------------

    _connectWebSocket() {
        try {
            this.ws = new WebSocket(this.config.wsUrl);
            this.ws.onopen = this._onOpen;
            this.ws.onclose = this._onClose;
            this.ws.onerror = this._onError;
            this.ws.onmessage = this._onMessage;
        } catch (e) {
            console.error('WebSocket connection failed:', e);
            this._updateStatus(false);
        }
    }

    _onOpen() {
        console.log('WebSocket connected');
        this.connected = true;
        this._updateStatus(true);

        // Subscribe to activations
        this.ws.send(JSON.stringify({
            type: 'subscribe',
            data: {
                type: 'top_k',
                filters: { k: 100 },
            },
        }));
    }

    _onClose() {
        console.log('WebSocket disconnected');
        this.connected = false;
        this._updateStatus(false);

        // Attempt reconnect
        setTimeout(() => this._connectWebSocket(), 3000);
    }

    _onError(error) {
        console.error('WebSocket error:', error);
        this._updateStatus(false);
    }

    _onMessage(event) {
        try {
            const msg = JSON.parse(event.data);
            this._handleMessage(msg);
        } catch (e) {
            console.error('Failed to parse message:', e);
        }
    }

    _handleMessage(msg) {
        switch (msg.type) {
            case 'activations':
                this._handleActivations(msg.data);
                break;
            case 'activations_batch':
                this._handleActivationsBatch(msg.data);
                break;
            case 'layer_snapshot':
                this._handleLayerSnapshot(msg.data);
                break;
            case 'statistics':
                this._handleStatistics(msg.data);
                break;
            case 'alert':
                this._handleAlert(msg.data);
                break;
            case 'heartbeat':
                // Connection alive
                break;
            default:
                console.log('Unknown message type:', msg.type);
        }
    }

    _handleActivations(data) {
        const { layer, features } = data;

        features.forEach(f => {
            const key = `${layer}:${f.index}`;
            this.activations.set(key, {
                layer,
                index: f.index,
                activation: f.activation,
                timestamp: Date.now(),
            });
        });

        this._updateVisualization();
    }

    _handleActivationsBatch(data) {
        const { layer, features } = data;

        features.forEach(f => {
            const key = `${layer}:${f.index}`;
            this.activations.set(key, {
                layer,
                index: f.index,
                activation: f.activation,
                timestamp: Date.now(),
            });
        });

        this._updateVisualization();
    }

    _handleLayerSnapshot(data) {
        // Update heatmap with complete layer data
        if (this.heatmapRenderer) {
            this.heatmapRenderer.updateLayer(data.layer, data.features);
        }
    }

    _handleStatistics(data) {
        // Update stats display
        this._updateStatistics(data);
    }

    _handleAlert(data) {
        console.warn('Alert:', data.message);
        // Could show a toast notification
    }

    // -------------------------------------------------------------------------
    // Update Loop
    // -------------------------------------------------------------------------

    _startUpdateLoop() {
        this.updateInterval = setInterval(() => {
            this._updateVisualization();
            this._updateHistory();
        }, this.config.updateInterval);
    }

    _updateVisualization() {
        const activationArray = Array.from(this.activations.values());

        // Update heatmap
        if (this.heatmapRenderer) {
            this.heatmapRenderer.update(activationArray);
        }

        // Update feature list
        this._updateFeatureList(activationArray);

        // Update statistics
        this._computeAndUpdateStats(activationArray);
    }

    _updateHistory() {
        const activationArray = Array.from(this.activations.values());
        const now = Date.now();

        // Compute metrics
        const activeCount = activationArray.filter(a => a.activation > 0.1).length;
        const activations = activationArray.map(a => a.activation);
        const mean = activations.length > 0
            ? activations.reduce((a, b) => a + b, 0) / activations.length
            : 0;
        const max = activations.length > 0
            ? Math.max(...activations)
            : 0;
        const sparsity = activations.length > 0
            ? activations.filter(a => a > 0).length / activations.length
            : 0;

        // Add to history
        this.history.activeFeatures.push({ time: now, value: activeCount });
        this.history.meanActivation.push({ time: now, value: mean });
        this.history.maxActivation.push({ time: now, value: max });
        this.history.sparsity.push({ time: now, value: sparsity });

        // Trim history
        const maxPoints = this.config.maxHistoryPoints;
        Object.keys(this.history).forEach(key => {
            if (this.history[key].length > maxPoints) {
                this.history[key] = this.history[key].slice(-maxPoints);
            }
        });

        // Update sparklines
        this._updateSparklines();
    }

    _computeAndUpdateStats(activations) {
        const activeCount = activations.filter(a => a.activation > 0.1).length;
        const values = activations.map(a => a.activation);
        const mean = values.length > 0
            ? values.reduce((a, b) => a + b, 0) / values.length
            : 0;
        const max = values.length > 0
            ? Math.max(...values)
            : 0;
        const sparsity = values.length > 0
            ? (1 - values.filter(v => v > 0).length / values.length) * 100
            : 100;

        // Update DOM
        const statActive = document.getElementById('statActiveFeatures');
        const statMean = document.getElementById('statMeanActivation');
        const statMax = document.getElementById('statMaxActivation');
        const statSparsity = document.getElementById('statSparsity');

        if (statActive) statActive.textContent = activeCount;
        if (statMean) statMean.textContent = mean.toFixed(3);
        if (statMax) statMax.textContent = max.toFixed(3);
        if (statSparsity) statSparsity.textContent = `${sparsity.toFixed(1)}%`;
    }

    _updateSparklines() {
        this._renderSparkline('sparklineActive', this.history.activeFeatures);
        this._renderSparkline('sparklineMean', this.history.meanActivation);
        this._renderSparkline('sparklineMax', this.history.maxActivation);
        this._renderSparkline('sparklineSparsity', this.history.sparsity);
    }

    _renderSparkline(id, data) {
        const svg = d3.select(`#${id}`);
        if (!svg.node() || data.length < 2) return;

        const width = svg.node().clientWidth || 100;
        const height = svg.node().clientHeight || 30;

        const values = data.map(d => d.value);
        const x = d3.scaleLinear()
            .domain([0, data.length - 1])
            .range([0, width]);
        const y = d3.scaleLinear()
            .domain([0, Math.max(...values) || 1])
            .range([height - 2, 2]);

        const line = d3.line()
            .x((d, i) => x(i))
            .y(d => y(d.value))
            .curve(d3.curveMonotoneX);

        const area = d3.area()
            .x((d, i) => x(i))
            .y0(height)
            .y1(d => y(d.value))
            .curve(d3.curveMonotoneX);

        svg.selectAll('*').remove();

        svg.append('path')
            .datum(data)
            .attr('class', 'sparkline-area')
            .attr('d', area);

        svg.append('path')
            .datum(data)
            .attr('class', 'sparkline-line')
            .attr('d', line);
    }

    _updateFeatureList(activations) {
        const container = document.getElementById('featureList');
        if (!container) return;

        // Sort by activation
        const sorted = activations
            .filter(a => a.activation > 0.01)
            .sort((a, b) => b.activation - a.activation)
            .slice(0, 50);

        container.innerHTML = sorted.map(a => {
            const activationClass = a.activation > 0.9 ? 'critical'
                : a.activation > 0.7 ? 'high'
                : '';
            return `
                <div class="feature-item" data-feature="${a.layer}:${a.index}">
                    <span class="feature-name">L${a.layer} F${a.index}</span>
                    <span class="feature-activation ${activationClass}">
                        ${a.activation.toFixed(4)}
                    </span>
                </div>
            `;
        }).join('');
    }

    _updateStatus(connected) {
        const dot = document.getElementById('statusDot');
        const text = document.getElementById('statusText');

        if (dot) {
            dot.classList.toggle('disconnected', !connected);
        }
        if (text) {
            text.textContent = connected ? 'Connected' : 'Disconnected';
        }
    }

    _updateStatistics(data) {
        // Handle statistics from server
        if (data.mean !== undefined) {
            const statMean = document.getElementById('statMeanActivation');
            if (statMean) statMean.textContent = data.mean.toFixed(3);
        }
    }

    // -------------------------------------------------------------------------
    // Event Handlers
    // -------------------------------------------------------------------------

    _onHeatmapHover(data, event) {
        if (!data) {
            this.tooltip.style.display = 'none';
            return;
        }

        this.tooltip.innerHTML = `
            <div class="tooltip-title">Layer ${data.layer} Feature ${data.index}</div>
            <div class="tooltip-value">${data.activation.toFixed(4)}</div>
        `;
        this.tooltip.style.display = 'block';
        this.tooltip.style.left = `${event.pageX + 10}px`;
        this.tooltip.style.top = `${event.pageY + 10}px`;
    }

    _onNetworkNodeClick(node) {
        console.log('Node clicked:', node);
        // Could show feature details panel
    }

    _filterLayer(layer) {
        if (this.heatmapRenderer) {
            this.heatmapRenderer.setLayerFilter(layer === 'all' ? null : parseInt(layer));
        }
    }

    _sortFeatures(sortBy) {
        // Re-render feature list with new sort
        const activations = Array.from(this.activations.values());

        if (sortBy === 'index') {
            activations.sort((a, b) => a.index - b.index);
        } else {
            activations.sort((a, b) => b.activation - a.activation);
        }

        this._updateFeatureList(activations);
    }
}

// =============================================================================
// Layer Heatmap Renderer
// =============================================================================

class LayerHeatmapRenderer {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            rows: options.rows || 10,
            cols: options.cols || 50,
            cellPadding: options.cellPadding || 1,
            onCellHover: options.onCellHover || (() => {}),
        };

        this.layerFilter = null;
        this._init();
    }

    _init() {
        const { width, height } = this.container.getBoundingClientRect();

        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        this.cellWidth = width / this.options.cols - this.options.cellPadding;
        this.cellHeight = height / this.options.rows - this.options.cellPadding;

        // Color scale
        this.colorScale = d3.scaleSequential(d3.interpolateYlOrRd)
            .domain([0, 1]);

        // Create cells
        this.cells = this.svg.selectAll('.heatmap-cell').data([]);
    }

    update(activations) {
        // Filter by layer if needed
        let filtered = activations;
        if (this.layerFilter !== null) {
            filtered = activations.filter(a => a.layer === this.layerFilter);
        }

        // Sort and limit
        const data = filtered
            .sort((a, b) => b.activation - a.activation)
            .slice(0, this.options.rows * this.options.cols);

        // Update cells
        this.cells = this.svg.selectAll('.heatmap-cell')
            .data(data, d => `${d.layer}:${d.index}`);

        // Enter
        const enter = this.cells.enter()
            .append('rect')
            .attr('class', 'heatmap-cell');

        // Update
        this.cells.merge(enter)
            .attr('x', (d, i) => (i % this.options.cols) * (this.cellWidth + this.options.cellPadding))
            .attr('y', (d, i) => Math.floor(i / this.options.cols) * (this.cellHeight + this.options.cellPadding))
            .attr('width', this.cellWidth)
            .attr('height', this.cellHeight)
            .attr('fill', d => this.colorScale(d.activation))
            .on('mouseover', (event, d) => this.options.onCellHover(d, event))
            .on('mouseout', () => this.options.onCellHover(null));

        // Exit
        this.cells.exit().remove();
    }

    updateLayer(layer, features) {
        const data = features.map((f, i) => ({
            layer,
            index: f.index,
            activation: f.activation,
        }));

        this.update(data);
    }

    setLayerFilter(layer) {
        this.layerFilter = layer;
    }
}

// =============================================================================
// Correlation Network Renderer
// =============================================================================

class CorrelationNetworkRenderer {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            nodeLimit: options.nodeLimit || 50,
            onNodeClick: options.onNodeClick || (() => {}),
        };

        this.showLabels = false;
        this._init();
    }

    _init() {
        const { width, height } = this.container.getBoundingClientRect();
        this.width = width;
        this.height = height;

        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        // Create groups
        this.linksGroup = this.svg.append('g').attr('class', 'links');
        this.nodesGroup = this.svg.append('g').attr('class', 'nodes');
        this.labelsGroup = this.svg.append('g').attr('class', 'labels');

        // Color scale for nodes
        this.colorScale = d3.scaleSequential(d3.interpolateViridis)
            .domain([0, 1]);

        // Create zoom behavior
        this.zoom = d3.zoom()
            .scaleExtent([0.5, 5])
            .on('zoom', (event) => {
                this.linksGroup.attr('transform', event.transform);
                this.nodesGroup.attr('transform', event.transform);
                this.labelsGroup.attr('transform', event.transform);
            });

        this.svg.call(this.zoom);

        // Force simulation
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(50))
            .force('charge', d3.forceManyBody().strength(-100))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(15));
    }

    update(nodes, links) {
        // Update simulation
        this.simulation.nodes(nodes);
        this.simulation.force('link').links(links);

        // Links
        const link = this.linksGroup.selectAll('.network-link')
            .data(links, d => `${d.source.id || d.source}-${d.target.id || d.target}`);

        link.enter()
            .append('line')
            .attr('class', 'network-link')
            .attr('stroke-width', d => Math.sqrt(d.value || 1));

        link.exit().remove();

        // Nodes
        const node = this.nodesGroup.selectAll('.network-node')
            .data(nodes, d => d.id);

        node.enter()
            .append('circle')
            .attr('class', 'network-node')
            .attr('r', d => 5 + (d.activation || 0) * 10)
            .attr('fill', d => this.colorScale(d.activation || 0))
            .on('click', (event, d) => this.options.onNodeClick(d))
            .call(this._drag());

        node.exit().remove();

        // Labels
        const label = this.labelsGroup.selectAll('.network-label')
            .data(this.showLabels ? nodes : [], d => d.id);

        label.enter()
            .append('text')
            .attr('class', 'network-label')
            .attr('dx', 12)
            .attr('dy', 4)
            .text(d => d.label || d.id);

        label.exit().remove();

        // Update positions on tick
        this.simulation.on('tick', () => {
            this.linksGroup.selectAll('.network-link')
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            this.nodesGroup.selectAll('.network-node')
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            this.labelsGroup.selectAll('.network-label')
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });

        this.simulation.alpha(1).restart();
    }

    _drag() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    }

    resetZoom() {
        this.svg.transition()
            .duration(750)
            .call(this.zoom.transform, d3.zoomIdentity);
    }

    toggleLabels() {
        this.showLabels = !this.showLabels;
        this.update(this.simulation.nodes(), this.simulation.force('link').links());
    }
}

// =============================================================================
// Feature Details Panel
// =============================================================================

class FeatureDetailsPanel {
    constructor(container) {
        this.container = container;
        this.selectedFeature = null;
    }

    show(feature) {
        this.selectedFeature = feature;
        this._render();
    }

    hide() {
        this.selectedFeature = null;
        this.container.innerHTML = '';
    }

    _render() {
        if (!this.selectedFeature) return;

        const f = this.selectedFeature;
        this.container.innerHTML = `
            <div class="feature-detail">
                <h3>Feature ${f.index}</h3>
                <p><strong>Layer:</strong> ${f.layer}</p>
                <p><strong>Activation:</strong> ${f.activation.toFixed(4)}</p>
                <p><strong>Label:</strong> ${f.label || 'Unlabeled'}</p>
            </div>
        `;
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        DarkTraceDashboard,
        LayerHeatmapRenderer,
        CorrelationNetworkRenderer,
        FeatureDetailsPanel,
    };
}
