/**
 * HoloLoom Interactive Diagrams Framework
 * Version: 1.0.0
 * Date: November 17, 2025
 *
 * Enhances Mermaid diagrams with:
 * - Click-to-expand node details
 * - Hover tooltips with code references
 * - Zoom/pan controls
 * - Search/filter functionality
 * - Permalink support
 *
 * Zero dependencies (except Mermaid.js from CDN)
 * Performance: <200ms initialization
 */

(function() {
  'use strict';

  // Configuration
  const CONFIG = {
    mermaidCDN: 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs',
    tooltipDelay: 300, // ms
    zoomStep: 0.1,
    minZoom: 0.5,
    maxZoom: 3.0,
    searchDebounce: 300 // ms
  };

  // Node metadata database (file references, descriptions, metrics)
  const NODE_METADATA = {
    // Layer 1: Input Processing
    'Input Processing': {
      file: 'HoloLoom/input/router.py:15',
      description: 'Multi-modal input routing system',
      metrics: { latency: '~20ms', complexity: 'Medium' }
    },
    'Text Processor': {
      file: 'HoloLoom/input/text_processor.py:30',
      description: 'Tokenization and entity extraction',
      metrics: { latency: '~15ms', complexity: 'Low' }
    },
    'Image Processor': {
      file: 'HoloLoom/input/image_processor.py:45',
      description: 'CLIP encoding and OCR (DeepSeek)',
      metrics: { latency: '~200ms', complexity: 'High' }
    },

    // Layer 2: Pattern Selection
    'Pattern Selection': {
      file: 'HoloLoom/loom/command.py:25',
      description: 'Selects BARE/FAST/FUSED execution mode',
      metrics: { latency: '~3ms', complexity: 'Low' }
    },

    // Layer 4: Memory Retrieval
    'Memory Retrieval': {
      file: 'HoloLoom/memory/cache.py:150',
      description: 'BM25 + semantic similarity search',
      metrics: { latency: '~45ms', complexity: 'Medium' }
    },
    'Yarn Graph': {
      file: 'HoloLoom/memory/graph.py:1',
      description: 'NetworkX-based knowledge graph',
      metrics: { latency: '~20ms', complexity: 'Medium' }
    },

    // Layer 5: Feature Extraction
    'Feature Extraction': {
      file: 'HoloLoom/resonance/shed.py:1',
      description: 'Motif + Embedding + Spectral extraction',
      metrics: { latency: '~60ms', complexity: 'High' }
    },
    'DotPlasma': {
      file: 'HoloLoom/documentation/types.py:45',
      description: 'Unified feature representation',
      metrics: { latency: '~5ms', complexity: 'Low' }
    },

    // Layer 7: Decision Making
    'Neural Policy': {
      file: 'HoloLoom/policy/unified.py:200',
      description: 'Transformer-based tool selection',
      metrics: { latency: '~35ms', complexity: 'High' }
    },
    'Thompson Sampling': {
      file: 'HoloLoom/policy/unified.py:450',
      description: 'Bayesian exploration/exploitation',
      metrics: { latency: '<1ms', complexity: 'Medium' }
    },
    'Convergence Engine': {
      file: 'HoloLoom/convergence/engine.py:1',
      description: 'Probability collapse to discrete action',
      metrics: { latency: '~5ms', complexity: 'Medium' }
    },

    // Layer 9: Learning
    'Reflection Buffer': {
      file: 'HoloLoom/reflection/buffer.py:1',
      description: 'Episodic memory for learning',
      metrics: { latency: '~5ms', complexity: 'Low' }
    },

    // Memory Backends
    'INMEMORY': {
      file: 'HoloLoom/memory/graph.py:1',
      description: 'NetworkX in-memory graph',
      metrics: { latency: '~20ms', persistence: 'No' }
    },
    'HYBRID': {
      file: 'HoloLoom/memory/neo4j_graph.py:1',
      description: 'Neo4j + Qdrant persistence',
      metrics: { latency: '~45ms', persistence: 'Yes' }
    },
    'HYPERSPACE': {
      file: 'HoloLoom/memory/hyperspace_backend.py:1',
      description: 'Gated multipass recursive',
      metrics: { latency: '~60ms', persistence: 'Yes' }
    }
  };

  // State management
  const STATE = {
    currentZoom: 1.0,
    panOffset: { x: 0, y: 0 },
    selectedNode: null,
    searchQuery: '',
    highlightedNodes: new Set()
  };

  /**
   * Initialize interactive features on all Mermaid diagrams
   */
  async function initializeInteractive() {
    console.log('[HoloLoom Interactive] Initializing...');
    const startTime = performance.now();

    // Initialize Mermaid
    await initializeMermaid();

    // Find all Mermaid diagrams
    const diagrams = document.querySelectorAll('.mermaid');
    console.log(`[HoloLoom Interactive] Found ${diagrams.length} diagrams`);

    // Enhance each diagram
    diagrams.forEach((diagram, index) => {
      enhanceDiagram(diagram, index);
    });

    // Initialize controls
    initializeControls();

    // Initialize search
    initializeSearch();

    const endTime = performance.now();
    console.log(`[HoloLoom Interactive] Initialized in ${(endTime - startTime).toFixed(2)}ms`);
  }

  /**
   * Initialize Mermaid library
   */
  async function initializeMermaid() {
    // Mermaid is loaded via CDN in HTML
    if (typeof mermaid === 'undefined') {
      console.error('[HoloLoom Interactive] Mermaid not loaded!');
      return;
    }

    mermaid.initialize({
      startOnLoad: true,
      theme: 'base',
      themeVariables: {
        primaryColor: '#FFD700',
        primaryTextColor: '#000',
        primaryBorderColor: '#000',
        lineColor: '#666',
        secondaryColor: '#90EE90',
        tertiaryColor: '#E6F3FF'
      },
      securityLevel: 'loose',
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis'
      }
    });
  }

  /**
   * Enhance a single diagram with interactive features
   */
  function enhanceDiagram(diagram, index) {
    // Wrap in container for zoom/pan
    const container = document.createElement('div');
    container.className = 'hololoom-diagram-container';
    container.dataset.diagramId = index;
    diagram.parentNode.insertBefore(container, diagram);
    container.appendChild(diagram);

    // Add zoom/pan wrapper
    const zoomWrapper = document.createElement('div');
    zoomWrapper.className = 'hololoom-zoom-wrapper';
    container.appendChild(zoomWrapper);
    zoomWrapper.appendChild(diagram);

    // Wait for Mermaid to render, then add interactivity
    setTimeout(() => {
      addNodeInteractivity(diagram, index);
      addZoomPanControls(container, zoomWrapper);
    }, 100);
  }

  /**
   * Add click/hover interactivity to nodes
   */
  function addNodeInteractivity(diagram, diagramIndex) {
    const svg = diagram.querySelector('svg');
    if (!svg) return;

    // Find all nodes (shapes with text)
    const nodes = svg.querySelectorAll('.node, g[id^="flowchart-"]');

    nodes.forEach((node, nodeIndex) => {
      const nodeId = `diagram-${diagramIndex}-node-${nodeIndex}`;
      node.setAttribute('data-node-id', nodeId);
      node.style.cursor = 'pointer';

      // Extract node label
      const textElement = node.querySelector('text, tspan');
      const label = textElement ? textElement.textContent.trim() : 'Unknown';

      // Get metadata
      const metadata = NODE_METADATA[label] || {
        file: 'Not specified',
        description: 'No description available',
        metrics: {}
      };

      // Click handler - expand details
      node.addEventListener('click', (e) => {
        e.stopPropagation();
        showNodeDetails(node, label, metadata, diagramIndex);
      });

      // Hover handler - show tooltip
      let hoverTimeout;
      node.addEventListener('mouseenter', () => {
        hoverTimeout = setTimeout(() => {
          showTooltip(node, label, metadata);
        }, CONFIG.tooltipDelay);
      });

      node.addEventListener('mouseleave', () => {
        clearTimeout(hoverTimeout);
        hideTooltip();
      });

      // Mark node as interactive
      node.classList.add('hololoom-interactive-node');
    });
  }

  /**
   * Show detailed panel for clicked node
   */
  function showNodeDetails(node, label, metadata, diagramIndex) {
    // Highlight selected node
    if (STATE.selectedNode) {
      STATE.selectedNode.classList.remove('hololoom-selected');
    }
    node.classList.add('hololoom-selected');
    STATE.selectedNode = node;

    // Create/update details panel
    let panel = document.getElementById('hololoom-details-panel');
    if (!panel) {
      panel = document.createElement('div');
      panel.id = 'hololoom-details-panel';
      panel.className = 'hololoom-panel';
      document.body.appendChild(panel);
    }

    // Render details
    panel.innerHTML = `
      <div class="hololoom-panel-header">
        <h3>${escapeHtml(label)}</h3>
        <button class="hololoom-close-btn" onclick="window.HoloLoomInteractive.closeDetailsPanel()">×</button>
      </div>
      <div class="hololoom-panel-content">
        <div class="hololoom-detail-section">
          <h4>📁 File Reference</h4>
          <code>${escapeHtml(metadata.file)}</code>
          <button class="hololoom-copy-btn" onclick="window.HoloLoomInteractive.copyToClipboard('${escapeHtml(metadata.file)}')">
            Copy
          </button>
        </div>

        <div class="hololoom-detail-section">
          <h4>📝 Description</h4>
          <p>${escapeHtml(metadata.description)}</p>
        </div>

        ${Object.keys(metadata.metrics).length > 0 ? `
          <div class="hololoom-detail-section">
            <h4>📊 Metrics</h4>
            <ul class="hololoom-metrics-list">
              ${Object.entries(metadata.metrics).map(([key, value]) => `
                <li><strong>${escapeHtml(key)}:</strong> ${escapeHtml(value)}</li>
              `).join('')}
            </ul>
          </div>
        ` : ''}

        <div class="hololoom-detail-section">
          <h4>🔗 Actions</h4>
          <button class="hololoom-action-btn" onclick="window.HoloLoomInteractive.jumpToCode('${escapeHtml(metadata.file)}')">
            Open in Editor
          </button>
          <button class="hololoom-action-btn" onclick="window.HoloLoomInteractive.shareNode('${escapeHtml(label)}', ${diagramIndex})">
            Share Link
          </button>
        </div>
      </div>
    `;

    panel.classList.add('hololoom-panel-visible');
  }

  /**
   * Show hover tooltip
   */
  function showTooltip(node, label, metadata) {
    let tooltip = document.getElementById('hololoom-tooltip');
    if (!tooltip) {
      tooltip = document.createElement('div');
      tooltip.id = 'hololoom-tooltip';
      tooltip.className = 'hololoom-tooltip';
      document.body.appendChild(tooltip);
    }

    const rect = node.getBoundingClientRect();
    tooltip.innerHTML = `
      <div class="hololoom-tooltip-header">${escapeHtml(label)}</div>
      <div class="hololoom-tooltip-body">
        <div><strong>File:</strong> <code>${escapeHtml(metadata.file)}</code></div>
        ${metadata.metrics.latency ? `<div><strong>Latency:</strong> ${escapeHtml(metadata.metrics.latency)}</div>` : ''}
      </div>
    `;

    tooltip.style.left = `${rect.left + rect.width / 2}px`;
    tooltip.style.top = `${rect.top - 10}px`;
    tooltip.classList.add('hololoom-tooltip-visible');
  }

  /**
   * Hide tooltip
   */
  function hideTooltip() {
    const tooltip = document.getElementById('hololoom-tooltip');
    if (tooltip) {
      tooltip.classList.remove('hololoom-tooltip-visible');
    }
  }

  /**
   * Add zoom/pan controls to diagram
   */
  function addZoomPanControls(container, zoomWrapper) {
    const controls = document.createElement('div');
    controls.className = 'hololoom-zoom-controls';
    controls.innerHTML = `
      <button class="hololoom-zoom-btn" data-action="zoom-in" title="Zoom In">+</button>
      <button class="hololoom-zoom-btn" data-action="zoom-out" title="Zoom Out">−</button>
      <button class="hololoom-zoom-btn" data-action="zoom-reset" title="Reset">⟲</button>
      <button class="hololoom-zoom-btn" data-action="fullscreen" title="Fullscreen">⛶</button>
    `;
    container.appendChild(controls);

    // Zoom button handlers
    controls.querySelectorAll('.hololoom-zoom-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const action = btn.dataset.action;
        handleZoomAction(action, zoomWrapper, container);
      });
    });

    // Mouse wheel zoom
    container.addEventListener('wheel', (e) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const delta = e.deltaY > 0 ? -CONFIG.zoomStep : CONFIG.zoomStep;
        zoomBy(delta, zoomWrapper);
      }
    });

    // Pan with mouse drag
    let isPanning = false;
    let startX, startY;

    container.addEventListener('mousedown', (e) => {
      if (e.target.closest('.hololoom-interactive-node')) return;
      isPanning = true;
      startX = e.clientX - STATE.panOffset.x;
      startY = e.clientY - STATE.panOffset.y;
      container.style.cursor = 'grabbing';
    });

    document.addEventListener('mousemove', (e) => {
      if (!isPanning) return;
      STATE.panOffset.x = e.clientX - startX;
      STATE.panOffset.y = e.clientY - startY;
      applyTransform(zoomWrapper);
    });

    document.addEventListener('mouseup', () => {
      if (isPanning) {
        isPanning = false;
        container.style.cursor = '';
      }
    });
  }

  /**
   * Handle zoom control actions
   */
  function handleZoomAction(action, zoomWrapper, container) {
    switch (action) {
      case 'zoom-in':
        zoomBy(CONFIG.zoomStep, zoomWrapper);
        break;
      case 'zoom-out':
        zoomBy(-CONFIG.zoomStep, zoomWrapper);
        break;
      case 'zoom-reset':
        STATE.currentZoom = 1.0;
        STATE.panOffset = { x: 0, y: 0 };
        applyTransform(zoomWrapper);
        break;
      case 'fullscreen':
        toggleFullscreen(container);
        break;
    }
  }

  /**
   * Zoom by delta
   */
  function zoomBy(delta, zoomWrapper) {
    const newZoom = Math.max(CONFIG.minZoom, Math.min(CONFIG.maxZoom, STATE.currentZoom + delta));
    STATE.currentZoom = newZoom;
    applyTransform(zoomWrapper);
  }

  /**
   * Apply transform to zoom wrapper
   */
  function applyTransform(zoomWrapper) {
    zoomWrapper.style.transform = `translate(${STATE.panOffset.x}px, ${STATE.panOffset.y}px) scale(${STATE.currentZoom})`;
  }

  /**
   * Toggle fullscreen
   */
  function toggleFullscreen(container) {
    if (!document.fullscreenElement) {
      container.requestFullscreen().catch(err => {
        console.error(`[HoloLoom Interactive] Fullscreen error: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
    }
  }

  /**
   * Initialize global controls (search, etc.)
   */
  function initializeControls() {
    // Create control bar if it doesn't exist
    if (document.getElementById('hololoom-controls')) return;

    const controls = document.createElement('div');
    controls.id = 'hololoom-controls';
    controls.className = 'hololoom-controls';
    controls.innerHTML = `
      <div class="hololoom-control-section">
        <input
          type="text"
          id="hololoom-search"
          class="hololoom-search-input"
          placeholder="Search diagrams... (Ctrl+F)"
        />
        <button id="hololoom-search-clear" class="hololoom-btn">Clear</button>
      </div>
      <div class="hololoom-control-section">
        <button id="hololoom-collapse-all" class="hololoom-btn">Collapse All</button>
        <button id="hololoom-export-svg" class="hololoom-btn">Export SVG</button>
      </div>
    `;

    // Insert at top of body
    document.body.insertBefore(controls, document.body.firstChild);
  }

  /**
   * Initialize search functionality
   */
  function initializeSearch() {
    const searchInput = document.getElementById('hololoom-search');
    const clearBtn = document.getElementById('hololoom-search-clear');

    if (!searchInput) return;

    let searchTimeout;
    searchInput.addEventListener('input', (e) => {
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => {
        performSearch(e.target.value);
      }, CONFIG.searchDebounce);
    });

    clearBtn.addEventListener('click', () => {
      searchInput.value = '';
      performSearch('');
    });

    // Keyboard shortcut: Ctrl+F
    document.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        e.preventDefault();
        searchInput.focus();
      }
    });
  }

  /**
   * Perform search across all diagrams
   */
  function performSearch(query) {
    STATE.searchQuery = query.toLowerCase();
    STATE.highlightedNodes.clear();

    if (!query) {
      // Clear all highlights
      document.querySelectorAll('.hololoom-search-highlight').forEach(node => {
        node.classList.remove('hololoom-search-highlight');
      });
      return;
    }

    // Find matching nodes
    document.querySelectorAll('.hololoom-interactive-node').forEach(node => {
      const textElement = node.querySelector('text, tspan');
      const label = textElement ? textElement.textContent.trim().toLowerCase() : '';

      if (label.includes(STATE.searchQuery)) {
        node.classList.add('hololoom-search-highlight');
        STATE.highlightedNodes.add(node);
      } else {
        node.classList.remove('hololoom-search-highlight');
      }
    });

    console.log(`[HoloLoom Interactive] Search: "${query}" - ${STATE.highlightedNodes.size} matches`);
  }

  /**
   * Utility: Escape HTML
   */
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Public API
  window.HoloLoomInteractive = {
    initialize: initializeInteractive,

    closeDetailsPanel: function() {
      const panel = document.getElementById('hololoom-details-panel');
      if (panel) {
        panel.classList.remove('hololoom-panel-visible');
      }
      if (STATE.selectedNode) {
        STATE.selectedNode.classList.remove('hololoom-selected');
        STATE.selectedNode = null;
      }
    },

    copyToClipboard: function(text) {
      navigator.clipboard.writeText(text).then(() => {
        alert(`Copied: ${text}`);
      });
    },

    jumpToCode: function(fileRef) {
      // In real implementation, this would open VS Code or editor
      alert(`Jump to: ${fileRef}\n\n(In production, this opens your editor)`);
    },

    shareNode: function(label, diagramIndex) {
      const url = `${window.location.href.split('#')[0]}#diagram-${diagramIndex}-${encodeURIComponent(label)}`;
      navigator.clipboard.writeText(url).then(() => {
        alert(`Permalink copied: ${url}`);
      });
    }
  };

  // Auto-initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeInteractive);
  } else {
    initializeInteractive();
  }

})();
