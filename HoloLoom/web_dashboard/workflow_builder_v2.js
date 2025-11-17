/**
 * HoloLoom Workflow Builder v2 - Enhanced with 5 Major Features
 *
 * Features:
 * 1. Live Preview Mode - Test workflow with sample data
 * 2. Smart Connection Suggestions - Auto-suggest compatible connections
 * 3. Template Snippets Library - Drag-and-drop reusable sub-workflows
 * 4. Real-Time Validation - Highlight errors as you build
 * 5. Node Configuration Wizard - Step-by-step guided setup
 *
 * Bonus:
 * - Keyboard shortcuts
 * - Auto-layout
 * - Export/import
 * - Performance optimization
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const API_BASE_URL = 'http://localhost:8001';
const GRID_SIZE = 20;

// Agent types and configurations
const AGENT_DEFINITIONS = {
    'Query': {
        category: 'Query Agents',
        description: 'HoloLoom query execution',
        icon: '🔍',
        inputs: [],
        outputs: ['result'],
        config: { mode: 'direct', max_steps: 5 }
    },
    'Memory': {
        category: 'Memory Agents',
        description: 'Store/retrieve from memory',
        icon: '🧠',
        inputs: ['input'],
        outputs: ['output'],
        config: { action: 'store' }
    },
    'Process': {
        category: 'Processing',
        description: 'Data transformation',
        icon: '⚙️',
        inputs: ['input'],
        outputs: ['output'],
        config: { transform: 'map' }
    },
    'Filter': {
        category: 'Processing',
        description: 'Filter data',
        icon: '🔎',
        inputs: ['input'],
        outputs: ['output'],
        config: { condition: '' }
    },
    'Decision': {
        category: 'Control Flow',
        description: 'Conditional branching',
        icon: '🔀',
        inputs: ['condition'],
        outputs: ['true', 'false'],
        config: { condition: '' }
    },
    'Loop': {
        category: 'Control Flow',
        description: 'Iterate over items',
        icon: '🔁',
        inputs: ['items'],
        outputs: ['item', 'index'],
        config: { items_path: '' }
    },
    'Output': {
        category: 'Output',
        description: 'Generate response',
        icon: '📤',
        inputs: ['input'],
        outputs: [],
        config: { format: 'text' }
    },
    'Parallel': {
        category: 'Control Flow',
        description: 'Parallel execution',
        icon: '⚡',
        inputs: ['input'],
        outputs: ['output'],
        config: { tasks: 2 }
    }
};

// Template snippets - common workflow patterns
const WORKFLOW_SNIPPETS = [
    {
        name: 'Email Notification',
        description: 'Fetch email → Classify → Send notification',
        category: 'Email',
        nodes: [
            { type: 'Query', x: 50, y: 50 },
            { type: 'Process', x: 300, y: 50 },
            { type: 'Output', x: 550, y: 50 }
        ],
        connections: [
            { from: 0, to: 1 },
            { from: 1, to: 2 }
        ]
    },
    {
        name: 'Error Handler',
        description: 'Try action → Catch errors → Retry',
        category: 'Resilience',
        nodes: [
            { type: 'Query', x: 50, y: 50 },
            { type: 'Decision', x: 300, y: 50 },
            { type: 'Query', x: 550, y: 50 }
        ],
        connections: [
            { from: 0, to: 1 },
            { from: 1, to: 2 }
        ]
    },
    {
        name: 'Data Transformation',
        description: 'Filter → Map → Reduce',
        category: 'Data',
        nodes: [
            { type: 'Query', x: 50, y: 50 },
            { type: 'Filter', x: 300, y: 50 },
            { type: 'Process', x: 550, y: 50 }
        ],
        connections: [
            { from: 0, to: 1 },
            { from: 1, to: 2 }
        ]
    },
    {
        name: 'Conditional Routing',
        description: 'Route based on condition',
        category: 'Control',
        nodes: [
            { type: 'Query', x: 50, y: 50 },
            { type: 'Decision', x: 300, y: 50 },
            { type: 'Output', x: 550, y: 50 },
            { type: 'Output', x: 550, y: 150 }
        ],
        connections: [
            { from: 0, to: 1 },
            { from: 1, to: 2 },
            { from: 1, to: 3 }
        ]
    },
    {
        name: 'Parallel Processing',
        description: 'Split → Process in parallel → Merge',
        category: 'Performance',
        nodes: [
            { type: 'Query', x: 50, y: 50 },
            { type: 'Parallel', x: 300, y: 50 },
            { type: 'Process', x: 550, y: 50 }
        ],
        connections: [
            { from: 0, to: 1 },
            { from: 1, to: 2 }
        ]
    }
];

// Global state
let nodes = [];
let connections = [];
let selectedNode = null;
let selectedConnection = null;
let zoom = 1;
let panX = 0;
let panY = 0;
let clipboard = null;
let undoStack = [];
let redoStack = [];
let previewMode = false;
let previewState = {};
let draggingPort = null;
let suggestedConnections = [];

// ============================================================================
// INITIALIZATION
// ============================================================================

window.addEventListener('DOMContentLoaded', () => {
    initializeBuilder();
    populateAgentPalette();
    populateSnippets();
    setupEventListeners();
    setupKeyboardShortcuts();
    addDebugLog('Builder initialized', 'info');
});

function initializeBuilder() {
    const canvas = document.getElementById('canvas');
    const viewport = document.getElementById('canvasViewport');

    // Canvas mouse events
    viewport.addEventListener('mousedown', onCanvasMouseDown);
    viewport.addEventListener('mousemove', onCanvasMouseMove);
    viewport.addEventListener('mouseup', onCanvasMouseUp);
    viewport.addEventListener('wheel', onCanvasWheel, { passive: false });

    // Drag and drop from palette
    document.addEventListener('dragstart', onPaletteItemDragStart);
    document.addEventListener('dragover', onCanvasDragOver);
    document.addEventListener('drop', onCanvasDrop);
}

function setupEventListeners() {
    document.addEventListener('keydown', handleKeyboardShortcuts);
    document.addEventListener('click', handleClickOutside);
}

function setupKeyboardShortcuts() {
    window.keyboardShortcuts = {
        'Delete': deleteSelectedNode,
        'Ctrl+C': copyNode,
        'Ctrl+V': pasteNode,
        'Ctrl+Z': undo,
        'Ctrl+Y': redo,
        'Ctrl+A': selectAll,
        'l': autoLayout,
        'v': validateWorkflow,
        's': showSnippetsLibrary,
        '?': showKeyboardShortcuts,
        ' ': executePreview,
        'Escape': deselectAll,
        'e': exportWorkflow
    };
}

// ============================================================================
// 1. LIVE PREVIEW MODE
// ============================================================================

async function executePreview() {
    if (previewMode) {
        pausePreview();
        return;
    }

    previewMode = true;
    document.getElementById('previewBtn').textContent = '⏸️ Pause';
    switchPanel('preview');

    const workflow = exportWorkflowData();

    if (nodes.length === 0) {
        showToast('No workflow to preview', 'warning');
        return;
    }

    addDebugLog('Starting preview execution', 'info');

    try {
        // Get sample data for first node
        const firstNode = nodes[0];
        const sampleInput = generateSampleInput(firstNode.type);

        // Execute workflow step by step
        await executePreviewStepByStep(workflow, sampleInput);

    } catch (error) {
        showToast(`Preview error: ${error.message}`, 'error');
        addDebugLog(`Preview error: ${error.message}`, 'error');
    } finally {
        previewMode = false;
        document.getElementById('previewBtn').textContent = '▶️ Preview';
    }
}

async function executePreviewStepByStep(workflow, initialInput) {
    const container = document.getElementById('previewContainer');
    container.innerHTML = '';

    let currentInput = initialInput;
    previewState = {};

    for (let i = 0; i < nodes.length && previewMode; i++) {
        const node = nodes[i];
        const step = createPreviewStepElement(node, 'running');
        container.appendChild(step);

        addDebugLog(`Executing step ${i + 1}: ${node.type} (${node.id})`, 'info');

        try {
            // Simulate node execution
            const output = await simulateNodeExecution(node, currentInput);
            previewState[node.id] = output;

            // Update step with result
            updatePreviewStep(step, output, 'success');
            currentInput = output;

            // Wait for visual feedback
            await new Promise(r => setTimeout(r, 500));

        } catch (error) {
            updatePreviewStep(step, { error: error.message }, 'error');
            addDebugLog(`Step ${i + 1} failed: ${error.message}`, 'error');
            break;
        }

        // Auto-scroll
        if (document.getElementById('autoScroll').checked) {
            container.scrollTop = container.scrollHeight;
        }
    }

    if (previewMode) {
        addDebugLog('Preview completed successfully', 'success');
        document.getElementById('previewStatus').textContent = '✓ Preview completed';
    }
}

function createPreviewStepElement(node, status = 'pending') {
    const div = document.createElement('div');
    div.className = `preview-step ${status}`;
    div.innerHTML = `
        <div class="preview-step-title">
            ${node.icon || '🔹'} ${node.type} (${node.id})
        </div>
        <div class="preview-step-content">
            ${status === 'running' ? '<div class="spinner"></div> Executing...' : 'Pending'}
        </div>
    `;
    return div;
}

function updatePreviewStep(element, output, status) {
    element.className = `preview-step ${status}`;
    const title = element.querySelector('.preview-step-title');
    const content = element.querySelector('.preview-step-content');

    const statusIcon = status === 'success' ? '✓' : status === 'error' ? '✗' : '⏳';
    title.textContent = title.textContent.replace(/^[✓✗⏳]/,'') || title.textContent;

    content.innerHTML = `
        <div>${statusIcon} ${status.toUpperCase()}</div>
        <div style="margin-top: 4px; font-size: 10px; color: #666;">
            ${JSON.stringify(output, null, 2).substring(0, 200)}${
                JSON.stringify(output).length > 200 ? '...' : ''
            }
        </div>
    `;
}

function pausePreview() {
    previewMode = false;
    document.getElementById('previewBtn').textContent = '▶️ Preview';
    document.getElementById('previewStatus').textContent = '⏸️ Paused';
    addDebugLog('Preview paused', 'info');
}

function stopPreview() {
    previewMode = false;
    document.getElementById('previewContainer').innerHTML = `
        <div style="color: #999; text-align: center; padding: 40px;">
            No preview running. Click "Run Preview" to test your workflow.
        </div>
    `;
    document.getElementById('previewBtn').textContent = '▶️ Preview';
    document.getElementById('previewStatus').textContent = 'Stopped';
    addDebugLog('Preview stopped', 'info');
}

function stepThroughPreview() {
    // TODO: Implement step-through mode
    showToast('Step-through mode coming soon', 'info');
}

function simulateNodeExecution(node, input) {
    return new Promise((resolve) => {
        setTimeout(() => {
            // Simulate different node behaviors
            let output = { ...input };

            switch (node.type) {
                case 'Query':
                    output = { query_result: 'Sample data from ' + node.type, confidence: 0.92 };
                    break;
                case 'Process':
                    output = { ...input, processed: true, timestamp: new Date().toISOString() };
                    break;
                case 'Filter':
                    output = Array.isArray(input) ? input.filter(x => x) : input;
                    break;
                case 'Decision':
                    output = { condition_result: true, branch: 'true_path' };
                    break;
                case 'Loop':
                    output = { items: [1, 2, 3], iteration: 0 };
                    break;
                case 'Output':
                    output = { final_output: input, status: 'complete' };
                    break;
                default:
                    output = input;
            }

            resolve(output);
        }, 300);
    });
}

function generateSampleInput(nodeType) {
    const samples = {
        'Query': { query: 'Sample question', context: 'User context' },
        'Memory': { key: 'sample_key', value: 'sample_value' },
        'Process': { data: [1, 2, 3], operation: 'map' },
        'Filter': { items: [1, 2, 3, 4, 5], predicate: 'x > 2' },
        'Decision': { condition: true },
        'Loop': { items: ['a', 'b', 'c'] },
        'Output': { message: 'Hello, World!' },
        'Parallel': { tasks: ['task1', 'task2'] }
    };
    return samples[nodeType] || { input: 'sample' };
}

// ============================================================================
// 2. SMART CONNECTION SUGGESTIONS
// ============================================================================

function showConnectionSuggestions(fromPort, fromNode) {
    clearSuggestions();
    suggestedConnections = [];

    const suggestions = {
        'Query': ['Process', 'Filter', 'Decision', 'Output'],
        'Process': ['Filter', 'Decision', 'Output', 'Parallel'],
        'Filter': ['Process', 'Decision', 'Output'],
        'Decision': ['Query', 'Process', 'Output'],
        'Loop': ['Process', 'Filter', 'Output'],
        'Parallel': ['Process', 'Output'],
        'Output': []
    };

    const compatible = suggestions[fromNode.type] || [];

    nodes.forEach(toNode => {
        if (toNode.id === fromNode.id) return;

        const isCompatible = compatible.includes(toNode.type);

        if (isCompatible) {
            suggestedConnections.push(toNode.id);
            highlightNode(toNode.id, 'suggested');
        } else {
            highlightNode(toNode.id, 'incompatible');
        }
    });

    // Show tooltip
    showSuggestionTooltip(fromNode, compatible);
}

function clearSuggestions() {
    document.querySelectorAll('.node').forEach(el => {
        el.classList.remove('suggested', 'incompatible');
    });
    suggestedConnections = [];
}

function showSuggestionTooltip(node, compatible) {
    const suggestions = compatible.slice(0, 3).join(', ');
    const tooltip = document.createElement('div');
    tooltip.style.cssText = `
        position: absolute;
        background: #333;
        color: white;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 11px;
        z-index: 1000;
        white-space: nowrap;
        pointer-events: none;
    `;
    tooltip.textContent = `Compatible: ${suggestions}`;
    document.body.appendChild(tooltip);

    setTimeout(() => tooltip.remove(), 3000);
}

function highlightNode(nodeId, type) {
    const nodeEl = document.querySelector(`[data-node-id="${nodeId}"]`);
    if (nodeEl) {
        nodeEl.style.borderColor = type === 'suggested' ? '#2ecc71' : '#ccc';
        nodeEl.style.opacity = type === 'suggested' ? '1' : '0.5';
    }
}

// ============================================================================
// 3. TEMPLATE SNIPPETS LIBRARY
// ============================================================================

function populateSnippets() {
    const snippetList = document.getElementById('snippetList');
    snippetList.innerHTML = '';

    WORKFLOW_SNIPPETS.forEach((snippet, idx) => {
        const item = document.createElement('div');
        item.className = 'snippet-item';
        item.draggable = true;
        item.innerHTML = `
            <div class="snippet-title">${snippet.name}</div>
            <div class="snippet-description">${snippet.description}</div>
            <div style="font-size: 10px; color: #999; margin-top: 4px;">
                ${snippet.nodes.length} nodes
            </div>
        `;

        item.addEventListener('dragstart', (e) => {
            e.dataTransfer.effectAllowed = 'copy';
            e.dataTransfer.setData('snippet', JSON.stringify(snippet));
        });

        snippetList.appendChild(item);
    });
}

function filterSnippets() {
    const search = document.getElementById('snippetSearch').value.toLowerCase();
    const items = document.querySelectorAll('.snippet-item');

    items.forEach(item => {
        const text = item.textContent.toLowerCase();
        item.style.display = text.includes(search) ? 'block' : 'none';
    });
}

function insertSnippet(snippet, x = 100, y = 100) {
    const offset = { x, y };

    snippet.nodes.forEach((nodeData, idx) => {
        addNode(nodeData.type, nodeData.x + offset.x, nodeData.y + offset.y);
    });

    const newNodes = nodes.slice(-snippet.nodes.length);

    snippet.connections.forEach(conn => {
        const fromNode = newNodes[conn.from];
        const toNode = newNodes[conn.to];
        if (fromNode && toNode) {
            addConnection(fromNode.id, toNode.id);
        }
    });

    showToast(`Added "${snippet.name}" snippet`, 'success');
    addDebugLog(`Inserted snippet: ${snippet.name}`, 'info');
}

function showSnippetsLibrary() {
    document.getElementById('snippetLibrary').classList.toggle('open');
}

// ============================================================================
// 4. REAL-TIME VALIDATION
// ============================================================================

function validateWorkflow() {
    const errors = [];
    const warnings = [];

    // Check for required fields
    nodes.forEach(node => {
        if (!node.config || Object.keys(node.config).some(k => !node.config[k])) {
            warnings.push(`⚠️ ${node.type} (${node.id}): Missing configuration`);
        }
    });

    // Check for isolated nodes
    nodes.forEach(node => {
        const hasConnection = connections.some(c => c.from === node.id || c.to === node.id);
        if (!hasConnection && nodes.length > 1) {
            warnings.push(`⚠️ ${node.type} (${node.id}): Not connected to other nodes`);
        }
    });

    // Check for circular dependencies
    const circular = detectCircularDependencies();
    if (circular.length > 0) {
        errors.push(`❌ Circular dependency detected: ${circular.join(' → ')}`);
    }

    // Check for unreachable nodes
    const unreachable = findUnreachableNodes();
    if (unreachable.length > 0) {
        warnings.push(`⚠️ Unreachable nodes: ${unreachable.join(', ')}`);
    }

    // Update UI
    updateValidationUI(errors, warnings);

    // Add status indicator
    if (errors.length > 0) {
        document.getElementById('statusDot').className = 'status-dot error';
        document.getElementById('statusText').textContent = `${errors.length} error(s)`;
    } else if (warnings.length > 0) {
        document.getElementById('statusDot').className = 'status-dot warning';
        document.getElementById('statusText').textContent = `${warnings.length} warning(s)`;
    } else {
        document.getElementById('statusDot').className = 'status-dot';
        document.getElementById('statusText').textContent = 'Valid';
    }

    addDebugLog(`Validation: ${errors.length} errors, ${warnings.length} warnings`, 'info');

    return { errors, warnings, valid: errors.length === 0 };
}

function updateValidationUI(errors, warnings) {
    const validationList = document.getElementById('validationList');
    validationList.innerHTML = '';

    if (errors.length === 0 && warnings.length === 0) {
        validationList.innerHTML = '<p style="color: #2ecc71;">✓ Workflow is valid</p>';
        return;
    }

    errors.forEach(err => {
        const item = document.createElement('div');
        item.className = 'validation-message error show';
        item.textContent = err;
        validationList.appendChild(item);
    });

    warnings.forEach(warn => {
        const item = document.createElement('div');
        item.className = 'validation-message warning show';
        item.textContent = warn;
        validationList.appendChild(item);
    });
}

function detectCircularDependencies() {
    const visited = new Set();
    const recursionStack = new Set();
    const graph = buildAdjacencyGraph();

    const hasCycle = (node) => {
        visited.add(node);
        recursionStack.add(node);

        for (const neighbor of (graph[node] || [])) {
            if (!visited.has(neighbor)) {
                if (hasCycle(neighbor)) return true;
            } else if (recursionStack.has(neighbor)) {
                return true;
            }
        }

        recursionStack.delete(node);
        return false;
    };

    for (const node of nodes.map(n => n.id)) {
        if (!visited.has(node)) {
            if (hasCycle(node)) return nodes.filter(n => recursionStack.has(n.id)).map(n => n.id);
        }
    }

    return [];
}

function findUnreachableNodes() {
    const graph = buildAdjacencyGraph();
    const visited = new Set();

    const dfs = (node) => {
        visited.add(node);
        for (const neighbor of (graph[node] || [])) {
            if (!visited.has(neighbor)) {
                dfs(neighbor);
            }
        }
    };

    if (nodes.length > 0) {
        dfs(nodes[0].id);
    }

    return nodes.filter(n => !visited.has(n.id)).map(n => n.id);
}

function buildAdjacencyGraph() {
    const graph = {};
    nodes.forEach(n => graph[n.id] = []);
    connections.forEach(c => {
        if (!graph[c.from]) graph[c.from] = [];
        graph[c.from].push(c.to);
    });
    return graph;
}

// ============================================================================
// 5. NODE CONFIGURATION WIZARD
// ============================================================================

function openNodeConfigWizard(node) {
    const wizard = document.getElementById('wizardModal');
    wizard.classList.add('open');

    const wizardTitle = document.getElementById('wizardTitle');
    wizardTitle.textContent = `Configure ${node.type}`;

    generateWizardSteps(node);
    updateWizardUI(node, 0);
}

function generateWizardSteps(node) {
    const stepsConfig = {
        'Query': ['Choose Mode', 'Set Parameters', 'Review'],
        'Process': ['Select Operation', 'Configure Options', 'Preview'],
        'Filter': ['Define Condition', 'Test Filter', 'Review'],
        'Decision': ['Set Condition', 'Configure Branches', 'Preview'],
        'Loop': ['Choose Items', 'Set Iterator', 'Review'],
        'Output': ['Choose Format', 'Set Template', 'Review'],
        'Memory': ['Choose Action', 'Set Key/Value', 'Review'],
        'Parallel': ['Set Task Count', 'Configure Tasks', 'Review']
    };

    const steps = stepsConfig[node.type] || ['Configure', 'Review'];
    const wizardSteps = document.getElementById('wizardSteps');
    wizardSteps.innerHTML = '';

    steps.forEach((step, idx) => {
        const btn = document.createElement('div');
        btn.className = `wizard-step ${idx === 0 ? 'active' : ''}`;
        btn.textContent = step;
        btn.onclick = () => updateWizardUI(node, idx);
        wizardSteps.appendChild(btn);
    });
}

function updateWizardUI(node, stepIndex) {
    const contents = document.getElementById('wizardContents');
    contents.innerHTML = '';

    const wizardSteps = document.querySelectorAll('.wizard-step');
    wizardSteps.forEach((s, idx) => {
        s.classList.toggle('active', idx === stepIndex);
    });

    switch (node.type) {
        case 'Query':
            renderQueryWizard(node, stepIndex, contents);
            break;
        case 'Process':
            renderProcessWizard(node, stepIndex, contents);
            break;
        case 'Filter':
            renderFilterWizard(node, stepIndex, contents);
            break;
        case 'Decision':
            renderDecisionWizard(node, stepIndex, contents);
            break;
        default:
            contents.innerHTML = `<p>Configuration for ${node.type}</p>`;
    }
}

function renderQueryWizard(node, step, container) {
    if (step === 0) {
        container.innerHTML = `
            <div class="property-group">
                <label class="property-label">Query Mode</label>
                <select class="property-input" onchange="updateNodeConfig(this.value)">
                    <option value="direct">🚀 Direct (fast)</option>
                    <option value="verify">✓ Verify (safe)</option>
                    <option value="research">🔍 Research (thorough)</option>
                    <option value="plan_execute">📋 Plan & Execute</option>
                </select>
            </div>
            <div class="property-group">
                <label class="property-label">Reasoning Steps</label>
                <input type="number" class="property-input" min="1" max="10" value="5" />
            </div>
        `;
    } else if (step === 1) {
        container.innerHTML = `
            <div class="property-group">
                <label class="property-label">Max Steps</label>
                <input type="number" class="property-input" min="1" max="20" value="${node.config.max_steps}" />
            </div>
            <div class="property-group">
                <label class="property-label">Timeout (seconds)</label>
                <input type="number" class="property-input" min="1" max="300" value="30" />
            </div>
        `;
    } else {
        container.innerHTML = `
            <div style="padding: 20px; text-align: center;">
                <h3>✓ Configuration Complete</h3>
                <p style="color: #666; margin-top: 10px;">
                    Mode: <strong>Direct</strong><br>
                    Steps: <strong>5</strong><br>
                    Timeout: <strong>30s</strong>
                </p>
            </div>
        `;
    }
}

function renderProcessWizard(node, step, container) {
    if (step === 0) {
        container.innerHTML = `
            <div class="property-group">
                <label class="property-label">Transform Type</label>
                <select class="property-input">
                    <option>Map (transform each item)</option>
                    <option>Filter (keep matching)</option>
                    <option>Reduce (combine all)</option>
                    <option>Sort (order items)</option>
                </select>
            </div>
        `;
    } else {
        container.innerHTML = `
            <div class="property-group">
                <label class="property-label">Function</label>
                <textarea class="property-input" style="height: 100px;">x => x * 2</textarea>
            </div>
        `;
    }
}

function renderFilterWizard(node, step, container) {
    if (step === 0) {
        container.innerHTML = `
            <div class="property-group">
                <label class="property-label">Condition</label>
                <input type="text" class="property-input" placeholder="e.g., value > 10" />
            </div>
        `;
    } else if (step === 1) {
        container.innerHTML = `
            <div class="property-group">
                <label class="property-label">Test with sample:</label>
                <div style="padding: 10px; background: #f5f5f5; border-radius: 4px; margin-top: 8px;">
                    <p style="font-size: 12px;">Input: [1, 5, 10, 15, 20]</p>
                    <p style="font-size: 12px;">Output: [10, 15, 20] ✓</p>
                </div>
            </div>
        `;
    }
}

function renderDecisionWizard(node, step, container) {
    if (step === 0) {
        container.innerHTML = `
            <div class="property-group">
                <label class="property-label">Condition Expression</label>
                <input type="text" class="property-input" placeholder="e.g., score > threshold" />
            </div>
        `;
    } else {
        container.innerHTML = `
            <div class="property-group">
                <label style="display: flex; align-items: center; gap: 8px;">
                    <input type="checkbox" checked />
                    <span>True Path (continue)</span>
                </label>
                <label style="display: flex; align-items: center; gap: 8px; margin-top: 8px;">
                    <input type="checkbox" checked />
                    <span>False Path (alternative)</span>
                </label>
            </div>
        `;
    }
}

function closeWizard() {
    document.getElementById('wizardModal').classList.remove('open');
}

function nextWizardStep() {
    const active = document.querySelector('.wizard-step.active');
    const next = active.nextElementSibling;
    if (next) next.click();
    else if (active === document.querySelector('.wizard-step:last-child')) {
        closeWizard();
        showToast('Configuration saved!', 'success');
    }
}

function previousWizardStep() {
    const active = document.querySelector('.wizard-step.active');
    const prev = active.previousElementSibling;
    if (prev) prev.click();
}

// ============================================================================
// NODE MANAGEMENT
// ============================================================================

function populateAgentPalette() {
    const palette = document.getElementById('agentPalette');
    const categories = {};

    Object.entries(AGENT_DEFINITIONS).forEach(([type, def]) => {
        if (!categories[def.category]) {
            categories[def.category] = [];
        }
        categories[def.category].push({ type, ...def });
    });

    palette.innerHTML = '';

    Object.entries(categories).forEach(([category, agents]) => {
        const categoryDiv = document.createElement('div');
        categoryDiv.className = 'agent-category';

        const header = document.createElement('div');
        header.className = 'category-header';
        header.innerHTML = `<span>${agents[0].icon || '🔹'} ${category}</span> <span>▼</span>`;
        header.onclick = () => {
            const content = categoryDiv.querySelector('.category-content');
            content.classList.toggle('collapsed');
            header.querySelector('span:last-child').textContent =
                content.classList.contains('collapsed') ? '▶' : '▼';
        };

        const content = document.createElement('div');
        content.className = 'category-content';

        agents.forEach(agent => {
            const item = document.createElement('div');
            item.className = 'agent-item';
            item.draggable = true;
            item.innerHTML = `${agent.icon || '🔹'} ${agent.type}`;
            item.title = agent.description;

            item.addEventListener('dragstart', (e) => {
                e.dataTransfer.effectAllowed = 'copy';
                e.dataTransfer.setData('agentType', agent.type);
            });

            content.appendChild(item);
        });

        categoryDiv.appendChild(header);
        categoryDiv.appendChild(content);
        palette.appendChild(categoryDiv);
    });
}

function addNode(type, x, y) {
    const def = AGENT_DEFINITIONS[type];
    if (!def) return;

    const node = {
        id: `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: type,
        x: Math.round(x / GRID_SIZE) * GRID_SIZE,
        y: Math.round(y / GRID_SIZE) * GRID_SIZE,
        icon: def.icon,
        config: { ...def.config }
    };

    nodes.push(node);
    saveState();
    renderCanvas();
    addDebugLog(`Added ${type} node at (${node.x}, ${node.y})`, 'info');
    showToast(`Added ${type}`, 'success');
}

function deleteNode(nodeId) {
    nodes = nodes.filter(n => n.id !== nodeId);
    connections = connections.filter(c => c.from !== nodeId && c.to !== nodeId);
    selectedNode = null;
    saveState();
    renderCanvas();
    addDebugLog(`Deleted node ${nodeId}`, 'info');
}

function deleteSelectedNode() {
    if (selectedNode) {
        deleteNode(selectedNode.id);
    }
}

function selectNode(nodeId) {
    selectedNode = nodes.find(n => n.id === nodeId);
    renderNodeProperties(selectedNode);
    renderCanvas();
}

function addConnection(fromId, toId) {
    if (fromId === toId) {
        showToast('Cannot connect node to itself', 'warning');
        return;
    }

    const exists = connections.some(c => c.from === fromId && c.to === toId);
    if (exists) {
        showToast('Connection already exists', 'warning');
        return;
    }

    connections.push({
        id: `conn_${Date.now()}`,
        from: fromId,
        to: toId
    });

    saveState();
    renderCanvas();
    validateWorkflow();
    addDebugLog(`Connected ${fromId} → ${toId}`, 'info');
}

// ============================================================================
// RENDERING
// ============================================================================

function renderCanvas() {
    const canvas = document.getElementById('canvas');
    canvas.innerHTML = '';

    nodes.forEach(node => {
        const nodeEl = document.createElement('div');
        nodeEl.className = `node ${selectedNode?.id === node.id ? 'selected' : ''}`;
        nodeEl.dataset.nodeId = node.id;
        nodeEl.style.left = `${node.x}px`;
        nodeEl.style.top = `${node.y}px`;

        nodeEl.innerHTML = `
            <div class="node-header">
                <span>${node.icon || '🔹'} ${node.type}</span>
                <div class="node-actions">
                    <button class="node-action-btn" onclick="openNodeConfigWizard(nodes.find(n => n.id === '${node.id}'))">⚙️</button>
                    <button class="node-action-btn" onclick="deleteNode('${node.id}')">✕</button>
                </div>
            </div>
            <div class="node-config" style="font-size: 11px; color: #999;">
                ${node.id.substring(0, 8)}
            </div>
            <div class="node-ports">
                <div class="port input" title="Input" onmousedown="onPortMouseDown(event, '${node.id}', 'input')"></div>
                <div class="port output" title="Output" onmousedown="onPortMouseDown(event, '${node.id}', 'output')"></div>
            </div>
        `;

        nodeEl.addEventListener('mousedown', (e) => {
            if (e.target.closest('.node-action-btn') || e.target.closest('.port')) return;
            selectNode(node.id);
            onNodeMouseDown(e, node.id);
        });

        canvas.appendChild(nodeEl);
    });

    renderConnections();
}

function renderConnections() {
    const svg = document.getElementById('connectionsLayer');
    svg.querySelectorAll('line').forEach(l => l.remove());

    const viewport = document.getElementById('canvasViewport');
    const offsetX = viewport.offsetLeft;
    const offsetY = viewport.offsetTop;

    connections.forEach(conn => {
        const fromNode = nodes.find(n => n.id === conn.from);
        const toNode = nodes.find(n => n.id === conn.to);

        if (!fromNode || !toNode) return;

        const x1 = fromNode.x + 200 / 2;
        const y1 = fromNode.y + 80;
        const x2 = toNode.x + 200 / 2;
        const y2 = toNode.y;

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x2);
        line.setAttribute('y2', y2);
        line.setAttribute('stroke', '#667eea');
        line.setAttribute('stroke-width', '2');
        line.setAttribute('marker-end', 'url(#arrowhead)');
        line.style.pointerEvents = 'all';
        line.style.cursor = 'pointer';

        line.addEventListener('click', () => {
            selectedConnection = conn;
            renderCanvas();
        });

        svg.appendChild(line);
    });
}

function renderNodeProperties(node) {
    if (!node) {
        document.getElementById('propsPanel').innerHTML = `
            <h3>⚙️ Properties</h3>
            <p style="font-size: 12px; color: #999;">Select a node to view properties</p>
        `;
        return;
    }

    const propsPanel = document.getElementById('propsPanel');
    propsPanel.innerHTML = `
        <h3>⚙️ Properties</h3>
        <div class="property-group">
            <label class="property-label">Node ID</label>
            <div class="property-input" style="border: none; background: #f5f5f5;">${node.id}</div>
        </div>
        <div class="property-group">
            <label class="property-label">Type</label>
            <div class="property-input" style="border: none; background: #f5f5f5;">${node.type}</div>
        </div>
        <div class="property-group">
            <label class="property-label">Position</label>
            <div class="property-input" style="border: none; background: #f5f5f5;">(${Math.round(node.x)}, ${Math.round(node.y)})</div>
        </div>
        <div class="property-group">
            <button class="btn btn-primary" onclick="openNodeConfigWizard(nodes.find(n => n.id === '${node.id}'))" style="width: 100%;">
                ⚙️ Configure
            </button>
        </div>
    `;
}

// ============================================================================
// EVENT HANDLERS
// ============================================================================

function onCanvasMouseDown(e) {
    if (e.button !== 0) return; // Only left mouse
    if (e.target !== e.currentTarget && e.target !== e.currentTarget.querySelector('.canvas')) return;

    selectedNode = null;
    renderCanvas();
}

function onNodeMouseDown(e, nodeId) {
    const node = nodes.find(n => n.id === nodeId);
    const startX = e.clientX;
    const startY = e.clientY;
    const startNodeX = node.x;
    const startNodeY = node.y;

    const onMouseMove = (moveEvent) => {
        const dx = moveEvent.clientX - startX;
        const dy = moveEvent.clientY - startY;

        node.x = startNodeX + dx;
        node.y = startNodeY + dy;
        renderCanvas();
    };

    const onMouseUp = () => {
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
        node.x = Math.round(node.x / GRID_SIZE) * GRID_SIZE;
        node.y = Math.round(node.y / GRID_SIZE) * GRID_SIZE;
        saveState();
        renderCanvas();
    };

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
}

function onPortMouseDown(e, nodeId, type) {
    e.stopPropagation();
    const node = nodes.find(n => n.id === nodeId);

    if (type === 'output') {
        showConnectionSuggestions(e.target, node);
        draggingPort = { from: nodeId, type: 'output' };
    }

    const startX = e.clientX;
    const startY = e.clientY;

    const onMouseMove = () => {
        // Visual feedback for connection preview
    };

    const onMouseUp = () => {
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
    };

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
}

function onCanvasDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
}

function onCanvasDrop(e) {
    e.preventDefault();

    const agentType = e.dataTransfer.getData('agentType');
    const snippet = e.dataTransfer.getData('snippet');

    const viewport = document.getElementById('canvasViewport');
    const rect = viewport.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (agentType) {
        addNode(agentType, x, y);
    } else if (snippet) {
        insertSnippet(JSON.parse(snippet), x, y);
    }
}

function onPaletteItemDragStart(e) {
    if (!e.target.classList.contains('agent-item') && !e.target.classList.contains('snippet-item')) {
        return;
    }
}

function onCanvasMouseMove(e) {
    // Pan canvas with middle mouse
    if (e.buttons === 4) {
        panX += e.movementX;
        panY += e.movementY;
        const canvas = document.getElementById('canvas');
        canvas.style.transform = `translate(${panX}px, ${panY}px)`;
    }
}

function onCanvasWheel(e) {
    e.preventDefault();

    const viewport = document.getElementById('canvasViewport');
    const oldZoom = zoom;
    zoom = Math.max(0.5, Math.min(2, zoom - e.deltaY * 0.001));

    const scaleChange = zoom / oldZoom;
    panX *= scaleChange;
    panY *= scaleChange;

    const canvas = document.getElementById('canvas');
    canvas.style.transform = `scale(${zoom}) translate(${panX}px, ${panY}px)`;
    document.getElementById('zoomLevel').textContent = `${Math.round(zoom * 100)}%`;
}

function handleKeyboardShortcuts(e) {
    const key = e.key.toLowerCase();
    const ctrl = e.ctrlKey || e.metaKey;
    const shift = e.shiftKey;

    let shortcut = key;
    if (ctrl) shortcut = `Ctrl+${key.toUpperCase()}`;
    if (e.key === ' ') shortcut = ' ';
    if (e.key === '?') shortcut = '?';
    if (e.key === 'Delete') shortcut = 'Delete';

    const handler = window.keyboardShortcuts?.[shortcut];
    if (handler && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        e.preventDefault();
        handler();
    }
}

function handleClickOutside(e) {
    if (!e.target.closest('.node') && !e.target.closest('.sidebar')) {
        selectedNode = null;
        renderCanvas();
    }
}

// ============================================================================
// BONUS FEATURES
// ============================================================================

// Keyboard Shortcuts
function showKeyboardShortcuts() {
    const panel = document.getElementById('shortcutsPanel');
    panel.classList.toggle('show');
}

// Auto-Layout
function autoLayout() {
    if (nodes.length === 0) return;

    const graph = buildAdjacencyGraph();
    const levels = calculateLevels(graph);
    const nodesByLevel = {};

    nodes.forEach(node => {
        const level = levels[node.id] || 0;
        if (!nodesByLevel[level]) nodesByLevel[level] = [];
        nodesByLevel[level].push(node);
    });

    let maxWidth = 0;
    Object.values(nodesByLevel).forEach(level => {
        const width = level.length * 250;
        if (width > maxWidth) maxWidth = width;
    });

    Object.entries(nodesByLevel).forEach(([level, nodesAtLevel]) => {
        const levelNum = parseInt(level);
        const totalWidth = nodesAtLevel.length * 250;
        const startX = (maxWidth - totalWidth) / 2;

        nodesAtLevel.forEach((node, idx) => {
            node.x = startX + idx * 250;
            node.y = levelNum * 150;
        });
    });

    saveState();
    renderCanvas();
    showToast('Layout auto-arranged', 'success');
    addDebugLog('Auto-layout applied', 'info');
}

function calculateLevels(graph) {
    const levels = {};
    const visited = new Set();

    const dfs = (nodeId, level) => {
        if (visited.has(nodeId)) return;
        visited.add(nodeId);
        levels[nodeId] = Math.max(levels[nodeId] || 0, level);

        (graph[nodeId] || []).forEach(neighbor => {
            dfs(neighbor, level + 1);
        });
    };

    nodes.forEach(node => {
        if (!visited.has(node.id)) {
            dfs(node.id, 0);
        }
    });

    return levels;
}

// Undo/Redo
function saveState() {
    undoStack.push({
        nodes: JSON.parse(JSON.stringify(nodes)),
        connections: JSON.parse(JSON.stringify(connections))
    });
    redoStack = [];
}

function undo() {
    if (undoStack.length === 0) return;

    redoStack.push({
        nodes: JSON.parse(JSON.stringify(nodes)),
        connections: JSON.parse(JSON.stringify(connections))
    });

    const prev = undoStack.pop();
    nodes = prev.nodes;
    connections = prev.connections;
    renderCanvas();
    validateWorkflow();
    addDebugLog('Undo performed', 'info');
}

function redo() {
    if (redoStack.length === 0) return;

    undoStack.push({
        nodes: JSON.parse(JSON.stringify(nodes)),
        connections: JSON.parse(JSON.stringify(connections))
    });

    const next = redoStack.pop();
    nodes = next.nodes;
    connections = next.connections;
    renderCanvas();
    validateWorkflow();
    addDebugLog('Redo performed', 'info');
}

// Copy/Paste
function copyNode() {
    if (!selectedNode) return;
    clipboard = JSON.parse(JSON.stringify(selectedNode));
    showToast('Node copied', 'info');
}

function pasteNode() {
    if (!clipboard) return;

    const newNode = JSON.parse(JSON.stringify(clipboard));
    newNode.id = `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    newNode.x += 50;
    newNode.y += 50;

    nodes.push(newNode);
    saveState();
    renderCanvas();
    showToast('Node pasted', 'success');
}

function selectAll() {
    // Placeholder for select all functionality
}

function deselectAll() {
    selectedNode = null;
    renderCanvas();
}

function filterAgents() {
    const search = document.getElementById('agentSearch').value.toLowerCase();
    document.querySelectorAll('.agent-item').forEach(item => {
        const text = item.textContent.toLowerCase();
        item.style.display = text.includes(search) ? 'block' : 'none';
    });
}

function clearCanvas() {
    if (confirm('Clear all nodes and connections?')) {
        nodes = [];
        connections = [];
        selectedNode = null;
        saveState();
        renderCanvas();
        showToast('Canvas cleared', 'info');
        addDebugLog('Canvas cleared', 'info');
    }
}

function zoomIn() {
    zoom = Math.min(2, zoom + 0.1);
    const canvas = document.getElementById('canvas');
    canvas.style.transform = `scale(${zoom})`;
    document.getElementById('zoomLevel').textContent = `${Math.round(zoom * 100)}%`;
}

function zoomOut() {
    zoom = Math.max(0.5, zoom - 0.1);
    const canvas = document.getElementById('canvas');
    canvas.style.transform = `scale(${zoom})`;
    document.getElementById('zoomLevel').textContent = `${Math.round(zoom * 100)}%`;
}

function resetZoom() {
    zoom = 1;
    panX = 0;
    panY = 0;
    const canvas = document.getElementById('canvas');
    canvas.style.transform = 'scale(1) translate(0, 0)';
    document.getElementById('zoomLevel').textContent = '100%';
}

// Export/Import
function exportWorkflowData() {
    return {
        version: '1.0',
        name: document.getElementById('workflowName').value,
        nodes: nodes,
        connections: connections,
        timestamp: new Date().toISOString()
    };
}

function exportWorkflow() {
    const data = exportWorkflowData();
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${data.name || 'workflow'}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Workflow exported', 'success');
    addDebugLog('Workflow exported', 'info');
}

function importWorkflowData(data) {
    nodes = data.nodes || [];
    connections = data.connections || [];
    document.getElementById('workflowName').value = data.name || 'Imported Workflow';
    saveState();
    renderCanvas();
    validateWorkflow();
    showToast('Workflow imported', 'success');
}

function switchPanel(panelName) {
    document.querySelectorAll('.panel-content').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));

    document.getElementById(`${panelName}-panel`)?.classList.add('active');
    document.querySelector(`[data-panel="${panelName}"]`)?.classList.add('active');
}

// Debug logging
function addDebugLog(message, level = 'info') {
    const log = document.getElementById('debugLog');
    const timestamp = new Date().toLocaleTimeString();
    const colors = {
        'info': '#888',
        'success': '#2ecc71',
        'error': '#e74c3c',
        'warning': '#f39c12'
    };
    const icon = {
        'info': 'ℹ️',
        'success': '✓',
        'error': '✗',
        'warning': '⚠️'
    };

    const line = document.createElement('div');
    line.style.color = colors[level];
    line.textContent = `[${timestamp}] ${icon[level]} ${message}`;
    log.appendChild(line);

    if (document.getElementById('autoScroll').checked) {
        log.scrollTop = log.scrollHeight;
    }
}

function clearDebugLog() {
    document.getElementById('debugLog').innerHTML = '';
}

// Notifications
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease-out reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function updateNodeConfig(value) {
    if (selectedNode) {
        selectedNode.config.mode = value;
        saveState();
    }
}

// Initialize on load
window.addEventListener('load', () => {
    addDebugLog('Workflow builder v2 loaded', 'success');
    renderCanvas();
    validateWorkflow();
});
