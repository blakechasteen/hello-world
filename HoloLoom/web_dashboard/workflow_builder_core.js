/**
 * HoloLoom Workflow Builder - Core Functionality
 *
 * Essential workflow builder features:
 * - Node management
 * - Connections
 * - Drag and drop
 * - Canvas interaction
 */

// ============================================================================
// GLOBAL STATE
// ============================================================================

window.nodes = [];
window.connections = [];
let selectedNode = null;
let draggedNode = null;
let connectionStart = null;
let nextNodeId = 1;
let nextConnectionId = 1;
let zoomLevel = 1.0;

// Agent type definitions (simplified from existing file)
const agentCategories = {
    query: [
        { id: 'hololoom', name: 'HoloLoom Query', icon: '🧵', color: '#667eea' },
        { id: 'search', name: 'Memory Search', icon: '🔍', color: '#667eea' },
        { id: 'multiquery', name: 'Multi-Query', icon: '🔀', color: '#667eea' }
    ],
    process: [
        { id: 'embedder', name: 'Matryoshka Embedder', icon: '📊', color: '#f093fb' },
        { id: 'synthesizer', name: 'Synthesizer', icon: '⚗️', color: '#f093fb' },
        { id: 'refiner', name: 'Recursive Refiner', icon: '✨', color: '#f093fb' }
    ],
    memory: [
        { id: 'store', name: 'Memory Store', icon: '💾', color: '#4facfe' },
        { id: 'retrieve', name: 'Context Retriever', icon: '📥', color: '#4facfe' },
        { id: 'fusion', name: 'Knowledge Fusion', icon: '🔗', color: '#4facfe' }
    ],
    decision: [
        { id: 'thompson', name: 'Thompson Sampler', icon: '🎲', color: '#43e97b' },
        { id: 'convergence', name: 'Convergence Engine', icon: '🎯', color: '#43e97b' },
        { id: 'safety', name: 'Safety Guardrails', icon: '🛡️', color: '#43e97b' }
    ],
    output: [
        { id: 'response', name: 'Response Generator', icon: '💬', color: '#fa709a' },
        { id: 'format', name: 'Format Converter', icon: '📄', color: '#fa709a' }
    ],
    control: [
        { id: 'conditional', name: 'Conditional Branch', icon: '🔀', color: '#30cfd0' },
        { id: 'loop', name: 'Loop Iterator', icon: '🔁', color: '#30cfd0' },
        { id: 'parallel', name: 'Parallel Executor', icon: '⚡', color: '#30cfd0' }
    ],
    llm: [
        { id: 'llm_prompt', name: 'LLM Prompt', icon: '🤖', color: '#ff6b6b' },
        { id: 'structured_llm', name: 'Structured LLM', icon: '📋', color: '#ff6b6b' },
        { id: 'rag_prompt', name: 'RAG Prompt', icon: '📚', color: '#ff6b6b' }
    ],
    tool: [
        { id: 'tool_call', name: 'Tool Call', icon: '🔧', color: '#ffa726' },
        { id: 'api_request', name: 'API Request', icon: '🌐', color: '#ffa726' },
        { id: 'code_executor', name: 'Code Executor', icon: '💻', color: '#ffa726' }
    ]
};

// ============================================================================
// INITIALIZATION
// ============================================================================

window.initializeAgentPalette = function() {
    const palette = document.getElementById('agentPalette');
    if (!palette) return;

    palette.innerHTML = '';

    for (const [categoryName, agents] of Object.entries(agentCategories)) {
        const categoryDiv = document.createElement('div');
        categoryDiv.className = 'agent-category';

        const titleDiv = document.createElement('div');
        titleDiv.className = 'category-title';
        titleDiv.textContent = categoryName.charAt(0).toUpperCase() + categoryName.slice(1);
        categoryDiv.appendChild(titleDiv);

        agents.forEach(agent => {
            const agentDiv = document.createElement('div');
            agentDiv.className = 'agent-template';
            agentDiv.draggable = true;
            agentDiv.dataset.type = categoryName;
            agentDiv.dataset.agent = agent.id;

            const iconDiv = document.createElement('div');
            iconDiv.className = 'agent-icon';
            iconDiv.style.background = agent.color;
            iconDiv.textContent = agent.icon;

            const nameDiv = document.createElement('div');
            nameDiv.className = 'agent-name';
            nameDiv.textContent = agent.name;

            agentDiv.appendChild(iconDiv);
            agentDiv.appendChild(nameDiv);

            // Drag events
            agentDiv.addEventListener('dragstart', handleAgentDragStart);
            agentDiv.addEventListener('dragend', handleAgentDragEnd);

            categoryDiv.appendChild(agentDiv);
        });

        palette.appendChild(categoryDiv);
    }
};

function filterAgents() {
    const searchTerm = document.getElementById('agentSearch').value.toLowerCase();
    const categories = document.querySelectorAll('.agent-category');

    categories.forEach(category => {
        const agents = category.querySelectorAll('.agent-template');
        let hasVisibleAgent = false;

        agents.forEach(agent => {
            const name = agent.querySelector('.agent-name').textContent.toLowerCase();
            if (name.includes(searchTerm)) {
                agent.style.display = '';
                hasVisibleAgent = true;
            } else {
                agent.style.display = 'none';
            }
        });

        category.style.display = hasVisibleAgent ? '' : 'none';
    });
}

window.filterAgents = filterAgents;

// ============================================================================
// DRAG AND DROP
// ============================================================================

function handleAgentDragStart(e) {
    const agentType = e.target.dataset.agent;
    const agentCategory = e.target.dataset.type;

    e.dataTransfer.effectAllowed = 'copy';
    e.dataTransfer.setData('agent', agentType);
    e.dataTransfer.setData('category', agentCategory);
}

function handleAgentDragEnd(e) {
    // Cleanup
}

// Setup canvas drop zone
function initializeCanvasDrop() {
    const canvas = document.getElementById('canvas');
    if (!canvas) return;

    canvas.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    });

    canvas.addEventListener('drop', (e) => {
        e.preventDefault();

        const agentType = e.dataTransfer.getData('agent');
        const agentCategory = e.dataTransfer.getData('category');

        if (!agentType) return;

        // Get drop position
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Create node
        createNode(agentType, agentCategory, x, y);
    });
}

// ============================================================================
// NODE MANAGEMENT
// ============================================================================

function createNode(agentType, category, x, y) {
    const nodeId = `node_${nextNodeId++}`;

    // Find agent definition
    let agentDef = null;
    if (agentCategories[category]) {
        agentDef = agentCategories[category].find(a => a.id === agentType);
    }

    const node = {
        id: nodeId,
        agentType: agentType,
        category: category,
        x: x,
        y: y,
        config: {}
    };

    window.nodes.push(node);

    // Render node
    renderNode(node, agentDef);

    return node;
}

function renderNode(node, agentDef) {
    const canvas = document.getElementById('canvas');
    if (!canvas) return;

    const nodeDiv = document.createElement('div');
    nodeDiv.className = 'workflow-node';
    nodeDiv.id = node.id;
    nodeDiv.style.left = `${node.x}px`;
    nodeDiv.style.top = `${node.y}px`;

    const icon = agentDef ? agentDef.icon : '📦';
    const name = agentDef ? agentDef.name : node.agentType;
    const color = agentDef ? agentDef.color : '#999';

    nodeDiv.innerHTML = `
        <div class="node-header">
            <div class="node-icon" style="background: ${color};">${icon}</div>
            <div class="node-title">${name}</div>
            <button class="node-delete" onclick="deleteNode('${node.id}')">×</button>
        </div>
        <div class="node-port input"></div>
        <div class="node-port output"></div>
    `;

    // Make draggable
    makeNodeDraggable(nodeDiv);

    // Click to select
    nodeDiv.addEventListener('click', (e) => {
        e.stopPropagation();
        selectNode(node.id);
    });

    // Connection ports
    const outputPort = nodeDiv.querySelector('.node-port.output');
    outputPort.addEventListener('mousedown', (e) => {
        e.stopPropagation();
        startConnection(node.id);
    });

    const inputPort = nodeDiv.querySelector('.node-port.input');
    inputPort.addEventListener('mouseup', (e) => {
        e.stopPropagation();
        endConnection(node.id);
    });

    canvas.appendChild(nodeDiv);
}

function makeNodeDraggable(nodeDiv) {
    let isDragging = false;
    let startX, startY, initialLeft, initialTop;

    nodeDiv.addEventListener('mousedown', (e) => {
        if (e.target.classList.contains('node-port') ||
            e.target.classList.contains('node-delete')) {
            return;
        }

        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        initialLeft = parseInt(nodeDiv.style.left);
        initialTop = parseInt(nodeDiv.style.top);

        nodeDiv.classList.add('dragging');

        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;

        const dx = e.clientX - startX;
        const dy = e.clientY - startY;

        nodeDiv.style.left = `${initialLeft + dx}px`;
        nodeDiv.style.top = `${initialTop + dy}px`;

        // Update connections
        updateConnections();
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            nodeDiv.classList.remove('dragging');

            // Update node position in state
            const node = window.nodes.find(n => n.id === nodeDiv.id);
            if (node) {
                node.x = parseInt(nodeDiv.style.left);
                node.y = parseInt(nodeDiv.style.top);
            }
        }
    });
}

function selectNode(nodeId) {
    // Deselect all
    document.querySelectorAll('.workflow-node').forEach(n => {
        n.classList.remove('selected');
    });

    // Select this one
    const nodeDiv = document.getElementById(nodeId);
    if (nodeDiv) {
        nodeDiv.classList.add('selected');
        selectedNode = nodeId;

        // Show properties
        showNodeProperties(nodeId);
    }
}

function deleteNode(nodeId) {
    // Remove from DOM
    const nodeDiv = document.getElementById(nodeId);
    if (nodeDiv) {
        nodeDiv.remove();
    }

    // Remove from state
    window.nodes = window.nodes.filter(n => n.id !== nodeId);

    // Remove connected connections
    window.connections = window.connections.filter(c =>
        c.from !== nodeId && c.to !== nodeId
    );

    updateConnections();
}

window.deleteNode = deleteNode;

function showNodeProperties(nodeId) {
    const node = window.nodes.find(n => n.id === nodeId);
    if (!node) return;

    const propsContent = document.getElementById('propertiesContent');
    if (!propsContent) return;

    propsContent.innerHTML = `
        <div>
            <h4 style="margin-bottom: 12px;">${node.agentType}</h4>
            <label>Node ID</label>
            <div style="background: #f3f4f6; padding: 8px; border-radius: 4px; font-family: monospace; font-size: 12px; margin-bottom: 12px;">
                ${node.id}
            </div>
            <label>Agent Type</label>
            <div style="background: #f3f4f6; padding: 8px; border-radius: 4px; font-size: 13px; margin-bottom: 12px;">
                ${node.agentType}
            </div>
            <label>Category</label>
            <div style="background: #f3f4f6; padding: 8px; border-radius: 4px; font-size: 13px;">
                ${node.category}
            </div>
        </div>
    `;
}

// ============================================================================
// CONNECTIONS
// ============================================================================

function startConnection(fromNodeId) {
    connectionStart = fromNodeId;
}

function endConnection(toNodeId) {
    if (!connectionStart) return;
    if (connectionStart === toNodeId) {
        connectionStart = null;
        return;
    }

    // Check if connection already exists
    const exists = window.connections.some(c =>
        c.from === connectionStart && c.to === toNodeId
    );

    if (!exists) {
        const connectionId = `conn_${nextConnectionId++}`;
        window.connections.push({
            id: connectionId,
            from: connectionStart,
            to: toNodeId
        });

        updateConnections();
    }

    connectionStart = null;
}

function updateConnections() {
    const svg = document.getElementById('connectionsLayer');
    if (!svg) return;

    // Clear existing connections
    const existingPaths = svg.querySelectorAll('path');
    existingPaths.forEach(path => path.remove());

    // Draw all connections
    window.connections.forEach(conn => {
        drawConnection(conn);
    });
}

function drawConnection(conn) {
    const fromNode = document.getElementById(conn.from);
    const toNode = document.getElementById(conn.to);

    if (!fromNode || !toNode) return;

    const svg = document.getElementById('connectionsLayer');

    const fromPort = fromNode.querySelector('.node-port.output');
    const toPort = toNode.querySelector('.node-port.input');

    const fromRect = fromPort.getBoundingClientRect();
    const toRect = toPort.getBoundingClientRect();
    const canvasRect = svg.getBoundingClientRect();

    const x1 = fromRect.left + fromRect.width / 2 - canvasRect.left;
    const y1 = fromRect.top + fromRect.height / 2 - canvasRect.top;
    const x2 = toRect.left + toRect.width / 2 - canvasRect.left;
    const y2 = toRect.top + toRect.height / 2 - canvasRect.top;

    // Create curved path
    const dx = x2 - x1;
    const dy = y2 - y1;
    const cx1 = x1 + dx * 0.5;
    const cy1 = y1;
    const cx2 = x1 + dx * 0.5;
    const cy2 = y2;

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', `M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`);
    path.setAttribute('class', 'connection-line');
    path.setAttribute('marker-end', 'url(#arrowhead)');
    path.setAttribute('data-connection-id', conn.id);

    // Click to delete
    path.style.pointerEvents = 'stroke';
    path.addEventListener('click', () => {
        window.connections = window.connections.filter(c => c.id !== conn.id);
        updateConnections();
    });

    svg.appendChild(path);
}

// ============================================================================
// CANVAS CONTROLS
// ============================================================================

function clearCanvas() {
    if (!confirm('Clear all nodes and connections?')) return;

    window.nodes = [];
    window.connections = [];

    const canvas = document.getElementById('canvas');
    if (canvas) {
        canvas.innerHTML = '';
    }

    updateConnections();

    const propsContent = document.getElementById('propertiesContent');
    if (propsContent) {
        propsContent.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">👈</div>
                <p>Select a node to view properties</p>
            </div>
        `;
    }
}

window.clearCanvas = clearCanvas;

function zoomIn() {
    zoomLevel = Math.min(zoomLevel + 0.1, 2.0);
    applyZoom();
}

function zoomOut() {
    zoomLevel = Math.max(zoomLevel - 0.1, 0.5);
    applyZoom();
}

function resetZoom() {
    zoomLevel = 1.0;
    applyZoom();
}

function applyZoom() {
    const canvas = document.getElementById('canvas');
    if (canvas) {
        canvas.style.transform = `scale(${zoomLevel})`;
        canvas.style.transformOrigin = 'top left';
    }

    const zoomDisplay = document.getElementById('zoomLevel');
    if (zoomDisplay) {
        zoomDisplay.textContent = `${Math.round(zoomLevel * 100)}%`;
    }

    updateConnections();
}

window.zoomIn = zoomIn;
window.zoomOut = zoomOut;
window.resetZoom = resetZoom;

// ============================================================================
// WORKFLOW EXECUTION
// ============================================================================

async function executeWorkflow() {
    if (window.nodes.length === 0) {
        showToast('No workflow to execute', 'warning');
        return;
    }

    try {
        showLoading('Executing workflow...');

        const workflow = {
            version: '1.0',
            name: document.getElementById('workflowName')?.value || 'Untitled Workflow',
            nodes: window.nodes.map(n => ({
                id: n.id,
                agentType: n.agentType,
                x: n.x,
                y: n.y,
                config: n.config || {}
            })),
            connections: window.connections.map(c => ({
                id: c.id,
                from: c.from,
                to: c.to
            }))
        };

        const response = await fetch(`${API_BASE_URL}/api/workflow/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workflow: workflow,
                input_data: {
                    query: prompt('Enter input query:') || 'Default query'
                }
            })
        });

        const result = await response.json();

        if (result.status === 'success') {
            showToast(`Workflow executed: ${result.nodes_executed} nodes`, 'success');

            // Track analytics
            if (window.trackExecution) {
                await window.trackExecution({
                    execution_id: `exec_${Date.now()}`,
                    workflow_id: currentWorkflowId || 'default',
                    workflow_name: workflow.name,
                    total_duration_ms: 0, // Would come from result
                    nodes_executed: result.nodes_executed,
                    nodes_failed: 0,
                    status: 'success'
                });
            }
        } else {
            throw new Error(result.detail || 'Execution failed');
        }
    } catch (error) {
        console.error('Execution error:', error);
        showToast(`Execution failed: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

window.executeWorkflow = executeWorkflow;

// ============================================================================
// WORKFLOW IMPORT/EXPORT
// ============================================================================

window.loadWorkflow = function(workflow) {
    // Clear current workflow
    clearCanvas();

    // Set name
    if (workflow.name && document.getElementById('workflowName')) {
        document.getElementById('workflowName').value = workflow.name;
    }

    // Create nodes
    if (workflow.nodes) {
        workflow.nodes.forEach(nodeData => {
            const node = {
                id: nodeData.id || `node_${nextNodeId++}`,
                agentType: nodeData.agentType,
                category: nodeData.category || 'query',
                x: nodeData.x || 100,
                y: nodeData.y || 100,
                config: nodeData.config || {}
            };

            window.nodes.push(node);

            // Find agent def
            let agentDef = null;
            for (const [cat, agents] of Object.entries(agentCategories)) {
                const found = agents.find(a => a.id === node.agentType);
                if (found) {
                    agentDef = found;
                    node.category = cat;
                    break;
                }
            }

            renderNode(node, agentDef);
        });
    }

    // Create connections
    if (workflow.connections) {
        window.connections = workflow.connections.map(connData => ({
            id: connData.id || `conn_${nextConnectionId++}`,
            from: connData.from || connData.from_node,
            to: connData.to
        }));
    }

    updateConnections();
};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('Workflow builder core initialized');

    initializeCanvasDrop();
    initializeAgentPalette();

    // Deselect on canvas click
    const canvas = document.getElementById('canvas');
    if (canvas) {
        canvas.addEventListener('click', () => {
            selectedNode = null;
            document.querySelectorAll('.workflow-node').forEach(n => {
                n.classList.remove('selected');
            });
        });
    }
});
