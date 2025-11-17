/**
 * HoloLoom Workflow Builder - Enhanced Features
 *
 * Integrates:
 * - AI Workflow Generation
 * - Analytics Dashboard
 * - Collaborative Editing
 * - Workflow Marketplace
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const API_BASE_URL = 'http://localhost:8001';
const WS_BASE_URL = 'ws://localhost:8001';

// Global state for enhanced features
let generatedWorkflow = null;
let collaborationWS = null;
let collaborationSession = null;
let analyticsData = {};
let marketplaceWorkflows = [];
let currentWorkflowId = null;

// ============================================================================
// PANEL SWITCHING
// ============================================================================

function switchPanel(panelName) {
    // Hide all panels
    document.querySelectorAll('.panel-content').forEach(panel => {
        panel.classList.remove('active');
    });

    // Remove active class from all tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected panel
    const panel = document.getElementById(`${panelName}-panel`);
    if (panel) {
        panel.classList.add('active');
    }

    // Activate corresponding tab
    const tab = document.querySelector(`[data-panel="${panelName}"]`);
    if (tab) {
        tab.classList.add('active');
    }

    // Load panel-specific data
    switch (panelName) {
        case 'analytics':
            loadAnalytics();
            break;
        case 'marketplace':
            loadMarketplace();
            break;
        case 'ai-gen':
            // AI panel doesn't need initial data
            break;
        case 'collaborate':
            // Collaboration panel waits for user action
            break;
    }
}

// ============================================================================
// AI WORKFLOW GENERATION
// ============================================================================

async function generateWorkflowAI() {
    const description = document.getElementById('aiDescription').value.trim();
    if (!description) {
        showToast('Please enter a workflow description', 'error');
        return;
    }

    const useLLM = document.getElementById('useLLM').checked;

    try {
        showLoading('Generating workflow...');

        const response = await fetch(`${API_BASE_URL}/api/workflow/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ description, use_llm: useLLM })
        });

        const data = await response.json();

        if (data.status === 'success') {
            generatedWorkflow = data.workflow;
            showWorkflowPreview(generatedWorkflow);
            showToast('Workflow generated successfully!', 'success');
        } else {
            throw new Error(data.detail || 'Generation failed');
        }
    } catch (error) {
        console.error('AI generation error:', error);
        showToast(`Generation failed: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

async function refineWorkflowAI() {
    const refinement = document.getElementById('aiRefinement').value.trim();
    if (!refinement) {
        showToast('Please enter refinement instructions', 'error');
        return;
    }

    if (!window.nodes || window.nodes.length === 0) {
        showToast('No workflow to refine. Create a workflow first.', 'error');
        return;
    }

    try {
        showLoading('Refining workflow...');

        const currentWorkflow = exportWorkflowData();

        const response = await fetch(`${API_BASE_URL}/api/workflow/refine`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workflow: currentWorkflow,
                refinement
            })
        });

        const data = await response.json();

        if (data.status === 'success') {
            generatedWorkflow = data.workflow;
            showWorkflowPreview(generatedWorkflow);
            showToast('Workflow refined successfully!', 'success');

            if (!data.valid) {
                showToast(`Warning: ${data.errors.join(', ')}`, 'warning');
            }
        } else {
            throw new Error(data.detail || 'Refinement failed');
        }
    } catch (error) {
        console.error('AI refinement error:', error);
        showToast(`Refinement failed: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

function showWorkflowPreview(workflow) {
    const preview = document.getElementById('aiPreview');
    preview.style.display = 'block';

    document.getElementById('previewNodes').textContent = `${workflow.nodes.length} nodes`;
    document.getElementById('previewConnections').textContent = `${workflow.connections.length} connections`;
}

function applyGeneratedWorkflow() {
    if (!generatedWorkflow) return;

    // Import the generated workflow into canvas
    importWorkflowData(generatedWorkflow);

    // Switch to canvas panel
    switchPanel('canvas');

    // Hide preview
    document.getElementById('aiPreview').style.display = 'none';
    document.getElementById('aiDescription').value = '';
    document.getElementById('aiRefinement').value = '';

    showToast('Workflow applied to canvas!', 'success');
}

function cancelGenerated() {
    generatedWorkflow = null;
    document.getElementById('aiPreview').style.display = 'none';
    document.getElementById('aiDescription').value = '';
    document.getElementById('aiRefinement').value = '';
}

// ============================================================================
// ANALYTICS DASHBOARD
// ============================================================================

async function loadAnalytics() {
    if (!currentWorkflowId) {
        currentWorkflowId = 'default_workflow';
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/analytics/${currentWorkflowId}`);

        if (response.status === 404) {
            // No analytics data yet
            showEmptyAnalytics();
            return;
        }

        const data = await response.json();
        analyticsData = data;
        renderAnalytics(data);
    } catch (error) {
        console.error('Analytics loading error:', error);
        showEmptyAnalytics();
    }
}

function showEmptyAnalytics() {
    document.getElementById('totalExecutions').textContent = '0';
    document.getElementById('successRate').textContent = '0%';
    document.getElementById('avgDuration').textContent = '0ms';
    document.getElementById('p95Duration').textContent = '0ms';
}

function renderAnalytics(data) {
    // Update summary metrics
    document.getElementById('totalExecutions').textContent = data.total_executions || 0;
    document.getElementById('successRate').textContent =
        `${((data.success_rate || 0) * 100).toFixed(1)}%`;
    document.getElementById('avgDuration').textContent =
        `${(data.avg_duration_ms || 0).toFixed(0)}ms`;
    document.getElementById('p95Duration').textContent =
        `${(data.p95_duration_ms || 0).toFixed(0)}ms`;

    // Render execution timeline (simple text-based for now)
    const timeline = document.getElementById('executionTimeline');
    timeline.innerHTML = `
        <div style="padding: 20px; text-align: center; color: #666;">
            <p>Total: ${data.total_executions}</p>
            <p>Success: ${data.successes} | Failed: ${data.failures}</p>
        </div>
    `;

    // Render node performance table
    renderNodePerformance(data.node_metrics || {});

    // Render bottleneck warnings
    renderBottleneckWarnings(data.bottleneck_nodes || []);
}

function renderNodePerformance(nodeMetrics) {
    const tbody = document.getElementById('nodePerformanceBody');
    tbody.innerHTML = '';

    if (Object.keys(nodeMetrics).length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No node data available</td></tr>';
        return;
    }

    for (const [nodeId, metrics] of Object.entries(nodeMetrics)) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${nodeId}</strong></td>
            <td>${(metrics.avg_duration_ms || 0).toFixed(1)}ms</td>
            <td>${((metrics.success_rate || 0) * 100).toFixed(0)}%</td>
            <td>${(metrics.p95_duration_ms || 0).toFixed(1)}ms</td>
            <td>
                ${metrics.success_rate >= 0.9 ? '✅' : '⚠️'}
            </td>
        `;
        tbody.appendChild(row);
    }
}

function renderBottleneckWarnings(bottlenecks) {
    const container = document.getElementById('bottleneckWarnings');
    container.innerHTML = '';

    if (bottlenecks.length === 0) {
        return;
    }

    bottlenecks.forEach(nodeId => {
        const warning = document.createElement('div');
        warning.className = 'warning-card';
        warning.innerHTML = `
            <h4>⚠️ Bottleneck Detected</h4>
            <p>Node <strong>${nodeId}</strong> is taking >40% of total execution time. Consider optimization.</p>
        `;
        container.appendChild(warning);
    });
}

async function refreshAnalytics() {
    await loadAnalytics();
    showToast('Analytics refreshed', 'success');
}

function exportAnalytics() {
    if (!analyticsData || Object.keys(analyticsData).length === 0) {
        showToast('No analytics data to export', 'warning');
        return;
    }

    // Convert to CSV
    let csv = 'Metric,Value\n';
    csv += `Total Executions,${analyticsData.total_executions}\n`;
    csv += `Success Rate,${(analyticsData.success_rate * 100).toFixed(1)}%\n`;
    csv += `Avg Duration,${analyticsData.avg_duration_ms}ms\n`;
    csv += `P95 Duration,${analyticsData.p95_duration_ms}ms\n`;

    // Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'workflow_analytics.csv';
    a.click();
    URL.revokeObjectURL(url);

    showToast('Analytics exported', 'success');
}

async function trackExecution(executionData) {
    try {
        await fetch(`${API_BASE_URL}/api/analytics/track`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(executionData)
        });
    } catch (error) {
        console.error('Analytics tracking error:', error);
    }
}

// ============================================================================
// COLLABORATIVE EDITING
// ============================================================================

async function joinCollaborativeSession() {
    const displayName = document.getElementById('displayName').value.trim();
    const sessionId = document.getElementById('sessionId').value.trim() || generateSessionId();

    if (!displayName) {
        showToast('Please enter your name', 'error');
        return;
    }

    try {
        // Connect to WebSocket
        const userId = `user_${Math.random().toString(36).substr(2, 9)}`;
        const wsUrl = `${WS_BASE_URL}/ws/collaborate/${sessionId}?user_id=${userId}&display_name=${encodeURIComponent(displayName)}`;

        collaborationWS = new WebSocket(wsUrl);

        collaborationWS.onopen = () => {
            console.log('Connected to collaborative session');
            collaborationSession = { sessionId, userId, displayName };

            // Show active session UI
            document.getElementById('sessionSetup').style.display = 'none';
            document.getElementById('sessionActive').style.display = 'block';
            document.getElementById('activeSessionId').textContent = sessionId;
            document.getElementById('connectionStatus').textContent = 'Connected';

            showToast(`Joined session: ${sessionId}`, 'success');
        };

        collaborationWS.onmessage = (event) => {
            const message = JSON.parse(event.data);
            handleCollaborationMessage(message);
        };

        collaborationWS.onerror = (error) => {
            console.error('WebSocket error:', error);
            showToast('Connection error', 'error');
        };

        collaborationWS.onclose = () => {
            console.log('WebSocket closed');
            document.getElementById('connectionStatus').textContent = 'Disconnected';
            showToast('Session ended', 'warning');
        };

    } catch (error) {
        console.error('Collaboration join error:', error);
        showToast(`Failed to join session: ${error.message}`, 'error');
    }
}

function handleCollaborationMessage(message) {
    switch (message.type) {
        case 'initial_state':
            // Received current workflow state
            if (message.workflow) {
                importWorkflowData(message.workflow);
            }
            updateActiveUsers(message.users || []);
            break;

        case 'user_joined':
            addActivityLog(`${message.display_name} joined`);
            updateActiveUsers(message.users);
            break;

        case 'user_left':
            addActivityLog(`${message.display_name} left`);
            updateActiveUsers(message.users);
            break;

        case 'operation':
            // Another user made a change
            applyRemoteOperation(message.operation);
            addActivityLog(`${message.display_name} ${message.operation.type}`);
            break;

        case 'cursor_move':
            // Update remote cursor position
            updateRemoteCursor(message.user_id, message.x, message.y);
            break;

        case 'node_locked':
            markNodeAsLocked(message.node_id, message.user_id);
            break;

        case 'node_unlocked':
            markNodeAsUnlocked(message.node_id);
            break;
    }
}

function updateActiveUsers(users) {
    const userList = document.getElementById('userList');
    const userCount = document.getElementById('userCount');

    userCount.textContent = users.length;
    userList.innerHTML = '';

    users.forEach(user => {
        const userItem = document.createElement('div');
        userItem.className = 'user-item';

        const color = stringToColor(user.user_id);
        userItem.innerHTML = `
            <div class="user-avatar" style="background-color: ${color};">
                ${user.display_name.charAt(0).toUpperCase()}
            </div>
            <div class="user-name">${user.display_name}</div>
            <div class="user-status">Active</div>
        `;

        userList.appendChild(userItem);
    });
}

function addActivityLog(message) {
    const log = document.getElementById('activityLog');
    const item = document.createElement('div');
    item.className = 'activity-item';

    const now = new Date().toLocaleTimeString();
    item.innerHTML = `
        <div>${message}</div>
        <div class="timestamp">${now}</div>
    `;

    log.insertBefore(item, log.firstChild);

    // Keep only last 20 items
    while (log.children.length > 20) {
        log.removeChild(log.lastChild);
    }
}

function broadcastOperation(operation) {
    if (collaborationWS && collaborationWS.readyState === WebSocket.OPEN) {
        collaborationWS.send(JSON.stringify({
            type: 'operation',
            user_id: collaborationSession.userId,
            data: operation
        }));
    }
}

function applyRemoteOperation(operation) {
    // Apply operation received from another user
    // This would integrate with the main workflow builder logic
    console.log('Applying remote operation:', operation);
}

function updateRemoteCursor(userId, x, y) {
    // Update cursor position for remote user
    console.log(`Remote cursor: ${userId} at (${x}, ${y})`);
}

function markNodeAsLocked(nodeId, userId) {
    // Mark node as locked by another user
    const node = document.getElementById(nodeId);
    if (node) {
        node.classList.add('locked');
        node.setAttribute('data-locked-by', userId);
    }
}

function markNodeAsUnlocked(nodeId) {
    // Unlock node
    const node = document.getElementById(nodeId);
    if (node) {
        node.classList.remove('locked');
        node.removeAttribute('data-locked-by');
    }
}

function leaveCollaborativeSession() {
    if (collaborationWS) {
        collaborationWS.close();
        collaborationWS = null;
    }

    collaborationSession = null;

    // Reset UI
    document.getElementById('sessionSetup').style.display = 'block';
    document.getElementById('sessionActive').style.display = 'none';
    document.getElementById('displayName').value = '';
    document.getElementById('sessionId').value = '';
    document.getElementById('userList').innerHTML = '';
    document.getElementById('activityLog').innerHTML = '';

    showToast('Left collaborative session', 'success');
}

function copySessionId() {
    const sessionId = document.getElementById('activeSessionId').textContent;
    navigator.clipboard.writeText(sessionId).then(() => {
        showToast('Session ID copied!', 'success');
    });
}

function generateSessionId() {
    return `session_${Math.random().toString(36).substr(2, 9)}`;
}

function stringToColor(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }

    const colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#ffa726', '#ff6b6b'];
    return colors[Math.abs(hash) % colors.length];
}

// ============================================================================
// MARKETPLACE
// ============================================================================

async function loadMarketplace() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/marketplace/list`);
        const data = await response.json();

        marketplaceWorkflows = data.workflows || [];
        renderMarketplace(marketplaceWorkflows);
    } catch (error) {
        console.error('Marketplace loading error:', error);
        showToast('Failed to load marketplace', 'error');
    }
}

function renderMarketplace(workflows) {
    const grid = document.getElementById('marketplaceGrid');
    grid.innerHTML = '';

    if (workflows.length === 0) {
        grid.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; padding: 60px 20px; color: #666;">
                <div style="font-size: 48px; margin-bottom: 16px;">📦</div>
                <p>No workflows in marketplace yet.</p>
                <p style="font-size: 14px; margin-top: 8px;">Be the first to publish!</p>
            </div>
        `;
        return;
    }

    workflows.forEach(item => {
        const card = createMarketplaceCard(item);
        grid.appendChild(card);
    });
}

function createMarketplaceCard(item) {
    const card = document.createElement('div');
    card.className = 'marketplace-item';

    const metadata = item.metadata || {};
    const rating = metadata.rating || 0;
    const downloads = metadata.downloads || 0;

    card.innerHTML = `
        <div class="item-header">
            <div class="item-title">${metadata.name || 'Untitled'}</div>
            <div class="item-rating">
                ⭐ ${rating.toFixed(1)}
            </div>
        </div>
        <div class="item-description">
            ${metadata.description || 'No description'}
        </div>
        <div class="item-meta">
            <span class="item-category">${metadata.category || 'custom'}</span>
            <span class="item-downloads">
                📥 ${downloads} downloads
            </span>
        </div>
    `;

    card.onclick = () => downloadMarketplaceWorkflow(item.workflow_id);

    return card;
}

async function downloadMarketplaceWorkflow(workflowId) {
    try {
        showLoading('Downloading workflow...');

        const response = await fetch(`${API_BASE_URL}/api/marketplace/${workflowId}`);
        const data = await response.json();

        // Import workflow
        importWorkflowData(data.workflow);

        // Switch to canvas
        switchPanel('canvas');

        showToast('Workflow downloaded!', 'success');
    } catch (error) {
        console.error('Download error:', error);
        showToast(`Download failed: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

function showPublishModal() {
    if (!window.nodes || window.nodes.length === 0) {
        showToast('No workflow to publish', 'error');
        return;
    }

    const modal = document.getElementById('publishModal');
    modal.classList.add('show');

    // Pre-fill name
    const workflowName = document.getElementById('workflowName').value;
    document.getElementById('publishName').value = workflowName;
}

async function publishToMarketplace() {
    const name = document.getElementById('publishName').value.trim();
    const description = document.getElementById('publishDescription').value.trim();
    const author = document.getElementById('publishAuthor').value.trim();
    const category = document.getElementById('publishCategory').value;
    const tags = document.getElementById('publishTags').value.split(',').map(t => t.trim());

    if (!name || !description || !author) {
        showToast('Please fill all required fields', 'error');
        return;
    }

    try {
        showLoading('Publishing workflow...');

        const workflow = exportWorkflowData();

        const response = await fetch(`${API_BASE_URL}/api/marketplace/publish`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workflow_id: `workflow_${Date.now()}`,
                workflow,
                metadata: { name, description, author, category, tags }
            })
        });

        const data = await response.json();

        if (data.status === 'success') {
            closeModal('publishModal');
            showToast('Workflow published successfully!', 'success');

            // Reload marketplace
            await loadMarketplace();
        } else {
            throw new Error(data.detail || 'Publish failed');
        }
    } catch (error) {
        console.error('Publish error:', error);
        showToast(`Publish failed: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

function searchMarketplace() {
    const query = document.getElementById('marketplaceSearch').value.toLowerCase();
    const filtered = marketplaceWorkflows.filter(item => {
        const metadata = item.metadata || {};
        return (metadata.name || '').toLowerCase().includes(query) ||
               (metadata.description || '').toLowerCase().includes(query) ||
               (metadata.tags || []).some(tag => tag.toLowerCase().includes(query));
    });

    renderMarketplace(filtered);
}

function filterMarketplace() {
    const category = document.getElementById('marketplaceCategory').value;

    if (category === 'all') {
        renderMarketplace(marketplaceWorkflows);
        return;
    }

    const filtered = marketplaceWorkflows.filter(item => {
        return (item.metadata?.category || '') === category;
    });

    renderMarketplace(filtered);
}

// ============================================================================
// TEMPLATES
// ============================================================================

async function showTemplatesModal() {
    const modal = document.getElementById('templatesModal');
    modal.classList.add('show');

    // Load templates
    try {
        const response = await fetch(`${API_BASE_URL}/api/templates/list`);
        const data = await response.json();

        renderTemplates(data.templates || []);
    } catch (error) {
        console.error('Templates loading error:', error);
        showToast('Failed to load templates', 'error');
    }
}

function renderTemplates(templates) {
    const list = document.getElementById('templatesList');
    list.innerHTML = '';

    if (templates.length === 0) {
        list.innerHTML = '<p class="empty-state">No templates available</p>';
        return;
    }

    templates.forEach(template => {
        const card = document.createElement('div');
        card.className = 'marketplace-item';
        card.innerHTML = `
            <div class="item-header">
                <div class="item-title">${template.name}</div>
                <span class="item-category">${template.difficulty}</span>
            </div>
            <div class="item-description">${template.description}</div>
            <div class="item-meta">
                <span>${template.nodes_count} nodes</span>
                <span>${template.connections_count} connections</span>
            </div>
        `;

        card.onclick = async () => {
            await loadTemplate(template.template_id);
            closeModal('templatesModal');
        };

        list.appendChild(card);
    });
}

async function loadTemplate(templateId) {
    try {
        showLoading('Loading template...');

        const response = await fetch(`${API_BASE_URL}/api/templates/${templateId}`);
        const template = await response.json();

        // Import template
        const workflow = {
            version: '1.0',
            name: template.name,
            nodes: template.nodes,
            connections: template.connections
        };

        importWorkflowData(workflow);

        // Switch to canvas
        switchPanel('canvas');

        showToast('Template loaded!', 'success');
    } catch (error) {
        console.error('Template loading error:', error);
        showToast(`Failed to load template: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

function filterTemplates() {
    // Template filtering logic (similar to marketplace)
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast show ${type}`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function showLoading(message = 'Loading...') {
    const overlay = document.getElementById('loadingOverlay');
    const text = document.getElementById('loadingText');
    text.textContent = message;
    overlay.style.display = 'flex';
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.style.display = 'none';
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.classList.remove('show');
}

// Close modals on background click
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.classList.remove('show');
    }
});

// ============================================================================
// INTEGRATION WITH MAIN WORKFLOW BUILDER
// ============================================================================

// These functions should integrate with the main workflow_builder_core.js

function exportWorkflowData() {
    // Export current workflow from canvas
    // This should be implemented in workflow_builder_core.js
    return {
        version: '1.0',
        name: document.getElementById('workflowName').value,
        nodes: window.nodes || [],
        connections: window.connections || []
    };
}

function importWorkflowData(workflow) {
    // Import workflow into canvas
    // This should be implemented in workflow_builder_core.js
    console.log('Importing workflow:', workflow);

    if (window.loadWorkflow) {
        window.loadWorkflow(workflow);
    } else {
        // Fallback: update global state
        window.nodes = workflow.nodes || [];
        window.connections = workflow.connections || [];
        document.getElementById('workflowName').value = workflow.name || 'Imported Workflow';
    }
}

function showVersionHistoryModal() {
    // Version history modal (existing functionality)
    const modal = document.getElementById('versionHistoryModal');
    modal.classList.add('show');
}

function exportWorkflow() {
    // Export workflow (existing functionality)
    const workflow = exportWorkflowData();
    const json = JSON.stringify(workflow, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${workflow.name || 'workflow'}.json`;
    a.click();
    URL.revokeObjectURL(url);

    showToast('Workflow exported', 'success');
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('Enhanced workflow builder initialized');

    // Default to canvas panel
    switchPanel('canvas');

    // Load agent palette
    if (window.initializeAgentPalette) {
        window.initializeAgentPalette();
    }
});

// Export functions for global access
window.switchPanel = switchPanel;
window.generateWorkflowAI = generateWorkflowAI;
window.refineWorkflowAI = refineWorkflowAI;
window.applyGeneratedWorkflow = applyGeneratedWorkflow;
window.cancelGenerated = cancelGenerated;
window.refreshAnalytics = refreshAnalytics;
window.exportAnalytics = exportAnalytics;
window.joinCollaborativeSession = joinCollaborativeSession;
window.leaveCollaborativeSession = leaveCollaborativeSession;
window.copySessionId = copySessionId;
window.showPublishModal = showPublishModal;
window.publishToMarketplace = publishToMarketplace;
window.searchMarketplace = searchMarketplace;
window.filterMarketplace = filterMarketplace;
window.showTemplatesModal = showTemplatesModal;
window.filterTemplates = filterTemplates;
window.closeModal = closeModal;
window.exportWorkflow = exportWorkflow;
window.showVersionHistoryModal = showVersionHistoryModal;
