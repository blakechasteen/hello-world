# Template Gallery × Workflow Builder - Integration Guide

**Purpose**: Connect the template gallery to the workflow builder for seamless template loading

## Quick Integration (2 Steps)

### Step 1: Add "Templates" Button to Workflow Builder Header

In `workflow_builder.html`, find the `.canvas-toolbar` section and add:

```html
<!-- Add this button to the toolbar (around line 520-540) -->
<a href="template_gallery.html" class="toolbar-btn" title="Browse workflow templates">
  📋 Templates
</a>
```

Full example:
```html
<div class="canvas-toolbar">
  <a href="template_gallery.html" class="toolbar-btn" title="Browse templates">
    📋 Templates
  </a>
  <button class="toolbar-btn primary" id="executeBtn">
    ▶ Execute
  </button>
  <button class="toolbar-btn" id="saveBtn">
    💾 Save
  </button>
</div>
```

### Step 2: Add Template Loading Logic to Workflow Builder

In `workflow_builder.js`, add this at the very beginning (after variable declarations):

```javascript
/**
 * Load template from URL parameter (e.g., ?template=research_pipeline.json)
 */
function loadTemplateFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const templateFile = params.get('template');

  if (templateFile) {
    console.log(`Loading template: ${templateFile}`);

    fetch(`example_workflows/${templateFile}`)
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to load template: ${response.status}`);
        }
        return response.json();
      })
      .then(workflow => {
        console.log('Template loaded successfully', workflow);
        loadWorkflow(workflow);

        // Optional: Update page title to show loaded template
        document.title = `${workflow.name} - HoloLoom Workflow Builder`;

        // Show success message
        showNotification(`✅ Loaded template: ${workflow.name}`, 'success');
      })
      .catch(error => {
        console.error('Error loading template:', error);
        showNotification(`❌ Failed to load template: ${error.message}`, 'error');
      });
  }
}

// Call this when page loads (add to existing init/setup function)
document.addEventListener('DOMContentLoaded', () => {
  // ... existing initialization code ...
  loadTemplateFromUrl();
});
```

If you don't have a `loadWorkflow()` function, create one:

```javascript
/**
 * Load a workflow into the builder
 * @param {Object} workflow - Workflow object with nodes and connections
 */
function loadWorkflow(workflow) {
  if (!workflow.nodes) {
    console.error('Invalid workflow: missing nodes');
    return;
  }

  // Clear existing workflow
  document.getElementById('canvas').innerHTML = '';
  nodes = [];
  connections = [];

  // Load nodes
  workflow.nodes.forEach(nodeData => {
    const node = createNode(nodeData.agentType, nodeData.x, nodeData.y);
    node.id = nodeData.id;
    node.config = nodeData.config || {};
    nodes.push(node);

    // Render node on canvas
    renderNode(node);
  });

  // Load connections
  if (workflow.connections) {
    workflow.connections.forEach(connData => {
      const fromNode = nodes.find(n => n.id === connData.from);
      const toNode = nodes.find(n => n.id === connData.to);

      if (fromNode && toNode) {
        connections.push({
          id: connData.id,
          from: fromNode.id,
          to: toNode.id
        });
      }
    });
    redrawConnections();
  }

  console.log(`Loaded workflow with ${nodes.length} nodes and ${connections.length} connections`);
}
```

## Full Integration Example

Here's what the workflow builder needs to support:

```javascript
// Inside workflow_builder.js

// 1. At the top (after other global variables)
let currentTemplate = null;

// 2. In the initialization function
function initBuilder() {
  // Existing init code...

  // New: Check for template parameter
  const params = new URLSearchParams(window.location.search);
  const templateFile = params.get('template');

  if (templateFile) {
    loadTemplate(templateFile);
  }
}

// 3. Add template loading function
async function loadTemplate(templateFilename) {
  try {
    const response = await fetch(`example_workflows/${templateFilename}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const workflow = await response.json();
    currentTemplate = workflow;

    // Load into builder
    loadWorkflow(workflow);

    // Update UI
    updateTemplateInfo(workflow);

  } catch (error) {
    console.error('Failed to load template:', error);
    alert(`Failed to load template: ${error.message}`);
  }
}

// 4. Add UI update function
function updateTemplateInfo(workflow) {
  // Optional: Show which template is loaded
  const titleElement = document.querySelector('.workflow-title');
  if (titleElement) {
    titleElement.textContent = workflow.name;
  }
}

// 5. Modify existing node creation to support loading
function loadWorkflow(workflow) {
  // Clear existing
  clearCanvas();

  // Create nodes from workflow
  workflow.nodes.forEach(nodeConfig => {
    const agent = AGENT_TYPES[nodeConfig.agentType];
    if (!agent) {
      console.warn(`Unknown agent type: ${nodeConfig.agentType}`);
      return;
    }

    const node = {
      id: nodeConfig.id || `node-${Date.now()}`,
      agentType: nodeConfig.agentType,
      x: nodeConfig.x,
      y: nodeConfig.y,
      config: nodeConfig.config || {},
      title: agent.name,
      icon: agent.icon
    };

    nodes.push(node);
    renderNode(node);
  });

  // Create connections
  if (workflow.connections) {
    connections = workflow.connections;
    redrawConnections();
  }
}
```

## Testing the Integration

### Test 1: Load Template from Gallery
```
1. Open template_gallery.html
2. Click "Use" on any template
3. Click "Use Template" in modal
4. Verify workflow_builder.html opens
5. Verify template nodes appear on canvas
6. Verify title shows template name
```

### Test 2: Direct URL
```
1. Open: workflow_builder.html?template=research_pipeline.json
2. Verify template loads automatically
3. Check browser console (no errors)
4. Verify nodes rendered correctly
5. Verify connections drawn
```

### Test 3: Button Link
```
1. Open workflow_builder.html
2. Click "📋 Templates" button in toolbar
3. Verify template_gallery.html opens
4. Load a template
5. Verify redirects back to builder with template
```

## Error Handling

Make sure to handle these edge cases:

```javascript
// 1. Missing file
try {
  const response = await fetch(`example_workflows/${templateFile}`);
  if (!response.ok) {
    throw new Error(`Template not found: ${templateFile}`);
  }
  // ...
} catch (error) {
  showError(`Failed to load template: ${error.message}`);
}

// 2. Invalid workflow structure
if (!workflow.nodes || !Array.isArray(workflow.nodes)) {
  throw new Error('Invalid workflow: missing nodes array');
}

// 3. Unknown agent types
workflow.nodes.forEach(nodeConfig => {
  if (!AGENT_TYPES[nodeConfig.agentType]) {
    console.warn(`Skipping unknown agent type: ${nodeConfig.agentType}`);
  }
});

// 4. Missing connections
if (workflow.connections && Array.isArray(workflow.connections)) {
  // Process connections
}
```

## Optional: Add "Back to Gallery" Button

In the workflow builder, add a way to go back:

```html
<!-- Add to canvas header -->
<a href="template_gallery.html" class="toolbar-btn" title="Back to templates">
  ← Gallery
</a>
```

Or keep templates open in a sidebar:

```javascript
// Optional: Show template browser as modal in builder
function showTemplatePanel() {
  const modal = document.createElement('div');
  modal.className = 'template-panel';
  modal.innerHTML = `
    <h3>Load Template</h3>
    <button onclick="closeTemplatePanel()">Close</button>
    <iframe src="template_gallery.html" style="width:100%;height:400px;"></iframe>
  `;
  document.body.appendChild(modal);
}
```

## File Locations

```
hololoom/web_dashboard/
├── template_gallery.html          ← Gallery UI
├── template_gallery.js            ← Gallery features
├── workflow_builder.html          ← Builder (add integration)
├── workflow_builder.js            ← Builder logic (add loading)
└── example_workflows/
    ├── research_pipeline.json
    ├── safety_gated_query.json
    ├── crm/
    │   ├── lead_scoring_simple.json
    │   ├── multi_factor_scoring.json
    │   └── daily_action_list.json
    └── llm/
        ├── customer_support_triage.json
        └── content_creation.json
```

## Sample Code for Different Scenarios

### Scenario 1: Minimal Integration (Template Loading Only)
```javascript
// In workflow_builder.js, at page load:
const params = new URLSearchParams(window.location.search);
const template = params.get('template');
if (template) {
  fetch(`example_workflows/${template}`)
    .then(r => r.json())
    .then(w => loadWorkflow(w))
    .catch(e => console.error('Failed to load template:', e));
}
```

### Scenario 2: Full Integration (With UI Updates)
```javascript
// In workflow_builder.js:
class WorkflowBuilder {
  constructor() {
    this.currentTemplate = null;
    this.nodes = [];
    this.connections = [];
    this.init();
  }

  init() {
    this.loadTemplateFromUrl();
    this.setupEventListeners();
  }

  loadTemplateFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const templateFile = params.get('template');

    if (templateFile) {
      this.loadTemplate(templateFile);
    }
  }

  async loadTemplate(filename) {
    try {
      const response = await fetch(`example_workflows/${filename}`);
      const workflow = await response.json();

      this.currentTemplate = workflow;
      this.loadWorkflow(workflow);
      this.updateUI();

    } catch (error) {
      console.error('Load failed:', error);
      this.showError(`Failed to load template: ${error.message}`);
    }
  }

  loadWorkflow(workflow) {
    this.nodes = workflow.nodes || [];
    this.connections = workflow.connections || [];
    this.renderCanvas();
  }

  updateUI() {
    document.querySelector('.workflow-title').textContent =
      this.currentTemplate.name;
  }

  renderCanvas() {
    const canvas = document.getElementById('canvas');
    canvas.innerHTML = '';

    this.nodes.forEach(node => this.renderNode(node));
    this.renderConnections();
  }

  // ... other methods ...
}
```

### Scenario 3: Advanced (With Session Storage)
```javascript
// Save current work before loading template
function loadTemplate(filename) {
  if (nodes.length > 0) {
    if (!confirm('Load template? Current work will be cleared.')) {
      return;
    }
    // Optionally save current workflow
    const current = {
      nodes: nodes,
      connections: connections,
      timestamp: new Date().toISOString()
    };
    sessionStorage.setItem('lastWorkflow', JSON.stringify(current));
  }

  // Now load template
  fetch(`example_workflows/${filename}`)
    .then(r => r.json())
    .then(workflow => {
      clearCanvas();
      loadWorkflow(workflow);
      showNotification(`Loaded: ${workflow.name}`);
    });
}

// Allow undoing template load
function restoreLastWorkflow() {
  const saved = sessionStorage.getItem('lastWorkflow');
  if (saved) {
    loadWorkflow(JSON.parse(saved));
    showNotification('Restored previous workflow');
  }
}
```

## Validation Checklist

Before deploying the integration:

- [ ] Gallery opens successfully
- [ ] Can click "Use" button
- [ ] Modal shows template details
- [ ] "Use Template" button redirects to builder
- [ ] Builder receives template via URL parameter
- [ ] Template loads correctly
- [ ] Nodes appear on canvas
- [ ] Connections are drawn
- [ ] No console errors
- [ ] Works on different browsers
- [ ] Mobile responsive
- [ ] "Templates" button added to builder
- [ ] Back navigation works

## Deployment Checklist

Before going live:

- [ ] All integration code tested
- [ ] Error handling in place
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] Browser compatibility verified
- [ ] Performance acceptable
- [ ] Accessibility checked
- [ ] Mobile tested
- [ ] Backup of original files
- [ ] Ready for production

## Support

If integration doesn't work:

1. **Check console** (F12 → Console) for errors
2. **Verify file paths** match `example_workflows/` structure
3. **Test directly** with: `workflow_builder.html?template=research_pipeline.json`
4. **Check permissions** that files are readable
5. **Clear cache** if files changed

---

**Integration Status**: Ready to implement
**Complexity**: Low (2-3 code additions)
**Time Estimate**: 15-30 minutes
**Testing Time**: 20-30 minutes

Next step: Follow the "Quick Integration (2 Steps)" section above!
