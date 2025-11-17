# HoloLoom Enhanced Workflow Builder - Quick Start Guide

Welcome to the HoloLoom Enhanced Workflow Builder! This guide will get you up and running in 5 minutes.

---

## 🚀 Quick Start (30 seconds)

### 1. Start the Backend

```bash
cd /home/user/hello-world/HoloLoom/web_dashboard
PYTHONPATH=../.. python workflow_executor.py
```

You should see:
```
Starting server...
📚 NEW FEATURES:
  • AI Workflow Generation
  • 5 Pre-built Templates
  • Real-time Analytics
  • Collaborative Editing
  • Workflow Marketplace
```

### 2. Open the Frontend

Open `workflow_builder_enhanced.html` in your browser:

```bash
# Option 1: Direct file open
open workflow_builder_enhanced.html

# Option 2: Local HTTP server (recommended)
python -m http.server 8080
# Then visit: http://localhost:8080/workflow_builder_enhanced.html
```

### 3. Start Building!

You'll see 5 tabs at the top:
- **🎨 Canvas** - Drag & drop workflow builder
- **🤖 AI Generator** - Natural language workflow creation
- **📊 Analytics** - Performance metrics
- **👥 Collaborate** - Real-time collaboration
- **🏪 Marketplace** - Share & discover workflows

---

## 📚 Feature Walkthroughs

### Canvas: Build Workflows Visually

**Create Your First Workflow**:

1. Drag an agent from the left sidebar onto the canvas
2. Drag another agent
3. Click the **output port** (right side) of the first node
4. Click the **input port** (left side) of the second node
5. Nodes are now connected!

**Execute Your Workflow**:

1. Click the **▶️ Execute** button in the top-right
2. Enter an input query when prompted
3. Watch the execution in real-time
4. Check the Analytics tab for performance metrics

**Save Your Work**:

- Click **💾 Export** to download as JSON
- Click **📚 Templates** to browse pre-built workflows
- Click **📜 History** to view version history

---

### AI Generator: Natural Language Workflows

**Generate a Workflow**:

1. Switch to the **🤖 AI Generator** tab
2. Describe your workflow in plain English:
   ```
   Create a workflow that analyzes Python code for security
   issues, generates a report, and suggests fixes
   ```
3. Click **✨ Generate Workflow**
4. Preview the generated workflow
5. Click **✅ Apply to Canvas**
6. Switch to **🎨 Canvas** tab to see the result!

**Refine an Existing Workflow**:

1. Build a workflow on the Canvas
2. Switch to **🤖 AI Generator**
3. Enter refinement instructions:
   ```
   Add error handling to all nodes and make processing parallel
   ```
4. Click **🔧 Refine Workflow**
5. Apply the refinements

---

### Analytics: Monitor Performance

**View Workflow Metrics**:

1. Execute a workflow at least once (from Canvas tab)
2. Switch to **📊 Analytics** tab
3. See key metrics:
   - Total executions
   - Success rate
   - Average duration
   - P95 latency
4. Check the **Node Performance** table for bottlenecks
5. Look for **⚠️ Bottleneck Warnings**

**Export Analytics**:

- Click **📥 Export CSV** to download metrics
- Click **🔄 Refresh** to update data

---

### Collaboration: Work Together

**Start a Collaborative Session**:

1. Switch to **👥 Collaborate** tab
2. Enter your name: `Alice`
3. Enter a session ID: `my-team-workflow` (or leave blank)
4. Click **🚀 Start/Join Session**
5. Share the session ID with teammates
6. See real-time updates as others join!

**Collaborate in Real-Time**:

- See **Active Users** list
- Watch the **Activity Feed** for changes
- Make edits on the Canvas
- Click **📋 Copy** to share the session ID
- Click **🚪 Leave Session** when done

**Open Multiple Tabs** (for testing):

1. Open the workflow builder in 2+ browser tabs
2. Join the same session in each tab
3. Make changes in one tab
4. See updates appear in other tabs automatically!

---

### Marketplace: Share & Discover

**Browse Workflows**:

1. Switch to **🏪 Marketplace** tab
2. Browse available workflows
3. Use the **search bar** to find specific workflows
4. Filter by **category** (RAG, Code, LLM, etc.)
5. Click a workflow card to download
6. Workflow loads automatically into Canvas!

**Publish Your Workflow**:

1. Create an awesome workflow on Canvas
2. Switch to **🏪 Marketplace**
3. Click **📤 Publish Current Workflow**
4. Fill in the form:
   - **Name**: "My Amazing Workflow"
   - **Description**: "Does amazing things..."
   - **Author**: "Your Name"
   - **Category**: Select appropriate category
   - **Tags**: "workflow, automation, awesome"
5. Click **Publish**
6. Your workflow appears in the marketplace!

---

## 🎯 Example Workflows

### Example 1: Simple Query → Response

**Goal**: Process a query and generate a response

**Steps**:
1. Drag **HoloLoom Query** to canvas
2. Drag **Response Generator** to canvas
3. Connect them
4. Execute with query: "What is Thompson Sampling?"

### Example 2: RAG Research Pipeline

**Use the Template**:
1. Click **📚 Templates**
2. Select **RAG Research Pipeline**
3. Workflow loads with 7 pre-configured nodes
4. Execute to see it work!

**Or Build Manually**:
1. **Multi-Query** → breaks question into sub-questions
2. **HoloLoom Query** (×5) → processes each sub-question
3. **Synthesizer** → combines results
4. **Response Generator** → final output

### Example 3: AI-Generated Code Analysis

**Use AI Generator**:
1. Go to **🤖 AI Generator**
2. Enter:
   ```
   Create a workflow that takes Python code as input,
   analyzes it for security vulnerabilities, extracts
   issues with line numbers, and generates a formatted
   report with fix suggestions
   ```
3. Click **✨ Generate Workflow**
4. Apply to canvas
5. Execute with your Python code!

---

## 🔧 Tips & Tricks

### Keyboard Shortcuts

- **Delete**: Delete selected node
- **Ctrl/Cmd + S**: Export workflow
- **Ctrl/Cmd + Enter**: Execute workflow
- **Escape**: Deselect all
- **?**: Show shortcuts help

### Zoom Controls

- **🔍+**: Zoom in
- **🔍-**: Zoom out
- **⊙**: Reset zoom to 100%
- **Mouse wheel**: Scroll to zoom (when implemented)

### Node Configuration

1. Click a node to select it
2. View properties in the **right sidebar**
3. Modify configuration as needed
4. Changes apply immediately

### Connection Management

- **Click output port → input port**: Create connection
- **Click connection line**: Delete connection
- Connections auto-update when moving nodes

### Templates

Pre-built templates available:
- **RAG Research**: Multi-query research with synthesis
- **Code Review**: Automated code analysis
- **Security Scanner**: Vulnerability detection
- **Data Pipeline**: ETL workflow
- **Simple Q&A**: Basic query → response

### Saving & Loading

**Export**:
- Click **💾 Export**
- Saves as JSON file
- Share with teammates

**Import**:
- Click **📁 Import**
- Select JSON file
- Workflow loads instantly

**Version Control**:
- Click **💾 Save** to create version
- Click **📜 History** to browse versions
- Click **🔀 Branch** to create experimental version

---

## 🐛 Troubleshooting

### Backend Not Running

**Error**: "Failed to connect to API"

**Solution**:
```bash
cd HoloLoom/web_dashboard
PYTHONPATH=../.. python workflow_executor.py
```

Make sure you see "Starting server..." message.

### CORS Errors

**Error**: "Cross-Origin Request Blocked"

**Solution**: Open the HTML file via HTTP server, not as `file://`:

```bash
python -m http.server 8080
# Visit: http://localhost:8080/workflow_builder_enhanced.html
```

### WebSocket Connection Failed

**Error**: "WebSocket connection failed"

**Solution**: Check backend is running on port 8001:
```bash
curl http://localhost:8001/health
```

Should return: `{"status":"healthy",...}`

### No Analytics Data

**Issue**: Analytics shows all zeros

**Reason**: No workflows executed yet

**Solution**:
1. Go to Canvas
2. Create a workflow
3. Click **▶️ Execute**
4. Go to Analytics to see metrics

### Marketplace Empty

**Issue**: No workflows in marketplace

**Reason**: Fresh installation

**Solution**: Be the first to publish!
1. Create a workflow
2. Switch to Marketplace
3. Click **📤 Publish**

---

## 📖 API Reference

For developers who want to integrate:

### API Base URL

```
http://localhost:8001
```

### Key Endpoints

```
GET  /health                      - Health check
POST /api/workflow/execute        - Execute workflow
POST /api/workflow/generate       - AI generation
POST /api/workflow/refine         - AI refinement
GET  /api/templates/list          - List templates
GET  /api/analytics/{id}          - Get analytics
WS   /ws/collaborate/{session}    - Collaboration
GET  /api/marketplace/list        - Browse marketplace
POST /api/marketplace/publish     - Publish workflow
```

See `API_INTEGRATION.md` for complete documentation.

---

## 🎓 Next Steps

### Learn More

- Read `INTEGRATION_REPORT.md` for detailed feature documentation
- Check `API_INTEGRATION.md` for API reference
- Explore `workflow_executor.py` for backend implementation

### Get Help

- Check the **📚 Templates** for examples
- Use **🤖 AI Generator** to build workflows automatically
- Join a **👥 Collaborative Session** with teammates

### Contribute

- Build amazing workflows
- Publish to **🏪 Marketplace**
- Share with the community

---

## 🌟 Pro Tips

1. **Start Simple**: Begin with 2-3 nodes, add complexity later
2. **Use Templates**: Don't reinvent the wheel
3. **AI is Your Friend**: Let AI generate complex workflows
4. **Monitor Performance**: Check Analytics regularly
5. **Collaborate Early**: Get feedback from teammates
6. **Publish & Share**: Help the community grow

---

## ✨ What's Next?

Planned features:
- **Visual cursor sync** in collaboration
- **Chart visualizations** in analytics
- **More templates** in marketplace
- **Keyboard shortcuts** for power users
- **Mobile improvements** for tablets
- **Export to Python** code generation

---

**Happy Workflow Building!** 🎉

For questions or issues, check the documentation or create an issue in the repository.

---

**Last Updated**: 2025-11-17
**Version**: 1.0.0
**Status**: Production Ready ✅
