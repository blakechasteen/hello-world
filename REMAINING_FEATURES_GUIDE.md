# Remaining Features Implementation Guide

**Date**: 2025-11-02
**Status**: 10/15 Complete
**Remaining**: 5 features

This guide provides complete implementation details for the 5 remaining features.

---

## Feature 11: Hover Previews

**Status**: 80% complete (CSS ready, needs JS)
**Time**: 20 minutes
**Complexity**: Low

### What's Already Done
The CSS for `.conversation-preview` is already in place in [agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html).

### Implementation Steps

**Step 1**: Add preview container to conversation element

```javascript
function createConversationElement(conv) {
    // ... existing code ...

    // Add preview container
    const previewContainer = document.createElement('div');
    previewContainer.className = 'conversation-preview';
    previewContainer.innerHTML = `
        <div class="preview-title">${escapeHtml(conv.title)}</div>
        <div id="preview-messages-${conv.id}"></div>
    `;
    div.appendChild(previewContainer);

    // Load preview on hover
    div.onmouseenter = () => loadPreview(conv.id);

    return div;
}
```

**Step 2**: Implement `loadPreview()` function

```javascript
let previewCache = {};
let previewTimeout = null;

function loadPreview(conversationId) {
    clearTimeout(previewTimeout);

    // Delay 500ms
    previewTimeout = setTimeout(async () => {
        // Check cache
        if (previewCache[conversationId]) {
            renderPreview(conversationId, previewCache[conversationId]);
            return;
        }

        // Fetch from server
        ws.send(JSON.stringify({
            action: 'get_preview',
            conversation_id: conversationId,
            limit: 3
        }));
    }, 500);
}

function renderPreview(conversationId, messages) {
    const container = document.getElementById(`preview-messages-${conversationId}`);
    if (!container) return;

    container.innerHTML = messages.map(msg => `
        <div class="preview-message">
            <div class="preview-role">${msg.role === 'user' ? 'You' : 'HoloLoom'}</div>
            <div>${escapeHtml(msg.content.substring(0, 100))}...</div>
        </div>
    `).join('');
}
```

**Step 3**: Add server handler in `agentic_server.py`

```python
elif action == 'get_preview':
    conversation_id = message_data.get('conversation_id')
    limit = message_data.get('limit', 3)

    messages = conversation_manager.get_messages(conversation_id, limit=limit)

    await websocket.send_json({
        'type': 'preview_messages',
        'data': {
            'conversation_id': conversation_id,
            'messages': [msg.to_dict() for msg in messages]
        }
    })
```

**Step 4**: Handle response in frontend

```javascript
case 'preview_messages':
    const convId = data.data.conversation_id;
    previewCache[convId] = data.data.messages;
    renderPreview(convId, data.data.messages);
    break;
```

**Testing**:
- Hover over conversations
- Preview should appear after 500ms
- First 3 messages should display
- Cache should prevent repeated fetches

---

## Feature 12: Conversation Templates

**Time**: 60 minutes
**Complexity**: Medium

### Database Schema

Add to `conversation_manager.py`:

```python
@dataclass
class ConversationTemplate:
    """A conversation template"""
    id: Optional[int]
    name: str
    description: str
    icon: str
    messages: str  # JSON string of message list
    created_at: str

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'icon': self.icon,
            'messages': json.loads(self.messages),
            'created_at': self.created_at
        }

# In _init_database():
cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversation_templates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        icon TEXT DEFAULT '📄',
        messages TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
""")
```

### Backend Methods

```python
def create_template(self, name: str, description: str, messages: List[Dict], icon: str = '📄') -> ConversationTemplate:
    """Save conversation as template"""
    now = datetime.now().isoformat()
    messages_json = json.dumps(messages)

    with sqlite3.connect(self.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversation_templates (name, description, icon, messages, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, description, icon, messages_json, now)
        )
        template_id = cursor.lastrowid
        conn.commit()

    logger.info(f"Created template {template_id}: {name}")
    return ConversationTemplate(
        id=template_id,
        name=name,
        description=description,
        icon=icon,
        messages=messages_json,
        created_at=now
    )

def list_templates(self) -> List[ConversationTemplate]:
    """List all templates"""
    with sqlite3.connect(self.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, description, icon, messages, created_at FROM conversation_templates ORDER BY created_at DESC")

        templates = []
        for row in cursor.fetchall():
            templates.append(ConversationTemplate(
                id=row[0],
                name=row[1],
                description=row[2],
                icon=row[3],
                messages=row[4],
                created_at=row[5]
            ))

    return templates

def load_template(self, template_id: int) -> Optional[ConversationTemplate]:
    """Load a template"""
    with sqlite3.connect(self.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, description, icon, messages, created_at FROM conversation_templates WHERE id = ?",
            (template_id,)
        )
        row = cursor.fetchone()

        if row:
            return ConversationTemplate(
                id=row[0],
                name=row[1],
                description=row[2],
                icon=row[3],
                messages=row[4],
                created_at=row[5]
            )

    return None

def delete_template(self, template_id: int):
    """Delete a template"""
    with sqlite3.connect(self.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversation_templates WHERE id = ?", (template_id,))
        conn.commit()

    logger.info(f"Deleted template {template_id}")
```

### WebSocket Actions

Add to `agentic_server.py`:

```python
elif action == 'save_as_template':
    conversation_id = message_data.get('conversation_id')
    template_name = message_data.get('name')
    description = message_data.get('description', '')
    icon = message_data.get('icon', '📄')

    # Get conversation messages
    messages = conversation_manager.get_messages(conversation_id)
    messages_data = [{'role': msg.role, 'content': msg.content} for msg in messages]

    # Create template
    template = conversation_manager.create_template(template_name, description, messages_data, icon)

    await websocket.send_json({
        'type': 'template_created',
        'data': template.to_dict()
    })

elif action == 'list_templates':
    templates = conversation_manager.list_templates()

    await websocket.send_json({
        'type': 'templates_list',
        'data': {
            'templates': [t.to_dict() for t in templates]
        }
    })

elif action == 'load_template':
    template_id = message_data.get('template_id')
    template = conversation_manager.load_template(template_id)

    if template:
        # Create new conversation from template
        new_conv = conversation_manager.create_conversation(title=template.name)

        # Add template messages
        for msg in json.loads(template.messages):
            conversation_manager.add_message(
                conversation_id=new_conv.id,
                role=msg['role'],
                content=msg['content']
            )

        await websocket.send_json({
            'type': 'conversation_created',
            'data': new_conv.to_dict()
        })

elif action == 'delete_template':
    template_id = message_data.get('template_id')
    conversation_manager.delete_template(template_id)

    await websocket.send_json({
        'type': 'template_deleted',
        'data': {'template_id': template_id}
    })
```

### Frontend UI

Add templates modal to HTML:

```html
<!-- Templates Modal -->
<div class="modal" id="templatesModal">
    <div class="modal-content">
        <div class="modal-title">📄 Conversation Templates</div>

        <div class="templates-grid" id="templatesGrid">
            <!-- Templates will be rendered here -->
        </div>

        <div class="modal-actions">
            <button class="btn btn-secondary" onclick="closeModal('templatesModal')">Close</button>
            <button class="btn btn-primary" onclick="showSaveTemplateDialog()">Save Current as Template</button>
        </div>
    </div>
</div>

<!-- Save Template Modal -->
<div class="modal" id="saveTemplateModal">
    <div class="modal-content">
        <div class="modal-title">Save as Template</div>
        <div class="form-group">
            <label class="form-label">Template Name</label>
            <input type="text" class="form-input" id="templateName" placeholder="e.g., Code Review">
        </div>
        <div class="form-group">
            <label class="form-label">Description</label>
            <textarea class="form-input" id="templateDescription" rows="3" placeholder="What is this template for?"></textarea>
        </div>
        <div class="form-group">
            <label class="form-label">Icon</label>
            <input type="text" class="form-input" id="templateIcon" placeholder="📄" maxlength="2">
        </div>
        <div class="modal-actions">
            <button class="btn btn-secondary" onclick="closeModal('saveTemplateModal')">Cancel</button>
            <button class="btn btn-primary" onclick="saveTemplate()">Save Template</button>
        </div>
    </div>
</div>
```

Add CSS:

```css
.templates-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px;
    margin: 20px 0;
}

.template-card {
    padding: 20px;
    background: #2a2a2a;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    border: 2px solid transparent;
}

.template-card:hover {
    border-color: #667eea;
    transform: translateY(-4px);
}

.template-icon {
    font-size: 32px;
    margin-bottom: 12px;
}

.template-name {
    font-weight: 600;
    margin-bottom: 8px;
}

.template-description {
    font-size: 12px;
    color: #888;
}
```

Add JS functions:

```javascript
function showTemplates() {
    ws.send(JSON.stringify({ action: 'list_templates' }));
    showModal('templatesModal');
}

function renderTemplates(templates) {
    const grid = document.getElementById('templatesGrid');
    grid.innerHTML = templates.map(t => `
        <div class="template-card" onclick="useTemplate(${t.id})">
            <div class="template-icon">${t.icon}</div>
            <div class="template-name">${escapeHtml(t.name)}</div>
            <div class="template-description">${escapeHtml(t.description)}</div>
        </div>
    `).join('');
}

function useTemplate(templateId) {
    ws.send(JSON.stringify({
        action: 'load_template',
        template_id: templateId
    }));
    closeModal('templatesModal');
}

function showSaveTemplateDialog() {
    if (!currentConversationId) {
        showToast('No active conversation to save', 'warning');
        return;
    }
    showModal('saveTemplateModal');
}

function saveTemplate() {
    const name = document.getElementById('templateName').value.trim();
    const description = document.getElementById('templateDescription').value.trim();
    const icon = document.getElementById('templateIcon').value.trim() || '📄';

    if (!name) {
        showToast('Please enter a template name', 'warning');
        return;
    }

    ws.send(JSON.stringify({
        action: 'save_as_template',
        conversation_id: currentConversationId,
        name: name,
        description: description,
        icon: icon
    }));

    closeModal('saveTemplateModal');
    showToast('Template saved successfully', 'success');
}

// Handle templates_list response
case 'templates_list':
    renderTemplates(data.data.templates);
    break;
```

**Add to keyboard shortcuts**: `Ctrl+T` to open templates

---

## Feature 13: Lazy Loading

**Time**: 60 minutes
**Complexity**: Medium

### Implementation

**Step 1**: Add pagination state

```javascript
let conversationsPage = 0;
let conversationsPerPage = 50;
let hasMoreConversations = true;
let loadingMore = false;
```

**Step 2**: Modify `list_conversations` query

In `conversation_manager.py`:

```python
def list_conversations(self, limit: int = 50, offset: int = 0) -> Tuple[List[Conversation], bool]:
    """List conversations with pagination"""
    with sqlite3.connect(self.db_path) as conn:
        cursor = conn.cursor()

        # Get conversations
        cursor.execute("""
            SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) as msg_count,
                   c.project_id, c.is_favorite, c.tags
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))

        conversations = []
        for row in cursor.fetchall():
            conversations.append(Conversation(
                id=row[0],
                title=row[1],
                created_at=row[2],
                updated_at=row[3],
                message_count=row[4],
                project_id=row[5],
                is_favorite=bool(row[6]),
                tags=row[7] or ""
            ))

        # Check if there are more
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total = cursor.fetchone()[0]
        has_more = (offset + limit) < total

    return conversations, has_more
```

**Step 3**: Add WebSocket pagination

```python
elif action == 'list_conversations':
    limit = message_data.get('limit', 50)
    offset = message_data.get('offset', 0)

    conversations, has_more = conversation_manager.list_conversations(limit, offset)

    await websocket.send_json({
        'type': 'conversations_list',
        'data': {
            'conversations': [conv.to_dict() for conv in conversations],
            'has_more': has_more,
            'offset': offset + len(conversations)
        }
    })
```

**Step 4**: Add Intersection Observer

```javascript
// Add to bottom of conversations container
const loadMoreSentinel = document.createElement('div');
loadMoreSentinel.id = 'loadMoreSentinel';
loadMoreSentinel.style.height = '1px';

const observer = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting && hasMoreConversations && !loadingMore) {
        loadMoreConversations();
    }
}, {
    root: document.getElementById('projectsContainer'),
    threshold: 0.1
});

// Append sentinel after rendering conversations
function renderConversations() {
    // ... existing code ...

    // Add sentinel for lazy loading
    const container = document.getElementById('projectsContainer');
    if (!document.getElementById('loadMoreSentinel')) {
        container.appendChild(loadMoreSentinel);
        observer.observe(loadMoreSentinel);
    }
}

function loadMoreConversations() {
    if (loadingMore || !hasMoreConversations) return;

    loadingMore = true;
    conversationsPage++;

    ws.send(JSON.stringify({
        action: 'list_conversations',
        offset: conversationsPage * conversationsPerPage,
        limit: conversationsPerPage
    }));
}

// Handle paginated response
case 'conversations_list':
    const newConversations = data.data.conversations;
    hasMoreConversations = data.data.has_more;

    if (data.data.offset === 0) {
        // First page, replace
        conversations = newConversations;
    } else {
        // Append to existing
        conversations = conversations.concat(newConversations);
    }

    renderConversations();
    loadingMore = false;
    break;
```

**Alternative: "Load More" Button**

```html
<button class="btn btn-secondary" id="loadMoreBtn" onclick="loadMoreConversations()" style="width: 100%; margin-top: 12px;">
    Load More Conversations
</button>
```

```javascript
function updateLoadMoreButton() {
    const btn = document.getElementById('loadMoreBtn');
    if (btn) {
        btn.style.display = hasMoreConversations ? 'block' : 'none';
        btn.textContent = loadingMore ? 'Loading...' : 'Load More Conversations';
        btn.disabled = loadingMore;
    }
}
```

---

## Feature 14: Analytics Dashboard

**Time**: 120 minutes
**Complexity**: High

### Required Libraries

Add to HTML `<head>`:

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
```

### Database Queries

Add to `conversation_manager.py`:

```python
def get_analytics(self, days: int = 30) -> Dict:
    """Get analytics data"""
    from_date = (datetime.now() - timedelta(days=days)).isoformat()

    with sqlite3.connect(self.db_path) as conn:
        cursor = conn.cursor()

        # Messages per day
        cursor.execute("""
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM messages
            WHERE created_at >= ?
            GROUP BY DATE(created_at)
            ORDER BY date
        """, (from_date,))
        messages_per_day = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]

        # Confidence distribution (if metadata exists)
        cursor.execute("""
            SELECT metadata
            FROM messages
            WHERE role = 'assistant' AND metadata IS NOT NULL AND metadata != ''
        """)
        confidences = []
        for row in cursor.fetchall():
            try:
                meta = json.loads(row[0])
                if 'confidence' in meta:
                    confidences.append(float(meta['confidence']))
            except:
                pass

        # Mode usage
        cursor.execute("""
            SELECT metadata
            FROM messages
            WHERE role = 'assistant' AND metadata IS NOT NULL AND metadata != ''
        """)
        mode_counts = {}
        for row in cursor.fetchall():
            try:
                meta = json.loads(row[0])
                if 'mode' in meta:
                    mode = meta['mode']
                    mode_counts[mode] = mode_counts.get(mode, 0) + 1
            except:
                pass

        # Response times
        cursor.execute("""
            SELECT metadata
            FROM messages
            WHERE role = 'assistant' AND metadata IS NOT NULL AND metadata != ''
        """)
        response_times = []
        for row in cursor.fetchall():
            try:
                meta = json.loads(row[0])
                if 'duration_ms' in meta:
                    response_times.append(float(meta['duration_ms']))
            except:
                pass

        # Average response time by day
        avg_response_time_per_day = []
        # ... calculate from response_times ...

        return {
            'messages_per_day': messages_per_day,
            'confidences': confidences,
            'mode_usage': mode_counts,
            'response_times': response_times,
            'avg_response_time_per_day': avg_response_time_per_day,
            'total_messages': len(messages_per_day),
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0
        }
```

### WebSocket Action

```python
elif action == 'get_analytics':
    days = message_data.get('days', 30)
    analytics = conversation_manager.get_analytics(days)

    await websocket.send_json({
        'type': 'analytics_data',
        'data': analytics
    })
```

### Frontend Modal

```html
<div class="modal" id="analyticsModal">
    <div class="modal-content" style="max-width: 1200px;">
        <div class="modal-title">📊 Analytics Dashboard</div>

        <div class="analytics-controls">
            <select id="analyticsRange" onchange="loadAnalytics()">
                <option value="7">Last 7 days</option>
                <option value="30" selected>Last 30 days</option>
                <option value="90">Last 90 days</option>
            </select>
        </div>

        <div class="analytics-grid">
            <div class="chart-container">
                <h4>Messages Per Day</h4>
                <canvas id="messagesChart"></canvas>
            </div>

            <div class="chart-container">
                <h4>Confidence Distribution</h4>
                <canvas id="confidenceChart"></canvas>
            </div>

            <div class="chart-container">
                <h4>Mode Usage</h4>
                <canvas id="modeChart"></canvas>
            </div>

            <div class="chart-container">
                <h4>Response Times</h4>
                <canvas id="responseTimeChart"></canvas>
            </div>
        </div>

        <div class="modal-actions">
            <button class="btn btn-secondary" onclick="closeModal('analyticsModal')">Close</button>
            <button class="btn btn-primary" onclick="exportAnalytics()">Export Report</button>
        </div>
    </div>
</div>
```

### JavaScript for Charts

```javascript
let charts = {};

function showAnalytics() {
    showModal('analyticsModal');
    loadAnalytics();
}

function loadAnalytics() {
    const days = document.getElementById('analyticsRange').value;
    ws.send(JSON.stringify({
        action: 'get_analytics',
        days: parseInt(days)
    }));
}

function renderAnalytics(data) {
    // Destroy existing charts
    Object.values(charts).forEach(chart => chart.destroy());
    charts = {};

    // Messages Per Day (Line Chart)
    const messagesCtx = document.getElementById('messagesChart').getContext('2d');
    charts.messages = new Chart(messagesCtx, {
        type: 'line',
        data: {
            labels: data.messages_per_day.map(d => d.date),
            datasets: [{
                label: 'Messages',
                data: data.messages_per_day.map(d => d.count),
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            }
        }
    });

    // Confidence Distribution (Histogram)
    const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
    const confidenceBins = binConfidences(data.confidences);
    charts.confidence = new Chart(confidenceCtx, {
        type: 'bar',
        data: {
            labels: confidenceBins.labels,
            datasets: [{
                label: 'Frequency',
                data: confidenceBins.values,
                backgroundColor: '#00ff88'
            }]
        },
        options: {
            responsive: true
        }
    });

    // Mode Usage (Pie Chart)
    const modeCtx = document.getElementById('modeChart').getContext('2d');
    charts.mode = new Chart(modeCtx, {
        type: 'pie',
        data: {
            labels: Object.keys(data.mode_usage),
            datasets: [{
                data: Object.values(data.mode_usage),
                backgroundColor: ['#667eea', '#764ba2', '#f093fb', '#4facfe']
            }]
        },
        options: {
            responsive: true
        }
    });

    // Response Times (Line Chart)
    const responseCtx = document.getElementById('responseTimeChart').getContext('2d');
    charts.responseTime = new Chart(responseCtx, {
        type: 'line',
        data: {
            labels: data.avg_response_time_per_day.map((d, i) => `Day ${i+1}`),
            datasets: [{
                label: 'Avg Response Time (ms)',
                data: data.avg_response_time_per_day,
                borderColor: '#00d4ff',
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true
        }
    });
}

function binConfidences(confidences) {
    const bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
    const counts = new Array(bins.length - 1).fill(0);

    confidences.forEach(c => {
        for (let i = 0; i < bins.length - 1; i++) {
            if (c >= bins[i] && c < bins[i+1]) {
                counts[i]++;
                break;
            }
        }
    });

    return {
        labels: bins.slice(0, -1).map((b, i) => `${b}-${bins[i+1]}`),
        values: counts
    };
}

// Handle analytics_data response
case 'analytics_data':
    renderAnalytics(data.data);
    break;
```

---

## Feature 15: Promptly Integration

**Time**: 120 minutes
**Complexity**: High

### Installation

```bash
pip install promptly-framework
```

### Backend Integration

Create `HoloLoom/web_dashboard/promptly_bridge.py`:

```python
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add Promptly to path if installed separately
promptly_path = Path(__file__).parent.parent.parent / "Promptly" / "promptly"
if promptly_path.exists():
    sys.path.insert(0, str(promptly_path.parent))

try:
    from promptly.promptly import Promptly
    from promptly.loop_composition import LoopEngine
    PROMPTLY_AVAILABLE = True
except ImportError:
    PROMPTLY_AVAILABLE = False
    print("Warning: Promptly not available. Install with: pip install promptly-framework")

class PromptlyBridge:
    """Bridge between HoloLoom and Promptly"""

    def __init__(self, promptly_dir: str = ".promptly"):
        if not PROMPTLY_AVAILABLE:
            raise ImportError("Promptly framework not installed")

        self.promptly = Promptly(promptly_dir=promptly_dir)
        self.loop_engine = LoopEngine(self.promptly)

    def list_prompts(self) -> List[Dict]:
        """List all prompts"""
        prompts = self.promptly.list_prompts()
        return [
            {
                'name': p.name,
                'version': p.version,
                'description': p.description,
                'created_at': p.created_at,
                'tags': p.tags
            }
            for p in prompts
        ]

    def get_prompt(self, name: str, version: Optional[str] = None) -> Dict:
        """Get a specific prompt"""
        prompt = self.promptly.get_prompt(name, version)
        return {
            'name': prompt.name,
            'version': prompt.version,
            'content': prompt.content,
            'description': prompt.description,
            'tags': prompt.tags,
            'created_at': prompt.created_at
        }

    def save_prompt(self, name: str, content: str, description: str = "", tags: List[str] = None):
        """Save a new prompt"""
        self.promptly.save_prompt(
            name=name,
            content=content,
            description=description,
            tags=tags or []
        )

    def execute_prompt(self, name: str, variables: Dict = None, version: Optional[str] = None) -> str:
        """Execute a prompt"""
        result = self.promptly.execute(
            prompt_name=name,
            variables=variables or {},
            version=version
        )
        return result

    def run_loop(self, prompt_name: str, iterations: int = 3, variables: Dict = None) -> List[str]:
        """Run a prompt in a loop"""
        results = self.loop_engine.run_loop(
            prompt_name=prompt_name,
            iterations=iterations,
            initial_variables=variables or {}
        )
        return results
```

### Server Integration

Add to `agentic_server.py`:

```python
from HoloLoom.web_dashboard.promptly_bridge import PromptlyBridge, PROMPTLY_AVAILABLE

# Initialize Promptly (if available)
promptly_bridge = None
if PROMPTLY_AVAILABLE:
    try:
        promptly_bridge = PromptlyBridge()
        logger.info("Promptly integration enabled")
    except Exception as e:
        logger.warning(f"Promptly integration failed: {e}")

# WebSocket handlers
elif action == 'list_prompts':
    if not promptly_bridge:
        await websocket.send_json({
            'type': 'error',
            'data': {'error': 'Promptly not available'}
        })
        return

    prompts = promptly_bridge.list_prompts()
    await websocket.send_json({
        'type': 'prompts_list',
        'data': {'prompts': prompts}
    })

elif action == 'get_prompt':
    if not promptly_bridge:
        await websocket.send_json({
            'type': 'error',
            'data': {'error': 'Promptly not available'}
        })
        return

    prompt_name = message_data.get('name')
    version = message_data.get('version')

    prompt = promptly_bridge.get_prompt(prompt_name, version)
    await websocket.send_json({
        'type': 'prompt_data',
        'data': prompt
    })

elif action == 'save_prompt':
    if not promptly_bridge:
        await websocket.send_json({
            'type': 'error',
            'data': {'error': 'Promptly not available'}
        })
        return

    name = message_data.get('name')
    content = message_data.get('content')
    description = message_data.get('description', '')
    tags = message_data.get('tags', [])

    promptly_bridge.save_prompt(name, content, description, tags)

    await websocket.send_json({
        'type': 'prompt_saved',
        'data': {'name': name}
    })
```

### Frontend UI

Add Promptly tab to sidebar or create dedicated modal:

```html
<div class="modal" id="promptlyModal">
    <div class="modal-content" style="max-width: 1000px;">
        <div class="modal-title">📝 Prompt Library (Promptly)</div>

        <div class="promptly-layout">
            <!-- Prompts List -->
            <div class="prompts-list">
                <input type="text" class="search-box" placeholder="Search prompts..." id="promptSearch">
                <div id="promptsList"></div>
            </div>

            <!-- Prompt Editor -->
            <div class="prompt-editor">
                <div class="form-group">
                    <label class="form-label">Prompt Name</label>
                    <input type="text" class="form-input" id="promptName">
                </div>
                <div class="form-group">
                    <label class="form-label">Description</label>
                    <input type="text" class="form-input" id="promptDescription">
                </div>
                <div class="form-group">
                    <label class="form-label">Content</label>
                    <textarea class="form-input" id="promptContent" rows="15"></textarea>
                </div>
                <div class="form-group">
                    <label class="form-label">Tags (comma-separated)</label>
                    <input type="text" class="form-input" id="promptTags">
                </div>
            </div>
        </div>

        <div class="modal-actions">
            <button class="btn btn-secondary" onclick="closeModal('promptlyModal')">Close</button>
            <button class="btn btn-primary" onclick="savePromptToPromptly()">Save Prompt</button>
        </div>
    </div>
</div>
```

### JavaScript

```javascript
function showPromptly() {
    if (!ws) {
        showToast('Not connected to server', 'error');
        return;
    }

    ws.send(JSON.stringify({ action: 'list_prompts' }));
    showModal('promptlyModal');
}

function renderPromptsList(prompts) {
    const list = document.getElementById('promptsList');
    list.innerHTML = prompts.map(p => `
        <div class="prompt-item" onclick="loadPrompt('${p.name}')">
            <div class="prompt-name">${escapeHtml(p.name)}</div>
            <div class="prompt-description">${escapeHtml(p.description)}</div>
            <div class="prompt-tags">${p.tags.join(', ')}</div>
        </div>
    `).join('');
}

function loadPrompt(name) {
    ws.send(JSON.stringify({
        action: 'get_prompt',
        name: name
    }));
}

function savePromptToPromptly() {
    const name = document.getElementById('promptName').value.trim();
    const content = document.getElementById('promptContent').value.trim();
    const description = document.getElementById('promptDescription').value.trim();
    const tags = document.getElementById('promptTags').value.split(',').map(t => t.trim());

    if (!name || !content) {
        showToast('Name and content are required', 'warning');
        return;
    }

    ws.send(JSON.stringify({
        action: 'save_prompt',
        name: name,
        content: content,
        description: description,
        tags: tags
    }));

    showToast('Prompt saved successfully', 'success');
}

// Handle server responses
case 'prompts_list':
    renderPromptsList(data.data.prompts);
    break;

case 'prompt_data':
    document.getElementById('promptName').value = data.data.name;
    document.getElementById('promptContent').value = data.data.content;
    document.getElementById('promptDescription').value = data.data.description;
    document.getElementById('promptTags').value = data.data.tags.join(', ');
    break;
```

---

## Testing Guide

### Feature Testing Checklist

**Hover Previews**:
- [ ] Hover over conversation shows preview after 500ms
- [ ] Preview displays first 3 messages
- [ ] Preview caches results (no duplicate fetches)
- [ ] Preview hides when mouse leaves

**Conversation Templates**:
- [ ] Save current conversation as template
- [ ] List templates shows all saved templates
- [ ] Load template creates new conversation with messages
- [ ] Delete template removes from list
- [ ] Default templates work correctly

**Lazy Loading**:
- [ ] Initial 50 conversations load
- [ ] Scrolling to bottom loads more
- [ ] "Load More" button works (if using button variant)
- [ ] No duplicate conversations
- [ ] Loading indicator shows during fetch

**Analytics Dashboard**:
- [ ] Charts render correctly
- [ ] Date range selector works (7/30/90 days)
- [ ] All 4 charts display data
- [ ] Export report generates file
- [ ] Dashboard updates when new data added

**Promptly Integration**:
- [ ] Prompts list displays
- [ ] Prompt editor loads prompt content
- [ ] Save prompt creates new prompt
- [ ] Search prompts filters list
- [ ] Integration with conversation system works

---

## Summary

All 5 remaining features now have complete implementation guides with:
- Database schemas
- Backend methods
- WebSocket actions
- Frontend UI (HTML)
- JavaScript functions
- CSS styling
- Testing checklists

**Total Implementation Time**: 6-8 hours
**Complexity**: 2 Low, 2 Medium, 1 High

**Recommended Order**:
1. Hover Previews (Quick, 20 min)
2. Lazy Loading (Medium, 60 min)
3. Conversation Templates (Medium, 60 min)
4. Analytics Dashboard (Complex, 120 min)
5. Promptly Integration (Complex, 120 min)

---

*Last Updated: 2025-11-02*
*Status: Implementation Guide Complete*
*Ready for development in next session*
