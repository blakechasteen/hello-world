# Multi-Threaded Chat - Advanced Enhancements

**Status**: 🚧 In Progress
**Target**: Transform basic multi-threaded chat into production-grade system

## Enhancement Overview

### Core Enhancements (Phase 1)
1. **Thread Search/Filter** - Find conversations by content, agent, date
2. **Export History** - JSON, Markdown, PDF formats
3. **Thread Bookmarking** - Star/favorite important threads
4. **Agent Performance** - Real-time charts and analytics

### UI/UX Enhancements (Phase 2)
5. **Mobile Responsive** - Touch gestures, adaptive layout
6. **Voice I/O** - Web Speech API integration

### Advanced Integration (Phase 3)
7. **Promptly Integration** - Complex prompt chaining, templates, workflows

---

## 1. Thread Search/Filter

### Backend API

**New Endpoints**:
```python
GET /threads/search
    ?q=revenue                 # Search message content
    &agent=budget              # Filter by agent
    &from=2025-11-01           # Date range start
    &to=2025-11-30             # Date range end
    &bookmarked=true           # Only bookmarked
    &min_confidence=0.8        # Minimum confidence
    &has_breakthroughs=true    # Only threads with breakthroughs
```

**Implementation**:
```python
@app.get("/threads/search")
async def search_threads(
    q: Optional[str] = None,
    agent: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    bookmarked: Optional[bool] = None,
    min_confidence: Optional[float] = None,
    has_breakthroughs: Optional[bool] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search threads with multiple filters.

    Returns:
        {
            "threads": [...],
            "total_count": 42,
            "filters_applied": {...}
        }
    """
    results = []

    for thread_id, thread in state.thread_manager.threads.items():
        # User filter
        if user_id and thread.user_id != user_id:
            continue

        # Agent filter
        if agent and thread.agent_name != agent:
            continue

        # Bookmarked filter
        if bookmarked and not thread.bookmarked:
            continue

        # Date range filter
        if from_date:
            if thread.created_at < parse_date(from_date):
                continue

        if to_date:
            if thread.created_at > parse_date(to_date):
                continue

        # Content search (search in messages)
        if q:
            if not any(q.lower() in msg.content.lower() for msg in thread.messages):
                continue

        # Confidence filter
        if min_confidence:
            avg_confidence = sum(
                msg.confidence for msg in thread.messages
                if hasattr(msg, 'confidence') and msg.confidence
            ) / len(thread.messages)

            if avg_confidence < min_confidence:
                continue

        # Breakthroughs filter
        if has_breakthroughs:
            if thread.breakthroughs_received == 0:
                continue

        results.append(thread)

    return {
        "threads": [serialize_thread(t) for t in results],
        "total_count": len(results),
        "filters_applied": {
            "q": q,
            "agent": agent,
            "from_date": from_date,
            "to_date": to_date,
            "bookmarked": bookmarked,
            "min_confidence": min_confidence,
            "has_breakthroughs": has_breakthroughs
        }
    }
```

### Frontend UI

**Search Bar**:
```html
<div class="search-bar">
    <input type="text" id="searchInput" placeholder="Search threads...">
    <button onclick="toggleAdvancedFilters()">🔍 Filters</button>
</div>

<div class="advanced-filters hidden" id="advancedFilters">
    <select id="filterAgent">
        <option value="">All Agents</option>
        <option value="budget">Budget</option>
        <option value="research">Research</option>
        <option value="architecture">Architecture</option>
    </select>

    <input type="date" id="filterFromDate" placeholder="From">
    <input type="date" id="filterToDate" placeholder="To">

    <label>
        <input type="checkbox" id="filterBookmarked"> Bookmarked only
    </label>

    <label>
        <input type="checkbox" id="filterBreakthroughs"> Has breakthroughs
    </label>

    <input type="range" id="filterConfidence" min="0" max="100" value="0">
    <span>Min confidence: <span id="confidenceValue">0%</span></span>

    <button onclick="applyFilters()">Apply</button>
    <button onclick="clearFilters()">Clear</button>
</div>
```

**Search Functionality**:
```javascript
async function searchThreads() {
    const query = document.getElementById('searchInput').value;
    const agent = document.getElementById('filterAgent').value;
    const fromDate = document.getElementById('filterFromDate').value;
    const toDate = document.getElementById('filterToDate').value;
    const bookmarked = document.getElementById('filterBookmarked').checked;
    const breakthroughs = document.getElementById('filterBreakthroughs').checked;
    const confidence = document.getElementById('filterConfidence').value / 100;

    const params = new URLSearchParams();
    if (query) params.append('q', query);
    if (agent) params.append('agent', agent);
    if (fromDate) params.append('from_date', fromDate);
    if (toDate) params.append('to_date', toDate);
    if (bookmarked) params.append('bookmarked', 'true');
    if (breakthroughs) params.append('has_breakthroughs', 'true');
    if (confidence > 0) params.append('min_confidence', confidence);

    const response = await fetch(`${API_BASE}/threads/search?${params}`);
    const data = await response.json();

    displaySearchResults(data.threads);
}
```

---

## 2. Export Conversation History

### Backend API

**New Endpoints**:
```python
GET /threads/{thread_id}/export?format=json|markdown|pdf
```

**Implementation**:
```python
from datetime import datetime
import json

@app.get("/threads/{thread_id}/export")
async def export_thread(
    thread_id: str,
    format: str = "json"  # json, markdown, pdf
):
    """Export thread conversation history"""
    thread = state.thread_manager.threads.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    if format == "json":
        return JSONResponse(content={
            "thread_id": thread.thread_id,
            "agent_name": thread.agent_name,
            "user_id": thread.user_id,
            "created_at": thread.created_at,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "confidence": getattr(msg, 'confidence', None),
                    "mode": getattr(msg, 'mode', None),
                    "timestamp": getattr(msg, 'timestamp', None)
                }
                for msg in thread.messages
            ],
            "stats": {
                "message_count": thread.message_count,
                "breakthroughs_received": thread.breakthroughs_received,
                "average_confidence": thread.get_avg_confidence()
            }
        })

    elif format == "markdown":
        md = f"# Conversation: {thread.agent_name}\n\n"
        md += f"**Thread ID**: {thread.thread_id}\n"
        md += f"**Created**: {datetime.fromtimestamp(thread.created_at).isoformat()}\n"
        md += f"**Messages**: {thread.message_count}\n\n"
        md += "---\n\n"

        for msg in thread.messages:
            role = "👤 User" if msg.role == "user" else f"🤖 {thread.agent_name}"
            md += f"## {role}\n\n"
            md += f"{msg.content}\n\n"

            if hasattr(msg, 'confidence'):
                md += f"*Confidence: {msg.confidence * 100:.0f}%*\n\n"

        return Response(content=md, media_type="text/markdown")

    elif format == "pdf":
        # Use reportlab or similar
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from io import BytesIO

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph(f"Conversation: {thread.agent_name}", styles['Title']))
        story.append(Spacer(1, 12))

        # Messages
        for msg in thread.messages:
            role = "User" if msg.role == "user" else thread.agent_name
            story.append(Paragraph(f"<b>{role}:</b>", styles['Heading2']))
            story.append(Paragraph(msg.content, styles['BodyText']))
            story.append(Spacer(1, 12))

        doc.build(story)
        buffer.seek(0)

        return Response(
            content=buffer.getvalue(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=thread_{thread_id}.pdf"
            }
        )
```

### Frontend UI

**Export Button**:
```html
<div class="thread-actions">
    <button onclick="exportThread('json')">📥 Export JSON</button>
    <button onclick="exportThread('markdown')">📄 Export Markdown</button>
    <button onclick="exportThread('pdf')">📕 Export PDF</button>
</div>
```

**Export Functionality**:
```javascript
async function exportThread(format) {
    if (!state.activeThreadId) return;

    const url = `${API_BASE}/threads/${state.activeThreadId}/export?format=${format}`;

    // Download file
    const a = document.createElement('a');
    a.href = url;
    a.download = `thread_${state.activeThreadId}.${format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    showToast('Export Complete', `Thread exported as ${format.toUpperCase()}`);
}
```

---

## 3. Thread Bookmarking

### Backend API

**Update Thread Model**:
```python
@dataclass
class ConversationThread:
    # ... existing fields ...
    bookmarked: bool = False
    bookmark_timestamp: Optional[float] = None
    tags: List[str] = field(default_factory=list)
```

**New Endpoints**:
```python
POST /threads/{thread_id}/bookmark
DELETE /threads/{thread_id}/bookmark
GET /threads/bookmarked
POST /threads/{thread_id}/tags
```

**Implementation**:
```python
@app.post("/threads/{thread_id}/bookmark")
async def bookmark_thread(thread_id: str):
    """Bookmark a thread"""
    thread = state.thread_manager.threads.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    thread.bookmarked = True
    thread.bookmark_timestamp = time.time()

    return {"status": "bookmarked", "thread_id": thread_id}

@app.delete("/threads/{thread_id}/bookmark")
async def unbookmark_thread(thread_id: str):
    """Remove bookmark from thread"""
    thread = state.thread_manager.threads.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    thread.bookmarked = False
    thread.bookmark_timestamp = None

    return {"status": "unbookmarked", "thread_id": thread_id}

@app.get("/threads/bookmarked")
async def get_bookmarked_threads(user_id: Optional[str] = None):
    """Get all bookmarked threads"""
    bookmarked = [
        thread for thread in state.thread_manager.threads.values()
        if thread.bookmarked and (not user_id or thread.user_id == user_id)
    ]

    # Sort by bookmark timestamp (most recent first)
    bookmarked.sort(key=lambda t: t.bookmark_timestamp or 0, reverse=True)

    return {
        "threads": [serialize_thread(t) for t in bookmarked],
        "total_count": len(bookmarked)
    }

@app.post("/threads/{thread_id}/tags")
async def add_tags(thread_id: str, tags: List[str]):
    """Add tags to thread"""
    thread = state.thread_manager.threads.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    thread.tags.extend(tags)
    thread.tags = list(set(thread.tags))  # Remove duplicates

    return {"tags": thread.tags}
```

### Frontend UI

**Bookmark Button on Tab**:
```html
<div class="thread-tab">
    <span class="bookmark-icon" onclick="toggleBookmark(threadId)">
        ☆  <!-- Empty star when not bookmarked -->
        ★  <!-- Filled star when bookmarked -->
    </span>
    <div class="tab-name">Budget</div>
    <button class="tab-close">×</button>
</div>
```

**Bookmarks Sidebar**:
```html
<div class="bookmarks-panel">
    <h3>⭐ Bookmarked Threads</h3>
    <div id="bookmarksList">
        <!-- Dynamically populated -->
    </div>
</div>
```

**Bookmark Functionality**:
```javascript
async function toggleBookmark(threadId) {
    const thread = state.threads.get(threadId);
    const method = thread.bookmarked ? 'DELETE' : 'POST';

    await fetch(`${API_BASE}/threads/${threadId}/bookmark`, { method });

    thread.bookmarked = !thread.bookmarked;
    updateThreadTabs();

    showToast(
        thread.bookmarked ? 'Bookmarked' : 'Unbookmarked',
        `Thread ${thread.agentName}`
    );
}

async function loadBookmarkedThreads() {
    const response = await fetch(`${API_BASE}/threads/bookmarked?user_id=${state.userId}`);
    const data = await response.json();

    displayBookmarks(data.threads);
}
```

---

## 4. Agent Performance Charts

### Backend API

**New Endpoints**:
```python
GET /stats/agents
GET /stats/agents/{agent_name}
GET /stats/threads/{thread_id}/timeline
```

**Implementation**:
```python
@app.get("/stats/agents")
async def get_agents_stats():
    """Get performance stats for all agents"""
    stats = []

    for agent_name, agent in state.orchestration_system.agents.items():
        # Calculate metrics
        total_queries = agent.total_queries_processed
        successes = int(total_queries * agent.success_rate)

        # Get recent performance (last 100 queries)
        recent_queries = agent.recent_queries[-100:]  # Need to track this
        recent_confidence = [q.confidence for q in recent_queries if q.confidence]

        # Negotiation stats
        neg_stats = state.orchestration_system.get_negotiation_stats(agent_name)

        stats.append({
            "agent_name": agent.agent_name,
            "total_queries": total_queries,
            "success_rate": agent.success_rate,
            "average_confidence": sum(recent_confidence) / len(recent_confidence) if recent_confidence else 0,
            "breakthroughs": agent.total_breakthroughs,
            "active_conversations": len(agent.active_conversations),
            "negotiation": {
                "creative_win_rate": neg_stats.get('creative_win_rate', 0),
                "qc_win_rate": neg_stats.get('qc_win_rate', 0),
                "compromise_rate": neg_stats.get('compromise_rate', 0)
            },
            "timeline": {
                "queries_per_hour": calculate_queries_per_hour(agent),
                "breakthroughs_per_100_queries": (agent.total_breakthroughs / max(total_queries, 1)) * 100
            }
        })

    return {"agents": stats}

@app.get("/stats/threads/{thread_id}/timeline")
async def get_thread_timeline(thread_id: str):
    """Get confidence timeline for thread"""
    thread = state.thread_manager.threads.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    timeline = []
    for msg in thread.messages:
        if hasattr(msg, 'confidence') and msg.confidence:
            timeline.append({
                "timestamp": msg.timestamp,
                "confidence": msg.confidence,
                "role": msg.role,
                "cached": getattr(msg, 'cached', False)
            })

    return {"timeline": timeline}
```

### Frontend UI

**Performance Dashboard**:
```html
<div class="performance-dashboard">
    <h2>📊 Agent Performance</h2>

    <div class="chart-grid">
        <!-- Agent comparison chart -->
        <div class="chart-container">
            <canvas id="agentComparisonChart"></canvas>
        </div>

        <!-- Confidence timeline -->
        <div class="chart-container">
            <canvas id="confidenceTimelineChart"></canvas>
        </div>

        <!-- Negotiation breakdown -->
        <div class="chart-container">
            <canvas id="negotiationChart"></canvas>
        </div>

        <!-- Breakthrough rate -->
        <div class="chart-container">
            <canvas id="breakthroughChart"></canvas>
        </div>
    </div>
</div>
```

**Chart Rendering (Chart.js)**:
```javascript
async function renderAgentPerformanceCharts() {
    const response = await fetch(`${API_BASE}/stats/agents`);
    const data = await response.json();

    // Agent Comparison Chart
    const agentChart = new Chart(document.getElementById('agentComparisonChart'), {
        type: 'bar',
        data: {
            labels: data.agents.map(a => a.agent_name),
            datasets: [{
                label: 'Success Rate',
                data: data.agents.map(a => a.success_rate * 100),
                backgroundColor: '#00d4ff'
            }, {
                label: 'Avg Confidence',
                data: data.agents.map(a => a.average_confidence * 100),
                backgroundColor: '#2a5298'
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true, max: 100 }
            }
        }
    });

    // Confidence Timeline Chart
    const timelineResponse = await fetch(`${API_BASE}/stats/threads/${state.activeThreadId}/timeline`);
    const timelineData = await timelineResponse.json();

    const timelineChart = new Chart(document.getElementById('confidenceTimelineChart'), {
        type: 'line',
        data: {
            labels: timelineData.timeline.map((_, i) => i + 1),
            datasets: [{
                label: 'Confidence Over Time',
                data: timelineData.timeline.map(t => t.confidence * 100),
                borderColor: '#00d4ff',
                fill: false
            }]
        }
    });
}
```

---

## 5. Mobile-Responsive UI

### Responsive CSS

```css
/* Mobile breakpoints */
@media (max-width: 768px) {
    /* Stack layout vertically */
    .main-content {
        flex-direction: column;
    }

    /* Full-width chat */
    .chat-area {
        width: 100%;
    }

    /* Hide sidebar by default, show as overlay */
    .notifications-sidebar {
        position: fixed;
        right: -280px;
        top: 0;
        bottom: 0;
        transition: right 0.3s;
        z-index: 100;
    }

    .notifications-sidebar.open {
        right: 0;
    }

    /* Horizontal scrolling thread tabs */
    .thread-tabs-container {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }

    /* Larger touch targets */
    .thread-tab {
        min-width: 120px;
        padding: 12px;
    }

    button {
        min-height: 44px;  /* iOS touch target size */
    }

    /* Bottom input area */
    .input-area {
        position: sticky;
        bottom: 0;
    }
}

/* Touch gestures */
.thread-tab {
    touch-action: pan-y;
}

.messages {
    -webkit-overflow-scrolling: touch;
    overscroll-behavior: contain;
}

/* Swipe indicators */
.swipe-indicator {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    opacity: 0;
    transition: opacity 0.2s;
}

.swipe-indicator.active {
    opacity: 1;
}
```

### Touch Gestures

```javascript
// Swipe to delete thread
let touchStartX = 0;
let touchCurrentX = 0;

function handleTouchStart(e, threadId) {
    touchStartX = e.touches[0].clientX;
}

function handleTouchMove(e, threadId) {
    touchCurrentX = e.touches[0].clientX;
    const diff = touchCurrentX - touchStartX;

    // Show delete indicator if swiped left
    if (diff < -50) {
        showDeleteIndicator(threadId);
    }
}

function handleTouchEnd(e, threadId) {
    const diff = touchCurrentX - touchStartX;

    // Delete if swiped far enough
    if (diff < -100) {
        closeThread(threadId);
    } else {
        hideDeleteIndicator(threadId);
    }
}

// Pull to refresh
let pullDistance = 0;

function handlePullToRefresh(e) {
    const messagesContainer = document.getElementById('messages');

    if (messagesContainer.scrollTop === 0) {
        pullDistance = e.touches[0].clientY - touchStartY;

        if (pullDistance > 100) {
            refreshThread();
        }
    }
}
```

---

## 6. Voice Input/Output

### Web Speech API Integration

```javascript
// Voice input (Speech Recognition API)
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.continuous = true;
recognition.interimResults = true;

function startVoiceInput() {
    recognition.start();

    recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
            .map(result => result[0].transcript)
            .join('');

        document.getElementById('messageInput').value = transcript;
    };

    recognition.onend = () => {
        // Auto-send when user stops speaking
        if (document.getElementById('autoSendVoice').checked) {
            sendMessage();
        }
    };
}

function stopVoiceInput() {
    recognition.stop();
}

// Voice output (Speech Synthesis API)
function speakMessage(text) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    // Use appropriate voice
    const voices = speechSynthesis.getVoices();
    utterance.voice = voices.find(v => v.lang === 'en-US') || voices[0];

    speechSynthesis.speak(utterance);
}

// Auto-speak assistant responses
function handleThreadMessage(threadId, data) {
    // ... existing code ...

    if (data.type === 'message' && data.role === 'assistant') {
        if (document.getElementById('autoSpeakResponses').checked) {
            speakMessage(data.content);
        }
    }
}
```

**Voice UI Controls**:
```html
<div class="voice-controls">
    <button id="voiceInputBtn" onclick="toggleVoiceInput()">
        🎤 Voice Input
    </button>

    <label>
        <input type="checkbox" id="autoSendVoice">
        Auto-send when done speaking
    </label>

    <label>
        <input type="checkbox" id="autoSpeakResponses">
        Speak assistant responses
    </label>
</div>
```

---

## 7. Promptly Integration

### Architecture

```
User Query
    ↓
Promptly Template Selection
    ├─ Simple Query → Direct template
    ├─ Analysis → Multi-step template
    ├─ Research → Chain-of-thought template
    └─ Complex → Custom workflow
    ↓
Template Execution
    ├─ Step 1: Context gathering
    ├─ Step 2: Analysis
    ├─ Step 3: Synthesis
    └─ Step 4: Verification
    ↓
Agent Execution (per step)
    ↓
Result Chaining
    ↓
Final Response
```

### Implementation

**See detailed implementation in next file...**

---

## Implementation Priority

### Phase 1: Core Features (4-6 hours)
1. ✅ Thread search/filter (2 hours)
2. ✅ Export history (1.5 hours)
3. ✅ Thread bookmarking (1 hour)
4. ✅ Performance charts (1.5 hours)

### Phase 2: UX Enhancements (3-4 hours)
5. ✅ Mobile responsive (2 hours)
6. ✅ Voice I/O (2 hours)

### Phase 3: Advanced Integration (4-6 hours)
7. ✅ Promptly integration (4-6 hours)

**Total Estimate**: 11-16 hours for complete enhancement suite

---

## Next Steps

1. Implement backend enhancements (search, export, bookmarks, stats)
2. Update frontend UI with new features
3. Add mobile responsive CSS
4. Integrate voice I/O
5. Build Promptly integration layer
6. Comprehensive testing
7. Documentation updates

Ready to implement?
