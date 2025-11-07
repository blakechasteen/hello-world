# Multi-Threaded Chat Integration - 8002 UI

**Status**: 🚧 In Progress
**Target**: Integrate multi-threaded conversation system into 8002 agentic dashboard UI

## Current State Analysis

### Backend (agentic_api_integrated.py - Port 8000/8002)

**Current Architecture**:
```
FastAPI Server
├─ REST Endpoints (/query, /stats, /audit-trail)
├─ WebSocket Endpoint (/ws)
│   └─ Single conversation stream
└─ Single Orchestrator Instance
    └─ Agentic reasoning (4 modes)
```

**Limitations**:
- ❌ Single WebSocket connection per session
- ❌ No persistent conversation threads
- ❌ No breakthrough sharing across conversations
- ❌ No agent selection (uses single orchestrator)
- ❌ No conversation history per thread

### Frontend (persistent_chat.html)

**Current UI**:
```
Single Chat Window
├─ Header (title + clear button)
├─ Messages Area (scrollable)
├─ Input Area (textarea + send button)
└─ Status Bar (connection status)
```

**Limitations**:
- ❌ Single conversation view
- ❌ No thread switching
- ❌ No agent selection
- ❌ No breakthrough notifications from other threads
- ❌ No multi-user awareness

## Target Architecture

### Multi-Threaded Backend

```
┌──────────────────────────────────────────────────────────────────┐
│ FastAPI Server (Port 8002)                                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  REST Endpoints                                                   │
│  ├─ POST /threads/create                                         │
│  ├─ GET /threads/list                                            │
│  ├─ DELETE /threads/{thread_id}                                  │
│  ├─ GET /agents/list                                             │
│  └─ GET /stats                                                    │
│                                                                   │
│  WebSocket Endpoints                                              │
│  ├─ /ws/thread/{thread_id}  ← Per-thread WebSocket             │
│  └─ /ws/notifications       ← Global notification stream        │
│                                                                   │
│  ConversationThreadManager (NEW)                                 │
│  ├─ Thread Pool                                                   │
│  │   ├─ Thread 1 (User A, Budget Agent)                         │
│  │   ├─ Thread 2 (User A, Research Agent)                       │
│  │   └─ Thread 3 (User B, Budget Agent) ← Same agent!          │
│  │                                                                │
│  ├─ Breakthrough Broadcasting                                     │
│  │   └─ Discovery in Thread 1 → Notify all threads              │
│  │                                                                │
│  └─ AgentOrchestrationSystem                                     │
│      └─ Persistent Agent Pool                                     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Multi-Threaded Frontend

```
┌──────────────────────────────────────────────────────────────────┐
│ Header                                                            │
│ ├─ HoloLoom Logo                                                 │
│ ├─ User ID                                                       │
│ ├─ + New Thread Button                                           │
│ └─ Settings                                                       │
├──────────────────────────────────────────────────────────────────┤
│ Thread Tabs (Horizontal Scroll)                                  │
│ ┌────────┐ ┌────────┐ ┌────────┐                               │
│ │Budget  │ │Research│ │Arch    │ + New                         │
│ │ • 5 ↑  │ │   12   │ │  🔔   │                                │
│ └────────┘ └────────┘ └────────┘                               │
│   Active     Unread   Breakthrough                               │
├──────────────────────────────────────────────────────────────────┤
│ Active Thread View                                                │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ Messages Area (scrollable)                                    ││
│ │                                                               ││
│ │ [User] What is Q4 revenue?                                   ││
│ │                                                               ││
│ │ [Budget Agent] Q4 revenue is $2.5M...                        ││
│ │   Confidence: 92% | Mode: verify                             ││
│ │                                                               ││
│ │ 💡 Breakthrough in Research thread:                          ││
│ │    "New pattern discovered in revenue analysis"              ││
│ │                                                               ││
│ └──────────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────────┤
│ Input Area                                                        │
│ ├─ Agent Selector [Budget ▼]                                    │
│ ├─ Mode Selector [Verify ▼]                                     │
│ ├─ Message Input (textarea)                                      │
│ └─ Send Button                                                    │
├──────────────────────────────────────────────────────────────────┤
│ Status Bar                                                        │
│ └─ Connected to Budget Agent | 3 active threads                 │
└──────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Backend Integration (2-3 hours)

#### Step 1.1: Integrate ConversationThreadManager

Create new `agentic_api_multithreaded.py`:

```python
from HoloLoom.web_dashboard.conversation_thread_manager import (
    create_conversation_thread_manager,
    ConversationThread
)
from HoloLoom.web_dashboard.adversarial_orchestration import (
    create_adversarial_orchestration_system
)

# Global state
thread_manager = None
orchestration_system = None

async def init_threading_system():
    """Initialize multi-threaded conversation system"""
    global thread_manager, orchestration_system

    kg = KG()
    emb = MatryoshkaEmbeddings(model_name='all-MiniLM-L6-v2', scales=[96, 192, 384])

    # Create orchestration with adversarial negotiation
    orchestration_system = await create_adversarial_orchestration_system(
        kg, emb,
        default_creativity=0.8,
        default_strictness=0.8
    )

    # Create thread manager
    thread_manager = await create_conversation_thread_manager(
        kg, emb,
        agent_pool=None,  # Uses orchestration_system
        enable_breakthrough_sharing=True
    )
```

#### Step 1.2: Add Thread Management Endpoints

```python
# Create thread
@app.post("/threads/create")
async def create_thread(
    user_id: str,
    agent_name: str,
    initial_message: Optional[str] = None
):
    """Create new conversation thread"""
    thread = await thread_manager.create_thread(
        user_id=user_id,
        agent_name=agent_name,
        websocket=None  # WebSocket connected later
    )

    return {
        "thread_id": thread.thread_id,
        "agent_name": agent_name,
        "created_at": thread.created_at
    }

# List threads
@app.get("/threads/list")
async def list_threads(user_id: Optional[str] = None):
    """List all threads (optionally filtered by user)"""
    threads = thread_manager.get_threads(user_id=user_id)

    return {
        "threads": [
            {
                "thread_id": t.thread_id,
                "user_id": t.user_id,
                "agent_name": t.agent_name,
                "message_count": t.message_count,
                "created_at": t.created_at,
                "last_activity": t.last_activity
            }
            for t in threads
        ]
    }

# Delete thread
@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete conversation thread"""
    await thread_manager.close_thread(thread_id)
    return {"status": "deleted"}

# List available agents
@app.get("/agents/list")
async def list_agents():
    """List available agents"""
    agents = orchestration_system.get_active_agents()

    return {
        "agents": [
            {
                "name": a.agent_name,
                "active_conversations": len(a.active_conversations),
                "total_queries": a.total_queries_processed,
                "success_rate": a.success_rate,
                "breakthroughs": a.total_breakthroughs
            }
            for a in agents
        ]
    }
```

#### Step 1.3: Update WebSocket Protocol

```python
# Per-thread WebSocket
@app.websocket("/ws/thread/{thread_id}")
async def thread_websocket(websocket: WebSocket, thread_id: str):
    """WebSocket for specific conversation thread"""
    await websocket.accept()

    # Attach WebSocket to thread
    thread = thread_manager.get_thread(thread_id)
    if not thread:
        await websocket.close(code=4004, reason="Thread not found")
        return

    thread.websocket = websocket

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message_text = data.get("message", "")
            mode = data.get("mode", "verify")

            # Send "thinking" status
            await websocket.send_json({
                "type": "status",
                "status": "thinking",
                "message": f"{thread.agent_name} is thinking..."
            })

            # Process query through thread
            result = await thread_manager.query_thread(
                thread_id=thread_id,
                query=Query(text=message_text),
                mode=mode
            )

            # Send result
            await websocket.send_json({
                "type": "message",
                "role": "assistant",
                "content": result.spacetime.metadata.get("response", ""),
                "confidence": result.spacetime.confidence,
                "mode": result.reasoning_mode.value,
                "thread_id": thread_id
            })

    except WebSocketDisconnect:
        thread.websocket = None
        logger.info(f"WebSocket disconnected from thread {thread_id}")

# Global notification stream (breakthroughs, etc.)
@app.websocket("/ws/notifications")
async def notifications_websocket(websocket: WebSocket):
    """WebSocket for global notifications (breakthroughs, etc.)"""
    await websocket.accept()

    # Subscribe to breakthrough notifications
    async def on_breakthrough(breakthrough):
        await websocket.send_json({
            "type": "breakthrough",
            "thread_id": breakthrough.source_thread_id,
            "agent_name": breakthrough.agent_name,
            "message": breakthrough.description,
            "impact_score": breakthrough.impact_score
        })

    thread_manager.subscribe_to_breakthroughs(on_breakthrough)

    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        thread_manager.unsubscribe_from_breakthroughs(on_breakthrough)
```

### Phase 2: Frontend Implementation (3-4 hours)

#### Step 2.1: Create Multi-Threaded UI

Create `ui/multithreaded_chat.html`:

**Key Features**:
- Tab-based thread switching
- Visual indicators (unread count, breakthrough notifications)
- Agent selector per thread
- Global notification sidebar
- Breakthrough toast notifications

**Component Structure**:
```html
<div class="app-container">
    <header class="header">
        <h1>HoloLoom Multi-Agent Chat</h1>
        <div class="user-info">User: <span id="userId">user_123</span></div>
        <button id="newThreadBtn">+ New Thread</button>
    </header>

    <div class="thread-tabs" id="threadTabs">
        <!-- Thread tabs dynamically added -->
    </div>

    <div class="main-content">
        <div class="active-thread-view" id="activeThreadView">
            <div class="messages" id="messages"></div>

            <div class="input-area">
                <select id="agentSelector">
                    <option value="budget">Budget Agent</option>
                    <option value="research">Research Agent</option>
                    <option value="architecture">Architecture Agent</option>
                </select>

                <select id="modeSelector">
                    <option value="direct">Direct</option>
                    <option value="verify">Verify</option>
                    <option value="research">Research</option>
                    <option value="plan_execute">Plan & Execute</option>
                </select>

                <textarea id="messageInput" placeholder="Type your message..."></textarea>
                <button id="sendBtn">Send</button>
            </div>
        </div>

        <div class="notifications-sidebar" id="notificationsSidebar">
            <h3>Recent Breakthroughs</h3>
            <div id="breakthroughsList"></div>
        </div>
    </div>

    <div class="status-bar" id="statusBar">
        Connected to Budget Agent | 3 active threads
    </div>
</div>
```

#### Step 2.2: JavaScript Thread Manager

```javascript
class ThreadManager {
    constructor() {
        this.threads = new Map();  // thread_id → thread object
        this.activeThreadId = null;
        this.userId = 'user_' + Math.random().toString(36).substr(2, 9);
        this.notificationWs = null;
    }

    async init() {
        // Connect to notification stream
        this.notificationWs = new WebSocket('ws://localhost:8002/ws/notifications');

        this.notificationWs.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'breakthrough') {
                this.handleBreakthrough(data);
            }
        };

        // Load existing threads
        await this.loadThreads();

        // If no threads, create default
        if (this.threads.size === 0) {
            await this.createThread('budget');
        }
    }

    async createThread(agentName, initialMessage = null) {
        // Create thread via API
        const response = await fetch('http://localhost:8002/threads/create', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                user_id: this.userId,
                agent_name: agentName,
                initial_message: initialMessage
            })
        });

        const data = await response.json();

        // Create thread object
        const thread = {
            id: data.thread_id,
            agentName: agentName,
            messages: [],
            ws: null,
            unreadCount: 0,
            hasBreakthrough: false
        };

        // Connect WebSocket
        thread.ws = new WebSocket(`ws://localhost:8002/ws/thread/${thread.id}`);

        thread.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            this.handleThreadMessage(thread.id, msg);
        };

        this.threads.set(thread.id, thread);
        this.renderThreadTab(thread);

        // Switch to new thread
        this.switchToThread(thread.id);

        return thread;
    }

    switchToThread(threadId) {
        this.activeThreadId = threadId;

        // Update UI
        this.renderActiveThread();
        this.updateThreadTabs();

        // Mark as read
        const thread = this.threads.get(threadId);
        thread.unreadCount = 0;
        thread.hasBreakthrough = false;
    }

    async sendMessage(message, mode = 'verify') {
        const thread = this.threads.get(this.activeThreadId);

        // Add user message to UI
        this.addMessage(thread.id, 'user', message);

        // Send via WebSocket
        thread.ws.send(JSON.stringify({
            message: message,
            mode: mode
        }));
    }

    handleThreadMessage(threadId, data) {
        const thread = this.threads.get(threadId);

        if (data.type === 'status') {
            this.showStatus(data.message);
        }
        else if (data.type === 'message') {
            this.addMessage(threadId, 'assistant', data.content, {
                confidence: data.confidence,
                mode: data.mode
            });

            // Increment unread if not active thread
            if (threadId !== this.activeThreadId) {
                thread.unreadCount++;
                this.updateThreadTabs();
            }
        }
    }

    handleBreakthrough(data) {
        // Show toast notification
        this.showBreakthroughToast(data);

        // Add to notifications sidebar
        this.addBreakthroughNotification(data);

        // Mark thread with breakthrough indicator
        const thread = this.threads.get(data.thread_id);
        if (thread && thread.id !== this.activeThreadId) {
            thread.hasBreakthrough = true;
            this.updateThreadTabs();
        }
    }

    renderThreadTab(thread) {
        const tabsContainer = document.getElementById('threadTabs');

        const tab = document.createElement('div');
        tab.className = 'thread-tab';
        tab.id = `tab-${thread.id}`;
        tab.innerHTML = `
            <div class="tab-name">${thread.agentName}</div>
            <div class="tab-indicators">
                <span class="unread-count" style="display: none;">0</span>
                <span class="breakthrough-icon" style="display: none;">🔔</span>
            </div>
            <button class="tab-close">×</button>
        `;

        tab.onclick = () => this.switchToThread(thread.id);
        tab.querySelector('.tab-close').onclick = (e) => {
            e.stopPropagation();
            this.closeThread(thread.id);
        };

        tabsContainer.appendChild(tab);
    }

    updateThreadTabs() {
        this.threads.forEach((thread, threadId) => {
            const tab = document.getElementById(`tab-${threadId}`);

            // Update active state
            tab.classList.toggle('active', threadId === this.activeThreadId);

            // Update unread count
            const unreadEl = tab.querySelector('.unread-count');
            if (thread.unreadCount > 0) {
                unreadEl.textContent = thread.unreadCount;
                unreadEl.style.display = 'block';
            } else {
                unreadEl.style.display = 'none';
            }

            // Update breakthrough indicator
            const breakthroughEl = tab.querySelector('.breakthrough-icon');
            breakthroughEl.style.display = thread.hasBreakthrough ? 'block' : 'none';
        });
    }
}

// Initialize
const threadManager = new ThreadManager();
threadManager.init();
```

### Phase 3: Migration Guide

#### Step 3.1: Run New Server

```bash
# Terminal 1: Start multi-threaded server
cd c:/Users/blake/OneDrive/Documents/mythRL
PYTHONPATH=. python HoloLoom/server/agentic_api_multithreaded.py

# Server starts on http://localhost:8002
```

#### Step 3.2: Test in Browser

```
1. Open http://localhost:8002/multithreaded_chat.html
2. Create first thread (Budget agent)
3. Send message: "What is Q4 revenue?"
4. Click "+ New Thread"
5. Create second thread (Research agent)
6. Send message: "Find breakthrough patterns"
7. If breakthrough detected in Research → See notification in Budget thread
```

### Phase 4: Features Checklist

**Backend ✅**:
- [ ] ConversationThreadManager integration
- [ ] Thread CRUD endpoints (/create, /list, /delete)
- [ ] Per-thread WebSocket (/ws/thread/{id})
- [ ] Global notification WebSocket (/ws/notifications)
- [ ] Breakthrough broadcasting
- [ ] Agent pool management

**Frontend ✅**:
- [ ] Tab-based thread UI
- [ ] Thread creation/deletion
- [ ] Agent selector per thread
- [ ] Mode selector per thread
- [ ] Unread message indicators
- [ ] Breakthrough notifications (toast + sidebar)
- [ ] Thread switching
- [ ] Message persistence per thread

**Advanced Features ⏭️**:
- [ ] Thread search/filter
- [ ] Thread history export
- [ ] Multi-user collaboration (see other users' agents)
- [ ] Agent performance metrics per thread
- [ ] Negotiation statistics display
- [ ] Thread bookmarking/favorites

## Benefits

### For Users
✅ **Multiple Simultaneous Conversations**: Budget + Research + Architecture agents in parallel
✅ **Cross-Thread Learning**: Breakthrough in one thread benefits all
✅ **Agent Specialization**: Right agent for each task
✅ **Context Preservation**: Each thread maintains its own history

### For System
✅ **Persistent Agents**: Agents reused across threads (efficient)
✅ **Breakthrough Sharing**: Discovery in Thread 1 → All threads accelerated
✅ **Adversarial Balance**: Creative vs QC negotiation per thread
✅ **Complete Monitoring**: Per-thread and global statistics

## Timeline Estimate

**Phase 1: Backend** (2-3 hours)
- [ ] Integrate ConversationThreadManager (1 hour)
- [ ] Add thread endpoints (30 minutes)
- [ ] Update WebSocket protocol (1-1.5 hours)

**Phase 2: Frontend** (3-4 hours)
- [ ] Multi-threaded UI layout (1 hour)
- [ ] JavaScript ThreadManager class (1.5 hours)
- [ ] WebSocket integration (1 hour)
- [ ] Styling and polish (30 minutes)

**Phase 3: Testing** (1-2 hours)
- [ ] Thread creation/deletion
- [ ] Cross-thread messaging
- [ ] Breakthrough notifications
- [ ] Multi-agent switching

**Total**: 6-9 hours for complete integration

## Next Steps

1. **Immediate**: Create `agentic_api_multithreaded.py` with ConversationThreadManager
2. **Then**: Build `multithreaded_chat.html` UI with tab interface
3. **Test**: Run full integration test with 3+ threads
4. **Deploy**: Replace single-chat UI with multi-threaded version

Ready to start implementation?
