# Complete Dashboard Roadmap - All Phases

**Date**: 2025-11-02
**Current Status**: Phase 3 Complete (Project Management)
**Total Estimated Time**: ~6-8 hours for all remaining features

---

## ✅ Completed: Phase 3 - Project Management & Backend

- [x] Database schema extension (project_id, is_favorite, tags)
- [x] 10 backend CRUD methods
- [x] 9 WebSocket action handlers
- [x] Complete UI with sidebar, search, favorites
- [x] Real-time updates via WebSocket
- [x] SQLite persistence

**Files Created/Modified**:
- [HoloLoom/web_dashboard/agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html) - 800 lines
- [HoloLoom/web_dashboard/conversation_manager.py](HoloLoom/web_dashboard/conversation_manager.py) - +250 lines
- [HoloLoom/web_dashboard/agentic_server.py](HoloLoom/web_dashboard/agentic_server.py) - +200 lines

---

## 🎯 Remaining Features (User-Requested)

### Phase 4A: Loading Spinners & UI Feedback (30-45 min)

**Add loading states for async operations**:

1. **Spinner Component** (HTML/CSS)
   - CSS3 spinning animation
   - 3 sizes: small (16px), medium (32px), large (48px)
   - Colors: primary (blue), success (green), warning (amber)

2. **Loading States**:
   - Message sending (inline spinner in input)
   - Conversation loading (skeleton UI)
   - Project operations (button spinner)
   - Search results (spinner in search box)

3. **Implementation**:
   ```javascript
   function showSpinner(location) {
       // Show spinner at location
   }

   function hideSpinner(location) {
       // Hide spinner
   }
   ```

**Files to Modify**:
- [agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html) - Add spinner CSS + JS

---

### Phase 4B: LLM Provider Dropdown (30-45 min)

**Add dropdown to select LLM provider**:

1. **Supported Providers**:
   - OpenAI (GPT-4, GPT-3.5)
   - Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
   - Local (Ollama - Llama 3, Mistral, etc.)

2. **UI Design**:
   ```html
   <select id="llmProvider">
       <option value="openai-gpt4">OpenAI GPT-4</option>
       <option value="anthropic-claude">Anthropic Claude 3.5</option>
       <option value="ollama-llama3">Ollama Llama 3</option>
   </select>
   ```

3. **Backend Support**:
   - Add `llm_provider` parameter to conversation
   - Update WebSocket action to include provider
   - Add provider routing in server

**Files to Modify**:
- [agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html) - Add dropdown UI
- [agentic_server.py](HoloLoom/web_dashboard/agentic_server.py) - Add provider routing
- [conversation_manager.py](HoloLoom/web_dashboard/conversation_manager.py) - Store provider preference

---

### Phase 4C: Bulk Operations (45-60 min)

**Select multiple conversations and perform bulk actions**:

1. **Selection UI**:
   - Checkbox on each conversation
   - "Select All" checkbox
   - Selection count badge
   - Bulk action toolbar (appears when > 0 selected)

2. **Bulk Actions**:
   - Delete multiple
   - Move to project
   - Add/remove tags
   - Export selected
   - Mark as favorite/unfavorite

3. **Implementation**:
   ```javascript
   let selectedConversations = new Set();

   function toggleSelection(conversationId) {
       // Add/remove from set
       updateBulkToolbar();
   }

   function bulkDelete() {
       // Delete all selected
   }
   ```

**Files to Modify**:
- [agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html) - Add selection UI + bulk toolbar
- [agentic_server.py](HoloLoom/web_dashboard/agentic_server.py) - Add bulk action handlers

---

### Phase 4D: Conversation Metadata (30 min)

**Show last message & timestamp for each conversation**:

1. **Metadata Display**:
   ```html
   <div class="conversation-item">
       <div class="conversation-title">Discussion about AI</div>
       <div class="conversation-meta">
           <span class="last-message">Sure, let me explain...</span>
           <span class="timestamp">2 hours ago</span>
       </div>
   </div>
   ```

2. **Data Structure**:
   - Modify `list_conversations` to include last message
   - Format timestamps as relative (2 hours ago, yesterday, etc.)

**Files to Modify**:
- [conversation_manager.py](HoloLoom/web_dashboard/conversation_manager.py) - Include last message in query
- [agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html) - Display metadata

---

### Phase 4E: Hover Previews (45 min)

**Show conversation preview on hover**:

1. **Preview Tooltip**:
   - Shows first 3 messages
   - Positioned above/below conversation item
   - Fade-in animation (200ms)
   - Delayed appearance (500ms hover)

2. **Implementation**:
   ```javascript
   function showPreview(conversationId) {
       // Fetch messages
       // Show tooltip with preview
   }
   ```

**Files to Modify**:
- [agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html) - Add preview tooltip + JS

---

### Phase 4F: Keyboard Shortcuts (60 min)

**Add global keyboard shortcuts**:

1. **Shortcuts**:
   - `Ctrl+N` / `Cmd+N`: New conversation
   - `Ctrl+K` / `Cmd+K`: Focus search
   - `Ctrl+/` / `Cmd+/`: Show shortcuts help
   - `Escape`: Clear search, close modals
   - `↑/↓`: Navigate conversations
   - `Enter`: Open selected conversation
   - `Delete`: Delete selected conversation (with confirmation)

2. **Implementation**:
   ```javascript
   document.addEventListener('keydown', (e) => {
       if (e.ctrlKey && e.key === 'n') {
           newConversation();
       }
       // ... other shortcuts
   });
   ```

3. **Help Modal**:
   - Shows all shortcuts
   - Accessible via `Ctrl+/`

**Files to Modify**:
- [agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html) - Add keyboard handler + help modal

---

### Phase 4G: Context Menus (60 min)

**Right-click options for conversations**:

1. **Context Menu Items**:
   - Rename
   - Delete
   - Move to Project (submenu)
   - Toggle Favorite
   - Add Tags
   - Export
   - Duplicate

2. **Implementation**:
   ```javascript
   conversationItem.addEventListener('contextmenu', (e) => {
       e.preventDefault();
       showContextMenu(e.clientX, e.clientY, conversationId);
   });
   ```

**Files to Modify**:
- [agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html) - Add context menu UI + handler

---

### Phase 4H: Drag-and-Drop (90 min)

**Drag conversations to move between projects**:

1. **Drag Source**:
   - Make conversation items draggable
   - Show ghost image during drag
   - Highlight valid drop targets

2. **Drop Targets**:
   - Project headers
   - Uncategorized section
   - Visual feedback on hover

3. **Implementation**:
   ```javascript
   conversationItem.draggable = true;

   conversationItem.ondragstart = (e) => {
       e.dataTransfer.setData('conversation_id', conversationId);
   };

   projectHeader.ondrop = (e) => {
       const convId = e.dataTransfer.getData('conversation_id');
       moveToProject(convId, projectId);
   };
   ```

**Files to Modify**:
- [agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html) - Add drag-drop handlers

---

### Phase 4I: Toast Notifications (45 min)

**Show feedback for all actions**:

1. **Toast Types**:
   - Success (green): "Conversation deleted"
   - Error (red): "Failed to delete conversation"
   - Info (blue): "Searching..."
   - Warning (amber): "Project already exists"

2. **Toast System**:
   ```javascript
   function showToast(message, type = 'info', duration = 3000) {
       const toast = createToast(message, type);
       toastContainer.appendChild(toast);
       setTimeout(() => toast.remove(), duration);
   }
   ```

3. **Positioning**:
   - Top-right corner
   - Stack multiple toasts
   - Fade in/out animations

**Files to Modify**:
- [agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html) - Add toast system

---

### Phase 5A: Export/Import (90 min)

**Export and import conversations as JSON**:

1. **Export**:
   - Single conversation
   - All conversations
   - Selected conversations (bulk)
   - Format: JSON with metadata

2. **Import**:
   - Upload JSON file
   - Validate format
   - Merge or replace option
   - Progress bar for large imports

3. **JSON Format**:
   ```json
   {
       "version": "1.0",
       "exported_at": "2025-11-02T...",
       "conversations": [
           {
               "id": 1,
               "title": "Discussion",
               "messages": [...],
               "metadata": {...}
           }
       ]
   }
   ```

**Files to Modify**:
- [agentic_server.py](HoloLoom/web_dashboard/agentic_server.py) - Add export/import endpoints
- [agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html) - Add export/import UI

---

### Phase 5B: Analytics Dashboard (120 min)

**Charts and insights for usage**:

1. **Metrics to Track**:
   - Messages per day (line chart)
   - Confidence distribution (histogram)
   - Most used reasoning modes (pie chart)
   - Average response time (line chart)
   - Token usage over time (area chart)

2. **Visualizations**:
   - Use Chart.js or D3.js (lightweight)
   - Interactive tooltips
   - Date range selector
   - Export charts as images

3. **Dashboard Layout**:
   ```
   ┌─────────────────────────────────────┐
   │  Analytics Dashboard                │
   ├──────────────┬──────────────────────┤
   │ Usage Chart  │  Confidence Dist.    │
   ├──────────────┼──────────────────────┤
   │ Mode Breakdown│  Response Times     │
   └──────────────┴──────────────────────┘
   ```

**Files to Modify**:
- [agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html) - Add analytics tab
- [agentic_server.py](HoloLoom/web_dashboard/agentic_server.py) - Add analytics endpoints
- [conversation_manager.py](HoloLoom/web_dashboard/conversation_manager.py) - Add analytics queries

---

### Phase 5C: Conversation Templates (60 min)

**Save and load conversation templates**:

1. **Template System**:
   - Save conversation as template
   - Name and describe template
   - Load template to start new conversation
   - Share templates (export/import)

2. **Template Gallery**:
   - Browse saved templates
   - Search templates
   - Preview template content
   - Use template button

3. **Default Templates**:
   - Code Review
   - Technical Discussion
   - Brainstorming
   - Research

**Files to Modify**:
- [conversation_manager.py](HoloLoom/web_dashboard/conversation_manager.py) - Add templates table
- [agentic_server.py](HoloLoom/web_dashboard/agentic_server.py) - Add template endpoints
- [agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html) - Add template UI

---

### Phase 6A: Lazy Loading (60 min)

**Optimize large conversation lists**:

1. **Virtual Scrolling**:
   - Only render visible conversations
   - Dynamically load more on scroll
   - Smooth scroll performance

2. **Pagination**:
   - Load 50 conversations at a time
   - "Load More" button
   - Infinite scroll option

3. **Implementation**:
   ```javascript
   const observer = new IntersectionObserver((entries) => {
       if (entries[0].isIntersecting) {
           loadMoreConversations();
       }
   });
   ```

**Files to Modify**:
- [agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html) - Add lazy loading logic
- [agentic_server.py](HoloLoom/web_dashboard/agentic_server.py) - Add pagination support

---

### Phase 6B: Debounced Search (15 min)

**Optimize search performance**:

1. **Debounce Function**:
   ```javascript
   let searchTimeout;
   function debouncedSearch(query) {
       clearTimeout(searchTimeout);
       searchTimeout = setTimeout(() => {
           performSearch(query);
       }, 300); // 300ms delay
   }
   ```

2. **Search Optimization**:
   - Cancel previous requests
   - Show "Searching..." indicator
   - Clear results on empty query

**Files to Modify**:
- [agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html) - Add debounce to search

---

### Phase 6C: Promptly Integration (120 min)

**Integrate Promptly prompt management framework**:

1. **What is Promptly?**:
   - Prompt versioning and management
   - Template system
   - A/B testing
   - Analytics

2. **Integration Points**:
   - Use Promptly templates for system prompts
   - Track prompt performance
   - Version control prompts
   - Share prompts across team

3. **UI Integration**:
   - "Prompt Library" tab
   - Browse/search prompts
   - Edit prompts inline
   - Compare prompt versions

**Files to Modify**:
- [agentic_server.py](HoloLoom/web_dashboard/agentic_server.py) - Import Promptly
- [agentic_dashboard.html](HoloLoom/web_dashboard/agentic_dashboard.html) - Add Promptly UI
- Create new: `HoloLoom/web_dashboard/promptly_integration.py`

---

## 📋 Implementation Checklist

### Quick Wins (Can do in parallel)
- [ ] Loading spinners (30 min)
- [ ] Debounced search (15 min)
- [ ] Conversation metadata (30 min)
- [ ] Toast notifications (45 min)

### Medium Features (Sequential)
- [ ] LLM provider dropdown (45 min)
- [ ] Keyboard shortcuts (60 min)
- [ ] Hover previews (45 min)
- [ ] Context menus (60 min)

### Complex Features (Sequential)
- [ ] Bulk operations (60 min)
- [ ] Drag-and-drop (90 min)
- [ ] Export/import (90 min)
- [ ] Conversation templates (60 min)
- [ ] Analytics dashboard (120 min)
- [ ] Lazy loading (60 min)
- [ ] Promptly integration (120 min)

---

## 🎯 Recommended Implementation Order

### Session 1: UI Polish & Feedback (3-4 hours)
1. Loading spinners
2. Toast notifications
3. Debounced search
4. Conversation metadata
5. Keyboard shortcuts
6. Hover previews

### Session 2: Advanced UX (3-4 hours)
7. Context menus
8. Bulk operations
9. Drag-and-drop
10. LLM provider dropdown

### Session 3: Power Features (3-4 hours)
11. Export/import
12. Conversation templates
13. Lazy loading
14. Analytics dashboard

### Session 4: Integration (2-3 hours)
15. Promptly integration
16. Final testing
17. Documentation

---

## 🚀 Current Priority

Based on user request, the priority order is:

1. **Loading spinners** - Critical for UX feedback
2. **LLM provider dropdown** - Core functionality
3. **Bulk operations** - Productivity boost
4. **Keyboard shortcuts** - Power user feature
5. **All remaining features** - As time allows

---

## 📝 Notes

- All features should maintain backward compatibility
- Database migrations should be safe (try/except pattern)
- WebSocket updates should be efficient (batch updates)
- UI should remain responsive with 1000+ conversations
- Mobile support can be added later (Phase 7)

---

*Last Updated: 2025-11-02*
*Total Features: 15*
*Estimated Total Time: ~20-25 hours*
*Priority Features: 1-5 (8-10 hours)*
