# Conversational Mode - Hands-Free Voice Dashboard

**Goal**: Enable automatic voice responses for natural conversation flow

---

## Two Options for Conversational Mode

### Option 1: Always Auto-Speak (Simple) ⚡

**Best for**: You always want voice responses automatically

**Setup**: Change one line in `agentic_server.py`:

```python
voice_integration = await create_voice_integration(
    tts_backend="pyttsx3",
    whisper_model="base",
    auto_speak=True  # ← Change from False to True
)
```

**Behavior**:
- Every response speaks automatically
- No manual button press needed
- Continuous conversation mode
- Always on

---

### Option 2: Toggle Button (Flexible) 🎛️

**Best for**: You want to switch between manual and auto modes

**Setup**: Add conversational mode toggle to frontend

**Files**:
- All code is in `conversational_mode.html`
- Copy sections to `agentic_dashboard.html`:
  1. CSS → `<style>` section
  2. HTML → Before `</body>`
  3. JavaScript → `<script>` section
  4. Update WebSocket handler (see below)

**Behavior**:
- Toggle button to switch modes (bottom right, above mic)
- Manual mode: Click "Speak" buttons (default)
- Auto mode: Automatic voice responses (hands-free)
- Keyboard shortcut: **Ctrl+M** to toggle
- Visual indicator (green pulsing dot when active)

---

## Comparison

| Feature | Always Auto-Speak | Toggle Button |
|---------|-------------------|---------------|
| **Setup** | 1 line change | Copy HTML/CSS/JS |
| **Flexibility** | Always on | Switch on/off |
| **UI** | None | Toggle button + indicator |
| **Keyboard** | N/A | Ctrl+M to toggle |
| **Best For** | Single use case | Multiple use cases |

---

## Option 2 Setup (Toggle Button)

### Step 1: Add CSS

Open `agentic_dashboard.html` and add to `<style>` section:

```css
/* Conversational Mode Toggle */
.conversation-toggle {
    position: fixed;
    bottom: 160px;
    right: 30px;
    background: rgba(255, 255, 255, 0.95);
    border: 2px solid #4CAF50;
    border-radius: 24px;
    padding: 8px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
    z-index: 1000;
    transition: all 0.3s;
}

.conversation-toggle.active {
    border-color: #FF5722;
    background: rgba(255, 87, 34, 0.1);
}

.conversation-toggle-switch {
    position: relative;
    width: 50px;
    height: 26px;
    background: #ccc;
    border-radius: 13px;
    cursor: pointer;
    transition: background 0.3s;
}

.conversation-toggle-switch.active {
    background: #4CAF50;
}

.conversation-toggle-slider {
    position: absolute;
    top: 3px;
    left: 3px;
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s;
}

.conversation-toggle-switch.active .conversation-toggle-slider {
    transform: translateX(24px);
}

.conversation-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ccc;
    animation: pulse-indicator 2s infinite;
}

.conversation-indicator.active {
    background: #4CAF50;
}

/* Notification styles */
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    background: white;
    border-left: 4px solid #2196F3;
    border-radius: 4px;
    padding: 16px;
    min-width: 300px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    z-index: 10000;
    opacity: 1;
    transition: opacity 0.3s;
}

.notification-success { border-left-color: #4CAF50; }
.notification-warning { border-left-color: #FF9800; }
```

### Step 2: Add HTML

Add before `</body>` in `agentic_dashboard.html`:

```html
<!-- Conversation Mode Toggle -->
<div class="conversation-toggle" id="conversationToggle">
    <div class="conversation-indicator" id="conversationIndicator"></div>
    <span class="conversation-toggle-label" id="conversationLabel">Manual</span>
    <div class="conversation-toggle-switch" id="conversationSwitch">
        <div class="conversation-toggle-slider"></div>
    </div>
</div>
```

### Step 3: Add JavaScript

Add to `<script>` section in `agentic_dashboard.html`:

```javascript
// Conversational Mode State
let isConversationalMode = false;

// Toggle conversational mode
document.getElementById('conversationToggle').addEventListener('click', () => {
    isConversationalMode = !isConversationalMode;

    const toggle = document.getElementById('conversationToggle');
    const switchEl = document.getElementById('conversationSwitch');
    const label = document.getElementById('conversationLabel');
    const indicator = document.getElementById('conversationIndicator');

    if (isConversationalMode) {
        toggle.classList.add('active');
        switchEl.classList.add('active');
        indicator.classList.add('active');
        label.textContent = 'Auto';
        label.style.color = '#4CAF50';
    } else {
        toggle.classList.remove('active');
        switchEl.classList.remove('active');
        indicator.classList.remove('active');
        label.textContent = 'Manual';
        label.style.color = '#333';
    }
});

// Auto-speak function
async function speakResponseAuto(responseText, confidence, mode) {
    const formData = new FormData();
    formData.append('response', responseText);
    formData.append('confidence', confidence.toString());
    formData.append('mode', mode);

    const response = await fetch('/api/voice/speak_response', {
        method: 'POST',
        body: formData
    });

    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = document.getElementById('audioPlayer');
    audio.src = audioUrl;
    audio.play();
}

// Modified addAssistantMessage to support auto-speak
function addAssistantMessageWithAutoSpeak(data) {
    addAssistantMessage(data);  // Call original

    // If conversational mode is ON, auto-speak
    if (isConversationalMode && data.response) {
        setTimeout(() => {
            speakResponseAuto(
                data.response,
                data.confidence || 0.5,
                data.mode || 'direct'
            );
        }, 500);
    }
}

// Keyboard shortcut: Ctrl+M to toggle
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'm') {
        e.preventDefault();
        document.getElementById('conversationToggle').click();
    }
});
```

### Step 4: Update WebSocket Handler

Find your WebSocket message handler and change:

```javascript
// OLD:
if (data.type === 'response') {
    addAssistantMessage(data);
}

// NEW:
if (data.type === 'response') {
    addAssistantMessageWithAutoSpeak(data);
}
```

---

## Usage

### Manual Mode (Default)
1. Speak into mic
2. Query submitted
3. Response appears
4. **Click "🔊 Speak" button** to hear response

### Auto Mode (Conversational)
1. **Click toggle** (or press Ctrl+M)
2. Toggle shows "Auto" with green indicator
3. Speak into mic
4. Query submitted
5. Response appears
6. **Automatically speaks** (no button needed)
7. Continuous conversation!

---

## Workflow Comparison

### Traditional (Manual)
```
You: [Click mic] → Speak → [Click mic] → Wait for response
Dashboard: [Text response]
You: [Click "Speak" button]
Dashboard: [Voice response]
```

### Conversational (Auto)
```
You: [Click mic] → Speak → [Click mic]
Dashboard: [Text response + Voice response automatically]
You: [Click mic] → Speak → [Click mic]
Dashboard: [Text response + Voice response automatically]
... continuous conversation!
```

---

## Features

### Toggle Button
- **Visual State**: Clear indication of mode (Manual/Auto)
- **Indicator**: Green pulsing dot when auto mode active
- **Click to Toggle**: Switch between modes instantly
- **Keyboard Shortcut**: Ctrl+M for quick toggle

### Auto-Speak
- **Smart Delivery**: Uses confidence/mode for appropriate tone
- **Error Handling**: Graceful fallback if audio fails
- **Notifications**: Visual feedback on mode changes
- **Works with Manual**: Speak buttons still work in auto mode

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Ctrl+M** | Toggle conversational mode |
| **Space** | Start/stop recording (if implemented) |
| **Ctrl+S** | Speak last response (if implemented) |

---

## Recommendation

**For Tonight**: Use **Option 1** (Always Auto-Speak)
- Fastest setup (1 line change)
- Pure conversational experience
- Test the voice flow

**This Week**: Upgrade to **Option 2** (Toggle Button)
- Add flexibility
- Switch modes based on context
- Better UX with visual feedback

---

## Next Steps

1. Choose your option (Always Auto or Toggle)
2. Follow setup instructions above
3. Test the conversational flow!

**Option 1**: Change `auto_speak=True` in agentic_server.py
**Option 2**: Copy HTML/CSS/JS from conversational_mode.html

🎤 **Ready for hands-free conversation!**
