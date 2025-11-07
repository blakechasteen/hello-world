# Chronos iOS Shortcuts

Voice-first time tracking via Siri. Works immediately with your existing Chronos server.

## Quick Start

1. **Set up SSH access** (see setup guide below)
2. **Import shortcuts** (instructions below)
3. **Say**: "Hey Siri, track time"

---

## Available Shortcuts

### 1. Track Time (Main Shortcut)
**Trigger**: "Hey Siri, track time"

**Menu Options**:
- Start Task
- Stop Task
- Add Note
- Check Status

**Usage**:
```
You: "Hey Siri, track time"
Siri: "What would you like to do?"
You: [Tap "Start Task" or say "Start"]
Siri: "What task?"
You: "garden work" (or dictate)
Siri: ✅ Started garden_work at 14:30:00
```

### 2. Quick Start (No Menu)
**Trigger**: "Hey Siri, start tracking"

Immediately asks for task name and starts tracking.

### 3. Quick Stop
**Trigger**: "Hey Siri, stop tracking"

Immediately stops current task.

### 4. Voice Note
**Trigger**: "Hey Siri, time note"

Add a note to current task via dictation.

---

## SSH Setup

### Prerequisites
- Mac or Linux server running Chronos
- SSH access to that server
- iOS Shortcuts app

### Option 1: SSH Key (Recommended)

1. **Generate SSH key on iOS** (via a-Shell app):
```bash
ssh-keygen -t ed25519 -f ~/.ssh/chronos_key
```

2. **Copy public key to server**:
```bash
ssh-copy-id -i ~/.ssh/chronos_key.pub user@your-server.com
```

3. **Test connection**:
```bash
ssh -i ~/.ssh/chronos_key user@your-server.com "python -m chronos status"
```

### Option 2: Tailscale (Zero-Config VPN)

If your server is behind NAT, use Tailscale for secure access:

1. Install Tailscale on server: https://tailscale.com
2. Install Tailscale on iOS
3. Connect both devices
4. Use Tailscale IP in shortcuts: `100.x.x.x`

---

## Shortcut Configuration

### Track Time (Main)

**Actions**:
```
1. Menu
   Prompt: "What would you like to do?"
   Options: Start Task | Stop Task | Add Note | Check Status

2. If "Start Task":
   - Dictate Text → taskName
   - Run Script Over SSH:
     Host: your-server.com
     User: your-username
     Script: python -m chronos voice "start [taskName]"
   - Show Notification: [SSH Output]

3. If "Stop Task":
   - Run Script Over SSH:
     Host: your-server.com
     User: your-username
     Script: python -m chronos voice "done"
   - Show Notification: [SSH Output]

4. If "Add Note":
   - Dictate Text → noteText
   - Run Script Over SSH:
     Host: your-server.com
     User: your-username
     Script: python -m chronos voice "note: [noteText]"
   - Show Notification: [SSH Output]

5. If "Check Status":
   - Run Script Over SSH:
     Host: your-server.com
     User: your-username
     Script: python -m chronos status
   - Show Notification: [SSH Output]
```

### Quick Start

**Actions**:
```
1. Ask for Input
   Prompt: "What task?"
   Type: Text

2. Run Script Over SSH:
   Host: your-server.com
   User: your-username
   Script: python -m chronos voice "start [Input]"

3. Show Notification:
   Title: "Time Tracking"
   Body: [SSH Output]
```

### Quick Stop

**Actions**:
```
1. Run Script Over SSH:
   Host: your-server.com
   User: your-username
   Script: python -m chronos voice "done"

2. Show Notification:
   Title: "Time Tracking"
   Body: [SSH Output]
```

---

## Apple Watch Support

All shortcuts work from Apple Watch:
- Raise wrist
- Say "Hey Siri, track time"
- Dictate task name
- Done!

**No phone needed.**

---

## Offline Mode (Future)

Currently requires SSH connection. For offline mode, see the native iOS app (coming soon).

---

## Troubleshooting

### "Can't connect to server"
- Check WiFi/cellular connection
- Verify SSH credentials
- Try: `ssh user@your-server.com` in terminal
- Consider Tailscale for easier setup

### "Command not found: python"
- Update script to use full path: `/usr/bin/python3 -m chronos`
- Or activate virtualenv first: `source ~/.venv/bin/activate && python -m chronos`

### "Permission denied"
- Check SSH key permissions
- Verify user has access to Chronos directory
- Try password auth first, then set up keys

### Voice recognition issues
- Speak clearly and slowly
- Use simple task names
- Dictation works better in quiet environments
- Can also type instead of dictate

---

## Tips

**Fast Workflow**:
- Morning: "Hey Siri, track time" → Start → "planning"
- Working: Just work
- Task switch: "Hey Siri, track time" → Start → "emails"
- End of day: "Hey Siri, stop tracking"

**Tags via voice**:
- Say: "garden work hashtag farm hashtag physical"
- Chronos parses: `garden_work #farm #physical`

**Quick notes**:
- While task running: "Hey Siri, time note" → dictate
- Automatically linked to current task

---

## Next Step

See NATIVE_APP.md for the full iOS app (local storage, widgets, offline mode).
