# Voice Commands

Control the Workflow Builder hands-free with natural language voice commands.

## Overview

The Workflow Builder includes a comprehensive voice control system with 18+ workflow-specific commands. Voice control enables:
- Hands-free workflow creation
- Rapid node manipulation
- Quick execution and debugging
- Accessibility for users with mobility limitations

## Enabling Voice Commands

### Browser Permissions

1. Click the **microphone icon** (🎤) in the toolbar
2. Allow microphone access when prompted
3. Green indicator shows voice is active

### Keyboard Shortcut

Press `V` to toggle voice input on/off.

### Requirements

- Modern browser (Chrome, Edge, Firefox)
- Microphone access permission
- Stable internet connection (for speech recognition)

## Available Commands

### Navigation Commands

| Command | Action |
|---------|--------|
| "zoom in" | Increase canvas zoom |
| "zoom out" | Decrease canvas zoom |
| "fit to view" | Fit all nodes in viewport |
| "center canvas" | Reset pan to center |
| "go to [node name]" | Focus on specific node |

### Node Creation

| Command | Action |
|---------|--------|
| "add query node" | Add HoloLoom Query node |
| "add memory node" | Add Memory Store node |
| "add decision node" | Add Thompson Sampler node |
| "add output node" | Add Response Generator node |
| "add [node type]" | Add specified node type |

**Node Type Keywords**:
- "query" → hololoom_query
- "search" → memory_search
- "multi query" → multi_query
- "embedder" → matryoshka_embedder
- "synthesizer" → synthesizer
- "refiner" → recursive_refiner
- "store" → memory_store
- "retriever" → context_retriever
- "fusion" → knowledge_fusion
- "sampler" → thompson_sampler
- "convergence" → convergence_engine
- "guardrails" → safety_guardrails
- "response" → response_generator
- "converter" → format_converter
- "branch" → conditional_branch
- "loop" → loop_iterator
- "parallel" → parallel_executor

### Node Selection

| Command | Action |
|---------|--------|
| "select [node name]" | Select node by label |
| "select all" | Select all nodes |
| "deselect" | Clear selection |
| "select next" | Select next node in flow |
| "select previous" | Select previous node |

### Node Manipulation

| Command | Action |
|---------|--------|
| "delete node" | Delete selected node |
| "duplicate node" | Duplicate selected node |
| "move left/right/up/down" | Move selected node |
| "rename to [name]" | Rename selected node |
| "configure" | Open properties panel |

### Connection Commands

| Command | Action |
|---------|--------|
| "connect [source] to [target]" | Create connection |
| "disconnect" | Remove selected connection |
| "disconnect all" | Remove all connections from node |

### Execution Commands

| Command | Action |
|---------|--------|
| "run workflow" | Execute the workflow |
| "run" | Execute the workflow |
| "stop" | Stop execution |
| "pause" | Pause at next breakpoint |
| "step" | Step to next node |
| "step over" | Execute current, pause at next |
| "continue" | Continue from breakpoint |

### Debugging Commands

| Command | Action |
|---------|--------|
| "set breakpoint" | Add breakpoint to selected |
| "clear breakpoint" | Remove breakpoint |
| "clear all breakpoints" | Remove all breakpoints |
| "show variables" | Open variable inspector |
| "show timeline" | Open execution timeline |

### File Commands

| Command | Action |
|---------|--------|
| "save workflow" | Save current workflow |
| "export JSON" | Export as JSON |
| "export Python" | Export as Python code |
| "export YAML" | Export as YAML |
| "new workflow" | Create new workflow |
| "open workflow" | Open file picker |

### Utility Commands

| Command | Action |
|---------|--------|
| "undo" | Undo last action |
| "redo" | Redo last action |
| "help" | Show voice command help |
| "close" | Close current panel |
| "toggle dark mode" | Switch theme |

## Voice Feedback

### Visual Indicators

```
┌─────────────────────────────────────────┐
│ Voice Status                            │
├─────────────────────────────────────────┤
│ 🎤 Listening...                         │
│                                         │
│ Last command: "add query node"          │
│ Status: ✓ Executed                      │
│                                         │
│ [Mute] [Settings]                       │
└─────────────────────────────────────────┘
```

### Audio Feedback

| Sound | Meaning |
|-------|---------|
| **Ding** | Command recognized |
| **Double ding** | Command executed |
| **Error tone** | Command not understood |
| **Click** | Voice toggled on/off |

## Advanced Usage

### Chained Commands

Execute multiple commands in sequence:

```
"add query node then connect to output"
```

### Conditional Phrases

Natural language variations are supported:

- "please add a query node"
- "can you add query node"
- "I want to add a query node"
- "create a query node"

### Named References

Reference nodes by their label:

```
"connect Research to Synthesizer"
"select the Query node"
"delete Response Generator"
```

### Context-Aware Commands

Commands adapt to current selection:

```
# With node selected:
"delete" → deletes selected node

# With connection selected:
"delete" → deletes selected connection

# Nothing selected:
"delete" → shows error
```

## Configuration

### Voice Settings Panel

Access via: **View** → **Voice Settings** or click gear icon next to microphone.

```
┌─────────────────────────────────────────┐
│ Voice Settings                       ×  │
├─────────────────────────────────────────┤
│ Language: [English (US)     ▼]          │
│                                         │
│ ☑ Audio feedback                        │
│ ☑ Visual confirmation                   │
│ ☐ Continuous listening                  │
│                                         │
│ Sensitivity: [▓▓▓▓▓▓░░░░] 60%          │
│                                         │
│ Keyword: "workflow" (optional wake word)│
│                                         │
│ [Test Microphone] [Reset to Defaults]   │
└─────────────────────────────────────────┘
```

### Options

| Setting | Description |
|---------|-------------|
| **Language** | Speech recognition language |
| **Audio feedback** | Enable/disable sound effects |
| **Visual confirmation** | Show command execution overlay |
| **Continuous listening** | Always listen (vs push-to-talk) |
| **Sensitivity** | Microphone sensitivity level |
| **Keyword** | Optional wake word before commands |

### Custom Commands

Define custom voice commands:

```javascript
// In workflow settings
{
  "voice_commands": {
    "custom": [
      {
        "phrase": "add my pipeline",
        "action": "add_template",
        "template_id": "my-custom-template"
      },
      {
        "phrase": "run tests",
        "action": "execute",
        "input": { "mode": "test" }
      }
    ]
  }
}
```

## Troubleshooting

### Voice Not Recognized

**Symptoms**: Commands not detected

**Solutions**:
1. Check microphone permissions in browser
2. Ensure microphone is not muted
3. Speak clearly and at normal pace
4. Check sensitivity settings
5. Try a different browser

### Wrong Command Executed

**Symptoms**: Different action than intended

**Solutions**:
1. Check command spelling/pronunciation
2. Use exact command phrases from reference
3. Avoid background noise
4. Add pause between commands

### Delayed Response

**Symptoms**: Long delay before execution

**Solutions**:
1. Check internet connection
2. Clear browser cache
3. Reduce audio feedback settings
4. Close other tabs using microphone

### Microphone Not Found

**Symptoms**: No microphone option available

**Solutions**:
1. Connect/enable microphone
2. Check system sound settings
3. Restart browser
4. Check browser permissions at system level

## Accessibility Features

Voice commands support accessibility:

- **Screen reader integration**: Commands announced
- **Keyboard alternatives**: Every voice command has a keyboard shortcut
- **High contrast mode**: Visual feedback works in all themes
- **Adjustable timing**: Configure command timeout

### Alternative Input Methods

| Method | Usage |
|--------|-------|
| Voice | Hands-free operation |
| Keyboard | Full shortcut coverage |
| Mouse | Standard point-and-click |
| Touch | Mobile/tablet support |

## Best Practices

### For Best Recognition

1. **Speak clearly**: Moderate pace, clear pronunciation
2. **Reduce noise**: Quiet environment improves accuracy
3. **Use standard phrases**: Stick to documented commands
4. **Pause between commands**: Allow processing time

### Workflow Tips

1. **Name nodes descriptively**: Easier voice reference
2. **Use templates**: "Add my research pipeline"
3. **Learn shortcuts**: Faster than voice for common actions
4. **Combine methods**: Voice for creation, mouse for positioning

### Performance

- Voice processing adds ~200-500ms latency
- Works offline with reduced accuracy (browser-dependent)
- Continuous listening uses minimal resources

## Keyboard Shortcuts Reference

Every voice command has a keyboard equivalent:

| Voice Command | Keyboard |
|---------------|----------|
| "zoom in" | `+` or `Ctrl+=` |
| "zoom out" | `-` or `Ctrl+-` |
| "fit to view" | `0` |
| "run workflow" | `Ctrl+Enter` |
| "stop" | `Escape` |
| "undo" | `Ctrl+Z` |
| "redo" | `Ctrl+Y` |
| "save workflow" | `Ctrl+S` |
| "select all" | `Ctrl+A` |
| "delete" | `Delete` |
| "duplicate" | `Ctrl+D` |

---

← [Debugging Tools](debugging.md) | [Export Formats](export-formats.md) →
