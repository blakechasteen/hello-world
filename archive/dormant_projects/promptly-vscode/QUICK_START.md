# Quick Start - Testing the Extension

## Step 1: Open the Extension Folder in VS Code

```bash
# Make sure you're in the extension folder
cd c:\Users\blake\OneDrive\Documents\mythRL\promptly-vscode

# Open in VS Code
code .
```

## Step 2: Launch Extension Development Host

In VS Code, press **F5** (or Run → Start Debugging)

You should see:
- ✅ A **new VS Code window** opens with `[Extension Development Host]` in title bar
- ✅ Original window shows "Running extension..." at the bottom

**If nothing happens:**
1. Open the Run panel (Ctrl+Shift+D)
2. Select "Run Extension" from dropdown
3. Click the green ▶️ play button

## Step 3: Open Promptly Chat

In the **Extension Development Host** window (the new window), try these methods:

### Method 1: Command Palette (Most Reliable)
1. Press `Ctrl+Shift+P`
2. Type "Promptly"
3. Select "Promptly: Open Chat"

### Method 2: Keyboard Shortcut
Press `Ctrl+Alt+P`

### Method 3: Status Bar
Look at the bottom-right of VS Code
Click the **"$(comment-discussion) Promptly"** button

## Step 4: Test Slash Commands

Once the chat panel opens, type:
```
/
```

You should see autocomplete suggestions appear!

Try:
```
/help
/gs
```

## Troubleshooting

### Extension Didn't Activate?

Check the **Debug Console** in your original VS Code window:
1. View → Debug Console
2. Look for: "Promptly extension activated"

If you see errors, share them!

### Keybinding Conflict?

VS Code might have another extension using `Ctrl+Alt+P`.

**Change the keybinding:**
1. In the extension folder, open `package.json`
2. Find line 52-55 (keybindings section)
3. Change to something else:
   ```json
   "key": "ctrl+shift+alt+p"
   ```
4. Recompile: `npm run compile`
5. Reload window: Ctrl+Shift+P → "Reload Window"

### Extension Not in Command Palette?

The extension might not have loaded. Check:
1. In Extension Development Host, press `Ctrl+Shift+P`
2. Type "Developer: Show Running Extensions"
3. Look for "Promptly" in the list

If not there:
1. Close Extension Development Host
2. In original window, check for errors in Debug Console
3. Press F5 again

## Quick Test

Here's the fastest way to verify everything works:

1. **Original VS Code**: Press F5
2. **Wait 2 seconds** for new window
3. **New window**: Press Ctrl+Shift+P
4. Type: **"Promptly: Open Chat"**
5. In chat, type: **/help**

If you see the help message, it works! 🎉

## Still Not Working?

Run this to check for errors:

```bash
cd promptly-vscode
npm run compile
```

Share any error messages you see!
