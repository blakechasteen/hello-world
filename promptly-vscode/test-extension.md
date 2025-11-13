# Testing the Extension - Checklist

Follow these steps EXACTLY:

## ✅ Step 1: Verify You're in the Right Folder

Run this in terminal:
```bash
pwd
# Should show: .../mythRL/promptly-vscode
```

If not:
```bash
cd c:\Users\blake\OneDrive\Documents\mythRL\promptly-vscode
```

## ✅ Step 2: Open in VS Code

```bash
code .
```

Wait for VS Code to fully load.

## ✅ Step 3: Check Files Exist

You should see in the Explorer (left sidebar):
- ✓ src/
- ✓ out/
- ✓ package.json
- ✓ node_modules/

If `out/` or `node_modules/` is missing:
```bash
npm install
npm run compile
```

## ✅ Step 4: Launch Extension

Press **F5** (or Run → Start Debugging)

**What should happen:**
1. Bottom status bar shows "Starting extension host..."
2. A **NEW VS Code window** opens (takes 3-5 seconds)
3. New window title says `[Extension Development Host]`

**What if nothing happens?**
1. Click the Run icon on left sidebar (▶️ with bug)
2. At the top, dropdown should show "Run Extension"
3. Click green ▶️ play button

## ✅ Step 5: Verify Extension Loaded

In the **ORIGINAL** VS Code window (where your code is):
1. Look at the bottom panel
2. Click "Debug Console" tab
3. You should see:
   ```
   Promptly extension activated
   ```

**If you see errors instead:**
- Copy the error message
- Share it with me!

## ✅ Step 6: Open Promptly Chat

In the **Extension Development Host** window (the NEW window):

### Method A: Command Palette (Recommended)
1. Press `Ctrl+Shift+P`
2. Type: `promptly`
3. Click: "Promptly: Open Chat"

### Method B: Status Bar
1. Look at bottom-right
2. Find: "$(comment-discussion) Promptly"
3. Click it

### Method C: Keyboard
- Press: `Ctrl+Alt+P`

## ✅ Step 7: Test Autocomplete

In the chat panel:
1. Type: `/`
2. Autocomplete menu should appear
3. Try: `/help`

## 🐛 Debugging Checklist

If it's not working, check these:

### Is the Extension Running?
In Extension Development Host, press `Ctrl+Shift+P` and type:
```
Developer: Show Running Extensions
```

Look for "Promptly" in the list.

### Any Compile Errors?
In your terminal:
```bash
cd promptly-vscode
npm run compile
```

Should complete with no errors.

### Check Debug Console
In your ORIGINAL VS Code window:
1. View → Debug Console
2. Look for errors (red text)
3. Share any errors you see

### Try Reloading
In Extension Development Host:
1. Press `Ctrl+Shift+P`
2. Type: `Reload Window`
3. Wait 3 seconds
4. Try opening Promptly again

## 📸 What You Should See

When working correctly:

**Original Window:**
- Debug Console shows: "Promptly extension activated"
- Bottom status bar shows: "Debugging"

**Extension Development Host:**
- Status bar (bottom-right) shows: "Promptly" button
- Command Palette shows: "Promptly: Open Chat" when you search
- Chat panel opens when you activate it
- Typing `/` shows autocomplete menu

## 🆘 Still Stuck?

Share this info:
1. Output of `npm run compile`
2. Screenshot of Debug Console (bottom panel in original window)
3. What happens when you press F5
