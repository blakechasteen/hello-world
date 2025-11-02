# 🎉 Ollama Quick Start

## ✅ Installation Complete!

Ollama version 0.12.6 is installed at:
`C:\Users\blake\AppData\Local\Programs\Ollama\ollama.exe`

## 🔄 Current Status

The `llama3.2:3b` model is downloading (this takes a few minutes for the first time).

## 🚀 Once Download Completes

### Option 1: Restart PowerShell (Recommended)
Close and reopen your terminal, then `ollama` will work directly:
```powershell
ollama --version
ollama list
ollama run llama3.2:3b "Hello!"
```

### Option 2: Use in Current Session
Load the helper script:
```powershell
. .\ollama_helper.ps1
ollama list
```

### Option 3: Use Full Path
```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list
```

## 🎯 Test UltraPrompt with Ollama

Once the model download completes, try:

```powershell
cd "c:\Users\blake\Documents\mythRL\Promptly\promptly"
python ultraprompt_ollama.py "write a Python function to calculate fibonacci numbers"
```

This will:
1. Use Promptly's ultraprompt template
2. Expand your simple request into a detailed prompt
3. Execute it with Ollama (locally, no API keys!)

## 📋 Useful Commands

```powershell
# Check what models you have
ollama list

# Run a prompt
ollama run llama3.2:3b "Explain Python decorators"

# Interactive chat
ollama run llama3.2:3b

# Check version
ollama --version

# Pull more models (when you want them)
ollama pull qwen2.5-coder:3b  # Great for coding
ollama pull llama3.1:8b        # Higher quality
```

## 🔧 Update the ultraprompt_ollama.py Script

The script uses the full path automatically if `ollama` isn't in PATH, so it should work right away!

## 🎬 Complete Workflow Example

```powershell
# 1. Simple request
python ultraprompt_ollama.py "create a REST API endpoint"

# 2. With execution (expands AND runs the expanded prompt)
python ultraprompt_ollama.py "create a REST API endpoint" --execute

# 3. With specific model
python ultraprompt_ollama.py "explain quantum computing" llama3.1:8b

# 4. Check model download status
ollama list
```

## 💡 Next Steps

1. ✅ Wait for llama3.2:3b to finish downloading
2. ✅ Restart PowerShell (optional, but makes `ollama` command work everywhere)
3. ✅ Test with: `python ultraprompt_ollama.py "your prompt here"`
4. 🎉 Enjoy free, private, local AI!

---

**Current model download in progress... check with:**
```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list
```
