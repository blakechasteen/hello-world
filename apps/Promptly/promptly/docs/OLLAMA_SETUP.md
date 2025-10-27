# Using Ollama with Promptly and LLM CLI

## What is Ollama?

Ollama lets you run large language models locally on your machine - no API keys needed! Perfect for:
- Privacy-sensitive work
- Offline development
- Free unlimited usage
- Fast local inference

## Installation

### 1. Install Ollama

Download and install from: https://ollama.ai/download

Or use winget:
```powershell
winget install Ollama.Ollama
```

### 2. Verify Installation

```powershell
ollama --version
```

### 3. Pull a Model

```powershell
# Recommended models (from smallest to largest)
ollama pull llama3.2:1b       # 1.3GB - Fast, good for simple tasks
ollama pull llama3.2:3b       # 2GB - Balanced
ollama pull qwen2.5:3b        # 2.3GB - Good reasoning
ollama pull llama3.1:8b       # 4.7GB - High quality
ollama pull mistral:7b        # 4.1GB - Great all-rounder
ollama pull llama3.1:70b      # 40GB - Best quality (if you have the RAM)

# Check what's installed
ollama list
```

## Integration with LLM CLI

The `llm` tool can work with Ollama models!

### 1. Install the Ollama Plugin for LLM

```powershell
llm install llm-ollama
```

### 2. Use Ollama Models with LLM

```powershell
# List available Ollama models
llm models list

# Use an Ollama model
llm -m ollama/llama3.2:3b "Explain Python decorators"

# Set as default
llm models default ollama/llama3.2:3b
```

## Using Ollama Directly

```powershell
# Simple prompt
ollama run llama3.2:3b "Write a Python function to sort a list"

# Interactive chat
ollama run llama3.2:3b

# With system prompt
ollama run llama3.2:3b --system "You are a helpful coding assistant"
```

## Integration with Promptly UltraPrompt

### Option 1: Using LLM CLI (Recommended)

```powershell
# Update the ultraprompt script to use Ollama
python ultraprompt_llm.py "write a sorting algorithm" ollama/llama3.2:3b
```

### Option 2: Direct Ollama Integration

We've created `ultraprompt_ollama.py` for direct Ollama usage (no API keys needed!):

```powershell
python ultraprompt_ollama.py "write a sorting algorithm"
python ultraprompt_ollama.py "explain quantum computing" llama3.1:8b
```

## Recommended Models for Different Tasks

| Task | Model | Size | Why |
|------|-------|------|-----|
| Code generation | qwen2.5-coder:3b | 2GB | Specialized for code |
| General chat | llama3.2:3b | 2GB | Fast and capable |
| Complex reasoning | llama3.1:8b | 4.7GB | Best balance |
| Writing | mistral:7b | 4.1GB | Creative writing |
| Maximum quality | llama3.1:70b | 40GB | Best results (needs 32GB+ RAM) |

## Example Workflows

### 1. Simple UltraPrompt with Ollama

```powershell
# First expand the request
ollama run llama3.2:3b "You are an expert assistant. Expand this into a detailed prompt: write a web scraper"

# Then use the expanded prompt
ollama run llama3.2:3b "[paste expanded prompt here]"
```

### 2. Automated with Promptly

```powershell
# Uses Promptly templates + Ollama
python ultraprompt_ollama.py "create a REST API with FastAPI"
```

### 3. Chain Multiple Prompts

```python
# In your Python scripts
import subprocess

def ollama_prompt(prompt, model="llama3.2:3b"):
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

# Use with Promptly templates
from promptly import Promptly
p = Promptly()
template = p.get('ultraprompt')
expanded = ollama_prompt(template['content'].format(request="your task"))
final_result = ollama_prompt(expanded)
```

## Performance Tips

1. **First run is slow** - Ollama loads the model into memory
2. **Keep Ollama running** - Models stay in memory for faster subsequent calls
3. **Use smaller models for iteration** - Switch to larger models for final output
4. **Adjust temperature** - Add `--temperature 0.7` for more creative outputs

## Troubleshooting

### Ollama not responding
```powershell
# Check if Ollama service is running
Get-Process ollama

# Restart Ollama
Stop-Process -Name ollama
ollama serve
```

### Out of memory
- Use smaller models (llama3.2:1b instead of llama3.1:70b)
- Close other applications
- Check available RAM: `Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory`

### Model not found
```powershell
# Pull the model first
ollama pull llama3.2:3b
```

## Advantages Over Cloud APIs

✅ **Free** - No API costs  
✅ **Private** - Your data stays local  
✅ **Fast** - No network latency  
✅ **Offline** - Works without internet  
✅ **No rate limits** - Use as much as you want  

## Disadvantages

❌ Requires good hardware (8GB+ RAM recommended)  
❌ Smaller models less capable than GPT-4/Claude  
❌ Takes up disk space  
❌ First load can be slow  

## Best Practices

1. Start with `llama3.2:3b` for testing
2. Use `qwen2.5-coder:3b` for code-specific tasks
3. Keep multiple models for different use cases
4. Upgrade to 8B or 70B models for production
5. Combine with Promptly for reproducible prompts
