# UltraPrompt with LLM CLI - Demo

## What We Created

We've set up a workflow that combines:
1. **Promptly** - For managing and versioning prompts
2. **llm CLI** - For calling LLM models from the command line

## The UltraPrompt Template

We added an "ultraprompt" template to Promptly that expands simple requests into comprehensive prompts:

```
"You are an expert assistant. Take the following request and expand it into a 
comprehensive, detailed prompt that will get the best results from an AI: {request}"
```

## How to Use It

### Basic Usage (Manual)

```bash
# 1. Get the ultraprompt template from Promptly
cd "c:\Users\blake\Documents\mythRL\Promptly\promptly"
python promptly.py get ultraprompt

# 2. Use it with llm CLI directly
llm "You are an expert assistant. Take the following request and expand it into a comprehensive, detailed prompt that will get the best results from an AI: write a sorting algorithm"
```

### Automated Usage (With Script)

```bash
# Use the ultraprompt_llm.py script
python ultraprompt_llm.py "write a sorting algorithm"

# With a specific model
python ultraprompt_llm.py "explain quantum computing" claude-3.5-sonnet
```

## Setting Up API Keys

If you see "No key found" error, you need to configure an API key:

```bash
# For OpenAI (default)
llm keys set openai
# Then paste your API key

# Or set environment variable
$env:OPENAI_API_KEY="your-key-here"

# For Anthropic Claude
llm keys set anthropic
```

## Example Commands

### Simple LLM Call
```bash
llm "Write a Python function to calculate fibonacci numbers"
```

### With Specific Model
```bash
llm -m claude-3.5-sonnet "Explain machine learning"
```

### Two-Step UltraPrompt (Manual)
```bash
# Step 1: Expand the prompt
llm "You are an expert assistant. Expand this request into a detailed prompt: write a web scraper"

# Step 2: Use the expanded prompt (copy from output above)
llm "Create a Python web scraper that...[expanded prompt]"
```

### Using Promptly for Versioning
```bash
# Add different versions of prompts
python promptly.py add coder "Write code for: {task}"
python promptly.py add coder "Write clean, well-documented code with tests for: {task}"

# View history
python promptly.py log --name coder

# Get specific version
python promptly.py get coder --version 1
```

## Advanced: Chain Multiple Prompts

```bash
# Create a chain
python promptly.py add outline "Create an outline for: {topic}"
python promptly.py add expand "Expand this outline into full text: {output}"
python promptly.py chain create writing-flow outline expand
```

## What's Cool About This Setup

1. **Version Control** - All your prompts are versioned automatically
2. **Branching** - Test experimental prompts without losing working ones
3. **Templating** - Reuse prompts with variables
4. **CLI Integration** - Combine with any CLI tool or script
5. **Evaluation** - Test prompt performance systematically

## Next Steps

1. Set up your API keys: `llm keys set openai`
2. Try the ultraprompt: `python ultraprompt_llm.py "your request"`
3. Create custom prompt templates in Promptly
4. Build prompt chains for complex workflows
5. Use evaluation to A/B test different prompt versions
