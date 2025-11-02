#!/usr/bin/env python3
"""
UltraPrompt with Ollama
Direct integration with Ollama for local LLM inference (no API keys needed!)
"""

import sys
import subprocess
import json
import tempfile
from pathlib import Path

# Add current directory to path to import promptly
sys.path.insert(0, str(Path(__file__).parent))
from promptly import Promptly


def get_ollama_path():
    """Get the path to ollama executable"""
    # Try ollama in PATH first
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return "ollama"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Try default Windows installation path
    import os
    default_path = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Ollama', 'ollama.exe')
    if os.path.exists(default_path):
        return default_path
    
    return None


def check_ollama():
    """Check if Ollama is installed and running"""
    ollama_path = get_ollama_path()
    if not ollama_path:
        return False
    
    try:
        result = subprocess.run(
            [ollama_path, "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def list_ollama_models():
    """List available Ollama models"""
    ollama_path = get_ollama_path()
    if not ollama_path:
        return None
    
    try:
        result = subprocess.run(
            [ollama_path, "list"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except Exception:
        return None


def ollama_run(prompt: str, model: str = "llama3.2:3b", system: str = None):
    """
    Run a prompt with Ollama
    
    Args:
        prompt: The prompt to send
        model: Ollama model name (default: llama3.2:3b)
        system: Optional system prompt (prepended to the prompt)
    """
    ollama_path = get_ollama_path()
    if not ollama_path:
        print("❌ Error: Ollama not found.")
        return None
    
    try:
        # Combine system prompt with user prompt if provided
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        
        print(f"\n🤖 Running with Ollama model: {model}")
        print(f"📏 Prompt length: {len(full_prompt)} characters")
        print("⏳ This may take a moment on first run...\n")
        
        # For very long prompts (>5000 chars), save to temp file and use stdin
        if len(full_prompt) > 5000:
            print("📝 Using temp file for long prompt...")
            # Save prompt to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(full_prompt)
                temp_file = f.name
            
            try:
                # Read from file and pipe to ollama
                with open(temp_file, 'r', encoding='utf-8') as f:
                    prompt_content = f.read()
                
                result = subprocess.run(
                    [ollama_path, "run", model],
                    input=prompt_content,
                    capture_output=True,
                    text=True,
                    timeout=180,  # 3 minutes for long prompts
                    encoding='utf-8'
                )
            finally:
                # Clean up temp file
                import os
                try:
                    os.unlink(temp_file)
                except:
                    pass
        else:
            # Build command - on Windows with full path, we need to handle it differently
            if sys.platform == "win32" and ollama_path.endswith('.exe'):
                # Use the command as a string for shell=True
                # Escape quotes in the prompt
                escaped_prompt = full_prompt.replace('"', '""')
                cmd_str = f'"{ollama_path}" run {model} "{escaped_prompt}"'
                result = subprocess.run(
                    cmd_str,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    shell=True
                )
            else:
                cmd = [ollama_path, "run", model, full_prompt]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
        
        print(f"✅ Process completed (return code: {result.returncode})")
        print(f"📤 Output length: {len(result.stdout)} chars")
        
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return None
        
        return result.stdout.strip()
        
    except subprocess.TimeoutExpired:
        print("❌ Error: Request timed out. Try a smaller model or simpler prompt.")
        return None
    except FileNotFoundError:
        print("❌ Error: Ollama not found. Please install from https://ollama.ai")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def ultraprompt_with_ollama(user_request: str, model: str = "llama3.2:3b", execute: bool = False, template: str = "ultraprompt"):
    """
    Takes a simple user request and uses Promptly + Ollama to:
    1. Expand it into a comprehensive prompt (ultraprompt)
    2. Optionally execute the expanded prompt
    
    Args:
        user_request: The simple request from the user
        model: Ollama model to use
        execute: If True, also execute the expanded prompt
        template: Which ultraprompt template to use ('ultraprompt' or 'ultraprompt-advanced')
    """
    
    # Check if Ollama is available
    if not check_ollama():
        print("❌ Ollama is not running or not installed.")
        print("\n📥 To install Ollama:")
        print("   1. Visit https://ollama.ai/download")
        print("   2. Download and install for Windows")
        print("   3. Run: ollama pull llama3.2:3b")
        print("   4. Try again!")
        return None
    
    # Initialize Promptly
    promptly = Promptly()
    
    # Get the ultraprompt template
    try:
        prompt_data = promptly.get(template)
        if not prompt_data:
            print(f"❌ Error: '{template}' template not found in Promptly")
            print("\n💡 Creating default ultraprompt template...")
            
            default_ultraprompt = (
                "You are an expert assistant. Take the following request and expand it "
                "into a comprehensive, detailed prompt that will get the best results "
                "from an AI: {request}"
            )
            
            promptly.add(template, default_ultraprompt)
            prompt_data = promptly.get(template)
            print("✅ Created ultraprompt template!")
            
    except Exception as e:
        print(f"❌ Error with Promptly: {e}")
        return None
    
    # Format the ultraprompt with the user's request
    # Use replace instead of format to avoid issues with other {} in the template
    # Support both {request} and {text} placeholders
    ultraprompt_content = prompt_data['content'].replace('{request}', user_request)
    ultraprompt_content = ultraprompt_content.replace('{text}', user_request)
    
    print("=" * 70)
    print("🚀 ULTRAPROMPT - Expanding your request with Ollama")
    print("=" * 70)
    print(f"\n📝 Original request: {user_request}")
    print(f"📋 Template: {prompt_data['name']} (v{prompt_data['version']})")
    print(f"🤖 Model: {model}")
    print("-" * 70)
    
    # Call Ollama to expand the prompt
    expanded_prompt = ollama_run(
        ultraprompt_content,
        model=model,
        system="You are a prompt engineering expert. Create detailed, effective prompts."
    )
    
    if not expanded_prompt:
        return None
    
    print("\n" + "=" * 70)
    print("✨ EXPANDED PROMPT:")
    print("=" * 70)
    print(expanded_prompt)
    print("=" * 70)
    
    # Optionally execute the expanded prompt
    if execute:
        print("\n" + "=" * 70)
        print("🔄 EXECUTING EXPANDED PROMPT...")
        print("=" * 70)

        # Ensure any placeholders in the expanded prompt are replaced with the original request
        # Support a handful of common placeholders that templates might include
        substitute_keys = ['{request}', '{text}', '{query}', '{original}']
        execute_prompt = expanded_prompt
        for key in substitute_keys:
            execute_prompt = execute_prompt.replace(key, user_request)

        # Append an explicit execution wrapper so the model produces the final answer
        # instead of staying in a prompt-engineering/meta mode.
        execution_wrapper = (
            "Below are the INSTRUCTIONS for completing the user's request."
            "\n\nINSTRUCTIONS:\n" + execute_prompt + "\n\n"
            "TASK: Using the INSTRUCTIONS above, produce the FINAL OUTPUT that fulfils the user's request:\n\""
            + user_request.replace('"', '\\"')
            + "\"\n\nOUTPUT RULES:\n1) Output only the final answer.\n2) Do NOT include any explanations, internal reasoning, steps, or meta commentary.\n3) Do NOT ask questions or request clarifications.\n4) If the request expects code, output only the code block (no surrounding commentary)."
        )

        # Strong system instruction to enforce final-answer-only behavior
        execution_system = (
            "You are a strict executor. Follow the user's instructions and the INSTRUCTIONS provided. "
            "Return ONLY the final output requested. Do not provide any analysis, commentary, or prompt-engineering notes."
        )

        final_result = ollama_run(execution_wrapper, model=model, system=execution_system)

        if final_result:
            print("\n" + "=" * 70)
            print("✅ FINAL RESULT:")
            print("=" * 70)
            print(final_result)
            print("=" * 70)
            return final_result
    
    return expanded_prompt


def main():
    if len(sys.argv) < 2:
        print("🎯 UltraPrompt with Ollama")
        print("=" * 70)
        print("\nUsage:")
        print("  python ultraprompt_ollama.py <request> [model] [--execute] [--advanced]")
        print("\nExamples:")
        print('  python ultraprompt_ollama.py "write a sorting function"')
        print('  python ultraprompt_ollama.py "explain quantum computing" llama3.1:8b')
        print('  python ultraprompt_ollama.py "create a REST API" --execute')
        print('  python ultraprompt_ollama.py "medical diagnosis guide" --advanced')
        print("\nFlags:")
        print("  --execute, -e   Execute the expanded prompt automatically")
        print("  --advanced, -a  Use ultraprompt-advanced (detailed, structured)")
        print("\nAvailable models (if installed):")
        
        models = list_ollama_models()
        if models:
            print(models)
        else:
            print("  (Run 'ollama list' to see installed models)")
            print("\n💡 Recommended starting models:")
            print("  - llama3.2:3b (fast, 2GB)")
            print("  - qwen2.5-coder:3b (great for code, 2GB)")
            print("  - llama3.1:8b (high quality, 4.7GB)")
        
        print("\n📖 This tool:")
        print("  1. Uses Promptly's 'ultraprompt' template")
        print("  2. Expands your simple request into a detailed prompt")
        print("  3. Optionally executes the expanded prompt (--execute flag)")
        print("\n✅ Benefits: Free, private, offline, no API keys needed!")
        sys.exit(1)
    
    # Parse arguments
    user_request = sys.argv[1]
    model = "llama3.2:3b"  # Default model
    execute = False
    template = "ultraprompt"  # Default template
    
    for arg in sys.argv[2:]:
        if arg == "--execute" or arg == "-e":
            execute = True
        elif arg == "--advanced" or arg == "-a":
            template = "ultraprompt-advanced"
        elif not arg.startswith("-"):
            model = arg
    
    # Run ultraprompt
    result = ultraprompt_with_ollama(user_request, model, execute, template)
    
    if result:
        print("\n✅ Done!")
        if not execute:
            print("\n💡 Tip: Add --execute to run the expanded prompt automatically")
    else:
        print("\n❌ Failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
