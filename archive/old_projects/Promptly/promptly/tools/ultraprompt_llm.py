#!/usr/bin/env python3
"""
UltraPrompt with LLM CLI
Combines Promptly prompt management with the llm CLI tool
"""

import sys
import subprocess
import json
from pathlib import Path

# Add current directory to path to import promptly
sys.path.insert(0, str(Path(__file__).parent))
from promptly import Promptly


def ultraprompt_with_llm(user_request: str, model: str = None):
    """
    Takes a simple user request and uses Promptly + llm CLI to:
    1. Expand it into a comprehensive prompt (ultraprompt)
    2. Execute the expanded prompt with llm CLI
    """
    
    # Initialize Promptly
    promptly = Promptly()
    
    # Get the ultraprompt template
    try:
        prompt_data = promptly.get('ultraprompt')
        if not prompt_data:
            print("Error: 'ultraprompt' template not found in Promptly")
            print("Please add it first with:")
            print('  python promptly.py add ultraprompt "Your ultraprompt template here"')
            return None
    except Exception as e:
        print(f"Error getting ultraprompt: {e}")
        return None
    
    # Format the ultraprompt with the user's request
    ultraprompt_content = prompt_data['content'].format(request=user_request)
    
    print("=" * 60)
    print("ULTRAPROMPT - Expanding your request...")
    print("=" * 60)
    print(f"\nOriginal request: {user_request}")
    print(f"\nUsing template: {prompt_data['name']} (v{prompt_data['version']})")
    print("-" * 60)
    
    # Call llm CLI to expand the prompt
    try:
        cmd = ["llm", ultraprompt_content]
        if model:
            cmd = ["llm", "-m", model, ultraprompt_content]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=False
        )
        
        if result.returncode != 0:
            print(f"Error calling llm: {result.stderr}")
            return None
        
        expanded_prompt = result.stdout.strip()
        
        print("\n✓ Expanded prompt:")
        print("=" * 60)
        print(expanded_prompt)
        print("=" * 60)
        
        return expanded_prompt
        
    except FileNotFoundError:
        print("Error: 'llm' command not found. Make sure it's installed and in your PATH.")
        return None
    except Exception as e:
        print(f"Error executing llm: {e}")
        return None


def simple_llm_call(prompt: str, model: str = None):
    """Simple wrapper to call llm CLI with a prompt"""
    try:
        cmd = ["llm", prompt]
        if model:
            cmd = ["llm", "-m", model, prompt]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=False
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"Error: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("UltraPrompt with LLM CLI")
        print("\nUsage:")
        print("  python ultraprompt_llm.py <your-request> [model]")
        print("\nExamples:")
        print('  python ultraprompt_llm.py "write a Python function to sort a list"')
        print('  python ultraprompt_llm.py "explain quantum computing" gpt-4')
        print("\nThis will:")
        print("  1. Use Promptly's 'ultraprompt' template to expand your request")
        print("  2. Call the llm CLI tool with the expanded prompt")
        sys.exit(1)
    
    user_request = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Expand the prompt using ultraprompt
    expanded = ultraprompt_with_llm(user_request, model)
    
    if expanded:
        print("\n" + "=" * 60)
        print("Would you like to execute this expanded prompt? (y/n)")
        print("=" * 60)
        
        # For non-interactive use, you can auto-execute:
        # response = simple_llm_call(expanded, model)
        # if response:
        #     print("\n=== FINAL RESPONSE ===")
        #     print(response)
