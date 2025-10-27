#!/usr/bin/env python3
"""
Debug test for Ollama
"""
import subprocess
import sys
import os

ollama_path = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Ollama', 'ollama.exe')
model = "llama3.2:3b"
prompt = "test prompt"

print(f"Ollama path: {ollama_path}")
print(f"Path exists: {os.path.exists(ollama_path)}")
print(f"Model: {model}")
print(f"Prompt: {prompt}")
print()

# Try method 1: shell=True with string
cmd_str = f'"{ollama_path}" run {model} "{prompt}"'
print(f"Command string: {cmd_str}")
print("Running with shell=True...")

try:
    result = subprocess.run(
        cmd_str,
        capture_output=True,
        text=True,
        timeout=30,
        shell=True
    )
    
    print(f"Return code: {result.returncode}")
    print(f"STDOUT length: {len(result.stdout)}")
    print(f"STDERR length: {len(result.stderr)}")
    
    if result.stdout:
        print(f"\nSTDOUT:\n{result.stdout}")
    
    if result.stderr:
        print(f"\nSTDERR:\n{result.stderr}")
        
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
