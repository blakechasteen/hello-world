#!/usr/bin/env python3
"""
Demo 1: Strange Loop about Strange Loops
The most meta demo - using a strange loop to explain strange loops!
"""

import sys
import os

# Add promptly to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'promptly'))

from execution_engine import ExecutionEngine, ExecutionConfig, ExecutionBackend
from recursive_loops import RecursiveEngine, LoopConfig, LoopType

print("\n" + "="*70)
print("DEMO: What is a strange loop?")
print("(Using a strange loop to explain strange loops - perfectly meta!)")
print("="*70)

# Setup Ollama
config = ExecutionConfig(
    backend=ExecutionBackend.OLLAMA,
    model="llama3.2:3b",
    temperature=0.7
)
engine_exec = ExecutionEngine(config)
executor = lambda prompt: engine_exec.execute_prompt(prompt, skill_name="demo").output

# Create recursive engine
engine = RecursiveEngine(executor)

# Configure Hofstadter loop
loop_config = LoopConfig(
    loop_type=LoopType.HOFSTADTER,
    max_iterations=4,
    enable_scratchpad=True
)

print("\nRunning 4 meta-levels of recursive thinking...")
print("(This will take ~30-40 seconds)")
print("\nProcessing...")

# Execute!
result = engine.execute_hofstadter_loop(
    task="What is a strange loop?",
    config=loop_config
)

# Show results
print("\n" + "="*70)
print("RESULTS:")
print("="*70)
print(result.to_report())

print("\n" + "="*70)
print("AMAZING: The tool explained itself by being itself!")
print("This is Hofstadter's GEB concept in action!")
print("="*70)
