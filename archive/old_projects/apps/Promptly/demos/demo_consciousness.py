#!/usr/bin/env python3
"""
Demo: Is consciousness a strange loop?
Hofstadter's core thesis from "I Am a Strange Loop"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'promptly'))

from execution_engine import ExecutionEngine, ExecutionConfig, ExecutionBackend
from recursive_loops import RecursiveEngine, LoopConfig, LoopType

print("\n" + "="*70)
print("DEMO: Is consciousness a strange loop?")
print("(Hofstadter's core thesis from 'I Am a Strange Loop')")
print("="*70)

config = ExecutionConfig(backend=ExecutionBackend.OLLAMA, model="llama3.2:3b", temperature=0.7)
engine_exec = ExecutionEngine(config)
executor = lambda prompt: engine_exec.execute_prompt(prompt, skill_name="demo").output

engine = RecursiveEngine(executor)
loop_config = LoopConfig(loop_type=LoopType.HOFSTADTER, max_iterations=5, enable_scratchpad=True)

print("\nRunning 5 meta-levels... (40-50 seconds)\n")

result = engine.execute_hofstadter_loop(
    task="Is consciousness a strange loop?",
    config=loop_config
)

print(result.to_report())
print("\n" + "="*70)
