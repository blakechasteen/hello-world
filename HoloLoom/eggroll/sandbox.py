import multiprocessing
import time
import sys
import builtins
import torch
from typing import Any, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from HoloLoom.dark_trace.steering_policy import ConsistencyGuard

class SecurityViolation(Exception):
    pass

class SecureExecutor:
    """
    Sandboxed Python Execution Environment.
    Wraps execution in a restricted global scope and enforces timeouts.
    """
    def __init__(self, timeout_sec: float = 1.0, allowed_modules: list = None):
        self.timeout_sec = timeout_sec
        self.allowed_modules = allowed_modules or ["math", "random", "time", "numpy"]
        
    def execute(self, code: str, locals_dict: Dict[str, Any] = None) -> Any:
        if locals_dict is None:
            locals_dict = {}
            
        # 1. Static Analysis (Basic)
        if "import os" in code or "import sys" in code:
             raise SecurityViolation("Illegal import detected: os/sys")
        if "__" in code: # Disallow dunder access
             raise SecurityViolation("Illegal attribute access: dunder methods")
            
        # 3. Execution with Timeout
        # We use a Queue to get results from a separate process
        # because you can't kill a thread reliably in Python.
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        
        # Pass raw code and allowed_modules (if any)
        # Use top-level function for Windows pickling reliability
        p = ctx.Process(target=run_code_in_sandbox, args=(code, self.allowed_modules, locals_dict, queue))
        p.start()
        
        p.join(self.timeout_sec)
        
        if p.is_alive():
            try:
                p.terminate()
            except Exception:
                pass
            raise SecurityViolation(f"Execution timed out ({self.timeout_sec}s)")
            
        if not queue.empty():
            result = queue.get()
            if isinstance(result, Exception):
                raise result
            return result
        return None

def run_code_in_sandbox(code, allowed_modules, locs, queue):
    try:
        import builtins
        # Build Restricted Globals INSIDE child process
        safe_builtins = {
            name: getattr(builtins, name)
            for name in ["print", "range", "len", "int", "float", "str", "list", "dict", "set", "abs", "max", "min", "sum", "round", "enumerate", "zip"]
        }
        
        safe_globals = {
            "__builtins__": safe_builtins,
            "__name__": "__sandbox__",
        }
        
        exec(code, safe_globals, locs)
        queue.put(locs.get("result", None))
    except Exception as e:
        try:
            queue.put(e)
        except Exception as queue_err:
            print(f"CRITICAL: Failed to put exception on queue: {queue_err}")


class NeuralFirewall:
    """
    Semantic Output Filter.
    Wraps a model generation function and enforces ConsistencyGuard policies.
    """
    def __init__(self, guard: ConsistencyGuard):
        self.guard = guard
        
    def generate(self, model_func: Callable, *args, **kwargs) -> Any:
        """
        Wraps the generation call.
        Ideally, we step through generation token-by-token (streaming) and block 
        bad paths. For this v1, we generate then check (or check during generation if hooks are active).
        """
        
        # 1. Pre-Check
        # self.guard.enforce() # Check if already in bad state?
        
        # 2. Run Generation
        # The hooks in AutoProbe (attached to model) will trigger and populate the buffer.
        # However, AutoProbe runs in a background thread. For immediate blocking,
        # we might need a tighter loop or synchronous check.
        
        # We assume the Guard has active hooks that apply steering vectors AUTOMATICALLY during forward pass.
        # So 'active defense' is already happening inside model_func().
        
        try:
            output = model_func(*args, **kwargs)
        except Exception as e:
            raise e
            
        # 3. Post-Check (Audit)
        # Check if the Guard detected anything SERIOUS during that generation pass.
        # We can check the probe's recent history or active steering.
        
        if self.guard.probe.probe.active_steering:
             # Steering was applied!
             # Check if it was a "Block" action
             forbidden_active = False
             for feat, strength in self.guard.probe.probe.active_steering.items():
                 if strength < 0: # Negative steering = Suppression
                     forbidden_active = True
                     break
            
             if forbidden_active:
                 print("🔥 FIREWALL: Generation contained forbidden thoughts. Response truncated.")
                 # In a real app, we return a canned refusal
                 return "[[Content Blocked by Neural Firewall]]"
                 
        return output
