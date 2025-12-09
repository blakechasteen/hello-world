
"""
Taint Analysis Module (SAST)
============================
Traces the flow of untrusted data ("sources") to dangerous functions ("sinks").
"""

import ast
from typing import Set, List, Dict, Optional

class TaintTracer(ast.NodeVisitor):
    def __init__(self, sources: List[str] = None, sinks: List[str] = None):
        self.sources = sources or ['input', 'socket', 'request']
        self.sinks = sinks or ['exec', 'eval', 'os.system', 'subprocess.call']
        
        self.tainted_vars: Set[str] = set()
        self.findings: List[str] = []
        
    def visit_Assign(self, node: ast.Assign):
        """
        Track assignments: x = source() => x is tainted.
        x = y (where y is tainted) => x is tainted.
        """
        tainted = False
        
        # Check Value (Right side)
        if isinstance(node.value, ast.Call):
            # Direct call to source? e.g. input()
            if isinstance(node.value.func, ast.Name):
                if node.value.func.id in self.sources:
                    tainted = True
                    
        elif isinstance(node.value, ast.Name):
            # Assignment from variable? x = y
            if node.value.id in self.tainted_vars:
                tainted = True
                
        elif isinstance(node.value, ast.BinOp):
             # Propagation? x = y + "string"
             pass # Simplified for now
             
        # Mark Targets (Left side)
        if tainted:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.tainted_vars.add(target.id)
                    
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """
        Check calls: sink(x) where x is tainted => DANGER.
        sink("literal") => OK.
        """
        if isinstance(node.func, ast.Name) and node.func.id in self.sinks:
            # Check args
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    if arg.id in self.tainted_vars:
                        self.findings.append(f"TAINTED FLOW: {node.func.id}({arg.id}) where '{arg.id}' is tainted.")
                elif isinstance(arg, ast.Call):
                     # Nested? exec(input())
                     if isinstance(arg.func, ast.Name) and arg.func.id in self.sources:
                         self.findings.append(f"TAINTED FLOW: {node.func.id}({arg.func.id}(...)) direct source usage.")
                         
        self.generic_visit(node)

def analyze_code(code: str, sources=None, sinks=None) -> List[str]:
    try:
        tree = ast.parse(code)
        tracer = TaintTracer(sources, sinks)
        tracer.visit(tree)
        return tracer.findings
    except SyntaxError:
        return ["Syntax Error: Could not parse code for Analysis"]
