import numpy as np
from typing import Any, List

class PerturbedModelWrapper:
    """
    Wraps the MirrorCore agent with a specific perturbation applied.
    """
    def __init__(self, core_agent, perturb_spec):
        self.core = core_agent
        self.spec = perturb_spec

    async def generate(self, prompt: str) -> str:
        """
        Generate text using the core agent, injecting perturbation info.
        """
        # In a real system, we would apply weights here.
        # For simulation, we note the perturbation in the system prompt.
        system_note = f"\n[Perturbation: rank={self.spec.rank}, scale={self.spec.scale}]"
        return await self.core.generate(prompt, system_prompt=system_note)

class LoomNode:
    """
    Federated Loom Node
    
    Responsibilities:
    * Generate low-rank perturbations.
    * Run inference-only forward passes.
    * Return scalar fitness + optional traces.
    """
    
    def __init__(self):
        pass

    def generate_low_rank(self, seed_A: int, seed_B: int, m: int, n: int, r: int):
        """
        Generate low-rank matrices A and B from seeds.
        """
        rng_A = np.random.default_rng(seed_A)
        A = rng_A.standard_normal((m, r))
        
        rng_B = np.random.default_rng(seed_B)
        B = rng_B.standard_normal((n, r))
        
        return A, B

    def apply_perturbation(self, model: Any, perturb_spec: Any) -> Any:
        """
        Apply perturbation to the model.
        """
        # Return a wrapper that simulates the perturbed model
        return PerturbedModelWrapper(model, perturb_spec)

    async def evaluate(self, model: Any, tasks: List[Any]) -> str:
        """
        Run inference on the model with the given tasks.
        """
        # For simplicity, just evaluate the first task
        if not tasks:
            return ""
            
        task = tasks[0]
        try:
            # Call generate on the perturbed model wrapper
            output = await model.generate(task)
            return output
        except Exception as e:
            return f"Evaluation failed: {e}"
