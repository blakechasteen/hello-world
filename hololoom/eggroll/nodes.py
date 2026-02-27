import numpy as np
import torch
from typing import Any, List, Dict, Tuple, Optional
from hololoom.eggroll.mirror_core import MirrorCoreAgent
from hololoom.warp.eggroll_warp import SpectralScorer

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
    * Holds a local copy of the MirrorCoreAgent.
    * Executes training/evaluation steps.
    * Computes architectural metrics (Sparsity, Spectral Radius).
    """
    
    def __init__(self, model_type: str = "standard", **model_kwargs):
        # distributed_backend passes worker_id, but MirrorCore/Architectures might not accept it.
        self.worker_id = model_kwargs.pop("worker_id", -1)
        
        print(f"[LoomNode {self.worker_id}] Initializing local model ({model_type})...")
        try:
            self.agent = MirrorCoreAgent(model_id=f"worker_{self.worker_id}", model_type=model_type, **model_kwargs)
        except Exception as e:
            print(f"[LoomNode {self.worker_id}] Init failed: {e}")
            self.agent = None
            
        self.model_type = model_type

    def step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> Tuple[float, Dict[str, float], str]:
        """
        Execute a single evolutionary step.
        Returns: (Fitness, Metrics, OutputSample)
        """
        if self.agent is None or not hasattr(self.agent, "custom_model"):
            return 0.0, {}, "[Agent Build Failed]"
            
        # 1. Forward Pass (Fitness = Negative Loss)
        # In a real scenario, we'd compute loss. Here we simulate or use a dummy forward.
        try:
            # Check if model supports standard forward
            if self.agent.custom_model and hasattr(self.agent.custom_model, "forward"):
                # Move inputs to device (if applicable)
                # input_ids = input_ids.to(self.agent.device)
                with torch.no_grad():
                    logits = self.agent.custom_model(input_ids)
                    # Simple pseudo-loss: maximize probability of something?
                    # For Evo, we often use task-based reward. 
                    # Let's verify it runs and return a dummy fitness based on logits variance (entropy)
                    fitness = float(torch.std(logits).item())
                    
                # Sample output for qualitative review
                output_sample = ""
                try:
                    gen_ids = self.agent.custom_model.generate(input_ids[:, :2], max_new_tokens=10)
                    output_sample = str(gen_ids[0].tolist())
                except Exception:
                    output_sample = "[Generation Failed]"
            else:
                fitness = 0.5
                output_sample = "[Simulated]"
        except Exception as e:
            print(f"Step failed: {e}")
            fitness = 0.0
            output_sample = f"Error: {e}"

        # 2. Compute Metrics
        metrics = self._compute_metrics()
        
        return fitness, metrics, output_sample

    def _compute_metrics(self) -> Dict[str, float]:
        """Analyze the local model's architecture."""
        stats = {}
        if self.agent is None or self.agent.custom_model is None:
            return stats
            
        model = self.agent.custom_model
        
        # Sparsity (for any model, check weights)
        # We grab the first available linear layer we find
        first_layer_weights = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                first_layer_weights = module.weight.detach().cpu().numpy()
                break
                
        if first_layer_weights is not None:
            stats["sparsity"] = SpectralScorer.analyze_sparsity(first_layer_weights, threshold=0.01)
            stats["spectral_radius"] = SpectralScorer.compute_spectral_radius(first_layer_weights)
        
        # Energy Efficiency (Mock based on Arch)
        if self.model_type == "spiking":
            # Count spikes? (Simulated)
            stats["energy"] = 0.1 # Very low energy
            stats["spike_count"] = 1240
        elif self.model_type == "liquid":
            stats["energy"] = 0.4 # Medium
        elif self.model_type == "moe":
            stats["energy"] = 0.3 # Sparse activation
            stats["expert_utilization"] = 0.8 # Balanced
        elif self.model_type == "large":
            stats["energy"] = 0.9 # High
            
        return stats

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
        return PerturbedModelWrapper(model, perturb_spec)

    async def evaluate(self, model: Any, tasks: List[Any]) -> str:
        """
        Run inference on the model with the given tasks.
        """
        if not tasks:
            return ""
        task = tasks[0]
        try:
            output = await model.generate(task)
            return output
        except Exception as e:
            return f"Evaluation failed: {e}"
