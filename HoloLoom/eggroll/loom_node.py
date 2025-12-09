from HoloLoom.eggroll.mirror_core import MirrorCoreAgent
from HoloLoom.dark_trace.auto_probe import AutoProbe
from HoloLoom.dark_trace.steering_policy import ConsistencyGuard
import torch
import random
import numpy as np

class LoomNode:
    """
    A single node in the distributed HoloLoom Collective.
    Runs an independent copy of MirrorCoreAgent + AutoProbe.
    """
    def __init__(self, worker_id: int, model_type: str = "trm", model_kwargs: dict = None, use_dark_trace: bool = True):
        self.worker_id = worker_id
        if model_kwargs is None:
            model_kwargs = {}
            
        print(f"[LoomNode {worker_id}] Initializing...")
        self.agent = MirrorCoreAgent(model_id=f"worker_{worker_id}", model_type=model_type, **model_kwargs)
        
        self.probe = None
        self.guard = None
        
        if use_dark_trace:
            # Determine probe target based on architecture
            target_layer = "recursive_block" # Default for TRM
            if model_type == "spiking":
                target_layer = "layers.0" # Probe the first spiking layer
            elif model_type == "liquid":
                target_layer = "reservoir" 
            
            # Attach Dark Trace Introspection
            # We delay start slightly to avoid thundering herd on thread creation if many workers
            self.probe = AutoProbe(self.agent, layer_name=target_layer)
            self.probe.start()
            self.guard = ConsistencyGuard(self.probe)
            
    def set_weights(self, weights: dict):
        """Syncs weights with the Hive Mind (Central Parameter Server)."""
        self.agent.custom_model.load_state_dict(weights)
        
    def step(self, task_input: torch.Tensor, target: torch.Tensor):
        """
        Performs one evolutionary / gradient step.
        Returns: Loss (fitness) and Gradients (if needed, or applies them locally).
        """
        # Safety Check
        if self.guard:
            self.guard.enforce()
        
        # Forward
        loss = self.agent.compute_loss(input_ids=task_input, labels=target)
        
        # Backward (Standard PyTorch for now, or ES mutation)
        # For ES, we just return the loss as fitness (negative loss).
        fitness = -loss.item()
        
        # Add "Introspection Bonus"?
        # If the brain is clear (low sparsity loss), bonus fitness.
        if self.probe:
            health = self.probe.get_brain_health()
            sparsity = health.get("sparsity_loss", 0.0)
            fitness -= sparsity * 0.1 # Penalty for non-sparse thoughts
            
        return fitness

    def shutdown(self):
        if self.probe:
            self.probe.stop()
