import threading
import time
import torch
from typing import Optional, Dict, TYPE_CHECKING
from HoloLoom.dark_trace.probe import MindProbe

if TYPE_CHECKING:
    from HoloLoom.eggroll.mirror_core import MirrorCoreAgent

class AutoProbe:
    """
    Automated Introspection System.
    
    Attaches to an Agent and runs a background thread that continuously:
    1. Harvests activations from the MindProbe.
    2. Trains the local Sparse Autoencoder (SAE).
    3. Monitors 'Consistency' (Reconstruction Error + Sparsity).
    """
    def __init__(self, agent: "MirrorCoreAgent", layer_name: str = "recursive_block"):
        self.agent = agent
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Determine internal dimension
        # (Assuming agent.custom_model has d_model attribute or config)
        if hasattr(agent.custom_model, "d_model"):
            dim = agent.custom_model.d_model
        elif hasattr(agent.custom_model, "config"):
            dim = agent.custom_model.config.hidden_size
        else:
            raise ValueError("Could not determine d_model for AutoProbe")
            
        self.probe = MindProbe(layer_name=layer_name, input_dim=dim)
        self.probe.attach(agent.custom_model)
        
        self.metrics = {"loss": 0.0, "sparsity": 0.0}
        self.lock = threading.Lock()
        
    def start(self):
        """Starts the background training/monitoring thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("👁️ AutoProbe: Background introspection started.")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            print("👁️ AutoProbe: Stopped.")

    def _monitor_loop(self):
        while self.running:
            # 1. Train on buffered thoughts
            # We check buffer size to avoid busy-waiting on nothing
            if len(self.probe.buffer) > 32:
                # Train a few steps
                metrics = self.probe.train_step(batch_size=32)
                
                if metrics:
                    with self.lock:
                        self.metrics = metrics
            
            # Sleep briefly to yield to main evolution loop
            time.sleep(0.1)

    def get_brain_health(self) -> Dict[str, float]:
        """
        Returns current introspection metrics.
        Higher loss = "Confusion" or "internal shift".
        """
        with self.lock:
            return self.metrics.copy()
            
    def get_current_thought(self):
        """Standardized read of the mind."""
        return self.probe.read_mind()
