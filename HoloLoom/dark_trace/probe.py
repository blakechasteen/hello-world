import torch
import torch.nn as nn
from typing import Optional, List, Dict, Callable
from collections import deque
from HoloLoom.dark_trace.sae import SparseAutoEncoder, DarkSaeTrainer

class MindProbe:
    """
    Hooks into a model's internal layers to extract activations and apply SAE analysis.
    The 'Periscope' into the LLM's mind.
    """
    def __init__(self, layer_name: str, input_dim: int):
        self.layer_name = layer_name
        self.input_dim = input_dim
        
        # Buffer for activations used for training the SAE
        self.buffer = deque(maxlen=1000) 
        
        # The SAE associated with this probe
        self.sae = SparseAutoEncoder(input_dim)
        self.trainer = DarkSaeTrainer(self.sae)
        
        self.hook_handle = None
        self.latest_activations = None
        
        # Steering: {feature_idx: strength}
        self.active_steering = {}
        
    def attach(self, model: nn.Module):
        """
        Locates the sub-module by name and registers a forward hook.
        """
        target_module = None
        for name, module in model.named_modules():
            if name == self.layer_name:
                target_module = module
                break
        
        if target_module is None:
            # Try approximate matching if exact fail
            for name, module in model.named_modules():
                if name.endswith(self.layer_name):
                    target_module = module
                    break
        
        if target_module is None:
            raise ValueError(f"Layer {self.layer_name} not found in model.")
            
        self.hook_handle = target_module.register_forward_hook(self._hook_fn)
        print(f"👁️ DarkTrace Probe attached to: {self.layer_name}")
        
    def set_steering(self, feature_idx: int, strength: float):
        """
        Sets a 'Control Vector' based on a specific SAE feature.
        This forces the model to 'think' about this concept.
        """
        self.active_steering[feature_idx] = strength
        print(f"🔧 Steering set: Feature {feature_idx} -> {strength}")
        
    def clear_steering(self):
        self.active_steering = {}

    def _hook_fn(self, module, input, output):
        """
        Capture activations AND apply steering.
        """
        # Handle tuple output (common in Transformers)
        was_tuple = False
        if isinstance(output, tuple):
            was_tuple = True
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            
        # 1. READ: Capture for analysis/training
        with torch.no_grad():
            self.latest_activations = hidden_states.detach().clone()
            
            # Store flattened activations for training
            flat_acts = self.latest_activations.view(-1, self.input_dim)
            if len(self.buffer) < self.buffer.maxlen:
                self.buffer.append(flat_acts)

        # 2. WRITE: Apply Steering (Mind Control)
        if self.active_steering:
            # We need to compute the direction vector for each active feature
            # The decoder weights of the SAE represent the 'direction' of that feature in model space.
            # decoder.weight is [Input_Dim, Latent_Dim] (or [Latent, Input] depending on impl)
            # In our SAE: decoder is Linear(Latent, Input), so weight is [Input, Latent]
            
            with torch.no_grad():
                steering_delta = torch.zeros_like(hidden_states)
                
                for feat_idx, strength in self.active_steering.items():
                    # Get the direction vector for this feature (column of decoder weight)
                    # weight shape: [out_features, in_features] -> [input_dim, latent_dim]
                    feature_vector = self.sae.decoder.weight[:, feat_idx] 
                    
                    # Add to all tokens? Or just the last?
                    # "Set number of tokens" -> We could limit this to the first K tokens or similar.
                    # For now, apply to all positions (Global Steering).
                    steering_delta += (feature_vector * strength).view(1, 1, -1)
                
                # Apply the delta
                hidden_states = hidden_states + steering_delta
        
        # Re-pack if necessary
        if was_tuple:
            return (hidden_states,) + rest
        return hidden_states

    def train_step(self, batch_size: int = 32):
        """
        Trains the SAE on buffered activations.
        """
        if len(self.buffer) < 1:
            return None
            
        # Collate from buffer
        data = torch.cat(list(self.buffer), dim=0) # [Total_Tokens, Dim]
        
        # Sample batch
        if data.size(0) > batch_size:
            indices = torch.randperm(data.size(0))[:batch_size]
            batch = data[indices]
        else:
            batch = data
            
        metrics = self.trainer.step(batch)
        
        # Clear buffer periodically to allow new distribution
        if len(self.buffer) >= self.buffer.maxlen:
            self.buffer.clear() # Primitive policy
            
        return metrics

    def read_mind(self) -> List[Dict[str, float]]:
        """
        Interprets the LATEST captured activation using the trained SAE.
        Returns a list of active features and their strengths for the last token.
        """
        if self.latest_activations is None:
            return []
            
        # Look at the last token of the first sequence in batch
        # shape: [Batch, Seq, Dim] -> [Dim]
        last_thought_vector = self.latest_activations[0, -1, :] 
        
        active_feats = self.sae.get_active_features(last_thought_vector.unsqueeze(0), threshold=0.1)
        
        # Format for display
        # In a real system, we would map feature IDs to "autolabel" descriptions.
        results = []
        for feat_id, val in active_feats.items():
            results.append({"feature": feat_id, "strength": val})
            
        # Sort by strength
        results.sort(key=lambda x: x["strength"], reverse=True)
        return results
