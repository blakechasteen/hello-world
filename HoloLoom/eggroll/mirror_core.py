import numpy as np
import asyncio
import os
from typing import Optional, Tuple, Any
from HoloLoom.awareness.llm_integration import create_llm, LLMProtocol

class MirrorCoreAgent:
    """
    MirrorCore Agent (The 'Self')
    
    Responsibilities:
    * Wraps the LLM (Ollama or Local PyTorch).
    * Manages the 'Self-Model' state (LoRA Adapters).
    * Applies evolutionary updates to its own weights.
    """
    
    def __init__(self, model_id: str = "gpt2"): # Default to small model for demo
        self.model_id = model_id
        self.version = 0
        self.adapter_path = f"adapters/{self.model_id}/v{self.version}"
        
        # Initialize LLM Connection (for generation)
        try:
            self.llm = create_llm("ollama")
            self.has_llm = True
        except ImportError:
            self.has_llm = False
            print("Warning: LLM integration not available.")

        # --- Real LoRA Integration (PEFT) ---
        self.use_peft = False
        self.peft_model = None
        self.tokenizer = None
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import get_peft_model, LoraConfig, TaskType
            
            print(f"[MirrorCore] Loading base model: {model_id}...")
            # Load small base model for demonstration of real weight updates
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            
            # Configure LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=16, 
                lora_alpha=32, 
                lora_dropout=0.1
            )
            
            self.peft_model = get_peft_model(base_model, peft_config)
            self.use_peft = True
            print(f"[MirrorCore] PEFT LoRA initialized. Trainable params: {self.peft_model.print_trainable_parameters()}")
            
            # Extract dimensions from the first LoRA layer found
            self.target_module = None
            for name, module in self.peft_model.named_modules():
                if "lora_A" in name:
                    self.target_module = name
                    self.dim = module.default.weight.shape[1] # In_features
                    self.rank = module.default.weight.shape[0] # Rank
                    print(f"[MirrorCore] Targeting LoRA module: {name} (Dim={self.dim}, Rank={self.rank})")
                    break
            
            if not self.target_module:
                raise ValueError("No LoRA modules found.")
                
        except Exception as e:
            print(f"[MirrorCore] PEFT initialization failed: {e}. Falling back to simulation.")
            self.use_peft = False
            self.rank = 16
            self.dim = 512
            self.adapter_A = np.random.randn(self.dim, self.rank) * 0.01
            self.adapter_B = np.random.randn(self.dim, self.rank) * 0.01

    @property
    def shape(self):
        """Weight matrix shape."""
        return (self.dim, self.dim)

    def update_weights(self, update_matrix: np.ndarray):
        """
        Integrate updates into active model weights using Projected Gradient Descent.
        """
        if self.use_peft:
            import torch
            target_layer = None
            # Find the parent module containing lora_A
            # self.target_module is like "base_model.model.decoder.layers.0.self_attn.q_proj.lora_A"
            # We need to access the module object
            
            # Simplification: Just iterate to find it again
            for name, module in self.peft_model.named_modules():
                if name == self.target_module:
                    target_layer = module
                    break
            
            if target_layer:
                with torch.no_grad():
                    lora_A = target_layer.default.weight # (rank, dim)
                    
                    # SIMPLIFICATION: Inject noise/update into weights
                    # In real run: delta_A = update_matrix @ B_inv ...
                    # Here we assume update_matrix is (dim, dim) and we project randomly
                    # just to prove we can modify the weights.
                    
                    # Slice update matrix to match (rank, dim)
                    update_slice = update_matrix[:self.rank, :self.dim]
                    noise = torch.tensor(update_slice, dtype=lora_A.dtype)
                    
                    lora_A.add_(noise * 0.01)
                    
                    print(f"[MirrorCore] Applied update to PEFT weights: {self.target_module}")
                    
                    # Save Adapter
                    self.version += 1
                    save_path = f"adapters/{self.model_id}/v{self.version}"
                    self.peft_model.save_pretrained(save_path)
                    print(f"[MirrorCore] Saved adapter v{self.version} to {save_path}")
                    
        else:
            # Simulation Fallback
            delta_A = update_matrix @ self.adapter_B
            learning_rate = 0.1
            self.adapter_A += learning_rate * delta_A
            
            self.version += 1
            print(f"[MirrorCore] Projected Update v{self.version} (Simulated)")

    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the wrapped LLM.
        """
        if not self.has_llm:
            return "LLM not available - MirrorCore simulation"
            
        system_prompt = kwargs.get('system_prompt', "You are a helpful AI.")
        system_prompt += f"\n[System Note: Running with MirrorCore Adapter v{self.version}]"
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs
            )
            return response.content
        except Exception as e:
            return f"Generation failed: {e}"
