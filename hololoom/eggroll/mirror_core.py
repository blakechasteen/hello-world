import numpy as np
import asyncio
import os
from typing import Optional, Tuple, Any
from hololoom.awareness.llm_integration import create_llm, LLMProtocol

# --- Imports & Conditional Dependencies ---
try:
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hololoom.eggroll.architectures import get_model
    
    # Try importing PEFT optionally
    try:
        from peft import get_peft_model, LoraConfig, TaskType
        params_peft_available = True
    except ImportError:
        params_peft_available = False
        # print("[MirrorCore] Warning: PEFT not installed. LoRA features disabled.")
        
except ImportError:
    # If Torch/Transformers missing, we run in limited mode
    torch = None
    nn = None
    AutoModelForCausalLM = None
    get_model = None
    params_peft_available = False

class MirrorCoreAgent:
    """
    MirrorCore Agent (The 'Self')
    
    Responsibilities:
    * Wraps the LLM (Ollama or Local PyTorch).
    * Manages the 'Self-Model' state (LoRA Adapters).
    * Applies evolutionary updates to its own weights.
    """
    
    def __init__(self, model_id: str = "gpt2", model_type: str = "standard", **model_kwargs):
        """
        Args:
            model_id: HuggingFace ID for standard models.
            model_type: 'standard', 'trm', or 'large'.
            **model_kwargs: Arguments for the specific architecture (e.g. recur_depth for TRM).
        """
        self.model_id = model_id
        self.version = 0
        self.adapter_path = f"adapters/{self.model_id}/v{self.version}"
        self.model_type = model_type
        
        # Initialize LLM Connection (for external generation if needed)
        try:
            self.llm = create_llm("ollama")
            self.has_llm = True
        except ImportError:
            self.has_llm = False
            # print("Warning: LLM integration not available.")

        # --- Model Initialization ---
        self.use_peft = False
        self.peft_model = None
        self.tokenizer = None
        self.custom_model = None
        
        try:
            # Imports are now global
            print(f"[MirrorCore] Initializing Agent with type: {model_type}")

            if torch is None:
                 raise ImportError("PyTorch not found (Global import failed)")

            if model_type == "standard":
                if not params_peft_available:
                     raise ImportError("PEFT is required for 'standard' model type.")
                     
                # Standard HF + LoRA Path (Existing Logic)
                print(f"[MirrorCore] Loading base model: {model_id}...")
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

            elif model_type in ["trm", "large", "liquid", "spiking", "moe", "sdm"]:
                # New Architectures
                self.custom_model = get_model(model_type, **model_kwargs)
                print(f"[MirrorCore] Custom architecture {model_type} loaded. Parameters: {self.custom_model.get_trainable_parameters()}")
                
                self.use_peft = False 
                
                # Placeholder params for evolution loop compatibility
                self.dim = 256 
                self.rank = 16 
                self.adapter_A = np.random.randn(self.dim, self.rank) * 0.01
                self.adapter_B = np.random.randn(self.dim, self.rank) * 0.01

        except Exception as e:
            print(f"[MirrorCore] Initialization failed: {e}. Falling back to simulation.")
            import traceback
            traceback.print_exc()
            self.use_peft = False
            self.custom_model = None
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

    def train(self):
        """Standard PyTorch train mode"""
        if self.custom_model:
            self.custom_model.train()
            
    def eval(self):
        """Standard PyTorch eval mode"""
        if self.custom_model:
            self.custom_model.eval()

    def compute_loss(self, input_ids, labels):
        """Computes Causal Language Modeling loss."""
        # Ensure model is on correct device if needed
        # Forward pass
        logits = self.custom_model(input_ids)
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

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
