import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from hololoom.eggroll.mirror_core import MirrorCoreAgent
from hololoom.eggroll.architectures import get_model, LiquidStateMachine, NeuromorphicNet

def run_test(name, model_type, **kwargs):
    print(f"\n⚡ --- Testing Architecture: {name} ({model_type}) ---")
    try:
        agent = MirrorCoreAgent(model_type=model_type, **kwargs)
        
        if agent.custom_model:
            print(f"✅ Loaded: {type(agent.custom_model).__name__}")
            
            # Param Count
            params = agent.custom_model.get_trainable_parameters()
            print(f"📊 Parameters: {params:,}")
            
            # Forward Pass Test
            dummy_input = torch.randint(0, 1000, (1, 16))
            with torch.no_grad():
                logits = agent.custom_model(dummy_input)
                print(f"🔄 Forward Pass Output: {logits.shape} (Matches [Batch, Seq, Vocab]?)")
                assert logits.shape == (1, 16, 50257) or logits.shape == (1, 16, agent.custom_model.head.out_features if hasattr(agent.custom_model, 'head') else 50257)

            # Generation Test
            print("📝 Attempting Generation...")
            gen_out = agent.custom_model.generate(dummy_input, max_new_tokens=5)
            print(f"✅ Generated Sequence Length: {gen_out.shape[1]}")
            
        else:
            print(f"❌ Failed to load {name} (Fallback to simulation?)")
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🚀 STARTING ARCHITECTURE LABORATORY 🚀")
    
    # 1. TRM (Recursive)
    run_test("Tiny Recursive", "trm", recur_depth=6, d_model=128)
    
    # 2. Liquid (Continuous Time)
    run_test("Liquid Neural Net", "liquid", reservoir_size=128)
    
    # 3. Spiking (Neuromorphic)
    run_test("Spiking Neuromorphic", "spiking", n_layers=4)
    
    # 4. Large (Mock)
    # 4. Large (Mock)
    run_test("Large Foundation", "large", model_name="gpt2-mock")

    # 5. Sparse MoE
    run_test("Sparse MoE", "moe", num_experts=8, experts_per_token=2)
    
    # 6. Sparse Distributed Memory (Kanerva)
    run_test("Kanerva SDM", "sdm", num_locations=512)

if __name__ == "__main__":
    main()
