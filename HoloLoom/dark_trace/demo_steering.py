from HoloLoom.eggroll.architectures import get_model
from HoloLoom.dark_trace.probe import MindProbe
import torch
import torch.nn.functional as F

def demo_steering():
    print("🌑 DARK TRACE: Consistency Control via Steering Vectors...")
    
    # 1. Init Model & Probe
    d_model = 128
    model = get_model("trm", vocab_size=1000, d_model=d_model, recur_depth=4)
    probe = MindProbe(layer_name="recursive_block", input_dim=d_model)
    probe.attach(model)
    
    # 2. Train SAE briefly to get meaningful features
    print("🧠 Warming up SAE...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(50):
        inputs = torch.randint(0, 1000, (4, 16))
        _ = model(inputs)
        probe.train_step()
    print("✅ SAE Trained.")

    # 3. Identify a 'Feature'
    # In a real scenario, we'd label these. Here we just pick a random active one.
    dummy_in = torch.randint(0, 1000, (1, 5))
    _ = model(dummy_in)
    concepts = probe.read_mind()
    target_feature = concepts[0]['feature'] # Pick strongest feature
    print(f"🎯 Targeted Feature for Steering: {target_feature}")
    
    # 4. Measure Baseline Consistency
    # We'll run the same input multiple times and measure variance in output LOGITS (approximation).
    # Since the model is deterministic without dropout, variance is 0.
    # To simulate "inconsistency", we'll inject noise into the model weights momentarily? 
    # Or better: We demonstrate that steering CHANGES the output consistently.
    
    test_prompt = torch.randint(0, 1000, (1, 10))
    
    print("\n--- Baseline Run (No Steering) ---")
    out_base = model(test_prompt)
    print(f"Base Output Mean: {out_base.mean().item():.4f}")
    
    # 5. Apply Steering
    print(f"\n--- 🔧 Applying Steering (Feature {target_feature} @ +10.0) ---")
    probe.set_steering(target_feature, 10.0)
    
    out_steered = model(test_prompt)
    print(f"Steered Output Mean: {out_steered.mean().item():.4f}")
    
    diff = (out_steered - out_base).abs().mean().item()
    print(f"Δ Change due to Steering: {diff:.4f}")
    
    if diff > 0.01:
        print("✅ Steering successfully altered model behavior.")
    else:
        print("❌ Steering had negligible effect.")
        
    # 6. "Set Number of Tokens" Interpretation
    # Let's clean up and demonstrate a "Thinking Budget"
    probe.clear_steering()
    print("\n--- ⏳ Interpreting 'Set Number of Tokens' as Thinking Budget ---")
    # This loop shows how we might iterate 10 times (10 'tokens' of thought)
    # refining the state before outputting.
    
    state = torch.zeros(1, d_model) # Initial thought
    thinking_budget = 10
    
    print(f"Allocating {thinking_budget} steps for recurrence...")
    # In the TRM, we already have `recur_depth`. We can dynamically adjust it.
    model.recur_depth = thinking_budget
    out_thoughtful = model(test_prompt)
    print("✅ Inference completed with extended recurrence budget.")

if __name__ == "__main__":
    demo_steering()
