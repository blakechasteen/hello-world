from hololoom.eggroll.architectures import get_model
from hololoom.dark_trace.probe import MindProbe
import torch
import time

def demo_dark_trace():
    print("🌑 DARK TRACE: Semantic Reverse Engineering Initialization...")
    
    # 1. Initialize Subject Model
    # We use our custom TRM (Tiny Recursive Model) as the subject
    d_model = 128
    model = get_model("trm", vocab_size=1000, d_model=d_model, recur_depth=4)
    print("🧠 Subject Model (TRM) Loaded.")
    
    # 2. Attach Probe
    # We want to probe the output of the recursive block.
    # In our TRM definition: self.recursive_block = nn.TransformerEncoder(...)
    # The hook inside MindProbe looks for layer names.
    probe = MindProbe(layer_name="recursive_block", input_dim=d_model)
    probe.attach(model)
    
    # 3. Simulate "Thinking" (Forward Passes)
    print("\n💭 Simulating Model Thought Process...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for step in range(100):
        # Random input simulation
        inputs = torch.randint(0, 1000, (4, 16)) # Batch 4, Seq 16
        
        # Forward pass (Probe captures automatically)
        _ = model(inputs)
        
        # 4. Train the SAE on the captured thoughts
        metrics = probe.train_step(batch_size=32)
        
        if step % 20 == 0 and metrics:
            print(f"   Step {step}: SAE Training Loss = {metrics['loss']:.4f} (Sparsity: {metrics['sparsity_loss']:.4f})")
            
    # 5. Peer into the Mind
    print("\n👁️ PEERING INTO THE MIND (Analysis of last state)...")
    
    # Run one specific input
    test_input = torch.randint(0, 1000, (1, 8))
    _ = model(test_input)
    
    concepts = probe.read_mind()
    
    print(f"Found {len(concepts)} active sparse features (concepts) in the last thought vector:")
    for concept in concepts[:10]: # Top 10
        print(f"  Feature {concept['feature']:04d}: Strength {concept['strength']:.4f}")
        
    print("\n✅ Dark Trace Probe Successful.")

if __name__ == "__main__":
    demo_dark_trace()
