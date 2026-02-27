import time
import torch
import torch.nn as nn
from hololoom.eggroll.mirror_core import MirrorCoreAgent
from hololoom.dark_trace.auto_probe import AutoProbe
from hololoom.dark_trace.steering_policy import ConsistencyGuard

def demo_auto_steering():
    print("🌑 AUTO-STEERING DEMO: The Neural Firewall")
    
    # 1. Setup Agent with TRM
    print("\n1. Initializing Agent...")
    agent = MirrorCoreAgent(model_id="agent_007", model_type="trm", d_model=128, recur_depth=4)
    
    # 2. Attach AutoProbe
    print("2. Attaching AutoProbe...")
    probe = AutoProbe(agent, layer_name="recursive_block")
    probe.start()
    
    # 3. Setup Guard
    print("3. Deploying ConsistencyGuard...")
    guard = ConsistencyGuard(probe, correction_strength=5.0)
    
    # 4. Burn-in Phase (Train SAE)
    print("\n🔥 Phase 1: Burn-in (Training SAE on random thoughts)...")
    inputs = torch.randint(0, 1000, (4, 16))
    optimizer = torch.optim.Adam(agent.custom_model.parameters(), lr=1e-3)
    
    for _ in range(30):
        # Simulate agent thinking
        _ = agent.custom_model(inputs)
        time.sleep(0.05) # Give the probe thread time to catch up
        
    metrics = probe.get_brain_health()
    print(f"   SAE Metrics: {metrics}")
    
    # 5. Identify a Target Feature to "Forbid"
    # We grab the strongest feature from the current state
    print("\n🧐 Phase 2: Identifying 'Forbidden' Thought...")
    try:
        current_thoughts = probe.get_current_thought()
        if not current_thoughts:
            print("   No thoughts captured yet. Retrying...")
            _ = agent.custom_model(inputs)
            time.sleep(0.2)
            current_thoughts = probe.get_current_thought()
            
        forbidden_feat = current_thoughts[0]['feature']
        print(f"   Target Acquired: Feature {forbidden_feat} (Simulated 'Deception')")
        
        guard.forbidden_features.add(forbidden_feat)
    except IndexError:
        print("   ❌ Failed to capture thoughts. Probe might need more training time.")
        probe.stop()
        return

    # 6. Verify Enforcement
    print(f"\n🛡️ Phase 3: Testing Neural Firewall on Feature {forbidden_feat}...")
    
    # Run ONE pass. The probe captures it.
    _ = agent.custom_model(inputs)
    
    # We assume the feature IS active because we just saw it.
    # Now we call enforce() which should see it and set steering.
    guard.enforce()
    
    if probe.probe.active_steering.get(forbidden_feat) == -5.0:
        print("   ✅ SUCCESS: Guard detected forbidden feature and set Negative Steering.")
    else:
        print("   ❌ FAILURE: Guard did not react.")
        print(f"   Active Steering: {probe.probe.active_steering}")

    # 7. Run again to see if steering effect (just print message)
    print("\n   (In a real loop, the NEXT forward pass would now be suppressed.)")
    
    probe.stop()
    print("\n✅ Demo Complete.")

if __name__ == "__main__":
    demo_auto_steering()
