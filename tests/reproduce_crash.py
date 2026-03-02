
import sys
import os
import torch
# Ensure we can import HoloLoom
sys.path.append(os.getcwd())

from hololoom.eggroll.mirror_core import MirrorCoreAgent
from hololoom.eggroll.architectures import get_model, NeuromorphicNet

def test_initialization():
    print("Testing MirrorCoreAgent Initialization with Spiking model...")
    
    # Simulate what LoomNode does
    worker_id = 1
    model_type = "spiking"
    # model_kwargs from integration init
    model_kwargs = {"d_model": 64, "n_layers": 2} 
    
    print(f"Simulating LoomNode param passing for worker {worker_id}")
    
    try:
        # DIRECT TEST of the suspected failure point
        agent = MirrorCoreAgent(
            model_id=f"worker_{worker_id}", 
            model_type=model_type, 
            **model_kwargs
        )
        print("SUCCESS: MirrorCoreAgent initialized.")
        
        if isinstance(agent.custom_model, NeuromorphicNet):
            print("SUCCESS: Underlying model is NeuromorphicNet")
        else:
            print(f"FAILURE: Unexpected model type: {type(agent.custom_model)}")
            
    except TypeError as e:
        print(f"CAUGHT EXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"CAUGHT UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_integration_flow():
    print("\nTesting EggrollIntegration flow (Mocking Backend)...")
    from hololoom.eggroll.integration import EggrollIntegration
    
    try:
        # We don't want to actually run the loop, just init
        # But we do want to verify LoomNode creation
        
        # Manually verify LoomNode init signature usage
        from hololoom.eggroll.loom_node import LoomNode
        
        # Simulate passing unexpected kwargs if any
        # The report said "unexpected keyword argument 'worker_id'"
        # This usually happens if 'worker_id' is passed to a function that doesn't accept it.
        # MirrorCoreAgent does accept **kwargs, but it passes them to get_model
        
        print("Manually invoking LoomNode...")
        node = LoomNode(
             worker_id=1,
             model_type="spiking",
             model_kwargs={"d_model": 64},
             use_dark_trace=False
        )
        print("SUCCESS: LoomNode initialized")
        
    except Exception as e:
         print(f"LoomNode Init Failed: {e}")
         import traceback
         traceback.print_exc()

if __name__ == "__main__":
    test_initialization()
    test_integration_flow()
